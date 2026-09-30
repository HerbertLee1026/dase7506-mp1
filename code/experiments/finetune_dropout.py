"""Continue a self-trained RoPE checkpoint with dropout; select on validation."""

import json
import math
import argparse
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, load_data, make_model, setup, sha
from evaluate import score


PARENT = ROOT / 'runs/student_rope_d6_w192_3600_s17/checkpoint.pt'
RUN_DIR = ROOT / 'runs/student_rope_dropout_d6_w192_ft3000_s41'
STEPS = 3000
BATCH = 32
SEED = 41


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--implementation', default='student_rope_dropout')
    parser.add_argument('--run-dir', type=Path, default=RUN_DIR)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--batch-size', type=int, default=BATCH)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--attention-dropout', type=float, default=None)
    parser.add_argument('--last-block-lr-multiplier', type=float, default=1.0)
    parser.add_argument('--eval-every', type=int, default=300)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    if (args.steps < 1 or args.batch_size < 1 or args.eval_every < 1
            or args.learning_rate <= 0 or args.last_block_lr_multiplier <= 0):
        parser.error('steps, batch size, validation interval, and learning rate must be positive')
    device, _ = setup('cpu', 'fp32', 4)
    torch.manual_seed(args.seed)
    data = load_data()
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    config = dict(parent['config'])
    config.setdefault('dropout', 0.1)
    if args.dropout is not None:
        if not 0 <= args.dropout < 1:
            parser.error('dropout must be in [0, 1)')
        config['dropout'] = args.dropout
    if args.attention_dropout is not None:
        if not 0 <= args.attention_dropout < 1:
            parser.error('attention-dropout must be in [0, 1)')
        config['attention_dropout'] = args.attention_dropout
    model, implementation_sha = make_model(args.implementation, config, device)
    loaded = model.load_state_dict(parent['model'], strict=False)
    allowed_missing = (
        {'expert_proj.weight', 'mixture_gate.weight', 'mixture_gate.bias'}
        if args.implementation == 'student_rope_mos'
        else {'head.bias'} if args.implementation in (
            'student_rope_dropout_headbias', 'student_rope_dropout_untied')
        else set()
    )
    if set(loaded.missing_keys) != allowed_missing or loaded.unexpected_keys:
        raise ValueError(f'Unexpected parent/model key mismatch: {loaded}')
    if args.last_block_lr_multiplier == 1:
        groups = [{'params': model.parameters(), 'lr_multiplier': 1.0}]
    else:
        last_prefix = f"blocks.{config['depth'] - 1}."
        common_params, last_params = [], []
        for name, parameter in model.named_parameters():
            (last_params if name.startswith(last_prefix) else common_params).append(parameter)
        if not last_params:
            raise ValueError('No last-block parameters found')
        groups = [{'params': common_params, 'lr_multiplier': 1.0},
                  {'params': last_params, 'lr_multiplier': args.last_block_lr_multiplier}]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, weight_decay=0.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    history = []
    validation_history = []
    best_bpb = float('inf')
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        lr = args.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_multiplier']
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            row = {'step': step, 'loss': loss.item(), 'lr': lr,
                   'seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = score(model, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            row = {'step': step, **result}
            validation_history.append(row)
            print(json.dumps({'validation': row}), flush=True)
            if result['bpb'] < best_bpb:
                best_bpb = result['bpb']
                torch.save({
                    'protocol': PROTOCOL, 'implementation': args.implementation,
                    'config': config, 'model': model.cpu().state_dict(),
                    'seed': args.seed, 'train_tokens': parent['train_tokens'] + step * args.batch_size * 256,
                    'parent_checkpoint_sha256': sha(args.parent),
                    'step': step,
                }, args.run_dir / 'checkpoint.pt')
                model.to(device)
            progress = {
                'implementation': args.implementation, 'config': config,
                'parent_checkpoint': str(args.parent),
                'parent_checkpoint_sha256': sha(args.parent),
                'seed': args.seed, 'planned_steps': args.steps,
                'steps_completed': step, 'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'dropout_override': args.dropout,
                'attention_dropout_override': args.attention_dropout,
                'last_block_lr_multiplier': args.last_block_lr_multiplier,
                'train_tokens_inherited': parent['train_tokens'],
                'train_tokens_added': step * args.batch_size * 256,
                'seconds': time.perf_counter() - started,
                'best_validation_bpb': best_bpb,
                'history': history, 'validation_history': validation_history,
                'checkpoint_sha256': sha(args.run_dir / 'checkpoint.pt'),
            }
            (args.run_dir / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    summary = {
        'implementation': args.implementation, 'config': config,
        'parent_checkpoint': str(args.parent),
        'parent_checkpoint_sha256': sha(args.parent),
        'implementation_sha256': implementation_sha,
        'train_tokens_inherited': parent['train_tokens'],
        'train_tokens_added': args.steps * args.batch_size * 256,
        'seed': args.seed, 'learning_rate': args.learning_rate,
        'dropout_override': args.dropout,
        'attention_dropout_override': args.attention_dropout,
        'last_block_lr_multiplier': args.last_block_lr_multiplier,
        'seconds': time.perf_counter() - started,
        'best_validation_bpb': best_bpb,
        'history': history, 'validation_history': validation_history,
        'checkpoint_sha256': sha(args.run_dir / 'checkpoint.pt'),
    }
    (args.run_dir / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('history', 'validation_history')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
