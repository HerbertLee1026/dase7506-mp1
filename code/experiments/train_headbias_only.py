"""Fit only a token-specific output bias on supplied training text."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--eval-every', type=int, default=100)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.learning_rate <= 0:
        parser.error('steps, batch-size, eval-every, and learning-rate must be positive')
    device, _ = setup('cpu', 'fp32', 4)
    torch.manual_seed(args.seed)
    data = load_data()
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL:
        raise ValueError('Protocol mismatch')
    model, implementation_sha = make_model('student_rope_dropout_headbias',
                                           parent['config'], device)
    loaded = model.load_state_dict(parent['model'], strict=False)
    if set(loaded.missing_keys) != {'head.bias'} or loaded.unexpected_keys:
        raise ValueError(f'Unexpected parent/model key mismatch: {loaded}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == 'head.bias')
    optimizer = torch.optim.Adam([model.head.bias], lr=args.learning_rate)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    started = time.perf_counter()
    history, validation_history = [], []
    best_bpb, best_step = float('inf'), None
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(batch[:, :-1]).flatten(0, 1), batch[:, 1:].flatten())
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            row = {'step': step, 'loss': loss.item(),
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
                best_bpb, best_step = result['bpb'], step
                torch.save({
                    'protocol': PROTOCOL,
                    'implementation': 'student_rope_dropout_headbias',
                    'config': parent['config'], 'model': model.cpu().state_dict(),
                    'seed': args.seed,
                    'train_tokens': parent['train_tokens'] + step * args.batch_size * 256,
                    'parent_checkpoint_sha256': sha(args.parent), 'step': step,
                }, args.run_dir / 'checkpoint.pt')
                model.to(device)
            progress = {
                'implementation': 'student_rope_dropout_headbias',
                'config': parent['config'], 'parent_checkpoint': str(args.parent),
                'parent_checkpoint_sha256': sha(args.parent),
                'implementation_sha256': implementation_sha,
                'seed': args.seed, 'planned_steps': args.steps,
                'steps_completed': step, 'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'train_tokens_inherited': parent['train_tokens'],
                'train_tokens_added': step * args.batch_size * 256,
                'seconds': time.perf_counter() - started,
                'best_step': best_step, 'best_validation_bpb': best_bpb,
                'checkpoint_sha256': sha(args.run_dir / 'checkpoint.pt'),
                'history': history, 'validation_history': validation_history,
            }
            (args.run_dir / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    (args.run_dir / 'metrics.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps({k: v for k, v in progress.items() if k not in (
        'history', 'validation_history')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
