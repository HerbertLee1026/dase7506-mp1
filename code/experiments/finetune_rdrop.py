"""Fine-tune a self-trained dropout model with two-view consistency."""

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, sha
from evaluate import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=89)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--consistency-weight', type=float, default=0.5)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--train-device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    if min(args.steps, args.batch_size, args.eval_every) <= 0:
        parser.error('steps, batch size, and eval interval must be positive')
    if args.learning_rate <= 0 or args.consistency_weight < 0:
        parser.error('learning rate must be positive and consistency weight nonnegative')
    torch.set_num_threads(4)
    device = torch.device(args.train_device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable')
    torch.manual_seed(args.seed)
    data = load_data()
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL:
        raise ValueError('Parent protocol mismatch')
    model, implementation_sha = make_model(parent['implementation'], parent['config'], device)
    model.load_state_dict(parent['model'])
    eval_model = (model if device.type == 'cpu'
                  else make_model(parent['implementation'], parent['config'], torch.device('cpu'))[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    history, validation_history = [], []
    best_bpb, best_step = float('inf'), None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        x, y = batch[:, :-1], batch[:, 1:]
        lr = args.learning_rate * (0.1 + 0.9 * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        log1 = F.log_softmax(model(x).float(), -1)
        log2 = F.log_softmax(model(x).float(), -1)
        hard = 0.5 * (
            F.nll_loss(log1.flatten(0, 1), y.flatten())
            + F.nll_loss(log2.flatten(0, 1), y.flatten()))
        kl = 0.5 * (
            (log1.exp() * (log1 - log2)).sum(-1).mean()
            + (log2.exp() * (log2 - log1)).sum(-1).mean())
        loss = hard + args.consistency_weight * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            row = {'step': step, 'loss': loss.item(), 'hard_loss': hard.item(),
                   'consistency_kl': kl.item(), 'lr': lr,
                   'seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if eval_model is not model:
                eval_model.load_state_dict({
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                })
            result = score(eval_model, *data['validation'], torch.device('cpu'), 'fp32')
            result.pop('window_nll_nats')
            row = {'step': step, **result}
            validation_history.append(row)
            print(json.dumps({'validation': row}), flush=True)
            if result['bpb'] < best_bpb:
                best_bpb, best_step = result['bpb'], step
                torch.save({
                    'protocol': PROTOCOL, 'implementation': parent['implementation'],
                    'config': parent['config'], 'model': eval_model.state_dict(),
                    'seed': args.seed,
                    'train_tokens': parent['train_tokens'] + 2 * step * args.batch_size * 256,
                    'sampled_train_targets': step * args.batch_size * 256,
                    'parent_checkpoint_sha256': sha(args.parent),
                    'step': step,
                }, args.run_dir / 'checkpoint.pt')
            progress = {
                'implementation': parent['implementation'], 'config': parent['config'],
                'parent_checkpoint': str(args.parent),
                'parent_checkpoint_sha256': sha(args.parent),
                'seed': args.seed, 'planned_steps': args.steps,
                'steps_completed': step, 'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'consistency_weight': args.consistency_weight,
                'train_device': args.train_device,
                'training_forward_backward_passes_per_batch': 2,
                'train_tokens_inherited': parent['train_tokens'],
                'train_tokens_added': 2 * step * args.batch_size * 256,
                'seconds': time.perf_counter() - started,
                'best_step': best_step, 'best_validation_bpb': best_bpb,
                'implementation_sha256': implementation_sha,
                'checkpoint_sha256': sha(args.run_dir / 'checkpoint.pt'),
                'history': history, 'validation_history': validation_history,
            }
            (args.run_dir / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    (args.run_dir / 'metrics.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps({k: v for k, v in progress.items() if k not in (
        'history', 'validation_history')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
