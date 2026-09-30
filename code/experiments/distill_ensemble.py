"""Continue a self-trained model using train-only teacher ensemble targets."""

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, load_data, make_model, setup, sha
from evaluate import score


def load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['protocol'] != PROTOCOL:
        raise ValueError(f'Protocol mismatch in {path}')
    model, implementation_sha = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    return saved, model, implementation_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--teacher-a', required=True, type=Path)
    parser.add_argument('--teacher-b', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--hard-fraction', type=float, default=0.5)
    parser.add_argument('--teacher-b-weight', type=float, default=0.4)
    parser.add_argument('--eval-every', type=int, default=300)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(args.run_dir)
    if not 0 <= args.hard_fraction <= 1:
        parser.error('hard-fraction must be in [0, 1]')
    if not 0 <= args.teacher_b_weight <= 1:
        parser.error('teacher-b-weight must be in [0, 1]')
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.learning_rate <= 0:
        parser.error('steps, batch-size, eval-every, and learning-rate must be positive')
    device, _ = setup('cpu', 'fp32', 4)
    torch.manual_seed(args.seed)
    data = load_data()
    parent, student, student_sha = load(args.parent, device)
    first, teacher_a, _ = load(args.teacher_a, device)
    second, teacher_b, _ = load(args.teacher_b, device)
    teacher_a.eval()
    teacher_b.eval()
    for teacher in (teacher_a, teacher_b):
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=0.1)
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    history, validation_history = [], []
    best_bpb = float('inf')
    best_step = None
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        x, y = batch[:, :-1], batch[:, 1:]
        lr = args.learning_rate * (0.1 + 0.9 * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        soft_target = None
        if args.hard_fraction < 1:
            with torch.no_grad():
                soft_target = (1 - args.teacher_b_weight) * teacher_a.predict_log_probs(x).exp()
                soft_target += args.teacher_b_weight * teacher_b.predict_log_probs(x).exp()
        student.train()
        optimizer.zero_grad(set_to_none=True)
        logits = student(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        hard_loss = F.nll_loss(log_probs.flatten(0, 1), y.flatten())
        soft_loss = (-(soft_target * log_probs).sum(-1).mean()
                     if soft_target is not None else hard_loss)
        loss = args.hard_fraction * hard_loss + (1 - args.hard_fraction) * soft_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            row = {'step': step, 'loss': loss.item(), 'hard_loss': hard_loss.item(),
                   'soft_loss': soft_loss.item(), 'lr': lr,
                   'seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = score(student, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            row = {'step': step, **result}
            validation_history.append(row)
            print(json.dumps({'validation': row}), flush=True)
            if result['bpb'] < best_bpb:
                best_bpb, best_step = result['bpb'], step
                torch.save({
                    'protocol': PROTOCOL, 'implementation': parent['implementation'],
                    'config': parent['config'], 'model': student.cpu().state_dict(),
                    'seed': args.seed,
                    'train_tokens': parent['train_tokens'] + step * args.batch_size * 256,
                    'parent_checkpoint_sha256': sha(args.parent),
                    'teacher_checkpoint_sha256': (sha(args.teacher_a), sha(args.teacher_b)),
                    'step': step,
                }, args.run_dir / 'checkpoint.pt')
                student.to(device)
            progress = {
                'implementation': parent['implementation'], 'config': parent['config'],
                'parent_checkpoint': str(args.parent),
                'teacher_a': str(args.teacher_a), 'teacher_b': str(args.teacher_b),
                'parent_checkpoint_sha256': sha(args.parent),
                'teacher_checkpoint_sha256': (sha(args.teacher_a), sha(args.teacher_b)),
                'seed': args.seed, 'planned_steps': args.steps, 'steps_completed': step,
                'batch_size': args.batch_size, 'hard_fraction': args.hard_fraction,
                'teacher_b_weight': args.teacher_b_weight,
                'learning_rate': args.learning_rate,
                'train_tokens_inherited': parent['train_tokens'],
                'train_tokens_added': step * args.batch_size * 256,
                'seconds': time.perf_counter() - started,
                'best_step': best_step, 'best_validation_bpb': best_bpb,
                'student_implementation_sha256': student_sha,
                'checkpoint_sha256': sha(args.run_dir / 'checkpoint.pt'),
                'history': history, 'validation_history': validation_history,
            }
            (args.run_dir / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    (args.run_dir / 'metrics.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps({k: v for k, v in progress.items() if k not in (
        'history', 'validation_history')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
