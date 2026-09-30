"""Train the neural part against a causal local-copy mixture; select on validation."""

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from common import PROTOCOL, ROOT, load_data, make_model, sha
from evaluate import score
from student_hidden_variable_ngram import HiddenRoPEGPT


def cache_target_terms(x, y, unigram_weight, bigram_weight):
    """Return neural mass and copied target mass from strictly earlier positions."""
    batch, length = x.shape
    positions = torch.arange(length, device=x.device)
    earlier = positions[None, :] < positions[:, None]
    following = torch.cat((x[:, 1:], x[:, -1:]), 1)
    correct = following[:, None, :] == y[:, :, None]
    match1 = (x[:, :, None] == x[:, None, :]) & earlier
    preceding = torch.cat((torch.full_like(x[:, :1], -1), x[:, :-1]), 1)
    match2 = (match1 & (preceding[:, :, None] == preceding[:, None, :])
              & (positions[:, None] >= 1) & (positions[None, :] >= 1))
    count1, count2 = match1.sum(-1), match2.sum(-1)
    w1, w2 = unigram_weight * (count1 > 0), bigram_weight * (count2 > 0)
    q1 = (match1 & correct).sum(-1).float() / count1.clamp_min(1)
    q2 = (match2 & correct).sum(-1).float() / count2.clamp_min(1)
    return 1 - w1 - w2, w1 * q1 + w2 * q2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.0005)
    parser.add_argument('--parent-bpb', type=float, required=True)
    parser.add_argument('--train-device', choices=('cpu', 'mps', 'cuda'), default='mps')
    parser.add_argument('--loss-mode', choices=('cache', 'neural'), default='cache')
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--optimizer', choices=('adamw', 'muon'), default='adamw')
    parser.add_argument('--muon-learning-rate', type=float, default=0.002)
    parser.add_argument('--muon-weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=0)
    args = parser.parse_args()
    trainer_hash = sha(__file__)
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if min(args.steps, args.batch_size, args.eval_every, args.patience) < 1:
        parser.error('Positive step, batch, evaluation interval and patience required.')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    device = torch.device(args.train_device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; check the remote GPU and PyTorch installation.')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_hidden_variable_ngram':
        raise ValueError('Expected a complete hidden-cache/ngram parent.')
    config = dict(parent['config'])
    if args.dropout is not None:
        if not 0 <= args.dropout < 1:
            parser.error('Dropout must be in [0, 1).')
        config['dropout'] = args.dropout
    evaluator, implementation_sha = make_model(parent['implementation'], config, torch.device('cpu'))
    evaluator.load_state_dict(parent['model'])
    neural = HiddenRoPEGPT(config).to(device)
    neural.load_state_dict(evaluator.base.base.state_dict())
    matrix_optimizer = None
    if args.optimizer == 'muon':
        from experiments.muon_local import SingleDeviceMuon
        matrices = [p for name, p in neural.named_parameters()
                    if name.startswith('blocks.') and p.ndim == 2]
        matrix_ids = {id(p) for p in matrices}
        auxiliary = [p for p in neural.parameters() if id(p) not in matrix_ids]
        matrix_optimizer = SingleDeviceMuon(
            matrices, lr=args.muon_learning_rate, weight_decay=args.muon_weight_decay)
        optimizer = torch.optim.AdamW(auxiliary, lr=args.learning_rate,
                                     betas=(0.9, 0.95), weight_decay=0.1)
    else:
        optimizer = torch.optim.AdamW(neural.parameters(), lr=args.learning_rate, weight_decay=0.1)
    data = load_data()
    train_tokens = data['train'][0]
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    sources = args.run_dir / 'source'
    sources.mkdir()
    for name in ('student_hidden_variable_ngram.py', 'student_variable_ngram.py',
                 'student_cache.py', 'student_rope_dropout.py', 'student_rope.py', 'model.py'):
        shutil.copy2(ROOT / name, sources / name)
    shutil.copy2(__file__, sources / Path(__file__).name)
    if matrix_optimizer is not None:
        for name in ('muon_local.py', 'MUON_LICENSE.txt'):
            shutil.copy2(ROOT / 'experiments' / name, sources / name)
    parent_hash = sha(args.parent)
    history, validations = [], []
    best_bpb = significant_best = args.parent_bpb
    best_step, stale = 0, 0
    started = time.perf_counter()
    checkpoint_path = args.run_dir / 'checkpoint.pt'
    shutil.copy2(args.parent, checkpoint_path)
    for step in range(1, args.steps + 1):
        starts = torch.randint(len(train_tokens) - 257, (args.batch_size,), generator=rng)
        batch = train_tokens[starts[:, None] + torch.arange(257)]
        xx, yy = batch[:, :-1], batch[:, 1:]
        if args.loss_mode == 'cache':
            neural_mass, copied = cache_target_terms(
                xx, yy, config['cache_unigram_weight'], config['cache_bigram_weight'])
            neural_mass, copied = neural_mass.to(device), copied.to(device)
        x, y = xx.to(device), yy.to(device)
        lr = args.learning_rate * (0.2 + 0.8 * 0.5 * (
            1 + math.cos(math.pi * (step - 1) / args.steps)))
        if args.warmup_steps > 0:
            lr *= min(1.0, step / args.warmup_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr
        if matrix_optimizer is not None:
            for group in matrix_optimizer.param_groups:
                group['lr'] = args.muon_learning_rate * lr / args.learning_rate
        neural.train()
        optimizer.zero_grad(set_to_none=True)
        if matrix_optimizer is not None:
            matrix_optimizer.zero_grad(set_to_none=True)
        logits = neural(x).float()
        if args.loss_mode == 'cache':
            logp = F.log_softmax(logits / config['base_temperature'], -1).gather(
                -1, y.unsqueeze(-1)).squeeze(-1)
            logp = torch.logaddexp(logp + neural_mass.log(), copied.clamp_min(1e-30).log())
            loss = -logp.mean()
        else:
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at step {step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neural.parameters(), 1.0)
        optimizer.step()
        if matrix_optimizer is not None:
            matrix_optimizer.step()
        if step % 50 == 0:
            row = {'step': step, 'loss': loss.item(), 'lr': lr,
                   'seconds': time.perf_counter() - started}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluator.base.base.load_state_dict({
                key: value.detach().cpu() for key, value in neural.state_dict().items()})
            result = score(evaluator, *data['validation'], torch.device('cpu'), 'fp32')
            result.pop('window_nll_nats')
            row = {'step': step, **result}
            validations.append(row)
            print(json.dumps({'validation': row}), flush=True)
            if result['bpb'] < best_bpb:
                best_bpb, best_step = result['bpb'], step
                torch.save({
                    'protocol': PROTOCOL, 'implementation': parent['implementation'],
                    'config': config, 'model': evaluator.state_dict(), 'seed': args.seed,
                    'train_tokens': parent['train_tokens'] + step * args.batch_size * 256,
                    'parent_checkpoint_sha256': parent_hash, 'step': step,
                }, checkpoint_path)
            if result['bpb'] < significant_best - args.min_delta:
                significant_best, stale = result['bpb'], 0
            else:
                stale += 1
            progress = {
                'protocol': PROTOCOL, 'implementation': parent['implementation'],
                'config': config, 'parent_checkpoint': str(args.parent),
                'parent_checkpoint_sha256': parent_hash,
                'parent_validation_bpb': args.parent_bpb,
                'arguments': {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
                'steps_completed': step, 'best_step': best_step,
                'best_validation_bpb': best_bpb,
                'train_tokens_inherited': parent['train_tokens'],
                'train_tokens_added': step * args.batch_size * 256,
                'seconds': time.perf_counter() - started,
                'implementation_sha256': implementation_sha,
                'trainer_sha256': trainer_hash,
                'checkpoint_sha256': sha(checkpoint_path),
                'history': history, 'validation_history': validations,
                'selection_split': 'validation', 'stale_validations': stale,
            }
            (args.run_dir / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
            if stale >= args.patience:
                progress['stop_reason'] = 'No material complete-predictor validation gain.'
                break
    (args.run_dir / 'metrics.json').write_text(json.dumps(progress, indent=2) + '\n')
    print(json.dumps({'best_step': best_step, 'best_validation_bpb': best_bpb,
                      'steps_completed': step}), flush=True)


if __name__ == '__main__':
    main()
