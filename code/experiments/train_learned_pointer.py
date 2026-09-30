"""Fit a causal prefix pointer on train; select complete predictor on validation."""

import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, load_data, make_model, sha
from evaluate import score
from student_hidden_variable_ngram import HiddenRoPEGPT
from student_learned_pointer import PrefixPointer
from experiments.finetune_cache_objective import cache_target_terms
from experiments.training_state import atomic_save, capture_rng, restore_rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--freeze-neural-steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--pointer-learning-rate', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=127)
    parser.add_argument('--train-device', choices=('cpu', 'mps', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    trainer_hash = sha(__file__)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=True)
        if resumed['trainer_sha256'] != trainer_hash:
            parser.error('Trainer source differs from saved state; review before migrating state.')
        for key, value in resumed['arguments'].items():
            if key not in ('run_dir', 'train_device', 'threads', 'resume'):
                setattr(args, key, Path(value) if key == 'parent' else value)
    if args.parent is None:
        parser.error('--parent is required for a new run.')
    if min(args.steps, args.batch_size, args.eval_every, args.save_every, args.patience) < 1:
        parser.error('Step, batch, evaluation, save interval and patience must be positive.')
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(args.seed)
    device = torch.device(args.train_device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS unavailable.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; check the remote GPU and PyTorch installation.')
    parent = torch.load(args.parent, weights_only=True, map_location='cpu')
    config = dict(parent['config'], pointer_dim=64, pointer_maximum=0.5,
                  hidden_cache_weight=0.0)
    evaluator, impl_hash = make_model('student_learned_pointer', config, torch.device('cpu'))
    missing, extra = evaluator.load_state_dict(parent['model'], strict=False)
    if extra or any(not key.startswith('pointer.') for key in missing):
        raise ValueError((missing, extra))
    neural = HiddenRoPEGPT(config).to(device)
    neural.load_state_dict(evaluator.base.base.state_dict())
    pointer = PrefixPointer(config).to(device)
    pointer.load_state_dict(evaluator.pointer.state_dict())
    optimizer = torch.optim.AdamW([
        {'params': neural.parameters(), 'lr': args.learning_rate, 'initial_lr': args.learning_rate},
        {'params': pointer.parameters(), 'lr': args.pointer_learning_rate,
         'initial_lr': args.pointer_learning_rate}], weight_decay=0.01)
    data = load_data()
    tokens = data['train'][0]
    rng = torch.Generator().manual_seed(args.seed)
    args.run_dir.mkdir(parents=True)
    source = args.run_dir / 'source'
    source.mkdir()
    for name in ('student_learned_pointer.py', 'student_hidden_variable_ngram.py',
                 'student_fast_hidden_ngram.py',
                 'student_variable_ngram.py', 'student_cache.py', 'student_rope_dropout.py',
                 'student_rope.py', 'model.py'):
        shutil.copy2(ROOT / name, source / name)
    for name in ('train_learned_pointer.py', 'finetune_cache_objective.py', 'training_state.py'):
        shutil.copy2(ROOT / 'experiments' / name, source / name)
    parent_hash = sha(args.parent)
    history, validations = [], []
    started = time.perf_counter()
    first_step, inherited_seconds = 1, 0.0
    arguments = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    path = args.run_dir / 'checkpoint.pt'

    def save(step):
        torch.save({'protocol': PROTOCOL, 'implementation': 'student_learned_pointer',
                    'config': config, 'model': evaluator.state_dict(), 'seed': args.seed,
                    'train_tokens': parent['train_tokens'] + step * args.batch_size * 256,
                    'parent_checkpoint_sha256': parent_hash, 'step': step}, path)
    if resumed is not None:
        if resumed['parent_checkpoint_sha256'] != parent_hash or resumed['config'] != config:
            raise ValueError('Resume parent/config mismatch.')
        neural.load_state_dict(resumed['neural'])
        pointer.load_state_dict(resumed['pointer'])
        optimizer.load_state_dict(resumed['optimizer'])
        best, significant = resumed['best'], resumed['significant']
        best_step, stale = resumed['best_step'], resumed['stale']
        history, validations = resumed['history'], resumed['validation_history']
        first_step = resumed['step'] + 1
        inherited_seconds = resumed['seconds']
        if first_step > args.steps:
            raise ValueError('Saved run is already complete.')
        best_source = args.resume.parent / 'checkpoint.pt'
        if sha(best_source) != resumed['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint no longer matches the resume snapshot.')
        shutil.copy2(best_source, path)
        same_device = restore_rng(resumed['rng'], rng, device)
        print(json.dumps({'resumed_step': first_step-1, 'same_device_rng': same_device,
                          'note': 'Different hardware can change floating-point results.'}), flush=True)
    else:
        initial = score(evaluator, *data['validation'], torch.device('cpu'), 'fp32')
        initial.pop('window_nll_nats')
        best = significant = initial['bpb']
        best_step, stale = 0, 0
        validations.append({'step': 0, **initial})
        print(json.dumps({'validation': validations[-1]}), flush=True)
        save(0)
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for step in range(first_step, args.steps + 1):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng)
        batch = tokens[starts[:, None]+torch.arange(257)]
        xx, yy = batch[:, :-1], batch[:, 1:]
        mass, copied = cache_target_terms(xx, yy, config['cache_unigram_weight'],
                                         config['cache_bigram_weight'])
        x, y, mass, copied = [value.to(device) for value in (xx, yy, mass, copied)]
        neural.train()
        pointer.train()
        optimizer.zero_grad(set_to_none=True)
        factor = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi*(step-1)/args.steps))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * factor
        with torch.set_grad_enabled(step > args.freeze_neural_steps):
            logits, hidden = neural.forward_with_hidden(x)
        base = F.softmax(logits.float()/config['base_temperature'], -1).gather(
            -1, y.unsqueeze(-1)).squeeze(-1)
        base = mass * base + copied
        attention, gate = pointer(hidden)
        following = torch.cat((x[:, 1:], x[:, -1:]), 1)
        correct = following[:, None, :] == y[:, :, None]
        target_copy = (attention * correct).sum(-1)
        probability = (1-gate)*base + gate*target_copy
        loss = -probability.clamp_min(1e-30).log().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(neural.parameters())+list(pointer.parameters()), 1.0)
        optimizer.step()
        if step % 50 == 0:
            row = {'step': step, 'loss': loss.item(), 'mean_gate': gate.mean().item(),
                   'seconds': inherited_seconds+time.perf_counter()-started}
            history.append(row)
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluator.base.base.load_state_dict({k:v.detach().cpu() for k,v in neural.state_dict().items()})
            evaluator.pointer.load_state_dict({k:v.detach().cpu() for k,v in pointer.state_dict().items()})
            result = score(evaluator, *data['validation'], torch.device('cpu'), 'fp32')
            result.pop('window_nll_nats')
            validations.append({'step':step, **result})
            print(json.dumps({'validation': validations[-1]}), flush=True)
            if result['bpb'] < best:
                best, best_step = result['bpb'], step
                save(step)
            if result['bpb'] < significant - .0005:
                significant, stale = result['bpb'], 0
            elif step > args.freeze_neural_steps:
                stale += 1
        if (step % args.save_every == 0 or step % args.eval_every == 0
                or step == args.steps or stopped[0]):
            elapsed = inherited_seconds+time.perf_counter()-started
            progress = {'arguments': arguments,
                        'config':config, 'steps_completed':step, 'best_step':best_step,
                        'best_validation_bpb':best, 'history':history, 'validation_history':validations,
                        'parent_checkpoint_sha256':parent_hash, 'trainer_sha256':trainer_hash,
                        'implementation_sha256':impl_hash, 'checkpoint_sha256':sha(path),
                        'train_tokens_inherited':parent['train_tokens'],
                        'train_tokens_added':step*args.batch_size*256,
                        'seconds':elapsed, 'selection_split':'validation',
                        'train_device':str(device), 'torch_version':str(torch.__version__)}
            atomic_save({'arguments': arguments, 'config': config, 'step': step,
                         'neural': neural.state_dict(), 'pointer': pointer.state_dict(),
                         'optimizer': optimizer.state_dict(), 'rng': capture_rng(rng, device),
                         'best': best, 'significant': significant, 'best_step': best_step,
                         'stale': stale, 'history': history, 'validation_history': validations,
                         'seconds': elapsed, 'trainer_sha256': trainer_hash,
                         'parent_checkpoint_sha256': parent_hash,
                         'best_checkpoint_sha256': sha(path)}, args.run_dir/'training_state.pt')
            (args.run_dir/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            if stopped[0] or stale >= args.patience:
                progress['stop_reason'] = ('User signal; resumable state saved.' if stopped[0]
                                         else 'No material validation gain.')
                break
    (args.run_dir/'metrics.json').write_text(json.dumps(progress,indent=2)+'\n')


if __name__ == '__main__':
    main()
