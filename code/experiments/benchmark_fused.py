"""Check and benchmark an algebraically equivalent fused projection on CPU."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from common import make_model, setup, sha


def convert_state(state, depth):
    state = dict(state)
    for layer in range(depth):
        prefix = f'base.base.blocks.{layer}.mlp.'
        for suffix in ('weight', 'bias'):
            state[prefix + 'gate_value.' + suffix] = torch.cat((
                state.pop(prefix + 'gate.' + suffix),
                state.pop(prefix + 'value.' + suffix)), dim=0)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device, _ = setup('cpu', 'fp32', 4)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    old, _ = make_model(parent['implementation'], parent['config'], device)
    old.load_state_dict(parent['model'])
    new, source_sha = make_model('student_fused_hidden_ngram', parent['config'], device)
    new.load_state_dict(convert_state(parent['model'], parent['config']['depth']))
    old.eval(); new.eval()
    torch.manual_seed(107)
    x = torch.randint(0, 2048, (32, 256))
    with torch.inference_mode():
        a, b = old.predict_log_probs(x), new.predict_log_probs(x)
        max_delta = (a - b).abs().max().item()
        if max_delta > 1e-4:
            raise ValueError(f'Prediction mismatch: {max_delta}')
        timings = {'original': [], 'fused': []}
        for iteration in range(8):
            pairs = [('original', old), ('fused', new)]
            if iteration % 2:
                pairs.reverse()
            for name, model in pairs:
                start = time.perf_counter()
                model.predict_log_probs(x)
                timings[name].append(time.perf_counter() - start)
    args.output.mkdir(parents=True)
    saved = dict(parent)
    saved.update(implementation='student_fused_hidden_ngram', model=new.state_dict(),
                 parent_checkpoint_sha256=sha(args.parent))
    torch.save(saved, args.output / 'checkpoint.pt')
    result = {'max_log_probability_delta': max_delta,
              'batch_timings_seconds': timings,
              'median_seconds': {k: statistics.median(v) for k, v in timings.items()},
              'implementation_sha256': source_sha,
              'parent_checkpoint_sha256': sha(args.parent),
              'checkpoint_sha256': sha(args.output / 'checkpoint.pt'),
              'note': 'Synthetic CPU batch microbenchmark; full idle validation required for budget.'}
    (args.output / 'benchmark.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
