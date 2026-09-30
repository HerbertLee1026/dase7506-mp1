"""Package the frozen 3600-step RoPE model with a causal window cache."""

import json
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, sha
from student_cache import build_model


PARENT = ROOT / 'runs/student_rope_d6_w192_3600_s17/checkpoint.pt'
OUTPUT = ROOT / 'runs/cache_unigram_bigram_w192_s17'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--unigram-weight', type=float, default=0.10)
    parser.add_argument('--bigram-weight', type=float, default=0.25)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'{args.output} already contains results')
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] not in (
        'student_rope', 'student_rope_dropout',
        'student_rope_dropout_headbias', 'student_rope_dropout_untied',
        'student_rope_mos', 'student_lstm'
    ):
        raise ValueError('Unexpected parent checkpoint')
    config = dict(parent['config'])
    config.update(cache_unigram_weight=args.unigram_weight,
                  cache_bigram_weight=args.bigram_weight,
                  base_temperature=args.temperature,
                  base_implementation=parent['implementation'])
    predictor = build_model(config)
    predictor.base.load_state_dict(parent['model'])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL,
        'implementation': 'student_cache',
        'config': config,
        'model': predictor.state_dict(),
        'seed': parent['seed'],
        'train_tokens': parent['train_tokens'],
        'parent_checkpoint_sha256': sha(args.parent),
    }, checkpoint)
    details = {
        'checkpoint_sha256': sha(checkpoint),
        'parent_checkpoint_sha256': sha(args.parent),
        'student_cache_sha256': sha(ROOT / 'student_cache.py'),
        'student_rope_sha256': sha(ROOT / 'student_rope.py'),
        'selection_split': 'validation',
        'cache_unigram_weight': config['cache_unigram_weight'],
        'cache_bigram_weight': config['cache_bigram_weight'],
        'base_temperature': config['base_temperature'],
    }
    (args.output / 'ancestry.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
