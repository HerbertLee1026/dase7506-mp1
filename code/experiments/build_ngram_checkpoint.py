"""Package a self-trained checkpoint with train-derived ngram counts."""

import json
import argparse
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, sha
from student_cache_ngram import build_model


PARENT = ROOT / 'runs/cache_unigram_bigram_w192_s17/checkpoint.pt'
OUTPUT = ROOT / 'runs/cache_train_ngram_w192_s17'
ASSET = ROOT / 'assets/train_ngrams.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--max-branches', type=int, default=100000)
    parser.add_argument('--bigram-weight', type=float, default=0.06)
    parser.add_argument('--trigram-weight', type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_cache':
        raise ValueError('Unexpected parent checkpoint')
    config = dict(parent['config'])
    config.update(
        ngram_asset='assets/train_ngrams.npz',
        ngram_asset_sha256=sha(ASSET),
        train_bigram_weight=args.bigram_weight,
        train_trigram_weight=args.trigram_weight,
        train_trigram_max_branches=args.max_branches,
    )
    predictor = build_model(config)
    predictor.base.load_state_dict(parent['model'])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL,
        'implementation': 'student_cache_ngram',
        'config': config,
        'model': predictor.state_dict(),
        'seed': parent['seed'],
        'train_tokens': parent['train_tokens'],
        'parent_checkpoint_sha256': sha(args.parent),
    }, checkpoint)
    details = {
        'checkpoint_sha256': sha(checkpoint),
        'parent_checkpoint_sha256': sha(args.parent),
        'ngram_asset_sha256': sha(ASSET),
        'ngram_asset_bytes': ASSET.stat().st_size,
        'implementation_sha256': sha(ROOT / 'student_cache_ngram.py'),
        'selection_split': 'validation',
        'train_bigram_weight': config['train_bigram_weight'],
        'train_trigram_weight': config['train_trigram_weight'],
        'train_trigram_max_branches': config['train_trigram_max_branches'],
    }
    (args.output / 'ancestry.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
