"""Package a causal cache checkpoint with compact train-derived ngrams."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, sha
from student_variable_ngram import build_model


ASSET = ROOT / 'assets/train_ngrams_3to5.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset', type=Path, default=ASSET)
    parser.add_argument('--trigram-weight', type=float, default=0.1)
    parser.add_argument('--fourgram-weight', type=float, default=0.05)
    parser.add_argument('--fivegram-weight', type=float, default=0.08)
    parser.add_argument('--max-branches', type=int, default=128)
    parser.add_argument('--later-max-branches', type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    asset = args.asset.resolve()
    asset_relative = asset.relative_to(ROOT)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_cache':
        raise ValueError('Unexpected parent checkpoint')
    config = dict(parent['config'])
    config.update(
        ngram_asset=str(asset_relative),
        ngram_asset_sha256=sha(asset),
        train_trigram_weight=args.trigram_weight,
        train_fourgram_weight=args.fourgram_weight,
        train_fivegram_weight=args.fivegram_weight,
        train_ngram_max_branches=args.max_branches,
        train_later_ngram_max_branches=args.later_max_branches,
    )
    predictor = build_model(config)
    predictor.base.load_state_dict(parent['model'])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL, 'implementation': 'student_variable_ngram',
        'config': config, 'model': predictor.state_dict(),
        'seed': parent['seed'], 'train_tokens': parent['train_tokens'],
        'parent_checkpoint_sha256': sha(args.parent),
    }, checkpoint_path)
    details = {
        'checkpoint_sha256': sha(checkpoint_path),
        'parent_checkpoint_sha256': sha(args.parent),
        'asset_sha256': sha(asset),
        'asset_bytes': asset.stat().st_size,
        'checkpoint_bytes': checkpoint_path.stat().st_size,
        'total_asset_bytes': asset.stat().st_size + checkpoint_path.stat().st_size,
        'implementation_sha256': sha(ROOT / 'student_variable_ngram.py'),
        'selection_split': 'validation',
        'train_trigram_weight': args.trigram_weight,
        'train_fourgram_weight': args.fourgram_weight,
        'train_fivegram_weight': args.fivegram_weight,
        'train_ngram_max_branches': args.max_branches,
        'train_later_ngram_max_branches': args.later_max_branches,
    }
    (args.output / 'ancestry.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
