"""Add a causal hidden-state copy rule to a frozen variable-ngram checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, ROOT, sha
from student_hidden_variable_ngram import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--weight', type=float, default=0.02)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--dim', type=int, default=320)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_variable_ngram':
        raise ValueError('Expected a same-protocol variable-ngram parent.')
    config = dict(parent['config'])
    config.update(hidden_cache_weight=args.weight, hidden_cache_beta=args.beta,
                  hidden_cache_dim=args.dim)
    model = build_model(config)
    model.load_state_dict(parent['model'])
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL, 'implementation': 'student_hidden_variable_ngram',
        'config': config, 'model': model.state_dict(),
        'seed': parent['seed'], 'train_tokens': parent['train_tokens'],
        'parent_checkpoint_sha256': sha(args.parent),
    }, checkpoint)
    asset = ROOT / config['ngram_asset']
    record = {
        'checkpoint_sha256': sha(checkpoint),
        'parent_checkpoint_sha256': sha(args.parent),
        'implementation_sha256': sha(ROOT / 'student_hidden_variable_ngram.py'),
        'asset_sha256': sha(asset),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'asset_bytes': asset.stat().st_size,
        'total_asset_bytes': checkpoint.stat().st_size + asset.stat().st_size,
        'hidden_cache_weight': args.weight, 'hidden_cache_beta': args.beta,
        'hidden_cache_dim': args.dim,
        'selection_split': 'validation',
    }
    (args.output / 'ancestry.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
