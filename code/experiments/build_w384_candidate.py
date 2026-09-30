"""Reuse the existing width-384 neural model with a smaller train-only asset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PROTOCOL, ROOT, sha
from student_hidden_variable_ngram import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_asset = ROOT / 'assets/train_ngrams_3to5_pruned128.npz'
    asset = ROOT / 'assets/train_ngrams_4to5_only.npz'
    if asset.exists():
        raise FileExistsError(asset)
    with np.load(source_asset) as saved:
        arrays = {k: saved[k] for k in saved.files if k.startswith(('four_', 'five_'))}
    np.savez(asset, **arrays)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_rope_dropout':
        raise ValueError('Expected raw dropout RoPE parent.')
    config = dict(parent['config'])
    config.update(base_implementation='student_rope_dropout', base_temperature=1.15,
                  cache_unigram_weight=0.05, cache_bigram_weight=0.25,
                  ngram_asset=str(asset.relative_to(ROOT)), ngram_asset_sha256=sha(asset),
                  train_trigram_weight=0.0, train_fourgram_weight=0.05,
                  train_fivegram_weight=0.08, train_ngram_max_branches=128,
                  train_later_ngram_max_branches=1000, hidden_cache_weight=0.02,
                  hidden_cache_beta=10.0, hidden_cache_dim=128)
    model = build_model(config)
    model.base.base.load_state_dict(parent['model'])
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': 'student_hidden_variable_ngram',
                'config': config, 'model': model.state_dict(), 'seed': parent['seed'],
                'train_tokens': parent['train_tokens'],
                'parent_checkpoint_sha256': sha(args.parent)}, checkpoint)
    record = {'parent_checkpoint': str(args.parent),
              'parent_checkpoint_sha256': sha(args.parent),
              'checkpoint_sha256': sha(checkpoint), 'ngram_asset_sha256': sha(asset),
              'parameters': sum(p.numel() for p in model.parameters()),
              'checkpoint_bytes': checkpoint.stat().st_size,
              'asset_bytes': asset.stat().st_size,
              'total_inference_bytes': checkpoint.stat().st_size + asset.stat().st_size,
              'asset_source_sha256': sha(source_asset), 'selection_split': 'validation'}
    (args.output / 'ancestry.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    if record['total_inference_bytes'] > 64 * 1024 ** 2:
        raise ValueError('Candidate exceeds asset budget; do not train.')


if __name__ == '__main__':
    main()
