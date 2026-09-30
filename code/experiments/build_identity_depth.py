"""Append an initially identity RoPE block to a self-trained checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, make_model, setup, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    device, _ = setup('cpu', 'fp32', 4)
    parent = torch.load(args.parent, map_location='cpu', weights_only=True)
    if parent['protocol'] != PROTOCOL or parent['implementation'] != 'student_rope_dropout':
        raise ValueError('Expected a regularized RoPE parent checkpoint')
    config = dict(parent['config'])
    old_depth = config['depth']
    config['depth'] = old_depth + 1
    model, implementation_sha = make_model(parent['implementation'], config, device)
    loaded = model.load_state_dict(parent['model'], strict=False)
    if loaded.unexpected_keys or not loaded.missing_keys or any(
        not key.startswith(f'blocks.{old_depth}.') for key in loaded.missing_keys
    ):
        raise ValueError(f'Unexpected parent/model key mismatch: {loaded}')
    block = model.blocks[-1]
    with torch.no_grad():
        block.proj.weight.zero_()
        block.proj.bias.zero_()
        block.mlp.proj.weight.zero_()
        block.mlp.proj.bias.zero_()
    old_model, _ = make_model(parent['implementation'], parent['config'], device)
    old_model.load_state_dict(parent['model'])
    old_model.eval()
    model.eval()
    x = torch.randint(parent['config']['vocab'], (2, 12))
    with torch.no_grad():
        max_delta = (old_model(x) - model(x)).abs().max().item()
    if max_delta > 1e-6:
        raise ValueError(f'Identity block changed logits by {max_delta}')
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'checkpoint.pt'
    torch.save({
        'protocol': PROTOCOL, 'implementation': parent['implementation'],
        'config': config, 'model': model.state_dict(),
        'seed': parent['seed'], 'train_tokens': parent['train_tokens'],
        'parent_checkpoint_sha256': sha(args.parent), 'step': 0,
    }, checkpoint)
    details = {
        'parent_checkpoint_sha256': sha(args.parent),
        'checkpoint_sha256': sha(checkpoint),
        'checkpoint_bytes': checkpoint.stat().st_size,
        'implementation_sha256': implementation_sha,
        'new_depth': config['depth'], 'initial_logit_max_delta': max_delta,
    }
    (args.output / 'ancestry.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(details, indent=2))


if __name__ == '__main__':
    main()
