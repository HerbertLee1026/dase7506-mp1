"""Validation-only weight interpolation between checkpoints on one trajectory."""

import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, load_data, make_model, setup
from evaluate import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.25, 0.5, 0.75])
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    if a['protocol'] != PROTOCOL or b['protocol'] != PROTOCOL:
        raise ValueError('Protocol mismatch')
    if a['implementation'] != b['implementation'] or a['config'] != b['config']:
        raise ValueError('Architecture mismatch')
    if a['model'].keys() != b['model'].keys():
        raise ValueError('State keys mismatch')
    model, _ = make_model(a['implementation'], a['config'], device)
    data = load_data()['validation']
    for alpha in args.alphas:
        if not 0 <= alpha <= 1:
            raise ValueError(alpha)
        blended = {}
        for key in a['model']:
            left, right = a['model'][key], b['model'][key]
            if not left.is_floating_point():
                if not torch.equal(left, right):
                    raise ValueError(f'Nonfloating state differs: {key}')
                blended[key] = left
            else:
                blended[key] = torch.lerp(left, right, alpha)
        model.load_state_dict(blended)
        result = score(model, *data, device, 'fp32')
        print(json.dumps({'alpha': alpha, 'bpb': result['bpb'],
                          'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
