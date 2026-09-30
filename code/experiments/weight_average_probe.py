"""Validation-only interpolation of two self-trained compatible checkpoints."""

import argparse
from pathlib import Path

import torch

from common import load_data, make_model, setup
from evaluate import score


ALPHAS = (0.25, 0.5, 0.75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    first = torch.load(args.first, map_location='cpu', weights_only=True)
    second = torch.load(args.second, map_location='cpu', weights_only=True)
    if first['implementation'] != second['implementation']:
        raise ValueError('Implementations differ')
    if (first['config']['vocab'], first['config']['width'], first['config']['depth']) != (
        second['config']['vocab'], second['config']['width'], second['config']['depth']
    ):
        raise ValueError('Architectures differ')
    if first['model'].keys() != second['model'].keys():
        raise ValueError('State keys differ')
    config = dict(second['config'])
    model, _ = make_model(second['implementation'], config, device)
    validation = load_data()['validation']
    for alpha in ALPHAS:
        state = {
            key: (1 - alpha) * first['model'][key] + alpha * second['model'][key]
            for key in first['model']
        }
        model.load_state_dict(state)
        result = score(model, *validation, device, 'fp32')
        print({'second_weight': alpha, 'bpb': result['bpb'],
               'scoring_seconds': result['seconds']}, flush=True)


if __name__ == '__main__':
    main()
