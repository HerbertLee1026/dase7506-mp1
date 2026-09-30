"""Measure validation scoring time for an untrained architecture, without saving results."""

import argparse
import json
import time
from pathlib import Path

import torch

from common import load_data, make_model, setup
from evaluate import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--implementation', default='student_rope_dropout')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', args.threads)
    torch.manual_seed(17)
    config = json.loads(args.config.read_text())
    model, _ = make_model(args.implementation, config, device)
    validation = load_data()['validation']
    started = time.perf_counter()
    result = score(model, *validation, device, 'fp32')
    print(json.dumps({
        'config': config, 'parameters': sum(p.numel() for p in model.parameters()),
        'scoring_seconds': result['seconds'],
        'process_seconds': time.perf_counter() - started,
    }, indent=2))


if __name__ == '__main__':
    main()
