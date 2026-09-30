"""Validation-only target-probability probe for two self-trained predictors."""

import argparse
import math
from pathlib import Path

import torch

from common import load_data, make_model, setup, windows


WEIGHTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', required=True, type=Path)
    parser.add_argument('--second', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    first, second = load(args.first, device), load(args.second, device)
    validation, bytes_count = load_data()['validation']
    scores = {weight: 0.0 for weight in WEIGHTS}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            first_prob = first.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
            second_prob = second.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
            for weight in WEIGHTS:
                mix = (1 - weight) * first_prob + weight * second_prob
                scores[weight] += -mix[valid].double().log().sum().item()
    for weight, nll in sorted(scores.items(), key=lambda item: item[1]):
        print({'second_weight': weight, 'bpb': nll / math.log(2) / bytes_count})


if __name__ == '__main__':
    main()
