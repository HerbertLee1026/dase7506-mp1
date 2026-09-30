"""Probe causal copy strength as a function of observed context count."""

import argparse
import math
from pathlib import Path

import torch

from common import load_data, make_model, setup, windows


GAINS = (-0.05, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--temperature', type=float, default=1.15)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = load_data()['validation']
    scores = {gain: 0.0 for gain in GAINS}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            batch, length = x.shape
            logits = model(x).float() / args.temperature
            base = (logits.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                    - logits.logsumexp(-1)).exp()
            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            match1 = (x[:, :, None] == x[:, None, :]) & earlier
            previous = torch.cat((torch.full((batch, 1), -1, dtype=x.dtype), x[:, :-1]), dim=1)
            match2 = match1 & (previous[:, :, None] == previous[:, None, :]) & (
                positions[:, None] >= 1) & (positions[None, :] >= 1)
            count1, count2 = match1.sum(-1), match2.sum(-1)
            copy1 = (match1 & correct).sum(-1).float() / count1.clamp_min(1)
            copy2 = (match2 & correct).sum(-1).float() / count2.clamp_min(1)
            weight1 = 0.05 * (count1 > 0)
            for gain in GAINS:
                weight2 = (0.25 + gain * torch.log2(count2.clamp_min(1))).clamp(0.0, 0.7)
                weight2 *= count2 > 0
                mixture = (1 - weight1 - weight2) * base + weight1 * copy1 + weight2 * copy2
                scores[gain] += -mixture[valid].double().log().sum().item()
    for gain, nll in sorted(scores.items(), key=lambda item: item[1]):
        print({'gain': gain, 'bpb': nll / math.log(2) / byte_count})


if __name__ == '__main__':
    main()
