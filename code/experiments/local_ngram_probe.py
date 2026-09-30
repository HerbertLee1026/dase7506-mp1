"""Probe longer causal copy patterns inside each validation window."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


WEIGHTS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = load_data()['validation']
    buckets = [[] for _ in range(5)]
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            base_prob = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
            batch, length = x.shape
            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            match = (x[:, :, None] == x[:, None, :]) & earlier
            results = []
            for lag in (1, 2, 3):
                prev = torch.cat((torch.full((batch, lag), -1, dtype=x.dtype), x[:, :-lag]), dim=1)
                match &= prev[:, :, None] == prev[:, None, :]
                match &= (positions[:, None] >= lag) & (positions[None, :] >= lag)
                count = match.sum(-1)
                probability = (match & correct).sum(-1).float() / count.clamp_min(1)
                if lag >= 2:
                    results.append((probability[valid].numpy(), (count > 0)[valid].numpy()))
            buckets[0].append(base_prob[valid].numpy())
            for i, (probability, available) in enumerate(results):
                buckets[1 + i * 2].append(probability)
                buckets[2 + i * 2].append(available)
    base, q3, m3, q4, m4 = [np.concatenate(parts) for parts in buckets]
    results = []
    for a3 in WEIGHTS:
        for a4 in WEIGHTS:
            if a3 + a4 >= 1:
                continue
            mix = (1 - a3 * m3 - a4 * m4) * base + a3 * m3 * q3 + a4 * m4 * q4
            bpb = -np.log(mix).sum() / math.log(2) / byte_count
            results.append((bpb, a3, a4))
    for bpb, a3, a4 in sorted(results)[:15]:
        print({'bpb': bpb, 'local_trigram_weight': a3,
               'local_fourgram_weight': a4})


if __name__ == '__main__':
    main()
