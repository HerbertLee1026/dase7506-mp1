"""Validation-only joint cache recency and weight selection."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import load_data, make_model, setup, windows


DECAYS = (None, 32.0, 64.0, 128.0)
UNIGRAM_WEIGHTS = (0.03, 0.05, 0.07)
BIGRAM_WEIGHTS = (0.2, 0.25, 0.3)


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
    base_rows = []
    probs = {order: {decay: [] for decay in DECAYS} for order in (1, 2)}
    active_rows = {order: [] for order in (1, 2)}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            batch, length = x.shape
            base = F.softmax(model(x).float() / 1.15, -1).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            base_rows.append(base[valid].numpy())
            positions = torch.arange(length, device=device)
            distance = positions[:, None] - positions[None, :]
            earlier = distance > 0
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            match1 = (x[:, :, None] == x[:, None, :]) & earlier
            preceding = torch.cat((torch.full((batch, 1), -1, dtype=x.dtype), x[:, :-1]), 1)
            match2 = match1 & (preceding[:, :, None] == preceding[:, None, :]) & (
                positions[:, None] >= 1) & (positions[None, :] >= 1)
            for order, match in ((1, match1), (2, match2)):
                active_rows[order].append((match.sum(-1) > 0)[valid].numpy())
                for decay in DECAYS:
                    weight = (torch.ones_like(distance, dtype=torch.float32)
                              if decay is None else torch.exp(-distance.float() / decay))
                    weighted = match.float() * weight
                    total = weighted.sum(-1)
                    q = (weighted * correct).sum(-1) / total.clamp_min(1)
                    probs[order][decay].append(q[valid].numpy())
    base = np.concatenate(base_rows).astype(np.float64)
    active = {order: np.concatenate(active_rows[order]).astype(np.float64)
              for order in (1, 2)}
    q = {order: {decay: np.concatenate(probs[order][decay]).astype(np.float64)
                 for decay in DECAYS} for order in (1, 2)}
    results = []
    for du in DECAYS:
        for db in DECAYS:
            for wu in UNIGRAM_WEIGHTS:
                for wb in BIGRAM_WEIGHTS:
                    p = ((1 - wu * active[1] - wb * active[2]) * base
                         + wu * active[1] * q[1][du]
                         + wb * active[2] * q[2][db])
                    bpb = -np.log(p).sum() / math.log(2) / byte_count
                    results.append((bpb, du, db, wu, wb))
    for bpb, du, db, wu, wb in sorted(results)[:25]:
        print(json.dumps({'bpb': bpb, 'unigram_decay': du,
                          'bigram_decay': db, 'unigram_weight': wu,
                          'bigram_weight': wb}), flush=True)
    base_result = next(r for r in results if r[1:] == (None, None, 0.05, 0.25))
    print(json.dumps({'unweighted_reference_bpb': base_result[0]}), flush=True)


if __name__ == '__main__':
    main()
