"""Validation-only probe of a train-derived six-token probability table."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


VOCAB = 2048
WEIGHTS = (0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2)


def contexts(ids):
    result = np.zeros(len(ids) - 5, dtype=np.int64)
    for offset in range(5):
        result = result * VOCAB + ids[offset:offset + len(result)]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64)
    train_context = contexts(train)
    context_keys, inverse, context_totals = np.unique(
        train_context, return_inverse=True, return_counts=True)
    successor_keys, successor_counts = np.unique(
        inverse.astype(np.int64) * VOCAB + train[5:], return_counts=True)
    print({'contexts': len(context_keys), 'successors': len(successor_keys),
           'repeated_contexts': int((context_totals >= 2).sum())}, flush=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = data['validation']
    scores = {(weight, minimum): 0.0 for weight in WEIGHTS for minimum in (1, 2, 3)}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = (y != -100).numpy()
            base = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
            xx, yy = x.numpy().astype(np.int64), y.clamp_min(0).numpy().astype(np.int64)
            batch, length = xx.shape
            query_context = np.zeros_like(xx, dtype=np.int64)
            for offset in range(5):
                query_context[:, 4:] = (query_context[:, 4:] * VOCAB
                                        + xx[:, offset:offset + length - 4])
            pos = np.searchsorted(context_keys, query_context)
            found = ((pos < len(context_keys))
                     & (context_keys[np.minimum(pos, len(context_keys)-1)] == query_context))
            found[:, :4] = False
            pos = np.minimum(pos, len(context_keys)-1)
            count = np.where(found, context_totals[pos], 0)
            query = pos.astype(np.int64) * VOCAB + yy
            kpos = np.searchsorted(successor_keys, query)
            matched = ((kpos < len(successor_keys))
                       & (successor_keys[np.minimum(kpos, len(successor_keys)-1)] == query))
            target_count = np.zeros_like(base)
            target_count[matched & found] = successor_counts[kpos[matched & found]]
            conditional = target_count / np.maximum(count, 1)
            for weight, minimum in scores:
                active = count >= minimum
                mixture = (1 - weight * active) * base + weight * active * conditional
                scores[weight, minimum] += -np.log(mixture[valid]).sum()
    for (weight, minimum), nll in sorted(scores.items(), key=lambda item: item[1])[:15]:
        print({'weight': weight, 'min_count': minimum,
               'bpb': nll / math.log(2) / byte_count})


if __name__ == '__main__':
    main()
