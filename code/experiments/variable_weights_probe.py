"""Tune compact train ngram interpolation on validation target probabilities."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


VOCAB = 2048


def table(saved, prefix, count_name):
    contexts = saved[prefix + 'context_keys'].astype(np.int64)
    offsets = saved[prefix + 'context_offsets'].astype(np.int64)
    successor = saved[prefix + 'next_tokens'].astype(np.int64)
    full_keys = np.repeat(contexts, np.diff(offsets)) * VOCAB + successor
    return contexts, saved[prefix + 'context_totals'], full_keys, saved[count_name]


def target_probs(x, y, order, parts):
    contexts, totals, full_keys, counts = parts
    batch, length = x.shape
    context = np.zeros_like(x, dtype=np.int64)
    first = order - 2
    for offset in range(order - 1):
        context[:, first:] = context[:, first:] * VOCAB + x[:, offset:offset + length - first]
    cpos = np.searchsorted(contexts, context)
    found = (cpos < len(contexts)) & (
        contexts[np.minimum(cpos, len(contexts) - 1)] == context)
    found[:, :first] = False
    total = np.where(found, totals[np.minimum(cpos, len(contexts) - 1)], 0)
    query = context * VOCAB + y
    spos = np.searchsorted(full_keys, query)
    matched = (spos < len(full_keys)) & (
        full_keys[np.minimum(spos, len(full_keys) - 1)] == query)
    n = np.where(matched, counts[np.minimum(spos, len(full_keys) - 1)], 0)
    return n / np.maximum(total, 1), found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-checkpoint', required=True, type=Path)
    parser.add_argument('--asset', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(args.cache_checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    with np.load(args.asset) as source:
        tables = [table(source, prefix, count_name) for prefix, count_name in (
            ('', 'trigram_counts'), ('four_', 'four_counts'), ('five_', 'five_counts'))]
    validation, utf8_bytes = load_data()['validation']
    all_values = [[] for _ in range(7)]
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = (y != -100).numpy()
            prob = model.predict_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).numpy()
            all_values[0].append(prob[valid])
            xx, yy = x.numpy().astype(np.int64), y.clamp_min(0).numpy().astype(np.int64)
            for order, parts, qi in zip((3, 4, 5), tables, (1, 2, 3)):
                q, active = target_probs(xx, yy, order, parts)
                all_values[qi].append(q[valid])
                all_values[qi + 3].append(active[valid])
    base, q3, q4, q5, a3, a4, a5 = [np.concatenate(rows).astype(np.float64)
                                      for rows in all_values]
    results = []
    for w3 in (0.05, 0.075, 0.1, 0.125, 0.15):
        tri = base + w3 * a3 * (q3 - base)
        for w4 in (0.0, 0.025, 0.05, 0.075, 0.1):
            for w5 in (0.0, 0.05, 0.08, 0.1, 0.12, 0.15):
                p = tri * (1 - w4 * a4 - w5 * a5) + w4 * a4 * q4 + w5 * a5 * q5
                bpb = -np.log(p).sum() / math.log(2) / utf8_bytes
                results.append((bpb, w3, w4, w5))
    for bpb, w3, w4, w5 in sorted(results)[:20]:
        print(json.dumps({'bpb': bpb, 'trigram': w3, 'fourgram': w4,
                          'fivegram': w5}), flush=True)


if __name__ == '__main__':
    main()
