"""Probe train-only 4/5-gram interpolation on the full validation split."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


VOCAB = 2048
WEIGHTS = (0.0, 0.03, 0.05, 0.08, 0.1, 0.15)


def counts_for_order(ids, order):
    context = np.zeros(len(ids) - order + 1, dtype=np.int64)
    for offset in range(order - 1):
        context = context * VOCAB + ids[offset:offset + len(context)]
    keys, counts = np.unique(context * VOCAB + ids[order - 1:], return_counts=True)
    contexts, totals = np.unique(context, return_counts=True)
    return keys, counts, contexts, totals


def target_prob(x, y, order, table):
    keys, counts, contexts, totals = table
    batch, length = x.shape
    context = np.zeros_like(x, dtype=np.int64)
    first = order - 2
    for offset in range(order - 1):
        context[:, first:] = context[:, first:] * VOCAB + x[:, offset:offset + length - first]
    query = context * VOCAB + y
    pos = np.searchsorted(keys, query)
    hit = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == query)
    target_counts = np.zeros_like(query, dtype=np.float64)
    target_counts[hit] = counts[pos[hit]]
    pos = np.searchsorted(contexts, context)
    hit = (pos < len(contexts)) & (contexts[np.minimum(pos, len(contexts) - 1)] == context)
    context_totals = np.zeros_like(query, dtype=np.float64)
    context_totals[hit] = totals[pos[hit]]
    context_totals[:, :first] = 0
    return target_counts / np.maximum(context_totals, 1), context_totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train_ids = data['train'][0].numpy().astype(np.int64)
    tables = {order: counts_for_order(train_ids, order) for order in (4, 5)}
    for order, table in tables.items():
        print({'order': order, 'unique_ngrams': len(table[0]),
               'unique_contexts': len(table[2])}, flush=True)

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    base_all, q4_all, q5_all, m4_all, m5_all = [], [], [], [], []
    validation, byte_count = data['validation']
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            base = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
            x_np, y_np = x.numpy().astype(np.int64), y.clamp_min(0).numpy().astype(np.int64)
            q4, m4 = target_prob(x_np, y_np, 4, tables[4])
            q5, m5 = target_prob(x_np, y_np, 5, tables[5])
            v = valid.numpy()
            base_all.append(base[v])
            q4_all.append(q4[v])
            q5_all.append(q5[v])
            m4_all.append(m4[v])
            m5_all.append(m5[v])
    base, q4, q5, m4, m5 = [np.concatenate(a) for a in
                            (base_all, q4_all, q5_all, m4_all, m5_all)]
    results = []
    for min_count in (1, 2, 3):
        available4, available5 = m4 >= min_count, m5 >= min_count
        for a4 in WEIGHTS:
            for a5 in WEIGHTS:
                mixture = ((1 - a4 * available4 - a5 * available5) * base
                           + a4 * available4 * q4 + a5 * available5 * q5)
                bpb = -np.log(mixture).sum() / math.log(2) / byte_count
                results.append((bpb, min_count, a4, a5))
    for bpb, min_count, a4, a5 in sorted(results)[:15]:
        print({'bpb': bpb, 'min_count': min_count,
               'fourgram_weight': a4, 'fivegram_weight': a5})
    for min_count in (1, 2, 3):
        bpb, _, a4, a5 = min(row for row in results if row[1] == min_count)
        print({'best_at_min_count': min_count, 'bpb': bpb,
               'fourgram_weight': a4, 'fivegram_weight': a5})
    for gain in (-0.01, 0.0, 0.01, 0.02, 0.04):
        available4, available5 = m4 >= 2, m5 >= 2
        weight4 = np.clip(0.05 + gain * np.log2(np.maximum(m4, 2) / 2), 0, 0.3) * available4
        weight5 = np.clip(0.08 + gain * np.log2(np.maximum(m5, 2) / 2), 0, 0.3) * available5
        mixture = ((1 - weight4 - weight5) * base
                   + weight4 * q4 + weight5 * q5)
        bpb = -np.log(mixture).sum() / math.log(2) / byte_count
        print({'count_gain': gain, 'bpb': bpb})


if __name__ == '__main__':
    main()
