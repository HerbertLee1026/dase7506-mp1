"""Probe train-derived trigram counts with the current cached model on validation."""

import math
import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


DEFAULT_CHECKPOINT = Path('runs/cache_unigram_bigram_w192_s17/checkpoint.pt')
VOCAB = 2048
ALPHAS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)
GATES = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, choices=(3, 4, 5), default=3)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train_ids = data['train'][0].numpy().astype(np.int64)
    contexts = np.zeros(len(train_ids) - args.order + 1, dtype=np.int64)
    for offset in range(args.order - 1):
        contexts = contexts * VOCAB + train_ids[offset:offset + len(contexts)]
    triples = contexts * VOCAB + train_ids[args.order - 1:]
    keys, counts = np.unique(triples, return_counts=True)
    contexts_unique, context_counts = np.unique(contexts, return_counts=True)
    print('unique_triples', len(keys), 'unique_contexts', len(contexts_unique), flush=True)

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = data['validation']
    scores = {(a, k): 0.0 for a in ALPHAS for k in GATES}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            valid = y != -100
            base_prob = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1).exp().numpy()
            x_np = x.numpy()
            y_np = y.clamp_min(0).numpy()
            context = np.zeros_like(x_np)
            for offset in range(args.order - 1):
                context[:, args.order - 2:] = (
                    context[:, args.order - 2:] * VOCAB
                    + x_np[:, offset:offset + x_np.shape[1] - (args.order - 2)]
                )
            queries = context * VOCAB + y_np
            positions = np.searchsorted(keys, queries)
            matched = (positions < len(keys)) & (keys[np.minimum(positions, len(keys)-1)] == queries)
            triple_counts = np.zeros_like(queries, dtype=np.float64)
            triple_counts[matched] = counts[positions[matched]]
            total_positions = np.searchsorted(contexts_unique, context)
            total_matched = ((total_positions < len(contexts_unique)) &
                             (contexts_unique[np.minimum(total_positions, len(contexts_unique)-1)] == context))
            totals = np.zeros_like(queries, dtype=np.float64)
            totals[total_matched] = context_counts[total_positions[total_matched]]
            conditional = triple_counts / np.maximum(totals, 1)
            totals[:, :args.order - 2] = 0
            for a, k in scores:
                weight = a * totals / np.maximum(totals + k, 1)
                mix = (1 - weight) * base_prob + weight * conditional
                scores[a, k] += -np.log(mix[valid.numpy()]).sum()

    for (a, k), nll in sorted(scores.items(), key=lambda item: item[1])[:20]:
        print('alpha', a, 'gate_k', k,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
