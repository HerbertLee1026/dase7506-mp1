"""Validation probe of skipping expensive high-branching trigram contexts."""

import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/cache_unigram_bigram_w192_s17/checkpoint.pt')
VOCAB = 2048
MAX_BRANCHES = (8, 16, 32, 64, 128, 256, 512, 100000)


def main():
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64)
    bigram = np.bincount(train[:-1] * VOCAB + train[1:],
                         minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
    bigram_total = bigram.sum(axis=-1)
    triples = (train[:-2] * VOCAB + train[1:-1]) * VOCAB + train[2:]
    tri_keys, tri_counts = np.unique(triples, return_counts=True)
    context_keys, starts, branches = np.unique(
        tri_keys // VOCAB, return_index=True, return_counts=True
    )
    context_totals = np.add.reduceat(tri_counts, starts)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = data['validation']
    scores = {limit: 0.0 for limit in MAX_BRANCHES}
    entries = {limit: 0 for limit in MAX_BRANCHES}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            valid = (y != -100).numpy()
            base = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1).exp().numpy()
            xx, yy = x.numpy(), y.clamp_min(0).numpy()
            big_total = bigram_total[xx]
            big_cond = bigram[xx, yy] / np.maximum(big_total, 1)
            wa = 0.06 * big_total / np.maximum(big_total + 100, 1)
            context = np.zeros_like(xx)
            context[:, 1:] = xx[:, :-1] * VOCAB + xx[:, 1:]
            queries = context * VOCAB + yy
            pos = np.searchsorted(tri_keys, queries)
            match = (pos < len(tri_keys)) & (tri_keys[np.minimum(pos, len(tri_keys)-1)] == queries)
            found_counts = np.zeros_like(base)
            found_counts[match] = tri_counts[pos[match]]
            cpos = np.searchsorted(context_keys, context)
            cmatch = (cpos < len(context_keys)) & (context_keys[np.minimum(cpos, len(context_keys)-1)] == context)
            total = np.zeros_like(base)
            total[cmatch] = context_totals[cpos[cmatch]]
            branch = np.zeros_like(base)
            branch[cmatch] = branches[cpos[cmatch]]
            total[:, 0] = 0
            tri_cond = found_counts / np.maximum(total, 1)
            for limit in scores:
                wb = 0.1 * ((total > 0) & (branch <= limit))
                mix = (1 - wa - wb) * base + wa * big_cond + wb * tri_cond
                scores[limit] += -np.log(mix[valid]).sum()
                entries[limit] += branch[(total > 0) & (branch <= limit)].sum()

    for limit, nll in scores.items():
        print('max_branches', limit, 'bpb', nll / math.log(2) / byte_count,
              'entries', entries[limit])


if __name__ == '__main__':
    main()
