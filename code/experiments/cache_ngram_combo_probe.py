"""Validation-only grid for training bigram and trigram probabilities."""

import math
import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/cache_unigram_bigram_w192_s17/checkpoint.pt')
VOCAB = 2048
BIGRAM_WEIGHTS = (0.0, 0.03, 0.06, 0.09, 0.12)
TRIGRAM_WEIGHTS = (0.0, 0.05, 0.1, 0.15, 0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train = data['train'][0].numpy().astype(np.int64)
    bigram = np.bincount(train[:-1] * VOCAB + train[1:],
                         minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
    bigram_total = bigram.sum(axis=-1)
    contexts = train[:-2] * VOCAB + train[1:-1]
    tri_keys, tri_counts = np.unique(contexts * VOCAB + train[2:], return_counts=True)
    context_keys, context_counts = np.unique(contexts, return_counts=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = data['validation']
    scores = {(a, b): 0.0 for a in BIGRAM_WEIGHTS for b in TRIGRAM_WEIGHTS}

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
            big_gate = big_total / np.maximum(big_total + 100, 1)
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
            total[cmatch] = context_counts[cpos[cmatch]]
            total[:, 0] = 0
            tri_cond = found_counts / np.maximum(total, 1)
            tri_gate = (total > 0).astype(np.float32)
            for a, b in scores:
                wa, wb = a * big_gate, b * tri_gate
                mix = (1 - wa - wb) * base + wa * big_cond + wb * tri_cond
                scores[a, b] += -np.log(mix[valid]).sum()

    for (a, b), nll in sorted(scores.items(), key=lambda item: item[1])[:20]:
        print('bigram', a, 'trigram', b,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
