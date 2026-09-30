"""Probe a compact training-only bigram distribution on validation."""

import math
from pathlib import Path

import numpy as np
import torch

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/student_rope_d6_w192_3600_s17/checkpoint.pt')
ALPHAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)
GATE_STRENGTHS = (0.0, 10.0, 100.0, 1000.0)
VOCAB = 2048


def main():
    device, _ = setup('cpu', 'fp32', 4)
    data = load_data()
    train_ids = data['train'][0].numpy()
    transitions = train_ids[:-1] * VOCAB + train_ids[1:]
    counts = np.bincount(transitions, minlength=VOCAB * VOCAB).reshape(VOCAB, VOCAB)
    counts = torch.from_numpy(counts.astype(np.float32))
    totals = counts.sum(dim=-1)

    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = data['validation']
    scores = {(a, k): 0.0 for a in ALPHAS for k in GATE_STRENGTHS}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            valid = y != -100
            base_prob = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1).exp()
            context_count = totals[x]
            bigram_prob = counts[x, y.clamp_min(0)] / context_count.clamp_min(1)
            for a, k in scores:
                weight = a * context_count / (context_count + k).clamp_min(1)
                mix = (1 - weight) * base_prob + weight * bigram_prob
                scores[a, k] += -mix[valid].double().log().sum().item()

    for (a, k), nll in sorted(scores.items(), key=lambda item: item[1])[:20]:
        print('alpha', a, 'gate_k', k,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
