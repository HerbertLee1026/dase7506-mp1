"""Tune unigram and bigram within-window cache weights on validation only."""

import math
import argparse
from pathlib import Path

import torch

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/student_rope_d6_w192_3600_s17/checkpoint.pt')
WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    tokens, byte_count = load_data()['validation']
    scores = {(a, b): 0.0 for a in WEIGHTS for b in WEIGHTS if a + b <= 0.65}

    with torch.no_grad():
        for x, y in windows(tokens, batch_size=32):
            x, y = x.to(device), y.to(device)
            batch, length = x.shape
            valid = y != -100
            base_prob = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1).exp()

            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            match1 = (x[:, :, None] == x[:, None, :]) & earlier
            preceding = torch.cat((torch.full((batch, 1), -1, device=device), x[:, :-1]), dim=1)
            match2 = match1 & (preceding[:, :, None] == preceding[:, None, :]) & (
                positions[:, None] >= 1
            ) & (positions[None, :] >= 1)

            caches = []
            actives = []
            for match in (match1, match2):
                total = match.sum(-1)
                caches.append((match & correct).sum(-1).float() / total.clamp_min(1))
                actives.append(total > 0)

            for (a, b) in scores:
                wa = a * actives[0]
                wb = b * actives[1]
                mix = (1 - wa - wb) * base_prob + wa * caches[0] + wb * caches[1]
                scores[a, b] += -mix[valid].double().log().sum().item()

    for (a, b), nll in sorted(scores.items(), key=lambda item: item[1])[:25]:
        print('unigram', a, 'bigram', b,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
