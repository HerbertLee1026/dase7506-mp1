"""Tune logit temperature and causal window-cache weights on validation only."""

import argparse
import math
from pathlib import Path

import torch

from common import load_data, make_model, setup, windows


TEMPERATURES = (1.0, 1.1, 1.15, 1.2, 1.25, 1.3)
UNIGRAM_WEIGHTS = (0.0, 0.05, 0.1, 0.15)
BIGRAM_WEIGHTS = (0.15, 0.2, 0.25, 0.3, 0.35)


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
    scores = {(t, a, b): 0.0 for t in TEMPERATURES for a in UNIGRAM_WEIGHTS
              for b in BIGRAM_WEIGHTS if a + b < 1}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            valid = y != -100
            batch, length = x.shape
            logits = model(x).float()
            target_logit = logits.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            match1 = (x[:, :, None] == x[:, None, :]) & earlier
            preceding = torch.cat((torch.full((batch, 1), -1, device=device), x[:, :-1]), dim=1)
            match2 = match1 & (preceding[:, :, None] == preceding[:, None, :]) & (
                positions[:, None] >= 1
            ) & (positions[None, :] >= 1)
            count1, count2 = match1.sum(-1), match2.sum(-1)
            copy1 = (match1 & correct).sum(-1).float() / count1.clamp_min(1)
            copy2 = (match2 & correct).sum(-1).float() / count2.clamp_min(1)
            active1, active2 = count1 > 0, count2 > 0
            for temperature in TEMPERATURES:
                base_prob = (target_logit / temperature -
                             torch.logsumexp(logits / temperature, dim=-1)).exp()
                for a in UNIGRAM_WEIGHTS:
                    wa = a * active1
                    for b in BIGRAM_WEIGHTS:
                        if (temperature, a, b) not in scores:
                            continue
                        wb = b * active2
                        mix = (1 - wa - wb) * base_prob + wa * copy1 + wb * copy2
                        scores[temperature, a, b] += -mix[valid].double().log().sum().item()

    for (temperature, a, b), nll in sorted(scores.items(), key=lambda item: item[1])[:25]:
        print('temperature', temperature, 'unigram', a, 'bigram', b,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
