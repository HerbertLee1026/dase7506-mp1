"""Validation-only probe of a causal hidden-state similarity cache."""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/cache_unigram_bigram_w192_s17/checkpoint.pt')
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
BETAS = (2.0, 5.0, 10.0, 20.0, 40.0)


def main():
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    validation, byte_count = load_data()['validation']
    scores = {(a, b): 0.0 for a in ALPHAS for b in BETAS}

    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            x, y = x.to(device), y.to(device)
            valid = y != -100
            base_prob = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1).exp()
            neural = model.base
            h = neural.token(x)
            for block in neural.blocks:
                h = block(h)
            h = F.normalize(neural.norm(h), dim=-1)
            sim = torch.matmul(h, h.transpose(1, 2))
            batch, length = x.shape
            pos = torch.arange(length, device=device)
            earlier = pos[None, :] < pos[:, None]
            next_token = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = next_token[:, None, :] == y[:, :, None]
            active = (pos > 0)[None, :]
            for beta in BETAS:
                weights = F.softmax(sim.masked_fill(~earlier, -1e9) * beta, dim=-1)
                weights = weights * earlier
                cache_prob = (weights * correct).sum(-1)
                for alpha in ALPHAS:
                    wa = alpha * active
                    mix = (1 - wa) * base_prob + wa * cache_prob
                    scores[alpha, beta] += -mix[valid].double().log().sum().item()

    for (alpha, beta), nll in sorted(scores.items(), key=lambda item: item[1])[:20]:
        print('alpha', alpha, 'beta', beta,
              'bpb', nll / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
