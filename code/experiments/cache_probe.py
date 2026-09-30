"""Measure causal within-window n-gram cache mixtures on validation targets."""

import math
from pathlib import Path

import torch

from common import load_data, make_model, setup, windows


CHECKPOINT = Path('runs/student_rope_d6_w192_3600_s17/checkpoint.pt')
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
GRAMS = (1, 2, 3)
DECAYS = (None, 64.0)


def main():
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    data = load_data()
    tokens, byte_count = data['validation']
    total_nll = {(n, decay, alpha): 0.0 for n in GRAMS for decay in DECAYS for alpha in ALPHAS}
    base_nll = 0.0

    with torch.no_grad():
        for x, y in windows(tokens, batch_size=32):
            x, y = x.to(device), y.to(device)
            batch, length = x.shape
            valid = y != -100
            base_logp = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            base_prob = base_logp.exp()
            base_nll += -base_logp[valid].double().sum().item()

            positions = torch.arange(length, device=device)
            distances = positions[:, None] - positions[None, :]
            earlier = distances > 0
            next_observed = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            matches_target = next_observed[:, None, :] == y[:, :, None]

            matches_context = x[:, :, None] == x[:, None, :]
            for n in GRAMS:
                if n > 1:
                    shift = n - 1
                    previous = torch.cat((
                        torch.full((batch, shift), -1, device=device, dtype=x.dtype),
                        x[:, :-shift],
                    ), dim=1)
                    matches_context = matches_context & (
                        previous[:, :, None] == previous[:, None, :]
                    ) & (positions[:, None] >= shift) & (positions[None, :] >= shift)
                for decay in DECAYS:
                    distance_weight = (
                        torch.ones_like(distances, dtype=torch.float32)
                        if decay is None else torch.exp(-distances.float() / decay)
                    )
                    weights = matches_context.float() * earlier * distance_weight
                    total = weights.sum(dim=-1)
                    correct = (weights * matches_target).sum(dim=-1)
                    cache_prob = correct / total.clamp_min(1e-30)
                    active = total > 0
                    for alpha in ALPHAS:
                        mix = torch.where(
                            active,
                            (1 - alpha) * base_prob + alpha * cache_prob,
                            base_prob,
                        )
                        total_nll[n, decay, alpha] += -mix[valid].double().log().sum().item()

    print('base_bpb', base_nll / math.log(2) / byte_count, flush=True)
    for (n, decay, alpha), nll in sorted(total_nll.items(), key=lambda item: item[1]):
        print('ngram', n, 'decay', decay, 'alpha', alpha,
              'bpb', nll / math.log(2) / byte_count, flush=True)


if __name__ == '__main__':
    main()
