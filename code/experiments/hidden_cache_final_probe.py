"""Validation-only hidden similarity cache on top of the full predictor."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import load_data, make_model, setup, windows


ALPHAS = (0.0, 0.01, 0.02, 0.03, 0.05)
BETAS = (5.0, 10.0, 20.0)
DIMS = (64, 128, 192, 320)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    neural = model.base.base
    validation, byte_count = load_data()['validation']
    scores = {(dim, alpha, beta): 0.0 for dim in DIMS
              for alpha in ALPHAS for beta in BETAS}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            base = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
            h = neural.embedding_dropout(neural.token(x))
            for block in neural.blocks:
                h = block(h)
            h = neural.norm(h)
            batch, length = x.shape
            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            active = (positions > 0)[None, :]
            for dim in DIMS:
                h_dim = F.normalize(h[..., :dim], dim=-1)
                sim = torch.matmul(h_dim, h_dim.transpose(1, 2))
                for beta in BETAS:
                    weights = F.softmax(sim.masked_fill(~earlier, -1e9) * beta, -1) * earlier
                    copy = (weights * correct).sum(-1)
                    for alpha in ALPHAS:
                        mix = torch.where(active, (1 - alpha) * base + alpha * copy, base)
                        scores[dim, alpha, beta] += -mix[valid].double().log().sum().item()
    for (dim, alpha, beta), nll in sorted(scores.items(), key=lambda row: row[1])[:30]:
        print(json.dumps({'dim': dim, 'alpha': alpha, 'beta': beta,
                          'bpb': nll / math.log(2) / byte_count}), flush=True)


if __name__ == '__main__':
    main()
