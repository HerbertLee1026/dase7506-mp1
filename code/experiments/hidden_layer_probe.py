"""Validation-only search for the best causal hidden-copy layer."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import load_data, make_model, setup, windows


ALPHAS = (0.0, 0.01, 0.02, 0.03)
BETAS = (5.0, 10.0, 20.0)
DIMS = (128, 320)


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
    selected_layers = (1, 2, 3, 4, 5, 6, 7)
    if len(neural.blocks) != 7:
        raise ValueError('This probe expects a seven-layer model.')
    validation, byte_count = load_data()['validation']
    scores = {(layer, dim, alpha, beta): 0.0
              for layer in selected_layers for dim in DIMS
              for alpha in ALPHAS for beta in BETAS}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            valid = y != -100
            base = model.predict_log_probs(x).gather(
                -1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
            batch, length = x.shape
            positions = torch.arange(length, device=device)
            earlier = positions[None, :] < positions[:, None]
            following = torch.cat((x[:, 1:], x[:, -1:]), dim=1)
            correct = following[:, None, :] == y[:, :, None]
            active = (positions > 0)[None, :]
            h = neural.embedding_dropout(neural.token(x))
            for layer, block in enumerate(neural.blocks, start=1):
                h = block(h)
                if layer not in selected_layers:
                    continue
                source = neural.norm(h) if layer == 7 else h
                for dim in DIMS:
                    unit = F.normalize(source[..., :dim], dim=-1)
                    sim = torch.matmul(unit, unit.transpose(1, 2))
                    for beta in BETAS:
                        weights = F.softmax(sim.masked_fill(~earlier, -1e9) * beta, -1)
                        copy = (weights * earlier * correct).sum(-1)
                        for alpha in ALPHAS:
                            mix = torch.where(active, (1 - alpha) * base + alpha * copy, base)
                            scores[layer, dim, alpha, beta] += -mix[valid].double().log().sum().item()
    for (layer, dim, alpha, beta), nll in sorted(scores.items(), key=lambda row: row[1])[:30]:
        print(json.dumps({'layer': layer, 'dim': dim, 'alpha': alpha,
                          'beta': beta, 'bpb': nll / math.log(2) / byte_count}), flush=True)


if __name__ == '__main__':
    main()
