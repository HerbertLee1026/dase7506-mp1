"""Choose a scalar logit temperature using only validation targets."""

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import load_data, make_model, setup, windows


TEMPERATURES = (0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5)


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
    nll = {t: 0.0 for t in TEMPERATURES}
    with torch.no_grad():
        for x, y in windows(validation, batch_size=32):
            logits = model(x.to(device)).float()
            valid = (y != -100).to(device)
            target = y.clamp_min(0).to(device)
            for temperature in TEMPERATURES:
                logp = F.log_softmax(logits / temperature, dim=-1)
                nll[temperature] += -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[valid].double().sum().item()
    for temperature, value in sorted(nll.items(), key=lambda item: item[1]):
        print('temperature', temperature, 'bpb', value / math.log(2) / byte_count)


if __name__ == '__main__':
    main()
