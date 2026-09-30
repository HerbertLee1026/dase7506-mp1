"""Temporary validation-only check of count-table interpolation."""

import numpy as np
import torch

from common import load_data, make_model, setup, windows
from experiments.variable_ngram_probe import counts_for_order, target_prob


def load(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model, _ = make_model(saved['implementation'], saved['config'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    return model


def main():
    device, _ = setup('cpu', 'fp32', 4)
    base = load('runs/cache_ngram_t115_dropout_p020_d6_w320_6000_s17/checkpoint.pt', device)
    extended = load('runs/variable_ngram_full45_t115_dropout_p020_d6_w320_6000_s17/checkpoint.pt', device)
    train = load_data()['train'][0].numpy().astype(np.int64)
    x, y = next(windows(load_data()['validation'][0], batch_size=32))
    valid = y != -100
    with torch.no_grad():
        p0 = base.predict_log_probs(x).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
        p1 = extended.predict_log_probs(x).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp().numpy()
    q4, c4 = target_prob(x.numpy().astype(np.int64), y.clamp_min(0).numpy().astype(np.int64), 4,
                         counts_for_order(train, 4))
    q5, c5 = target_prob(x.numpy().astype(np.int64), y.clamp_min(0).numpy().astype(np.int64), 5,
                         counts_for_order(train, 5))
    m4, m5 = c4 >= 2, c5 >= 2
    expected = (1 - 0.05 * m4 - 0.08 * m5) * p0 + 0.05 * m4 * q4 + 0.08 * m5 * q5
    delta = np.abs(expected[valid.numpy()] - p1[valid.numpy()])
    print('max_delta', delta.max(), 'mean_delta', delta.mean(),
          'expected_nll', -np.log(expected[valid.numpy()]).sum(),
          'actual_nll', -np.log(p1[valid.numpy()]).sum())
    locations = np.argwhere(np.abs(expected - p1) > 0.0001)
    for batch, time in locations[:10]:
        print(batch, time, 'target', int(y[batch, time]),
              'base', p0[batch, time], 'new', p1[batch, time],
              'expected', expected[batch, time], 'q4', q4[batch, time],
              'count4', c4[batch, time], 'q5', q5[batch, time],
              'count5', c5[batch, time])


if __name__ == '__main__':
    main()
