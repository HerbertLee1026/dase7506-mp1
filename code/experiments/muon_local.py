"""FP32 single-device Muon for hidden matrices; embeddings/biases use AdamW.

Adapted from Keller Jordan's Muon (accessed 2026-09-29):
https://github.com/KellerJordan/Muon/blob/master/muon.py
Changes: FP32 Newton-Schulz for MPS compatibility, no distributed operations,
and skip absent gradients. See MUON_LICENSE.txt for the upstream MIT license.
"""

import torch


def orthogonal_update(gradient, steps=5):
    if gradient.ndim != 2:
        raise ValueError('Muon requires a two-dimensional hidden weight.')
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        correction = -4.7750 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + correction @ x
    if transposed:
        x = x.T
    return x * max(1.0, gradient.shape[0] / gradient.shape[1]) ** 0.5


class SingleDeviceMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.002, momentum=0.95, weight_decay=0.01):
        if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError('Invalid optimizer parameters.')
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                     weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError('Route embeddings and nonmatrix parameters to AdamW.')
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                momentum = state['momentum_buffer']
                momentum.lerp_(p.grad, 1 - group['momentum'])
                update = orthogonal_update(p.grad.lerp(momentum, group['momentum']))
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
        return loss
