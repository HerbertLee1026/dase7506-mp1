"""Dropout RoPE language model with a two-component softmax mixture head."""

import torch
from torch import nn
from torch.nn import functional as F

from student_rope_dropout import DropoutRoPEGPT


class MixtureRoPEGPT(DropoutRoPEGPT):
    def __init__(self, config):
        super().__init__(config)
        width = config['width']
        self.expert_proj = nn.Linear(width, width, bias=False)
        self.mixture_gate = nn.Linear(width, 2)
        with torch.no_grad():
            nn.init.eye_(self.expert_proj.weight)
            self.expert_proj.weight.add_(0.01 * torch.randn_like(self.expert_proj.weight))
            nn.init.zeros_(self.mixture_gate.weight)
            nn.init.zeros_(self.mixture_gate.bias)

    def forward(self, ids):
        h = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        gate = F.log_softmax(self.mixture_gate(h).float(), dim=-1)
        logp0 = F.log_softmax(self.head(h).float(), dim=-1)
        logp1 = F.log_softmax(self.head(self.expert_proj(h)).float(), dim=-1)
        return torch.logaddexp(logp0 + gate[..., 0, None],
                               logp1 + gate[..., 1, None])

    def predict_log_probs(self, ids):
        return self(ids)


def build_model(config):
    return MixtureRoPEGPT(config)
