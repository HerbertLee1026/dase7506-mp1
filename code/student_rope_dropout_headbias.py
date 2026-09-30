"""Regularized RoPE model with a learned token-specific output bias."""

from torch import nn

from student_rope_dropout import DropoutRoPEGPT


class BiasedDropoutRoPEGPT(DropoutRoPEGPT):
    def __init__(self, config):
        super().__init__(config)
        self.head = nn.Linear(config['width'], config['vocab'], bias=True)
        self.head.weight = self.token.weight
        nn.init.zeros_(self.head.bias)


def build_model(config):
    return BiasedDropoutRoPEGPT(config)
