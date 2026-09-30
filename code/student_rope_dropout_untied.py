"""Regularized RoPE model with separate input and output token weights."""

from torch import nn

from model import GPT
from student_rope_dropout import DropoutRoPEGPT


class UntiedDropoutRoPEGPT(DropoutRoPEGPT):
    def __init__(self, config):
        super().__init__(config)
        self.head = nn.Linear(config['width'], config['vocab'], bias=True)
        self.head.apply(GPT.initialize)
        nn.init.zeros_(self.head.bias)


def build_model(config):
    return UntiedDropoutRoPEGPT(config)
