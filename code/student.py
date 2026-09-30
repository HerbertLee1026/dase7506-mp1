"""GPT with a parameter-matched SwiGLU feed-forward layer."""

from torch import nn
from torch.nn import functional as F

from model import GPT


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Three projections at this width have about the same parameter count
        # as the baseline's two projections with a 4 * width hidden layer.
        hidden = (8 * width) // 3
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))


class StudentGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.blocks:
            block.mlp = SwiGLU(config['width'])
            block.mlp.apply(self.initialize)


def build_model(config):
    return StudentGPT(config)
