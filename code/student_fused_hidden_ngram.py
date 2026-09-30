"""Same predictor with SwiGLU gate/value projections computed in one linear call."""

from torch import nn
from torch.nn import functional as F

from student_hidden_variable_ngram import HiddenVariableNgramGPT


class FusedSwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = (8 * width) // 3
        self.gate_value = nn.Linear(width, 2 * hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.proj(F.silu(gate) * value)


class FusedHiddenNgramGPT(HiddenVariableNgramGPT):
    def __init__(self, config):
        super().__init__(config)
        for block in self.base.base.blocks:
            block.mlp = FusedSwiGLU(config['width'])


def build_model(config):
    return FusedHiddenNgramGPT(config)
