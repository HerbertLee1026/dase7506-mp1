"""GPT with SwiGLU feed-forward layers and rotary attention positions."""

import torch
from torch import nn
from torch.nn import functional as F

from model import Block, GPT


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = (8 * width) // 3
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.proj = nn.Linear(hidden, width)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.value(x))


def rotate_pairs(x):
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


class RoPEBlock(Block):
    def __init__(self, width, heads, context):
        super().__init__(width, heads)
        head_dim = width // heads
        if width % heads or head_dim % 2:
            raise ValueError('RoPE requires an even attention head dimension.')
        self.mlp = SwiGLU(width)
        frequencies = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        angles = torch.arange(context).float()[:, None] * frequencies[None, :]
        self.register_buffer('cos', angles.cos().repeat_interleave(2, dim=-1), persistent=False)
        self.register_buffer('sin', angles.sin().repeat_interleave(2, dim=-1), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        cos = self.cos[:length].to(dtype=q.dtype)[None, None]
        sin = self.sin[:length].to(dtype=q.dtype)[None, None]
        q = q * cos + rotate_pairs(q) * sin
        k = k * cos + rotate_pairs(k) * sin
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class RoPEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            RoPEBlock(width, config['heads'], self.context)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(GPT.initialize)
        self.head.weight = self.token.weight

    def forward(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return RoPEGPT(config)
