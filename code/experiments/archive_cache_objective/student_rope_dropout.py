"""RoPE GPT with residual dropout for small-data training."""

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT
from student_rope import RoPEBlock, SwiGLU, rotate_pairs


class DropoutRoPEBlock(RoPEBlock):
    def __init__(self, width, heads, context, dropout, attention_dropout=0.0):
        super().__init__(width, heads, context)
        self.mlp = SwiGLU(width)
        self.residual_dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        cos = self.cos[:length].to(dtype=q.dtype)[None, None]
        sin = self.sin[:length].to(dtype=q.dtype)[None, None]
        q = q * cos + rotate_pairs(q) * sin
        k = k * cos + rotate_pairs(k) * sin
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0)
        x = x + self.residual_dropout(
            self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        )
        return x + self.residual_dropout(self.mlp(self.norm2(x)))


class DropoutRoPEGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        width = config['width']
        dropout = float(config.get('dropout', 0.1))
        attention_dropout = float(config.get('attention_dropout', 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1).')
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError('attention_dropout must be in [0, 1).')
        self.token = nn.Embedding(config['vocab'], width)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            DropoutRoPEBlock(width, config['heads'], self.context, dropout, attention_dropout)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(GPT.initialize)
        self.head.weight = self.token.weight

    def forward(self, ids):
        x = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return DropoutRoPEGPT(config)
