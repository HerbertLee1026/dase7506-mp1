"""Window-reset, unidirectional LSTM language model with tied embeddings."""

import torch
from torch import nn
from torch.nn import functional as F

from model import GPT


class LockedDropout(nn.Module):
    """Use one dropout mask per example and feature across a full window."""

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        mask = x.new_empty(x.shape[0], 1, x.shape[2]).bernoulli_(
            1 - self.probability
        ) / (1 - self.probability)
        return x * mask


class WindowLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        width = config['width']
        depth = config['depth']
        dropout = float(config.get('dropout', 0.2))
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1).')
        self.token = nn.Embedding(config['vocab'], width)
        self.embedding_dropout = LockedDropout(dropout)
        self.lstm = nn.LSTM(
            input_size=width, hidden_size=width, num_layers=depth,
            batch_first=True, dropout=dropout if depth > 1 else 0.0,
            bidirectional=False,
        )
        self.output_dropout = LockedDropout(dropout)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.token.apply(GPT.initialize)
        self.head.weight = self.token.weight

    def forward(self, ids):
        embedded = self.embedding_dropout(self.token(ids))
        hidden, _ = self.lstm(embedded)
        return self.head(self.norm(self.output_dropout(hidden)))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return WindowLSTM(config)
