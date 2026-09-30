"""Trainable causal prefix pointer, inspired by Merity et al. (2016).

https://arxiv.org/abs/1609.07843 . This is a new Transformer adaptation;
no external weights or training data are used. All pointer state is per window.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from student_hidden_variable_ngram import HiddenVariableNgramGPT
from student_fast_hidden_ngram import FastHiddenCacheGPT


class PrefixPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, dim = config['width'], config.get('pointer_dim', 64)
        self.query = nn.Linear(width, dim, bias=False)
        self.key = nn.Linear(width, dim, bias=False)
        self.gate = nn.Linear(width, 1)
        self.maximum = float(config.get('pointer_maximum', 0.5))
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.0)))
        with torch.no_grad():
            identity = torch.eye(dim, width)
            self.query.weight.copy_(identity)
            self.key.weight.copy_(identity)
            self.gate.weight.zero_()
            self.gate.bias.fill_(math.log(0.02 / (self.maximum - 0.02)))

    def forward(self, hidden):
        q = F.normalize(self.query(hidden.float()), dim=-1)
        k = F.normalize(self.key(hidden.float()), dim=-1)
        length = hidden.shape[1]
        positions = torch.arange(length, device=hidden.device)
        earlier = positions[None, :] < positions[:, None]
        similarity = q @ k.transpose(-1, -2)
        scores = similarity * self.log_beta.exp().clamp_max(100)
        attention = F.softmax(scores.masked_fill(~earlier, -1e9), -1) * earlier
        gate = self.maximum * torch.sigmoid(self.gate(hidden.float())).squeeze(-1)
        gate = gate * (positions > 0)
        return attention, gate


class LearnedPointerGPT(HiddenVariableNgramGPT):
    def __init__(self, config):
        super().__init__(config)
        self.base = FastHiddenCacheGPT(config)
        self.pointer = PrefixPointer(config)

    def predict_log_probs(self, ids):
        mixture, hidden = self.base.predict_probs_with_hidden(ids)
        trigram, fourgram, fivegram = self.weights
        if trigram:
            rows, next_tokens, values, valid = self._ngram_entries(ids, 3, '')
            mixture *= 1 - trigram * valid.unsqueeze(-1)
            mixture.reshape(-1).scatter_add_(
                0, rows * self.vocab + next_tokens, trigram * values)
        later, scale = [], None
        for order, prefix, weight in ((4, 'four_', fourgram), (5, 'five_', fivegram)):
            if weight:
                rows, next_tokens, values, valid = self._ngram_entries(ids, order, prefix)
                contribution = weight * valid
                scale = contribution if scale is None else scale + contribution
                later.append((rows, next_tokens, values, weight))
        if scale is not None:
            mixture *= 1 - scale.unsqueeze(-1)
            for rows, next_tokens, values, weight in later:
                mixture.reshape(-1).scatter_add_(
                    0, rows * self.vocab + next_tokens, weight * values)
        attention, gate = self.pointer(hidden)
        batch, length = ids.shape
        following = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        mixture *= 1 - gate.unsqueeze(-1)
        mixture.scatter_add_(-1, following[:, None, :].expand(batch, length, length),
                             gate.unsqueeze(-1) * attention)
        return (mixture / mixture.sum(-1, keepdim=True)).log()


def build_model(config):
    return LearnedPointerGPT(config)
