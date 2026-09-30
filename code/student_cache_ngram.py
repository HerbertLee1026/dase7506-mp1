"""RoPE model with causal window cache and training-only ngram counts."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import sha
from student_cache import CacheGPT


class CacheNgramGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.vocab = config['vocab']
        self.base = CacheGPT(config)
        self.bigram_weight = float(config.get('train_bigram_weight', 0.06))
        self.trigram_weight = float(config.get('train_trigram_weight', 0.10))
        self.trigram_max_branches = int(config.get('train_trigram_max_branches', 100000))
        if self.bigram_weight < 0 or self.trigram_weight < 0:
            raise ValueError('Ngram weights must be nonnegative.')
        if self.bigram_weight + self.trigram_weight >= 1:
            raise ValueError('Ngram weights must sum to less than one.')

        asset = Path(__file__).resolve().parent / config['ngram_asset']
        if sha(asset) != config['ngram_asset_sha256']:
            raise ValueError('Training-derived ngram asset hash mismatch.')
        with np.load(asset) as saved:
            arrays = {name: saved[name] for name in saved.files}
        for name, array in arrays.items():
            dtype = (torch.long if name in ('context_keys', 'context_offsets', 'next_tokens')
                     else torch.int32)
            self.register_buffer(name, torch.as_tensor(array, dtype=dtype), persistent=False)

    def forward(self, ids):
        return self.base(ids)

    def predict_log_probs(self, ids):
        base_prob = self.base.predict_log_probs(ids).exp()
        batch, length = ids.shape
        bigram_total = self.bigram_totals[ids].float()
        bigram_prob = self.bigram_counts[ids].float() / bigram_total.clamp_min(1).unsqueeze(-1)
        bigram_weight = self.bigram_weight * bigram_total / (bigram_total + 100).clamp_min(1)

        context = torch.zeros_like(ids)
        context[:, 1:] = ids[:, :-1] * self.vocab + ids[:, 1:]
        query = context.reshape(-1).contiguous()
        positions = torch.searchsorted(self.context_keys, query)
        valid = positions < self.context_keys.numel()
        positions = positions.clamp_max(self.context_keys.numel() - 1)
        valid &= self.context_keys[positions] == query
        valid = valid.view(batch, length)
        valid[:, 0] = False
        valid = valid.reshape(-1)
        starts = self.context_offsets[positions]
        ends = self.context_offsets[positions + 1]
        valid &= (ends - starts) <= self.trigram_max_branches
        lengths = torch.where(valid, ends - starts, 0)
        rows = torch.repeat_interleave(torch.arange(query.numel(), device=ids.device), lengths)
        prefix = torch.cumsum(lengths, 0) - lengths
        entry = starts[rows] + torch.arange(rows.numel(), device=ids.device) - torch.repeat_interleave(prefix, lengths)
        next_token = self.next_tokens[entry]
        next_count = self.trigram_counts[entry].float()
        trigram_prob = torch.zeros_like(base_prob).reshape(-1)
        trigram_prob.scatter_add_(0, rows * self.vocab + next_token, next_count)
        trigram_total = self.context_totals[positions].float()
        trigram_prob = trigram_prob.view(batch, length, self.vocab) / trigram_total.clamp_min(1).view(batch, length, 1)
        trigram_weight = self.trigram_weight * valid.view(batch, length)

        mixture = (
            (1 - bigram_weight - trigram_weight).unsqueeze(-1) * base_prob
            + bigram_weight.unsqueeze(-1) * bigram_prob
            + trigram_weight.unsqueeze(-1) * trigram_prob
        )
        return mixture.log()


def build_model(config):
    return CacheNgramGPT(config)
