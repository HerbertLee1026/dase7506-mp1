"""Causal window cache with compact train-only 3/4/5-gram distributions."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import sha
from student_cache import CacheGPT


class VariableNgramGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        self.vocab = config['vocab']
        self.base = CacheGPT(config)
        self.weights = (
            float(config.get('train_trigram_weight', 0.1)),
            float(config.get('train_fourgram_weight', 0.05)),
            float(config.get('train_fivegram_weight', 0.08)),
        )
        self.max_branches = int(config.get('train_ngram_max_branches', 128))
        self.later_max_branches = int(config.get(
            'train_later_ngram_max_branches', self.max_branches))
        if any(weight < 0 for weight in self.weights) or sum(self.weights) >= 1:
            raise ValueError('Ngram weights must be nonnegative and sum below one.')
        asset = Path(__file__).resolve().parent / config['ngram_asset']
        if sha(asset) != config['ngram_asset_sha256']:
            raise ValueError('Training-derived ngram asset hash mismatch.')
        with np.load(asset) as saved:
            for name in saved.files:
                value = torch.as_tensor(saved[name], dtype=(
                    torch.long if name.endswith(('keys', 'offsets', 'tokens'))
                    else torch.int32))
                self.register_buffer(name, value, persistent=False)

    def forward(self, ids):
        return self.base(ids)

    def _ngram_entries(self, ids, order, prefix):
        batch, length = ids.shape
        first = order - 2
        if length <= first:
            empty = torch.empty(0, device=ids.device, dtype=torch.long)
            return empty, empty, empty.float(), torch.zeros(
                (batch, length), device=ids.device, dtype=torch.bool)
        context = torch.zeros_like(ids)
        for offset in range(order - 1):
            context[:, first:] = (
                context[:, first:] * self.vocab
                + ids[:, offset:offset + length - first]
            )
        queries = context.reshape(-1).contiguous()
        keys = getattr(self, prefix + 'context_keys')
        offsets = getattr(self, prefix + 'context_offsets')
        totals = getattr(self, prefix + 'context_totals')
        next_tokens = getattr(self, prefix + 'next_tokens')
        counts = getattr(self, 'trigram_counts' if order == 3 else prefix + 'counts')
        positions = torch.searchsorted(keys, queries)
        valid = positions < keys.numel()
        positions = positions.clamp_max(keys.numel() - 1)
        valid &= keys[positions] == queries
        valid = valid.view(batch, length)
        valid[:, :first] = False
        valid = valid.reshape(-1)
        starts = offsets[positions]
        ends = offsets[positions + 1]
        valid &= (ends - starts) <= (
            self.max_branches if order == 3 else self.later_max_branches)
        sizes = torch.where(valid, ends - starts, 0)
        rows = torch.repeat_interleave(torch.arange(queries.numel(), device=ids.device), sizes)
        prefix_sizes = torch.cumsum(sizes, 0) - sizes
        entries = starts[rows] + torch.arange(rows.numel(), device=ids.device) - (
            torch.repeat_interleave(prefix_sizes, sizes)
        )
        values = counts[entries].float() / totals[positions][rows].clamp_min(1)
        return rows, next_tokens[entries], values, valid.view(batch, length)

    def predict_log_probs(self, ids):
        base_prob = self.base.predict_probs(ids)
        mixture = base_prob.clone()
        base_for_later = None
        for order, prefix, weight in zip((3, 4, 5), ('', 'four_', 'five_'), self.weights):
            if order == 4:
                base_for_later = mixture.clone()
            if weight == 0:
                continue
            rows, next_tokens, values, valid = self._ngram_entries(ids, order, prefix)
            reference = base_prob if order == 3 else base_for_later
            mixture -= weight * valid.unsqueeze(-1) * reference
            mixture.reshape(-1).scatter_add_(
                0, rows * self.vocab + next_tokens, weight * values)
        return (mixture / mixture.sum(-1, keepdim=True)).log()


def build_model(config):
    return VariableNgramGPT(config)
