"""RoPE GPT with causal within-window unigram and bigram copy distributions."""

import importlib

import torch
from torch import nn
from torch.nn import functional as F

class CacheGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context = config['context']
        base_implementation = config.get('base_implementation', 'student_rope')
        if base_implementation not in ('student_rope', 'student_rope_dropout',
                                       'student_rope_dropout_headbias',
                                       'student_rope_dropout_untied',
                                       'student_rope_mos', 'student_lstm'):
            raise ValueError('Unsupported cache base implementation.')
        self.base = importlib.import_module(base_implementation).build_model(config)
        self.unigram_weight = float(config.get('cache_unigram_weight', 0.10))
        self.bigram_weight = float(config.get('cache_bigram_weight', 0.25))
        self.base_temperature = float(config.get('base_temperature', 1.0))
        if self.base_temperature <= 0:
            raise ValueError('Base temperature must be positive.')
        if self.unigram_weight < 0 or self.bigram_weight < 0:
            raise ValueError('Cache mixture weights must be nonnegative.')
        if self.unigram_weight + self.bigram_weight >= 1:
            raise ValueError('Cache mixture weights must sum to less than one.')

    def forward(self, ids):
        return self.base(ids)

    def predict_probs(self, ids):
        # Position t may copy only tokens observed at positions <= t.
        base_prob = F.softmax(self.base(ids).float() / self.base_temperature, dim=-1)
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device)
        earlier = positions[None, :] < positions[:, None]
        following = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        scatter_index = following[:, None, :].expand(batch, length, length)

        match1 = (ids[:, :, None] == ids[:, None, :]) & earlier
        preceding = torch.cat((
            torch.full((batch, 1), -1, device=ids.device, dtype=ids.dtype),
            ids[:, :-1],
        ), dim=1)
        match2 = match1 & (preceding[:, :, None] == preceding[:, None, :]) & (
            positions[:, None] >= 1
        ) & (positions[None, :] >= 1)

        count1 = match1.sum(-1)
        count2 = match2.sum(-1)
        copy1 = torch.zeros_like(base_prob).scatter_add_(
            -1, scatter_index, match1.to(base_prob.dtype)
        ) / count1.clamp_min(1).unsqueeze(-1)
        copy2 = torch.zeros_like(base_prob).scatter_add_(
            -1, scatter_index, match2.to(base_prob.dtype)
        ) / count2.clamp_min(1).unsqueeze(-1)
        weight1 = self.unigram_weight * (count1 > 0)
        weight2 = self.bigram_weight * (count2 > 0)
        mixture = (
            (1 - weight1 - weight2).unsqueeze(-1) * base_prob
            + weight1.unsqueeze(-1) * copy1
            + weight2.unsqueeze(-1) * copy2
        )
        return mixture

    def predict_log_probs(self, ids):
        return self.predict_probs(ids).log()


def build_model(config):
    return CacheGPT(config)
