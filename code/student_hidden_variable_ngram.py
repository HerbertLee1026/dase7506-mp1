"""Causal hidden-state copy on top of the compact train-only ngram predictor."""

import torch
from torch import nn
from torch.nn import functional as F

from student_cache import CacheGPT
from student_rope_dropout import DropoutRoPEGPT
from student_variable_ngram import VariableNgramGPT


class HiddenRoPEGPT(DropoutRoPEGPT):
    def forward_with_hidden(self, ids):
        h = self.embedding_dropout(self.token(ids))
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h), h

    def forward(self, ids):
        return self.forward_with_hidden(ids)[0]


class HiddenCacheGPT(CacheGPT):
    def __init__(self, config):
        super().__init__(config)
        if config.get('base_implementation') != 'student_rope_dropout':
            raise ValueError('Hidden cache requires dropout RoPE base model.')
        self.base = HiddenRoPEGPT(config)

    def predict_probs_with_hidden(self, ids):
        logits, hidden = self.base.forward_with_hidden(ids)
        base_prob = F.softmax(logits.float() / self.base_temperature, dim=-1)
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
        mixture = ((1 - weight1 - weight2).unsqueeze(-1) * base_prob
                   + weight1.unsqueeze(-1) * copy1
                   + weight2.unsqueeze(-1) * copy2)
        return mixture, hidden


class HiddenVariableNgramGPT(VariableNgramGPT):
    def __init__(self, config):
        super().__init__(config)
        self.base = HiddenCacheGPT(config)
        self.hidden_weight = float(config.get('hidden_cache_weight', 0.02))
        self.hidden_beta = float(config.get('hidden_cache_beta', 10.0))
        self.hidden_dim = int(config.get('hidden_cache_dim', config['width']))
        if not 0 <= self.hidden_weight < 1 or self.hidden_beta <= 0:
            raise ValueError('Invalid hidden cache weight or beta.')
        if not 1 <= self.hidden_dim <= config['width']:
            raise ValueError('Invalid hidden cache dimension.')

    def predict_log_probs(self, ids):
        mixture, hidden = self.base.predict_probs_with_hidden(ids)
        trigram, fourgram, fivegram = self.weights
        if trigram:
            rows, next_tokens, values, valid = self._ngram_entries(ids, 3, '')
            mixture *= 1 - trigram * valid.unsqueeze(-1)
            mixture.reshape(-1).scatter_add_(
                0, rows * self.vocab + next_tokens, trigram * values)
        later = []
        scale = None
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
        if self.hidden_weight:
            h = F.normalize(hidden.float()[..., :self.hidden_dim], dim=-1)
            sim = torch.matmul(h, h.transpose(1, 2))
            batch, length = ids.shape
            positions = torch.arange(length, device=ids.device)
            earlier = positions[None, :] < positions[:, None]
            weights = F.softmax(sim.masked_fill(~earlier, -1e9) * self.hidden_beta, -1)
            weights = weights * earlier
            following = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
            active_weight = self.hidden_weight * (positions > 0)
            mixture *= (1 - active_weight)[None, :, None]
            mixture.scatter_add_(
                -1, following[:, None, :].expand(batch, length, length),
                self.hidden_weight * weights)
        return (mixture / mixture.sum(-1, keepdim=True)).log()


def build_model(config):
    return HiddenVariableNgramGPT(config)
