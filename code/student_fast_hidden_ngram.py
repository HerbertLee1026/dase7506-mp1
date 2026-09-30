"""Equivalent local-copy mixture using one vocabulary-sized scatter buffer."""

import torch
from torch.nn import functional as F
from student_hidden_variable_ngram import HiddenCacheGPT, HiddenVariableNgramGPT


class FastHiddenCacheGPT(HiddenCacheGPT):
    def predict_probs_with_hidden(self, ids):
        logits, hidden = self.base.forward_with_hidden(ids)
        probability = F.softmax(logits.float() / self.base_temperature, dim=-1)
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device)
        earlier = positions[None, :] < positions[:, None]
        following = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
        match1 = (ids[:, :, None] == ids[:, None, :]) & earlier
        preceding = torch.cat((torch.full_like(ids[:, :1], -1), ids[:, :-1]), dim=1)
        match2 = (match1 & (preceding[:, :, None] == preceding[:, None, :])
                  & (positions[:, None] >= 1) & (positions[None, :] >= 1))
        count1, count2 = match1.sum(-1), match2.sum(-1)
        weight1 = self.unigram_weight * (count1 > 0)
        weight2 = self.bigram_weight * (count2 > 0)
        copy = (match1 * (weight1 / count1.clamp_min(1)).unsqueeze(-1)
                + match2 * (weight2 / count2.clamp_min(1)).unsqueeze(-1))
        probability *= (1 - weight1 - weight2).unsqueeze(-1)
        probability.scatter_add_(-1, following[:, None, :].expand(batch, length, length), copy)
        return probability, hidden


class FastHiddenVariableNgramGPT(HiddenVariableNgramGPT):
    def __init__(self, config):
        super().__init__(config)
        self.base = FastHiddenCacheGPT(config)


def build_model(config):
    return FastHiddenVariableNgramGPT(config)
