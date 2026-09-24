"""Permutation invariant entity networks; no information is read outside ``obs``.

Entities start with two type indicators and two relative position coordinates.
This common schema is an adapter convention, not an extra observation channel.
The centralized MLP, probability-floor mixture and bounded zero-mean direction
head retain the archived cooperative_navigation/models/__init__.py definitions;
entity encoding, relations and action readout replace its fixed-input backbone.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn


def tensor_observation(obs: Mapping, device=None) -> dict[str, torch.Tensor]:
    """Convert an adapter observation without altering its batch dimensions."""
    return {key: torch.as_tensor(value, device=device,
                               dtype=torch.bool if key == "mask" else torch.float32)
            for key, value in obs.items() if key in ("self", "entities", "mask")}


class RelationAttention(nn.Module):
    """A.5 residual attention on the robot's own authorized entity records.

    The declared edge set is the complete graph of valid observed nodes, with
    self-loops. ``r[j,k] = position[k] - position[j]`` uses only their already
    observed coordinates; it is not a communication edge or a critic feature.
    Pair-game records have no geometric relation and supply zero coordinates.
    Each head adds a relation score and a relation-dependent value message.
    """
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.relation_score = nn.Sequential(nn.Linear(2, hidden, bias=False), nn.Tanh(),
                                            nn.Linear(hidden, heads, bias=False))
        self.relation_value = nn.Linear(2, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())

    @staticmethod
    def relation_features(positions, valid):
        positions = torch.where(valid[..., None], positions, torch.zeros_like(positions))
        relations = positions[:, None, :, :] - positions[:, :, None, :]
        edges = valid[:, :, None] & valid[:, None, :]
        return torch.where(edges[..., None], relations, torch.zeros_like(relations)), edges

    def forward(self, tokens, positions, valid):
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        batch, nodes, hidden = tokens.shape
        relations, edges = self.relation_features(positions, valid)
        def split(projected):
            return projected.reshape(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        query, key, value = split(self.query(tokens)), split(self.key(tokens)), split(self.value(tokens))
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits + self.relation_score(relations).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~edges[:, None, :, :], float("-inf"))
        # Valid queries always retain their self-loop. Invalid queries must not
        # form an all-infinite softmax or retain a residual output.
        logits = torch.where(valid[:, None, :, None], logits, torch.zeros_like(logits))
        weights = torch.where(edges[:, None, :, :], logits.softmax(-1), torch.zeros_like(logits))
        relation_values = self.relation_value(relations).reshape(
            batch, nodes, nodes, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        messages = torch.matmul(weights, value) + (weights[..., None] * relation_values).sum(-2)
        messages = messages.transpose(1, 2).reshape(batch, nodes, hidden)
        result = tokens + self.update(torch.cat([tokens, messages], -1))
        return torch.where(valid[..., None], result, torch.zeros_like(result))


class TypedActionReadout(nn.Module):
    """Separate action-conditioned pooling for each authorized entity type."""
    def __init__(self, hidden, types=2):
        super().__init__()
        self.keys = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(types)])
        self.hidden = hidden

    def forward(self, queries, tokens, type_masks):
        outputs = []
        for projection, valid in zip(self.keys, type_masks):
            output = tokens.new_zeros((queries.shape[0], queries.shape[1], self.hidden))
            present = valid.any(-1)
            # Empty types are exactly zero and never enter an empty softmax.
            if bool(present.any()):
                values = torch.where(valid[present, :, None], tokens[present], torch.zeros_like(tokens[present]))
                logits = torch.matmul(queries[present], projection(values).transpose(-1, -2)) / math.sqrt(self.hidden)
                logits = logits.masked_fill(~valid[present, None, :], float("-inf"))
                output[present] = torch.matmul(logits.softmax(-1), values)
            outputs.append(output)
        return outputs


class EntityEncoder(nn.Module):
    """Shared entity encoding, optional relations, and action-dependent readout.

    A self token is always valid, including when there are no observed entities.
    Padding never enters the attention, pooling or entity counts. ``summary``
    averages raw records of each type before encoding; it cannot secretly retain
    the original individual entities through a nonlinear pooled embedding.
    """
    def __init__(self, self_dim, entity_dim, action_dim=5, hidden=128, heads=4,
                 relation_layers=2, mode="full"):
        super().__init__()
        if mode == "no_relation":
            mode, relation_layers = "full", 0
        if mode not in ("full", "summary", "nearest2"):
            raise ValueError(f"Unknown observation mode: {mode}")
        if hidden % heads or hidden <= 0 or heads <= 0:
            raise ValueError("hidden must be positive and divisible by heads")
        if entity_dim < 4 or self_dim < 1 or action_dim < 2 or relation_layers < 0:
            raise ValueError("Invalid entity network dimensions")
        self.self_dim, self.entity_dim = int(self_dim), int(entity_dim)
        self.action_dim, self.mode = int(action_dim), mode
        self.self_encoder = nn.Sequential(nn.Linear(self_dim, hidden), nn.Tanh(),
                                          nn.Linear(hidden, hidden), nn.Tanh())
        self.entity_encoder = nn.Sequential(nn.Linear(entity_dim + self_dim, hidden),
                                            nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.relations = nn.ModuleList([RelationAttention(hidden, heads) for _ in range(relation_layers)])
        self.action_queries = nn.Parameter(torch.randn(action_dim, hidden) / math.sqrt(hidden))
        self.query_from_self = nn.Linear(hidden, hidden)
        self.readout = TypedActionReadout(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 4 + 4, hidden), nn.Tanh(),
                                  nn.Linear(hidden, 1))
        nn.init.orthogonal_(self.head[-1].weight, gain=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _prepare(self, entities, mask):
        # Mask values before any arithmetic: even NaN/inf padding is inert.
        entities = torch.where(mask[..., None], entities, torch.zeros_like(entities))
        type_masks = [mask & (entities[..., j] > 0.5) for j in range(2)]
        counts = torch.stack([valid.sum(-1) for valid in type_masks], -1).to(entities.dtype)
        if self.mode == "summary":
            means = [(entities * valid[..., None]).sum(-2) /
                     valid.sum(-1, keepdim=True).clamp_min(1) for valid in type_masks]
            return torch.stack(means, -2), counts > 0, counts
        if self.mode == "nearest2":
            selected = torch.zeros_like(mask)
            # Deterministic lexicographic ties make selection independent of input order.
            order = torch.arange(entities.shape[1], device=entities.device).expand(mask.shape)
            for column in range(entities.shape[-1] - 1, -1, -1):
                values = entities[..., column].gather(1, order)
                order = order.gather(1, values.argsort(dim=1, stable=True))
            distance = entities[..., 2:4].square().sum(-1)
            for valid in type_masks:
                keys = distance.masked_fill(~valid, float("inf")).gather(1, order)
                nearest = order.gather(1, keys.argsort(dim=1, stable=True))[:, :2]
                typed_selection = torch.zeros_like(mask)
                typed_selection.scatter_(1, nearest, valid.gather(1, nearest))
                selected |= typed_selection
            mask = selected
            # This ablation keeps only the nearest two records per type. Counts
            # describe the retained observation, not the discarded full roster.
            counts = torch.stack([(selected & typed).sum(-1) for typed in type_masks], -1).to(entities.dtype)
        return entities, mask, counts

    def forward(self, obs):
        own, entities, mask = obs["self"], obs["entities"], obs["mask"].bool()
        leading = own.shape[:-1]
        if own.shape[-1] != self.self_dim or entities.shape[-1] != self.entity_dim:
            raise ValueError("Observation feature dimensions do not match this network")
        if entities.shape[:-2] != leading or mask.shape != entities.shape[:-1]:
            raise ValueError("self, entities and mask batch dimensions disagree")
        own = own.reshape(-1, self.self_dim)
        entities = entities.reshape(own.shape[0], entities.shape[-2], self.entity_dim)
        mask = mask.reshape(own.shape[0], -1)
        entities, mask, counts = self._prepare(entities, mask)
        # Unselected nearest-neighbor rows also cannot influence learned features.
        entities = torch.where(mask[..., None], entities, torch.zeros_like(entities))
        own_embedding = self.self_encoder(own)
        repeated = own[:, None, :].expand(-1, entities.shape[1], -1)
        tokens = self.entity_encoder(torch.cat([entities, repeated], -1))
        tokens = torch.cat([own_embedding[:, None, :], tokens], 1)
        valid = torch.cat([torch.ones((own.shape[0], 1), device=mask.device, dtype=torch.bool), mask], 1)
        # The self node is at the local origin. Both other-node coordinates and
        # their differences come only from fields 2:4 of the masked observation.
        positions = torch.cat([entities.new_zeros((own.shape[0], 1, 2)), entities[..., 2:4]], 1)
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        for relation in self.relations:
            tokens = relation(tokens, positions, valid)
        own_context = tokens[:, 0, :]
        query = torch.tanh(self.query_from_self(own_context)[:, None, :] + self.action_queries[None, :, :])
        type_masks = [mask & (entities[..., j] > 0.5) for j in range(2)]
        pooled = self.readout(query, tokens[:, 1:, :], type_masks)
        count_features = counts.log1p()[:, None, :].expand(-1, self.action_dim, -1)
        empty_features = (counts == 0).to(entities.dtype)[:, None, :].expand(-1, self.action_dim, -1)
        # Retain action identity in the head even when only the self token is
        # visible. Empty types supply a zero readout and an explicit empty bit.
        own_context = own_context[:, None, :].expand(-1, self.action_dim, -1)
        scores = self.head(torch.cat([own_context, query, *pooled, count_features, empty_features], -1)).squeeze(-1)
        return scores.reshape(*leading, self.action_dim)
