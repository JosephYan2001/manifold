"""Dictionary-view facade for an Actor trained by the restored flat collector."""
import torch
from torch import nn


class DeploymentActor(nn.Module):
    def __init__(self, actor, config):
        super().__init__()
        self.actor, self.config = actor, config
        self.model_config = dict(mode=config.get('observation_mode', 'full'),
                                 relation_layers=config.get('relation_layers', 2))

    def forward(self, obs):
        own, entities = obs['self'], obs['entities']
        if self.config.get('environment', 'navigation') == 'navigation':
            n = (entities.shape[-2]+1)//2
            raw = torch.cat([own[..., :4], entities[..., :n, 2:4].flatten(-2),
                             entities[..., n:, 2:4].flatten(-2), entities[..., n:, 4:6].flatten(-2)], -1)
        else:
            raw = torch.cat([own[..., :-1], entities.flatten(-2)], -1)
        extra = raw.new_zeros((*raw.shape[:-1], 6))
        extra[..., -1] = 1
        return self.actor(torch.cat([raw, extra, 1-own[..., -1:]], -1))
