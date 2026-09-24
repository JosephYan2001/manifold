import torch
from torch import nn


def entity_input(x, config=None):
    """Decode native full MPE fields carried by the unchanged History sampler.

    History appends previous action (5), frame validity (1), elapsed time (1).
    Current-frame models deliberately ignore previous actions; their only clock
    is remaining task time. List length varies; learned parameter shapes do not.
    """
    raw = x[..., :-7]
    config = config or {}
    if config.get('environment', 'navigation') != 'navigation':
        own_dim = config['entity_self_dim']-1
        width = config['entity_record_dim']
        entities = raw[..., own_dim:].reshape(*raw.shape[:-1], -1, width)
        return {'self': torch.cat([raw[..., :own_dim], 1-x[..., -1:]], -1),
                'entities': entities,
                'mask': torch.ones(entities.shape[:-1], dtype=torch.bool, device=x.device)}
    if raw.shape[-1] % 6 or raw.shape[-1] < 12:
        raise ValueError('Full native navigation observation must have 6*N fields')
    n = raw.shape[-1] // 6
    goals = raw[..., 4:4+2*n].reshape(*raw.shape[:-1], n, 2)
    peers = raw[..., 4+2*n:4+2*n+2*(n-1)].reshape(*raw.shape[:-1], n-1, 2)
    comm = raw[..., 4+2*n+2*(n-1):].reshape(*raw.shape[:-1], n-1, 2)
    gtype = goals.new_zeros((*goals.shape[:-1], 2)); gtype[..., 0] = 1
    ptype = peers.new_zeros((*peers.shape[:-1], 2)); ptype[..., 1] = 1
    entities = torch.cat([torch.cat([gtype, goals, torch.zeros_like(goals)], -1),
                          torch.cat([ptype, peers, comm], -1)], -2)
    return {'self': torch.cat([raw[..., :4], 1-x[..., -1:]], -1),
            'entities': entities,
            'mask': torch.ones(entities.shape[:-1], dtype=torch.bool, device=x.device)}


def entity_encoder(config):
    from .entities import EntityEncoder
    return EntityEncoder(config.get('entity_self_dim', 5), config.get('entity_record_dim', 6),
                         config.get('action_dim', 5), config['hidden'], config.get('heads', 4),
                         config.get('relation_layers', 2), config.get('observation_mode', 'full'))


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


def load_actor(state, config=None):
    """Restore an evaluation actor by checkpoint format, including legacy models."""
    config = config or state['config']
    kind = state.get('actor_kind', 'local_mlp_v1')
    if kind in ('author_mappo_v1', 'author_ippo_v1', 'entity_mappo_v1', 'entity_ippo_v1'):
        from ..training.author_ppo import AuthorActor
        actor = AuthorActor(state['input_dim'], config, condition=kind.split('_')[1])
    elif kind in ('local_mlp_v1', 'entity_v1'):
        actor = Actor(state['input_dim'], config)
    else:
        raise ValueError(f'未知 Actor 实现: {kind}')
    actor.load_state_dict(state['actor'])
    return actor.eval()


class Actor(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.entity = config.get('observation_protocol') == 'entities_v1'
        self.config = dict(config)
        self.action_dim = config.get('action_dim', 5)
        self.net = entity_encoder(config) if self.entity else mlp(input_dim, config['hidden'], 5)
        self.beta = config['beta']

    def forward(self, x):
        inputs = entity_input(x, self.config) if self.entity and not isinstance(x, dict) else x
        return (1-self.beta)*torch.softmax(self.net(inputs), dim=-1)+self.beta/self.action_dim

    def evaluate_actions(self, x, rnn_states, actions, masks, available_actions=None, active_masks=None):
        x = torch.as_tensor(x, dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        actions = torch.as_tensor(actions, device=x.device, dtype=torch.long)
        distribution = torch.distributions.Categorical(probs=self(x))
        entropy = distribution.entropy()
        if active_masks is not None:
            active = torch.as_tensor(active_masks, device=x.device).reshape(entropy.shape)
            entropy = (entropy*active).sum()/active.sum().clamp_min(1)
        else:
            entropy = entropy.mean()
        return distribution.log_prob(actions.squeeze(-1)).unsqueeze(-1), entropy


class Direction(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.entity = config.get('observation_protocol') == 'entities_v1'
        self.config = dict(config)
        self.net = entity_encoder(config) if self.entity else mlp(input_dim, config['hidden'], 5)
        self.q_max = config['q_max']

    def forward(self, x):
        inputs = entity_input(x, self.config) if self.entity and not isinstance(x, dict) else x
        u = torch.tanh(self.net(inputs))
        return self.q_max/2*(u-u.mean(dim=-1, keepdim=True))


class Critic(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.net = mlp(input_dim, config['hidden'], 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LocalCritic(nn.Module):
    """IPPO's value network uses exactly the same authorized local fields."""
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = dict(config)
        self.net = entity_encoder(config)

    def forward(self, x):
        return self.net(entity_input(x, self.config)).mean(-1)
