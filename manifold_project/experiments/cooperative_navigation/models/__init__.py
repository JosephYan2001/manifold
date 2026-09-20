import torch
from torch import nn


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


def load_actor(state, config=None):
    """Restore an evaluation actor by checkpoint format, including legacy models."""
    config = config or state['config']
    kind = state.get('actor_kind', 'local_mlp_v1')
    if kind in ('author_mappo_v1', 'author_ippo_v1'):
        from ..training.author_ppo import AuthorActor
        actor = AuthorActor(state['input_dim'], config, condition=kind.split('_')[1])
    elif kind == 'local_mlp_v1':
        actor = Actor(state['input_dim'], config)
    else:
        raise ValueError(f'未知 Actor 实现: {kind}')
    actor.load_state_dict(state['actor'])
    return actor.eval()


class Actor(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.net = mlp(input_dim, config['hidden'], 5)
        self.beta = config['beta']

    def forward(self, x):
        return (1-self.beta)*torch.softmax(self.net(x), dim=-1)+self.beta/5


class Direction(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.net = mlp(input_dim, config['hidden'], 5)
        self.q_max = config['q_max']

    def forward(self, x):
        u = torch.tanh(self.net(x))
        return self.q_max/2*(u-u.mean(dim=-1, keepdim=True))


class Critic(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.net = mlp(input_dim, config['hidden'], 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)
