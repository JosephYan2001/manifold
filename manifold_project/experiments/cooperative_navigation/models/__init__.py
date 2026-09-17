import torch
from torch import nn


def mlp(input_dim, hidden, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


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
