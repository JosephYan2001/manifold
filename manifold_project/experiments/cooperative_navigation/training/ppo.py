"""Shared numerical objectives; episode is the minibatch grouping unit."""
import torch


def returns(rewards, gamma):
    out = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        tail = rewards[:, t] + gamma*tail
        out[:, t] = tail
    return out


def gae(rewards, values, gamma, lam):
    advantage = torch.zeros_like(values)
    tail = torch.zeros_like(values[:, 0])
    next_value = torch.zeros_like(tail)
    for t in reversed(range(rewards.shape[1])):
        reward = rewards[:, t]
        while reward.ndim < tail.ndim:
            reward = reward.unsqueeze(-1)
        delta = reward + gamma*next_value-values[:, t]
        tail = delta + gamma*lam*tail
        advantage[:, t] = tail
        next_value = values[:, t]
    return advantage.detach()


def weighted(value, gamma):
    # E,T,N -> sum normalized over episodes, time and agents.
    w = torch.pow(torch.as_tensor(gamma, device=value.device, dtype=value.dtype), torch.arange(value.shape[1], device=value.device))
    return (value*w.reshape(1, -1, *([1]*(value.ndim-2)))).sum()/(w.sum()*value.shape[0]*(value.shape[2] if value.ndim == 3 else 1))


def direction_terms(q, mu, actions):
    center = (mu*q).sum(-1)
    score = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)-center
    fisher = (mu*q.square()).sum(-1)-center.square()
    return score, fisher


def ppo_loss(actor, batch, advantage, config):
    p = actor(batch['x'])
    action = batch['actions'].unsqueeze(-1)
    ratio = p.gather(-1, action).squeeze(-1)/batch['mu'].gather(-1, action).squeeze(-1)
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-config['ppo_clip'], 1+config['ppo_clip'])*advantage)
    entropy = -(p*p.log()).sum(-1)
    loss = -weighted(objective+config['entropy_coef']*entropy, config['gamma'])
    with torch.no_grad():
        kl = weighted((batch['mu']*(batch['mu'].log()-p.log())).sum(-1), config['gamma'])
        clipped = ((ratio-1).abs() > config['ppo_clip']).float().mean()
    return loss, {'ppo_kl': float(kl), 'clip_fraction': float(clipped)}
