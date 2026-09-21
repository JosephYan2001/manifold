"""Shared numerical objectives; episode is the minibatch grouping unit."""
import torch


def _expand_step(value, template):
    while value.ndim < template.ndim:
        value = value.unsqueeze(-1)
    return value


def returns(rewards, gamma, *, next_values=None, terminated=None, truncated=None):
    """Reward-to-go, optionally bootstrapped at cutoffs; E,T[,N].

    A terminal transition has no future value. A timeout uses the value of its
    final observation and never incorporates rewards from the following reset.
    Without next_values this remains the observed finite-window return (logging).
    """
    out = torch.zeros_like(rewards if next_values is None else next_values)
    tail = torch.zeros_like(out[:, 0]) if next_values is None else next_values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        if truncated is not None:
            final = torch.zeros_like(tail) if next_values is None else next_values[:, t]
            tail = torch.where(_expand_step(truncated[:, t].bool(), tail), final, tail)
        if terminated is not None:
            tail = torch.where(_expand_step(terminated[:, t].bool(), tail), 0., tail)
        tail = _expand_step(rewards[:, t], tail) + gamma*tail
        out[:, t] = tail
    return out


def gae(rewards, values, gamma, lam, *, next_values=None, terminated=None, truncated=None):
    advantage = torch.zeros_like(values)
    tail = torch.zeros_like(values[:, 0])
    if next_values is None:
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    for t in reversed(range(rewards.shape[1])):
        terminal = torch.zeros_like(tail, dtype=torch.bool) if terminated is None else _expand_step(terminated[:, t].bool(), tail)
        timeout = torch.zeros_like(terminal) if truncated is None else _expand_step(truncated[:, t].bool(), tail)
        delta = _expand_step(rewards[:, t], tail) + gamma*torch.where(terminal, 0., next_values[:, t])-values[:, t]
        # Bootstrap at a timeout, but cut the advantage trace across reset.
        tail = delta + gamma*lam*torch.where(terminal | timeout, 0., tail)
        advantage[:, t] = tail
    return advantage.detach()


def value_targets(batch, critic, config, local=False):
    """Frozen labels for direction methods and the legacy local PPO backend."""
    from ..configs import continuing_task, arrival_task
    with torch.no_grad():
        values = critic(batch['x'] if local else batch['state'])
        terminal = batch['terminated']
        if continuing_task(config) or arrival_task(config):
            final = critic(batch['final_x'] if local else batch['final_state'])
        else:
            final = torch.zeros_like(values[:, 0])
            terminal = terminal | batch['truncated']
        next_values = torch.cat([values[:, 1:], final.unsqueeze(1)], dim=1)
        if arrival_task(config):
            valid = batch['valid']
            terminal = terminal | ~valid
            for i, length in enumerate(batch['lengths'].tolist()):
                if length:
                    next_values[i, length-1] = final[i]
        target = returns(batch['reward'], config['gamma'], next_values=next_values,
                         terminated=terminal, truncated=batch['truncated'])
        advantage = gae(batch['reward'], values, config['gamma'], config['gae_lambda'],
                        next_values=next_values, terminated=terminal, truncated=batch['truncated'])
        if arrival_task(config):
            mask = _expand_step(valid, values)
            values, target, advantage = [torch.where(mask, v, 0.) for v in (values, target, advantage)]
        return values.detach(), target.detach(), advantage.detach()


def weighted(value, gamma, valid=None, horizon=None):
    # E,T,N -> sum normalized over episodes, time and agents.
    w = torch.pow(torch.as_tensor(gamma, device=value.device, dtype=value.dtype), torch.arange(value.shape[1], device=value.device))
    if valid is not None:
        value = torch.where(_expand_step(valid, value), value, 0.)
    denominator = w.sum() if horizon is None else torch.pow(w.new_tensor(gamma), torch.arange(horizon, device=w.device)).sum()
    return (value*w.reshape(1, -1, *([1]*(value.ndim-2)))).sum()/(denominator*value.shape[0]*(value.shape[2] if value.ndim == 3 else 1))


def batch_weighted(value, batch, config):
    # Absorbing post-terminal steps have zero contribution. Use the common task
    # horizon, never the realized episode length, to preserve the return objective.
    from ..configs import arrival_task
    return weighted(value, config['gamma'], batch.get('valid'),
                    config['task_horizon'] if arrival_task(config) else None)


def valid_values(value, batch):
    return value[batch['valid']] if 'valid' in batch else value


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
    loss = -batch_weighted(objective+config['entropy_coef']*entropy, batch, config)
    with torch.no_grad():
        kl = batch_weighted((batch['mu']*(batch['mu'].log()-p.log())).sum(-1), batch, config)
        clipped = valid_values(((ratio-1).abs() > config['ppo_clip']).float(), batch).mean()
    return loss, {'ppo_kl': float(kl), 'clip_fraction': float(clipped),
                  'policy_entropy': float(batch_weighted(entropy.detach(), batch, config))}
