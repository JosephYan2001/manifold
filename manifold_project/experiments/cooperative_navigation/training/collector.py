from collections import Counter
import numpy as np
import torch
from ..envs.navigation import Navigation
from ..observations.history import History
from .storage import seed_for


def episode(actor, config, reset_seed, action_seed, n=None, retain=False, on_step=None):
    env = Navigation(config, n)
    history = History(env.n, env.obs_dim, config['history'], env.horizon, include_time=env.include_time)
    rng = np.random.default_rng(action_seed)
    device = next(actor.parameters()).device
    rows = {key: [] for key in ('x', 'state', 'actions', 'mu', 'reward', 'terminated', 'truncated')}
    rewards, collisions, robot_collisions = [], [], []
    coverages, full_coverages = [], []
    try:
        obs = env.reset(reset_seed)
        previous = None
        for t in range(env.horizon):
            x = history.append(obs, previous, t)
            with torch.no_grad():
                p = actor(torch.as_tensor(x, device=device)).cpu().numpy()
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
                raise FloatingPointError('非法 Actor 概率')
            cumulative = p.astype(np.float64).cumsum(-1)
            cumulative /= cumulative[:, -1:]
            actions = (rng.random(env.n)[:, None] >= cumulative).sum(-1)
            state = env.state() if retain else None
            obs, reward, terminated, truncated = env.step(actions)
            if on_step:
                on_step()
            if not np.isfinite(reward):
                raise FloatingPointError('非有限环境奖励')
            metrics = env.metrics()
            collisions.append(metrics['collision_pairs'])
            robot_collisions.append(metrics['collisions_per_agent'])
            rewards.append(reward)
            coverages.append(metrics['coverage'])
            full_coverages.append(metrics['all_covered'])
            if retain:
                for key, value in zip(rows, (x, state, actions, p, reward, terminated, truncated)):
                    rows[key].append(value)
            previous = actions
        assert terminated or truncated
        metrics.update(J=float(np.dot(np.power(config['gamma'], np.arange(env.horizon)), rewards)),
                       collision_pairs=float(np.mean(collisions)),
                       collisions_per_agent=float(np.mean(robot_collisions)),
                       mean_reward=float(np.mean(rewards)), mean_coverage=float(np.mean(coverages)),
                       tail_coverage=float(np.mean(coverages[env.horizon//2:])),
                       tail_all_covered=float(np.mean(full_coverages[env.horizon//2:])))
        if not retain:
            return metrics
        batch = {key: np.asarray(value) for key, value in rows.items()}
        # Construct s_T / h_T using the last action, before closing/resetting the env.
        batch.update(final_x=history.append(obs, previous, env.t), final_state=env.state())
        return batch
    finally:
        env.close()


class Collector:
    def __init__(self, config, condition, seed, log=None):
        self.config, self.condition, self.seed, self.log = config, condition, seed, log
        self.counts, self.costs = Counter(), Counter()

    @property
    def used(self):
        return sum(self.costs.values())

    def collect(self, actor, count, purpose, retain=True):
        if self.used+count*self.config['horizon'] > self.config['budget']:
            raise RuntimeError('源预算不足，禁止超支采样')
        episodes = []
        for _ in range(count):
            index = self.counts[purpose]
            self.counts[purpose] += 1
            if self.log:
                self.log('sampling_start', purpose=purpose, episode=index,
                         reserved_through=self.used+self.config['horizon'])
            def debit():
                self.costs[purpose] += 1
            episodes.append(episode(actor, self.config,
                seed_for('train', self.condition, self.seed, purpose, index, 'reset'),
                seed_for('train', self.condition, self.seed, purpose, index, 'action'),
                retain=retain, on_step=debit))
            if self.log:
                self.log('sampling', purpose=purpose, episode=index, used=self.used)
        if not retain:
            return episodes
        device = next(actor.parameters()).device
        return {key: torch.as_tensor(np.stack([e[key] for e in episodes]), device=device,
                                     dtype=(torch.long if key == 'actions' else
                                            torch.bool if key in ('terminated', 'truncated') else torch.float32))
                for key in episodes[0]}
