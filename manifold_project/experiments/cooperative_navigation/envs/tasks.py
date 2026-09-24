"""Additional task adapters plug into the original navigation collector/Runner."""
import numpy as np
from .navigation import Navigation as NativeNavigation


def Navigation(config, n=None):
    if config.get('environment', 'navigation') == 'navigation':
        return NativeNavigation(config, n)
    return EntityTask(config, n)


class EntityTask:
    def __init__(self, config, n=None):
        from ...applications.envs import make_env
        self.config = config
        self.task = make_env(config['environment'], config['current_config'], n_agents=n)
        self.n, self.horizon = self.task.n_agents, self.task.horizon
        self.include_time, self.end_reason = True, None
        self.state_dim = self.task.state_dim
        # Shape probe does not consume the collector's separately seeded scenes.
        obs, _ = self.task.reset(seed=0)
        self.obs_dim = self.flatten(obs).shape[-1]
        self.t = 0

    @staticmethod
    def flatten(obs):
        return np.concatenate([obs['self'][:, :-1], obs['entities'].reshape(len(obs['self']), -1)], -1)

    def reset(self, seed):
        obs, self.info = self.task.reset(seed)
        self.t, self.end_reason = 0, self.task.end_reason
        return self.flatten(obs)

    def step(self, actions):
        obs, reward, terminal, truncated, self.info = self.task.step(actions)
        self.t, self.end_reason = self.task.t, self.task.end_reason
        return self.flatten(obs), reward, terminal, truncated

    def state(self):
        return self.task.state()

    def metrics(self):
        # Expose task metrics only. Navigation quantities are not invented for W/P.
        return {k: v for k,v in self.info.items() if isinstance(v, (int, float, np.number))}

    def close(self):
        self.task.close()
