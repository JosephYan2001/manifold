"""Pinned native MPE2 dynamics; physical diagnostics never enter the actor."""
import importlib.metadata
import numpy as np


class Navigation:
    def __init__(self, config, n=None):
        from mpe2 import simple_spread_v3
        if importlib.metadata.version('mpe2') != '1.1.1':
            raise RuntimeError('本协议锁定 mpe2==1.1.1，请安装 requirements.txt')
        self.n = n or config['n_agents']
        self.horizon = config['horizon']
        self.env = simple_spread_v3.parallel_env(
            N=self.n, local_ratio=config['local_ratio'], max_cycles=self.horizon,
            continuous_actions=False, num_agent_neighbors=config['agent_neighbors'],
            num_landmark_neighbors=config['landmark_neighbors'],
            terminate_on_success=False, curriculum=False, dynamic_rescaling=False)
        self.names = self.env.possible_agents
        self.obs_dim = self.env.observation_space(self.names[0]).shape[0]
        self.state_dim = self.obs_dim * self.n + 1
        assert self.env.action_space(self.names[0]).n == 5
        self.t = 0

    @property
    def world(self):
        return self.env.unwrapped.world

    def reset(self, seed):
        observations, _ = self.env.reset(seed=int(seed))
        self.t = 0
        return np.stack([observations[a] for a in self.names])

    def state(self):
        return np.concatenate([self.env.state(), [self.t / self.horizon]]).astype(np.float32)

    def metrics(self):
        agents, landmarks = self.world.agents, self.world.landmarks
        distances = np.linalg.norm(np.array([a.state.p_pos for a in agents])[:, None, :]
                                   - np.array([l.state.p_pos for l in landmarks])[None, :, :], axis=-1).min(axis=0)
        pairs = sum(np.linalg.norm(a.state.p_pos-b.state.p_pos) < a.size+b.size
                    for i, a in enumerate(agents) for b in agents[i+1:])
        return {'distance': float(distances.mean()), 'coverage': float((distances < .1).mean()),
                'all_covered': float((distances < .1).all()), 'collision_pairs': float(pairs),
                'collisions_per_agent': float(2*pairs/self.n)}

    def step(self, actions):
        observations, rewards, terminated, truncated, _ = self.env.step(dict(zip(self.names, map(int, actions))))
        self.t += 1
        if set(rewards) != set(self.names):
            raise RuntimeError('原生环境提前移除机器人')
        done = all(terminated[a] or truncated[a] for a in self.names)
        if done != (self.t == self.horizon):
            raise RuntimeError('终点与固定时域不一致')
        obs = np.stack([observations[a] for a in self.names]) if not done else None
        return obs, float(np.mean(list(rewards.values()), dtype=np.float64)), done

    def close(self):
        self.env.close()
