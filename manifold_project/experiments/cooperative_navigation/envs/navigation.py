"""Pinned native MPE2 dynamics; physical diagnostics never enter the actor."""
import importlib.metadata
import numpy as np
from ..configs import continuing_task, arrival_task, episode_horizon


class Navigation:
    def __init__(self, config, n=None):
        from mpe2 import simple_spread_v3
        if importlib.metadata.version('mpe2') != '1.1.1':
            raise RuntimeError('本协议锁定 mpe2==1.1.1，请安装 requirements.txt')
        self.n = n or config['n_agents']
        self.horizon = episode_horizon(config)
        self.first_arrival = arrival_task(config)
        self.end_reason = None
        self.include_time = not continuing_task(config)
        self.env = simple_spread_v3.parallel_env(
            N=self.n, local_ratio=config['local_ratio'], max_cycles=self.horizon,
            continuous_actions=False, num_agent_neighbors=config['agent_neighbors'],
            num_landmark_neighbors=config['landmark_neighbors'],
            terminate_on_success=False, curriculum=False, dynamic_rescaling=False)
        self.names = self.env.possible_agents
        self.obs_dim = self.env.observation_space(self.names[0]).shape[0]
        self.state_dim = self.obs_dim * self.n + int(self.include_time)
        assert self.env.action_space(self.names[0]).n == 5
        self.t = 0

    @property
    def world(self):
        return self.env.unwrapped.world

    def reset(self, seed):
        observations, _ = self.env.reset(seed=int(seed))
        self.t = 0
        self.end_reason = 'success' if self.first_arrival and self.metrics()['all_covered'] else None
        return np.stack([observations[a] for a in self.names])

    def state(self):
        state = self.env.state()
        if self.include_time:
            state = np.concatenate([state, [self.t / self.horizon]])
        return state.astype(np.float32)

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
        if self.t >= self.horizon or self.end_reason is not None:
            raise RuntimeError('采样窗口已结束，请 reset 或在新 rollout 中设置更长 horizon')
        observations, rewards, terminated, truncated, _ = self.env.step(dict(zip(self.names, map(int, actions))))
        self.t += 1
        if set(rewards) != set(self.names):
            raise RuntimeError('原生环境提前移除机器人')
        # This scenario ends all agents together and has no success termination.
        if len(set(terminated.values())) != 1 or len(set(truncated.values())) != 1:
            raise RuntimeError('当前团队采样器不支持机器人异步结束')
        terminal, timeout = all(terminated.values()), all(truncated.values())
        if terminal or timeout != (self.t == self.horizon):
            raise RuntimeError('原生结束标志与关闭成功终止的固定采样窗口不一致')
        # The task definition belongs to this adapter. MPE supplies dynamics,
        # native rewards and final observations; its clock remains a truncation.
        if self.first_arrival:
            success = bool(self.metrics()['all_covered'])
            self.end_reason = 'success' if success else 'deadline' if timeout else None
            terminal, timeout = self.end_reason is not None, False
        # MPE2 supplies the pre-reset final observation even after a timeout.
        obs = np.stack([observations[a] for a in self.names])
        return obs, float(np.mean(list(rewards.values()), dtype=np.float64)), terminal, timeout

    def close(self):
        self.env.close()
