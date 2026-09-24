"""Dictionary interface to the restored native navigation adapter; no second dynamics loop."""
import numpy as np
from ...cooperative_navigation.envs.navigation import Navigation as LegacyNavigation


class Navigation(LegacyNavigation):
    self_dim, entity_dim, action_dim = 5, 6, 5

    def __init__(self, config, n_agents=None, render_mode=None):
        c = dict(config, observation_protocol='entities_v1', task_mode='first_arrival',
                 task_horizon=config.get('horizon', 200), horizon=config.get('horizon', 200),
                 agent_neighbors=None, landmark_neighbors=None, local_ratio=config.get('local_ratio', .5),
                 render_mode=render_mode)
        super().__init__(c, n_agents)
        self.package_version = '1.1.1'
        self.n_agents = self.n
        self.coverage_radius = config.get('coverage_radius', .1)
        self.local_ratio = c['local_ratio']
        self.initial_scale = config.get('initial_scale', 1.)
        self.initial_done, self._ready = False, False

    def _observation(self, raw):
        n = self.n_agents
        own = np.empty((n, self.self_dim), dtype=np.float32)
        entities = np.zeros((n, 2 * n - 1, self.entity_dim), dtype=np.float32)
        for i, name in enumerate(self.names):
            vector = np.asarray(raw[name], dtype=np.float32)
            expected = 4 + 2 * n + 4 * (n - 1)
            if vector.shape != (expected,):
                raise RuntimeError(f"Unexpected native observation shape {vector.shape}; expected {(expected,)}")
            own[i, :4], own[i, 4] = vector[:4], 1 - self.t / self.horizon
            entities[i, :n, 0] = 1
            entities[i, :n, 2:4] = vector[4:4 + 2 * n].reshape(n, 2)
            entities[i, n:, 1] = 1
            start = 4 + 2 * n
            entities[i, n:, 2:4] = vector[start:start + 2 * (n - 1)].reshape(n - 1, 2)
            entities[i, n:, 4:6] = vector[start + 2 * (n - 1):].reshape(n - 1, 2)
        return {"self": own, "entities": entities,
                "mask": np.ones((n, 2 * n - 1), dtype=bool)}


    def reset(self, seed=None):
        raw = super().reset(0 if seed is None else seed)
        self.initial_done, self._ready = self.end_reason == 'success', True
        return self._observation(dict(zip(self.names, raw))), self._info(self.metrics())

    def metrics(self):
        metrics = super().metrics()
        metrics.update(success=bool(metrics['all_covered']), mean_goal_distance=metrics['distance'],
            collision_participations=2*metrics['collision_pairs'],
            reward_distance=-(1-self.local_ratio)*self.n*metrics['distance'],
            reward_collision=-self.local_ratio*2*metrics['collision_pairs']/self.n)
        return metrics

    def _info(self, metrics):
        return dict(metrics, end_reason=self.end_reason, initial_success=self.initial_done,
                    elapsed_steps=self.t, completion_time_capped=self.t if metrics['success'] else self.horizon,
                    n_agents=self.n, entity_count=2*self.n-1)

    def state(self):
        if not self._ready:
            raise RuntimeError('reset must be called before state')
        return super().state()

    def step(self, actions):
        actions = np.asarray(actions)
        if (not self._ready or actions.shape != (self.n,) or not np.isfinite(actions).all()
                or not np.all((actions >= 0) & (actions < 5) & (actions == np.floor(actions)))):
            raise ValueError('Expected one valid native action per robot after reset')
        raw, reward, terminal, truncated = super().step(actions)
        return self._observation(dict(zip(self.names, raw))), reward, terminal, truncated, self._info(self.metrics())

    def protocol(self):
        return {"environment": "mpe2.simple_spread_v3", "version": self.package_version,
                "n_agents": self.n_agents, "n_landmarks": self.n_agents, "horizon": self.horizon,
                "actions": ["noop", "left", "right", "down", "up"],
                "self_fields": ["self_velocity_x", "self_velocity_y", "self_position_x", "self_position_y", "remaining_time_fraction"],
                "entity_fields": ["is_landmark", "is_peer", "relative_x", "relative_y", "communication_0", "communication_1"],
                "critic_fields": ["all_agent_positions_and_velocities", "all_landmark_positions", "elapsed_time_fraction"],
                "observation": "native_full_lists_no_truncation", "local_ratio": self.local_ratio,
                "reward_aggregation": "mean_native_agent_rewards", "success": "first_simultaneous_coverage",
                "coverage_radius": self.coverage_radius, "deadline_is_terminal": True,
                "initial_position_range": [-self.initial_scale, self.initial_scale],
                "agent_radius": [float(a.size) for a in self.world.agents],
                "agent_acceleration": [a.accel for a in self.world.agents],
                "resolved_discrete_action_force": [5.0 if a.accel is None else float(a.accel) for a in self.world.agents],
                "agent_max_speed": [a.max_speed for a in self.world.agents],
                "agent_mass": [float(a.mass) for a in self.world.agents],
                "physics_dt": float(self.world.dt), "physics_damping": float(self.world.damping),
                "contact_force": float(self.world.contact_force),
                "contact_margin": float(self.world.contact_margin),
                "transport_classification": "B_empirical_extrapolation"}


    def render(self):
        return self.env.render()
