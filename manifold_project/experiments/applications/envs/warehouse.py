"""RWARE finite delivery task, preserving all native local grid observations."""
from __future__ import annotations

import hashlib
from importlib.metadata import version

import numpy as np


# Native tiny: shelf_rows=1, shelf_columns=3, column_height=8. Explicit layout
# avoids silently changing the map when package registration defaults change.
TINY_LAYOUT = "\n".join(["..........", *([".xx....xx."] * 8), "..........", "....gg...."])
TINY_LAYOUT_SHA256 = hashlib.sha256(TINY_LAYOUT.encode("utf-8")).hexdigest()


class Warehouse:
    self_dim = 9
    entity_dim = 11
    action_dim = 5

    def __init__(self, config, n_agents=None, render_mode=None):
        from rware.warehouse import Warehouse as NativeWarehouse, RewardType, ObservationType
        self.package_version = version("rware")
        if self.package_version != "2.0.0":
            raise RuntimeError("Warehouse protocol requires rware==2.0.0")
        self.config = dict(config)
        self.n_agents = int(n_agents if n_agents is not None else config.get("n_agents", 4))
        self.n = self.n_agents
        self.horizon = int(config.get("horizon", 1000))
        self.sensor_range = int(config.get("sensor_range", 1))
        self.request_queue_size = int(config.get("request_queue_size", 4))
        if self.sensor_range != 1:
            raise ValueError("The W-v1 protocol fixes sensor_range=1; use a separate protocol for other views")
        if config.get("layout", TINY_LAYOUT) != TINY_LAYOUT:
            raise ValueError("The W-v1 protocol fixes the explicit native tiny layout")
        if config.get("layout_sha256", TINY_LAYOUT_SHA256) != TINY_LAYOUT_SHA256:
            raise ValueError("Warehouse layout hash does not match the pinned ASCII layout")
        if self.horizon < 1 or not 1 <= self.n_agents <= 110:
            raise ValueError("Invalid warehouse horizon or robot count")
        self.shelf_count = TINY_LAYOUT.count("x")
        if not 1 <= self.request_queue_size < self.shelf_count:
            raise ValueError("Queue must leave an unrequested shelf available for replacement")
        self.env = NativeWarehouse(
            shelf_columns=3, column_height=8, shelf_rows=1, n_agents=self.n_agents,
            msg_bits=0, sensor_range=1, request_queue_size=self.request_queue_size,
            max_inactivity_steps=None, max_steps=None, reward_type=RewardType.GLOBAL,
            layout=TINY_LAYOUT, observation_type=ObservationType.DICT,
            normalised_coordinates=False, render_mode=render_mode)
        self.state_dim = 7 * self.n_agents + 3 * self.shelf_count + 1
        self.t = 0
        self.initial_done = False
        self.end_reason = None
        self._ready = False
        self.deliveries_total = 0
        self.first_delivery_step = None
        self.no_delivery_streak = 0
        self.longest_no_delivery_streak = 0

    def _observation(self, raw):
        own = np.empty((self.n_agents, self.self_dim), dtype=np.float32)
        entities = np.empty((self.n_agents, 9, self.entity_dim), dtype=np.float32)
        onehot = np.eye(4, dtype=np.float32)
        offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        for i, record in enumerate(raw):
            s = record["self"]
            own[i] = [*s["location"], float(s["carrying_shelf"][0]),
                      *onehot[int(s["direction"])], float(s["on_highway"][0]),
                      1 - self.t / self.horizon]
            if len(record["sensors"]) != 9:
                raise RuntimeError("The pinned warehouse protocol requires all nine native grid records")
            for j, (cell, (dx, dy)) in enumerate(zip(record["sensors"], offsets)):
                # Empty/out-of-bounds fields are preserved exactly as emitted by
                # RWARE, including the default 'UP' direction of an empty cell.
                # No boundary flag, nearby robot load, or global request is added.
                entities[i, j] = [1, 0, dx, dy, float(cell["has_agent"][0]),
                                   *onehot[int(cell["direction"])],
                                   float(cell["has_shelf"][0]), float(cell["shelf_requested"][0])]
        return {"self": own, "entities": entities,
                "mask": np.ones((self.n_agents, 9), dtype=bool)}

    def reset(self, seed=None):
        raw, _ = self.env.reset(seed=None if seed is None else int(seed))
        self.t, self.end_reason, self._ready = 0, None, True
        self.deliveries_total, self.first_delivery_step = 0, None
        self.no_delivery_streak = self.longest_no_delivery_streak = 0
        return self._observation(raw), self._info(0, 0)

    def state(self):
        if not self._ready:
            raise RuntimeError("reset must be called before state")
        onehot = np.eye(4, dtype=np.float32)
        state = []
        for agent in self.env.agents:
            state.extend([agent.x, agent.y, *onehot[agent.dir.value],
                          0 if agent.carrying_shelf is None else agent.carrying_shelf.id / self.shelf_count])
        requested = {s.id for s in self.env.request_queue}
        for shelf in self.env.shelfs:
            state.extend([shelf.x, shelf.y, float(shelf.id in requested)])
        state.append(1 - self.t / self.horizon)
        return np.asarray(state, dtype=np.float32)

    def _info(self, deliveries, blocked):
        return {"deliveries": int(deliveries), "deliveries_total": self.deliveries_total,
                "first_delivery_step": self.first_delivery_step,
                "first_delivery_time_capped": self.horizon if self.first_delivery_step is None else self.first_delivery_step,
                "no_delivery_streak": self.no_delivery_streak,
                "longest_no_delivery_streak": self.longest_no_delivery_streak,
                "blocked_forward_actions": int(blocked), "elapsed_steps": self.t,
                "end_reason": self.end_reason, "n_agents": self.n_agents,
                "entity_count": 9, "request_queue_size": self.request_queue_size,
                "team_throughput": self.deliveries_total / self.horizon,
                "per_robot_throughput": self.deliveries_total / (self.horizon * self.n_agents)}

    def step(self, actions):
        if not self._ready or self.end_reason is not None:
            raise RuntimeError("Task is inactive or already terminal; call reset before step")
        actions = np.asarray(actions)
        if (actions.shape != (self.n_agents,) or not np.all(np.isfinite(actions))
                or not np.all((actions >= 0) & (actions < 5) & (actions == np.floor(actions)))):
            raise ValueError("Expected one action in [0,4] per robot")
        before = [(agent.x, agent.y) for agent in self.env.agents]
        raw, rewards, native_terminal, native_truncated, _ = self.env.step(actions.astype(int).tolist())
        if native_terminal or native_truncated:
            raise RuntimeError("Unbounded native warehouse ended unexpectedly; adapter controls deadline")
        self.t += 1
        if not np.allclose(rewards, rewards[0]):
            raise RuntimeError("Expected identical GLOBAL rewards for all warehouse robots")
        reward = float(np.mean(rewards, dtype=np.float64))
        deliveries = int(round(reward))
        if not np.isclose(reward, deliveries) or deliveries < 0:
            raise RuntimeError("GLOBAL reward should count completed shelf deliveries exactly once")
        self.deliveries_total += deliveries
        if deliveries:
            if self.first_delivery_step is None:
                self.first_delivery_step = self.t
            self.no_delivery_streak = 0
        else:
            self.no_delivery_streak += 1
            self.longest_no_delivery_streak = max(self.longest_no_delivery_streak, self.no_delivery_streak)
        blocked = sum(int(action) == 1 and old == (agent.x, agent.y)
                      for action, old, agent in zip(actions, before, self.env.agents))
        if self.t >= self.horizon:
            self.end_reason = "deadline"
        return self._observation(raw), reward, self.end_reason is not None, False, self._info(deliveries, blocked)

    def protocol(self):
        return {"environment": "rware.Warehouse", "version": self.package_version,
                "layout": TINY_LAYOUT, "layout_sha256": TINY_LAYOUT_SHA256, "layout_shape": [11, 10],
                "n_agents": self.n_agents, "horizon": self.horizon,
                "request_queue_size": self.request_queue_size, "sensor_range": 1,
                "actions": ["noop", "forward", "left_turn", "right_turn", "toggle_load"],
                "self_fields": ["native_x", "native_y", "carrying_shelf", "direction_onehot_4", "on_highway", "remaining_time_fraction"],
                "entity_fields": ["is_grid_cell", "unused_type", "relative_dx", "relative_dy", "has_agent", "native_direction_onehot_4", "has_shelf", "shelf_requested"],
                "cell_order": "native_row_major_dy_then_dx_including_empty_and_boundary_padding",
                "critic_fields": ["all_agent_positions_directions_carried_shelf_ids", "all_shelf_positions_request_flags", "remaining_time_fraction"],
                "boundary_marker_added": False, "reward_type": "GLOBAL", "reward_aggregation": "mean_identical_global_rewards",
                "max_inactivity_steps": None, "deadline_is_terminal": True,
                "transport_classification": "empirical_extrapolation_unverified_coverage"}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
