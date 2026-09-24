"""Exact one-step pair and genuinely sequential two-step environments."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class PairGame:
    n: int = 4
    plus_type_probability: float = 0.6
    variant: str = "normalized"
    source_n: int = 4
    interaction_strength: float = 2.0

    n_states = 2
    horizon = 1
    gamma = 1.0
    initial_policy = np.array([0.3, 0.7])  # types -1,+1, actions -1,+1
    bias = np.array([-0.1, 0.2])

    def __post_init__(self):
        if self.n < 2 or self.source_n < 2:
            raise ValueError("Pair experiments require at least two agents")
        if not 0 < self.plus_type_probability < 1:
            raise ValueError("Both types must have positive probability")
        if self.variant not in ("normalized", "fixed-edge", "composition"):
            raise ValueError(f"Unknown pair variant {self.variant}")
        if not np.isfinite(self.interaction_strength) or self.interaction_strength < 0:
            raise ValueError("interaction_strength must be finite and nonnegative")

    @property
    def type_probabilities(self):
        return np.array([1 - self.plus_type_probability, self.plus_type_probability])

    @property
    def edge_weight(self):
        return self.interaction_strength / (self.source_n - 1 if self.variant == "fixed-edge" else self.n - 1)

    @property
    def z(self):
        return 1.0

    @property
    def return_bound(self):
        return self.n * float(np.max(abs(self.bias))) + abs(self.edge_weight) * self.n * (self.n - 1) / 2

    @property
    def label_return_bound(self):
        return self.return_bound

    def rewards(self, types, actions):
        a = 2 * np.asarray(actions) - 1
        return (self.bias[types] * a).sum(-1) + self.edge_weight * ((a.sum(-1) ** 2 - self.n) / 2)

    def state_value(self, types, policy):
        means = 2 * np.asarray(policy)[types] - 1
        pair = (means.sum(-1) ** 2 - (means * means).sum(-1)) / 2
        return (self.bias[types] * means).sum(-1) + self.edge_weight * pair

    def sample(self, policy, episodes, rng):
        if episodes < 1:
            raise ValueError("Sampling requires complete positive episode count")
        types = (rng.random((episodes, self.n)) < self.plus_type_probability).astype(int)
        p = np.asarray(policy)[types]
        actions = (rng.random(p.shape) < p).astype(int)
        reward = self.rewards(types, actions)
        return {"x": types[:, None, :], "actions": actions[:, None, :],
                "old": p[:, None, :], "rewards": reward[:, None], "returns": reward[:, None],
                "weights": np.full((episodes, 1, self.n), 1 / (episodes * self.n)),
                "exact_advantage": (reward - self.state_value(types, policy))[:, None]}

    def oracle(self, policy, direction=None):
        p = np.asarray(policy)
        d, m = self.type_probabilities, 2 * p - 1
        mean = float(d @ m)
        c = self.bias + self.edge_weight * (self.n - 1) * mean
        j = self.n * float(d @ (self.bias * m)) + self.edge_weight * self.n * (self.n - 1) * mean ** 2 / 2
        variance = float(d @ (m - mean) ** 2)
        permission_gap = float(np.sum(d * 4 * p * (1 - p)) * self.edge_weight ** 2 * (self.n - 1) * variance)
        return _oracle_metrics(self, p, d, c, j, direction, permission_gap)

    def enumerate(self, policy):
        """Independent enumeration oracle; no training caller uses it for choices."""
        if 4 ** self.n > 1_100_000:
            raise ValueError("Enumeration exceeds one million type/action events")
        p = np.asarray(policy)
        rows, probabilities = [], []
        for events in product(range(4), repeat=self.n):
            events = np.asarray(events)
            x, a = events // 2, events % 2
            probability = float(np.prod(self.type_probabilities[x] * np.where(a, p[x], 1 - p[x])))
            rows.append((x, a, self.rewards(x, a), self.state_value(x, p)))
            probabilities.append(probability)
        x = np.asarray([row[0] for row in rows])[:, None, :]
        a = np.asarray([row[1] for row in rows])[:, None, :]
        rewards = np.asarray([row[2] for row in rows])[:, None]
        advantage = rewards - np.asarray([row[3] for row in rows])[:, None]
        batch = {"x": x, "actions": a, "old": p[x], "rewards": rewards,
                 "returns": rewards.copy(), "weights": np.full(x.shape, 1 / (len(x) * self.n)),
                 "exact_advantage": advantage}
        return batch, np.asarray(probabilities)


@dataclass(frozen=True)
class TwoStepGame:
    gamma: float = 0.9
    n = 2
    n_states = 3
    horizon = 2
    initial_policy = np.array([0.65, 0.75, 0.35])  # t0, t1 s=+, t1 s=-

    def __post_init__(self):
        if not 0 < self.gamma <= 1:
            raise ValueError("Two-step gamma must be in (0, 1]")

    @property
    def z(self):
        return 1 + self.gamma

    @property
    def return_bound(self):
        return 0.15 + self.gamma

    @property
    def label_return_bound(self):
        return max(1.0, self.return_bound)

    def state_values(self, policy):
        p = np.asarray(policy)
        m = 2 * p - 1
        vplus, vminus = 0.7 * m[1] + 0.3 * m[1] ** 2, -0.7 * m[2] + 0.3 * m[2] ** 2
        plus = p[0] ** 2 + (1 - p[0]) ** 2
        vzero = 0.15 * m[0] ** 2 + self.gamma * (plus * vplus + (1 - plus) * vminus)
        return np.array([vzero, vplus, vminus])

    def _batch(self, policy, a0, a1):
        p = np.asarray(policy)
        episodes = len(a0)
        physical0, physical1 = 2 * a0 - 1, 2 * a1 - 1
        s = physical0.prod(axis=1)
        state = np.where(s == 1, 1, 2)
        x = np.stack([np.zeros_like(a0), np.repeat(state[:, None], 2, axis=1)], axis=1)
        actions = np.stack([a0, a1], axis=1)
        r0 = 0.15 * s
        r1 = 0.7 * s * physical1.mean(axis=1) + 0.3 * physical1.prod(axis=1)
        returns = np.stack([r0 + self.gamma * r1, r1], axis=1)
        values = self.state_values(p)
        advantage = np.stack([r0 + self.gamma * values[state] - values[0], r1 - values[state]], axis=1)
        weights = np.broadcast_to(np.array([1, self.gamma])[None, :, None] / (episodes * self.n * self.z), x.shape).copy()
        return {"x": x, "actions": actions, "old": p[x], "rewards": np.stack([r0, r1], axis=1),
                "returns": returns, "weights": weights, "exact_advantage": advantage}

    def sample(self, policy, episodes, rng):
        a0 = (rng.random((episodes, 2)) < policy[0]).astype(int)
        state = np.where((2 * a0 - 1).prod(axis=1) == 1, 1, 2)
        a1 = (rng.random((episodes, 2)) < np.asarray(policy)[state, None]).astype(int)
        return self._batch(policy, a0, a1)

    def enumerate(self, policy):
        actions = np.asarray(list(product(range(2), repeat=4)))
        batch = self._batch(policy, actions[:, :2], actions[:, 2:])
        probabilities = np.prod(np.where(batch["actions"], batch["old"], 1 - batch["old"]), axis=(1, 2))
        return batch, probabilities

    def oracle(self, policy, direction=None):
        p = np.asarray(policy)
        m, values = 2 * p - 1, self.state_values(p)
        plus = p[0] ** 2 + (1 - p[0]) ** 2
        nu = np.array([1, self.gamma * plus, self.gamma * (1 - plus)]) / self.z
        c = np.array([m[0] * (0.15 + self.gamma * (values[1] - values[2]) / 2),
                      0.35 + 0.3 * m[1], -0.35 + 0.3 * m[2]])
        # Current authorized inputs are the full Markov public state.
        return _oracle_metrics(self, p, nu, c, values[0], direction, 0.0)


def _oracle_metrics(env, policy, occupancy, c, value, direction, permission_gap):
    q = np.zeros_like(c) if direction is None else np.asarray(direction)
    fisher = occupancy * 4 * policy * (1 - policy)
    local_energy = float(fisher @ (c * c))
    linear = float(fisher @ (c * q))
    norm = float(fisher @ (q * q))
    return {"expected_return": float(value), "per_agent_return": float(value / env.n),
            "local_direction": c, "occupancy": occupancy, "local_energy": local_energy,
            "permission_loss": float(permission_gap), "full_energy": local_energy + permission_gap,
            "direction_error": float(fisher @ ((c - q) ** 2)), "direction_norm": norm,
            "score": 2 * linear - norm, "normalized_derivative": linear,
            "return_derivative": env.n * env.z * linear}


def labels_for(batch, label, env, policy, baseline=None):
    """Adapt archived training/labels.py finite validation and immutable labels.

    The old single reward vector is extended to complete time-indexed MC returns
    and separately named exact labels; baseline is already fit and frozen.
    """
    if label == "exact_advantage":
        result = batch["exact_advantage"].copy()
    elif label == "mc_raw":
        result = batch["returns"].copy()
    elif label == "mc_exact_baseline":
        if isinstance(env, PairGame):
            result = batch["exact_advantage"].copy()
        else:
            result = batch["returns"] - env.state_values(policy)[batch["x"][:, :, 0]]
    elif label == "mc_learned_baseline":
        if baseline is None:
            raise ValueError("Learned baseline must be frozen before the training batch")
        if isinstance(env, PairGame):
            result = batch["returns"] - float(baseline[0])
        else:
            result = batch["returns"] - np.asarray(baseline)[batch["x"][:, :, 0]]
    else:
        raise ValueError(f"Unknown label protocol {label}")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite labels")
    result.setflags(write=False)
    return result


def fit_baseline(env, batch):
    """Least-squares MC baseline on an independent complete-episode batch.

    Pair uses a constant baseline; T uses its three public Markov states.
    Missing states fall back to the batch-wide mean and are counted explicitly.
    """
    if isinstance(env, PairGame):
        return np.array([batch["returns"].mean()]), 0
    x = batch["x"][:, :, 0]
    baseline = np.full(env.n_states, batch["returns"].mean())
    missing = 0
    for state in range(env.n_states):
        mask = x == state
        if mask.any():
            baseline[state] = batch["returns"][mask].mean()
        else:
            missing += 1
    return baseline, missing


def pair_source(config):
    """Construct the declared source; input-only variants have their own protocol."""
    return PairGame(n=int(config.get("n_agents", 4)),
                    plus_type_probability=float(config.get("type_probability", 0.6)),
                    variant=str(config.get("pair_variant", "normalized")),
                    source_n=int(config.get("source_agents", 4)),
                    interaction_strength=float(config.get("interaction_strength", 2.0)))
