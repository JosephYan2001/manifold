"""One-step cooperative game with private iid types and binary actions.

Action indices 0/1 have physical values -1/+1. Observation row i is only
agent i's type, not the whole type vector. No Gym/OpenSpiel dependency.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def _integer_vector(values, size, upper, name):
    array = np.asarray(values)
    if (array.shape != (size,) or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array >= upper)):
        raise ValueError(f"{name} must contain {size} integer indices in [0, {upper})")
    return array.astype(np.int64, copy=True)


@dataclass(frozen=True)
class PairConfig:
    n_agents: int = 4
    type_probs: tuple = (0.6, 0.4)
    local_bias: tuple = (-0.2, 0.3)
    pair_payoff: tuple = ((1.0, -0.5), (-0.5, 1.0))
    interaction_strength: float = 0.7

    def __post_init__(self):
        if type(self.n_agents) is not int or self.n_agents < 2:
            raise ValueError("n_agents must be an integer >= 2")
        p = np.asarray(self.type_probs, dtype=float)
        b = np.asarray(self.local_bias, dtype=float)
        c = np.asarray(self.pair_payoff, dtype=float)
        if (p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all()
                or np.any(p < 0) or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)):
            raise ValueError("type_probs must be nonnegative and sum to one")
        if b.shape != p.shape or not np.isfinite(b).all():
            raise ValueError("local_bias must have one finite entry per type")
        if (c.shape != (len(p), len(p)) or not np.isfinite(c).all()
                or not np.allclose(c, c.T, rtol=0, atol=1e-12)):
            raise ValueError("pair_payoff must be a finite symmetric type-by-type table")
        rho = float(self.interaction_strength)
        if not np.isfinite(rho) or rho < 0:
            raise ValueError("interaction_strength must be finite and nonnegative")
        # Immutable copies prevent later edits to caller-owned lists or arrays.
        object.__setattr__(self, "type_probs", tuple(float(x) for x in p))
        object.__setattr__(self, "local_bias", tuple(float(x) for x in b))
        object.__setattr__(self, "pair_payoff", tuple(tuple(float(x) for x in row) for row in c))
        object.__setattr__(self, "interaction_strength", rho)

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def n_types(self):
        return len(self.type_probs)

    @property
    def reward_bound(self):
        """Deterministic absolute bound on the one-step team reward."""
        return (self.n_agents * max(abs(x) for x in self.local_bias)
                + self.interaction_strength * self.n_agents / 2
                * max(abs(x) for row in self.pair_payoff for x in row))


class PairCoordinationEnv:
    """reset -> (local_types, info); step -> standard five-element tuple.

    Reward is one shared scalar. The collector must pass only local_types[i]
    to actor i. state() exposes the joint types for centralized training.
    reset(types=...) is an explicit diagnostic override, not a training reset.
    """

    n_actions = 2
    action_values = (-1, 1)

    def __init__(self, config=None, seed=None):
        self.config = config if config is not None else PairConfig()
        self._rng = np.random.default_rng(seed)
        self._types = None
        self._done = True

    def reset(self, *, seed=None, types=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if types is None:
            sampled = self._rng.choice(self.config.n_types, self.config.n_agents,
                                       p=self.config.type_probs)
        else:
            sampled = _integer_vector(types, self.config.n_agents,
                                      self.config.n_types, "types")
        self._types = sampled.copy()
        self._done = False
        return self._types.copy(), {"diagnostic_reset": types is not None}

    def state(self):
        if self._types is None:
            raise RuntimeError("Call reset before requesting state")
        return self._types.copy()

    def reward(self, types, actions):
        """Pure deterministic team reward, also used by exact enumeration."""
        x = _integer_vector(types, self.config.n_agents, self.config.n_types, "types")
        a = _integer_vector(actions, self.config.n_agents, 2, "actions")
        return float(self.reward_batch(x, a[None, :])[0])

    def reward_batch(self, types, joint_actions):
        x = _integer_vector(types, self.config.n_agents, self.config.n_types, "types")
        a = np.asarray(joint_actions)
        if (a.ndim != 2 or a.shape[1] != self.config.n_agents
                or a.dtype.kind not in "iu" or np.any(a < 0) or np.any(a > 1)):
            raise ValueError("joint_actions must have shape (batch, n_agents), integer 0/1")
        signs = 2.0 * a - 1.0
        result = signs @ np.asarray(self.config.local_bias)[x]
        pair = np.asarray(self.config.pair_payoff)
        weight = self.config.interaction_strength / (self.config.n_agents - 1)
        for i in range(self.config.n_agents):
            for j in range(i + 1, self.config.n_agents):
                result += weight * pair[x[i], x[j]] * signs[:, i] * signs[:, j]
        return result

    def step(self, actions):
        if self._done:
            raise RuntimeError("Call reset before step; episodes have exactly one step")
        reward = self.reward(self._types, actions)
        self._done = True
        return self._types.copy(), reward, True, False, {}


def validate_policy(config, policy):
    p = np.asarray(policy, dtype=float)
    if (p.shape != (config.n_types, 2) or not np.isfinite(p).all()
            or np.any(p <= 0)
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("policy must be a strictly positive normalized (n_types, 2) table")
    return p.copy()


def sample_episodes(config, policy, n_episodes, seed=0):
    """Source-only iid episodes; no exact advantage or oracle in the batch.

    Arrays retain episode/agent axes. local_types[j, i] is actor i's input.
    policy is snapshotted so old_probs remain fixed for subsequent fitting.
    """
    if type(n_episodes) is not int or n_episodes <= 0:
        raise ValueError("n_episodes must be a positive integer")
    p = validate_policy(config, policy)
    streams = np.random.SeedSequence(seed).spawn(2)
    env = PairCoordinationEnv(config, seed=streams[0])
    action_rng = np.random.default_rng(streams[1])
    types = np.empty((n_episodes, config.n_agents), dtype=np.int64)
    actions = np.empty_like(types)
    rewards = np.empty(n_episodes)
    for episode in range(n_episodes):
        x, _ = env.reset()
        a = (action_rng.random(config.n_agents) >= p[x, 0]).astype(np.int64)
        _, rewards[episode], _, _, _ = env.step(a)
        types[episode], actions[episode] = x, a
    return {"local_types": types, "actions": actions, "team_rewards": rewards,
            "old_probs": p[types], "weights": np.full(types.shape, 1.0 / types.size)}
