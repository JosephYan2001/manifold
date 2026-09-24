"""P-TB: the one-step pair game with an authorized complete peer-type list.

This is a distinct observation protocol from the own-type finite P experiments.
The number of peer records reveals n; different n therefore has disjoint full
input support. Compatible execution is empirical extrapolation, not a proof of
the covered-input transport hypothesis.
"""
from __future__ import annotations

import numpy as np

from ...finite.environments import PairGame


class PairEntities:
    self_dim = 3
    entity_dim = 6
    action_dim = 2
    horizon = 1

    def __init__(self, config, n_agents=None, render_mode=None):
        self.config = dict(config)
        self.n_agents = int(n_agents if n_agents is not None else config.get("n_agents", 4))
        self.n = self.n_agents
        if self.n_agents < 2:
            raise ValueError("P-TB requires at least two agents")
        self.variant = str(config.get("pair_variant", "normalized")).replace("_", "-")
        if self.variant not in ("normalized", "fixed-edge"):
            raise ValueError("pair_variant must be normalized or fixed-edge")
        self.type_probability = float(config.get("type_positive_probability", 0.6))
        if not 0 < self.type_probability < 1:
            raise ValueError("Both declared types must have positive probability")
        # P and P-TB differ in observation rights, not in physical payoff. Keep
        # one reward implementation so the roster experiment cannot drift away
        # from its exact finite counterpart when either module is maintained.
        self.game = PairGame(n=self.n_agents, plus_type_probability=self.type_probability,
                             variant=self.variant, source_n=4)
        self.edge_weight = self.game.edge_weight
        self.state_dim = 2 * self.n_agents + 1
        self.t = 0
        self.initial_done = False
        self.end_reason = None
        self.types = None
        self.rng = np.random.default_rng()

    def _observation(self):
        onehot = np.eye(2, dtype=np.float32)[(self.types > 0).astype(int)]
        own = np.concatenate([onehot, np.full((self.n_agents, 1), 1 - self.t, dtype=np.float32)], 1)
        records = np.zeros((self.n_agents, self.n_agents - 1, self.entity_dim), dtype=np.float32)
        records[..., 0] = 1
        for i in range(self.n_agents):
            records[i, :, 4:6] = np.concatenate([onehot[:i], onehot[i + 1:]], 0)
        return {"self": own, "entities": records,
                "mask": np.ones((self.n_agents, self.n_agents - 1), dtype=bool)}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.types = np.where(self.rng.random(self.n_agents) < self.type_probability, 1, -1)
        self.t, self.end_reason = 0, None
        return self._observation(), self._info(0.0)

    def state(self):
        if self.types is None:
            raise RuntimeError("reset must be called before state")
        onehot = np.eye(2, dtype=np.float32)[(self.types > 0).astype(int)]
        return np.concatenate([onehot.reshape(-1), [1 - self.t]]).astype(np.float32)

    def team_reward(self, types, signed_actions):
        types, actions = np.asarray(types), np.asarray(signed_actions)
        if types.shape != (self.n_agents,) or actions.shape != (self.n_agents,):
            raise ValueError("Expected one type and one signed action per agent")
        if not np.isin(types, [-1, 1]).all() or not np.isin(actions, [-1, 1]).all():
            raise ValueError("Types and physical actions must be -1 or +1")
        return float(self.game.rewards((types > 0).astype(int), (actions > 0).astype(int)))

    def _info(self, reward):
        return {"elapsed_steps": self.t, "end_reason": self.end_reason,
                "n_agents": self.n_agents, "entity_count": self.n_agents - 1,
                "team_reward": float(reward), "per_agent_reward": float(reward / self.n_agents),
                "pair_variant": self.variant, "edge_weight": self.edge_weight}

    def step(self, actions):
        if self.types is None or self.end_reason is not None:
            raise RuntimeError("Task is inactive or already terminal; call reset before step")
        actions = np.asarray(actions)
        if actions.shape != (self.n_agents,) or not np.isin(actions, [0, 1]).all():
            raise ValueError("Expected one action index 0/1 per agent")
        reward = self.team_reward(self.types, 2 * actions.astype(int) - 1)
        self.t, self.end_reason = 1, "one_step_complete"
        return self._observation(), reward, True, False, self._info(reward)

    def protocol(self):
        return {"environment": "pair_entities", "version": "P-TB-v1", "n_agents": self.n_agents,
                "horizon": 1, "pair_variant": self.variant, "edge_weight": self.edge_weight,
                "type_positive_probability": self.type_probability,
                "local_bias": {"negative_type": -0.1, "positive_type": 0.2},
                "actions": [-1, 1], "reward_aggregation": "one_team_sum",
                "self_fields": ["own_type_negative", "own_type_positive", "remaining_time"],
                "entity_fields": ["is_peer", "unused_type", "zero_x", "zero_y", "peer_type_negative", "peer_type_positive"],
                "critic_fields": ["all_type_onehots", "remaining_time"],
                "observation": "own_type_and_complete_peer_type_multiset_without_actions_or_ids",
                "transport_classification": "B_empirical_disjoint_complete_list_support"}

    def render(self):
        raise NotImplementedError("P-TB is a one-step diagnostic with tabular output, not a physical renderer")

    def close(self):
        pass
