"""Centralized table baseline, fit only on an independent source batch."""
import numpy as np


class TableCritic:
    def __init__(self, n_agents, n_types, bound):
        if n_types**n_agents > 1_000_000:
            raise ValueError("Critic table exceeds one million states")
        if not np.isfinite(bound) or bound < 0:
            raise ValueError("Invalid baseline bound")
        self.n_agents, self.n_types, self.bound = n_agents, n_types, float(bound)
        self.values = np.zeros(n_types**n_agents)

    def indices(self, states):
        x = np.asarray(states)
        if (x.ndim != 2 or x.shape[1] != self.n_agents or x.dtype.kind not in "iu"
                or np.any(x < 0) or np.any(x >= self.n_types)):
            raise ValueError("Critic requires valid complete pre-action type vectors")
        return x @ (self.n_types ** np.arange(self.n_agents))

    def fit(self, states, rewards):
        indices = self.indices(states)
        y = np.asarray(rewards, float)
        if y.shape != indices.shape or not np.isfinite(y).all():
            raise ValueError("Invalid critic labels")
        counts = np.bincount(indices, minlength=len(self.values))
        sums = np.bincount(indices, weights=y, minlength=len(self.values))
        self.values = np.clip(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0),
                              -self.bound, self.bound)
        self.values.setflags(write=False)
        return self

    def predict(self, states):
        return self.values[self.indices(states)].copy()
