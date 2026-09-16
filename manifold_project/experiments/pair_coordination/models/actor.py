"""Shared binary actor: one trainable logit per local type."""
import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0., -np.asarray(z, dtype=float)))


class TableActor:
    def __init__(self, n_types, beta=0.02, logits=None):
        if type(n_types) is not int or n_types < 1 or not 0 < beta < 1:
            raise ValueError("Require n_types >= 1 and 0 < beta < 1")
        self.n_types, self.beta = n_types, float(beta)
        self.logits = np.zeros(n_types) if logits is None else np.asarray(logits, float).copy()
        if self.logits.shape != (n_types,) or not np.isfinite(self.logits).all():
            raise ValueError("Invalid actor logits")

    @classmethod
    def initial(cls, n_types, beta=0.02):
        probability = np.linspace(.35, .65, n_types) if n_types > 1 else np.array([.5])
        raw = (probability - beta/2)/(1-beta)
        if np.any(raw <= 0) or np.any(raw >= 1):
            raise ValueError("beta cannot represent the initial policy")
        return cls(n_types, beta, np.log(raw/(1-raw)))

    def table(self):
        p1 = self.beta/2 + (1-self.beta)*sigmoid(self.logits)
        return np.column_stack([1-p1, p1])

    @property
    def parameters(self):
        return self.logits

    def parameter_gradient(self, logit_gradient):
        return logit_gradient

    def probabilities(self, local_types):
        return self.table()[local_types]

    def copy(self):
        return TableActor(self.n_types, self.beta, self.logits)

    def state(self):
        return {"kind": "binary_type_table", "n_types": self.n_types,
                "beta": self.beta, "logits": self.logits.tolist()}

    @classmethod
    def from_state(cls, state):
        if state.get("kind") != "binary_type_table":
            raise ValueError("Unsupported actor kind")
        return cls(state["n_types"], state["beta"], state["logits"])
