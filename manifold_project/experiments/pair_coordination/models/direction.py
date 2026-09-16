"""A bounded zero-sum direction, indexed by local type only."""
import numpy as np


class TableDirection:
    def __init__(self, n_types, q_max=1.0):
        if type(n_types) is not int or n_types < 1 or not np.isfinite(q_max) or q_max <= 0:
            raise ValueError("Invalid direction size or bound")
        self.q_max = float(q_max)
        self.parameters = np.zeros(n_types)

    def table(self):
        t = self.q_max * np.tanh(self.parameters)
        return np.column_stack([-t, t])

    def parameter_gradient(self, table_gradient):
        dt = self.q_max * (1 - np.tanh(self.parameters)**2)
        return (table_gradient[:, 1] - table_gradient[:, 0]) * dt

    def state(self):
        return {"kind": "bounded_type_direction", "q_max": self.q_max,
                "parameters": self.parameters.tolist(), "table": self.table().tolist()}
