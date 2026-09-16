"""Float64 MLP with explicit backpropagation and one flat parameter vector."""
import copy
import numpy as np
from .actor import TableActor, sigmoid


class MLP:
    def __init__(self, input_size, width=32, depth=2, seed=0):
        if any(type(v) is not int or v < 1 for v in (input_size, width, depth)):
            raise ValueError("Network dimensions must be positive integers")
        self.input_size, self.width, self.depth = input_size, width, depth
        sizes = [input_size]+[width]*depth+[1]
        self.shapes = []
        rng = np.random.default_rng(seed)
        chunks = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.shapes.extend([(a, b), (b,)])
            chunks.extend([rng.normal(0, np.sqrt(2/(a+b)), (a, b)).ravel(), np.zeros(b)])
        self.parameters = np.concatenate(chunks)

    def arrays(self):
        offset, arrays = 0, []
        for shape in self.shapes:
            size = int(np.prod(shape))
            arrays.append(self.parameters[offset:offset+size].reshape(shape))
            offset += size
        return arrays

    def forward(self, x):
        a = np.asarray(x, float)
        cache = [a]
        arrays = self.arrays()
        for layer in range(self.depth+1):
            z = a @ arrays[2*layer]+arrays[2*layer+1]
            a = np.tanh(z) if layer < self.depth else z
            cache.append(a)
        return a[:, 0], cache

    def backward(self, cache, gradient):
        delta = np.asarray(gradient, float)[:, None]
        arrays, gradients = self.arrays(), []
        for layer in range(self.depth, -1, -1):
            gradients[0:0] = [(cache[layer].T @ delta).ravel(), delta.sum(axis=0)]
            if layer:
                delta = (delta @ arrays[2*layer].T)*(1-cache[layer]**2)
        return np.concatenate(gradients)

    def state(self):
        return {"input_size": self.input_size, "width": self.width, "depth": self.depth,
                "parameters": self.parameters.tolist()}

    @classmethod
    def from_state(cls, state):
        model = cls(state["input_size"], state["width"], state["depth"])
        values = np.asarray(state["parameters"], float)
        if values.shape != model.parameters.shape or not np.isfinite(values).all():
            raise ValueError("Invalid MLP checkpoint")
        model.parameters[:] = values
        return model


class MLPActor:
    def __init__(self, n_types, beta=.02, width=32, depth=2, seed=0):
        TableActor(n_types, beta)  # shared probability-bound validation
        self.n_types, self.beta = n_types, beta
        self.network = MLP(n_types, width, depth, seed)

    @classmethod
    def initial(cls, n_types, beta=.02, width=32, depth=2, seed=0):
        model = cls(n_types, beta, width, depth, seed)
        # Match the table actor's starting probabilities for controlled comparisons.
        target = TableActor.initial(n_types, beta).logits
        _, cache = model.network.forward(np.eye(n_types))
        features = np.column_stack([cache[-2], np.ones(n_types)])
        solution = np.linalg.lstsq(features, target, rcond=None)[0]
        if not np.allclose(features @ solution, target, atol=1e-10):
            raise ValueError("MLP width cannot reproduce the common initial actor")
        arrays = model.network.arrays()
        arrays[-2][:, 0], arrays[-1][:] = solution[:-1], solution[-1]
        return model

    @property
    def parameters(self):
        return self.network.parameters

    @property
    def logits(self):
        return self.network.forward(np.eye(self.n_types))[0]

    def parameter_gradient(self, logit_gradient):
        _, cache = self.network.forward(np.eye(self.n_types))
        return self.network.backward(cache, logit_gradient)

    def table(self):
        p = self.beta/2+(1-self.beta)*sigmoid(self.logits)
        return np.column_stack([1-p, p])

    def probabilities(self, local_types):
        return self.table()[local_types]

    def copy(self):
        return copy.deepcopy(self)

    def state(self):
        return {"kind": "mlp_actor", "n_types": self.n_types, "beta": self.beta,
                "network": self.network.state()}

    @classmethod
    def from_state(cls, state):
        model = cls(state["n_types"], state["beta"])
        model.network = MLP.from_state(state["network"])
        if model.network.input_size != model.n_types:
            raise ValueError("Checkpoint input size mismatch")
        return model


class MLPDirection:
    def __init__(self, n_types, q_max=1., width=32, depth=2, seed=0):
        if not np.isfinite(q_max) or q_max <= 0:
            raise ValueError("Invalid direction bound")
        self.n_types, self.q_max = n_types, q_max
        self.network = MLP(n_types, width, depth, seed)
        self.network.arrays()[-2][:] = 0
        self.network.arrays()[-1][:] = 0

    @property
    def parameters(self):
        return self.network.parameters

    def table(self):
        z, _ = self.network.forward(np.eye(self.n_types))
        t = self.q_max*np.tanh(z)
        return np.column_stack([-t, t])

    def parameter_gradient(self, table_gradient):
        z, cache = self.network.forward(np.eye(self.n_types))
        grad = (table_gradient[:, 1]-table_gradient[:, 0])*self.q_max*(1-np.tanh(z)**2)
        return self.network.backward(cache, grad)

    def state(self):
        return {"kind": "mlp_direction", "q_max": self.q_max,
                "network": self.network.state(), "table": self.table().tolist()}


def load_actor(state):
    if state.get("kind") == "mlp_actor":
        return MLPActor.from_state(state)
    return TableActor.from_state(state)
