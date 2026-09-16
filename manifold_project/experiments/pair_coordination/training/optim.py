"""Small deterministic Adam optimizer for the finite table models."""
import numpy as np


class Adam:
    def __init__(self, shape, lr):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be positive")
        self.lr, self.t = lr, 0
        self.m, self.v = np.zeros(shape), np.zeros(shape)

    def step(self, parameters, gradient):
        if gradient.shape != parameters.shape or not np.isfinite(gradient).all():
            raise ValueError("Non-finite or incorrectly shaped gradient")
        self.t += 1
        self.m = .9*self.m + .1*gradient
        self.v = .999*self.v + .001*gradient**2
        update = self.lr*(self.m/(1-.9**self.t))/(np.sqrt(self.v/(1-.999**self.t))+1e-8)
        parameters -= update
        if not np.isfinite(parameters).all():
            raise FloatingPointError("参数更新后出现 NaN 或 Inf")
