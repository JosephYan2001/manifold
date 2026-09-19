import numpy as np


class History:
    def __init__(self, n, obs_dim, length, horizon, include_time=True):
        self.frames = np.zeros((n, length, obs_dim + 6), dtype=np.float32)
        self.horizon = horizon
        self.include_time = include_time

    def append(self, obs, previous_action, t):
        self.frames[:, :-1] = self.frames[:, 1:].copy()
        previous = np.zeros((len(obs), 5), dtype=np.float32)
        if previous_action is not None:
            previous[np.arange(len(obs)), previous_action] = 1
        self.frames[:, -1] = np.concatenate([obs, previous, np.ones((len(obs), 1))], axis=-1)
        x = self.frames.reshape(len(obs), -1).copy()
        if self.include_time:
            x = np.concatenate([x, np.full((len(obs), 1), t/self.horizon)], axis=-1)
        return x.astype(np.float32)
