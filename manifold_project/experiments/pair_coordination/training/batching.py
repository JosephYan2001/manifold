"""按完整团队回合打乱和分批，保留回合内全部机器人。"""
import numpy as np


def episode_batches(batch, labels, batch_size, epochs, seed):
    n = len(batch["local_types"])
    size = batch_size or n
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, size):
            indices = order[start:start+size]
            part = {k: v[indices].copy() for k, v in batch.items()}
            part["weights"] = np.full(part["local_types"].shape, 1/part["local_types"].size)
            yield epoch+1, part, None if labels is None else labels[indices]
