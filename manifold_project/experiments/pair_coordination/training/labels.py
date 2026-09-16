"""One label per complete episode; critic fitting is performed beforehand."""
import numpy as np


def make_labels(batch, critic=None):
    labels = np.asarray(batch["team_rewards"], float).copy()
    if critic is not None:
        labels -= critic.predict(batch["local_types"])
    if not np.isfinite(labels).all():
        raise ValueError("Non-finite labels")
    labels.setflags(write=False)
    return labels
