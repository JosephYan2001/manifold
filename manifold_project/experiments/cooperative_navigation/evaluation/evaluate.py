import numpy as np
from ..training.collector import episode
from ..training.storage import seed_for


def evaluate(actor, config, condition, seed, purpose, count, n=None):
    n = n or config['n_agents']
    rows = [episode(actor, config,
                    seed_for('evaluation-reset', purpose, n, i),
                    seed_for('evaluation-action', condition, seed, purpose, n, i))
            for i in range(count)] if n == config['n_agents'] else [
                episode(actor, config, seed_for('evaluation-reset', purpose, n, i),
                        seed_for('evaluation-action', condition, seed, purpose, n, i), n=n)
                for i in range(count)]
    return {key: float(np.mean([r[key] for r in rows], dtype=np.float64)) for key in rows[0]}, rows
