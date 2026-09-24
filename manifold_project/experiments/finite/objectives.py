"""Active adaptations of archived direction and forward-KL objectives.

Sources inside archive/legacy_experiments_20260923.zip:
  manifold_project/experiments/pair_coordination/training/direction_loss.py
  manifold_project/experiments/pair_coordination/training/actor_fit.py

The AN/SA vector formulas are retained. Arrays now keep an explicit time axis;
the returned coefficient gradient contracts q=(-c,c). Forward KL retains the
legacy probability/occupancy validation. Its derivative uses beta=0 because the
new common finite Actor is an ordinary Bernoulli logit, not the legacy mixed
probability-floor head. Optimization budgets and Actor classes are callers'
responsibility, avoiding dependencies on the old experiment runner.
"""
import numpy as np


def direction_records(c, batch, labels, method="AN"):
    if method not in ("AN", "SA"):
        raise ValueError("Unknown direction objective")
    x, actions, pplus = batch["x"], batch["actions"], batch["old"]
    table = np.column_stack([-c, c])
    q = table[x]
    p = np.stack([1 - pplus, pplus], axis=-1)
    a = np.asarray(labels, float)
    if a.shape != x.shape[:-1]:
        raise ValueError("Labels must have one entry per episode and time")
    scores = np.eye(2)[actions] - p
    f = np.sum(scores * q, axis=-1)
    fq = p * (q - np.sum(p * q, axis=-1, keepdims=True))
    if method == "AN":
        quadratic = np.sum(q * fq, axis=-1)
        grad = 2 * fq - 2 * a[..., None, None] * scores
        curvature = 4 * pplus * (1 - pplus)
    else:
        quadratic = f * f
        grad = 2 * (f - a[..., None])[..., None] * scores
        curvature = 4 * (actions - pplus) ** 2
    losses = quadratic - 2 * a[..., None] * f
    coefficient_gradient = grad[..., 1] - grad[..., 0]
    return -losses, coefficient_gradient, curvature


def forward_kl_and_logit_gradient(policy, target, weights):
    policy, target, weights = np.asarray(policy, float), np.asarray(target, float), np.asarray(weights, float)
    if (policy.ndim != 1 or target.shape != policy.shape or weights.shape != policy.shape
            or not np.isfinite(policy).all() or np.any((policy <= 0) | (policy >= 1))
            or not np.isfinite(target).all() or np.any((target <= 0) | (target >= 1))
            or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1)):
        raise ValueError("Invalid fitting target or occupancy weights")
    p = np.column_stack([1 - policy, policy])
    target_table = np.column_stack([1 - target, target])
    # Archived chain rule: weights*(p1-target1)/(p0*p1)*sigmoid'(logit).
    # beta=0 cancels the factors exactly, avoiding division near saturation.
    gradient = weights * (policy - target)
    loss = np.sum(weights[:, None] * target_table * (np.log(target_table) - np.log(p)))
    return float(loss), gradient
