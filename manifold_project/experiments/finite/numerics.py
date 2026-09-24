"""Small, inspectable optimizers for finite mechanism experiments.

Policies contain P(action=+1); a direction coefficient c denotes q=(-c,c).
Every sample weight includes the agent mean and normalized time discount.
No oracle is used by the optimizers or empirical acceptance checks.
"""
from __future__ import annotations

import numpy as np

from ..common.reporting import write_csv
from .objectives import direction_records, forward_kl_and_logit_gradient
from ..pair_coordination.training.optim import Adam


def sigmoid(logits):
    """Stable archived pair_coordination/models/actor.py sigmoid."""
    logits = np.asarray(logits, dtype=float)
    return np.exp(-np.logaddexp(0.0, -logits))


def logit(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError("All finite policies must have strictly positive action probabilities")
    return np.log(p) - np.log1p(-p)


def tilted(p, c, eta):
    result = sigmoid(logit(p) + 2.0 * float(eta) * np.asarray(c))
    # A numerical guard, common to all methods, not a learned probability floor.
    return np.clip(result, 1e-12, 1 - 1e-12)


def kl(p, r, weights):
    p, r, weights = np.asarray(p), np.asarray(r), np.asarray(weights)
    r = np.clip(r, 1e-12, 1 - 1e-12)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(weights * (p * np.log(p / r) + (1 - p) * np.log((1 - p) / (1 - r)))))


def occupancy(batch, n_states):
    return np.bincount(batch["x"].ravel(), weights=batch["weights"].ravel(), minlength=n_states)


def direction_terms(c, batch, labels, method="AN"):
    return direction_records(np.asarray(c), batch, labels, method)


def episode_scores(c, batch, labels, method="AN"):
    scores, _, _ = direction_terms(np.asarray(c), batch, labels, method)
    # batch weights sum to 1; multiply by E for one independent episode score.
    return (scores * batch["weights"]).sum(axis=(1, 2)) * len(scores)


def direction_gradient(c, batch, labels, method="AN"):
    scores, gradient, _ = direction_terms(np.asarray(c), batch, labels, method)
    grad = np.bincount(batch["x"].ravel(),
                       weights=(gradient * batch["weights"]).ravel(), minlength=len(c))
    return -float(np.sum(batch["weights"] * scores)), grad


def fit_direction(batch, labels, n_states, method, steps=200, lr=0.03, q_bound=4.0):
    """Same zero initialization, smooth bound and optimizer for AN and SA."""
    if q_bound <= 0 or steps < 0:
        raise ValueError("q_bound must be positive and optimization steps nonnegative")
    from ..pair_coordination.models.direction import TableDirection
    from ..pair_coordination.training.direction_loss import fit_direction as original_fit
    n = batch['x'].shape[-1]
    # Flatten time only for the full-batch optimizer. Original episode axes stay
    # intact in sampling, hashes, variance and bootstrap calculations.
    old = batch['old'].reshape(-1, n)
    adapted = dict(local_types=batch['x'].reshape(-1,n), actions=batch['actions'].reshape(-1,n),
                   old_probs=np.stack([1-old,old],-1), weights=batch['weights'].reshape(-1,n))
    model = TableDirection(n_states, q_bound)
    original_fit(model, adapted, np.asarray(labels).reshape(-1),
                 'analytic' if method == 'AN' else 'sampled', int(steps), lr, max(1,int(steps)))
    c = model.table()[:, 1]
    loss, _ = direction_gradient(c, batch, labels, method)
    return c, {"direction_loss": loss, "direction_steps": int(steps), "q_bound": q_bound}


def fit_actor(old, target, weights, steps=100, lr=0.03, restricted=False):
    """Forward KL; a restricted actor can only add one shared logit offset.

    Both classes contain the same old policy. They differ only in the candidate
    parameter tangent space, avoiding an unequal starting-policy comparison.
    """
    old, target, weights = np.asarray(old), np.asarray(target), np.asarray(weights)
    if not restricted:
        from ..pair_coordination.models.actor import TableActor
        from ..pair_coordination.training.actor_fit import fit_actor as original_fit
        beta = 1e-14
        raw = (old-beta/2)/(1-beta)
        actor = TableActor(len(old), beta, logit(raw))
        adapted = dict(local_types=np.arange(len(old))[None,:], weights=weights[None,:])
        candidate, _ = original_fit(actor,np.column_stack([1-target,target]),adapted,int(steps),lr)
        return candidate.table()[:,1]
    design = np.ones((len(old), 1)) if restricted else np.eye(len(old))
    offset = np.zeros(design.shape[1])
    optimizer = Adam(offset.shape, lr)
    for _ in range(int(steps)):
        candidate = sigmoid(logit(old) + design @ offset)
        _, gradient = forward_kl_and_logit_gradient(candidate, target, weights)
        optimizer.step(offset, design.T @ gradient)
    candidate = np.clip(sigmoid(logit(old) + design @ offset), 1e-12, 1 - 1e-12)
    return candidate


def project_coefficient(old, c, weights, restricted=False):
    if not restricted:
        return np.asarray(c).copy()
    fisher_weight = np.asarray(weights) * 4 * old * (1 - old)
    value = float(np.dot(fisher_weight, c) / max(fisher_weight.sum(), 1e-15))
    return np.full_like(c, value)


def fit_direct(old, batch, labels, steps=100, lr=0.03, restricted=False):
    """Unclipped old-policy-ratio surrogate, not stale score/log-prob epochs.

    The first gradient at the old actor equals the corresponding score gradient.
    Its occupancy normalization is explicit and the same as the direction loss.
    """
    old = np.asarray(old)
    design = np.ones((len(old), 1)) if restricted else np.eye(len(old))
    offset = np.zeros(design.shape[1])
    optimizer = Adam(offset.shape, lr)
    x, a = batch["x"], batch["actions"]
    old_action_probability = np.where(a, batch["old"], 1 - batch["old"])
    for _ in range(int(steps)):
        p = sigmoid(logit(old) + design @ offset)
        ratio = np.where(a, p[x], 1 - p[x]) / old_action_probability
        contributions = -batch["weights"] * labels[..., None] * ratio * (a - p[x])
        gradient = np.bincount(x.ravel(), weights=contributions.ravel(), minlength=len(old))
        optimizer.step(offset, design.T @ gradient)
    return np.clip(sigmoid(logit(old) + design @ offset), 1e-12, 1 - 1e-12)
