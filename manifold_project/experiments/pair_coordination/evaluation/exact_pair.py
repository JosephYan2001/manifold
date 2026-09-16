"""Exact expectations for small pair-coordination games; evaluation only."""

from itertools import product
import numpy as np

from ..envs.pair_coordination import PairCoordinationEnv, validate_policy


def exponential_target(policy, direction, eta):
    p, q = np.asarray(policy, float), np.asarray(direction, float)
    if (p.ndim != 2 or q.shape != p.shape or not np.isfinite(p).all()
            or np.any(p <= 0) or not np.isfinite(q).all() or not np.isfinite(eta)
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Expected a positive policy, same-shaped finite direction and finite eta")
    logits = np.log(p) + eta * q
    if not np.isfinite(logits).all():
        raise ValueError("Exponent step overflow; reduce eta/direction")
    logits -= logits.max(axis=1, keepdims=True)
    tilted = np.exp(logits)
    tilted /= tilted.sum(axis=1, keepdims=True)
    if np.any(tilted == 0):
        raise ValueError("Exponent step underflow; reduce eta/direction")
    return tilted


def exact_evaluate(config, policy, direction=None, max_events=1_000_000):
    """Enumerate types/actions, returning pooled-agent Fisher quantities.

    This first version uses iid types. Zero-mass types have local_direction=0
    as a storage convention; supported_types marks where it is identified.
    Complexity O((2 * number_of_supported_types)**n_agents); guard before allocation.
    """
    p = validate_policy(config, policy)
    support = np.flatnonzero(np.asarray(config.type_probs) > 0)
    event_count = (len(support) * 2) ** config.n_agents
    if type(max_events) is not int or max_events < 1 or event_count > max_events:
        raise ValueError(f"Exact enumeration requires {event_count} events; limit={max_events}")
    q = np.zeros_like(p) if direction is None else np.asarray(direction, dtype=float)
    if (q.shape != p.shape or not np.isfinite(q).all()
            or not np.allclose(q.sum(axis=1), 0, rtol=0, atol=1e-10)):
        raise ValueError("direction must be finite, zero-sum, and have shape (n_types, 2)")
    n = config.n_agents
    env = PairCoordinationEnv(config)
    fisher = np.array([np.diag(row) - np.outer(row, row) for row in p])
    inverse = np.array([np.linalg.pinv(f, rcond=1e-12) for f in fisher])
    joint_actions = np.array(list(product(range(2), repeat=n)), dtype=np.int64)
    type_probs = np.asarray(config.type_probs)
    nu = np.zeros(config.n_types)
    m_weighted = np.zeros_like(p)
    full_energy = linear = value = advantage_square = regression_square = mass = 0.0
    eye = np.eye(2)
    for types in product(support, repeat=n):
        x = np.asarray(types)
        ph = float(np.prod(type_probs[x]))
        pa = np.prod(p[x[None, :], joint_actions], axis=1)
        rewards = env.reward_batch(x, joint_actions)
        vh = float(pa @ rewards)
        advantage = rewards - vh
        value += ph * vh
        mass += ph * pa.sum()
        advantage_square += ph * (pa @ (advantage ** 2))
        for i, xi in enumerate(x):
            scores = eye[joint_actions[:, i]] - p[xi]
            mh = (pa * advantage) @ scores
            vf = inverse[xi] @ mh
            weight = ph / n
            nu[xi] += weight
            m_weighted[xi] += weight * mh
            full_energy += weight * (vf @ fisher[xi] @ vf)
            linear += weight * (mh @ q[xi])
            regression_square += weight * (pa @ ((advantage - scores @ q[xi]) ** 2))
    conditional_m = np.divide(m_weighted, nu[:, None], out=np.zeros_like(p),
                              where=nu[:, None] > 0)
    local = np.einsum("xij,xj->xi", inverse, conditional_m)

    def norm(a):
        return float(np.einsum("x,xi,xij,xj->", nu, a, fisher, a))

    local_energy = norm(local)
    return {"expected_return": float(value), "probability_mass": float(mass),
            "event_count": event_count, "local_type_probs": nu,
            "supported_types": nu > 0, "fisher": fisher,
            "conditional_m": conditional_m, "local_direction": local,
            "full_energy": float(full_energy), "local_energy": local_energy,
            "information_gap": float(full_energy - local_energy),
            "direction_error": norm(q - local), "direction_norm": norm(q),
            "score": float(2 * linear - norm(q)),
            "return_derivative": float(n * linear),
            "advantage_second_moment": float(advantage_square),
            "regression_risk": float(regression_square)}


def closed_form_expected_return(config, policy):
    """Exact iid team return in O(n_types**2), without joint enumeration.

    Each local term has the same mean; the n*(n-1)/2 pair terms
    share a mean and carry interaction_strength/(n-1).
    Evaluation only: this score must not drive training acceptance.
    """
    p = validate_policy(config, policy)
    weighted_means = np.asarray(config.type_probs) * (p[:, 1] - p[:, 0])
    local = weighted_means @ np.asarray(config.local_bias)
    pair = weighted_means @ np.asarray(config.pair_payoff) @ weighted_means
    return float(config.n_agents * (local + config.interaction_strength * pair / 2))


def closed_form_local_direction(config, policy):
    """Independent analytic oracle for iid types and normalized complete graph."""
    p = validate_policy(config, policy)
    action_means = p @ np.array([-1., 1.])
    signal = (np.asarray(config.local_bias) + config.interaction_strength
              * (np.asarray(config.pair_payoff) @ (np.asarray(config.type_probs) * action_means)))
    return signal[:, None] * np.array([-1., 1.])
