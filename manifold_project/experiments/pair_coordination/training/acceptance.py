"""Finite-sample Hoeffding checks and an explicitly labeled empirical variant."""
import numpy as np


def alpha_per_check(total_alpha, rounds, attempts):
    if not 0 < total_alpha < 1 or rounds < 1 or attempts < 1:
        raise ValueError("Invalid multiple-testing budget")
    return total_alpha/(rounds*(1+attempts))


def check_mean(values, lower, upper, alpha, mode="hoeffding", threshold=0.):
    x = np.asarray(values, float)
    if (x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all()
            or not np.isfinite([lower, upper, threshold]).all() or lower > upper
            or not 0 < alpha < 1 or mode not in ("hoeffding", "empirical")):
        raise ValueError("Invalid acceptance inputs")
    if np.any(x < lower-1e-9) or np.any(x > upper+1e-9):
        raise ValueError("Observed statistic violates the asserted deterministic bound")
    mean = float(x.mean())
    radius = float((upper-lower)*np.sqrt(np.log(1/alpha)/(2*len(x))))
    decision_value = mean-radius if mode == "hoeffding" else mean
    return {"mean": mean, "radius": radius, "lower_confidence_bound": mean-radius,
            "decision_value": decision_value, "threshold": threshold,
            "accepted": bool(decision_value > threshold), "mode": mode,
            "alpha": alpha, "episodes": len(x)}


def direction_check(scores, advantage_bound, q_max, alpha, mode):
    return check_mean(scores, -4*advantage_bound*q_max-q_max*q_max,
                      4*advantage_bound*q_max, alpha, mode)


def return_check(old_rewards, candidate_rewards, reward_bound, alpha, mode):
    if np.shape(old_rewards) != np.shape(candidate_rewards):
        raise ValueError("Expected equally sized independent return batches")
    differences = np.asarray(candidate_rewards)-np.asarray(old_rewards)
    return check_mean(differences, -2*reward_bound, 2*reward_bound, alpha, mode)
