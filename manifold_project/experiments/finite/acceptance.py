"""Migrated bounded episode checks, with explicitly empirical training use.

Source: archive/legacy_experiments_20260923.zip ::
manifold_project/experiments/pair_coordination/training/acceptance.py.
Preserves validation, episode means, Hoeffding calculation and strict default
comparison. Adaptations: configurable thresholds, an explicitly named inclusive
tie rule for the new return protocol, and paired old/new episode terminology.
No high-confidence training claim is enabled by merely providing this helper.
"""
import numpy as np


def check_mean(values, lower, upper, alpha, mode="hoeffding", threshold=0., inclusive=False):
    x = np.asarray(values, float)
    if (x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all()
            or not np.isfinite([lower, upper, threshold]).all() or lower > upper
            or not 0 < alpha < 1 or mode not in ("hoeffding", "empirical")):
        raise ValueError("Invalid acceptance inputs")
    if np.any(x < lower - 1e-9) or np.any(x > upper + 1e-9):
        raise ValueError("Observed statistic violates the asserted deterministic bound")
    mean = float(x.mean())
    radius = float((upper - lower) * np.sqrt(np.log(1 / alpha) / (2 * len(x))))
    decision_value = mean - radius if mode == "hoeffding" else mean
    accepted = decision_value >= threshold if inclusive else decision_value > threshold
    return {"mean": mean, "radius": radius, "lower_confidence_bound": mean - radius,
            "decision_value": decision_value, "threshold": threshold,
            "accepted": bool(accepted), "mode": mode,
            "alpha": alpha, "episodes": len(x)}


def direction_check(scores, advantage_bound, q_max, alpha, mode, threshold=0.):
    return check_mean(scores, -4 * advantage_bound * q_max - q_max * q_max,
                      4 * advantage_bound * q_max, alpha, mode, threshold)


def return_check(old_rewards, candidate_rewards, reward_bound, alpha, mode,
                 threshold=0., inclusive=False):
    if np.shape(old_rewards) != np.shape(candidate_rewards):
        raise ValueError("Expected equally sized return batches of independent episode pairs")
    differences = np.asarray(candidate_rewards) - np.asarray(old_rewards)
    return check_mean(differences, -2 * reward_bound, 2 * reward_bound,
                      alpha, mode, threshold, inclusive)
