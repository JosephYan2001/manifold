"""Fixed-base experiments; exact quantities are diagnostics, never selectors."""
from __future__ import annotations

from itertools import product
from time import perf_counter
import hashlib

import numpy as np

from .acceptance import direction_check, return_check
from .environments import PairGame, TwoStepGame, labels_for, pair_source
from .numerics import (direction_terms, episode_scores, fit_actor, fit_direct,
                       fit_direction, kl, logit, occupancy, project_coefficient, tilted)


def settings(config):
    seed = int(config.get("seed", 100))
    result = {"seed": seed, "sample_sizes": list(config.get("sample_sizes", [32, 128, 512, 2048])),
            "data_seeds": list(config.get("data_seeds", range(1000, 1020))),
            "direction_steps": int(config.get("direction_steps", 200)),
            "direction_lr": float(config.get("direction_lr", 0.03)),
            "actor_lr": float(config.get("actor_lr", 0.03)),
            "q_bound": float(config.get("q_bound", 4.0)),
            "fit_steps": list(config.get("fit_steps", [1, 10, 100, 1000])),
            "evaluation_episodes": int(config.get("evaluation_episodes", 512)),
            "step_sizes": list(config.get("step_sizes", [0.2, 0.1, 0.05, 0.025])),
            "methods": list(config.get("methods", ["AN", "SA", "DA"])),
            "target_sizes": list(config.get("target_sizes", [2, 6])),
            "direction_check": bool(config.get("direction_check", True)),
            "return_check": bool(config.get("return_check", True)),
            "direction_threshold": float(config.get("direction_threshold", 0.0)),
            "return_tolerance": float(config.get("return_tolerance", 0.0))}
    result["actor_fit_steps"] = int(config.get("actor_fit_steps", max(result["fit_steps"])))
    result["check_episodes"] = int(config.get("check_episodes", result["evaluation_episodes"]))
    for key in ("sample_sizes", "fit_steps", "step_sizes", "data_seeds"):
        if not result[key]:
            raise ValueError(f"{key} must not be empty")
    if any(int(n) != n or n < 1 for n in result["sample_sizes"]):
        raise ValueError("sample_sizes must contain positive whole episode counts")
    if any(int(n) != n or n < 0 for n in result["fit_steps"]):
        raise ValueError("fit_steps must contain nonnegative integers")
    if any(not np.isfinite(eta) or eta <= 0 for eta in result["step_sizes"]):
        raise ValueError("step_sizes must be positive and finite")
    if min(result["direction_lr"], result["actor_lr"], result["q_bound"], result["evaluation_episodes"]) <= 0:
        raise ValueError("Learning rates, q_bound and evaluation_episodes must be positive")
    if result["direction_steps"] < 0:
        raise ValueError("direction_steps must be nonnegative")
    if not result["methods"] or set(result["methods"]) - {"AN", "SA", "DA"}:
        raise ValueError("Finite methods must be a nonempty subset of AN, SA and DA")
    if result["actor_fit_steps"] < 0 or result["check_episodes"] < 1:
        raise ValueError("actor_fit_steps must be nonnegative and check_episodes positive")
    if any(not isinstance(n, int) or n < 2 for n in result["target_sizes"]):
        raise ValueError("Finite target_sizes must contain integers >= 2")
    return result


def configured_labels(config, key, default, learning=False):
    aliases = {"exact": "exact_advantage", "mc_zero": "mc_raw",
               "mc": "mc_learned_baseline" if learning else "mc_exact_baseline"}
    result = [aliases.get(label, label) for label in config.get(key, default)]
    supported = {"exact_advantage", "mc_raw", "mc_learned_baseline" if learning else "mc_exact_baseline"}
    if not result or set(result) - supported:
        raise ValueError(f"Unsupported {key}: {result}")
    return list(dict.fromkeys(result))


def _fit(batch, labels, env, method, cfg):
    return fit_direction(batch, labels, env.n_states, method, cfg["direction_steps"],
                         cfg["direction_lr"], cfg["q_bound"])


def _columns(prefix, values):
    return {f"{prefix}_{i}": float(value) for i, value in enumerate(values)}


def dataset_hash(batch):
    """Adapt archived experiments/mechanisms.py dataset_hash for time batches.

    Retains contiguous array hashing, adds old probabilities/weights and shape
    tags so paired objectives can verify their actual shared source data.
    """
    digest = hashlib.sha256()
    for key in ("x", "actions", "rewards", "old", "weights"):
        value = np.ascontiguousarray(batch[key])
        digest.update(f"{key}:{value.dtype}:{value.shape}".encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def estimation(config):
    cfg, env = settings(config), pair_source(config)
    methods = [method for method in cfg["methods"] if method != "DA"]
    if not methods:
        raise ValueError("P-E estimates direction and requires AN or SA; DA has no q")
    old, rows = env.initial_policy.copy(), []
    for n in cfg["sample_sizes"]:
        for seed in cfg["data_seeds"]:
            batch = env.sample(old, int(n), np.random.default_rng(seed))
            fingerprint = dataset_hash(batch)
            validation = env.sample(old, cfg["evaluation_episodes"], np.random.default_rng(seed + 1_000_000))
            for label in configured_labels(config, "labels", ("exact", "mc_zero")):
                labels, check_labels = labels_for(batch, label, env, old), labels_for(validation, label, env, old)
                for method in methods:
                    start = perf_counter()
                    c, fitted = _fit(batch, labels, env, method, cfg)
                    oracle = env.oracle(old, c)
                    scores = episode_scores(c, validation, check_labels, "AN")
                    _, gradient, curvature = direction_terms(c, validation, check_labels, method)
                    gradients = np.zeros((len(scores), env.n_states))
                    for x in range(env.n_states):
                        gradients[:, x] = np.sum(gradient * (validation["x"] == x) * validation["weights"], axis=(1, 2)) * len(scores)
                    # A second variance diagnostic keeps the reference q fixed
                    # across AN/SA; learned-q variances above mix optimization
                    # outcomes with the respective estimator's noise.
                    reference_q = env.oracle(old)["local_direction"]
                    fixed_scores = episode_scores(reference_q, validation, check_labels, method)
                    _, fixed_gradient, _ = direction_terms(reference_q, validation, check_labels, method)
                    fixed_gradients = np.zeros_like(gradients)
                    for x in range(env.n_states):
                        fixed_gradients[:, x] = np.sum(fixed_gradient * (validation["x"] == x) * validation["weights"], axis=(1, 2)) * len(scores)
                    row = {"experiment": "P-E", "method": method, "label": label, "data_seed": seed,
                           "dataset_sha256": fingerprint,
                           "sample_episodes": n, "source_steps_total": n + len(scores),
                           "source_steps_training": n, "source_steps_validation": len(scores),
                           "direction_error": oracle["direction_error"], "direction_norm": oracle["direction_norm"],
                           "exact_score": oracle["score"], "validation_score": float(scores.mean()),
                           "validation_score_variance": float(scores.var(ddof=1)) if len(scores) > 1 else 0.0,
                           "validation_gradient_trace_variance": float(gradients.var(axis=0, ddof=1).sum()) if len(scores) > 1 else 0.0,
                           "fixed_oracle_q_score_variance": float(fixed_scores.var(ddof=1)) if len(scores) > 1 else 0.0,
                           "fixed_oracle_q_gradient_trace_variance": float(fixed_gradients.var(axis=0, ddof=1).sum()) if len(scores) > 1 else 0.0,
                           "curvature_min": float(curvature.min()), "curvature_max": float(curvature.max()),
                           "optimization_seconds": perf_counter() - start, **fitted, **_columns("q_coefficient", c)}
                    rows.append(row)
    return rows


def actor_realization(config):
    cfg, env = settings(config), pair_source(config)
    old, seed = env.initial_policy.copy(), cfg["seed"]
    n = int(config.get("sample_episodes", max(cfg["sample_sizes"])))
    batch = env.sample(old, n, np.random.default_rng(seed))
    validation = env.sample(old, cfg["evaluation_episodes"], np.random.default_rng(seed + 1_000_000))
    weights, validation_weights = occupancy(batch, env.n_states), occupancy(validation, env.n_states)
    label = "mc_raw"
    labels = labels_for(batch, label, env, old)
    direction_method = next((method for method in ("AN", "SA") if method in cfg["methods"]), None)
    # ONE learned direction, shared by every fit. AN is the default reference;
    # a SA-only request uses one SA direction, never a separate q per fit count.
    c = _fit(batch, labels, env, direction_method, cfg)[0] if direction_method else np.zeros(env.n_states)
    eta = float(config.get("eta", cfg["step_sizes"][0]))
    target, rows = tilted(old, c, eta), []
    old_value = env.oracle(old)["expected_return"]
    for restricted in (False, True):
        projection = project_coefficient(old, c, env.type_probabilities, restricted)
        projection_oracle = env.oracle(old, projection)
        for steps in cfg["fit_steps"]:
            start = perf_counter()
            candidate = fit_actor(old, target, weights, steps, cfg["actor_lr"], restricted)
            actual = (logit(candidate) - logit(old)) / (2 * eta)
            exact = env.oracle(old, actual)
            row = {"experiment": "P-A", "method": direction_method, "direction_method": direction_method, "label": label,
                   "seed": seed, "actor_class": "shared_offset" if restricted else "table", "fit_steps": steps,
                   "sample_episodes": n, "source_steps_total": n + len(validation["x"]), "eta": eta,
                   "ideal_return_gain": env.oracle(target)["expected_return"] - old_value,
                   "actual_return_gain": env.oracle(candidate)["expected_return"] - old_value,
                   "learned_direction_derivative": env.oracle(old, c)["return_derivative"],
                   "projected_direction_derivative": projection_oracle["return_derivative"],
                   "projection_reference": "exact_population_Fisher_projection_diagnostic",
                   "realized_path_derivative": exact["return_derivative"],
                   "forward_kl_training": kl(target, candidate, weights),
                   "forward_kl_validation": kl(target, candidate, validation_weights),
                   "forward_kl_exact": kl(target, candidate, env.type_probabilities),
                   "old_new_kl": kl(old, candidate, env.type_probabilities),
                   "direction_realization_residual": float(np.sum(env.type_probabilities * 4 * old * (1 - old) * (actual - c) ** 2)),
                   "seconds": perf_counter() - start, **_columns("learned_q", c), **_columns("candidate_p", candidate)}
            if direction_method:
                rows.append(row)
            if "DA" not in cfg["methods"]:
                continue
            direct = fit_direct(old, batch, labels, steps, cfg["actor_lr"], restricted)
            direct_path = (logit(direct) - logit(old)) / 2
            rows.append({"experiment": "P-A", "method": "DA", "label": label, "seed": seed,
                         "actor_class": row["actor_class"], "fit_steps": steps,
                         "sample_episodes": n, "source_steps_total": n + len(validation["x"]),
                         "actual_return_gain": env.oracle(direct)["expected_return"] - old_value,
                         "realized_path_derivative": env.oracle(old, direct_path)["return_derivative"],
                         "old_new_kl": kl(old, direct, env.type_probabilities),
                         "direction_status": "not_applicable", **_columns("candidate_p", direct)})
    return rows


def empirical_update(env, old, batch, labels, label_name, baseline, method, cfg, rng,
                     check_episodes=None, max_source_steps=None):
    """Select solely with fresh source episodes; exact functions are absent here."""
    check_n = int(check_episodes or cfg["check_episodes"])
    spent = {"direction_check": 0, "return_check": 0}
    direction_score, direction, candidate = None, None, old.copy()
    details = {"accepted": False, "reason": "return_rejected", "eta": 0.0,
               "attempts": 0, "forward_kl_training": None}

    def fits(extra):
        return max_source_steps is None or sum(spent.values()) + extra <= max_source_steps

    if method in ("AN", "SA"):
        direction, fit = _fit(batch, labels, env, method, cfg)
        details.update(fit)
        if cfg["direction_check"]:
            if not fits(env.horizon * check_n):
                details["reason"] = "budget_exhausted"
                return candidate, direction, spent, details
            check = env.sample(old, check_n, rng)
            spent["direction_check"] += check_n * env.horizon
            scores = episode_scores(direction, check, labels_for(check, label_name, env, old, baseline))
            label_bound = env.label_return_bound * (1 if label_name == "mc_raw" else 2)
            check_result = direction_check(scores, label_bound, cfg["q_bound"], .05, "empirical",
                                           threshold=cfg["direction_threshold"])
            direction_score = check_result["mean"]
            details["direction_check_score"] = direction_score
            details["direction_check_mode"] = check_result["mode"]
            if not check_result["accepted"]:
                details["reason"] = "direction_rejected"
                return candidate, direction, spent, details
        direct_logit_delta = None
    elif method == "DA":
        direct = fit_direct(old, batch, labels, cfg["actor_fit_steps"], cfg["actor_lr"])
        direct_logit_delta = logit(direct) - logit(old)
    else:
        raise ValueError(f"Unknown method {method}")

    for eta in cfg["step_sizes"]:
        if cfg["return_check"] and not fits(2 * env.horizon * check_n):
            details["reason"] = "budget_exhausted"
            break
        if direction is not None:
            target = tilted(old, direction, eta)
            proposal = fit_actor(old, target, occupancy(batch, env.n_states), cfg["actor_fit_steps"], cfg["actor_lr"])
            details["forward_kl_training"] = kl(target, proposal, occupancy(batch, env.n_states))
        else:
            proposal = tilted(old, direct_logit_delta / 2, eta)
        if not cfg["return_check"]:
            candidate = proposal
            details.update({"accepted": True, "reason": "accepted_return_check_disabled", "eta": float(eta),
                            "attempts": details["attempts"] + 1})
            break
        # Common random numbers reduce paired differences; both policies are charged.
        check_seed = int(rng.integers(0, 2**63 - 1))
        old_eval = env.sample(old, check_n, np.random.default_rng(check_seed))
        new_eval = env.sample(proposal, check_n, np.random.default_rng(check_seed))
        spent["return_check"] += 2 * check_n * env.horizon
        difference = new_eval["returns"][:, 0] - old_eval["returns"][:, 0]
        check_result = return_check(old_eval["returns"][:, 0], new_eval["returns"][:, 0],
                                    env.return_bound, .05, "empirical",
                                    threshold=-cfg["return_tolerance"], inclusive=True)
        details.update({"attempts": details["attempts"] + 1, "return_check_gain": float(difference.mean()),
                        "return_check_mode": check_result["mode"],
                        "return_check_standard_error": float(difference.std(ddof=1) / np.sqrt(check_n)) if check_n > 1 else 0.0})
        if check_result["accepted"]:
            candidate = proposal
            details.update({"accepted": True, "reason": "accepted_empirical", "eta": float(eta)})
            break
    return candidate, direction, spent, details


def transfer(config):
    cfg, source = settings(config), pair_source(config)
    old, seed = source.initial_policy.copy(), cfg["seed"]
    n = int(config.get("sample_episodes", max(cfg["sample_sizes"])))
    batch = source.sample(old, n, np.random.default_rng(seed))
    labels = labels_for(batch, "mc_raw", source, old)
    rows = []
    for method in cfg["methods"]:
        candidate, learned, spent, acceptance = empirical_update(source, old, batch, labels, "mc_raw", None,
                                                               method, cfg, np.random.default_rng(seed + 100_000))
        actual = (logit(candidate) - logit(old)) / (2 * acceptance["eta"]) if acceptance["eta"] else np.zeros_like(old)
        source_true = source.oracle(old)["local_direction"]
        source_fisher = source.type_probabilities * 4 * old * (1 - old)
        for variant in ("normalized", "fixed-edge", "composition"):
            for target_n in sorted(set([source.n, *cfg["target_sizes"]])):
                target_p = 0.4 if variant == "composition" and target_n != source.n else source.plus_type_probability
                target = PairGame(target_n, target_p, variant, source.n, source.interaction_strength)
                truth, actual_metrics = target.oracle(old), target.oracle(old, actual)
                nu = target.type_probabilities
                fisher = nu * 4 * old * (1 - old)
                kappa = float(np.max(nu / source.type_probabilities))
                drift = float(fisher @ ((truth["local_direction"] - source_true) ** 2))
                row = {"experiment": "P-T", "method": method, "seed": seed, "variant": variant,
                       "source_n": source.n, "target_n": target_n, "target_plus_type_probability": target_p,
                       "coverage_class": "A_finite_supported", "support_covered": True,
                       "density_ratio": kappa, "neighbor_weight": (target_n - 1) * target.edge_weight,
                       "semantic_drift": drift, "target_local_energy": truth["local_energy"],
                       "target_path_derivative": actual_metrics["return_derivative"],
                       "target_normalized_path_derivative": actual_metrics["normalized_derivative"],
                       "old_return": truth["expected_return"], "return": target.oracle(candidate)["expected_return"],
                       "return_gain": target.oracle(candidate)["expected_return"] - truth["expected_return"],
                       "source_steps_total": n + sum(spent.values()), "source_steps_training": n,
                       "source_steps_direction_check": spent["direction_check"], "source_steps_return_check": spent["return_check"],
                       **acceptance, **_columns("target_true_coefficient", truth["local_direction"]),
                       **_columns("frozen_actor_p", candidate)}
                if learned is not None:
                    delta_source = float(source_fisher @ ((learned - source_true) ** 2))
                    fit_residual = float(source_fisher @ ((actual - learned) ** 2))
                    bound = np.sqrt(kappa) * (np.sqrt(delta_source) + np.sqrt(fit_residual)) + np.sqrt(drift)
                    row.update({"source_direction_error": delta_source, "source_realization_residual": fit_residual,
                                "target_direction_error": float(fisher @ ((actual - truth["local_direction"]) ** 2)),
                                "transport_error_upper_bound": bound ** 2,
                                "normalized_derivative_lower_bound": truth["local_energy"] - np.sqrt(truth["local_energy"]) * bound,
                                "bound_object": "specified_source_logit_path_through_finite_candidate"})
                else:
                    row["direction_status"] = "not_applicable"
                rows.append(row)
    return rows


def information(config):
    probabilities = np.array([0.2, 0.5, 0.9])
    means = 2 * probabilities - 1
    mappings = {
        "full_multiset": lambda own, others: (own, *sorted(others)),
        "mean_and_count": lambda own, others: (own, sum(others), len(others)),
        "sorted_first_two": lambda own, others: (own, *sorted(others)[:2]),
        "drop_independent_nuisance": lambda own, others: (own, *sorted(others)),
    }
    rows = []
    full_energy = 0.0
    cases = []
    for types in product(range(3), repeat=4):
        own, others = types[0], types[1:]
        c = float(means[list(others)].sum() / 3)
        fisher = 4 * probabilities[own] * (1 - probabilities[own])
        weight = 1 / 81
        full_energy += weight * fisher * c * c
        cases.append((own, others, c, fisher, weight))
    for name, mapping in mappings.items():
        groups = {}
        for own, others, c, fisher, weight in cases:
            group = groups.setdefault(mapping(own, others), [0.0, 0.0])
            group[0] += weight
            group[1] += weight * c
        error = local = 0.0
        for own, others, c, fisher, weight in cases:
            mass, moment = groups[mapping(own, others)]
            predicted = moment / mass
            error += weight * fisher * (predicted - c) ** 2
            local += weight * fisher * predicted ** 2
        rows.append({"experiment": "P-I", "method": name, "input_states": len(groups),
                     "permission_loss": 0.0, "preprocessing_loss": error,
                     "raw_information_energy": full_energy, "local_direction_energy": local,
                     "decomposition_residual": abs(full_energy - local - error),
                     "learned_direction_status": "exact_input_mapping_only",
                     "source_steps_total": 0, "oracle_only": True})
    return rows


def boundaries(config):
    env, rng = PairGame(), np.random.default_rng(settings(config)["seed"])
    p, c = np.full(2, 0.5), np.array([0.4, -0.2])
    batch = env.sample(p, 32, rng)
    labels = labels_for(batch, "mc_raw", env, p)
    an, gan, _ = direction_terms(c, batch, labels, "AN")
    sa, gsa, _ = direction_terms(c, batch, labels, "SA")
    source = env.oracle(env.initial_policy)["local_direction"]
    normalized = PairGame(n=2).oracle(env.initial_policy)["local_direction"]
    fixed = PairGame(n=2, variant="fixed-edge").oracle(env.initial_policy)["local_direction"]
    return [
        {"experiment": "P-B", "case": "uniform_binary_equivalence", "score_max_difference": float(abs(an - sa).max()),
         "gradient_max_difference": float(abs(gan - gsa).max()), "source_steps_total": 32,
         "interpretation": "special_point_not_general_AN_SA_ranking"},
        {"experiment": "P-B", "case": "zero_first_order_positive_second_order", "old_p": 0.5, "candidate_p": 0.6,
         "true_first_derivative": 0.0, "old_return": 0.0, "candidate_return": 0.04,
         "q_coefficient": 0.5, "nonzero_q_score": -0.25,
         "empirical_or_exact_check": "exact_direction_check_rejects_nonzero_q",
         "source_steps_total": 0, "interpretation": "R=a1*a2_zero_gradient_cannot_capture_coordinated_second_order_gain"},
        {"experiment": "P-B", "case": "same_source_different_target_mechanisms",
         "source_minus_type_coefficient": float(source[0]), "normalized_target_minus_coefficient": float(normalized[0]),
         "fixed_edge_target_minus_coefficient": float(fixed[0]), "source_steps_total": 0,
         "interpretation": "identical_four_agent_source_does_not_identify_unrestricted_target_mechanism"},
    ]


def temporal_variance(config):
    cfg, env = settings(config), TwoStepGame(gamma=float(config.get("gamma", 0.9)))
    methods = [method for method in cfg["methods"] if method != "DA"]
    if not methods:
        raise ValueError("T-V compares direction estimators and requires AN or SA")
    old, c = env.initial_policy.copy(), np.array([0.4, 0.9, -0.6])
    exact_batch, mass = env.enumerate(old)
    derivative = env.oracle(old, c)["return_derivative"]
    eps = 1e-5
    fd = (env.oracle(tilted(old, c, eps))["expected_return"] - env.oracle(tilted(old, c, -eps))["expected_return"]) / (2 * eps)
    rows = []
    for label in configured_labels(config, "labels", ("exact", "mc", "mc_zero")):
        exact_labels = labels_for(exact_batch, label, env, old)
        an_scores = episode_scores(c, exact_batch, exact_labels, "AN")
        sa_scores = episode_scores(c, exact_batch, exact_labels, "SA")
        quadratic_noise = an_scores - sa_scores
        noise_mean = float(mass @ quadratic_noise)
        noise_variance = float(mass @ ((quadratic_noise - noise_mean) ** 2))
        score_noise_covariance = float(mass @ ((an_scores - mass @ an_scores) * (quadratic_noise - noise_mean)))
        variance_difference = float(mass @ ((an_scores - mass @ an_scores) ** 2) - mass @ ((sa_scores - mass @ sa_scores) ** 2))
        for method in methods:
            score_terms, _, _ = direction_terms(c, exact_batch, exact_labels, method)
            time_scores = score_terms.mean(axis=2) * np.array([1, env.gamma])[None, :] / env.z
            scores = time_scores.sum(axis=1)
            mean = float(mass @ scores)
            variance = float(mass @ ((scores - mean) ** 2))
            time_means = mass @ time_scores
            cross_covariance = float(np.sum(mass * (time_scores[:, 0] - time_means[0]) * (time_scores[:, 1] - time_means[1])))
            label_error = float(np.sum(mass[:, None] * np.array([1, env.gamma])[None, :] / env.z * (exact_labels - exact_batch["exact_advantage"]) ** 2))
            common = {"experiment": "T-V", "method": method, "label": label,
                      "exact_episode_score_mean": mean, "exact_episode_score_variance": variance,
                      "exact_raw_episode_score_variance": variance * env.z ** 2,
                      "discount_normalizer": env.z, "cross_time_covariance": cross_covariance,
                      "quadratic_noise_episode_variance": noise_variance,
                      "AN_score_quadratic_noise_covariance": score_noise_covariance,
                      "variance_identity_residual": variance_difference - (2 * score_noise_covariance - noise_variance),
                      "true_return_derivative": derivative, "finite_difference_derivative": fd,
                      "label_mse_against_exact_joint_advantage": label_error}
            for n in cfg["sample_sizes"]:
                for seed in cfg["data_seeds"]:
                    batch = env.sample(old, int(n), np.random.default_rng(seed))
                    labels = labels_for(batch, label, env, old)
                    sample_scores = episode_scores(c, batch, labels, method)
                    learned, fit = _fit(batch, labels, env, method, cfg)
                    rows.append({**common, "data_seed": seed, "sample_episodes": n,
                                 "dataset_sha256": dataset_hash(batch),
                                 "source_steps_total": 2 * n, "sample_score_mean": float(sample_scores.mean()),
                                 "sample_score_variance": float(sample_scores.var(ddof=1)) if n > 1 else 0.0,
                                 "learned_direction_error": env.oracle(old, learned)["direction_error"], **fit})
    return rows
