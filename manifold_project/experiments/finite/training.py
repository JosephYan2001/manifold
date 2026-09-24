"""Complete-episode finite learning, including all source decision costs."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from .diagnostics import configured_labels, empirical_update, settings
from .environments import PairGame, TwoStepGame, fit_baseline, labels_for, pair_source
from .numerics import episode_scores, kl, write_csv


def _checkpoint(path, policy, env, seed, steps, label, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, format_version=np.array("finite_v1"), policy_plus=policy,
                        environment=np.array(type(env).__name__), seed=np.array(seed),
                        source_steps_total=np.array(steps), label=np.array(label), method=np.array(method),
                        source_n=np.array(env.n))


def load_policy(path):
    """Load an Actor-only checkpoint; evaluation does not create/update a Critic."""
    if Path(path).suffix == '.json':
        import json
        from ..pair_coordination.models.mlp import load_actor
        return load_actor(json.loads(Path(path).read_text(encoding='utf-8'))['actor']).table()[:,1]
    with np.load(path, allow_pickle=False) as state:
        if str(state["format_version"]) != "finite_v1":
            raise ValueError("Unsupported finite checkpoint version")
        policy = state["policy_plus"].astype(float).copy()
        if not np.isfinite(policy).all() or np.any((policy <= 0) | (policy >= 1)):
            raise ValueError("Invalid checkpoint policy")
        return policy


def _report_evaluation(env, policy, method, label, seed, episodes, variant):
    # This report-only evaluation cannot affect checkpoint selection or training.
    batch = env.sample(policy, episodes, np.random.default_rng(seed))
    return [{"method": method, "label": label, "training_seed": seed - 10_000_000,
             "evaluation_seed": seed, "episode": i, "target_n": env.n,
             "variant": variant, "policy_mode": "stochastic", "ended_by": "task_complete",
             "discounted_return": float(batch["returns"][i, 0]),
             "undiscounted_return": float(batch["rewards"][i].sum()), "episode_length": env.horizon}
            for i in range(episodes)]


def train(config, output: Path, temporal=False):
    if not temporal:
        from ..pair_coordination.current_protocol import train as original_pair_train
        return original_pair_train(config, output)
    cfg = settings(config)
    env = TwoStepGame(gamma=float(config.get("gamma", 0.9))) if temporal else pair_source(config)
    experiment = "T-L" if temporal else "P-L"
    budget = int(config.get("budget", 65536))
    if budget < 1:
        raise ValueError("budget must be a positive number of joint environment steps")
    n = int(config.get("batch_episodes", min(max(cfg["sample_sizes"]), 128)))
    check_n = int(config.get("check_episodes", min(cfg["evaluation_episodes"], 128)))
    critic_n = int(config.get("critic_episodes", n))
    monitor_n = int(config.get("monitor_episodes", config.get("source_eval_episodes", min(check_n, 64))))
    if min(n, check_n, critic_n, monitor_n) < 1:
        raise ValueError("All training episode counts must be positive")
    label_groups = configured_labels(config, "learning_labels", ("exact", "mc") if temporal else ("mc",), learning=True)
    records, training_rows, evaluation_rows = [], [], []
    for label in label_groups:
        for method in cfg["methods"]:
            policy = env.initial_policy.copy()
            rng = np.random.default_rng(cfg["seed"])
            counts = {"training": 0, "critic": 0, "direction_check": 0, "return_check": 0, "monitor": 0}
            total, update, accepts, direction_rejections, return_rejections = 0, 0, 0, 0, 0
            best_value, best_policy, best_steps = -np.inf, None, 0
            start = perf_counter()
            # Reserve one candidate check and one monitor. Further backtracks fit
            # only if the remaining budget permits complete old/new episodes.
            minimum = env.horizon * (n + (critic_n if label == "mc_learned_baseline" else 0)
                                     + (check_n if method != "DA" and cfg["direction_check"] else 0)
                                     + (2 * check_n if cfg["return_check"] else 0) + monitor_n)
            if budget < minimum:
                raise ValueError(f"{experiment}/{label}/{method}: budget {budget} is below one complete update ({minimum}); reduce batch/check/critic/monitor episodes")
            while total + minimum <= budget:
                baseline, missing = None, 0
                if label == "mc_learned_baseline":
                    baseline_batch = env.sample(policy, critic_n, rng)
                    baseline, missing = fit_baseline(env, baseline_batch)
                    counts["critic"] += critic_n * env.horizon
                batch = env.sample(policy, n, rng)
                counts["training"] += n * env.horizon
                total = sum(counts.values())
                labels = labels_for(batch, label, env, policy, baseline)
                old = policy.copy()
                candidate, direction, spent, details = empirical_update(
                    env, old, batch, labels, label, baseline, method, cfg, rng,
                    check_episodes=check_n, max_source_steps=budget - total - monitor_n * env.horizon)
                for purpose, value in spent.items():
                    counts[purpose] += value
                policy = candidate
                accepts += int(details["accepted"])
                direction_rejections += int(details["reason"] == "direction_rejected")
                return_rejections += int(details["reason"] == "return_rejected")
                monitor = env.sample(policy, monitor_n, rng)
                counts["monitor"] += monitor_n * env.horizon
                total, update = sum(counts.values()), update + 1
                monitor_return = float(monitor["returns"][:, 0].mean())
                if monitor_return > best_value:
                    best_value, best_policy, best_steps = monitor_return, policy.copy(), total
                exact = env.oracle(policy)
                row = {"experiment": experiment, "method": method, "label": label,
                       "oracle_label_group": label == "exact_advantage", "seed": cfg["seed"], "update": update,
                       "source_steps_total": total, **{f"source_steps_{k}": v for k, v in counts.items()},
                       "source_complete_episodes": total // env.horizon,
                       "agent_decisions": total * env.n, "budget": budget,
                       "training_return": float(batch["returns"][:, 0].mean()),
                       "monitor_return": monitor_return, "exact_source_return_diagnostic": exact["expected_return"],
                       "actual_return_gain_diagnostic": exact["expected_return"] - env.oracle(old)["expected_return"],
                       "old_new_kl": kl(old, policy, env.oracle(old)["occupancy"]),
                       "baseline_kind": "constant_independent_mc" if isinstance(env, PairGame) else "public_state_tabular_independent_mc",
                       "baseline_missing_states": missing,
                       "entropy": float(np.sum(exact["occupancy"] * (-policy * np.log(policy) - (1 - policy) * np.log1p(-policy)))),
                       "seconds": perf_counter() - start, **details}
                if label == "exact_advantage":
                    row["baseline_kind"] = "exact_joint_advantage_oracle_diagnostic"
                elif label == "mc_raw":
                    row["baseline_kind"] = "zero_baseline"
                if baseline is not None:
                    predicted = baseline[0] if isinstance(env, PairGame) else baseline[batch["x"][:, :, 0]]
                    residual = batch["returns"] - predicted
                    row["critic_validation_mse"] = float(np.mean(residual ** 2))
                    row["label_mse_against_exact_joint_advantage"] = float(np.mean((labels - batch["exact_advantage"]) ** 2))
                    row["critic_explained_variance"] = float(1 - np.var(residual) / max(np.var(batch["returns"]), 1e-15))
                if direction is not None:
                    truth = env.oracle(old, direction)
                    row.update({"direction_error": truth["direction_error"], "direction_norm": truth["direction_norm"],
                                "exact_direction_score_diagnostic": truth["score"]})
                training_rows.append(row)
                if config.get("verbose", True):
                    score = "n/a" if direction is None else f"{details.get('direction_check_score', float('nan')):.4g}"
                    print(f"{experiment} {label} {method} seed={cfg['seed']} round={update} source_steps={total}/{budget} "
                          f"train_R={row['training_return']:.4g} monitor_R={monitor_return:.4g} "
                          f"direction={score} result={details['reason']}", flush=True)
            checkpoint_dir = output / label / f"{method}_s{cfg['seed']}"
            final_path, best_path = checkpoint_dir / "final.npz", checkpoint_dir / "best_source.npz"
            _checkpoint(final_path, policy, env, cfg["seed"], total, label, method)
            if best_policy is not None:
                _checkpoint(best_path, best_policy, env, cfg["seed"], best_steps, label, method)
            records.append({"experiment": experiment, "method": method, "label": label,
                            "row_kind": "training_summary",
                            "oracle_label_group": label == "exact_advantage", "seed": cfg["seed"],
                            "source_steps_total": total, **{f"source_steps_{k}": v for k, v in counts.items()},
                            "budget": budget, "unused_budget": budget - total, "updates": update,
                            "accepted_updates": accepts, "direction_rejections": direction_rejections,
                            "return_rejections": return_rejections,
                            "final_source_return_diagnostic": env.oracle(policy)["expected_return"],
                            "best_source_empirical_monitor_return": best_value,
                            "final_checkpoint": str(final_path.relative_to(output)),
                            "best_source_checkpoint": str(best_path.relative_to(output)),
                            "seconds": perf_counter() - start,
                            "source_report_only_steps": cfg["evaluation_episodes"] * env.horizon,
                            "target_report_only_steps": 0 if temporal else 3 * len(cfg["target_sizes"]) * cfg["evaluation_episodes"],
                            "checkpoint_selection": "final_primary_best_by_independent_source_monitor_only"})
            evaluation_rows.extend(_report_evaluation(env, policy, method, label, cfg["seed"] + 10_000_000,
                                                     cfg["evaluation_episodes"], "source"))
            if not temporal:
                for variant in ("normalized", "fixed-edge", "composition"):
                    for target_n in cfg["target_sizes"]:
                        target = PairGame(target_n, 0.4 if variant == "composition" else env.plus_type_probability,
                                          variant, env.n, env.interaction_strength)
                        evaluation_rows.extend(_report_evaluation(target, policy, method, label, cfg["seed"] + 10_000_000,
                                                                 cfg["evaluation_episodes"], variant))
                        records.append({"experiment": "P-L-final-transfer", "method": method, "label": label,
                                        "row_kind": "frozen_transfer", "source_cost_scope": "shared_with_training_summary_not_additional",
                                        "seed": cfg["seed"], "variant": variant, "source_n": env.n, "target_n": target_n,
                                        "frozen_actor_return_diagnostic": target.oracle(policy)["expected_return"],
                                        "source_steps_total": total, "target_training_steps": 0,
                                        "coverage_note": "input_support_covered_final_performance_not_local_update_theorem"})
    write_csv(output / "training.csv", training_rows)
    write_csv(output / "evaluation_episodes.csv", evaluation_rows)
    return records, training_rows, evaluation_rows
