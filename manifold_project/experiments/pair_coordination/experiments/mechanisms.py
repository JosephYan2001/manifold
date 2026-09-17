"""Finite-game mechanisms. Oracle quantities are restricted to these experiments."""
from dataclasses import replace
import hashlib
import math
import time
import numpy as np
from ..envs.pair_coordination import PairConfig, sample_episodes, validate_policy
from ..evaluation.exact_pair import (closed_form_expected_return, closed_form_local_direction,
                                    exponential_target)
from ..models.actor import TableActor
from ..models.direction import TableDirection
from ..training.actor_fit import kl_and_gradient
from ..training.direction_loss import fit_direction, loss_and_gradient, episode_scores
from ..training.optim import Adam
from ..training.acceptance import alpha_per_check, direction_check, return_check


def fisher_norm(config, policy, values):
    p = validate_policy(config, policy)
    a = np.asarray(values, float)
    return float(np.asarray(config.type_probs) @ ((p*a*a).sum(-1)-(p*a).sum(-1)**2))


def direction_metrics(config, policy, q):
    v = closed_form_local_direction(config, policy)
    error = fisher_norm(config, policy, q-v)
    return {"direction_error": error, "score": fisher_norm(config, policy, v)-error}


def exact_advantages(config, policy, batch):
    """R - E[R | joint types], not R - unconditional policy value."""
    x = batch["local_types"]
    means = (policy[:, 1]-policy[:, 0])[x]
    baseline = (np.asarray(config.local_bias)[x]*means).sum(-1)
    pairs = np.asarray(config.pair_payoff)
    for i in range(config.n_agents):
        for j in range(i+1, config.n_agents):
            baseline += (config.interaction_strength/(config.n_agents-1)
                         * pairs[x[:, i], x[:, j]] * means[:, i]*means[:, j])
    return batch["team_rewards"]-baseline


def dataset_hash(batch):
    digest = hashlib.sha256()
    for key in ("local_types", "actions", "team_rewards"):
        digest.update(np.ascontiguousarray(batch[key]).tobytes())
    return digest.hexdigest()


def direction_estimation(source, config):
    rows = []
    for actor_id, probabilities in enumerate(config["frozen_probabilities"]):
        policy = np.column_stack((1-np.asarray(probabilities), probabilities))
        for size in config["sample_sizes"]:
            for seed in config["data_seeds"]:
                batch = sample_episodes(source, policy, size, seed)
                fingerprint = dataset_hash(batch)
                for label_name in ("exact_advantage", "return"):
                    labels = (exact_advantages(source, policy, batch) if label_name == "exact_advantage"
                              else batch["team_rewards"])
                    for method in ("analytic", "sampled"):
                        model = TableDirection(source.n_types, config["q_max"])
                        fit_direction(model, batch, labels, method, config["direction_steps"],
                                      config["direction_lr"], config["direction_steps"])
                        q = model.table()
                        rows.append({"actor_id": actor_id, "p0": float(policy[0, 1]),
                                     "p1": float(policy[1, 1]), "sample_size": size, "seed": seed,
                                     "label": label_name, "method": method, "dataset_sha256": fingerprint,
                                     "q0": float(q[0, 1]), "q1": float(q[1, 1]),
                                     "training_loss": loss_and_gradient(q, batch, labels, method)[0],
                                     "updates": config["direction_steps"],
                                     "amplitude_fraction": float(abs(q).max()/config["q_max"]),
                                     **direction_metrics(source, policy, q)})
                print(f"P-M1 actor={actor_id} N={size} seed={seed}", flush=True)
    return rows


def kl(policy, target, weights):
    return float(np.sum(np.asarray(weights)[:, None]*target*(np.log(target)-np.log(policy))))


def actor_realization(source, config):
    rows = []
    policy = np.full((source.n_types, 2), .5)
    weights = np.asarray(source.type_probs)
    direction = closed_form_local_direction(source, policy)
    initial_return = closed_form_expected_return(source, policy)
    # dJ/deta = n * <v,q>_F at eta=0, q=v here.
    derivative = source.n_agents*fisher_norm(source, policy, direction)
    beta = config["beta"]
    for eta in config["step_sizes"]:
        target = exponential_target(policy, direction, eta)
        target_gain = closed_form_expected_return(source, target)-initial_return
        for kind in ("table", "tied_logit"):
            values = np.zeros(source.n_types if kind == "table" else 1)
            optimizer = Adam(values.shape, config["actor_lr"])
            optimal_p1 = np.clip(target[:, 1] if kind == "table" else
                                 np.repeat(weights @ target[:, 1], source.n_types), beta/2, 1-beta/2)
            optimum = np.column_stack((1-optimal_p1, optimal_p1))
            reference_kl = kl(optimum, target, weights)
            for step in range(1, max(config["fit_steps"])+1):
                actor = TableActor(source.n_types, beta, values if kind == "table" else
                                   np.repeat(values[0], source.n_types))
                _, gradient = kl_and_gradient(actor, target, weights)
                optimizer.step(values, gradient if kind == "table" else np.array([gradient.sum()]))
                if step not in config["fit_steps"]:
                    continue
                actor = TableActor(source.n_types, beta, values if kind == "table" else
                                   np.repeat(values[0], source.n_types))
                actual_kl = kl(actor.table(), target, weights)
                rows.append({"eta": eta, "actor_class": kind, "updates": step,
                             "target_gain": target_gain,
                             "actor_gain": closed_form_expected_return(source, actor.table())-initial_return,
                             "fit_kl": actual_kl, "best_class_kl": reference_kl,
                             "optimization_excess_kl": actual_kl-reference_kl,
                             "target_min_probability": float(target.min()),
                             "probability_floor": beta/2,
                             "target_difference_quotient": target_gain/eta,
                             "exact_derivative": derivative})
    return rows


def transport_metrics(source, target, policy, q):
    if source.n_types != target.n_types or source.local_bias != target.local_bias or source.pair_payoff != target.pair_payoff:
        raise ValueError("Transport requires identical type/action semantics and reward functions")
    if min(source.type_probs) <= 0:
        raise ValueError("Source must cover all target types")
    v0, ve = closed_form_local_direction(source, policy), closed_form_local_direction(target, policy)
    error0 = fisher_norm(source, policy, q-v0)
    error = fisher_norm(target, policy, q-ve)
    shift = fisher_norm(target, policy, ve-v0)
    kappa = float(max(np.asarray(target.type_probs)/source.type_probs))
    # Lambda = rho * iid type distribution; signed-measure variation is L1,
    # not half-L1 probability total variation. R_int=max |pair_payoff|.
    variation = float(np.abs(target.interaction_strength*np.asarray(target.type_probs)
                             - source.interaction_strength*np.asarray(source.type_probs)).sum())
    shift_bound = float(np.max(np.abs(source.pair_payoff))**2*variation**2)
    bound = (math.sqrt(max(0, kappa*error0))+math.sqrt(max(0, shift)))**2
    mechanism_bound = (math.sqrt(max(0, kappa*error0))+math.sqrt(shift_bound))**2
    return {"source_error": error0, "target_error": error, "direction_shift": shift,
            "coverage_kappa": kappa, "neighbor_variation": variation, "shift_bound": shift_bound,
            "transport_bound": bound, "mechanism_transport_bound": mechanism_bound,
            "bound_holds": error <= bound+1e-10 and shift <= shift_bound+1e-10,
            "team_return": closed_form_expected_return(target, policy),
            "return_per_agent": closed_form_expected_return(target, policy)/target.n_agents}


def direction_transport(source, config, snapshots):
    targets = [("source", source)]
    targets += [(f"agents_{n}", replace(source, n_agents=n)) for n in (2, 6, 8)]
    targets += [(f"composition_{p}", replace(source, type_probs=(p, 1-p))) for p in (.3, .9)]
    targets += [(f"strength_{rho}", replace(source, interaction_strength=rho)) for rho in (0., .35, 1.4)]
    selected = [r for r in snapshots if int(r["actor_id"]) == 1 and r["method"] == "analytic"
                and r["label"] == "return" and int(r["sample_size"]) == config["transport_sample_size"]]
    if len(selected) != len(config["data_seeds"]):
        raise ValueError("P-M3 requires one matching P-M1 source snapshot for each data seed")
    rows = []
    for row in selected:
        p1 = np.array([float(row["p0"]), float(row["p1"])])
        policy = np.column_stack((1-p1, p1))
        t = np.array([float(row["q0"]), float(row["q1"])])
        q = np.column_stack((-t, t))
        for name, target in targets:
            rows.append({"target": name, "seed": int(row["seed"]), "n_agents": target.n_agents,
                         "type0_probability": target.type_probs[0], "rho": target.interaction_strength,
                         "source_dataset_sha256": row["dataset_sha256"],
                         **transport_metrics(source, target, policy, q)})
    return rows


def second_order(config):
    source = PairConfig(n_agents=2, type_probs=(1.,), local_bias=(0.,),
                        pair_payoff=((1.,),), interaction_strength=1.)
    uniform = np.full((1, 2), .5)
    zero = float(np.abs(closed_form_local_direction(source, uniform)).max())
    rows = []
    for epsilon in config["epsilons"]:
        policy = np.array([[.5-epsilon, .5+epsilon]])
        gain = closed_form_expected_return(source, policy)
        rows.append({"epsilon": epsilon, "direction_max_abs": zero, "actual_gain": gain,
                     "expected_gain": 4*epsilon**2,
                     "identity_holds": zero < 1e-12 and abs(gain-4*epsilon**2) < 1e-12})
    return rows


def check_rules(source, config, training, progress=None):
    policy = np.full((source.n_types, 2), .5)
    v = closed_form_local_direction(source, policy)
    alpha = alpha_per_check(training.alpha, training.rounds, training.attempts)
    rows = []
    total_episodes = 9*sum(config['check_sizes'])*len(config['data_seeds'])
    total_batches = 6*len(config['check_sizes'])*len(config['data_seeds'])
    completed_episodes = completed_batches = 0
    started = last_report = time.perf_counter()
    if progress is not None:
        progress(dict(completed_episodes=0,total_episodes=total_episodes,
                      completed_batches=0,total_batches=total_batches,elapsed_seconds=0.,eta_seconds=None))
    for kind in ("direction", "return"):
        for sign in (-1, 0, 1):
            q = sign*v
            candidate = exponential_target(policy, q, .1)
            truth = (direction_metrics(source, policy, q)["score"] if kind == "direction" else
                     closed_form_expected_return(source, candidate)-closed_form_expected_return(source, policy))
            for size in config["check_sizes"]:
                for seed in config["data_seeds"]:
                    old = sample_episodes(source, policy, size, seed*2+100000)
                    new = (sample_episodes(source, candidate, size, seed*2+100001)
                           if kind == "return" else None)
                    batch_id = dataset_hash(old) + (":"+dataset_hash(new) if new is not None else "")
                    for mode in ("empirical", "hoeffding"):
                        if kind == "direction":
                            result = direction_check(episode_scores(q, old, old["team_rewards"]),
                                                     source.reward_bound, config["q_max"], alpha, mode)
                        else:
                            result = return_check(old["team_rewards"], new["team_rewards"],
                                                  source.reward_bound, alpha, mode)
                        rows.append({"kind": kind, "truth_class": sign, "true_value": truth,
                                     "sample_size": size, "seed": seed, "mode": mode,
                                     "evaluation_batch_id": batch_id,
                                     "mean": result["mean"], "radius": result["radius"],
                                     "decision_value": result["decision_value"], "alpha": alpha,
                                     "accepted": result["accepted"],
                                     "false_accept": result["accepted"] and truth <= 1e-12,
                                     "false_reject": not result["accepted"] and truth > 1e-12,
                                     "evaluation_episodes": size*(2 if kind == "return" else 1)})
                    completed_episodes += size*(2 if kind == 'return' else 1)
                    completed_batches += 1
                    now = time.perf_counter()
                    if progress is not None and (now-last_report >= 5 or completed_batches == total_batches):
                        elapsed = now-started
                        progress(dict(completed_episodes=completed_episodes,total_episodes=total_episodes,
                                      completed_batches=completed_batches,total_batches=total_batches,
                                      elapsed_seconds=elapsed,
                                      eta_seconds=elapsed*(total_episodes-completed_episodes)/completed_episodes,
                                      kind=kind,truth_class=sign,sample_size=size,seed=seed))
                        last_report = now
    return rows
