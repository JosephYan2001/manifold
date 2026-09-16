"""Resolve experiment identities and budgets before starting any training."""
from dataclasses import replace
import math
import numpy as np
from ..training.config import TrainConfig
from ..training.plan import normalize

EXPERIMENTS = ("P-C", "P-A", "P-M1", "P-M2", "P-M3", "P-M4", "P-H")
CONDITIONS = ("ours", "sampled", "pg", "no_direction_check", "no_return_check", "fit_quarter")


def selected_experiments(names):
    names = list(EXPERIMENTS) if "all" in names else names
    if not names or any(name not in EXPERIMENTS for name in names):
        raise ValueError("Unknown or empty experiment selection")
    requested = set(names)
    if "P-M3" in requested:
        requested.add("P-M1")  # Explicit source dependency; reused when both are selected.
    return [name for name in EXPERIMENTS if name in requested]


def conditions_for(experiments):
    names = set()
    if "P-C" in experiments:
        names.update(("ours", "sampled", "pg"))
    if "P-A" in experiments:
        names.update(("ours", "sampled", "no_direction_check", "no_return_check", "fit_quarter"))
    return [name for name in CONDITIONS if name in names]


def base_training(config):
    values = normalize(config["training"])
    # Do not let a round cap stop a cheaper ablation before its source budget.
    values["rounds"] = values["budget"]//values["episodes"]+1
    return TrainConfig(**values)


def condition_settings(base, name, seed):
    values = {"seed": seed, "algorithm": "direction", "method": "analytic",
              "direction_check_enabled": True, "return_check_enabled": True}
    if name == "sampled":
        values["method"] = "sampled"
    elif name == "pg":
        values.update(algorithm="pg", direction_check_enabled=False, return_check_enabled=False)
    elif name == "no_direction_check":
        values["direction_check_enabled"] = False
    elif name == "no_return_check":
        values["return_check_enabled"] = False
    elif name == "fit_quarter":
        key = "actor_epochs" if base.actor_epochs else "fit_steps"
        values[key] = max(1, math.ceil(getattr(base, key)/4))
    elif name != "ours":
        raise ValueError(f"Unknown condition: {name}")
    return replace(base, **values)


def validate_protocol(config, source):
    if source.n_types != 2 or min(source.type_probs) <= 0:
        raise ValueError("This suite requires two source types with positive coverage")
    # The target grid and source-controlled experiments in the document use this source.
    if source.n_agents != 4 or not np.allclose(source.type_probs, [.6, .4]) or source.interaction_strength != .7:
        raise ValueError("Suite source must have n=4, type_probs=(.6,.4), rho=.7")
    base = base_training(config)
    if base.mode != "policy" or base.baseline != "zero" or base.acceptance != "empirical":
        raise ValueError("Main comparison requires policy mode, zero baseline, empirical checks")
    for key, values in (("training_seeds", config["training_seeds"]),
                        ("data_seeds", config["mechanisms"]["data_seeds"])):
        if not values or len(values) != len(set(values)) or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(f"{key} must be unique nonnegative integer seeds")
    m = config["mechanisms"]
    for key in ("sample_sizes", "fit_steps", "check_sizes"):
        if not m[key] or len(set(m[key])) != len(m[key]) or any(type(x) is not int or x < 1 for x in m[key]):
            raise ValueError(f"{key} must contain unique positive integers")
    for key in ("direction_steps",):
        if type(m[key]) is not int or m[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if m["transport_sample_size"] not in m["sample_sizes"]:
        raise ValueError("Transport snapshot size must be present in P-M1 sample_sizes")
    if m["frozen_probabilities"] != [[.5, .5], [.2, .8]]:
        raise ValueError("The prescribed frozen actors are [.5,.5] and [.2,.8]")
    for key in ("direction_lr", "actor_lr", "q_max", "beta"):
        if not math.isfinite(m[key]) or m[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if m["beta"] >= 1 or m["q_max"] < max(abs(x) for x in source.local_bias):
        raise ValueError("Invalid probability floor or insufficient check direction bound")
    for key in ("step_sizes", "epsilons"):
        if not m[key] or len(set(m[key])) != len(m[key]) or any(not math.isfinite(x) or x <= 0 for x in m[key]):
            raise ValueError(f"Invalid {key}")
    if max(m["epsilons"]) >= .5:
        raise ValueError("Second-order epsilon must be below .5")
    for key in ("bootstrap_repeats", "curve_points"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    return base
