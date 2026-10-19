"""One validated flat configuration shared by every entry point."""
from __future__ import annotations

import json
import math
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
PROFILES = ("smoke", "pilot", "formal")


def _merge(base: dict, values: dict) -> dict:
    result = dict(base)
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(environment: str, profile: str, overrides: str | Path | None = None) -> dict:
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    common = json.loads((CONFIG_ROOT / "defaults.json").read_text(encoding="utf-8-sig"))
    env = json.loads((CONFIG_ROOT / f"{environment}.json").read_text(encoding="utf-8-sig"))
    profiles = json.loads((CONFIG_ROOT / "profiles.json").read_text(encoding="utf-8-sig"))
    result = _merge(common, env)
    result = _merge(result, profiles[profile]["common"])
    result = _merge(result, profiles[profile]["finite" if environment in ("pair", "two_step") else "application"])
    if overrides:
        path = Path(overrides)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration not found: {path.resolve()}")
        extra = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(extra, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = sorted(set(extra) - set(result))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        result = _merge(result, extra)
    result["environment"] = environment
    result["source_agents"] = result["n_agents"]
    result["profile"] = profile
    result["protocol_version"] = (("navigation-success-v3" if result.get('train_reset_horizon')
                                   else "navigation-success-v2") if environment == 'navigation'
                                  and result.get('navigation_task') == 'success_only' else "direction-v1")
    result["smoke_only"] = profile == "smoke"
    if environment == 'navigation' and result.get('collection_mode') == 'episodes':
        result['protocol_version'] = 'navigation-episodes-v1'
        result['train_reset_horizon'] = result['horizon']
    if environment == 'navigation' and result.get('history', 1) > 1:
        result['protocol_version'] += '-history-v1'
    validate_config(result)
    return result


def validate_config(c: dict) -> None:
    from .actor_schedule import validate_actor_schedule
    validate_actor_schedule(c)
    if c.get('actor_fit_selection', 'last') not in ('last', 'best_train_kl'):
        raise ValueError('actor_fit_selection must be last or best_train_kl')
    if c['environment'] in ('pair', 'two_step') and c.get('actor_fit_selection', 'last') != 'last':
        raise ValueError('best_train_kl currently applies to the application Runner, not finite table experiments')
    for key, default in (('actor_fit_selection_atol', 1e-12), ('actor_fit_selection_rtol', 1e-6)):
        value = c.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    rounds=c.get('direction_diagnostic_rounds',[])
    if not isinstance(rounds,list) or any(type(r) is not int or r<1 for r in rounds) or len(set(rounds))!=len(rounds):
        raise ValueError('direction_diagnostic_rounds must contain distinct positive integers')
    count=c.get('direction_diagnostic_episodes',16)
    if type(count) is not int or count<1:
        raise ValueError('direction_diagnostic_episodes must be a positive integer')
    if type(c.get('save_direction_snapshots',False)) is not bool:
        raise ValueError('save_direction_snapshots must be boolean')
    for key in ('direction_check','return_check'):
        if type(c[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if c['environment'] == 'navigation':
        chunk = c.get('network_chunk_size', 256)
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise ValueError('network_chunk_size must be a positive integer')
        if c.get('collection_mode', 'streaming') not in ('streaming', 'episodes'):
            raise ValueError('collection_mode must be streaming or episodes')
        history = c.get('history', 1)
        if not isinstance(history, int) or isinstance(history, bool) or history < 1:
            raise ValueError('history must be a positive integer')
        if history > 1 and c.get('navigation_task') != 'success_only':
            raise ValueError('Observation history currently supports success-only navigation')
        if c.get('navigation_task') not in ('success_only', 'first_arrival'):
            raise ValueError('navigation_task must be success_only or first_arrival')
        if not isinstance(c.get('evaluation_horizon'), int) or isinstance(c['evaluation_horizon'], bool) or c['evaluation_horizon'] <= 0:
            raise ValueError('evaluation_horizon must be a positive integer')
        if c['navigation_task'] == 'success_only' and not 0 < c['gamma'] < 1:
            raise ValueError('Success-only navigation requires gamma < 1')
        reset_horizon = c.get('train_reset_horizon')
        if reset_horizon is not None and (not isinstance(reset_horizon, int)
                                          or isinstance(reset_horizon, bool) or reset_horizon <= 0):
            raise ValueError('train_reset_horizon must be a positive integer or null (legacy unbounded collection)')
    for key in ("budget", "n_agents", "horizon", "hidden", "heads", "rollout_steps", "minibatch_size",
                "actor_fit_steps", "direction_steps", "critic_steps", "ppo_epochs", "evaluation_episodes",
                "source_eval_episodes", "check_episodes"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("actor_lr", "critic_lr", "direction_lr", "q_bound", "max_grad_norm"):
        if not isinstance(c[key], (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("batch_episodes", "critic_episodes", "torch_threads", "eval_every_steps", "plot_every"):
        if not isinstance(c[key], int) or isinstance(c[key], bool) or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(c["relation_layers"], int) or c["relation_layers"] < 0:
        raise ValueError("relation_layers must be a nonnegative integer")
    if not 0 <= c["probability_floor"] < 1:
        raise ValueError("probability_floor is a uniform mixture mass in [0,1)")
    for key in ("gamma", "gae_lambda"):
        if not 0 < c[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 < c["ppo_clip"] < 1:
        raise ValueError("ppo_clip must be in (0, 1)")
    if c["hidden"] % c["heads"]:
        raise ValueError("hidden must be divisible by heads")
    if c["observation_mode"] not in ("full", "summary", "nearest2"):
        raise ValueError("observation_mode must be full, summary, or nearest2")
    if c["label_mode"] not in ("mc", "gae", "exact"):
        raise ValueError("label_mode must be mc, gae, or exact")
    if c["environment"] in ("navigation", "warehouse") and c["label_mode"] == "exact":
        raise ValueError("Application environments have no exact-advantage oracle")
    if c["device"] != "cpu" and c["device"] != "cuda" and not str(c["device"]).startswith("cuda:"):
        raise ValueError("device must be cpu, cuda or cuda:N")
    for key in ("sample_sizes", "fit_steps", "target_sizes", "seeds", "data_seeds"):
        if not c[key] or not all(isinstance(x, int) and not isinstance(x, bool) and x >= (0 if key.endswith("seeds") else 1) for x in c[key]):
            raise ValueError(f"{key} must be a nonempty integer list")
    if not c["step_sizes"] or any(x <= 0 for x in c["step_sizes"]):
        raise ValueError("step_sizes must contain positive values")
    expected_n = 2 if c["environment"] == "two_step" else 4
    if c["n_agents"] != expected_n:
        raise ValueError(f"Protocol {c['environment']} fixes the single source at n={expected_n}")
    if c["environment"] == "pair" and c["horizon"] != 1:
        raise ValueError("Pair coordination is a one-step episode")
    if c["environment"] == "two_step" and c["horizon"] != 2:
        raise ValueError("The sequential diagnostic has exactly two steps")
