"""Small CSV artifacts and seed-level summaries; no per-run plot explosion."""
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap(values, repeats=10000):
    values = np.asarray(values, float)
    if not len(values):
        return {"count": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = float(values.mean())
    if len(values) == 1:
        return {"count": 1, "mean": mean, "std": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(20260916)
    means = values[rng.integers(len(values), size=(repeats, len(values)))].mean(-1)
    return {"count": len(values), "mean": mean, "std": float(values.std(ddof=1)),
            "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975))}


def run_metrics(directory, settings, condition, seconds, curve_points):
    directory = Path(directory)
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    policies = read_jsonl(directory/"policy.jsonl")
    rounds = read_jsonl(directory/"rounds.jsonl")
    batches = read_jsonl(directory/"batches.jsonl")
    xs = np.array([r["source_episodes"] for r in policies])
    ys = np.array([r["expected_return"] for r in policies])
    deltas = np.diff(ys)
    accepted = np.array([r["accepted"] for r in policies[1:]], bool)
    declines = accepted & (deltas < -1e-10)  # Exclude floating-point ties.
    # Piecewise-constant committed policy, never interpolate a future actor.
    area = float(np.sum(np.diff(np.append(xs, settings.budget))*ys)/settings.budget)
    fits = [r["fit_kl"] for r in rounds if r["fit_kl"] is not None]
    checks = read_jsonl(directory/"checks.jsonl")
    return_checks = [r for r in checks if r["kind"] == "return" and not r.get("check_skipped")]
    metrics = {"condition": condition, "seed": settings.seed, "status": "complete",
               "run_dir": str(directory.resolve()), "seconds": seconds,
               "source_episodes": summary["source_episodes"], "budget": settings.budget,
               "unused_budget": settings.budget-summary["source_episodes"],
               "rounds": summary["rounds"], "stop_reason": summary["stop_reason"],
               "initial_return": float(ys[0]), "final_return": float(ys[-1]),
               "budget_normalized_auc": area,
               "accepted_updates": int(accepted.sum()), "declining_updates": int(declines.sum()),
               "decline_fraction": float(declines.sum()/accepted.sum()) if accepted.any() else None,
               "total_decline": float(-deltas[declines].sum()),
               "candidate_count": sum(r["candidate_count"] for r in rounds),
               "return_check_rejections": sum(not r["accepted"] for r in return_checks),
               "mean_fit_kl": float(np.mean(fits)) if fits else None}
    for purpose in ("direction_train", "pg_train", "critic", "direction_check", "return_old", "return_candidate"):
        metrics[purpose+"_episodes"] = sum(r["episodes"] for r in batches if r["purpose"] == purpose)
    batches_per_epoch = int(np.ceil(settings.episodes/(settings.batch_size_episodes or settings.episodes)))
    direction_updates = batches_per_epoch*settings.direction_epochs if settings.direction_epochs else settings.direction_steps
    fit_updates = batches_per_epoch*settings.actor_epochs if settings.actor_epochs else settings.fit_steps
    metrics["optimizer_updates"] = (len(rounds) if settings.algorithm == "pg" else
                                    len(rounds)*direction_updates + metrics["candidate_count"]*fit_updates)
    curves = []
    for grid in np.linspace(0, settings.budget, curve_points+1).astype(int):
        index = int(np.searchsorted(xs, grid, side="right")-1)
        curves.append({"condition": condition, "seed": settings.seed, "budget_checkpoint": int(grid),
                       "model_source_episodes": int(xs[index]), "model_round": policies[index]["round"],
                       "expected_return": float(ys[index])})
    return metrics, curves


def summarize_runs(rows, repeats):
    result = []
    metrics = ("final_return", "budget_normalized_auc", "seconds", "source_episodes",
               "decline_fraction", "candidate_count", "mean_fit_kl", "optimizer_updates")
    for condition in dict.fromkeys(row["condition"] for row in rows):
        all_rows = [r for r in rows if r["condition"] == condition]
        complete = [r for r in all_rows if r["status"] == "complete"]
        for key in metrics:
            values = [float(r[key]) for r in complete if r.get(key) is not None]
            result.append({"condition": condition, "metric": key, "planned_runs": len(all_rows),
                           "failed_or_interrupted": len(all_rows)-len(complete),
                           "statistics_scope": "completed_runs_only", **bootstrap(values, repeats)})
    return result


def summarize_mechanism(name, rows, repeats):
    definitions = {
        "P-M1": (("actor_id", "label", "sample_size", "method"), ("direction_error", "score")),
        "P-M3": (("target",), ("target_error", "direction_shift", "transport_bound")),
        "P-H": (("kind", "truth_class", "sample_size", "mode"), ("accepted", "false_accept", "false_reject")),
    }
    if name not in definitions:
        return []  # Deterministic calculations do not get seed confidence intervals.
    keys, metrics = definitions[name]
    result = []
    groups = list(dict.fromkeys(tuple(r[k] for k in keys) for r in rows))
    for group in groups:
        selected = [r for r in rows if tuple(r[k] for k in keys) == group]
        for metric in metrics:
            result.append({**dict(zip(keys, group)), "metric": metric,
                           **bootstrap([float(r[metric]) for r in selected], repeats)})
    return result
