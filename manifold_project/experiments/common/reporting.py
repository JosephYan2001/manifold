"""Compact CSV artifacts and seed-level summaries; no per-round JSON files."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

from ..cooperative_navigation.training.storage import atomic
from .statistics import bootstrap


def _cell(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    def write(temporary):
        with Path(temporary).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: _cell(value) for key, value in row.items()} for row in rows)
    atomic(path, write)


def read_csv(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: str | Path, row: dict) -> None:
    """Append under a stable header; evolve occasional new metric columns safely."""
    path = Path(path)
    if not path.exists():
        write_csv(path, [row])
        return
    with path.open(newline="", encoding="utf-8-sig") as handle:
        columns = next(csv.reader(handle))
    if set(row) - set(columns):
        write_csv(path, read_csv(path) + [row])
    else:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writerow({k: _cell(v) for k, v in row.items()})


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic(path, lambda temporary: Path(temporary).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"))


GROUP_KEYS = (
    "experiment", "method", "variant", "pair_variant", "label", "label_mode", "label_kind",
    "actor_class", "actor_kind", "observation_mode", "stage", "reference", "metric_kind",
    "sample_size", "sample_episodes", "samples", "n_samples", "n_episodes", "fit_steps", "n_agents", "source_n",
    "target_n", "source_agents", "target_agents", "source_size", "target_size", "step_size",
    "input_mode", "projection", "case", "direction_method", "training_label", "policy_mode",
    "kind", "normalization", "budget_fraction", "relation_layers", "direction_check", "return_check",
    "source_horizon", "source_budget", "source_profile", "source_smoke_only", "smoke_only", "source_experiment",
)
NON_METRICS = set(GROUP_KEYS) | {"seed", "data_seed", "run_seed", "batch_seed", "iteration", "round", "update", "success", "completed"}


def summarize_records(records: list[dict], bootstrap_seed: int = 913) -> list[dict]:
    """First average within independent seed/batch, then bootstrap those units."""
    import numpy as np
    groups = {}
    for row in records:
        tags = {k: _cell(row[k]) for k in GROUP_KEYS if k in row}
        key = tuple(sorted(tags.items()))
        bucket = groups.setdefault(key, {"tags": tags, "metrics": {}})
        unit = (row.get("seed", row.get("run_seed", "exact")), row.get("data_seed", row.get("batch_seed", "exact")))
        for metric, value in row.items():
            if metric in NON_METRICS or isinstance(value, (str, bool, list, tuple, dict)) or value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                bucket["metrics"].setdefault(metric, {}).setdefault(unit, []).append(number)
    output = []
    for group in groups.values():
        for metric, units in group["metrics"].items():
            values = np.array([np.mean(v) for v in units.values()], dtype=float)
            count = len(values)
            stats = bootstrap(values, seed=bootstrap_seed)
            output.append({**group["tags"], "metric": metric, "mean": float(values.mean()),
                           "std": stats["std"],
                           "independent_units": count, "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                           "interval_unit": "training_seed_or_data_batch", "small_sample": count < 5})
    return output


def paired_summaries(records: list[dict]) -> list[dict]:
    """Adapt legacy paired_direction_summary/summaries to the new experiment IDs."""
    groups = {}
    for row in records:
        method = row.get("method")
        if method not in ("AN", "SA", "DA", "MAPPO-E", "IPPO-E"):
            continue
        tags = {key: _cell(row[key]) for key in GROUP_KEYS
                if key not in ('method','direction_check','return_check') and key in row}
        unit = (row.get("seed", "exact"), row.get("data_seed", "exact"))
        if unit == ("exact", "exact"):
            continue
        group = groups.setdefault(tuple(sorted(tags.items())), {"tags": tags, "units": {}})
        pair = group["units"].setdefault(unit, {})
        if method in pair:
            raise ValueError(f"Duplicate paired record for {method}/{unit}; missing experiment grouping field")
        pair[method] = row
    output = []
    for group in groups.values():
        for other in ("SA", "DA", "MAPPO-E", "IPPO-E"):
            differences = {}
            for pair in group["units"].values():
                if "AN" not in pair or other not in pair:
                    continue
                left, right = pair["AN"], pair[other]
                if group["tags"].get("experiment") == "P-E" and left.get("dataset_sha256") != right.get("dataset_sha256"):
                    raise ValueError("AN/SA paired estimation must use the same archived data hash")
                for metric in left.keys() & right.keys():
                    if metric in NON_METRICS:
                        continue
                    a, b = left[metric], right[metric]
                    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
                        if math.isfinite(a) and math.isfinite(b):
                            differences.setdefault(metric, []).append(a - b)
            for metric, values in differences.items():
                stats = bootstrap(values)
                output.append({**group["tags"], "method": f"AN - {other}", "metric": metric,
                               "mean": stats["mean"], "std": stats["std"], "independent_units": stats["count"],
                               "ci95_low": stats["ci_low"], "ci95_high": stats["ci_high"],
                               "interval_unit": "paired_training_seed_or_data_batch", "small_sample": len(values) < 5})
    return output


def plot_overview(directory: str | Path, records: list[dict] | None = None) -> Path | None:
    """Lazy dashboard entry; importing reporting does not import pyplot."""
    from .plotting import plot_overview as render
    return render(directory, records)
