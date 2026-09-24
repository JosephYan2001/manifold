"""P-EA: replay saved P-E directions without fitting a direction again."""
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from ..common.reporting import read_csv
from .diagnostics import actor_realization, configured_labels, dataset_hash
from .environments import pair_source


def linked_realization(config):
    source = Path(config["direction_source"])
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entry = manifest.get("experiments", {}).get("P-E", {})
    if manifest.get("status") != "completed" or entry.get("status") != "completed":
        raise ValueError("P-EA requires a completed P-E source suite")
    original = entry["config"]
    if original.get("protocol_version") != config["protocol_version"]:
        raise ValueError("P-E protocol version mismatch")
    for key in ("environment", "n_agents", "horizon", "type_probability",
                "interaction_strength", "source_agents", "pair_variant"):
        if original.get(key) != config.get(key):
            raise ValueError(f"P-E source environment mismatch: {key}")
    if original.get("smoke_only") and not config.get("smoke_only"):
        raise ValueError("Cannot use smoke directions as non-smoke evidence")
    path = (source / entry["directory"] / "diagnostics.csv").resolve()
    if not path.is_relative_to(source.resolve()):
        raise ValueError("P-E diagnostics must be inside the source suite")
    records = read_csv(path)
    index = {}
    for r in records:
        key = (r["method"], r["label"], int(r["sample_episodes"]), int(r["data_seed"]))
        if key in index:
            raise ValueError(f"Duplicate P-E direction: {key}")
        index[key] = r
    labels = configured_labels(config, "labels", ("exact", "mc_zero"))
    keys = list(product(config["methods"], labels, config["sample_sizes"], config["data_seeds"]))
    missing = [key for key in keys if key not in index]
    if missing:
        raise ValueError(f"Requested P-E directions missing: {missing[:5]}")
    env = pair_source(config)
    prepared = []
    for key in keys:
        method, label, n, seed = key
        r = index[key]
        batch = env.sample(env.initial_policy, n, np.random.default_rng(seed))
        # Includes old policy probabilities and rewards: verifies replay semantics.
        if dataset_hash(batch) != r["dataset_sha256"]:
            raise ValueError(f"P-E dataset/policy replay mismatch: {key}")
        q = np.array([float(r[f"q_coefficient_{i}"]) for i in range(env.n_states)])
        error = float(r["direction_error"])
        if not np.isfinite(q).all() or not np.isfinite(error):
            raise ValueError(f"Non-finite saved direction: {key}")
        if not np.isclose(env.oracle(env.initial_policy, q)["direction_error"], error, rtol=1e-9, atol=1e-12):
            raise ValueError(f"P-E direction definition/oracle mismatch: {key}")
        prepared.append((key, r, q))
    rows = []
    csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for (method, label, n, seed), r, q in prepared:
        c = dict(config, seed=seed, sample_episodes=n, methods=[method])
        for row in actor_realization(c, saved_direction=q, saved_label=label):
            row.update(experiment="P-EA", data_seed=seed,
                       source_direction_error=float(r["direction_error"]),
                       source_exact_score=float(r["exact_score"]),
                       dataset_sha256=r["dataset_sha256"], direction_retrained=False,
                       direction_source=str(source), source_csv_sha256=csv_hash,
                       source_manifest_sha256=manifest_hash,
                       source_code_sha256=manifest["metadata"]["experiment_code_sha256"],
                       additional_training_environment_steps=0,
                       replayed_source_episodes=n,
                       report_validation_episodes=config["evaluation_episodes"])
            rows.append(row)
    return rows
