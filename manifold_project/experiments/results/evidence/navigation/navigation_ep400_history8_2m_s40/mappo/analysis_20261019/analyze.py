"""Recompute the historical MAPPO reference from saved logs; no model execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "N-C_source_learning/full/MAPPO-E_s40"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def is_success(row: dict[str, str]) -> bool:
    return row["success"].lower() in {"true", "1", "1.0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = read_json(ROOT / "manifest.json")
    config = read_json(RUN / "config.json")["config"]
    summary = read_json(RUN / "summary.json")
    episodes = read_csv(RUN / "final_evaluation/evaluation_episodes.csv")
    training = read_csv(RUN / "training.csv")
    events = [json.loads(line) for line in (RUN / "events.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    monitoring = [event["record"] for event in events
                  if event.get("stream") == "evaluation"]
    aggregate = {row["metric"]: row for row in read_csv(ROOT / "summary.csv")}

    successful = [row for row in episodes if is_success(row)]
    final = {
        "checkpoint": "final.pt",
        "episodes": len(episodes),
        "scene_seeds": [int(row["scene_seed"]) for row in episodes],
        "policy_mode": sorted({row["policy_mode"] for row in episodes}),
        "mean_discounted_return": mean(float(row["discounted_return"]) for row in episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes),
        "restricted_mean_steps": mean(int(row["steps"]) for row in episodes),
        "successful_mean_steps": mean(int(row["steps"]) for row in successful),
        "successes_by_400": sum(int(row["steps"]) <= 400 for row in successful),
        "evaluation_steps": sum(int(row["steps"]) for row in episodes),
    }
    # Verify the published aggregate against independent per-episode records.
    for metric, value in {
        "discounted_return": final["mean_discounted_return"],
        "success_rate": final["success_rate"],
        "restricted_mean_steps": final["restricted_mean_steps"],
        "source_report_only_steps": final["evaluation_steps"],
    }.items():
        assert math.isclose(float(aggregate[metric]["mean"]), value,
                            rel_tol=1e-12, abs_tol=1e-10), metric
    assert len(episodes) == 20 and len(set(final["scene_seeds"])) == 20
    assert final["scene_seeds"] == list(range(1000000, 1000020))
    assert final["policy_mode"] == ["stochastic"]
    assert all(Path(row["checkpoint"].replace("\\", "/")).name == "final.pt"
               for row in episodes)
    assert all(int(row["evaluation_horizon"]) == 1000 for row in episodes)
    assert len(training) == summary["rounds"] == 295
    assert len(monitoring) == 32
    assert sum(summary["costs"].values()) == summary["source_steps"]
    assert config["budget"] - summary["source_steps"] == summary["unused_budget"]
    assert sum(round(row["episodes"] * row["episode_steps"]) for row in monitoring
               ) == summary["costs"]["source_monitor"]
    assert sum(int(row["train_batch_steps"]) for row in training
               ) == summary["costs"]["train"]
    assert math.isclose(monitoring[-1]["J"], summary["J"], abs_tol=1e-12)

    reference_nodes = {0, 262144, 458752, 1048576, 1966080, 2000000}
    monitor_fields = ("budget_checkpoint", "actor_source_steps", "episodes", "J",
                      "success_rate", "restricted_mean_steps")
    selected = [{key: row[key] for key in monitor_fields}
                for row in monitoring if row["budget_checkpoint"] in reference_nodes]
    for row in selected:
        row["scene_set"] = ("source-final" if row["budget_checkpoint"] == 2000000
                            else "source-grid")
    raw_files = sorted(path for path in ROOT.rglob("*") if path.is_file()
                       and "analysis_20261019" not in path.relative_to(ROOT).parts)
    result = {
        "analysis_date": "2026-10-19",
        "scope": "saved_results_only_no_new_environment_or_optimizer_steps",
        "training_seed": config["seeds"],
        "experiment_code_sha256": manifest["metadata"]["experiment_code_sha256"],
        "implementation": summary["implementation"],
        "protocol": {key: config[key] for key in (
            "n_agents", "history", "train_reset_horizon", "evaluation_horizon",
            "train_episodes", "minibatch_episodes", "ppo_epochs", "actor_lr",
            "critic_lr", "ppo_value_normalization", "ppo_value_clipping",
            "ppo_huber_loss", "ppo_normalize_advantage")},
        "final_independent_evaluation": final,
        "source_monitoring": {
            "nodes": len(monitoring),
            "selected_nodes": selected,
            "final_summary_is_8_scene_monitor_not_20_scene_report": True,
        },
        "costs": {
            "source_budget": config["budget"],
            "source_steps": summary["source_steps"],
            "training_steps": summary["costs"]["train"],
            "source_monitoring_steps": summary["costs"]["source_monitor"],
            "unused_source_budget": summary["unused_budget"],
            "final_report_steps": final["evaluation_steps"],
            "total_environment_steps": summary["source_steps"] + final["evaluation_steps"],
            "rounds": summary["rounds"],
            "optimizer_steps_by_module": summary["optimizer_steps_by_module"],
        },
        "training_diagnostics_first_last": [
            {key: float(row[key]) for key in (
                "source_steps", "explained_variance", "policy_entropy", "train_J")}
            for row in (training[0], training[-1])],
        "raw_files_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                             for path in raw_files},
        "limitations": [
            "One training seed; final report contains only 20 scenarios.",
            "Historical code differs from the planned six AN runs.",
            "Future AN evaluation uses different scenarios; no paired comparison yet.",
            "Monitoring nodes and final independent report are separate scenario sets.",
            "No checkpoint loading or inference compatibility check was performed.",
        ],
    }
    output = Path(__file__).with_name("metrics.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"final": final, "costs": result["costs"],
                      "raw_file_count": len(raw_files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
