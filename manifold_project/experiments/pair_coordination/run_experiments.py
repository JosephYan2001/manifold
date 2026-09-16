"""Run the pair-coordination experiments by document ID, with shared runs."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import hashlib
from pathlib import Path
import platform
import sys
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "manifold_project.experiments.pair_coordination"

from .envs.pair_coordination import PairConfig
from .training.checkpoints import atomic_json
from .training.plan import training_plan
from .experiments.protocol import (EXPERIMENTS, selected_experiments, conditions_for,
                                   condition_settings, validate_protocol)
from .experiments.reporting import (write_csv, read_csv, run_metrics, summarize_runs,
                                    summarize_mechanism)

ROOT = Path(__file__).resolve().parent

#   python manifold\manifold_project\experiments\pair_coordination\run_experiments.py --profile pilot --experiments P-C P-A --plot

def build_parser():
    parser = argparse.ArgumentParser(description="成对协作实验：按文档编号执行、复用、统计")
    parser.add_argument("--experiments", nargs="+", choices=[*EXPERIMENTS, "all"], default=["all"])
    parser.add_argument("--profile", choices=["smoke", "pilot", "formal"], default="smoke")
    parser.add_argument("--suite-config", type=Path, help="自定义实验协议 JSON，覆盖 profile")
    parser.add_argument("--output-dir", type=Path, help="新实验目录")
    parser.add_argument("--resume-suite", action="store_true", help="复用同协议目录中已完成的实验，不自动重跑失败训练")
    parser.add_argument("--dry-run", action="store_true", help="只显示条件、依赖和预算，不导入 PyTorch、不创建结果")
    parser.add_argument("--plot", action="store_true", help="整个实验目录仅生成一张总览 PNG")
    parser.add_argument("--device", help="覆盖训练设备，例如 cpu 或 cuda")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU 线程数，默认 1 适合小网络")
    return parser


def resolve_plan(args):
    config_path = args.suite_config or ROOT/f"configs/suite_{args.profile}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.device:
        config["training"]["device"] = args.device
    source = PairConfig.from_json(ROOT/"configs/pair_source.json")
    base = validate_protocol(config, source)
    experiments = selected_experiments(args.experiments)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    jobs = []
    conditions = conditions_for(experiments)
    requested = config.get("training_conditions")
    if requested is not None:
        if (not isinstance(requested, list) or not requested
                or any(name not in conditions for name in requested)
                or len(set(requested)) != len(requested)):
            raise ValueError("training_conditions must be unique conditions belonging to the selected experiments")
        conditions = [name for name in conditions if name in requested]
    for condition in conditions:
        for seed in config["training_seeds"]:
            settings = condition_settings(base, condition, seed)
            plan = training_plan(source, settings)
            if plan["minimum_to_start_round"] > settings.budget:
                raise ValueError(f"Budget cannot start {condition}")
            jobs.append({"condition": condition, "seed": seed,
                         "settings": asdict(settings), "minimum_round_cost": plan["minimum_to_start_round"]})
    m = config["mechanisms"]
    counts = {"training_runs": len(jobs),
              "training_source_episode_cap": sum(job["settings"]["budget"] for job in jobs)}
    if "P-M1" in experiments:
        counts["P-M1_fits"] = len(m["frozen_probabilities"])*len(m["sample_sizes"])*len(m["data_seeds"])*4
        counts["P-M1_unique_sampled_episodes"] = len(m["frozen_probabilities"])*sum(m["sample_sizes"])*len(m["data_seeds"])
    if "P-M2" in experiments:
        counts["P-M2_records"] = len(m["step_sizes"])*len(m["fit_steps"])*2
    if "P-M3" in experiments:
        counts["P-M3_records"] = 9*len(m["data_seeds"])
    if "P-M4" in experiments:
        counts["P-M4_records"] = len(m["epsilons"])
    if "P-H" in experiments:
        counts["P-H_records"] = 2*3*len(m["check_sizes"])*len(m["data_seeds"])*2
        counts["P-H_unique_sampled_episodes"] = 9*sum(m["check_sizes"])*len(m["data_seeds"])
    return source, base, config, {"experiments": experiments, "counts": counts, "jobs": jobs,
                                  "config_path": str(config_path.resolve()), "threads": args.threads}


def execute(args, source, base, config, plan):
    import numpy as np
    from .experiments import mechanisms
    from .training.config import TrainConfig
    directory = (args.output_dir or ROOT/"results"/f"suite_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    identity = {"config": config, "source": asdict(source), "threads": args.threads}
    # Normalize tuples to JSON arrays before comparing a persisted protocol.
    identity = json.loads(json.dumps(identity))
    if args.resume_suite:
        if args.output_dir is None:
            raise ValueError("--resume-suite requires --output-dir")
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Existing suite has a different protocol; use a new output directory")
        state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        manifest["experiments"] = selected_experiments([*manifest["experiments"], *plan["experiments"]])
        combined_args = argparse.Namespace(**vars(args))
        combined_args.experiments = manifest["experiments"]
        manifest["plan"] = resolve_plan(combined_args)[3]
    else:
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"identity": identity, "experiments": plan["experiments"], "plan": plan,
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                "executable": sys.executable},
                    "selection": "final only; oracle diagnostics never select training candidates",
                    "protocol_status": config["protocol_status"]}
        manifest["code_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for folder in ("experiments", "training", "models", "evaluation", "envs")
                                    for path in sorted((ROOT/folder).glob("*.py"))}
        manifest["code_sha256"]["run_experiments.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        state = {"jobs": {}}
    atomic_json(directory/"manifest.json", manifest)
    atomic_json(directory/"status.json", state)
    if plan["jobs"]:
        import torch
        torch.set_num_threads(args.threads)
        manifest["runtime"]["torch"] = torch.__version__
        atomic_json(directory/"manifest.json", manifest)
    print(f"实验输出：{directory}", flush=True)

    def run_job(key, function, training=False):
        previous = state["jobs"].get(key)
        if previous and previous["status"] == "complete":
            print(f"复用：{key}", flush=True)
            return
        if training and previous:
            # Failed seeds remain in the report; do not rerun until success.
            if previous["status"] == "running":
                previous.update(status="interrupted", error="Interrupted training; retained without automatic retry")
            atomic_json(directory/"status.json", state)
            return
        state["jobs"][key] = {"status": "running"}
        atomic_json(directory/"status.json", state)
        started = time.perf_counter()
        try:
            result = function()
            state["jobs"][key] = {"status": "complete", "seconds": time.perf_counter()-started, **result}
        except KeyboardInterrupt:
            state["jobs"][key] = {"status": "interrupted", "seconds": time.perf_counter()-started}
            atomic_json(directory/"status.json", state)
            raise
        except Exception as error:
            state["jobs"][key] = {"status": "failed", "error_type": type(error).__name__,
                                   "error": str(error), "seconds": time.perf_counter()-started}
            print(f"失败并保留记录：{key}: {error}", flush=True)
        atomic_json(directory/"status.json", state)

    for job in plan["jobs"]:
        name, seed = job["condition"], job["seed"]
        key = f"training/{name}/seed_{seed}"
        def train(job=job, key=key):
            from .training.runner import run_training
            settings = TrainConfig(**job["settings"])
            started = time.perf_counter()
            run_training(source, settings, directory/key)
            metrics, curves = run_metrics(directory/key, settings, job["condition"],
                                           time.perf_counter()-started, config["curve_points"])
            return {"metrics": metrics, "curves": curves}
        run_job(key, train, training=True)

    functions = {"P-M1": lambda: mechanisms.direction_estimation(source, config["mechanisms"]),
                 "P-M2": lambda: mechanisms.actor_realization(source, config["mechanisms"]),
                 "P-M3": lambda: mechanisms.direction_transport(source, config["mechanisms"],
                                      read_csv(directory/"P-M1/records.csv")),
                 "P-M4": lambda: mechanisms.second_order(config["mechanisms"]),
                 "P-H": lambda: mechanisms.check_rules(source, config["mechanisms"], base)}
    for name in plan["experiments"]:
        if name not in functions:
            continue
        def mechanism(name=name):
            if name == "P-M3" and state["jobs"].get("P-M1", {}).get("status") != "complete":
                raise ValueError("P-M1 dependency did not complete")
            rows = functions[name]()
            write_csv(directory/name/"records.csv", rows)
            summary = summarize_mechanism(name, rows, config["bootstrap_repeats"])
            if summary:
                write_csv(directory/name/"summary.csv", summary)
            return {"records": len(rows)}
        run_job(name, mechanism)

    records, curves = [], []
    for key, value in state["jobs"].items():
        if not key.startswith("training/"):
            continue
        if value["status"] == "complete":
            records.append(value["metrics"])
            curves.extend(value["curves"])
        else:
            _, condition, seed_name = key.split("/")
            failure_path = directory/key/"failure.json"
            failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.exists() else {}
            records.append({"condition": condition, "seed": int(seed_name.removeprefix("seed_")),
                            "status": value["status"], "error": value.get("error", "interrupted"),
                            "run_dir": str(directory/key), "seconds": value.get("seconds"),
                            "source_episodes": failure.get("sampled_source_episodes")})
    if records:
        write_csv(directory/"training_results.csv", records)
        write_csv(directory/"training_summary.csv", summarize_runs(records, config["bootstrap_repeats"]))
        write_csv(directory/"learning_curves.csv", curves)
    if args.plot:
        from .experiments.plotting import plot_suite
        plot_suite(directory)
    failed = [key for key, value in state["jobs"].items() if value["status"] != "complete"]
    print(f"已完成 {len(state['jobs'])-len(failed)} 项；失败/中断 {len(failed)} 项。结果：{directory}", flush=True)
    return 1 if failed else 0


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        source, base, config, plan = resolve_plan(args)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"协议状态：{config['protocol_status']}｜{config['description']}", flush=True)
    print(json.dumps({"experiments": plan["experiments"], "counts": plan["counts"],
                      "conditions": list(dict.fromkeys(job["condition"] for job in plan["jobs"])),
                      "source_budget_per_run": base.budget}, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    try:
        return execute(args, source, base, config, plan)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
