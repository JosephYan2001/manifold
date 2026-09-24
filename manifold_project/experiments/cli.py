"""Single entry point for mechanism, application, and frozen evaluation suites."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

from .common.config import load_config, validate_config
from .common.reporting import read_csv, summarize_records, paired_summaries, write_csv, write_json, plot_overview
from .common.runtime import configure_runtime, metadata
from .registry import APPLICATION, EXPERIMENTS, get_experiment


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direction-learning experiments v1 (source-only training).")
    p.add_argument("--experiments", nargs="+", default=["P-E", "P-A", "P-T"], choices=list(EXPERIMENTS))
    p.add_argument("--profile", choices=["smoke", "pilot", "formal"], default="pilot")
    p.add_argument("--methods", nargs="+", choices=list(APPLICATION))
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--data-seeds", nargs="+", type=int)
    p.add_argument("--sample-sizes", nargs="+", type=int)
    p.add_argument("--fit-steps", nargs="+", type=int)
    p.add_argument("--budget", type=int, help="Source decision joint steps PER method/seed/label group; fixed-base experiments use sample-sizes instead.")
    p.add_argument("--evaluation-episodes", type=int)
    p.add_argument("--device", default=None, help="cpu, cuda, or cuda:N")
    p.add_argument("--threads", type=int)
    p.add_argument("--config", type=Path, help="JSON overrides, unknown keys are rejected.")
    p.add_argument("--output", type=Path)
    p.add_argument("--source", type=Path, help="Existing source suite containing final.pt; required for transfer/replay unless source training is in this invocation.")
    p.add_argument("--checkpoints", nargs="+", type=Path, help="Explicit source checkpoints (never selected by target performance).")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-every", type=int)
    p.add_argument("--allow-duplicate-openmp", action="store_true", help="Explicitly enable the previously used KMP_DUPLICATE_LIB_OK workaround.")
    p.add_argument("--dry-run", action="store_true", help="Print resolved experiments/configurations; do not create results.")
    p.add_argument("--list", action="store_true", help="List experiment IDs and prerequisites.")
    return p


def resolve(args):
    plan = []
    for name in dict.fromkeys(args.experiments):
        exp = get_experiment(name)
        c = load_config(exp.environment, args.profile, args.config)
        if args.profile == "smoke":
            if exp.environment in ("navigation", "warehouse"):
                c["horizon"] = 8
            c["batch_episodes"] = 8
            c["critic_episodes"] = 8
        for key in ("seeds", "data_seeds", "sample_sizes", "fit_steps", "budget", "evaluation_episodes", "device", "plot_every"):
            value = getattr(args, key)
            if value is not None:
                c[key] = value
        if args.threads is not None:
            c["torch_threads"] = args.threads
        c["plot"] = args.plot
        c["allow_duplicate_openmp"] = args.allow_duplicate_openmp or c["allow_duplicate_openmp"]
        c["methods"] = [method for method in (args.methods or exp.methods) if method in exp.methods] if exp.methods else ["AN", "SA", "DA"]
        if exp.methods and not c["methods"]:
            raise ValueError(f"{name} has no method in the requested selection; allowed: {exp.methods}")
        if name == "P-A" and not set(c["methods"]) & {"AN", "SA"}:
            raise ValueError("P-A needs AN or SA to learn the fixed reference direction; DA can be added as a comparator.")
        c["experiment_id"] = name
        if name == "P-EA":
            if not args.source:
                raise ValueError("P-EA requires --source pointing to a completed P-E suite")
            if args.seeds is not None:
                raise ValueError("P-EA selects P-E data seeds with --data-seeds, not --seeds")
            c["direction_source"] = str(args.source.resolve())
        if name == "N-R":
            c["relation_layers"] = 0
        elif name == "N-Gd":
            c["direction_check"] = False
        elif name == "N-Gr":
            c["return_check"] = False
        if exp.environment == "pair_entities" and args.budget is None:
            c["budget"] = {"smoke": 256, "pilot": 8192, "formal": 65536}[args.profile]
        validate_config(c)
        plan.append((exp, c))
    requested = {exp.id for exp, _ in plan}
    for exp, _ in plan:
        if exp.kind in ("evaluate", "replay") and not args.source and not args.checkpoints and exp.source_experiment not in requested:
            raise ValueError(f"{exp.id} requires --source SOURCE_SUITE or --checkpoints SOURCE_FINAL.pt (or include {exp.source_experiment}).")
    # Source learning must precede evaluation even if the command lists it later.
    return sorted(plan, key=lambda item: 2 if item[0].kind in ("evaluate", "replay")
                  else 0 if item[0].id in ("N-C", "W-C") else 1)


def _require_dependencies(plan):
    needed = {"numpy"}
    if any(exp.kind != "finite" or exp.id == 'P-L' for exp, _ in plan):
        needed.add("torch")
    if any(exp.environment == "navigation" for exp, _ in plan):
        needed.add("mpe2")
    if any(exp.environment == "warehouse" for exp, _ in plan):
        needed.add("rware")
    if any(c["plot"] for _, c in plan):
        needed.add("matplotlib")
    missing = [name for name in sorted(needed) if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Missing dependencies in {sys.executable}: {', '.join(missing)}. "
                           "Install manifold_project/experiments/requirements.txt in the active interpreter.")


def _select_checkpoints(exp, args, generated):
    paths = list(args.checkpoints or [])
    if args.source:
        if not args.source.exists():
            raise FileNotFoundError(f"Source results not found: {args.source.resolve()}")
        if args.source.is_file():
            paths.append(args.source)
        elif (args.source / "manifest.json").is_file():
            source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8-sig"))
            entry = source_manifest.get("experiments", {}).get(exp.source_experiment)
            if not entry or entry.get("status") != "completed":
                raise ValueError(f"Source manifest needs a completed {exp.source_experiment}; choose explicit --checkpoints for an ablation.")
            for item in entry.get("checkpoints", []):
                path = Path(item)
                paths.append(path if path.is_absolute() else args.source / path)
        else:
            paths.extend(sorted(args.source.rglob("final.pt")))
    paths.extend(generated.get(exp.source_experiment, []))
    paths = list(dict.fromkeys(Path(path).resolve() for path in paths))
    if not paths:
        raise ValueError(f"No source final checkpoints for {exp.id}; archived checkpoints are incompatible.")
    return paths


def _attach(rows, exp, config, **tags):
    for row in rows:
        row.setdefault("experiment", exp.id)
        row.setdefault("environment", exp.environment)
        row.setdefault("smoke_only", config["smoke_only"])
        row.update(tags)
    return rows


def _run_finite(exp, config, directory):
    from .finite import run_experiment
    all_rows, runs = [], []
    if exp.id in ("P-A", "P-T", "P-L", "T-L"):
        for seed in config["seeds"]:
            c = dict(config, seed=seed)
            result = run_experiment(exp.id, c, directory / f"seed_{seed}")
            all_rows.extend(_attach(result["records"], exp, c))
            runs.append({"seed": seed, "status": result["status"], "config": c})
    else:
        result = run_experiment(exp.id, config, directory)
        all_rows.extend(_attach(result["records"], exp, config))
        runs.append({key: value for key, value in result.items() if key != "records"})
    return all_rows, runs, []


def _run_train(exp, config, directory, args, generated):
    from .cooperative_navigation.current_protocol import train
    from .applications.evaluation import evaluate_checkpoint, load_actor
    all_rows, runs, checkpoints = [], [], []
    reusable = {}
    if exp.kind == "input_train" and (args.source or args.checkpoints or generated.get("N-C")):
        for path in _select_checkpoints(get_experiment("N-T"), args, generated):
            actor, saved = load_actor(path, config["device"])
            old = saved["config"]
            if saved["environment"] == exp.environment and old.get("observation_mode", "full") == "full":
                key = (saved["method"], old["seed"])
                if key in reusable:
                    raise ValueError(f"More than one full-input source checkpoint for {key}; select --checkpoints explicitly")
                reusable[key] = (path, actor, saved)
    modes = ("full", "summary", "nearest2") if exp.kind == "input_train" else (config["observation_mode"],)
    for mode in modes:
        for method in config["methods"]:
            for seed in config["seeds"]:
                c = dict(config, seed=seed, observation_mode=mode)
                run_dir = directory / mode / f"{method}_s{seed}"
                reused = mode == "full" and (method, seed) in reusable
                if mode == "full" and reusable and not reused:
                    raise ValueError(f"N-I source reuse is missing {method}/seed {seed}; use matching --methods and --seeds")
                if reused:
                    source_path, source_actor, saved = reusable[(method, seed)]
                    reporting_only = {'experiment_id','methods','seeds','target_sizes','plot','plot_every',
                                      'evaluation_episodes','evaluation_seed','deterministic_evaluation',
                                      'save_fit_diagnostics','fit_snapshot_fractions','fit_steps'}
                    keys = (set(saved['config']) | set(c))-reporting_only
                    mismatches = [key for key in keys if saved["config"].get(key) != c.get(key)]
                    if mismatches:
                        raise ValueError(f"N-I full-input source differs in {mismatches}; align configuration before reusing it")
                    last_rows = read_csv(source_path.parent / "training.csv") or read_csv(source_path.parent.parent / "training.csv")
                    last = last_rows[-1] if last_rows else {}
                    record = dict(method=method, seed=seed, source_steps_total=saved["source_steps_total"], budget=c["budget"],
                                  actor_parameters=sum(p.numel() for p in source_actor.parameters()),
                                  additional_source_training_steps=0, source_training_reused=True)
                    for key, value in last.items():
                        if key.startswith("source_") and key.endswith("steps"):
                            record[key] = float(value)
                    result = dict(records=[record], checkpoint=str(source_path), protocol=saved["protocol"])
                else:
                    result = train(exp.environment, method, c, run_dir)
                checkpoint = Path(result["checkpoint"])
                checkpoints.append(checkpoint)
                # Report-only evaluation: not used for best/stopping/hyperparameters.
                with_targets = exp.kind in ('train_transfer', 'input_train') or exp.id == 'N-R'
                eval_config = dict(c, target_sizes=c['target_sizes'] if with_targets else [])
                evaluated = evaluate_checkpoint(checkpoint, eval_config, run_dir / "final_evaluation", experiment=exp.id)
                source = next(row for row in evaluated["records"] if row["target_n"] == c["n_agents"])
                record = dict(result["records"][0])
                for key, value in source.items():
                    if key not in ("environment_steps", "evaluation_steps", "wall_seconds", "checkpoint"):
                        record[key] = value
                record["source_report_only_steps"] = source["evaluation_steps"]
                record["target_report_only_steps"] = evaluated["evaluation_steps"] - source["evaluation_steps"]
                record["report_evaluation_steps"] = evaluated["evaluation_steps"]
                all_rows.extend(_attach([record], exp, c, observation_mode=mode))
                if with_targets:
                    all_rows.extend(_attach([r for r in evaluated["records"] if r["target_n"] != c["n_agents"]], exp, c, observation_mode=mode))
                runs.append({"method": method, "seed": seed, "config": c, "protocol": result["protocol"],
                             "checkpoint": str(checkpoint), "status": "completed", "frozen_actor_verified": evaluated["frozen_actor_verified"],
                             "source_training_reused": reused,
                             "evaluation_protocols": evaluated["protocols"]})
    return all_rows, runs, checkpoints


def _run_evaluation(exp, config, directory, args, generated):
    from .applications.evaluation import evaluate_checkpoint, load_actor
    rows, runs = [], []
    for index, checkpoint in enumerate(_select_checkpoints(exp, args, generated)):
        _, payload = load_actor(checkpoint, config["device"])
        if payload["environment"] != exp.environment or payload["method"] not in config["methods"]:
            continue
        if args.seeds and payload["config"]["seed"] not in args.seeds:
            continue
        source_id = payload["config"].get("experiment_id")
        if not args.checkpoints and source_id and source_id != exp.source_experiment:
            continue
        source_smoke = bool(payload["config"].get("smoke_only", False))
        if config["profile"] == "formal" and source_smoke:
            raise ValueError(f"Formal evaluation cannot use a smoke checkpoint: {checkpoint}")
        run_dir = directory / f"{payload['method']}_s{payload['config']['seed']}_{index}"
        if exp.kind == "replay":
            if payload["method"] != "AN":
                continue
            raw = read_csv(checkpoint.parent / "diagnostics.csv") or read_csv(checkpoint.parent.parent / "diagnostics.csv")
            if not raw:
                raise ValueError(f"No fixed-direction fit diagnostics at {checkpoint.parent}; source run needs save_fit_diagnostics=true.")
            converted = []
            for row in raw:
                converted_row = {}
                for key, value in row.items():
                    try:
                        converted_row[key] = float(value) if value != "" else None
                    except (TypeError, ValueError):
                        converted_row[key] = value
                converted_row.update(experiment=exp.id, source_checkpoint=str(checkpoint), used_for_selection=False,
                                     source_profile=payload["config"].get("profile", "unknown"),
                                     smoke_only=source_smoke or config["smoke_only"])
                converted.append(converted_row)
            write_csv(run_dir / "diagnostics.csv", converted)
            rows.extend(_attach(converted, exp, config))
            runs.append({"checkpoint": str(checkpoint), "status": "completed", "replayed_from_source_diagnostics": True,
                         "extra_training_steps": 0, "config": payload["config"]})
        else:
            result = evaluate_checkpoint(checkpoint, config, run_dir, experiment=exp.id)
            rows.extend(_attach(result["records"], exp, config))
            runs.append({"checkpoint": str(checkpoint), "status": "completed", "frozen_actor_verified": result["frozen_actor_verified"],
                         "evaluation_steps": result["evaluation_steps"], "protocols": result["protocols"]})
    if not rows:
        raise ValueError(f"No checkpoint matches {exp.environment}/{config['methods']} for {exp.id}.")
    return rows, runs, []


def _write_analysis(directory, exp, records, summary, config):
    count = len({row.get("seed", row.get("data_seed")) for row in records})
    text = [f"# {exp.id} {exp.title}", "", f"状态：执行完成。协议：{config['protocol_version']}；profile：{config['profile']}。", ""]
    if config["smoke_only"]:
        text += ["**这是缩小模型/期限/预算的工程烟雾测试，不是论文实验结果，也不用于方法排名。**", ""]
    text += [f"记录 {len(records)} 行，独立种子/批次标识数 {count}。详见 diagnostics.csv、各运行 training.csv 或 evaluation_episodes.csv。", "",
             "summary.csv 为长表：每行是一项指标，mean/std 是独立训练种子或数据批次间的统计；independent_units 是实际重复数。单次精确计算不伪造置信区间。", "",
             "A 类覆盖结论仅适用于可精确核对完整输入支持与方向偏移的有限模型；实体列表跨人数及应用结果按经验外推解释。", "",
             "自动汇总不判定算法优越或训练收敛；应同时检查任务指标、源学习、拟合误差与全部交互/计算成本。", ""]
    (directory / "analysis.md").write_text("\n".join(text), encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        for e in EXPERIMENTS.values():
            print(f"{e.id:5} {e.environment:14} {e.kind:14} {e.title}  methods={','.join(e.methods) or 'exact'}" +
                  (f"  source={e.source_experiment}" if e.source_experiment else ""))
        return 0
    try:
        plan = resolve(args)
        if args.dry_run:
            print(json.dumps({e.id: c for e, c in plan}, ensure_ascii=False, indent=2))
            return 0
        configure_runtime(any(c["allow_duplicate_openmp"] for _, c in plan))
        _require_dependencies(plan)
        if any(e.kind != "finite" for e, _ in plan):
            from .common.runtime import configure_torch
            configure_torch(plan[0][1])
        default_name = "_".join(e.id for e, _ in plan) + f"_{args.profile}_v1"
        output = (args.output or Path(__file__).parent / "results" / default_name).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is nonempty: {output}. Choose a new --output; existing runs are never overwritten.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"protocol_version": "direction-v1", "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata(), "command_arguments": vars(args), "experiments": {},
                    "archive_sha256": "3872988f36a6a94a9dc4527ccd4b0ec894be217e0fa01b3f733709824efe09c6"}
        write_json(output / "manifest.json", manifest)
        generated, suite_rows = {}, []
        for exp, c in plan:
            directory = output / f"{exp.id}_{exp.title}"
            directory.mkdir(parents=True, exist_ok=True)
            print(f"\n{exp.id}: {exp.title} [{c['profile']}] -> {directory}", flush=True)
            entry = {"status": "running", "config": c, "directory": str(directory.relative_to(output))}
            manifest["experiments"][exp.id] = entry
            write_json(output / "manifest.json", manifest)
            started = time.perf_counter()
            try:
                if exp.kind == "finite":
                    rows, runs, checkpoints = _run_finite(exp, c, directory)
                elif exp.kind in ("train", "input_train", "train_transfer"):
                    rows, runs, checkpoints = _run_train(exp, c, directory, args, generated)
                else:
                    rows, runs, checkpoints = _run_evaluation(exp, c, directory, args, generated)
                generated[exp.id] = checkpoints
                summary = summarize_records(rows) + paired_summaries(rows)
                write_csv(directory / "summary.csv", summary)
                _write_analysis(directory, exp, rows, summary, c)
                if c["plot"]:
                    plot_overview(directory, rows)
                suite_rows.extend(rows)
                entry.update(status="completed", runs=runs, elapsed_seconds=time.perf_counter() - started,
                             records=len(rows), checkpoints=[str(x.relative_to(output)) if x.is_relative_to(output) else str(x) for x in checkpoints])
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
                manifest["status"] = "failed"
                write_json(output / "manifest.json", manifest)
                raise
            write_json(output / "manifest.json", manifest)
        write_csv(output / "summary.csv", summarize_records(suite_rows) + paired_summaries(suite_rows))
        if args.plot:
            plot_overview(output, suite_rows)
        manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "manifest.json", manifest)
        print(f"Completed: {output}", flush=True)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
