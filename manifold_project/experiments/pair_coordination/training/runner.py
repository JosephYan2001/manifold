"""Single-source table learning. Exact enumeration writes diagnostics only."""
from dataclasses import asdict
from pathlib import Path
import platform
import numpy as np
from ..models.actor import TableActor
from ..models.direction import TableDirection
from ..models.critic import TableCritic
from ..models.mlp import MLPActor, MLPDirection, load_actor
from .checkpoints import save_checkpoint, atomic_json
import json
import hashlib
from ..evaluation.exact_pair import exact_evaluate, exponential_target
from .labels import make_labels
from .direction_loss import fit_direction, episode_scores
from .actor_fit import fit_actor
from .acceptance import alpha_per_check, direction_check, return_check
from .logging import SourceSampler, save_json, append_json
from .plan import training_plan, public_config
from .progress import Progress


def diagnostics(source, policy, direction=None):
    support = sum(p > 0 for p in source.type_probs)
    if (2*support)**source.n_agents > 1_000_000:
        return {"exact_skipped": True}
    result = exact_evaluate(source, policy, direction)
    return {k: float(result[k]) for k in
            ("expected_return", "direction_error", "score", "information_gap")}


def _run_training(source, settings, output_dir, resume=None, stop_after_round=None, requested_config=None, compact_output=False):
    from ..models.torch_models import TorchModel, resolve_device
    import torch
    device = resolve_device(settings.device)
    directory = Path(output_dir)
    actor = (MLPActor.initial(source.n_types, settings.beta, settings.hidden_width,
                             settings.hidden_depth, settings.seed)
             if settings.actor_model == "mlp" else TableActor.initial(source.n_types, settings.beta))
    restored = None
    if settings.initial_probabilities is not None:
        if settings.actor_model != 'table':
            raise ValueError('Explicit initial probabilities require the finite table Actor')
        p = np.asarray(settings.initial_probabilities, dtype=float)
        raw = (p-settings.beta/2)/(1-settings.beta)
        if p.shape != (source.n_types,) or np.any(raw <= 0) or np.any(raw >= 1):
            raise ValueError('Initial probabilities are not representable')
        actor.logits[:] = np.log(raw/(1-raw))
    if resume:
        restored = json.loads(Path(resume).read_text(encoding="utf-8"))
        if restored.get("format_version") != 1 or restored.get("boundary") != "completed_round":
            raise ValueError("Resume requires a versioned completed-round checkpoint")
        from .config import TrainConfig
        previous = asdict(TrainConfig(**restored["training"]))
        current = asdict(settings)
        previous.pop("device")
        current.pop("device")
        if previous != current or restored["source"] != json.loads(json.dumps(asdict(source))):
            raise ValueError("Resume must preserve source and training settings, including testing budgets")
        actor = load_actor(restored["actor"])
    actor = TorchModel(actor, device)
    minimum = training_plan(source, settings)["minimum_to_start_round"]
    if settings.budget < minimum + (settings.monitor_episodes if settings.checkpoint_selection == 'empirical' else 0):
        raise ValueError(f"Budget must allow at least one round ({minimum} episodes)")
    directory.mkdir(parents=True, exist_ok=False)
    if compact_output:
        (directory/'events.jsonl').touch()
    runtime = {"backend": "pytorch", "torch": torch.__version__, "cuda_build": torch.version.cuda,
               "requested_device": settings.device, "device": str(device), "dtype": "float64",
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    print(f"训练设备：{device}；PyTorch {torch.__version__}；float64")
    plan = training_plan(source, settings)
    metadata = {"source": asdict(source), "training": asdict(settings),
                "python": platform.python_version(), "numpy": np.__version__}
    extras = {"runtime": runtime,
              "requested_config": requested_config or {"programmatic": asdict(settings)},
              "resolved_config": {"source": asdict(source), "training": public_config(settings)},
              "training_plan": plan}
    if compact_output:
        metadata.update(extras)
    else:
        for name, value in extras.items():
            save_json(directory/f"{name}.json", value)
    save_json(directory/"config.json", metadata)
    sampler = SourceSampler(source, settings, directory)
    progress = Progress(directory, sampler)
    completed = restored["round"] if restored else 0
    if restored:
        sampler.used, sampler.calls = restored["source_episodes"], restored["sampler_calls"]
        save_json(directory/"resume_from.json", {"path": str(Path(resume).resolve()), "round": completed})
    pg_optimizer = None
    if settings.algorithm == "pg":
        from .policy_gradient import make_optimizer, optimizer_snapshot, step as pg_step
        if restored and completed and "optimizer_state" not in restored:
            raise ValueError("PG resume requires saved Adam state")
        pg_optimizer = make_optimizer(actor, settings.fit_lr,
                                      restored.get("optimizer_state") if restored else None)
    best = restored.get("best_checkpoint") if restored else None
    checkpoint = save_checkpoint(directory, source, settings, actor, completed, sampler,
                                 restored.get("accepted", False) if restored else False, best,
                                 optimizer_snapshot(pg_optimizer, actor) if pg_optimizer else None,
                                 legacy_alias=not compact_output)
    best = checkpoint["best_checkpoint"]
    alpha = alpha_per_check(settings.alpha, settings.rounds, settings.attempts)
    append_json(directory/"policy.jsonl", {"round": completed, "source_episodes": sampler.used,
                "accepted": False, **diagnostics(source, actor.table())})
    rounds = 1 if settings.mode == "direction" else settings.rounds
    last_direction = {}
    for round_index in range(completed+1, rounds+1):
        if sampler.used + minimum > settings.budget:
            break
        old = actor.copy()
        round_start_cost = sampler.used
        progress.start(round_index, "冻结策略")
        mu = old.table()
        critic = None
        if settings.baseline == "table":
            progress.start(round_index, "critic 采样与拟合")
            data = sampler.sample(mu, settings.critic_episodes, "critic", round_index)
            critic = TableCritic(source.n_agents, source.n_types, source.reward_bound)
            critic.fit(data["local_types"], data["team_rewards"])
            if settings.save_direction_snapshots:
                save_json(directory/f"critic_{round_index:03d}.json", {"values": critic.values.tolist()})
        progress.start(round_index, "训练采样")
        batch = sampler.sample(mu, settings.episodes,
                               "pg_train" if pg_optimizer else "direction_train", round_index)
        append_json(directory/"data.jsonl", {"round": round_index,
                    "type_counts": np.bincount(batch["local_types"].ravel(), minlength=source.n_types).tolist(),
                    "episodes": len(batch["team_rewards"]), "reward_mean": float(batch["team_rewards"].mean()),
                    "actor_min_probability": float(mu.min())})
        labels = make_labels(batch, critic)
        if pg_optimizer is not None:
            progress.start(round_index, "直接策略梯度")
            pg_record = pg_step(actor, pg_optimizer, batch, labels)
            completed = round_index
            progress.start(round_index, "提交结果")
            append_json(directory/"policy.jsonl", {"round": completed, "source_episodes": sampler.used,
                        "accepted": True, **diagnostics(source, actor.table())})
            checkpoint = save_checkpoint(directory, source, settings, actor, completed, sampler, True,
                                         best, optimizer_snapshot(pg_optimizer, actor), legacy_alias=not compact_output)
            best = checkpoint["best_checkpoint"]
            progress.finish()
            append_json(directory/"rounds.jsonl", {"round": completed, "source_episodes": sampler.used,
                        "round_source_episodes": sampler.used-round_start_cost, "direction_passed": None,
                        "candidate_count": 0, "fit_kl": None, "accepted": True, **pg_record})
            print(f"PG 第 {completed}/{rounds} 轮｜源回合 {sampler.used}/{settings.budget}", flush=True)
            if stop_after_round is not None and completed >= stop_after_round:
                break
            continue
        if settings.algorithm == 'direct':
            from .policy_gradient import ratio_fit
            accepted, candidate_count = False, 0
            reserve = settings.monitor_episodes if settings.checkpoint_selection == 'empirical' else 0
            for attempt in range(settings.attempts):
                if sampler.used+(2*settings.check_episodes if settings.return_check_enabled else 0)+reserve > settings.budget:
                    break
                candidate = ratio_fit(old, batch, labels, settings.fit_steps, settings.fit_lr/(2**attempt))
                gate = {'accepted': True, 'disabled': True}
                if settings.return_check_enabled:
                    before = sampler.sample(mu, settings.check_episodes, 'return_old', round_index)
                    after = sampler.sample(candidate.table(), settings.check_episodes, 'return_candidate', round_index)
                    gate = return_check(before['team_rewards'], after['team_rewards'], source.reward_bound, alpha, settings.acceptance,
                                        threshold=-settings.return_tolerance)
                candidate_count += 1
                append_json(directory/'checks.jsonl', dict(kind='return', round=round_index, attempt=attempt, **gate))
                if gate['accepted']:
                    actor, accepted = candidate, True
                    break
            completed = round_index
            checkpoint = save_checkpoint(directory, source, settings, actor, completed, sampler, accepted, best,
                                         legacy_alias=not compact_output)
            best = checkpoint['best_checkpoint']
            append_json(directory/'policy.jsonl', dict(round=completed, source_episodes=sampler.used,
                        accepted=accepted, **diagnostics(source, actor.table())))
            append_json(directory/'rounds.jsonl', dict(round=completed, source_episodes=sampler.used,
                        direction_passed=None, candidate_count=candidate_count, fit_kl=None, accepted=accepted))
            progress.finish()
            if stop_after_round is not None and completed >= stop_after_round:
                break
            continue
        model = (MLPDirection(source.n_types, settings.q_max, settings.hidden_width,
                              settings.hidden_depth, settings.seed+round_index)
                 if settings.direction_model == "mlp" else TableDirection(source.n_types, settings.q_max))
        model = TorchModel(model, device)

        def record(step, loss, table):
            batches_per_epoch = int(np.ceil(settings.episodes/(settings.batch_size_episodes or settings.episodes)))
            append_json(directory/"direction.jsonl", {
                "round": round_index, "step": step, "loss": loss,
                "training_episodes": settings.episodes,
                "batch_size_episodes": min(settings.batch_size_episodes or settings.episodes, settings.episodes),
                "completed_epochs": step//batches_per_epoch if settings.direction_epochs else None,
                "batches_per_epoch": batches_per_epoch if settings.direction_epochs else None,
                "source_episodes": sampler.used,
                "amplitude_fraction": float(np.max(np.abs(table))/settings.q_max),
                **diagnostics(source, mu, table)})

        progress.start(round_index, "方向学习")
        fit_direction(model, batch, labels, settings.method, settings.direction_steps,
                      settings.direction_lr, settings.log_every, record,
                      settings.batch_size_episodes, settings.direction_epochs, settings.seed+round_index,
                      update_callback=(lambda row: append_json(directory/"updates.jsonl", {"round": round_index, "module": "direction", **row})) if settings.save_update_logs else None)
        q = model.table()
        if settings.save_direction_snapshots:
            save_json(directory/f"round_{round_index:03d}.json", {
                "reference_actor": old.state(), "direction": model.state(),
                "round": round_index, "source": asdict(source)})
        accepted = False
        direction_passed = None
        candidate_count = 0
        last_fit = None
        if settings.mode == "policy":
            gate = {"accepted": True}
            if settings.direction_check_enabled:
                progress.start(round_index, "独立方向检查")
                check = sampler.sample(mu, settings.check_episodes, "direction_check", round_index)
                gate = direction_check(episode_scores(q, check, make_labels(check, critic)),
                                       source.reward_bound*(2 if critic else 1), settings.q_max,
                                       alpha, settings.acceptance, threshold=settings.direction_threshold)
                direction_passed = gate["accepted"]
                append_json(directory/"checks.jsonl", {"kind": "direction", "round": round_index, "source_episodes": sampler.used, **gate})
            if gate["accepted"]:
                for attempt in range(settings.attempts):
                    reserve = settings.monitor_episodes if settings.checkpoint_selection == 'empirical' else 0
                    if settings.return_check_enabled and sampler.used + 2*settings.check_episodes + reserve > settings.budget:
                        break
                    eta = settings.step_sizes[attempt] if settings.step_sizes is not None else settings.eta/(2**attempt)
                    progress.start(round_index, "目标构造与 actor 拟合", attempt)
                    target = exponential_target(mu, q, eta)
                    candidate, fit = fit_actor(old, target, batch, settings.fit_steps, settings.fit_lr,
                                              settings.batch_size_episodes, settings.actor_epochs,
                                              settings.seed+round_index,
                                              callback=lambda row: append_json(directory/"actor_fit.jsonl", {"round": round_index, "attempt": attempt, **row}),
                                              update_callback=(lambda row: append_json(directory/"updates.jsonl", {"round": round_index, "attempt": attempt, "module": "actor", **row})) if settings.save_update_logs else None)
                    candidate_count += 1
                    last_fit = fit["fit_kl"]
                    gate = {"accepted": True, "check_skipped": True, "mode": "disabled"}
                    if settings.return_check_enabled:
                        progress.start(round_index, "独立回报检查", attempt)
                        old_data = sampler.sample(mu, settings.check_episodes, "return_old", round_index)
                        new_data = sampler.sample(candidate.table(), settings.check_episodes,
                                                  "return_candidate", round_index)
                        gate = return_check(old_data["team_rewards"], new_data["team_rewards"],
                                            source.reward_bound, alpha, settings.acceptance, threshold=-settings.return_tolerance)
                    append_json(directory/"checks.jsonl", {
                        "kind": "return", "round": round_index, "attempt": attempt,
                        "eta": eta, "source_episodes": sampler.used, **fit, **gate,
                        "candidate_id": f"round_{round_index:04d}_attempt_{attempt}",
                        "candidate_sha256": hashlib.sha256(json.dumps(candidate.state(), sort_keys=True).encode()).hexdigest(),
                        "actor_checkpoint": None,  # Rolling files are not immutable candidate snapshots.
                        "target_diagnostics": diagnostics(source, target),
                        "candidate_diagnostics": diagnostics(source, candidate.table())})
                    if gate["accepted"]:
                        actor, accepted = candidate, True
                        break
        completed = round_index
        progress.start(round_index, "提交结果")
        append_json(directory/"policy.jsonl", {
            "round": round_index, "source_episodes": sampler.used, "accepted": accepted,
            **diagnostics(source, actor.table())})
        checkpoint = save_checkpoint(directory, source, settings, actor, round_index, sampler, accepted, best,
                                     legacy_alias=not compact_output)
        best = checkpoint["best_checkpoint"]
        progress.finish()
        append_json(directory/"rounds.jsonl", {"round": round_index, "source_episodes": sampler.used,
                    "round_source_episodes": sampler.used-round_start_cost, "direction_passed": direction_passed,
                    "candidate_count": candidate_count, "fit_kl": last_fit, "accepted": accepted})
        print(f"第 {round_index}/{rounds} 轮｜源回合 {sampler.used}/{settings.budget}｜"
              f"方向更新 {plan['direction_updates']} 次，检查={direction_passed}｜"
              f"候选 {candidate_count} 个，拟合 KL={last_fit}｜接受={accepted}｜"
              f"checkpoint：{directory/'checkpoints/final.json'}", flush=True)
        last_direction = diagnostics(source, mu, q)
        if stop_after_round is not None and completed >= stop_after_round:
            break
    summary = {"rounds": completed, "source_episodes": sampler.used,
               "stop_reason": ("requested_pause" if stop_after_round is not None and completed >= stop_after_round and completed < rounds
                               else "round_limit" if completed == rounds else "source_budget"),
               "mode": settings.mode, "method": settings.method, "algorithm": settings.algorithm,
               "acceptance": settings.acceptance, "actor": actor.state(),
               "final_actor_diagnostics": {k: v for k, v in diagnostics(source, actor.table()).items()
                                           if k in ("expected_return", "exact_skipped")},
               "checkpoint": str((directory/"checkpoints/final.json").resolve()),
               "best_checkpoint": str((directory/"checkpoints/best.json").resolve()),
               "best_round": best["round"], "best_selection_metric": best["selection_metric"],
               "best_selection_score": best["selection_score"],
               "best_expected_return": best["selection_score"] if settings.checkpoint_selection == 'exact' else None,
               "last_direction_at_reference_actor": last_direction}
    saved_summary = {k: v for k, v in summary.items() if k != "actor"} if compact_output else summary
    save_json(directory/"summary.json", saved_summary)
    if compact_output and summary["stop_reason"] != "requested_pause":
        for name in ("progress.json", "sampling_progress.json"):
            (directory/name).unlink(missing_ok=True)
    print(f"停止原因：{ {'round_limit': '达到轮数上限', 'source_budget': '源预算不足', 'requested_pause': '按请求暂停'}[summary['stop_reason']]}；"
          f"完成 {completed} 轮；已用 {sampler.used}，剩余 {settings.budget-sampler.used} 回合；"
          f"下一轮至少需要 {minimum} 回合。", flush=True)
    return summary


def run_training(source, settings, output_dir, resume=None, stop_after_round=None, requested_config=None, compact_output=False):
    directory = Path(output_dir)
    existed = directory.exists()
    try:
        return _run_training(source, settings, output_dir, resume, stop_after_round, requested_config, compact_output)
    except (Exception, KeyboardInterrupt) as error:
        if not existed and directory.exists():
            def read(name):
                path = directory/name
                return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            checkpoint = read("checkpoints/final.json")
            sampled = read("sampling_progress.json").get("source_episodes", checkpoint.get("source_episodes", 0))
            report = {"stop_reason": "user_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                      "error_type": type(error).__name__, "message": str(error),
                      "stage": getattr(error, "training_stage", None) or read("progress.json"),
                      "committed_round": checkpoint.get("round"), "sampled_source_episodes": sampled,
                      "uncommitted_source_episodes": sampled-checkpoint.get("source_episodes", 0),
                      "latest_checkpoint": str((directory/"checkpoints/final.json").resolve()) if checkpoint else None}
            atomic_json(directory/"failure.json", report)
            print(f"训练中断：{report['stop_reason']}；阶段 {report['stage']}；"
                  f"最近提交模型 {report['latest_checkpoint']}；未提交源回合 {report['uncommitted_source_episodes']}", flush=True)
        raise
