"""按已提交训练轮次汇总，不用候选回报替代实际执行策略回报。"""
import json
from pathlib import Path


def snapshot_records(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines)-1:
                break  # 正在运行时最后一行可能尚未写完。
            raise
    return rows


def collect_overview(directory):
    directory = Path(directory)
    policies = {r["round"]: r for r in snapshot_records(directory/"policy.jsonl")}
    # policy 日志先于 checkpoint 写入，只展示已提交轮次。
    latest = directory/"checkpoints/final.json"
    if not latest.exists():
        latest = directory/"checkpoints/latest.json"  # Older runs.
    limit = json.loads(latest.read_text(encoding="utf-8"))["round"] if latest.exists() else max(policies, default=0)
    directions = {}
    for row in snapshot_records(directory/"direction.jsonl"):
        previous = directions.get(row["round"])
        if previous is None or row["step"] >= previous["step"]:
            directions[row["round"]] = row
    checks, fits = {}, {}
    for row in snapshot_records(directory/"checks.jsonl"):
        if row["kind"] == "direction":
            checks[row["round"]] = row
        elif row["kind"] == "return":
            previous = fits.get(row["round"])
            if previous is None or row["attempt"] >= previous["attempt"]:
                fits[row["round"]] = row
    result = []
    for round_index in sorted(policies):
        if round_index > limit:
            continue
        policy = policies[round_index]
        direction, check, fit = directions.get(round_index, {}), checks.get(round_index, {}), fits.get(round_index, {})
        result.append({"round": round_index, "source_episodes": policy["source_episodes"],
                       "expected_return": policy.get("expected_return"),
                       "direction_final_loss": direction.get("loss"),
                       "direction_final_error": direction.get("direction_error"),
                       "direction_updates": direction.get("step"),
                       "direction_accepted": check.get("accepted"),
                       "last_candidate_fit_kl": fit.get("fit_kl"),
                       "last_candidate_attempt": fit.get("attempt"),
                       "actor_accepted": policy["accepted"] if direction else None})
    return result


def plot_round_overview(directory, export_data=False, svg=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_training import chinese_fonts
    chinese_fonts(plt)
    directory = Path(directory)
    rows = collect_overview(directory)
    if not rows:
        return
    if export_data:
        (directory/"round_overview.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), layout="constrained")
    metrics = [("expected_return", "实际执行策略：平均团队回报"),
               ("direction_final_loss", "每轮方向训练结束：整批 loss"),
               ("direction_final_error", "每轮方向训练结束：真实方向误差"),
               ("last_candidate_fit_kl", "本轮最后一个候选：最终拟合 KL"),
               ("actor_accepted", "本轮是否采用候选 actor"),
               ("source_episodes", "累计源回合数（含所有检查）")]
    for axis, (key, title) in zip(list(axes.flat)[:6], metrics):
        xs = [r["round"] for r in rows]
        ys = [float(r[key]) if r[key] is not None else float("nan") for r in rows]
        if key == "actor_accepted":
            axis.scatter(xs, ys, s=12)
            axis.set_yticks([0, 1], ["保留旧 actor", "采用候选"])
            axis.set_ylim(-.15, 1.15)
        else:
            axis.plot(xs, ys, linewidth=1.5, marker="." if len(rows) < 50 else None)
        if all(r[key] is None for r in rows):
            axis.text(.5, .5, "无对应记录／尚未执行该阶段", ha="center", transform=axis.transAxes)
        axis.set(title=title, xlabel="外层训练轮次 round")
        axis.grid(alpha=.2)
    limit = rows[-1]["round"]
    checks = [r for r in snapshot_records(directory/"checks.jsonl")
              if r["kind"] == "direction" and r["round"] <= limit]
    axis = axes[2, 0]
    if checks:
        for key, label in (("mean", "样本平均分数"), ("decision_value", "接受判断值")):
            axis.plot([r["round"] for r in checks], [r[key] for r in checks], label=label)
        axis.axhline(0, color="gray", linewidth=1)
        axis.legend(fontsize=9)
    else:
        axis.text(.5, .5, "无方向检查记录", ha="center", transform=axis.transAxes)
    axis.set(title="独立方向检查（判断值 > 0 时通过）", xlabel="外层训练轮次 round")
    axis.grid(alpha=.2)
    stages = [r for r in snapshot_records(directory/"stages.jsonl") if r["round"] <= limit]
    names = list(dict.fromkeys(r["stage"] for r in stages))
    for axis, key, title in zip(axes[2, 1:], ("source_episodes", "seconds"),
                                ("各阶段源回合成本", "各阶段耗时（秒）")):
        if stages:
            axis.barh(names, [sum(r[key] for r in stages if r["stage"] == name) for name in names])
            axis.tick_params(axis="y", labelsize=8)
        else:
            axis.text(.5, .5, "无阶段记录", ha="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.grid(axis="x", alpha=.2)
    fig.suptitle(f"训练总览｜截至已提交第 {limit} 轮\n"
                 "方向 loss 对应各轮不同的数据与旧策略；缺失值表示无对应记录；阶段成本仅统计本运行段", fontsize=14)
    for extension in (("png", "svg") if svg else ("png",)):
        fig.savefig(directory/f"round_overview.{extension}", dpi=160)
    plt.close(fig)
    return directory/"round_overview.png"
