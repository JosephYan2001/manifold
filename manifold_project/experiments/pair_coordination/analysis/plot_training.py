"""读取训练日志，绘制中文指标曲线。"""
import argparse
import json
from pathlib import Path
from ..training.descriptions import VALUES


def chinese_fonts(plt):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_records(path):
    from ..training.logging import read_records as read_stream
    return read_stream(path, tolerate_partial=True)


def representative_rounds(rows, limit=6):
    """按轮次位置均匀选取，保留首末轮；不根据指标好坏选择。"""
    rounds = sorted({r["round"] for r in rows})
    if len(rounds) <= limit:
        return rounds
    return [rounds[round(i*(len(rounds)-1)/(limit-1))] for i in range(limit)]


def plot_training(directory, detailed=False):
    """默认仅保存一张总览；详细分图用于按需排查。"""
    if not detailed:
        from .round_overview import plot_round_overview
        return plot_round_overview(directory)
    return plot_detailed_training(directory)


def plot_detailed_training(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chinese_fonts(plt)
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))["training"]
    rows = read_records(directory/"direction.jsonl")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.72, wspace=.32)
    selected_rounds = representative_rounds(rows)
    for round_index in selected_rounds:
        selected = [r for r in rows if r["round"] == round_index]
        for axis, key, title in zip(axes, ["loss", "direction_error", "score"],
                                    ["整批方向损失", "真实方向误差", "总体方向分数"]):
            points = [r for r in selected if key in r]
            if points:
                axis.plot([r["step"] for r in points], [r[key] for r in points], label=f"第 {round_index} 轮")
            elif round_index == selected_rounds[0]:
                axis.text(.5, .5, "已跳过精确枚举", ha="center", transform=axis.transAxes)
            axis.set(xlabel="方向参数更新次数", ylabel=title)
            axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .86), ncol=6, frameon=False)
    total_rounds = len({r['round'] for r in rows})
    fig.suptitle(f"轮内方向学习：{VALUES[config['method']]}｜随机种子 {config['seed']}\n"
                 f"共 {total_rounds} 轮，按位置展示 {len(selected_rounds)} 轮；完整数据保存在日志中", y=.98)
    for extension in ("png", "svg"):
        fig.savefig(directory/f"direction_curves.{extension}", dpi=160)
    plt.close(fig)
    if config["mode"] == "policy":
        rows = read_records(directory/"policy.jsonl")
        checks = read_records(directory/"checks.jsonl")
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
        exact = [r for r in rows if "expected_return" in r]
        axes[0].plot([r["source_episodes"] for r in exact], [r["expected_return"] for r in exact], marker="o")
        axes[0].set_ylabel("精确平均团队回报（越高越好）")
        fits = [r for r in checks if r["kind"] == "return"]
        if fits:
            axes[1].plot([r["source_episodes"] for r in fits], [r["fit_kl"] for r in fits], marker="o")
        else:
            axes[1].text(.5, .5, "方向检查未通过\n尚未执行 actor 拟合", ha="center", transform=axes[1].transAxes, wrap=True)
        axes[1].set_ylabel("actor 拟合 KL（越接近零越好）")
        rates = [sum(s["accepted"] for s in rows[1:i+1])/i for i in range(1, len(rows))]
        axes[2].plot([r["source_episodes"] for r in rows[1:]], rates, marker="o")
        axes[2].set(ylabel="累计更新接受比例", ylim=(-.05, 1.05))
        for axis in axes:
            axis.set_xlabel("累计源团队回合数（含所有检查）")
            axis.grid(alpha=.2)
        fig.suptitle(f"策略学习：{VALUES[config['acceptance']]}｜随机种子 {config['seed']}")
        for extension in ("png", "svg"):
            fig.savefig(directory/f"policy_curves.{extension}", dpi=160)
        plt.close(fig)
    plot_debug(directory)
    from .round_overview import plot_round_overview
    plot_round_overview(directory, export_data=True, svg=True)


def plot_debug(directory):
    """从阶段及 epoch 日志绘图，旧运行缺少文件时跳过对应图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chinese_fonts(plt)
    directory = Path(directory)

    def save(fig, name):
        for suffix in ("png", "svg"):
            fig.savefig(directory/f"{name}.{suffix}", dpi=160)
        plt.close(fig)

    if read_records(directory/"actor_fit.jsonl"):
        rows = read_records(directory/"actor_fit.jsonl")
        checks = read_records(directory/"checks.jsonl")
        decisions = {(r["round"], r["attempt"]): r["accepted"] for r in checks if r["kind"] == "return"}
        fig, axis = plt.subplots(figsize=(9, 4), layout="constrained")
        keys = sorted({(r["round"], r["attempt"]) for r in rows})
        for key in keys[-10:]:
            selected = [r for r in rows if (r["round"], r["attempt"]) == key]
            use_epoch = all(r["epoch"] is not None for r in selected)
            status = "接受" if decisions.get(key) else "拒绝或未完成检查"
            axis.plot([r["epoch"] if use_epoch else r["step"] for r in selected],
                      [r["fit_kl"] for r in selected], label=f"轮 {key[0]} 候选 {key[1]+1}：{status}")
        axis.set(xlabel="epoch（兼容模式为更新次数）", ylabel="整批输入上的拟合 KL",
                 title="actor 拟合过程（最近 10 个候选；全部记录保存在日志中）")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        save(fig, "actor_fit_epochs")
    if read_records(directory/"checks.jsonl"):
        rows = [r for r in read_records(directory/"checks.jsonl") if r["kind"] == "direction"]
        fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        axis.plot([r["round"] for r in rows], [r["mean"] for r in rows], marker="o", label="独立样本平均分数")
        axis.plot([r["round"] for r in rows], [r["decision_value"] for r in rows], label="实际接受判断值")
        axis.axhline(0, color="gray", linewidth=1)
        axis.set(xlabel="训练轮次", ylabel="方向检查分数", title="独立方向检查（判断值大于零时通过）")
        axis.legend()
        save(fig, "direction_checks")
    if read_records(directory/"stages.jsonl"):
        rows = read_records(directory/"stages.jsonl")
        stages = sorted({r["stage"] for r in rows})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for axis, key, title in zip(axes, ("source_episodes", "seconds"), ("源回合成本", "耗时（秒）")):
            axis.barh(stages, [sum(r[key] for r in rows if r["stage"] == stage) for stage in stages])
            axis.set_xlabel(title)
        fig.suptitle("本运行段各阶段总成本")
        save(fig, "stage_costs")


def plot_aggregate(directory):
    """展示独立种子的均值和标准差，误差棒表示离散程度。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chinese_fonts(plt)
    directory = Path(directory)
    groups = json.loads((directory/"aggregate.json").read_text(encoding="utf-8"))["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method in sorted({g["method"] for g in groups}):
        selected = sorted([g for g in groups if g["method"] == method], key=lambda g: g["training_episodes_per_round"])
        for axis, metric, title in zip(axes, ["direction_error", "expected_return"],
                                       ["最后一轮旧策略处的方向误差", "最终平均团队回报"]):
            valid = [g for g in selected if g[metric]["mean"] is not None]
            if valid:
                line, = axis.plot([g["training_episodes_per_round"] for g in valid],
                          [g[metric]["mean"] for g in valid], marker="o", label=VALUES[method])
                repeated = [g for g in valid if g[metric]["std_across_seeds"] is not None]
                axis.errorbar([g["training_episodes_per_round"] for g in repeated],
                              [g[metric]["mean"] for g in repeated],
                              yerr=[g[metric]["std_across_seeds"] for g in repeated], fmt="none",
                              ecolor=line.get_color(), capsize=4)
            axis.set(xlabel="每轮训练回合数", ylabel=title)
            axis.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("独立种子比较：均值与标准差")
    for extension in ("png", "svg"):
        fig.savefig(directory/f"comparison.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--overview-only", action="store_true", help="只绘制已完成轮次总览，可用于正在运行的训练")
    parser.add_argument("--detailed", action="store_true", help="额外生成详细分图、SVG 和总览数据")
    args = parser.parse_args()
    directory = args.run_dir
    if args.overview_only:
        from .round_overview import plot_round_overview
        plot_round_overview(directory)
    elif (directory/"aggregate.json").exists():
        plot_aggregate(directory)
    else:
        plot_training(directory, detailed=args.detailed)
