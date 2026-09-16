"""One nine-panel overview for the whole suite; all raw observations stay in CSV."""
import argparse
import json
from pathlib import Path
import numpy as np
from .reporting import read_csv, bootstrap

LABELS = {"ours": "完整方法", "sampled": "采样平方", "pg": "PG",
          "no_direction_check": "去方向检查", "no_return_check": "去回报检查", "fit_quarter": "1/4 拟合"}


def plot_suite(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..analysis.plot_training import chinese_fonts
    chinese_fonts(plt)
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    repeats = manifest["identity"]["config"]["bootstrap_repeats"]

    def read(name):
        path = directory/name
        return read_csv(path) if path.exists() else []

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), layout="constrained")
    axes = list(axes.flat)
    curves = read("learning_curves.csv")
    for condition in dict.fromkeys(r["condition"] for r in curves):
        selected = [r for r in curves if r["condition"] == condition]
        xs = sorted({int(r["budget_checkpoint"]) for r in selected})
        stats = [bootstrap([float(r["expected_return"]) for r in selected if int(r["budget_checkpoint"]) == x], repeats) for x in xs]
        line, = axes[0].step(xs, [r["mean"] for r in stats], where="post", label=LABELS[condition])
        if stats[0]["count"] > 1:
            axes[0].fill_between(xs, [r["ci_low"] for r in stats], [r["ci_high"] for r in stats],
                                 step="post", color=line.get_color(), alpha=.12)
    axes[0].set(title="P-C / P-A：精确源回报", xlabel="累计源回合预算", ylabel="团队期望回报")

    runs = [r for r in read("training_results.csv") if r["status"] == "complete"]
    conditions = list(dict.fromkeys(r["condition"] for r in runs))
    bottoms = np.zeros(len(conditions))
    for key, label in (("direction_train_episodes", "方向训练"), ("pg_train_episodes", "PG训练"),
                       ("direction_check_episodes", "方向检查"), ("return_old_episodes", "旧策略检查"),
                       ("return_candidate_episodes", "候选检查")):
        heights = [np.mean([float(r[key]) for r in runs if r["condition"] == c]) for c in conditions]
        if sum(heights) == 0:
            continue
        axes[1].bar([LABELS[c] for c in conditions], heights, bottom=bottoms, label=label)
        bottoms += heights
    axes[1].tick_params(axis="x", rotation=25, labelsize=8)
    axes[1].set(title="P-C / P-A：实际源采样成本", ylabel="平均源回合数")

    estimates = read("P-M1/summary.csv")
    for axis, label in zip(axes[2:4], ("return", "exact_advantage")):
        for method in ("analytic", "sampled"):
            selected = sorted([r for r in estimates if r["actor_id"] == "1" and r["label"] == label
                               and r["method"] == method and r["metric"] == "direction_error"],
                              key=lambda r: int(r["sample_size"]))
            if selected:
                xs = [int(r["sample_size"]) for r in selected]
                line, = axis.plot(xs, [float(r["mean"]) for r in selected], marker="o", label=method)
                if all(r["ci_low"] for r in selected):
                    axis.fill_between(xs, [float(r["ci_low"]) for r in selected],
                                      [float(r["ci_high"]) for r in selected], alpha=.15, color=line.get_color())
        axis.set(title="P-M1：偏置策略 / "+("回报标签" if label == "return" else "精确优势"),
                 xlabel="独立训练回合数", ylabel="真实方向误差", xscale="log")

    fits = read("P-M2/records.csv")
    if fits:
        max_steps = max(int(r["updates"]) for r in fits)
        for kind in ("table", "tied_logit"):
            rows = sorted([r for r in fits if r["actor_class"] == kind and int(r["updates"]) == max_steps],
                          key=lambda r: float(r["eta"]))
            xs = [float(r["eta"]) for r in rows]
            axes[4].plot(xs, [float(r["actor_gain"]) for r in rows], marker="o", label=kind)
            axes[5].plot(xs, [float(r["fit_kl"]) for r in rows], marker="o", label=kind)
        axes[4].plot(xs, [float(r["target_gain"]) for r in rows], "k--", label="指数目标")
    axes[4].set(title="P-M2：最大拟合预算下的回报增量", xlabel="步幅", ylabel="精确回报增量")
    axes[5].set(title="P-M2：最大拟合预算下的目标实现", xlabel="步幅", ylabel="总体前向 KL")

    transports = read("P-M3/records.csv")
    if transports:
        xs = [float(r["transport_bound"]) for r in transports]
        ys = [float(r["target_error"]) for r in transports]
        axes[6].scatter(xs, ys, alpha=.6, label="源方向 / 目标环境")
        maximum = max(max(xs), max(ys), 1e-10)
        axes[6].plot([0, maximum], [0, maximum], "k--", label="误差 = 上界")
    axes[6].set(title="P-M3：运输界核验", xlabel="理论目标误差上界", ylabel="实际目标误差")
    boundary = read("P-M4/records.csv")
    if boundary:
        xs = [float(r["epsilon"]) for r in boundary]
        axes[7].plot(xs, [float(r["actual_gain"]) for r in boundary], "o", label="精确回报增量")
        axes[7].plot(xs, [float(r["expected_gain"]) for r in boundary], "--", label="4ε²")
    axes[7].set(title="P-M4：零一阶方向的二阶改进", xlabel="共同概率扰动 ε", ylabel="回报增量")
    checks = read("P-H/summary.csv")
    for sign, label in (("1", "真实改善"), ("0", "零改善"), ("-1", "真实退化")):
        for mode, style in (("empirical", "-"), ("hoeffding", "--")):
            rows = sorted([r for r in checks if r["kind"] == "return" and r["truth_class"] == sign
                           and r["mode"] == mode and r["metric"] == "accepted"], key=lambda r: int(r["sample_size"]))
            if rows:
                axes[8].plot([int(r["sample_size"]) for r in rows], [float(r["mean"]) for r in rows],
                             style, marker=".", label=f"{label}/{mode}")
    axes[8].set(title="P-H：固定候选的回报检查", xlabel="每策略检查回合数", ylabel="接受比例", ylim=(-.05, 1.05))
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
        elif not axis.has_data():
            axis.text(.5, .5, "本次未执行／无完成记录", ha="center", transform=axis.transAxes)
        axis.grid(alpha=.2)
    state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    failed = sum(r["status"] != "complete" for r in state["jobs"].values())
    fig.suptitle(f"成对协作实验总览｜{manifest['protocol_status']}｜失败或中断任务 {failed}\n"
                 "训练图仅绘制完成运行；单种子无置信区间；机制总览展示指定切片，完整数据见 CSV", fontsize=13)
    path = directory/"overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    plot_suite(parser.parse_args().directory)
