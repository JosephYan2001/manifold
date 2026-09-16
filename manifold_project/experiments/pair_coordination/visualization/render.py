"""Render a diagnostic episode and data flow. No learned policy is implied."""
import argparse
from pathlib import Path

import numpy as np

from ..envs.pair_coordination import PairConfig, PairCoordinationEnv


def episode_contributions(config, types, actions):
    """Validate using the real environment before deriving visual contributions."""
    env = PairCoordinationEnv(config)
    reward = env.reward(types, actions)
    x = np.asarray(types, dtype=int)
    a = np.asarray(actions, dtype=int)
    signs = 2 * a - 1
    local = np.asarray(config.local_bias)[x] * signs
    pairs = []
    for i in range(config.n_agents):
        for j in range(i + 1, config.n_agents):
            value = (config.interaction_strength / (config.n_agents - 1)
                     * config.pair_payoff[x[i]][x[j]] * signs[i] * signs[j])
            pairs.append((i, j, float(value)))
    if not np.isclose(local.sum() + sum(v for _, _, v in pairs), reward,
                      atol=1e-10, rtol=1e-12):
        raise AssertionError("Visual reward decomposition disagrees with the environment")
    return {"types": x, "actions": a, "signs": signs, "local": local,
            "pairs": pairs, "team_reward": reward}


def _plotting():
    # Core environment imports/tests do not require the optional plotting dependency.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun")
             if f in available]
    if not fonts:
        raise RuntimeError("Install a Chinese font (e.g. Noto Sans CJK SC) before rendering")
    plt.rcParams.update({"font.sans-serif": fonts + ["DejaVu Sans"],
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "font.size": 11, "figure.facecolor": "#ffffff",
                         "axes.facecolor": "#ffffff"})
    return plt


def _save(figure, output_dir, stem, plt):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path.resolve())
    plt.close(figure)
    return paths


def render_episode(config, types, actions, output_dir):
    if config.n_agents > 8:
        raise ValueError("Episode figure supports at most 8 agents to preserve legibility")
    data = episode_contributions(config, types, actions)
    plt = _plotting()
    from matplotlib.patches import Circle
    count = config.n_agents
    figure, (network, bars) = plt.subplots(1, 2, figsize=(13, max(6, count * .9)),
                                          gridspec_kw={"width_ratios": [1.05, 1.2]})
    figure.subplots_adjust(top=.83, bottom=.13, wspace=.40)
    positive, negative = "#18756c", "#b44931"
    local_sum = float(data["local"].sum())
    pair_sum = sum(v for _, _, v in data["pairs"])
    figure.suptitle("单回合协作与奖励分解", fontsize=19, y=.98)
    figure.text(.5, .915,
                f"个体收益 {local_sum:+.3f}  +  交互收益 {pair_sum:+.3f}  =  团队奖励 {data['team_reward']:+.3f}",
                ha="center", fontsize=13)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2*np.pi, count, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    for i, j, value in data["pairs"]:
        left, right = positions[i], positions[j]
        network.plot([left[0], right[0]], [left[1], right[1]],
                     color=positive if value >= 0 else negative,
                     linestyle="-" if value >= 0 else "--", linewidth=1.8,
                     alpha=.65, zorder=1)
    for i, (px, py) in enumerate(positions):
        network.add_patch(Circle((px, py), .29, facecolor="#f1f5f6",
                                 edgecolor="#536674", linewidth=1.3, zorder=3))
        network.text(px, py+.07, f"机器人{i+1}", ha="center", va="center", zorder=4, fontsize=10)
        network.text(px, py-.07, f"类型 {data['types'][i]}", ha="center", va="center", zorder=4, fontsize=10)
        network.text(px*1.4, py*1.4, f"行为 {data['signs'][i]:+d}",
                     ha="left" if px > .5 else "right" if px < -.5 else "center",
                     va="center", color="#1e2934", fontsize=10)
    network.set(xlim=(-1.85, 1.85), ylim=(-1.85, 1.85), aspect="equal")
    network.axis("off")
    network.set_title("节点：机器人｜连线：本回合交互收益", pad=14)
    network.text(.5, -.06, "实线：非负收益    虚线：负收益\n排布只表示关系，不表示物理位置",
                 transform=network.transAxes, ha="center", va="top", color="#536674")

    labels = [f"机器人 {i+1} · 个体" for i in range(count)]
    labels += [f"机器人 {i+1}—{j+1} · 交互" for i, j, _ in data["pairs"]]
    values = list(data["local"]) + [v for _, _, v in data["pairs"]]
    scale = max(max(abs(v) for v in values), .1)
    y = np.arange(len(values))
    bars.barh(y, values, color=[positive if v >= 0 else negative for v in values], height=.65)
    bars.set_yticks(y, labels)
    bars.invert_yaxis()
    bars.axvline(0, color="#81909a", linewidth=.8)
    bars.axhline(count-.5, color="#d9dfe3", linewidth=.8)
    for yi, value in enumerate(values):
        bars.text(value + (.035*scale if value >= 0 else -.035*scale), yi,
                  f"{value:+.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=10)
    bars.set_xlim(min(0, min(values))-.45*scale, max(0, max(values))+.45*scale)
    bars.set_xlabel("对团队奖励的贡献")
    bars.set_ylabel("奖励项")
    bars.set_title("每一项均由当前配置、类型与动作计算", pad=14)
    bars.spines[["top", "right", "left"]].set_visible(False)
    bars.tick_params(axis="y", length=0)
    return _save(figure, output_dir, "episode-rewards", plt)


def render_flow(output_dir):
    plt = _plotting()
    figure, axes = plt.subplots(1, 2, figsize=(12, 10))
    figure.subplots_adjust(top=.87, bottom=.06, wspace=.28)
    figure.suptitle("从环境回合到策略更新", fontsize=20, y=.975)
    figure.text(.5, .935, "本地类型决定动作；源经验用于学习方向与拟合 actor", ha="center", color="#536674")
    execution = [
        ("环境重置", "源配置与随机种子 → 全队类型 H"),
        ("按机器人读取本地输入", "机器人 i 只读取自身类型 Xi"),
        ("冻结概率表与独立采样", "μ(·|Xi) → 各机器人的动作 Ui"),
        ("环境执行联合动作", "全部类型、动作 → 共同团队奖励 R"),
        ("回合记录 D", "本地类型、动作、旧概率、团队奖励"),
    ]
    training = [
        ("源回合 D 与冻结 actor", "本轮旧概率始终固定"),
        ("团队评价标签", "R 减去动作前基线 → Â"),
        ("训练本地方向", "Xi、Â、旧概率和动作 → qω"),
        ("独立方向检查", "新源回合评价；通过后继续"),
        ("指数目标", "旧概率 μ、方向 qω、步幅 η → 目标分布"),
        ("拟合实际 actor", "本地输入与固定目标 → 候选参数"),
        ("独立回报检查", "接受候选或保留旧 actor → 下一轮"),
    ]
    for ax, nodes, heading in zip(axes, [execution, training],
                                  ["执行与采样", "类型表参数更新"]):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        ax.set_title(heading, fontsize=14, pad=18)
        ys = np.linspace(.93, .12 if ax is axes[1] else .36, len(nodes))
        for index, ((title, detail), ypos) in enumerate(zip(nodes, ys)):
            ax.text(.5, ypos, title + "\n" + detail, ha="center", va="center", fontsize=11,
                    linespacing=1.65, bbox={"boxstyle": "round,pad=.65", "facecolor": "#f1f5f6",
                                         "edgecolor": "#b7c6ca", "linewidth": .8})
            if index:
                ax.annotate("", xy=(.5, ypos+.052), xytext=(.5, ys[index-1]-.052),
                            arrowprops={"arrowstyle": "->", "color": "#526670", "lw": 1.4})
        if ax is axes[0]:
            ax.text(.5, .17, "精确评价（独立诊断）\n配置 + 冻结策略 → 真实回报与方向",
                    ha="center", va="center", linespacing=1.7, color="#18756c")
            ax.text(.5, .055, "真值用于误差核对\n普通采样训练不读取真实方向", ha="center",
                    va="center", fontsize=10, color="#536674")
    return _save(figure, output_dir, "environment-training-flow", plt)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/pair_source.json")
    parser.add_argument("--types", type=int, nargs="+", default=[0, 0, 1, 1])
    parser.add_argument("--actions", type=int, nargs="+", default=[0, 1, 1, 1])
    parser.add_argument("--output-dir", type=Path, default=root / "results/visualization")
    args = parser.parse_args()
    config = PairConfig.from_json(args.config)
    for path in render_episode(config, args.types, args.actions, args.output_dir) + render_flow(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
