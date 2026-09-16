"""支持文件路径及模块方式启动训练，命令行参数覆盖 JSON 配置。"""
import argparse
from dataclasses import fields, asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys

# 直接执行文件时，根据脚本位置补齐包上下文；不改变用户配置路径的解析目录。
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "manifold_project.experiments.pair_coordination"

from .envs.pair_coordination import PairConfig
from .training.config import TrainConfig
from .training.runner import run_training
from .training.logging import save_json
import itertools
import numpy as np
from .training.descriptions import PARAMETERS, METRICS
from .training.plan import ALIASES, normalize, training_plan, print_plan, public_config

# python manifold_project\experiments\pair_coordination\train.py --config manifold_project/experiments/pair_coordination/configs/pair_source.json --training-config manifold_project/experiments/pair_coordination/configs/train_minibatch.json --acceptance empirical --plot
def build_parser():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description="单源成对协作训练；配置优先级：命令行 > JSON > 默认值")
    parser.add_argument("--config", type=Path, default=root/"configs/pair_source.json", help="源环境规则配置文件")
    parser.add_argument("--training-config", type=Path, help="训练参数 JSON 配置文件")
    parser.add_argument("--output-dir", type=Path, help="保存结果的新目录")
    parser.add_argument("--resume", type=Path, help="从已完成轮次 checkpoint 恢复，输出到新目录，沿用原配置")
    parser.add_argument("--episodes-per-round", type=int, help="每轮采集回合数，等价于 --episodes")
    parser.add_argument("--max-rounds", type=int, help="最多训练轮数")
    parser.add_argument("--max-source-episodes", type=int, help="全部源交互回合预算，含所有检查")
    parser.add_argument("--actor-lr", type=float, help="actor 拟合学习率（恒定）")
    parser.add_argument("--dry-run", action="store_true", help="检查配置与预算并展示训练计划，随后退出")
    parser.add_argument("--n-agents", type=int, help="覆盖源团队人数，每次训练内固定")
    parser.add_argument("--seeds", type=int, nargs="+", help="指定多个独立随机种子")
    parser.add_argument("--methods", nargs="+", choices=["analytic", "sampled"], help="批量比较两种方向损失")
    parser.add_argument("--sample-sizes", type=int, nargs="+", help="批量比较每轮训练回合数")
    parser.add_argument("--print-config", action="store_true", help="显示生效配置并退出")
    parser.add_argument("--plot", action="store_true", help="训练后生成中文 PNG/SVG 曲线，需要 Matplotlib")
    choices = {"algorithm": ["direction", "pg"], "actor_model": ["mlp", "table"], "direction_model": ["mlp", "table"],
               "mode": ["direction", "policy"], "method": ["analytic", "sampled"],
               "baseline": ["zero", "table"], "acceptance": ["hoeffding", "empirical"]}
    groups = {name: parser.add_argument_group(name) for name in
              ("模型", "数据与优化", "检查与预算", "保存与日志")}
    for field in fields(TrainConfig):
        default_hint = (f"新运行未指定计数方式时：{128 if field.name == 'batch_size_episodes' else 25}；兼容模式见启动计划"
                        if field.name in ("batch_size_episodes", "direction_epochs", "actor_epochs")
                        else f"默认：{field.default}")
        group = groups["模型" if field.name in ("actor_model", "direction_model", "hidden_width", "hidden_depth", "beta", "q_max")
                       else "保存与日志" if field.name in ("save_batches", "save_direction_snapshots", "checkpoint_every", "log_every")
                       else "检查与预算" if field.name in ("check_episodes", "acceptance", "attempts", "alpha", "eta", "budget", "rounds")
                       else "数据与优化"]
        if type(field.default) is bool:
            group.add_argument("--"+field.name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                                default=None, help=PARAMETERS[field.name])
            continue
        group.add_argument("--"+field.name.replace("_", "-"), type=type(field.default), default=None,
                            choices=choices.get(field.name), help=argparse.SUPPRESS if field.name in
                            ("episodes", "rounds", "budget", "fit_lr", "direction_steps", "fit_steps")
                            else f"{PARAMETERS[field.name]}（{default_hint}）")
    return parser


def resolve_configs(args):
    if args.resume:
        if (any(getattr(args, f.name) is not None for f in fields(TrainConfig) if f.name != "device")
                or args.training_config or args.n_agents is not None or args.seeds or args.methods
                or args.sample_sizes or args.episodes_per_round is not None
                or args.max_rounds is not None or args.max_source_episodes is not None or args.actor_lr is not None
                or args.config != Path(__file__).parent/"configs/pair_source.json"):
            raise ValueError("恢复时沿用 checkpoint 配置，请移除训练参数覆盖")
        state = json.loads(args.resume.read_text(encoding="utf-8"))
        settings = TrainConfig(**state["training"])
        if args.device is not None:
            settings = replace(settings, device=args.device)
        return PairConfig(**state["source"]), [settings]
    values = normalize(json.loads(args.training_config.read_text(encoding="utf-8"))) if args.training_config else {}
    cli = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name) is not None}
    cli.update({k: getattr(args, k) for k in ALIASES if getattr(args, k) is not None})
    values.update(normalize(cli))
    legacy_names = sorted(set(cli) & {"episodes", "rounds", "budget", "fit_lr", "direction_steps", "fit_steps"})
    if legacy_names:
        print(f"兼容参数：{', '.join(legacy_names)}；新配置建议使用规范名称与 epoch 计数。", file=sys.stderr)
    for epoch, steps in (("direction_epochs", "direction_steps"), ("actor_epochs", "fit_steps")):
        if values.get(epoch, 0) > 0 and steps in values:
            raise ValueError(f"不能混用 {epoch} 与 {steps}，请删除旧 steps 参数")
    if not any(k in values for k in ("direction_steps", "fit_steps", "direction_epochs", "actor_epochs")):
        values.update(direction_epochs=25, actor_epochs=25)
        values.setdefault("batch_size_episodes", 128)
    unknown = set(values)-{f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"未知配置参数：{sorted(unknown)}")
    settings = TrainConfig(**values)
    if settings.baseline == "zero" and args.critic_episodes is not None:
        raise ValueError("baseline=zero 时 critic-episodes 不生效，请开启 --baseline table 或移除此参数")
    source = PairConfig.from_json(args.config)
    if args.n_agents is not None:
        source = replace(source, n_agents=args.n_agents)
    seeds, methods, sizes = args.seeds or [settings.seed], args.methods or [settings.method], args.sample_sizes or [settings.episodes]
    for name, items in [("seeds", seeds), ("methods", methods), ("sample-sizes", sizes)]:
        if len(set(items)) != len(items):
            raise ValueError(f"Duplicate {name}")
    configs = [replace(settings, seed=seed, method=method, episodes=size)
               for method, size, seed in itertools.product(methods, sizes, seeds)]
    return source, configs


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        source, configs = resolve_configs(args)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if args.print_config:
        print(json.dumps({"source": asdict(source), "runs": [asdict(c) for c in configs]}, indent=2))
        return
    try:
        from .models.torch_models import resolve_device
        for config in configs:
            print(f"PyTorch 训练设备：{resolve_device(config.device)}")
    except (ImportError, ValueError, RuntimeError) as error:
        parser.error(f"PyTorch 环境检查失败：{error}；请使用已安装 PyTorch 的 Python 环境")
    for config in configs:
        plan = training_plan(source, config)
        if config.budget < plan["minimum_to_start_round"]:
            parser.error(f"预算不足以开始一轮，至少需要 {plan['minimum_to_start_round']} 回合")
        print(f"训练配置文件：{args.training_config.resolve() if args.training_config else '程序默认值或恢复配置'}")
        print_plan(plan)
    if args.dry_run:
        return
    root = Path(__file__).parent
    output = args.output_dir or root/"results"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if len(configs) > 1:
        output.mkdir(parents=True, exist_ok=False)
        save_json(output/"plan.json", {"source": asdict(source), "runs": [asdict(c) for c in configs]})
    results = []
    for config in configs:
        path = output if len(configs) == 1 else output/f"{config.method}_N{config.episodes}_seed{config.seed}"
        result = run_training(source, config, path, resume=args.resume,
                              requested_config={"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                                "file_contents": json.loads(args.training_config.read_text(encoding="utf-8")) if args.training_config else None})
        results.append((config, result))
        print(f"可加载的最终 actor：{result['checkpoint']}")
        print(json.dumps({"结果目录": str(path), "已用源回合数": result["source_episodes"],
                          "最终执行策略": {METRICS[k]: v for k, v in result["final_actor_diagnostics"].items()},
                          "最后一轮旧策略处的方向": {METRICS[k]: v for k, v in result["last_direction_at_reference_actor"].items()}},
                         ensure_ascii=False, indent=2))
        if args.plot:
            from .analysis.plot_training import plot_training
            plot_training(path)
    if len(configs) > 1:
        groups = []
        for method, size in sorted({(c.method, c.episodes) for c in configs}):
            selected = [(c, r) for c, r in results if c.method == method and c.episodes == size]
            group = {"method": method, "training_episodes_per_round": size,
                     "seeds": [c.seed for c, _ in selected], "repeats": len(selected)}
            for metric, location in [("expected_return", "final_actor_diagnostics"),
                                     ("direction_error", "last_direction_at_reference_actor")]:
                data = [r[location][metric] for _, r in selected if metric in r[location]]
                group[metric] = {"count": len(data), "mean": float(np.mean(data)) if data else None,
                                 "std_across_seeds": float(np.std(data, ddof=1)) if len(data) > 1 else None}
            groups.append(group)
        save_json(output/"aggregate.json", {"groups": groups,
                  "direction_reference": "Each run's last frozen actor; policy-mode references may differ"})
        if args.plot:
            from .analysis.plot_training import plot_aggregate
            plot_aggregate(output)
    print(f"结果已保存：{output.resolve()}")


if __name__ == "__main__":
    main()
