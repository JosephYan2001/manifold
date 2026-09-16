"""配置名称转换与启动计划；内部字段兼容已保存的 checkpoint。"""
from dataclasses import asdict
import math

ALIASES = {"episodes_per_round": "episodes", "max_rounds": "rounds",
           "max_source_episodes": "budget", "actor_lr": "fit_lr"}


def normalize(values):
    values = dict(values)
    for new, old in ALIASES.items():
        if new in values:
            if old in values and values[old] != values[new]:
                raise ValueError(f"配置冲突：{new} 与 {old}")
            values[old] = values.pop(new)
    return values


def public_config(settings):
    values = asdict(settings)
    for new, old in ALIASES.items():
        values[new] = values.pop(old)
    if settings.direction_epochs:
        values.pop("direction_steps")
    if settings.actor_epochs:
        values.pop("fit_steps")
    return values


def training_plan(source, settings):
    n, m = settings.episodes, settings.check_episodes
    c = settings.critic_episodes if settings.baseline == "table" else 0
    batches = math.ceil(n/(settings.batch_size_episodes or n))
    policy = settings.mode == "policy"
    pg = settings.algorithm == "pg"
    direction_cost = m if policy and not pg and settings.direction_check_enabled else 0
    return_cost = 2*m if policy and not pg and settings.return_check_enabled else 0
    one_cost = n+c+direction_cost+return_cost
    rounds = settings.rounds if policy else 1
    return {"episode_steps": 1, "n_agents": source.n_agents, "episodes_per_round": n,
            "batch_size_episodes": min(settings.batch_size_episodes or n, n),
            "batches_per_epoch": batches, "direction_epochs": settings.direction_epochs,
            "actor_epochs": settings.actor_epochs,
            "direction_updates": 0 if pg else (batches*settings.direction_epochs if settings.direction_epochs else settings.direction_steps),
            "actor_updates_per_candidate": 1 if pg else ((batches*settings.actor_epochs if settings.actor_epochs else settings.fit_steps) if policy else 0),
            "direction_lr": settings.direction_lr, "actor_lr": settings.fit_lr,
            "learning_rate_schedule": "constant", "baseline": settings.baseline,
            "actor_model": settings.actor_model, "direction_model": settings.direction_model,
            "hidden_width": settings.hidden_width, "hidden_depth": settings.hidden_depth,
            "algorithm": settings.algorithm,
            "direction_rejected_cost": n+c+direction_cost,
            "one_candidate_cost": one_cost,
            "maximum_round_cost": n+c+direction_cost+settings.attempts*return_cost,
            "minimum_to_start_round": one_cost,
            "max_rounds": rounds, "max_source_episodes": settings.budget,
            "one_candidate_total_cost": rounds*one_cost}


def print_plan(plan):
    print(f"源任务：{plan['n_agents']} 个机器人，每 episode {plan['episode_steps']} 步", flush=True)
    print(f"模型：actor={plan['actor_model']}，方向={plan['direction_model']}；"
          f"隐藏层 {plan['hidden_width']}×{plan['hidden_depth']}；基线={plan['baseline']}", flush=True)
    print(f"每轮采集 {plan['episodes_per_round']} 回合；每批 {plan['batch_size_episodes']} 回合；"
          f"每 epoch {plan['batches_per_epoch']} 批", flush=True)
    print(f"方向更新 {plan['direction_updates']} 次，学习率 {plan['direction_lr']}；"
          f"每候选 actor 更新 {plan['actor_updates_per_candidate']} 次，学习率 {plan['actor_lr']}（均为恒定学习率）", flush=True)
    if plan['baseline'] == 'table':
        print("critic：独立源批次的类型表均值拟合，无梯度学习率。", flush=True)
    if plan['direction_epochs'] == 0:
        print("兼容模式：方向使用全批次 steps 计数。", flush=True)
    print(f"上限：{plan['max_rounds']} 轮 / {plan['max_source_episodes']} 源回合；"
          f"每轮方向拒绝成本 {plan['direction_rejected_cost']}，一个候选成本 {plan['one_candidate_cost']}，"
          f"最大成本 {plan['maximum_round_cost']}", flush=True)
    if plan['one_candidate_total_cost'] > plan['max_source_episodes']:
        print(f"预算提示：每轮检查一个候选时，全部轮次需要 {plan['one_candidate_total_cost']} 回合；"
              "当前预算可能提前结束训练。", flush=True)
