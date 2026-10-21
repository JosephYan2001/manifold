# 六组 AN 2M 训练与冻结 MAPPO-E 评价命令

2026-10-19 编写，2026-10-22 更新状态：六组 AN 训练及各 100 场景终点评价已完成，见[结果分析](../../../results/evidence/navigation/navigation_batch_lr_2m_s40_20261019_v1/analysis_20261022/六组训练与旧基线对比.md)。MAPPO-E 同场景冻结评价尚未提供。下列命令保留用于复现，已有结果目录不要再次普通启动覆盖。[CPU 工程验收](../../../results/checks/navigation_batch_lr_acceptance_20261019_v1/验收报告.md)与[原六组方案](../../../results/notes/导航采样量与学习率六组实验方案_20261019.md)分别记录工程检查和实验定义。

## 设置

| 配置 | 每轮采样回合数 | Actor 学习率 |
|---|---:|---|
| n16_fixed.json | 16 | 固定 3e-4 |
| n16_linear.json | 16 | 按源交互进度从 3e-4 衰减至 3e-5 |
| n32_fixed.json | 32 | 固定 3e-4 |
| n32_linear.json | 32 | 同上衰减 |
| n64_fixed.json | 64 | 固定 3e-4 |
| n64_linear.json | 64 | 同上衰减 |

全部从头初始化，seed 40，每组 2M 源预算，包含训练和源监控。方向 4 epoch、Actor 16 epoch、Critic 10 epoch；每小批 2 回合。Actor 使用 best_train_kl，包含零更新候选。不开方向/回报门控，不额外采样诊断数据。

`actor_lr_schedule` 只控制 AN Actor 拟合；每轮采样前读取源计数并计算学习率，同轮所有拟合步骤使用同一值。`actor_lr_final=3e-5`、`actor_lr_decay_steps=2000000` 固定保存。首轮日程原点 s0 在采样前持久化，恢复继承；将来显式增加预算也不会重置原日程。方向和 Critic 学习率、教师 η 不变。

## PowerShell 执行

先激活可用的 Python / CUDA 环境，再切换到包含 `manifold_project` 的项目根目录。以下命令使用 `python`，没有 `$py`；需要在同一个 PowerShell 会话中执行变量定义。

```powershell
Set-Location D:\manifold
$cfg = "manifold_project/experiments/configs/navigation/batch_lr_2m"
$out = "manifold_project/experiments/results/evidence/navigation/navigation_batch_lr_2m_s40_20261019_v1"
```

顺序运行六组；每组结束后，公共入口自动在场景 20262000–20262099 上评价 final.pt，最多 1,000 步，只评价源规模 4 人，无需另跑一次 AN 终点评价。出现错误时循环停止，不继续消耗下一组预算。

```powershell
foreach ($run in @("n16_fixed", "n16_linear", "n32_fixed", "n32_linear", "n64_fixed", "n64_linear")) {
    python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/$run.json" --budget 2000000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/$run" --plot --plot-every 5 --allow-duplicate-openmp
    if ($LASTEXITCODE -ne 0) { throw "$run 执行失败；请先检查该组日志，并使用 --resume 恢复。" }
}
```

如果只运行其中一组，例如 N32 衰减：

```powershell
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/n32_linear.json" --budget 2000000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/n32_linear" --plot --plot-every 5 --allow-duplicate-openmp
```

已开始或已完成的目录不能再次使用普通启动命令覆盖。只查看完整配置时，在对应命令末尾加 `--dry-run`；不会训练或创建结果目录。

然后执行一次历史 MAPPO-E 的冻结评价。`N-T` 是现有公共冻结评价入口；此配置限定 `target_sizes=[4]`，因此不运行 2/6/8 人迁移，不拟合 Actor，不更新 PPO。

```powershell
$checkpoint = "manifold_project/experiments/results/evidence/navigation/navigation_ep400_history8_2m_s40/mappo/N-C_source_learning/full/MAPPO-E_s40/checkpoints/final.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "找不到历史 MAPPO-E final.pt，请确认本地检查点路径。" }
python -m manifold_project.experiments --experiments N-T --profile pilot --methods MAPPO-E --config "$cfg/mappo_reference.json" --checkpoints "$checkpoint" --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/mappo_reference_evaluation" --allow-duplicate-openmp
```

该评价与 AN 使用相同的 100 个场景、随机动作规则和 1,000 步上限，保留历史模型与任务输入协议。原始 MAPPO-E 目录只读。新增总交互上限 12.7M：六组源预算 12M、六组最终评价 0.6M、MAPPO-E 冻结评价 0.1M；历史训练成本另列。

## 中断恢复

例如 N32 衰减组中断后：

```powershell
python -m manifold_project.experiments --resume "$out/n32_linear" --allow-duplicate-openmp
```

恢复时不要再传 `--config`，也不要用恢复入口切换 N、日程或其他变体。已提交边界恢复与不中断运行一致；未提交采样按现有账本计费，保留 s0，下一轮学习率随实际源计数推进。恢复该组后，手动继续尚未开始的组，不从六组循环开头重跑。

冻结评价不是训练恢复任务。如果它中断，保留失败目录及评价账本，另用新输出后缀重试；失败成本单列。公共训练恢复过程中若重复最终评价，`evaluation_audit_001.json` 等各次审计与逐回合账本均保留，不能只计算最后一次成功评价的成本。

## 查看结果

- AN：`<组名>/N-C_source_learning/full/AN_s40/training.csv`，检查 `actor_lr_used`、`fit_actor_lr`、`actor_lr_source_steps`、完整训练批次 Critic 指标、`direction_train_*`、所选 epoch 和实际 KL。
- Checkpoint：保存 `actor_lr_schedule_state` 与 `initial_model_sha256`；六组初始 Actor/Critic 哈希已在工程验收中验证一致。
- 最终评价：每组 `final_evaluation/evaluation_episodes.csv`，以及冻结 MAPPO-E 输出中的逐场景 CSV。
- `evaluation_audit.json` 是最近一次评价状态；编号审计和 `evaluation_ledger_*.jsonl` 保留每次尝试的模型哈希、场景、实际评价成本、失败时未完成回合预留上限。训练采样与优化次数为零。

运行总报告只比较预先规定的全部对照。MAPPO-E 是历史适配策略的参照，不标为原版 MAPPO 复现。本轮工程验收不属于训练性能证据。
