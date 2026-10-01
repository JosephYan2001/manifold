# 实验运行说明

统一入口为 `python -m manifold_project.experiments`。成对学习复用 pair_coordination 训练器，导航、仓库和实体成对任务共用 cooperative_navigation 的 Runner；有限机制诊断由 finite 组织。完整研究安排见[总体实验方案](../docs/总体实验方案.md)，代码职责见[项目结构与代码说明](../docs/项目结构与代码说明.md)。

## 1. 安装与入口

在包含 manifold_project 的目录运行，本机为 D:\manifold。使用已有 PyTorch/CUDA 环境，或激活项目环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r manifold_project/experiments/requirements.txt
python -m manifold_project.experiments --list
```

MPE2 固定 1.1.1，RWARE 固定 2.0.0。入口在计算库导入前设置 CUDA 确定性所需环境变量，尊重调用者已设置的值；运行清单记录实际运行环境。此前使用的 OpenMP 重复库兼容开关可通过 `--allow-duplicate-openmp` 显式开启，不表示依赖冲突已经根治。

## 2. 去哪里找实验命令

| 任务 | 执行文档 | 当前用途 |
|---|---|---|
| 协作导航 | [导航实验手册](cooperative_navigation/README.md) | 新协议单种子源训练、四方法比较、冻结评价和 GIF |
| 成对机制与连续学习 | [后续对照执行说明](finite/后续对照执行说明.md)、[有限预算四组](configs/finite_budget_study/README.md) | KL 对照、检查机制、P-L/T-L 和预算消融 |
| 全部实验编号与预期结果 | [总体实验方案](../docs/总体实验方案.md) | P/T/N/W 的研究问题、方法与指标 |
| 已有证据 | [结果索引](results/README.md) | 已完成结果、历史边界和分析入口 |

导航的唯一完整运行说明在导航 README。当前先运行 AN 无检查与 MAPPO-E，再比较 AN 双检查和保留回报检查的 DA。success-v3 的 200 步训练段、1000 步外部训练重置与评估观察上限分别设置；训练重置保留末状态自举。旧限时与 v2 无限延续结果不与修正后的结果混用。

其他入口示例：

```powershell
# 固定基点机制诊断；只解析时追加 --dry-run
python -m manifold_project.experiments --experiments P-E P-A P-T --profile pilot --output manifold_project/experiments/results/pair_mechanisms_new --plot

# P-EA 直接使用已有 P-E 的方向，不重新学习
python -m manifold_project.experiments --experiments P-EA --profile formal --source manifold_project/experiments/results/pair_estimation_20seeds_v1 --config manifold_project/experiments/configs/pair_ea_mc.json --sample-sizes 2048 --output manifold_project/experiments/results/pair_ea_new --plot

# 仓库源训练与冻结评价
python -m manifold_project.experiments --experiments W-C --profile pilot --device cuda --output manifold_project/experiments/results/warehouse_source_new --plot
python -m manifold_project.experiments --experiments W-T W-L --profile pilot --source manifold_project/experiments/results/warehouse_source_new --device cuda --output manifold_project/experiments/results/warehouse_transfer_new --plot
```

这些是入口示例，不要求重做已完成的成对实验。已有目录不覆盖；P-EA 使用 data-seeds，普通训练使用 seeds。N-F/W-F 等拟合诊断需要源训练提前保存所需数据，不能从未启用诊断的运行事后补出。

## 3. 配置的组织

合并顺序为 `configs/defaults.json → configs/<环境>.json → profiles.json → --config → 命令参数与实验设置`，随后 current_protocol.py 映射到底层训练器。以 `--dry-run` 和运行内保存的完整配置为准。

| 配置层 | 应修改什么 |
|---|---|
| defaults.json | 全部实验共享的默认键和值 |
| navigation.json 等环境文件 | 任务、输入、优化、监控的共同参数 |
| profiles.json | pilot/formal 的默认预算和种子，smoke 的工程缩减 |
| navigation/checked.json、no_checks.json | 导航比较组的检查开关；不复制共同参数 |
| 有限环境专题配置目录 | 已有预算或机制实验所需覆盖 |
| 环境目录下的底层 configs | 原训练器加载和旧入口兼容，不直接作为当前矩阵的实验配置 |

`--budget` 是每方法、每训练种子、每标签组的源联合环境步预算，不是 optimizer 步数。应用 direction_steps/actor_fit_steps/critic_steps 映射为 epoch；固定基点 fit_steps 表示拟合优化步数。是否等成本应看实际采样和计算账本。

smoke 只做工程验证。pilot 当前应用预算默认 2M，formal 默认 20M 和种子 100–104；profile 本身不保证收敛或足够评价精度。正式运行前必须冻结共同参数、预算与评价样本量。

## 4. 结果与恢复

| 文件 | 用途 |
|---|---|
| manifest.json | 完整配置、代码/依赖信息、协议版本、运行状态和来源 |
| summary.csv、analysis.md | 跨种子汇总和解释边界 |
| training.csv | 逐轮学习与最近源监控指标；具体回报含义按环境手册 |
| diagnostics.csv | 启用时的机制诊断，不是每次运行必有 |
| evaluation_episodes.csv | 冻结策略逐场景报告数据 |
| overview.png | 更新覆盖的总图 |
| events.jsonl | 每个训练运行一个采样、检查与恢复账本 |
| checkpoints/final.pt、best.pt | 应用训练最终状态与源监控辅助 Actor |
| checkpoints/final.json、best.json 或 .npz | 成对与两步模型使用的格式 |

主比较使用 final。训练、检查、源选模监控计入源预算；仅用于报告的最终评价及额外机制诊断单列。目标不训练 Critic，不按目标成绩选源模型。同一种子的多个场景或采样段不算多个独立训练种子。

统一矩阵入口没有自动续训开关。底层保存了恢复所需状态，但旧套件的外层清单不同，不能将其恢复命令直接套到矩阵目录。更换观测或终止协议后不能直接续训旧模型。历史结果中的 JSON 是必要溯源数据，不属于重复模板。

## 5. 可视化与工程检查

当前矩阵模型使用 `python -m manifold_project.experiments.visualize` 生成 GIF，完整导航示例见导航手册。cooperative_navigation/tools 中的 monitor/replay/stability 只面向原套件格式。

```powershell
python -m pip install -r manifold_project/experiments/requirements-test.txt
python -m pytest manifold_project/experiments/tests -q
```

测试包括数学、输入权限、环境边界、采样连续性、恢复、冻结执行及图表。测试数量以实际输出为准；通过不代表算法收敛。源码来源与归档复用记录见 [reuse_audit.csv](reuse_audit.csv)，实际训练无需解压旧 ZIP。
