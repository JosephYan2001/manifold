# 实验运行说明

统一入口为 `python -m manifold_project.experiments`。成对学习复用 pair_coordination 训练器，导航、仓库和实体成对任务共用 cooperative_navigation 的 Runner；有限机制诊断由 finite 组织。完整研究安排见[总体实验方案](../docs/总体实验方案.md)，代码职责见[项目结构与代码说明](../docs/项目结构与代码说明.md)。

## 环境模块说明

各环境的模块、观测空间、模型结构和算法数据流分别维护如下；文档路径遵循实际实现位置，两步环境在 `finite/`，仓库适配器在 `applications/envs/`。

| 环境 | 模块文档 | 覆盖范围 |
|---|---|---|
| 成对协作 P | [模块组成说明](pair_coordination/模块组成说明.md) | 自身类型协议、P-L 类型表训练、固定基点诊断及 P-TB 实体分支 |
| 两步协作 T | [两步环境模块组成说明](finite/两步环境模块组成说明.md) | 公开状态观测、三状态模型、标签和时间权重、T-V/T-L |
| 协作导航 N | [模块组成说明](cooperative_navigation/模块组成说明.md) | 全名单与 8 帧历史、空间/时间网络、episode 采样与截断自举 |
| 仓库 W | [仓库环境模块组成说明](applications/envs/仓库环境模块组成说明.md) | 九格观测、单帧实体网络、有限任务终止及共用 Runner 接入 |

## 1. 安装与入口

在包含 manifold_project 的目录运行，本机为 D:\manifold。使用已有 PyTorch/CUDA 环境，或激活项目环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r manifold_project/experiments/requirements.txt
python -m manifold_project.experiments --list
```

MPE2 固定 1.1.1，RWARE 固定 2.0.0。入口在计算库导入前设置 CUDA 确定性所需环境变量，尊重调用者已设置的值；运行清单记录实际运行环境。此前使用的 OpenMP 重复库兼容开关可通过 `--allow-duplicate-openmp` 显式开启，不表示依赖冲突已经根治。

## 2. 去哪里找实验命令

导航训练的中断恢复及累计预算扩展统一使用 `--resume 原结果目录`。省略预算时继承原配置、原地恢复；增加 `--budget` 时必须指定独立的新 `--output`。详见[导航手册：恢复与延长训练](cooperative_navigation/README.md#8-中断恢复与增加预算续训)。

| 任务 | 执行文档 | 当前用途 |
|---|---|---|
| 协作导航 | [导航实验手册](cooperative_navigation/README.md) | 新协议单种子源训练、四方法比较、冻结评价和 GIF |
| 成对机制与连续学习 | [后续对照执行说明](finite/后续对照执行说明.md)、[有限预算四组](configs/finite_budget_study/README.md) | KL 对照、检查机制、P-L/T-L 和预算消融 |
| 全部实验编号与预期结果 | [总体实验方案](../docs/总体实验方案.md) | P/T/N/W 的研究问题、方法与指标 |
| 已有证据 | [结果索引](results/README.md) | 已完成结果、历史边界和分析入口 |

导航的唯一完整运行说明在导航 README。当前先运行 AN 无检查与 MAPPO-E，再比较 AN 双检查和保留回报检查的 DA。共同使用 8 帧历史，每个 episode 成功即结束，否则 400 步硬重置；收齐 16 个 episode 后更新。超时保留末状态自举，独立评估观察上限为 1000 步。旧导航数据已清理，当前协议需要从头训练。

其他入口示例：

```powershell
# 固定基点机制诊断；只解析时追加 --dry-run
python -m manifold_project.experiments --experiments P-E P-A P-T --profile pilot --output manifold_project/experiments/results/evidence/finite/pair_mechanisms_new --plot

# P-EA 直接使用已有 P-E 的方向，不重新学习
python -m manifold_project.experiments --experiments P-EA --profile formal --source manifold_project/experiments/results/evidence/finite/pair_estimation_20seeds_v1 --config manifold_project/experiments/configs/pair_ea_mc.json --sample-sizes 2048 --output manifold_project/experiments/results/evidence/finite/pair_ea_new --plot

# 仓库源训练与冻结评价
python -m manifold_project.experiments --experiments W-C --profile pilot --device cuda --output manifold_project/experiments/results/evidence/warehouse/warehouse_source_new --plot
python -m manifold_project.experiments --experiments W-T W-L --profile pilot --source manifold_project/experiments/results/evidence/warehouse/warehouse_source_new --device cuda --output manifold_project/experiments/results/evidence/warehouse/warehouse_transfer_new --plot
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

## 5. 可视化与配置检查

当前矩阵模型使用 `python -m manifold_project.experiments.visualize` 生成 GIF，完整导航示例见导航手册。训练过程由控制台、training.csv 和 overview.png 监控。原导航套件入口、分析脚本及 monitor/replay/stability 工具已删除，避免与当前入口混用。

```powershell
python -m manifold_project.experiments --list
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config manifold_project/experiments/configs/navigation/no_checks.json --device cuda --dry-run
```

工程测试源码和测试专用依赖已按用户要求清理，统一安装清单为 experiments/requirements.txt。dry-run 只核对配置，不证明收敛。理论数学核验工具保留在 manifold_project/tools/verification/。源码来源与归档复用记录见 [reuse_audit.csv](reuse_audit.csv)，实际训练无需解压旧 ZIP。
