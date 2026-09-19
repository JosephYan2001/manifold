# 三个实验环境

每个环境的配置、实现、训练、评价、测试和依赖保存在自己的目录中。

更新：2026-09-19。跨环境文档的职责和阅读顺序见[研究文档导航](../docs/README.md)。本页负责实现进度与直接入口；环境中的设计稿、可运行代码和已完成结果是三种不同状态。

| 目录 | 环境 | 当前状态 |
|---|---|---|
| [pair_coordination](pair_coordination/README.md) | 可枚举成对协作 | 83次开发训练及三组确认补充实验已完成；新增60次训练、640条方向/180条运输/9600条检查记录 |
| [cooperative_navigation](cooperative_navigation/README.md) | MPE2 合作导航 | 已改为持续任务，旧结果按用户要求清理；新协议尚无训练结果，正式参数未冻结 |
| [rware](rware/README.md) | RWARE 仓库 | 目录说明与跨环境任务方案已有；独立详细设计、环境封装和训练尚未完成 |

研究动机见[实验方案](../docs/策略流形研究设计.实验方案.md)，做哪些实验见[实验总览](../docs/实验执行矩阵.md)，方法、计账与统计共识见[共同执行协议](../docs/实验执行矩阵_详细协议.md)，工程验收见[构建指南](../docs/三个实验环境构建指南.md)。具体参数与命令以各环境最新协议及实际配置为准；docs/archive 中的旧稿不作为执行依据。

## 当前执行入口

成对新增 P-T 已完成：60 个 final 模型、540 条精确评价。当前结论是近约束最优部署与相对当前 PG 的目标回报优势，不是普遍迁移优越性；详细数据见[完整实验判断第 7.1 节](pair_coordination/results/完整实验判断_20260918.md)。

| 环境 | 接下来做什么 | 直接阅读 |
|---|---|---|
| 成对协作 | 论文结论稿与统一数值报告已整理，预定三组补充已完成，无需默认重跑 | [论文结论与后续验证](pair_coordination/results/论文结论与后续验证.md)、[完整实验判断](pair_coordination/results/完整实验判断_20260918.md)、[已有结果索引](pair_coordination/results/结果目录索引.md) |
| 合作导航 | 新持续协议下MAPPO/IPPO与no_checks多步回报/GAE各从头训练2000万步，再做500步连续评价；旧结果已清理 | [运行说明](cooperative_navigation/实验运行说明.md)、[当前判断](cooperative_navigation/results/当前实验判断.md)、[结果索引](cooperative_navigation/results/README.md) |
| 仓库 | 先明确独立执行设计、版本、地图和接口，再实现封装与训练 | [仓库说明](rware/README.md) |

当前可推进导航源pilot。成对开发与确认结果不支持“完整方法普遍领先”或“检查已证明具有保护作用”；相应负结果与成本取舍应在论文中如实保留。

## 可运行入口与输出

在仓库根目录运行现有环境：

```powershell
python -m manifold_project.experiments.pair_coordination.run_experiments --profile smoke --experiments all --dry-run
python -m unittest discover -s manifold_project/experiments/pair_coordination/tests -v
```

第一条只打印计划，不创建结果；需要实际 smoke 时移除 --dry-run 并按需添加 --plot。旧 run_demo.py 已不存在。导航入口为 `python manifold_project/experiments/cooperative_navigation/run_experiments.py --profile pilot --experiments N-C N-A --plot`；仓库仍无对应训练入口。

成对套件每个成功子运行保存 4 个 JSON 和 1 个 events.jsonl；整体报告在 suite 根目录。已有结果按实验内容命名，新运行可用 --output-dir 指定新目录。导航每个成功子运行保存 2 个 JSON、1 个 events.jsonl 和 best/final 两个 PT；套件目录自动编号，也可用 --output 指定新目录。仓库格式在实现时冻结。历史研究结果数值与原始记录保留，文档更新不改写旧统计。

论文通用数学核验位于 `manifold_project/tools/verification/`，独立于三个环境。测试数量会随实现变化，文档不再维护固定数字，以实际测试输出为准。
