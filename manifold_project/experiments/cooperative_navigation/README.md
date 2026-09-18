# 合作导航实验

更新：2026-09-18。已实现 MPE2 1.1.1 环境封装、七种训练条件、独立源评价、冻结人数迁移、可选候选审计与标签对齐。参数仍为开发候选，smoke 通过不代表已有论文结果。

首批 nav_01_pilot 已完成21次源训练（7条件×3种子、每次20万团队步）。[首批实验判断](results/pilot_01_实验判断.md)已核对日志与汇总：ours优于sampled的开发趋势明确，但仍落后于MAPPO/IPPO，覆盖率较低；尚无N-T/N-D结果。下一步是共同50万步预算开发，不直接冻结formal。本批源码指纹与当前工作区不同，补跑需核对原训练代码和依赖。

| 阅读需求 | 文档 |
|---|---|
| 实验问题、比较组与结论边界 | [实验设计](实验设计.md) |
| 安装、命令、训练组织与结果文件 | [实验运行说明](实验运行说明.md) |
| 代码结构、张量、算法与恢复 | [模块说明](模块说明.md) |
| CSV 解读、预算与调参 | [训练指标与调参说明](训练指标与调参说明.md) |
| 本地验证范围与限制 | [实现验收记录](实现验收记录.md) |

任务配置：源环境 4 个智能体、4 个地标；固定观察最近的 2 个其他智能体和 2 个地标，并使用 8 步本地历史；5 个离散动作；100 个联合步；团队奖励取原生个体奖励均值。冻结后评价规模 3、4、6、8，其中 4 为源规模参照。

实测原生观测 16 维、Actor 输入 177 维、源集中 critic 输入 65 维。集中 state 是各机器人原生观测的拼接，不称为无限制全物理状态。

在仓库根目录、已安装 [requirements.txt](requirements.txt) 的 Python 环境中运行：

```powershell
python manifold_project/experiments/cooperative_navigation/run_experiments.py --profile smoke --experiments N-C N-A --plot
python manifold_project/experiments/cooperative_navigation/run_experiments.py --profile pilot --experiments N-C N-A --dry-run
python manifold_project/experiments/cooperative_navigation/run_experiments.py --profile pilot --experiments N-C N-A --plot
```

N-C 与 N-A 共用参考条件，合计七条件，不重复训练。第二条只打印计划，第三条实际启动 21 次 pilot。默认先做源实验，不用目标规模结果选超参数。

当前下一步：3 种子 pilot → 冻结参数 → 70 次正式训练 → 四主方法冻结迁移。200 万源步仍是正式预算候选。N=4 复用 final 源评价，N=3/6/8 另行评价；正式源节点每模型 100 回合，final/每目标规模 200 回合，评价成本单列。

结果自动命名为 `results/nav_XX_smoke`、`nav_XX_pilot` 等。每个正常训练只保留两个 JSON、一个统一 JSONL、best/final 两个 PT。图片集中为套件级 `overview.png` 和可选 `transfer_overview.png`，不按每个种子分散保存。

本目录详细设计维护导航专属参数；[共同执行协议](../../docs/实验执行矩阵_详细协议.md)维护方法定义、计账与统计；[文档导航](../../docs/README.md)维护进度和职责。旧稿的“三方法”“每节点 50 回合”等建议已由当前设计替代，不同时执行。首版使用 empirical 检查，不继承 Hoeffding 置信保证。
