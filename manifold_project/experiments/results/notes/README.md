# 当前计划与历史判断入口

**[导航六组实验](导航采样量与学习率六组实验方案_20261019.md)已完成，最新判断见[六组与旧基线对比](../evidence/navigation/navigation_batch_lr_2m_s40_20261019_v1/analysis_20261022/六组训练与旧基线对比.md)第 6 节。** 六组内部只改变 N 与日程；相对旧基线还改变了方向 epoch 与候选选择。归因审查后撤回默认追加两组 500k 的建议；下一项只考虑固定数据的 Critic 数值机制验证，尚未实施，不扩展训练矩阵。

[Critic 价值尺度诊断命令](../../cooperative_navigation/Critic价值尺度诊断执行说明.md)已接入原 `diagnose` 入口：同一批数据上的四条 Critic 拟合轨迹，最多采集 25,600 步，不训练 Actor/方向。14 项相关 CPU 检查通过，正式数据尚未生成。

状态更新于 2026-10-22：六组源训练与最终评价实际合计 **12,580,529** 步，均 0/100 成功。日程和候选选择按定义执行，但没有恢复旧基线水平。计划中的 MAPPO 同 100 场景冻结评价尚未提供结果；历史[28 项 CPU 验收](../checks/navigation_batch_lr_acceptance_20261019_v1/验收报告.md)与[六组命令](../../configs/navigation/batch_lr_2m/README.md)保留用于复现。Critic 输出近似常数是明确的批次观测，不等于已定位唯一根因。

## 已完成的证据链

| 阶段 | 当前结论与边界 |
|---|---|
| [历史 MAPPO 2M 参照复算](../evidence/navigation/navigation_ep400_history8_2m_s40/mappo/analysis_20261019/MAPPO_2M参照分析.md) | final.pt 为 −54.32、7/20 成功，约 0.46M 已有监控进展；提供预算与性能参照，不能冒充同版本算法对照 |
| [Actor 拟合工程验收](../checks/actor_fit_acceptance_20261017_v1/验收报告.md) | 最佳候选、零更新、固定目标和方向预算控制已验收；不是任务性能证据 |
| [E1 固定目标](../evidence/navigation/actor_fit_e1_20261017_v1/analysis_20261017/E1结果分析.md) | 部分固定目标的拟合改善明确，任务收益未确认 |
| [数据种子 43/44 确认](../evidence/navigation/actor_fit_e1_confirm_20261018_v1/analysis_20261018/新种子确认分析.md) | 小学习率的非零验证改善复现；不能当作独立训练种子 |
| [固定教师复核](../evidence/navigation/teacher_return_review_20261018_v1/analysis_20261018/固定教师复核分析.md) | 两个教师的新评价区间均跨零；一次封顶复核已结束 |
| [N64 四格短训练](../evidence/navigation/actor_fit_short_n64_u128_500k_s40_20261018_v1/analysis_20261019/四格短训练分析.md) | 固定 3e-5 在本轮较差；选择器保留进展但整体收益未确认；每组仅 16 轮 |

小学习率在已训练基点上有拟合收益，不代表从头全程应使用 3e-5。N64 在一个固定基点上改善过方向验证分数，不代表相同交互预算下的连续学习最优。这两点正是下一轮六组对照要检验的假设。

## 历史草稿与清理

- [2026-10-16 归因草稿](../archive/notes/manifold_当前问题归因与原稿修订决策_20261016.md)：按当时证据保留，不作为当前待办。
- [历史执行建议](../archive/notes/历史执行建议_20261019.md)：从旧报告移出的 D1/D2、E1、教师复核和条件性扩展建议。
- [2026-10-19 整理记录](maintenance_20261019/整理记录.md)：派生重复图的保留位置和原始文件核验。

新研究运行放入 `results/evidence/navigation/<新运行名>/`，工程验收放入 `results/checks/<新验收名>/`。结果未生成前，计划不列入 evidence；失败、零更新和负收益结果不按成绩删除。
