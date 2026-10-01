# 实验结果索引

按研究问题和协议版本读取结果。同一种子的多个目标规模、场景或重复报告不算独立训练种子；原始 manifest、配置、CSV、日志和检查点保持原始含义。

## 成对与两步环境

当前主要结论见[262k 四组预算与检查消融报告](finite_budget_study/四组预算与检查消融报告.md)及[冻结跨规模迁移分析](finite_budget_study/冻结跨规模迁移分析.md)。AN–SA 研究估计方式的有限样本差异，方向方法–DA 研究候选质量与连续学习价值，二者分开解释。

完整 DA 保留回报检查，源与目标表现良好；P-L 两个源失败种子只属于关闭回报检查的 DA。当前充分学好的方法都能迁移到共享最优表策略的目标，不能将源训练失败直接解释为独立迁移能力差距。

| 目录 | 用途与入口 |
|---|---|
| pair_estimation_20seeds_v1/ | P-E 方向估计，20 个数据种子；P-EA 的直接输入 |
| mechanisms/pair_ea_mc2048_20seeds_v1/ | [方向到执行联合分析](mechanisms/pair_ea_mc2048_20seeds_v1/方向到执行联合分析.md) |
| mechanisms/pair_ea_mc32_128_512_20seeds_v1/ | 小样本 P-EA 补充，与 2048 档联合解释 |
| pair_kl_mc2048_20seeds_v1/ | 同源 KL 的全部候选比较；不等于全程学习效率比较 |
| mechanisms/pair_fixed_gate_kl3e4_20seeds_v1/ | 固定候选的独立重复检查 |
| finite_budget_study/ | 262k 完整版、两个单检查消融和无检查主结果 |
| mechanisms/pair_checks/ | [检查敏感性分析](mechanisms/pair_checks/pair_transfer_check512_10seeds_v1/检查敏感性实验分析.md) |
| baselines/finite_learning_mc65k_checks512_10seeds_v1/ | [65k 阶段基线](baselines/finite_learning_mc65k_checks512_10seeds_v1/阶段综合报告.md) |
| baselines/pair_realization_transfer_10seeds_v1/ | [P-A/P-T 多种子基线](baselines/pair_realization_transfer_10seeds_v1/多种子复验分析.md) |
| archive/ | 首批 pilot 等历史探索，不与扩种子结果重复计数 |

P-E 与 KL 源目录保留原位，因为其他实验直接引用。其余已按表格分组，历史 manifest 中的当时命令和路径不改写。当前执行配置见[有限预算四组说明](../configs/finite_budget_study/README.md)。

## 协作导航

| 数据 | 协议与使用边界 |
|---|---|
| navigation_trial/ | 旧 200 步硬期限、含时间输入的四组初训；[初始分析](navigation_trial/初始训练分析.md)只对该协议有效 |
| navigation_success_v2_2m_s40/ | AN 已完成、MAPPO 记录到第 71 轮；无限延续导致状态漂移，见[异常诊断](navigation_success_v2_2m_s40/异常诊断.md)。保留作采样问题证据，不据此排名算法 |
| navigation_success_v3_2m_s40/ | 修正后预定输出；成功终止、200 步分段、1000 步外部采样重置及末状态自举、无时间输入。待实际运行 |

旧四组最终均为 0/20 次成功；MAPPO-E 回报改善，无检查 AN 改善较慢，检查版实际更新少。这些观察不能直接判断新版性能。新版数据应单独分析，源任务尚未学好时的目标低分不能直接归因于迁移失败。

任务定义、配置和四组命令统一见[导航实验手册](../cooperative_navigation/README.md)。本次 v2 记录证实未成功场景不重置会使训练长期偏离初始状态分布；v3 只修正采样流程，不能预先声称性能问题全部解决。活动配置已经精简，历史数据内的 JSON 仍用于重现当时设置，不能作为当前启动模板。

## 数据使用规则

优先读取运行 manifest 和协议字段，再读 summary 与逐场景结果。不同任务终止语义、观测、采样重置、观察上限或回报定义不能直接混合均值。新版导航汇总按 protocol_version、task_mode、train_reset_horizon、evaluation_horizon 和 return_scope 分组。

清理活动模板不删除原始结果。失败运行、未完成场景和检查成本都属于实验记录；最终报告与训练使用的采样分别计费。
