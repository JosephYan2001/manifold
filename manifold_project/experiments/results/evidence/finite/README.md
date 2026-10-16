# 成对与两步环境

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
| [../../archive/](../../archive/README.md) | 首批 pilot 等历史探索，不与扩种子结果重复计数 |

P-E 与 KL 源目录已随本组证据迁入 evidence/finite，维护中的命令和文档使用新路径。历史 manifest 中的当时命令和路径不改写，对应关系见[迁移表](../../path_map.json)。当前执行配置见[有限预算四组说明](../../../configs/finite_budget_study/README.md)。

