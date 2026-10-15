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

2026-10-15 已完成固定 checkpoint D1/D2：[问题排查及后续诊断](navigation_fixed_diagnosis_20261014_v1/analysis_20261015/问题排查.md)。D1 方向 64 epoch 的训练/验证代理分数分离，D2 的 3e-4 拟合出现震荡；较小学习率降低拟合残差。40 个配对场景的收益区间均跨零，尚未证明真实回报改善。

2026-10-15 [修复验收及 D1/D2 命令](repair_acceptance_20261015_v2/验收报告.md)：12/12 CPU 工程测试通过，覆盖方法/开关、分块整批统计、冻结快照与梯度隔离、拟合起点、KL 口径及只读监控。该结果只用于代码验收，不计入正式训练性能比较。

2026-10-02 按用户要求删除旧导航的限时版、无限延续版和跨轮分段版数据，以及对应 checkpoint、GIF、曲线和判断报告；成对/两步数据及原始源码 ZIP 保留。

当前协议为 `navigation-episodes-v1-history-v1`：8 帧历史，每个 episode 成功即结束，否则 400 步截断硬重置；截断处使用重置前末状态自举。2026-10-14 已有以下完成结果，均为训练种子 40；A0/A1 每批16回合，A2每批64回合。

| 结果目录 | 声明累计预算 | 实际源步数 A0/A1/A2 | final 独立20场景回报 A0/A1/A2 | 成功数 |
|---|---:|---|---|---|
| navigation_diagnosis_20261010_2m_s40_v1 | 2M | 1996700 / 1996452 / 1996461 | -104.273 / -105.217 / -102.629 | 各0/20 |
| navigation_diagnosis_20261012_4m_s40_v1 | 4M | 3999337 / 3998440 / 3992501 | -98.996 / -92.818 / -101.002 | 各0/20 |
| navigation_ep400_history8_2m_s40/mappo | 2M | 1997221 | MAPPO-E：-54.320 | 7/20 |

2M AN 的实验源码指纹以 `657d55d1` 开头，4M续训以 `c4671df2` 开头，旧MAPPO以 `6820ad62` 开头；完整值以各运行 manifest 为准。源步数包含监控，最终独立20场景评价另计。4M为同一模型续训，不是新的独立训练种子。最终报告使用 final.pt；best.pt 是源监控选择的诊断模型，不替代 final 报告。8场景监控与20场景最终报告不得混用。

最新[数据、结论与诊断命令](../cooperative_navigation/续训结果与诊断执行清单.md)、[统计与GIF](navigation_diagnosis_20261012_4m_s40_v1/analysis_20261014/)、[整改验收记录](../cooperative_navigation/方向问题整改记录.md)。历史 `an_core_memfix` 保留在 `navigation_ep400_history8_2m_s40`，不与后续A0重复计算训练重复。尚未运行的计划目录不列为完成结果。

## 数据使用规则

优先读取运行 manifest 和协议字段，再读 summary 与逐场景结果。不同任务终止语义、观测、采样重置、观察上限或回报定义不能直接混合均值。新版导航汇总按 protocol_version、task_mode、train_reset_horizon、evaluation_horizon 和 return_scope 分组。

通常保留当前协议的原始结果；本次旧导航数据清理是用户明确指定的协议更替清理。新实验的失败运行、未完成场景和检查成本仍须记录，不能按成绩选择性删除；最终报告与训练采样分别计费。
