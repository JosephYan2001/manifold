# P-L source_learning

状态：执行完成。协议：direction-v1；profile：formal。

记录 210 行，独立种子/批次标识数 10。详见 diagnostics.csv、各运行 training.csv 或 evaluation_episodes.csv。

summary.csv 为长表：每行是一项指标，mean/std 是独立训练种子或数据批次间的统计；independent_units 是实际重复数。单次精确计算不伪造置信区间。

A 类覆盖结论仅适用于可精确核对完整输入支持与方向偏移的有限模型；实体列表跨人数及应用结果按经验外推解释。

自动汇总不判定算法优越或训练收敛；应同时检查任务指标、源学习、拟合误差与全部交互/计算成本。
