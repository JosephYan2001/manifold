# 临时测试与工程验收

这里存放检查代码是否按定义工作的结果。回报高低不决定验收是否通过，这些运行也不计入算法性能比较。

| 目录 | 状态与用途 |
|---|---|
| [navigation_batch_lr_acceptance_20261019_v1](navigation_batch_lr_acceptance_20261019_v1/验收报告.md) | 六组配置、Actor 源交互学习率日程、恢复和冻结 MAPPO-E 评价，加既有回归共 28/28 CPU 测试通过；当前最新工程验收 |
| [teacher_evaluation_acceptance_20261018_v1](teacher_evaluation_acceptance_20261018_v1/验收报告.md) | 固定教师零训练评价、配对统计、冻结与成本记录，加既有回归共 21/21 CPU 测试通过；历史验收记录 |
| [actor_fit_acceptance_20261017_v1](actor_fit_acceptance_20261017_v1/验收报告.md) | Actor 候选选择、固定目标重放及既有回归共 17/17 CPU 测试通过；历史验收记录 |
| [repair_acceptance_20261015_v2](repair_acceptance_20261015_v2/验收报告.md) | 10 月 15 日修复验收记录，12/12 CPU 测试通过；覆盖冻结、统计、KL、方法开关和只读监控 |
| [repair_acceptance_20261015_v1](repair_acceptance_20261015_v1/) | 首次验收记录；v2 收紧了实际路径的 KL 数值容差，读取结论时优先看 v2 |

后续 smoke test、dry-run 输出、回归验收使用本目录下独立的新名称。只读 dry-run 本身不产生结果时，不创建空运行目录。用于研究判断的固定 checkpoint 诊断放在 [evidence/navigation](../evidence/navigation/README.md)，不能因预算小就视作临时测试。

测试失败也保留日志，重跑使用新目录；不要用新结果覆盖失败过程。
