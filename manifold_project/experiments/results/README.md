# 实验结果索引

更新：2026-09-25。按证据职责阅读，不把同种子的重复实验或目标规模算成独立样本。原始 manifest、CSV、图片均保留。

最新结论入口：[262k四组预算与检查消融报告](finite_budget_study/四组预算与检查消融报告.md)。此前[65k阶段报告](baselines/finite_learning_mc65k_checks512_10seeds_v1/阶段综合报告.md)保留为预算基线。AN–SA用于估计方式的有限预算机制比较；AN/SA–DA用于方向路线的候选质量与完整训练比较，二者不混为同一问题。

最新迁移判断：[冻结跨规模迁移分析](finite_budget_study/冻结跨规模迁移分析.md)。源与当前六个目标共享最优表策略，充分学好的方法均迁移成功；无检查DA的总体目标差距主要来自两个源失败种子，不能当作独立迁移能力差距。

**DA口径：完整DA保留回报检查，源与目标均表现良好；两个失败种子只属于关闭回报检查的DA。方向方法尚未证明优于完整DA。后续保留这一强基线，先核对协作导航协议，再组织共同预算试跑。**

共同预算/检查消融四组已全部完成，配置和命令见 [finite_budget_study](../configs/finite_budget_study/README.md)。主结果集中于 `finite_budget_study/`。

阅读顺序：**当前主结果**看finite_budget_study；**机制证据**看mechanisms及两个保留原位的源数据目录；**对照基线**看baselines；**历史探索**看archive。P-E与KL候选被其他实验直接引用，保留原路径；其余数据已实际分组移动。

| 目录 | 用途 | 当前判断/入口 |
|---|---|---|
| `pair_estimation_20seeds_v1/` | P-E 方向估计，20 数据种子；P-EA 的直接输入 | 固定方向方差改善稳定；最终方向误差优势尚未确立 |
| `baselines/pair_realization_transfer_10seeds_v1/` | 原 P-A、P-T 多种子基线 | [多种子复验分析](baselines/pair_realization_transfer_10seeds_v1/多种子复验分析.md) |
| `mechanisms/pair_ea_mc2048_20seeds_v1/` | P-E→Actor 完整证据链，MC/2048 | [联合分析](mechanisms/pair_ea_mc2048_20seeds_v1/方向到执行联合分析.md) |
| `mechanisms/pair_ea_mc32_128_512_20seeds_v1/` | 小样本P-E→Actor补充，已完成 | 与2048档联合解释，机制扫描收束 |
| `pair_kl_mc2048_20seeds_v1/` | 同源KL的所有候选比较 | AN/SA局部候选增益优于当前DA；不等于全程效率优势 |
| `mechanisms/pair_fixed_gate_kl3e4_20seeds_v1/` | 固定候选独立重复检查 | 方向检查有保护作用，回报平局放行存在问题 |
| `baselines/finite_learning_mc65k_checks512_10seeds_v1/` | 65k连续训练及冻结迁移，预算基线 | 此预算下DA最终回报更高，方向方法末段仍上升 |
| `finite_budget_study/` | 当前主结果：262k完整、单检查消融、无检查 | 无检查方向方法在两任务均接近最优；效率与稳定性需分开解释 |
| `mechanisms/pair_checks/` | 四组检查敏感性对照，集中管理 | [联合检查分析](mechanisms/pair_checks/pair_transfer_check512_10seeds_v1/检查敏感性实验分析.md) |
| `archive/` | 首批 pilot，历史探索记录 | 不与扩种子后的相同记录重复计数 |

## 本次整理

- `pair_mechanisms_pilot_v1` 移至 `archive/pair_mechanisms_pilot_v1`。
- 两个P-EA运行、固定候选检查、pair_checks整体移入mechanisms；原65k和P-A/P-T多种子运行移入baselines。上述表格给出准确新路径，运行文件夹原名保留。
- P-E源目录与KL候选源目录保持原路径。历史manifest中的原始运行命令/路径保持原样，不把历史记录改写成新命令。移动后的358个非Markdown文件已逐一核对SHA256保持不变。
- 已修复Markdown相对链接及说明中的完整运行路径。旧聊天中的被移动路径不会自动更新，以此索引为准。
- 没有删除原始实验数据。部分种子虽重复，整体运行仍是可复现来源，不做跨运行物理去重。

小样本P-EA与四组预算消融均已完成。下一阶段以冻结规模评价和合适的后续环境验证为主，具体见最新四组报告。KL与检查诊断使用finite.probes入口，训练使用统一矩阵入口。
