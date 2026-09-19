# 协作导航结果索引

更新：2026-09-19。当前研究结论只在[当前实验判断](当前实验判断.md)维护；新实验命令只在[实验运行说明](../实验运行说明.md)维护。200万步的PPO尚未确认收敛，下一轮采用2000万步长预算及更充分的评价，不能沿用早期终点作为最终能力排名。

## 当前目录

| 位置 | 内容 | 用途 |
|---|---|---|
| [reference/nav_03_pilot](reference/nav_03_pilot/) | 完整ours，MC，1000万步，seed40 | 现有完整方法参考 |
| [reference/nav_04_pilot](reference/nav_04_pilot/) | no_checks，MC，1000万步，seed40 | 检查模块对照、方向诊断父模型 |
| [diagnostics/nav_01_continue_reuse](diagnostics/nav_01_continue_reuse/) | 同一nav_04终态，方向epochs 16/4，各续训100万步 | 已完成的平台诊断，不作为新主算法结果 |
| [archive/nav_01_pilot](archive/nav_01_pilot/) | 7条件×3种子，每次20万步 | 早期开发记录，退出当前实验待办 |
| [archive/nav_02_pilot](archive/nav_02_pilot/) | 7条件×1种子，每次200万步 | 学习趋势参照，不能作为已收敛的最终排名 |

历史判断随对应数据归档：[首批20万步](archive/pilot_01_实验判断.md)、[第二批200万步](archive/pilot_02_预算与学习率判断.md)、[完整方法1000万步](reference/nav_03_pilot_实验判断.md)。不删除负结果，不把归档解释为实验无效。

## 新一轮结果

以下目录由训练命令实际启动时创建，不预建空结果：

- `learning_20m_mc_s40/`：MAPPO、IPPO、no_checks MC，每组2000万步。
- `learning_20m_gae_s40/`：no_checks GAE，2000万步。

两份配置位于 `configs/learning_20m/`；每组seed40、CUDA、41评价节点、每节点1024回合。两份配置只差方向标签，PPO保持自身GAE、优势标准化及熵项。正式跨种子确认尚未开始。

## 搬移记录与文件保留

原根目录的nav_01/nav_02移入archive，nav_03/nav_04移入reference，nav_01_continue_reuse移入diagnostics；内部套件目录名和数据内容保持不变。原 `nav_04_pilot_双检查消融判断.md` 更名为 `当前实验判断.md`，配图移至reference。可用上表从旧套件名定位新路径。

保留manifest/config/summary/status、三张CSV、原始events、best/final模型及现有总览/监控图。它们分别支持参数追溯、汇总读取、机制分析和模型复查，不能仅因文件数量多而删掉。旧绝对路径作为运行时的来源记录原样保留，不修改历史manifest或checkpoint；本地默认读取入口已改为新目录。已完成套件无需恢复训练；若需复现，应使用记录的软件版本和实际新路径，而非改写旧指纹。

本次盘点没有发现失败记录或临时结果文件。五个套件共193个原始文件，整理时按相对路径和文件SHA256校验搬移前后内容。只减少当前根目录的杂项，不重复生成数据副本。
