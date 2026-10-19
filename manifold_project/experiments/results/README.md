# 实验结果索引

**当前唯一执行安排：[导航六组实验方案](notes/导航采样量与学习率六组实验方案_20261019.md)。** N16/N32/N64 × 固定/线性衰减 Actor 学习率，同一初始化 seed 40，每组 2M。[配置与命令](../configs/navigation/batch_lr_2m/README.md)已就绪，[28 项 CPU 工程验收](checks/navigation_batch_lr_acceptance_20261019_v1/验收报告.md)通过；正式训练未启动，不把计划列为已完成结果。

最新已完成证据是 [N64 四格短训练分析](evidence/navigation/actor_fit_short_n64_u128_500k_s40_20261018_v1/analysis_20261019/四格短训练分析.md)：各 16 轮，A/B/C/D 最终回报 -179.04/-186.05/-176.15/-185.72，均 0/100 成功。小学习率两组在本轮较差，C 的优势未确认；最佳候选生效但整体收益尚未确认。此前固定教师复核已按预算结束，不再重复设为连续训练门槛。

2M 性能参照见[历史 MAPPO 复算](evidence/navigation/navigation_ep400_history8_2m_s40/mappo/analysis_20261019/MAPPO_2M参照分析.md)：final.pt 为 −54.32、7/20 成功，实际源步数 1,997,221。六组方案已纳入一次同场景冻结 MAPPO 评价，新增总预算上限 12.7M；不新增 MAPPO 训练，复评尚未执行。历史结果与新 AN 的版本、评价场景差异须明确保留。

## 按用途查找

| 目录 | 放什么 | 什么时候看 |
|---|---|---|
| [evidence/navigation](evidence/navigation/README.md) | 导航训练、D1/D2、补测、分析、GIF 和检查点 | 当前排查的主要证据 |
| [evidence/finite](evidence/finite/README.md) | 成对与两步环境的机制、学习、迁移和基线结果 | 需要核对原研究主张时 |
| [checks](checks/README.md) | 临时测试、修复验收和回归检查 | 判断代码有没有按定义工作 |
| [archive](archive/README.md) | 首批 pilot、早期 AN 等暂不参与当前主对照的结果 | 追溯历史时 |
| [notes](notes/README.md) | 问题归因草稿和下一步计划 | 看待验证假设与执行顺序 |

## 当前先看这四项

1. [六组完整方案](notes/导航采样量与学习率六组实验方案_20261019.md)：唯一当前计划，包含变量、预算、评价、日程定义与启动前验收。
2. [最新短训练结果](evidence/navigation/actor_fit_short_n64_u128_500k_s40_20261018_v1/analysis_20261019/四格短训练分析.md)：本轮计划的直接依据。
3. [导航证据索引](evidence/navigation/README.md)：固定目标、教师复核、历史 2M/4M 和 GIF，按阶段查找。
4. [整理记录](notes/maintenance_20261019/整理记录.md)：历史建议归档、19 张重复概览图清理及原始证据哈希核验。

## 路径与保存规则

2026-10-16 按用途整理目录，原运行名称保留。旧路径到新路径见 [path_map.json](path_map.json)，按最长匹配前缀查找；新命令使用新路径。历史 manifest、运行配置、CSV、日志、检查点和 GIF 保留原内容，其中的当时路径和命令不改写。2026-10-19 仅清理曲线区域逐像素相同、顶部目录标题不同的父级概览图，保留位置与旧图哈希见[清理清单](notes/maintenance_20261019/cleanup_manifest.json)。历史结论的数值部分保留，过期执行建议移到归档；固定 commit 的 GitHub 来源链接仍指向当时版本。

新研究运行放入 `evidence/<环境>/<新运行名>/`，工程测试放入 `checks/<新运行名>/`。每次运行使用独立目录，采样与检查成本全部保留；失败和低分运行不能按成绩选择性删除。尚未运行的计划写在 notes，不列为已完成实验。

优先读 manifest 和协议字段，再读汇总与逐场景结果。不同终止语义、观测、观察上限或回报定义不能直接混合均值。导航汇总按 protocol_version、task_mode、train_reset_horizon、evaluation_horizon 和 return_scope 区分。

同一种子的不同规模、场景或重复报告不算独立训练种子；4M 续训不算新的训练重复；工程验收和固定基点诊断不替代完整训练性能比较。正式报告与训练采样分别计费。
