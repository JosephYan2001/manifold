# 实验结果索引

2026-10-18 已完成 [A0 数据种子 43、44 的确认分析](evidence/navigation/actor_fit_e1_confirm_20261018_v1/analysis_20261018/新种子确认分析.md)：两组小学习率最佳候选均产生非零验证拟合改善，但尚未确认回报提高，seed 44 的教师有待复核的负向信号。固定教师纯评价入口已接入，[最新 CPU 工程验收](checks/teacher_evaluation_acceptance_20261018_v1/验收报告.md) 21/21 通过；正式 100 对收益复核尚未运行，需使用训练机器上的完整快照，见[当前安排](notes/README.md)。此前 [E1 开发结果](evidence/navigation/actor_fit_e1_20261017_v1/analysis_20261017/E1结果分析.md)和 17/17 工程验收作为既有证据保留。

## 按用途查找

| 目录 | 放什么 | 什么时候看 |
|---|---|---|
| [evidence/navigation](evidence/navigation/README.md) | 导航训练、D1/D2、补测、分析、GIF 和检查点 | 当前排查的主要证据 |
| [evidence/finite](evidence/finite/README.md) | 成对与两步环境的机制、学习、迁移和基线结果 | 需要核对原研究主张时 |
| [checks](checks/README.md) | 临时测试、修复验收和回归检查 | 判断代码有没有按定义工作 |
| [archive](archive/README.md) | 首批 pilot、早期 AN 等暂不参与当前主对照的结果 | 追溯历史时 |
| [notes](notes/README.md) | 问题归因草稿和下一步计划 | 看待验证假设与执行顺序 |

## 当前先看这四项

1. [最新补测原因分析](evidence/navigation/navigation_followup_20261015_v1/analysis_20261016/补测原因分析.md)：逐步拟合与 N×U 证据。
2. [第一轮 D1/D2 分析](evidence/navigation/navigation_fixed_diagnosis_20261014_v1/analysis_20261015/问题排查.md)：已有对照，避免重复实验。
3. [4M 训练统计与 GIF](evidence/navigation/navigation_diagnosis_20261012_4m_s40_v1/analysis_20261014/)：A0/A1/A2 的结果和下一轮固定基点来源。
4. [当前归因草稿](notes/manifold_当前问题归因与原稿修订决策_20261016.md)：仍待验证，不能替代原始证据。

## 路径与保存规则

2026-10-16 按用途整理目录，原运行名称保留。旧路径到新路径见 [path_map.json](path_map.json)，按最长匹配前缀查找；新命令使用新路径。历史 manifest、运行配置、CSV、日志、检查点及图片保留原内容，其中的当时路径和命令不改写。维护中的说明文档和分析脚本已经跟随新目录更新；固定 commit 的 GitHub 来源链接仍指向当时版本。

新研究运行放入 `evidence/<环境>/<新运行名>/`，工程测试放入 `checks/<新运行名>/`。每次运行使用独立目录，采样与检查成本全部保留；失败和低分运行不能按成绩选择性删除。尚未运行的计划写在 notes，不列为已完成实验。

优先读 manifest 和协议字段，再读汇总与逐场景结果。不同终止语义、观测、观察上限或回报定义不能直接混合均值。导航汇总按 protocol_version、task_mode、train_reset_horizon、evaluation_horizon 和 return_scope 区分。

同一种子的不同规模、场景或重复报告不算独立训练种子；4M 续训不算新的训练重复；工程验收和固定基点诊断不替代完整训练性能比较。正式报告与训练采样分别计费。
