# 协作导航结果索引

更新：2026-09-22。已按清理要求删除旧持续任务训练套件，仅保留其首达诊断及来源记录；新首达任务长训练尚未启动。

## 现有记录

[legacy_ppo_arrival_s40_h500/](legacy_ppo_arrival_s40_h500/)仅含4个文件：

| 文件 | 用途 |
|---|---|
| episodes.csv | 两个旧PPO模型各200场景的首次全覆盖评价 |
| summary.csv | 成功率、完成步数、碰撞及原模型SHA256 |
| overview.png | 首达诊断总图 |
| provenance.json | 原manifest、两组运行配置/训练汇总、文件摘要与清理记录，合并为一份来源文件 |

此目录是历史诊断，不是可恢复的训练套件，也不是新first_arrival协议的训练成绩。原模型权重已删除，不能直接续训、重新回放或重跑该历史评价。结论见[当前实验判断](当前实验判断.md)。

## 本次清理

删除learning_20m_author_mpe_s40旧套件的17个文件，共142,370,534字节（135.78 MiB），包括两组best/final检查点、两个events.jsonl、旧训练曲线、表格、监控图和套件状态文件。必要来源信息已合并保存，不再分散保留旧训练JSON。

三份首达诊断从旧套件移出，移动前后SHA256一致；删除前原final摘要与诊断记录一致。本次仅清理协作导航过时训练产物，未修改成对协作结果和新实验配置。

## 待新建实验

| 目录 | 用途 |
|---|---|
| arrival_20m_author_mpe_s40/ | first_arrival作者MAPPO/IPPO |
| arrival_20m_ours_checks_mc_s40/ | first_arrival完整ours与no_checks |
| arrival_20m_no_checks_gae_s40/ | 需要时仅改变核心方法方向标签 |
| arrival_20m_single_ablations_mc_s40/ | 后续单模块消融，继承选定ours参数 |

以上目录尚未生成，不代表已有成绩。启动和恢复命令统一见[运行说明](../实验运行说明.md)。