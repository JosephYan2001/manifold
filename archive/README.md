# 旧实验归档

归档日期：2026-09-23。当前研究入口是[项目说明](../manifold_project/README.md)。本目录仅用于追溯，归档中的“当前”“下一步”、实验编号和命令均无现行效力。

[legacy_experiments_20260923.zip](legacy_experiments_20260923.zip) 保存本轮清理前的实际文件，包含未提交的文档改动。压缩包内保持相对于仓库根目录的路径。

| 原位置 | 内容 | 文件数 |
|---|---|---:|
| `manifold_project/experiments/` | 成对与导航旧实现、训练与评价入口、监控/回放/续训工具、旧 JSON 配置、依赖说明、配套测试、MAPPO 第三方代码与许可证、RWARE 占位包，以及两组机制数据 | 168 |
| `manifold_project/docs/archive/` | 前期实验方案、论文与实验对应表、总体架构 | 3 |
| 合计 | 从工作目录移出的历史文件 | 171 |

原文件共 4,153,712 字节；压缩包为 1,471,051 字节。包内额外包含 `ARCHIVE_MANIFEST.json`，记录每个原文件的路径、大小与 SHA256。归档后逐文件比较内容哈希，并校验 ZIP 完整性。

压缩包 SHA256：

```text
3872988f36a6a94a9dc4527ccd4b0ec894be217e0fa01b3f733709824efe09c6
```

两组历史成对机制数据位于包内原 `pair_coordination/results/suite_10_论文方向估计与运输_20数据种子/` 和 `suite_11_论文检查规则_200数据种子/`，共 13 文件，原始内容未修改。它们不是当前新主张的正式性能证据。

问题审查引用的旧文件也在包内：`experiments/cooperative_navigation/实现说明.md`、`experiments/pair_coordination/实现说明.md`，以及 `docs/archive/20260923_前期方案/` 下三份计划；这些路径均以 `manifold_project/` 为前缀。

需要复核历史内容时，将压缩包解压到独立审查目录，不直接覆盖当前修改。归档没有纳入 `.git/`、编辑器设置、当前理论稿或参考论文。

2026-09-24 按用户要求，将旧成对和导航源码直接恢复为活动实验主体，再修改当前协议不一致处。历史结果和计划仍只留在本包，原 ZIP 内容及哈希未变。活动文件与原归档的逐文件核对见 [reuse_audit.csv](../manifold_project/experiments/reuse_audit.csv)，运行说明见[实验模块](../manifold_project/experiments/README.md)。

[rejected_framework_20260924.zip](rejected_framework_20260924.zip) 保存本次恢复前的重写框架快照，便于追溯；该框架已停用，不是第二套运行入口，也不作为结果来源。

随后按用户要求清理活动目录：移除旧成对套件、绑定历史结果的脚本和过时协议测试，保留关键旧函数、训练主体与必要检查；导航辅助工具归入环境 `tools/`。`reuse_audit.csv` 现在同时记录原路径、活动路径和移除状态，不能将初次恢复数量理解成当前仍全部保留。
