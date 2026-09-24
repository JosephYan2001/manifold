# 单源 CTDE 的方向函数学习

在规定的本地信息和一个源规模下，利用团队经验学习动作概率怎样调整，再让共享 Actor 实现调整，分析规模变化后何时继续有效。

**2026-09-24：实验模块改为直接复用旧版主体。** 保留解析 Fisher 估计、方向到 Actor 的实现、条件跨规模运输三个贡献候选。恢复成对与导航原目录和训练器，仅修改当前协议需要的观测、模型接口、采样边界与预算，并补充旧包没有的两步模型和仓库适配。正式预算训练及性能结论尚待开展。

## 阅读入口

| 文档 | 职责 |
|---|---|
| [CTDE 方向学习主稿（从这里开始）](docs/策略流形研究设计.CTDE本地策略改进版.md) | 当前理论主稿及四组研究问题；附录 C 含修改对照和任务验收 |
| [总体实验方案](docs/总体实验方案.md) | 环境、实验编号、算法、指标、预算、结果保存及旧代码复用；开发和运行入口 |
| [项目结构与代码说明](docs/项目结构与代码说明.md) | 各层级职责、对应文件/函数、调用流程、配置关系及修改位置；阅读代码从这里开始 |
| [实验模块与运行命令](experiments/README.md) | 当前代码、归档迁用来源、配置、训练/冻结评价/GIF命令及验证范围 |
| [研究主张与开发路线](docs/研究主张与开发路线.md) | 工作进度、阶段交付与完成条件 |
| [问题审查](docs/理论与实验对应问题审查.md) | 信息压缩、源可识别性、方向变化及实际 Actor 等缺口 |
| [相关论文与主张对照](../references/README.md) | 本地论文、原始来源与研究定位 |

## 当前目录

```text
manifold_project/
  README.md
  docs/                 理论、研究路线与问题记录
  experiments/          原成对/导航训练主体、当前机制诊断及任务适配
  tools/
    verification/       独立数学核验，不依赖旧实验代码
    extract_math_check.py
references/             论文 PDF 与文献索引
archive/                原始旧实验归档、停用重写框架快照及说明
```

目录内各层级、对应代码和调用关系统一见[项目结构与代码说明](docs/项目结构与代码说明.md)。日常入口是 `python -m manifold_project.experiments`；调当前实验参数先看 `experiments/configs/`，查训练算法看两个环境目录的 `training/`，做工程检查运行 `python -m pytest manifold_project/experiments/tests -q`。

2026-09-23 的旧实现、两组机制数据和历史说明仍完整保存在[归档说明](../archive/README.md)所列压缩包。当前直接复用旧训练主体，并已清理重复实现、旧成对套件和历史脚本。旧结果没有恢复为当前证据。逐文件保留、移动与移除记录见 [reuse_audit.csv](experiments/reuse_audit.csv)。

## 可运行的数学核验

以下命令在仓库根目录 `D:\manifold` 运行，不训练模型，也不生成结果文件。

```powershell
python -B manifold_project/tools/verification/check_cross_scale_foundations.py
python -B manifold_project/tools/verification/check_ctde_extensions.py
```

- `check_cross_scale_foundations.py`：仅使用 Python 标准库，自包含检查 P0/F1/F2/F3，分别对应受限模型的规模不变性、信息缺失、源不可识别和方向翻转；这些编号仅为脚本内的模型标识。
- `check_ctde_extensions.py`：需要 NumPy；先执行 `check_ctde_local_policy.py`，再核验方差、参数投影、成对交互、二阶边界及本次新增推论的有限例子。
- `extract_math_check.py`：默认提取当前 CTDE 主稿公式，可传入其他 Markdown 路径；按需生成 `build/mathcheck.tex`，供本机 TeX 工具检查，生成目录已被 Git 忽略。

数值核验只支持列出的模型与恒等式；新版模块的工程验证也不等于收敛或导航/仓库满足迁移条件。下一步先运行 P-E（方向估计）、P-A（Actor 实现）、P-T（条件迁移）的 pilot，命令见实验模块说明，再确定正式源配置。
