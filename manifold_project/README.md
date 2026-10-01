# 单源 CTDE 的方向函数学习

项目研究在规定的本地信息和一个源规模下，利用团队经验学习动作概率的改进方向，再由共享 Actor 实现，并检验规模变化后的冻结执行表现。

研究分为两条比较：AN 与 SA 比较同一总体方向目标的有限样本估计差异；方向方法与 DA/PPO 比较“先学方向再实现”的训练价值。AN 与 SA 接近不等于方向路线失败，输入维度兼容也不等于跨规模迁移保证。

## 阅读与运行

| 入口 | 职责 |
|---|---|
| [CTDE 方向学习主稿](docs/策略流形研究设计.CTDE本地策略改进版.md) | 理论主张、假设、命题与算法候选 |
| [总体实验方案](docs/总体实验方案.md) | 环境职责、实验编号、比较方法、指标和预期结果 |
| [项目结构与代码说明](docs/项目结构与代码说明.md) | 层级职责、代码调用、配置映射与修改位置 |
| [公共实验运行说明](experiments/README.md) | 安装、统一入口、配置层次、输出与恢复 |
| [协作导航实验手册](experiments/cooperative_navigation/README.md) | 当前导航协议、四组训练命令、指标与冻结评价 |
| [研究主张与开发路线](docs/研究主张与开发路线.md) | 当前证据和下一阶段顺序 |
| [结果索引](experiments/results/README.md) | 已有数据与分析报告 |
| [问题审查](docs/理论与实验对应问题审查.md)、[相关论文](../references/README.md) | 理论适用缺口与文献依据 |

成对和两步环境已形成多批机制、连续学习及迁移记录。当前应用工作是协作导航 success-v2 的源训练：成功才终止、200 步仅切分采样、轨迹跨段延续、无时间输入。旧限时导航结果作为历史记录；新协议的收敛和跨规模优势仍需新数据。

## 项目层级

```text
manifold_project/
  docs/                   理论、总体实验方案、结构说明
  experiments/
    configs/              当前矩阵配置
    pair_coordination/    成对训练主体
    cooperative_navigation/ 共用应用训练主体与导航手册
    applications/         任务适配与冻结评价
    finite/               有限机制诊断和两步模型
    common/               配置、统计、绘图与运行信息
    tests/                工程验证
    results/              实验记录与判断
  tools/verification/     独立数学核验
references/               论文与文献索引
archive/                  原始实现和历史计划归档
```

日常入口为 `python -m manifold_project.experiments`，在包含 manifold_project 的目录运行。原成对与导航训练主体直接复用归档后修订，实际执行不依赖 ZIP；来源见[归档说明](../archive/README.md)和[文件复用记录](experiments/reuse_audit.csv)。

## 数学核验

以下命令不训练模型：

```powershell
python -B manifold_project/tools/verification/check_cross_scale_foundations.py
python -B manifold_project/tools/verification/check_ctde_extensions.py
```

数值核验支持具体有限例子和恒等式，不代替一般定理或环境条件证明。文稿公式提取工具为 tools/extract_math_check.py；工程测试命令见实验运行说明。
