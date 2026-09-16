# 固定本地信息下的多智能体策略改进

研究严格单源 CTDE 条件下的方向估计、共享 actor 实现与跨规模适用条件。

## 当前入口

- [研究设计](docs/策略流形研究设计.CTDE本地策略改进版.md)
- [实验方案](docs/策略流形研究设计.实验方案.md)
- [三个实验环境](experiments/README.md)
- [后续构建指南](docs/三个实验环境构建指南.md)

当前已实现一步成对交互环境、MLP 与类型表方向学习、actor 拟合、独立接受检查、多轮训练、冻结评价、多种子比较、曲线及自动测试。直接策略更新对照和应用基准训练尚待实现。

## 目录

```text
docs/                       研究文档与实验方案
  archive/                  重构备份
experiments/
  pair_coordination/        可枚举协作：配置、环境、评价、测试
  cooperative_navigation/   合作导航：环境代码与后续训练
  rware/                    仓库：环境代码与后续训练
tools/                      文档检查工具
  verification/             既有论文核验脚本
```

早期文档保留用于追溯。当前研究入口以以上 CTDE 设计及实验方案为准。

## 在仓库根目录运行

```powershell
python -m manifold_project.experiments.pair_coordination.run_demo
python -m unittest discover -s manifold_project/experiments/pair_coordination/tests -v
python manifold_project/tools/verification/check_ctde_reconstruction_20260909.py
```

可枚举环境依赖 NumPy，见 `experiments/pair_coordination/requirements.txt`。演示中的改进方向来自精确评价器，该输出属于环境与数学检查。
