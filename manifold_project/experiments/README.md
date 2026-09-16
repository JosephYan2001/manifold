# 三个实验环境

每个环境的配置、实现、训练、评价、测试和依赖保存在自己的目录中。

| 目录 | 环境 | 当前状态 |
|---|---|---|
| [pair_coordination](pair_coordination/README.md) | 可枚举成对协作 | MLP 与类型表学习、actor 拟合、独立检查、多种子比较、曲线与 40 项测试已实现 |
| [cooperative_navigation](cooperative_navigation/README.md) | MPE2 合作导航 | 已建立目录；环境封装和训练待实现 |
| [rware](rware/README.md) | RWARE 仓库 | 已建立目录；环境封装和训练待实现 |

后续开发遵循[三个实验环境构建指南](../docs/三个实验环境构建指南.md)。研究问题、对照方法和统计协议见[实验方案](../docs/策略流形研究设计.实验方案.md)。

第一个环境的计分任务和实验安排见[设计文档](pair_coordination/设计文档.md)，代码功能与开发待办见[模块说明](pair_coordination/模块说明.md)。

在仓库根目录运行现有环境：

```powershell
python -m manifold_project.experiments.pair_coordination.run_demo
python -m unittest discover -s manifold_project/experiments/pair_coordination/tests -v
```

论文通用数学核验位于 `manifold_project/tools/verification/`，独立于三个环境。
