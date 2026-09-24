# 成对协作：恢复的原实现

本目录各层级、对应函数和 P-L 调用流程见[项目结构与代码说明](../../docs/项目结构与代码说明.md)第 5 节；启动命令统一见[实验运行说明](../README.md)。

本目录直接恢复自 `archive/legacy_experiments_20260923.zip`，保留环境、表格/MLP、训练器、采样、检查、保存和旧入口。逐文件变更见 [reuse_audit.csv](../reuse_audit.csv)。

当前论文矩阵通过 `current_protocol.py` 设置 P-L 的源奖励、初始策略、独立基线、经验检查与源监控选模，再调用原 `training/runner.py::run_training`。固定基点机制模块直接使用原方向拟合和 Actor 拟合函数。新增 DA 比率代理作为当前方案对照，原 PG 仍保留兼容。

保留 `train.py` 单次训练、`evaluate.py` 精确评价，以及 `envs/models/training/evaluation/analysis` 运行模块。旧 `run_experiments.py`、嵌套 `experiments/`、旧套件配置和历史结果整理脚本已移除；所需 bootstrap/Wilson 统计合并到公共 `common/statistics.py`。关键测试集中在 `../tests/test_pair_*.py`。当前 P-E/P-A/P-T/P-L 命令统一见[实验模块说明](../README.md)。
