# 协作导航：恢复的原训练主体

本目录各层级、共用应用训练流程和工具适用范围见[项目结构与代码说明](../../docs/项目结构与代码说明.md)第 6 节；任务适配见第 7 节，启动命令统一见[实验运行说明](../README.md)。

本目录直接恢复自 `archive/legacy_experiments_20260923.zip`。原 `training/runner.py`、采样器、作者 PPO、监控、轮次保存与恢复继续承担训练；`current_protocol.py` 只转换当前论文配置并汇总结果。

当前观测使用原生完整实体列表，模型入口通过 `models/entities.py` 编码、关系交互和动作读取，支持变人数。旧展平模型保留用于历史配置及 checkpoint 读取；两种模型不能互相续训。导航仍使用原生 MPE2 动力学与奖励，首次同时覆盖全部目标结束，期限到达记为超时；当前 MC 更新采完整回合。

`envs/tasks.py` 为仓库及完整名单成对任务提供旧采样器接口，复用同一个 Runner，不另起训练框架。MAPPO/IPPO 保留原作者更新及 ValueNorm，仅适配共同 Actor 和集中/本地 Critic；部署目标规模时不加载 Critic。

`run_experiments.py --conditions ...` 底层工作流和恢复入口保留；原根目录监控、回放和曲线检查移入 `tools/`。旧实验分支续训、旧候选搜索、延长期限评价等脚本已从活动目录移除。原 `tests/` 中与当前协议有关的数学、恢复及作者源码完整性检查并入公共 `../tests/`，过时协议测试不再维护。当前命令与目录职责统一见[实验模块说明](../README.md)，逐文件来源见 [reuse_audit.csv](../reuse_audit.csv)。
