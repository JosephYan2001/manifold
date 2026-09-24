# 有限机制实验

本模块与原成对训练器的分工见[项目结构与代码说明](../../docs/项目结构与代码说明.md)第 8 节；全项目调用关系与配置映射也统一在该说明维护。

当前有限模块直接调用恢复的 `pair_coordination/` 源码，补充总体方案中的诊断与两步模型。P-L 调用旧 `run_training`，没有另一套成对多轮训练循环。运行不读 ZIP、不载入旧结果。

| 文件 | 职责及复用关系 |
|---|---|
| `environments.py` | 一步成对、两步模型、完整回合和精确真值；补规模及组成变体 |
| `numerics.py` | 方向拟合直接调用旧 `TableDirection/fit_direction`；不受限 Actor 拟合调用旧 `TableActor/fit_actor`；仅受限偏移类、时间轴和诊断另行适配 |
| 优化器 | 直接从 `pair_coordination/training/optim.py` 导入 Adam，不保留转发文件 |
| `objectives.py`、`acceptance.py` | 多时间步的目标函数核验、经验检查及边界诊断；与旧函数做数值对照 |
| `diagnostics.py` | P-E/P-A/P-T/P-I/P-B/T-V 固定基点实验 |
| `training.py` | P-L 转交旧成对训练器；T-L 实现旧包没有的两步训练 |
| `../pair_coordination/current_protocol.py` | 当前 P-L 配置、独立源监控和成本映射；读取旧事件账本汇总 CSV |

P 的输入为自身类型；T 为首步及第二步正/负公开状态。策略数组是动作 `+1` 的概率，方向系数对应 `q=(-c,c)`。T 一个回合消耗两个联合环境步。同数据 AN/SA 仅改变二次项；P-A 从同一个学得方向和旧 Actor 出发比较拟合预算与函数类。

P-I 当前完成精确输入映射及损失分解，同数据学习对照仍待补充；P-TB 的完整实体名单实验另走实体网络与旧导航 Runner，不能用自身类型表格冒充。

P-L/T-L 的预算按方法、标签组分别核算，包含训练、独立基线、检查和源选模监控；纯报告评价另计。MC 学习标签的基线由独立源样本拟合并冻结，精确优势只属 oracle 诊断。经验检查不是高置信改进保证。固定基点实验由样本档、数据批次数及拟合档控制。

P-L 保留原 `checkpoints/final.json、best.json` 与单个事件账本，含原训练恢复字段；当前协议 best 用独立源监控选取。T-L 保存 `final.npz、best_source.npz`，只含有限策略而非完整续训状态。主比较均用 final，`load_policy` 支持两种格式，不逐轮新增文件。

`actor_fit_steps` 控制实际 Actor 优化，`fit_steps` 列表是 P-A 的拟合预算轴；`source_eval_episodes` 参与选模并计费，`evaluation_episodes` 只做最后报告。有限网络不使用应用的 `hidden/heads/relation_layers`，但恢复的旧成对训练模块会导入 PyTorch。

命令统一见[实验模块说明](../README.md)，逐文件来源见 [reuse_audit.csv](../reuse_audit.csv)。验证：

```powershell
python -m pytest manifold_project/experiments/tests/test_finite.py manifold_project/experiments/tests/test_pair_learning.py -q
```
