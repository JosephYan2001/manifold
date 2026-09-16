# pair_coordination：训练与调试

这是一个多人同时选择 A/B 的一步协作任务。每个机器人只读取自己的类型，环境根据个人选择和两两交互计算共同奖励。

## 阅读入口

按实验文档批量运行算法比较、消融和机制验证，请先读[实验运行说明](实验运行说明.md)。统一入口为 run_experiments.py；原 train.py 继续用于单次训练。

| 文档 | 内容 |
|---|---|
| [设计文档](设计文档.md) | 环境规则、计分例子与实验安排 |
| [模块说明](模块说明.md) | actor、方向函数、基线与检查模块 |
| [训练指标与调参说明](训练指标与调参说明.md) | 指标含义、学习率及预算排查 |
| 本页 | 配置、训练命令、输出文件与恢复 |

训练入口使用 PyTorch，实现 MLP 或类型表的自动微分与 Adam 更新，支持 CUDA。采样、精确枚举、统计接受检查和类型表 critic 在 CPU 上执行。NumPy 模型保留为数值核验与旧权重读取实现。直接策略更新对照和正式容量比较仍待完成。

先激活已安装 PyTorch 的环境。本机可用环境为：

~~~powershell
conda activate 'D:\python_exercise_program\dqn+pso\.conda'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
~~~

`--device auto` 为默认设置，CUDA 可用时使用 GPU；`--device cuda` 强制使用 GPU，不可用时报错；`--device cpu` 使用 CPU，也支持 `--device cuda:0`。学习损失、反向传播和参数更新在所选设备上执行。当前使用 float64 核对数学实验，pair 网络较小，GPU 不保证比 CPU 更快。

每次运行的 `runtime.json` 记录 PyTorch 版本、CUDA 构建版本、实际设备、GPU 名称和计算精度。checkpoint 继续使用 JSON 保存权重，可加载已有模型。恢复时允许用 `--device` 切换设备。旧 NumPy 训练迁移为 PyTorch 后可继续训练；跨实现或跨设备的后续数值不保证逐位相同。优化器每轮重新初始化，checkpoint 对应已完成轮次。

## 1. 先检查本次训练计划

在 D:\manifold 执行：

~~~powershell
python manifold_project/experiments/pair_coordination/train.py --training-config manifold_project/experiments/pair_coordination/configs/train_minibatch.json --dry-run
~~~

该命令打印实际源环境、网络、基线、每轮回合数、每批大小、更新次数、两个学习率和预算估算，然后退出，不创建训练目录。

当前配置中的两个学习率均为 0.001，训练中保持恒定。方向使用 direction_lr，actor 拟合使用 actor_lr。类型表 critic 采用条件样本均值拟合，没有梯度学习率。

当前文件设置最多 500 轮、500000 源回合。每轮检查一个候选时，一轮消耗 4096 回合，500 轮需要 2048000 回合；两项停止上限分别生效。以每轮一个候选估算，现有预算只能支持约 122 轮。实际次数取决于检查分支。

## 2. 开始训练

~~~powershell
python manifold_project/experiments/pair_coordination/train.py --training-config manifold_project/experiments/pair_coordination/configs/train_minibatch.json --max-rounds 5 --acceptance empirical --plot
~~~

empirical 使用独立源数据的经验均值筛选，没有 Hoeffding 置信保证。省略该覆盖时采用文件中的 hoeffding 检查；当前样本量可能不足以通过保守检查。

支持直接脚本与 python -m manifold_project.experiments.pair_coordination.train 两种启动方式。手动指定的相对路径均相对于终端当前目录。

## 3. 常用参数与生效规则

| 参数 / JSON 字段 | 含义 |
|---|---|
| --episodes-per-round / episodes_per_round | 每轮新采集的训练 episode 数 |
| --batch-size-episodes / batch_size_episodes | 每次梯度更新的完整 episode 数 |
| --direction-epochs / direction_epochs | 方向网络遍历数据的次数 |
| --actor-epochs / actor_epochs | 每个候选 actor 遍历数据的次数 |
| --direction-lr / direction_lr | 方向 Adam 学习率 |
| --actor-lr / actor_lr | actor 拟合 Adam 学习率 |
| --max-rounds / max_rounds | 外层轮数上限 |
| --max-source-episodes / max_source_episodes | 总源回合上限，包含 critic 和检查 |
| --check-episodes / check_episodes | 每批独立检查的回合数 |
| --baseline zero/table | 零基线或集中类型表基线 |
| --actor-model、--direction-model | 分别选择 mlp 或 table |
| --hidden-width、--hidden-depth | MLP 隐藏宽度、层数 |
| --eta、--attempts | 初始目标步幅和每轮候选数上限 |
| --seed | 采样和初始化种子 |

命令行覆盖 JSON，JSON 覆盖默认值。新运行未指定优化计数方式时，默认使用小批次 epoch 模式。episodes、rounds、budget、fit_lr 为兼容名称；direction_steps、fit_steps 仅用于旧全批次模式，已从常用帮助隐藏。正 epoch 与对应旧 steps 同时提供会报错；冲突别名和未知字段也会报错。--print-config 可查看内部兼容字段；正式运行的 resolved_config.json 使用规范名称。

当前 1024 回合、每批 128 回合、25 epochs，意味着每个 epoch 8 次更新，共 200 次。同一回合全部机器人保持在同一批；每个 epoch 重新打乱，最后不足一批的回合仍参与训练。方向和 actor 分别更新自己的参数。

## 4. 怎样阅读训练过程

每轮依次执行冻结 actor、可选 critic、训练采样、方向学习、方向检查、候选拟合、回报检查、提交结果。终端打印方向更新次数、候选数量、拟合 KL、接受决定及最新 checkpoint 路径。方向未通过时，候选数为零。

停止时显示原因、完成轮数、已用预算、剩余预算及下一轮最低启动成本。当前控制器会为启动新一轮预留训练、方向检查和至少一个候选回报检查的预算。

| 文件 | 用途 |
|---|---|
| requested_config.json | 用户参数和配置文件原文；程序调用则保存输入配置 |
| resolved_config.json | 规范名称的完整生效配置 |
| config.json | 内部兼容配置、源规则与 Python/NumPy 版本 |
| runtime.json | PyTorch 版本、实际 CPU/CUDA 设备和 float64 精度 |
| training_plan.json | 批次数、更新次数、成本估算、恒定学习率 |
| updates.jsonl | 每次梯度更新的小批次损失、梯度范数、实际学习率 |
| direction.jsonl | 整批方向损失、精确误差与分数；包含 epoch 边界 |
| actor_fit.jsonl | 每个候选在整批输入上的 epoch 拟合 KL |
| data.jsonl | 类型出现次数、奖励均值和旧 actor 最小概率 |
| checks.jsonl | 独立检查指标、接受决定、候选摘要及可用模型路径 |
| rounds.jsonl、policy.jsonl | 每轮执行结果、源成本与真实回报 |
| stages.jsonl | 本运行段各阶段耗时与新增源回合数 |
| batches.jsonl | 每批源数据的用途、种子和成本账目 |
| summary.json | 最终模型、停止原因和诊断结果 |
| failure.json | 异常或中断的阶段、最近 checkpoint、未提交采样成本 |

日志不保存候选网络参数。网络身份使用摘要核对；没有保留的历史模型路径为 null。精确指标只用于诊断，不参与候选选择。

updates.jsonl 的大小随梯度更新次数增长，所有记录均为标量诊断。长实验可依据需要在后续扩展更新级日志采样频率。

## 5. 曲线

默认只生成 round_overview.png，一张 3×3 总览包含实际执行回报、每轮最终方向 loss、方向误差、最后候选拟合 KL、接受结果、累计源成本、独立方向检查，以及各阶段采样成本和耗时。无候选的轮次不填充拟合误差；候选被拒绝时回报仍来自旧执行策略。默认不额外生成 SVG 或可从日志重新计算的 round_overview.json。

训练过程中也可以单独生成截至最近提交轮次的总览：

~~~powershell
python -m manifold_project.experiments.pair_coordination.analysis.plot_training "运行目录" --overview-only
~~~

此命令读取日志快照，不等待整次训练结束，也不修改模型。最新一条未写完的日志会暂时跳过。恢复运行时只展示当前目录记录的运行段。

--plot 默认生成上述单张总览；也可以对已有目录调用：

~~~powershell
python -m manifold_project.experiments.pair_coordination.analysis.plot_training "运行目录"
~~~

需要排查轮内学习过程时，在上述绘图命令末尾添加 --detailed，才会额外生成详细分图、SVG 和 round_overview.json（批量实验的 comparison 仍单独生成）：

- direction_curves：轮内方向学习的整批损失、真实误差和总体分数。最多按轮次位置均匀展示 6 轮，含首末轮，避免长实验的图例挤占绘图区；全部数据仍在日志中。横轴是本轮方向参数更新次数。
- direction_checks：各轮独立方向检查的均值与实际判断值。
- actor_fit_epochs：最近 10 个候选的 epoch 拟合过程及接受状态；所有候选数据保存在日志中。
- policy_curves：实际执行策略回报、候选最终拟合 KL、接受比例。
- stage_costs：本运行段各阶段的交互成本与耗时。
- comparison：批量实验的独立种子均值与标准差。

固定 direction 模式不更新 actor，所以回报保持不变。候选拒绝时继续使用旧 actor。不同轮的方向对应不同旧策略，分数无需跨轮单调增加。缺少拟合日志表示该运行尚未进行候选拟合；旧运行缺少新增日志时跳过相应新图。

绘图使用 Matplotlib 和中文字体，例如 Microsoft YaHei、SimHei 或 Noto Sans CJK SC。

## 6. critic 对照

默认 baseline=zero，团队奖励直接作标签。--baseline table 会先用独立源批次拟合全队类型条件下的平均回报，然后冻结基线，标签使用奖励减去该预测值。critic 不接收方向损失的梯度，部署时也不使用 critic。

在当前一步任务中，critic 不是必需模块。它可能影响估计方差，但拟合质量和额外采样成本都需要测量；有界基线还会增大当前保守检查使用的残差界。

对照时分别运行 --baseline zero 和 --baseline table，固定 actor/方向结构、优化预算和种子，比较方向误差、梯度波动及累计源成本。相同总源预算下另行比较最终回报。不同基线运行的数据流调用数不同，单靠相同 seed 不保证方向训练批次逐条相同，不能宣称已经构成共享数据的配对实验。

## 7. 保存模型与恢复

- --save-batches：保存完整采样 NPZ，默认关闭。
- --save-direction-snapshots：保存旧 actor 与方向及可选 critic 快照，默认关闭。
- --checkpoint-every：仅兼容旧配置，不再控制保存频率。

checkpoints 目录仅保留两个模型：best.json 保存源环境精确期望团队回报最高的执行策略（含初始策略，回报相同时保留较早的模型）；final.json 每个完成轮次覆盖更新，训练结束时即为最后模型，中断后也可用于恢复。评分按本环境独立同分布类型和成对奖励的解析公式计算，不需要联合枚举或额外采样，不参与训练接受检查；被拒绝的候选不参与最优模型评选。actor.json 保留为最后模型的兼容入口。

模型使用原子替换，不再生成 actor_round_*.json 或 latest.json。final.json 内携带最优快照，单独复制后恢复也能延续历史最优记录；从旧格式 checkpoint 恢复时，以该恢复点为起点重新记录最优。已有历史结果不会自动删除。summary.json 记录 best_checkpoint、best_round 和 best_expected_return。

~~~powershell
python manifold_project/experiments/pair_coordination/train.py --resume "原运行目录/checkpoints/final.json" --output-dir "新运行目录"
~~~

恢复仅支持已完成轮次边界，沿用原配置及检查预算。每轮优化器重新创建，后续初始化、采样与分批均由确定性种子控制，已测试恢复与连续运行一致。未完成轮次需重新执行；failure.json 记录中断前已采集但未提交的成本，正式总成本还需加入这些消耗。恢复后的日志只记录续跑段，原日志仍保留。

## 8. 批量比较与冻结评价

~~~powershell
python -m manifold_project.experiments.pair_coordination.train --mode direction --seeds 1 2 3 --methods analytic sampled --sample-sizes 128 512 2048 --direction-lr 0.001 --plot
python -m manifold_project.experiments.pair_coordination.evaluate "运行目录/checkpoints/final.json" --n-agents 8
~~~

每次批量运行独立使用预算。aggregate.json 保存跨种子的均值与样本标准差。方向误差对应各自最后一轮旧 actor；policy 模式下不同运行的旧 actor 可能不同。目标结果不参与源训练选择。

configs/pair_conflict.json、pair_second_order.json 分别用于偏好冲突和二阶协作边界；pair_composition_target.json 用于冻结后的组成变化评价。

## 9. 测试与依赖

~~~powershell
python -m unittest discover -s manifold_project/experiments/pair_coordination/tests -v
python -m pip install -r manifold_project/experiments/pair_coordination/requirements.txt
python -m pip install -r manifold_project/experiments/pair_coordination/requirements-visualization.txt
python -m manifold_project.experiments.pair_coordination.run_demo
python -m manifold_project.experiments.pair_coordination.visualization.render
~~~

环境演示和静态图展示固定策略与计分例子。学习结果由 train 产生。三环境的总体安排见[实验方案](../../docs/策略流形研究设计.实验方案.md)。
