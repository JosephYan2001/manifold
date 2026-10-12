# 协作导航实验手册

针对现有 AN/MAPPO 结果，下一轮先执行的三组 AN 对照及后续网络命令见[下一轮实验执行清单](下一轮实验执行清单.md)。

代码模块、观测字段与张量维度、实体/历史网络和各算法数据流见[导航模块组成说明](模块组成说明.md)。本手册维护任务协议与执行命令；仓库共用训练器的接入及自身协议见[仓库模块说明](../applications/envs/仓库环境模块组成说明.md)。

本环境研究方向学习能否转化为实际协作能力，并在源规模学好后检验冻结 Actor 的跨规模表现。当前默认协议为 `navigation-episodes-v1-history-v1`：每个训练 episode 从新场景开始，成功即结束，否则最多 400 步硬重置；收齐 16 个 episode 后更新网络。Actor 使用最近 8 帧局部观测和上一动作。超时仍使用重置前末状态自举，Actor 与 Critic 均不接收任务时钟。本文件统一维护导航的任务定义、训练流程、配置、命令和数据解释。

跨环境研究安排见[总体实验方案](../../docs/总体实验方案.md)，公共安装与入口见[实验运行说明](../README.md)。当前阶段先做单种子源训练，不能把工程测试通过解释为收敛或迁移优势。

针对“方向尚未可靠，Actor 拟合是否传播或放大误差”的问题，见[方向学习与 Actor 拟合问题诊断方案](../../docs/方向学习与Actor拟合问题诊断方案.md)。该方案规定旧 Actor、直接执行的目标策略、拟合 Actor 三组比较；新增诊断尚待实现，不改变本手册的训练协议。

## 1. 任务和信息协议

| 项目 | 定义 |
|---|---|
| 环境 | MPE2 1.1.1 simple_spread；原生物理过程和五离散动作：停、左、右、下、上 |
| 源与目标 | 源 4 个机器人/4 个目标；目标 2/2、6/6、8/8，初始位置范围固定，因此数量和密度同时变化 |
| 奖励 | 原生 agent 奖励的平均值，local_ratio=0.5；距离和碰撞成本，不额外加成功奖励或截止罚分 |
| 成功 | 同一时刻所有目标到最近机器人的距离均小于 0.1，首次满足即终止 |
| 成功的边界 | 不要求停稳、持续驻留或严格一一匹配；不是各目标曾经分别被访问 |
| 采样 | collection_mode=episodes；每批 16 个独立重置 episode，每个最多 horizon=400 步，提前成功不补齐步数，批次最多 6400 联合环境步 |
| 重置 | 成功时 terminated=true；未成功但达到 400 步时 truncated=true、terminated=false，保留末状态自举。下一 episode 清空历史，从新场景开始，不跨轮继续旧场景 |
| Actor 自身输入 | 二维速度和二维绝对位置，共 4 维，无时间字段 |
| 实体输入 | 所有目标和同伴，每条 6 维：两种类型标记、二维相对位置、两维原生通信；当前静默机器人通信为零 |
| 历史 | 最近 8 帧（含当前帧），每帧包含自身和全部实体观测、上一动作的五维 one-hot、有效帧标记；初始缺帧补零并屏蔽 |
| 信息限制 | 不直接提供同伴速度，可利用历史变化学习运动信息；有限历史与集合聚合不保证恢复完整状态 |
| 网络与规模 | 每帧用共享实体编码、关系注意力和动作相关读取，再按从旧到新的顺序融合 8 帧特征；不按实体槽位跨帧对齐，不引入全局信息或 GRU；实体数量改变不影响参数尺寸 |
| Critic | 集中全局位置、速度及目标位置，6N 维，源规模下 24 维，无时间字段 |
| 评估 | 独立场景中冻结 Actor，默认观察到 1000 步；成功提前结束，未完成记为 evaluation_cutoff |

同一网络能读取可变实体数，只说明维度兼容。全名单的聚合是否保留决策信息、源经验是否约束目标行为，仍需独立证据；导航按 B 类经验外推环境解释。

`configs/navigation.json` 的 `history=8` 对 AN、SA、DA、MAPPO-E、IPPO-E 统一生效；AN/SA 方向网络也使用同一历史协议。每帧先得到动作相关空间特征，再拼接该帧上一动作与有效标记，通过时间融合 MLP 输出当前分数。历史仅在同一 episode 内累积，重置清空；开头不足 8 帧用有效位屏蔽，仍保留这些真实转移用于学习，不额外等待 8 步。末状态自举只读取历史快照；训练检查、冻结评价、跨规模评价和 GIF 使用同样的帧构造。集中 Critic 仍读取当前全局状态；IPPO 的局部 Critic 使用相同历史。

设置 `history=1` 可做单帧对照，保留旧单帧网络与输入口径。8 帧版新增时间融合参数，必须从头训练，不能直接续训旧单帧 checkpoint；已有旧模型仍按保存配置评价和回放。历史增加推理和训练计算成本，不增加已计费的环境步，也不保证收敛或消除感知压缩问题。

400 步是训练采样上限，不是任务价值归零的期限，不添加失败奖励。成功终止与超时重置都结束本次 episode，但只有成功令后续价值为零。`horizon` 是 episode 模式的训练上限，适配器据此同步 `train_reset_horizon`；后者仅在旧 streaming 模式下独立控制跨轮重置。评估观察上限仍独立设为 1000 步，不改成训练用的 400 步。[Gymnasium 的时间限制说明](https://gymnasium.farama.org/v0.26.3/tutorials/handling_time_limits/)同样区分真终止与需保留自举的外部截断。

## 2. 一轮训练如何组织

1. 冻结本轮旧 Actor 和 Critic。
2. 用同一冻结 Actor 收集 16 个 episode。每个从新场景开始，成功立即结束，未成功到 400 步硬重置。全部收集完成后，组织成带有效转移 mask 的变长批次。
3. 用冻结 Critic 计算标签，再训练 Critic。成功边界后续值为零，截断边界使用真实末状态价值。
4. AN/SA 学习方向，再拟合 Actor；DA 直接优化策略代理；MAPPO-E 使用作者 PPO 更新。
5. 开启的检查使用额外独立场景，不改变已经收集的训练批次。接受候选后提交 Actor。
6. 记录实际采样成本、episode 长度、超时和成功数，保存 final 及 best；完成更新后丢弃本批采样，用新 Actor 收集下一批。没有跨更新复用的经验回放池。

`batch_episodes=16` 表示每次更新前收集 16 个 episode，不保证 16 次任务成功。一轮最多 6400 个环境步；若提前成功则更少，预算按实际步数计费。启动一轮前为整批最大成本及剩余监控预留预算，余额不足时停止。历史窗口长度 8 和训练批次大小 16 是两个不同参数。

AN/DA 的 Critic 标签是截断处自举的 n-step reward-to-go；策略标签默认 GAE。MAPPO-E 保留作者 GAE、优势标准化、ValueNorm、价值损失与 PPO 优化。适配层逐段计算目标后打包，GAE 不跨其他轨迹的重置或截断边界。这些训练流程并非完全相同。

检查版从独立重置场景采至成功或 400 步。回报检查比较实际奖励加截断处冻结旧 Critic 的尾部估值，方向检查也使用近似标签；两者都是经验筛选，不能宣称完整任务真实收益保证。

独立重置解决了跨轮延续场景的问题，但 400 步截断及近似 Critic 自举仍不等于无限任务的完整回报、精确优势或无偏标签。它们是需要用实验检验的应用近似。

## 3. 配置与四组比较

配置按 `defaults.json → navigation.json → profile → --config → 命令参数` 合并。

| 文件 | 职责 |
|---|---|
| [configs/navigation.json](../configs/navigation.json) | 导航任务、输入、网络、优化和监控的共同参数 |
| [configs/navigation/no_checks.json](../configs/navigation/no_checks.json) | 仅关闭两个经验检查 |
| [configs/navigation/checked.json](../configs/navigation/checked.json) | 仅开启方法适用的经验检查 |
| [configs/profiles.json](../configs/profiles.json) | pilot/formal 的预算和种子默认值 |
| [底层 configs](configs/) | Runner 加载及旧套件兼容所需；不是当前实验的第二套用户配置 |

当前启动只使用共同参数与两份检查开关配置。底层配置用于训练器加载，不作为另一套运行命令。

| 实验组 | 方法 | 覆盖配置 | 方向检查 | 回报检查 |
|---|---|---|---|---|
| AN 核心版 | AN | no_checks.json | 关 | 关 |
| AN 完整版 | AN | checked.json | 开 | 开 |
| DA 完整版 | DA | checked.json | 不适用 | 开 |
| MAPPO-E | MAPPO-E | no_checks.json | 不适用 | 不适用 |

共同设置：种子 40、总源预算 200 万步、隐藏宽度 128、4 头和 2 层关系交互；Actor/Critic/方向学习率分别为 3e-4/1e-3/3e-4。方向与 Actor 拟合各 16 epoch，Critic 10 epoch，PPO 10 epoch。`minibatch_size=3200` 按 400 步和 4 个机器人换算为每小批 2 个 episode；提前成功时实际样本更少。作者 PPO 将本批所有真实转移打包后划分 8 个小批次（样本极少时相应减少），不把补齐位置用于损失。

显存分块由 `network_chunk_size=256` 控制，单位是机器人的完整 8 帧历史样本，不是 episode。网络大批量前向分块计算；反向采用激活重计算，只保存输入和必要结果。关系值消息先加权再线性投影，避免展开巨大的逐实体对高维张量。上述处理保留网络参数、完整历史、训练批次、损失权重和优化器更新次数，代价是额外重算时间；不等同于缩小 PPO 小批次或减少采样。旧实现的整批关系张量单个可达 6.25 GiB，显存分配器选项不能消除这种实际张量需求。

2026-10-02 在 8 GB RTX 4060 Laptop、PyTorch 2.7.1+cu118 上验证：AN 无检查与 MAPPO 各实际采集 16×400=6400 步，保留 8 帧、隐藏宽度 128、4 头和两层关系网络，各模块只运行 1 个 epoch；均完成更新。PyTorch 峰值已分配显存分别为 0.491/0.485 GiB，峰值预留为 0.611/0.541 GiB，不含驱动及其他进程。另有 49 项数值等价、梯度、更新、采样、恢复和入口检查通过。这是完整批次的显存路径验证，不是完整预算训练或收敛结果。

每 65536 源步进行一次 8 场景监控，观察上限 1000 步；这组 200 万预算最多为监控预留 256000 步，包含在源预算内。最终 20 场景评价另计费。检查开启时每批 16 个窗口。样本量用于当前试跑，不代表正式评价已充分。

## 4. 当前训练命令

在包含 `manifold_project` 的目录运行，激活已经安装依赖的 Python 环境。旧导航结果已清理，以下均按新协议从头训练。四组相互独立，单张 GPU 依次执行。

```powershell
$cfg = "manifold_project/experiments/configs/navigation"
$out = "manifold_project/experiments/results/navigation_ep400_history8_2m_s40"
$budget = 2000000

# 先运行：AN 无检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/no_checks.json" --budget $budget --seeds 40 --device cuda --output "$out/an_core" --plot --plot-every 5 --allow-duplicate-openmp

# 先运行：MAPPO-E
python -m manifold_project.experiments --experiments N-C --profile pilot --methods MAPPO-E --config "$cfg/no_checks.json" --budget $budget --seeds 40 --device cuda --output "$out/mappo" --plot --plot-every 5 --allow-duplicate-openmp

# 检查开销对照：AN 双检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/checked.json" --budget $budget --seeds 40 --device cuda --output "$out/an_checked" --plot --plot-every 5 --allow-duplicate-openmp

# 检查开销对照：DA 保留回报检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods DA --config "$cfg/checked.json" --budget $budget --seeds 40 --device cuda --output "$out/da_checked" --plot --plot-every 5 --allow-duplicate-openmp
```

加 `--dry-run` 只检查配置。输出目录已有数据时不会覆盖，需要使用新目录；统一矩阵入口尚无自动续训开关。200 万步是观察点，不是已确认的收敛预算。所有方法在源环境完成设置选择后，再统一扩预算与增加种子。

命令保留此前使用的 `--allow-duplicate-openmp` 兼容开关。若改预算，同步修改 `$budget` 和 `$out` 中的预算名称；每个方法使用相同总源预算，不要求更新轮数相同。

OOM 后更新代码并结束已报错的 Python 进程，再启动新进程。失败目录保留诊断记录，不直接覆盖；例如将 AN 无检查命令的输出改为 `$out/an_core_memfix`，后续评价/GIF 的源路径同步替换。仅调整分块仍不够时先检查 GPU 是否被其他训练占用，不同时运行四组。

`--evaluation-horizon 1000` 可显式指定观察长度，不改变任务终止或 Actor 输入。`--evaluation-episodes 500` 增加最终报告场景，不改变源监控的 8 场景；源监控数量要调整共同配置中的 source_eval_episodes。正式实验需重新确定预算和评价精度，不能只换成 formal 就认为证据已充分。

## 5. 冻结评价与任务可视化

源策略学好后，用 final Actor 做目标规模评价。以下示例接续上面的 AN 核心版，其他组替换方法、检查配置和源目录。

```powershell
python -m manifold_project.experiments --experiments N-T --profile pilot --methods AN --config "$cfg/no_checks.json" --source "$out/an_core" --evaluation-episodes 500 --evaluation-horizon 1000 --device cuda --output "$out/an_core_transfer" --plot
python -m manifold_project.experiments.visualize --checkpoint "$out/an_core/N-C_source_learning/full/AN_s40/checkpoints/final.pt" --agents 4 --seed 1000000 --output "$out/an_core/replay_n4.gif"
```

导航 GIF 使用旧版轨迹画法：带编号的彩色机器人、黑色目标叉号、完整运动轨迹，以及当前覆盖率和碰撞对数；坐标范围在整段回放中固定。默认每一步一帧、10 FPS（`--frame-stride 1 --fps 10`），增大 `--frame-stride` 会加速播放。默认随机采样动作，`--deterministic` 改为选择最大概率动作。末帧显示成功或评估截断原因。

目标只执行 Actor，不运行目标 Critic、不训练、不按目标分数挑源模型。GIF 是单场景示例，不能替代成功率。N-I/N-R/N-Gd/N-Gr 等后续实验见总体方案；N-F 需要源训练提前启用固定方向诊断，当前关闭该诊断的运行不能事后凭空生成 N-F 数据。

## 6. 如何读日志和结果

| 内容 | 解释 |
|---|---|
| train_episode_return；兼容 train_J/train_discounted_return | 本轮 16 个 episode 的观测折扣回报均值，超时后的预测不加入日志回报 |
| train_mean_reward、train_end_goal_distance、train_agent_radius_max | 真实步均奖励、各段末目标最近距离均值、段末机器人距原点最大值；用于发现漂移，不进入 Actor 输入 |
| train_task_age_max、train_reset_count、train_success_count | 本批最大 episode 长度、超时次数及成功次数，最大长度不得超过 400 |
| train_episode_count、train_episode_steps_mean、train_batch_steps | 本批 episode 数、平均长度、实际联合环境步数 |
| critic_mse、explained_variance | 原奖励尺度的价值误差及解释方差；MAPPO 优化用的 ValueNorm 损失另见 author_stats，不能仅凭原尺度 MSE 判断归一化失效 |
| monitor_return、eval_success、monitor_budget_checkpoint | 最近一次独立源监控的回报、成功率和对应节点；不是每轮重新评价 |
| events.jsonl 的 train_episode | 每个 episode 的长度、回报、成功或超时、自举方式；旧 streaming 日志的 train_segment/completed_task/training_reset 仅作历史兼容 |
| evaluation_episodes.csv | 冻结策略的逐场景观测回报、成功或截尾、观察上限、覆盖及碰撞 |
| summary.csv | 按训练种子汇总，再计算种子间统计；按协议、任务、观察上限和回报定义分组 |
| final.pt 与 best.pt | final 是主结果及恢复状态，best 是源监控选出的辅助 Actor |
| overview.png | episode 回报与独立源监控分面显示；监控曲线从事件账本取实际评价节点，含训练结束后的最终监控。总图覆盖更新 |

单步团队奖励取原生 agent 奖励均值；`discounted_return` 是 gamma=0.99 的折扣和，`return_undiscounted` 是实际奖励总和，`mean_reward` 是步均奖励。独立评价回报绝不加入 Critic 的预测。

日志 `episode_J` 来自更新前旧策略收集的本批 episode，每个最多 400 步；图中 `monitor_return` 来自冻结策略在独立场景中最多 1000 步的监控，每次 8 场景。长度、样本和策略版本可能不同，数值不必一致。最终报告另外使用 20 场景。旧 streaming 数据保留原 `segment_J/train_segment_return` 名称。

成功率指观察上限内首次完成的比例，同时报告 200/500/1000 步成功率、成功样本平均完成步数、未完成比例和截尾平均步数。截尾平均步数给未完成样本记观察上限，不等于估计它们真实的最终完成时间。有限观察不能证明永不成功。

训练、方向检查、旧/候选回报检查、源监控均按实际联合环境步计入预算；最终报告评价和额外机制诊断单列。同一轨迹的多个段不是独立训练种子。回报提高也不自动等于任务成功，应同时看距离、覆盖、碰撞和成功率。

## 7. 代码位置与历史兼容

| 模块 | 职责 |
|---|---|
| current_protocol.py | 统一配置映射、监控预留、训练记录汇总 |
| envs/navigation.py | 原生动力学与奖励、成功判定、全局状态 |
| training/collector.py | 独立 episode 收集、400 步截断、变长批次和实际成本；兼容旧 streaming 采样恢复 |
| training/ppo.py | 截断自举、GAE、标签、样本权重及方向损失项 |
| training/runner.py | 各方法更新、检查、监控、模型与采样状态保存 |
| training/author_ppo.py | 保留作者更新，适配逐段价值目标与批次 |
| models/__init__.py、models/entities.py、models/deployment.py | 训练输入解码、可变实体网络与冻结部署 |
| ../applications/evaluation.py | 独立场景观察、截尾记录、冻结目标评价 |

2026-10-02 已按用户要求删除旧导航结果和判断报告。原始源码 ZIP 保留作实现来源，不能作为当前运行配置。历史协议读取分支保留用于代码兼容；当前所有训练命令统一使用本手册的 400 步 episode 批次协议，结果状态见[结果目录](../results/README.md)。

工程检查覆盖终止与截断、无时间输入、逐段 GAE、跨轮连续性、CPU/CUDA 短训练、精确恢复和冻结变规模执行；通过测试不代表性能结论。运行验证的方法见实验运行说明。

## 8. 中断恢复与增加预算续训

两种操作统一使用 `python -m manifold_project.experiments --resume SUITE`，其中 `SUITE` 是包含 `manifest.json` 的结果目录，不是 `final.pt`。原方法、种子、batch、方向训练次数、设备和评估设置均从结果继承，无须重复传入 `--config`、`--profile` 或 `--methods`；恢复时禁止改变训练设置。

```powershell
Set-Location D:\manifold
$src = "manifold_project/experiments/results/navigation_diagnosis_20261010_2m_s40_v1"
$dst = "manifold_project/experiments/results/navigation_diagnosis_20261012_4m_s40_v1"

# 中断恢复：原目录、原累计预算，从最近一次提交的 checkpoint 继续。
python -m manifold_project.experiments --resume "$src/an_base" --allow-duplicate-openmp

# 增加预算：已完成的 200 万步运行，延长到累计 400 万步，写入新目录。
python -m manifold_project.experiments --resume "$src/an_base" --budget 4000000 --output "$dst/an_base" --allow-duplicate-openmp
python -m manifold_project.experiments --resume "$src/an_direction64" --budget 4000000 --output "$dst/an_direction64" --allow-duplicate-openmp
python -m manifold_project.experiments --resume "$src/an_batch64" --budget 4000000 --output "$dst/an_batch64" --allow-duplicate-openmp

# 延长训练中途退出后，恢复新目录；不再重新指定旧来源。
python -m manifold_project.experiments --resume "$dst/an_base" --allow-duplicate-openmp
```

任一命令加 `--dry-run` 只校验和打印恢复计划，不创建或修改结果。`--budget` 始终是累计源预算，包含已消耗的训练与源监测步骤。增加预算保留历史监测节点，并按原间隔追加新节点；不会重新采集历史监测。最终报告评价仍单独计费。

模型、Critic、优化器、随机状态、采样序号、采样器状态和累计成本通过原 Runner 恢复；中断但尚未提交的采样按原账本规则计费，不会删除实际消耗。已经完成的运行在同预算恢复时不再更新模型，但可以重建最终评估和汇总。增加预算要求源训练已经完成，且新输出目录独立于原目录。

当前公共恢复支持导航 `train`/`train_transfer` 套件；不支持有限实验、输入消融套件或 `save_fit_diagnostics=true` 的运行（FitProbe 尚未保存完整恢复状态）。启动后尚未生成任何 checkpoint 的已初始化运行会报错，不会悄悄从头训练。运行清单的 `recovery_history` 保存来源 checkpoint 哈希、配置和恢复前后的源码指纹；不同指纹会提示，但不能仅凭指纹差异判定算法变化。

输出继续采用标准套件目录：`N-C_source_learning/full/AN_s40/checkpoints/final.pt`，以及原有的训练曲线、最终评估和套件汇总。`tools/extend_navigation.py` 仅保留为转发到公共入口的兼容包装，`tools/extend_navigation_diagnosis.ps1` 也已改用公共入口，不再维护独立续训流程。

CPU 集成验证：`python -m unittest manifold_project.experiments.tests.test_recovery -v`。测试使用临时小规模环境，检查中断恢复与连续训练的参数一致性、预算扩展、扩展中断后的恢复、来源文件保护和 dry-run 无写入；不启动正式训练。
