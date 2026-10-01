# 协作导航实验手册

本环境研究方向学习能否转化为实际协作能力，并在源规模学好后检验冻结 Actor 的跨规模表现。当前采样协议为 `navigation-success-v3`：成功才真正终止，200 步切分训练段，连续采样到 1000 步未成功时按外部截断重置场景，使用重置前末状态自举。Actor 与 Critic 均不接收任务时钟。本文件统一维护导航的任务定义、训练流程、配置、命令和数据解释。

跨环境研究安排见[总体实验方案](../../docs/总体实验方案.md)，公共安装与入口见[实验运行说明](../README.md)。当前阶段先做单种子源训练，不能把工程测试通过解释为收敛或迁移优势。

## 1. 任务和信息协议

| 项目 | 定义 |
|---|---|
| 环境 | MPE2 1.1.1 simple_spread；原生物理过程和五离散动作：停、左、右、下、上 |
| 源与目标 | 源 4 个机器人/4 个目标；目标 2/2、6/6、8/8，初始位置范围固定，因此数量和密度同时变化 |
| 奖励 | 原生 agent 奖励的平均值，local_ratio=0.5；距离和碰撞成本，不额外加成功奖励或截止罚分 |
| 成功 | 同一时刻所有目标到最近机器人的距离均小于 0.1，首次满足即终止 |
| 成功的边界 | 不要求停稳、持续驻留或严格一一匹配；不是各目标曾经分别被访问 |
| 采样 | 每条轨迹每轮最多采 200 步；段末自举，未到训练重置上限则下一轮继续原状态 |
| 重置 | 成功后立即重置；未成功但累计采到 train_reset_horizon=1000 时外部截断并重置，terminated=false，保留末状态自举。预算用完仅停止实验 |
| Actor 自身输入 | 二维速度和二维绝对位置，共 4 维，无时间字段 |
| 实体输入 | 所有目标和同伴，每条 6 维：两种类型标记、二维相对位置、两维原生通信；当前静默机器人通信为零 |
| 信息限制 | 不含同伴速度，不使用历史帧或上一动作；全位置感知不等于完整全局状态 |
| 网络与规模 | 实体列表变长，共享编码、关系注意力与动作头的参数尺寸不变；实体有效 mask 不是动作 mask |
| Critic | 集中全局位置、速度及目标位置，6N 维，源规模下 24 维，无时间字段 |
| 评估 | 独立场景中冻结 Actor，默认观察到 1000 步；成功提前结束，未完成记为 evaluation_cutoff |

同一网络能读取可变实体数，只说明维度兼容。全名单的聚合是否保留决策信息、源经验是否约束目标行为，仍需独立证据；导航按 B 类经验外推环境解释。

任务终止、采样段结束、训练重置是三件事。1000 步训练重置用于更新初始场景、限制无边界空间中的长期漂移，不是任务完成期限，也不添加失败奖励。它与评估观察上限分别配置。[Gymnasium 的时间限制说明](https://gymnasium.farama.org/v0.26.3/tutorials/handling_time_limits/)同样区分真终止与需保留自举的外部截断。1000 是当前采样设计起点，不是已验证的最优值或收敛保证。

## 2. 一轮训练如何组织

1. 冻结本轮旧 Actor 和 Critic。
2. 从 16 条环境轨迹分别采集 200 步，共 3200 联合环境步。轨迹可跨轮延续；成功或累计采到 1000 步时重置，补齐该轨迹本轮剩余额度。重置前后的记录分成不同段。
3. 用冻结 Critic 计算标签，再训练 Critic。成功边界后续值为零，截断边界使用真实末状态价值。
4. AN/SA 学习方向，再拟合 Actor；DA 直接优化策略代理；MAPPO-E 使用作者 PPO 更新。
5. 开启的检查使用独立环境，检查和监控都不推进那 16 条训练轨迹。接受候选后提交 Actor。
6. 记录成本、监控和训练指标，保存 final 及 best；final 同时保存未完成轨迹状态及待重置标记，支持相同协议下的底层精确恢复。

`batch_episodes=16` 是沿用字段名，在本协议下表示 16 条采样通道，不表示每轮完成 16 个任务。一次场景采样可跨最多 5 个常规轮次；训练中成功任务的累计回报可能跨策略版本，只能作过程描述，不能当作冻结 Actor 的独立性能。外部重置不计为成功或任务失败。

AN/DA 的 Critic 标签是截断处自举的 n-step reward-to-go；策略标签默认 GAE。MAPPO-E 保留作者 GAE、优势标准化、ValueNorm、价值损失与 PPO 优化。适配层逐段计算目标后打包，GAE 不跨其他轨迹的重置或截断边界。这些训练流程并非完全相同。

检查版仍从独立重置场景采 200 步窗口。回报检查比较实际窗口奖励加冻结旧 Critic 的尾部估值，方向检查也使用近似标签；两者都是经验筛选。检查场景与持续训练轨迹的分布可能不同，不能宣称完整任务真实收益保证。

连续批次、自举标签及跨策略延续不直接满足主稿的独立完整回合、精确优势或无偏标签假设。它们是需要用实验检验的应用近似。

## 3. 配置与四组比较

配置按 `defaults.json → navigation.json → profile → --config → 命令参数` 合并。

| 文件 | 职责 |
|---|---|
| [configs/navigation.json](../configs/navigation.json) | 导航任务、输入、网络、优化和监控的共同参数 |
| [configs/navigation/no_checks.json](../configs/navigation/no_checks.json) | 仅关闭两个经验检查 |
| [configs/navigation/checked.json](../configs/navigation/checked.json) | 仅开启方法适用的经验检查 |
| [configs/profiles.json](../configs/profiles.json) | pilot/formal 的预算和种子默认值 |
| [底层 configs](configs/) | Runner 加载及旧套件兼容所需；不是当前实验的第二套用户配置 |

原 `configs/navigation_trial/` 四份大配置已由上述共同参数与两份开关配置替代。运行数据内保存的配置是溯源记录，不属于待清理模板。

| 实验组 | 方法 | 覆盖配置 | 方向检查 | 回报检查 |
|---|---|---|---|---|
| AN 核心版 | AN | no_checks.json | 关 | 关 |
| AN 完整版 | AN | checked.json | 开 | 开 |
| DA 完整版 | DA | checked.json | 不适用 | 开 |
| MAPPO-E | MAPPO-E | no_checks.json | 不适用 | 不适用 |

共同设置：种子 40、总源预算 200 万步、隐藏宽度 128、4 头和 2 层关系交互；Actor/Critic/方向学习率分别为 3e-4/1e-3/3e-4。方向与 Actor 拟合各 16 epoch，Critic 10 epoch，PPO 10 epoch。`minibatch_size=3200` 在适配层按机器人样本换算，当前对应 4 条长度 200 的轨迹为一个小批次；不能将它与联合环境步混淆。

每 65536 源步进行一次 8 场景监控，观察上限 1000 步；这组 200 万预算最多为监控预留 256000 步，包含在源预算内。最终 20 场景评价另计费。检查开启时每批 16 个窗口。样本量用于当前试跑，不代表正式评价已充分。

## 4. 当前训练命令

在包含 `manifold_project` 的目录运行，激活已经安装依赖的 Python 环境。以下均从头训练；v2 的无限延续异常运行保留作诊断，不混入本版方法比较。四组相互独立，单张 GPU 建议依次执行。

```powershell
$cfg = "manifold_project/experiments/configs/navigation"
$out = "manifold_project/experiments/results/navigation_success_v3_2m_s40"

# 先运行：AN 无检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/no_checks.json" --budget 2000000 --seeds 40 --device cuda --output "$out/an_core" --plot --plot-every 5

# 先运行：MAPPO-E
python -m manifold_project.experiments --experiments N-C --profile pilot --methods MAPPO-E --config "$cfg/no_checks.json" --budget 2000000 --seeds 40 --device cuda --output "$out/mappo" --plot --plot-every 5

# 检查开销对照：AN 双检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/checked.json" --budget 2000000 --seeds 40 --device cuda --output "$out/an_checked" --plot --plot-every 5

# 检查开销对照：DA 保留回报检查
python -m manifold_project.experiments --experiments N-C --profile pilot --methods DA --config "$cfg/checked.json" --budget 2000000 --seeds 40 --device cuda --output "$out/da_checked" --plot --plot-every 5
```

加 `--dry-run` 只检查配置。输出目录已有数据时不会覆盖，需要使用新目录；统一矩阵入口尚无自动续训开关。200 万步是观察点，不是已确认的收敛预算。所有方法在源环境完成设置选择后，再统一扩预算与增加种子。

`--evaluation-horizon 1000` 可显式指定观察长度，不改变任务终止或 Actor 输入。`--evaluation-episodes 500` 增加最终报告场景，不改变源监控的 8 场景；源监控数量要调整共同配置中的 source_eval_episodes。正式实验需重新确定预算和评价精度，不能只换成 formal 就认为证据已充分。

## 5. 冻结评价与任务可视化

源策略学好后，用 final Actor 做目标规模评价。以下示例接续上面的 AN 核心版，其他组替换方法、检查配置和源目录。

```powershell
python -m manifold_project.experiments --experiments N-T --profile pilot --methods AN --config "$cfg/no_checks.json" --source "$out/an_core" --evaluation-episodes 500 --evaluation-horizon 1000 --device cuda --output "$out/an_core_transfer" --plot
python -m manifold_project.experiments.visualize --checkpoint "$out/an_core/N-C_source_learning/full/AN_s40/checkpoints/final.pt" --agents 4 --seed 1000000 --output "$out/an_core/replay_n4.gif"
```

目标只执行 Actor，不运行目标 Critic、不训练、不按目标分数挑源模型。GIF 是单场景示例，不能替代成功率。N-I/N-R/N-Gd/N-Gr 等后续实验见总体方案；N-F 需要源训练提前启用固定方向诊断，当前关闭该诊断的运行不能事后凭空生成 N-F 数据。

## 6. 如何读日志和结果

| 内容 | 解释 |
|---|---|
| train_segment_return；兼容 train_J/train_discounted_return | 本轮采样段折扣回报均值，不是完整任务回报 |
| train_mean_reward、train_end_goal_distance、train_agent_radius_max | 真实步均奖励、各段末目标最近距离均值、段末机器人距原点最大值；用于发现漂移，不进入 Actor 输入 |
| train_task_age_max、train_reset_count、train_success_count | 本批轨迹的最大累计年龄、外部重置次数及成功次数。常规配置下年龄不得超过 1000 |
| critic_mse、explained_variance | 原奖励尺度的价值误差及解释方差；MAPPO 优化用的 ValueNorm 损失另见 author_stats，不能仅凭原尺度 MSE 判断归一化失效 |
| monitor_return、eval_success、monitor_budget_checkpoint | 最近一次独立源监控的回报、成功率和对应节点；不是每轮重新评价 |
| events.jsonl 的 train_segment | 每条轨迹的段起止步、长度、奖励和结束原因 |
| events.jsonl 的 completed_task | 成功训练任务累计步数与回报，可能跨策略版本 |
| events.jsonl 的 training_reset | 外部采样上限引起的重置，单列于真正完成任务；末状态用于 bootstrap |
| evaluation_episodes.csv | 冻结策略的逐场景观测回报、成功或截尾、观察上限、覆盖及碰撞 |
| summary.csv | 按训练种子汇总，再计算种子间统计；按协议、任务、观察上限和回报定义分组 |
| final.pt 与 best.pt | final 是主结果及恢复状态，best 是源监控选出的辅助 Actor |
| overview.png | 总图覆盖更新，不按每轮新增图片 |

单步团队奖励取原生 agent 奖励均值；`discounted_return` 是 gamma=0.99 的折扣和，`return_undiscounted` 是实际奖励总和，`mean_reward` 是步均奖励。独立评价回报绝不加入 Critic 的预测。

成功率指观察上限内首次完成的比例，同时报告 200/500/1000 步成功率、成功样本平均完成步数、未完成比例和截尾平均步数。截尾平均步数给未完成样本记观察上限，不等于估计它们真实的最终完成时间。有限观察不能证明永不成功。

训练、方向检查、旧/候选回报检查、源监控均按实际联合环境步计入预算；最终报告评价和额外机制诊断单列。同一轨迹的多个段不是独立训练种子。回报提高也不自动等于任务成功，应同时看距离、覆盖、碰撞和成功率。

## 7. 代码位置与历史兼容

| 模块 | 职责 |
|---|---|
| current_protocol.py | 统一配置映射、监控预留、训练记录汇总 |
| envs/navigation.py | 原生动力学与奖励、成功判定、全局状态 |
| training/collector.py | 保留轨迹、成功或采样上限重置、分段采样、成本及采样状态恢复 |
| training/ppo.py | 截断自举、GAE、标签、样本权重及方向损失项 |
| training/runner.py | 各方法更新、检查、监控、模型与采样状态保存 |
| training/author_ppo.py | 保留作者更新，适配逐段价值目标与批次 |
| models/__init__.py、models/entities.py、models/deployment.py | 训练输入解码、可变实体网络与冻结部署 |
| ../applications/evaluation.py | 独立场景观察、截尾记录、冻结目标评价 |

归档 `legacy_experiments_20260923.zip` 的默认配置是 continuing、最近 2 个同伴/目标、8 帧历史，代码另支持限时首达；其 episode 调用会重新建环境。`results/navigation_trial/` 四组采用全实体、时间输入和 200 步硬期限。v2 改为无时钟、成功终止，但只在成功后重置，已在 AN/MAPPO 数据中出现严重漂移；详见[异常诊断](../results/navigation_success_v2_2m_s40/异常诊断.md)。v3 保留无时钟和成功终止，增加有自举的外部训练重置，不能将 v2 结果改名冒充 v3。

统一字段含义、成本和分组规则，不重写历史 manifest、CSV 或 checkpoint，也不把旧 deadline 改名为新 cutoff。旧价值标签已经不同，改名不能修复。旧数据和报告保留为历史证据，索引见[结果目录](../results/README.md)。

工程检查覆盖终止与截断、无时间输入、逐段 GAE、跨轮连续性、CPU/CUDA 短训练、精确恢复和冻结变规模执行；通过测试不代表性能结论。运行验证的方法见实验运行说明。
