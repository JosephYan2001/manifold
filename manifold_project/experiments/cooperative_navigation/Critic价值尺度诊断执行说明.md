# Critic 价值尺度诊断执行说明

2026-10-22。**正式诊断已完成，见[结果分析](../results/evidence/navigation/critic_scale_probe_20261022_v1/analysis_20261022/Critic尺度诊断结果分析.md)。** 归一化改善新 Critic 的训练拟合和饱和，但独立回合预测变差；已有饱和模型未恢复。两个起点均未通过预定筛选，本轮结束，不推进完整训练。下文保留执行前实验定义与命令，已有输出目录不得覆盖。本轮没有连续策略训练或任务回报评价。

## 实验对象与四个分支

冻结新六组中的 `n16_fixed/final.pt` Actor 和旧 Critic，使用数据种子 45 采集 32 个训练回合、32 个独立验证回合，每回合最多 400 个联合环境步。全部分支共享这 64 个回合，新增交互最多 **25,600 步**。

价值目标按现有 AN 的计算定义生成：原始折扣 reward-to-go，采样截断处用冻结旧 Critic 自举，成功终止处为零；拟合期间不重新计算目标。验证集来自同一个冻结策略与源重置分布，使用独立的采样用途随机种子。

| 分支 | Critic 起点 | 拟合的目标 | 回答的问题 |
|---|---|---|---|
| fresh_raw | 新初始化的 Critic | 原始价值目标 | 在这份数据上，现有优化能否重新学出状态差异？ |
| fresh_fixed_normalized | 同一个新初始化 Critic | 固定归一化目标 | 归一化能否改善重新拟合、避免进入饱和？ |
| checkpoint_raw | checkpoint 中已饱和的 Critic | 原始价值目标 | 现有拟合能否使它恢复？ |
| checkpoint_fixed_normalized | 同一个 checkpoint Critic | 固定归一化目标 | 已经饱和后，归一化还能否救回它？ |

这是同一批数据上的四条 Critic 拟合轨迹，不是四次环境训练。fresh 只重置待拟合 Critic，Actor、采样策略、标签来源均不变；不能把它解释成从随机策略开始的完整训练。

主比较是每种起点内 normalized 对 raw。不得将 fresh_normalized 对 checkpoint_raw 的差别全部归因于归一化；二者还改变了初始化。

## 冻结条件和预算

- 网络沿用原来的两层 tanh Critic，不增加正交初始化、输入归一化、Huber、价值裁剪或网络宽度。
- 归一化均值 m、标准差 s 仅从训练价值目标按现有折扣权重计算，标准差下限固定为 1e-3。两者整条拟合轨迹保持不变，验证集不参与计算。
- normalized 分支拟合 `(target-m)/s`，评价使用反变换后的 `V=s*z+m`。初始化时将最后线性层变为 `W/s`、`(b-m)/s`，使同一起点两分支的初始实际价值预测在声明容差内一致；不是把初始预测直接搬到目标均值附近。
- 每分支均使用新 Adam，**不继承源 Critic 的优化器动量**。它是受控重新拟合，不是训练恢复。归一化包含输出参数化改变，不等同于简单把整个 loss 除以一个常数。
- 学习率继承该 checkpoint 的 1e-3，value_coef=0.5、全局梯度裁剪 0.5、每小批 2 个完整回合。共享相同的 80 epoch 回合排列，各分支 **1,280 次优化**，四分支合计 **5,120 次**；Actor 和方向优化次数均为零。
- 在 epoch 0 和每个 epoch 结束记录完整训练/验证指标，完整跑完预算，以第 80 epoch 为主要终点。不早停，不用验证集或任务回报挑选候选。

这只是一次固定尺度干预，不是 MAPPO 的动态 ValueNorm 完整移植。固定自举目标也不是真实 V 的独立真值；改善拟合不能直接证明优势标签或任务回报改善。

## 运行命令

在已安装项目训练依赖的 Python 环境中执行。命令不使用 `$py`、`$src` 或 `$out`，避免未定义变量导致路径从根目录开始。

```powershell
Set-Location D:\manifold
python -m manifold_project.experiments.cooperative_navigation.diagnose --mode critic --checkpoint "manifold_project/experiments/results/evidence/navigation/navigation_batch_lr_2m_s40_20261019_v1/n16_fixed/N-C_source_learning/full/AN_s40/checkpoints/final.pt" --output "manifold_project/experiments/results/evidence/navigation/critic_scale_probe_20261022_v1" --seeds 45 --init-seed 40 --batch-episodes 32 --validation-episodes 32 --critic-epochs 80 --critic-starts fresh checkpoint --device cuda --allow-duplicate-openmp
```

这一条命令完成一次采集、保存固定包和四个分支拟合。可先在末尾加 `--dry-run` 核对路径、预算与参数；dry-run 不创建输出目录。已有输出目录会拒绝覆盖。

`--init-seed 40` 控制本实验 fresh Critic 的独立初始化，并不宣称复现历史完整训练的初始 Critic 参数。成对比较使用同一份保存的初始参数。

若希望只保存数据后停下，在主命令末尾加 `--prepare-only`。如果拟合阶段中断、或需核对复现，可以从固定包重放，使用新输出目录：

```powershell
Set-Location D:\manifold
python -m manifold_project.experiments.cooperative_navigation.diagnose --mode critic --critic-data "manifold_project/experiments/results/evidence/navigation/critic_scale_probe_20261022_v1/critic_data.pt" --output "manifold_project/experiments/results/evidence/navigation/critic_scale_probe_20261022_replay_v1" --device cuda --allow-duplicate-openmp
```

重放不构造 Actor、不采样、不重算目标，继承固定包中的初始化、归一化统计、小批次顺序与全部优化预算。它会重新完成四条拟合轨迹，不是从已完成 epoch 继续；新增拟合计算仍单列计费。只有数据包已经成功保存后才能重放；采集期间失败的实际与预留成本保留在原目录账本中，不能把失败运行删除后当作零成本。

## 看哪些结果和如何结束

| 文件 | 用途 |
|---|---|
| `summary.json` | 每个起点内 normalized 相对 raw 的验证 MSE 降幅、EV 增量及四个终点指标 |
| `diagnostics.csv` | 四条轨迹最终的原始量纲 MSE、解释方差、预测/目标方差与饱和率 |
| `critic_trace.csv` | epoch 0–80 的整批拟合轨迹、梯度范数与裁剪比例 |
| `episode_errors.csv` | 终点逐回合原始量纲误差，保留配对数据，不把各时间步当独立样本 |
| `critic_data.pt` | 紧凑固定包：状态、原始目标、冻结旧价值/GAE、标签 Critic、两个初始 Critic、顺序与哈希；不保存大体积 Actor 历史输入 |
| `candidates/*.pt` | 四个最终 Critic 及反归一化信息 |
| `manifest.json`、`collection_ledger.jsonl` | 源检查点、代码与数据指纹、实验状态、采样成本及优化次数 |

主要看**独立验证误差是否下降**，同时看 EV 和预测方差；仅从常数变成变化很大的错误输出不算改善。辅助看第二层 tanh 饱和和梯度裁剪，验证推测的机制是否伴随变化。

预先设定描述性筛选线：同一起点下 normalized 的最终验证 MSE 比 raw 至少低 10%，验证 EV 至少高 0.05。阈值用于限制后续投入，不是统计显著性、算法成功标准或继续追加样本的理由。全程只有一个数据单元和一个 fresh 初始化，不据此声称跨种子稳定。

| 结果 | 判断与行动 |
|---|---|
| fresh_raw 出现失效，fresh_normalized 改善验证拟合并减轻饱和 | 支持尺度处理影响拟合过程；才考虑一个受控短训练，检验能否改善标签、教师和回报 |
| fresh_normalized 改善，checkpoint 两分支仍差 | 说明重新学习与救回已饱和网络不同；不直接对旧模型加归一化后续训 |
| raw 本身已能拟合，normalized 没有增益 | 这份固定数据没有支持尺度是主因；不据此添加归一化主训练补丁 |
| 只改善训练误差，验证没有改善；或两种起点均无明确改善 | 本轮结束，不自动追加 epoch、种子或归一化组合，不推进 2M 重训 |

即使筛选通过，也不能把“Critic 能拟合固定自举目标”改写为“AN 已能追上 MAPPO”。如果后续价值指标改善而教师/回报没有增量，应否定 Critic 足以解释全部差距的判断。

## 工程验证

新增 3 项 CPU 测试通过：输出尺度变换不改变初始实际预测且统计排除 padding；raw 分支与生产 Critic 优化路径逐参数一致；四分支共享固定包、源文件不变、无方向/Actor 更新、零采样重放得到相同参数且拒绝篡改/覆盖。原方向/Actor 拟合和教师评价的 11 项相关回归测试通过。正式 CUDA 拟合已完成；固定包、实际候选、目标与成本复核见上方结果分析，工程通过不代表研究筛选通过。

```powershell
python -m unittest manifold_project.experiments.tests.test_critic_probe manifold_project.experiments.tests.test_navigation_diagnose manifold_project.experiments.tests.test_actor_fit_selection manifold_project.experiments.tests.test_teacher_evaluation -v
```

## 长回报与 λ-return 的下一项对照

归一化诊断未通过后，用已有固定包单独比较价值目标。`diagnose --mode critic-target` 已接入，正式目标对照尚未运行。A 组直接加载原 `fresh_fixed_normalized` 最终模型；只新增 B 组拟合，不重跑 A、不采样。源目录必须包含已完成的 `manifest.json`、`diagnostics.csv`、`critic_data.pt` 和 `candidates/fresh_fixed_normalized.pt`。

B 使用固定包保存的 `frozen_old_values + frozen_gae`，继承 λ=0.95、fresh 初始参数、80 epoch/1,280 次更新、小批次顺序、新 Adam、学习率和裁剪。**归一化仍使用原长回报训练均值 −163.9381 和标准差 33.8661**，不为 λ-return 重新计算尺度。Actor 与方向均不更新。

```powershell
Set-Location D:\manifold
python -m manifold_project.experiments.cooperative_navigation.diagnose --mode critic-target --critic-data "manifold_project/experiments/results/evidence/navigation/critic_scale_probe_20261022_v1/critic_data.pt" --output "manifold_project/experiments/results/evidence/navigation/critic_target_probe_20261022_v1" --device cuda --allow-duplicate-openmp
```

在末尾加 `--dry-run` 可核对来源、对照模型、数据指纹和继承预算，不创建输出目录。不要额外传 `--seeds`、`--critic-epochs` 或初始化参数；入口会拒绝改变冻结条件。中断后使用新输出目录重跑，仍为零采样，但新增拟合成本单列；本入口不提供逐 epoch 中断恢复。

主要终点是第 80 epoch 的 **`long_validation_mse` 和 `long_validation_explained_variance`**，A/B 均面对同一个已保存的长回报验证目标。`lambda_validation_*` 仅说明 λ-return 自身是否容易拟合，不决定实验成功。

预定描述性筛选：B 的共同长回报验证 MSE 至少比训练均值常数参照 930.22 低 10%（约低于 837.20），且 EV>0.05。只优于较差的 A 组不足以通过；只在自己的 λ-return 上表现好也不足以通过。结果见 `summary.json`、`diagnostics.csv`、`critic_trace.csv`、`episode_errors.csv` 和 `manifest.json`，实际新增交互为零、Critic 更新为 1,280 次。

这批验证数据已经查看过，本轮仅用于探索机制；不把它当作新的独立确认。共同长回报仍含冻结旧 Critic 的自举误差，也不是真实价值。通过后才考虑封顶短训练与新评价场景；未通过不继续扩大归一化、epoch 或网络组合。
