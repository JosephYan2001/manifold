# Actor 拟合最小干预实验

2026-10-17。最佳候选选择已接入正式 Runner，诊断复用同一个拟合函数；固定目标保存、重放和显式方向更新预算已接入原 `diagnose` 入口。下一步先做下面两组固定基点实验，不重跑完整训练。

## 已完成的实现与验收

正式训练配置 `actor_fit_selection` 支持 `last` 和 `best_train_kl`，默认仍为 `last`。后者在完整训练批次上，以归一化 FP64 的 KL(目标∥候选) 比较 epoch 0 至最后一个 epoch；全部优化完成后才提交所选参数。若最小值相对起点的改善不超过 `actor_fit_selection_atol + actor_fit_selection_rtol * initial_KL`，选择旧 Actor；默认容差分别为 1e-12 和 1e-6。

验证集和执行回报不参与拟合候选选择。原有回报检查是否启用仍由原开关决定。零更新单独记录为 `actor_changed=false`，并保留全部采样、优化成本；no_checks 不会因此额外尝试其他 η。Actor 的优化损失、Adam 初始化、梯度裁剪和优化预算保持原定义，整批候选检查会增加计算时间。

公共训练入口可读取 [actor_bestfit.json](../configs/navigation/diagnosis/actor_bestfit.json) 或 [actor_bestfit_lr3e5.json](../configs/navigation/diagnosis/actor_bestfit_lr3e5.json)。它们供后续短训练使用；当前固定目标实验使用下方 diagnose 命令。此选择规则作用于应用 Runner 的 AN/SA 目标拟合，不改变有限表策略、DA 或 PPO 的优化规则。公共 `--resume` 仍继承原配置，不允许用恢复操作悄悄切换选择规则。

CPU 工程验收 17/17 通过，包含中间最佳、零更新、实际参数提交、全预算计费、旧优化轨迹一致、固定目标重放以及原有恢复/梯度隔离检查。证据见[验收记录](../results/checks/actor_fit_acceptance_20261017_v1/验收报告.md)。真实 A2、A0 checkpoint 的下列命令已通过 dry-run；尚未执行这些完整 CUDA 诊断，因此没有新的性能结论。

## 先运行这两组

两组都生成两条拟合轨迹，每条各取最终和最佳结果，形成四组候选；额外评价旧 Actor 和原始教师目标。不匹配或缩放实际 KL，不调整 η。方向在每个数据种子下只训练一次，随后两档学习率共享同一目标、数据和小批次顺序。

| 运行 | 用途 | 方向预算 | 每档 Actor 预算 |
|---|---|---|---|
| A2、数据种子 41 | 已知失败设置的回归检查 | N64、U512，16 epoch | 16 epoch、512 次更新 |
| A0、数据种子 40 | 本轮主要新增对照 | N64、U128，4 epoch | 16 epoch、512 次更新 |

在 PowerShell 中先执行路径设置，再顺序运行。这里使用已经完成 CPU 验收的 Python 环境；输出目录必须不存在。

```powershell
Set-Location D:\manifold
$py = 'D:\python_exercise_program\dqn+pso\.conda\python.exe'
$src = 'manifold_project/experiments/results/evidence/navigation/navigation_diagnosis_20261012_4m_s40_v1'
$out = 'manifold_project/experiments/results/evidence/navigation/actor_fit_e1_20261017_v1'

# A2：回归检查，确认旧 Actor 或更好中间候选能被保留
& $py -m manifold_project.experiments.cooperative_navigation.diagnose --mode fit --checkpoint "$src/an_batch64/N-C_source_learning/full/AN_s40/checkpoints/final.pt" --output "$out/a2_s41_u512" --seeds 41 --init-seed 40 --batch-episodes 64 --validation-episodes 32 --evaluation-episodes 40 --evaluation-seed 20261017 --direction-updates 512 --fit-epochs 16 --actor-lrs 0.0003 0.00003 --fit-trace-every-updates 1 --device cuda --allow-duplicate-openmp

# A0：主要对照，显式使用 U128，不能沿用 checkpoint 默认的 U512
& $py -m manifold_project.experiments.cooperative_navigation.diagnose --mode fit --checkpoint "$src/an_base/N-C_source_learning/full/AN_s40/checkpoints/final.pt" --output "$out/a0_s40_u128" --seeds 40 --init-seed 40 --batch-episodes 64 --validation-episodes 32 --evaluation-episodes 40 --evaluation-seed 20261017 --direction-updates 128 --fit-epochs 16 --actor-lrs 0.0003 0.00003 --fit-trace-every-updates 1 --device cuda --allow-duplicate-openmp
```

加 `--dry-run` 只打印计划，不采样、不创建结果。`--direction-updates` 与 `--direction-epochs` 二选一；U 必须能被 `ceil(N / minibatch_episodes)` 整除，fit 模式一次只指定一个方向预算。

`--evaluation-seed 20261017` 明确指定一组新的评价场景；同一实验单元内所有候选使用配对场景与动作随机数。它不是训练种子。每组最多新增 278400 个环境步，其中固定数据最多 38400 步，评价最多 240000 步；完全相同的候选复用评价并单独标记，实际成本以 manifest 为准。这些诊断成本与原 4M 训练分开记录。

## 固定目标如何保存和重放

上述命令都会在 Actor 拟合开始前自动保存：

```text
a2_s41_u512/fixed_targets/data_s41_direction_u512.pt
a0_s40_u128/fixed_targets/data_s40_direction_u128.pt
```

文件包含旧 Actor/Critic、方向、训练与验证批次、标签、旧概率、q、目标概率、有效位和历史、η、配置与模型代码指纹、随机状态和完整的 Actor 小批次顺序。源 checkpoint 不被修改。没有历史完整快照的旧诊断不能据此冒充历史批次重放。

如果只想先生成目标，在上方某条命令末尾增加 `--prepare-only`：它会保存目标并结束，不拟合 Actor，也不执行回报评价。之后运行下面的重放命令。若原拟合中断，但固定目标文件已完整保存，也可以用它重新执行两条拟合轨迹。

```powershell
& $py -m manifold_project.experiments.cooperative_navigation.diagnose --mode fit --fixed-target "$out/a0_s40_u128/fixed_targets/data_s40_direction_u128.pt" --output "$out/a0_s40_u128_replay" --actor-lrs 0.0003 0.00003 --fit-trace-every-updates 1 --device cuda --allow-duplicate-openmp
```

重放继承固定数据、方向、η、初始化与小批次顺序，不需要原 checkpoint，也不重新采样训练/验证数据或生成标签。不能再传 `--seeds`、`--batch-episodes`、`--direction-updates`、`--eta` 等改变固定问题的参数。Actor 预算必须与保存值一致；回报评价仍会运行并计费。重放从旧 Actor 重新开始拟合，不是从中断的优化步恢复；原运行已发生的成本仍保留在原目录，不能把它抹去。

## 看哪些文件

| 文件 | 重点 |
|---|---|
| `manifest.json` | 来源、方向预算、实际 Actor 更新数、固定目标哈希、采样与评价成本 |
| `fit_trace.csv` | 初始、逐步和逐 epoch 的训练/验证目标 KL、旧→候选实际 KL；best 只在 epoch 边界选择 |
| `diagnostics.csv` | 教师、两档学习率的最终/最佳候选、选择 epoch、是否零更新、独立验证 KL、原始实际 KL 和回报 |
| `fit_candidates/*.pt` | 每条轨迹的最终与最佳完整参数、选择历史、小批次顺序；旧 Actor 位于对应固定目标文件 |

相同参数候选的执行评价复用记录通过 `evaluation_reused_from` 标注；不要把 CSV 中复用的行再次加到环境成本中。

先确认非零候选在独立验证数据上的拟合也改善，再补少量未用于选择配置的数据种子。确认阶段预先固定配置，并另用一组评价场景，例如 `--evaluation-seed 20261018`。只选择旧 Actor 避免退化，不算已经解决学习问题；是否转化为持续回报改善，要由后续短训练回答。
