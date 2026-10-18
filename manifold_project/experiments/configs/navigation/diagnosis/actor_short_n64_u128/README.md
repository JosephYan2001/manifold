# Actor 四格短训练运行命令

2026-10-18：固定教师收益复核已结束，本轮观察 Actor 学习率和候选选择能否影响连续学习。四组从相同随机初始化开始，训练种子均为 40；不加载历史 4M checkpoint，也不依赖固定目标大文件。这回答早期学习问题，不等同于 4M 状态附近的分支续训。

| 组别 | Actor 学习率 | 候选选择 | 配置 |
|---|---:|---|---|
| A | 3e-4 | 最终候选 | [a_lr3e4_last.json](a_lr3e4_last.json) |
| B | 3e-5 | 最终候选 | [b_lr3e5_last.json](b_lr3e5_last.json) |
| C | 3e-4 | 最佳训练 KL，含零更新 | [c_lr3e4_best.json](c_lr3e4_best.json) |
| D | 3e-5 | 最佳训练 KL，含零更新 | [d_lr3e5_best.json](d_lr3e5_best.json) |

四组共用 N64/U128：每轮 64 回合，每个小批次 2 回合；方向 4 epoch × 32 小批次 = 128 次优化，Actor 16 epoch × 32 小批次 = 512 次优化。最佳选择也跑完相同拟合预算，最后提交所选候选。方向检查和回报检查均关闭，不开启额外固定目标采样。环境、8 帧历史、400 步训练上限、网络、Critic、GAE 和首个步幅 η=0.1 继承当前 navigation/pilot 设置。

同一轮的随机数生成规则相同，但策略在后续轮次分化后，实际训练轨迹会不同；这已经是连续学习实验，不能再要求四组一直面对同一张固定目标表。

## 命令

在包含 `manifold_project` 的根目录执行，先同步本目录的四份配置。单张 GPU 依次运行四条。训练机器历史路径为 `D:\manifold\manifold`，本工作区为 `D:\manifold`；以下使用相对路径，不强行改变当前目录。

```powershell
$cfg = "manifold_project/experiments/configs/navigation/diagnosis/actor_short_n64_u128"
$out = "manifold_project/experiments/results/evidence/navigation/actor_fit_short_n64_u128_500k_s40_20261018_v1"

# A：3e-4，提交最后候选。
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/a_lr3e4_last.json" --budget 500000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/a_lr3e4_last" --plot --plot-every 5 --allow-duplicate-openmp

# B：3e-5，提交最后候选。
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/b_lr3e5_last.json" --budget 500000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/b_lr3e5_last" --plot --plot-every 5 --allow-duplicate-openmp

# C：3e-4，提交最佳训练 KL 候选，包含零更新。
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/c_lr3e4_best.json" --budget 500000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/c_lr3e4_best" --plot --plot-every 5 --allow-duplicate-openmp

# D：3e-5，提交最佳训练 KL 候选，包含零更新。
python -m manifold_project.experiments --experiments N-C --profile pilot --methods AN --config "$cfg/d_lr3e5_best.json" --budget 500000 --seeds 40 --evaluation-episodes 100 --device cuda --output "$out/d_lr3e5_best" --plot --plot-every 5 --allow-duplicate-openmp
```

四条命令已通过公共入口 dry-run，并核对映射到 Runner 后的方向/Actor 更新预算；解析后的配置仅 `actor_lr` 和 `actor_fit_selection` 不同。未启动训练。任一命令末尾可加 `--dry-run` 再检查本机配置，它不创建训练结果，也不验证 CUDA 实际执行。

## 预算与读取结果

每组源预算上限 500,000 步，包含训练和源监控。源监控保留每 65,536 步、每次 8 回合的现有规则，加起点和终点共 9 个监控节点，最坏预留 72,000 步。若训练和评价回合均跑满，约完成 16 轮训练；这轮用于观察早期进展，不能凭未收敛就断定方法无效。

四组终点评价使用同一组 100 个场景，重置种子从 20261020 起，单回合上限 1,000 步，只评价源人数 4。终点评价另计，每组最多 100,000 步；四组源预算合计最多 2,000,000 步，含终点评价总上限为 **2,400,000 步**。成功提前结束时记实际步数，不按中途结果追加预算或挑选终点评价模型。

主要读取每组 `N-C_source_learning/full/AN_s40/` 下的：

- `training.csv` 和 `events.jsonl`：独立源监控、实际策略 KL、拟合起点/最佳/最终/所选 KL、所选 epoch 和零更新标记。
- `summary.json`：`actor_changed_rounds`、`selected_zero_update_rounds`、优化次数与总成本。
- `final_evaluation/evaluation_episodes.csv`：固定 final Actor 的 100 回合结果，四组按场景配对比较。

单个训练种子用于探索，不作正式跨种子优越性结论。观察非零更新是否带来任务回报或成功率进展，不能只看训练 KL 或是否避免下降。

如果某组中断，保留该组目录，使用原公共恢复入口。例如恢复 D 组：

```powershell
python -m manifold_project.experiments --resume "$out/d_lr3e5_best" --allow-duplicate-openmp
```

恢复继承该组原配置，不用更改配置的方式恢复另一变体。首次启动失败且尚无 checkpoint 时，先排查错误，再使用新输出目录。
