# 共同预算与检查消融

P-L、T-L，MC学习标签，种子40–49，源联合步数上限262144，为此前65536的四倍。所有未列参数保持formal有限环境配置。检查未关闭时每批512回合。配置不会自动开始训练。

| 配置 | 方向检查 | 回报检查 | 方法 |
|---|---|---|---|
| full | 开 | 开 | AN SA DA |
| no_direction | 关 | 开 | AN SA |
| no_return | 开 | 关 | AN SA DA |
| no_checks | 关 | 关 | AN SA |

DA本身没有方向检查。因此no_direction中的DA等同full，no_checks中的DA等同no_return，直接引用对应组，避免重复训练。结果汇总时需明确这两处基线复用，不能将其误读为缺失方法。

```powershell
$exp = "manifold_project/experiments"
python -m manifold_project.experiments --experiments P-L T-L --profile formal --methods AN SA DA --config "$exp/configs/finite_budget_study/full.json" --output "$exp/results/finite_budget_study/full_262k" --plot
python -m manifold_project.experiments --experiments P-L T-L --profile formal --methods AN SA --config "$exp/configs/finite_budget_study/no_direction.json" --output "$exp/results/finite_budget_study/no_direction_262k" --plot
python -m manifold_project.experiments --experiments P-L T-L --profile formal --methods AN SA DA --config "$exp/configs/finite_budget_study/no_return.json" --output "$exp/results/finite_budget_study/no_return_262k" --plot
python -m manifold_project.experiments --experiments P-L T-L --profile formal --methods AN SA --config "$exp/configs/finite_budget_study/no_checks.json" --output "$exp/results/finite_budget_study/no_checks_262k" --plot
```

在直接包含manifold_project的目录使用此前Python环境执行。建议顺序执行，避免同时占用资源。统一入口不自动续训，这些是从相同初始化重跑，不是从旧65k检查点接续。目录非空时停止，勿覆盖原结果。

比较顺序：full与旧65k结果观察预算效应；同262k预算比较full/no_direction、full/no_return分别识别模块影响；no_checks检验无门控核心路线。关闭检查会释放采样预算并增加更新机会，这是端到端效率对照的一部分，不强行匹配轮次数。所有优化超参数和监控设置保持一致。

无检查时仍学习方向、拟合Actor，按预设第一档步长执行；没有接受门控保证。只关闭检查，不关闭用于记录的源监控。保留回报检查的组仍沿用当前经验平局规则，本次不同时更改该规则。单项消融减少检查成本，但不等于已设计更高效的自适应检查算法。

分析实际总源步数下的回报曲线、最终回报、优化数据/检查/监控开销、接受率、退化幅度、各种子离散程度。不同组轮数不同，不能按轮次横轴比较样本效率。不用目标迁移成绩选择配置；冻结迁移只作报告。预算增大不保证已收敛，旧65k预算下的效率结论保留。
