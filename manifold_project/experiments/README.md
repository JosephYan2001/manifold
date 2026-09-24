# 实验模块：在旧版代码上修订

当前代码以 `archive/legacy_experiments_20260923.zip` 中的成对协作和协作导航模块为主体。复用原训练器、采样器、检查点、作者 PPO、监控和回放工具，按[总体实验方案](../docs/总体实验方案.md)修改不一致之处。已清理停用套件、重复实现和历史测试；旧结果不作为当前证据。

此前重写框架已留存到 `archive/rejected_framework_20260924.zip`，不参与运行。原始归档未改。这里保留的公共入口只编排实验、转换配置和汇总报告，**不再实现另一套应用训练循环**。

## 1. 代码说明与运行入口

各目录职责、关键文件/函数和完整调用关系统一见[项目结构与代码说明](../docs/项目结构与代码说明.md)。本文件维护安装、命令、配置使用和输出，避免重复维护结构清单。

当前入口为 `python -m manifold_project.experiments`。P-L 调用原成对训练器；导航、仓库和实体成对 P-TB 调用原导航 Runner；有限固定基点诊断和 T-L 由 `finite/` 组织。原文件的保留、移动、合并和移除记录见 [reuse_audit.csv](reuse_audit.csv)。

P-I 当前只完成精确输入映射与损失分解，同数据学习对照仍待补充。各入口可执行与工程检查通过，不等于正式预算训练已完成。

## 2. 运行准备

在仓库根目录 `D:\manifold` 运行。已有 PyTorch/CUDA 环境可直接使用；本机项目虚拟环境也已配置：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r manifold_project/experiments/requirements.txt
```

保留旧版明确采用的 `KMP_DUPLICATE_LIB_OK=TRUE` 兼容设置，并在导入计算库前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；调用者显式设置的值优先。运行清单记录实际设置。OpenMP 开关是沿用的兼容措施，不代表重复运行库已根治。

MPE2 固定 1.1.1，RWARE 固定 2.0.0。导航和仓库 Actor 输出保持原生离散动作语义；跨人数评价只加载 Actor，不创建目标 Critic。

## 3. 当前方案的命令

统一入口与环境入口最终调用上述旧训练器。先用 `--dry-run` 检查解析后的配置：

```powershell
python -m manifold_project.experiments --experiments P-E P-A P-T --profile pilot --dry-run
python -m manifold_project.experiments --experiments P-E P-A P-T --profile pilot --output manifold_project/experiments/results/pair_mechanisms_pilot_v1 --plot
```

然后是成对多轮、输入边界和两步模型：

补充机制链实验 `P-EA` 直接读取已完成 P-E 的方向系数，分别交给完整和受限 Actor 拟合，不重新学习方向。推荐先用 MC 标签、2048 回合、AN/SA 和 20 个 P-E 数据种子：

```powershell
python -m manifold_project.experiments --experiments P-EA --profile formal --source manifold_project/experiments/results/pair_estimation_20seeds_v1 --config manifold_project/experiments/configs/pair_ea_mc.json --sample-sizes 2048 --output manifold_project/experiments/results/pair_ea_mc2048_20seeds_v1 --plot
```

这里 `formal` 提供默认数据种子 1000–1019；不执行长期训练，也不重新运行 P-E。使用 `--data-seeds` 筛选源方向，不能使用 `--seeds`。默认拟合步数为 1、10、100、1000，共 320 条记录。去掉 `--sample-sizes 2048` 可覆盖默认全部四档样本量；去掉 MC 配置参数可同时覆盖精确与 MC 标签。请求的每条方向必须已经存在，否则报错，不会悄悄补训。

P-EA 核对源 manifest 完成状态、协议与环境参数，并重建源采样批次核对包含基础策略的哈希，再核对方向误差定义。输出保存源 manifest/CSV/代码指纹以及每条方向的数据种子、方法、标签、样本量。旧源码指纹可以与新实现不同，但重建哈希或方向语义核对不通过时停止。

重点对照 `source_direction_error → ideal_return_gain → direction_realization_residual → actual_return_gain`，并记录实际 `old_new_kl`。不同方法沿用共同 eta，并非实际 KL 匹配实验。拟合使用重建的同一源经验，不增加训练环境样本；`additional_training_environment_steps=0`。`source_steps_total` 沿用 P-A 的“源批次加验证批次”规模口径，不是新增交互成本；重放和报告验证另有显式字段。各拟合步数不是独立重复。

```powershell
python -m manifold_project.experiments --experiments P-I P-B P-L T-V T-L --profile pilot --output manifold_project/experiments/results/finite_learning_pilot_v1 --plot
python -m manifold_project.experiments --experiments P-TB --profile pilot --device cuda --output manifold_project/experiments/results/pair_entities_pilot_v1 --plot
```

导航源训练与冻结迁移：

```powershell
python -m manifold_project.experiments --experiments N-C --profile pilot --device cuda --output manifold_project/experiments/results/nav_source_pilot_v1 --plot --plot-every 10
python -m manifold_project.experiments --experiments N-T N-F N-D --profile pilot --source manifold_project/experiments/results/nav_source_pilot_v1 --device cuda --output manifold_project/experiments/results/nav_frozen_pilot_v1 --plot
```

输入、关系及单检查消融：

```powershell
python -m manifold_project.experiments --experiments N-I --profile pilot --source manifold_project/experiments/results/nav_source_pilot_v1 --device cuda --output manifold_project/experiments/results/nav_inputs_pilot_v1 --plot
python -m manifold_project.experiments --experiments N-R N-Gd N-Gr --profile pilot --device cuda --output manifold_project/experiments/results/nav_ablations_pilot_v1 --plot
```

N-I 复用同种子、同训练配置的 N-C 完整输入模型，仅重训摘要和最近两项；输入与关系消融同时做冻结目标评价。N-F 从源训练预定位置的固定方向诊断读取数据，不通过目标结果选择模型。

仓库复用同一个 Runner：

```powershell
python -m manifold_project.experiments --experiments W-C --profile pilot --device cuda --output manifold_project/experiments/results/warehouse_source_pilot_v1 --plot
python -m manifold_project.experiments --experiments W-T W-L W-F --profile pilot --source manifold_project/experiments/results/warehouse_source_pilot_v1 --device cuda --output manifold_project/experiments/results/warehouse_frozen_pilot_v1 --plot
```

`--methods AN SA DA MAPPO-E IPPO-E` 可以筛选本次方法，`--seeds 40` 指定源种子，`--budget 20000000` 指定每方法/种子/标签组的源联合步数。P-E/P-A/P-T 等固定基点诊断由样本档、拟合档和独立批次数控制。2M pilot 不意味着已收敛；是否提高共同预算只看源曲线。

`smoke` 只用于缩小期限、模型和预算的工程验证。`formal` 默认应用 20M、种子 100–104；在确认源设置后再运行。没有执行正式预算实验，也没有据此得出方法优越性。

## 4. 配置与旧入口

当前矩阵配置为 `configs/defaults.json → 环境 JSON → profile → --config → 命令参数`；`current_protocol.py` 将其转换为旧 Runner 的配置，实际转换结果保存在运行内。AN/SA 仅改变方向二次项；MAPPO/IPPO 使用同一 Actor 输入、网络和概率头。

应用中的 `direction_steps / actor_fit_steps / critic_steps` 对应旧训练器的 `direction_epochs / actor_epochs / critic_epochs`；每个 epoch 的真实小批次数和优化次数由旧日志记录。有限固定基点的 `fit_steps` 仍表示优化步数。不要把这些计数直接当作相同的环境成本。

`pair_coordination/train.py` 及导航 `--conditions / --resume-suite` 底层工作流保留。旧成对 P-M/P-H/P-C 等套件驱动、`suite_*.json` 和绑定历史结果的分析脚本已移除，仅在原归档中保留；当前实验使用上方矩阵命令。底层默认配置不能和当前论文配置混用。

原训练器保留完整轮次 checkpoint 和恢复逻辑。单次成对入口和导航原套件入口保留相应恢复参数；当前统一矩阵入口尚不提供自动续训开关。更换观测结构后，旧展平网络 checkpoint 不能直接当作实体网络继续训练。停用重写框架的 `entity_experiments_v1` checkpoint 已不再由当前评价入口加载。

## 5. 文件、监控与回放

现有数据的职责、归档位置与最新分析统一见 [results/README.md](results/README.md)。首批 pilot 已归档至 `results/archive/`，四组检查敏感性结果集中于 `results/pair_checks/`；原始运行清单保留当时路径。

- `manifest.json`：套件配置、代码与依赖信息、实验完成状态。
- `summary.csv`、`analysis.md`：跨种子指标、配对差值和解释边界；单种子不生成虚假的种子置信区间。
- `training.csv`、`diagnostics.csv`、`evaluation_episodes.csv`：分别记录学习过程、机制诊断、纯报告评价。
- `overview.png`：覆盖更新的总图，不每轮另存图片。
- 每个旧训练运行保留一个 `events.jsonl` 账本及必要配置/摘要；没有逐轮新增 JSON 文件。账本用于追踪采样和断点恢复，不再复制出一组分模块 JSONL。
- 成对 P-L 保存 `checkpoints/final.json、best.json`；应用保存 `checkpoints/final.pt、best.pt`。当前协议 best 只用独立源评价，主比较使用 final。T-L 的有限新模型保存 `.npz`。

导航或仓库的矩阵运行可生成冻结任务 GIF：

```powershell
python -m manifold_project.experiments.visualize --checkpoint manifold_project/experiments/results/nav_source_pilot_v1/N-C_source_learning/full/AN_s40/checkpoints/final.pt --agents 6 --seed 1000000 --output manifold_project/experiments/results/nav_source_pilot_v1/replay_n6.gif
```

原导航辅助工具归入 `cooperative_navigation/tools/`，用于原套件格式：

```powershell
python -m manifold_project.experiments.cooperative_navigation.tools.monitor --help
python -m manifold_project.experiments.cooperative_navigation.tools.replay --help
python -m manifold_project.experiments.cooperative_navigation.tools.stability --help
```

矩阵运行使用前述 `experiments.visualize`；旧根目录 `monitor_navigation.py`、`visualize_navigation.py` 等路径不再保留转发副本。

验证命令：

```powershell
python -m pip install -r manifold_project/experiments/requirements-test.txt
python -m pytest manifold_project/experiments/tests -q
```

测试集中为 13 个文件、102 项检查，本次均已通过：当前入口与统计、P/T 数学机制、实体与环境协议、应用训练/恢复/冻结、必要图表检查，以及原成对环境、梯度、PyTorch、原子保存、断点和底层入口检查。旧持续导航、最近邻槽位、历史配置迁移、旧套件编号、重复图表布局等测试已移除；保留的数学检查直接检验当前实现，不从 ZIP 加载历史默认配置。测试通过不等于正式性能结论。
