# 相关论文与本项目主张对照

整理日期：2026-09-23；研究定位更新：2026-09-24。

本次提及的 8 篇论文已下载到本目录，并核对了 PDF 标题、页数及下文涉及的主要定义、命题和实验段落。下载来自出版方或作者的 arXiv 版本。本文维护文献对照；当前方向函数研究见[CTDE 主稿](../manifold_project/docs/策略流形研究设计.CTDE本地策略改进版.md)，工作进度见[开发路线](../manifold_project/docs/研究主张与开发路线.md)。

**当前定位：以规定本地信息下的方向函数学习为主线，保留解析 Fisher 估计、方向到共享 Actor 的实现、条件跨规模运输三个贡献候选。下列表示、可识别性和外推文献用于审查边界，不再要求先完成一个通用表示联合命题。网络可处理变长输入不等于信息充分，零样本迁移本身也不是新问题。**

## 1. 下载清单

点击论文标题打开本地 PDF；“原始来源”用于核对公开版本。页数为文件实际页数。

| 编号 | 本地论文 | 版本与页数 | 原始来源 |
|---|---|---|---|
| R1 | [On the Limitations of Representing Functions on Sets](2019_Wagstaff_Limitations_of_Set_Representations.pdf) | Wagstaff 等，ICML 2019，8 页 | [PMLR](https://proceedings.mlr.press/v97/wagstaff19a.html) |
| R2 | [Action-Sufficient State Representation Learning for Control with Structural Constraints](2022_Huang_Action_Sufficient_State_Representations.pdf) | Huang 等，ICML 2022，20 页 | [PMLR](https://proceedings.mlr.press/v162/huang22f.html) |
| R3 | [Approximate Information State for Approximate Planning and Reinforcement Learning in Partially Observed Systems](2022_Subramanian_Approximate_Information_State.pdf) | Subramanian 等，JMLR 2022，83 页 | [JMLR](https://jmlr.org/papers/v23/20-1165.html) |
| R4 | [The Value Equivalence Principle for Model-Based Reinforcement Learning](2020_Grimm_Value_Equivalence_Principle.pdf) | Grimm 等，NeurIPS 2020，12 页 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) |
| R5 | [Policy Gradient Methods in the Presence of Symmetries and State Abstractions](2024_Panangaden_Policy_Gradient_State_Abstractions.pdf) | Panangaden 等，JMLR 2024，57 页 | [JMLR](https://www.jmlr.org/papers/v25/23-1415.html) |
| R6 | [How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks](2021_Xu_How_Neural_Networks_Extrapolate.pdf) | Xu 等，ICLR 2021；下载 arXiv v5，52 页 | [arXiv v5](https://arxiv.org/abs/2009.11848v5) |
| R7 | [Learning Transferable Cooperative Behavior in Multi-Agent Teams](2019_Agarwal_Transferable_Cooperative_Behavior.pdf) | Agarwal、Kumar、Sycara，2019；下载 arXiv v1，10 页 | [arXiv v1](https://arxiv.org/abs/1906.01202v1) |
| R8 | [Invariance in Policy Optimisation and Partial Identifiability in Reward Learning](2023_Skalse_Invariance_and_Partial_Identifiability.pdf) | Skalse 等，ICML 2023，26 页 | [PMLR](https://proceedings.mlr.press/v202/skalse23a.html) |

R7 在此按已下载的 arXiv 预印本登记，不标成 ICML 主会论文。R4 为出版方主文 PDF；清单不表示每篇论文的独立补充文件均已下载。

本目录原有的 [ROMA](wang20f.pdf) 与 [When Is Generalizable Reinforcement Learning Tractable?](NeurIPS-2021-when-is-generalizable-reinforcement-learning-tractable-Paper.pdf) 保留，未改写。

## 2. 对照所用的本项目研究对象

> 在规定的本地信息和一个源规模下，用团队经验学习动作概率怎样调整，再让实际 Actor 实现，分析人数变化后这些调整继续有效的条件。

论文首先分析同一个旧策略附近的一阶方向。输入权限、固定预处理、方向网络能力和 Actor 实现分别处理。变长实体结构是待验证实现；当前无需证明任意规模都存在可学习的充分表示，才开展方向估计实验。

跨规模定理使用完整实际输入的覆盖与密度比条件，以及条件方向偏移。新名单长度等违反覆盖时，保留为经验外推；更强源外理论需要另证结构条件。最终训练性能与相对其他算法优势均需独立实验。研究状态和四组实验见主稿第 6—7 章。

## 3. 每篇论文已经回答什么，我们还想回答什么

| 论文 | 论文已经建立或展示的内容 | 与我们重叠的部分 | 我们需要额外完成的内容 |
|---|---|---|---|
| R1：集合表示限制 | 在连续求和分解等明确设置下，普遍表示集合函数需要足够的潜空间维度 | 压缩可能丢失决定行为的信息 | 对指定任务说明应保留哪些区别、容许多少损失；不能由其结论推出所有固定维表示都不可用 |
| R2：动作充分表示 ASR | 利用生成模型的结构条件找出对策略学习充分的状态因素，并学习相应表示 | 压缩应服务决策，而非重建一切 | 区分当前策略方向信息与策略充分状态；本稿不另承诺通用充分表示学习 |
| R3：近似信息状态 AIS | 历史压缩满足奖励与预测近似条件时，建立规划性能界；含分散多智能体扩展 | 信息损失怎样影响决策质量 | 对比其规划性能对象与本稿固定旧策略的方向投影，不将方向误差说成最优回报差 |
| R4：价值等价 | 对指定策略集和价值函数集保留 Bellman 更新，可避免恢复完整动力学 | 只需识别与使用目的相关的规律 | 若改为局部改进方向，须证明它带来何种不同条件或可验证收益；更换被保留的量不是充分创新 |
| R5：抽象下策略梯度 HPG | 连续 MDP 同态下的价值等价和策略梯度结果，以及学习抽象的算法 | 表示、策略改进与实际算法的连接 | 对比方向、参数实现及其前提；当前不声称推广同态定理 |
| R6：网络外推 | 在特定网络和数据条件下分析外推；研究图网络结构与目标计算的匹配 | 为什么训练范围以外的行为受到约束 | 处理策略诱导的数据分布、团队优势与 Actor 更新，不能只靠网络接受可变长输入 |
| R7：可迁移协作 | 代理—实体图、共享策略与消息传递；包含不同团队规模的零样本实验 | 可变数量实体表示、协作与跨规模冻结执行 | 对齐信息权限和训练来源；本稿差异应落到方向估计、Actor 实现及条件运输，不能只称首次跨规模迁移 |
| R8：部分可识别性 | 刻画奖励学习的数据歧义，并分析其对策略优化与跨环境迁移的限制 | 源数据无法区分的解释可能在目标中产生冲突 | 研究有团队奖励反馈时的表示、方向与跨规模歧义；不能把一般不可识别性论证当作原创机制 |

以上“需要额外完成”描述对应的比较工作，不表示原论文完全不涉及相邻问题，也不表示本项目已经得到更强结论。依据与重要限制如下。

### R1：维度兼容不等于信息足够

重点见 PDF 第 4–5 页、定理 4.1/4.3/4.4。维度下界针对连续集合函数的普遍表示与相应分解结构；特定简单函数可以使用更低维表示。它要求我们明确需要保留的任务函数族，而不是直接否定所有池化或注意力编码器。[原文](https://proceedings.mlr.press/v97/wagstaff19a.html)

### R2：决策相关压缩已有明确理论路线

重点见 PDF 第 2–3 页、定义 1 和命题 1。ASR 基于潜在生成模型的图结构；命题使用 Markov 与忠实性等条件识别对策略学习充分的状态因素。潜在因素充分不等于机器人已从实际局部观测中获得这些因素。本稿用权限与预处理损失区分这些层次，不把获得充分状态设为方向学习前提。[原文](https://proceedings.mlr.press/v162/huang22f.html)

### R3：AIS 已经覆盖分散多智能体，不能按“单智能体”排除

重点见信息状态/AIS 定义与性能界，以及第 5 节。其多智能体部分利用部分历史共享和公共信息方法构造虚拟协调问题，再建立近似信息状态。虚拟协调者是数学等价转换，不等于执行时额外提供中央控制器。因此“我们是 CTDE/多智能体”不足以形成区别；应逐项比较固定策略的方向误差与其规划误差对象、信息协议和前提。[原文](https://jmlr.org/papers/v23/20-1165.html)

### R4：只恢复行为相关规律，不是新的原则

重点见 PDF 第 2–4 页、定义 1。价值等价相对于明确的策略集与价值函数集定义；这些集合越丰富，允许的模型歧义越少。我们若提出“方向相关的等价或约束”，必须解释保留对象为什么足够、源数据怎样学到它，以及策略改变后是否仍成立。[原文](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)

### R5：这是理论上应优先逐项对照的工作

重点见 PDF 第 10 页定义 11、第 15 页定理 17。同态要求奖励不变、转移经过映射后相容，并有相应数学正则条件。附录 C.3/图 19 还展示了切换任务后的继续学习曲线，不能说它没有迁移实验。该实验协议不同于冻结 Actor 的跨团队规模评价。我们的局部改进目标也比完整价值等价更窄，条件不同须逐项证明。[原文](https://www.jmlr.org/papers/v25/23-1415.html)

### R6：外推需要结构和数据条件

重点见关于 ReLU 网络外推和图网络计算结构的理论、实验。论文并未保证任意 GNN 在任意大图上泛化。对本项目的启发是把可识别的关系与具体函数类写清楚；实际网络若允许源外任意变化，不能借用受限函数类的外推结论。[原文](https://arxiv.org/abs/2009.11848v5)

### R7：这是任务上最接近的先例

重点见第 3.1 节、第 4.4–4.5 节和表 4。论文允许机器人在回合开始知道全部实体位置，并通过图进行消息传递；执行权限须与我们的方案明确对齐。第 4.5 节将 5 个机器人规模的策略直接用于其他人数、不微调；第 4.4 节另有跨规模课程训练。正文不足以完全确认表 4 模型的全部训练及选择来源，不能据此认定它满足或违反我们的严格单源协议。[原文](https://arxiv.org/abs/1906.01202v1)

### R8：源内歧义会影响迁移，已有明确反例与定理

重点见第 4.2 节、定理 4.2：奖励学习中，源动力学下不可区分的奖励可能在新动力学下产生不同后果；该强结论有奖励依赖后继状态等具体条件。本项目直接收到训练奖励，研究对象还包括方向和信息压缩，不能直接套用其结论。但“源一致、目标冲突”的论证范式应明确引用前人。[原文](https://proceedings.mlr.press/v202/skalse23a.html)

## 4. 三个贡献候选怎样对照先行工作

| 当前候选 | 应比较的直接先例 | 本稿需提供的证据 |
|---|---|---|
| 解析 Fisher 方向估计 | Natural Actor-Critic、Fisher 与采样外积分析 | 同数据、同标签、同函数类的 AN/SA 经验差值、回合级条件和方向误差；无无条件优势主张 |
| 方向到共享 Actor | MPO、AMPO、自然梯度及近似投影 | 前向 KL 的有限拟合与理论切空间的区别，同一学得方向进入实际 Actor 后的收益 |
| 条件跨规模运输 | 可迁移协作、外推、单源歧义研究 | 完整输入覆盖、条件方向偏移、一步交互机制与同一源参数更新；覆盖失效明确标成经验外推 |

这些直接方法的原始出处、推导对应与完整引用见主稿第 2、8 章。表示相关的八篇论文帮助解释信息和外推边界，不能代替对直接方向学习与策略拟合方法的新颖性审查。

机器人执行时分不清，与源经验无法区分目标延拓，是两个保留的问题。前者由信息误差及其细分说明，后者限制源外保证；当前不将二者合并成必须先完成的大型联合理论。一般神经网络怎样在源外保留方向、有限拟合的误差界和跨轮最终性能，仍按主稿列为未解决任务。

## 5. 贡献陈述的边界

可采用的表述是：研究严格单源 CTDE 中规定本地信息上的改进方向，分析解析 Fisher 经验估计、共享 Actor 实现以及成对交互规模变化对方向有效性的影响，并用受控实验检验这些关系。

不应声称首次研究控制信息压缩、首次保持抽象策略梯度、首次实现跨人数零样本迁移、源训练保证任意目标表现，或已证明优于相关方法。主稿保留三个贡献候选；候选网络尚未实现，四组新实验尚未运行。

本索引保留此前八篇论文的来源与范围核对，不声称已完成全面优先权审查。当前阅读顺序宜先按主稿第 2 章比较自然 Actor-Critic、MPO、AMPO 和 Fisher 估计，再按输入与跨规模问题查 R1—R8。本目录的 PDF 和用户补充的中文精翻保持原文件，不因项目定位变化改写论文内容。
