# 作者 MAPPO 源码

来源：[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac)，固定提交 `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`。保留 [MIT LICENSE](LICENSE)；[UPSTREAM.json](UPSTREAM.json) 记录原文件 SHA256。

仅收录前馈MAPPO/IPPO训练所需模块及其导入依赖。原文件唯一修改是把 `from onpolicy...` 改为本项目私有vendor命名空间，避免与已安装的同名包冲突。两种PPO共用作者的更新、GAE/λ-return、小批次生成、Actor/Critic和ValueNorm实现。没有运行作者环境或安装其旧版PyTorch、Gym、WandB等依赖。

任务适配集中在[training/author_ppo.py](../../training/author_ppo.py)：共同MPE2场景、采样预算、随机动作采样、超时最终输入和统一独立评价。MAPPO Critic读集中状态，IPPO Critic只读本地历史；两者均关闭RNN。参数及保存说明见[模块说明](../../模块说明.md#4-mappoippo-适配)。本版本称“基于作者实现、适配统一MPE2持续任务协议的MAPPO/IPPO”，不等同于复现作者论文原始场景及成绩。

更新上游时应重新核对许可、导入依赖、终点语义与全部适配测试，并重建 UPSTREAM.json；不能覆盖此目录后沿用旧实验指纹。
