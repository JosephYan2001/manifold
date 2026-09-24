# 实验结果索引

更新：2026-09-25。按证据职责阅读，不把同种子的重复实验或目标规模算成独立样本。原始 manifest、CSV、图片均保留。

| 目录 | 用途 | 当前判断/入口 |
|---|---|---|
| `pair_estimation_20seeds_v1/` | P-E 方向估计，20 数据种子；P-EA 的直接输入 | 固定方向方差改善稳定；最终方向误差优势尚未确立 |
| `pair_realization_transfer_10seeds_v1/` | 原 P-A、P-T 多种子基线 | [多种子复验分析](pair_realization_transfer_10seeds_v1/多种子复验分析.md) |
| `pair_ea_mc2048_20seeds_v1/` | P-E→Actor 完整证据链，MC/2048 | [最新联合分析及下一步命令](pair_ea_mc2048_20seeds_v1/方向到执行联合分析.md) |
| `pair_checks/` | 四组检查敏感性对照，集中管理 | [联合检查分析](pair_checks/pair_transfer_check512_10seeds_v1/检查敏感性实验分析.md) |
| `archive/` | 首批 pilot，历史探索记录 | 不与扩种子后的相同记录重复计数 |

## 本次整理

- `pair_mechanisms_pilot_v1` 移至 `archive/pair_mechanisms_pilot_v1`。
- `pair_transfer_check32_10seeds_v1`、`pair_transfer_check128_10seeds_v1`、`pair_transfer_check512_10seeds_v1`、`pair_transfer_no_direction_check8_10seeds_v1` 原名移入 `pair_checks/`。
- P-E 源目录和 P-EA 保持原路径，避免破坏方向复用。历史 manifest 中的原始运行命令/路径保持原样，不把历史记录改写成新命令。
- 没有删除原始实验数据。部分种子虽重复，整体运行仍是可复现来源，不做跨运行物理去重。

下一组结果计划为 `pair_ea_mc32_128_512_20seeds_v1/`，与 2048 档互补。新实验依旧通过顶层统一入口运行。
