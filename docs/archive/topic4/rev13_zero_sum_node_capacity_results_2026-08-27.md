# Topic 4 rev13 zero-sum Node capacity 正式结果

日期：2026-08-27

## 1. 科学问题

rev13 测试一个限定很窄的模型容量假设：在冻结的 data-driven 静态连续 Node
场上，加入一个总阈值偏移为零、由近期活动驱动的场内可达性重分配，是否能在同一张网络内稳定
增强两个相反方向的 causal-family 结构。

本轮不是 patient fit，也不是 ictal experiment。运行期间没有读取 patient labels、
patient prototypes 或 contact classifier；EE、E-to-I 和 Z/M 全部关闭。只有先通过
model-internal capacity，才允许打开 patient-training target。

## 2. 正式输入与产物

正式 artifact 根目录：

`results/topic4_sef_hfo/data_driven_node_dualmode_rev13/node_zero_sum_recovery_canary/`

承重产物：

- `parity/exact_off_parity_audit.json`
- `aggregate/model_internal_per_run.json`
- `aggregate/model_internal_paired.json`
- `analysis/model_internal_decision.json`

正式设计为 6 arms × 3 network seeds（2311、2312、2313），共 18 runs：

- `exact_off`
- `zero_sum_c010`
- `zero_sum_c020`
- `zero_sum_c040`
- `raise_only_c020`
- `spatial_shift_c020`

其中只有 `zero_sum_c020` 具有同系数 raise-only 和 spatial-shift matched
controls，因此只有它能得到正式 capacity pass/fail。`c010` 和 `c040` 仅提供剂量信号。

## 3. 工程验收

### 3.1 Exact-off parity

`exact_off` seed 2291 与历史 Stage-AK artifact 的 20 s common prefix 审计为
`PASS`。以下内容一致：

- 冻结 E-neuron positions、Node field、`delta_vtheta`、edge coefficients 和
  contact geometry；
- whole-sheet active fraction；
- contact envelope；
- sheet activity counts；
- common-prefix 内 35 个可比较 causal-family events 的 onset、offset、return、
  contact onsets/ranks、source onset maps、fragment count 和 root identity。

这关闭了“rev13 新路径没有复现冻结 Node-only 基线”的工程替代解释。

### 3.2 正式运行完整性

- 18/18 runs 完成；
- 18/18 无 runaway；
- patient labels/prototypes 均未加载；
- 所有 active controller 均满足 zero-sum、无 widespread saturation、无 widespread
  static sign flip。

## 4. 修正事件单位后的样本支持

正式分析单位是完整 causal-root family，不是 detector fragment。任何属于
time-overlap connected episode 的 family 都被整体排除。下表给出排除前 returned/evaluable
family 数和排除后的 isolated family 数：

| Arm | Seed 2311 | Seed 2312 | Seed 2313 |
|---|---:|---:|---:|
| `exact_off` | 41 → 26 | 45 → 31 | 55 → 29 |
| `raise_only_c020` | 35 → 24 | 50 → 34 | 37 → 24 |
| `spatial_shift_c020` | 55 → 45 | 55 → 28 | 50 → 38 |
| `zero_sum_c010` | 49 → 39 | 45 → 31 | 40 → 28 |
| `zero_sum_c020` | 41 → 25 | 51 → 38 | 45 → 35 |
| `zero_sum_c040` | 41 → 33 | 44 → 26 | 35 → 29 |

所有 18 个 run 都达到至少 24 个 isolated families 的正式可评价下限。`c020`
与 paired off/controls 做 event-count matching 后，三个网络分别使用 24、28、24 个
families。因此本轮阴性结果不是由样本不足造成的。

## 5. `zero_sum_c020` matched-control 结果

正式连续端点是三等时长时间块 held-out 的 K2-K1 log-likelihood improvement，
单位为每个 matched causal family。候选必须同时超过 paired `exact_off`、
`raise_only_c020` 和 `spatial_shift_c020`。

| Seed | Matched n | `c020` K2-K1 | 相对 off | 相对 raise-only | 相对 spatial-shift | 结果 |
|---:|---:|---:|---:|---:|---:|---|
| 2311 | 24 | +0.486 | +0.073 | -0.013 | +0.409 | 未超过 raise-only |
| 2312 | 28 | +0.244 | -0.094 | +0.189 | +0.448 | 未超过 exact-off |
| 2313 | 24 | +0.220 | +0.109 | -0.127 | +0.124 | 未超过 raise-only |

三张网络中的 `c020` 都有正的 K2-K1、同一网络内相反方向、至少 20% 少数方向、
足够的方向一致性与时间块复现，也没有 forced alternation 或 runaway。但这些是候选自身的
描述，不足以证明 zero-sum controller 带来特异增益。它在 0/3 网络中同时超过静态底物和
两个 matched controls，低于预注册的至少 2/3 标准。

## 6. 正式裁定

```text
ZERO_SUM_NODE_CAPACITY_NOT_OBSERVED
```

安全结论是：在当前冻结静态连续 Node 底物、`tau_a=250 ms` 和正式
`c=0.2` 条件下，field-gated zero-sum activity-dependent redistribution 没有显示出
超过 paired static Node、raise-only 和空间错配对照的可复制双方向容量增益。

不能据此声称：

- 所有动态 Node 机制都无效；
- 当前 SNN 没有双模式容量；
- patient 两种模式不真实；
- EE、E-to-I 或 Z/M 应该被打开来“补救”本轮结果。

## 7. 未执行内容

由于 model-internal capacity 未通过，以下步骤均未启动：

- patient-training objective；
- patient held-out；
- patient alignment 与完整四层 loss；
- fresh-network Node confirmation；
- same-checkpoint hotspot intervention；
- EE、E-to-I 或 Z/M 调参；
- `tau_a={100,500 ms}` 或更多 `c` 扫描。

这不是工程中断，而是按预注册 stop rule 正常收口。

## 8. 下一步

下一步回到静态连续 Node 场本身，使用本轮修正后的事件单位做诊断，而不是继续扫
zero-sum controller 参数：

1. 以完整 causal-root family 取代 detector fragment；
2. 排除 overlap-connected episode 的全部成员；
3. 在每张网络内检查两个方向是否来自独立完整事件，而不是同一长事件的切片；
4. 量化静态场的起始位置、传播方向、事件支持与重叠结构；
5. 先解释静态连续 Node 场为什么偏向当前事件组织，再决定是否需要修改 Node 场表示。

在这一诊断闭合前，不打开 patient target/held-out，也不进入 EE、E-to-I 或 Z/M
cross-state 调参。
