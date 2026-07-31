# Z/M Phase C tonic-branch identity — post-result futility stop（2026-07-31）

## 1. 一句话结论

Phase C 没有在已访问的 \(Z/M/S_G\) 慢状态邻域中找到发作样非-tonic carrier。用户授权停止时，seed 1 的 primary 邻域已完成 59/60 个 continuation；corrected-v2 重判后 59/59 都是 `tonic_non_AI`，而且所有 10 个 cell 即使把唯一缺失 run 假设成阳性，也不可能达到预注册的 5/6 cell 门。因此原注册的 primary maturation GO 已经逻辑不可达，继续跑 seed 3/4、shell、gain 或 \(dt/2\) 不再回答核心问题。

这不是完整的 Phase C bounded negative。安全结论是：

> 在已观测的 seed-1 primary frozen slow-field neighbourhood 内，移动 \(Z/M/S_G\) 坐标只把网络留在同一个局部高率、低调制 tonic fast branch；它没有产生所需的非-tonic ictal carrier。下一步应改 fast inhibitory/membrane feedback，而不是继续扩大 slow-field 网格。

## 2. 测了什么

Phase C 原来要区分两件事：

1. 自然 Z/M 轨迹访问到的 frozen states 上，持续态究竟是 AI tonic、refractory/tonic，还是有稳定周期/爆发结构的 carrier；
2. 沿预注册 primary convex neighbourhood 移动 \(z_i,m_i,S_G\) 后，能否出现跨相邻 cell、跨 seed、并通过 \(dt/2\) 的非-tonic maturation window。

每个 cell 有两种 fast phase × 三种未来噪声 continuation，共 6 个 run。cell-level 阳性要求至少 5/6 run 属于同一种 `periodic_non_tonic_carrier` 或 `clonic_or_bursting_carrier`，且两种 fast phase 都有支持；primary GO 还要求 native seeds 1 和 3 都支持 homologous window。

## 3. 数据与执行状态

- C0 已完成 153/153 exact tasks；seeds 1/3/4 均为 `mixed_or_indeterminate_tonic_branch`。
- C1 原完整 base matrix 为 204 runs。
- 用户授权停止时：
  - 已完成 59 runs；
  - 全部来自 seed 1 的 10 个 primary cells；
  - 9 个 cell 完成 6/6；
  - `primary__peak__bounded_late` 完成 5/6；
  - 其余 145 runs 未完成。
- 协调器用自身 ownership 表清理 12 个 inflight workers；没有残留仿真进程。
- 59 个已发布 part 均有相邻 resource receipt，锁定采样中 worker `VmSwap=0` 与 pre-publish self snapshot。
- partial-abort record：
  `results/topic4_sef_hfo/zm_phase_c_tonic_identity/coordinator_runs/phasec1_base_partial_abort_20260731T005704_p1220175.json`。

本轮没有生成 `phasec1_base_atlas_dt.json`，没有伪装成 204/204 complete；conditional gain、\(dt/2\)、modal 和 full Phase-C adjudication 都不再授权。

## 4. P0 诚实性修正

原 v1 phenotype classifier 把 pathology-core `source_rate_hz` 喂给了语义上属于 whole-sheet 的 250 Hz runaway gate。由于当前持续态的核心率约 440 Hz，它被错误叫成 runaway；但 production NPZ 已经独立保存 `carrier_gate_r_all_hz` 与 `carrier_gate_bin_ms`，不需要重跑 SNN。

修正版保持所有 source temporal morphology 判据不变，只把 runaway/tail-trend 映射到 all-sheet E-rate：

- pathology-core mean：435.55–442.58 Hz；
- all-sheet mean：140.12–160.17 Hz；
- all-sheet 全部低于 250 Hz gate；
- corrected runaway scope：59/59 `all_sheet_E`；
- corrected phenotype：59/59 `tonic_non_AI`。

原生产 manifest、原 classifier、原 analyzer 和原始 SNN parts 保持 byte-identical。修正通过独立 v2 adapter 和 futility verdict 绑定。

## 5. 为什么现在可以停，而不是继续 C1

单 cell 需要至少 5 个非-tonic positive runs。

- 9 个完整 cell：6 个已观测 run 全阴，最大可能阳性数 = 0；
- 1 个不完整 cell：5 个已观测 run 全阴，即使唯一缺失 run 阳性，最大可能阳性数 = 1；
- 因而 10/10 cell 都已经 mathematically unrescuable。

primary GO 明确要求 native seed 1 与 seed 3 同时支持 homologous window。seed 1 已不可能产生一个 positive primary cell，更不可能形成两个相邻 positive cells。因此 seed 3/4 的任何结果都不能恢复 primary GO；secondary shell 也只允许作 extrapolative sensitivity，不能建立 primary reachability。

这个停止规则是在结果出现后根据逻辑不可达性触发的，所以必须叫 `post_result_futility_stopped_incomplete`，不能回写成预注册 full-matrix negative stop。

## 6. 核心动力学读数

59 个 run 的 corrected-v2 汇总：

| 读数 | min | median | max |
|---|---:|---:|---:|
| fine-rate modulation depth | 0.0251 | 0.0341 | 0.0445 |
| pathology-core E rate (Hz) | 435.55 | 439.23 | 442.58 |
| all-sheet E rate (Hz) | 140.12 | 149.66 | 160.17 |
| active-core \(\rho_{80}\) | 1.0 | 1.0 | 1.0 |
| refractory-ISI fraction | 0.084 | 0.154 | 0.182 |

这些值说明当前态不是 whole-sheet runaway，也不是典型的 refractory-ISI-locked saturation；它是一个核心神经元几乎持续贴近高率上限、群体包络却只轻微起伏的局部 tonic branch。调制深度上限 0.0445，只有注册非-tonic 门 0.20 的约 22%。

因此当前 slow coordinates 能改变工作点和空间占据，但没有改变 fast subsystem 的吸引子类型。继续调 \(z,m,S_G\) 更可能在 tonic plateau、silence 与 runaway 之间移动，而不是凭空产生新的 oscillatory branch。

## 7. 图的批注

主图：
`results/topic4_sef_hfo/zm_phase_c_tonic_identity/figures/fig_phasec_futility_seed1_primary.png`。

- **a**：橙色 59 格都是 `tonic_non_AI`；唯一灰格只是 stopped-before-run。它直接显示缺失 run 不足以挽救任何 cell。
- **b**：核心率约 440 Hz，而全场约 150 Hz；旧 250 Hz 竖线展示错误 scope，横线才是 whole-sheet runaway gate。点全部在横线下。
- **c**：所有 run 的 modulation depth 聚集在 0.025–0.045，远离 0.20 非-tonic 门；slow-state path 上没有向 carrier bifurcation 接近的可见趋势。
- **d**：代表性 continuation 的核心与全场率在 8 s 内近乎平坦；这是 persistent frozen tonic branch 的直接波形，但尚未经过扰动回归，不能称为已证明的 attractor，也不是 sustained 30–80 Hz ictal-energy carrier。

## 8. 能写与不能写

### 能写

- seed-1 primary slow-state neighbourhood 的 59 个已完成 continuation 均停留在 corrected `tonic_non_AI` branch；
- 原注册 primary maturation GO 已逻辑不可达，因此继续完整 C1 对核心问题没有信息增益；
- 当前瓶颈定位到 fast inhibitory/membrane dynamics，而不是 slow-coordinate coverage。

### 不能写

- “三个 seeds 的 primary neighbourhood 已完整 NO-GO”；
- “整个 Z/M SNN 不存在任何 ictal carrier”；
- “发作吸引子已经建立”；
- entry、offset、recovery 或完整 lifecycle；
- 临床发作机制或 Abbott 机制已被证明。

## 9. 下一步

Phase C 到此停止。下一节点为独立 Phase D：
`docs/superpowers/specs/2026-07-31-topic4-zm-snn-fast-carrier-repair-design.md`。

Phase D 保留 E1146 two-end anisotropic SNN、per-neuron Z/M、虚拟 SEEG 和全部 E→E 权重/核；只修改抑制如何进入膜方程，并加入预注册的亚秒 dynamic-threshold arm。先用 frozen state-fork 判定 fast subsystem 能否产生有界、持续、空间非同步的 carrier；只有 carrier 过门，才重新打开动态 Z/M 去测 entry、offset 与 recovery。

## 10. 机器判决

- `results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_futility_verdict.json`
- schema：`zm_phasec_post_result_futility_stop_v1_2026-07-31`
- status：`post_result_futility_stopped_incomplete`
- verdict SHA：以 artifact 内 `verdict_sha256` 为准。
