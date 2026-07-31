# Z/M Phase C completion audit（2026-08-01）

## 1. 一句话判断

**验收：`ACCEPT_PHASEC_POST_RESULT_FUTILITY_STOP`。** Phase C 已经完成其可支持的
科学决策：C0 完整判定了 visited tonic branch 的 identity；C1 在用户授权下，
以 seed-1 primary 59/60 runs 的严格逻辑 futility stop 收口。它不是 204/204
完整 atlas，也不是三-seed bounded negative，更不是 ictal lifecycle。

这份 audit 不补跑唯一缺失格，也不把后来 Phase-D engine 混入旧合同。原因不是
节省算力，而是缺失格即使为阳性也无法挽救任何 cell，且结果后的用户指令已经
明确改 spec；继续生产只会扩大一个已经不能改变 registered GO 的面板。

## 2. 逐项验收

| 目标 | 权威证据 | 状态 |
|---|---|---|
| 正确 Z/M SNN 衬底 | `phasec_manifest.json`：per-neuron Z/M、seed 1/3/4、冻结 slow fields、E→E 不变 | 通过 |
| C0 tonic identity | native C0 `153/153` tasks；`n_missing=0`；3 seeds 均为 `mixed_or_indeterminate_tonic_branch` | 完成 |
| C1 morphology maturation | seed-1 primary 59/60；corrected-v2 为 59/59 `tonic_non_AI`；10/10 cells 均不可达到 5/6 non-tonic 门 | 以 futility stop 完成决策 |
| 固定 observation panels | 每 seed activity-independent analysis panel 1024 E cells、pairwise panel 256 E cells，均有独立 self-hash | 通过 |
| 分层 bootstrap | 5000 draws；500 ms blocks、core/surround neuron strata、固定 pair census、matched shift-null strata、continuation/phase 层级均实现并 mutation-tested | 通过 |
| provenance | C0 153/153 resource receipts 完整；C1 59 part/NPZ/receipt hashes 逐项复核无错，59 个 runtime SHA 均为 `7f061b96...` | 通过 |
| 资源安全 | C1 59 个 child pre-publish `VmSwap=0`；coordinator 只清理自身 12 个 inflight workers | 通过 |
| 图与归档 | `fig_phasec_futility_seed1_primary.{png,pdf,json}` + 中文 README + 2026-07-31 futility archive | 通过 |
| commit / push | Phase-C futility archive commit `939691da`，已位于远端分支历史 | 通过 |

## 3. C0 验收细节

`results/topic4_sef_hfo/zm_phase_c_tonic_identity/c0_identity_summary_dt.json`
报告：

- expected/validated resource entries：153/153；
- receipt issues：0；
- missing tasks：0；
- hierarchical bootstrap draws：5000；
- seed 1/3/4 均为 `mixed_or_indeterminate_tonic_branch`；
- aggregate verdict：`mixed_or_indeterminate_tonic_branch`；
- resolution gate：`not_required_without_native_positive`。

这回答的是 visited frozen tonic state 的 identity，而不是证明一个发作吸引子。

## 4. C1 为什么 59/60 足以停止，但不等于完整 negative

每个 primary cell 有两种 fast phase × 三种 future noise，共 6 runs；cell-level
阳性要求至少 5/6 runs 为同一种 non-tonic carrier。

- 9 个 cell 已完成 6/6，全部阴性；
- `primary__peak__bounded_late` 完成 5/6，5 个已观测 run 全部阴性；
- 唯一缺失格是
  `seed1 / primary__peak__bounded_late / peak / noise_resample_2`；
- 即使缺失格为阳性，该 cell 最多也只有 1/6 阳性；
- 因而 seed 1 不可能产生任何 positive primary cell，而 registered GO 又要求
  native seeds 1 和 3 同时存在 homologous window。

所以继续 seed 3/4、secondary shell、gain、modal 或 dt/2 不可能恢复 primary
GO。机器状态必须继续写成 `post_result_futility_stopped_incomplete`，不能改写成
`complete_phasec1_negative`。

## 5. fixed panel 与 bootstrap 修复是否真实

最终 manifest 对每个 seed 锁定：

- `activity_independent=true`；
- analysis panel 1024 E cells；
- pairwise panel 256 E cells；
- panel IDs 由 config/seed/anatomical stratum 决定，而不是按结果重新选；
- native/dt2 复用同一个 anatomy panel contract。

hierarchical analyzer 的承重规则为：

1. CV2 neuron resampling 在 core/surround strata 内进行；
2. refractory fraction 从抽中的 500 ms block numerator/denominator 重新计算；
3. fixed pair panel 因共享 neuron 被当作 dependent design census，不作为 IID
   bootstrap 轴；
4. observed pair correlation 与 100-draw circular-shift null 先在相同 stratum、
   相同 block aggregation 下比较；
5. fast phase 是固定设计 stratum；continuation 才是最高层重复单位。

针对 pair pseudoreplication、漏 hierarchical field、core/surround 混用、null
先取分位数再聚合、refractory ratio 取 per-neuron median 等错误路径均有测试。

## 6. provenance 的当前态修复

Phase C 的 production manifest 锁定了当时完整 engine closure。Phase D 后来以
off-by-default 方式合法修改了 `kick_probe.py` 和 `slow_field.py`，因此在当前
HEAD 上重新调用 Phase-C builder 必须报 `live engine hash drift`。这说明旧合同
在 fail closed，不是旧结果失效。

原测试把“live engine 永远等于 Phase-C producer”误当成永久 CI 条件。当前修复
只改变测试隔离方式：

- 在历史 runtime commit `7f061b96` 上，真实 builder/producer closure 全量运行；
- 在后续 HEAD 上，先确认 live drift 被拒绝，再加载并纯验证 closed input
  manifest，继续执行 threshold/panel/coordinate mutation tests；
- final-lock synthetic tests 在后续 HEAD 上显式 mock 回同一个 closed input，
  不重建、不 re-bless、不改旧 manifest。

验证结果：历史 runtime commit 上 Phase-C suite **266 passed**；当前 HEAD 在
保留 live-drift fail-closed 语义后，完整 Phase-C suite **277 passed**。

## 7. 最终科学口径

### 可以写

- visited frozen states 支持的是局部高率、低调制 tonic continuation；
- C0 identity 在三 seeds 上是 mixed/indeterminate，而非已确认 AI 或 refractory
  saturation；
- seed-1 primary slow-coordinate neighbourhood 没有向 non-tonic carrier
  maturation 的证据，registered primary GO 已逻辑不可达；
- 因此改 fast inhibitory/membrane mechanism 比继续扩大同一 slow-field 网格更
  有信息价值。

### 不可以写

- 三 seeds 的 C1 neighbourhood 已完整扫描；
- Z/M SNN 任何位置都不存在 carrier；
- tonic continuation 已经是严格证明的 attractor；
- 已建立 bounded ictal oscillation、entry、offset、recovery 或 lifecycle；
- 后续 Phase D 的 baseline NO-GO 能回填为 Phase C 的证据。

## 8. Goal closure 解释

Phase C 的“完成”是**问题判定完成**，不是“强行跑完一张已无法改变结论的
204-run 表”。用户在看到逻辑 futility 后明确授权停止 C1 并修改下一版 spec；
`phasec_futility_verdict.json::user_authorization` 已绑定该决定。原目标中的 fixed
panel、hierarchical bootstrap、provenance、资源安全、图文归档、commit/push 均
已有直接证据，因此本阶段可以关闭；核心 ictal lifecycle 目标继续由后续独立
机制节点承担。
