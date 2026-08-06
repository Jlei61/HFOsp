# 审阅结论：MZ inhibitory-reserve 二维 frozen corridor

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

状态：**R0b fixed-q geometry supported；仅解锁 R1 的 `Ubar/q_res` 映射，不代表 autonomous reserve lifecycle 已成立。**

## 1. 一句话判断

R0a/R0b 已证明当前 P=3 current-based scaffold 在固定 effective inhibition `q` 时，存在一段可容纳 bounded CCO、平滑 additive-M exit 和同一低态恢复的二维 `(q,A)` corridor；正式通过区间是 **`.835–.845`**。`.825` failure 与 `.830` safe 只构成后验 confirmed-anchor bracket，`.8275` 仍 unresolved，不能把它们写成正式 safe strip 或已定位的单调分岔边界。

本节点没有积分 `D_I/q_res` 动力学，没有证明背景事件可自主进入、退出、reset 或 retrigger，也没有改变 E→E weight、kernel、delay、recurrent divisor 或 membrane conductance。bath 的 `q=.90` 是为 frozen-oracle parity 固定的 mask，不是 emergent spatial containment。

## 2. 完成程度

> **完成度：92/100**

已完成：

- R0a base-dt、四 phase 的二维 `q–A` step discovery；
- R0b 五节点 formal strip 的双 dt、四 phase source/step/smooth-ramp/recovery confirm；
- `.825/.830/.8325` lower-ramp sentinel 与 `.8275 unresolved` 的边界记录；
- 主 CSV、summary、图之间的独立一致性核对；
- R0b 13 个 formal gate 与 sentinel 5 个 anchor-bracket gate 全部通过；
- reserve 方程代数、R0/R0b gate 与底层 latch/patch 相关测试，当前至少 22 个通过。

尚未完成：

- 在 CCO 上测周期平均 inhibitory-use `Ubar_CCO(q,A=0)`；
- 由 q-nullcline 反算合法 `q_res`；
- 将 `D_I/q_res` 真正接入背景事件驱动的 slow loop；
- 验证 autonomous entry、四 returns 后 termination、reset、early/late retrigger；
- 将固定 bath mask 提升为 field/continuous-space 中的动态 containment。

## 3. P0 / P1 关键问题

### P0

当前产物没有发现会推翻 R0 fixed-q 结论的 P0。正式 strip 的原始表完整、无重复，图与 summary 的标签一致。

### P1：formal strip 与 anchor bracket 必须永久分开

正式 R0b safe strip 是：

```text
q = .8350, .8375, .8400, .8425, .8450
width = .0100
spacing = .0025
n_nodes = 5
```

`.825/.830` sentinel 只支持：

- `.825`：smooth ramp 8/8 `physical_or_numerical_failure`；约在 767 ms 离开 transfer support，`Amax-A_SN≈.02546 mV`；
- `.830`：8/8 回 LLL，但 smooth ramp 只越过 fold `≈.00260–.00312 mV`；
- `.8325`：8/8 回 LLL，越过 fold `≈.03588–.03640 mV`；
- `.8275`：source base-dt Poincaré closure `2.24–2.30e-5`，略高于锁定的 `2e-5`，因此保持 unresolved，没有重标为 safe 或 failed。

因此允许写“最高 confirmed failing anchor 为 `.825`、最低 confirmed safe anchor 为 `.830`，两者之间仍含 `.8275 unresolved`”。禁止写“safe strip 从 `.830` 开始”“边界已定位在 `.825–.830`”或“已证明该边界单调”。

### 已关闭的 P1：R0b 与 sentinel gate 已改为 fail-closed

此前识别出的 Cartesian completeness、配置 margin/bracket、实际 safe-mask interval 与空集合假阳性风险已经修复。canonical R0b 当前 13 个 gate 全部通过：

1. `tables_form_complete_cartesian_products`；
2. `source_CCO_all_q_phase_dt`；
3. `instantaneous_bracket_width_within_gate`；
4. `step_below_fold_remains_CCO`；
5. `step_above_fold_reaches_LLL`；
6. `step_registered_margin_reaches_LLL`；
7. `smooth_M_ramp_fixed_q_reaches_LLL`；
8. `smooth_M_ramp_crosses_low_fold`；
9. `formal_rows_have_zero_failclosed_violations`；
10. `effective_q_is_exactly_frozen_during_ramp`；
11. `parameter_restoration_returns_same_LLL_basin`；
12. `base_half_dt_labels_match`；
13. `continuous_safe_q_strip_from_outcomes_meets_gate`。

runner 明确核对 source/step/ramp/recovery 的 expected/observed rows 为 `56/56`、`160/160`、`56/56`、`40/40`；instantaneous bracket 为 `[-.005,+.005] mV`、宽度 `.010 mV`，registered margin 为 `+.025 mV`；safe interval 由实际 per-q outcomes 重建，唯一 formal interval 为 `.835–.845`。

sentinel 当前状态为 `R0B_LOWER_RAMP_CONFIRMED_ANCHOR_BRACKET`，以下 5 个 gate 也全部通过：

1. `sentinel_cartesian_product_complete`；
2. `known_failure_anchor_cartesian_product_complete`；
3. `known_failure_anchor_confirmed_in_canonical_r0b`；
4. `at_least_one_failclosed_safe_anchor`；
5. `unresolved_q_lies_strictly_between_confirmed_anchors`。

因此 gate 实现不再是当前阻断项；剩余限制是 fixed-q 科学层级，而不是 formal acceptance 的工程缺口。

### P1：R1 解锁必须使用窄定义

本轮只解锁：在 formal strip 下/中/上节点测 `Ubar_CCO`、反算 `q_res`、检查 q-nullcline 能否落入 strip。它没有解锁“reserve 已产生自发发作—终止周期”的结论，也不允许先调 M 或回头扫描 E→E。

## 4. 科学性问题与已支持结论

### 4.1 R0a 说明瞬时 step 不是承重证据

R0a 在 q 轴 `.8555→.8200` 的 11 个节点上，A=0 source 均为 bounded CCO；从 low-root fold 开始的瞬时 A-step 也都能回 LLL，并在注册轴上获得至少 `.20 mV` 的 right-censored safe margin。

但这只是 base-dt discovery。瞬时 step 会绕过旧 autonomous arm 中 q 与 A 同时缓慢移动的路径，因此不能单独解锁 reserve dynamics。

### 4.2 R0b 的 smooth ramp 证明 step-vs-ramp 差异是真实的

正式 `.835–.845` strip 中：

- source CCO：40/40 bounded；core/annulus 均满足 closure、period-CV、peak-drift 合同；bath 0 returns，最大峰值 `10.17 Hz`；
- threshold step：`A_SN-.005` 的 40/40 保持 CCO；`A_SN`、`A_SN+.005`、`A_SN+.025` 共 120/120 回 LLL；
- fixed-q smooth M ramp：40/40 回 LLL，无 support/bound/nonfinite violation；`Amax-A_SN` 为 `.0692–.2028 mV`；
- fixed-q 数值误差最大 `2.86e-8`，低于 `1e-7` gate；
- restoration 到 `q=.90,A=0`：40/40 回同一 LLL 分类，final fast RHS 远低于 `1e-8/ms`；
- base/half-dt 与四 phase outcome labels 一致。

关键反例是 `.825`：R0a 的瞬时 step 看起来安全，但原 225-ms、joint-occupancy-gated smooth M ramp 8/8 越出 transfer support。由此可见，承重结论必须来自 smooth path，而不是 step fiber。

### 4.3 当前动力学解释边界

安全解释是：固定 q 消除了旧 additive latch 中持续移动的 `A_exit(q)` 目标，使 M 有机会平滑穿越近似固定的 exit fiber；在 `.835–.845` 内，这条 fast geometry path 对双 dt 和四 phase 稳健。

这仍不是完整 slow-fast loop。当前 ramp 从 established CCO checkpoint 开始，并预置 regional latch；q 被人为 frozen，bath resource 也被固定。它证明“reserve-compatible geometry 有空间”，不证明真实 reserve 轨迹一定会进入、停留并退出该空间。

`.825` 的失败应写为 transfer-support/nonfinite oracle failure，不应升级为生理 runaway、Hopf、torus 或特定 bifurcation 类型证明。

## 5. 工程性问题

- R0a 产物：
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor/r0a_summary.json`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor/r0a_q_fibers.csv`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor/figures/mz_inhibitory_reserve_corridor_r0a.png`
- R0b formal confirm：
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_summary.json`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_source_cco.csv`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_step_threshold.csv`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_smooth_ramp.csv`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_recovery.csv`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/figures/mz_inhibitory_reserve_corridor_r0b.png`
- 后验 lower-boundary sentinel：
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_lower_boundary_sentinel.json`
  - `results/topic4_sef_hfo/mz_inhibitory_reserve_corridor_r0b/r0b_lower_boundary_sentinel.csv`

R0b 主 summary 已包含 sentinel 的简要 anchor-bracket synopsis，但主图没有展示后验 `.830/.8325` sentinel。凡详细引用 lower anchor bracket，仍应同时引用 sentinel JSON/CSV，不能声称主图已经展示该边界。

当前相关测试至少 22 个通过，已覆盖完整 Cartesian、配置驱动 margin/bracket、实际 safe-mask interval、zero fail-closed 与 sentinel anchor-bracket 等本轮修复合同；不再保留“尚缺 R0b gate 测试”的旧结论。

## 6. 最小修改路线

1. 以 `.835/.840/.845` 为 R1 下/中/上节点测 `Ubar_CCO`，不要用 `.830` 作为首轮 hold target；
2. 用锁定的 q-nullcline 反算每个 hold node 对应的 `q_res`，拒绝越界、单点精调或无法跨节点复现的 floor；
3. 只在 q-nullcline 能以余量落入 formal strip 后，接入固定背景事件 replay 与真实 `D_I/q_res` dynamics；
4. 保留 old no-reserve arm 与原 four-return M arm，R1 首轮不调 M、不扩大 E→E；
5. autonomous run 必须重新验收 entry、至少四 returns、无 support escape 的 exit、reset 与 early/late retrigger；fixed-q R0 不能替代这些门。

## 7. 下一步建议

**有条件 GO 到 R1 mapping，暂不 GO 到 autonomous lifecycle。**

下一节点应回答一个窄问题：在 `.835/.840/.845` 的 established CCO 上，周期平均 inhibitory use 是否能反推出物理合法、非精调且能把 q-nullcline稳定放进 formal strip 的 `q_res`。若不能，reserve 主路线按预注册 stop rule 关闭；若能，再用固定背景事件和原 M arm检验 entry—containment—exit，而不是扩大 E→E 或 M 参数网格。

本报告只审阅当前 current-based inhibitory-reserve 路线。它与并行 recurrent-conductance/E→E fast-gain shaping 线保持独立；本节点未触及 E→E 连接，也没有把固定 bath mask写成空间机制结论。
