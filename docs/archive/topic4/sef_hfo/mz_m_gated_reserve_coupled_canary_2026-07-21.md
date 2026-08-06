# 审阅结论：MZ M-gated reserve coupled center canary（R3）

日期：2026-07-21

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**预注册的 R3 coupled center canary 是 numerically clean 的 formal no-go：闭合 q-use feedback 后，第 5 次而不是第 6 次背景事件首先跨过 entry fold，因此必须按 stop rule 停在 center cell；但若改用真实 entry 对齐，第 5 次事件后到第 6 次事件前确实出现了 4 次 core/annulus 自主配对 returns，随后 M 把第 6 次 within-train immediate post-clonic challenge 压低。这是值得进入 R4 center closure 的描述性 lifecycle candidate，不是已经完成 recovery、same-basin return 或 retrigger 的 ictal lifecycle。**

当前 canonical status 为：

```text
R3_COUPLED_CLEAN_NO_GO_PREMATURE_EVENT5_ENTRY
```

当前 canonical decision 为：

```text
stop_before_grid_and_reassess_coupled_event_map
```

## 2. 完成程度

> **R3 coupled Segment-A center canary：100/100 完成；完整可恢复 lifecycle：约 45/100，未验收**

已完成：

- 只运行注册中心点 `tau_slow=90 s, q_hold=.8425, tau_fast=20 s, dt=.125 ms`；
- 在原六次固定背景事件下联合积分 fast regional state、q、persistence 与 M；
- 检查真实 entry ordering、latch、区域 returns、finite/support/bound、bath mask 与资源合同；
- 在 formal stop 后，没有运行其余 17 条 base/half-dt paths，也没有运行 Segments B-D、retrigger 或 ablation；
- 保存 summary、CSV、NPZ、2x3 机制图与中文 figure README；
- 本轮定向验证 `20 passed in 0.71 s`。

尚未完成：

- latch 的 true-to-false reset；
- M 的自然 release；
- 回到原 interictal fixed point 所在的同一 basin；
- recovery 期间的 early challenge 与自然恢复后的 late retrigger；
- fixed bath mask 之外的连续空间 recruitment、wavefront stall/annihilation；
- continuous field、full SNN 与 smooth Hopf/torus/limit-cycle 证明。

## 3. P0 / P1 关键问题

### P0：闭环后 entry ordering 改变，formal canary 必须判 no-go

注册 gate 要求前 5 次事件都不越过 regional entry fold，且第 6 次事件是第一次 entry。真实结果中，六次事件窗口的区域 q minimum 依次为：

```text
.8858801, .8745442, .8659458, .8594670, .8547559, .8452614
```

已知 entry fold 为 `q=.8558315843`，因此第 5 次事件窗口的 `.8547559` 已在 fold 下方；first entry time 为 `7620 ms`，对应 `entry_event_index=5`。pre-entry M 最大绝对值仍为 0，说明这不是 M 提前泄漏造成的假失败，而是 q 与 fast event response 闭环后的真实 event-map 改变。

**为什么严重**：R3 scalar/path 层的 event-6 parity 来自 fixed sensor / feed-forward replay。它不能保证 q 回馈 fast rates 后仍保持同一 entry ordering。若忽略这个差异继续跑 grid，就会把两种不同 onset protocol 混成一个机制 corridor。

**怎么改**：保留 formal clean no-go，不用调 q mapping 把入口重新推回 event 6；下一节点改为 actual-entry-aligned 的 R4 center closure，先验证同一条 event-5 entry 轨迹能否完成 finite fast exit、真实 reset、M release、same-basin return 与 late retrigger。

### P0：4 次 returns 是 actual-entry-aligned 诊断，不能覆盖 formal failure

formal 分析按原合同从最后一次事件后 `11035 ms` 开始，因此 formal pulse-free returns 为 core `0`、annulus `0`，`at_least_four_pulse_free_returns_each_region=false`。这与注册 gate 一致。

但第 5 次事件在 `7531-7551 ms` 后已经触发真实 entry。排除 100 ms evoked response，从 `7651 ms` 重新对齐，到第 6 次事件 `10915 ms` 前，NPZ 轨迹中出现 4 组无外部脉冲的区域 returns：

| return | core (ms) | annulus (ms) |
|---:|---:|---:|
| 1 | 8826.344 | 8833.603 |
| 2 | 9537.764 | 9544.708 |
| 3 | 10170.460 | 10177.268 |
| 4 | 10805.147 | 10811.904 |

这些 returns 说明 center trace 中已有 bounded recurrent bursting / clonic-like structure，而不是单一高率平台。它们发生在真实 entry 之后、下一次外驱之前，因此可以作为 autonomous paired returns 报告；但它们不满足原来“final pulse 后 returns”的 formal gate，`formal_acceptance_override=false` 必须保持。

### P0：fast activity 有限退出不等于 slow lifecycle 已闭环

persistence latch 在 `10233.375 ms` set。M 随后升到 `.2409702`，对应 additive M 最大值 `.3855523 mV`。第 6 次事件到达时 additive 仍为 `.3855523 mV`，regional fast peak 仅 `6.778 Hz`，没有再产生 section return；`11035 ms` 时 core/annulus fast rates 已降至约 `.559/.531 Hz`，数值有限且无 support、state-bound 或 nonfinite failure。

但是 `20 s` 末态仍是：

```text
q_core=q_annulus=.8545557 < q_safe=.885
m_core=m_annulus=.2409702
A_core=A_annulus=.3855523 mV
latch=[true, true, false]
persistence_core/annulus approximately 8.79e-7 / 6.52e-7
```

所以当前只证明了一个 finite low-rate fast exit candidate。latch 尚未 reset，M/A 尚未 release，也没有证明状态回到原来的 stable interictal root。不能把 `clean_low_onset=11035 ms` 或 `finite_clean_exit_without_support_or_bounds_failure=true` 写成完整 termination-and-recovery lifecycle。

### P1：第 6 次事件不是 early-recovery probe

第 6 次事件属于预先固定的同一六事件 background train，发生在第 4 次 actual-entry-aligned return 后约 `103 ms`，且此时 latch 和 M 仍处于 active plateau。它只能命名为 **within-train immediate post-clonic challenge**。

该响应被压到 `6.778 Hz` 支持“即时再次进入被抑制”的描述性观察，但它没有经过 latch reset 或 M release，也没有注册 recovery delay。因此不能称为 early-recovery retrigger test，更不能与未来 late retrigger 作恢复曲线比较。

### P1：当前空间结果仍是 three-patch fixed-mask diagnostic

core 与 annulus 的 4 次 returns 时间相差约 `6.8-7.3 ms`，bath peak 为 `18.915 Hz`，低于注册上限 `20 Hz`。但是 bath q 被 depletion mask 强制固定在 `.90`，最大数值误差仅 `2.38e-8`；这不是自组织产生的 spatial containment。

**怎么改**：R4 仍只做 center temporal closure。只有 same-basin recovery 与 late retrigger 成立后，才允许进入 coarse continuous field，并把 local recruitment、front stall 和 bath release 作为新的独立空间 gate。

## 4. 科学性问题与动力学反思

### 4.1 什么做对了

这一步第一次把 R3 的 q–M slow path 放回真实 regional fast model，使 q 的变化能够改变 fast rates、inhibitory use、occupancy、persistence 与后续 M build。结果没有数值 runaway，也没有变成平坦高率平台：实际 entry 后出现 4 次 core/annulus 配对 returns，之后 active M 把下一次 within-train challenge 压低。

因此，先前“q 提供 entry、bounded regional orbit 提供内环、M 提供 fast exit、M-gated q recovery 尝试完成外层 reset”的分工仍有部分数值支持。右侧的 clonic returns 不是由额外 E-E 或 conductance 修改产生，也没有借用并行工作线的参数。

### 4.2 什么还不够：feed-forward nullcline 推断没有包含 event map 的闭环增益

冻结 `(m,Ubar)` 时，R3 的 q-nullcline仍只有一个稳定点：

\[
q^*(m,\bar U)=
{r_{rec}(m)q_0+(\bar U/\tau_D)q_{res}
\over
r_{rec}(m)+\bar U/\tau_D},
\qquad
\lambda_q=-\left[r_{rec}(m)+{\bar U\over\tau_D}\right]<0.
\]

这部分分析本身没有错，但它把 `Ubar` 当成外给量。真实 coupled event map 中，q 下降会放大下一次 fast response，fast response 又提高 inhibitory use U，U 再进一步降低 q。第 5 次事件的提前 entry 正是这条正反馈在离散 event-to-event 尺度上的表现：

```text
q lower -> event response/use larger -> q depletion larger -> fold crossed earlier
```

因此，R3 scalar/path oracle 对 post-entry recovery 仍是有用的 necessary screen，却不能预测 coupled onset ordering。下一阶段必须围绕真实 entry 定义，而不能继续要求 coupled trace复刻 feed-forward sensor 的事件编号。

### 4.3 当前更合理的动力学图景

这条 trace 支持的是一个 hybrid slow-fast candidate，而不是已经证明的 Hopf 或 torus：

1. 固定背景事件逐步降低 q，第 5 次事件让 fast state跨过 entry fold；
2. regional fast subsystem 在 q 较低且 M 尚未升高时产生 4 次 bounded returns；
3. persistence latch set 后，M/additive A 把 fast subsystem推出 bursting window；
4. 随后 fast state进入有限低率区，但 q、latch 与 M 仍停在未恢复的慢状态；
5. 外层慢环是否能通过 q recovery、latch reset 与 M release 回到同一 interictal basin，尚未计算。

所以当前可以画成“内环已出现、外环只完成前半段”；还不能声称大环闭合，也不能从单 trace 推断 smooth bifurcation type。

### 4.4 与并行 E-E / conductance 工作线的独立性

本 canary 没有修改：

- `W_EE`、E->E kernel、delay 或 recurrent saturation；
- conductance-based membrane equation；
- presynaptic relay；
- M 的 `225 ms/12 s` 时间尺度、Amax 或 persistence threshold；
- R2 `q_res/tau_D` mapping；
- three-patch geometry 与 fixed bath-resource mask。

因此这一结果独立回答“现有 bounded CCO 的 inhibitory slow path 能否闭合 entry-maintenance-exit-reset 外环”。下一步仍应保持这条边界，不能通过借用并行线的 E-E 或 conductance 参数来修复 event ordering。

## 5. 工程性问题

### 已通过的工程合同

- 13 个上游输入以 SHA-256 锁定；
- default-off implementation 保留旧 slow RHS/integration 语义，并显式返回 `final_latch_state`；
- center only、单进程、单 BLAS 线程，stop rule 生效；
- 只执行 `1/18` 注册路径，其余 17 条未启动；
- `support_violation_count=0`、`state_bound_violation_count=0`，无 nonfinite；
- trace 估算 `2,620,131 bytes`，压缩 NPZ 为 `573,331 bytes`，低于 `64 MiB`；
- runtime `93.712 s`，peak RSS `321,064 KiB`（约 `313.5 MiB`），低于 `1.5 GiB`；
- 本轮定向测试：

```text
python -m pytest -q -p no:cacheprovider \
  tests/test_topic4_mz_spatial_autonomous_latch.py \
  tests/test_topic4_mz_m_gated_reserve_coupled_canary.py

20 passed in 0.71s
```

### 仍需保留的工程边界

- `actual_entry_aligned_diagnostic` 是从同一 canonical NPZ 重分类出的描述性层，不改变 formal status；
- 不能把 final-pulse analysis 的 `0 returns` 与 actual-entry-aligned 的 `4 returns` 混成同一个验收字段；
- clonic bursting 中存在超过 `250 ms` 的 inter-burst low interval，R4 不能仅用“连续低率 250 ms”定义 termination；
- R4 应用“最后一次 autonomous return 后不再 return + finite low fast state + latch/reset state”组合定义退出；
- analytic recovery bridge 若使用，只能在 zero-U low interval 内，并必须配 full-fast sentinels；
- fixed bath mask 与 three-patch aggregation 继续作为明确的模型边界。

核心产物：

- summary：`results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/coupled_canary_summary.json`
- outcome：`results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/segment_a_center_canary.csv`
- canonical trace：`results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/segment_a_center_canary_trace.npz`
- 主图：`results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/figures/mz_m_gated_reserve_coupled_canary.png`
- 图说明：`results/topic4_sef_hfo/mz_m_gated_reserve_coupled_canary/figures/README.md`
- 锁定设计：`docs/superpowers/specs/2026-07-20-topic4-mz-m-gated-reserve-coupled-canary-design.md`

## 6. 最小修改路线

1. 将 R3 formal 结果锁定为 `PREMATURE_EVENT5_ENTRY` clean no-go，不运行剩余 17 条 grid paths；
2. 新建 actual-entry-aligned R4 center spec，保留当前 canonical q mapping 与所有 fast/E-E/conductance 参数；
3. 以第 5 次实际 entry 为 onset，只接受至少 4 次 paired autonomous returns、finite numeric fast exit 与第 6 次 within-train challenge suppression；
4. 从 `20 s` 的真实末态继续 protected q recovery，不手动重置 q、p、M 或 latch；
5. 在 q 接近 `.885` 前后用 full-fast sentinels，要求现有状态机产生一次真实 latch true-to-false reset；
6. reset 后让 M 按原 `12 s` 时间常数自然 release，并验证 low-rate、low RHS、向原 interictal root 收敛；
7. 只有同一 basin recovery 成立，才从恢复中的早期 checkpoint 与自然恢复末态分别做注册 challenge / late retrigger；
8. R4 center closure 失败则停止本线，不通过重标 q、修改 M 时序或借用 E-E/conductance 机制救援。

## 7. 下一步建议

**GO 到 R4 actual-entry-aligned center closure；NO-GO 到 R3 grid、continuous spatial field、full SNN 和完整 lifecycle claim。**

当前最安全的结论是：

> R3 的 full coupled center trace 否定了 feed-forward sensor 所预测的 event-6-first ordering：q-use feedback 使第 5 次事件在 `7620 ms` 首先跨入 ictal branch，因此 formal canary 必须 clean no-go。与此同时，按真实 event-5 entry 对齐，可在下一次外驱前观察到 4 次 core/annulus autonomous paired returns，随后 active M 将第 6 次 within-train immediate post-clonic challenge 压低到 `6.778 Hz`。这说明模型已有 bounded fast bursting 与 finite low-rate exit 的候选前半环；但 latch 仍开启、q 未回到 `.885`、M/A 未释放，same-basin return 与 late retrigger 尚未测试。因此下一步只应做不重标 q、不碰 E-E/conductance 的 R4 center closure。
