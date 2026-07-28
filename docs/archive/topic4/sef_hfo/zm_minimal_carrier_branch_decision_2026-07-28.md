# Z/M minimal-carrier branch decision（2026-07-28）

## 0. 一句话判决

**本轮确认了访问到的慢状态上存在一个冻结慢变量后的源空间 sustained tonic
carrier，top-level verdict = `carrier_at_visited_states`；但 observation-space
身份被真实 returning-event reference 缺失阻断，entry 边界未解，现有慢坐标没有给出
可用 offset，动态 Z/M 9/9 runaway，因此 bounded ictal oscillation、可恢复 lifecycle
和 exit actuator 均未建立。**

这不是上一轮“carrier 不存在”的简单翻转。上一轮检验的是自然 Z/M+\(S_G\) 轨迹和
revised carrier gate；本轮用 exact checkpoint/resume 把快系统与慢变量漂移分开，
问的是：**自然轨迹访问过的慢状态若被固定，下面是否已经存在可持续快态。**答案是
“源空间存在”，但“自然慢流是否进入并离开它”仍是否定或未解。

机器判决：
`results/topic4_sef_hfo/zm_branch_decision/branch_verdict.json`。

---

## 1. 科学问题与范围

本轮执行 revision 3.1 branch-decision spec：
`docs/superpowers/specs/2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md`。

它没有构造新的 E→E 机制，也没有把旧 q_I/g_K 沙盒迁回主线。衬底仍是正确的
per-neuron Z/M 各向异性二维 E/I SNN；\(z\) 表示抑制资源耗竭/恢复，\(m\) 表示慢适应，
\(S_G\) 是 recurrent excitation 上的共享除法抑制。

本轮依次回答四个问题：

1. 自然 Z/M 轨迹访问过的慢状态下，冻结慢变量后是否存在有界、持续的快态 carrier；
2. 该 carrier 的源空间和虚拟电极形态是否足以叫 ictal carrier；
3. \(Z/M/S_G\) 中哪些慢方向控制 entry 和 offset；
4. 自然动态 Z/M 是否真的把系统带入并带出这个 carrier。

---

## 2. 实现与验收合同

### 2.1 状态与可复现性门

- `state_gate = ok`；
- `exact_resume = ok`；
- seeds 1、3、4 均提供合格 anchor；
- snapshot/restore 后继续轨迹通过 paired-future-noise 合同；
- 默认路径和 checkpoint idle 路径保持既有 parity；
- 最终 adjudicator 重新执行 19 组相关测试，`202 passed, 4 warnings`。

因此本轮结论来自同一 SNN 状态的可重复 fork，不是从相近参数重启后拼接出来的轨迹。

### 2.2 Carrier fork

对自然轨迹的 `bounded_early`、`bounded_mid`、`bounded_late` 以及
trough/rising/peak 快相位保存状态，再比较：

- `freeze_all`：冻结全部慢变量，只保留快 E/I SNN 演化；
- `freeze_z`；
- `freeze_zm`；
- `freeze_zsg`；
- 对照的动态 replay / dynamic-z-only。

每个 cell 使用 paired noise replay/resample；carrier window 需要跨 seed、快相位和相邻
慢状态 bin 的兼容支持，不能由单条漂亮轨迹通过。

### 2.3 Boundary audit

- entry：在 onset-context \(M/S_G\) 固定时扫描 conditional actual-field \(Z\) slice；
- offset：分别扫描 `M_alone`、`M_SG`、`M_Z_recovery`，并增加完整
  `dynamic_ZM` 检验；
- offset 成功只认从 carrier basin 进入持续 low/rest basin；短暂降率或零星
  `rest_return` 不等于 recovery；
- 任一非单调、未 bracket、覆盖不足或缺少 seed-wise 支持均 fail closed。

---

## 3. Carrier：源空间存在，但它是 frozen tonic carrier

### 3.1 Positive result

`freeze_all` 得到 `carrier_window`：

- 9 个 positive cells；
- 覆盖 seeds 1、3、4；
- 覆盖 rising/peak 两个快相位；
- 覆盖相邻 `bounded_early` 与 `bounded_mid` bins；
- seeds 1、3 提供跨相位、跨相邻 bin 的兼容 witness；
- seeds 1、3 的 `bounded_mid__peak` 在 20 s 长时和独立 \(dt/2\) 8 s 确认中均通过。

因此最小 positive subsystem 是 `carrier_fast_only`：自然轨迹已经访问到一个慢状态区域，
在该区域冻结慢变量后，快 E/I 子系统可以持续停留在有界高活动态。

### 3.2 为什么不能写成 bounded ictal oscillation

确认轨迹在 25 ms 粗粒度下均为 `tonic_at_25ms`。20 s 确认中 population-rate CV
约为 \(9.3\times10^{-4}\)，更像几乎平坦的 tonic branch，而不是已经证明的
bursting orbit 或 limit cycle。

细粒度源空间分类也没有跨 seed 统一：

- seed 1、3：`asynchronous_or_irregular_candidate`；
- seed 4：`phase_staggered_periodic_candidate`；
- aggregate status：`class_disagreement`。

因此 modal/operator audit 按预注册规则停止，不能从现有数据宣称固定点、Floquet
周期轨道或统一的 ictal rhythm class。

### 3.3 Observation layer 被阻断

E1146 geometry 和 6 个 early-ictal raw windows 可定位，但缺少 canonical real
returning-group-event SEEG-window index：

- early ictal windows：6；
- returning group-event windows：0；
- contract 要求：至少 3 个；
- generic seizure-free background 和模型自身 interictal trace 均被禁止替代。

所以 `observation_space_carrier = blocked_reference_artifacts`。虚拟电极上的频谱或持续
能量只能作为 descriptive morphology，不能通过 empirical carrier identity gate，也不能
生成 paper-ready lifecycle 图。

---

## 4. Functional rank：当前是缺证据，不是低秩结论

Z 和 \(S_G\) 方向的 probe 可计算，但 \(M\) 的 central physical pair 在三个 seeds
都不完整：

- seed 1：\(M\) 仅 2/6 rows valid；
- seed 3：\(M\) 仅 2/6 rows valid；
- seed 4：\(M\) 0/6 rows valid。

因此 `effective_rank = no_evidence_incomplete_central_pairs`，没有任何 complete
seed microstate。当前不能写“slow control 是 rank-1”或“rank-2”；只能写完整 Z/M/\(S_G\)
functional rank 在当前物理可行 displacement 下不可识别。

---

## 5. Entry：conditional Z boundary 未解

entry audit 覆盖 seeds 1、3、4，共 23 rows。所有 sampled \(q\) 点的 posterior median
均高于 0.5，但曲线被判为 `nonmonotonic`，没有有效 \(q_{1/2}\) 或 bootstrap CI；自然
轨迹也没有通过已解析边界的方向 crossing。

因此：

- verdict = `conditional_Z_entry_boundary_unresolved`；
- 不能写 global Z sufficiency；
- 不能把自然轨迹出现 onset/runaway 等同于已找到 entry bifurcation；
- 这条 conditional slice 也不提供 offset、recovery 或 lifecycle 证据。

---

## 6. Offset：现有慢坐标没有给出安全退出

最终 offset verdict 为 `no_evidence`，诊断为
`static_M_Z_recovery_curve_nonmonotonic_dynamic_ZM_all_runaway`。

### 6.1 Static families

| family | 边界状态 | low-basin persistence | 可写结论 |
|---|---|---:|---|
| `M_alone` | `unbracketed` | 0 | 已测试范围内未找到可达 offset |
| `M_SG` | `unbracketed` | 0 | 加 \(S_G\) 未建立 offset basin |
| `M_Z_recovery` | `nonmonotonic` | 0 | 局部降率/返回不能组成稳定边界 |

三个 families 均没有 basin coexistence，没有 bootstrap-supported \(q_{1/2}\)，也没有
自然方向可达的 crossing。`M_Z_recovery` 在 \(\lambda=1\) 附近出现从 carrier 分类离开的
现象，但修复后逐 cell 重跑显示 raw runner 没有进入 `dead_in_rest_basin`；事件只被
classifier 标成 `rest_return`，不能升级为稳定低盆地。

### 6.2 Dynamic Z/M

完整动态检验覆盖：

- 3 seeds；
- 每 seed 3 个 paired replicates；
- 共 9/9 trajectories。

结果为：

- offset success：0/9；
- end reason：9/9 `runaway`；
- posterior median \(P(\mathrm{offset})=0.0243\)；
- \(P[P(\mathrm{offset})>0.8]=4.57\times10^{-8}\)。

这说明自然 Z/M 慢流在当前测试中不是把 frozen carrier 安全带回低态，而是继续把系统
推向 runaway。该结果直接阻止把 frozen carrier 写成自然存在的可恢复 ictal attractor。

### 6.3 诚实性修复

执行中发现并修复了两条会制造假阳性的路径：

1. 原 adjudicator 可能在 dynamic grid 未按 seed/replicate 完整覆盖时聚合判定，并把
   ambiguous/nonmonotonic surface 默认路由到 Phase 3；
2. continuation summary 曾优先使用 classifier 的 `rest_return`，可能遮住 raw runner
   `end_reason`，使短暂返回被误读为 stable rest-basin evidence。

修复后：

- dynamic grid 必须 3 seeds × 3 replicates 完整；
- offset 逐 seed 计算 posterior；
- raw `end_reason` 优先，classifier reason 单独保留；
- ambiguous surface 一律 `no_evidence`；
- 8 个受影响的 `M_Z_recovery, \lambda=1` cells 强制重跑并合并；
- 没有发现被旧分类器遮住的 stable low basin。

因此最终 negative offset verdict 不是缓存或分类字段造成的假阴/假阳。

---

## 7. 最终科学口径

### 7.1 当前可以写

1. 正确 Z/M SNN 的自然轨迹访问过一个慢状态区域；冻结慢变量后，快 E/I 子系统支持
   多 seed、多快相位、相邻 slow-bin 的 sustained source-space carrier。
2. 该 carrier 的 population-rate envelope 是 tonic，而不是已经证明的 ictal bursting
   或 limit cycle。
3. carrier 的细粒度源空间 rhythm class 跨 seed 不一致。
4. conditional \(Z\)-entry boundary 未解析。
5. 已测试的 \(M\)、\(M+S_G\)、\(M+Z\)-recovery 坐标没有给出可达、可复现的
   offset basin；完整动态 Z/M 为 9/9 runaway。
6. 当前 top-level verdict 是 `carrier_at_visited_states`，不是 lifecycle candidate。

### 7.2 当前不能写

- “已经存在 bounded ictal oscillation/limit cycle”；
- “自然 Z/M 轨迹进入并退出同一个 ictal attractor”；
- “虚拟 SEEG 已匹配真实 early-ictal carrier”；
- “\(M\) 或 \(S_G\) 已构成 seizure offset mechanism”；
- “已经恢复到原 returning interictal-event regime”；
- “functional rank 已证明为一维或二维”；
- 任何 paper-ready Figure 5 lifecycle claim。

---

## 8. 工程与产物

- final gate：`202 passed, 4 warnings in 106.60s`；
- state inventory、checkpoint/restore、noise-bank、boundary、morphology、rank、verdict
  均有 machine-readable provenance；
- offset contract audit 通过，required seeds = 1/3/4；
- durable resource log 最低 `MemAvailable = 100136 MB`，未触发 OOM guard；
- 本线保持与 E→E-modification worktree 独立。

核心产物：

- 总判决：`results/topic4_sef_hfo/zm_branch_decision/branch_verdict.json`
- carrier matrix：`results/topic4_sef_hfo/zm_branch_decision/figures/carrier_subsystem_matrix.png`
- native morphology：
  `results/topic4_sef_hfo/zm_branch_decision/figures/native_confirmation_spatiotemporal_morphology.png`
- source rhythm：
  `results/topic4_sef_hfo/zm_branch_decision/source_rhythm/source_rhythm_summary.json`
- functional rank：
  `results/topic4_sef_hfo/zm_branch_decision/effective_rank/effective_rank_summary.json`
- entry：
  `results/topic4_sef_hfo/zm_branch_decision/boundaries/entry/entry_boundary_summary.json`
- offset：
  `results/topic4_sef_hfo/zm_branch_decision/boundaries/offset/offset_boundary_summary.json`
- phase status：
  `results/topic4_sef_hfo/zm_branch_decision/figures/phase_completion_status.png`

---

## 9. 下一步与停止点

本 spec 到这里应 **stop-and-review**：

1. 不运行 Phase 3 driver comparison；
2. 不实现 \(P/A\) exit actuator；
3. 不做 \(M\) calibration；
4. 不把 frozen carrier 包装成 lifecycle。

下一轮需要单独锁定新的机制问题：为什么自然慢流在已经存在 frozen tonic carrier 的
情况下仍走向 runaway，以及怎样产生一个既改变快系统稳定结构、又能在退出后允许
returning interictal events 重现的慢反馈。任何新机制都应继续保留原 Z/M、各向异性
空间连接和虚拟 SEEG readout，并以“自然 entry → bounded non-tonic ictal state →
offset → returning interictal events”作为最终验收，而不是只优化某一个 frozen state。
