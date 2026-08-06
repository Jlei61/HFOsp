# 审阅结论：MZ actual-entry-aligned regional lifecycle closure（R4）

日期：2026-07-21

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**R4 第一次在不改 E→E、conductance 或 relay 的前提下，把同一个 regional center state 数值闭合成“event-5 entry → 4 次 bounded paired bursts → finite exit → 真实 latch reset → M 自然释放 → 同一间期 basin → early/late 同挑战分离”的完整外环；但它仍是 fixed-bath、three-patch、离散 hysteretic latch 驱动的 center-point hybrid lifecycle，不是平滑 Hopf/torus、连续空间波前、零输入自发发作或 full SNN 验证。**

当前 canonical status 为：

```text
R4_ACTUAL_ENTRY_REGIONAL_HYBRID_LIFECYCLE_CENTER_SUPPORTED
```

当前 canonical decision 为：

```text
unlock_fully_coupled_q_map_recalibration_and_local_robustness_only
```

## 2. 完成程度

> **R4 center closure：100/100 完成；最终时空可恢复发作模型：约 60/100，尚未验收**

已完成：

- hash-lock R3 coupled trace、summary 与 config，并从同一轨迹重新确认 event-5 actual entry、4 次 core/annulus paired returns、finite exit 与 event-6 within-train suppression；
- 从真实 `20 s` 末态继续 q/p/M/latch，不人工 reset 任一状态；
- 用三段 500-ms full-fast sentinel 验证 active-latch low interval 中 use/occupancy 为 0 后，才使用解析 q-recovery bridge；
- 由现有 state machine 在 `q>=.885, p<=.03, rE_fast<=5 Hz` 时完成一次且仅一次 true-to-false reset；
- reset 后让 M 以原 `12 s` 时间常数自然释放，并在 A=`.10/.02/.002 mV` 处插入 full-fast sentinels；
- 在 `q>=.899` 后连续完整积分 4 s，验证低率、低 fast-RHS、向原 LLL root 收敛及无 latch reactivation；
- 对 protected checkpoint 与自然恢复末态施加完全相同的六事件 challenge，并使用同一个 response-excluded lifecycle classifier；
- 从共同 R3 checkpoint 开始的 recovery 与 recovered-challenge 在 `dt=.125/.0625 ms` 下标签和关键 crossing timing 一致；R0–R4 扩展回归 82 个测试通过；
- canonical run 单进程、单 BLAS，`654.383 s / 175,412 KiB`，未接近 OOM 门。

尚未完成：

- fully coupled event-map 的局部参数 corridor 与 causal ablation；
- 把离散 latch 替换或逼近为连续慢动力学后仍保留 lifecycle；
- bath 资源自由演化下的 local recruitment、front stall/annihilation 与 refractory wake；
- continuous field、full SNN、跨 seed 稳健性；
- 零输入噪声自发 onset；
- Hopf、SNIC、torus 或 smooth full-system limit cycle 的正式分岔证明。

## 3. P0 / P1 关键问题

### P0：这是 hybrid reset closure，不是平滑分岔已经形成

在 latch 开启且 fast occupancy 已归零时，当前 pooled M 方程为 `dm/dt=0`；q 则按 M 选择的恢复率向 `q0` 回升。只有 q、p 与 fast rate 同时到达 reset guard，离散状态 `L:1→0` 才改变向量场；此后才有 `dm/dt=-m/tau_M`。

因此外环依赖一个明确的离散 hysteresis switch：

```text
L=1, occupancy=0: M frozen, q recovers
q>=.885 and p<=.03 and r_fast<=.005: L 1->0
L=0: M decays, q returns toward q0
```

**为什么严重**：这个结构可以产生可靠的 relaxation lifecycle，却不能直接叫 Hopf、torus 或平滑极限环；active-low 段还包含一条 M 的中性方向。若论文机制目标要求连续生理状态变量而不是 operational controller，必须在下一层 falsify/replace 这个 latch。

**怎么改**：下一步先做局部 robustness 与 causal ablation，确认闭环不是单点巧合；随后单独测试 continuous persistence / two-pool inhibitory recovery，要求在无离散 bit 的情况下保留 entry、4 returns、finite exit、same-basin recovery 和 retrigger separation。

### P0：early challenge 抑制的是 autonomous lifecycle，不是所有电响应

protected checkpoint 本来已经 `q<q_fold`，因此不能用“是否再次跨 q fold”判 retrigger。R4 的共同 classifier 改为逐个 event 的 response-excluded window 搜索至少 4 次 paired returns，并同时要求 finite supported low tail。

两套 dt 的 early challenge 都是：

- `actual_entry_lifecycle_candidate=false`；
- paired autonomous returns = 0；
- 最后 250 ms 为持续低态；
- 无 bounded-high/runaway tail。

但第 6 次刺激仍各自产生一次 core/annulus section crossing：base-dt 为 `10954.08/10963.92 ms`，half-dt 为 `10953.82/10963.62 ms`。

**为什么严重**：若把这一结果写成“early retrigger 完全被抑制”或“electrical silence”，会把一次可诱发响应误当成没有响应。当前安全结论只是：**protected state 对完整 ictal lifecycle refractory，但不是对每个 evoked response 静默。**

### P1：空间信息仍由 three-patch 与 fixed bath mask 强加

source lifecycle 中 core 与 annulus 四次 return 的 lag 约 `6.8–7.3 ms`，说明 regional fast inner orbit 保留了粗空间顺序；但是 bath q 被固定在 `.90`，没有让资源耗竭、全局抑制或波前自行决定传播边界。

**怎么改**：只有局部 robustness 通过后，才进入 coarse continuous field。新空间 gate 必须把 bath q 解冻，并分别验证 local recruitment、annulus handoff、front stall/annihilation、postictal refractory wake；不能用 three-patch lag 代替 wavefront。

### P1：onset 仍由固定六事件序列触发

R4 证明的是同一短刺激序列可在恢复前后分别落入不同动力学结果；没有测试零外驱噪声下的自然 basin crossing。这里的“自发”仅指 entry 后 response-excluded interval 中的自主 returns，不是 zero-input spontaneous seizure onset。

### P1：恢复到 late-retrigger checkpoint 约需 312 s，尚未生物标定

base-dt 的 latch 在约 `74.1 s` reset，但满足注册的 `A<=.002 mV, q>=.899` 并完成 4-s full check
要到约 `311.7 s`。这个时间主要由 `tau_slow=90 s` 和靠近 `q0=.90` 的 `.899` same-basin
阈值共同决定。它可以作为分钟级 postictal refractory 的模型候选，但当前没有用真实 seizure
recovery 数据标定，也没有扫描中间 challenge delays。

**怎么改**：local robustness 阶段同时报告 continuous recovery curve，在多个预注册 delay 上重复同一
challenge，区分“reset 已发生”“M 已释放”“q 接近 q0”和“lifecycle susceptibility 恢复”四个时间点；
在这之前不能把 `311.7 s` 写成生理 refractory period。

### P1：same-basin 是 operational gate，不是严格 basin continuation

当前 gate 同时要求 `q/A/p` 回到注册邻域、4 s full integration 保持低率、fast RHS 足够小、且到原
LLL root 的 fast-state 距离下降。这足以排除 reset 后仍停在高平台或数值假稳态，但没有对整个
basin boundary 做 continuation，也没有证明恢复轨迹与最初状态逐点同宿。

**怎么改**：下一阶段从多个恢复 checkpoint 做 state-fork，并在 frozen slow coordinates 上继续低态
branch 与 basin boundary；在这之前使用“回到原 LLL basin 的 operational neighborhood”，不写严格
global basin proof。

### P1：half-dt 收敛不覆盖原始 R3 Segment A

原始 event-5 entry 与四次 source returns 来自 hash-lock 的 `.125 ms` R3 trace。R4 的 `.0625 ms` arm
从同一个 `20 s` checkpoint 开始，独立重跑 checkpoint 后 recovery；在自然恢复后，它又完整重跑同一
六事件 challenge，并复现 event-5 entry、四次 paired returns 与有限退出。因此当前 half-dt 证据覆盖
的是 **checkpoint 后外环 + recovered lifecycle**，不是原始 R3 Segment A 的 half-dt 重积分。

**怎么改**：local robustness 若继续使用 source Segment A，需把其 base/half-dt provenance 单列；不能用
`source_phenotype_common_hash_locked=true` 代替 source trace 的 half-dt convergence。

## 4. 科学性问题与动力学反思

### 4.1 什么做对了，为什么这次能够闭环

这条线做对的不是“又加了一个负电流”，而是把四个动力学任务分开：

1. q 的事件间累积耗竭把 fast subsystem 推过 localized entry fold；
2. 既有 regional CCO 提供有界 bursting 内环，而不是高率饱和平台；
3. persistence/recruitment latch 只在 core 与 annulus 已共同进入持续活动后打开 M；
4. additive M 把 fast state 推出 CCO window；M-gated q recovery 随后只负责达到 safe reset，reset 后 M 自然释放并回到 LLL basin。

这避免了旧 additive-only 设计的主要矛盾：同一即时负电流若足以 prevention，通常就不能延迟到 4 次 burst 后再 termination；若足够弱，又只得到 plateau。当前 M 的进入由 history/latch 延迟，q recovery 与 M additive exit 的职责也被 causal 分离。

### 4.2 q nullcline没有制造 Hopf；它只移动唯一稳定点并改变收缩速度

冻结 `(m,U)` 后：

\[
\dot q=r_{rec}(m)(q_0-q)-{U\over\tau_D}(q-q_{res}),
\qquad
r_{rec}(m)={1-m\over\tau_{slow}}+{m\over\tau_{fast}}.
\]

唯一 q-nullcline 为：

\[
q^*(m,U)=
{r_{rec}(m)q_0+(U/\tau_D)q_{res}
\over r_{rec}(m)+U/\tau_D},
\qquad
\lambda_q=-\left[r_{rec}(m)+{U\over\tau_D}\right]<0.
\]

因此 M 增大时，q fixed point 向 `q0` 上移且 q 方向更快收缩；这一维不会自己产生第二稳定点或 Hopf。真正把外环闭合的是 fast fold crossing、bounded CCO、M exit 和 latch guard 之间的 hybrid composition，而不是 q-nullcline发生了振荡分岔。

### 4.3 coupled event map 解释了为什么 feed-forward R3 会提前一个事件失败

R3 scalar/path oracle 固定 fast sensor，因此只看到 q 在给定 U(t) 下的变化。真实 coupled map 多了一条正反馈：

```text
q lower -> inhibition weaker -> next event response/use larger
        -> q depletion larger -> entry fold crossed earlier
```

实际六个 event window 的 minimum q 为：

```text
.885880, .874544, .865946, .859467, .854756, .845261
```

第 5 次已低于 fold `.8558316`，所以 event-6-first 的 feed-forward calibration 不能保留。R4 没有为了恢复旧编号而重新拟合 q；它按实际 entry 对齐，因而把“onset calibration 错一事件”与“模型是否存在完整 lifecycle”两个问题分开。这是本轮最重要的方法学修正。

### 4.4 数值上闭合的是“大环套小环”的 relaxation 图景

source trace 在 event 5 后出现 4 次 core/annulus paired returns：

| return | core (ms) | annulus (ms) |
|---:|---:|---:|
| 1 | 8826.344 | 8833.603 |
| 2 | 9537.764 | 9544.708 |
| 3 | 10170.460 | 10177.268 |
| 4 | 10805.147 | 10811.904 |

最后 joint downcross 为 `10890.699 ms`，持续低态从 `10922 ms` 开始。base-dt 中真实 latch reset 为 `74117.302 ms`；自然释放与 full sentinels 后，`311720.875 ms` 的末态为：

```text
q=.89904347, p≈1.05e-175, M≈6.06e-10, A≈9.70e-10 mV
rE_max=.823154 Hz, rE_fast_max=.823156 Hz
fast RHS norm=1.56e-10 /ms
distance to original root: .0003005 -> .00004170
```

恢复后的 late challenge 再次在 event 5 entry，并产生 4 次 paired autonomous returns 后有限退出；checkpoint-forward half-dt 给出相同标签，关键时间均在 20 ms 容差内。这个结果支持“fast bursting 小环 + slow hybrid recovery 大环”的 center-point existence proof，但不决定平滑系统的分岔类型。

### 4.5 与并行 E→E/conductance 工作线保持独立

R4 没有修改：

- `W_EE`、E→E kernel、delay 或 recurrent saturation；
- conductance membrane；
- presynaptic E→E relay；
- M 的 `.225 s / 12 s` 时间尺度、Amax 与 persistence threshold；
- R2/R3 `q_res/tau_D` mapping；
- P3 geometry 与 fixed bath mask。

所以两条线回答不同问题：并行线改变 fast recurrent/membrane structure，寻找更自然的 ictal attractor；本线测试现有 bounded regional orbit 能否由 inhibitory slow path 完成 entry–exit–reset。后续不能把并行线参数借来“修复”本线的 event ordering。

### 4.6 解析 bridge 的独立 frozen-branch 数值审计

为检查 sparse full-fast sentinels 之间是否可能隐藏 low branch 失稳，独立只读审计调用现有
`find_regional_equilibria()` / `regional_fast_jacobian()`，沿同一解析路径做了两组 continuation：

- protected path：固定 `A=.385552 mV`，从 `q=.854556` 到 `.884500` 取 9 点；9/9 都连接到稳定低率根，regional Emax 从约 `.926` 降到 `.604 Hz`；
- released path：从 `(q=.885121,A=.372943 mV)` 到 `(q=.899,A≈0)` 取 13 点；13/13 都连接到稳定低率根，regional Emax 约 `.609-.910 Hz`；
- 两条路径所有点的 leading real 均约 `-.0125/ms`。

这个 post-hoc audit 显著降低了解析 gap 中 low branch 暗中消失的担忧，但它只 continuation 已知低支，
没有穷举共存吸引子，也不是全程 coupled ODE integration；因此不升级 R4 的正式 acceptance tier。

## 5. 工程性问题

### 已通过的工程合同

- 三个直接输入逐一 SHA-256 锁定，上游 R3 provenance 继续由 source config 递归锁定；
- config 明确禁止 q recalibration、parameter grid、continuous field、full SNN、E→E/conductance/relay 修改；
- base-dt 必须先通过，才运行 half-dt；
- analytic bridge 前后均有 500-ms full-fast sentinel，且 use/occupancy、low branch、latch state 全部 fail-closed；
- latch reset 是现有 state machine 的真实 transition，不是脚本改 bit；
- early/late 使用完全同一 schedule 和 classifier；classifier 已覆盖“初始 q 已在 fold 下方”的情况，并拒绝 sustained-high tail；
- late acceptance 显式要求 final sustained-low 与 no bounded-high/runaway tail，不能用短暂 250-ms low exit 代替最终恢复；
- summary 对两份 trace NPZ、gate CSV 与 endpoint CSV 保存 SHA-256；reporting-only refresh 必须先验完整四文件 manifest，从 hash-lock 的 gate CSV 继承 canonical numeric gate，并要求重算后的 challenge gate 字典与 CSV 完全一致，不能构造 `finite=True` 或静默改变 acceptance；
- strict JSON 禁止 NaN/Infinity；CSV、NPZ、PNG、PDF 与中文 README 齐全；
- 两份 trace 各约 1 MiB，peak RSS `175,412 KiB`，资源余量充分；
- closure gate table 为 `168/168` true；R0–R4 相关扩展回归测试 `82 passed in 1.75 s`；canonical 运行中的一次既有 transfer `logaddexp` warning 未伴随 support、bound 或 nonfinite failure，checkpoint-forward 两套 dt 标签一致。

### 仍需保留的工程边界

- 长时间 analytic gap 不是完整 fast trajectory，只有 sentinel 采样支持 zero-use 近似；若后续空间场中任一区域 use/occupancy 非零，必须回到数值积分；
- `entry_event_index=0` 在 protected fork 表示初始 q 已低于 fold，不等于 time-zero 新 onset；lifecycle 判据必须读 `candidate_trigger_event_index` 和 response-excluded paired returns；
- `early event6_no_section_crossing=false` 不能被隐藏，报告和图注必须明确仍有 evoked response；
- result 是 ignored artifact，提交时必须显式 `git add -f`，否则 clean worktree 不能证明结果存在。

核心产物：

- summary：`results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/actual_entry_lifecycle_closure_summary.json`
- gate table：`results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/closure_gate_table.csv`
- endpoint table：`results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/hybrid_endpoint_table.csv`
- representative traces：`results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/representative_traces_dt0p125.npz`、`representative_traces_dt0p0625.npz`
- 主图：`results/topic4_sef_hfo/mz_actual_entry_lifecycle_closure/figures/mz_actual_entry_lifecycle_closure.png`
- 锁定设计：`docs/superpowers/specs/2026-07-21-topic4-mz-actual-entry-lifecycle-closure-design.md`

## 6. 最小修改路线

1. 将 R4 锁为 center-point regional hybrid lifecycle support，不恢复三条下游 workflow；
2. 先做 fully coupled q-map 的局部 robustness：围绕当前 q mapping、`tau_slow` 与 `tau_fast` 做小范围 continuation，要求 base/half-dt 标签一致；
3. 同时做 coupled causal ablation：q-use off、M additive off、M-gated recovery off、latch clamped on/off；每个 ablation 只回答 entry、maintenance、exit、reset 哪一环丢失；
4. 若只有孤立单点或任一关键因果分工不成立，停止本线，不上 continuous field；
5. 若形成局部 corridor，再测试 continuous persistence / separate two-pool recovery，判断离散 latch 是否必要；
6. 只有 continuous 或明确保留的 hybrid mechanism 在 coarse field 中产生 local recruitment、front stall 与 recovery wake，才移植 full SNN；
7. 三条 workflow 只在时空模型通过后恢复为下游 readout。

## 7. 下一步建议

**GO 到 fully coupled q-map local robustness + causal ablation；暂时 NO-GO 到 continuous field、full SNN、三条 workflow 和 smooth-bifurcation claim。**

当前最安全的结论是：

> 在不改变 E→E、conductance 或 relay 的 regional three-patch fixed-bath 模型中，实际 event-5 entry 后出现 4 次 core/annulus autonomous paired bursts；已有 M 机制完成有限 fast exit，M-gated q recovery 使现有 latch 在约 74.1 s 真实 reset，随后 M 自然释放，系统回到原 LLL basin。相同六事件 challenge 在 protected state 只产生 evoked responses而不重建 autonomous lifecycle，在自然恢复后重新产生 event-5 entry 与 4 次 paired bursts；checkpoint 后 recovery 与 recovered lifecycle 在 `.125/.0625 ms` 一致，原 Segment A 仍为 `.125 ms` source。这是一个 center-point hybrid relaxation lifecycle existence proof，不是零输入 spontaneous onset、连续空间 wavefront、平滑 Hopf/torus 或 full-SNN seizure mechanism。
