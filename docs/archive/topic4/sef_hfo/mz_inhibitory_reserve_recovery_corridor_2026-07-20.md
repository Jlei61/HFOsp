# 审阅结论：MZ inhibitory-reserve recovery-timescale corridor（R2）

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**R2 是一个数值完整、机制边界清楚的 clean no-go：常数 `tau_r` 的确能把 entry 从第 5 事件推迟到第 6 事件，但入口需要慢恢复、reset 又需要快恢复，最后只有孤立的 `80 s` 节点在全部三个 `q_hold` 上通过，未达到预注册的“至少 3 个连续节点”门，因此禁止运行 coupled R2。**

当前 canonical status 为：

```text
R2_RECOVERY_TIMESCALE_CORRIDOR_CLEAN_NO_GO_REGISTERED_GATES
```

当前 canonical decision 为：

```text
do_not_run_coupled_R2_and_proceed_to_two_pool_resource_design
```

## 2. 完成程度

> **R2 scalar diagnostic 完成度：92/100；完整 autonomous lifecycle：仍未通过**

已完成：

- 锁定 `tau_r=[20,40,50,60,70,80,90,100,120,160] s`、`q_hold=[.8400,.8425,.8450]` 的 `10 x 3` continuation；
- 对每个 cell 仅用完整 CCO hold 与不变的 event endpoint 重新求 `q_res/tau_d`，没有针对 entry 或 handoff 反调参数；
- 30/30 cells 均找到唯一、单调、物理合法的 root；
- 完成 4,800 行注册 root scan，numeric error 为 0；
- 完成 base/half event replay、完整 `phase x source-dt` periodic oracle、locked schedule probes；
- 按真实 latch 语义完成两阶段 postictal `q-M` handoff predictor：reset 前 additive M 冻结，reset 后才按 `tau_m_down=12 s` 衰减；
- 完成 preregistered `72/88 s` fixed-parameter sensitivity，4/4 event rows 与 16/16 periodic rows 完整且 gate 通过；
- canonical producer、严格 JSON、图、CSV/NPZ 和中文 `figures/README.md` 已落盘；
- 相关验证为 29 tests passed；canonical run 约 `13.02 s`，peak RSS `254,932 KiB`（约 `249 MiB`，按运行记录取整可记为约 `255 MiB`），单进程、单 BLAS 线程、0 swap、无 OOM。

尚未完成：

- q 对 fast regional model 的闭环反馈；
- dynamic M-gated recovery 或真正的双资源池；
- coupled termination、same-basin recovery、early/late retrigger；
- bath mask 移除后的局部招募、wavefront stall/annihilation；
- continuous field 与 full SNN；
- smooth bifurcation / limit-cycle 证明。

## 3. P0 / P1 关键问题

### P0：只有 `80 s` 是全 q-hold 通过点，不是鲁棒 corridor

预注册验收要求：至少 3 个连续 `tau_r` 节点同时通过全部三个 `q_hold` 的 entry、periodic hold、handoff 与 preferred-q schedule contract，而且该 component 必须包含 `80 s`。

实际结果为：

```text
passing_tau_nodes = [80.0]
preferred_component_tau_nodes = [80.0]
```

虽然 30 个 cell 中有 9 个通过，完整 tau-node 只有 `80 s` 一个。单点命中不能排除边界拟合，也不能支撑 coupled 生命周期。必须保留 clean no-go，不能事后把连续节点门降成“存在一个点”。

**怎么改**：关闭 constant-`tau_r` coupled arm；下一步先在 scalar/path 层检验 state-dependent recovery 是否能把入口与 reset 两个相反约束分开。

### P0：入口和 reset 对 `tau_r` 的要求方向相反

低 `tau_r` 区恢复过快，事件间记忆不够：

- `20 s`：三个 `q_hold` 都在第 5 事件提前 crossing；
- `40 s`：首次 crossing 已推迟到第 6 事件，但 pre-last minimum 只有 `.85605-.85666`，低于 `q_entry+.00125=.8570816`，仍不满足安全的 event-6-first entry；
- `50 s`：只有 `q_hold=.8400` 通过；
- `60/70 s`：`.8400/.8425` 通过，但高 `q_hold=.8450` 仍缺 pre-last margin；
- `80 s`：三个 `q_hold` 才全部通过 entry。

高 `tau_r` 区则 postictal recovery 太慢：

- `90 s`：`.8400/.8425` 到 `q=.885` 的 reset time 分别为 `124.96/121.15 s`，超过注册的 `120 s` horizon；只有 `.8450` 仍在 `117.16 s` 内；
- `100 s` 及以上：三个 `q_hold` 全部超过 reset horizon；
- `120/160 s` 的 sparse schedule 还会在第 6 事件进入，进一步破坏 registered schedule selectivity。

**怎么改**：不能继续把一个常数恢复时间同时当作 preictal event-memory timescale 和 postictal reset timescale；必须引入状态依赖的恢复速度，或在其失败后进入真正的双 pool 设计。

### P1：`72/88 s` sensitivity 通过不能推翻 clean no-go

`72/88 s` sensitivity 固定使用 primary `80 s,q_hold=.8425` 解出的 `q_res/tau_d`，其用途是验证单点附近的 local parameter robustness。两臂均保持第 6 事件首次 entry、post-event fold margin 与 periodic range，说明 `80 s` 不是纯数值尖点。

但它没有重新覆盖三个 `q_hold` 的完整 Cartesian gate，也没有改变 registered tau axis 上只有一个全通过节点的事实。因此它支持“局部轨迹不脆弱”，不支持“存在机制级 corridor”。

### P1：当前仍是 frozen-sensor + analytic handoff，不是 autonomous seizure

q 使用 hash-locked CCO/event `U(t)` replay，并未反馈到 fast regional system；handoff 也是假定低 occupancy 后的解析路径。它能定位慢变量合同冲突，但不能证明动态网络会沿该路径退出，也不能证明空间 containment。

**怎么改**：只有新的 scalar/path 机制先形成宽 corridor，才允许一个短、分段、低资源的 coupled state-fork；否则不得恢复 field/SNN workflow。

## 4. 科学性问题与动力学反思

### 4.1 什么做对了

R2 纠正了上一版最重要的设计漏洞：原先把 `tau_r=20 s` 当作固定背景参数，却没有先检查不规则事件间隔怎样改变 q 的 event map。R2 没有加新电流、没有碰 E-E，也没有靠新阈值制造结果，而是先检验原方程中最直接的时间尺度自由度。

这一步确实解释了 entry-ordering failure。quiet gap 内 `U=0` 时：

\[
q_{k+1}^{-}=q_0-(q_0-q_k^{+})
\exp\left(-{\Delta_k\over\tau_r}\right).
\]

第 5 到第 6 事件的 gap 最长（`3.384 s`）。增大 `tau_r` 会减少这段 gap 内的恢复，保留更多累积耗竭，使第 6 事件而不是第 5 事件成为首次 entry。结果从 `20 s` 的 event-5 crossing，依次过渡到 `40-70 s` 的 event-6 但 margin 不完整，再到 `80 s` 的三 q-hold 全通过，与这个 event-map 推断一致。

R2 也正确修复了 postictal M 语义。现有 latch 不是“退出后 M 立刻衰减”；在 low occupancy 但 reset 条件未满足时，`dM/dt=0`，只有 low rate、persistence off 且 `q>=.885` 后才 reset，再按 `12 s` 释放 M。报告的 handoff 因而对应真实实现的两阶段路径，而不是一个过于乐观的 immediate-decay 近似。

最后，完整 periodic oracle 30/30 cells 全通过，说明 no-go 不是 bounded CCO hold 消失；失败来自 event entry 与 postictal reset 两段慢流不能被同一个常数 `tau_r` 同时满足。

### 4.2 nullcline / 稳定点层面的准确解释

原方程为：

\[
\dot q={q_0-q\over\tau_r}
-{(q-q_{res})U\over\tau_d}.
\]

冻结平均 use `Ubar` 后，唯一 q-nullcline 为：

\[
q^*(\bar U)=
{q_0/\tau_r+q_{res}\bar U/\tau_d
\over
1/\tau_r+\bar U/\tau_d}.
\]

其 q 方向线性特征值为：

\[
\lambda_q=-\left({1\over\tau_r}+{\bar U\over\tau_d}\right)<0.
\]

因此，单纯改变 `tau_r` 不会产生第二个 q fixed point，也不会让 q 方向失稳。R2 又在每个 cell 重新映射 `q_res/tau_d`，把 CCO mean 锁回原 `q_hold`；所以它主要改变的是沿 slow manifold 的速度和有限 event-to-event memory，而不是创造新的 fast branch。

更重要的是，这个 R2 只 replay q，并未改 fast subsystem 的 Jacobian。它不可能据此证明 fast Hopf，也没有制造一个新的 SNIC、homoclinic 或 permanent limit cycle。当前能支持的动力学图景仍只是：fast subsystem 已有 bounded CCO 小环，q 负责跨 entry boundary，additive M 负责跨 exit boundary，而 q/M 的慢漂移是否真能闭合成外层 relaxation loop 仍待 coupled 验证。

### 4.3 为什么 constant `tau_r` 必然形成窄夹层

入口需要在多次事件间保留耗竭，因此偏好更大的 `tau_r`；退出后的 latch reset 需要 q 在有限 horizon 内恢复到 `.885`，因此偏好更小的 `tau_r`。在 `U=0` 时：

\[
t_{reset,q}=-\tau_r
\log\left({q_0-.885\over q_0-q_{start}}\right),
\]

所以 reset wait 与 `tau_r` 近似成正比。R2 的结果不是偶然参数失败，而是同一个常数同时承担两个相反任务后的结构性夹逼：入口下界大约推到 `80 s`，handoff 上界又在 `80-90 s` 之间切断全部 q-hold robustness。

这也解释了为什么继续加密 `75-85 s` 网格没有科学价值。它可能找到更多数值点，但不能证明对 q-hold 与 protocol 扰动存在有限宽度机制区域；预注册连续节点门正是用来阻止这种事后插值。

### 4.4 下一候选：M-gated state-dependent recovery，只做 cheap-first

R2 no-go 指向的最小新假设不是再加一个 additive current，也不是改 E-E，而是让 q 的恢复速度依赖现有 M 状态。例如：

\[
r_{rec}(m)={1-m\over\tau_{slow}}+{m\over\tau_{fast}},
\]

\[
\dot q=(q_0-q)r_{rec}(m)
-{(q-q_{res})U\over\tau_d},
\qquad 0\le m\le1,
\]

其中 `tau_slow > tau_fast`。预期分工是：

- interictal / preictal 的 `m` 低，使用慢恢复，保留跨事件耗竭记忆；
- ictal exit 期间 M 已被招募，恢复速度增大，使 q 能在既有 reset horizon 内回到安全区；
- M 同时保留原 additive exit 作用，q 恢复则逐步恢复抑制，不需要改 recurrent E-E。

冻结 `(U,m)` 时仍只有一个 q fixed point：

\[
q^*(\bar U,m)=
{r_{rec}(m)q_0+q_{res}\bar U/\tau_d
\over
r_{rec}(m)+\bar U/\tau_d}.
\]

当 M 上升、`r_rec(m)` 增大时，该 fixed point 向 `q_0` 上移，q 方向收缩也加快。这正好有机会把“慢 entry memory”和“快 postictal reset”解耦，但它仍不会自动制造 fast Hopf。其第一轮只能做 preregistered scalar/path continuation，检查是否出现跨多个 `tau_slow`、全部 q-hold 和 `tau_fast` sensitivity 的有限宽度 corridor。

这里必须保持三个限制：

1. 这是下一步假设，不是 R2 结果；任何未落盘 pilot 不能进入 evidence table 或结论；
2. 若 baseline M 不够低、M 上升太早，快速 q recovery 可能直接 prevention，必须显式检查 interictal leakage 与 onset ordering；
3. q 若仍叫“presynaptic GABA reserve”，M 直接加速其恢复的生物学含义偏弱；在机制确认前更安全的名称是 `effective inhibitory capacity`，或失败后改为两个含义清楚的 resource pools。

### 4.5 与并行 E-E / conductance 线的边界

本工作线只处理 inhibitory-resource entry/exit slow path，不修改：

- E-E weight、kernel、delay；
- recurrent excitation saturation `rec_sat_g`；
- conductance-based membrane equation；
- relay `y/x`；
- P3 geometry 或并行线的 connection mechanism。

因此两条线回答的问题不同：并行线检验 fast recurrent-excitation / membrane coupling 能否产生更丰富的 ictal attractor；本线检验已有 bounded CCO 的 inhibitory-resource slow path 能否形成可进入、可退出的外层 relaxation loop。即使未来二者都通过，也必须先分别完成 ablation，不能把两条线的修改一次性合并后归因。

## 5. 工程性问题

### 已通过的工程合同

- 6 个上游输入全部用 SHA-256 锁定，并校验 R0b/R1 provenance；
- 注册轴、primary point、sensitivity、reset 常数与 forbidden scope 均有 fail-closed config validation；
- 30/30 mapping roots 唯一、单调、物理合法；
- root scan 为完整 `30 x 160 = 4,800` 行，numeric error 为 0；
- event base/half、periodic `4 phase x 2 source-dt`、schedule 与 sensitivity 产品均检查 Cartesian completeness；
- strict JSON 使用 `allow_nan=False`，中间 CSV/NPZ 在绘图前落盘；
- figure 目录包含中文 README，主图为 2×3 mechanism figure；
- 29 tests passed；约 `13.02 s / 255 MiB`，明显低于 1 GiB 合同，没有 OOM 风险。

### 仍需保留的工程边界

- analytic handoff 用 R0b frozen folds 的 piecewise-linear interpolation，不等同于 dynamic fast-model proof；
- fixed bath mask 仍是外加边界条件，不能当作 emergent spatial containment；
- coupled arm 若未来解锁，必须继续限制单进程/单 BLAS，并先跑短 state-fork，不能直接扩成全轴 SNN grid；
- 结果目录和主图只能作为 R2 scalar diagnostic，不得被下游脚本误标为 autonomous lifecycle。

核心产物：

- summary：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/recovery_corridor_summary.json`
- tau acceptance：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/recovery_corridor_tau_acceptance.csv`
- event entry：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/recovery_corridor_event_entry.csv`
- periodic oracle：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/recovery_corridor_periodic_oracle.csv`
- hybrid handoff：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/recovery_corridor_hybrid_handoff.csv`
- 主图：`results/topic4_sef_hfo/mz_inhibitory_reserve_recovery_corridor/figures/mz_inhibitory_reserve_recovery_corridor.png`

## 6. 最小修改路线

1. 把 constant-`tau_r` R2 锁为 clean no-go，不运行 `[60,70,80] s` coupled arm；
2. 预注册一个独立的 M-gated recovery scalar/path node，只改变 q recovery rate，不改 E-E、conductance、relay 或空间连接；
3. 先要求至少 3 个连续 `tau_slow` 节点在全部 q-hold、完整 periodic/schedule、handoff 与 `tau_fast` sensitivity 上通过；
4. 同时检查 baseline M leakage、onset 前是否被 prevention、M 上升后 q-nullcline是否按预期上移；
5. 只有上述 corridor 成立，才运行少量 tau 节点的 coupled regional state-fork，验收 entry、bounded CCO、termination、same-basin recovery 与 early/late retrigger；
6. 若 M-gated recovery 仍无宽 corridor，停止共享单 q 状态，进入生物学含义分开的 two-pool inhibitory resource；
7. local field / continuous field / full SNN 继续冻结，直到 coupled regional lifecycle 通过。

## 7. 下一步建议

**NO-GO 到 constant-`tau_r` coupled R2；GO 仅到 M-gated state-dependent recovery 的 cheap-first scalar/path 验证。**

当前最安全的核心结论是：

> 延长 inhibitory-reserve recovery 确实修复了锁定六事件的 entry ordering，但同一个常数 `tau_r` 无法同时提供足够长的 preictal event memory 和足够快的 postictal reset。完整全-q-hold通过只出现在孤立的 `80 s` 节点，因此它不是可接受的动力学 corridor，也不能解锁 autonomous、spatial field 或 full SNN。下一步应让恢复速度受现有 M 状态调节，先检验能否形成有限宽度的外层 slow-loop corridor；这条线继续与并行 E-E/conductance 机制严格分开。
