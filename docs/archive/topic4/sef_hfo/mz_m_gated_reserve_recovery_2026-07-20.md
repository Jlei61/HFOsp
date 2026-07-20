# 审阅结论：MZ M-gated effective-inhibitory-capacity recovery（R3）

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**R3 在注册的 scalar/path 层得到支持：已有 M 状态调节 q 恢复速度后，`tau_slow=80/90/100 s` 在 `tau_fast=20/15/25 s` 三组无重拟合条件下形成完全一致的三节点 corridor；但这只是 fixed-q fast sensor 加 feed-forward q–M path oracle，只解锁一个有“少于 4 次 return 立即停止”硬门的 short P3 coupled canary，不能写成已实现可恢复发作 lifecycle。**

当前 canonical status 为：

```text
R3_M_GATED_RESERVE_RECOVERY_PATH_SUPPORTED_SHORT_P3_FORK_UNLOCKED
```

当前 canonical decision 为：

```text
run_short_P3_state_fork_tau_slow_80_90_100_all_qhold_base_half_dt
```

## 2. 完成程度

> **R3 scalar/path diagnostic 完成度：95/100；coupled autonomous lifecycle：尚未验收**

已完成：

- 锁定且 hash-check R0b、R2 的所有输入，R3 没有重新拟合 `q_res/tau_D`；
- 重跑 `3 q_hold x 4 phase x 2 dt=24` 个 fixed-q M-ramp sensor，24/24 通过 source/ramp label、有限性、support/bound、M 单调性、固定 q 与 R0b final-A parity；
- 完成 `3 tau_fast x 6 tau_slow x 3 q_hold x 4 phase x 2 dt=432` 行 path oracle，432/432 path 本身通过；
- primary `tau_fast=20 s` 与固定 sensitivity `15/25 s` 均只接受注册的 `tau_slow=[80,90,100] s`；
- 原边界未被放松：`70 s,q_hold=.845` 仍为 entry failure，`120/160 s` 仍为 schedule failure；
- fixed-sensor 最大 q excursion 为 `.0002437338`，远低于注册上限 `.00125`；
- recovery-gate-only 的最大 q-nullcline 上界为 `.8493216201`，低于 entry fold `.8558315843`；
- gate-off 18/18 精确复刻 R2，maximum numeric error 为 0；
- 8 tests passed；最终 canonical run 为 `704.269331 s`，peak RSS `238,104 KiB`（约 `232.5 MiB`），明显低于 `1.5 GiB` 上限。

尚未完成：

- q 对 fast regional dynamics 的闭环反馈；
- 同一条真实轨迹上的 event-6 entry、至少 4 次 pulse-free returns、finite exit；
- latch 的真实 true-to-false reset、自然 M release、same-basin recovery；
- early/late retrigger；
- 移除 fixed bath mask 后的局部招募、wavefront stall/annihilation；
- continuous field、full SNN 与 fast Hopf/torus/limit-cycle 证明。

## 3. P0 / P1 关键问题

### P0：scalar/path 通过不等于 coupled lifecycle 通过

R3 的 fast sensor 在固定 q 下生成 `U(t),m(t)`，随后 q 只在该测量轨迹上 feed-forward replay，并不反过来改变 fast rates、occupancy 或 M build。432/432 path 说明候选慢路径在 frozen-sensor 近似内自洽，只是 coupled run 的必要条件。

**为什么严重**：一旦 q 在真实仿真中恢复，它会改变 fast branch 和 occupancy；退出时间、M build 时间、return 数量及 latch reset 都可能与 feed-forward 结果不同。因此当前证据不能支持“已经形成可进入、维持、退出并回到同一 basin 的发作态”。

**怎么改**：只运行 conditional spec 中的 center canary，先联合积分 fast + q + M；若少于 4 次 pulse-free regional returns，立即记为 `R3_COUPLED_CLEAN_NO_GO_EARLY_M_EXIT`，不得调 M 或扩网格。

### P0：现有 timing 已预示 M 可能在第 4 次 return 前过早终止

现有 base-dt regional trace 的 timing sentinel 为：

```text
pulse-free returns 1/2: 11.988 / 12.547 s
latch set:             12.612625 s
pulse-free returns 3/4: 13.036996 / 13.889377 s
```

旧 `225 ms` arm 的 A 约在 `13.079 s` 越过 R0b `q=.840` exit fold：这只比第 3 次 return 晚约 `42 ms`，却比第 4 次 return 早约 `810 ms`。而 R3 把 q 保留在 R0b corridor 附近，所需退出 A 还可能低于旧的 continuing-Z depletion path。

**为什么严重**：center canary 很可能只有 3 次 return。若如此，R3 虽能修复 reset，却以缩短 ictal episode 为代价破坏 registered maintenance gate。

**怎么改**：center cell 固定为 `tau_slow=90 s,q_hold=.8425,dt=.125 ms`；少于 4 次 return 就终止整条 coupled R3，不跑其余 17 paths，不改变 `tau_m_up=225 ms`、Amax、latch threshold 或 q mapping。

### P1：恢复门的生物学映射仍是 phenomenological

活动与残余 presynaptic calcium 调节 readily releasable vesicle pool 的恢复速度有直接实验依据，例如高频输入可加速 release-ready pool replenishment，且这种加速依赖 Ca²⁺；其他实验也分离出 Ca²⁺-dependent 与 Ca²⁺-independent replenishment pathway。[Wang and Kaczmarek, 1998](https://doi.org/10.1038/28645)；[Liu et al., 2014](https://doi.org/10.7554/eLife.01524)

但当前 M 是现有模型里的无量纲慢效应器，并非显式 presynaptic calcium、synaptotagmin、calmodulin 或 vesicle-pool state。文献只能支持“恢复速度可随状态变化”这一 generic premise，不能支持当前精确的线性 `M -> r_rec` 耦合，更不能证明 postsynaptic/additive M 同时就是 presynaptic GABA reserve 的恢复传感器。

**怎么改**：当前变量统一称为 `effective inhibitory capacity q`；R3 成功也只能称 state-dependent phenomenological recovery。若 coupled canary 失败，下一步转向生物学含义分离的 two-pool / separate recovery-effector 设计，而不是继续给同一个 M 添加多重解释。

### P1：fixed bath mask 仍是 imposed boundary

R3 只读取既有 core/annulus/bath regional scaffold，bath 固定在 `q=.90` 且不耗竭。它没有产生连续空间前沿，也没有证明全局抑制随 recruited area 自发限制传播。

**怎么改**：即使 short P3 canary 全通过，下一门仍是 coarse spatial field 的 local recruitment、front stall 与 refractory wake；在此之前不能声称 spatial containment 或 wavefront annihilation。

## 4. 科学性问题与动力学反思

### 4.1 什么做对了

R2 的结构性冲突是：preictal event memory 需要慢恢复，而 postictal reset 需要快恢复。R3 没有再加一个负电流，也没有改变 E-E；它只让已有 M 状态选择 q 的恢复时间尺度：

\[
r_{rec}(m)={1-m\over\tau_{slow}}+{m\over\tau_{fast}},
\]

\[
\dot q=r_{rec}(m)(q_0-q)
-{U\over\tau_D}(q-q_{res}),
\qquad \tau_{fast}<\tau_{slow}.
\]

在 onset 前 `m=0`，所以 entry 与 schedule 必须逐项继承 R2。结果确实保留了 `70 s` 的高-q entry failure 和 `120/160 s` 的 schedule failure，说明 R3 没有通过改变 onset contract 偷换结果。M 被招募后，恢复速度才上升，使 R2 中 `90/100 s` 的 reset failure 转为通过，从而形成事先注册的 `80/90/100 s` corridor。

这比单纯加密 `tau_r` 网格更有解释力：它直接把两个方向相反的时间尺度任务分开，而且 `tau_fast=15/20/25 s` 三组在不重拟合 `q_res/tau_D` 时得到完全相同的 corridor。

### 4.2 nullcline、稳定点与“不是 Hopf”的准确解释

冻结 `(m,\bar U)` 后，q-nullcline 是：

\[
q^*(m,\bar U)=
{r_{rec}(m)q_0+(\bar U/\tau_D)q_{res}
\over
r_{rec}(m)+\bar U/\tau_D}.
\]

当 `tau_fast<tau_slow` 且 `q_0>q_res` 时：

\[
{\partial q^*\over\partial m}>0,
\qquad
\lambda_q=-\left[r_{rec}(m)+{\bar U\over\tau_D}\right]<0.
\]

所以 M 上升只把唯一稳定 q fixed point 向 `q_0` 上移，同时加快 q 方向收缩；它不会创造第二个 q fixed point，也不会让 q 方向失稳。当前 feed-forward oracle 中 M 由测量 sensor 给定，q–M block 等效为 triangular slow path，其对角收缩项保持负值。因此 R3 没有、也不应声称产生 Hopf、torus、SNIC 或 smooth autonomous limit cycle。

当前更准确的“大环套小环”图景是：

- fast regional subsystem 已提供 bounded CCO 内环；
- q 的慢耗竭负责跨入 fast entry fold；
- additive M，而不是 recovery gate，负责跨出 fast exit fold；
- M-gated q recovery 负责在 exit 后及时达到 `q>=.885`，让 latch reset；
- reset 后 M 以 `12 s` 释放，候选外环才可能闭合回 interictal basin。

这仍是 hybrid relaxation candidate，不是已经证明的 smooth bifurcation object。

### 4.3 causal controls 支持了怎样的最小因果结论

`additive on / recovery gate off` 的 18 个 cell 与 R2 handoff 完全一致，numeric error 为 0：关闭 M-gated recovery 后，`90/100 s` 的原 reset 边界回来。说明 R3 的改进确实来自 state-dependent recovery，而不是 runner 或 fold interpolation 漂移。

`additive off / recovery gate on` 则给出所有 54 个组合的 `q^*(m=1)<q_entry`，最坏上界 `.8493216201 < .8558315843`。因此恢复门单独不能把 CCO 推过 fast exit；它不是 termination mechanism。当前唯一安全归因是：

> additive M supplies fast exit；M-gated q recovery supplies timely reset。

二者不能互换，也不能合并写成一个“全局抑制终止发作”的笼统机制。

### 4.4 432/432 path pass 为什么仍只接受 80/90/100

432 条 feed-forward path 全部通过 sensor replay、q excursion、fold margin、120-s handoff 与 post-reset release。这说明在给定 fast-exit sensor 后，M-gated recovery 对注册 axis 的恢复路径本身很宽。

但完整 tau-node acceptance 还必须乘上 onset 前继承的 entry/periodic/schedule contract：

- `70 s` 在 `q_hold=.845` 缺 entry robustness；
- `80/90/100 s` 三个 q-hold、phase、dt 与 schedule 均通过；
- `120/160 s` 仍破坏 sparse schedule selectivity。

所以最终 accepted set 恰好是 `[80,90,100] s`，不是从 432/432 中事后挑选。这也说明 R3 只修复了它预注册要修的 postictal reset，不修 entry 或 schedule 的边界。

### 4.5 与并行 conductance / E-E 线的独立性

本工作线只改变 inhibitory slow-path 的 q recovery coefficient，明确未修改：

- `W_EE`、E→E kernel 或 delay；
- recurrent saturation `rec_sat_g`；
- conductance-based membrane equation；
- presynaptic relay `y/x`；
- P3 geometry 与 fixed bath-resource mask。

因此两条线的科学问题仍可分辨：并行线检验 fast recurrent/membrane structure 能否生成更自然的 ictal attractor；本线检验已有 bounded CCO 的 inhibitory slow path 能否形成 entry–maintenance–exit–reset 外环。short canary 在本分支独立失败或通过，都不应借用并行线参数补救。

## 5. 工程性问题

### 已通过的工程合同

- 10 个上游输入均以 SHA-256 锁定，R0b 与 R2 provenance fail-closed；
- config 锁死 `q_hold`、`tau_slow`、`tau_fast`、Amax、phase、dt、corridor 与 forbidden scope；
- 24-cell sensor Cartesian product 无缺失、无重复，base/half dt 顺序运行；
- fixed-q error 最大约 `2.86e-8`，final A 对 R0b error 为 0；
- 432/432 path 具备有限值、动态 fold margin、reset horizon 与 post-reset monotonic recovery；
- gate-off 与 gate-only controls 完整落盘；
- strict JSON 使用 `allow_nan=False`，CSV/NPZ 在绘图前保存；
- figure 目录含中文 README，主图采用 2×3 mechanism layout；
- 8 tests passed；单进程、单 BLAS 线程，canonical run 为 `704.269331 s / 238,104 KiB`，没有 OOM 风险。

### 仍需保留的工程边界

- `m_ramp_sensor` 是 fixed-q producer，不能被下游误读为 coupled q–M simulation；
- q replay 使用 `<=1 ms` reporting step，但 fast sensor 本身来自 base/half dt 的既有 regional integrator；
- path acceptance 与 upstream entry/schedule acceptance 是两层合同，不能只读 `path_pass=True`；
- coupled canary 必须保持旧 `10P+2` state layout 的 default-off backward compatibility，并显式输出 `final_latch_state`；
- canary 单 trace 上限 `64 MiB`、RSS 上限 `1.5 GiB`，center failure 时不得继续跑 grid。

核心产物：

- summary：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/m_gated_reserve_recovery_summary.json`
- M-ramp sensor：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/m_ramp_sensor.csv`
- path oracle：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/m_gated_path_oracle.csv`
- tau acceptance：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/m_gated_tau_acceptance.csv`
- gate-off control：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/control_gate_off_r2_parity.csv`
- gate-only control：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/control_gate_only_nullcline.csv`
- 主图：`results/topic4_sef_hfo/mz_m_gated_reserve_recovery/figures/mz_m_gated_reserve_recovery.png`
- conditional canary spec：`docs/superpowers/specs/2026-07-20-topic4-mz-m-gated-reserve-coupled-canary-design.md`

## 6. 最小修改路线

1. 锁定 R3 scalar/path 为 `SUPPORTED`，但不把它写成 coupled lifecycle；
2. 只实现 default-off 的 q reserve recovery equation 与 `final_latch_state` 输出，不碰 E-E、conductance、relay 或空间核；
3. 先跑 center canary `tau_slow=90 s,q_hold=.8425,dt=.125 ms`；
4. 若少于 4 次 pulse-free returns，立即 clean no-go，停止 R3 coupled grid并进入 biologically separated two-pool resource design；
5. 只有 center 通过，才顺序跑 9 个 base-dt，再跑 9 个 half-dt；
6. 只有 entry、至少 4 returns、finite exit、真实 latch reset、same-basin recovery、early/late retrigger 全部通过，才允许进入 coarse spatial field；
7. continuous field、full SNN 和三条下游 workflow 继续冻结。

## 7. 下一步建议

**GO 仅到 short P3 center canary；NO-GO 到 broad coupled grid、continuous field、full SNN 和完整 ictal lifecycle claim。**

当前最安全的核心结论是：

> 让已有 M 状态只调节 effective inhibitory capacity q 的恢复速度，能够在 frozen-sensor / feed-forward path 层把 R2 的单点 `80 s` 扩成预注册的 `80/90/100 s` 三节点 corridor，并在 `tau_fast=15/20/25 s` 下无重拟合复现。nullcline 分析与 causal controls 表明该门只加快唯一稳定 q 状态的恢复；additive M 仍负责 fast exit，M-gated recovery 只负责及时 reset。这不是 Hopf 或完整 lifecycle。现有 timing 还预示 M 可能在第 4 次 return 前过早退出，因此下一步必须先跑一个 center coupled canary，少于 4 returns 就立即停止。
