# 审阅结论：MZ entry/exit nullcline 与 additive-recovery leverage

日期：2026-07-20

> 后续执行更新：formal attracting-cycle continuation 与 persistence dual-sensor Stage B 已完成；最新判定见 `mz_additive_orbit_persistence_stage_ab_2026-07-20.md`。本文保留 entry/exit cheap geometry 的阶段性证据，不再代表当前执行状态。

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

## 1. 一句话判断

**当前快系统其实已经有“low fixed point ↔ 大振幅 fast cycle”的 entry/exit 骨架；没有回收的结构性原因是进入周期后 Z 仍单向下降。加性 M 并非没有 exit leverage，但必须改成 persistence-gated、有界且能追上 Z 漂移的 recovery field。**

## 2. 完成程度

> **完成度：72/100**

已经完成：

- 用既有 Stage 0C 九维 rate system，而不是另造 Wilson–Cowan 模型；
- 画出固定 Z/A 下的 E/I nullcline；
- 解出 low-state saddle-node；
- 用完整 9D Jacobian 核对 fold 的零特征方向；
- 对 Z 和 additive-current 两个方向做 base/half-`dt` frozen cycle state fork；
- 定量比较旧 QI/JK、当前 `S_G/T_G` 与 proposed recovery 的机制方向；
- 修正 current-stage Figure 5 中 centered onset 被误写成 causal onset 的问题；
- 锁定与并行 E→E conductance/relay 线不重叠的新方程。

仍缺：

- periodic-orbit pseudo-arclength continuation，故 state-fork boundary 还不能等同正式 `B_C`；
- 完整 `(z,p,m)` slow nullcline 与同轨 lifecycle；
- spatial persistence/recovery field、leading/trailing front 与 annihilation；
- primary-seed SNN 验证。

## 3. P0 / P1 关键问题

### P0：当前 Z 在 ictal state 中只会远离出口

现方程为：

\[
\tau_z\dot z=H(I_{th}-I_I)-z.
\]

ictal activity 中 `I_I>I_th`，因此 `dot z=-z/tau_z<0`。entry 后没有任何慢向量把轨迹推回 low-state boundary，这直接解释了当前 20 s capture 中 `z` 持续下降、无 offset。

**怎么改**：不要再给 E→E 加第三个 divisor；增加只在 established ictal state 后 build 的 bounded M effector，使 `A=A_max m` 移动 fast fold，并在 offset 后保持到 Z 回到安全侧。

### P0：`0.3165 mV` 只是冻结 Z 下界，不能直接作为动态终止参数

在 `z=.85` 冻结时，约 `0.31651 mV` 重建 low+saddle，state fork 在 `0.31→0.32 mV` 由周期转低。但允许 Z 持续下降后，退出成本快速上升：保留 3 个周期约需 `0.89 mV`，按当前 persistence sensor 的约 `2.76 s` delay 则约需 `1.28 mV`。

**怎么改**：下一步必须在 `(z,p,m)` 同一轨迹上比较 `A_max m(t)` 与 `A_SN[z(t)]`；冻结-Z fork 只作 leverage oracle。

### P1：强 SNIC-like 证据仍不是正式分岔证明

精确 low fold：

\[
z_{SN}=0.87447467,
\quad(r_E,r_I)=(2.0264,7.0559)\,\mathrm{Hz}.
\]

周期从 `z=.85` 的约 `605 ms` 增至 `z=.87445` 的约 `5.5 s`，inverse-square-root 拟合 `R²=0.9993`；`z=.87450` state fork 回低。它强烈符合 SNIC/saddle-node-on-cycle 图景，但没有正式 shooting continuation。

**怎么改**：对 `(z,A)` 做 periodic-orbit pseudo-arclength continuation，保存 period/amplitude/Floquet multipliers；明确区分 `B_C` 与 basin/state-fork strip。

### P1：加性出口附近有 peak-rate ceiling 风险

`alpha_G=15` 在 `A=.30/.31 mV` 的 transition-near cycle peak 约 `102 Hz`；`alpha_G=16` 在 half-`dt` 约 `99.8/99.4 Hz`，但 base-`dt` 仍略高于 100 Hz。mean rate 只有约 3–5 Hz，不能掩盖峰值风险。

**怎么改**：formal continuation 时逐点保存 stepwise peak/occupancy；必须同时过 base/half `dt`，不能后验放宽 100-Hz operating envelope。

### P1：现有 persistence separation 只是单 seed pilot

当前 capture 的 `T_G` 在 causal onset 前最大约 `0.0600`，onset 后 2 s 的下四分位约 `0.0834`，pilot threshold `~0.0717` 非空，并在 onset 后约 `2.76 s` 首次跨越。这说明 persistence gating 有可行信号，但不能直接锁参数。

**怎么改**：用完整 repeated-IED history、interval jitter、low-root noise、fast-cycle history和 primary seeds 重算共同 separation interval；区间为空则单一 persistence sensor no-go。

## 4. 科学性问题

### 哪些地方做对了

1. **Z 的 entry 作用是对的**：它把 inhibitory efficacy 变成局部慢坐标，能够让 returning events 逐步跨入不同 fast regime。
2. **delayed `S_G` divisor 的 fast 作用是对的**：它弯折高率 E-nullcline、删除约 303-Hz saturation attractor，并形成 unstable focus 外围的大振幅周期。
3. **当前图保留了真实空间读出**：returning-event field、逐细胞 recruited-window rate、轴向时空图均来自同一 SNN capture，不是由 population rate 伪造。
4. **entry、inner cycle 与 exit 已开始分工**：现在可以把 Z=entry、`S_G`=inner-cycle shaping、M=exit/refractory 分开检验，而不是继续用一个变量承担全部职责。

### 哪些地方还不够

1. **旧 linear M 的 nullcline 从 IED 就开始移动**：它测试的是 prevention，不是 established-state termination。
2. **`T_G` 仍作用于 recurrent-E denominator**：它与旧 M4 fast mechanism 同构，也与并行 E→E line 重叠。
3. **空间层只有 fast scaffold，没有 slow tissue-state field**：当前 causal onset 后的 axial sweep 为 `48/48 bins / 130 ms`，但轴向 Spearman 仅 `+0.20`，不能称为稳定有序的 recruitment front。
4. **global scalar 没有位置记忆**：不能产生局部 refractory wake、leading/trailing front 或 intrinsic annihilation。
5. **真正目标不是稳定 torus**：稳定 torus更容易预测永久 quasiperiodic bursting；这里需要的是有限时间沿 fast-cycle family 旋转，再由慢向量横穿出口回到 low basin。

### 与旧 QI/JK 的准确区别

旧 `q_I+g_K` screen 是 negative；真正产生 bounded third state 的是 `q_I + delayed S_G recurrent-E divisor`，且 `g_K` 当时关闭。当前九维 fast cycle 正是同一 denominator 机制的解析版。因此本线不能再靠另一个 `D_R` 冒充独立更新。

新路线只保留该 denominator 作为 inherited inner-cycle generator；新 recovery 通过 additive E current 移动 low/cycle boundary，且由 persistence gate 决定何时生效。这才与并行 E→E relay 线正交。

## 5. 工程性问题

- 新 runner 单进程、单线程 BLAS；未启动完整 SNN，未接触并行 conductance worktree。
- producer 全量重放耗时 `6:41.68`、峰值 RSS `355180 kB`、无 swap；四个上游输入均以 SHA-256 锁定，输入漂移会直接停止运行。
- exact-Siegert transfer 全程 no-clip；fold 使用 smooth cubic table，state fork 使用原 extra-fine table。
- 所有 state fork 同时跑 `dt=.125/.0625 ms`；结果边界一致。
- 新 additive implementation 在 `A=0` 与原 Stage 0C RHS array-exact parity。
- current-stage figure 现在同时标记 retrospective centered onset `13.8539 s` 与 causal trailing onset `13.9788 s`，fast sweep 改锚 causal onset。
- 仍有一个独立数值债：Stage 0F v1.1 whole-return gate 对大 `epsilon=1e-3` 非线性敏感；本轮没有修改已冻结 verdict，也没有用较小 epsilon 后验改写 PASS。

## 6. 最小修改路线

1. **先做 formal `(z,A)` boundary**：fixed-point + periodic-orbit continuation，锁 `B_L/B_C`；不跑 SNN。
2. **再做 persistence separation**：在全 pre-onset/cycle history 上锁 `tau_p,p_r`，区间为空直接 no-go。
3. **只跑三档 `A_max={0.9,1.3,1.6} mV`**：检查 M 能否在 3–5 cycles 内追上 `A_SN[z(t)]`，不做宽网格。
4. **完成一条 0D lifecycle**：IED load→entry→≥3 cycles→exit→same-low return→early/late retrigger。
5. **再上 1D spatial field**：先 `gamma_p=0` 检验 local wake，再加 finite-width broad kernel 检验 front stall；无 wake 或只有全场 prevention 都停止。
6. **最后移植完整 SNN**：只在 0D/1D gate 通过后恢复时间、空间和 early-ictal 三条下游 workflow。

## 7. 下一步建议

下一版本锁定为：

> **existing Z entry + inherited delayed-`S_G` inner cycle + causal persistence sensor p + bounded additive recovery field m。**

它不改 E→E 权重、kernel、delay、relay 或 conductance，因此与并行线保持独立。核心验收不再是“平均率下降”，而是 slow trajectory 是否横穿正式 cycle boundary、回到同一个间期 basin，并在空间上形成可量化的 front/wake/stall。

## 8. 产物

- 主诊断图：`results/topic4_sef_hfo/mz_entry_exit_nullclines/figures/mz_entry_exit_nullcline_diagnostic.png`
- summary：`results/topic4_sef_hfo/mz_entry_exit_nullclines/entry_exit_summary.json`
- fold surface：`results/topic4_sef_hfo/mz_entry_exit_nullclines/fixed_point_fold_surface.csv`
- Z state forks：`results/topic4_sef_hfo/mz_entry_exit_nullclines/cycle_z_state_forks.csv`
- additive state forks：`results/topic4_sef_hfo/mz_entry_exit_nullclines/additive_current_state_forks.csv`
- timing oracle：`results/topic4_sef_hfo/mz_entry_exit_nullclines/timing_leverage_oracle.csv`
- 新设计合同：`docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`
