# 审阅结论：MZ additive fast boundary 与 persistence Stage A/B

日期：2026-07-20

工作线：`.worktrees/topic4-mz-divisive-lifecycle`

并行边界：本线未修改 `W_EE`、E→E kernel/delay、presynaptic relay 或 conductance membrane；不与 `.worktrees/topic4-mz-conductance` 的 E→E/conductance 路线合并解释。

## 1. 一句话判断

**Additive M 的“退出方向”是成立的，但原先的 memoryless scalar persistence gate 不成立。**快系统存在与 low saddle-node 对齐的 strong SNIC-like cycle boundary；真正阻断 lifecycle 的是同一个标量在真实空间 SNN 中开门过晚、在 homogeneous cycle 中又开门过早，因此当前版本不能进入完整 0D 或空间 SNN。

## 2. 完成程度

> **完成度：78/100**

已经完成：

- 在锁定的九维 Stage 0C 快系统上实现 additive-current event-restarted Poincaré map；
- 沿 `A` 追踪与 `A=0` 连通的 attracting cycle branch；
- 保存 period、peak、orbit-to-fold distance、P/P² closure、finite-difference transverse multipliers；
- 用 exact transfer 与 base/half `dt` 复核选定点；
- 从 raw SNN capture 重算严格 causal onset 与完整 pre-onset persistence threshold；
- 用完全相同参数并列检验 `recorded_SNN_UTG_replay` 和 alpha-15 `endogenous_phase_sensor`；
- 执行 memoryless primary 与不能改判 primary 的 latched-gate 机制诊断；
- 生成两张四面板图、CSV/JSON/NPZ、图目录 README 和单元测试。

尚未完成：

- unstable/disconnected periodic branch 的排除与 formal SNIC 证明；
- transition-near derivative matrix platform；
- primary seeds 3/4 的 `A_G/U_TG` spatial histories；
- 显式 spatial recruitment coordinate；
- full `Z-p-latch-M` trajectory、same-low return、retrigger；
- coarse spatial field 与完整 SNN lifecycle。

扣分原因不是 runner 没跑完，而是 Stage B 按预注册 stop rule 得到 no-go；继续积分原 0D 只会把已知错误的 sensor mapping 包装成更长轨迹。

## 3. P0 / P1 关键问题

### P0：一个 homogeneous scalar 不能同时表示“局部成熟振荡”和“空间招募范围”

phase-source 周期从第一圈就是成熟全局 oscillator。严格按 SNN 定义重建：

\[
A_G=\Psi(r_{E,fast}),
\qquad
U_p=\operatorname{slow\_gate\_drive}(A_G),
\]

四个 phase offsets 在 `0.03–0.81` 个 A=0 baseline-period equivalents 内开门/越过 exit oracle。真实 SNN 的 `A_G/U_TG` 还包含 recruited area 逐步扩大的信息，同一阈值到 `4.56` 个 equivalents 才打开。这一层只是 fixed-period timing oracle，不是 dynamic slow trajectory 的真实 return count。

**为什么严重**：如果按 homogeneous 输入调 threshold，模型会在 entry 前后立刻 prevention；如果按真实空间历史调，又留不出足够 build 时间。这里不是参数网格没扫够，而是 0D state 缺了空间坐标。

**怎么改**：下一版必须显式保留 recruited fraction / neighborhood recruitment field；不能再把 mature-cycle phase source 当作空间 onset 的完整替代。

### P0：memoryless detection 与 effector build 被错误绑成同一个 gate

primary `tau_p=750 ms,p_r=.0722287` 下，SNN gate 在 `2758.5 ms` 才首次打开。到第 5 个 A=0 baseline-period equivalent：

- `Amax=1.6 mV,tau_up=125 ms`：available-minus-required `=-.257 mV`；
- `Amax=1.6 mV,tau_up=100 ms`：available-minus-required `=-.132 mV`。

p 在 bursting trough 中再次跌到阈值下，memoryless M 只短暂 build。主合同的 15 个 `tau_p × arm` 组合全部没有共同 dual-sensor candidate。

**怎么改**：检测和维持必须分开。post-no-go latch diagnostic 在第一次 crossing 后保留 build gate，SNN 的 `tau_up=125/100 ms` 分别于 `4.958/4.864` 个 baseline-period equivalents 追上 exit oracle，说明幅度杠杆尚在；但 primary-125 只比 5-equivalent 上限早约 `0.042 equivalent`、margin 仅约 `.034 mV`，不满足 robust candidate。latch 必须由 `z_safe`/recovery 状态释放，不能写成永久开关。

### P1：边界高度支持 SNIC-like，但还不能正式命名 SNIC

`z=.85,alpha_G=15`：

- fixed-point fold：`A_SN=.3165099207 mV`；
- stable branch last accepted：`.31645 mV`；
- first failed return：`.31648 mV`；
- period：`604.8 ms → 11.35 s`；
- fold-state distance 缩小约 `8445×`；
- `A>=.28 mV` inverse-square-root period fit：`R²=.999987`；
- transverse spectral radius 约 `1e-4–3e-3`，没有趋近 `+1`。

这反对 finite-period cycle fold、支持 saddle-node-on-cycle/infinite-period 图景。但 `.31/.316 mV` 的 FD matrix platform 未通过，exact transfer 在 `.3164 mV` 未于 `12 s` 内返回，而且 directed iteration 不能排除 unstable/disconnected cycle。

**怎么改**：当前标签固定为 `strong_SNIC_like_candidate_formal_label_open`；只有需要正式分岔命名时，再做更长 return window、boundary-aware variational derivative 与必要的 collocation/pseudo-arclength。

### P1：transition-near cycle 有 peak-rate operating-envelope 风险

smooth/exact branch 从 `A=.1/.2 mV` 起 peak 接近或超过 `100 Hz`，exact base 在 `.2–.316 mV` 约 `102 Hz`，half `dt` 约 `101–102 Hz`。虽然占空短、mean rate 低，但不能用 mean 掩盖峰值。

**怎么改**：下一空间节点继续逐点保存 peak 和 `>=100 Hz` occupancy；若 coarse field 只有靠该 ceiling 才产生 front/exit，则该工作点不能升格。

### P1：正式 persistence lock 仍只有 seed 1

本轮完整 causal `A_G/U_TG` history 只有 seed 1。`tau_p=500/750/1000 ms` 的 full-pre-onset midpoint 分别为 `.0793150/.0722287/.0667676`，只能作为 pilot；seeds 3/4 现有文件只有 population rate 和 event history。

**怎么改**：在任何 SNN 参数确认前补采 seeds 3/4 的 raw spatial sensor history；不能用 population rate 伪造 recruited-area 语义。

## 4. 科学性问题

### 做对了什么

1. **加性 current 不是“只压平均率”**：它移动 E-nullcline、low saddle-node 和 cycle bottleneck，方向上是有效 exit coordinate。
2. **快慢分工更清楚**：Z 负责 entry，已有 `S_G` 负责 inner fast cycle，M 负责 exit；没有再增加第三个 recurrent-E divisor。
3. **边界证据比 state fork 强**：period divergence、fold alignment、orbit-to-fold approach 与 multiplier not-`+1` 来自同一个 event-restarted object。
4. **dual-sensor 反证有效**：没有只选择看起来成功的 SNN 或 0D 一侧。
5. **latch ablation 只作定位**：它证明 SNN 晚门控后仍有幅度窗口，但没有被拿来事后改判 primary。

### 哪些想法被结果推翻

1. **“给 additive M 加一个普通 persistence low-pass 就能完成 lifecycle”被推翻。**检测阈值和持续 build 不能共用 memoryless gate。
2. **“homogeneous mature cycle 足以代表发作建立过程”被推翻。**它没有 recruited area，天然从第一圈就是全局成熟状态。
3. **“`Amax=1.6 mV,tau_up≈125 ms` 在真实 sensor 下必然够用”被推翻。**只有理想持续开门或 latch 时够用，原 p 会回落。
4. **“下一步先积分完整 0D”不再合理。**在缺 spatial coordinate 时，0D 通过只能来自人为延迟或 threshold retuning。

### entry / exit 向量场解释

定义：

\[
D(z,m)=z-z_{B_C}(A_{max}m),
\qquad
F(z,m)=A_{max}m-A_{B_C}(z).
\]

entry 需要 `D=0,dot D<0`；exit 等价于 `F=0,dot F>0`。

参考 fold 附近 `z'_B(A)≈-.079 mV^{-1}`，unopposed entry `dot z≈-3.12e-5 ms^{-1}`。若 `Amax=1.6 mV,tau_up=125 ms` 的 gate 提前打开，初始 boundary-shift 项约为：

\[
-z'_B A_{max}\dot m
\approx .079\times\frac{1.6}{125}
\approx1.0\times10^{-3}\ \mathrm{ms}^{-1},
\]

约为 depletion entry vector 的 `32×`。要仅靠放慢 M 避免翻转 entry，`tau_up` 需到约 `4 s`，又必然追不上下降的 Z。因此 entry 前要求的不是“较小 M”，而是 **严格 `dot m=0`**。

- homogeneous phase sensor 很早使 `dot m>0`，在真正 entry 前把 `dot D` 推向正侧，所以表现为 prevention；
- recorded SNN memoryless sensor 的 `dot m` 出现太晚且不连续，`F` 到第 5 个 baseline-period equivalent 仍小于零；
- latch diagnostic 使 crossing 后 `dot m` 保持正值，`F` 才在约第 5 个 equivalent 从负变正。

因此下一版必须让 spatial recruitment 决定 **何时允许 latch set**，而由 `z_safe` 决定 **何时 reset**。单纯增加 A 只改变 `F` 的幅度，不修复 entry vector 的方向错误。

## 5. 工程性问题

- orbit runner：1 process、BLAS 1 thread，wall `37:01`，peak RSS `302264 kB`，无 swap/OOM；
- persistence runner：1 process、BLAS 1 thread，约 `10–27 s`，peak RSS 首轮 `279616 kB`；
- 上游 capture、entry/exit、orbit summary/trace 均 SHA-256 fail-closed；
- `A=0` additive return map 与冻结 Stage 0E parity 测试通过；
- `A=.31` 的 `>1200 ms` return 测试防止旧窗口误删长周期；
- causal onset 从 raw rate 的 strictly trailing 250-ms envelope 重算，未消费 centered sidecar；
- phase drive 使用 raw `recruitment_sensor(rE_fast)`，未错误使用已低通 `mu_G`；
- results 下两个新 `figures/` 均有中文 README；
- 未启动完整 SNN，未接触另一 worktree，资源余量充足。

仍需注意：orbit source 是 smooth transfer 的 `A=0` cycle；选定 checkpoint 有 exact/base-half 复核，但 persistence phase replay 本身不是 exact-transfer cycle sensitivity。

## 6. 最小修改路线

1. **冻结当前 Stage A/B**：不扩大 `Amax/p_r/tau_p` 网格，不运行原完整 0D。
2. **补 spatial sensor artifacts**：对 primary seeds 保存 raw `A_G/U_TG`、局部 active fraction、历史 recruited fraction；先验证共同 causal separation。
3. **先做 `0D+rho` replay oracle**：只检验 area-weighted local occupancy 能否解释 late gate；外加 logistic rho 不能算机制成功。
4. **实现通用 P-patch、先执行 P=1/2**：local fast state 为 7 维，加 local `z/p/m`；全域只能共享一对 `mu_G/S_G`，连续 state shape=`10P+2`。P=1/uniform-P 必须复刻原 Stage 0C。
5. **把 detection 与 latch 分离**：local persistence 与 actual neighborhood recruited fraction 做 AND-set，`z>z_safe + local-low + p<p_off` 做 reset；禁止 ordinary IED/core-only 提前置位。
6. **先过 core–surround 两区**：只检查 `LL→CL→CC→low`、memoryless/latch、rho off/on、m off、cross-zone coupling off；不扫 E→E 参数。
7. **两区通过后才开 coarse 1D field**：原参数直接移植，先 local wake，再单一 broad arm 检查 stall/annihilation。
8. **只有 coarse field 通过才上完整 SNN**：随后再检查 same-low return、early/late retrigger和 seed sensitivity。

## 7. 下一步建议

下一节点不应叫“更强 additive M”，而应叫：

> **spatial-recruitment-set / Z-safe-reset additive recovery latch**。

它仍属于本线：postsynaptic additive recovery + fixed existing fast scaffold；不修改 E→E relay、weight、kernel、delay 或 conductance。核心可证伪问题是：真实 recruited-area coordinate 能否让 latch 在局部 onset 后、全场 prevention 前置位，并在第 3–5 个**真实 Poincaré return**内把局部轨迹横穿 exit boundary。near-boundary period 已达到 `1.6–11.3 s`，后续不能再用固定 `604.8 ms × cycle count` 代替真实 return。

## 8. 产物

### Stable-cycle boundary

- 图：`results/topic4_sef_hfo/mz_additive_orbit_continuation/figures/mz_additive_orbit_continuation.png`
- summary：`results/topic4_sef_hfo/mz_additive_orbit_continuation/orbit_continuation_summary.json`
- branch：`results/topic4_sef_hfo/mz_additive_orbit_continuation/stable_cycle_branch.csv`
- exact checkpoints：`results/topic4_sef_hfo/mz_additive_orbit_continuation/exact_transfer_checkpoints.csv`
- derivative ladder：`results/topic4_sef_hfo/mz_additive_orbit_continuation/poincare_derivative_ladder.csv`
- traces/matrices：`results/topic4_sef_hfo/mz_additive_orbit_continuation/{cycle_states_and_traces,poincare_matrices}.npz`

### Persistence Stage B

- 图：`results/topic4_sef_hfo/mz_persistence_feasibility/figures/mz_persistence_dual_sensor_feasibility.png`
- summary：`results/topic4_sef_hfo/mz_persistence_feasibility/persistence_feasibility_summary.json`
- thresholds：`results/topic4_sef_hfo/mz_persistence_feasibility/persistence_threshold_audit.csv`
- all races：`results/topic4_sef_hfo/mz_persistence_feasibility/dual_sensor_races.csv`
- arm verdicts：`results/topic4_sef_hfo/mz_persistence_feasibility/dual_sensor_arm_verdicts.csv`
- traces：`results/topic4_sef_hfo/mz_persistence_feasibility/dual_sensor_traces.npz`

设计与执行合同：`docs/superpowers/specs/2026-07-20-topic4-mz-persistence-gated-additive-spatial-lifecycle-design.md`。
