# Topic 4 MZ-FCXR：逐细胞 spike-load → electrogenic pump 生命周期设计

日期：2026-07-26

状态：**DESIGN LOCKED FOR P0–P2 ONLY；P3/P4 为 gated roadmap，不自动执行**

前置验收：
`docs/archive/topic4/sef_hfo/mz_fcxr_heo_line_acceptance_2026-07-26.md`

本版根据 2026-07-26 两轮外部审阅完成六项承重修订：

1. pump membrane effect 改为 distributionally baseline-centered，不再使用 positive-part rectification；
2. primary load 改成无量纲 activity-dependent spike-only mass balance，不宣称为具体 Na concentration；
3. Gate I 拆成硬前置的 I-a instrument 与非阻塞 I-b diagnostic，真实 target 归入 Gate E；
4. 40k P0 改为 matched-common-noise empirical finite-time response operator，exact Floquet 降为条件性项目；
5. frozen topology 统一为 `Z×P` pump-activation map，并增加 branch-conditioned slow flow；
6. 因果门新增 Z-frozen/pump-only 与 pump-off/Z-only 分解，区分独立、协同和 Z-dominant termination。

## 0. 核心科学目标

目标不是制造固定节律，也不是把当前 16 Hz 状态调成另一条永久振荡。目标是在同一 E1146 空间 scaffold 上得到一次自然、有限、有空间结构的 excursion：

```text
稀疏、不规则 IED 的稳定统计邻域
→ 有界的 ictal-like high/bursting excursion
→ postictal suppression
→ 回到原有 IED 稳定概率分布
```

“回到间期”不是回到周期轨道，而是：

> 回到能够按原有统计规律产生稀疏、不规则 IED 的状态空间邻域。

最终目标图包含：

1. virtual-SEEG 的间期—发作—爆后—间期；
2. 慢变量相图与 branch-conditioned slow flow；
3. 间期和发作早期能量场的 scaffold alignment；
4. 不同阶段的 empirical spatial response modes / stimulation susceptibility。

本 P0–P2 只锁生命周期。真实 E1146 的约 3–8 Hz 尖波、`1–80 Hz` 宽带形态属于 Gate E；pump-only 即使闭环但仍窄带，也只能称 lifecycle scaffold。

## 1. 已锁定 substrate 与变量职责

### 1.1 Fast substrate 不再重调

- `L=20`，`N=40000`，E1146 registered scaffold / narrow montage；
- connectivity seeds `1,3`；
- `dt=0.05 ms`；
- external/feedforward AMPA 为 additive current；
- recurrent E→E 为 reversal-aware conductance；
- recurrent-only smooth saturation：
  `g_rec_eff=g_sat·tanh(g_rec_raw/g_sat)`，`g_sat=21.6`；
- HEO1 cooperative recurrent transform 使用已选 16 Hz anchor；
- `M=off`，`X≡1`，无新的 global/area brake；
- 新机制全部 off-by-default；pump off 时 byte-parity。

当前高态只称：

**bounded coherent common-mode oscillatory branch**。

其约 15.6 Hz、跨电极 coherence 约 0.97、相位跨度约 168°。它不是零相位同步，但仍是 common-mode dominated，也尚未跨 seed 证明为 full-state deterministic periodic orbit。

### 1.2 慢变量职责

| 变量 | 职责 | P0–P2 |
|---|---|---|
| `Z_i` | inhibition robustness / permissivity；onset arm | 保留 |
| `M_i` | 2–4 Hz burst envelope / waveform shaping | 关闭 |
| `X_i` | within-burst relay collapse | 固定 `1` |
| `u_i` / pump | termination + postictal memory | 唯一新增机制 |
| broad/area feedback | common-mode selective restraint | 不实现 |

这三个问题不得混写：

- lifecycle topology；
- ictal waveform；
- spatial-mode selection。

## 2. 无量纲 load–pump 方程

### 2.1 Primary：spike-only load

令 `u_i≥0` 为 E-cell 的无量纲 **activity-dependent intracellular load（Na/pump-inspired）**：

\[
\phi(u)=\frac{u^h}{1+u^h},
\qquad h=3\ \text{（primary fixed）}.
\]

连续形式：

\[
\dot u_i
=
a_{\mathrm{load}}S_i(t)
-\frac{1}{\tau_N}\phi(u_i).
\]

其中 `S_i(t)` 是 spike train；load clearance 与 electrogenic current 必须由同一个 `φ(u)` 驱动。

第一阶段自由参数仅为：

- `a_load`：每个 E spike 的 load jump；
- `tau_N`：pump-mediated load release time；
- `I_max`：膜上的最大 electrogenic effect。

`h=3` 不扫。Tier A 通过后才允许 `h∈{2,4}` sensitivity。

### 2.2 离散更新与因果顺序

每步：

\[
u_i(t+\Delta t)
=
\max\left[
0,\,
u_i(t)
+a_{\mathrm{load}}N_i^{spike}(t,t+\Delta t)
-\frac{\Delta t}{\tau_N}\phi(u_i(t))
\right].
\]

锁定顺序：

1. membrane step 使用 `u_i(t^-)`；
2. 生成本步 spike；
3. clearance 与 spike jump 更新到 `u_i(t^+)`；
4. 新 spike 只从下一步开始影响 pump current。

`a_load` 是每 spike jump，不再乘或除 `dt`。任何 safety cap 只作 fail-fast；候选撞 cap 即失效。

### 2.3 Distributionally baseline-centered membrane effect

在独立 sensor-only baseline calibration 中估计：

\[
p_{0,i}
=
\mathbb E_{\mathrm{baseline}}\left[\phi(u_i)\right].
\]

E-cell 膜方程写为：

\[
\tau_{m,E}\dot V_i
=
F_{\mathrm{FCXR-HEO}}(V_i,\mathrm{inputs},Z_i)
-I_{\max}\phi(u_i)
+I_{\max}p_{0,i}.
\]

即：

\[
I^{pump,excess}_i
=
I_{\max}\left[\phi(u_i)-p_{0,i}\right].
\]

**禁止 positive part**。`+Imax·p0` 是对原 FCXR baseline 已隐含稳态的补偿，不是 pump 反向工作；负的 excess 表示 pump activation 低于 baseline reference，而不是生理 pump 反向运转。这样既消除随机 baseline 波动造成的正平均偏置，也保留局部响应分析所需的光滑性。

### 2.4 `p0_i` shrinkage

不把单 seed、短 baseline 的 32000 个 raw per-cell means 直接固化为空间参数场。

锁定方案：

1. sensor-only baseline 分 calibration / final held-out blocks；
2. 细胞按 pump-off baseline firing-rate decile 分组，不按 source/sink/空间位置分组；
3. raw per-cell `mean φ(u_i)` 向 decile mean 做 empirical shrinkage；
4. shrinkage weight 只在 calibration blocks 内以预注册的 inner block-CV prediction error 选择；
5. final held-out block 不参与 grouping、shrinkage strength、margin 或任何 threshold 的拟合；
6. 选择规则在 lifecycle 运行前锁定；
7. final held-out baseline 上只做一次 pump-on vs pump-off equivalence。

必须保存 raw、group mean、shrunken `p0_i` 与 shrinkage strength。`p0_i` 不得形成预设 source→sink 梯度。

### 2.5 Synaptic influx 降为 sensitivity

primary 不使用 `g_rec_raw`。它既没有 driving force，也没有经过实际 `tanh` saturation，不能解释为具体离子 influx。

Tier A 通过后才允许三个 sensitivity：

1. spike-only；
2. excitatory-charge-only：
   \[
   Q_{E,i}=
   [I^{ff}_{E,i}+g^{rec,eff}_{E,i}(E_E-V_i)]_+;
   \]
3. combined。

不得在 primary 同时扫描 spike/synaptic ratio。

## 3. 总 gate 架构

```text
Gate I-a  pump instrument validity              # hard prerequisite
Gate I-b  response-operator diagnostic          # non-blocking for lifecycle
→ Gate T topology + branch-conditioned slow flow
→ Gate C causal lifecycle
→ Gate S spatial scaffold preservation
→ Gate E empirical seizure compatibility
```

正式依赖链是 `I-a → T → C → S → E`。`I-b` 可与 P1/P2 并行；失败不阻止生命周期实验，但禁止 response-mode/eigenmode claim 和最终图的右侧 susceptibility panel。

只有 `I-a+T+C+S` 才称 **lifecycle scaffold**。

只有再通过 `E` 才称 **data-consistent seizure lifecycle candidate**。

## 4. Gate I-a / I-b：instrument 与 diagnostic 分层

### I1. Sensor-only baseline calibration

对每个候选 `(a_load,tau_N)`：

- `Imax=0`；
- burn-in 后收集 event-count-driven baseline；
- 输出 `u_i`、`φ(u_i)`、`p0_i` shrinkage；
- isolated IED 必须产生可测 load excursion；
- typical IED 不得长期把 `φ(u)`钉在 1。

若 IED 对 load 完全不可见，机制退化成隐藏 ictal gate；若 baseline 已 pump-saturated，机制不兼容。

### I2. Pump-on baseline equivalence

使用 final held-out baseline noise：

- pump off；
- pump on + `p0_i` compensation；
- common random numbers；
- 不允许重估 `p0_i`。

验收使用 calibration blocks 的 pump-off block-to-block variability 定义 equivalence margin，不用“不显著”冒充等价。每个 primary metric 的 margin 在查看 final held-out pump-on 结果前写入 `baseline_variability.json`；final held-out 只作一次判定，不反向改 margin。

至少比较：

- IED rate / IEI median / IEI CV；
- duration / participation / peak；
- source/axis/off-axis activity；
- virtual-SEEG band-power distribution；
- forward/reverse template readout。

任一 primary metric 超出预锁 equivalence margin，Gate I-a fail。

### I3. Virtual-SEEG contribution audit

当前 `LFPRecorder` 是 E-cell `|I_E|+|I_I|` proxy。P0 必须明确它不是临床电位的完整 forward model。

同时输出以下**虚拟 SEEG proxy**：

```text
V_legacy              = existing |I_E|+|I_I| proxy
V_no_direct_pump      = signed/weighted synaptic components, no pump term
V_all                 = same readout + direct pump-current sensitivity
```

这里的 `V_*` 是为了与后续 figure contract 对齐的 artifact label，不代表已经得到物理单位为伏特的完整 forward solution。P0 必须先核查：非 blessed observer 实际能否取得带符号、带 driving-force 的所需分量。若只能取得 `|I_E|+|I_I|` 或其他幅值 proxy，则必须把输出和结论显式标为 proxy；不得补想象中的符号，也不得为了让 Gate I-a 通过而静默修改 blessed `lfp.py`。若缺少必要状态使 `V_no_direct_pump` 无法构造，Gate I-a 以 `READOUT_NOT_IDENTIFIABLE` fail，并另开受审阅的 engine-change sprint。

并分解：

- excitatory synaptic；
- inhibitory synaptic；
- adaptation（P0–P2 为 0）；
- pump。

Gate E primary 使用 `V_no_direct_pump`。去掉 direct pump 后，网络活动本身仍须产生目标频谱重构；不得靠慢 pump current 直接制造低频功率。

### Gate I-a GO

- pump-off byte parity 与既有 Z/M/X update-order contract 均通过；
- baseline compensation 在 final held-out 上等价；
- pump/load 的逐步因果顺序可验证；
- readout contribution 可审计；若 readout 不可识别，不允许伪造 signal component，Gate I-a 不通过。

只有 Gate I-a PASS 解锁 Gate T。真实 E1146 target extraction 和 response operator 都不是 Gate I-a 的阻塞项。

### I-b0. Dynamical regime classification

当前 15.6 Hz branch 不预先称为 attractor/limit cycle。对 low/high/kick-release 三类 IC，在重复 noise 与 deterministic/replayed-drive 条件下分类为：

1. `DETERMINISTIC_PERIODIC_CANDIDATE`；
2. `STOCHASTIC_OSCILLATORY_REGIME`；
3. `METASTABLE_HIGH_ACTIVITY`；
4. `UNRESOLVED`。

分类使用 cycle-to-cycle return、phase/frequency diffusion、不同 IC 收敛、双时间窗 persistence 与 noise-replay sensitivity。只有第一类才具有另开 exact Floquet 的资格；其他类别只报告 empirical finite-time response。

### I-b1. 40k empirical finite-time response operator

40k stochastic hybrid LIF 的必做对象不是 full-state Jacobian，而是：

> **empirical finite-time response operator**。

它是 coarse observables 在 matched common noise 下的经验有限时响应映射，不是 stochastic hybrid SNN 的 exact mathematical Jacobian。

定义固定 coarse observable：

\[
y(t)=
\Pi_{\mathrm{coarse}}
[r_E(x,t),r_I(x,t),g_E^{eff}(x,t),g_I(x,t)].
\]

primary 使用 `20×20` spatial bins；`32×32` 仅作 sensitivity。输入 basis 至少包含：

- common；
- source-vs-sink/core-differential；
- axial low-k phase pair；
- transverse low-k phase pair；
- source-localized / sink-localized / off-axis matched probes。

用相同 connectivity、相同 noise realization、`±ε` paired perturbations 估计：

\[
\delta y(t+\Delta)\approx A_\Delta(t)\delta y(t).
\]

报告：

- eigenvalues/eigenvectors of the fitted coarse map；
- singular vectors / finite-time gain；
- common/axial/transverse projection；
- perturbation decay/amplification；
- amplitude linearity与重复性；
- left/right response asymmetry。

### I-b2. Small-network validation；exact Floquet 条件化

必做：

- 先在同构 `N≈1000` 网络验证 coarse operator 对 `ε`、noise replay 和 binning 的稳定性；
- 若 mode 排序不稳，再做 `N≈4000` sensitivity；
- naive full-state finite difference 不作为 ground truth。

只有满足以下条件，才另开 exact Floquet：

- deterministic/repeated drive；
- Poincaré section 收敛；
- full-state periodic orbit，而非 population quasicycle；
- reset-aware/saltation tangent 在小网络通过。

exact Floquet 不是 pump sprint 的阻塞 gate，不得在 40k stochastic SNN 上直接宣称 full-state multiplier。

### Gate I-b disposition

- `PASS`：operator 对 amplitude/noise/binning 可重复，可进入 spatial response-mode claim；
- `UNRESOLVED/FAIL`：仍可进入 Gate T/C/S-core，但不得声称 eigenmode/Floquet，不得生成最终图右侧 susceptibility panel；
- I-b 失败时只允许修 observable/basis/hook，不得用它反向调 pump lifecycle 参数。

## 5. Gate T：fast topology + slow flow

### T1. Activity-shaped load fields

在 `Imax=0` 的 sensor-only high branch 上记录：

\[
u_i^{high}(t_s),\quad t_s\in\{0.5,1,2,3\}\ \mathrm{s}.
\]

用 matched baseline field `u_i^0` 构造：

\[
u_i(\rho_u)
=
u_i^0+\rho_u[u_i^{high}(t_s)-u_i^0].
\]

其中 `rho_u` 只用于生成 activity-shaped field，不是相图坐标。fast system 的相图坐标统一写成 excess pump activation：

\[
\bar p
=
\frac1{N_E}\sum_i[\phi(u_i)-p_{0,i}],
\]

而不是模糊的 `N` 或非法的 `n=0`。

### T2. 三组 frozen `Z×P` pump-activation maps

对已有 activity-shaped Z depletion field 与 pump field 做：

```text
Primary: activity-shaped per-cell pump field
Control 1: mean-matched uniform pump field
Control 2: value-matched spatial shuffle
```

每个 `(rho_Z, bar_p)` cell 跑 low/high/kick-release IC，分类：

- low only；
- high only；
- coexistence/bistable-like；
- bounded periodic-like；
- metastable transient；
- unsafe。

uniform 和 shuffle 必须匹配 `mean[φ(u)-p0]`，不是匹配 raw `u`。

### T3. Branch-conditioned slow flow

对每个 frozen-map cell 和每个存在的 fast branch，评估：

\[
G_{\mathrm{branch}}(Z,P)
=
\left\langle
\dot{\bar Z},\,
\dot{\bar p}
\right\rangle_{\mathrm{branch}}.
\]

同时保存空间投影：

\[
U_\parallel=\langle(u-u_0)\psi_\parallel\rangle,\qquad
U_\perp=\langle(u-u_0)\psi_\perp\rangle,
\]

以及 core A/B difference、spatial CV、participation ratio、common/axial/transverse overlap。

必须检查：

1. low branch 上存在稳定 interictal slow-flow neighborhood，`Z` 不持续单调倒计时；
2. high branch 上 pump flow 指向 exit surface；
3. offset 后 low branch 上 Z flow 指向恢复；
4. pump release 前系统已进入 `u→u0` reset 后仍 low-stable 的区域；
5. 不出现非目标 deterministic recurrent-seizure oscillator。

### Gate T GO

- healthy low branch 保留；
- impaired Z + low pump 存在高支；
- pump 增加选择性移除高支；
- exit 附近 low branch 仍存在；
- branch-conditioned slow flow 指向可闭合 excursion；
- topology 不是 cap / early-stop artifact。

不预设 entry/exit 必须是 Hopf 或 fold。

## 6. Gate C：causal lifecycle

### C1. Interictal stationarity

`no kick` 只是必要条件，不足以证明 spontaneous onset。

观察窗之前先 burn-in，直到：

- `Z_i`、`u_i`、IED rate、IEI 达到 block-stationary；
- event count 达到 baseline-variability pilot 的要求；
- `bar Z` 没有 typical-IED 下的持续单调漂移。

要求：

1. onset latency 随 noise seed 变化，不固定在初始化后的同一秒；
2. onset 前出现 IED clustering 或特定 slow-state excursion；
3. observation-start 平移不固定 onset latency；
4. matched mean-rate、declustered Z-sensor replay 显著降低 onset probability。

若只得到确定性慢漂移，允许保留，但标签降为：

**autonomous slow-variable-driven excursion**，不得称 spontaneous transition。

### C2. 参数反推，不做大网格

- `a_load/tau_N` 由 sensor-only IED 与 high-state accumulation 时间反推；
- `Imax` 由 Gate T exit surface 反推；
- dynamic Z 参数必须先过 stationarity；
- primary candidate 锁定后只做 one-axis-at-a-time `±20%`；
- 禁止 `Z×a_load×tau_N×Imax` Cartesian sweep。

### C3. Primary dynamic run

开发阶段：

- connectivity seed 1；
- 一个显式 development noise seed；
- no kick；
- `M=off`、`X=1`；
- run length 由事件数和 lifecycle 决定，不把 40 s 当 confirmatory 上限。

候选必须有：

- stationary interictal pre-window；
- bounded high excursion；
- autonomous offset；
- postictal suppression；
- late statistical return；
- no clip/nonfinite/cap。

### C4. Paired causal counterfactuals

所有对照从 t=0 以同一 network、同一后续 noise 运行；scheduled intervention 前必须 byte-identical，记录 prefix hash。

**Necessity 1 — pump-current knockout**

- pre-offset 时令 membrane pump current 为 0；
- 保留 `u_i` dynamics；
- offset 应显著延迟或消失。

**Necessity 2 — load reset**

- pre-offset 时令 `u_i→u_i^0`；
- 高态应重新可持续或 offset 延迟。

**Z–pump termination decomposition**

从同一个 established-high snapshot、同一后续 noise 做四臂：

```text
combined     native Z recovery + native pump
pump_only    freeze Z at the ictal/permissive field + native pump
Z_only       allow Z recovery + reset u to u0 + disable pump current
neither      freeze Z at ictal field + reset u to u0 + disable pump current
```

判读：

- `pump_only` 仍能 offset，且 `Z_only` 不能或明显更晚：`PUMP_DOMINANT_EXIT`；
- combined 能 offset、两个单独臂均失败或显著更弱：`COOPERATIVE_Z_PUMP_EXIT`；
- `Z_only` 与 combined 等效、`pump_only` 失败：`Z_DOMINANT_EXIT`，不得称 pump termination；
- neither 也 offset：snapshot/noise 自终止，当前反事实无辨识力。

四臂只复用同一个 candidate snapshot，不扩参数网格。

**Sufficiency**

- 记录真实 candidate 的 `u_i^preoff`；
- 提前施加到 matched established-high snapshot/replay time；
- 不改 Z/connectivity；
- 应提前终止。

**Postictal-memory causality**

- offset 后立即 `u_i→u_i^0`；
- postictal suppression / early retrigger resistance 应缩短或消失。

**Spatial attribution**

- per-cell preoff field；
- mean-matched uniform field；
- spatial shuffle。

artifact：
`causal_counterfactuals_seed{connectivity}_noise{noise}.json`。

### C5. Time ordering 改为 surface + counterfactual

报告：

```text
t_onset
t_u_departure              # descriptive，可早于 onset
t_pump_exit_surface
t_offset
t_z_low_safe
t_pump_release
t_first_postictal_IED
t_statistical_return
```

硬门：

```text
t_pump_exit_surface ≤ t_offset + tolerance
pump-current knockout delays/prevents offset
termination_attribution ∈ {PUMP_DOMINANT_EXIT, COOPERATIVE_Z_PUMP_EXIT}
at pump release, u-reset counterfactual remains in the low basin
no rebound before statistical return
```

不再强制 `u` 必须 onset 后才上升，也不强制 Z 必须 offset 后才恢复。

### C6. 事件数驱动统计恢复

baseline 与 late-recovery 窗：

\[
T_{\mathrm{window}}
=
\max(8\ \mathrm{s},\ \text{收集预锁 }N_{\mathrm{IED}}\text{ 个事件所需时间}).
\]

`N_IED` 由 baseline variability pilot 在 lifecycle 前锁定；最低不得少于 20 个事件，若 template/direction CI 尚未稳定则继续采集。

统计方法：

- baseline 划分独立 blocks；
- block-to-block variability 定 equivalence margin；
- time-block bootstrap，不把事件当独立样本；
- late recovery 必须通过 equivalence，不用“未显著不同”。

至少验收：

- event rate / IEI median / CV；
- duration / participation / peak；
- band-power distribution；
- source/axis/off-axis activity；
- forward/reverse template similarity 与 direction ratio。

### C7. Holdout replication

RNG 分开记录：

```text
connectivity_seed
noise_seed
initial_state_seed
perturbation_seed
```

参数选择只用：

- connectivity seed 1；
- development noise seed。

锁参后 confirmatory 最低配置：

```text
connectivity seeds {1,3}
× 每个 connectivity 3 个未见 noise seeds
= 6 trajectories
```

报告：

- lifecycle success probability；
- onset latency distribution；
- excursion duration；
- postictal duration；
- statistical-return probability。

所有 paired causal controls 使用 common random numbers。

### Gate C GO

- stationarity 通过；
- onset 不是初始化倒计时；
- pump necessity + sufficiency 成立；
- Z–pump decomposition 排除 `Z_DOMINANT_EXIT` 和无辨识力的 spontaneous decay；
- postictal memory 对 load reset 敏感；
- late window equivalently returns；
- holdout trajectories 支持同方向结论。

## 7. Gate S：spatial scaffold preservation

Tier A 增加 `Tier A-S`：

1. onset 不能退化成全 sheet simultaneous ignition；
2. early recruitment 优先沿 E1146 behavioral axis；
3. early ictal field 与 interictal scaffold alignment 不低于 pump-off substrate 的预锁 margin；
4. `p0_i` calibration 不编码永久 source→sink 方向；
5. recovery 后 forward/reverse IED probe 回到 baseline band；
6. off-axis broadening 只有真实 E1146 分布支持时才作硬目标；
7. activity-shaped pump field 必须优于或可区别于 mean-matched uniform/shuffle。

保存：

- `U_parallel / U_perp`；
- core A/B load difference；
- spatial CV；
- participation ratio；
- common/axial/transverse response projection；
- source/sink/off-axis matched perturbation response。

Gate S fail 时，即使时间轨迹闭合，也不得称 lifecycle scaffold。

## 8. Gate E：empirical seizure compatibility

### E0. 真实 E1146 target distribution

target extraction 是 Gate E 的前置分析，不阻塞 Gate I-a、Gate T 或 Gate C。

不再把单个 seizure、单个精确六频段向量当 target。建立：

- 多 seizure；
- 多 contacts；
- interictal / onset / early ictal / established ictal 多窗口；
- band power、PLV、recruitment、sharpness、burst rate 的分布与 confidence interval。

若有足够 eligible seizures，留至少一场为 holdout；若不能形成 holdout，只允许 exploratory Gate E。

模型—数据状态向量：

```text
[P1-4, P4-8, P8-13, P13-30, P30-80,
 PLV, recruitment, sharpness, burst_rate]
```

`80–150 Hz` 单列描述，不作必须抬升的 target。

Gate E 完全使用 E0 预先锁定的真实分布与 holdout；target extraction 失败只会使 Gate E `UNRESOLVED`，不会否定 pump instrument 或 lifecycle。

要求：

- 模型状态轨迹进入真实 E1146 9D ictal manifold；
- 约 3–8 Hz 尖锐 burst；
- `1–80 Hz` 多频带抬升；
- phase/recruitment trajectory 与真实阶段序列兼容；
- primary `V_no_direct_pump` 仍通过；
- 结论不由 direct pump readout 人工制造；
- held-out real seizure 不参与 threshold 定义。

Gate E 不通过，但 I-a/T/C/S 通过：

> lifecycle scaffold PASS；data-consistent seizure compatibility NOT PASS。

## 9. P0–P2 执行边界

### P0：Gate I-a + non-blocking Gate I-b

- load/pump baseline calibration；
- virtual-SEEG decomposition；
- Gate I-a adjudication；
- empirical finite-time operator（I-b diagnostic，不阻塞 P1/P2）；
- conditional dynamical-regime/Floquet eligibility classification。

E0 real-target distribution 可以并行构建，但只在 Gate E 使用，不属于 Gate I-a。

### P1：Gate T

- activity-shaped frozen map；
- uniform/shuffle controls；
- branch-conditioned slow flow；
- entry/exit topology。

### P2：Gate C + Gate S + Gate E readout

- dynamic stationarity；
- lifecycle candidate；
- causal counterfactuals；
- event-count equivalence；
- 6-trajectory holdout；
- spatial preservation；
- empirical compatibility readout。

P0、P1 或 causal development candidate 未通过时，不进入大规模 holdout。

## 10. Deferred roadmap：不属于本 executable lock

### P3：`M_i` burst morphology

仅在 I-a/T/C/S 通过后另写 spec。目标是 2–4 Hz burst envelope，不得重新承担 termination。

### P4：broad/area common-mode feedback

仅在 lifecycle 已闭合、empirical operator 仍显示 common mode 压倒 transverse/axial modes 后另写 spec。必须有 rank-one/common-mode projection 与 area-matched controls。

P3/P4 不自动执行，不包含在本 plan 的预算或授权内。

## 11. Artifact 与图合同

结果根：

```text
results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle/
```

必须有：

```text
STATUS.md
run_manifest.json
baseline_variability.json
pump_baseline_calibration.json
pump_baseline_equivalence.json
gate_Ia.json
gate_Ib.json
real_target_distribution.json
virtual_seeg_component_audit.json
dynamical_regime_classification.json
finite_time_operator.json
frozen_topology_map.json
branch_slow_flow.json
stationarity_gate.json
lifecycle_verdict_*.json
causal_counterfactuals_*.json
spatial_preservation_*.json
holdout_summary.json
resource_log.jsonl
figures/README.md
```

诊断图：

1. `instrument_baseline_equivalence.png`
2. `real_target_trajectory_distribution.png`
3. `virtual_seeg_component_audit.png`
4. `finite_time_response_modes.png`
5. `frozen_topology_and_slow_flow.png`
6. `lifecycle_and_counterfactuals.png`
7. `statistical_return_equivalence.png`
8. `spatial_scaffold_preservation.png`

只有 `I-a+T+C+S` 通过才生成 lifecycle candidate figure；只有再过 E 且 I-b 支持 response-mode claim，才生成包含右侧 susceptibility panel 的 paper-ready 四栏图。

## 12. 工程、OOM 与 detached-run

### 12.1 Worktree

- 新 sprint 使用独立 `codex/topic4-mz-fcxr-pump-lifecycle` worktree；
- 记录 base commit、dirty state、blessed hashes；
- 不修改 sibling worktree；
- guarded engine edit 若不可避免，必须先停并报告，不自主 re-bless。

### 12.2 Worker

既有 40k run 峰值 RSS 约 20 GB：

- `T<20 s`：最多 2 workers；
- `T≥20 s`：最多 1 worker；
- `OMP/OPENBLAS/MKL/NUMEXPR=1`；
- 不保存 `n_steps×N_cell` dense state；只存 streaming summaries 与少量 landmark fields。

### 12.3 OOM stop

- launch 前记录 `MemAvailable` / swap baseline；
- swap delta `>256 MiB`：停止提交；
- swap delta `>512 MiB` 且继续上升，或 `MemAvailable<2×` 单 run peak：只停自己的最新任务；
- 写 `RESOURCE_PAUSED.json` / `ABORTED.json`；
- 不杀 sibling / user process。

### 12.4 Nohup

```bash
setsid nohup <runner> > <run_dir>/nohup.log 2>&1 < /dev/null &
```

写：

- `launcher.pid`
- `RUNNING.json`
- `DONE.json` / `FAILED.json`
- `resource_log.jsonl`

恢复会话后用 PID、command line、sentinel 三重核对。

## 13. Stop rules

出现任一情况停止扩网格：

1. pump off parity 失败；
2. `p0_i` compensation 未实现 held-out baseline equivalence；
3. IED 完全不驱动 load，或 baseline pump 饱和；
4. readout 的“宽带改善”主要来自 direct pump term；
5. empirical operator 对 amplitude/noise/binning 不稳定：停止 I-b 扩展并撤回 response-mode claim，但不自动停止 lifecycle；
6. frozen topology 没有 exit-only corridor；
7. slow flow 不指向闭合 excursion；
8. dynamic onset 是 Z 初始化倒计时；
9. pump necessity/sufficiency counterfactual 失败；
10. postictal memory 对 load reset 不敏感；
11. late window 未等价回到 baseline；
12. onset 退化成全 sheet simultaneous ignition；
13. 需要同时调 drive/connectivity/cooperative gain/M/X/global brake；
14. `u` 或 conductance 撞 safety cap；
15. OOM/swap gate 触发。

Pump no-go 后不得自动开启 P3/P4。

## 14. 允许的结论层级

### Gate I-a only

“load/pump instrument 与 baseline compensation 通过；尚未证明 topology 或 lifecycle。”

### Gate I-a+T

“frozen fast topology 与 branch-conditioned slow flow 支持/不支持 pump-mediated exit corridor；尚未证明动态因果闭环。”

### Gate I-a+T+C

“pump 对 termination 与 postictal memory 具有/不具有独立、协同或非承重作用；空间 scaffold 尚需 Gate S。”

### Gate I-a+T+C+S

“在 holdout stochastic trajectories 上得到保留 E1146 scaffold 的 lifecycle scaffold；真实波形兼容性仍待 Gate E。”

### Gate I-a+T+C+S+E

“得到与真实 E1146 状态序列兼容的、可恢复时空 seizure-like lifecycle candidate。”

Gate I-b 只决定能否追加 empirical response-mode/susceptibility claim；它不得被混写成 lifecycle gate。

工程 green、单条漂亮轨迹、两个 seed 成功、rate 下降或图已生成，都不能单独升级科学结论。
