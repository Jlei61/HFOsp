# FCXR-LC2 — SNN-calibrated hysteretic carrier, delayed-load offset, and interictal recovery

日期：2026-08-01

状态：**DESIGN LOCK CANDIDATE — REVIEW REQUIRED BEFORE EXECUTION**

中文名：**FCXR-LC2：局部迟滞高态—持续负荷终止—间期统计恢复**

Plan：`docs/superpowers/plans/2026-08-01-topic4-fcxr-lc2-phase0-phase3.md`

> **代际边界**：LC2 从 FCXR-HYB2 的正式收口提交 `3c2fc86a` 分出；它不是 HYB3。
> HYB2 已 `EXECUTED · CLOSED`，不得在本分支降低 Z、移动 A0 窗口、提高 occupied-volume
> ceiling 或扩展 ELR 参数。

> **授权边界**：第一份 plan 只允许 Phase 0–3：上游冻结、真实目标锁、SNN-calibrated reduced
> model、H/M/X/Z 的低维几何，以及缩小但动力学匹配的空间 SNN 七门。**不授权 40k lifecycle、
> K/Na、HYB2/ELR、recruited-area integrator、eigenmode 主分析或最终论文图。**

---

## 0. 一句话目标

在不改变 E1146 患者特异各向异性 E→E 病理轴、双端低阈值 core 和 RC1 有界膜方程的条件下，
证明重复间期事件可以通过 Z 将网络推过一个**由局部慢 recurrent conductance H 支持的迟滞
onset surface**；逐细胞 M_i 只塑造 high-state 内部的宽带与错相活动；持续负荷 X 使 high branch
自主消失并维持 postictal protection，最终返回能按原统计规律产生稀疏、不规则 IED 的状态空间邻域。

核心链条固定为：

\[
\text{repeated IEDs}\to Z\downarrow\to \Sigma_{on}\to H\text{ carrier}
\to M_i\text{ morphology}\to X\uparrow\to\Sigma_{off}
\to\text{postictal protection}\to Z\uparrow\to\text{IED statistics return}.
\]

“恢复”不是回到一个固定数值点或周期轨道，而是回到**同一个间期统计邻域**。

---

## 1. 正式冻结的上游资产与负结果

### 1.1 可复用且不得重新争论的资产

1. **RC1 fast membrane**：feedforward AMPA additive；只把 recurrent E→E 改为 reversal-aware
   conductance；`g_sat*tanh(g_raw/g_sat)` 平滑饱和；两个 primary seed 与 `g_sat±20%` 保住间期。
2. **空间衬底**：L=20、各向异性 E→E 病理轴、两个低阈值 core、原连接/延迟/权重归一化。
3. **Z**：现有 `MZSlowVars` 的逐 E 细胞抑制可用度；高抑制负荷使 z 下降，低负荷使 z 恢复；
   非对称 `tau_z_down/up` 已实现并有 off-parity。
4. **X termination authority**：LC1 的 persistence sensor `y_j` 与 presynaptic E→E relay
   availability `x_relay,j` 已在两个 seed 上终止一次持续高态。
5. **M_i**：逐细胞 adaptation history 的差异承载部分宽带化与去同步；mean-field M 不能替代。
6. **观测层**：current-based virtual SEEG、15-contact E1146 montage、空间 source/core/axis/off-axis
   读出、snapshot/restart 和数值安全合同。
7. **六个 blessed engine 文件**：`kick_probe.py / params.py / model.py / connectivity.py /
   connectivity_rot.py / lfp.py` 保持逐字节冻结。

### 1.2 只作机制组件、不得包装成 lifecycle 的结果

- 约 16 Hz 高相干共同模态不是目标 ictal carrier；
- X 终止一次高态不等于统计恢复；
- 钾反馈的额外参与细胞/半径不等于已证明传播；
- HYB1 是 concentration-memory baseline no-go；
- HYB2 是 baseline-safe but `A0_UNDECIDABLE_ALL_LEVELS`；
- kick、hard reset、外部 parameter step 只可作 basin/causality probe，不计入 lifecycle acceptance。

证据索引：

- `docs/archive/topic4/sef_hfo/mz_fcxr_heo_line_acceptance_2026-07-26.md`
- `docs/archive/topic4/sef_hfo/mz_fcxr_lc1_bounded_negative_2026-07-23.md`
- `docs/archive/topic4/sef_hfo/fcxr_hyb1_baseline_disturbed_2026-07-31.md`
- `docs/archive/topic4/sef_hfo/fcxr_hyb2_a0_undecidable_2026-08-01.md`

---

## 2. 角色分解与坐标语义

| 模块 | 唯一职责 | 明确禁止 |
|---|---|---|
| `Z` | event-driven entry coordinate | 不负责 termination；不从 X 得到直接恢复项 |
| `H` | 低/高分支、carrier existence、迟滞 | 不独自承担 3–8 Hz/宽带/空间错相 |
| `M_i` | high-state morphology、节律、逐细胞相位异质 | 不决定 carrier 是否存在；不负责 lifecycle offset |
| `X` | sustained-load offset 与 postictal protection | 不读离线 seizure label；不 hard-reset H/Z |
| anisotropic E→E | onset/recruitment axis | 不新增 ictal mask 或新 E→E 边 |

### 2.1 符号必须与现有代码一致

- `z_i` 是 inhibitory availability；定义 depletion `d_i=1-z_i`。
- 现有 `x_relay,j` 是 presynaptic E→E availability，活动持续时**下降**；LC2 reduced 负荷坐标定义为
  \(x=1-\overline{x_{relay}}\)，所以 termination load 上升时 x 上升。
- `m_i` 是逐细胞 adaptation state。
- 新 `h_i` 是**局部慢 recurrent-conductance state**；不得复用 HEO1 瞬时 `coop_A` 的名字或语义。
- LC2 primary 中 `coop_A=0`。HEO1 algebraic cooperative gate 只允许作为已知 16 Hz bad-data control。

---

## 3. 真实 E1146 目标锁

### 3.1 唯一 primary 记录与 montage

- subject：`epilepsiae_1146` / `pat_114602`
- recording：`rec_114600102`
- EEG onset：`2009-04-24 07:46:49.316406`
- EEG offset：`2009-04-24 07:47:45.947266`
- contacts：`SCL6–SCL9 + ICL1–ICL11`（15 contacts）
- reference：15-contact local CAR；模型仍用 current-based virtual SEEG，二者只比较各自 baseline-normalized
  频段结构、时间尺度和空间次序，不比较绝对电压。

现有 loader/先例：`scripts/run_heo_gate_on_real_seizure.py`。它已锁定真实发作约 3 Hz、1–80 Hz
宽带增强、间歇尖波而非持续 30–150 Hz 平台。LC2 必须扩展 onset/offset/postictal，而不是重写这条结论。

### 3.2 Phase-0 必须先生成的五份 lock

```text
results/topic4_sef_hfo/fcxr_lc2/empirical_lock/
  fcxr_lc2_empirical_target_lock.json
  fcxr_lc2_dynamotype_lock.json
  returning_event_window_index.json
  early_ictal_window_index.json
  postictal_recovery_window_index.json
```

要求：

1. 至少 3 个真实 returning-interictal population-event windows，使用同一冻结 detector，不能从模型结果
   反选；每个窗保存原始 block、绝对/相对时间、contacts、reference、checksum。
2. early-ictal 至少含 onset `[0,3] s` 与 established `[3,18] s` 两层；最终指标只用 Phase-0 锁定窗。
3. offset 至少含 `[-5,0] s`、postictal `[0,15] s` 和 recovery `[15,60] s`；若 raw block 不足，必须
   通过相邻 block 的 SQL/时间连续性证明后拼接，否则标 `POSTICTAL_INPUT_INCOMPLETE`。
4. sharp pulse-comb null 与真实 returning IED null 分开；pulse-comb 只检验谐波伪宽带。
5. 所有阈值用真实 baseline 的 median/MAD/quantile 冻结；不得用模型 candidate 调阈值。

### 3.3 dynamotype 的证据边界

记录 onset amplitude-from-zero、ISI trend、offset slowing/abruptness、DC availability。若采集链或 head
metadata 不能证明 DC 可用，则输出合并类别，不强分 saddle-node 与 subcritical Hopf。LC2 只针对该
E1146 记录的目标表型，不声称统一所有临床 seizure dynamotypes。

---

## 4. SNN-calibrated reduced model

### 4.1 禁止任意 Wilson–Cowan 参数

`Phi_E/Phi_I`、RC1 recurrent input-output curve、E/I delay、噪声尺度、refractory ceiling、Z/M/X
sensitivity 必须来自实际 LIF/RC1 response probe。`src/sef_hfo_lif.py` 可作解析 sanity/null，不能替代
LC2 的 SNN-calibrated surface。

主状态：

\[
q=(r_E,r_I,h,m,x,d),\qquad d=1-z,\quad x=1-\bar x_{relay}.
\]

快率方程：

\[
\tau_E\dot r_E=-r_E+\Phi_E(U_E),
\]

\[
U_E=I_0+F_{RC1}(r_E;r_I)
+J_H S_H(h)-J_{EI}(1-d)r_I-g_Mm-g_Xx,
\]

\[
\tau_I\dot r_I=-r_I+\Phi_I(J_{IE}r_E-J_{II}r_I+I_I).
\]

`F_RC1` 不是自由线性 `J_A r_E`：它必须包含实际 recurrent conductance 的 reversal dependence 与
`g_sat*tanh` saturation，并由 transfer artifact 固定。

### 4.2 H：只负责 carrier geometry

\[
\tau_H\dot h=-h+Q_H(r_E),
\qquad
S_H(h)=\frac{1}{1+\exp[-(h-\theta_H)/k_H]}.
\]

`J_H S_H(h)` 通过 `r_E -> h -> r_E` 形成平滑正反馈。迟滞必须由连续方程自然产生；禁止 Schmitt
hard threshold、状态标签开关或手工 on/off surface。

### 4.3 M：reduced 只检验 carrier survival

\[
\tau_M\dot m=-m+Q_M(r_E).
\]

Reduced M 只问平均 adaptation load 是否把 high branch 消灭。3–8 Hz、宽带和去同步必须在逐细胞
`M_i` 空间 SNN 中验收。Primary 形态规则沿 HEO2.1：`tau_M=250 ms`，强度按**新 high branch
recurrent drive 的 10% force-match**解析重标；不是沿用旧 16 Hz 锚点的绝对 `eta_m=0.354`。

### 4.4 X：load 上升，relay availability 下降

\[
\tau_y\dot y=-y+r_E,
\]

\[
x_\infty(y)= (1-x_{min})\,
\frac{[y-y_{gate}]_+^{n_X}}{K_X^{n_X}+[y-y_{gate}]_+^{n_X}},
\]

\[
\dot x=\frac{x_\infty-x}{\tau_X^{rise/decay}}.
\]

映射到 SNN 时 `x = 1-x_relay`；termination 通过降低 presynaptic recurrent E→E availability，不是
额外抽象关闭 H。Primary 复用 LC1：`tau_y=120 ms, K_X=5, n_X=4, x_min(relay)=0.1,
tau_x_down=1000 ms`。postictal recovery 只允许比较 `tau_x_up={5000,10000} ms` 这一条轴；
`y_gate=baseline y Q99.9` 的规则固定，值由新 baseline 生成。

### 4.5 Z：entry coordinate，不加新机制

\[
\dot z=\frac{z_\infty(r_E,r_I)-z}{\tau_Z^{down/up}},
\qquad d=1-z.
\]

`z_inf` 必须由现有 pre-z GABA sensor 的 SNN-calibrated survival curve给出。H-only continuation 先把
`d` 当冻结坐标；闭环阶段再用同一 Z 方程。禁止 `dot Z += gamma_X X`。唯一 `I_th_EI` 由运行前规则
解析选定：baseline replay 在 8 s 前不能跨 `d_on`，在最长开发窗内需具有非零跨越概率；若规则无解，
判 `Z_ENTRY_CALIBRATION_UNRESOLVED`，不扫更多阈值。

---

## 5. H 在 SNN 中的唯一 v1 实现

### 5.1 与原 E→E 拓扑严格同源

现有 `membrane_terms` 已收到 per-target `I_E_rec` 并形成 `gErec_raw`。对于线性突触滤波，先沿 W_EE
scatter 再低通，与先低通 presynaptic train 再沿相同 W_EE scatter 等价。因此 v1 用同一
`gErec_raw_i(t)` 的慢低通实现 local H：

\[
h_i(t+dt)=h_i(t)e^{-dt/\tau_H}+
(1-e^{-dt/\tau_H})g^{A,raw}_{rec,i}(t),
\]

\[
g^{H}_i=\rho_H S_H(h_i),
\]

\[
g^{eff}_{rec,i}=g_{sat}\tanh\left[
\frac{g^{A,raw}_{rec,i}+g^H_i}{g_{sat}}
\right].
\]

它不增加新边、不读 global rate、不读 core mask、不读取 seizure label。新状态放在非 blessed
`src/snn_engine/mz_slow_vars.py`，所有参数 off-by-default；`rho_H=0` 必须逐比特回到 RC1。

### 5.2 本版锁 `B(V)=1`

当前 slow hook 不接收膜电位 V；为加入显式 magnesium/voltage gate 而修改 blessed `kick_probe.py`
会把 H geometry 与引擎接口同时改变。LC2 v1 因此锁 `B(V)=1`。RC1 的 `(E_E-V)` reversal term 已保留
电压依赖。只有结果 `H_HYSTERESIS_NO_GO` 且诊断明确指向缺失的 voltage cooperativity，下一代才可
单独审阅 `B(V)`；本 plan 禁止。

### 5.3 H 参数的预锁规则

- `tau_H` 候选固定 `{50,100,200} ms`；
- 每个 `tau_H` 的 `theta_H` 来自 H-off interictal `h_i` 分布的 Q99.9；
- `k_H=0.1*theta_H`；
- `rho_H/g_sat` 候选固定 `{0.25,0.50,0.75}`；
- 先在 reduced system 判断，不因空间 SNN 结果扩大盒子；
- 多个 H 点通过时取最小 `rho_H`，再取最短 `tau_H`，最后字典序，不以 waveform 挑点。

---

## 6. 必须证明的动力学几何

令 d 为横轴、x 为 offset load。目标是低/高共存的非零区域，不是一条 gain slope：

```text
low d, low x: stable interictal branch
      repeated IED -> d rises
cross Sigma_on(d,x)
      stable finite H-supported high branch
      sustained load -> x rises
cross Sigma_off(d,x), high branch disappears or loses its basin
      stable low/postictal branch
      x remains high while z recovers
      x decays -> interictal statistical neighborhood
```

### 6.1 continuation 证据

必须同时保存：

- forward/backward equilibrium branches；
- Jacobian eigenvalues；
- fold candidates 经 pseudo-arclength continuation 与最小奇异值复核；
- Hopf candidate 的 complex-pair crossing；
- stable/unstable branch label；
- time integration from both basins；
- solver tolerance、step-size sensitivity、bootstrap transfer-surface uncertainty。

仅靠长 transient、classifier 标签或一次 sweep 跳变不得称 fold/Hopf/bistability。

### 6.2 H-only hard gate

必须同时满足：

1. baseline d 下只有稳定 low branch；
2. 病理 d 范围存在 stable finite high branch；
3. low/high 在非零 d 区间共存，`Sigma_on != Sigma_off` 超过 continuation 和 transfer bootstrap
   uncertainty；
4. high branch 不在 refractory ceiling，且局部 input-output gain 非零；
5. high branch 在 M=X=0 时存在；
6. 16 Hz common Hopf 不能是唯一 high-state solution；
7. RC1 saturation 始终有界。

失败标签：`H_HYSTERESIS_NO_GO`，不得进入空间 SNN。

### 6.3 M survival/morphology gate

- reduced：force-matched M 打开后 high branch 仍存在；
- small SNN：在同一 active window 内，3–8 Hz envelope、1–80 Hz broadband、phase dispersion、
  high-energy duty 同时进入 Phase-0 经验范围；
- mean-field M matched-load 对照保留 carrier，但 broadband/entropy/phase dispersion 应下降；
- 若只有 burst–silence 且 carrier 消失，标签 `CARRIER_POSITIVE_MORPHOLOGY_NEGATIVE`。

### 6.4 X offset/recovery gate

1. IED baseline 中 x 低占空；
2. high branch 中 x 单调积累；
3. x 使 high branch 消失或不可达；
4. 同一 d,x 点 low branch 稳定；
5. X-off matched control 明显延长 high state或达到 simulation cap；
6. `T_X,protect > T_Z,recover-to-safe`；
7. x 衰减后 low/interictal branch 仍稳定。

若 X 能 offset 但不能 recovery，先比较锁定的 5/10 s recovery；仍失败才收口为
`OFFSET_POSITIVE_RECOVERY_NEGATIVE`。本代不加 X→Z。

### 6.5 closed slow path gate

接回真实 `dot z / dot x` 后，不允许 parameter step、kick、hard reset。必须从同一 autonomous ODE
得到 low→high→offset→postictal→low 的闭合慢轨迹，并满足前后 interictal distribution distance
在 bootstrap acceptance band 内。否则 reduced phase 不通过。

---

## 7. 缩小空间 SNN 的相似性合同

候选规模 `{N=4000,8000,16000}`，选择**满足下列条件的最小 N**，不是选择最容易产生 seizure 的 N：

1. E/I 比例不变；
2. `L/sigma_kernel`、core radius/L、两 core separation/L、anisotropy ratio 不变；
3. E→E/E→I/I→E/I→I expected in-degree、权重一阶均值和方差、delay distribution 与 40k RC1 匹配；
4. 若通过增大概率会出现 `p>1`，该 N 判 scaling infeasible；不得截断后继续；
5. H-off baseline 的 event rate/IEI/duration/participation 与 accepted RC1 band 相容；
6. frozen-state transfer surface 与 Phase-1 surface 在预注册误差内；
7. exact snapshot/restart、seed、current-based vSEEG 与 source-space readout全部保留。

第一轮模块顺序只能是 RC1→local H→per-cell M→validated X→dynamic Z。不得加入 K、ELR、A、
新 E→E 边或 global seizure sensor。

---

## 8. 小型空间 SNN 七门

| Gate | 问题 | 最低语义 |
|---|---|---|
| G0 baseline | 新机制是否破坏原间期 | IED rate/IEI/duration/participation/axis 与 RC1 band 相容；H/X 不被代码关闭 |
| G1 onset | 是否无 kick 自发进入 | dynamic Z 进入；matched Z-frozen baseline 不进入 |
| G2 carrier | 是否有限高态 | 1–5 s；非 runaway、非 refractory plateau、非单次事件串 |
| G3 morphology | 是否接近 E1146 | 3–8 Hz envelope、1–80 Hz broadband、非 pulse-comb、低于 HEO1 coherence |
| G4 spatial | 是否局部起始和轴向招募 | first-passage 非 whole-sheet flash；source/core/axis/off-axis 均可判 |
| G5 offset | X 是否因果终止 | X-on offset；matched X-off 明显延长或不终止 |
| G6 recovery | 是否回原间期邻域 | postictal protection 后 ≥8 s returning IED，联合统计距离回归 |

### 8.1 G2 tonic-saturation 排除

必须保存单细胞 ISI CV、Fano factor、refractory-ceiling fraction、pairwise correlation、E/I balance、
distance-to-threshold、fine-bin PSD、virtual-SEEG PSD 和 local gain。贴近 refractory ceiling 的 branch
标 `SATURATED_TONIC_BRANCH`，即使持续有界也不通过。

### 8.2 G3 同窗合取

所有 morphology 指标必须在同一个 200–300 ms active window 内合取；whole-run 分别平均不得拼成
成功。阈值只来自 Phase-0 lock。sharp pulse-comb、returning IED 和 HEO1 16 Hz state 都是 bad-data
regression：合格 classifier 必须分别拒绝。

### 8.3 G4 不再用总 occupied-volume

主读出：first-passage latency、newly recruited area、front velocity、axis/off-axis latency difference、
recruitment gradient。总 occupied-volume 只作描述，不能投票，避免 HYB2 的空间天花板重演。

### 8.4 G6 多变量统计恢复

定义并锁定：

\[
d_{rest}(t)=d[r_E,\;event\ rate,\;IEI,\;duration,\;participation,
E_{vSEEG},\;spatial\ rank,\;z,m,h,x].
\]

pre 与 recovered post 使用同一 estimator/bootstrap。必须返回“能继续产生稀疏不规则 IED”的邻域；
永久静默、固定周期、快速复燃均失败。early retrigger 应受抑，late retrigger 恢复仅作 causality probe，
不替代 spontaneous lifecycle。

---

## 9. 开发/确认与因果边界

- development connection seeds：2 个；
- confirmation connection/noise seeds：至少 3 个，参数锁后才能揭盲；
- confirmation 不参与 H/M/X/Z 选点；
- lifecycle acceptance 无 kick；
- kick 只用于 basin mapping/retrigger，单独标记；
- 第一份 plan 的 matched controls 限于七门所必需的 H-off、X-off、mean-field-M、Z-frozen；完整 Phase-4
  ablation、axis rotation、isotropic control 延后，不在本 plan 自主扩展。

---

## 10. 明确停止规则

1. transfer surface 无法在 held-out 点复现 SNN response → `TRANSFER_CALIBRATION_UNRESOLVED`。
2. H 只有 continuous ramp/runaway/saturation/common Hopf → `H_HYSTERESIS_NO_GO`。
3. H 有迟滞但 high branch 为 refractory plateau → `FAST_TRANSFER_FUNCTION_REPAIR_REQUIRED`；下一代才
   允许 AdEx/EIF 或 voltage-dependent slow conductance。
4. H carrier 成立而 M morphology 失败 → `CARRIER_POSITIVE_MORPHOLOGY_NEGATIVE`；不加 K/A/ELR。
5. X offset 成立、recovery 失败 → `OFFSET_POSITIVE_RECOVERY_NEGATIVE`；本代不加 X→Z。
6. 正确缩放的 small SNN 不可行 → `SMALL_SNN_SCALING_BLOCKED`；不得直接跳 40k。
7. 生命周期成立但 spatial recruitment 不足 → 本 plan 收口；下一代才允许重访 K/ELR，且只能用
   first-passage/front 指标。
8. 任何一项需要 kick、hard reset、参数 step 或 outcome-driven threshold 才通过 → lifecycle FAIL。

---

## 11. 延后阶段（本 plan 明确不授权）

### 11.1 Phase 4 完整因果消融

H-off、X-off、M_i→mean-field、Z-frozen、E→E isotropic/axis-rotation 的完整矩阵只在 nominal small
SNN 七门全过后另写 plan。七门内的最小 matched controls 不等于完成 Phase 4。

### 11.2 Phase 5 eigenmode

只有完整 lifecycle 后才在 interictal/pre-onset/early-ictal/established/pre-offset/postictal/recovered
七状态估计 coarse E/I linear-response operator、spectral abscissa、finite-time gain 和 axial selectivity。
不得在不完整 carrier 上提前包装 eigenmode 结论。

### 11.3 Phase 6 40k E1146

不重新搜索参数；仅允许解析 scaling；至少 3 confirmation seeds 和预注册 controls。第一份 plan 禁止。

---

## 12. 工程合同

- 所有新能力 off-by-default；H-off 对 RC1 逐比特；
- 六个 blessed engine SHA 每 stage 核验；
- 新 H exact exponential update、restart/snapshot、determinism、same-W-path equivalence 必测；
- reduced continuation 必有 synthetic saddle-node/Hopf/bistable/false-long-transient 回归；
- 所有 artifact 写 schema、units、seed、source sha256、code commit、threshold provenance；
- 结果根：`results/topic4_sef_hfo/fcxr_lc2/`；图目录必须有中文 `figures/README.md`；
- 任何 spec/plan 冲突均为 execution blocker，不设“plan 优先”或“spec 优先”。

---

## 13. 文献定位（原则约束，不作参数拼盘）

- Jirsa et al., Epileptor：fast/intermediate/slow time scales 与 seizure onset/offset dynamotype。
  https://academic.oup.com/brain/article/137/8/2210/2847958
- Proix et al.：慢 recruitment wavefront 与更快 ictal dynamics 分离。
  https://www.nature.com/articles/s41467-018-02973-y
- Krishnan & Bazhenov：持续负荷、Na/K pump、termination 与 postictal depression 的机制联系。
  https://pubmed.ncbi.nlm.nih.gov/21677171/
- Wang：慢 NMDA-like excitation 对 persistent/bursting state 的支持及过强时转 tonic 的风险。
  https://pubmed.ncbi.nlm.nih.gov/12748642/
- Saggio et al.：临床 onset/offset dynamotype 的多样性与证据边界。
  https://elifesciences.org/articles/55632
- Liou et al.：tonic wavefront、clonic discharges 与 recruited-territory-dependent termination 分离。
  https://elifesciences.org/articles/50927

这些工作只约束职责分离、时间尺度与可检验几何；LC2 不复制其状态变量或跨论文拼接参数。
