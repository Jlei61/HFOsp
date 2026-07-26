# Topic 4 MZ-FCXR：逐细胞 activity/Na-like load → pump-equivalent recovery 生命周期设计

日期：2026-07-26

状态：**DESIGN LOCK CANDIDATE — 先完成 P0 诊断，不自动启动 40k 长网格**

前置验收：`docs/archive/topic4/sef_hfo/mz_fcxr_heo_line_acceptance_2026-07-26.md`

## 0. 核心科学目标

最终目标不是制造一个固定节律，也不是把当前 16 Hz 振荡调成另一条永久振荡。目标是同一块 E1146 空间 scaffold 上出现一次自然、有限、有空间结构的状态 excursion：

```text
稀疏、不规则 IED 的稳定统计邻域
→ 有界的 ictal-like oscillatory/bursting excursion
→ postictal suppression
→ 回到原有 IED 统计邻域
```

“回到间期”不要求回到一条周期轨道。验收对象是：

> 返回能够按原有统计规律产生稀疏、不规则 IED 的状态空间邻域 / 稳定概率分布。

最终 paper-ready 图必须能同时承载四类证据：

1. virtual-SEEG：间期 → 发作 → 爆后 → 间期；
2. 慢变量相图：轨迹穿过 onset / offset 边界并形成 postictal memory；
3. 间期与发作早期能量场：用与真实 E1146 相同的读出说明二者共享空间 scaffold；
4. 空间 eigenmode / finite-time response：说明不同阶段的可刺激模态如何改变。

本 sprint 先解决 **lifecycle topology**。真实 E1146 的约 3 Hz 尖锐、`1–80 Hz` 宽带波形是后续独立 gate；pump-only 若闭合生命周期但仍窄带，只能叫 lifecycle scaffold，不得叫最终 seizure model。

## 1. 当前起点与不能再混写的问题

### 1.1 已锁定的 fast substrate

冻结以下已经验收的 FCXR-HEO substrate，不重新调参：

- `L=20`，`N=40000`，E1146 registered scaffold / montage；
- primary seeds `1,3`；
- external/feedforward AMPA = additive current；
- recurrent E→E = reversal-aware conductance；
- recurrent-only smooth saturation：
  `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`，`g_sat=21.6`；
- HEO1 cooperative recurrent transform 保留在已选定的 16 Hz 高分支锚点；该锚点的跨 seed 吸引子身份由 P0 重新确认；
- `dt=0.05 ms`；
- 既有 `Z` 通路保留；
- 所有新增机制 off-by-default，off 时必须 byte-parity。

当前高态的正式标签是：

**bounded coherent common-mode oscillatory branch**。

它约 15.6 Hz、跨电极 coherence 约 0.97、相位跨度约 168°。这说明它是相位锁定的 traveling-wave-like 公共模分支，不是零相位同步，也不是与真实 E1146 一致的发作分支。

### 1.2 三个问题分开

| 问题 | 当前状态 | 本 spec 是否主攻 |
|---|---|---|
| 生命周期：进入、有限持续、终止、爆后、统计恢复 | 未完成 | **是** |
| 波形：约 3 Hz 尖波、1–80 Hz 宽带 | 未完成 | pump 后单独做 |
| 空间模态：公共模 vs transverse/axial/core-differential | 公共模占优 | P0 诊断；必要时后续做 |

不得再用一个参数网格同时声称解决三件事。

## 2. 慢变量职责锁

下一版采用清楚的时间尺度分工：

| 变量 | 职责 | 本 sprint 状态 |
|---|---|---|
| `Z_i` | inhibition robustness / permissivity；推动 onset | 保留 |
| `M_i` | 中间 2–4 Hz burst envelope / waveform shaping | **关闭** |
| `X_i` | within-burst recurrent relay collapse | 固定 `X≡1` |
| `N_i` / pump | termination + postictal memory + late release | **唯一新增机制** |
| broad/area feedback `H` | 选择性压制 common mode | 不实现，仅保留后续 gate |

原因：

- LC1 已证明 X 能终止一次持续高活动，但 Z 不恢复、没有 postictal memory；
- HEO2/3 已证明均匀 `M` 只能在 16 Hz、爆发—静默和压死之间移动；
- 下一次必须让一个**独立的逐细胞负荷状态**在 activity 已下降后仍保持恢复电流。

## 3. 新状态变量与最小方程

### 3.1 逐细胞 absolute load

对每个 E cell 增加 absolute load `n_i ≥ 0`。它可解释为 intracellular Na-like / metabolic burden，但本 sprint 只声称 pump-equivalent mechanism，不声称完成离子浓度模型。

\[
\frac{dn_i}{dt}
=
\alpha_{\mathrm{spk}}\sum_k\delta(t-t_i^k)
+\alpha_{\mathrm{syn}}\,g^{rec,raw}_{E,i}(t)
-\beta_n P(n_i),
\]

\[
P(n_i)=\frac{n_i^h}{K_n^h+n_i^h}.
\]

其中 `n_i^0` 是 same-seed slow-off 间期工作点的 load equilibrium，满足该细胞 baseline 平均 influx 与 pump clearance 的平衡。它由 baseline trace 离线标定，不在线追踪。

进一步约束：

- spike influx 与 recurrent-synaptic influx 的相对比例只在 P0 离线标定一次，不作为二维 sweep；
- `P(n)` 连续、无 ictal label、无 onset detector；
- isolated IED 必须产生可测 `n_i-n_i^0` 瞬变；持续高态依靠质量累积跨过 pump 的高增益区，而不是靠发作传感器开门；
- 不再额外加入 `-(n-n_0)/tau` 线性泄漏；否则新变量容易退化成另一条一阶 adaptation；
- `n_i` 数值安全 cap 只作 fail-fast，任何候选若撞 cap 即判无效，不把 cap 当生理饱和。

### 3.2 pump-equivalent outward current

\[
I^{pump}_i(t)=I_{\max}\,[P(n_i)-P(n_i^0)]_+,
\]

\[
\tau_{m,E}\dot V_i =
F_{\mathrm{FCXR\mbox{-}HEO}}(V_i,\mathrm{inputs},Z_i)
- I^{pump}_i.
\]

`n_i=n_i^0` 时 excess pump current 为 0，因此不改变已验收 baseline。第一版只作用 E cells，避免同时改变 I-cell gain。不得把 pump 写成：

- 对 signed net current 的整体除法；
- 高率阈值触发的 seizure sensor；
- population-mean `n̄(t)`；
- 与 X 同时承担 termination 的混合机制。

### 3.3 为什么它不同于现有 `M`

`M` 基本是 firing/activity 的一阶 adaptation filter；`n/pump` 同时提供：

1. 活动引起的逐细胞负荷累积；
2. 非线性 pump activation；
3. pump 对负荷的清除；
4. activity 已低时仍存在的外向恢复电流；
5. 可验证的 postictal release 时间。

最关键时序是：

\[
T_{Z,\mathrm{recovery}} < T_{N,\mathrm{release}}.
\]

即有效抑制先恢复，而 pump 后解除；否则容易在 pump 一松开时立即 rebound。

### 3.4 与旧 M4-3 `n→a` 设计的关系

仓库已有
`docs/superpowers/specs/2026-07-09-sef-hfo-m4-3-continuous-shunting-axis-coordinate-design.md`
提出抽象 `n→a` recovery。新 spec 不是把旧设计换名字重跑：

- 复用其三条正确纪律：baseline-centered membrane effect、不得整体除 signed net current、early/late retrigger；
- 不复用其 M4 `q_I + S_G` substrate、Gaussian/graph slow field 或全局 shunt；
- 当前主变量是 FCXR-HEO 上的 **per-cell mass-balance load**，pump 同时清除 load 并产生 outward current；
- 第一阶段不做 graph kernel、不做固定 source/sink parameter field。

因此后续代码应优先复用旧 spec 的测试思想，但必须建立新的 FCXR off-parity 与 per-cell causality 合同。

## 4. 目标慢快图景

用群体汇总坐标 `(\bar z,\bar n)` 只作可视化，不替代逐细胞状态：

1. **Interictal**：`z` 高、`n` 低；系统在不规则 IED 统计邻域；
2. **Pre-onset**：IED cluster / slow Z depletion 使 `z` 下降；
3. **Onset**：轨迹穿过 fast high-branch entry boundary；
4. **Ictal-like excursion**：高活动使 `n` 持续积累；
5. **Offset**：高 `n/pump` 使高分支消失或失去可持续性，此时 `z` 仍低；
6. **Postictal**：activity 已低，`n/pump` 仍高；
7. **Recovery**：`z` 先恢复，`n` 后释放；
8. **Return**：晚窗回到 same-seed 原始 IED 分布，而不是回到固定周期。

该图景必须先通过 frozen `(\bar z,\bar n)` attractor map，再运行动态 lifecycle。若 frozen map 中 pump 只能同时压死 low 与 high state，就不允许用长动态网格“碰运气”。

## 5. P0：不加新机制前的正式诊断

P0 是必做，不是装饰。

### P0.1 真实 E1146 与模型读出合同

先把相同处理链用于真实 E1146 与 virtual-SEEG：

- 同一 montage / reference；
- 同一采样率、滤波、窗长、hop；
- 同一 baseline normalization；
- 同一六频段，但真实主 gate 是 `1–80 Hz`；`80–150 Hz` 只作描述，因为真实值约 `−1.2 dB`；
- 同一 sharpness、burst rate、recruitment、pairwise PLV 指标。

真实 E1146 参考向量：

```text
[1–4, 4–8, 8–13, 13–30, 30–80, 80–150] Hz
≈ [+12.0, +10.4, +8.6, +8.3, +5.0, −1.2] dB
```

不再使用“持续 30–150 Hz 全平台”作为真实目标。

模型—数据比较采用时间轨迹状态向量：

```text
[P1-4, P4-8, P8-13, P13-30, P30-80,
 PLV, recruitment, sharpness, burst_rate]
```

先在真实 seizure 上确定 interictal → onset → early ictal → established ictal 的经验状态序列与容差，再问模型是否进入相同状态邻域。旧的“四项必须同一 1 s 窗同时过固定阈值”保留为诊断，不再单独决定真实一致性。

### P0.2 virtual-SEEG forward-model audit

必须写清每个虚拟触点由什么生成。优先从局部 synaptic current / conductance、inhibitory current、adaptation/pump current和几何权重形成 LFP-like readout；不得只用 firing-rate envelope 与临床 SEEG 比频带。

最低交付：

- 当前 `LFPRecorder` 的逐项来源与符号；
- firing-rate proxy vs current-based readout 对照；
- 同一运行下 band ΔdB / PLV / sharpness 的敏感性；
- 若读出定义改变，旧 HEO1–3 结果只作旧读出下的机制证据，不静默继承。

### P0.3 当前高分支的 attractor identity

在 frozen slow state 下，对 low / medium / high IC 与 kick-then-release 重跑：

- 至少 3 个 IC；
- 至少 20 个候选周期或 8 s，取更长者；
- 检查振幅、周期、相位关系是否收敛；
- 做参数上扫 / 下扫的最小 hysteresis test；
- 区分 stable periodic branch、quasiperiodic/chaotic branch、metastable long transient。

只有周期与相位截面收敛后，才称 periodic orbit 并进入 Floquet。否则用 finite-time tangent propagator，不强行套 Floquet。

### P0.4 eigenmode / Floquet / nonnormal 诊断

连续时间 fixed point 看：

\[
\alpha(J)=\max_k \operatorname{Re}\lambda_k(J),
\]

不是 spectral radius。

周期分支看 one-cycle stroboscopic map：

\[
\mu_k=\operatorname{eig}\Phi(T),\qquad
\gamma_k=\log|\mu_k|/T.
\]

使用 matrix-free JVP + Arnoldi，不构造 `40000×40000` dense Jacobian。若 full-state JVP 成本过高，先用固定 32×32 或 40×40 coarse spatial basis，并用源层 trace 验证。

每个阶段至少跟踪：

- common mode；
- source-vs-sink / core-differential mode；
- axial mode；
- transverse / off-axis mode；
- participation ratio / localization；
- left/right mode overlap；
- numerical abscissa 与 finite-time singular gain，避免 nonnormal 系统只看 eigenvector。

P0 的目标不是证明“已经是 seizure”，而是检验当前高态是否确实为 transverse-stable common-mode periodic branch。

### P0 gate

P0 完成后必须能给出：

1. 当前高态的吸引子类型；
2. 主导 common / transverse 稳定性差；
3. virtual-SEEG 是否足以承载真实频带比较；
4. 真实 E1146 的经验状态序列。

P0 不通过时可修仪器，但不得先加 pump 再用新机制掩盖分类错误。

## 6. P1：frozen `Z×N` attractor map

### P1.1 先测 load sensor，不跑 lifecycle

在 slow-off 间期和 frozen 16 Hz 高分支上离线标定：

- median IED 后 `n_i-n_i^0` 瞬变必须高于数值噪声并自行消退；
- 单个 IED 不应把 median E cell 推入 `P(n)≈1`；
- 持续高分支应在 0.5–3 s 内进入明显 pump activation；
- spike 与 synaptic influx 的相对权重固定后不再随结果调整。

若 IED 对 `n` 完全不可见，Hill 非线性实际变成隐藏 seizure gate，判设计失败。若普通 IED 已让 pump 长期饱和，判 baseline incompatibility。

### P1.2 冻结二维图

在小网格上冻结 `Z` 与 uniform clamp `n`，只跑 fast subsystem：

- `Z`：healthy workpoint、onset-near、HEO1 high-branch anchor、strongly impaired；
- `n`：0、预计 onset level、两个 exit-near level、high；
- IC：low / high / kick-release；
- seed1 先跑，只有出现 topology separation 才用 seed3 确认。

`n` clamp 只用于画存在边界，不是最终机制；dynamic run 必须恢复 per-cell `n_i`。

### P1.3 topology GO

必须同时看到：

1. healthy `Z` + low `n` 保持原间期工作点；
2. impaired `Z` + low `n` 存在可持续高分支；
3. 同一 impaired `Z` 下，`n` 增加可使高分支消失 / 失稳 / 落回低态；
4. low/interictal state 在 exit 附近仍存在，不被 pump 同时压死；
5. exit boundary 对小幅参数变化连续，不靠 cap 或 early-stop。

若没有第 3–4 条，停止 pump 动态网格，写 clean NO-GO。

## 7. P2：dynamic Z + per-cell N/pump

### P2.1 先独立锁定 onset 速度

LC1 的 q75 太慢、q50 太快。允许只在二者之间做一个预注册的 3 点 `Z` 速率 bracket，pump off：

- 目标是 no-kick 下保留 ≥8 s 间期统计窗；
- 随后进入可持续至少 1 s 的已知高分支；
- 不要求自然终止；
- 选最慢满足 onset 的点并冻结。

这一步只标定 `Z` 的 onset 角色，不与 pump 参数联合搜索。

### P2.2 pump 参数由 frozen exit surface 反推

不做 `alpha × tau × Imax × h` 全笛卡尔网格：

1. `h` 固定一个生理上平滑的值；
2. `Imax` 从 P1 frozen exit boundary 反推；
3. `alpha_spk/alpha_syn` 的共同尺度从目标高态累积时间反推；
4. `beta_n` 从目标 postictal release 时间反推；
5. 只对每个反推量做中心与 `±20%` 小楔形。

第一轮 `M=off`、`X=1`、无 area feedback。

### P2.3 primary lifecycle

primary run：

- no kick；
- seeds 1、3；
- `T≥40 s`；
- 前 8 s 必须能量化原间期统计；
- 高活动须有界、非 clip、非 refractory ceiling；
- offset 后至少 8 s postictal + 8 s late-recovery 观测；
- 只允许自己的 watchdog 提前终止 nonfinite / OOM / hard numerical safety 事件。

kick 仅用于 basin / retrigger probe，不得替代 spontaneous onset。

## 8. 生命周期验收合同

### 8.1 分层判决

**Tier A — lifecycle topology pass**

- no-kick 自然 onset；
- pre-interictal ≥8 s；
- 有界高活动持续 ≥1 s；
- activity 下降发生在 `n/pump` 已明显积累之后；
- `Z` 先恢复，pump 后释放；
- 有 postictal suppression；
- late window 回到原间期 IED 统计邻域；
- seed1/3 同方向复现。

**Tier B — empirical ictal-manifold pass**

- 模型轨迹进入真实 E1146 的 9 维状态容差；
- 约 3–8 Hz 尖锐 burst；
- `1–80 Hz` 多频带抬升，`80–150 Hz` 不作必须上升；
- spatial recruitment / PLV 的阶段序列与真实数据兼容。

Pump sprint 的最低阳性终点是 Tier A。只有 Tier A + Tier B 才能写“data-consistent seizure lifecycle”。

### 8.2 统计恢复，不是固定节律恢复

同 seed slow-off baseline 预先保存：

- IED event rate；
- inter-event interval median / CV；
- event duration；
- participation；
- peak rate；
- source / sink / axis / off-axis activity；
- propagation-template rank similarity 与方向比例；
- virtual-SEEG band-power 分布。

late recovery 以这些量回到 baseline bootstrap / replicate band 为准。end-rate 变低、出现一个晚期事件、或肉眼“看起来安静”都不算恢复。

### 8.3 因果时序

必须报告：

```text
t_onset
t_n_rise
t_pump_effect
t_offset
t_z_recovered
t_n_release
t_first_postictal_IED
t_statistical_return
```

硬要求：

```text
t_onset < t_n_rise < t_offset
t_offset < t_z_recovered < t_n_release
t_n_release ≤ t_statistical_return
```

时序失败时不得用最终 rate 下降代替机制解释。

### 8.4 retrigger

- early postictal probe：同强度扰动应衰减 / 不传播；
- late recovery probe：同扰动应恢复到 baseline response band；
- early 与 late 都只作匹配扰动对照，不进入自然 onset 主分母。

## 9. P3 / P4 只在前一门通过后解锁

### P3：把 `M_i` 重定位为 2–4 Hz burst envelope

只有 Tier A 通过后才重新开启 per-cell `M_i`：

- 目标是把 16 Hz carrier 分组成约 3 Hz 尖锐 burst；
- 不改变已锁定的 Z onset 与 N/pump termination；
- 必须有 mean-matched static control；
- 必须保持 per-cell heterogeneity；
- 若 `M` 只能造成长静默 / fragment，不继续扫 `tau_adp`。

### P4：选择性 broad/area feedback

只有 lifecycle 已闭合而 common Floquet mode 仍压倒其他空间模态时才考虑：

\[
H(t)\propto \int A(x,t)\,dx
\]

或其非局部 inhibitory 等价项。目标不是全局压低 rate，而是让 common multiplier 更向内、缩小 common–transverse stability gap，同时保留 axial / core-differential response。

必须有：

- area-matched static control；
- rank-one/common-mode projection 分析；
- 对 transverse/axial modes 的选择性；
- 不允许复用 population-mean `M` 冒充 area feedback。

## 10. 空间机制边界

- E1146 轴定义 orientation，不自动定义 source→sink 方向；
- 固定 source-fast / sink-slow 条带不是最终机制；
- H3.1b 只证明空间摆放改变权衡，未证明唯一方向；
- 下一版空间差异应由局部 activity history 生成的 `n_i(t)` 自然形成；
- 对正向/反向间期模板必须 report-both，不能把单一固定方向写进参数场。

空间刺激分析除右 eigenvector 外还要报告：

- left eigenvector / controllability；
- finite-time singular vector；
- common / axial / transverse response gain；
- source / sink / off-axis matched perturbation；
- phase-specific stimulation on the periodic-like branch。

## 11. 图与 artifact 合同

新结果根：

```text
results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle/
```

必须有：

- `STATUS.md`
- `run_manifest.json`
- `baseline_contract_seed{1,3}.json`
- `real_target_contract.json`
- `virtual_seeg_audit.json`
- `attractor_identity.json`
- `frozen_zn_map.json`
- `lifecycle_verdict_seed{1,3}.json`
- `resource_log.jsonl`
- `figures/README.md`

诊断图：

1. `current_branch_dynamics.png`：多 IC、hysteresis、mode/Floquet 或 finite-time 诊断；
2. `real_vs_virtual_readout.png`：同 pipeline 的真实 / 模型 9 维轨迹；
3. `frozen_zn_attractor_map.png`：low/high/orbit/return 分类与 entry/exit boundary；
4. `pump_lifecycle_diagnostic.png`：rate、Z、N、pump、分区 activity、关键时间点；
5. `statistical_return.png`：pre / late IED 分布，而非单条 trace。

只有 Tier A 通过才生成 lifecycle candidate figure；只有 Tier A+B 通过才生成目标 paper-ready 四栏图：

```text
virtual-SEEG lifecycle
| (Z,N) phase portrait
| interictal vs early-ictal energy field
| stage-specific spatial modes / stimulation
```

负结果不得伪装成 paper-ready lifecycle 图。

## 12. 工程、OOM 与 detached-run 合同

### 12.1 worktree 隔离

- 新 sprint 使用独立 `codex/topic4-mz-fcxr-pump-lifecycle` worktree；
- 不修改 sibling worktree；
- 不 merge / rebase / push 他人分支；
- 先记录 base commit、dirty state、blessed hashes；
- guarded engine 若非改不可，先停并报告；不得自主 re-bless。

### 12.2 worker 上限

基于既有 40k run 峰值 RSS 约 20 GB：

- `T < 20 s`：最多 2 个 40k workers；
- `T ≥ 20 s`：最多 1 个 worker；
- `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；
- 不缓存多份完整 per-cell time matrix；在线聚合，landmark 才存稀疏 trace/snapshot。

### 12.3 内存 stop rule

launcher 前记录 `MemAvailable` 与 swap baseline：

- swap delta `>256 MiB`：停止提交新任务；
- swap delta `>512 MiB` 且连续两个采样仍增加，或 `MemAvailable < 2×` 单 run 实测峰值：只终止自己的最新任务；
- 不杀 sibling / 用户进程；
- 资源异常写 `RESOURCE_PAUSED.json` / `ABORTED.json`，不静默重启。

### 12.4 nohup / sentinel

长 run 使用：

```bash
setsid nohup <runner> > <run_dir>/nohup.log 2>&1 < /dev/null &
```

立即写：

- `launcher.pid`
- `RUNNING.json`
- `resource_log.jsonl`

完成写 `DONE.json`，失败写 `FAILED.json`；恢复会话后先用 PID + command line + sentinel 三重核对，不能只看 log 尾部。

## 13. 停机条件

出现任一情况，停止扩网格并写 bounded-negative：

1. pump-off baseline parity 失败；
2. isolated IED 完全不能驱动 `n`，说明机制退化成隐藏 gate；
3. 普通 IED 已让 pump 长期饱和，工作点被 tonic suppression 替代；
4. frozen map 中 high 与 low state 同时被压死，没有 exit-only corridor；
5. dynamic pump 只能产生 fragment / repeated reset，没有 postictal memory；
6. offset 由 X、M、hard cap、early-stop 或数值天花板造成；
7. `n` 撞 safety cap；
8. Z 未先恢复，pump 一松即 rebound；
9. late window 未回到 baseline IED 分布；
10. 需要同时调 drive、connectivity、cooperative gain、Z、M、X 才能得到候选。

禁止在 pump no-go 后自动加 M 或 broad feedback；每个新机制必须是独立 sprint。

## 14. 本阶段安全结论模板

### P0 only

“当前 FCXR-HEO 高态被形式化为 `<attractor type>`；其主导 mode / virtual-SEEG 边界为 `<result>`。尚未加入 lifecycle 机制。”

### Pump bounded-negative

“逐细胞 activity/Na-like load → pump-equivalent recovery 在已锁 substrate 上 `<能够/不能>` 提供选择性 exit 与 postictal memory；未得到 confirmed lifecycle。”

### Tier A pass

“模型在两个 primary seeds 上完成 no-kick 的 interictal-statistical-neighborhood → bounded high excursion → postictal suppression → statistical return。该结果验收为 lifecycle scaffold；真实 E1146 波形一致性仍待 Tier B。”

### Tier A+B pass

只有同时满足真实状态轨迹与空间读出合同，才允许写：

“在同一 E1146 scaffold 上得到与真实发作状态序列兼容的、可恢复时空 seizure-like lifecycle candidate。”

在任何阶段都不得仅凭工程 green、图已生成或 rate 降低声称科学 PASS。
