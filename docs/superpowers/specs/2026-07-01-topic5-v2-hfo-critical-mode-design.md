# Topic 5 V2 设计 — 间期 HFO 几何作为候选病理临界模态

date 2026-07-01 · rev2（post-review 收紧）· 状态：design（待 writing-plans）· 前身=V1 network-skeleton 线（见 §1）

> **rev2 变更（依 2026-07-01 审阅）**：(1) G_HFO 降为 **candidate** mode，升级需证据阶梯（§1.1）；(2) go/no-go 拆成 **三个 gate**（§3.4）；(3) null 拆成 **三层**（空间平滑 / HFO-rate-preserving timing-order / confound），SEEG 用 contact-shaft-constrained 非 spin-test（§5.2）；(4) null 复刻完整 `|corr|`→A/B max→窗→发作→**被试**聚合，subject 为主单位；(5) 保留 **signed + Spearman** 方向度量；(6) 1/f 分成 **aperiodic 校正** 与 **common-field 残差化** 两类；(7) band 改名 + **80–250 composite 为 primary HFO anchor**；(8) **line-noise 谐波 mask + 512Hz edge flag**；(9) SOZ/HFO-rate/baseline confound 控制入 5.2；(10) phenotype **channel→seizure 两层 + morphology + coarse family**，`hfo_rich` 须独立来源防循环，标注 **blind to alignment**，`preictal_multiband_shift` 移入 state 层；(11) PAC 从"必须"降为 **mechanistic secondary + surrogate**；(12) Phase 2 VAR/DMD/avalanche 加控制、avalanche 不估 exponent。

---

## 0. 摘要（朴素话）

我们看：癫痫病人在**两次发作之间**，那些短暂的高频异常放电（HFO 事件）总是**按一条固定的先后顺序**在电极间传开——像一条走熟了的小路。V2 要问三件递进的事：

1. 这条小路**是不是**一个"最容易塌方的地方"（一个病理临界模态）——**注意是"是不是"，不是默认它就是**；
2. 系统快要发作时，是不是**先在这条小路上"变脆"**（临界指标沿这条路升高）；
3. 真发作时，是**哪一种频率**的活动沿这条路被点亮——不同频率背后是不同的发作生理机制。

一句话：HFO 几何给出"哪条路**可能**容易失稳"，临界指标给出"什么时候接近阈值"，频带扫描给出"失稳通过什么神经生理机制表达"。**但"这条路是临界模态"是要被证/证伪的假设，不是前提。**

---

## 1. V1 与 V2 的关系

**V1（已收口）**：间期网络轴 = 患者内稳定网络骨架 readout（shared coarse anchor），非发作特异 path-replay。V1 把"轴是**稳定 scaffold**"钉住了。

**V2（本 spec）**：问这条 scaffold 是否**同时**是一个病理临界模态。**V2 不推翻 V1，也不默认把 stable scaffold 升级成 unstable mode**——升级要过证据阶梯（§1.1）。

触发 V2 的关键厘清（本 session 建立）：
- 发作"场"喂进统计的量 = 宽带功率 baseline-robust-z = **能量/群体招募代理**（`bb_zt = baseline_robust_z(band_power_trace(1–45Hz))`），非 z-ER 峰值时序。
- 间期轴 = **HFO（高频事件）**峰值传播顺序（`typical_rank`/lagPat）。
- 所以"发作能量贴间期几何"是**跨频带对齐**问题（间期几何来自高频事件，发作相似度却在宽带+HFA 上算）→ 必须扫。
- 现有 field similarity 逐窗逐发作，非平均发作场。

### 1.1 证据阶梯（G_HFO = candidate → pathological mode 的升级条件）

```text
V1 result:
    G_HFO 是稳定的 interictal timing scaffold。

V2 hypothesis:
    G_HFO 可能是一条 candidate pathological susceptibility mode。

升级为 "pathological critical mode" 需要（全部）：
    1. early ictal energy 对齐 G_HFO 超过【空间平滑 null】；
    2. 该效应不被 【interictal HFO rate / SOZ / baseline power / 沿杆距离 / 粗解剖】 解释；
    3. 频带特异效应在 【aperiodic(1/f) + common-field 控制】 后存活；
    4. preictal susceptibility / dynamic mode 也投影到 G_HFO（Phase 2）。
```
在 1–4 未全部满足前，文档与 manuscript 一律写 **candidate mode**，不写 pathological/unstable critical mode。

---

## 2. 三层框架

```text
trait / scaffold  (固定)
    G_HFO[ch] = 间期 HFO timing 几何（typical_rank / lag order）
             = **candidate** 病理临界模态（升级见 §1.1）

state / control   (随时间变)
    C_t[ch]  = 临界性 / 易感性场（variance, AR1, line length, 1/f slope,
               网络 eigenmode, VAR1/DMD λmax, avalanche branching …）

expression / readout  (发作时)
    E_{b,t}[ch] = 频带 b 的发作能量场 = 这次发作以什么生理机制表达
```

派生量：`A_b = align(E_{b,t}, G_HFO)`；`K_t = align(C_t, G_HFO)`；`M_t = align(leading_dynamic_mode_t, G_HFO)`。

三段主假设：H1（发作前 `C_t` 投影到 G_HFO）/ H2（发作早期某频带 `E_{b,t}` 沿 G_HFO 放大）/ H3（**哪个频带**决定机制解释，§8）。

---

## 3. 已锁决定 + 数据现实约束

### 3.1 已锁（LOCKED）
- 参考几何 = **间期 HFO timing 顺序场，固定**。不用间期同频带能量场（未定义）。
- field similarity 沿用 `align_maxab` 机制，扩到多频带；但**新增 signed + Spearman 方向度量**（§5.2）。

### 3.2 数据现实
- **cache**：`results/topic5_ictal_recruitment/ictal_field_long_cache`，含 `bb_zt`/`hfa_zt` 全窗 baseline-robust-z 轨迹 + `relt`。加频带 = `band_power_trace(band=(lo,hi))` 重算入 cache（FFT-bin 口径）。
- **采样率**：多数 1024Hz；`139/253` 是 **512Hz**（Nyquist 256）。**共同天花板 ~250Hz 但 512Hz 被试的 150–250 段临近 Nyquist、脆弱（§5.2 edge flag）**；fast-ripple>250 只 1024 子集、secondary/exploratory。
- **工频**：本队列 UKLFR/Europe → **50 Hz mains**，谐波 50/100/150/200/250 Hz 直接落进 gamma/ripple 段 → 必须 mask（§5.2）。
- **preictal 长度**：cache pre 段 ≤300s，**无 hours-to-days 长 preictal** → 临界性层写成 **short-timescale peri-ictal susceptibility，非 forecasting**。
- **n**：broad=9、narrow=7、union≈13。**分 onset phenotype 后每格 1–3** → phenotype 分层**描述性/per-subject，非 cohort claim**；cohort 模型只用 coarse family（§5.3–5.4）。
- **onset phenotyping**：repo 无临床标注 → 半自动（morphology+spectrogram）+ 人工复核，**blind to alignment**；是待验证子组件。

### 3.3 沿用质疑
- field similarity 在 **2D 投影+平滑网格**上算；对所有频带一视同仁、不污染频带间比较，但 axis_partition/similarity 应补 **3D robustness**（[[project_topic5_ictal_field_dynamics_plan]] 老账）。

### 3.4 三个 gate（取代单一 go/no-go）

```text
Gate A — spatial alignment gate:
    任何 early ictal field 对齐 G_HFO 是否超过【空间平滑 null】？
    过 → 不是纯平滑场自相关，可继续。
    不过 → 退回 common-field / smoothing artifact，V2 表达层不启动。

Gate B — frequency-specific gate:
    是否某频带对齐超过【broadband / common-field baseline】？
    过 → 可谈 frequency-specific ictal expression。
    不过 → 只能说 "G_HFO predicts broadband recruitment"（仍是有价值结果，见下），不谈特定频带机制。

Gate C — HFO-specific gate:
    80–250Hz（ripple_full）对齐峰是否在【aperiodic + common-field 控制】后仍成立？
    过 → 才谈 HFO-specific / ripple-band recruitment。
    不过 → 不等于 V2 全死；只是 HFO-specific 机制不成立。
```

**关键**：Gate B/C 不过**不是失败**。若 null-corrected 对齐存在但 1/f-residual 后消失，这是一个**正结果**——"HFO geometry 预测宽带发作招募 / firing-rate-like recruitment，而非窄带 HFO replay"（broadband LFP shift 是比窄带更好的放电率代理，Manning 2009）。**每层 gate 各有其可发表的阴性/中性结论。**

---

## 4. 相位分解

| Phase | 内容 | 层 | 状态 |
|---|---|---|---|
| **0** | 重定义 + 现有数值 + 锁定 | 基础 | 已建（handoff 2026-07-01） |
| **1** | onset-phenotype 条件化频带扫描 + 三层 null + 1/f/confound 控制 + PAC(secondary) | expression | 主线，next（§5） |
| **2** | 临界性 state 层（susceptibility 场、动力学主模态、avalanche）投影 G_HFO | state | §6，多为 exploratory |
| **3** | 时间结构（沿几何渐次招募）+ 三层联合 | synthesis | §7，exploratory |

---

## 5. Phase 1 详细设计 — onset-phenotype 条件化频带扫描

核心改写：**不是"哪个频带相关最高"，而是"不同发作起始机制，是否在各自特征频带上沿 G_HFO 招募"。频带=发作机制的 readout。**

### 5.1 频带集（带生理标签 + 命名修正 + 谐波/edge 处理）

```python
EPILEPSY_BANDS = [
    ("delta_HYP_slow",      1,   4),   # HYP / periodic spike / clonic slow
    ("theta_preictal_PAC",  4,   8),   # preictal / ictal ripple 相位调制
    ("alpha_sharp_leq13",   8,  13),   # sharp ≤13Hz / spike-wave 边界
    ("beta_LVFA_low",      13,  30),   # LVFA 低端
    ("gamma_LVFA",         30,  80),   # LVFA / recruiting fast（50Hz bin 由 mask 剔除）
    ("hg_low_ripple",      80, 150),   # high-gamma + low-ripple（100Hz bin 剔除）
    ("ripple_high",       150, 250),   # 上 ripple（150/200/250 bin 剔除；512Hz edge-risk）
]
COMPOSITES = [
    ("low_HYP_1_13",        1,  13),
    ("LVFA_13_80",         13,  80),
    ("ripple_full_80_250",  80, 250),  # ★ PRIMARY HFO hypothesis anchor（不是 150–250）
    ("legacy_bb_1_45",      1,  45),
]
FAST_RIPPLE_SECONDARY_1024HZ = [("fast_ripple_low",250,350), ("fast_ripple_high",350,450)]

PRIMARY_HFO_CONTRAST = "ripple_full_80_250"
SECONDARY_SPLIT = ["hg_low_ripple_80_150", "ripple_high_150_250"]

# 工频谐波 mask（本队列 50Hz）——从每个 band 的 FFT-bin 积分里剔除
LINE_NOISE_MASK = {"mains_hz": 50, "harmonics": [50,100,150,200,250],
                   "exclusion_halfwidth_hz": 2.0,
                   "note": "谐波剔除会吃掉 ripple 段大量 bin → 报告每 band 有效带宽占比"}

# 512Hz 被试高频 edge 风险
def effective_hi(fs, hi):
    return min(hi, 0.43*fs)   # 512Hz -> ~220Hz；对 fs==512 且 hi>220 打 flag
# 主结果双报告：full-cohort 保守 ripple 80–200/220 + 1024子集 80–250 + 1024子集 FR 探索
```

### 5.2 每频带 field similarity + 三层 null + 1/f/confound 控制（Gate 命门所在）

**度量（三个都存，不只 |corr|）**：

```python
ALIGN_METRICS = {
  "align_abs_maxab":    "legacy 兼容 primary descriptive（|corr|, A/B mirror max）",
  "align_signed_oriented":"机制方向度量——能量高在 early-HFO 端还是 late-HFO 端",
  "align_spearman_rank": "G_HFO 是 rank/lag order（序数）→ Spearman/robust rank 作 sensitivity",
}
```

**能量特征（三类，分开报告）**：

```python
POWER_FEATURES = {
  "raw_band_z":               "band power baseline-robust-z；问：发作能量在哪招募",
  "aperiodic_residual_band_z":"每 channel-window 拟合 PSD（specparam/FOOOF 类）去 aperiodic offset/slope，"
                              "取 residual/oscillatory 分量；问：是否有超出 1/f 的 oscillatory band 分量",
  "common_field_residual_band_z":"把每 band 场对 broadband/common recruitment 场回归取残差；"
                              "问：该 band 是否有超出全局招募的空间特异性",
}
# 注：common_field_residual 很保守——真 HFO 常与 broadband firing 同时发生，
# residual 后消失只说明"不支持 narrowband-specific HFO claim"，不说明"无生理意义"。
```

**三层 null（缺一不可，否则 timing geometry 会退化成 HFO rate topography）**：

```python
NULLS_REQUIRED = {
  "spatial_smoothness_null":
     "SEEG 用 contact-level constrained permutation（within-subject，优先 within-shaft / 近距离 bin，"
     "同 contact mask，同 |corr|/A-B-max 操作）为 PRIMARY；variogram/distance-preserving surrogate 为 secondary；"
     "2D grid rotation/reflection/toroidal shift 仅作当前平滑场实现的 robustness，NOT 唯一 null。"
     "（不默认 spin-test：SEEG 是稀疏、不规则、深部 3D，不在连续皮层球面上）",
  "HFO_rate_preserving_order_null":
     "保留每触点 interictal HFO rate/count/detection topography，打乱 event peak-time / channel rank order → "
     "重建 G_HFO_null，再算对齐。分开'timing order 有信息' vs '只是贴 HFO-rich 触点'。",
  "confound_null":
     "检验 G_HFO 是否超出 SOZ/resection、interictal HFO rate、baseline band power、broadband 1–250 power、"
     "沿杆位置/距 onset 触点距离、channel SNR、earliest-detection bias。",
}
```

**confound 残差化（map-level，把"timing geometry"和"HFO rate/SOZ topography"分开）**：

```python
G_HFO_resid = residualize(G_HFO, covariates=[
    HFO_rate_map, baseline_power_map, broadband_1_250_map,
    SOZ_or_resection_label, distance_to_onset_contact, shaft_position])
# 主 claim 用 align(E_b, G_HFO_resid)：贴的是 timing order，不是 rate/SOZ topography。
```

**null 必须复刻完整统计流程 + subject 为主单位（防 pseudo-replication）**：

```python
for perm in permutations:
  for subject, seizure, valid_early_window, band:
     A_null[...] = align_maxab(E_band[...], G_HFO_null_or_resid[perm, subject])
  seizure_null   = median_over_windows(...)     # 窗 → 发作
  subject_null   = median_over_seizures(...)     # 发作 → 被试（★主单位）
cohort_null = median_over_subjects(subject_null) # 被试 → cohort
# 不把 window/seizure 当独立样本。
```

**输出列（固定）**：

```python
OUTPUT_COLUMNS = ["subject","seizure","band","epoch",
  "align_raw_abs","align_signed","align_spearman",
  "align_null_median","align_null_mad","align_null_z","align_delta","align_empirical_p",
  "aperiodic_resid_align","common_field_resid_align",
  "n_windows","n_contacts","band_effective_bandwidth_frac","fs_edge_flag"]
```

**主窗**：early ictal（onset→~20s；`ictal_fraction≥0.5`）。late ictal 仅 sensitivity（可能 supercritical 饱和）。

### 5.3 onset phenotyping（channel→seizure 两层 + morphology + coarse family）

**不能只靠 band power**（sharp spike/HYP 都带高频泄漏、LVFA 关键含 low-voltage/electrodecrement）→ 用 morphology + spectrogram + raw bipolar 复核。

```python
# A. channel-level（先 channel，再聚合；SOZ pattern ≠ spread pattern）
CHANNEL_ONSET_PHENOTYPE = {
  "subject","seizure","channel","is_initial_contact","onset_latency_sec",
  "pattern_subtype": one_of(["LVFA","HYP_periodic_spike","sharp_leq_13Hz","spike_wave",
                             "polyspike","delta_brush","burst_suppression","uncertain"]),
  "pattern_family":  one_of(["fast_onset","slow_synchronous","mixed_slow_fast","uncertain"]),
  "confidence": one_of(["high","medium","low"]),
}
PHENOTYPE_FEATURES = ["amplitude_change(low-voltage drop / high-amp)","dominant_frequency/rhythmicity",
  "line_length(sharp/spike burden)","periodicity(spike interval)","morphology(spike-wave/polyspike/delta-brush)",
  "spectrogram(time-freq onset)","raw_bipolar_review"]

# B. seizure-level（只从最早 active contacts 推 dominant，不从整场所有通道）
SEIZURE_ONSET_PHENOTYPE = {"dominant_initial_family","dominant_initial_subtype",
  "mixed_pattern","confidence","n_initial_contacts"}

# C. hfo_rich 必须独立来源（防循环：不能用同一个 ripple-power outcome 定义再去验 ripple 对齐）
HFO_RICH_ONSET = {
  "interictal_hfo_rate_axis": "来自 interictal HFO event rate，独立于 ictal band-power outcome（可用作 confirmatory）",
  "ictal_event_hfo_detector":"来自 event-level HFO 检测器，非 raw ripple power（可用作 confirmatory）",
  "ripple_power_rich":       "descriptive only，NOT confirmatory stratification",
}
```

**标注纪律**：标注者只看 onset morphology/spectrogram/raw trace，**blind to `A_b`/alignment**；low-confidence/mixed → uncertain/mixed；标注与分析分成两个脚本/两阶段，避免"看到结果调标签"。

### 5.4 分析模型层级（防把 phenotype 分层写成 confirmatory cohort）

```text
Primary（cohort）:      alignment ~ band            # 不含 phenotype 交互
Secondary（descriptive）: alignment profiles by ONSET_FAMILY（fast / slow_synchronous / mixed / uncertain，4 类）
Exploratory:            subtype-level per-subject 例子 / phenotype 预测核对（7 subtype）
```
**不**把 `alignment ~ band × onset_subtype` 当 confirmatory cohort model（n 不够）。

### 5.5 PAC（mechanistic secondary，非硬 gate）

HFO 可能被低频相位调制而非与之竞争。但**非正弦 sharp wave 会人为制造伪 PAC**（同一尖波的频谱展开），且 0–20s 对 δ 相位样本偏少（1Hz 仅 ~20 cycle）。故 PAC = **secondary，且必须带 surrogate**：

```python
PAC_FIELDS = [("delta_phase","ripple_amp"),("theta_phase","ripple_amp"),("alpha_phase","ripple_amp")]
PAC_CONTROLS = ["phase-shuffled surrogate","time-shifted amplitude surrogate",
  "event-locked spike-waveform control","compare PAC vs line-length/sharpness",
  "exclude windows dominated by single large transient"]
# 用途：主要给 slow_synchronous / delta_brush 例子；问 PAC 空间场是否贴 G_HFO。
```

---

## 6. Phase 2 概要 — 临界性/state 层（多为 exploratory，加控制）

**定位**：short-timescale peri-ictal susceptibility，**非 forecasting**（≤300s pre）；~10ch SEEG 上噪声大。

```python
CRITICALITY_FEATURES = {
 "passive_univariate": ["variance","lag1_autocorr","line_length","skewness","DFA_Hurst","aperiodic_1f_slope"],
 "network":            ["mean_pairwise_corr","correlation_length","phase_synchrony",
                        "participation_ratio","cov_eigen_slope","leading_eigenvalue_fraction"],
 "linear_dynamics":    ["VAR1_lambda_max","DMD_lambda_max","recovery_time_tau","leading_dynamic_mode"],
 "avalanche":          ["branching_ratio_sigma","ATM(activation-order)"],  # 不估 size/dur exponent
}
```

**VAR/DMD 三坑 → 改 exploratory + 控制**：(1) envelope/band-power 平滑本身生成强 AR1；(2) ~10ch 拟合 VAR 不稳；(3) pre 300s 非平稳，λmax→1 可能是滤波/窗/慢漂移而非临界。控制：

```python
VAR_DMD_CONTROLS = ["ridge-regularized VAR","cross-validated one-step prediction",
  "block-shuffled temporal surrogate","phase-randomized surrogate",
  "matched interictal baseline windows if available","同一 preprocessing/smoothing 跨所有窗"]
```

**Avalanche**：~10 contacts 估 power-law exponent 基本不稳 → **不主打 exponent**；主打 `branching_ratio`（描述性）+ **ATM/activation-order 对齐 G_HFO_lag_order**。

主检验：`M_t = corr(leading_dynamic_mode, G_HFO)`；`early_ictal_alignment[b] ~ HFO_axis_criticality + onset_family + band`。**λmax→1 + τ↑ + v_max→G_HFO** 才谈"接近失稳的空间模态"（借 Pachitariu/Stringer 2026 critically-normalized 思想，仅**类比**，manuscript 不写成癫痫证据）。

---

## 7. Phase 3 概要 — synthesis（exploratory）

- **时间结构**：能量招募是否沿 G_HFO 渐次。**很难**——发作会变成过度同步非线性系统（ictal core 饱和），渐次被同步淹没 → 仅 exploratory，对标 Schevon core/penumbra、Proix Epileptor-field。
- **三层联合** + claim 分级。

---

## 8. 结论语言分级（对齐三 gate）+ 判读矩阵

| 档 | 条件 | 写法 |
|---|---|---|
| **最强** | ripple_full 80–250 峰 + 过空间/order/confound null + aperiodic&common-field 残差存活 + early 最强 +（独立）HFO-rich/LVFA phenotype 更强 (+ Phase2 preictal criticality/λmax→1 沿 G_HFO) | early ictal ripple-band recruitment 优先沿间期 HFO 几何 → HFO 传播标记一条被发作起始招募的病理微环路 |
| **中** | 13–80/80–150 峰 + raw/residual 部分成立 + LVFA family 最强 | HFO 几何标记一条以 LVFA/fast 招募表达的致痫通路，非 ripple-specific |
| **broadband-recruitment（Gate B 不过）** | 过空间 null 但频带不特异 / 1-f residual 后 flat | **G_HFO 预测宽带发作招募 / firing-rate-like recruitment，非频带特异 HFO 机制**（正结果） |
| **低频/PAC** | 1–13 峰或 δ/θ-ripple PAC 场（过 surrogate）贴 G_HFO | HFO generator 嵌在慢 epileptiform scaffold，经 cross-frequency gating，非直接 replay |
| **弱/negative（Gate A 不过）** | raw 高但过不了空间/order null | 表观对齐来自平滑空间场 / HFO-rate topography / common tissue，非 timing-geometry 机制 |

**判读矩阵（band × 机制）**：δ/1–13=slow control gate HFO（中，偏 slow）；13–80 LVFA=fast subsystem 失稳，HFO 轴=易点火微环路（强，非 HFO-specific）；80–150 hg=local firing/core（中，含 1/f 需 Gate C）；150–250/80–250 ripple=病理 micro-avalanche/core cascade（最强，须过 Gate C）；250–450 FR（1024 子集）=病理 microcluster（探索）；低频 phase×ripple PAC=慢波门控 HFO（中，须过 surrogate）；late ictal 全频带同步=supercritical 饱和（描述性）。

---

## 9. 文献锚

- **onset phenotype/频带**：Perucca 2014 *Brain*（morphology 分类：LVFA 43%/periodic spike 21%/≤13Hz sharp 15%）；Gnatkovsky & de Curtis；Jasper NBK609868（LVF vs HYP）；Ferrari-Marinho（onset↔HFO：sharp≤13 最低 HFO、polyspike/delta-brush 最高）。
- **HFO 定义/网络**：Jasper NBK609890（ripple 80–250 / FR 250–500-600；HFO event ≠ band energy）；Jacobs/Zijlmans/Bragin/Worrell；Staba（FR/R ratio↔海马体积↓）。
- **broadband=firing-rate**：Manning 2009 *JNS*；Miller；Ray & Maunsell（high-γ vs broadband；broadband transient ≠ oscillation）。
- **1/f 分离**：Donoghue（specparam/FOOOF，canonical band 会混 periodic/aperiodic）；Gerster（FOOOF vs IRASA）。
- **空间 map null**：spin/permutation null 用于 map-to-map spatial correspondence（Alexander-Bloch 2018）——但为**皮层表面**设计，SEEG 用 contact/shaft-constrained + variogram surrogate。
- **ictal core/行波**：Schevon 2012 *Nat Commun*（core/penumbra）；Smith 2016（wavefront）；Proix 2018（Epileptor field 多时间尺度）。
- **PAC**：Weiss（SOZ spike-phase×ripple-amp↑）；ictal θ/δ–HFA PAC；PAC 伪迹（非正弦波形）。
- **临界性**：Jirsa Epileptor；Lepeu 2024 *Nat Commun*（critical transition；active probing 优于 passive——我们无 active probing）；Maturana 2020（critical slowing biomarker）；Wilkat/Lehnertz 2019 *Chaos*（CSD 反例→写 susceptibility 非 forecasting）；Meisel（SOC failure）；Corsi 2024（avalanche/ATM in TLE）；Pachitariu/Stringer 2026 *Nature*（critically-normalized matrix，仅类比，非癫痫证据）。

---

## 10. 决定项 status

- **[LOCKED]** 参考几何=间期 HFO timing 顺序场；metric 逐窗逐发作 + subject 主单位。
- **[LOCKED, rev2]** 三 gate（A/B/C）；三层 null；signed+Spearman；aperiodic 与 common-field 分开；band rename（80–250 composite 为 primary anchor）；line-noise mask + 512Hz edge flag；confound 残差化；phenotype channel→seizure + coarse family + blind + hfo_rich 独立；PAC secondary+surrogate。
- **[OPEN]** 频带天花板 full-cohort 保守 80–200/220 vs 1024 子集 80–250 + FR 探索的最终报告口径。
- **[OPEN]** onset phenotyping 具体阈值 + 是否拉临床 onset-pattern 标注复核。
- **[GATE]** Gate A（空间+order null）未过 → 表达层退回 smoothing/topology artifact。
- **[EXPLORATORY]** Phase 2（criticality/avalanche，short-timescale，加控制）；Phase 3（时间结构）；能量 vs SOZ（broad+narrow 合并，SOZ 非关键）。

---

## 11. 核心命题（一句话，rev2 降一级）

> **间期 HFO timing 几何定义了一条稳定的病理【候选】模态。若发作前 susceptibility / dynamic mode 逐渐投影到该模态，并且发作早期某个频带的能量场在【空间 null + HFO-rate-preserving timing-order null + aperiodic/1-f + common-field 控制】后仍沿该模态放大，则可支持该 HFO 几何作为发作临界失稳通路的解释。频带身份决定机制：低频=慢变量门控 HFO，beta-gamma=LVFA fast transition 微环路，high-gamma/ripple=发作核心招募。在证据阶梯（§1.1）未满足前，一律称 candidate mode。**
