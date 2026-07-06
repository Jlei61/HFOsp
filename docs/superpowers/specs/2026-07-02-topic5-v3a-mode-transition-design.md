# Topic 5 V3a 设计 — 发作是否从"沿间期 HFO 轴"转入"非轴向不稳定模态"（data-side mode transition）

date 2026-07-02 · **rev2（post-review 收紧）** · 状态：design（待 writing-plans）· 前身 = V2a（`docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md`，restricted axial sanity check）· 姊妹 spec = V3b（M3B 模型–数据一致性，另写）

> **rev2 变更（依审阅 6 blocking + 8–16）**：(1) primary/gate 对齐——H3a=Δβ_axis_strength **supportive-only**，承重阳性 = H3b OR H3c；(2) 主转变锚 **P3→I1**，O 只作 buffer/descriptive；(3) 动力学三层——**direct 2D VAR** 为主算子（well-posed）、**low-rank VAR 承 H3c 子空间**、full VAR 仅 sensitivity；(4) 非正规系统最大放大方向用**奇异向量**非特征向量；(5) 离散 VAR reactivity 用 **σ_max(A^k)**，连续 numerical abscissa 仅在 `logm(A)` 稳定时且更名；(6) H3c 用**子空间 P_N/P_A** 非单个 e_nonaxis；(8) avalanche 用 **null-corrected flux surplus** + i≠j + lag0 common-drive control；(9) K primary = **line-length rate**（+β 可靠性阈）；(10) **I1=[+10,+30]s** 锁定（+短发作 fallback）；(11) clinical onset provenance-aware sensitivity（不阻塞）；(12) offset/termination 拆为 **Secondary V3a-T**；(13) 判定改 **tier 0–5**；(14) schema 补 QC/provenance 列 + geometry-insufficient；(15) 实现顺序 = pilot QC 先行。

> **一句话缘起**：V2a 问"发作前临界性是否压在间期 HFO 小路上"，受限实现下没支持。V3a 把问题**从"是否沿轴"改成"是否从轴向搬到非轴向"**——阴性不再是"轴上没东西"，真正的阳性是"非轴向活动/流/可放大模态放大"。这是 07-02 模型–数据 criticality 纲领的**数据腿(D)**；模型侧对接(H3d)在 V3b。

---

## 0. 摘要（朴素话）

病人在两次发作之间，短暂高频异常放电按一条固定先后顺序在电极间传开——像一条走熟的小路（代号 `G_HFO`）。V2a 假设发作前"变脆"迹象也压在这条小路上，受限实现下没看到，还证明了"接近失稳"的读数（谱半径≈0.95）是宽带能量平滑的假象。

V3a 换更贴模型的问题：**发作真正启动时，系统会不会先把"沿小路的有序传播"松掉，转而在小路**之外**的电极上、沿小路之外的方向越来越容易被点亮/放大？** 三件事：(1) 把"变脆地图"跟小路做**有符号**对齐，看它随时间从有序变涣散（H3a，**只作辅助，单独不算支持**）；(2) 用局部线性系统看"最容易被放大的方向"落在小路**子空间**还是小路外**子空间**——因为模型已证轴向机制是"非正规瞬态"而非一个特征值，我们用**有限时窗奇异放大方向**（不是特征向量），只看谱半径会漏（H3c）；(3) 把连锁激活拆成轴内/轴外两区，看**从轴向漏到非轴向的净流量**（扣掉"非轴向触点多"的随机基线后）是否随发作启动上升（H3b）。承重转变只看**发作前 −30~−10s（P3）到 早发作 +10~+30s（I1）**；onset 附近 ±10s（O）只作缓冲、不进主检验。全部跟"打乱后的随机版本"比，被试为单位，broad/narrow 不混，**narrow 主 broad 复制**。

（内部代号：`G_HFO`=间期 HFO typical_rank 顺序场；`β_axis`=有符号轴向对齐强度；非轴向=纯间期 HFO 参与度定义的 off-template clean 触点集 + 子空间 `P_N`；`net_offaxis_flux_surplus`=rate-preserving-null 校正的 A→N 净流；`mode_shift`=`A_lowrank^{k*}` 主奇异向量在非轴向 vs 轴向子空间的投影差；`λ_surplus`=σ_max/|eig| − surrogate 中位数；tier 0–5=证据强度分层。）

---

## 1. 与 V2a 的关系 + honest coupling + 与 07-02 纲领的关系

- **V2a 降级不删**：冻结为 restricted-axial baseline，工程产物 = V3a 的 **axis-only 回归对照臂**。V2a 方法学定律升级为 V3a 全程约束：λmax/VAR/DMD/Jacobian 一律报 `λ_surplus`，**不报 raw**。
- **V3a 数据侧**：真数据检验 H3a/b/c。模型侧 H3d（M3B eigenmode/非正规瞬态一致性）→ **V3b**，exploratory confirmatory（模型均质 L=20 衬底↔真 SEEG 几何桥脆，不拖累数据结论）。
- **honest coupling**：V3a 阳性 = "数据里存在 axial→non-axial 重组"；升级成机制主张需 V3b + Phase 1 表达层。**无 forecasting**。

---

## 2. 主假设 + 数据侧子假设（primary 与 gate 一致）

**H3（主）**：从 late preictal 到 ictal early，系统"最容易被放大的方向 / 连锁流向"从 HFO-axis 转向 non-axis。

| 子假设 | 朴素话 | **primary 指标（P3→I1 的 Δ，每假设唯一）** | 方向 | 角色 |
|---|---|---|---|---|
| **H3a 轴向减弱** | 沿小路的有序变化变弱 | `Δβ_axis_strength = |β_axis|_I1 − |β_axis|_P3`（line-length-rate 基） | **< 0** | **supportive/contextual only** |
| **H3b 非轴向流放大** | 连锁从轴向漏向非轴向增多 | `Δnet_offaxis_flux_surplus`（rate-preserving-null 校正） | **> 0** | **承重 primary** |
| **H3c 模态转移** | 最易放大方向转向非轴向子空间 | `Δmode_shift`，`mode_shift = ‖P_N u₁‖² − ‖P_A u₁‖²`（u₁=`A_lowrank^{k*}` 主右奇异向量，k*=3） | **> 0** | **承重 primary** |
| H3d 模型–数据 | 模型预测的非轴向模态在数据里也更被放大 | → **V3b** | — | — |

**H3a 单独不特异**（轴向减弱可由发作淹没/SNR/饱和/覆盖不足造成）→ **H3a 显著只加强解释，不能单独定义 support**。承重阳性必须来自 H3b 或 H3c。其余（variance/AR1/skewness 时间慢化、λ_surplus、reactivity、direction_index、leak_index、2D mode-shift…）全 secondary/descriptive。**时间慢化只作 secondary**（Wilkat-Lehnertz/Milanowski negative；V3a 主张是空间重组非时间慢化）。

---

## 3. 已锁决定（本 session + 审阅对齐；可回改）

- **[LOCKED D1]** 拆两 spec：本 = V3a（H3a/b/c 数据侧）；V3b = M3B 一致性（H3d）。
- **[LOCKED D2]** 动力学**三层**：primary 算子 = **direct 2D projected VAR**（`z=Qᵀx`，`z_{t+1}=B_direct z_t`，well-posed）承 λ_surplus/gain/reactivity；**H3c 子空间 mode-shift 由 low-rank all-clean VAR `A_lowrank` 承载**（实测 all-clean 60–124 触点，full VAR 病态；DMD/SVD-VAR 降到 top-k 后 well-posed）；full ridge-VAR **仅 sensitivity/supplement**。**narrow 主，broad 复制。**
- **[LOCKED D3]** 非轴向 primary = **纯间期 HFO 参与度**（clean − 模板触点，QC-good，对 ictal 全盲，复用 `load_subject_propagation_events`+`typical_rank`+`axis_partition`）；用**子空间 P_N/P_A**（非单个 e_nonaxis）。M3B 方向 → V3b；Phase-1 residual（LOSO）→ sensitivity。
- **[LOCKED D4]** 线性指标全 grid 算（描述轨迹），**承重统计锚 P3→I1**；中晚 ictal（I2/I3）线性只描述，confirmatory 由 flux 承。
- **[LOCKED D5]** 主转变 = **P3→I1**；**O（±10s）只作 buffer/descriptive/sensitivity，不进 primary**（否则等于在 onset 混合区找最大效应）。
- **[LOCKED D6]** **I1 = [eeg_onset+10, +30]s**，eligibility ≥+35s 可用 ictal；短发作 fallback `I1=[+10, min(+30, +0.25·dur)]` 要求 ≥1 个整 10s 窗；`I1_norm`(ictal 0–25%) 仅 sensitivity；固定窗与百分比窗不同时作 primary。
- **[LOCKED D7]** K primary metric = **bb-envelope line-length rate**（envelope roughness；variance/AR1/skewness/spatial-corr 全 secondary）；`β_axis` 只在 `|β_axis|_P3` 过可靠性阈时可判"减弱"。
- **[LOCKED D8]** avalanche H3b primary = **null-corrected `net_offaxis_flux_surplus`**（非 raw）；ATM 排 self-transition（i≠j）+ lag0 common-drive control（sensitivity）。
- **[LOCKED D9]** 判定改 **tier 0–5**（§7），`state_v3_supported = tier ≥ 3`；V3a 最高 tier 4，tier 5 留 V3b。
- **[LOCKED D10]** offset/termination = **Secondary V3a-T**，**不能挽救 negative onset primary**。
- **[LOCKED]** V3a **自建 null**（shaft-spatial / rate-preserving / label），不依赖 stalled Phase-1。
- **[LOCKED]** 有限时窗 **k\*=3**（0.3s @0.1s hop）为 primary，报 `{1,2,3,5}` profile。
- **[LOCKED impl]** 实现顺序 = pilot QC 先行（§9.1）。

---

## 4. 时间设计（eeg-onset/offset 锚定；P3→I1 primary；regime-metric matching）

**数据现实（已核对 cache）**：每发作有 `eeg_onset_rel`/`eeg_offset_rel`/`eeg_duration_sec`，relt 覆盖 ictal+postictal（如 [-152,169]/[-160,260]）。**坑**：cache `relt=0` ≠ 电生理 onset，`eeg_onset_rel`≠0（139=−3.75s）——V2a 的 relt<0 晚窗混进了 early-ictal。**V3a 每窗按各发作 `eeg_onset_rel`/`eeg_offset_rel` 锚定。**

**Event table**：`subject, seizure, t_eeg_onset, t_eeg_offset, eeg_duration, usable_{pre,ictal,post}_sec, offset_quality, clinical_onset_available, clinical_minus_eeg_onset_sec`。

**滑窗** 10s/5s，覆盖 `[eeg_onset−120, eeg_offset+60]`。

| phase | 相对 eeg_onset/offset | 角色 |
|---|---|---|
| P0/P1/P2 | −120~−90 / −90~−60 / −60~−30 | preictal 轨迹（描述） |
| **P3** | **−30~−10** | **primary 起点** |
| **O** | **−10~+10** | **peri-onset buffer——只作轨迹连续性 + sensitivity，不进 primary Δ** |
| **I1** | **+10~+30**（见 D6） | **primary 终点** |
| I2 | ictal 25–75% | maintenance（线性只描述） |
| I3 | pre-offset（offset−30~offset / 75–100%） | **Secondary V3a-T**（termination；flux 承） |
| Post | offset~+60 | postictal recovery |

**primary Δ = median(I1) − median(P3)。** O sensitivity：若效应只在 O 有、I1 没有 → **不算 primary support**。
**regime-metric matching**：线性算子 P3/I1 承重、I2/I3 描述；flux 对非线性稳健，承 I2/I3 + pre-offset（V3a-T）。
**onset jitter**：`eeg_onset` 及 ±{5,10,15}s 平移重算，**±10s 方向不变**为通过，±15s 压力测试。
**clinical onset（provenance-aware, 不阻塞）**：主分析用 eeg_onset；writing-plan 含 `try_fetch_clinical_onset_from_SQL` + flag；可得则做 sensitivity，不可得则主文档诚实写"clinical onset 缺，onset 不确定用 ±5/10/15s jitter 处理"。

---

## 5. 空间设计（signed axis + 非轴向集合 + 子空间）

**5.1 有符号轴 β_axis**：`rank_forward_i`=G_HFO early→late 缩放 −1~+1；`β_axis(metric,t)=Spearman(metric_i(t), rank_forward_i)`（axis 触点上）。`|β|`=轴向组织强度、`β` 随时间变=轴向方向转移/减弱。

**5.2 非轴向（primary=纯间期 HFO，防循环）**
- **axis 集**：进入间期 HFO 模板者（有限 typical_rank / 高参与度）。
- **non-axis strict（primary 集）**：QC-good clean 触点中不在模板、参与度低于阈者。**对 ictal 全盲。**
- **子空间**：`P_A`=投影到 axis 触点、`P_N`=投影到 non-axis strict 触点（**H3c 用 P_A/P_N，不押单个 e_nonaxis**）。
- **e_axis/e_nonaxis**（仅 2D 可视化 + reduced VAR）：`e_axis`=axis 按 rank_forward 加权；`e_nonaxis`=非轴向参与度地形正交分量（Gram-Schmidt 去 e_axis）。
- **confirmatory/sensitivity**：M3B 方向（V3b）、Phase-1 residual（LOSO）、`axis_partition` 几何 non_axial（within-matched 对照）。

**5.3 geometry-sufficiency 门**：`n_nonaxis_strict ≥ 3` AND `n_axis ≥ 5` AND ≥1 根杆同时有 axis + 邻近 non-axis（供 spatial/label null）。不满足 → 标 **`geometry_insufficient`，不是 negative**。
**5.4 防循环铁律**：非轴向集/方向**不得**用同批发作 criticality 结果。

---

## 6. 三条腿

### 6.1 susceptibility K_t（H3a supportive）
每 contact 每窗算 **line-length rate（bb-envelope roughness，primary）** + variance/AR1/skewness/spatial-corr（secondary）。
- `K_axis_strength(t)=|β_axis|`（line-length-rate 基）；**H3a primary=`Δ|β_axis|` P3→I1 预期 <0**。
- 报告用 `K_nonaxis_contrast`/`K_nonaxis_projection` 作 secondary 收敛证据。
- **可靠性阈**：`|β_axis|_P3` 低于阈 → "无轴向结构可减弱"，H3a 不可判（避免 0.05→0.01 机械叫 weakening）。
- null：shaft-spatial + order（打乱 rank_forward 保参与度）+ axis/non-axis label。

### 6.2 dynamics（direct 2D VAR 主算子 + low-rank 子空间 H3c + 非正规奇异放大）
数据 = all-clean contacts × time（每 10–20s 窗）。
- **primary 算子 = direct 2D VAR**：`z_t=Qᵀx_t`（`Q=[e_axis,e_nonaxis]`），拟合 `z_{t+1}=B_direct z_t`（2D，well-posed，不需先估 60–124 维 A）。→ **`λ_surplus`**（=σ_max(B_direct) 或 |eig| − phase/block surrogate 中位数，**永不报 raw**）、`gain_axis=‖B e_axis‖`/`gain_nonaxis`、离散 reactivity、+ **2D mode-shift 一致性检验**（`|u1_2D_nonaxis|²−|u1_2D_axis|²`）。
- **H3c primary 载体 = low-rank all-clean VAR**：SVD/DMD 降到 top-k → `A_lowrank`；取 `A_lowrank^{k*}`（k*=3）主右奇异向量 u₁，映回触点空间 → **`mode_shift = ‖P_N u₁‖² − ‖P_A u₁‖²`**；**H3c primary = `Δmode_shift` P3→I1 预期 >0**。（用奇异向量非特征向量——非正规系统最大有限时放大方向是奇异向量。）
- **离散 reactivity（改公式）**：`one_step_gain=σ_max(A)`、`finite_time_gain(k)=σ_max(A^k)`（k∈{1,2,3,5}，主 k*=3）。连续解释仅当 `J=logm(A)/Δt` 稳定（无病态复分支）：`reactivity_continuous_approx=λ_max((J+Jᵀ)/2)`（**更名，只在 logm_quality_flag ok 时报**）。
- **eigenvector 投影 = descriptive/secondary only**（符号任意，非放大方向）。
- 低秩 DMD/SVD-VAR 已在癫痫 EEG 模式分解中用作数据驱动动态模式提取 → 作 H3c 载体 + 与 2D 一致性核对合理。full ridge-VAR 仅 supplement。
- null：phase + block surrogate（V2a 已建，扩到 B_direct、A_lowrank、reactivity、mode_shift）。

### 6.3 avalanche compartment flux（H3b；rate-preserving null 是命门）
compartment `A_early/A_late/N`；每窗 ATM `ATM[i,j]=P(j@t+δ|i@t)`，**排 self-transition（i≠j）**。
- `net_offaxis_flux_raw = flux_{A→N} − flux_{N→A}`；`net_offaxis_flux_surplus = raw − median(rate_preserving_null)`；`_z=(obs−median)/MAD`。
- **H3b primary = `Δnet_offaxis_flux_surplus` P3→I1(/I2) 预期 >0**（surplus 而非 raw → 直接排除"非轴向触点多"）。
- secondary：`axis_forward_flow`(A 内 early→late，H3a 描述)、`leak_index`、`branching_N`、`N_self_sustain`。
- **common-drive control（sensitivity）**：`ATM_lag0[i,j]=P(j@t|i@t)`，`ATM_lag1_specific=ATM_lag1−ATM_lag0` → 排全局 burst 假传播。
- null（4）：time-block / **rate-preserving（保每 contact 激活率打乱目标）** / shaft-spatial / axis-nonaxis label。**不需要 axis_forward>0**。

---

## 7. 统计骨架 + tier 判定

- **subject 为单位**（窗→发作→被试→队列中位数，不把窗/发作当独立样本）；**narrow 主 broad 复制，永不 pool**。
- **每假设唯一 primary**（§2），其余 exploratory。null 全 V3a 自建；经验 p=(1+越界)/(1+n_perm)；对齐双侧、方向/趋势单侧。
- **subject-level strong support** =
  `(H3b_primary 显著 OR H3c_primary 显著)` AND 方向正确 AND 对应 null 过 AND onset ±10s jitter 方向稳 AND 非单一 contact 驱动 AND axis-only 对照不能解释。
  **H3a 显著只加强解释，单独不构成 support。** geometry_insufficient 的被试标记，不计 negative。
- **cohort-level**：分 broad/narrow 报 subject-median effect + sign test / Wilcoxon + bootstrap CI + 显著被试数。
- **tier（取代二元 state_leg_supported）**：

| tier | 定义 |
|---|---|
| 0 | 无支持 |
| 1 | 仅描述性方向，null 不显著 |
| 2 | ≥1 被试 subject-level 支持，但无 cohort 方向 |
| 3 | **narrow cohort primary 支持** |
| 4 | narrow 支持 + broad 同向复制 |
| 5 | tier 4 + V3b 模型–数据一致性（**仅 V3b**） |

**`state_v3_supported = tier ≥ 3`；V3a 最高 tier 4。**

---

## 8. 输出 schema（每 subject 一行 + phase-resolved 明细另表）
```
# phase/eligibility
subject, cohort, n_seizures_total, n_seizures_used_{pre,ictal,offset},
n_windows_{P0,P1,P2,P3,O,I1,I2,I3,Post}, duration_median, i1_definition_used,
n_short_seizures_excluded_I1, onset_anchor, onset_jitter_pass,
clinical_onset_available, clinical_minus_eeg_onset_sec
# geometry quality
n_contacts_all_clean, n_axis, n_axis_early, n_axis_late, n_nonaxis_strict,
nonaxis_threshold, axis_nonaxis_ratio, n_shaft_with_axis_and_nonaxis, geometry_insufficient
# K_t (H3a supportive)
K_primary_metric(=line_length_rate), beta_axis_{P3,I1}, beta_axis_P3_reliable,
delta_beta_axis_strength, beta_axis_delta_null_z
# dynamics
dynamics_primary_model(=direct_2D_VAR), dynamics_support_model(=lowrank_DMD), rank_DMD_selected,
lambda_surplus_{P3,I1}, gain_axis_delta, gain_nonaxis_delta, cross_a2n_delta,
one_step_gain_{P3,I1}, finite_gain_k_used(=3), finite_gain_{axis,nonaxis}_{P3,I1},
reactivity_cont_available, logm_quality_flag,
mode_shift_{P3,I1}, delta_mode_shift, mode_shift_2D_consistency, p_{phase,block}
# avalanche (H3b)
atm_lag_sec, activation_threshold, self_transition_excluded, lag0_common_drive_control,
net_offaxis_flux_raw_{P3,I1}, net_offaxis_flux_surplus_{P3,I1}, delta_net_offaxis_flux_surplus,
net_offaxis_flux_z, leak_index_delta, branching_N_delta, p_{rate,spatial,label}
# verdict
axis_weakening_supportive, nonaxis_amplification_supported(H3b), mode_transition_supported(H3c),
tier, state_v3_supported
```

---

## 9. 脚本/测试 + 实现顺序

```
scripts/_topic5_v3_event_windows.py         # eeg-onset/offset 锚定 phase grid + jitter
scripts/_topic5_v3_geometry_axis_nonaxis.py # signed β_axis + 纯间期 HFO 非轴向集 + P_A/P_N + e_axis/e_nonaxis
scripts/_topic5_v3_surrogates.py            # shaft-spatial / rate-preserving / label（自建）
scripts/_topic5_v3_dynamics_utils.py        # direct 2D VAR + low-rank DMD/SVD-VAR + σ_max(A^k) + subspace mode_shift
scripts/run_topic5_v3_{susceptibility_timegrid,dynamics_2d,avalanche_offaxis,summary}.py
scripts/plot_topic5_v3_{timecourses,mode_transition,avalanche_flux}.py
tests/ test_v3_event_windows_onset_offset / _signed_axis_convention / _nonaxis_no_circularity /
       _2d_projection_known_matrix / _singular_finite_time_gain / _discrete_reactivity_gain /
       _subspace_mode_shift / _surrogate_rate_preserving / _avalanche_compartment_flux /
       _geometry_insufficient_flag / _subject_level_aggregation_tier
```

**§9.1 实现顺序（pilot QC 先行，防返工）**：
- **Step 0** geometry/time feasibility pilot：每 subject 出 `n_axis / n_nonaxis / n_windows_P3,I1 / duration 分布 / onset,offset sanity`；确认 narrow 主队列有足够 non-axis 触点 + I1 窗。
- **Step 1** 冻结 + 测试 `_topic5_v3_event_windows.py` + `_topic5_v3_geometry_axis_nonaxis.py`（定义定死后不再改）。
- **Step 2** avalanche H3b（最贴"非轴向增强"、不需解 VAR/DMD，先做）。
- **Step 3** dynamics：direct 2D VAR + σ_max(A^k) 有限时奇异增益（低秩 DMD/SVD-VAR + 子空间 mode_shift 第二轮）。
- **Step 4** K_t（最易被指标选择/baseline/幅度干扰，最后做作收敛证据）。

---

## 10. 判读纪律 / 禁止 claim
不能说"没有临界性 / 发作前没有 state projection"；raw λmax 不当临界（只 λ_surplus）；rank-coupling 不当传播（只 flux）；**eigenvector 不当最大放大方向（用奇异向量）**；**离散 A 直接算 (A+Aᵀ)/2 不叫 numerical abscissa（要先 logm）**；**O 窗不进 primary Δ**；**H3a 轴向减弱单独不算 support**；**raw net_offaxis_flux 不当 primary（用 surplus）**；中晚 ictal 线性 λ 只描述；**termination(V3a-T) 不能挽救 negative onset primary**；非轴向定义不得用发作结果；**geometry_insufficient ≠ negative**；−120~−30 差分不当 onset-proximal（按 eeg_onset 锚）。

## 11. 决定项 status
- **[LOCKED]** D1–D10 + 自建 null + narrow 主 + eeg_onset 锚 + λ_surplus-only + P3→I1 primary + tier + k*=3 + pilot-first。
- **[OPEN, pilot 定]** 非轴向参与度阈值具体值；β_axis 可靠性阈具体值；DMD/SVD-VAR 秩 k；短发作是否多到需 fallback。
- **[V3b]** H3d 模型–数据 eigenmode/非正规瞬态一致性（exploratory confirmatory）。

## 12. 核心命题（一句话）
> **V3a 的 primary endpoint 是 P3→I1，O 窗只作 onset 缓冲/描述。H3a primary = line-length-rate envelope roughness 的 `Δβ_axis_strength` 下降，但 H3a 单独不构成 support；H3b primary = rate-preserving-null 校正的 `Δnet_offaxis_flux_surplus` 上升；H3c primary = 有限时窗奇异放大方向在非轴向子空间 vs 轴向子空间的投影差 `Δmode_shift` 上升。动力学 primary 算子用 direct 2D VAR 直接拟合 `z_{t+1}=B z_t`，H3c 子空间 mode-shift 由 low-rank all-clean VAR 承载，full ridge-VAR 仅 sensitivity；离散 VAR 的主 reactivity 用 `σ_max(A^k)`，连续 numerical abscissa 仅在 `J=logm(A)/Δt` 质量合格时作 secondary。判定用 tier 0–5，`state_v3_supported=tier≥3`，V3a 最高 tier 4，tier 5 留 V3b。narrow 主、broad 复制；模型侧一致性(H3d)在 V3b。**
