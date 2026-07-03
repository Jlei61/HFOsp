# Topic 5 V3p 设计 — 发作前(preictal-only)非轴向轨迹:小路之外的活动/流/可放大模态是否随发作临近逐渐爬升

date 2026-07-03 · **rev0（brainstorm 收口，待 writing-plans）** · 前身 = V3a（`docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md`，P3→I1 onset 对比 → tier 2 偏阴性）· 姊妹 = V2a（restricted axial sanity，`docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md`）

> **一句话缘起**：V3a 用「发作前 −30~−10s(P3) 对比 发作后 +10~+30s(I1)」两时刻差问"发作是否从轴向搬到非轴向"，只有个别被试(1125)像、凑不成队列(tier 2)。问题在于 I1 落在发作**已经点着之后**——信噪比塌、饱和、全场淹没，这个终点本身脏，两时刻差不干净。V3p **把终点从"发作后"挪走，只看发作真正开始之前的两分钟(P0→P3, −120~−10s)干净时段**，问一个 V3a 没问的形状：非轴向的活动/流/可放大方向是否随发作临近**逐渐爬升**（斜率>0），而且这爬升**专门集中在小路之外的触点**、不是"发作前大家一起升温"的假象。

> **锁定决定（本 session，用户 2026-07-03 三选一确认）**：(1) **worktree** = off V3a HEAD `ac042f3`（分支 `topic5-v3p-preictal-trajectory`），V3p **只加新文件**（`src/topic5_v3p_*` + scripts + tests），**read-only import** V3a 的 `topic5_v3_mode_transition` 模块；(2) **co-primary** = `mode_shift_density_surplus_slope`(H3p-c) + `net_offaxis_flux_surplus_slope`(H3p-b)（`surplus_slope` = obs_slope − label-null 斜率中位数），Holm 校正；activation-burden / N→N 自持 / gain_nonaxis / 轴向减弱 全 secondary/supportive；(3) **残差化落地** = **同杆 label-null 主裁**（天然控全局升温）+ 对 `global_energy(t)`+`axial_energy(t)` 的**回归残差斜率**作 sensitivity。

---

## 0. 摘要（朴素话）

病人两次发作之间，短暂高频异常放电按一条固定先后顺序在电极间传开——一条走熟的小路（间期传播轴 `G_HFO`）。有人猜发作启动时系统会把"沿小路的有序传播"松掉、转到小路**之外**去。这个猜想在本仓库已经问过三次都弱/不稳：发作内场动力学（broad 有暗示、narrow 扩队列证否）、V2a 受限临界性（偏阴性）、V3a onset 两时刻差（tier 2）。

V3p 换**最干净的一个窗口**再问一次：**只看发作真正开始之前的两分钟**（`P0..P3` = 发作前 −120~−10s，按每次发作的电生理 onset 锚定，完全不碰发作本身）。在这段还没发作、信号还干净的时间里，看小路**之外**的触点（non-axis strict）是不是**一路缓慢爬升**（不是突然跳一下）出现三件事：(1) **激活负担**变重（小路外触点越来越频繁放电）；(2) **自我维持**（小路外点着后更容易接着点小路外的下一个，形成小路外的连锁）；(3) **最容易被放大的方向**越来越偏离小路、指向小路之外。

**命门：必须扣掉"发作前大家一起升温"。** 发作前普遍有个全局兴奋度上升（已知现象，所有电极都涨）。若小路外涨了、但小路上和全局同样涨，那只是"整体升温"不是"搬到小路外"。真正的裁判是**同杆内 label null**：把"哪些触点算小路外"在同一根电极杆内随机打乱、重算整条斜率——只是整体升温的话，打乱后斜率一样大→实测不显著；只有上升**专门集中在真正的小路外触点**上，实测斜率才超过打乱分布。另外每个指标再对"全局能量随时间的趋势"+"轴向能量随时间的趋势"做回归、看残差还剩多少斜率，作稳健性核对。

**诚实预期**：这是"轴向→非轴向"假设的**最后一枪**（preictal-only 无信噪比塌 + 残差化 + null 裁判）。**若阴性**——诚实结论是"数据不支持非轴向搬迁"，反而**加固** Topic 5 主线（间期轴=共享粗骨架，非发作特异 replay）；**若出现干净的 preictal 非轴向爬坡**——是目前最强阳性证据。两种结果都必须写成**探索性**，预注册的阴性是可接受结局。

（内部代号：`G_HFO`=间期 HFO typical_rank 顺序场；non-axis strict=纯间期 HFO 参与度低的 clean 触点集 + 子空间 `P_N`；`slope`=preictal 窗上 Theil-Sen 稳健斜率；`label-null-of-slope`=同杆打乱轴/非轴标签重算斜率的置换零假设；`surplus_slope`=obs_slope − label-null 斜率中位数；tier 0–5=证据强度分层。）

---

## 1. 与 V3a 的关系 + honest framing

- **V3a 不删、不改**：V3p **read-only 继承** V3a 已建全部纯数学机制（相位窗 `phase_bin_range`、几何 `classify_contacts`/`P_A`/`P_N`/`axis_nonaxis_vectors`、三种 null `shaft_constrained_permute`/`rate_preserving_shuffle`/`label_permute`、avalanche `atm_offdiag`/`compartment_flux`/`net_offaxis_flux`、dynamics `lowrank_var`/`dominant_right_singular_vector`/`map_lowrank_vector_to_contacts`/`subspace_mode_shift`/`finite_time_gain`/`project_2d`/`direct_2d_var`/`demean_window`），io `_topic5_v3_io.py::load_subject_phase_envelopes`/`classify_subject_contacts`。V3p 新增文件只加"preictal 限定 + 斜率 + 残差化 + N→N 自持"这层。
- **同一科学问题的第三个角度**：§3.6（发作内场动力学）在**场层**问、V3a 在 **onset 两时刻差层**问、V3p 在 **preictal 轨迹层**问。三者共用非轴向定义（防循环），V3p 是设计上最能规避 SNR/饱和混淆的一枪。
- **honest coupling**：V3p 阳性 = "数据里，发作前非轴向活动/流/可放大模态存在**专门集中在非轴向触点**的上升趋势"。**这不是发作预测/forecasting**——是对一段 preictal 时间窗内趋势的**描述性 + null 裁判**，不是前向分类器，不出任何 lead-time / AUC / 预测主张（见 §11 禁止 claim）。升级成机制主张需模型侧对接（V3b 的 preictal 类比，本 spec 外）+ 表达层。

---

## 2. 主假设 + 子假设（primary 与 gate 一致）

**H3p（主）**：从 far-preictal(`P0`, −120s) 到 late-preictal(`P3`, −10s)，非轴向的连锁流向 / 最易放大方向随发作临近**逐渐上升**，且**专门集中在非轴向触点**（label-null 裁判）。

| 子假设 | 朴素话 | **primary 指标（preictal 窗上唯一斜率）** | 方向 | 角色 | V3a 对应 |
|---|---|---|---|---|---|
| **H3p-a 轴向减弱** | 沿小路的有序组织随发作临近变弱 | `beta_axis_strength_slope`（`|β_axis|` line-length-rate 基，Theil-Sen over `P0..P3`） | **< 0** | **supportive/contextual only** | H3a |
| **H3p-b 非轴向流放大** | 连锁从轴向漏向非轴向随时间增多 | `net_offaxis_flux_surplus_slope`（= obs_slope − label-null 斜率中位数） | **> 0** | **承重 co-primary** | H3b |
| **H3p-c 模态转移** | 最易放大方向随时间转向非轴向子空间 | `mode_shift_density_surplus_slope`（`mode_shift_density = ‖P_N u₁‖²/rank(P_N) − ‖P_A u₁‖²/rank(P_A)`，u₁=`A_lowrank^{k*=3}` 主**奇异向量**映回触点空间；斜率减 label-null 中位数） | **> 0** | **承重 co-primary** | H3c |
| H3p-d 非轴向负担/自持/增益 | 小路外激活变重、小路外自持连锁增多、小路外可放大增益上升 | `nonaxis_activation_burden_slope`（残差化 vs global）、`N_self_sustain_slope`、`gain_nonaxis_surplus_slope` | **> 0** | secondary/convergent | 无（V3p 新增） |

**support = H3p-b OR H3p-c**（cohort 级 **Holm** 校正 2 个 co-primary），方向正确、**对应 label-null 过**、onset ±10s jitter 方向稳、非单一 contact 驱动、axis-only relabel 不能解释。**H3p-a 显著只加强解释，单独不算 support**（轴向减弱可由 coverage/SNR/饱和造成）。H3p-d 全 secondary/convergent，Step-last，Step2/3 阴性则不过度解释。

---

## 3. 已锁决定（本 session；可回改）

- **[LOCKED L1]** **preictal-only**：primary 时间腿只用 `P0/P1/P2/P3`（−120~−10s），**完全不碰** `O/I1/I2/I3/Post`。V3p 不做 onset-straddling 对比。
- **[LOCKED L2]** **趋势 = 斜率**（不是两窗 Δ）：每发作 preictal 窗上 **Theil-Sen 稳健斜率**（primary 效应量），报 Spearman ρ(metric, t) 作 scale-free 单调性伴随、OLS 斜率作 alt。被试值 = 各发作斜率的中位数。
- **[LOCKED L3]** **co-primary = H3p-b(flux) + H3p-c(mode-shift)**，Holm 校正；H3p-a supportive-only（`module_support_flag` 恒 False）；H3p-d secondary。**与 V3a co-primary 结构对齐**。
- **[LOCKED L4]** **残差化 = label-null 主裁 + 回归残差 sensitivity**。primary p = **同杆 label-null-of-slope**（重算整条斜率）；每指标另报对 `global_energy(t)`+`axial_energy(t)` 回归后的 `*_slope_resid` 作稳健列（**conservative floor**，见 §7 collinearity 注）。
- **[LOCKED L5]** **eeg-onset 锚定**（各发作 `eeg_onset_rel`，非 cache relt=0）；继承 V3a `phase_bin_range` 的 `P0..P3`。onset ±{5,10,15}s jitter；preictal 远离 onset，±10s 方向稳即通过。
- **[LOCKED L6]** **几何/非轴向定义 = 继承 V3a**：three-class（axis/non-axis-strict/ambiguous），`P_A/P_N` 用 axis+non-axis-strict，非轴向=纯间期 HFO 参与度、**对 ictal 与 preictal 结果全盲**（防循环）。pilot-locked 阈值（`nonaxis_hfo_participation_max`、`beta_axis_reliability_min=0.20`、`lowrank=6`、`finite_horizon_k=3`、`single_contact_energy_frac_max=0.50`）**直接沿用 V3a `config/topic5_v3.yaml` 的锁值**（同几何同 dynamics，无需重 pilot）。
- **[LOCKED L7]** **λ 只报 surplus**（继承 V2a→V3a 方法学定律）；mode-shift 用**奇异向量**非特征向量；VAR/DMD **within-window demean 不 standardize**；ATM `i≠j`。
- **[LOCKED L8]** **subject 为单位**（窗→发作 Theil-Sen→被试中位数→队列）；**narrow 主 broad 复制，永不 pool**。
- **[LOCKED L9]** 判定 **tier 0–5**（§8），`state_v3p_supported = tier ≥ 3`，V3p 最高 tier 4（tier 5 = 模型侧一致性，本 spec 外）。
- **[LOCKED L10]** **EXPLORATORY**，**无 forecasting/prediction 主张**；预注册阴性可接受、不救 1125。
- **[LOCKED impl]** 实现顺序 = feasibility(preictal 窗计数)先行 → 纯数学 → trajectory runner → summary → figure。
- **[OPEN, pilot 定]** `min_windows_for_slope`（默认 8）；slope 估计的 preictal span 是否需 fallback（短 pre-onset 记录）；proximal-only `[−60,−10]` sensitivity 是否入表。

---

## 4. 时间设计（preictal-only；eeg-onset 锚定；斜率）

**只用 preictal 相位**（继承 V3a `phase_bin_range`，anchor = `eeg_onset_rel + onset_shift`）：

| phase | 相对 eeg_onset | 角色 |
|---|---|---|
| P0 | −120~−90 | preictal 轨迹起点 |
| P1 | −90~−60 | 轨迹 |
| P2 | −60~−30 | 轨迹 |
| P3 | −30~−10 | **primary 轨迹终点**（最贴 onset 的干净窗） |
| ~~O/I1/I2/I3/Post~~ | — | **V3p 不用**（离开 onset-straddling 与 ictal 淹没区） |

- **滑窗** 10s/5s 覆盖 `[eeg_onset−120, eeg_onset−10]` → 满记录时每发作 ~21 窗，每窗中心时刻 `t_w`。
- **趋势统计**（每发作）：`slope = TheilSen(metric_w, t_w)`（primary 效应量，稳健于离群窗），`rho = Spearman(metric_w, t_w)`（scale-free 单调性伴随），`slope_ols`（alt）。**被试值 = median over seizures**。
- **min_windows_for_slope**（默认 8，pilot 定）：某发作 preictal 窗数 < 阈 → 该发作不进斜率（类比 V3a `i1_eligible`）；被试 0 usable 发作 → feasibility-insufficient（**≠ negative**）。
- **primary span = 全 `[−120,−10]`**（对齐 P0→P3）；**proximal-only `[−60,−10]` slope 作 sensitivity**（趋势是否集中在临近 onset 段）。
- **onset jitter**：anchor ±{5,10,15}s 重算；preictal 远离 onset，**±10s 方向不变**为通过，±15s 压力测试。

---

## 5. 空间设计（继承 V3a，防循环铁律不变）

- **axis / non-axis-strict / ambiguous 三分类**：`_topic5_v3_io.py::classify_subject_contacts`（single source of truth）。axis=有限 typical_rank 或 `axis_partition` source/mid/end；non-axis-strict=QC-good clean ∧ ¬axis ∧ 间期 HFO 参与度 < 阈；ambiguous=clean ∧ ¬axis ∧ 参与度 ≥ 阈（进 all-clean VAR X，但**不进** `P_A/P_N`）。
- **子空间** `P_A/P_N`（`subspace_projectors`）、`e_axis_mean/e_axis_grad/e_nonaxis_mean`（`axis_nonaxis_vectors`，uniform 非参与度加权）。
- **防循环铁律**：非轴向集**不得**用同批发作的 ictal / preictal criticality 结果——纯间期 HFO 参与度定义，对下游全盲。
- **geometry-sufficiency 门**（`geometry_sufficient`）：`n_axis ≥ 5` AND `n_nonaxis ≥ 3` AND ≥1 杆同时有 axis+non-axis（供 shaft-label null）。不满足 → 标 `geometry_insufficient`，**不是 negative**。

---

## 6. 四条腿（preictal 窗上逐窗指标 → 斜率）

所有腿共用**一次 preictal 窗循环**（一个 trajectory runner，避免重复 load；§10）。每窗产出一组标量，跨窗拟合斜率。

### 6.1 H3p-b 非轴向流放大（co-primary）
- 每窗：阈化激活（`z > z_threshold`）→ `atm = atm_offdiag(active_bool)`（`i≠j`）→ `net_offaxis_flux(atm, axis_idx, nonaxis_idx, "source_mean")`。
- 每发作：`obs_slope = TheilSen(net_offaxis_flux_w, t_w)`；被试 = median over seizures。
- **primary 裁判 = label-null-of-slope**：1000 次同杆 `label_permute` → 每次用置换后的 axis/nonaxis 索引重算整条 flux 轨迹 + 斜率 → null 斜率分布。`net_offaxis_flux_surplus_slope = obs_slope − median(null)`；`p_label_slope = (1+#{null≥obs})/(1+n_perm)` 单侧（期望>0）。**label-null 天然控全局升温**（全场同涨 → 换标签同斜率 → 不显著）。
- secondary null：`rate_preserving_shuffle`-of-slope（控每-contact-rate 驱动的假趋势）、`shaft_constrained_permute`-of-slope。
- secondary metric：`leak_index`、`branching_N`、common-drive 敏感度（`atm_lag0` → lag1-specific）。

### 6.2 H3p-c 模态转移（co-primary）
- 每窗：all-clean X（含 ambiguous）`demean_window` → `A_lowrank, U_r = lowrank_var(X, rank, alpha)` → `u_r = dominant_right_singular_vector(A_lowrank, k*=3)` → `u_c = map_lowrank_vector_to_contacts(u_r, U_r)` → `mode_shift_density = subspace_mode_shift(u_c, P_N, P_A, "density")`。**逐窗计算 = V3a H3c 逐窗算法逐字复用。**
- 每发作：`obs_slope = TheilSen(mode_shift_density_w, t_w)`；被试 = median。
- **primary 裁判 = label-null-of-slope**（同 6.1；label null 同时控 density 的维度计数偏置）。`mode_shift_density_surplus_slope = obs_slope − median(null)`；`p_label_slope` 单侧（期望>0）。
- secondary null：phase-randomize / block-shuffle（V3a dynamics null，扩到斜率）。
- 一致性核对：2D-VAR（`project_2d`→`direct_2d_var`）主奇异模态-shift 斜率（`mode_shift_2D_consistency_slope`）应同号。

### 6.3 H3p-a 轴向减弱（supportive-only）
- 每窗：contact susceptibility = **bb-envelope line-length rate**（V2 `contact_susceptibility`，roughness）→ `beta_axis(metric_by_name, rank_forward)` 得 `|β_axis|`（axis 触点上）。
- 每发作：`beta_axis_strength_slope = TheilSen(|β_axis|_w, t_w)`；期望 **< 0**。
- **可靠性门**：`|β_axis|` 在 P0..P3 中位数 < `beta_axis_reliability_min` → "无轴向结构可减弱"，H3p-a 不可判。
- **`module_support_flag` 恒 False**（H3p-a 永不单独定义 support）；显著只加强解释。
- null：shaft-spatial + order（打乱 rank_forward 保参与度）+ label。

### 6.4 H3p-d 非轴向负担 / 自持 / 增益（secondary/convergent；Step-last）
- **nonaxis_activation_burden**：每窗 non-axis-strict 触点的平均激活率 → 斜率。**这里残差化是命门**：报 `_slope_raw` + `_slope_resid`（对 `global_activation_rate(t)` 回归后残差斜率）；label-null 主裁。
- **N_self_sustain**：小路外→小路外自持连锁 = 每窗 non-axis 源在 non-axis 目标上的 `atm_offdiag` 平均质量（V3p 新增小 helper `within_compartment_flux(atm, idx)`，N×N block source_mean）→ 斜率。
- **gain_nonaxis_surplus**：每窗 2D-VAR `B_direct` 的 `gain_nonaxis = ‖B e_nonaxis_mean‖`，减 phase/block surrogate 中位数 → surplus → 斜率。
- **判读**：Step2/3(co-primary) 阴性时 H3p-d 不过度解释（收敛证据，非独立主张）。

---

## 7. 残差化 + null（命门）

**两个正交混淆，两套控制：**

1. **"发作前全局升温"（趋势层混淆）→ label-null-of-slope（主裁，L4）**。同杆内随机重指派 axis/non-axis 标签，重算**整条**指标轨迹 + 斜率。若上升是全场同步的，换标签得同斜率 → 不显著；只有上升**专门压在真非轴向触点**上才超 null。**这就是"残差化全局能量"的置换化落地**——比回归更不怕 time↔energy 共线性。
2. **"非轴向触点多/基率高"（水平层混淆）→ 指标本身已是 density/source-normalized 对比**（`mode_shift_density` ÷rank、`net_offaxis_flux` source_mean），且 label-null 保杆内计数。

**回归残差（sensitivity，L4）**：每发作 preictal 窗上，
- `global_energy_w` = 全 clean 触点平均 bb-envelope 能量（或平均激活率）；
- `axial_energy_w` = axis 触点平均能量；
- 每指标 OLS `m_w ~ 1 + global_energy_w + axial_energy_w` 取残差 `r_w` → `TheilSen(r_w, t_w) = *_slope_resid`。
- **collinearity 注（写进判读纪律）**：若 `global_energy_w` 本身随 `t_w` 单调（发作前升温常见），回归会把"与全局同向的那部分非轴向上升"一并吸走 → `*_slope_resid` 是**保守下界**，可能过度剥离。**因此 label-null 是裁判，回归残差只作稳健核对**：`slope_resid` 同号 = 加强；`slope_resid≈0` **不**推翻 label-null 阳性（只说明"非轴向上升与全局上升相关"，不说明"非轴向没有专属上升"）。

**cohort 聚合用标准化效应**：`slope_label_z = (obs_slope − median(null))/MAD(null)`（跨被试/跨指标可比），Wilcoxon signed-rank on 被试 `slope_label_z`（方向正确）+ 显著被试数；co-primary Holm 校正。

---

## 8. 统计骨架 + tier 判定

- **subject 为单位**；**每假设唯一 primary 斜率**（§2），其余 exploratory。null 全继承 V3a 自建（shaft-spatial/rate-preserving/label）扩到斜率；经验 `p=(1+越界)/(1+n_perm)`；趋势单侧（H3p-a 反向单侧）。
- **subject_support** = `(H3p-b OR H3p-c 的 module_support_flag)` AND 方向正确 AND label-null 过 AND onset ±10s jitter 稳 AND 非单一 contact 驱动（`top_contact_energy_fraction ≤ single_contact_energy_frac_max` / leave-one-contact 斜率符号存活）AND axis-only relabel 不能解释。**H3p-a 只加强，单独不算。** `geometry_insufficient` 不计入分母。
- **cohort**：分 narrow/broad；co-primary H3p-b/H3p-c 的被试 `slope_label_z` 做 **Holm 校正** Wilcoxon；报 subject-median effect + sign test + bootstrap CI + 显著被试数。
- **tier（同 V3a）**：

| tier | 定义 |
|---|---|
| 0 | 无支持 |
| 1 | 仅描述性方向，null 不显著 |
| 2 | ≥1 被试 subject-level 支持，无 cohort 方向 |
| 3 | **narrow cohort co-primary 支持（Holm 过）** |
| 4 | narrow 支持 + broad 同向复制 |
| 5 | tier 4 + 模型侧一致性（本 spec 外） |

**`state_v3p_supported = tier ≥ 3`；V3p 最高 tier 4。**

---

## 9. 输出 schema（每 subject 一行 + phase/window 明细另表）
```
# phase/eligibility
subject, cohort, n_seizures_total, n_seizures_used, n_windows_{P0,P1,P2,P3}_median,
usable_pre_sec_median, min_windows_for_slope, onset_anchor, onset_jitter_pass
# geometry quality（继承 V3a）
n_contacts_all_clean, n_axis, n_nonaxis, n_ambiguous, n_shaft_with_axis_and_nonaxis, geometry_insufficient
# H3p-b flux (co-primary)
net_offaxis_flux_slope_raw, net_offaxis_flux_surplus_slope, net_offaxis_flux_slope_resid,
net_offaxis_flux_slope_z, p_label_slope_b, p_rate_slope_b, p_spatial_slope_b,
proximal_flux_slope, leave_one_contact_flux_pass, axis_only_flux_control_pass,
lag1_specific_slope, common_drive_sensitive, module_support_flag_b, module_direction_correct_b, module_null_pass_b
# H3p-c mode-shift (co-primary)
mode_shift_density_slope_raw, mode_shift_density_surplus_slope, mode_shift_density_slope_resid,
mode_shift_density_slope_z, p_label_slope_c, p_phase_slope_c, p_block_slope_c,
mode_shift_2D_consistency_slope, top_contact_energy_fraction, single_contact_driven,
leave_one_contact_mode_pass, axis_only_mode_control_pass, rank_used, k_star,
module_support_flag_c, module_direction_correct_c, module_null_pass_c
# H3p-a axial weakening (supportive-only)
K_primary_metric(=line_length_rate), beta_axis_strength_slope, beta_axis_reliable,
beta_axis_slope_z, p_label_slope_a, module_support_flag_a(=False by construction)
# H3p-d secondary
nonaxis_activation_burden_slope_raw, nonaxis_activation_burden_slope_resid, burden_slope_z, p_label_burden,
N_self_sustain_slope, N_self_sustain_slope_z, p_label_selfsustain,
gain_nonaxis_surplus_slope, gain_axis_slope, gain_nonaxis_slope_z
# trend companions
trend_estimator(=theil_sen), spearman_rho_flux, spearman_rho_mode, slope_span(=full/proximal)
# verdict
axis_weakening_supportive, nonaxis_flux_amplification_supported(H3p-b), mode_transition_supported(H3p-c),
tier, state_v3p_supported
```
per-window 明细另表 `v3p_window_detail.csv`：`subject, cohort, seizure_idx, phase, t_center, net_offaxis_flux, mode_shift_density, nonaxis_activation_rate, global_energy, axial_energy, N_self_sustain`。

---

## 10. 脚本/测试 + 实现顺序 + 复用边界

**新增文件（V3p，import V3a read-only）**：
```
config/topic5_v3p.yaml                       # preictal span / trend estimator / min_windows / residualization covariates / co-primary / tier（几何+dynamics 锁值引用 topic5_v3.yaml）
src/topic5_v3p_preictal_trajectory.py        # 纯数学: theil_sen_slope, spearman_trend, slope_over_windows,
                                             #   within_compartment_flux(N→N 自持), residualize_slope(vs global+axial),
                                             #   null_slope(permute_fn, 轨迹重算 callback → label/rate/spatial 三种共用),
                                             #   global_axial_energy
scripts/run_topic5_v3p_feasibility.py        # preictal 窗计数 + 几何门 pilot（confirm ≥4 narrow qualify）
scripts/run_topic5_v3p_trajectory.py         # 单 runner: 一次 preictal 窗循环产全指标轨迹→斜率→null→残差
scripts/run_topic5_v3p_summary.py            # Holm co-primary + tier + cohort JSON（tier 只在此）
scripts/plot_topic5_v3p_summary.py           # 2–3 独立问题 panel（CLAUDE.md §7）
tests/test_topic5_v3p_preictal_trajectory.py # 纯函数
tests/test_topic5_v3p_integration.py         # two-tier（skipped-ok + eligible-runs，复用 V3a integration subject 253）
```
输出：`results/topic5_ictal_recruitment/v3p_preictal_trajectory/{narrow,broad}/`。

**复用边界（继承 V3a，read-only，不改共享文件——保合并干净）**：
- `src/topic5_v3_mode_transition.py`：相位窗、几何、三 null、avalanche、dynamics helper——**全部直接 import**。
- `scripts/_topic5_v3_io.py`：`classify_subject_contacts`、`load_subject_phase_envelopes`（传 `phases=["P0","P1","P2","P3"]`）——直接调。
- `config/topic5_v3.yaml`：pilot-locked 几何/dynamics 阈值——V3p config 引用其锁值，不重 pilot。
- **只新增**：preictal 限定循环、Theil-Sen/Spearman 斜率、N→N 自持 helper、回归残差、label-null-of-slope 编排、global/axial energy。

**实现顺序（feasibility 先行，防返工）**：
- **Step 0** config + module skeleton（`load_v3p_config`）。
- **Step 1** 纯函数（斜率/趋势/残差/within-compartment/label-null-of-slope 编排）+ 纯测试。
- **Step 2** feasibility pilot：每 subject 出 `n_windows_P0..P3 / n_axis / n_nonaxis / usable_pre_sec / geometry_sufficient`；确认 narrow ≥4 qualify + lock `min_windows_for_slope`。**<4 → STOP + 报告**。
- **Step 3** trajectory runner（co-primary H3p-b/H3p-c 先，H3p-a/H3p-d 随后同循环）+ two-tier integration。
- **Step 4** summary + tier（Holm）+ figure（render→eyeball→fix，写 `figures/README.md` 中文 + append `results/FIGURE_INDEX.md`）。

---

## 11. 判读纪律 / 禁止 claim
- **禁 forecasting/prediction**：不出 lead-time、AUC、前向分类、"发作前 X 秒可预警"任何主张——V3p 是**趋势描述 + null 裁判**，非预测器。
- 不能说"没有临界性 / 发作前没有 state projection"；raw λmax 不当临界（只 `λ_surplus`）；rank-coupling 不当传播（只 flux）；**eigenvector 不当最大放大方向（用奇异向量）**。
- **H3p-a 轴向减弱单独不算 support**；只在 `beta_axis_reliable` 时可判。
- **回归残差 `*_slope_resid≈0` 不推翻 label-null 阳性**（collinearity 保守下界，§7）；label-null 是裁判。
- **label-null 是"专门集中在非轴向"的唯一裁判**：obs_slope 大但 label-null 不显著 = "只是全局升温"，**不算 support**。
- 非轴向定义不得用发作 / preictal criticality 结果（防循环）；`geometry_insufficient ≠ negative`；短 pre-onset 记录 → feasibility-insufficient ≠ negative。
- **预注册阴性可接受**：若队列级 co-primary 不过，诚实写"数据不支持 preictal 非轴向搬迁"，**加固**共享粗骨架主线；**不救 1125**（个别被试只作描述性 case-series，不升队列主张）。
- EXPLORATORY 全程；tier 只在 summary 判。

## 12. 决定项 status
- **[LOCKED]** L1–L10 + preictal-only + Theil-Sen 斜率 + co-primary(flux+mode) + label-null 主裁 + 回归残差 sensitivity + eeg-onset 锚 + narrow 主 + tier + 继承 V3a 几何/dynamics/null + read-only import。
- **[OPEN, pilot 定]** `min_windows_for_slope`（默认 8）；proximal-only sensitivity 是否入 primary 表；短记录 fallback。
- **[本 spec 外]** 模型侧 preictal 一致性（V3b 类比）；表达层。

## 13. 核心命题（一句话）
> **V3p 只在发作真正开始前的两分钟（`P0..P3` = −120~−10s，eeg-onset 锚定、不碰 ictal）问一个 V3a 没问的形状：非轴向的连锁流（`net_offaxis_flux`，H3p-b）与最易放大方向（`mode_shift_density`，H3p-c）是否随发作临近**逐渐上升**（Theil-Sen 斜率>0），且这上升**专门集中在非轴向触点**——由**同杆 label-null-of-slope** 裁判（天然控"发作前全局升温"），对 `global+axial energy(t)` 的回归残差斜率作保守 sensitivity。co-primary = H3p-b + H3p-c（Holm）；H3p-a 轴向减弱 supportive-only；H3p-d 负担/自持/增益 secondary。被试为单位、narrow 主 broad 复制、判定 tier 0–5、`state_v3p_supported=tier≥3`、V3p 最高 tier 4。全程 EXPLORATORY、**无 forecasting**、预注册阴性可接受不救 1125。read-only 继承 V3a 全部机制，只加 preictal 限定 + 斜率 + 残差化 + N→N 自持。**
