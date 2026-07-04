# Topic 5 V3p 设计 — 发作前(preictal-only)非轴向轨迹:小路之外的活动/流/可放大模态是否随发作临近逐渐爬升

date 2026-07-03 · **rev1（post-review 收紧，2026-07-04）** · 前身 = V3a（`docs/superpowers/specs/2026-07-02-topic5-v3a-mode-transition-design.md`，P3→I1 onset 对比 → tier 2 偏阴性）· 姊妹 = V2a（restricted axial sanity）

> **rev1 变更（依用户 review "accept with targeted revisions"）**：(1) **onset guard 双轨**——headline primary span 保留 `−120~−10s`，另加 jitter-safe guard span `−120~−20s`（`guard_sec=max(10,jitter)`），强支持须**两轨同过**，只 full 过 = `near_onset_dependent` 降级；(2) **H3p-b support 硬门**——`p_rate_slope_b<α` 从 secondary 升为硬门 + `lag1_specific_slope>0`（lag1−lag0 flux 斜率）硬门；(3) **rate-preserving null 逐窗**——保每-contact **每窗**激活率、只打乱窗内 target timing（不破 preictal burden 轨迹）；(4) **H3p-c support 硬门**——加 `p_phase_slope_c<α`（label+phase+block 三合取=strong、label+一个 temporal=weak）；(5) **gain 改相对**——`gain_shift_slope=slope(gain_nonaxis−gain_axis)` 主看，`gain_nonaxis_surplus_slope` 保留；(6) **QC 列**——`mode_singular_gap_median`/`mode_vector_stable`/`cv_r2`（H3c 奇异向量稳定性）、`h3b_activation_sufficient`/`n_activation_events_pre`（H3b 稀疏保护）、`N_self_sustain_lag1_specific_slope`；(7) **label-null 可置换性 QC**——`n_label_permutable_shafts`/`n_unique_label_permutations_est`/`label_null_underpowered`（实测 narrow 1146 shaftBoth=1、1096/1125=2，真问题），underpowered 出强阳性分母；(8) **time-order null**——窗序循环移位 secondary sensitivity（`time_order_p_{b,c}`）；(9) cohort 实测干净 + 可扩候选见 §10。

> **一句话缘起**：V3a 用「发作前 −30~−10s(P3) 对比 发作后 +10~+30s(I1)」两时刻差，只有个别被试(1125)像、凑不成队列(tier 2)，因为 I1 落在发作**已点着之后**（信噪比塌/饱和/淹没）终点脏。V3p **把终点从"发作后"挪走，只看发作前干净时段**，问非轴向的活动/流/可放大方向是否随发作临近**逐渐爬升**、且**专门集中在非轴向触点**（同杆 label-null 裁判、控"发作前全场升温"）。

---

## 0. 摘要（朴素话）

病人两次发作之间，短暂高频异常放电按一条固定先后顺序在电极间传开——一条走熟的小路（间期传播轴 `G_HFO`）。有人猜发作启动时系统会把"沿小路的有序传播"松掉、转到小路**之外**去。这个猜想在本仓库已问过三次都弱/不稳（发作内场动力学 broad 暗示/narrow 证否、V2a 受限临界性偏阴性、V3a onset 两时刻差 tier 2）。

V3p 换**最干净的一个窗口**再问一次：**只看发作真正开始之前的两分钟**（`P0..P3`，按每次发作电生理 onset 锚定，完全不碰发作本身）。在这段还没发作、信号还干净的时间里，看小路**之外**的触点（non-axis strict）是不是**一路缓慢爬升**（不是突然跳）出现三件事：(1) **激活负担**变重；(2) **自我维持**（小路外→小路外连锁增多）；(3) **最容易被放大的方向**越来越偏离小路。

**命门:必须扣掉"发作前大家一起升温"。** 真正裁判是**同杆 label-null**（把"哪些算小路外"在杆内打乱重算整条斜率——全场同涨则打乱后斜率一样大→不显著；只有上升**专门压在真非轴向触点**上才超 null）。另外 flux 还要过"rate 保持+lag0 剔除"硬门（区分真·延迟传播 vs 只是激活多/同步爆），mode-shift 还要过时间 surrogate 硬门（区分真·模态转移 vs 频谱/平滑假象）。

**onset guard（rev1 关键）**:V3p 卖点是"不碰 ictal"。但把 anchor 往后平移 +10s，原 P3 `−30~−10s` 会变成 `−20~0s` 贴到 onset 边界。所以**强支持要求在 jitter-safe 的 `−120~−20s` 段也成立**；只在 `−120~−10s` 成立、`−120~−20s` 不成立的 → 标 `near_onset_dependent` 降级。

**诚实预期**:这是"轴向→非轴向"假设的最后一枪。**若阴性**——诚实结论"数据不支持非轴向搬迁"，**加固** Topic 5 主线（间期轴=共享粗骨架）；**若干净爬坡**——最强阳性证据。全程 EXPLORATORY、**无 forecasting**、预注册阴性可接受、**不救 1125**。

（内部代号：`G_HFO`=间期 HFO typical_rank 顺序场；non-axis strict=纯间期 HFO 参与度低的 clean 触点 + 子空间 `P_N`；`slope`=preictal 窗上 Theil-Sen 稳健斜率；`label-null-of-slope`=同杆打乱轴/非轴标签重算斜率的置换零假设；`surplus_slope`=obs_slope − label-null 中位数；`lag1_specific`=lag1−lag0 flux；`gain_shift`=gain_nonaxis−gain_axis；tier 0–5=证据强度分层。）

---

## 1. 与 V3a 的关系 + honest framing

- **V3a 不删、不改**：V3p **read-only 继承** V3a 全部纯数学机制（`src.topic5_v3_mode_transition`：相位窗 / 几何 / 三 null / avalanche / dynamics）+ io（`_topic5_v3_io`：`classify_subject_contacts` / `load_subject_phase_envelopes`）。V3p 只加新文件（preictal 限定 + 斜率 + 残差化 + N→N 自持 + rev1 硬门/QC）。
- **同一科学问题的第三个角度**：§3.6（场层）、V3a（onset 两时刻差层）、V3p（preictal 轨迹层）。V3p 设计上最能规避 SNR/饱和混淆。
- **honest coupling**：V3p 阳性 = "数据里，发作前非轴向活动/流/可放大模态存在**专门集中在非轴向触点**的上升趋势"。**不是发作预测/forecasting**——是对一段 preictal 窗内趋势的描述性 + null 裁判，不出 lead-time/AUC/前向分类。

---

## 2. 主假设 + 子假设（primary 与 gate 一致）

**H3p（主）**：从 far-preictal(`P0`, −120s) 到 late-preictal(`P3`, −10s / guard −20s)，非轴向的连锁流向 / 最易放大方向随发作临近**逐渐上升**，且**专门集中在非轴向触点**（label-null 裁判）。

| 子假设 | 朴素话 | **primary 指标（preictal 窗上唯一斜率）** | 方向 | 角色 | V3a 对应 |
|---|---|---|---|---|---|
| **H3p-a 轴向减弱** | 沿小路的有序组织随发作临近变弱 | `beta_axis_strength_slope`（窗内 line-length-rate 基） | **< 0** | **supportive only** | H3a |
| **H3p-b 非轴向流放大** | 连锁从轴向漏向非轴向随时间增多 | `net_offaxis_flux_surplus_slope`（= obs_slope − label-null 中位数） | **> 0** | **承重 co-primary** | H3b |
| **H3p-c 模态转移** | 最易放大方向随时间转向非轴向子空间 | `mode_shift_density_surplus_slope`（`A_lowrank^{k*=3}` 主奇异向量映回、density-归一 P_N−P_A、斜率减 label-null 中位数） | **> 0** | **承重 co-primary** | H3c |
| H3p-d 负担/自持/增益 | 小路外激活重、自持连锁增多、相对可放大增益上升 | `nonaxis_activation_burden_slope`（残差化）、`N_self_sustain_lag1_specific_slope`、`gain_shift_slope`（=slope(gain_nonaxis−gain_axis)） | **> 0** | secondary/convergent | 无（V3p 新增） |

**support = H3p-b OR H3p-c**（cohort 级 **Holm** 校正 2 个 co-primary），**且各自过 rev1 硬门**（§6）+ 两轨同过（§4）+ onset ±10s jitter 稳 + 非单一 contact + axis-only relabel 不能解释 + label-null 非 underpowered。**H3p-a 显著只加强解释、单独不算 support**。H3p-d secondary/convergent，Step-last，co-primary 阴性则不过度解释。

---

## 3. 已锁决定（本 session + rev1 review；可回改）

- **[LOCKED L1]** **preictal-only**：primary 只用 `P0/P1/P2/P3`，**不碰** `O/I1/I2/I3/Post`。
- **[LOCKED L1b（rev1）]** **onset guard 双轨**：headline `full=[−120,−10]`（用户点名保留）+ jitter-safe `guard=[−120,−20]`（`guard_sec=max(10, jitter_primary=10)=10` → primary_end=onset−20）。**两轨都算 co-primary 指标**，强支持须**两轨同过**；只 full 过 → `near_onset_dependent=True` 降级。proximal `[−60,−10]` 作 sensitivity。
- **[LOCKED L2]** **趋势 = Theil-Sen 斜率**（primary）；报 Spearman ρ 单调性伴随 + OLS alt。被试值 = median over seizures。
- **[LOCKED L3]** **co-primary = H3p-b + H3p-c**，Holm；H3p-a supportive-only（`module_support_flag_a` 恒 False）；H3p-d secondary。
- **[LOCKED L4]** **残差化 = label-null 主裁 + 回归残差 sensitivity**（conservative floor，`slope_resid≈0` 不推翻 label-null 阳性）。
- **[LOCKED L4b（rev1）]** **H3p-b 硬门** = `direction ∧ p_label_slope_b<α ∧ p_rate_slope_b<α ∧ lag1_specific_slope>0`（rate + lag0-common-drive 硬门，区分真流 vs 激活多/同步爆）。
- **[LOCKED L4c（rev1）]** **rate-preserving null 逐窗**：保每-contact **每窗**激活计数、只打乱**窗内** target timing（破跨-contact 延迟配对，不破 preictal burden 轨迹）。**不是**把整段 preictal 打散（那会破坏 per-contact rate slope → 假阳性）。
- **[LOCKED L4d（rev1）]** **H3p-c 硬门** = `direction ∧ p_label_slope_c<α ∧ p_phase_slope_c<α ∧ p_block_slope_c<α`（strong）；label + 一个 temporal surrogate = weak。理由：V2a 教训——宽带 envelope 平滑本身造高自相关/近临界假象，label-null 只答"标签重不重要"，phase/block 才答"是不是频谱/平滑/块结构"。
- **[LOCKED L5]** **eeg-onset 锚定**；onset ±{5,10,15}s jitter，±10s 方向稳过。
- **[LOCKED L6]** **几何/非轴向定义 = 继承 V3a**（three-class、`P_A/P_N`、纯间期 HFO 参与度、对 ictal+preictal 全盲防循环）；阈值沿用 V3a `config/topic5_v3.yaml` 锁值。
- **[LOCKED L7]** λ 只报 surplus；mode-shift 用奇异向量；VAR within-window demean 不 standardize；ATM `i≠j`。
- **[LOCKED L7b（rev1）]** **H3c 奇异向量稳定性 QC**：`mode_singular_gap_median=median(σ1/σ2 of A^{k*})`、`mode_vector_stable=gap≥阈`、`cv_r2`（重建质量）；gap 小 → mode_shift slope 标 low-confidence（**输出，non-hard 默认**）。
- **[LOCKED L7c（rev1）]** **H3b 激活充分性 QC**：`n_activation_events_pre`/`n_active_windows_pre`/`h3b_activation_sufficient`（激活太稀疏别把 0 当阴性）。
- **[LOCKED L7d（rev1）]** **gain 改相对**：`gain_shift_slope=slope(gain_nonaxis−gain_axis)` 为 H3p-d gain 主看（保 `gain_nonaxis_surplus_slope` 供参考，避免"全系统 gain 都涨"误读）。
- **[LOCKED L7e（rev1）]** **N_self_sustain 硬化**：`N_self_sustain_lag1_specific_slope=N_self_sustain_lag1_slope−N_self_sustain_lag0_slope`；≤0 → "非轴向同步共激活增强"非"自持连锁增强"。
- **[LOCKED L8]** **subject 为单位**；**narrow 主 broad 复制，永不 pool**。
- **[LOCKED L8b（rev1）]** **label-null 可置换性 QC**：`n_label_permutable_shafts`/`n_label_permutable_{axis,nonaxis}`/`n_unique_label_permutations_est`/`label_null_entropy`/`label_null_underpowered`（有效置换 <100 → True）。underpowered subject **不计入强阳性分母**（或降级）。实测 narrow 1146 shaftBoth=1、1096/1125=2 → 真问题。
- **[LOCKED L8c（rev1）]** **time-order null（secondary sensitivity）**：每发作窗序循环移位/打乱、保 metric 值+标签、重算斜率 → `time_order_p_{b,c}`；答"slope 是否真依赖越近 onset 越强的顺序"（非硬门，尤其 H3p-c）。
- **[LOCKED L9]** 判定 **tier 0–5**（§8），`state_v3p_supported=tier≥3`，V3p 最高 tier 4。
- **[LOCKED L10]** **EXPLORATORY**，**无 forecasting**；预注册阴性可接受、不救 1125。
- **[OPEN, pilot 定]** `min_windows_for_slope`（默认 8；**实测每发作 ~17-18 窗、两队列每发作都 ≥8，非约束**）；奇异 gap / label-perm underpowered / 激活充分 的具体阈值。
- **[OPEN, 用户定]** **cohort 是否扩**（§10：6 个候选可扩，但非原 swap-curated 队列 → 需 axis-quality 门）。

---

## 4. 时间设计（preictal-only；eeg-onset 锚定；斜率；onset guard 双轨）

**只用 preictal 相位**（继承 V3a `phase_bin_range`，anchor = `eeg_onset_rel + onset_shift`）：

| phase | 相对 eeg_onset | 角色 |
|---|---|---|
| P0 | −120~−90 | 轨迹起点 |
| P1 | −90~−60 | 轨迹 |
| P2 | −60~−30 | 轨迹 |
| P3 | −30~−10 | full-span 轨迹终点 |
| ~~O/I1/I2/I3/Post~~ | — | **V3p 不用** |

**onset guard 双轨（rev1 L1b）**：
- **`full` span = `[−120, −10]`**（headline primary，用户点名保留）。
- **`guard` span = `[−120, −20]`**（jitter-safe，`guard_sec=max(10, jitter_primary_sec=10)`；等价 P3 收到 `−30~−20`）。
- **两轨都算 co-primary 斜率 + null**；strong support 须**两轨同方向同过 null**；只 full 过、guard 不过 → `near_onset_dependent=True`（tier 降级，见 §8）。
- **proximal `[−60, −10]`** slope 作 sensitivity（趋势是否集中在临近 onset 段）。

- **滑窗** 10s/5s；**实测每发作 full-span ~17-18 窗**（cache relt 覆盖 [−152,...]，preictal 数据充裕）。
- **趋势统计**（每发作）：`slope = TheilSen(metric_w, t_w)` primary，`rho = Spearman(metric_w,t_w)` 伴随，`slope_ols` alt。被试 = median over seizures。
- **min_windows_for_slope**（默认 8）：某发作窗数 < 阈 → 不进斜率；被试 0 usable → feasibility-insufficient（**≠ negative**）。**实测两队列每发作都 ≥8，非约束**。
- **onset jitter**：anchor ±{5,10,15}s 重算；**±10s 方向不变**过，±15s 压力。guard 双轨本身就是 jitter-safety 的结构化落地。

---

## 5. 空间设计（继承 V3a，防循环铁律不变）+ 实测几何

- **three-class**（`classify_subject_contacts` single source of truth）：axis=有限 typical_rank 或 `axis_partition` 端点；non-axis-strict=clean ∧ ¬axis ∧ 间期 HFO 参与度<阈；ambiguous=clean ∧ ¬axis ∧ 参与度≥阈（进 all-clean VAR X，不进 `P_A/P_N`）。
- **子空间** `P_A/P_N`；`e_axis_mean/e_axis_grad/e_nonaxis_mean`（uniform）。
- **防循环铁律**：非轴向集**不得**用同批发作 ictal / preictal criticality 结果。
- **geometry-sufficiency 门**：`n_axis≥5` AND `n_nonaxis≥3` AND ≥1 杆同时有 axis+non-axis。不满足 → `geometry_insufficient`（**≠ negative**）。
- **实测（2026-07-04，V3a io）**：broad 9/9 + narrow 7/7 全 geometry_sufficient，n_nonaxis 充裕（10–104）。**label-null 隐患**：narrow `1146` shaftBoth=**1**、`1096`/`1125`=**2** → 同杆置换空间小，须 `label_null_underpowered` QC（L8b）；broad shaftBoth 3–9 健康。

---

## 6. 四条腿（preictal 窗上逐窗指标 → 斜率）+ rev1 硬门/QC

所有腿共用**一次 preictal 窗循环**（一个 trajectory runner）。每窗产一组标量，跨窗拟合斜率。**每指标在 full + guard 两 span 各算一次。**

### 6.1 H3p-b 非轴向流放大（co-primary）
- 每窗：阈化激活（`z>z_threshold`）→ `atm=atm_offdiag`（`i≠j`）→ `net_offaxis_flux(...,"source_mean")`（lag1）+ `atm_lag0` 版（同时共激活）。
- 每发作斜率 → 被试 median。**`net_offaxis_flux_surplus_slope = obs_slope − median(label-null slope)`**；`p_label_slope_b`（单侧 >0）。
- **硬门（rev1 L4b）**：`module_support_flag_b = direction ∧ p_label_slope_b<α ∧ p_rate_slope_b<α ∧ lag1_specific_slope>0`。
  - `lag1_specific = lag1_net_offaxis_flux − lag0_net_offaxis_flux`；`lag1_specific_slope`=其斜率（>0 = 真·延迟 A→N 流增强，非同步爆）。
  - `p_rate_slope_b` = rate-preserving-null-of-slope（**逐窗** rate 保持，L4c）。
  - **降级**：label 过但 rate/lag0 不过 → 结论 = "非轴向 activation burden / common-drive 上升"，**非**明确 axis→nonaxis flow。
- secondary null：`shaft_constrained_permute`-of-slope。secondary metric：`leak_index`、`branching_N`。
- **激活充分性 QC（L7c）**：`n_activation_events_pre`/`n_active_windows_pre`/`h3b_activation_sufficient`；激活太稀疏 → 0 flux **不当阴性**。

### 6.2 H3p-c 模态转移（co-primary）
- 每窗：all-clean X（含 ambiguous）`demean_window` → `lowrank_var` → `dominant_right_singular_vector(A_lowrank,k*=3)` → `map_lowrank_vector_to_contacts` → `subspace_mode_shift(...,"density")`。**逐窗 = V3a H3c 算法逐字复用。**
- 每发作斜率 → 被试 median。**`mode_shift_density_surplus_slope = obs_slope − median(label-null slope)`**；`p_label_slope_c`（单侧 >0）。
- **硬门（rev1 L4d）**：`module_support_flag_c = direction ∧ p_label_slope_c<α ∧ p_phase_slope_c<α ∧ p_block_slope_c<α`（**strong**）；`label + 一个 temporal surrogate` = **weak**（分级输出，非直接 support）。
- **奇异向量稳定性 QC（L7b）**：`mode_singular_gap_median=median(σ1/σ2 of A^{k*})`、`mode_vector_stable`、`cv_r2`；gap 小 → `u1` 不稳、mode_shift slope 标 low-confidence。
- 一致性核对：2D-VAR 主奇异模态-shift 斜率（`mode_shift_2D_consistency_slope`）应同号。

### 6.3 H3p-a 轴向减弱（supportive-only）
- 每窗 per-contact **line-length rate（roughness，内联算 `mean|diff(env_i)|`，非 V2 `contact_susceptibility` 的 late−early Δ，§6.1 问题不匹配）** → `beta_axis(...)` 得 `|β_axis|`。
- 斜率期望 **<0**；`beta_axis_reliable = median(|β_axis|)≥beta_axis_reliability_min`（否则不可判）。
- **`module_support_flag_a` 恒 False**（永不单独 support）；`p_label_slope_a`（单侧 <0）。

### 6.4 H3p-d 非轴向负担 / 自持 / 相对增益（secondary/convergent；Step-last）
- **nonaxis_activation_burden**：每窗 non-axis-strict 平均激活率 → 斜率。**残差化命门**：报 `_raw` + `_label_surplus` + `_resid`（对 `global_activation_rate(t)` 回归）。只 burden 阳性 → "小路外负担有 preictal increase，但不足以说 flow / 模态转移"。
- **N_self_sustain（rev1 L7e 硬化）**：每窗 non-axis→non-axis `within_compartment_flux`（lag1）+ lag0 版；`N_self_sustain_lag1_specific_slope = lag1_slope − lag0_slope`；≤0 → "非轴向同步共激活"非"自持连锁"。
- **gain_shift（rev1 L7d）**：每窗 2D-VAR `gain_nonaxis=‖B e_nonaxis_mean‖`、`gain_axis=‖B e_axis_mean‖`；`gain_shift_slope=slope(gain_nonaxis−gain_axis)` 主看；`gain_nonaxis_surplus_slope`（减 phase/block null）保留供参考。

---

## 7. 残差化 + null（命门）+ rev1 QC

**三个正交混淆，三套控制**：
1. **"发作前全局升温"（趋势层）→ label-null-of-slope（主裁，L4）**。同杆重指派 axis/non-axis 标签、重算整条斜率。全场同涨 → 换标签同斜率 → 不显著；只有专门压真非轴向才超 null。**这就是"残差化全局能量"的置换化落地**（比回归更不怕 time↔energy 共线性）。
2. **"非轴向触点多/基率高"（水平层）→ 指标已是 density/source-normalized 对比 + label-null 保杆内计数**。
3. **"真延迟流 vs 激活多/同步爆"（H3b 特有）→ rate-preserving 逐窗 null（L4c）+ lag0 common-drive 剔除（lag1_specific，L4b）**。
4. **"真模态转移 vs 频谱/平滑假象"（H3c 特有）→ phase + block temporal surrogate 硬门（L4d）**。

**回归残差（sensitivity，L4）**：每发作 preictal 窗，`global_energy_w`（全 clean 平均能量）+ `axial_energy_w`（axis 平均），每指标 OLS `m_w~1+global+axial` 取残差 → `TheilSen → *_slope_resid`。**collinearity 注**：`global_energy_w` 随 `t_w` 单调时回归会吸走"与全局同向的非轴向上升" → `*_slope_resid` 是**保守下界**；`slope_resid≈0` **不**推翻 label-null 阳性。

**label-null 可置换性 QC（rev1 L8b）**：`n_label_permutable_shafts`（同时有 axis+nonaxis 的杆数）、`n_label_permutable_{axis,nonaxis}`、`n_unique_label_permutations_est`（∏ 各杆杆内标签排列数的对数估计）、`label_null_entropy`、`label_null_underpowered=（有效置换 <100）`。underpowered subject **不计强阳性分母 / 降级**。

**time-order null（rev1 L8c，secondary sensitivity）**：每发作窗序循环移位、保 metric 值+标签、重算斜率 → `time_order_p_{b,c}`（答"是否真依赖越近 onset 越强"）。非硬门，尤其看 H3p-c。

**cohort 聚合**：`slope_label_z=(obs−median(null))/MAD(null)`；Wilcoxon signed-rank on 被试 z（方向正确）+ 显著被试数；co-primary Holm。

---

## 8. 统计骨架 + tier 判定

- **subject 为单位**；每假设唯一 primary 斜率（§2）；null 继承 V3a 自建扩到斜率；`p=(1+越界)/(1+n_perm)`；趋势单侧。
- **subject_support** = `(H3p-b 硬门 OR H3p-c 硬门)` AND **两轨(full+guard)同过** AND onset ±10s jitter 稳 AND 非单一 contact（`top_contact_energy_fraction ≤ 阈` / leave-one-contact 符号存活）AND axis-only relabel 不能解释 AND **`label_null_underpowered=False`**。**H3p-a 只加强。** `geometry_insufficient` / feasibility-insufficient 不计分母。
- **降级**：只 full 过 guard 不过 → `near_onset_dependent=True`，tier 封顶 2（不进 cohort 强支持）。`label_null_underpowered=True` → 该被试不计强阳性分母。H3p-c 只 weak（缺 phase 或 block）→ 不算硬门 support。
- **cohort**：分 narrow/broad；co-primary H3p-b/H3p-c 被试 z 做 **Holm** Wilcoxon；报 subject-median effect + sign test + bootstrap CI + 显著被试数。
- **tier**：0 无 / 1 仅描述方向 null 不显著 / 2 ≥1 被试硬门 support 无 cohort 方向 / 3 **narrow cohort co-primary Holm 过（两轨）** / 4 narrow + broad 同向复制 / 5 模型侧（本 spec 外）。`state_v3p_supported=tier≥3`，V3p 最高 tier 4。narrow+broad 永不 pool。
- **honest-negative**：tier ≤1 → `pre_registered_negative=True`（加固共享粗骨架主线）。

---

## 9. 输出 schema（每 subject 一行 + window 明细另表）
```
# phase/eligibility/geometry
subject, cohort, n_seizures_total, n_seizures_used, n_windows_full_median, n_windows_guard_median,
min_windows_for_slope, onset_anchor, onset_jitter_pass, primary_span_end_sec(=-10 headline),
n_contacts_all_clean, n_axis, n_nonaxis, n_ambiguous, n_shaft_with_axis_and_nonaxis, geometry_insufficient
# label-null permutability QC (rev1)
n_label_permutable_shafts, n_label_permutable_axis, n_label_permutable_nonaxis,
n_unique_label_permutations_est, label_null_entropy, label_null_underpowered
# H3p-b flux (co-primary) — full + guard
net_offaxis_flux_slope_raw_{full,guard}, net_offaxis_flux_surplus_slope_{full,guard}, net_offaxis_flux_slope_resid,
net_offaxis_flux_slope_z_{full,guard}, p_label_slope_b_{full,guard}, p_rate_slope_b, p_spatial_slope_b,
lag1_specific_slope, common_drive_sensitive, proximal_flux_slope, spearman_rho_flux,
n_activation_events_pre, n_active_windows_pre, h3b_activation_sufficient,
leave_one_contact_flux_pass, axis_only_flux_control_pass, time_order_p_b,
near_onset_dependent_b, module_support_flag_b, module_direction_correct_b, module_null_pass_b
# H3p-c mode-shift (co-primary) — full + guard
mode_shift_density_slope_raw_{full,guard}, mode_shift_density_surplus_slope_{full,guard}, mode_shift_density_slope_resid,
mode_shift_density_slope_z_{full,guard}, p_label_slope_c_{full,guard}, p_phase_slope_c, p_block_slope_c,
mode_shift_2D_consistency_slope, mode_singular_gap_median, mode_vector_stable, cv_r2,
top_contact_energy_fraction, single_contact_driven, leave_one_contact_mode_pass, axis_only_mode_control_pass,
time_order_p_c, rank_used, k_star, spearman_rho_mode, h3c_support_grade(strong/weak/none),
near_onset_dependent_c, module_support_flag_c, module_direction_correct_c, module_null_pass_c
# H3p-a axial weakening (supportive-only)
K_primary_metric(=line_length_rate), beta_axis_strength_slope, beta_axis_reliable,
beta_axis_slope_z, p_label_slope_a, module_support_flag_a(=False)
# H3p-d secondary
nonaxis_activation_burden_slope_raw, nonaxis_activation_burden_slope_label_surplus, nonaxis_activation_burden_slope_resid, burden_slope_z, p_label_burden,
N_self_sustain_lag1_slope, N_self_sustain_lag0_slope, N_self_sustain_lag1_specific_slope, N_self_sustain_slope_z, p_label_selfsustain,
gain_axis_slope, gain_nonaxis_slope, gain_shift_slope, gain_nonaxis_surplus_slope, gain_shift_slope_z
# trend companions + verdict
trend_estimator(=theil_sen), slope_span(=full+guard), tier, state_v3p_supported, pre_registered_negative,
axis_weakening_supportive, nonaxis_flux_amplification_supported(H3p-b), mode_transition_supported(H3p-c)
```
window 明细 `v3p_window_detail.csv`：`subject, cohort, seizure_idx, span, phase, t_center, net_offaxis_flux_lag1, net_offaxis_flux_lag0, mode_shift_density, mode_singular_gap, nonaxis_activation_rate, global_energy, axial_energy, N_self_sustain_lag1, N_self_sustain_lag0, gain_axis, gain_nonaxis`。

---

## 10. cohort 实测 + 可扩性（2026-07-04）

**当前两队列干净**（V3a io，`config/topic5_v3.yaml` 锁值）：

| cohort | n | geometry_sufficient | preictal 充裕 | label-null 隐患 |
|---|---|---|---|---|
| broad | 9（139,253,1077,1096,1125,1150,620,635,916） | 9/9 | 每发作 17–18 窗，全 ≥8 | shaftBoth 3–9 健康 |
| narrow | 7（1096,1125,1146,253,384,442,958） | 7/7 | 每发作 17–18 窗，全 ≥8 | **1146=1、1096/1125=2 → underpowered 候选** |

**可扩候选**（cache 20 − roster 并集 13 = 7 个非 roster）：**6 个 broad-context 下 geom=OK + preictal 充裕** → `1084`（70 sz）、`583`（21）、`590`（11）、`922`（24）、`yuquan_xuxinyi`（3）、`yuquan_zhangkexuan`（3）；`548` 缺模板文件跑不了。
**[OPEN, 用户定]** 这 6 个**不在原 swap-curated 队列**（broad=swap-positive 8 + E916），其 axis 虽可构（ax≈20）但未经 swap/compact-core 策展 → 若扩须先过 **axis-quality 门**（如 narrow compact-core 构轴通过 + 稳定模板）。**是否扩、扩进 narrow 还 broad、axis-quality 门定义 = 待用户决策**（改的是预注册队列）。

---

## 11. 判读纪律 / 禁止 claim
- **禁 forecasting/prediction**：不出 lead-time、AUC、前向分类、"发作前 X 秒预警"。
- 不能说"没有临界性 / 发作前没有 state projection"；raw λmax 不当临界（只 `λ_surplus`）；rank-coupling 不当传播（只 flux）；**eigenvector 不当最大放大方向（用奇异向量）**。
- **H3p-a 单独不算 support**；只在 `beta_axis_reliable` 时可判。
- **回归残差 `*_slope_resid≈0` 不推翻 label-null 阳性**（collinearity 保守下界）。
- **label-null 是"专门集中在非轴向"唯一裁判**：obs_slope 大但 label-null 不显著 = "只是全局升温"，**不算 support**。
- **H3p-b：label 过但 rate/lag0 不过 → "非轴向负担/共激活上升"，非明确 axis→nonaxis flow**。
- **H3p-c：缺 phase 或 block surrogate → weak，不算硬门 support**；`mode_vector_stable=False` → 低置信不过度解释。
- **N_self_sustain：lag1-specific≤0 → "同步共激活"非"自持连锁"**。
- **gain：只看 `gain_nonaxis` 上升可能是全系统 gain 涨 → 主看 `gain_shift`**。
- **near_onset_dependent（只 full 过 guard 不过）→ 不算 preictal-only 强支持**。
- **label_null_underpowered → 不计强阳性分母**。
- 非轴向定义不得用发作/preictal criticality 结果（防循环）；`geometry_insufficient` / 短记录 ≠ negative。
- **预注册阴性可接受**（加固共享粗骨架）；**不救 1125**（个别被试只描述性 case）。
- EXPLORATORY 全程；tier 只在 summary。

## 12. 决定项 status
- **[LOCKED]** L1–L10 + rev1 L1b/L4b/L4c/L4d/L7b–e/L8b/L8c + preictal-only + Theil-Sen + co-primary(flux+mode) + label-null 主裁 + onset guard 双轨 + 继承 V3a + read-only。
- **[OPEN, pilot 定]** `min_windows_for_slope`（实测非约束）；奇异 gap / label-perm underpowered / 激活充分 阈值。
- **[OPEN, 用户定]** cohort 是否扩（§10：6 候选，需 axis-quality 门）。
- **[本 spec 外]** 模型侧 preictal 一致性；表达层。

## 13. 核心命题（一句话）
> **V3p 只在发作前干净时段（`full=[−120,−10]` headline + `guard=[−120,−20]` jitter-safe 双轨，eeg-onset 锚、不碰 ictal）问：非轴向连锁流（`net_offaxis_flux`，H3p-b）与最易放大方向（`mode_shift_density`，H3p-c）是否随发作临近**逐渐上升**（Theil-Sen 斜率）、且**专门集中在非轴向触点**——**同杆 label-null-of-slope** 主裁（控全局升温）+ 回归残差保守 sensitivity；H3p-b 加 rate 逐窗 + lag0 硬门（真流非同步爆），H3p-c 加 phase+block 硬门（真模态非平滑假象），两轨同过 + label-null 非 underpowered 才算强支持。co-primary=H3p-b+H3p-c（Holm）；H3p-a supportive-only；H3p-d 负担/自持(lag1-specific)/相对增益(gain_shift) secondary。被试为单位、narrow 主 broad 复制、tier 0–5、`state_v3p_supported=tier≥3`、最高 tier 4。全程 EXPLORATORY、无 forecasting、预注册阴性可接受不救 1125。cohort 实测 broad 9/9 + narrow 7/7 干净、6 候选可扩（待用户定 axis-quality 门）。**
