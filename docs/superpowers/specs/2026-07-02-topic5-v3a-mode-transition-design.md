# Topic 5 V3a 设计 — 发作是否从"沿间期 HFO 轴"转入"非轴向不稳定模态"（data-side mode transition）

date 2026-07-02 · 状态：design（待 writing-plans）· 前身 = V2a（`docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md`，restricted axial sanity check）· 姊妹 spec = V3b（M3B 模型–数据一致性，另写）

> **一句话缘起**：V2a 问"发作前临界性是否压在间期 HFO 小路上"，在受限实现下没支持。V3a 把问题**从"是否沿轴"改成"是否从轴向搬到非轴向"**——阴性不再是"轴上没东西"，真正的阳性是"轴向组织减弱 + 非轴向活动/流/模态放大"。这是 07-02 模型–数据 criticality 纲领的**数据腿(D)**；模型侧对接(H3d)在 V3b。

---

## 0. 摘要（朴素话）

病人在两次发作之间，那些短暂高频异常放电按一条固定先后顺序在电极间传开——像一条走熟的小路（代号 `G_HFO`）。V2a 假设：真发作前系统"变脆"的迹象也会压在这条小路上。结果：在只看发作前两分钟、只用两段窗、只在小路匹配触点上算、没有显式非轴向假设的受限实现下，**没看到**；而且证明了当时"系统接近失稳"的读数（谱半径≈0.95）其实是宽带能量本身平滑造成的假象。

V3a 换一个更贴模型的问题：**发作真正启动时，系统会不会先把"沿小路的有序传播"松掉，转而在小路**之外**的电极上、沿小路之外的方向变得越来越容易被点亮/放大？** 我们做三件事：(1) 把"变脆地图"跟小路做**有符号**对齐，看它随时间是不是**从有序变涣散**；(2) 用一个局部线性系统看"最容易被放大的方向"是**落在小路上还是小路外**，并且——因为模型早告诉我们轴向机制是个"非正规瞬态"而不是一个特征值——**不只看谱半径，还看瞬时放大率**；(3) 把连锁激活拆成"轴内 / 轴外"两个区，看**从轴向漏到非轴向的净流量**是不是随发作启动上升。全部跟"打乱后的随机版本"比，被试为单位，broad/narrow 不混。**narrow 为主队列**（更聚焦的轴，轴/非轴对比更干净），broad 作复制。

（内部代号：`G_HFO`=间期 HFO typical_rank 顺序场；signed axis β_axis；非轴向=纯间期 HFO 参与度定义的 off-template clean 触点；2D 投影算子 = 把全系统投到 [e_axis,e_nonaxis] 平面；reactivity = numerical abscissa / 有限时窗放大；compartment flux = A→N/N→A ATM 质量；λ_surplus = 观测谱半径 − surrogate 中位数。）

---

## 1. 与 V2a 的关系 + honest coupling + 与 07-02 纲领的关系

- **V2a 降级不删**：V2a 冻结为 restricted-axial baseline，其工程产物变成 V3a 的 **axis-only 回归对照臂**（证明新信号来自非轴向而非实现变化）。V2a 那条**方法学定律保留并升级为 V3a 全程约束**：所有 λmax/VAR/DMD/Jacobian 一律报 `λ_surplus`（观测 − surrogate 中位数），**不报 raw**。
- **V3a 是数据侧**：只用真数据检验 H3a/b/c（轴向减弱 / 非轴向放大 / 模态转移）。**模型侧 H3d（M3B Jacobian eigenmode 与数据 VAR/DMD 模态、非正规瞬态的一致性）拆到 V3b**，定 exploratory confirmatory（模型是均质 L=20 衬底，其"非轴向模态"在 1D 链上，和真 SEEG 几何的桥是脆的，不拖累数据结论）。
- **honest coupling**：V3a 单独也不能宣称"发作机制 = 非轴向临界模态"。V3a 阳性 = "数据里存在 axial→non-axial 重组"；要升级成机制主张，需 V3b 的模型–数据一致性 + Phase 1 能量表达层。**无 forecasting**（发作前记录 ≤ 几百秒）。

---

## 2. 主假设 + 数据侧子假设

**H3（主）**：从 late preictal 到 ictal early/mid，系统"最容易被放大的方向"从 HFO-axis mode 转向 non-axis mode。

| 子假设 | 朴素话 | primary 指标（每假设只钉一个） | 方向预期 |
|---|---|---|---|
| **H3a 轴向减弱** | 原来沿小路的有序变化变弱 | `β_axis_strength` 的 P3→O/I1 变化（susceptibility 侧）+ `axis_forward_flow` 变化（avalanche 侧，co-descriptive） | 下降 |
| **H3b 非轴向放大** | 小路外越来越容易被点亮/放大 | `net_offaxis_flux` 的 P3→O/I1 变化（avalanche，**H3b primary**） | 上升 |
| **H3c 模态转移** | 主导可放大方向从轴向转非轴向 | `mode_projection_nonaxis − mode_projection_axis` 的 P3→O/I1 变化（dynamics 2D 算子，**H3c primary**） | 上升 |
| H3d 模型–数据 | 模型预测的可放大非轴向模态在数据里也更被放大 | → **V3b**（subspace angle / gain 相关） | — |

其余指标（variance/AR1/skewness 时间慢化、λ_surplus、reactivity、direction_index、leak_index…）全部 secondary/descriptive。**时间慢化指标只作 secondary**——文献（Wilkat-Lehnertz negative、Milanowski-Suffczynski "无共同签名"）对发作前临界慢化本身有争议；V3a 的主张是**空间重组**而非时间慢化，主指标必须是空间的。

---

## 3. 已锁决定（本 session 与用户对齐，可回改）

- **[LOCKED D1]** 拆两个 spec：本 = V3a（数据侧 H3a/b/c）；V3b = M3B 模型–数据一致性（H3d）。
- **[LOCKED D2]** 动力学主算子 = **2D 投影算子 + reactivity/非正规**（well-posed、对齐假设）；"多做几个"= 并行低秩 DMD/SVD-VAR（支撑）+ axis-only VAR（V2a 对照）；**full all-contact ridge-VAR 只作高维 sensitivity**（实测 all-clean=60–124 触点，full VAR 病态）。**narrow = 主队列**，broad 复制。
- **[LOCKED D3]** 非轴向 primary = **纯间期 HFO 参与度定义**（clean cache 触点 − 间期 HFO 模板触点，QC-good；对 ictal 结果全盲 → 零循环风险；复用既有 `load_subject_propagation_events` 参与度 + 模板 `typical_rank` + `axis_partition`）。M3B 模型方向 → V3b confirmatory；Phase-1 expression residual（严格 leave-one-seizure-out）→ sensitivity。
- **[LOCKED D4]** 线性动力学指标（β_axis、2D gain、λ_surplus、reactivity、mode projection）**在完整 time-grid 上都算**（描述性轨迹，看它怎么随发作演化）；**但承重统计检验锚在 P3→O/I1 转变**（线性假设在此可辩护）；**中晚 ictal（I2/I3）线性值只作描述**（饱和态线性假设弱化），confirmatory 由 avalanche/flux 承担。
- **[LOCKED]** V3a **自建 null**（shaft-constrained spatial / rate-preserving order / axis-nonaxis label），**不再依赖 stalled Phase-1** 的 `spatial_constrained_permute`/`order_null_rank_pair`。

---

## 4. 时间设计（onset/offset 锚定 + regime-metric matching）

**数据现实（已核对 cache）**：每次发作有 `eeg_onset_rel`、`eeg_offset_rel`、`eeg_duration_sec`；relt 覆盖到 ictal + postictal（如 [-152,169]、[-160,260]）。**关键坑**：cache `relt=0` 不是电生理 onset，`eeg_onset_rel`≠0（如 139=−3.75s）——V2a 的"relt<0=preictal"晚窗混进了几秒 early-ictal。**V3a 每个窗按每次发作各自的 `eeg_onset_rel`/`eeg_offset_rel` 锚定，不用 relt=0。**

**Event table（每次发作一行）**：`subject, seizure, t_eeg_onset, t_eeg_offset, eeg_duration, usable_pre_sec, usable_ictal_sec, usable_post_sec, offset_quality`。（clinical onset：cache 无——**onset sensitivity 只能用 eeg_onset + jitter**，除非从 SQL 元数据补；spec 假定只有 eeg_onset。）

**滑窗**：10s 窗 / 5s 步，覆盖 `[eeg_onset−120s, eeg_offset+60s]`（无可靠 offset 的发作不进 offset 分析，但仍进 onset/peri-ictal）。

**Phase bins（按 eeg_onset 锚定；O 窗单列作 onset 不确定缓冲）**：

| phase | 定义（相对 eeg_onset/offset） | 用途 |
|---|---|---|
| P0 | −120～−90 | 早期参考（**不叫 baseline**） |
| P1 | −90～−60 | preictal early |
| P2 | −60～−30 | preictal middle |
| P3 | −30～−10 | late preictal（不直接贴 onset） |
| **O** | −10～+10 | peri-onset 不确定缓冲，**单独报告，不并入 pre/ictal slope** |
| I1 | +10～+30 或 ictal 0–25% | early ictal（**线性指标主检验落点**） |
| I2 | ictal 25–75% | ictal maintenance（线性只描述） |
| I3 | offset−30～offset 或 75–100% | pre-offset termination（线性只描述；flux 承重） |
| Post | offset～+60 | postictal recovery |

**regime-metric matching（承 D4）**：线性算子（VAR/2D-gain/λ_surplus/reactivity/mode-projection）在 O/I1 承重、I2/I3 描述；flux/avalanche 对非线性稳健，承 I2/I3 + pre-offset 终止转变（Kramer termination 那条线）。

**onset jitter 验收**：每个主结论在 `eeg_onset` 与 `onset ± {5,10,15}s` 平移下重算；**±10s 内方向不变**为通过，±15s 作压力测试。

---

## 5. 空间设计（signed axis + 非轴向）

**5.1 有符号轴 β_axis（取代 V2a 反向的 K_signed_oriented 约定）**
- `rank_forward_i` = G_HFO 小路 early→late，缩放到 −1～+1（early=−1，late=+1）。
- `β_axis(metric, t) = Spearman(metric_i(t), rank_forward_i)`（在 axis 触点上）。约定固定：`β>0` 指标偏 late 端、`β<0` 偏 early 端、`|β|` = 轴向组织强度、`β` 随时间变 = 轴向方向是否转移/减弱。旧 `K_signed_oriented` 保留为兼容列。

**5.2 非轴向定义（primary = 纯间期 HFO，防循环）**
- **axis 触点**：进入间期 HFO 传播模板者（有限 `typical_rank` / 高 HFO 参与度；即 `load_context` 的 matched/mapped ∪ `axis_partition` 的 source_core/axial_mid/axis_end）。
- **non-axis strict（primary）**：cache 里 QC-good 的 clean 触点中，**不在** HFO 模板、且间期 HFO 参与度低于阈的触点。**只用间期信息，对 ictal 结果全盲。** 输出 `is_axis / is_nonaxis_strict / hfo_participation / n_axis / n_nonaxis`。
- **confirmatory / sensitivity**（非 primary）：M3B 模型方向（V3b）；Phase-1 expression residual 方向（严格 LOSO，防循环）；`axis_partition` 的几何 non_axial（作 within-matched 对照）。
- **residual 非轴向方向 e_nonaxis**（给 2D 算子用）：在 all-clean 触点上，先构 `e_axis`（axis 触点按 rank_forward 加权、非 axis 为 0），再取一个与 `e_axis` 正交的非轴向单位向量（来源：间期 HFO 参与度地形的次主成分 / 或 V3b 的模型方向）；对 `e_axis` 施 Gram-Schmidt 保证正交。

**5.3 防循环铁律**：非轴向集/方向的定义**不得**用同一批发作的 criticality 结果；主定义纯间期，其余用 LOSO 或模型先验。

---

## 6. 三条腿重设计

### 6.1 susceptibility K_t（time-grid + 有符号 + 非轴向对比）
每 contact 每窗算 variance / lag-1 AR / **line-length rate（注：bb envelope 上是 envelope roughness，非 raw-EEG line length，主文档须注明）** / skewness(补) / spatial-corr(网络级补)。
主指标：
- `K_axis_strength(t)=|β_axis|`（H3a：P3→O/I1 预期**降**）；
- `K_nonaxis_contrast(t)=median(metric_nonaxis)−median(metric_axis)`（P3→O/I1 预期**升**）；
- `K_nonaxis_projection(t)=corr(metric_i, e_nonaxis_i)`（升）；`K_shift_index=nonaxis_projection−axis_strength`（升）。
null：shaft-constrained spatial + order（打乱 rank_forward 保参与度）+ axis/non-axis label（匹配 shaft/触点数/baseline variance 交换标签）。

### 6.2 dynamics（2D 投影算子 + 非正规 reactivity；narrow 主）
数据矩阵 = all-clean contacts × time（每 10–20s 窗）。三版本：**2D 投影算子（主）** / 低秩 DMD/SVD-VAR（支撑）/ axis-only VAR（V2a 对照）。full ridge-VAR 只作高维 sensitivity。
2D：`Q=[e_axis,e_nonaxis]`，`B_w=QᵀA_w Q`。报告（全 grid 描述，P3→O/I1 承重）：
- `gain_axis=|A e_axis|`、`gain_nonaxis=|A e_nonaxis|`（H3? 描述）；
- `cross_axis_to_nonaxis=e_nonaxisᵀA e_axis`（轴向扰动漏到非轴向，预期升）；
- `mode_projection_axis / _nonaxis`（leading mode 对 e_axis/e_nonaxis 投影；**H3c primary = nonaxis−axis 的 P3→O/I1 升**）；
- **λ_surplus**（谱半径 − phase/block surrogate 中位数，永不报 raw）；
- **reactivity**（非正规：`max eig((A+Aᵀ)/2)` 或 numerical abscissa）+ **有限时窗放大**（`max_k ||A^k e||/||e||`，沿 axis/nonaxis 分别）——因 M3B-R2 已证轴向机制是非正规瞬态而非特征值，只看 λ 会漏。
- **eigenvector 符号任意** → 不用 leading-eigvec 符号解释方向；只用"落在 axis 还是 nonaxis"（投影）+"扰动放大到哪"（operator response）。
null：phase + block surrogate（同 V2a，已建，扩到 2D 算子与 reactivity）。

### 6.3 avalanche compartment flux（非轴向流；rate-preserving null 是命门）
compartment：`A_early / A_late / N`（N=非轴向 strict）。每窗建 ATM `ATM[i,j]=P(j@t+δ | i@t)`。
主指标：`axis_forward_flow`(A 内 early→late，H3a 描述，预期降)、**`net_offaxis_flux=fluxA→N − fluxN→A`（H3b primary，P3→O/I1 升）**、`N_self_sustain`(N→N)、`branching_N`、`leak_index=A→N / A 所有出流`。
null（4 类）：time-block shuffle / **rate-preserving shuffle（保每 contact 激活率打乱目标——命门：否则"非轴向 flux 高"只因非轴向触点多）** / shaft-constrained spatial / axis-nonaxis label。
**验收不需要 axis_forward_flow>0**；真支持 = `net_offaxis_flux` 在 P3→O/I1(/I2) 升且超 rate-preserving+spatial+label null。

---

## 7. 统计骨架

- **subject 为单位**：窗→发作→被试→队列，逐层中位数；不把窗/发作当独立样本。**narrow 主、broad 复制，永不 pool。**
- **每假设一个 primary 指标**（§2 表），其余 exploratory，控 garden-of-forking-paths。
- **null 全部 V3a 自建**（不依赖 stalled Phase-1）；经验 p = (1+越界数)/(1+n_perm)；对齐类双侧、方向/趋势类单侧。
- **两层验收门**：
  - subject-level 支持 = `net_offaxis_flux` 或 `K_nonaxis_contrast` 或 `mode_projection(nonaxis−axis)` 至少一个 primary 显著 + 方向符合(P3→O/I1 增) + 相应 null 过 + onset ±10s jitter 方向稳 + 非单一 contact 驱动 + axis-only 不能单独作支持。
  - cohort-level = 分 broad/narrow 报 subject-median effect + sign test / Wilcoxon + bootstrap CI + 显著 subject 数；不把 seizure/window 当独立样本。
- **`state_v3_supported`**（取代旧 `state_leg_supported`）= 上述 subject 门 + cohort 复制 + jitter 稳健。

---

## 8. 输出 schema（每 subject 一行；phase-resolved 明细另表）
`subject, cohort, n_seizures_total, n_seizures_used_{pre,ictal,offset}, n_contacts_all_clean, n_axis, n_nonaxis, onset_anchor, onset_jitter_pass,`
`K_axis_strength_{P3,O,I1}, K_nonaxis_contrast_{P3,O,I1}, K_nonaxis_delta_I1_P3, K_nonaxis_p_{spatial,order,label},`
`lambda_surplus_{P3,I1}, gain_axis_delta, gain_nonaxis_delta, cross_a2n_delta, reactivity_{P3,I1}, finite_time_amp_{axis,nonaxis}, mode_proj_axis_delta, mode_proj_nonaxis_delta, p_{phase,block},`
`axis_forward_delta, net_offaxis_flux_delta, leak_index_delta, branching_N_delta, p_{rate,spatial,label},`
`axis_weakening_supported, nonaxis_amplification_supported, mode_transition_supported, avalanche_offaxis_supported, state_v3_supported, tier`

---

## 9. 脚本 + 测试结构（复用 V2 io/surrogate/ATM primitives；新建时间/几何/null）
```
scripts/_topic5_v3_event_windows.py         # eeg_onset/offset 锚定 phase grid + jitter
scripts/_topic5_v3_geometry_axis_nonaxis.py # signed β_axis + 纯间期 HFO 非轴向 + e_axis/e_nonaxis
scripts/_topic5_v3_surrogates.py            # shaft-spatial / rate-preserving / label null（自建）
scripts/_topic5_v3_dynamics_utils.py        # 2D 投影算子 + reactivity + 低秩 DMD
scripts/run_topic5_v3_susceptibility_timegrid.py
scripts/run_topic5_v3_dynamics_2d.py        # narrow 主；2D 主 + DMD 支撑 + axis-only 对照
scripts/run_topic5_v3_avalanche_offaxis.py
scripts/run_topic5_v3_summary.py
scripts/plot_topic5_v3_{timecourses,mode_transition,avalanche_flux}.py
tests/ test_v3_event_windows_onset_offset / _signed_axis_convention / _nonaxis_no_circularity /
       _2d_projection_known_matrix / _reactivity_nonnormal / _surrogate_rate_preserving /
       _avalanche_compartment_flux / _subject_level_aggregation
```

## 10. 判读纪律 / 禁止 claim（承 V2a §4.4）
不能说"没有临界性"/"发作前没有 state projection"；不能把 raw λmax≈0.95 当临界证据（只 λ_surplus）；不能把 avalanche rank-coupling 当传播证据（只 forward/flux）；不能把 −120～−30 差分当 onset-proximal transition（按 eeg_onset 锚）；不能把 axis-only 结果当全脑模态；不能用 eigenvector 符号解释方向；不能把中晚 ictal 的线性 λ 当可靠临界读数（饱和态描述）；非轴向定义**不得**用发作结果（防循环）。

## 11. 决定项 status
- **[LOCKED]** D1–D4（§3）+ V3a 自建 null + narrow 主队列 + eeg_onset 锚定 + λ_surplus-only + 每假设一 primary。
- **[OPEN]** 非轴向 HFO 参与度阈值具体值（pilot 定）；I1 用固定 [+10,+30] 还是 ictal 0–25%（按 duration 分布定）；clinical onset 是否从 SQL 补做第二锚。
- **[V3b]** H3d 模型–数据 eigenmode/非正规瞬态一致性（另 spec，exploratory confirmatory）。

## 12. 核心命题（一句话）
> **间期 HFO 给出一条稳定的轴向传播几何；发作启动可能不是"这条轴更强对齐"，而是"轴向组织减弱、同时非轴向触点/流/可放大模态增强"。V3a 用有符号轴向强度、2D 轴/非轴投影算子（含非正规 reactivity）、轴内→非轴向 compartment flux 三条数据侧证据，在 eeg-onset 锚定的 peri-onset→early-ictal 转变上、经自建空间/顺序/label/rate-preserving null 与 onset-jitter 检验，看系统是否发生 axial→non-axial 模态转移。narrow 主、broad 复制；模型侧一致性(H3d)在 V3b。**
