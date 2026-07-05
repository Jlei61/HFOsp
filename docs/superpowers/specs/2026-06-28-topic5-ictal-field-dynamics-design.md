# Topic 5 — 发作内 field 动力学 pilot (ictal field dynamics) — design

> 状态: design (2026-06-28)。subject-level exploratory PILOT,**非 cohort claim**。
> 上游: 发作早期 field 投影分析 (`scripts/plot_topic5_field_vs_ictal_swap.py` +
> `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`)。

## 0. 朴素话定位(测什么 / 怎么测 / 揭示什么)

**测什么** — 一次发作从开始(onset)到结束(offset),电活动的"空间形状"怎么随时间变。把
触点分三块分开看:① 两个假设病灶(间期传播每个模板**最早**的触点 = source 焦点)本身;
② 这两个病灶**"之间"的走廊**(间期波平时沿它传播的中段路径);③ 离走廊远的**横向/离轴**触点。

**怎么测** — 每隔几秒取一个时间窗,算每个触点相对发作前 baseline 的 robust-z(和发作早期
分析同一口径);按上面三块分组,比较各块的相对活动、整体同步度、以及活动场的方向是否漂移;
再额外按 offset 对齐看终止前后。

**揭示什么(期待但不预设)** — 期待"走廊(两病灶之间)的活动相对全场变弱 + 离轴活动相对变强
+ 整体更同步 + 场方向可能漂移";但 pilot **只描述这些量随发作进程怎么走**,不下机制/cohort
结论,不预设终止动力学,不把任何单 subject 现象写成规律。

**为什么 source 不作减弱检验** — source 是着火点(假设病灶),预计一次发作里**一直亮**;测它
"减弱"没意义。所以 source 只作参照(确认它确实恒高),真正的检验对象是"两病灶之间的中段走廊"。

(内部代号: maxAB sign-free alignment, rank-displacement decision_k / rank_a/b_dense_full,
baseline robust-z, eligibility analysis_eligible, narrow substrate。)

## 1. 输入与口径(全部沿用现有,不新发明)

### 1.1 高质量发作 (eligibility)
- 主表: `results/topic5_ictal_recruitment/t0_eligibility_audit.csv`,取 `analysis_eligible == True`。
- **附加门槛**: `has_complete_eeg_interval == True`(offset 已知,长窗 + 终止对齐都依赖它)。
- onset/offset 时间: `results/epilepsiae_seizure_inventory.csv` 的 `eeg_onset_epoch` / `eeg_offset_epoch`
  与 `eeg_duration_sec`。**MVP 用 EEG(电生理) offset**;clinical offset (`clin_offset_epoch`)
  仅作未来 sensitivity,不进 MVP。
- 6 个 subject 的 `analysis_eligible` 计数(供规模参考): 442:22, 548:26, 583:22, 384:12,
  958:12, 1084:72(其中再被 `has_complete_eeg_interval` 收一遍)。

### 1.2 baseline robust-z(与发作早期**完全一致**,不改)
- 沿用现有自适应口径(不是固定 [-90,-60]): per-channel `z = (trace - median(base)) / (1.4826*MAD(base))`,
  baseline window = `[-pre, -60s]`(pre 自适应 ≥120s),guard `[-60,0]s`。
- 复用 `src.ictal_onset_extraction.extract_seizure_window` + `src.topic5_ictal_recruitment.baseline_robust_z`
  + `band_power_trace`,hop = 0.1s,reference = "car"。
- band: **broadband 1–45 Hz 为 primary**(与 onset 图一致);HFA 60–100 Hz 同时存盘作 secondary,
  MVP 不主分析。

### 1.3 间期模板轴 / source 焦点 (narrow substrate)
- rank-displacement: `results/interictal_propagation_masked/rank_displacement/per_subject/<ds>.json`,
  取 `primary_pair`(回退 `pairs[0]`) 的 `rank_a_dense_full` / `rank_b_dense_full` / `joint_valid`
  / `channel_names` / `swap_sweep.decision_k`。
- 几何: `results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/<ds>_t_a.json`
  与 `<ds>_t_b.json`(提供 frame + per-contact 坐标)。
- **不用旧 PR-2 source 字段;不 gate `swap_class`**(本分析用 endpoint 层而非 swap 角色互换层,
  6/6 subject 全可用,其中 4 个 swap_class="none" 不受影响 —— 见 CLAUDE.md §6.2 分辨率层级)。

### 1.4 subjects (pilot, 6 个几何干净 ECoG)
`epilepsiae_{442, 548, 583, 384, 958, 1084}`(来源 `docs/archive/topic5/ictal_direction_clustering_2026-06-27.md`:
electrode_type==ECoG ∧ coord_aspect≥0.15)。其中:
- **442** = 发作早期双方向簇但未对上间期 A/B 的个案(Δ_AB=147°,轴可解释)。
- **1084** = 退化轴**负控**(间期两模板方向 Δ_AB=6°,几乎重合)。
- 384(swap candidate)/958(swap strict)/548/583 余四个。

### 1.5 substrate 决策
**narrow**(非 broad)。理由: 这 6 个 subject 的几何 + rank-displacement 在 narrow 齐全(6/6);
broad 仅 384/583 有几何且均 swap="none"。上游那张 swap 环图是 broad、是另一批 subject。
长窗 cache 本身与 substrate 无关(它是发作期录波,按通道名匹配几何)。

## 2. 长窗 ictal field cache(新建,parallel dir,复用现有函数)

现有 onset cache 已存完整 robust-z 时间序列(`bb_zt` + `bb_relt`),但只到 onset+20s —— 对
80–113s 的发作远不到 offset。故新建长窗 cache。

- thin builder(新脚本,**复用** `extract_seizure_window` / `baseline_robust_z` / `band_power_trace`),
  输出 `results/topic5_ictal_recruitment/ictal_field_long_cache/<ds>_<sid>.npz` + `<ds>_<sid>.json`(meta)。
- 每次 eligible(且 complete-eeg-interval)发作: `pre_sec = 130`,`span = max(eeg_offset_rel, eeg_duration_sec)`,
  `post_sec = ceil(span) + 90`。**P1: eeg_onset 可能晚于 clin_onset(如 384 `eeg_onset_rel≈+36s`),
  只用 duration 会少抽到 offset** —— 用 span(相对 clin onset 的 offset)保证覆盖 eeg offset+60。
  `span > 600s`(`MAX_ICTAL_SEC`,疑似 status epilepticus)→ drop `duration_too_long_for_pilot`(亦防 OOM)。
- 存盘(每发作): per-contact full robust-z trace `bb_zt` + secondary `hfa_zt`、对应相对 clin onset 的
  bin 时间 `bb_relt`/`hfa_relt`、`bb_auc`/`hfa_auc`([0,10]s parity 用)、channel names、`fs`、
  `eeg_onset_rel`/`eeg_offset_rel`、`pre_sec`/`post_sec`;meta 写 `baseline={guard_sec:60}`(baseline=[-pre_sec,-60] adaptive)。
- 失败/缺 offset/越 block/baseline 不足/too-long → 写 drop reason,跳过该发作。
- **baseline parity gate(P1c,强制)**: builder 完成后,对每发作把长窗 cache 的 `[0,10]s` window 值
  (与现有 cache 同一 reduction)与现有 `results/topic5_ictal_recruitment/t0_feature_cache_v2_windows`
  同发作 `[0,10]s` bb(及 hfa)值比对;`max|Δ|` 超容差(`<1e-3` 绝对)→ 该发作 `parity_fail`,**不进指标**。
  即用数值**证明**"baseline 完全一致",而非口头声称。
- builder 不重画图、不动现有 `t0_feature_cache*`(parallel dir,旧目录不删 —— AGENTS.md results 规范)。

## 3. 触点四分区(source 线段投影,每 subject 算一次,间期态)

1. **匹配**: cache channel names ∩ 几何 contacts(按名),得每触点 frame 坐标 (x, y)(复用
   `scripts.plot_topic5_swap_nodes_fields._arrays` / `plot_contact_plane_static._display_points`,
   x = 间期传播轴方向 mm, y = 横向 mm)。仅保留 finite 坐标且在 cache 中有 trace 的触点为 "mapped"。
   **三个来源(cache / rank-displacement / 几何 record)一律按通道名匹配,不靠 index**
   (AGENTS.md channel_names ordering 警告;raw 顺序可能不同)。
2. **source core 焦点(compact,不依赖 decision_k)** — 每侧(A/B)独立:
   - 候选 = 模板 A `rank_a_dense_full`(B 用 `rank_b_dense_full`)在 `joint_valid` 且**已 mapped** 子集内
     按 rank 升序(最早在前)的触点名(`derive_swap_endpoint` 的 source-侧排序逻辑)。
   - 取最早 2 个;若二者 frame 间距 `< 15 mm` → `source_core` = 这 2 个,centroid = 均值。
   - 否则 `source_focus_uncertain[side] = True` → `source_core` = **仅最早单个 mapped 触点**,centroid = 该点;
     top2/top3 仅作可视化候选,不进 centroid。
   - `decision_k` **只作 rank-displacement provenance 记录,不参与 source core 大小**(P0:dk 是 swap 尺度非小灶尺度)。
3. **轴**: `P_A` = source-A core centroid,`P_B` = source-B core centroid。轴向量 `u = P_B - P_A`,轴长 `L = |u|`。
4. **退化检测**: `bbox_diag` = mapped 触点 bounding-box 对角线长。若 `L < 0.15 * bbox_diag` →
   `axis_degenerate = True`,该 subject 分区不可信(仅出诊断,不进主结论)。**1084 预计落此 = 负控自证**。
5. **投影**: 每 mapped 触点 c: 沿轴位置 `t_c = ((c - P_A)·u) / L²`(归一,0≈P_A,1≈P_B);
   垂距 `d_c =` 点到直线 P_A–P_B 的垂直距离。`med_d = median(d over 非 source_core mapped)`。
6. **MECE 四分区(P1:source 与端区分开,sanity 只看 source_core)**:
   - **source_core**(参照/sanity) = 步骤 2 的两侧 core 触点(预期恒高;主 sanity = `source_core_minus_all`)。
   - **axis_end_noncore**(辅助) = 非 source_core 且 `d_c ≤ med_d` 且 `t_c∉[0.25,0.75]`(贴轴但靠端)。
   - **axial_mid**(检验对象) = 非 source_core 且 `d_c ≤ med_d` 且 `t_c∈[0.25,0.75]`(两焦点之间、贴轴走廊)。
   - **non_axial**(对照) = 非 source_core 且 `d_c > med_d`(离轴线远)。
   - 四组互斥且覆盖全部 mapped 触点;各组触点数记入输出。

### 3.1a 实测 source 散布(支持 P0 compact gate;narrow,frame mm)

| subject | swap | dk | bbox_diag | srcA top2 | srcB top2 | 结论 |
|---|---|---|---|---|---|---|
| 442 | none | 7 | 68.7 | 1.7 | **32.3** | A 双点 compact / B 单点 |
| 548 | none | 6 | 64.1 | (1 mapped) | (1 mapped) | 两侧单点(几何 mapping gap) |
| 583 | none | 2 | 60.0 | 13.0 | 9.7 | 两侧双点 compact |
| 384 | cand | 2 | 63.0 | **30.6** | **42.6** | 两侧单点 |
| 958 | strict | 3 | 73.1 | **28.7** | **45.8** | 两侧单点 |
| 1084 | none | 4 | 43.1 | 15.0 | 15.0 | 边界→单点(退化轴负控) |

单位 mm;粗体 = 超 15mm compact 阈 → 单点 fallback。两个 swap-positive 主角(384/958)两侧都散 →
若仍用 centroid(更别说 decision_k 尺度)轴就是几何 artifact;故 compact gate + 单点 fallback 必需。

## 4. 指标(每个 window 一行,4 个 family)

window 内每 mapped 触点有: window-mean robust-z(标量)与 window 内 robust-z trace(时间序列,
从 `{band}_zt` 按 `{band}_relt` 切片)。**两个 band 都分析**(P1-4):`bb`=primary、`hfa`=secondary,
每窗 × 每 band 出一行(`band` 列);图默认 bb,HFA 可选。每窗另记 `ictal_fraction`(窗内 bin 落在
`[0, eeg_offset_rel]` 的比例)与 `post_offset_overlap`。

- **A. field_axis_alignment** — window 活动场 vs 间期 A/B 场的 sign-free maxAB 对齐。
  per-contact window-mean z → 在 subject 内 rank → 建 field record → `R_smooth_rank`(81×81 归一平面)
  → `corr_pair_mirror_invariant` 对 `F_inter_A` 与 `F_inter_B` 各算,取 `max(|corr_A|, |corr_B|)`。
  口径与 `scripts/run_topic5_axis_alignment.py`(`statistic="max_ab"`) 一致,仅把活动向量从
  "mean 0–10s" 换成 "mean over window"。常量 `S_THRESH=0.15`, `OVERLAP_MIN=25`。
  **interictal 场 = 几何 record 的 `typical_rank`(t_a/t_b),与 run_topic5_axis_alignment 同源**。
  注意分辨率层区别(CLAUDE.md §6.2): §3.2 的 source core 用 rank-displacement `rank_*_dense_full`,
  本处 alignment 场用几何 `typical_rank` —— 二者都是模板 A/B 传播序但层不同,各按其既有口径用,不混。
- **B. field_direction_drift** — window 活动在 frame (x,y) 上的梯度方向漂移。
  对 mapped 触点(finite z)做 **uniform OLS** 线性拟合 `z ~ a*x + b*y` → 梯度 `(a,b)`,`grad_angle = atan2(b,a)`,
  `grad_mag = hypot(a,b)`。报: `drift_vs_onset`(本窗 grad_angle 与第一个 [0,10] 窗 grad_angle 的角距,
  fold 到 [0,90])、`angle_to_interictal_axis`(grad_angle 与 frame x 轴=0 的角距,fold [0,90])、`grad_mag`
  (梯度量级,场变平/同步时变小)。
- **C. 四分区相对活动** — 对 source_core / axis_end_noncore / axial_mid / non_axial 各算:
  `mean_z`、`pos_share`(z>0 占比)、`p95_z`(non_axial 尤其关注)、
  `positive_mass_share = Σ max(z,0)_group / Σ max(z,0)_all-mapped`(有界 [0,1],四组和=1)。
  **用 positive_mass_share 替代 mean/mean 比值**(Extra1:robust-z 均值可能近 0/为负 → 比值会爆掉或反号)。派生:
  - 主假设读数: `axial_mid` 的 `positive_mass_share` 随进程是否**下降**、`non_axial` 是否**上升**。
  - `axialmid_minus_nonaxial = mean_z(axial_mid) - mean_z(non_axial)`(差值,正=中段更亮;保留)。
  - sanity: `source_core_minus_all = mean_z(source_core) - mean_z(all-mapped)`(应恒为正、恒高;**只用 source_core,不含端区**)。
- **D. field_synchrony** — window 内各 mapped 触点 robust-z trace 两两 Pearson 的 **median pairwise corr**;
  `participation = (window-mean z > 2 的 mapped 触点占比)`(同步/泛化招募读数)。

## 5. 时间窗

- **onset-aligned 滑窗**: `[0,10],[5,15],...`(长 10s, step 5s),直到窗右端 ≥ eeg offset。每窗记
  `t_center_rel_onset` + `ictal_fraction` + `post_offset_overlap`。**P1-2: 短发作(如 1084 中位~4.4s)
  的 [0,10] 窗大半在 offset 之后** → summary/plot 默认只用 `ictal_fraction ≥ 0.5` 的 onset 窗作 ictal 轨迹;
  低于的行保留作诊断,不当 ictal trajectory。
- **归一化进程**: 每发作把 onset→offset 线性压到 `progress∈[0,1]`,每窗记 `progress_frac`;画图时按
  `0/25/50/75/100%` 分箱(取最近窗)对齐不同 duration。
- **offset-aligned 终止窗**: 按 eeg offset 对齐,固定 `[-60,-30],[-30,-10],[-10,0],[0,30]s`(相对 offset),
  各算同样 4 family 指标。需 trace 覆盖该窗(post_sec 已保证 offset+60;onset 前覆盖 130s)。
  **边界(Extra2)**: 若某终止窗左缘早于 onset(短发作,如 dur<60s 时 `[-60,-30]` 落到 onset 前)→
  标 `pre_onset_overlap=True`,**不进终止 summary、不算 ictal termination 窗**(行保留作诊断)。
- MVP **不**加 pre-onset 参照窗(baseline robust-z 已是参照),不预设终止动力学。

## 6. 输出

- `results/topic5_ictal_recruitment/field_dynamics/per_seizure_metrics.csv` — 一行 = subject × seizure × window。
  列: `ds_sid, subject, seizure_idx, seizure_id, window_kind(onset_slide|offset_aligned), t_center_rel_onset,
  t_center_rel_offset, progress_frac, pre_onset_overlap, post_offset_overlap, ictal_fraction,
  parity_fail, band, n_matched, n_source_core, n_axis_end_noncore, n_axial_mid, n_non_axial, axis_degenerate,
  source_focus_uncertain_a, source_focus_uncertain_b, align_maxab, drift_vs_onset,
  angle_to_interictal_axis, grad_mag, source_core_mean_z, axis_end_noncore_mean_z, axial_mid_mean_z,
  non_axial_mean_z, axialmid_minus_nonaxial, source_core_minus_all, pms_source_core,
  pms_axis_end_noncore, pms_axial_mid, pms_non_axial, source_core_pos_share, axial_mid_pos_share,
  non_axial_pos_share, non_axial_p95_z, sync_median_corr, participation`(`pms_*` = positive_mass_share)。
- `results/topic5_ictal_recruitment/field_dynamics/per_subject/<ds>.json` — drop reasons、
  n_eligible / n_complete_interval / n_used / n_perseizure_fig、parity_fail 条数、offset coverage、
  轴信息(`L, bbox_diag, axis_degenerate, source_core_a, source_core_b, source_focus_uncertain_{a,b},
  source_top2_dist_{a,b}_mm, decision_k_provenance`)、每指标 summary(median 轨迹 + onset vs offset 对比)。
- 图(**两层**, narrow,先 render→目视→改 再定稿;复用 field-vs-ictal 图的 display frame / 坐标轴 /
  平滑参数 / 色标逻辑):
  - **per-seizure(每条合格发作 1 张 multi-panel composite,`figures/per_seizure/<ds>/<ds>_szN.png`)**:
    8 panel = 4 field snapshots(progress≈0/33/66/100% 活动场 `rank01(window-mean z)`,`_smooth_rank_field_mm`,
    叠四分区着色 + 轴参照线)+ metric trajectory(随 progress)+ 四分区 positive_mass_share(随 progress)
    + offset zoom(终止对齐窗)+ 同步(随 progress)。**仅 `eeg_duration ≥ 40s` 且非 parity_fail 的发作出**;
    短发作(如 1084)只进 subject-level 聚合。每 subject 报 per-seizure 出图条数。trajectory panel 默认
    `band=bb` + `ictal_fraction≥0.5`。
  - **subject-level(每 subject 4 张,`figures/`)**:① `<ds>_progress.png`(spaghetti+median:align_maxab /
    pms_axial_mid / pms_non_axial / sync_median_corr,横轴 progress);② `<ds>_offset.png`(offset summary:
    rel-offset 窗 median±散点,排除 pre_onset_overlap);③ `<ds>_seizure_heatmap.png`(行=发作,列=progress bin,
    值=pms_axial_mid 等关键指标);④ `<ds>_geometry_qc.png`(四分区 + source_core + uncertain flag + 轴线)。
    (drop "representative atlas" —— 与 per-seizure field snapshots 重复,CLAUDE.md §7 多 panel 去冗。)
- `results/topic5_ictal_recruitment/field_dynamics/figures/README.md`(**中文**,逐图 2–4 句 + 末行 `**关注点**:`)。
- 新结论图目录 append `results/FIGURE_INDEX.md`。

## 7. 锁定参数(spec body 即合同)

| 参数 | 值 |
|---|---|
| substrate | narrow |
| subjects | epilepsiae_{442,548,583,384,958,1084} |
| eligibility | `analysis_eligible==True` ∧ `has_complete_eeg_interval==True` |
| offset 口径 | eeg offset(`eeg_offset_epoch`) |
| baseline | 现有自适应 `[-pre,-60]`(pre≥120), per-ch median/MAD*1.4826 |
| band | bb 1–45 Hz primary + hfa 60–100 Hz secondary,**两者都分析**(每窗×band 一行) |
| cache 窗 | `pre_sec=130`, `post_sec=ceil(max(eeg_offset_rel,eeg_duration))+90`;`span>600s`→drop too-long |
| ictal 轨迹 | onset 窗 summary/plot 默认 `ictal_fraction≥0.5`(短发作跨 offset 窗仅诊断) |
| source core | 每侧 top2 if maxdist<15mm 否则 top1(单点);`decision_k` 仅 provenance |
| compact gate | 15 mm/侧;≥ → `source_focus_uncertain`(单点 fallback) |
| 分区 | 4 组: source_core / axis_end_noncore / axial_mid / non_axial |
| 组占比 | `positive_mass_share = Σmax(z,0)_g / Σmax(z,0)_all`(四组和=1) |
| on-axis 阈 | `d ≤ median(d over 非 source_core mapped)` |
| mid-zone | `t∈[0.25,0.75]` |
| 退化轴 | `L < 0.15 * bbox_diag` → axis_degenerate |
| 滑窗 | 长 10s, step 5s, 到 offset |
| 终止窗 | `[-60,-30],[-30,-10],[-10,0],[0,30]s` rel offset;左缘<onset → `pre_onset_overlap`(排除 summary) |
| baseline parity | [0,10]s vs t0_feature_cache_v2_windows, `max|Δ|<1e-3` → `parity_fail`(剔除) |
| per-seizure 详图 | 仅 `eeg_duration≥40s` 且非 parity_fail |
| alignment | maxAB `|corr_pair_mirror_invariant|`, S_THRESH=0.15, OVERLAP_MIN=25 |
| participation | window-mean z>2 占比 |
| synchrony | window z-trace 两两 Pearson median |

## 8. 健康检查(light, pilot)

- **source_core sanity**: 各 subject 多数窗 `source_core_minus_all > 0`(source 确实恒高;**只用 source_core,不含端区**)。否则 flag(可能 source/几何错配)。
- **baseline parity(强制)**: `parity_fail` 发作不进指标;per_subject 报 parity_fail 条数。
- **source compactness**: per_subject 报每 template 的 `source_focus_uncertain` + top2 距离(实测 384/958/442-B 必 uncertain → 单点 anchor)。
- **负控**: 1084 应 `axis_degenerate==True`。否则回查 `0.15*bbox_diag` 阈。
- **覆盖**: per_subject 报"完整 onset-to-offset 高质量发作(eeg_duration≥40s)"条数。≥3 个 subject 各 ≥8 条 → 触发是否扩 SEEG / 正式 subject-first 统计的讨论(不在本 MVP)。
- **bad-data**: window 内 mapped<6 或全 NaN → 跳过该窗并记 reason。

## 9. 不做 (YAGNI)

- 不做 cross-subject cohort 统计 / 显著性。
- 不做 per-window permutation null(描述为主;后续可加)。
- 不做 clinical offset、不做 Yuquan(无 ictal cache)。
- 不预设、不命名终止动力学;不把单 subject 现象写成机制/规律。
- 不动现有 `t0_feature_cache*` 与上游 swap 图。

## 10. 复用清单(impl 时按此 import,勿重造)

- `src.ictal_onset_extraction.extract_seizure_window`, `src.topic5_ictal_recruitment.baseline_robust_z` / `band_power_trace`
- `src.propagation_contact_plane_readout.{corr_pair_mirror_invariant, R_smooth_rank, smooth_field}`
- `scripts.run_topic5_axis_alignment` 的 per-snapshot field 构造(`make_field_record`/`make_plane_grid` 等,impl 时定位精确符号)
- `scripts.plot_topic5_swap_nodes_fields.{_subject_data 的几何加载逻辑, _arrays, _ring, _legend_handles, SUBSTRATE}`
  —— 但**绕过 swap_class gate**(本分析需要 swap="none" 的 subject;写一个不 gate 的轻量 loader 复用其几何/frame 部分)
- `scripts.plot_contact_plane_static.{_subject_display_frame, _display_points, _smooth_rank_field_mm}`
- `src.rank_displacement.derive_swap_endpoint`(参考其 source-侧最早排序;**但 source core 用 §3.2 compact-core + 单点 fallback,不用 decision_k**)
- 图层复用上游 `scripts/plot_topic5_field_vs_ictal_swap.py` 的 frame/轴/平滑/色标逻辑(P1b 要求)
