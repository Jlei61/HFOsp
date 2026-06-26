# Topic 5 A-line — Event-Resolved Interictal axis_bias (Secondary) — Design Spec **v2**

> **日期**: 2026-06-25
> **版本**: v2（v1 经 advisor-proxy 四镜头复审后整体改写；v1 的"层3 bootstrap 测 std + K-scan"框架已**作废**，原因见 §9）
> **状态**: design（pre-execution）；pilot-first；cohort verdict 是 hard-stop（advisor=用户，需回来 sign-off）
> **关系**: 在现有 A-line primary（scaffold 主结果）**之外新开** secondary，不替换、不修改 primary。
> **上游 primary**: `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md` + `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`
> **配套 plan**: `docs/superpowers/plans/2026-06-25-topic5-event-resolved-axis-bias.md`
> **tier**: 全部 **secondary / exploratory**。本 spec 不产出任何 cohort-level claim 用以扩展或推翻 A-line primary。

---

## 0. 白话摘要（CLAUDE.md §8 三段式）

**测了什么** — 现在 A-line 把一个病人平时成千上万次 HFO 传播"压成一张平均的空间梯度图"，再看它跟发作头几秒的活动梯度图是不是落在同一条空间轴上（共线，不分方向）。一压平均，类内部的差异（有的事件像、有的不像）就没了。这次不压平均：把已经被自动分成 A、B 两类的**每一次具体间期事件**单独拿出来，逐次问"这一次事件的空间梯度像不像发作早期那张梯度图"，于是能看清：① 同一类里有多散（直接量**事件级离散度**）；② A、B 两类是不是各有偏向；③（后续 Stage C）某类事件之后，下一段时间同类/反类事件是否被**短期压低、随间隔恢复**——**这才是用户口中的"短期抑制（STD = short-term depression）"机制问题，本轮（Stage A）未做**。注意：本轮测的"离散度"≠ STD 机制，别混。

**怎么测的** — 关键现实：一次间期事件只点亮几个触点。在原始窄通道集上（6–16 通道）一半事件只点亮 3 个触点，逐事件根本铺不成一张图。但在**broad 通道集**上（20–67 通道）同一个病人每次事件能点亮中位约 13 个触点——足够铺一张小梯度图。所以底物用 broad。每次事件铺成它自己的小梯度图，用**和 primary 同一个"尺子"**（镜像不变、符号自由的场相关算法本身）跟发作梯度图比，得到这一次事件的一个 alignment 数。**注意"同一个"只指场相关这把尺子；发作场的估计量本模块用 subject-mean（比 primary 的 per-seizure 折叠更平均），是 secondary 口径，cohort 前需补 per-seizure-median 敏感性。** 把同一类所有事件的这些数收成一条分布：
- **分布有多宽（直接的 IQR/标准差）= 类内散度**（不是 v1 那个会自动随样本变窄的 bootstrap 宽度——那是均值的标准误，不是散度，已废弃）。
- **A、B 两条分布的位置差 = 类间偏向**；判它是否真有结构，要拿**A/B 标签打乱**（按"块"打乱，因为事件在时间上扎堆不独立）当对照，而不是只拿"发作图触点打乱"。
- **真实分布是否比"发作图随机排布"更靠右** = 是否超过随机；这个用发作激活在该事件参与触点内打乱做对照。
- 还有一个**1D 伴随量**（事件内触点先后顺序 vs 发作强弱的秩相关绝对值），它对稀疏更耐受、能在窄通道高通道数病人上算，但它**更接近"重放"那个被 primary 刻意回避的构念**，所以只当稳健性旁证、明确标注、只看类级分布、不报单个事件、不留方向符号。

**揭示了什么** — 本 spec 只定方法与判据，不预判结果。可能形态：A、B 两类分布**位置接近、宽度有别**（共享同一根粗轴、事件级散度不同），或**位置就有偏向**。无论哪种，口径都停在"类内/类间散度的描述"，且明确：(a) 这是 broad 子队列（≠ A-line primary 的 18 epi 全队列，可比性是注脚不是缺陷）；(b) **不**升级成"某类事件驱动发作 / 逐点重放 / 方向一致 / 某类压制某类"。

（内部归档代号：A-line primary = `corr_pair_mirror_invariant` 2D field + 4-null + FDR；本 secondary = event-resolved axis_bias；A/B = `adaptive_cluster.labels` stable_k=2；phantom 掩膜 = `mask_phantom_ranks`；底物 = `interictal_propagation_masked_broad` + `propagation_geometry_broad`；块自相关 = `block_ids`/`block_time_ranges`。）

---

## 1. 已核验的数据现实（2026-06-25 实测；execution 地基）

| 事实 | 证据（实测） | 含义 |
|---|---|---|
| **逐事件 A/B 标签在盘、与重载事件按位对齐（长度）** | narrow `epilepsiae_1077`：`load_subject_propagation_events`+`_valid_event_indices(bools,3)` → n_valid=54432 == `adaptive_cluster.labels` 长度 | 标签对齐合同成立（但只证了长度，**位置**须另证，见 §C1） |
| **逐事件时间戳/块结构现成** | loader 直接返回 `event_abs_times`/`event_rel_times`/`block_ids`/`block_time_ranges`；JSON `event_metadata.block_boundaries[*].has_packed_times=True` | Stage B/C 不被时间戳阻塞；**块自相关**可被显式建模（§C7） |
| **窄底物稀疏墙** | narrow 1077（6ch）参与触点 min/median/max=3/**3**/6；≥5 占 20%、≥6 占 4.6% | 窄底物逐事件**场**不可行；逐事件场必须走 broad |
| **broad 底物存在、密、stable_k=2** | `results/interictal_propagation_masked_broad/per_subject/`：**12 epi + 17 yuquan**，nch=20（epi）/20–67（yuquan），全 stable_k=2（litengsheng k=3 除外）；broad 几何在 `results/spatial_modulation/propagation_geometry_broad` | broad = 逐事件场的可行底物（feasibility 镜头实测 broad 1077 中位 13 触点/事件） |
| **broad 代价：事件少 + 队列部分覆盖** | broad nvalid 范围 345（139）–19816（1125）；broad epi 只 12 个（缺 A-line 的 1073/1084/1146/442/548/590/958） | broad 子队列 ≠ A-line primary 18 epi 全队列 → 可比性注脚（§6）；少事件 → 报告每类 n_events + **n_blocks** |
| **ictal field 现成** | `results/topic5_ictal_recruitment/t0_feature_cache_v2_windows/<ds_sid>.npz` 的 `bb_auc__<idx>`；`eligible_idxs` 在 **`{ds_sid}.json` sidecar**（不在 npz，§C6） | 逐事件对照目标 = subject-mean bb_auc 场（标注为"比 primary 更平均的估计量"，§C6） |
| **seizure onset 现成** | `results/epilepsiae_seizure_inventory.csv` + `results/dataset_inventory/yuquan_seizure_inventory.csv` + block inventory | Stage B/C 窗口可落地 |

**底物决策（pilot 经验定，默认如下）**：逐事件**场**度量（primary M）走 **broad**；1D 伴随量（M1d）可在 narrow 高通道（n_ch≥9）病人补算。pilot Phase 0 先审 broad 覆盖 + 实测每个 pilot 被试 broad 的参与触点分布与可用比例，再确认底物。

---

## 2. 假设与 tier（pre-registration，planning 锁定，结果里不升级）

> 纪律：A-line primary 是 cohort claim（scaffold）。本 secondary 全部 **exploratory**，目的为**刻画类内/类间散度**与**逐事件 ictal-likeness**，不构成新 cohort 主张。tier 拆细以堵 v1 "descriptive 标签下偷跑推断机器"的漏洞。

- **S1a（purely descriptive，对应 Q1 + 事件级离散度）**：在每个被试内、每类，逐事件 alignment 的**位置（median）与直接离散度（IQR/std）**；列出最像/最不像发作场的若干**具体事件**（idx、abs_time、block_id）。**不**做 null 比较、**不**做 cohort 聚合（只逐被试报数）。这层就是用户 Q1 的字面答案 + 正确的离散度度量（**注：此处"离散度"是统计散布，≠ 用户口中的短期抑制 STD；STD 机制是 S3/Stage C**）。
- **S1b（null-relative，inference 机器，须明示）**：
  - **类间分离**：A−B 的 Δmedian 与散度比，对 **A/B 标签块级打乱 null**（保类大小）+ 类大小匹配（大类下采样到 n_min）。
  - **超随机**：逐事件 alignment vs 发作激活打乱 null（参与触点内）。
  - cohort 层若出任何陈述：只能写"k/N 个被试观测值落在被试内 null 带之外，**未校正、exploratory、family 大小已披露、不花 alpha**"（或若要正式则继承 primary 的 FDR）。**禁止** cohort 动词（"A 类更对齐 / 显著 / cohort 超 null"）。
- **S2（Q2，Stage B，pilot-gated，exploratory）**：A/B 两类事件在 pre-ictal/post-ictal/far-background 窗口的出现占比与 alignment 是否随窗变。窗口复用 primary 的 background/peri-ictal 口径。
- **S3（Q3，Stage C，pilot-gated，exploratory）**：给定前一事件类别，紧随时间窗内同类 vs 反类事件的 reach（参与触点数）、rank-field、事件率是否被压低/偏转。**仅描述"是否存在条件依赖"**，model-sufficiency≠causal；禁"某类压制某类"。

**禁止升级**：S1–S3 任何信号再强都不得写成 cohort claim、不得写"驱动/重放/方向一致/压制"。允许/禁止措辞见 §6。

---

## 3. 核心度量（spec lock；按此立 TDD）

设被试（broad 底物）有通道集 `C`（loader `channel_names`）；A、B 两类各有自己的聚合平面（A→`t_a`、B→`t_b`，几何来自 `propagation_geometry_broad`，各由 `compute_axis_frame` 在该类自己的 source/sink core 上建；**不共用同一平面**，§C3）；subject-mean 发作激活 `a:C→ℝ`（bb_auc over `eligible_idxs`，仅 matched 触点有值）。

逐事件原始：`ranks,bools (|C|,n_ev)`（loader）。掩膜 `masked = mask_phantom_ranks(ranks,bools,normalize=True)`（参与触点局部重排[0,1]，非参与=NaN；**直接 2D helper 取列；禁 build_masked_kmeans_features 的 0.5-impute 入场**，§C5）。valid+标签：`valid_ev=_valid_event_indices(bools,3)`，`labels[i]↔valid_ev[i]`（位置由 §C1 证）。

### 3.1 Primary 度量 M — 逐事件镜像不变场相关（与 A-line 同构念，broad 底物）
对每个 valid 事件 `e`（类 `g`，参与触点 `P_e`，`n_e=|P_e|`）：
```
plane_g          = 类 g 的聚合平面记录（x_norm,y_norm；§C3）
support_e[c]     = 1.0 if c∈P_e else NaN        # 该事件自己的参与，不用聚合 support（§C4）
F_e, S_e         = R_smooth_rank( make_field_record(plane_g 上 matched∩P_e 的触点, masked[P_e,e]),
                                  X, Y, sigma_xy = sigma_g_full )   # sigma_xy 钉死为类 g 全通道模板值（§C4）
align_e          = | corr_pair_mirror_invariant(F_e, S_e, F_ictal, S_ictal) |   # 符号自由、镜像不变（同 primary）
```
- `F_ictal` = subject-mean bb_auc 场（§C6：标注"比 primary 更平均的估计量"，**不**写"与 primary 同构造"；per-seizure-median 版 = sensitivity）。
- 门控：`n_overlap ≥ OVERLAP_MIN`（=25，沿用 primary）否则 `insufficient_overlap`，计入并**报告 usable fraction**（CLAUDE.md §6 不沉默丢弃）。
- 输出每事件 `{align, n_part, label, event_idx, abs_time, block_id, status}`。

### 3.2 Companion 度量 M1d — 逐事件 1D 共线（replay-adjacent，明确标注）
对 valid 事件 `e`，**eligibility**：`MIN_PART_EVENT ≤ n_e ≤ n_ch − CHANNEL_HEADROOM`（事件须"留出≥3 个未参与触点"以携带事件特异信息；故只在 `n_ch ≥ 8` 病人有事件合格，§feasibility 镜头 BLOCKER）：
```
r_e = masked[P_e,e]; a_e = a[P_e]               # a_e 须全 finite
align1d_e = | spearman(r_e, a_e) |               # 符号自由；不存 sign（§6 删 replay 旁路）
```
- **per-event null（专配，不借场级 4-null）**：在 `P_e` 内打乱 `a_e`，给该事件该触点数的 chance |spearman|。
- 报告纪律：**只**作类级分布的稳健性旁证；**不**逐事件点名"哪个事件最重放"；措辞按 §6 replay 禁忌门控。

### 3.3 Readouts（全部块感知；effective N = n_blocks，§C7）
- **R1（S1a）**：每类 align 分布的 median + 直接 IQR/std；最像/最不像的 top/bottom 事件（idx,abs_time,block_id）。
- **R2（S1b 类间）**：A−B Δmedian + 散度比，vs **A/B 标签块级打乱 null**（保类大小）+ 类大小匹配。
- **R3（S1b 超随机）**：align vs 发作激活打乱 null（channel=primary；within_shaft/anchor/joint=sensitivity），块感知聚合。
- **M1d 伴随**（仅 n_ch≥8 病人）：类级 align1d 分布 + per-event null，replay-adjacent 标注。

### 3.4 cohort 汇总（仅 pilot 通过 + advisor sign-off 后；本 spec 不下判）
- per-subject 摘要 → 按 `subject_id` **配对** A vs B（§C9）；descriptive；**不**做 binary PASS/FAIL；family 大小披露；不花 alpha（或继承 primary FDR 若正式化）。

---

## 4. 冻结参数（lock；改动须回本 spec 改并记 rationale）

```
substrate_primary    = broad (interictal_propagation_masked_broad + propagation_geometry_broad + *_lagPat NPZ broad pool)
substrate_companion  = narrow (仅 M1d，n_ch≥8 病人)
min_participating     = 3        # valid-event gate（与上游 labels 一致）
MIN_PART_EVENT        = 5        # M1d 逐事件下限
CHANNEL_HEADROOM      = 3        # M1d：n_part ≤ n_ch−3（须留≥3 触点未参与）；故 n_ch≥8 才有合格事件
OVERLAP_MIN           = 25       # M 场相关像素重叠（沿用 primary）
S_THRESH              = 从 propagation_contact_plane_readout 导入，不复制字面量
sigma_xy              = 钉死为各类全通道模板的 sigma（不逐事件重算，§C4/§C7）
ictal_reference       = subject-mean bb_auc over eligible_idxs（标注更平均；per-seizure-median = sensitivity）
activation_primary    = broadband (bb_auc)；hfa = sensitivity
class_sep_null        = A/B 标签**块级**打乱，保类大小，N_PERM=1000
chance_null           = 发作激活 channel-shuffle（primary）；within_shaft/anchor/joint = sensitivity
cluster_map_margin    = 0.30     # signed diag−offdiag 边际（§C2），不足→ambiguous→剔除
RNG_SEED              = 20260625
PILOT_SUBJECTS        = [epilepsiae_1077(broad20/narrow6), epilepsiae_1125(broad20/narrow8,边界),
                         epilepsiae_922(broad20/narrow8,多事件), yuquan_zhangbichen(broad67)]
```

**度量优先级**：M（broad 逐事件场）= primary 读数；R1 直接离散度 = 事件级离散度主答（**≠ STD 短期抑制；STD = Stage C 序列问题，未做**）；M1d = replay-adjacent 稳健旁证。**v1 的 bootstrap/K-scan 已删**（§9）。**用户 pivot（2026-06-25）：另出"每类全事件投影+加权归一化"的场图（A 场 | B 场 | 发作场），左图形式 → `scripts/plot_topic5_event_resolved_fields.py`。**

---

## 5. 数据/复用合同（CLAUDE.md §5/§6/§6.1；实现前逐条核对、逐条立 TDD）

- **§C1 位置级标签对齐（load-bearing，三条硬 raise；2026-06-25 已在 broad 1077/1125/922 实测 PASS）**：(1) 重载 `channel_names` 与该被试 broad JSON 的 `channel_names` **逐元素相等**；(2) 由 `labels` 重算每类事件计数 == JSON `adaptive_cluster.clusters[k].n_events`；(3) **复现 producer 的模板** `argsort(argsort(_legacy_hist_mean_rank(ranks[:, valid_ev[labels==k]], bools[:, valid_ev[labels==k]])))` 与 JSON `clusters[k].template_rank` **整数列逐元素相等**（实测三被试全 exact；容差兜底用 rank-corr ≥0.99）。**注意**：用 loader 的**原始 `ranks`**（非 masked）复现模板——producer 用的是 `_legacy_hist_mean_rank`（raw ranks + bools 选参与值），不是 masked nanmean；masked nanmean 是不同聚合（实测只到 0.83/0.61，故**不可**用作 clincher）。任一不符 → raise（仅长度相等不够；底物/glob 漂移会"长度巧合相等而事件错位"）。
- **§C2 cluster_id↔t_a/t_b 映射**：**signed** max-corr，组 2×2，要求 `diag_mean − offdiag_mean ≥ cluster_map_margin` 且 argmax 为双射；否则 `cluster_template_map=ambiguous` → 该被试**剔出**配对比较（不静默选）。记录两侧 corr。（forward/reverse 近镜像是本数据常态，故必须 signed + 边际。）
- **§C3 每类用自己平面**：A 事件铺在 `t_a` 平面、B 事件铺在 `t_b` 平面（各 `compute_axis_frame` 于该类 source/sink core）。**禁**单一共享平面（A-line primary 只验过 t_a；强加 t_a 给 B 会用错轴污染）。
- **§C4 逐事件 support + sigma**：support 用**该事件自己的参与**（参与=1，其余 NaN 丢弃），**禁**用聚合 support（否则把要剖析的"平均"又带回来）。`sigma_xy` 钉死为该类全通道模板的 sigma（显式传入），不逐事件重算（否则平滑尺度与 ictal 场不可比）。
- **§C5 phantom 掩膜**：单事件=`mask_phantom_ranks(ranks,bools)[:,e]`（NaN-drop，非 0.5-impute）；**禁** `build_masked_kmeans_features`（其 0.5 是给 KMeans Euclidean 的，入场会污染）；任何 nanmean 聚合前先丢 `n_part<2` 列（杜绝单参与 0.5 哨兵进模板）。
- **§C6 ictal 估计量 + eligible_idxs 来源**：`ictal_reference` = subject-mean 场，**显式标注 ≠ primary 的 per-seizure-median**（per-seizure 版列 sensitivity）。`eligible_idxs` 从 `{ds_sid}.json` sidecar 读（**不在** npz），断言存在。
- **§C7 块自相关一等公民**：所有散度/分布/null 报告须带 **n_blocks**（effective N）；标签打乱 null **按 block 打乱**（事件块内相关，按事件打乱会低估 null 散度）；**禁**任何"n=54432 事件"式逐事件计数精度措辞。
- **§C8 selection cost 对称**：real 与 null 走**同一个** mirror-invariant+sign-free reduction（M 的 max-over-{identity,mirror}、M1d 的 abs）；TDD 断言 null 用同一 reduction 实例。
- **§C9 paired-cohort key**：cohort 配对按 `subject_id`（`sorted(set keys)` + assert 集合相等），禁 dict 顺序。
- **§C10 stub**：Stage B/C 入口 `raise NotImplementedError`，不返回 plausible 值。

---

## 6. 允许/禁止措辞（archive/main doc/对用户复盘）

**允许**：
- "在 broad 子队列被试内，逐事件 ictal-alignment 有可观类内散度（直接 IQR/std）；A、B 分布位置[接近/有偏向]、宽度[相近/不同]。"
- "这是 broad 子队列（≠ A-line primary 18 epi 全队列），事件数与 n_blocks 已报，结论为 exploratory、未校正。"
- "M1d（1D 共线）作为稳健旁证，是比 primary 更接近重放的构念，只看类级分布。"

**禁止**（沿用 A-line + swap-geometry 纪律）：
- "某类间期事件驱动/触发发作"
- "发作沿某类事件逐点重放 / 方向一致 / 前向传播一致"
- "A 类事件抑制/压制 B 类事件"（Stage C 即便见条件依赖，也只能说"见到条件依赖"，≠ causal）
- 把 S1–S3 写成 cohort claim 或 primary 升级；cohort 用"显著/更对齐"等动词
- "HFA 是 primary"（broadband 才是）
- **逐事件点名"事件 X 重放了发作"**（M1d 只报类级分布）

---

## 7. 文件结构（plan 按此 decompose）

**新模块（纯函数 + TDD）** `src/topic5_event_resolved_alignment.py`：
- `load_event_labels_ranks(dataset, subject, *, broad=True) -> dict`（重载 broad NPZ + broad 标签 JSON；执行 §C1 三条硬核对；返回 masked/bools/labels/valid_ev/event_abs_times/block_ids/channel_names）
- `map_clusters_to_templates(labels, masked, valid_ev, t_a_rank, t_b_rank, *, margin) -> dict`（§C2，含 ambiguous 判定）
- `per_event_field_alignment(masked, bools, valid_ev, labels, plane_a, plane_b, ictal_field, *, sigma_a, sigma_b, overlap_min) -> dict`（M，§3.1，§C3/C4/C8）
- `per_event_1d_alignment(masked, bools, valid_ev, labels, ictal_by_ch, channel_names, *, min_part, headroom, rng) -> dict`（M1d，§3.2）
- `class_separation_block_null(align_by_event, labels, block_ids, *, n_perm, rng) -> dict`（R2，§C7）
- `participation_diagnostics(bools, labels, block_ids) -> dict`（参与触点分布 + usable fraction + n_blocks/类）
- `stage_b_window_bias(...) / stage_c_sequential_effects(...)` → `raise NotImplementedError`（§C10）

**驱动** `scripts/run_topic5_event_resolved_alignment.py`：`--pilot` / `--subjects` / `--substrate {broad,narrow}` / `--activation` / `--out`；产 per_subject JSON + pilot_summary（cohort 仅 sign-off 后）。

**图** `scripts/plot_topic5_event_resolved_alignment.py`：每被试一图——(a) M 逐事件 A/B alignment 直方/小提琴 + usable fraction + n_blocks；(b) R2 A−B Δ 与标签打乱 null 带；(c) 参与触点分布（broad vs narrow 对照）。每 panel 一个独立问题（§7 纪律）。配 `figures/README.md`（中文）。

**测试** `tests/test_topic5_event_resolved_alignment.py`：§C1 位置对齐（合成错位须 raise）、§C2 近镜像 ambiguous、§C3 B 事件用 t_b、§C4 support=事件参与且 sigma 钉死、§C5 0.5 不入场、§C7 标签按块打乱、§C8 同 reduction、门控+usable fraction、§C9 配对 key。

**输出** `results/topic5_ictal_recruitment/event_resolved_alignment/`（`per_subject/`、`figures/`、`pilot_summary.json`、`cohort_summary.json`）。

---

## 8. 执行边界（hard stops；用户离开期间生效）

1. **PILOT-FIRST**：Phase 0 先审 broad 覆盖 + 实测 4 个 pilot 被试 broad 参与触点分布与 usable fraction；再实现 M + TDD；再在 pilot 被试跑 M（+ M1d 于 narrow 高通道）并出图。
2. **pilot gate（围绕 Q1 可估性，非 bootstrap 收敛）**：每被试报 (i) broad vs narrow 参与触点分布；(ii) M 的 usable fraction（n_overlap≥25）≥ 阈值（pilot 目视定，初设 ≥0.3）；(iii) 每类 align 分布在足够 events + **n_blocks** 下可估；(iv) 图可目视。**不**用"分布随 K 收敛"作判据。
3. **advisor=用户 是 cohort 判决 hard stop**（调查确认 advisor 为人）。pilot 完成后**停在 cohort run 之前**，写 pilot recap（§8 三段式）等 sign-off，再决定 (a) cohort run、(b) 底物/度量微调、(c) 进 Stage B/C。
4. 离开期间**允许**：写 spec/plan、advisor-proxy 复审、实现模块+TDD、跑 pilot、出 pilot 图。**禁**：cohort 级判决结论、改 A-line primary、长 cohort run。

---

## 9. v1→v2 变更账（为什么改；CLAUDE.md §5 re-read 纪律）

advisor-proxy 四镜头（stats / contracts / feasibility / scope）一致 BLOCKER：
1. **删 v1 层3 bootstrap + K-scan 当 std**：bootstrap-of-means 宽度 = 均值标准误（~1/√K），**不是**类内 std；K-scan 收窄是 CLT 必然，零异质也会"通过"。→ std 改为**逐事件分布的直接 IQR/std**（R1）。
2. **Layer 1 升为答 Q1 的层**（v1 误降级）：逐事件标量才回答"哪些具体事件更像"。但 1D 共线 replay-adjacent → 删 sign、标注、只报类级；同时新增 channel-headroom 门控（n_ch≥8 才合格）。
3. **逐事件场度量改走 broad**：窄底物逐事件场 dead-on-arrival（4.6% 事件够 6 触点）；broad（中位 13 触点）使其可行且**留在 primary 的镜像不变场家族**（最强的反 over-claim 防线）→ 定为 primary M。
4. **新增 A/B 标签块级打乱 null**（v1 缺）：原 4-null 只测"超随机排布"，不测"A≠B"；类大小悬殊（如 A=16462/B=37970）会机械造成宽度差。
5. **块自相关一等公民**（v1 只在 loader 提）：所有散度/null 块级、报 n_blocks、禁逐事件计数精度。
6. **每类用自己平面 + 逐事件 support + 钉死 sigma**（v1 共用 t_a + 聚合 support + 重算 sigma 三处错）。
7. **cluster↔template signed+边际+双射门控**（v1 仅"most correlated"，近镜像会误映）。
8. **ictal 估计量明示 ≠ primary**（v1 误称"同构造"；primary 是 per-seizure-median，本 spec 是 mean-field）。
