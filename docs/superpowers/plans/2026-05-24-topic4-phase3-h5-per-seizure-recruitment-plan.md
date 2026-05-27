# Phase 3 H5 — Per-seizure recruitment / expansion plan (v1.0.7 spec)

> **状态**：v0 plan draft 2026-05-24, **pending advisor review** before implementation
> **framework**：`docs/topic4_sef_itp_framework.md` §3.5 v1.0.7 H5 sub-amendment
> **前置**：Phase 2 v1.1.0 (Stage B) cohort 收口完成 (cohort_run_2026-05-24.md)
> **目标 archive**：`docs/archive/topic4/sef_itp_phase3/cohort_run_<TBD>.md`
> **module target**：`src/sef_itp_phase3.py` (new) — reuses `_per_cluster_template_rank`, `compute_local_rank_endpoint`, `compute_endpoint_spatial_radius`, `compute_source_sink_centroid_distance`, `compute_decision_k_drift` from `src/sef_itp_phase2.py` v1.1.0
> **核心朴素话**：每次 seizure 是一次独立的测量机会, 看 seizure 前后或两次 seizure 之间的端点形状和数量, 跟同病人正常背景时段比, 有没有"招募新通道"或"空间扩张"的迹象。SEF-ITP 区分性预测: 真正的病理核心受发作驱动会扩散, 不只是触发频率升高。

---

## 0. 一个真正测什么的朴素话

间期我们假设有一个固定形状的病理区, 随机触发后涟漪扫过附近通道。但发作邻近会发生什么? SEF-ITP 的预测:

- 如果发作只是 "现有病理区被触发频率升高" → 端点 identity / 空间范围相对 background 不变, 只有事件率升高
- 如果发作伴随 "病理核心招募新通道" 或 "空间扩散到更大区域" → 端点 identity 漂、加新通道; 核心区数量 (decision_k) 升; source / sink 通道集合空间范围扩大

Phase 3 测的就是后者: 在每次 seizure 前后 (或两次 seizure 之间的特定时段), 跟同病人 time-of-day matched baseline 比, swap-k endpoint 的数量 + 空间范围 + identity 是否系统性 increase / change。

**关键设计**: per-seizure 是 primary reporting unit, NOT subject-level aggregate。Topic 5 PR-1 / PR-4 已证明 subject 内 seizure type 差异巨大、间隔波动大、相对时间不能解决 — subject 级 mean 会把这些 within-subject heterogeneity 抹平。

---

## 1. Locked contract (user-return v3 + v4 catches 2026-05-23/24 ratified, framework v1.0.7)

| 项 | Lock |
|---|---|
| Primary reporting unit | **per-seizure** (每次 seizure 的 peri-ictal window vs matched baseline) |
| **Primary cohort restriction (v4 catch 2026-05-24 lock)** | **strict ∪ candidate subjects only** (在背景间期 full-data swap_sweep 给出 swap_class ∈ {strict, candidate} 的人, 即真有 stable swap-core 的人); none = negative control 独立报, all-cohort = sensitivity 独立报 |
| Cohort inference (primary) | **cluster-robust SE on subject id** (statsmodels OLS `cov_type='cluster'` sandwich estimator) |
| Cohort inference (sensitivity, **required when n_qualifying_subjects < 10**) | **wild cluster bootstrap** (resample subjects) OR subject-level bootstrap — primary 是 cluster-robust SE, sensitivity 必跑 |
| Cohort inference (sensitivity, robustness) | mixed-effect model (seizure nested in subject) 或 GEE — sensitivity only, primary 以 cluster-robust SE 为准 |
| **p-value direction lock (v4 catch 2026-05-24)** | **one-sided directional p** (Δk > 0, Jaccard ↓, new_node_fraction ↑, radius ↑); 所有 SUPPORTED branch 用 one-sided; reverse-direction (FAIL branch) 用对称 one-sided。**禁止**写 two-sided p 然后讨论 direction (混淆) |
| Stratification (mandatory) | seizure type / seizure pair / inter-seizure interval — 分层报, 不 collapse |
| **BH-FDR family boundaries (v4 catch 2026-05-24 lock)** | pre-ictal 和 post-ictal **各自独立 BH-FDR family** (不合并); ISI 分析单独 family (不与 peri-ictal 合并); 每 family 内 6 primary metrics 做 BH-FDR q<0.10 |
| Subject-level summary | secondary (between-subject heterogeneity sensitivity only) |
| Multi-seizure independence | **禁止**把同一 subject 多次 seizure 当独立样本 (cluster-robust SE 必要原因) |
| Primary 主问题 | "在 seizure-adjacent window 或两次 seizure 之间，swap-k endpoint 的数量和空间范围是否相对 matched baseline 增加？" |
| swap_class 分层 | **strict / candidate 子集 = primary inference set**; none subset = negative-control (per-window decision_k 是 noise / not 真实核心大小; 期望 Δk ≈ 0; 出现显著 Δk → 红 flag); all-cohort = sensitivity |
| **B0 eligibility audit gate (v4 catch 2026-05-24 lock)** | **必须先跑 audit 再锁 data gate**: 统计每 subject 的 (n_seizures, n_qualifying_peri_windows, n_matched_baselines, n_events_per_window, n_events_per_cluster_per_window) 分布。**在 audit 完成前不锁 `min_events_per_window`**（plan §3.3 当前的 100 是占位，必须 audit 后才能锁实际阈值） |

**Primary metrics (lock, with v4 catch 2026-05-24 update)**:

| # | 指标 | 类型 | 期望 SEF-ITP 方向 |
|---|---|---|---|
| 1a | **Δdecision_k = k(peri) − median(k(baseline_i))** | primary (sign-check, NOT independent gate vote) | Δk > 0 (core recruitment) |
| 1b | **Δdecision_k / baseline_k** (normalized; baseline_k = median across baseline windows) | sign-check companion of 1a; **NOT 独立 primary BH-FDR gate** | same sign as 1a |
| 2a | **swap-k endpoint identity Jaccard (peri vs each baseline window i, then median across i)** | primary | Jaccard ↓ (招募新通道) |
| 2b | **new_node_fraction = median over baseline_i of \|S_peri \ S_base_i\| / \|S_peri\|** | primary (**v4 catch 2026-05-24**: identity recruitment 不只用 Jaccard, 加 new_node fraction; Jaccard ↓ 与 new_node_fraction ↑ 联合解读 = "招募" 比单独 Jaccard 更稳健) | new_node_fraction > 0 |
| 3a | **Δsource centroid RMS = source_RMS(peri) − median(source_RMS(baseline_i))** | primary | Δ > 0 (source 侧空间扩张) |
| 3b | **Δsource mean pairwise** (same baseline aggregation) | primary | Δ > 0 |
| 4a | **Δsink centroid RMS** (same) | primary | Δ > 0 (sink 侧空间扩张) |
| 4b | **Δsink mean pairwise** (same) | primary | Δ > 0 |
| 5 | **HFO rate Δ = rate(peri) − median(rate(baseline_i))** | **secondary** | rate ↑ — descriptive excitability marker only, **不**入 SUPPORTED gate |

**Baseline 聚合规则 (v4 catch 2026-05-24 lock — 不能 median endpoint set, 必须 per-window 算再聚合)**:
- **decision_k**: median across baseline windows (scalar 可 median)
- **swap-k endpoint set**: **不能 median set**; peri set vs each baseline window i 算 Jaccard / new_node_fraction, 然后跨 baseline windows 取 median
- **spatial radius (per side, per metric)**: 每个 baseline window 算自己的 radius (scalar), peri radius − median(baseline radii)
- **rate**: scalar median across baseline windows

**Δdecision_k 与 normalized Δk 不算两个独立 BH-FDR gate** (v4 catch 2026-05-24): 1a (raw Δk) 和 1b (normalized Δk/baseline_k) 是 sign-check 联合, 报为 "Δk median > 0 且 normalized Δk 同号" = "concordant k trend"; 不分两个独立 vote, 不进 BH-FDR family。**BH-FDR family** 仅含 {Jaccard, new_node_fraction, source_RMS, source_pairwise, sink_RMS, sink_pairwise} = **6 个 primary tests per side**。

**v1.0.6 §3.5 primary metric tier 旧表 (4 primary) → v1.0.7 §1 lock 现表 (6 primary BH-FDR + Δk sign-check) 是 user-return v4 catch 2026-05-24 的 refinement**, 不是新增 primary。

**MEB (min enclosing ball) 降为 sensitivity** (user-locked 2026-05-24): primary spatial radius 用 centroid RMS + mean pairwise (这两个对任意 k 数学上无歧义); MEB 当前实现对 k>3 不完整, 不补 4-point sphere, 也不进 Phase 3 primary。

**Verdict threshold (user-locked 2026-05-24, NOT "3/3 全过")**:

> **SUPPORTED**: identity recruitment (指标 3) **OR** radius expansion (指标 4 source side OR 指标 5 sink side) 至少一项 cluster-robust p<0.05 (BH-FDR q<0.10 across primary metrics)
> + **AND** Δk (指标 1) / 与 Δk/baseline_k (指标 2) 同方向 (Δk_median > 0 + 对应 normalized 同号)
> + ≥6 qualifying subjects with ≥1 qualifying seizure each
> rate Δ (secondary) 伴随报告, **不**入 SUPPORTED gate

> **NULL**: identity recruitment 与 radius expansion 都不显著 + Δk 方向不一致

> **FAIL**: identity Jaccard ↑ (identity 收紧而不是招募) 或 radius ↓ (空间收缩而不是扩张) 显著反向 → 与 SEF-ITP 预测反向

> **UNDERPOWERED**: <6 qualifying subjects with ≥1 qualifying seizure

**关键 framing 锁** (避免 over-read):
- ❌ "Δk + Δk/normalized Δk + Jaccard + radius source + radius sink = 5 个 primary, 3/5 显著 → PASS" — Δk 和 Δk/baseline_k 联合解读 (一个是 raw 变化, 一个是规模化效应), **不算两个独立 vote**
- ❌ "rate Δ 显著 → PASS" — rate 是 secondary, 永远不入 SUPPORTED gate
- ✅ "swap-k node 招募 OR 空间半径扩张 至少一项显著 AND core size (Δk) 方向一致" → 才是真正的 SEF-ITP recruitment / expansion claim

---

## 2. 数据 + helper 复用 (CLAUDE.md §6.1 question-match)

| Phase 3 需要的事 | 复用来源 | 问题匹配？|
|---|---|---|
| Seizure timestamps (Yuquan) | `results/seizure_detection/yuquan/<sid>.json` (PR-1) | ✅ per-seizure start/end times |
| Seizure timestamps (Epilepsiae) | SQL `seizure` table (per-subject .sql) + Phase 1 `epilepsiae_inventory` | ✅ per-seizure start/end + sub-block clinical labels (when available) |
| HFO event times | `loaded["event_abs_times"]` from `load_subject_propagation_events` | ✅ 同 Phase 2 H4 ingest |
| Per-cluster events | `loaded["bools"]` + PR-2 masked `labels` | ✅ 同 Phase 2 H4 ingest |
| Per-window per-cluster template rank | `src.sef_itp_phase2._per_cluster_template_rank` (复用) | ✅ 同 Phase 2 B1/B3 canonical helper |
| Per-window swap_sweep → decision_k + swap_class | `src.rank_displacement.compute_swap_score_sweep` (复用) | ✅ 同 Phase 2 B3, 不同窗口 (peri-ictal 而非 epoch); 用 per-window rank_a/rank_b/valid_a/valid_b |
| **Per-window swap-k endpoint node set** (v4 catch 2026-05-24 修正; 原错: 写成复用 `compute_local_rank_endpoint`) | `src.rank_displacement.derive_swap_endpoint(channel_names, rank_a_dense, decision_k)` (复用; rank_a_dense 来自 per-window `_per_cluster_template_rank` → `argsort(argsort)` 或用 `compute_swap_score_sweep` 返回 rank_a_dense 字段) | ✅ 同 Phase 2 H2 spatial 层 input; 不是 fixed source/sink endpoint (那是 `compute_local_rank_endpoint`, PR-6 anchoring 风格), 而是 **variable-k swap-k node set** (Topic 4 H2 layer 的 channel-label 输入) |
| Spatial radius per side (centroid RMS + mean pairwise) | `src.sef_itp_phase2.compute_endpoint_spatial_radius` (复用; **忽略 MEB return field**, 不用作 primary) | ✅ 同 Phase 2 B2, 不同窗口 + MEB ignored |
| 3D coords (mm) | `src.seeg_coord_loader.load_subject_coords` (复用 Phase 1) | ✅ |
| Subject swap_class (primary stratification) | `results/interictal_propagation_masked/rank_displacement/per_subject/<...>.json::pairs[0].swap_sweep.swap_class` | ✅ 同 Phase 2 runner |
| Time-of-day matched baseline picker | **新写**, 没有 Phase 1/2 helper | ⚠️ Phase 3 specific |
| Cluster-robust SE (sandwich estimator) | **新写**, 用 `statsmodels.regression.linear_model.OLS().fit(cov_type='cluster', cov_kwds={'groups': subject_id})` | ⚠️ Phase 3 specific |

---

## 3. Per-seizure window slicer + matched baseline picker

### 3.1 Window definitions (continue v1.0.5 lock)

每次 seizure 定义 3 类窗口:
- **Pre-ictal**: `[onset − 60min, onset − 5min]` (55 min duration, 跳过 5 min 缓冲避免污染 seizure onset zone)
- **Post-ictal**: `[end + 5min, end + 60min]` (55 min duration)
- **Inter-seizure interval (ISI) window** (新, v1.0.7 amendment 朝 "两次 seizure 之间" 直接扩展): 当一对相邻 seizure 之间间隔 ≥ 6 小时, 取 ISI 的中段 `[onset₁ + 3h, onset₂ − 3h]` 内的所有 60-min sliding window (可多个 windows per ISI pair)。**Phase 3 v0 secondary, Phase 3 v1 主线只用 peri-ictal**; 等 peri-ictal 走通后再加 ISI 分析。

### 3.2 Matched baseline picker (continue v1.0.5 lock)

每个 peri-ictal window, baseline 候选规则:
1. 同一 subject
2. 同 hour-of-day ± 2h (控制 circadian)
3. 距任何 seizure ≥ **12 hours** (避免污染)
4. 每 seizure 至少配 **≥5 个独立 baseline windows** (用于 within-seizure variability 估计)

不满足任一条 → 该 seizure 从 Phase 3 排除 (不强凑)。

### 3.3 Data gate — **PENDING B0 eligibility audit (v4 catch 2026-05-24 lock)**

**当前 plan 阈值是占位，必须先跑 B0 audit 后才能锁定**：

- ~~per channel × per window `n_events ≥ 30`~~ → **PENDING audit**: 这条 v1.0.5 sustained spec 对 endpoint / swap-k 分析可能会偏向高参与通道, 把 recruitment 信号 (= 新通道被招募 → 这些通道在 baseline 中可能 n_events < 30) 过滤掉。改用:
  - **窗口总事件数** + **每 cluster 最小事件数** 作 primary gate
  - **channel 级 valid** 用 bool participation (channel.any() in window) 而不是 n_events ≥ 30
  - n_events ≥ 30 per channel 作 **sensitivity audit** (在 main analysis 跑完后看排除 channel 数对 verdict 影响)
- ~~per window 总事件数 < `min_events_per_window = 100`~~ → **PENDING audit**: 100 是占位; B0 audit 后看 cohort 上每窗事件数分布的中位 + IQR + 25th percentile 再锁
- ~~每 seizure 至少配 ≥5 个独立 baseline windows~~ → **保留** v1.0.5 lock (这条是 baseline variability 估计的最低要求, 不依赖 audit)

**B0 audit 输出 (lock 阈值前必须有)**:
- 每 subject 的 (n_seizures_total, n_qualifying_peri_windows_pre, n_qualifying_peri_windows_post, n_matched_baselines_per_seizure_distribution)
- 每窗的 (n_events_total, n_events_per_cluster_a, n_events_per_cluster_b) 分布 (median, IQR, p25, p75)
- 每 channel 在 typical window 的 n_events 分布 (用于评估 ≥30 阈值是否过严)
- 输出 csv: `results/topic4_sef_itp/phase3_ictal_adjacent/diagnostics/b0_eligibility_audit_<date>.csv`

**Audit 后才决定**: min_events_per_window (e.g., 50 / 75 / 100), min_events_per_cluster (e.g., 20 / 30 / 50), 是否保留 channel-level n_events ≥ 30 还是降为 sensitivity。

---

## 4. Δmetric computation per seizure

For each (subject, seizure, side ∈ {pre, post}) tuple:

### 4.1 Compute per-window metrics (per-window pipeline — v4 catch 2026-05-24 修正)

For peri-ictal window AND each matched baseline window i, run the **canonical 4-step per-window pipeline**:

```
window's events = events with event_abs_times in [t_start, t_end)
cluster_a / cluster_b = primary_pair from full-data rank_displacement JSON (固定 = 0, 1 for stable_k=2)

Step 1 — per-cluster template rank:
  rank_a, valid_a = _per_cluster_template_rank(ranks, bools, window_events ∩ {labels == cluster_a})
  rank_b, valid_b = _per_cluster_template_rank(ranks, bools, window_events ∩ {labels == cluster_b})

Step 2 — swap_sweep:
  sweep = compute_swap_score_sweep(
      rank_a=rank_a.astype(float),
      rank_b=rank_b.astype(float),
      valid_mask_a=valid_a,
      valid_mask_b=valid_b,
      n_perm=500,  # Phase 2 B3 同 budget
      seed=...,
  )
  swap_class_window = sweep["swap_class"]
  decision_k_window = sweep["decision_k"]
  T_obs_window = sweep["T_obs"]

Step 2.5 — **per-window swap quality gate (advisor catch 2 2026-05-24)**:
  # `decision_k` 在 `T_obs < score_floor` (0.5) 时是 noise — sweep 仍返回一个 argmin k
  # 让 swap_score_obs == T_obs, 但这是从噪声中找的"最佳"k, 不是真实 swap signal。
  # Subject 级 strict 但 per-window 可能 weak (小 n_events, noisy template), 该 window 的
  # decision_k 不该进 Δk_median 计算; 不 gate 会让 Δk 被 noise driven。
  if T_obs_window < 0.5:  # score_floor 与 compute_swap_score_sweep default 一致
      decision_k_window = np.nan  # Δk 计算时跳过这个 window
      swap_k_endpoint_channels = []  # Jaccard / radius 也跳过
      continue  # window 标 sub-floor, 进 diagnostic 但不进 metric aggregation

Step 3 — rank_a_dense recipe (advisor catch 1 2026-05-24 lock):
  # `compute_swap_score_sweep` 不返回 rank_a_dense; 必须显式重算 — recipe lock:
  joint_valid = valid_a & valid_b  # n_valid channels in joint pool
  channel_names_joint = [channel_names[i] for i in np.flatnonzero(joint_valid)]
  # dense rank within joint_valid subset (0..n_valid-1), method='dense' 处理 ties:
  rank_a_dense_joint = scipy.stats.rankdata(rank_a[joint_valid], method='dense') - 1

  # Calibration gate (smoke test before B7 cohort): for each strict/candidate subject,
  # run this recipe on full-data (event_indices = all events) and verify the produced
  # rank_a_dense_joint == subject's existing `pairs[0]["rank_a_dense_full"]` (only at
  # joint_valid positions). Mismatch → recipe wrong, stop. Equivalent firm-test as
  # Phase 2 B1 calibration 0/23.

Step 4 — swap-k endpoint node set (v4 catch lock, NOT compute_local_rank_endpoint):
  swap_k_endpoint_channels = derive_swap_endpoint(
      channel_names=channel_names_joint,        # joint_valid 后的 channel names
      rank_a_dense=rank_a_dense_joint,           # joint_valid 内的 dense rank
      decision_k=decision_k_window,
  )
  # Returns list of channel names; convert to indices via name_to_idx (using ORIGINAL
  # channel_names, not joint_valid 子集) for coord lookup

Step 4b — split swap-k into source side + sink side:
  # derive_swap_endpoint internal: top decision_k by rank_a_dense ascending = source;
  # bottom decision_k = sink (per §8 main figure swap-marker convention)
  order_joint = np.argsort(rank_a_dense_joint, kind="stable")
  source_side_joint = order_joint[:decision_k_window]
  sink_side_joint = order_joint[-decision_k_window:]
  source_endpoint_names = [channel_names_joint[i] for i in source_side_joint]
  sink_endpoint_names = [channel_names_joint[i] for i in sink_side_joint]

Step 5 — spatial radius per side (ignore MEB):
  source_radius = compute_endpoint_spatial_radius(
      [name_to_idx[n] for n in source_endpoint_names], coords
  )  # {centroid_rms, mean_pairwise, min_enclosing_radius (ignored), n_points}
  sink_radius = compute_endpoint_spatial_radius(...)

Step 6 — rate:
  rate_window = len(window_events) / window_duration_hours
```

**Notes**:
- 在 v4 catch 之前的 plan 错误用 `compute_local_rank_endpoint`, 那给的是 PR-6 anchoring 风格 fixed-k source/sink (template rank top-k / bottom-k), 不是 swap-k node set。修正后用 `derive_swap_endpoint` (rank-displacement variable-k swap-k 节点) — 这才是 Topic 4 H2 spatial 层 channel-label 来源。
- swap_class_window 可能跟 subject-level full-data swap_class 不同 (per-window 样本量小, swap_sweep p_fw 不稳定)。**不要**用 per-window swap_class 重新分层 — primary cohort 仍按 subject-level full-data swap_class 分层 (strict / candidate / none); per-window swap_class 仅作 diagnostic 字段记录。

### 4.2 Aggregate baseline windows per seizure (v4 catch 2026-05-24 修正 — 不能 median endpoint set)

对每 (subject, seizure, side ∈ {pre, post}) tuple, baseline 聚合**分两类**:

**类 1 — scalar metrics (decision_k, rate, spatial radius)**: median across baseline windows
- `decision_k_baseline = median([decision_k_baseline_i for i in baseline_windows])`
- `rate_baseline = median([rate_baseline_i for i in baseline_windows])`
- `source_centroid_rms_baseline = median([source_radius_i["centroid_rms"] for i in baseline_windows])`
- 同理 source_mean_pairwise / sink_centroid_rms / sink_mean_pairwise

**类 2 — set/structure metrics (swap-k endpoint identity)**: **per-baseline-window 算 Jaccard 再 median**, NOT median endpoint set
- `identity_jaccard_peri_vs_baseline = median([Jaccard(swap_k_peri, swap_k_baseline_i) for i in baseline_windows])`
- `new_node_fraction = median([|swap_k_peri \ swap_k_baseline_i| / |swap_k_peri| for i in baseline_windows])` (peri 中有多少通道不在 baseline 中)

**为什么 set 不能 median**: 通道集合是离散结构, median(channels) 数学上未定义 (不能"中位数化一组集合"); 必须先 per-window 算 set 相似性的 scalar (Jaccard / fraction), 然后这些 scalar 才能聚合 median。

### 4.3 Compute Δmetrics per (subject, seizure, side) — v4 catch 2026-05-24 updated

| # | Metric | Formula | Per (subject, seizure, side) | BH-FDR family |
|---|---|---|---|---|
| 1a | Δdecision_k | `k(peri) − median(k(baseline_i))` | scalar | sign-check only, NOT in BH-FDR |
| 1b | Δdecision_k / baseline_k (normalized) | `(k(peri) − k_base) / max(k_base, 1)` | scalar | sign-check companion of 1a, NOT in BH-FDR |
| 2a | Identity Jaccard | `median([Jaccard(S_peri, S_base_i) for i in baseline_windows])` (per-baseline-window Jaccard 再 median; **不能** median set) | scalar (lower = more recruitment) | **YES — primary BH-FDR** |
| 2b | new_node_fraction (v4 catch 新增) | `median([\|S_peri \ S_base_i\| / \|S_peri\| for i in baseline_windows])` | scalar (higher = more recruitment) | **YES — primary BH-FDR** |
| 3a | Δsource centroid RMS | `source_rms(peri) − median(source_rms(baseline_i))` | scalar | **YES** |
| 3b | Δsource mean pairwise | same | scalar | **YES** |
| 4a | Δsink centroid RMS | same | scalar | **YES** |
| 4b | Δsink mean pairwise | same | scalar | **YES** |
| 5 | Rate Δ | `rate(peri) − median(rate(baseline_i))` (events/h) | scalar | **NO — secondary descriptive only** |

**BH-FDR family** per side (pre 和 post 各一个 family): **6 primary tests** = {2a, 2b, 3a, 3b, 4a, 4b}; 1a + 1b 是 sign-check (concordance) 不入 BH-FDR; 5 (rate Δ) 是 secondary 永远不入 BH-FDR。

### 4.4 Per-seizure record (output schema)

```json
{
  "subject": "yuquan_xxx",
  "seizure_id": "seizure_0",
  "seizure_onset_t": <epoch float>,
  "seizure_end_t": <epoch float>,
  "swap_class_full_data": "strict",
  "global_decision_k": 7,
  "pre_ictal": {
    "peri_window": {"t_start": ..., "t_end": ..., "n_events": ...},
    "n_baseline_windows": 5,
    "metrics": {
      "decision_k_peri": 9,
      "decision_k_baseline_median": 7,
      "delta_decision_k": 2,
      "delta_decision_k_normalized": 0.286,
      "jaccard_peri_vs_baseline": 0.45,
      "source_centroid_rms_peri": 14.2,
      "source_centroid_rms_baseline_median": 9.8,
      "delta_source_centroid_rms": 4.4,
      "source_mean_pairwise_peri": 21.5,
      "source_mean_pairwise_baseline_median": 15.9,
      "delta_source_mean_pairwise": 5.6,
      "sink_centroid_rms_peri": ...,
      "sink_centroid_rms_baseline_median": ...,
      "delta_sink_centroid_rms": ...,
      "sink_mean_pairwise_peri": ...,
      "sink_mean_pairwise_baseline_median": ...,
      "delta_sink_mean_pairwise": ...,
      "rate_peri": ...,
      "rate_baseline_median": ...,
      "delta_rate": ...
    },
    "exit_reason": "ok"
  },
  "post_ictal": { ... same structure ... }
}
```

---

## 5. Cohort inference: cluster-robust SE primary, mixed model sensitivity

### 5.1 Primary inference (v4 catch 2026-05-24 lock — cluster-robust SE + one-sided directional p + primary cohort = strict ∪ candidate)

**Primary cohort filter**: `df_primary = df[df['swap_class_full_data'].isin(['strict', 'candidate'])]` (排除 none 子集; v4 catch lock)。

对每 primary metric m ∈ {Jaccard, new_node_fraction, source RMS Δ, source pairwise Δ, sink RMS Δ, sink pairwise Δ} (6 个 BH-FDR primary, 不含 Δk 1a/1b sign-check):

```python
import statsmodels.api as sm

# y = Δmetric per seizure (long format)
# direction[m] = +1 for ↑ (radius, new_node_fraction) or -1 for ↓ (Jaccard)
# rows = seizures (strict ∪ candidate cohort only), grouping by subject_id
y = df_primary[m].values
X = np.ones_like(y)
model = sm.OLS(y, X)
result = model.fit(cov_type='cluster', cov_kwds={'groups': df_primary['subject_id']})

# One-sided directional p (v4 catch lock):
mean_est = result.params[0]
se = result.bse[0]
t_stat = mean_est / se
if direction[m] > 0:
    # one-sided: testing mean > 0
    p_one_sided = 1 - scipy.stats.t.cdf(t_stat, df=result.df_resid)
else:
    # direction[m] < 0 (Jaccard): testing mean < 0
    p_one_sided = scipy.stats.t.cdf(t_stat, df=result.df_resid)

ci_95 = result.conf_int(alpha=0.05)[0]
```

BH-FDR q<0.10 across **6 primary metrics per side** (pre-ictal 一个 family, post-ictal 另一个 family — v4 catch lock: pre/post **不**合并 BH-FDR family)。

**Δdecision_k (1a) 和 normalized Δk (1b) sign-check**: 不入 BH-FDR family, 单独报 `Δk_median + sign(Δk) == sign(Δk_normalized)`。

### 5.2 Sensitivity inference 1 — wild cluster bootstrap (required when n_qualifying_subjects < 10, advisor catch 3 2026-05-24 lock)

primary cohort = strict ∪ candidate 可能 < 10 (Stage B 数字: 5 strict + 4 candidate = 9 subjects)。**当 n_qualifying_subjects < 10, 必须跑 wild cluster bootstrap as sensitivity**。

**算法 (Cameron-Gelbach-Miller 2008, "Bootstrap-Based Improvements for Inference with Clustered Errors", procedure 1 — null-imposed wild cluster bootstrap with Rademacher weights)**:

```python
# Per metric m, per side (pre / post), primary cohort = strict ∪ candidate only

# Step 1 — Fit UNrestricted model + extract observed t-stat (one-sided, direction[m]):
X = np.ones(n)              # intercept only (test H0: mean Δm == 0)
y = df_primary[m].values
result_obs = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': subjects})
beta_obs = result_obs.params[0]
se_obs = result_obs.bse[0]
t_obs = beta_obs / se_obs

# Step 2 — Fit RESTRICTED model (mean = 0 imposed):
# Under H0 (mean=0), restricted fit gives β_R = 0, residuals = y itself
residuals_restricted = y.copy()  # since intercept-only restricted is y - 0

# Step 3 — Bootstrap loop with Rademacher cluster-level weights:
n_boot = 2000
t_boot = np.zeros(n_boot)
rng = np.random.default_rng(seed)
for b in range(n_boot):
    # Sample one Rademacher sign per CLUSTER (not per observation):
    cluster_signs = rng.choice([-1.0, 1.0], size=n_subjects)
    # Map cluster sign back to each observation by subject_id:
    obs_signs = np.array([cluster_signs[subject_to_idx[s]] for s in subjects])
    # Generate bootstrap y under H0 by flipping restricted residuals:
    y_boot = obs_signs * residuals_restricted
    # Refit with cluster-robust SE on bootstrap sample:
    result_b = sm.OLS(y_boot, X).fit(cov_type='cluster', cov_kwds={'groups': subjects})
    t_boot[b] = result_b.params[0] / result_b.bse[0]

# Step 4 — Compute one-sided p (matching direction lock from §5.1):
if direction[m] > 0:
    p_wild_one_sided = (np.sum(t_boot >= t_obs) + 1) / (n_boot + 1)
else:  # direction[m] < 0 (Jaccard)
    p_wild_one_sided = (np.sum(t_boot <= t_obs) + 1) / (n_boot + 1)
```

**关键 — 为什么 restricted residuals**: CGM 2008 §2.2 证明 unrestricted residuals 在 small-cluster 场景下让 bootstrap 分布太宽 (size distortion); restricted residuals (imposing H0) 给正确 size。对 n≈9 clusters 这是 must, 不是 nice-to-have。

**Library options**:
- 手写如上 (Phase 3 v0 用这个, 完全 transparent + reproducible)
- `wildboottestpy` (PyPI, 同 CGM 2008 实现; B4 实施时 evaluate 是否引入依赖)

primary p 和 wild bootstrap p 方向相反 → flag, primary 以 cluster-robust SE 为准 (user-locked); 两个同向 → 增强 confidence。两者都进 archive `cohort_summary.json::verdicts.pre_ictal/post_ictal.sensitivity_wild_bootstrap`。

### 5.3 Sensitivity inference 2 — mixed-effect model

```python
import statsmodels.formula.api as smf
md = smf.mixedlm(f"{m} ~ 1", df_primary, groups=df_primary['subject_id']).fit()
```

primary vs mixed model p 方向相反 → flag; primary 以 cluster-robust SE 为准。

### 5.4 Stratification (mandatory)

主 verdict 报 pre-ictal + post-ictal **各自独立** 的 verdict (两个独立 BH-FDR families)。然后:
- 按 seizure type (focal / generalized / focal-to-bilateral) 分层报
- 按 inter-seizure interval (短 / 中 / 长 — bins TBD by B0 audit) 分层报
- 按 swap_class (strict vs candidate) 分层报 (primary cohort 内细分; none 已经在 §5.5 negative control 独立报)

每层独立 verdict, **不**collapse 跨层。

### 5.5 swap_class subset 分析 (v4 catch lock)

- **strict ∪ candidate 子集 (primary)**: §5.1 主分析跑在这子集上
- **none 子集 (negative control)**: 独立跑同一 metric 集合; 期望全部 NULL (Δk ≈ 0, Jaccard ≈ baseline, radius 无显著 Δ); 出现显著 Δk 或 radius expansion → 红 flag (measurement artifact 或 baseline 不 matched 或 SEF-ITP 在 none 子集也有信号 — 后者颠覆 swap_class 作为 primary inclusion criterion 的假设)
- **all-cohort (sensitivity)**: §5.1 主分析也跑全 23 cohort 一遍; 主 verdict 仍以 strict ∪ candidate 为准, all-cohort verdict 作 sensitivity report (检查 primary cohort restriction 是否过严)

---

## 6. Verdict logic (user-locked 2026-05-24)

### 6.1 Cohort-level verdict per side (pre-ictal / post-ictal 各 verdict, v4 catch lock)

```python
def phase3_verdict(
    p_one_sided: Dict[str, float],   # 6 BH-FDR primary tests; key in {jaccard, new_node_fraction,
                                      # source_rms, source_pairwise, sink_rms, sink_pairwise}
    delta_k_median: float,            # Δk_median (raw)
    delta_k_normalized_median: float, # Δk/baseline_k median
    n_qualifying_subjects: int,
    n_qualifying_seizures: int,
):
    """v4 catch 2026-05-24 lock — verdict logic."""
    # Underpowered gate
    if n_qualifying_subjects < 6:
        return "UNDERPOWERED"

    # BH-FDR across 6 primary one-sided directional p
    p_corrected = bh_fdr(p_one_sided, q=0.10)  # 6 tests per side

    # Direction lock (one-sided p built in; significance = corrected < 0.10 AND direction matches)
    identity_recruitment_sig = (
        p_corrected['jaccard'] < 0.10  # Jaccard ↓ direction baked in
        OR p_corrected['new_node_fraction'] < 0.10  # new_node_fraction ↑ direction baked in
    )
    radius_expansion_sig = any(
        p_corrected[m] < 0.10 for m in
        ['source_rms', 'source_pairwise', 'sink_rms', 'sink_pairwise']
    )

    # Core size (Δk) sign-check: NOT a BH-FDR vote; sign-agreement only
    delta_k_concordant = (
        delta_k_median > 0
        AND np.sign(delta_k_median) == np.sign(delta_k_normalized_median)
    )

    # Reverse-direction FAIL branches (one-sided reverse p; use 1 - p_one_sided)
    # for FAIL, switch direction sign and re-test (or use two-sided p as auxiliary):
    # if identity Jaccard signals strongly POSITIVE (identity 收紧, opposite of recruitment):
    p_reverse_jaccard = 1 - p_one_sided['jaccard']  # one-sided reverse direction
    # similar for other metrics
    if bh_fdr({'jaccard_reverse': p_reverse_jaccard, ...})['jaccard_reverse'] < 0.10:
        return "FAIL_IDENTITY_CONTRACTION"
    # similar branches for radius contraction (one-sided reverse direction)

    # SUPPORTED (v4 catch lock — identity recruitment OR radius expansion + Δk concordant)
    if (identity_recruitment_sig OR radius_expansion_sig) AND delta_k_concordant:
        return "SUPPORTED"

    # SUPPORTED-but-k-trend (identity recruitment OR radius expansion, but Δk not concordant)
    if (identity_recruitment_sig OR radius_expansion_sig) AND NOT delta_k_concordant:
        return "SUPPORTED_WITHOUT_K_CONCORDANCE"
        # 措辞: "identity recruitment / radius expansion 显著, 但核心数量 (Δk) 不显著 ↑;
        #        with concordant k trend cannot be claimed"

    return "NULL"
```

**关键 framing 锁 (v4 catch 2026-05-24, 防止 over-read)**:
- ❌ "Δk 不显著但 identity Jaccard 下降 → 核心数量增加" — Δk 不显著就不能这样写, 只能写 "识别招募显著, 核心数量趋势同向 (descriptive concordance)"
- ❌ "Δk 显著但 identity / radius 都不显著 → SUPPORTED" — 单 Δk 不够 (Δk 是 noisy scalar, identity 或 radius 是 SEF-ITP 的真正 discriminative claim)
- ✅ "swap-k node 招募 (identity Jaccard 显著 ↓ 或 new_node_fraction 显著 ↑) 或 空间半径扩张 至少一项显著, AND Δk median > 0 与 normalized Δk 同方向" — SUPPORTED
- ✅ "如 Δk 不显著只方向同向, 写 'with concordant k trend', 不能写 '核心数量显著增加'" (user-locked 2026-05-24)

### 6.2 Reporting alongside

不论 cohort verdict 是什么, **都报**:
- rate Δ verdict (cluster-robust p, direction) — descriptive secondary, 不入 SUPPORTED gate
- per-subject heterogeneity diagnostic (median + IQR + outlier subject list)
- per-stratum verdict (seizure type / interval / swap_class)

### 6.3 SUPPORTED + SOZ relation (secondary, only if SUPPORTED)

如果 cohort SUPPORTED, 在显著的 recruitment 通道集合上查:
- 这些通道是否更靠近 clinical SOZ?
- 是否更靠近 data-driven SOZ (PR-T3-1 Layer B, 当前不存在, 待 PR-T3-1 Layer B 启动)?
- 是否重叠 ictal early propagation channel (Topic 3 PR-T3-1 Layer A)?
- 个体 DTI 可用时 SC out-degree 关系?

**两套 SOZ 标签都查, 不融合**; **禁止**在 UNDERPOWERED / NULL / FAIL case 上做 SOZ 关系分析。

---

## 7. 实现 (`src/sef_itp_phase3.py`)

### 7.1 模块组织

```
src/sef_itp_phase3.py
├── SubjectPhase3Data dataclass
├── _load_seizure_times(dataset, subject_id) — Yuquan + Epilepsiae 统一接口
├── _enumerate_peri_ictal_windows(seizure_times, pre_minutes=55, post_minutes=55, buffer_minutes=5)
├── _pick_matched_baseline_windows(seizure_times, peri_window, hour_tolerance=2, min_separation_hours=12, n_baseline_min=5)
├── compute_window_metrics(window, event_abs_times, labels, bools, ranks, coords, primary_pair) — 拿 (decision_k, swap_k_endpoint, radius_source, radius_sink, rate) tuple
├── compute_delta_metrics(peri_record, baseline_records) — per-seizure side
├── compute_cohort_cluster_robust_inference(df_per_seizure, metric_names, group_col='subject_id')
├── compute_cohort_mixed_model_sensitivity(df_per_seizure, metric_names, group_col='subject_id')
├── compute_phase3_verdict(p_values, directions, n_qualifying_subjects, alpha=0.10) — BH-FDR + 6.1 logic
├── enumerate_stratifications(df_per_seizure) — seizure_type / swap_class / ISI stratify
```

### 7.2 Runner

```
scripts/run_sef_itp_phase3.py
├── --subject yuquan_<sid> | --all
├── --window-side pre | post | both (default both, 各自独立 verdict)
├── --include-isi (default off; Phase 3 v0 主线只用 peri-ictal)
├── 输出 per-subject JSON: results/topic4_sef_itp/phase3_ictal_adjacent/per_subject/<dataset>_<sid>.json
├── 输出 cohort summary: results/topic4_sef_itp/phase3_ictal_adjacent/cohort_summary.json
└── 输出 figures README per AGENTS.md 中文规范
```

### 7.3 Per-subject JSON schema (proposed)

```json
{
  "schema_version": "sef_itp_phase3_v1_2026_05_XX",
  "dataset": "yuquan",
  "subject_id": "...",
  "swap_class_full_data": "strict",
  "global_decision_k": 7,
  "n_seizures_total": 4,
  "n_seizures_qualifying": 3,
  "per_seizure": [...],  // see §4.4
  "exit_reason": "ok"
}
```

### 7.4 Cohort summary JSON schema

```json
{
  "schema_version": "sef_itp_phase3_cohort_v1_2026_05_XX",
  "n_subjects_total": 23,
  "n_subjects_qualifying": <N>,
  "n_seizures_total": <S>,
  "n_seizures_qualifying": <S_qual>,
  "verdicts": {
    "pre_ictal": {
      "primary_cluster_robust": {
        "delta_k": {"p": ..., "ci": [...], "direction": ...},
        "delta_k_normalized": {...},
        "jaccard": {...},
        "source_centroid_rms": {...},
        ...
      },
      "primary_bh_fdr": {...},
      "sensitivity_mixed_model": {...},
      "integrated_verdict": "SUPPORTED" | "NULL" | "FAIL_IDENTITY_CONTRACTION" | "FAIL_RADIUS_CONTRACTION" | "UNDERPOWERED"
    },
    "post_ictal": {... same structure ...}
  },
  "stratifications": {
    "by_seizure_type": {...},
    "by_swap_class": {"strict": {...}, "candidate": {...}, "none": {...}},
    "by_isi": {...}
  },
  "rate_secondary": {...}  // never enters integrated_verdict
}
```

---

## 8. TDD plan

### 8.1 Unit tests (`tests/test_sef_itp_phase3.py`)

```python
def test_enumerate_peri_ictal_windows_basic()
def test_enumerate_peri_ictal_windows_overlap_buffer()
def test_pick_matched_baseline_excludes_within_12h_of_seizure()
def test_pick_matched_baseline_hour_of_day_tolerance()
def test_pick_matched_baseline_insufficient_returns_empty()
def test_compute_window_metrics_returns_expected_schema()
def test_compute_delta_metrics_median_baseline()
def test_cluster_robust_se_handles_subject_clustering()  # synthetic 30 seizures across 6 subjects with within-subject correlation
def test_mixed_model_sensitivity_matches_cluster_robust_direction()  # both should agree on direction
def test_phase3_verdict_supported_recruitment_only()  # Jaccard ↓ + Δk consistent
def test_phase3_verdict_supported_radius_only()  # radius ↑ + Δk consistent
def test_phase3_verdict_null_no_signal()
def test_phase3_verdict_fail_identity_contraction()
def test_phase3_verdict_fail_radius_contraction()
def test_phase3_verdict_underpowered_below_6_subjects()
def test_phase3_verdict_delta_k_sign_disagreement_blocks_supported()  # 不 SUPPORTED if Δk and normalized Δk 异号
def test_stratify_by_swap_class_negative_control_check()  # none 子集出现显著 Δk → red flag
```

### 8.2 Integration tests

- End-to-end on 1 strict subject with ≥2 qualifying seizures, verify JSON schema + verdict logic
- End-to-end on 1 candidate subject
- End-to-end on 1 none subject (verify negative-control behavior)

---

## 9. Pending advisor review questions (post-v4 catch reduced)

User-return v4 catch 2026-05-24 已 lock 大部分原 question. 剩余需要 advisor review 的项:

1. **Wild cluster bootstrap 具体形式**: Rademacher weights (±1) vs Mammen weights vs Webb's 6-point distribution? 哪个对 n≈9 strict ∪ candidate cohort 最 power-stable? statsmodels 没有 built-in wild bootstrap, 需要手写 — advisor review 实现细节。

2. **Peri-ictal window 5min 缓冲是否足够**: PR-1 seizure detection 精度有限 (Yuquan 算法 vs Epilepsiae SQL clinical onset 标签精度不同), 5min buffer 在 Epilepsiae 上够吗? B0 audit 是否需要 per-subject manual onset 校准 sub-step?

3. **identity Jaccard / new_node_fraction effect size floor**: 当前 verdict 只用 BH-FDR q<0.10 + direction, 没设 effect size floor。peri baseline Jaccard 本来就 < 1 (window-to-window noise), 是否需要 Cohen's d ≥ 0.3 on `1 - Jaccard` 作 floor? (建议 advisor 给意见; floor 加严减少 false positive, 但小 cohort 上可能 inflate underpowered)

4. **B0 audit 后 lock 阈值的决策规则**: B0 audit 输出 cohort 分布, plan §3.3 说 "audit 后才决定 min_events_per_window"。具体决策规则是什么? (建议: median × 0.5? p25? 还是 advisor 给定 absolute floor 如 50?)

5. **swap_class per-seizure vs subject-level (旧 Q5 sustain)**: 一个 strict 病人某次 seizure 的 peri 窗 swap signal 弱 — 是否需要 per-seizure swap_class 重新评估? (建议第一版用 subject-level, B7 cohort run 后再考虑 per-seizure refinement)

6. **ISI analysis 启动条件**: peri-ictal v0 跑通后才启 ISI; 但 ISI window 长度 (60 min sliding? 30 min sliding?) 和 baseline matching 规则 (同样 hour-of-day ± 2h ≥ 12h to any seizure?) 是否照搬 peri-ictal? advisor review。

7. **Sensitivity inference 优先级**: §5.2 wild cluster bootstrap + §5.3 mixed model + §5.4 stratify by seizure type + §5.5 negative control + sensitivity all-cohort — 跑哪个用什么 budget? (建议: B7 cohort run 先 all sensitivity 跑一遍 + 全部 stratification, archive 写时按 advisor 决定哪些进 verdict main, 哪些 sensitivity-only)

8. **Sequence advisor on B0 audit results**: B0 audit 跑完后再调一次 advisor sign-off lock 阈值 + lock primary cohort N (strict + candidate 实际 qualifying subjects 数), 再启 B1-B7 主实施。这是双重 advisor gate (B0 audit 前 + B0 audit 后) — advisor agree (2026-05-24)。三 gate 节奏 (§9 sign-off + post-B0 audit + post-B7 cohort run) advisor 同意是正确纪律, 不该 fold。

### User-decision items (advisor 2026-05-24 surfaced; §6.1 verdict commits 前必须 user ratify)

9. **Identity recruitment: Jaccard OR new_node_fraction vs AND with sign concordance?** User 原话 "Jaccard 下降 + new_node_fraction 上升, 比单独 Jaccard 更像招募" 在两种语义间 ambiguous:
   - (a) OR is fine — both metrics together is just stronger evidence, 一个 BH-FDR 显著 即可触发 identity recruitment branch (当前 plan §6.1 用此)
   - (b) AND with sign concordance — 必须 Jaccard 显著 ↓ **AND** new_node_fraction 显著 ↑ 才算 identity recruitment, 防止单 metric noise driven
   - 需要 user 拍板。建议 (a) — 但接受 (b) 如果 user 希望 stricter specificity。

10. **Pre-ictal vs post-ictal integration: 报独立 verdict 还是合成 Phase 3 整体 verdict?** SEF-ITP biophysics 在 pre vs post 不同 (build-up vs recovery/refractory):
    - (a) **Either side SUPPORTED → Phase 3 SUPPORTED** (broad recruitment signal anywhere)
    - (b) **Both required → SUPPORTED; 一边 SUPPORTED → PARTIAL_SUPPORTED**
    - (c) **Strictly independent — 不合 Phase 3 整体 label, 只报 pre/post 各自 verdict**
    - 当前 plan §6.1 implicit 用 (c), 但 archive doc 写时往往会想 collapse 成一个判读。需要 user 拍板。

---

## 10. Pending implementation gates (must clear before TDD starts)

- [ ] advisor review on §9 questions
- [ ] user ratify on §6 verdict logic (尤其 "至少一项 identity recruitment OR radius expansion + Δk 方向一致" 阈值)
- [ ] Phase 2 cohort_run_2026-05-24.md 收口 + advisor sign-off (independent gate)
- [ ] Phase 2 framework v1.0.7 banner lock 确认无 dangling
- [ ] confirm `statsmodels` available in project env (check requirements.txt; if absent, add)

---

## 11. Codename mapping (CLAUDE.md §8 朴素话 — 朴素描述在 §0-§6, 这里只是 reverse lookup)

| 代号 | 朴素话 |
|---|---|
| swap-k | "在两个反向模板间换位的通道集合" (来自 PR-6 supplementary rank-displacement swap_sweep) |
| decision_k | "该 subject 自适应最稳的 swap-endpoint 通道数 (核心区数量估计)" |
| peri-ictal window | "发作邻近的时段 (前 5-60 分钟 / 后 5-60 分钟)" |
| baseline window | "同 subject 同 hour-of-day ± 2h、距任何 seizure ≥ 12h 的对照时段" |
| Δdecision_k | "peri 窗 decision_k 减去 baseline 中位 decision_k" (core 招募量) |
| Δdecision_k / baseline_k | "Δdecision_k 相对于 baseline_k 的比例" (normalized effect, 跨 subject 可比) |
| identity Jaccard ↓ | "peri 窗 swap-k 通道集合跟 baseline 重叠下降" (招募新通道) |
| spatial radius expansion | "peri 窗 source / sink 通道空间分布比 baseline 更分散" (空间扩张) |
| cluster-robust SE | "考虑同一 subject 多次 seizure 之间非独立的标准误估计" (sandwich estimator) |
| ISI | inter-seizure interval (两次 seizure 之间的时段) |
| qualifying subject | "至少有 1 次 seizure 满足所有 data gate (n_events / baseline 充足)" |
| BH-FDR | Benjamini-Hochberg false discovery rate 多重比较校正 |

---

## 12. 实施序列 (v4 catch 2026-05-24 — 双重 advisor gate + B0 audit first)

**Sequence (after advisor sign-off on §9 questions before B0 starts)**:

0. **(pending) advisor pass 1**: §9 review questions sign-off → locked spec 更新进 framework v1.0.7 §3.5 amendment (append "Phase 3 v0 spec lock 2026-05-XX")

1. **B0 — eligibility audit (NEW, v4 catch lock)**:
   - 写 `scripts/phase3_b0_eligibility_audit.py` — 不跑主统计, 只统计 per-subject seizure / window / events 分布
   - 输出 `results/topic4_sef_itp/phase3_ictal_adjacent/diagnostics/b0_eligibility_audit_<date>.csv` + summary 文本
   - **gate 内容**: per-subject (n_seizures_total, n_qualifying_peri_windows_pre, n_qualifying_peri_windows_post, n_matched_baselines_distribution); per-window (n_events_total, n_events_per_cluster_a, n_events_per_cluster_b 中位 + IQR + p25); per-channel n_events 分布

2. **(pending) advisor pass 2**: B0 audit 结果出来后 advisor sign-off lock data gate 阈值 (`min_events_per_window`, `min_events_per_cluster`, channel-level n_events 阈值 / 是否保留作 sensitivity), lock primary cohort qualifying N (strict + candidate 实际 qualifying subjects 数)

3. **B1**: TDD `_load_seizure_times` (Yuquan PR-1 JSON + Epilepsiae SQL 统一接口) + `_enumerate_peri_ictal_windows` + `_pick_matched_baseline_windows` (unit tests + 1 real subject smoke)

4. **B2**: TDD `compute_window_metrics` — per-window 4-step pipeline (§4.1: `_per_cluster_template_rank` → `compute_swap_score_sweep` → `derive_swap_endpoint` → `compute_endpoint_spatial_radius` ignore MEB)

5. **B3**: TDD `compute_delta_metrics_per_seizure` — per-baseline-window Jaccard / new_node_fraction 再 median, scalar metrics median 聚合 (§4.2 v4 catch lock)

6. **B4**: TDD `compute_cohort_cluster_robust_inference` (primary, one-sided directional p, BH-FDR per side) + sensitivity wild cluster bootstrap (if n<10) + sensitivity mixed model

7. **B5**: TDD `compute_phase3_verdict` (§6.1 all branches: SUPPORTED / SUPPORTED_WITHOUT_K_CONCORDANCE / NULL / FAIL_IDENTITY_CONTRACTION / FAIL_RADIUS_CONTRACTION / UNDERPOWERED)

8. **B6**: runner + summarizer + figures README per AGENTS.md 中文规范

9. **B7**: cohort rerun (primary cohort strict ∪ candidate + sensitivity all-cohort + negative control none) + new archive `docs/archive/topic4/sef_itp_phase3/cohort_run_<date>.md`

10. **(pending) advisor pass 3**: B7 cohort run 后 archive 落字前 advisor 校准 cohort 解读 (防止 over-read), 检查 sensitivity vs primary 方向一致, 检查 stratification 内部一致

11. **B8**: framework banner v1.0.8 (Phase 3 收口 marker) + Topic 4 main doc cross-cite + paper_overview Topic 4 行 update
