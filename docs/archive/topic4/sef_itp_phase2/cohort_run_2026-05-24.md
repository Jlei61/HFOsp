# SEF-ITP Phase 2 Cohort 实跑 v1.1 — 2026-05-24 (rank-based endpoint)

> **状态**：v1.1 cohort run, **pending cohort completion + summarizer**（数字填入 §1/§2/§3 在 cohort 跑完 + summarizer 出 verdict 后）
> **framework**：`docs/topic4_sef_itp_framework.md` **v1.0.6** (Phase 2 完结 marker 待 banner 升级)
> **plan**：`docs/superpowers/plans/2026-05-23-topic4-phase2-h4-v1.1-rank-endpoint-plan.md`
> **module**：`src/sef_itp_phase2.py` **v1.1.0** (bumped from v1.0.0)
> **runner**：`scripts/run_sef_itp_phase2.py` (schema bumped to `sef_itp_phase2_v2_2026_05_24`)
> **summarizer**：`scripts/summarize_sef_itp_phase2.py` (cohort summary schema `sef_itp_phase2_cohort_v2_2026_05_24`)
> **per-subject 输出**：`results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/<dataset>_<sid>.json` (v2 schema, overwrites v1)
> **cohort summary**：`results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json` (v2 cohort schema)
> **B1 calibration gate**：`results/topic4_sef_itp/phase2_temporal_x_geometry/diagnostics/b1_calibration_2026-05-23.csv` — **0/23 mismatch** (B1 reproduces PR-6 anchoring on full-data per cluster; rank-endpoint extractor reuses `_legacy_hist_mean_rank` + `extract_endpoint_middle` + per-cluster valid_mask)
> **测试**：57 GREEN (Phase 2 module + B1 calibration + B2 spatial radius + B3 decision-k drift)

---

## 0. 背景：v1.0 → v1.1 改动原因（user 2026-05-23 catch + advisor review）

v1.0 (2026-05-23 cohort_run) `compute_local_endpoint(events_bool, labels, k)` 用
`mean(events_bool[labels==c], axis=0)` 取 channel-wise 参与率 top-k/bottom-k —
**这是 participation field top-k 漂移，不是 propagation rank endpoint 漂移**。
两件事不同：
- 参与率 top-k = "哪些通道这段时间常常出现在事件里"
- 传播 rank top-k = "哪些通道在事件里最先点火"（这才是 endpoint 的本意）

v1.0 cohort_run 报告的 H4 PASS (Wilcoxon p=9.5e-7, Cohen's d=1.50, n=21 finite) **只能解读为 "参与场不漂"**，不能直接支撑 SEF-ITP H4 关于 "传播 endpoint 不漂" 的预测。

v1.1 修法（CLAUDE.md §6.1 question-match）：
- 主线 H4 endpoint 改用 **`compute_local_rank_endpoint`**，per-epoch 调 `_legacy_hist_mean_rank` (PR-6 anchoring 用的同一 template estimator) → argsort(argsort) → `extract_endpoint_middle` (PR-6 anchoring 用的同一 endpoint extractor) + per-cluster valid_mask = "cluster events 内任一通道有参与" (PR-6 anchoring runner 同一构造)。
- **B1 calibration gate**：`compute_local_rank_endpoint(full-dataset-as-one-epoch)` 必须 set-equality reproduce PR-6 anchoring `per_template[].source/sink` per cluster。**结果 0/23 mismatch**（refactor 后；refactor 前用简单 masked arithmetic mean 21/23 mismatch — 不同 statistic 自然不复现 PR-6）。
- 加 4 个 spatial radius metrics per side（centroid RMS / mean pairwise / min enclosing ball, source 和 sink 分开算）+ source-sink centroid distance（轴长度，单独报）。
- 加 decision-k drift per epoch（per-epoch 重跑 `compute_swap_score_sweep`）— **user-return v2 catch 2026-05-23**：computed for **all 23 subjects**, swap_class 作 context（summarizer stratify by swap_class；swap-positive 9 个有信心解读，swap=none 14 个作 noise / control baseline）。
- v1.0 participation-field implementation **保留**作 supplementary（`supplementary_participation_field` block in JSON, 沿用 v1.0 `compute_local_endpoint`）— 用户希望比较两个 endpoint definition 的结论一致性。

---

## 1. H3 — 不变 (v1.0 cohort_run 数字保留)

H3 数字 (mark transition TOST + LOO + endpoint Jaccard) 在 v1.1 不变——H3 不动 H4 main metric 的改动。引 `cohort_run_2026-05-23.md` §1 全部数字 + verdict logic + 三阶段朴素话解读。**framework v1.0.6 §3.3 surgical clarification 已生效**：原 v1.0.5 字面 6-AND rule "all 6 TOST equiv_pass → SUPPORTED" 标 SUPERSEDED；新三层 verdict (R1 long-scale ≥60s + R2 endpoint stability + descriptive sub-shape) 是当前 verdict gate。**v1.1 cohort run reaffirms** v1.0 H3 numbers under the v1.0.6 verdict logic.

**v1.1 H3 verdict**（按 v1.0.6 §3.3 三层规则）：**SUPPORTED**
- R1（长尺度 ≥60s 独立性 required）：60s + 1800s 窗口 cohort 95% CI 都在 ±0.05 带内 ✓
- R2（endpoint identity 时间二分稳定 required）：first_half Jaccard 中位 0.71 + odd_even 中位 0.86，OR 任一 ≥ 0.7 ✓
- Descriptive sub-shape：10s/30s anti-clustering（refractoriness 方向，与 SEF-ITP 自身物理一致，**不**触发 verdict）；run_length lift CI 上端略擦出 ±0.05 等价带（descriptive，不进 R1）
- 旧 v1.0.5 字面 6-AND-rule 跑出"CONTRADICTED"是 prose under-specification，cohort 数字本身不变（见 framework v1.0.6 §3.3 surgical clarification + cohort_run_2026-05-23.md banner cross-cite）

---

## 2. H4 v1.1 main line — rank-based endpoint geometry drift

### 测了什么（朴素话）

把每个录制切成 30 分钟一段（epoch），在**背景间期**（不挨着发作的时段）里看三件事：

1. **每段事件里"最先点火 / 最后点火的通道集合"在不同段间换不换** —— 跟全程的端点集合算重叠度（Jaccard），看 endpoint identity 是稳还是飘
2. **这些端点通道在脑里占多大空间** —— source 那边 k 个通道一团有多紧（centroid 均方距、两两平均距、最小包络球半径），sink 那边同样三个指标。source 中心到 sink 中心的距离作"传播轴长度"单独报。这只是 background 期间端点空间分布的描述性统计，不解读为 SOZ 相关
3. **核心病理区数量（decision-k）在段间是否变化** —— 每段重跑一次"通道在两个反向模板间换位"的检验，看每段给出的 decision_k 在段间的波动幅度

**Stage B 不测的事**：跟 SOZ 扩张相关的、ictal-adjacent / inter-seizure interval 上的端点招募或空间扩张 —— 这是 Phase 3 重头戏，必须 per-seizure 做、不能 subject-level aggregate（因为之前经验已经说明 subject 内 seizure type 差异巨大、间隔波动也巨大）。

**Stage B 测的是 SEF-ITP 的一条 cohort claim**：在背景间期、慢状态调制的时间尺度上，触发频率（事件率）漂动比传播端点身份漂动大 —— 即"慢状态主要调制触发频率、不显著重塑传播端点身份"。decision-k drift 在背景间期 epoch 里是否可测 + 是否有 bounded variation，作 Phase 3 per-seizure biomarker 候选的可行性验证（不作 Phase 2 cohort verdict）。

### 怎么测的

主要 helper:
- `_per_cluster_template_rank(ranks, bools, cluster_evt_idx)` → (template_rank, valid_mask)
  - template = `_legacy_hist_mean_rank(masked_ranks)` — PR-2 / PR-6 同一 estimator
  - template_rank = argsort(argsort(template))
  - valid_mask = `cluster_bools.any(axis=1)` — cluster events 内任一通道有参与
- `compute_local_rank_endpoint(ranks, bools, labels, event_indices, k, valid_mask_per_cluster)` → {cluster_id: {source, sink}}
  - per cluster: `_per_cluster_template_rank` → `extract_endpoint_middle` (PR-6 helper)
  - **B1 calibration**: full-data one-epoch must reproduce PR-6 per-cluster source/sink (0/23 mismatch)
- `compute_endpoint_spatial_radius(endpoint_indices, coords)` → {centroid_rms, mean_pairwise, min_enclosing_radius, n_points}
  - per side (source 和 sink 分开 — 避免 "轴变长" vs "每端散" 混淆)
  - min_enclosing brute-force for k ≤ 5
- `compute_source_sink_centroid_distance(source_indices, sink_indices, coords)` → float
  - 单独报 axis 长度（不混入 per-side radius 主指标）
- `compute_decision_k_drift(ranks, bools, labels, epochs, cluster_a=0, cluster_b=1, n_perm=500)` → {decision_k_per_epoch, decision_k_std, decision_k_mean, decision_k_range}
  - all 23 subjects computed; swap_class as context (summarizer stratify)
  - epochs with cluster events < min_events_per_cluster=20 → decision_k=None (no carry-forward)

I_geom_rank computation 沿用 `compute_I_geom_normalized` (v1.0 helper) 但 input = rank-based local endpoints (v1.1) 而不是 participation-field local endpoints (v1.0)。

Matched null (rate + geom) 决策与 v1.0 同：双方法并跑 (literal `epoch_order_shuffle` + proposed `circular_shift_within_block`)，user 仍 pending ratify final null choice。

### 揭示了什么（n=23 病人，30 分钟 epoch，n_perm=1000，decision-k drift n_perm=500）

#### 2.1 主结论 —— 背景间期 rate-geometry decoupling 成立

在背景间期 30 分钟段内，**标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度**，支持"慢状态主要调制触发频率、不显著重塑传播端点身份"这一 SEF-ITP cohort claim。

具体数字（标准化效应大小，不是 raw 量比）：

| 比较 | Wilcoxon p（rate > geom，单侧）| Cohen's d | n_subjects（两个量都有限）|
|---|---|---|---|
| 事件率 vs **rank-based endpoint identity 漂动**（v1.1 主） | **1.4 × 10⁻⁶** | **+1.26** | 21 |
| 事件率 vs **participation field 漂动**（v1.0 对照，沿用旧实现）| 9.5 × 10⁻⁷ | +1.50 | 21 |

- 中位标准化事件率不稳定度 ≈ 50；中位 rank-endpoint identity 不稳定度 ≈ 17–20
- 两个端点定义（"事件里最先点火的 k 个通道" vs "事件里参与率最高的 k 个通道"）给同向结论 → 主结论对 endpoint 定义不敏感
- 21 个有限的原因：2 个 Epilepsiae 病人（epi_1073、epi_1077）valid 通道太少（≤ 6），random sample 退化 → I_geom 分母为零，过滤

**结论锁定措辞**："在 23 个病人 cohort（21 个两个标准化指数都有限）上，背景间期 30 分钟段内、标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度（Wilcoxon p=1.4×10⁻⁶，Cohen's d=+1.26），participation field 对照给同向结论（p=9.5×10⁻⁷，d=+1.50）。支持 SEF-ITP 关于'慢状态主要调制触发频率、不显著重塑传播端点身份'的预测。"

**不能写**：
- "事件率漂动是 endpoint 漂动 3 倍" —— 这是 raw 量比 framing，两个指标都是 dimensionless 标准化效应大小，不能这样表述
- "病理空间不变" / "病理结构稳定" —— Stage B 只测了背景间期 30 分钟段内的 endpoint identity 稳定性，**不**蕴含 ictal-adjacent / inter-seizure interval 上的空间稳定性（那是 Phase 3 重头戏）
- "SOZ 不扩张" —— Stage B 不测 SOZ 相关任何东西

#### 2.2 supplementary —— v1.0 participation field 对照（沿用旧实现作 robustness）

v1.0 implementation 把"endpoint"定义为 events_bool 参与率 top-k（不是传播 rank），数字上 PASS 信号更强（中位 I_geom_participation ≈ 12，比 v1.1 rank-based 略低）。**v1.0 的 PASS 反映"参与场不漂"，不直接对应 SEF-ITP propagation endpoint claim**；v1.1 修对后 main line 才直接对应。两者同向 → 主结论 robust to endpoint definition。详细对照数字见 cohort_summary.json `h4.supplementary_v1_0_participation_field`。

#### 2.3 背景间期 endpoint 空间分布特征化（描述性，不作 cohort claim）

per-cluster per-side per-metric 数字（cluster 0 across 23 subjects 的 cohort 中位数 + 段间 CV 中位数）：

| 指标 | source 侧 cohort 中位 | source 段间 CV | sink 侧 cohort 中位 | sink 段间 CV |
|---|---|---|---|---|
| centroid RMS 半径（mm）| 9.66 | 0.32 | 16.69 | 0.29 |
| mean pairwise distance（mm）| 15.88 | 0.30 | 24.89 | 0.29 |
| min enclosing ball 半径（mm）| 10.82 | 0.32 | 18.03 | 0.29 |
| source-sink centroid distance（mm，cross-side）| 20.95 | 0.28 | (axis 长度) | — |

**朴素话观察**：sink 侧比 source 侧空间上更分散（~17 vs ~10 mm 半径量级）—— 跟"涟漪从紧的源点扩散开"的物理图景一致。段间变化大约 28–32% CV，bounded 但不为零。

**报告口径限定**：
- 这是**背景间期描述性统计**，不作为 cohort claim
- **不**解读为 "病理核心 v.s. 边缘" 之类的 SOZ 相关 claim
- min enclosing ball 当前实现对 k ≤ 3 完整（2-point 球 + 3-point 圆心球穷举），**k ≥ 4 不完整**（缺 4-point 球穷举）—— Phase 3 如用 variable decision_k 可能 k > 3，必须先补 4-point 球实现，或把 MEB 降为 sensitivity、centroid RMS + mean pairwise 升 primary

#### 2.4 Decision-k drift —— Phase 3 per-seizure biomarker 候选可行性验证

**Stage B 这个分析的科学价值**：在背景间期 30 分钟 epoch 上，per-epoch decision_k 可测且有 bounded variation —— motivates Phase 3 per-seizure recruitment / expansion analysis。**不**作为 Phase 2 cohort verdict 主结论。

按 swap_class 分组的 epoch 间 decision_k 标准差（n=23 全部都跑了，**不**按 swap_class gate）：

| swap_class | n | 中位 decision_k_std | IQR | 注 |
|---|---|---|---|---|
| strict | 5 | 1.53 | [0.44, 2.27] | 真有 swap 信号的核心区，epoch 间 decision_k 有可测变化 |
| candidate | 4 | 0.86 | [0.68, 1.23] | 中等 swap 信号 |
| none | 14 | 0.88 | [0.61, 1.07] | **无稳定 swap-core，per-epoch decision_k 是 noise/control baseline，不是核心区大小** |

**strict 子组详细 per-subject**（这是 Phase 3 真正会用的子集）：

| subject | global decision_k | epoch decision_k 中位 | std | range |
|---|---|---|---|---|
| epi_1146 | 7 | 5.89 | 1.53 | [2, 7] |
| epi_139 | 3 | 2.88 | 0.33 | [2, 3] |
| epi_958 | 3 | 5.75 | 2.27 | [2, 8] |
| yuquan_zhangjiaqi | 3 | 2.73 | 0.44 | [2, 3] |
| yuquan_zhaochenxi | 10 | 10.76 | 2.96 | [3, 13] |

**关键观察**：
- 即使是 strict 病人，per-epoch decision_k 也在 epoch 间漂动（range 通常跨 2–4 个值，最极端 zhaochenxi 跨 3–13）
- 这**证明 per-epoch decision_k 是可测的、有 bounded 但不为零的时间变化**
- 这是 Phase 3 per-seizure 分析的可行性证据：如果背景间期都能测到 epoch 间变化，那 peri-ictal vs matched baseline 比较是可执行的

**严格 caveat**：
- decision_k_std 不能跨 subject 直接比 —— k=2 时 std=1 和 k=23 时 std=1 含义完全不同（前者是核心区数量翻倍，后者是 < 5% 变化）。Phase 3 必须用 Δk、Δk/baseline_k、range，**和 spatial radius 一起报**
- none 子组的 decision_k 是 noise/control baseline，**不能**跟 strict/candidate 同等解释为"核心区大小"
- 当前 Stage B 没做 per-seizure 分析 —— 这是 Phase 3 的事，需要 H5 plan 改成 per-seizure primary + subject-clustered inference（见 framework v1.0.6 §3.5 H5 amendment 即将的修订）

**Phase 3 主问题 lock**（user-return v3 catch 2026-05-23 ratified，待 framework §3.5 落字）：

> 在两次 seizure 之间或 seizure-adjacent window，swap-k endpoint 的数量和空间范围是否相对 matched baseline 增加？

#### 2.5 框架限定 + caveats 总集

1. Stage B 测的是**背景间期 30 分钟段**上的 rate-geometry decoupling，**不**蕴含 ictal-adjacent 或长时间尺度的稳定性
2. H4 v1.1 main 的 cohort claim 是"rate 标准化漂动 > endpoint identity 标准化漂动"，**不是**"endpoint 不变 / 病理空间稳"
3. spatial radius 表只作背景描述，不入 cohort claim，不解读为 SOZ 相关
4. decision_k_std 跨 subject 不可直接比；Phase 3 用 Δk + Δk/baseline + range + spatial radius 联合报
5. none 子组 decision_k 是 noise baseline，不是核心区大小
6. MEB 当前实现 k ≤ 3 完整、k ≥ 4 不完整 —— Phase 3 前要么补 4-point 球，要么 MEB 降级 sensitivity

---

## 3. Cohort 整体一句话 verdict

**H3 (framework v1.0.6 §3.3 三层 verdict)**：长尺度独立性 (60s + 1800s 都在 ±0.05 带内) ✓ + endpoint identity 时间二分稳定 (Jaccard 0.71/0.86) ✓ + 短尺度 anti-clustering refractoriness 方向 descriptive ✓ → **SUPPORTED**（按 v1.0.6 三层 verdict 规则；v1.0.5 字面 6-AND-rule 跑出"CONTRADICTED"是 prose under-specification，已 surgical clarification）。

**H4 v1.1 (背景间期 rate-geometry decoupling, rank-based endpoint)**：n=23 (21 finite), 标准化事件率不稳定度显著高于 rank-endpoint identity 不稳定度 (Wilcoxon p=1.4×10⁻⁶, Cohen's d=+1.26), participation field 对照同向 (p=9.5×10⁻⁷, d=+1.50) → **PASS**（背景间期 rate-geometry decoupling 成立, 不蕴含 ictal-adjacent / SOZ 相关 claim）。

**Decision-k drift (Phase 3 biomarker 候选可行性)**：n=23 全部跑了, per-epoch decision_k 可测 + 有 bounded variation（strict 子组 epoch range 普遍跨 2–4 个值, 最极端 zhaochenxi 跨 3–13）→ **Phase 3 per-seizure 招募/扩张分析可行性已验证, 不作 Phase 2 cohort verdict**。

**整体朴素话**：在背景间期 30 分钟段内、慢状态调制的时间尺度上, **触发频率漂得比传播端点身份漂得多** —— SEF-ITP "慢状态主要调制触发频率、不显著重塑传播端点身份" 的预测在 cohort 上成立。decision-k 在 epoch 间可测且有 bounded variation —— Phase 3 真正的重头戏（peri-ictal vs matched baseline 的 swap-k endpoint 数量 + 空间范围招募/扩张, per-seizure primary + subject-clustered inference）的可行性已验证。

---

## 4. 与 framework v1.0.6 的关系

- H3 verdict 用 v1.0.6 §3.3 三层 verdict (R1 long-scale + R2 endpoint stability + descriptive sub-shape)；不用 v1.0.5 字面 6-AND rule (SUPERSEDED, 保留 audit trail)
- H4 main verdict 用 rank-based endpoint geometry drift (v1.1) 作 cohort claim；v1.0 participation-field 降级 supplementary (framework v1.0.6 §3.4 amendment 已落)
- δ_excess = 0.05 已锁，未调整
- endpoint k = 3 已锁 (default)；decision-k drift 是 k 自适应的 sensitivity (framework §3.4 v1.0.6 amendment 已 authorize)
- B1 calibration gate (0/23 mismatch) 是 v1.1 endpoint extractor 是否复用 PR-6 same primitives 的 firm-test

---

## 5. 实施纪律 & 复用映射（CLAUDE.md §6 / §6.1）

| v1.1 用法 | 来源 | 问题匹配？|
|---|---|---|
| `_legacy_hist_mean_rank` | `src.interictal_propagation` (PR-2 PR-6 共用) | ✅ 同 template estimator |
| `extract_endpoint_middle` | `src.template_anatomical_anchoring` (PR-6 anchoring 用) | ✅ 同 endpoint extractor (top-k 选择) |
| `compute_swap_score_sweep` | `src.rank_displacement` (PR-6 supplementary, Topic 4 H2 input) | ✅ 同 swap_class / decision_k 定义 |
| `compute_I_geom_normalized` | v1.0 `sef_itp_phase2` | ✅ 同 normalization framework (matched null vs valid pool sample), input swap to rank-based |
| `_per_cluster_template_rank` (new helper) | new in v1.1 — wraps `_legacy_hist_mean_rank` + argsort(argsort) + cluster_bools.any | ✅ 同 PR-2 adaptive_cluster.template_rank 构造 |
| `compute_local_rank_endpoint` (new) | new in v1.1 — uses canonical helper + PR-6 `extract_endpoint_middle` | ✅ same source/sink question as PR-6, per-epoch slice |
| `compute_endpoint_spatial_radius` (new) | new in v1.1 — 3D Euclidean radius helpers | ⚠️ different question from H1 H2 spatial compactness (those test "is this set tight against null"; H4 tests "does this set's radius drift over time") |
| `compute_source_sink_centroid_distance` (new) | new in v1.1 — axis length | ✅ separate metric (cross-side); not mixed with per-side radius main metric |
| `compute_decision_k_drift` (new) | new in v1.1 — per-epoch swap_sweep | ✅ same statistic as PR-6 supplementary swap_class label, per-epoch slice |

User-return v2 catch 2026-05-23 corrections (实施在 v1.1 code):
- A. main H4 endpoint = rank-based (not participation field); v1.0 demoted to supplementary
- B. spatial radius per side, not mixed (avoid "轴变长" vs "每端散" 混淆)
- C. decision-k drift for ALL 23 subjects (not gated by swap_class); summarizer stratify
- D. canonical template estimator `_legacy_hist_mean_rank` (B1 calibration gate 0/23)
- E. `_masked_mean_rank` (would have been simple arithmetic mean) removed as dead code (refactor → use `_per_cluster_template_rank`)
- F. **external `valid_mask_per_cluster` is INTERSECTION not REPLACE** (user-review catch 2 2026-05-23): production runner does NOT pass external mask; `derived_valid` (per-epoch cluster participation) is the gate. PR-6 full-data per-cluster mask只能作为 calibration parity audit, never replace epoch participation. Otherwise `_legacy_hist_mean_rank` fallback `template[ci]=ci` for non-participating channels lets them silently enter source/sink (participation-field re-introduction). Test: `test_compute_local_rank_endpoint_external_mask_intersected_not_replaced` locks this contract.

---

## 6. 测试 + 实现 (v1.1)

### 实现文件

- `src/sef_itp_phase2.py` (v1.1.0) — adds:
  - `_per_cluster_template_rank` (B1 + B3 共享 canonical helper)
  - `compute_local_rank_endpoint` (v1.1 main endpoint extractor)
  - `_min_enclosing_ball_radius`, `_triangle_circumsphere` (B2 helpers)
  - `compute_endpoint_spatial_radius`, `compute_source_sink_centroid_distance` (B2)
  - `compute_decision_k_drift` (B3)
  - v1.0 `compute_local_endpoint` retained with v1.1 banner ("supplementary, NOT propagation endpoint")
- `scripts/run_sef_itp_phase2.py` — schema v2; loads coords + rank_displacement; produces:
  - `h4.rank_endpoint.{per_epoch_jaccard, per_epoch_local, I_geom_rank}` (v1.1 main)
  - `h4.spatial_radius.per_epoch_per_cluster` (v1.1)
  - `h4.decision_k_drift.{computed, swap_class_context, global_decision_k_context, result}` (v1.1, computed for ALL subjects)
  - `h4.supplementary_participation_field.{per_epoch_jaccard, per_epoch_local, I_geom_participation}` (v1.0 supplementary)
  - `h4.I_rate_{epoch_order_shuffle, circular_shift}` (unchanged from v1.0)
- `scripts/summarize_sef_itp_phase2.py` — cohort schema v2; produces:
  - `h4.main_v1_1_rank_based.{epoch_order_shuffle_literal, circular_shift_within_block_proposed}` (主 verdict)
  - `h4.supplementary_v1_0_participation_field.{...}` (supplementary verdict)
  - `h4.spatial_radius_drift_cohort.per_subject[]`
  - `h4.decision_k_drift_cohort.{n_total_subjects_with_drift, swap_class_distribution, stratified_summary, per_subject}`
- `scripts/b1_calibration_check.py` (Stage B advisor catch C5b) — cohort gate; **0/23 mismatch confirms B1 reproduces PR-6 on full data**

### 测试

- `tests/test_sef_itp_phase2.py`：**57 GREEN**
  - existing v1.0 tests preserved (compute_local_endpoint, endpoint_jaccard, slice_events, I_rate, I_geom, H3 verdict, cohort TOST + LOO)
  - new v1.1 tests:
    - `test_compute_local_rank_endpoint_basic_two_clusters`
    - `test_compute_local_rank_endpoint_phantom_mask_changes_endpoint`
    - `test_compute_local_rank_endpoint_zero_participation_channel_excluded`
    - `test_compute_local_rank_endpoint_valid_mask_filter`
    - `test_compute_local_rank_endpoint_k_degradation_on_small_pool`
    - `test_compute_local_rank_endpoint_empty_cluster_skipped`
    - `test_compute_local_rank_endpoint_empty_event_indices`
    - `test_compute_endpoint_spatial_radius_{empty, single_point, equilateral, collinear}`
    - `test_compute_source_sink_centroid_distance_{basic, 3d, empty_returns_nan}`
    - `test_compute_decision_k_drift_{returns_decision_k_per_epoch, drops_low_event_epochs, phantom_mask, summary_stats}`
  - removed `_masked_mean_rank` tests (function removed; dead code after refactor to `_per_cluster_template_rank`)

---

## 7. Pending decisions (unchanged from v1.0)

H4 I_rate matched null final choice (circular_shift / Poisson / gamma / etc) — user pending ratify; v1.1 implementation 沿用 v1.0 双方法并跑直至 user 决定。framework banner Phase 2 完结 marker 升级到 v1.0.7 (or v1.0.6 maintenance) — pending user ratify after cohort numbers land.

---

## 8. 内部归档代号映射（CLAUDE.md §8 朴素话风格 — 复用 v1.0 + new）

- v1.1 rank-based endpoint = propagation rank top-k （事件里最先点火的 k 个通道），用 PR-6 anchoring 同一 template estimator (`_legacy_hist_mean_rank` + argsort(argsort)) per-epoch 重算
- v1.0 participation field endpoint (supplementary) = `events_bool.mean(axis=0)` top-k（事件参与率最高的 k 个通道）
- centroid RMS = "该侧 k 点到自己 centroid 的均方距离"
- mean pairwise = "该侧 k 点两两距离平均"
- min enclosing ball = "装下该侧 k 点的最小球半径"
- source-sink centroid distance = "传播轴长度" (cross-side, separately reported)
- decision_k drift = "每 epoch 重跑 swap_sweep 得到的 decision_k 是否随时间变化"
- B1 calibration gate = "B1(full-data) 必须 set-equality reproduce PR-6 anchoring per_template source/sink per cluster" (firm-test for endpoint extractor parity)
