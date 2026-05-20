# lagPatRank Phantom Pseudo-Rank — Audit-First Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This plan is audit-first**: the discriminator AMI gate (Step 3) decides whether Steps 4 (cosmetic) or Step 5 (re-derivation) executes — do NOT run Step 5 before the audit completes.

## 0. 现状摘要 — Bug 已经被实证确认（2026-05-20）

### 0.1 Bug 定义

`lagPatRank` 在 legacy producer 中按 `argsort(argsort(per-event channel centers))` 的方式生成，**对所有通道**无差别排名。non-participating 通道的 `center_lagPat` 来源于 spectrogram 该时间窗的噪声 centroid，不是真实事件激活时间——但它仍会得到一个 `[0, n_ch-1]` 范围内的有限整数排名。

- **Legacy 源代码位置**：`ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/hfo_net.py:289`
  ```python
  center_lagPat_rank = np.array([np.argsort(np.argsort(x)) for x in center_lagPat.T]).T
  ```
  无 `eventsBool` 掩码。`p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py:462` 把这个结果当 `lagPatRank` 写入 `*_lagPat_withFreqCent.npz`。
- **HFOsp 继承**：loader (`src/interictal_propagation.py:335`) 直接读 `lp["lagPatRank"]`，下游 KMeans feature 矩阵在四个位置都用 `np.where(np.isfinite(...), ..., 0.0)` 作 "NaN guard"。**因为 phantom ranks 是有限整数，`np.isfinite` 捕不到，guard 等同于 no-op。**

### 0.2 实证证据（2026-05-20 在本 session 验证）

- 文件：`/mnt/yuquan_data/yuquan_24h_edf/chengshuai/FC10477Y_lagPat_withFreqCent.npz`
- 形状：`n_ch=8` (E11, K3, K5, K6, K7, K8, K9, K10), `n_ev=3513`
- `ranks` 中 NaN 数：**0**；最小值 0.0，最大值 7.0，取值 `{0..7}`
- 9064 个 non-participating cells (`bools==False`) **全部有有限整数 rank**：
  ```
  rank=7: 1728 (19.1%)
  rank=0: 1612 (17.8%)
  rank=6: 1181 (13.0%)
  rank=1: 1114 (12.3%)
  rank=2:  921 (10.2%)
  rank=5:  869 ( 9.6%)
  rank=4:  832 ( 9.2%)
  rank=3:  807 ( 8.9%)
  ```
  **U-shape pollution：两端（0/7）合计 36.9%，中段每档 ~10%**。"只看 high-rank 端 / low-rank 端"不能逃避污染，两端都被偏置。

### 0.3 受污染的 KMeans 调用点（已枚举）

| # | 位置 | 函数 | 影响 |
|---|---|---|---|
| 1 | `src/interictal_propagation.py:1215-1217` | `compute_cluster_stereotypy` | legacy fixed-k=2 cluster |
| 2 | `src/interictal_propagation.py:1469-1470` | `compute_adaptive_cluster_stereotypy` | **stable_k 选择 + cluster labels（PR-2 主路径）** |
| 3 | `src/interictal_propagation.py:1817-1818, 1825-1826` | `compute_time_split_reproducibility` | **PR-2.5 split-half / odd-even** |
| 4 | `src/interictal_propagation.py:2138-2139` | `compute_pr6_step6_held_out` | **PR-6 Step 6 held-out template** |
| 5 | `src/cluster_geometry.py:compute_pca_embedding` | `compute_pca_embedding` | cluster geometry PCA 视图 |
| 6 | `src/topic4_attractor_diagnostics.py:43-77` | `build_rank_feature_matrix` | **Topic 4 attractor (本周新增)** |

`_kmeans_stability_for_k` 用上游 feature 矩阵，间接受污染（在 #2 中调用）。

### 0.4 不受污染的部分（已审查）

- `_legacy_hist_mean_rank` (`src/interictal_propagation.py:1109-1138`)：per-channel bools-masked 后再 histogram。**唯一污染点**是 `template[ci]=ci` fallback（仅当通道在该 cluster 里完全不参与时）—— AGENTS.md "Cross-PR Contract Lookups" 已标注。
- `_center_rank_matrix:693`：有 `min_participation` 通道 gate。
- `_multi_seed_tau_summary`：tau 计算用 bools mask 限定 shared channels。
- PR-6 endpoint extraction：`valid_mask` 从 raw bools 派生（AGENTS.md 已合同化）。

### 0.5 下游受影响的科学结论（待审计）

- **stable_k 分布**：`{2:35, 4:2, 5:2, 6:1}`（PR-2 主表）
- **PR-3 per-cluster MI**：cluster_events 由 polluted labels 派生
- **PR-4A** day/night template occupancy timeline / dominant_fraction Wilcoxon
- **PR-4B/D** rate-state coupling (L1/L2/L3)
- **PR-5 / PR-5-B** template recruitment shift（dominant_global post−baseline +65.46 events/h, p=0.00128）
- **PR-6 H1 endpoint anchoring / H2 forward-reverse swap / Step 6 held-out template**
- **PR-7** template antagonistic pairing
- **Topic 4 attractor** Step 1 全部数字（10/34 label-transition λ₂ 通过）

---

## 1. 计划目标 & 边界

**Goal**：以最低 churn 验证现有 cohort 结论是否被 phantom rank 污染，仅在确证污染显著时（AMI gate 失败）才回溯重算。

**In scope**：
- 实现 masked-rank 重排 + NaN-aware KMeans feature 准备
- 单 helper 函数 + 单测，避免 surgical 改动多个 KMeans 调用点
- 跑 cohort 级 AMI(original_labels, masked_labels) 比较审计
- AMI 通过：写一份 audit-pass 归档，修复 heatmap 可视化；**不动**已发表口径
- AMI 不通过：列出需要重跑的下游 PR，分批替换 + 验证

**Out of scope**：
- 重新生成 `*_lagPat*.npz`（legacy producer 重跑代价过高且生态不许）。修复在 HFOsp 消费侧。
- 修改 `_legacy_hist_mean_rank` 的 fallback 行为（已被 AGENTS.md 合同化，且属于 template 而非 KMeans 路径）
- PR-5/PR-6/PR-7 plan-of-record 的 hypothesis tier 调整（CLAUDE.md §5：sensitivity gates 通过才动主文档）

**Architecture**：
- 新增 `src/lagpat_rank_audit.py`：放置 (a) `mask_phantom_ranks(ranks, bools)` 重排函数；(b) `build_masked_kmeans_features(ranks, bools, impute='event_median')` ；(c) `kmeans_label_ami_audit(...)` 跨 subject 比较。
- 新增 `scripts/audit_kmeans_phantom_rank.py`：单脚本，loop 40 subjects，每 subject 跑 original vs masked KMeans → AMI → emit `results/lagpatrank_audit/<sid>.json` + cohort summary CSV + per-subject diagnostic figure。
- 不动 `compute_adaptive_cluster_stereotypy` 主路径；若审计 fail，再做最小侵入式补丁（增加 `use_masked_features=False` 默认参数，gate 在 caller）。

---

## 2. 修复策略 — Masked re-rank + median impute

### 2.1 数学定义

Given raw `ranks: (n_ch, n_ev)` and `bools: (n_ch, n_ev)`:

1. **Per-event 重排**：对每个事件 `e`，只在 `bools[:, e] == True` 的通道上做 `argsort(argsort(...))`，结果范围 `[0, n_participating(e) - 1]`。non-participating 通道置 `NaN`。
2. **归一化**（避免 n_participating 差异带来的轴 scale 不一致）：对每个事件 `e`，把参与通道的 rank 线性映射到 `[0, 1]`：`r_norm[c, e] = r[c, e] / (n_participating(e) - 1)` if `n_participating(e) > 1`，否则 `r_norm[c, e] = 0.5`。
3. **NaN 填补**（feed KMeans 用）：non-participating 通道填 `0.5`（事件归一化轴的中点），不偏向两端。

为什么 median impute 不用「per-channel 中位数」？因为 KMeans Euclidean 在 channel 维度求和，per-channel 中位数会把该通道的 cluster identity 信号挤压。用「per-event 中点」对所有事件均匀偏置，最不引入新的伪聚类轴。

### 2.2 备选 impute 策略（sensitivity）

- `event_median = 0.5`（normalized scale）— 主路径
- `channel_median`（per-channel 在所有参与事件上的中位排名归一化值）— sensitivity
- `random_uniform` — sensitivity，验证 phantom 是否仅作为"noise rank"
- `drop_low_participation` — 仅在 `n_participating(e) >= ceil(0.7 * n_ch)` 事件上跑 KMeans —— 当 n_part 接近 n_ch 时 phantom 自然消失

主路径 + 1 个 sensitivity 即可，无需全跑 4 个。

### 2.3 重要 disclaimer — masked impute 不解开「参与模式 vs 排序」纠缠

把 non-participating cell 从 phantom rank 改为常数 `0.5` 仅去除了**端点 U-shape** 的方向性偏置；KMeans Euclidean 仍然把「该通道是否参与」当作一条特征轴看（参与 → 任意 [0,1] 值；不参与 → 恒定 0.5），二者在 channel-wise 距离上仍可区分。

这意味着本 audit 只回答：**「phantom rank 的端点排序偏置是否驱动了 cluster identity？」**——**不**回答「PR-2 cluster 究竟是排序聚类还是参与模式聚类」。后者是另一道问题，应留给 cluster geometry 后续工作（`src/cluster_geometry.py:1-24` 已经把这条 caveat 写进了文档）。

如果未来想做"排序 vs 参与模式"分离 audit，需要的是：把每事件的参与集合作为单独特征（bitmask hashing 或 one-hot），与 rank 特征分开跑 KMeans，比较两者各自的 silhouette + 与原 labels 的 AMI。这不在本 plan 范围。

---

## 3. Step-by-step plan

### Step 1 — Helper 模块 + 单测（TDD）

- [ ] **TDD 1.1**：`tests/test_lagpat_rank_audit.py::test_mask_phantom_ranks_per_event`
  - Setup: 手工构造 `ranks=[[0,1,2],[1,2,0],[2,0,1]]`（3ch × 3ev）, `bools=[[T,T,F],[T,F,T],[T,T,T]]`
  - Expected: `masked_ranks` 在 (1,1) 和 (2,0) 是 NaN；其余每事件重排为 `argsort(argsort(participating))`。事件 1 有 2 通道参与 → 重排范围 `{0, 1}`；事件 2 有 1 通道参与 → 该通道置 0；事件 0 有 3 通道参与 → 重排范围 `{0, 1, 2}`。
- [ ] **TDD 1.2**：`test_normalize_event_rank_scale`
  - Verify per-event normalized range `[0, 1]`；n_part=1 → 0.5；n_part=0 → 整列 NaN（事件被外排）
- [ ] **TDD 1.3**：`test_build_masked_kmeans_features_shape_and_impute`
  - Verify output shape `(n_ev, n_ch)`、no NaN、impute 值 == 0.5、order 与 raw bools 一致
- [ ] **TDD 1.4**：`test_kmeans_label_ami_audit_synthetic_phantom`
  - 构造 5ch × 200ev 合成数据：100 事件 "early-K1 + K3-K5 participate" cluster A，100 事件 "late-K1 + K2-K4 participate" cluster B；non-participating channels 用 U-shape phantom 填。original KMeans labels vs masked KMeans labels AMI → expect AMI < 0.8（phantom 引入伪轴）。
- [ ] **TDD 1.5**：`test_kmeans_label_ami_audit_no_phantom_pass_through`
  - 在所有 cells `bools=True` 的全参与合成数据上跑：original vs masked AMI → expect `>= 0.99`（fully degenerate 等价）
- [ ] 1.6 实现 `src/lagpat_rank_audit.py`：
  - `mask_phantom_ranks(ranks, bools, normalize=True) -> np.ndarray`（含 NaN）
  - `build_masked_kmeans_features(ranks, bools, impute='event_median') -> (n_ev, n_ch)`
  - `kmeans_label_ami_audit(ranks, bools, k, n_init=10, random_state=0) -> dict`（返回 original_labels, masked_labels, ami, n_events, phantom_fraction）
- [ ] 1.7 跑 `pytest tests/test_lagpat_rank_audit.py -v` 确认 5 项全过

### Step 2 — Cohort audit driver + AMI noise floor

**关键设计 (advisor 2026-05-20)**：单跑 `AMI(original_labels, masked_labels)` 有 ceiling confounded by 共享的 participation pattern。必须先建 **AMI noise floor** = 同一 feature matrix 下不同 KMeans seed 之间的 AMI，再相对噪声底比较 phantom audit AMI。

- [ ] 2.1 实现 `scripts/audit_kmeans_phantom_rank.py`
  - Loop 40 subjects（20 yuquan + 20 epilepsiae，复用 `scripts/audit_topic4_step0.py` 的 subject 列表）
  - 每 subject：
    - 用 `load_subject_propagation_events` 载入 ranks + bools
    - 从 PR-2 per-subject JSON (`results/interictal_propagation/<sid>.json`) 读 `adaptive_cluster.chosen_k`、`adaptive_cluster.labels`、`stable_k`、`scan_results`
    - **Noise floor**：在 **original** feature matrix 上跑 KMeans 5 seeds (random_state ∈ {0..4})，计算 10 个 pairwise AMI 的中位数 `ami_seed_floor_original`
    - **Masked noise floor**：在 **masked** feature matrix 上同样跑 5 seeds，计算 pairwise AMI 中位数 `ami_seed_floor_masked`
    - **Audit signal**：跨 feature matrix 在固定 seed=0 下计算 `ami_audit = AMI(original_labels_seed0, masked_labels_seed0)`
    - **Per-k**：对每个 k ∈ scan_range 跑 audit_signal + noise_floor（仅 chosen_k 跑 full sweep；其他 k 用 single-seed 即可）
    - 记录 phantom_fraction = `(~bools).sum() / bools.size`
  - 写 `results/lagpatrank_audit/<sid>.json` per subject + cohort_summary.csv 列：
    `sid, dataset, n_ch, n_ev, phantom_fraction, chosen_k, ami_audit_chosen, ami_seed_floor_original, ami_seed_floor_masked, ami_audit_minus_floor (= ami_audit - min(ami_seed_floor_original, ami_seed_floor_masked)), masked_stable_k_candidate, stable_k_changed`
- [ ] 2.2 跑 cohort：`python scripts/audit_kmeans_phantom_rank.py --all`（预期 < 30 min for 40 subjects with 5× seed re-cluster）
- [ ] 2.3 三张诊断图 `results/lagpatrank_audit/figures/`：
  - `ami_vs_noise_floor.{png,pdf}` (核心)：scatter `ami_seed_floor_original` (x) vs `ami_audit_chosen` (y)；对角线 = "audit AMI = seed noise floor"（cosmetic 区域 = 点在对角线上方或紧贴）
  - `phantom_fraction_vs_ami_audit.{png,pdf}`：scatter phantom_fraction (x) vs ami_audit_minus_floor (y)；负值 = phantom 偏置确实损害 cluster identity
  - `stable_k_confusion.{png,pdf}`：counts of (original_stable_k, masked_stable_k_candidate)；off-diagonal = phantom 扭曲了 stable_k 选择
  - 写对应 README，**明确指出 noise-floor 是 gate 阈值的 anchor**

### Step 3 — Discriminator AMI gate（relative to noise floor）

- [ ] 3.1 读 cohort_summary.csv，计算：
  - cohort-median `ami_audit_minus_floor`（核心 metric）
  - n_subjects with `ami_audit_minus_floor < -0.10 / -0.05` （phantom 损害 cluster identity 超过 seed jitter）
  - n_subjects whose masked_stable_k differs from original_stable_k
  - phantom_fraction 与 `ami_audit_minus_floor` 的 Spearman ρ
- [ ] 3.2 **Gate rule (pre-registered 2026-05-20, freeze before running Step 2)**：

  阈值锚定到 noise floor：cosmetic = "phantom audit 没把 cluster identity 推得比 seed jitter 更远"。
  
  - **Cosmetic outcome**：cohort-median `ami_audit_minus_floor >= -0.05` AND n_subjects(`ami_audit_minus_floor < -0.10`) <= 2 AND no subject changes `stable_k` → **Step 4**。
  - **Mixed outcome**：cohort-median `ami_audit_minus_floor ∈ [-0.15, -0.05)` OR 3-8 subjects with `ami_audit_minus_floor < -0.10` OR 1-3 subjects change `stable_k` → **Step 5 narrow** (case-series re-derivation on impacted subjects only)。
  - **Re-derivation outcome**：cohort-median `ami_audit_minus_floor < -0.15` OR >8 subjects with `ami_audit_minus_floor < -0.10` OR ≥4 subjects change `stable_k` → **Step 5 broad**。
  - **Edge case**：若 `ami_seed_floor_original < 0.7` 的 subject 数 ≥ 3，说明 KMeans 在该 cohort 上本就 seed-不稳定，本 audit metric 不适用——先解决 PR-2 的 seed 稳定性问题再回来跑 audit。
  
  *理由*：absolute 0.70 / 0.85 阈值无法在面对 seed 噪声主导的 cohort 时辩护（advisor 反馈 2026-05-20）。Noise-floor relative 阈值能区分 "phantom 影响" vs "本就 seed-不稳"。

### Step 4 — Cosmetic outcome path

- [ ] 4.1 修可视化：在 `scripts/plot_interictal_propagation.py:147, 178, 189, 832` 把 `pcolormesh(display_ranks, cmap="viridis")` 改成 `pcolormesh(masked_ranks, cmap="viridis")` where `masked_ranks = np.where(bools, display_ranks, np.nan)`，并设置 `cmap.set_bad('lightgray')` 让 phantom cells 灰显
- [ ] 4.2 重生成 30 张 per-subject propagation heatmap（用最新 `--pr3` mode），确认视觉上 phantom 灰带与原 U-shape 分布一致
- [ ] 4.3 写归档：`docs/archive/topic1/propagation/lagpatrank_phantom_audit_2026-05-XX.md`
  - §1 Bug definition + empirical numbers（搬本 plan §0）
  - §2 AMI gate cohort result + per-subject table
  - §3 Cosmetic verdict 推理（AMI 高 → cluster identity 由真实通道驱动，phantom 是中位 noise）
  - §4 Heatmap fix before/after 对比（含 chengshuai 一张前后图）
  - §5 不动 PR-2/PR-3/PR-4*/PR-5/PR-6/PR-7/Topic 4 的口径
- [ ] 4.4 Topic 1 主文档 §10 历史索引追加新 archive 的 back-link；**不动** §2 / §4 当前结论

### Step 5 — Re-derivation path（仅当 gate 不通过）

执行顺序严格按依赖：PR-2 → PR-3 → PR-4* → PR-5 → PR-6 → PR-7 → Topic 4。

#### Step 5a — PR-2 re-cluster with masked features
- [ ] 5a.1 增加 `compute_adaptive_cluster_stereotypy` 参数 `use_masked_features: bool = False` 默认 False（保旧调用兼容），True 时调用 `build_masked_kmeans_features`
- [ ] 5a.2 增加同款参数到 `compute_cluster_stereotypy` 和 `compute_time_split_reproducibility`
- [ ] 5a.3 增加单测：原参数 default=False 时所有现有 test 通过；True 时在合成 phantom-rich 数据上 AMI > 0.8 vs ground-truth labels
- [ ] 5a.4 改 `scripts/run_interictal_propagation.py` 增加 `--masked-features` flag；默认 False；为 audit 重跑专用 True
- [ ] 5a.5 用 `--masked-features` 重跑 40 subjects；写到 `results/interictal_propagation_masked/`（不覆盖既有 `results/interictal_propagation/`）
- [ ] 5a.6 输出对照表：original vs masked 各 subject 的 `stable_k`、`chosen_k`、cluster fractions、scan_results AMI matrix

#### Step 5b — PR-2.5 split-half reproducibility on masked
- [ ] 5b.1 跑 `compute_time_split_reproducibility(use_masked_features=True)` 对 40 subjects
- [ ] 5b.2 与 original 比对：split-half + odd-even 各方向 `forward_reverse_reproduced` 是否仍 8/9（OR 规则，per AGENTS.md cross-PR contract）

**Checkpoint A (after 5a + 5b)**：advisor consult 以下两个问题：
- 5a stable_k 分布相对 original `{2:35, 4:2, 5:2, 6:1}` 的变化幅度
- 5b 8/9 forward/reverse reproduced subjects 集合是否稳定
若 5a stable_k flip 超过 5 个 subject 或 5b reproduced 集合大幅重组，**暂停 5c–5h，回头审 masking 策略本身**（可能 event_median impute 不够好，需要 channel_median 或 drop_low_participation sensitivity）。

#### Step 5c — PR-3 per-cluster MI on masked labels
- [ ] 5c.1 用 masked labels 重跑 `compute_legacy_mi`；30 个 MI 数字是否仍 30/30 显著
- [ ] 5c.2 within-cluster centered tau bias_fraction 是否仍 ~86%

#### Step 5d — PR-4A/B/C/D on masked labels
- [ ] 5d.1 PR-4A `compute_temporal_cluster_dynamics`：用 masked templates；day/night Wilcoxon p 与 TV distance 是否同符号
- [ ] 5d.2 PR-4B `compute_rate_state_coupling` L1/L2/L3：跟原 cohort 数字符号是否一致
- [ ] 5d.3 PR-4C seizure proximity：5 指标 cohort Wilcoxon 是否仍 null
- [ ] 5d.4 PR-4D `compute_template_rate_decomposition`：rate-burst seizure-enrich 9 subjects 是否仍 strict-match

**Checkpoint B (after 5d)**：advisor consult on PR-4B/D direction reversal — 这两个是 Topic 1 主文档 §4 主结论里的描述层。若 5d 出现 NULL → SIGNIFICANT 或 SIGNIFICANT → NULL 翻转，**暂停 5e–5h，先写一份 reconcile 文档讨论 masking 影响**。

#### Step 5e — PR-5 / PR-5-B on masked
- [ ] 5e.1 `compute_template_recruitment_shift` 用 masked labels；dom_global_share post−baseline 是否仍 +65.46 events/h、p=0.00128
- [ ] 5e.2 PR-5-A novel-template gate 是否仍 PASS（n=23/22）

#### Step 5f — PR-6 endpoint anchoring on masked clusters
- [ ] 5f.1 H1 endpoint vs middle frac_SOZ：cohort Wilcoxon 数字
- [ ] 5f.2 H2 forward-reverse swap：swap_class 分布是否变化（关注 swap_n_perm null 重算）
- [ ] 5f.3 Step 6 held-out template `compute_pr6_step6_held_out(use_masked_features=True)`：tier 分布
- [ ] 5f.4 PR-6 supplementary rank displacement：F_norm cohort 中位 0.800、ρ(F_norm, τ)=-0.916 是否稳定（rank displacement 用 `template_rank` 即 polluted templates 派生，间接受影响）

#### Step 5g — PR-7 antagonistic pairing on masked clusters
- [ ] 5g.1 H1 三 metric NULL 是否仍 NULL
- [ ] 5g.2 burst diagnostic 是否结论不变

#### Step 5h — Topic 4 attractor on masked features
- [ ] 5h.1 修 `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix`：增加 `mask_phantom: bool = False` 参数，True 时调用 `build_masked_kmeans_features`
- [ ] 5h.2 重跑 `scripts/run_attractor_step1.py --masked-features`，emit `results/topic4_attractor_masked/`
- [ ] 5h.3 关键比较：label transition λ₂ 通过 p<0.001 & λ₂>0 的 subject 数（原 10/34）是否变化；GOF pass 率（原 33/34）是否变化
- [ ] 5h.4 修补 `docs/archive/topic1/propagation/topic4_attractor_diagnostics_step1_results_2026-05-10.md`：在 §5 加 "phantom-rank audit re-run results" 段

#### Step 5i — 主文档 + 归档更新
- [ ] 5i.1 创建 `docs/archive/topic1/propagation/lagpatrank_phantom_audit_results_2026-05-XX.md` 含全部 5a–5h 对比表
- [ ] 5i.2 修 Topic 1 主文档 §2 / §3.1 / §4 / §10：把 cohort 数字更新到 masked 版本；旧数字下沉 archive
- [ ] 5i.3 修 AGENTS.md "Cross-PR Contract Lookups" 节加新条目：**「lagPatRank 是 phantom-contaminated，KMeans feature 必须走 `build_masked_kmeans_features`」**
- [ ] 5i.4 修 CLAUDE.md §5 / §6 examples 加 phantom-rank case 作为「end-to-end fix」实例
- [ ] 5i.5 把 `use_masked_features` 默认从 False 改成 True（broad-rederivation 路径下旧默认是 bug）；deprecate 老路径

---

## 4. 不在本 plan 内的 follow-ups（记账）

- 重跑 legacy producer 修 `hfo_net.py:289`：代价过高（需要重新跑 24h spectrogram on 40 subjects），且会破坏与老 paper 的字节级可重现。本 plan 选择在消费侧修。
- 重审老 paper 的 Fig 2 polar plot：与本 repo 主线无关，留给后续叙事 framing 时一句脚注。
- BHPN-fit (PR-T4-2) 启动时若用 metastable label transition 作 fit target，必须基于 masked 重算的 λ₂（如果 Step 5h 走到了）。

---

## 5. 验收合同 / Failure modes

- **AMI gate 出口必须 pre-registered**：阈值（cohort-median `ami_audit_minus_floor >= -0.05` cosmetic / ∈ [-0.15, -0.05) mixed / < -0.15 broad）必须在跑 Step 2 之前 freeze。改阈值 = 数据驱动决策 = p-hacking。
- **Step 5 任何子步骤如果出现「原 cohort 数字 vs masked cohort 数字」符号反转**：必须 STOP，写一份独立 review 文档讨论；不能默默更新主文档结论。
- **PR-7 P3 INCONCLUSIVE-locked 是 paper-framework-level decision**（`paper1_framework_sba.md` v1.1.2 + `pr7_addendum_p3_equivalence_2026-05-01.md`）。若 Step 5g masked re-derivation 把 P3 推到 PASS 或 FAIL（任意一侧），这是 framework revision 而非 numerical update —— 必须先停下来发起 framework 修订 review，**不**直接重写 P3 结论。
- **Topic 4 Step 1 数字最敏感**：本周（2026-05-10）新增、未入主文档。Step 5h 完成前不允许在 Topic 1 主文档把 attractor diagnostics 升级到主结论 tier。
- **Step 4 / Step 5 分支必须互斥**：AMI gate 推出 Cosmetic 后**不**主动跑 Step 5；推出 Re-derivation 后必须跑完所有 5a-5h 再发布。
- **Step 5 内 advisor checkpoint A (after 5a+5b) + checkpoint B (after 5d) 是 hard stops**：未通过 checkpoint 不准继续。

---

## 6. 工作量估计

| Step | 估时 | 依赖 |
|---|---|---|
| 1 (helper + TDD) | 半天 | 无 |
| 2 (cohort AMI audit) | 半天 | 1 |
| 3 (gate 评估) | 1 小时 | 2 |
| 4 (cosmetic path) | 半天 | 3 → cosmetic |
| 5a–5h (broad re-derivation) | 1–2 周 | 3 → re-derivation |
| 5i (doc updates) | 半天 | 5a–5h |

Cosmetic 路径预算 1.5 天；Broad re-derivation 路径预算 2–3 周（含 review checkpoints）。

---

## 7. 启动建议 — chengshuai-first 闪电诊断

chengshuai 是 memory 标注的"worst case"（114 distinct participation patterns over 3513 events，n_ch=8 phantom 浓度极高）。**先在 chengshuai 一个 subject 上把 noise floor + audit AMI 跑出来，能在 1–2 小时内大致看出 cohort 走向哪个出口**。

下一步：
1. **半天**：实现 Step 1 helper + TDD（5 项单测必须先过）。
2. **1–2 小时**：在 chengshuai 上跑：5 seeds × original features → `ami_seed_floor_original`；5 seeds × masked features → `ami_seed_floor_masked`；single-seed cross → `ami_audit`。
3. **判读**：
   - 若 `ami_audit ≥ ami_seed_floor_original` （即 phantom 对 cluster identity 影响小于 seed 噪声）：cohort 几乎必然走 Cosmetic，可直接转 Step 4。chengshuai 是 worst case，过了它别的 subject 应该都过。
   - 若 `ami_audit << ami_seed_floor_original` 但 stable_k 不变：cohort 大概率走 Mixed，准备 Step 5 narrow。
   - 若 chengshuai 的 masked stable_k 翻转：跑全 cohort Step 2 再判，可能要走 Broad。
4. **不要先动** PR-2 / PR-6 / Topic 4 的源代码，直到 chengshuai 闪电诊断 + 全 cohort Step 2 + Step 3 gate 给出明确出口。
5. **闪电诊断结果出来后，立即更新本 plan 的 §0.6 加上 chengshuai 数字**，并在 commit message 里把判读路径记下来。
