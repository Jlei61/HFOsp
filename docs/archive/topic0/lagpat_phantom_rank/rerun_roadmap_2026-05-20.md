# lagPatRank Phantom — 重跑路线图（Step 5a–5i）

> 状态：路线已锁，2026-05-20 由 user 决定系统性重跑。
> 主入口：`docs/topic0_methodology_audits.md`
> 计划合同：`docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md` §5
> 白话总览：`./plain_chinese_report_2026-05-20.md` §4

---

## 0. 原则

1. **旧结果不删**。每一步的修过版输出走 `_masked` parallel dir。
2. **修过版函数路径** 通过新增 `use_masked_features` / `mask_phantom` 默认参数实现；旧 default = False，保兼容；驱动脚本加 `--masked-features` flag 显式打开。
3. **Checkpoint 是 hard stops**：未通过 advisor consult 不进入下一阶段。
4. **PR-7 P3 翻转 = framework revision**，必须先发起 `docs/paper1_framework_sba.md` 评议，不可默改。
5. **每个 Step 跑完更新 Topic 0 主文档 §3.1 表**，把对应行从 "待重跑" 移到 "重跑完，方向 = 一致/翻转/NULL flip"。

---

## Step 5a — PR-2 re-cluster on masked features

### 范围
- 40 个 subject
- `compute_adaptive_cluster_stereotypy` + `compute_cluster_stereotypy` 加 `use_masked_features` 参数
- 调用 `src/lagpat_rank_audit.build_masked_kmeans_features(ranks, bools, impute="event_median")`
- 输出 → `results/interictal_propagation_masked/per_subject/<sid>.json`

### TDD
- [ ] `test_compute_adaptive_cluster_stereotypy_use_masked_features_smoke`
  - 同一 subject，`use_masked_features=False` vs `True` 都不报错
  - chosen_k 都返回有效值，labels 长度 = n_valid_events
- [ ] `test_compute_cluster_stereotypy_use_masked_features_smoke`

### 输出
- `results/interictal_propagation_masked/per_subject/<sid>.json`
- `results/interictal_propagation_masked/_cohort_summary.csv`（同 schema 加 `_masked` 后缀字段）

### 对比汇总
- stable_k 翻转表（pre-loaded from `results/lagpatrank_audit/cohort_summary.csv`，验证一致）
- 每 subject cluster fractions before/after
- 每 subject label-level Jaccard 与 contingency table

---

## Step 5b — PR-2.5 split-half / odd-even on masked

### 范围
- 同 cohort
- `compute_time_split_reproducibility` 加 `use_masked_features` 参数
- 输出同 dir，文件名加 split-half / odd-even tag

### TDD
- [ ] `test_compute_time_split_reproducibility_use_masked_features_smoke`

### 对比
- `forward_reverse_reproduced` (split-half OR odd-even per AGENTS.md cross-PR contract) 集合 vs 原（原 = 8/9）
- per-subject reproducibility tier (strong/moderate/weak) 是否翻转

---

## Checkpoint A（advisor consult）

**触发**：5a + 5b 跑完。

**审什么**：
1. masked stable_k 与 lagpatrank_audit `masked_stable_k_candidate` 是否一致（应该 100% 一致；不一致说明修法本身漂移）
2. stable_k flip 数（主线 cohort 是 1 个 916：2→4；高 k cohort 3 个）是否符合预期
3. PR-2.5 forward/reverse reproduced 集合变化幅度——大幅重组说明 masked 把信号也削掉了

**Gate**：
- 通过 → 进 5c
- 主线 cohort flip 超 3 个 / reproduced 集合大幅重组 → 暂停，先反思 impute 策略（试 `channel_median` 或 `drop_low_participation` sensitivity）

---

## Step 5c — PR-3 per-cluster MI on masked labels

### 范围
- 用 5a 的 masked cluster labels
- `compute_legacy_mi` 不动（已 bools-masked，per memory `_legacy_hist_mean_rank` 也只有 fallback 污染）
- 重算每 subject 的 30 MI + permutation
- 输出 → `results/interictal_propagation_masked/per_subject/<sid>.json` 的 `legacy_mi` 字段

### 对比
- 30/30 仍 MI > 0 且显著？
- within-cluster centered tau bias_fraction 是否仍 ~86%？

---

## Step 5d — PR-4A/B/C/D on masked

### Step 5d.1 — PR-4A day/night occupancy
- `compute_temporal_cluster_dynamics` 用 masked templates / labels
- 输出 → `results/interictal_propagation_masked/pr4a_daynight/`
- 对比：day/night Wilcoxon p / TV distance / template projection agreement median

### Step 5d.2 — PR-4B rate-state coupling L1/L2/L3
- `compute_rate_state_coupling` 用 masked
- 输出 → `results/interictal_propagation_masked/pr4b_rate_coupling/`
- 对比：L1 / L2 / L3 各 cohort Wilcoxon 方向 + p

### Step 5d.3 — PR-4C seizure proximity
- 5 指标 cohort Wilcoxon
- 输出 → `results/interictal_propagation_masked/pr4c_seizure_proximity/`

### Step 5d.4 — PR-4D template-rate decomposition
- 输出 → `results/interictal_propagation_masked/pr4d_template_rate/`
- 对比：rate-burst seizure-enrich 9 subject strict-match 是否仍一致

---

## Checkpoint B（advisor consult）

**触发**：5d 跑完。

**审什么**：
- PR-4B/D 的 cohort 方向是否反转（这两条是 Topic 1 主文档 §4 主结论描述层支柱）
- 任何 NULL → SIGNIFICANT 或 SIGNIFICANT → NULL 都触发 advisor consult

**Gate**：
- 通过 → 进 5e
- 翻转 → 写 reconcile 文档，讨论 masking 是否过度

---

## Step 5e — PR-5 / PR-5-B on masked

- `compute_template_recruitment_shift` 用 masked
- PR-5-A novel-template gate 重跑
- 输出 → `results/interictal_propagation_masked/template_share_switching/`
- 重新生成 fig_a / fig_b paper-grade 图 + 写 README
- 对比：dominant_global post−baseline events/h 是否仍 +65.46, p=0.00128

---

## Step 5f — PR-6 endpoint / swap / Step 6 / rank displacement on masked

### Step 5f.1 — PR-6 H1 endpoint vs middle frac_SOZ
- 输出 → `results/interictal_propagation_masked/pr6_template_anchoring/`

### Step 5f.2 — PR-6 H2 forward-reverse swap
- swap_class 分布对比（注意：H2 在 framework 中是 directional mechanism sanity，不是 cohort claim，按 AGENTS.md 不允许升级 tier）

### Step 5f.3 — PR-6 Step 6 held-out
- `compute_pr6_step6_held_out` 用 masked
- tier 分布 (strong/moderate/weak/fail) 对比
- 输出 → `results/interictal_propagation_masked/pr6_step6_held_out_template/`

### Step 5f.4 — PR-6 supplementary rank displacement
- `compute_swap_score_sweep` 用 masked templates（template_rank 派生自 cluster templates，间接受影响）
- F_norm cohort median / Kendall τ / Spearman ρ(F_norm, τ) 对比
- 输出 → `results/interictal_propagation_masked/rank_displacement/`

---

## Step 5g — PR-7 antagonistic pairing on masked

- H1 三 metric (10s primary + 30s sensitivity + sign test) 重算
- P3 cohort-level TOST equivalence test 重算
- N0–N4 surrogate hierarchy 重算
- 输出 → `results/interictal_propagation_masked/pr7_template_pairing/`

**⚠️ 重要**：如果 P3 INCONCLUSIVE-locked 翻到 PASS 或 FAIL（任意一侧），**STOP**——这是 framework-level revision，必须先发起 `docs/paper1_framework_sba.md` v1.1.2 修订评议，不允许默改。

---

## Step 5h — Topic 4 attractor on masked

- `src/topic4_attractor_diagnostics.py:build_rank_feature_matrix` 加 `mask_phantom: bool = False` 参数
- `scripts/run_attractor_step1.py --masked-features`
- 输出 → `results/topic4_attractor_masked/`
- 对比：
  - GOF pass 率（原 33/34，916 是 GOF-fail 的那一个；916 stable_k flip 2→4 后是否落到 stable_k>2 cohort 之外）
  - Coordinate-free PR-2 label λ₂ 通过 p<0.001 & λ₂>0 的 subject 数（原 10/34）
  - max_iter sensitivity 重做

---

## Step 5i — 主文档 + AGENTS.md cross-PR 合同更新

### 5i.1 创建 Topic 0 second archive
- `docs/archive/topic0/lagpat_phantom_rank/rerun_results_2026-05-XX.md`
- 含 5a-5h 所有对比表 + before/after 关键图链接

### 5i.2 修 Topic 1 主文档
- §2 一句话结论：把 cohort 数字更新到 masked 版本
- §3 / §4 / §10 同步更新；旧数字下沉到本 archive

### 5i.3 修 Topic 4 主文档（若已建）
- 同上

### 5i.4 AGENTS.md 更新
- "Cross-PR Contract Lookups" 节加一条：
  > **`lagPatRank` 是 phantom-contaminated，KMeans feature 必须走 `build_masked_kmeans_features` 或显式 `use_masked_features=True`。详见 `docs/topic0_methodology_audits.md` §3.1。**
- "Current Code Map" 节标注 `src/interictal_propagation.py` 中 4 个 KMeans 调用点的新合同

### 5i.5 CLAUDE.md 更新
- §5 / §6 examples 加 phantom-rank case 作为 "end-to-end fix" 实例

### 5i.6 把 `use_masked_features` 默认从 False 改成 True
- broad-rederivation 路径下旧默认 False = 重复 bug
- 旧路径标 `DeprecationWarning`，留 1 个版本兼容窗口

### 5i.7 Topic 0 主文档 §3.1 "受影响 PR" 表全部从 "待重跑" 移到 "重跑完"

---

## 工作量估算

| Step | 内容 | 预算 |
|---|---|---|
| 5a | PR-2 re-cluster | 1 天 |
| 5b | PR-2.5 split-half | 1 天 |
| Checkpoint A | advisor consult | 0.5 天 |
| 5c | PR-3 MI | 0.5 天 |
| 5d | PR-4A/B/C/D | 3–4 天 |
| Checkpoint B | advisor consult | 0.5 天 |
| 5e | PR-5 | 1–2 天 |
| 5f | PR-6 全套 | 2–3 天 |
| 5g | PR-7（含 framework gate） | 2–3 天 + framework review 不可控 |
| 5h | Topic 4 attractor | 1–2 天 |
| 5i | 主文档更新 | 1 天 |

**总预算 2–3 周**（连续工作）。每步可独立提交、可独立 advisor consult。

---

## 启动顺序（按依赖）

立即可启动：
1. **5a 实现**（添加参数 + TDD + per-subject 运行）— 是所有下游的前置
2. **5b 实现**（同 5a 一组）

5a + 5b 全 cohort 跑完 → Checkpoint A → 5c → 5d 系列 → Checkpoint B → 5e/5f/5g 三者并行（无相互依赖）→ 5h → 5i 收尾。
