# Agent Guide

This repo is rebuilding the legacy Yuquan HFO pipeline into a maintainable codebase. Do not guess where figures or artifacts come from.

## Read This First

Before tracing any result, read these in order:

1. `docs/LEGACY_YUQUAN_CODEBASE_MAP.md`
2. `docs/LEGACY_YUQUAN_FIGURE_ASSET_MAP.md`
3. `docs/LEGACY_PAPER_TIFF_CHAIN.md`
4. `docs/OLD_vs_NEW_algorithm_comparison.md`
5. `docs/yuquan_24h_dataset_structure.md`
6. `config/default.yaml`

## Canonical Topic Docs

For any scientific-status question, go to the topic doc; do not rely on summaries in this file.

- `docs/paper_overview.md` — total index + one-line conclusions for all topics
- **`docs/topic0_methodology_audits.md` — READ FIRST before any Topic 1–5 number** (currently: `lagPatRank` phantom pseudo-rank fix; phase 0 broad re-derivation closed 2026-05-21)
- `docs/topic1_within_event_dynamics.md` — within-event dynamics (propagation + synchrony)
- `docs/topic2_between_event_dynamics.md` — event-between-event timing
- `docs/topic3_spatial_soz_modulation.md` — where / SOZ spatial attribution
- `docs/topic5_seizure_subtyping.md` — within-subject seizure subtyping (exploratory)

## 文档与输出形式

- Topic 1/2/3 走"分层中文 markdown + archive 归档"，不走 Cursor canvas / React 面板（用户**明确**点名 canvas 才考虑）。
- 主文档 `docs/topic{1,2,3}_*.md` 只保留正式口径（当前接受结论、风险点、最小合同、下一步），语言中文，英文术语 / 变量名保持原样。
- 审阅 / 阶段性报告 / 全量数值表归档到 `docs/archive/<topic>/<descriptive>_<YYYY-MM-DD>.md`。
- 主文档**不复制 archive 全量数值表**；只保留摘要 + 链接，章节末尾回链 archive。

## Canonical Roots

- Current codebase: `HFOsp/`
- Canonical artifact root: `/mnt/yuquan_data/yuquan_24h_edf`
- Epilepsiae raw/sql/artifact root: `/mnt/epilepsia_data`
- Historical legacy tree: `/home/honglab/leijiaxin/HFOsp/ReplayIED` (Yuquan mainline at `inter_events/yuquan_24h_perPatientAnalysis_dropRef/`)

If the legacy tree is not mounted, stop and ask the user for its real path. Do not invent one.

## Golden Rule

Trace by **artifact**, not by figure number.

1. Identify which artifact the figure consumes: `<record>_gpu.npz` / `_refineGpu.npz` / `<record>_packedTimes*.npy` / `<record>_lagPat*.npz` / network outputs.
2. Map that artifact back to the producing stage.
3. Find the plotting script only after the data source is clear.

Exception: when the user explicitly references a paper TIFF in `ReplayIED/tiffs`, read `docs/LEGACY_PAPER_TIFF_CHAIN.md` first; even then, use TIFF visual theme + asset chain, not basename alone. Do not search by `Fig7` / `fig7` / `7b`.

## Legacy Pipeline Map (short)

Detection (`p16_cuda_24h_bipolar.py`) → `_gpu.npz`; refine (`p16_refine_chns_bySyn.py`) → `_refineGpu.npz`; pack (`p16_packGroupEvents*.py` driving `hfo_net.get_packedEventsTimes_overThresh`) → `_packedTimes*.npy`; lag/freq (`p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` calling `return_massCenterPat`) → `_lagPat*.npz`; 24h summary (`p16_merge24h_lagPat.py`) → `hist_meanX.npz`. Plotting in `plotting_fig{4,5,8}_*`.

**Drift**: the `_withFreqCenter` packer writes `_lagPat_withFreqCent.npz` + `_packedTimes_withFreqCent.npy`; legacy plotters still load old names. Verify which packer actually produced the artifact before tracing. Full detail: `docs/LEGACY_YUQUAN_CODEBASE_MAP.md`.

## Current Code Map (modules only)

Entry point: `scripts/run_pipeline.py`. Detection batch: `scripts/run_hfo_detection.py` (per-subject params in `config/subject_params.json`). Modules (look up specific functions via grep — names drift):

- `src/preprocessing.py` — Yuquan EDF + Epilepsiae `.data/.head` loaders, spatial-extent seizure detection
- `src/hfo_detector.py` — detection + legacy-compatible `_gpu.npz` writer, Nyquist hard gate
- `src/group_event_analysis.py` — refine / packing / centroid / lag-rank (legacy-aligned)
- `src/interictal_synchrony*.py` — event-level synchrony (PR4–PR6); CLI in `scripts/{pr6_interictal_sync_figures,run_*_interictal_synchrony,aggregate_*,compute_region_stratified_synchrony}.py`
- `src/event_periodicity.py` — pulse-train PSD, specparam, surrogates, Phase 2 tools; CLI in `scripts/{run_event_periodicity,run_surrogates_batch,run_periodicity_phase2,plot_*}.py`
- `src/interictal_propagation.py` — lagPatRank loader, KMeans cluster stereotypy + adaptive k-scan, template anchoring, rate-state coupling; CLI in `scripts/{run_interictal_propagation,plot_interictal_propagation,run_pr*,run_rank_displacement}.py`
- `src/rank_displacement.py` — PR-6 supplement; CLI in `scripts/{run,plot}_rank_displacement.py`
- `src/topic4_attractor_diagnostics.py` — Topic 4 attractor; CLI in `scripts/{run,summarize,augment,audit}_*attractor*.py`
- `src/lagpat_rank_audit.py` — Topic 0 phantom-rank fix; `build_masked_kmeans_features()` is the canonical masked-feature constructor
- `src/epilepsiae_dataset.py`, `src/network_analysis.py`, `src/visualization.py`

For per-PR scientific status, read the topic doc — NOT this file.

## Epilepsiae Contract

Read `docs/epilepsiae_dataset_structure.md` before any Epilepsiae question.

- Dataset contract: raw `*.data + *.head`; metadata truth `all_data_sqls/*.sql`; interictal artifacts under `interilca_inter_results/all_data_lns/<subject>/all_recs`.
- Trust order: SQL `recording / block / seizure` > `.head.start_ts` (block-level validation only) > legacy scripts (hints only).
- Do **not** treat `vigilance` as day/night. Current mount = `UKLFR` → `Europe/Berlin`; day/night rule = `08:00–20:00 = day`.
- Aggregation rule: synchrony is computed per event from 1h lagPat blocks. If an event's parent block crosses seizure / post-ictal / day-night / nontrivial gap boundaries, **exclude** the event — do not force-assign.
- Inventories + SOZ JSON live under `results/` (see `results/epilepsiae_*_inventory.csv`, `results/{epilepsiae,yuquan}_soz_core_channels.json`, `results/epilepsiae_electrode_focus_rel.json`).

## HFO Detector v2 (canonical since 2026-05-05)

Canonical artifact root: `results/hfo_detector_v2/`. Specs: `docs/archive/hfo_detector_v2/{v2_specification,v2_validation_contract,v2_cohort_rebuild_plan_2026-05-05}.md`.

Do **not** compare v2 events 1:1 against `results/_legacy_2021_readonly/` — that backup is historical citation only. v2 is deterministic on modern stacks (CPU=GPU, float32=float64); the 2021 cusignal vintage cannot be bit-reproduced and is not a parity target.

## Source-of-Truth Order

When answers conflict, trust in this order:

1. Files in `/mnt/yuquan_data/yuquan_24h_edf`
2. Legacy producer scripts
3. Current `HFOsp` ports of those producer scripts
4. Existing docs
5. Variable names like `fig7` / `fig8`

## Stop Conditions

Stop and ask the user instead of guessing when:

- the legacy `ReplayIED` tree is not present
- a figure clearly depends on a legacy plotting script not in this repo
- a key artifact is missing from `/mnt/yuquan_data/yuquan_24h_edf`
- an Epilepsiae request needs sub-block clinical labels that cannot be justified from 1h block outputs

## Cross-PR Contract Lookups

When a downstream PR consumes a field defined by an earlier PR, look up the accepted definition in the earlier PR's archive doc before using it — the JSON field name alone is not the contract. Frequent lookups follow.

**`lagPatRank` is phantom-contaminated (Topic 0 §3.1)** — every non-participating channel in `*_lagPat*.npz` carries a finite int rank from the legacy producer (`hfo_net.py:289` `argsort(argsort(x))` is unmasked). The `np.where(np.isfinite, ranks, 0.0)` guard in HFOsp's 4 KMeans call sites is a no-op for these phantom int values. Cohort audit (`results/lagpatrank_audit/cohort_summary.csv`, n=40) shows **40/40 subjects fail the cosmetic gate**; cohort-median AMI(original, masked) − seed_floor = -0.599. Any KMeans feature matrix derived from `lagPatRank` must go through `src.lagpat_rank_audit.build_masked_kmeans_features(ranks, bools, impute='event_median')`, or use the **`use_masked_features=True`** parameter on `compute_adaptive_cluster_stereotypy` / `compute_cluster_stereotypy` / `compute_time_split_reproducibility` / `compute_held_out_endpoint_validation` (PR-6 Step 6), or **`mask_phantom=True`** on `src.topic4_attractor_diagnostics.build_rank_feature_matrix` (Topic 4 attractor; same intent, different parameter name kept for module-local convention). **2026-05-21 phase 0 broad re-derivation completed for all downstream PR (5a–5h + Checkpoint A + Checkpoint B advisor consult), see Topic 0 §5 row-by-row**. Phase 0 verdict: large structure (K=2 dominant, within-cluster stereotypy, 86%→92% identity bias) survives phantom fix; one primary metric flipped significant→NULL (PR-4B Step 23 L3 high-confidence Pearson r, n=8, p=0.016 → 0.547, written into Topic 1 §2); three secondary metrics flipped significant→NULL (PR-5-B composition share, PR-5-B fig_a extended, PR-6 Step 4b node anatomy h1_eligible Wilcoxon p=0.014→0.059); two metrics strengthened (PR-6 Step 6 swap_class concordance 0.69→0.82, Topic 4 λ₂ 10/34→13/34). Source: `docs/topic0_methodology_audits.md` §3.1 + `docs/archive/topic0/lagpat_phantom_rank/{diagnostic_2026-05-20.md, step5{a,b,c,d.1,d.2.0,d.2.1,d.2.2,d.3,e,f,g,h}_*_2026-05-{20,21}.md, checkpoint_b_report_2026-05-21.md}`. **Runner discipline**: any new script that consumes PR-2 cluster labels MUST add `--masked-features` flag and follow the `_apply_masked_paths()` 5-line global path-swap pattern used by all 8 existing scripts (`run_interictal_propagation.py`, `run_pr6_template_anchoring.py`, `run_pr6_step6.py`, `run_rank_displacement.py`, `run_pr7_template_pairing.py`, `pr7_addendum_p3_equivalence.py`, `plot_pr7_template_pairing.py`, `run_pr5b_share_extended.py`, `run_pr5_transition_windows.py`, `plot_template_share_switching.py`, 5 `*attractor*.py` scripts).

**`forward_reverse_reproduced` (PR-2.5)** — accepted rule is **split-half OR odd-even** (8/9 subjects). The per-subject JSON exposes both `time_split_reproducibility.splits.first_half_second_half.forward_reverse_reproduced` and `splits.odd_even_block.forward_reverse_reproduced`; downstream consumers must take the OR. Checking only split-half undercounts. Source: `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md` PR-2.5 section.

**`template_rank` (PR-2 adaptive cluster)** — `adaptive_cluster.clusters[k].template_rank` is `argsort(argsort(template))`. Channels that never participate in this cluster's events still get a rank because `_legacy_hist_mean_rank` fallback assigns `template[ci] = ci`. Downstream code that picks rank extremes (source/sink, top-N) **must** derive a per-cluster `valid_mask` from raw bools and exclude non-participating channels — otherwise non-participating channels can be silently picked as endpoint members. Use `_load_bools_and_channels` (or `load_subject_propagation_events`) on the **`*_lagPat_withFreqCent.npz`** files (10ch full set), not `*_lagPat.npz` (older 7ch legacy slice).

**`channel_names` ordering** — JSON `channel_names` and any downstream `template_rank` / `template_valid_mask` indices are aligned to the same channel ordering, but raw lagPat NPZ may order them differently per block. Always re-derive the union ordering and compare against JSON `channel_names` before indexing. Mismatch means template_rank indices map to the wrong channels.

**Pre-registered hypothesis tier** — every PR plan archive declares hypothesis tiers (primary / secondary / mechanism sanity / sensitivity). Look up the tier in the plan archive when writing results; do not infer it from the data's strength. PR-6 H2 forward/reverse swap is registered as **directional mechanism sanity, not cohort claim** in `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §3.3 — never report it as "independent finding" regardless of swap_score magnitude.

**`valid_mask` semantics in PR-6 helpers** — `extract_endpoint_middle` and `compute_template_anchoring` accept `valid_mask`. Two consumer modes: split-half consumers pass `-1` sentinels in the rank vector and rely on the default mask derivation; full-data consumers must compute `valid_mask` per cluster from raw bools and pass it explicitly. **Default `valid_mask=None` for full-data input restores the buggy "all channels valid" path** — this is silent and only catchable by audit.

**Audit eligibility tiers (PR-6)** — `endpoint_defined` (n_ch ≥ 6) and `h1_primary_eligible` (n_ch ≥ 7) are orthogonal. `pass = h1_primary_eligible`. Never collapse them to a single "pass" without losing the n_ch=6 case-series subjects.

## Results Directory Standards

### 目录命名规则

- **按 topic 分类，不使用 PR 编号命名。** `pr6_analysis/` 这类命名是坏味道，应是 `interictal_synchrony/analysis/combined/`。
- 新建结果目录时，目录名必须能独立传达"这是什么分析的什么阶段输出"。

### 方法学审计后重跑的并行目录约定（Topic 0）

当某个 Topic 0 audit 触发广泛重跑时，**修复版结果走 parallel dir，旧结果不删**：

```
results/interictal_propagation/            ← 旧（phantom-contaminated），保留作为 archive evidence
results/interictal_propagation_masked/     ← 新（masked re-rank）

results/topic4_attractor/                  ← 旧
results/topic4_attractor_masked/           ← 新
```

- **`_masked` 后缀** 是 `lagPatRank` phantom audit 的命名（参考 `docs/topic0_methodology_audits.md` §4）。未来其他 audit 用其他对应词缀。
- 修复版目录里的图文件名**不重复加 `_masked`**——目录已区分。
- 对比图（before-vs-after）放专门的 `<topic>_vs_masked/` dir。
- 任何 PR 重跑时优先级：旧目录 path 在归档 doc 里仍然有效（不要 dangling），新目录 path 在新 PR 文档里使用。

### 优先级分层

1. **图（`figures/` 子目录）— 最高优先级**
   - 每次生成后用户需亲自目视检查。
   - 每个含图的目录**必须**有一个 `figures/README.md`，用中文逐图说明"这张图在展示什么，关注点在哪里"。
   - README.md 格式：`### filename` 开头，正文 2–4 句，末尾一行 `**关注点**：`。
   - 不需要每次重新生成图时重写 README，但当图的含义发生根本改变时必须更新。

2. **聚合 CSV / JSON 统计（次优先）**
   - 放在主目录下（与 `figures/` 同级），不单独建子目录。
   - 文件名体现 topic，不体现 PR 号（`pr6_statistics_summary.json` 可以保留内容但要放在正确目录）。

3. **中间 JSON / per-subject 文件（最低优先）**
   - 放在 `per_subject/`、`phase2/`、`epilepsiae/`、`yuquan/` 等子目录中，不散落在主目录。
   - 不需要 README，有 `cohort_summary.json` 提供索引即可。

### 当前规范目录结构

```
results/
├── dataset_inventory/          (epilepsiae/yuquan 元数据 inventory CSV/JSON)
├── seizure_detection/          (pr1 per-subject seizure JSON + validation)
├── event_periodicity/
│   ├── figures/
│   │   ├── README.md           ← 必须存在，中文
│   │   ├── *_cohort_psd_stack.png
│   │   ├── *_iei_summary.png
│   │   ├── epilepsiae/         (per-subject PSD)
│   │   └── yuquan/             (per-subject PSD)
│   ├── epilepsiae/             (per-subject JSON — 次优先)
│   ├── yuquan/                 (per-subject JSON — 次优先)
│   └── phase2/                 (5个实验 JSON — 次优先)
├── interictal_synchrony/
│   ├── analysis/
│   │   ├── combined/           (Epilepsiae+Yuquan 合并统计)
│   │   │   ├── figures/
│   │   │   │   └── README.md   ← 必须存在，中文
│   │   │   └── *.csv / *.json
│   │   └── yuquan/             (Yuquan 独立统计)
│   │       ├── figures/
│   │       │   └── README.md   ← 必须存在，中文
│   │       └── *.csv / *.json
│   ├── epilepsiae_ready_full_artifacts/
│   └── yuquan_soz/
├── refine_soz_validation/
│   ├── refine_soz_summary.json
│   └── figures/
│       ├── README.md             ← 必须存在，中文
│       ├── yuquan_refine_soz_cohort.png
│       └── per_subject/
├── spatial_modulation/
│   ├── gpu_audit.csv
│   ├── relaxed_refine_channel_counts.csv
│   ├── per_channel_metrics/
│   │   ├── yuquan/
│   │   └── epilepsiae/
│   └── soz_comparison/
│       ├── figures/
│       │   └── README.md ← 必须存在，中文
│       └── *.csv / *.json
├── run_logs/
└── seizure_onset/
```

### Agent 行为规范

- 生成新的图目录时，**同时生成** `figures/README.md`，不得只生成图不写说明。
- README.md 必须在图实际生成后写，不得提前占位写空内容。
- 引用结果路径时使用上述规范路径，不得出现 `pr1_`/`pr4_`/`pr6_` 开头的**目录名**（文件名内有 PR 编号可以接受，目录名不行）。
