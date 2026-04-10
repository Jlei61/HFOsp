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

## Canonical Roots

- Current codebase: `HFOsp/`
- Canonical artifact root: `/mnt/yuquan_data/yuquan_24h_edf`
- Epilepsiae raw/sql/artifact root: `/mnt/epilepsia_data`
- Historical legacy tree:
  - repo root: `/home/honglab/leijiaxin/HFOsp/ReplayIED`
  - Yuquan mainline: `/home/honglab/leijiaxin/HFOsp/ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/`

If the legacy tree is not mounted in the current workspace, stop and ask the user for its real path. Do not invent one.

## Golden Rule

Trace by artifact, not by figure number.

Exception:

- when the user explicitly references a paper TIFF in `ReplayIED/tiffs`, read `docs/LEGACY_PAPER_TIFF_CHAIN.md` first
- even then, use TIFF visual theme plus asset chain, not TIFF basename alone

Bad:

- search `Fig7`, `Fig8`, `7b`
- assume `fig7` in a script means paper Figure 7

Good:

1. Identify which artifact the figure consumes:
   - `<record>_gpu.npz`
   - `_refineGpu.npz`
   - `<record>_packedTimes.npy`
   - `<record>_lagPat.npz`
   - `<record>_lagPat_withFreqCent.npz`
   - network outputs
2. Map that artifact back to the producing stage
3. Find the plotting script only after the data source is clear

## Legacy Pipeline Map

- Detection:
  - `p16_cuda_24h_bipolar.py`
  - helper logic often in `highEvents_yuquan0910_utils.py`
  - writes `<record>_gpu.npz`
- Legacy refine:
  - `p16_refine_chns_bySyn.py`
  - writes `_refineGpu.npz`
- Packed group events:
  - `hfo_net.py::get_packedEventsTimes_overThresh`
  - driven by `p16_packGroupEvents*.py`
  - writes `<record>_packedTimes.npy`
- Lag/frequency:
  - `return_massCenterPat`
  - `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py`
  - writes `<record>_lagPat*.npz`
- 24h core-channel summary:
  - `p16_merge24h_lagPat.py`
  - writes `hist_meanX.npz`
- Plotting:
  - `plotting_fig4_*`
  - `plotting_fig5_*`
  - `plotting_fig8_*`

Important drift:

- `p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py` writes
  - `<record>_lagPat_withFreqCent.npz`
  - `<record>_packedTimes_withFreqCent.npy`
- In that script, old `<record>_packedTimes.npy` writing is commented out.
- Many legacy plotting scripts still load old names like `<record>_lagPat.npz` and `<record>_packedTimes.npy`.
- Therefore, always verify which pack script actually produced the artifact before tracing a figure.

## Current Code Map

- Entry point: `scripts/run_pipeline.py`
- Batch HFO detection: `scripts/run_hfo_detection.py` — produces `*_gpu.npz` + `_refineGpu.npz` for both datasets
  - Per-subject parameters: `config/subject_params.json`
- Preprocess: `src/preprocessing.py`
  - includes `detect_seizure_by_spatial_extent()` (Yuquan EDF) and `detect_seizure_by_spatial_extent_epilepsiae()` (*.data/*.head)
  - includes `load_epilepsiae_block()` — full signal loader for Epilepsiae `.data/.head` (CAR or bipolar)
- HFO detection: `src/hfo_detector.py`
  - `save_detection_as_gpu_npz()` — persist detection result as legacy-compatible `*_gpu.npz`
  - `build_legacy_alias_map()` — bipolar→left-contact alias for backward compatibility
  - Nyquist hard gate: `sfreq/2 <= band_upper` raises `ValueError`
- Legacy-aligned refine / packing / lag:
  - `src/group_event_analysis.py`
  - look for:
    - `legacy_refine_channels_from_detections()`
    - `legacy_refine_counts_from_detection_sets()`
    - `save_refine_gpu_npz()` — produce + persist `_refineGpu.npz`
    - `_legacy_rehist_events_by_packing()`
    - `build_windows_from_detections()`
    - `build_windows_from_packed_times()`
    - `compute_centroid_matrix_spectrogram()`
    - `lag_rank_from_centroids()`
- Interictal synchrony (event-level, PR4–PR6):
  - `src/interictal_synchrony.py` — event row export (`build_event_rows_from_result`), legacy lagPat consumer, `select_core_penumbra_mask` (with zero-overlap fallback)
  - `src/interictal_synchrony_aggregation.py` — interval annotation, day/night, exclusion rules
  - `src/interictal_synchrony_analysis.py` — fixed-window tests, trajectory analysis, event rate test, region-stratified analysis, Figures A–F, `run_pr6_analysis()`
  - `scripts/pr6_interictal_sync_figures.py` — PR6 CLI
  - `scripts/run_epilepsiae_interictal_synchrony.py` / `scripts/run_yuquan_interictal_synchrony.py` — event CSV export (both support `--soz-core-json`)
  - `scripts/aggregate_epilepsiae_interictal_synchrony.py` / `scripts/aggregate_yuquan_interictal_synchrony.py` — aggregation
  - `scripts/compute_region_stratified_synchrony.py` — augment event CSV with per-region (i/l/e) synchrony columns
- Event periodicity (Fig 3C / S7 / S13 verification):
  - `src/event_periodicity.py` — pulse train PSD, specparam, IEI MLE, ISI-shuffle/Gamma surrogate, Phase 2 tools (hazard, return map, packing sweep, centroid bypass, propagation stereotypy)
  - `scripts/run_event_periodicity.py` — Phase 1 dual-dataset batch driver
  - `scripts/run_surrogates_batch.py` — group-only surrogate batch
  - `scripts/plot_event_periodicity.py` — Phase 1 cohort figures
  - `scripts/run_periodicity_phase2.py` — Phase 2 experiments (10 experiments: exp1–5 artifact localization, exp6 PR-1, exp7 PR-2, exp7b PR-2.5, exp7c PR-2.6, exp7d PR-2.7)
  - `scripts/plot_periodicity_phase2.py` — Phase 2 visualization (exp1–7d)
  - `tests/test_event_periodicity.py` — PR-2 / PR-2.5 / PR-2.6 / PR-2.7 function unit tests
  - `docs/event_periodicity_analysis.md` — main results, code map, and current conclusion (Phase 4+5 + PR-1 to PR-2.7)
  - `docs/event_periodicity_phase2_review_2026-04-05.md` — detailed scientific/statistical review of Phase 2
  - `docs/interictal_population_event_methodological_review.md` — collaborator-facing narrative update and next-step framing
- Spatial modulation / SOZ analysis (Where question):
  - `docs/spatial_modulation_soz_analysis.md` — plan and results for per-channel SOZ spatial attribution + HFO detection infrastructure (§8)
  - `scripts/audit_gpu_npz.py` — Step 0 data audit (Yuquan PASS 11/18, Epilepsiae FAIL 0/20)
  - `scripts/run_spatial_modulation.py` — PR-1 batch driver (Yuquan-only, 9 valid pairs)
  - `scripts/plot_spatial_modulation.py` — PR-1 figures
  - `scripts/run_hfo_detection.py` — batch HFO detection for both datasets (`--dataset yuquan/epilepsiae --subject/--all`)
  - `src/event_periodicity.py` — `load_perchannel_events_relaxed()`, `compute_perchannel_metrics()`, `annotate_channels_soz()`, `match_bipolar_soz()`
- Epilepsiae dataset: `src/epilepsiae_dataset.py`
- Network: `src/network_analysis.py`
- Plotting in current repo:
  - `src/visualization.py`
  - `scripts/visualize_run.py`
- Seizure validation:
  - `scripts/pr2_seizure_validation.py` — dual-dataset (yuquan + epilepsiae) detector validation
  - `scripts/pr25_loso_validation.py` — LOSO threshold search scaffold

Do not assume current plotting covers all legacy paper figures. Many old paper figures were produced by legacy `plotting_fig*` scripts, not by the new visualization module.

## Interictal Synchrony Analysis — Current Status (2026-04-04)

**Read `docs/interictal_synchrony_preliminary_report_2026-04-03.md` for the full statistical report.**

- PR4–PR6 **completed for both Epilepsiae + Yuquan**. Overall conclusion: **population-level null** for phase synchrony.
  - Combined: 29 subjects / 1,468,780 event rows / 253 intervals / 141 fixed-window intervals
  - Phase_all Post vs Pre: p = 0.279, r = 0.106. Phase_core: p = 0.967.
  - Within-interval trajectory: phase_all p = 0.589, phase_core p = 0.643.
  - Event rate (events/hour): post vs pre p = 0.361 — no change in HFO group event density across windows.
- **Region-stratified analysis (i/l/e)** completed on Epilepsiae:
  - SOZ (`i`): p = 0.646, no effect. Lesion (`l`): p = 0.543, no effect.
  - **Extra-focal (`e`): p = 0.012, r = 0.31** (pre > post, medium effect). Bonferroni-corrected p = 0.037.
  - Trajectory for `e`: p = 0.129 (same direction, not significant).
  - Interpretation: seizures may transiently disrupt extra-focal synchrony; SOZ itself is unaffected. Exploratory-significant, needs larger sample validation.
- Analysis is **interval-first** (seizure_interval as statistical unit). Primary artifact: `*_interictal_sync_events.csv`.
- Metric hierarchy: **phase** (primary scientific) > **legacy** (paper-comparable) > **span** (appendix).
- Legacy metric "0.6 wall" at n_participating=3 is a **mathematical artifact** (≈0.5918), not biology.
- SOZ definitions:
  - Epilepsiae: `electrode.focus_rel == 'i'` from SQL → `results/epilepsiae_soz_core_channels.json`, `results/epilepsiae_electrode_focus_rel.json` (i/l/e per channel)
  - Yuquan: `p16_subs_info.py` hand-annotated → `results/yuquan_soz_core_channels.json` (20 subjects with non-empty SOZ)
- Paper Figure 7B/C (subject 548 = E14) **reproduced exactly**: r=0.147, p=3.2e−14 (single-subject effect).
- Next scientific directions: (1) validate phase_e effect with bootstrap CI, (2) per-subject case series on phase_e, (3) event-timestamp resolution, (4) prediction framing.

## Epilepsiae Contract

Read `docs/epilepsiae_dataset_structure.md` before answering any Epilepsiae question.

- Dataset contract:
  - raw signal: `*.data + *.head`
  - metadata truth: `all_data_sqls/*.sql`
  - interictal artifacts: `interilca_inter_results/all_data_lns/<subject>/all_recs`
- Trust order inside Epilepsiae:
  1. SQL `recording / block / seizure`
  2. `.head.start_ts` only for block-level validation
  3. legacy scripts only as hints
- Do not treat `vigilance` as day/night.
- Current mounted data resolves to `hospital=UKLFR` -> `Europe/Berlin`.
- Day/night rule for current Epilepsiae analysis: `08:00-20:00 = day`, else `night`.
- Reusable current interfaces:
  - `src.epilepsiae_dataset`
  - `src.interictal_synchrony`
  - `src.interictal_synchrony_aggregation`
  - `src.interictal_synchrony_analysis`
- Current machine-readable outputs:
  - `results/epilepsiae_subject_inventory.csv`
  - `results/epilepsiae_recording_inventory.csv`
  - `results/epilepsiae_block_inventory.csv`
  - `results/epilepsiae_seizure_inventory.csv`
  - `results/epilepsiae_sync_subject_manifest.csv`
  - `results/epilepsiae_soz_core_channels.json` — SOZ (i-labeled) channels per subject
  - `results/epilepsiae_electrode_focus_rel.json` — per-channel i/l/e labels per subject
  - `results/yuquan_soz_core_channels.json` — Yuquan SOZ channels per subject
  - `results/interictal_synchrony/epilepsiae_ready_full_artifacts/`
  - `results/interictal_synchrony/epilepsiae_ready_full_artifacts/aggregated/`
  - `results/interictal_synchrony/epilepsiae_ready_full_artifacts/epilepsiae_region_stratified_events.csv`
  - `results/interictal_synchrony/yuquan_soz/` — Yuquan SOZ-stratified event CSV + aggregated
  - `results/interictal_synchrony/analysis/combined/pr6_statistics_summary.json` — combined cohort statistics
  - `results/interictal_synchrony/analysis/combined/figures/` — Figures B–F + per-subject timelines
  - `results/interictal_synchrony/analysis/yuquan/figures/` — Yuquan-only Figures A–E
- Aggregation rule:
  - synchrony is computed per event from 1h lagPat blocks; analysis consumes event-level rows
  - do not invent sub-block seizure / post-ictal / day-night labels
  - if an event's parent block crosses seizure, post-ictal, day-night, or nontrivial gap boundaries, exclude the event instead of force-assigning it

## Source-of-Truth Order

When answers conflict, trust them in this order:

1. Files in `/mnt/yuquan_data/yuquan_24h_edf`
2. Legacy producer scripts
3. Current `HFOsp` ports of those producer scripts
4. Existing docs
5. Variable names like `fig7`, `fig8`

## Stop Conditions

Stop and ask the user instead of guessing when:

- the legacy `ReplayIED` tree is not present
- a figure clearly depends on a legacy plotting script not in this repo
- a key artifact is missing from `/mnt/yuquan_data/yuquan_24h_edf`
- an Epilepsiae request needs sub-block clinical labels that cannot be justified from 1h block outputs

## Fast Path For Common Questions

- "What did the interictal synchrony analysis find?"
  - Read `docs/interictal_synchrony_preliminary_report_2026-04-03.md`
  - Short answer: **population-level null** on 29 subjects (Epilepsiae + Yuquan); individual heterogeneity dominates
  - **One exploratory finding**: extra-focal (`e`) phase synchrony is pre > post (p=0.012, r=0.31); SOZ (`i`) and lesion (`l`) show no effect
  - Paper 548/E14 reproduced exactly; not generalizable to cohort

- "What metrics should I use for synchrony?"
  - **phase** (`sync_phase_global`) = primary scientific metric (no low-channel discretization artifact)
  - **legacy** (`sync_legacy_global`) = only for backward compatibility with old paper
  - **span** (`sync_span_global`) = appendix / sensitivity only
  - See DEVELOP_PLAN.md § "指标层级判定"

- "Is the ~2Hz event periodicity real?"
  - Read `docs/event_periodicity_analysis.md`
  - Then read `docs/interictal_population_event_methodological_review.md` if the question is about scientific narrative / mechanism
  - Short answer: **NO.** The ~2Hz PSD peak is not evidence for an intrinsic oscillator; current evidence supports a refractory / dead-time artifact plus slow rate modulation.
  - Gamma renewal null (matching firing rate + refractory period) explains 15/21 subject peaks with specparam peaks
  - Analytic renewal PSD overlay (PR-1 exp6A): 16/21 |Δf| < 1 Hz; **two independent paths cover 19/21 (90%)**
  - Escaping 2/21 (1084, 1096) **resolved by PR-2.5 backfill**: detrended PSD shows peaks completely disappear → slow rate modulation, not oscillator
  - ISI-shuffle shows peaks are distribution-shape artifacts, not temporal-order effects
  - IEI distribution is lognormal (30/30), NOT power-law as old paper claimed
  - Only 1/30 subjects passes both surrogate tests (huanghanwen, n=484, likely false positive)
  - Per-channel vs group peak frequencies are inconsistent (packing artifact)
  - SOZ dead-time (PR-1 exp6B): SOZ < non-SOZ (p=0.008, n=8, exploratory); 22/30 subjects have nearly zero pure non-SOZ group events
  - Phase 2 (artifact localization):
    - f_peak ≠ 1/W across 9 window sizes → packing window size is not the direct cause
    - Centroid bypass: within the current legacy `lagPatRaw -> absolute time` mapping, most subjects keep similar peaks; this is not yet a fully independent timestamp reconstruction
    - IEI serial correlation is **positive for all 30 subjects** on log-IEI; formal reporting should use subject-level direction consistency / sign test, not naive within-subject Pearson p-values
    - Hazard curves are qualitative dead-time visualizations, not formal survival-analysis estimates
    - Propagation stereotypy is the part most likely to reflect real network structure, but SOZ > non-SOZ is still exploratory rather than definitive
  - PR-2 (exp7, 2026-04-08): lag-k serial correlation deep dive on 30 subjects:
    - Half-life median = 107.5s ≈ 1.8 min; 6/30 never reach half (persistent slow modulation)
    - 600s rolling-median detrending: **~72% of positive correlation is slow rate drift** (>10 min), **~28% is short-range dependency**; 27/30 still positive after detrending
    - Within-block pooled: 30/30 positive (cross-block contamination ruled out)
    - SOZ vs non-SOZ: SOZ median 0.302 > nonSOZ 0.132, p=0.055 (n=9, borderline)
  - PR-2.5 (exp7b, 2026-04-08): multi-scale modulation anatomy on 30 subjects:
    - Δ_frac nearly flat (0.080–0.147) → broad-band (1/f-like) modulation, no single dominant timescale; minor peak ~329s (~5.5 min)
    - n_participating Spearman autocorrelation: cross-corr with IEI decay median 0.742 (18/30 > 0.7) → **single global state variable confirmed**
    - Day/night stratified detrending: 28/30 still positive after within-segment detrending (day median 0.094, night 0.086) → short-range dependency is genuine, not day/night boundary artifact
    - Backfill for escape subjects 1084, 1096: **peaks completely disappear after 600s detrending** → Layer 3 gap closed, 21/21 specparam peaks fully explained by renewal + slow modulation
  - PR-2.6 (exp7c, 2026-04-09): continuous long-timescale analysis on 30 subjects:
    - Yuquan 10/10 subjects provide near-24h continuous trajectories; Epilepsiae longest continuous run median = 75.1h, observed duration median = 158.4h
    - Continuous-time binned rate autocorrelation (5-min bins) remains positive from 0.5h to 8h lags in Epilepsiae (median 0.108 at 8h); Yuquan positive to 4h but ~0 at 8h. Note: this is binned rate autocorrelation, not a direct extension of IEI lag-k serial correlation
    - Continuous day/night segment analysis: Yuquan 9/10 and Epilepsiae 17/20 remain positive on both sides → short-range dependency survives inside continuous day/night segments
  - PR-2.7 (exp7d, 2026-04-09): rate-trace spectral characterization + seizure proximity on 30 subjects:
    - Rate-trace PSD β: cohort median = 0.64 (range 0.04–1.62), r² median 0.709 → sub-pink-noise but still confirms long-range dependence beyond white noise; consistent with PR-2.5 broad-band Δ_frac
    - Rate × n_participating coherence: after fixing multi-span spectral averaging bug, median = 0.358 (4/26 > 0.5) → only weak/moderate frequency-domain coupling; do NOT overclaim a strong single global state variable
    - **Seizure-triggered rate average: pre-ictal [-6h,-1h] vs baseline [-12h,-6h] Wilcoxon p=0.019, 16/21 pre > baseline, but post > pre (p=0.016)** → supports seizure-centered broad rate elevation, not yet a pure pre-ictal ramp. Primarily Epilepsiae-driven (21 subjects, 328 usable seizure windows)
  - Next: PR-3 (stereotypy robustness with centering SOZ-erasure diagnostic)
  - Method caveats incorporated: detrend_fraction curve behavior depends on window vs modulation timescale (use Δ_frac for band localization); n_participating must use Spearman not Pearson-on-log; centered rank tau must check SOZ source node preservation
  - See `results/event_periodicity/` and `results/event_periodicity/phase2/` for full results

- "Where does the slow IEI modulation occur? Is it SOZ-specific?"
  - Read `docs/spatial_modulation_soz_analysis.md`
  - Short answer: **PR-1 completed (Yuquan-only, n=9)**. Raw serial corr shows **no SOZ difference** (p=1.0). But **detrend_fraction is lower in SOZ** (7/9 subjects, p=0.129) — SOZ channels have more short-range memory, less slow drift. SOZ median IEI is shorter (p=0.055, marginal).
   - Epilepsiae gpu.npz are all corrupt stubs (216 bytes); per-channel analysis requires HFO re-detection via `scripts/run_hfo_detection.py --dataset epilepsiae --all`
  - Per-channel approach (relaxed refine k=0.0) successfully expands channel set from lagPat ~10 to ~33, with 9/11 subjects forming valid SOZ/nonSOZ pairs
  - Key insight: lagPat-based SOZ > nonSOZ serial corr (PR-2 exp7D, p=0.055) was confounded by SOZ's higher event rate → global modulation effect. Per-channel detrending separates this into global (no difference) + local (SOZ has more short-range memory).
  - Epilepsiae three-tier (i/l/e) gradient analysis deferred to PR-2

- "Why is legacy synchrony always ~0.6?"
  - Mathematical artifact: 3-channel uniform lag pattern → theoretical limit ≈ 0.5918
  - Not biology. Filter by `n_participating >= 5` or use **phase** metric instead.

- "Where does `_refineGpu.npz` come from?"
  - Read `docs/LEGACY_YUQUAN_CODEBASE_MAP.md`
  - Then inspect `p16_refine_chns_bySyn.py`
  - In current code, inspect `legacy_refine_counts_from_detection_sets()`

- "Where does `lagPat` come from?"
  - Read `docs/OLD_vs_NEW_algorithm_comparison.md`
  - Then inspect `p16_packGroupEvents*.py`
  - In current code, inspect `compute_centroid_matrix_spectrogram()`

- "Which script made a paper figure?"
  - First identify the artifact class from axis labels and annotations
  - Then map artifact -> producer -> plotter
  - Never start from figure numbering alone

## Results Directory Standards

### 目录命名规则

- **按 topic 分类，不使用 PR 编号命名。** `pr6_analysis/` 这类命名是坏味道，应是 `interictal_synchrony/analysis/combined/`。
- 新建结果目录时，目录名必须能独立传达"这是什么分析的什么阶段输出"。

### 优先级分层

1. **图（`figures/`子目录）— 最高优先级**
   - 每次生成后用户需亲自目视检查。
   - 每个含图的目录**必须**有一个 `figures/README.md`，用中文逐图说明"这张图在展示什么，关注点在哪里"。
   - README.md 格式：`### filename`开头，正文2–4句，末尾一行`**关注点**：`。
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
