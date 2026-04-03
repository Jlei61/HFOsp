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
- Preprocess: `src/preprocessing.py`
  - includes `detect_seizure_by_spatial_extent()` (Yuquan EDF) and `detect_seizure_by_spatial_extent_epilepsiae()` (*.data/*.head)
- HFO detection: `src/hfo_detector.py`
- Legacy-aligned refine / packing / lag:
  - `src/group_event_analysis.py`
  - look for:
    - `legacy_refine_channels_from_detections()`
    - `legacy_refine_counts_from_detection_sets()`
    - `_legacy_rehist_events_by_packing()`
    - `build_windows_from_detections()`
    - `build_windows_from_packed_times()`
    - `compute_centroid_matrix_spectrogram()`
    - `lag_rank_from_centroids()`
- Interictal synchrony (event-level, PR4–PR6):
  - `src/interictal_synchrony.py` — event row export (`build_event_rows_from_result`), legacy lagPat consumer
  - `src/interictal_synchrony_aggregation.py` — interval annotation, day/night, exclusion rules
  - `src/interictal_synchrony_analysis.py` — fixed-window tests, trajectory analysis, Figures A–E, `run_pr6_analysis()`
  - `scripts/pr6_interictal_sync_figures.py` — PR6 CLI
  - `scripts/run_epilepsiae_interictal_synchrony.py` / `scripts/run_yuquan_interictal_synchrony.py` — event CSV export
  - `scripts/aggregate_epilepsiae_interictal_synchrony.py` / `scripts/aggregate_yuquan_interictal_synchrony.py` — aggregation
- Epilepsiae dataset: `src/epilepsiae_dataset.py`
- Network: `src/network_analysis.py`
- Plotting in current repo:
  - `src/visualization.py`
  - `scripts/visualize_run.py`
- Seizure validation:
  - `scripts/pr2_seizure_validation.py` — dual-dataset (yuquan + epilepsiae) detector validation
  - `scripts/pr25_loso_validation.py` — LOSO threshold search scaffold

Do not assume current plotting covers all legacy paper figures. Many old paper figures were produced by legacy `plotting_fig*` scripts, not by the new visualization module.

## Interictal Synchrony Analysis — Current Status (2026-04-03)

**Read `docs/plans/interictal_synchrony_analysis_v4.plan.md` § "当前科学结论" for full evidence.**

- PR4–PR6 (Epilepsiae side) **completed**. Conclusion: **population-level null**.
  - 16 subjects / ~1,280,824 event rows / 232 intervals
  - Fixed-window Post vs Pre: all three metrics p > 0.35 (legacy 0.529, phase 0.380, span 0.947)
  - Within-interval trajectory: legacy p=0.290, phase p=0.933, span p=0.053 (borderline but **direction opposes** hypothesis)
  - Individual heterogeneity dominates; only 3/16 subjects weakly fit resynchronization hypothesis
- Analysis is **event-level**, not block-mean. Primary artifact: `*_interictal_sync_events.csv`.
- Metric hierarchy: **phase** (primary scientific) > **legacy** (paper-comparable) > **span** (appendix).
- Legacy metric "0.6 wall" at n_participating=3 is a **mathematical artifact** (≈0.5918), not biology.
- Core/Global are **indistinguishable** on Epilepsiae: lagPat channels ≡ legacy high-event channels, no clinical SOZ labels consumed. SQL `electrode.focus_rel` (`i`/`e`/`l`) available but not yet used.
- Paper Figure 7B/C (subject 548 = E14) **reproduced exactly**: r=0.147, p=3.2e−14 (single-subject effect).
- Yuquan PR6: event export done; interval analysis blocked on PR3 (seizure interval inventory).
- Next scientific directions: (1) focus_rel SOZ mask, (2) n_participating covariate, (3) subject stratification, (4) event-timestamp resolution, (5) prediction framing. See v4 plan for details.

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
  - `results/interictal_synchrony/epilepsiae_ready_full_artifacts/`
  - `results/interictal_synchrony/epilepsiae_ready_full_artifacts/aggregated/`
  - `results/pr6_analysis/stats/pr6_analysis_stats.json`
  - `results/pr6_analysis/figures/`
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
  - Read `docs/plans/interictal_synchrony_analysis_v4.plan.md` § "当前科学结论"
  - Short answer: **population-level null** on Epilepsiae 16 subjects; individual heterogeneity dominates
  - Paper 548/E14 reproduced exactly; not generalizable to cohort

- "What metrics should I use for synchrony?"
  - **phase** (`sync_phase_global`) = primary scientific metric (no low-channel discretization artifact)
  - **legacy** (`sync_legacy_global`) = only for backward compatibility with old paper
  - **span** (`sync_span_global`) = appendix / sensitivity only
  - See DEVELOP_PLAN.md § "指标层级判定"

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
