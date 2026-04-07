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
  - `scripts/run_periodicity_phase2.py` — Phase 2 experiments (5 experiments)
  - `scripts/plot_periodicity_phase2.py` — Phase 2 visualization
  - `docs/event_periodicity_analysis.md` — full results and conclusions (Phase 4+5)
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
  - `results/pr6_analysis_soz/pr6_statistics_summary.json` — combined cohort PR6 statistics
  - `results/pr6_analysis_soz/figures/` — Figures A–F
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
  - Short answer: **NO.** The ~2Hz PSD peak is a refractory-period artifact.
  - Gamma renewal null (matching firing rate + refractory period) explains 15/30 subject peaks entirely
  - ISI-shuffle shows peaks are distribution-shape artifacts, not temporal-order effects
  - IEI distribution is lognormal (30/30), NOT power-law as old paper claimed
  - Only 1/30 subjects passes both surrogate tests (huanghanwen, n=484, likely false positive)
  - Per-channel vs group peak frequencies are inconsistent (packing artifact)
  - Phase 2 (artifact localization):
    - f_peak ≠ 1/W across 9 window sizes → packing window size is not the direct cause
    - Centroid bypass: 3 event definitions give identical peaks → not grid quantization
    - IEI serial correlation is **positive for all 30 subjects** → slow rate modulation, not oscillator
    - Propagation stereotypy: SOZ events more stereotyped (tau=0.119 vs 0.048, p=0.077)
  - See `results/event_periodicity/` and `results/event_periodicity/phase2/` for full results

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
