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
- Network: `src/network_analysis.py`
- Plotting in current repo:
  - `src/visualization.py`
  - `scripts/visualize_run.py`

Do not assume current plotting covers all legacy paper figures. Many old paper figures were produced by legacy `plotting_fig*` scripts, not by the new visualization module.

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

## Fast Path For Common Questions

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
