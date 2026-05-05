# Cursor Plan: Legacy Scientific Alignment (ReplayIED -> HFOsp)

> Scope: yuquan24 pipeline scientific alignment for `chengshuai/FC10477Q`, then generalize.
> Owner: Cursor agent execution plan
> Mode: small, reviewable PRs with hard acceptance gates
> Date: 2026-03-27

---

## 0) Non-Negotiable Constraints

1. **Never break userspace**: existing `run_pipeline.py` usage and output names remain valid.
2. **Keep both lag domains**:
   - absolute lag (for legacy plotting/statistics)
   - relative lag (for current network/causal pipeline)
3. **GPU must be first-class test path**: all PRs must pass at least one `cuda_env` smoke test.
4. **No hidden parameter overrides**: detector thresholds only come from `hfo_detection.config`.
5. **Each PR must be independently mergeable** with deterministic checks.

---

## 1) Target Scientific Equivalence

Alignment target is not line-by-line code cloning; it is **behavioral equivalence** on core science outputs:

- Event detection sensitivity/specificity knobs: `rel/abs`, `side_thresh`, `min_last`, `max_last`
- Group window semantics (legacy packed windows/refine loop)
- Centroid/Lag semantics:
  - spectrogram mass center (`50-300Hz`, hamming, Gaussian sigma=1.5, power=3)
  - absolute lag retained for downstream analysis
- Downstream comparability:
  - lag span distribution
  - pairwise order consistency / Matching Index
  - legacy plotting inputs available without manual conversion

---

## 2) Execution Baseline (Environment + Data)

### 2.1 GPU environment (mandatory)

- Conda env: `cuda_env`
- Python: `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`
- GPU: RTX 3090 (CUDA visible)
- Sanity command:

```bash
conda run --no-capture-output -n cuda_env python -c "import cupy as cp; print(cp.__version__); print(cp.cuda.runtime.getDeviceCount())"
```

### 2.2 Dataset baseline

- Root: `/mnt/yuquan_data/yuquan_24h_edf`
- Primary regression record: `chengshuai/FC10477Q.edf`
- Legacy reference artifacts:
  - `FC10477Q_gpu.npz`
  - `FC10477Q_packedTimes.npy`
  - `FC10477Q_lagPat.npz`
  - optional: `FC10477Q_lagPat_withFreqCent.npz`

### 2.3 Smoke config baseline

- `config/smoke_gpu_chengshuai.yaml`
- Crop window: `120s` for fast iteration

---

## 3) PR Plan (File-by-File + Acceptance)

## PR-1: Preprocess/Output Contract Hardening

### Goal

Add missing preprocess legacy behavior and formalize lag outputs for both domains (absolute + relative).

### File modification list

- `src/preprocessing.py`
  - add optional legacy outer-contact dropping mode (per shaft edge drop)
  - ensure behavior is configurable and explicit in metadata
- `config/default.yaml`
  - add config flag for outer-contact drop mode
  - document default and legacy-aligned value
- `scripts/run_pipeline.py`
  - plumb new preprocess flag from config
  - include output schema version in summary
- `src/group_event_analysis.py`
  - standardize output fields:
    - `lag_abs` (absolute centroid time)
    - `lag_raw` (relative aligned lag, existing)
    - `lag_rank`
    - `lag_freq` (when available)
  - keep backward-compatible aliases where needed
- `docs/OLD_vs_NEW_algorithm_comparison.md`
  - update with explicit output contract delta closed
- `tests/` (new)
  - unit tests for lag absolute/relative consistency

### Execution standard

- no breaking CLI changes
- no removal of existing output keys without alias
- add deterministic tests for lag conversion

### Acceptance criteria

1. `groupAnalysis.npz` contains both absolute and relative lag fields.
2. With legacy-controlled inputs (old gpu + old packedTimes + old channels), shape and channel order exactly match legacy.
3. Absolute-lag aligned error median `< 5ms` on `FC10477Q` 120s comparison.
4. GPU smoke run succeeds in `cuda_env`.

---

## PR-2: Legacy Closed-Loop Refine Channel Selection

### Goal

Implement legacy-style refine loop so from-scratch runs converge toward legacy event/window counts.

### File modification list

- `src/group_event_analysis.py`
  - add refine pipeline:
    1. initial high-count channel pick
    2. build provisional group windows from picked channels
    3. recount all channels in windows
    4. produce refined core channels
  - export refine diagnostics (counts, thresholds, selected channels)
- `scripts/run_pipeline.py`
  - add optional `core_channels.source=legacy_refine` execution path
  - persist refine artifacts path in run summary
- `config/default.yaml`
  - refine params block (`pickChn_thresh`, window/count thresholds)
- `tests/` (new)
  - refine loop deterministic test with synthetic detections

### Execution standard

- refine mode opt-in, default behavior stable
- full traceability of channel selection decisions

### Acceptance criteria

1. Refine output reproducible across reruns.
2. Using same legacy `_gpu`, refined core channel list agrees with legacy baseline (exact or documented tolerance).
3. From-scratch 120s run event count moves significantly toward legacy baseline (quantified in PR report).
4. No regression in non-refine path tests.

---

## PR-3: `lagPatFreq` Standard Output + Legacy Plot Adapter

### Goal

Make frequency centroid a first-class output and provide direct compatibility path for legacy plotting stack.

### File modification list

- `src/group_event_analysis.py`
  - compute/store `lag_freq` in standard output for spectrogram mode
  - define null behavior for non-spectrogram mode
- `scripts/run_pipeline.py`
  - include `lag_freq` in `_lagPat_pipeline.npz`
  - write explicit schema metadata
- `scripts/` (new adapter script)
  - add `export_legacy_lagpat_bundle.py`:
    - exports legacy-like keys (`lagPatRaw`, `lagPatRank`, `lagPatFreq`, `eventsBool`, `chnNames`)
- `docs/` (new short spec)
  - key mapping old vs new output fields
- `tests/` (new)
  - output-key compatibility tests

### Execution standard

- legacy plot scripts can consume exported bundle without manual edits
- keep current new-format outputs intact

### Acceptance criteria

1. `_lagPat_pipeline.npz` includes frequency centroid field.
2. Adapter output can be loaded by at least one representative legacy plotting script path without key errors.
3. Frequency/time centroid arrays share same `(n_channels, n_events)` shape.

---

## PR-4: Matching Index Module + Statistical Validation Harness

### Goal

Implement MI in new codebase with eventwise and aggregate modes, and validate scientific comparability.

### File modification list

- `src/metrics/` (new)
  - `matching_index.py`
    - pairwise order agreement
    - eventwise MI
    - template MI / aggregate MI
- `src/group_event_analysis.py` or `src/network_analysis.py`
  - optional MI computation hook and save path
- `scripts/` (new)
  - `compute_mi_report.py` (legacy-vs-new comparison report)
- `tests/` (new)
  - MI formula parity tests vs known examples

### Execution standard

- MI implementation independent of visualization code
- clear separation: metric computation vs plotting

### Acceptance criteria

1. Eventwise MI results are deterministic and unit-tested.
2. On controlled comparison (`old gpu + old packedTimes`), pairwise order agreement mean `> 0.8`.
3. MI report generated for `FC10477Q` smoke and attached to PR description.

---

## 4) Cross-PR QA Matrix

Each PR must report the following matrix:

1. **Env check**
   - `cuda_env` + cupy device count
2. **Smoke run**
   - `config/smoke_gpu_chengshuai.yaml`
   - run success + output artifacts exist
3. **Controlled legacy comparison**
   - old `_gpu` + old `packedTimes` + old core channels
   - compare:
     - event count
     - participation channels
     - lag absolute error
     - lag span distribution
     - pairwise order agreement
4. **Regression**
   - existing tests still pass

---

## 5) PR Review Checklist Template

Use this in each PR description:

- [ ] Scope limited to this PR plan item
- [ ] File list matches plan
- [ ] Config docs updated
- [ ] GPU smoke run passed (`cuda_env`)
- [ ] Controlled legacy comparison attached
- [ ] No hidden parameter override introduced
- [ ] Absolute lag preserved and exported
- [ ] Backward compatibility checked

---

## 6) Exit Criteria (Plan Done)

Plan is considered complete only when all are true:

1. From-scratch run + controlled run both available and documented.
2. Absolute lag and frequency centroid are standard outputs.
3. Legacy refine loop available and validated.
4. Matching Index available with reproducible report.
5. Core scientific behavior matches legacy on agreed metrics, not just shape.

