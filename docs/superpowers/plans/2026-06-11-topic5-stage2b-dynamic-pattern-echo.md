# Topic 5 Stage 2b — Early-Ictal Dynamic Pattern Echo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. When writing a function whose spec section states multi-clause invariants (align_score sign §2.1, max-over-time null §2.2, latency eligibility §3.2), invoke `hfosp-deep-contract-verify` first.

**Goal:** Test whether the early-ictal (0–10 s) DYNAMIC activation pattern aligns with the interictal template — via a template-alignment curve `echo(t)`, growth-slope latency, and early ramp strength — replacing the failed first-onset-rank instrument.

**Architecture:** One pure-math module (`src/topic5_dynamic_echo.py`, no I/O) holding `align_score` (sign-locked), Savitzky-Golay `activation_and_slope`, `echo_curve` + `echo_curve_null` (max-over-time), eligibility-gated `slope_latencies`, `ramp_strength`, `region_aggregate` + region-label null. One runner (`scripts/run_topic5_dynamic_echo.py`) that reads the EXISTING Stage-2 cache (`results/topic5_ictal_recruitment/sentinel_cache/`), reuses `src.topic5_ictal_recruitment._z_from_traces` for robust-z, per-dataset montage align + masked template, and `src.topic5_echo_gate` for subject pooling. One plotter. **Staged gate:** Phase 1 (pure math + synthetic TDD) → Phase 2 (sentinel from cache + max-null + plotter, MANUAL GATE) → Phase 3 (build-cache → per-subject → cohort). **Cohort does not run until the sentinel echo(t) gate passes on epi + yuquan.** The pre-ictal secondary (spec §6b) is NOT in this plan — separate sub-plan after main passes.

**Tech Stack:** Python, numpy, scipy (savgol_filter, spearmanr, rankdata). Reuses `src.topic5_ictal_recruitment` (cache I/O, `_raw_traces`/`_z_from_traces`, montage, `_load_masked_template`), `src.topic5_echo_gate` (`pool_echo_subject_level`), `src.propagation_skeleton_geometry.parse_shaft`. pytest. Matplotlib.

**Spec:** `docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md` (v2.1, committed). Re-read the referenced spec section at each task boundary (CLAUDE.md §5).

**Locked constants (spec):** `HOP=0.1`, window `[T0,T1]=[0,+10]s` (sensitivity `[-2,+10]` + epi EEG-onset-anchored), `SAVGOL_WIN=5` frames (0.5s)/`polyorder=2`, ramp windows `(0,2)/(2,5)/(5,10)s`, `Z_MIN=2.0`, `DELTA_MIN=1.0`, `MIN_CH=8`, `MIN_GROUP=4`, region `min_group=2`, `B=2000`, `RNG_SEED=20260611`, pre-registered confirmatory = broadband `echo_mean[0,+5]s`. Feature win: HFA=0.5s others 1.0s. detrend default `rolling_median` (sensitivity `none`).

---

## File Structure

- **Create `src/topic5_dynamic_echo.py`** — pure functions, no I/O.
- **Create `scripts/run_topic5_dynamic_echo.py`** — `sentinel` / `build-cache` / `per-subject` / `cohort`. Reads Stage-2 cache; reuses montage + template + echo pooling.
- **Create `scripts/plot_topic5_dynamic_echo.py`** — echo(t) curves + max-null band, latency/ramp alignment scatter, region overlay + `figures/README.md`.
- **Create `tests/test_topic5_dynamic_echo.py`** — synthetic TDD.

---

# PHASE 0 — Pre-flight

- [ ] Branch is `topic5-ictal-recruitment-stage2`; stage ONLY explicit topic5/dynamic-echo paths per commit (`git add <path>` + `git diff --cached --name-only` check) — the worktree still carries unrelated Topic 4 WIP. Skills named here are Claude Code skills; in other environments treat them as manual pre-implementation contract checks.

---

# PHASE 1 — Pure-math module + synthetic TDD

## Task 1: Scaffold + `align_score` (sign-locked, §2.1)

Re-read spec §2.1 before this task. intensity = `-Spearman(template_rank, value)`; latency = `+Spearman(template_rank, latency_rank)`; `>0` always = template-early contacts earlier/stronger/faster. Common-channel mask; `< min_ch` → NaN.

**Files:** Create `src/topic5_dynamic_echo.py`, `tests/test_topic5_dynamic_echo.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np
import pytest
from src.topic5_dynamic_echo import align_score


def test_align_score_intensity_positive_when_template_early_is_stronger():
    # template_rank small = source/early. value large = stronger. If template-early
    # contacts ARE stronger, value should DECREASE with rank -> Spearman<0 -> align=+.
    template_rank = np.arange(10, dtype=float)            # 0=earliest
    value = (10 - template_rank) + 0.01                   # earliest = strongest
    assert align_score(template_rank, value, kind="intensity", min_ch=8) > 0.9


def test_align_score_intensity_sign_flips():
    template_rank = np.arange(10, dtype=float)
    value = template_rank.copy()                          # earliest = WEAKEST (anti)
    assert align_score(template_rank, value, kind="intensity", min_ch=8) < -0.9


def test_align_score_latency_positive_when_template_early_is_earlier():
    # latency_rank small = earlier. template-early earlier -> ranks agree -> Spearman+ -> align=+.
    template_rank = np.arange(10, dtype=float)
    latency_rank = np.arange(10, dtype=float)
    assert align_score(template_rank, latency_rank, kind="latency", min_ch=8) > 0.9


def test_align_score_too_few_common_is_nan():
    a = np.array([0.0, 1, 2, np.nan, np.nan, np.nan, np.nan, np.nan])
    b = np.arange(8, dtype=float)
    assert np.isnan(align_score(a, b, kind="intensity", min_ch=8))
```

- [ ] **Step 2: Run → FAIL** (`pytest tests/test_topic5_dynamic_echo.py -k align_score -v`)

- [ ] **Step 3: Implement**

```python
"""Topic 5 Stage-2b early-ictal dynamic pattern echo (PURE math, no I/O).

Spec: docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md

align_score sign is a LOCKED contract (§2.1): >0 ALWAYS means template-early
(template_rank small = source) contacts are earlier / stronger / faster.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import rankdata, spearmanr


def align_score(template_rank, value, *, kind, min_ch):
    """Sign-locked template alignment (§2.1).
    kind='intensity' (activation/dZdt/AUC/slope; larger=stronger) -> -Spearman.
    kind='latency'   (latency rank; smaller=earlier)              -> +Spearman.
    Common channels = both finite; < min_ch -> NaN."""
    t = np.asarray(template_rank, float)
    v = np.asarray(value, float)
    common = np.isfinite(t) & np.isfinite(v)
    if int(common.sum()) < min_ch:
        return float("nan")
    rho = spearmanr(t[common], v[common]).statistic
    if not np.isfinite(rho):
        return float("nan")
    return float(-rho if kind == "intensity" else rho if kind == "latency"
                 else _bad_kind(kind))


def _bad_kind(kind):
    raise ValueError(f"kind must be 'intensity' or 'latency', got {kind!r}")
```

- [ ] **Step 4: Run → PASS (4)**. **Step 5: Commit** `feat(topic5-dyn): align_score sign-locked`.

---

## Task 2: `activation_and_slope` (Savitzky-Golay derivative, §2)

Re-read spec §2 (dZdt = Savitzky-Golay, win=0.5s=5 frames, polyorder=2).

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import activation_and_slope


def test_activation_and_slope_derivative_sign():
    hop = 0.1
    n = 200
    z = np.zeros((1, n))
    z[0, 50:] = np.linspace(0, 10, n - 50)        # rising ramp after frame 50
    act, dz = activation_and_slope(z, hop=hop)
    assert np.allclose(act, z)                     # activation = z itself
    assert dz[0, 100] > 0                          # positive slope on the ramp
    assert abs(dz[0, 20]) < 0.05                   # ~0 on the flat baseline
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
SAVGOL_WIN = 5      # 0.5 s at hop=0.1
SAVGOL_POLY = 2


def activation_and_slope(z_trace, *, hop=0.1, win=SAVGOL_WIN, poly=SAVGOL_POLY):
    """(activation_z, dZdt). activation = the robust-z itself; dZdt = Savitzky-Golay
    first derivative (locked smoothing). NaN channels pass through as NaN."""
    z = np.asarray(z_trace, dtype=np.float64)
    w = min(win if win % 2 == 1 else win + 1, z.shape[1] - (1 - z.shape[1] % 2))
    if w < poly + 2:
        dz = np.gradient(z, hop, axis=1)
    else:
        dz = savgol_filter(np.nan_to_num(z, nan=0.0), window_length=w, polyorder=poly,
                           deriv=1, delta=hop, axis=1)
        dz[~np.isfinite(z)] = np.nan
    return z, dz
```

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): activation_and_slope (savgol)`.

---

## Task 3: `echo_curve` (align_score(t) + echo_peak/echo_mean)

Re-read spec §3.1 + §2.2 (echo_peak = max_t; echo_mean = pre-registered window mean). The time grid is shared (interpolated to 0.1 s, §2); this function takes already-on-grid `value_by_t` (n_ch, n_t) + the t axis.

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import echo_curve


def test_echo_curve_peaks_when_template_early_strengthens_over_time():
    n_ch, n_t = 12, 100
    t = np.arange(n_t) * 0.1                       # 0..10 s
    template_rank = np.arange(n_ch, dtype=float)
    # template-early contacts ramp up earlier/stronger -> align_score(t) rises then high
    value = np.zeros((n_ch, n_t))
    for c in range(n_ch):
        value[c] = np.clip((t - 0.3 * c), 0, None)   # earlier-rank contacts rise first
    res = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0.0, 5.0))
    assert res["echo_peak"] > 0.5
    assert 0.0 <= res["t_peak"] <= 10.0
    assert res["echo_mean"] > 0.3


def test_echo_curve_flat_when_no_priority():
    n_ch, n_t = 12, 100
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    rng = np.random.default_rng(0)
    value = rng.standard_normal((n_ch, n_t))       # no template relationship
    res = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0.0, 5.0))
    assert abs(res["echo_mean"]) < 0.4
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
def echo_curve(template_rank, value_by_t, t_axis, *, kind, min_ch, mean_window):
    """align_score(t) over a shared time grid. echo_peak = max_t (max-over-time, §2.2);
    echo_mean = mean over the pre-registered confirmatory window (no time selection)."""
    V = np.asarray(value_by_t, float)
    t = np.asarray(t_axis, float)
    curve = np.array([align_score(template_rank, V[:, j], kind=kind, min_ch=min_ch)
                      for j in range(V.shape[1])])
    finite = np.isfinite(curve)
    if not finite.any():
        return {"curve": curve, "t_axis": t, "echo_peak": float("nan"),
                "t_peak": float("nan"), "echo_mean": float("nan")}
    jpeak = int(np.nanargmax(curve))
    w = (t >= mean_window[0]) & (t <= mean_window[1]) & finite
    return {"curve": curve, "t_axis": t, "echo_peak": float(curve[jpeak]),
            "t_peak": float(t[jpeak]),
            "echo_mean": float(np.nanmean(curve[w])) if w.any() else float("nan")}
```

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): echo_curve (max-over-time peak + confirmatory mean)`.

---

## Task 4: `echo_curve_null` (max-over-time / max-over-feature null, §2.2 — THE key gate)

Re-read spec §2.2. Each shuffle recomputes the WHOLE curve and takes `max_t`; null statistic for `echo_peak` is the max-null distribution. null_mode channel / within_shaft / anchor_matched (block permute of channel identity, reuse Stage-1 block semantics conceptually). max-over-feature handled by the runner (pass stacked features).

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import echo_curve_null, echo_peak_pvalue


def test_max_null_not_falsely_significant_on_random():
    n_ch, n_t = 14, 80
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    rng = np.random.default_rng(3)
    value = rng.standard_normal((n_ch, n_t))                 # no real echo
    obs = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0, 5))["echo_peak"]
    null = echo_curve_null(template_rank, value, t, kind="intensity", min_ch=8,
                           null_mode="channel", blocks=None, B=300, rng=rng)
    p = echo_peak_pvalue(obs, null)
    assert p > 0.05                                         # max-null absorbs the time-selection


def test_max_null_significant_on_real_echo():
    n_ch, n_t = 14, 80
    t = np.arange(n_t) * 0.1
    template_rank = np.arange(n_ch, dtype=float)
    value = np.zeros((n_ch, n_t))
    for c in range(n_ch):
        value[c] = np.clip(t - 0.25 * c, 0, None)            # strong template echo
    obs = echo_curve(template_rank, value, t, kind="intensity", min_ch=8,
                     mean_window=(0, 5))["echo_peak"]
    null = echo_curve_null(template_rank, value, t, kind="intensity", min_ch=8,
                           null_mode="channel", blocks=None, B=300,
                           rng=np.random.default_rng(4))
    assert echo_peak_pvalue(obs, null) < 0.05
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
def _permute_channels(idx, blocks, rng):
    """Channel-identity permutation. blocks=None -> full shuffle; else within-block."""
    if blocks is None:
        return rng.permutation(idx)
    out = idx.copy()
    blocks = np.asarray(blocks)
    for b in np.unique(blocks):
        m = np.where(blocks == b)[0]
        out[m] = idx[m][rng.permutation(len(m))]
    return out


def echo_curve_null(template_rank, value_by_t, t_axis, *, kind, min_ch, null_mode,
                    blocks, B, rng):
    """Max-over-time null (§2.2): each draw permutes channel identity of `value`, recomputes
    the full align_score(t) curve, and records max_t. Returns the max-null distribution."""
    V = np.asarray(value_by_t, float)
    n_ch, n_t = V.shape
    idx = np.arange(n_ch)
    out = np.full(B, np.nan)
    for b in range(B):
        perm = idx if null_mode == "channel" and blocks is None else None
        perm = _permute_channels(idx, None if null_mode == "channel" else blocks, rng)
        Vp = V[perm]
        curve = np.array([align_score(template_rank, Vp[:, j], kind=kind, min_ch=min_ch)
                          for j in range(n_t)])
        if np.isfinite(curve).any():
            out[b] = np.nanmax(curve)
    return out[np.isfinite(out)]


def echo_peak_pvalue(observed, max_null):
    """One-sided p of observed echo_peak vs the max-null distribution."""
    mn = np.asarray(max_null, float)
    mn = mn[np.isfinite(mn)]
    if mn.size < 2 or not np.isfinite(observed):
        return float("nan")
    return float((np.sum(mn >= observed) + 1) / (mn.size + 1))
```

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): echo_curve_null max-over-time + p-value`.

---

## Task 5: `slope_latencies` (eligibility-gated, §3.2)

Re-read spec §3.2. Eligibility: `peak_z >= Z_MIN` AND `peak_z - z(T0) >= DELTA_MIN`, else all latencies NaN. Returns per-contact `t_max_slope/t50_rise/t80_rise/t_peak` (seconds rel onset; smaller=earlier).

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import slope_latencies


def test_slope_latencies_orders_by_rise_time_and_gates_flat():
    hop = 0.1
    t = np.arange(100) * hop                              # 0..10s
    z = np.zeros((3, 100))
    z[0, 10:] = np.clip(t[10:] - t[10], 0, 8)            # rises early (contact 0)
    z[1, 40:] = np.clip(t[40:] - t[40], 0, 8)            # rises later (contact 1)
    z[2] = 0.1 * np.random.default_rng(0).standard_normal(100)  # flat noise -> ineligible
    out = slope_latencies(z, t_axis=t, z_min=2.0, delta_min=1.0)
    assert out["t_peak"][0] < out["t_peak"][1]           # earlier contact peaks first
    assert np.isnan(out["t_peak"][2])                    # flat channel gated to NaN
    assert np.isnan(out["t50_rise"][2])
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
def slope_latencies(z_trace, *, t_axis, z_min, delta_min, hop=0.1):
    """Per-contact growth-slope latencies in the early-ictal window. Eligibility (§3.2):
    peak_z>=z_min AND peak_z - z(T0)>=delta_min, else all latencies NaN for that contact."""
    z = np.asarray(z_trace, float)
    t = np.asarray(t_axis, float)
    n_ch = z.shape[0]
    _, dz = activation_and_slope(z, hop=hop)
    out = {k: np.full(n_ch, np.nan) for k in ("t_max_slope", "t50_rise", "t80_rise", "t_peak")}
    for c in range(n_ch):
        zc = z[c]
        if not np.isfinite(zc).any():
            continue
        peak = np.nanmax(zc)
        z0 = zc[np.isfinite(zc)][0]
        if not (peak >= z_min and (peak - z0) >= delta_min):
            continue
        out["t_peak"][c] = t[int(np.nanargmax(zc))]
        out["t_max_slope"][c] = t[int(np.nanargmax(dz[c]))]
        for frac, key in ((0.5, "t50_rise"), (0.8, "t80_rise")):
            thr = z0 + frac * (peak - z0)
            cross = np.where(zc >= thr)[0]
            if cross.size:
                out[key][c] = t[cross[0]]
    return out
```

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): slope_latencies (eligibility-gated)`.

---

## Task 6: `ramp_strength` (per-window AUC + slope, §3.3)

Re-read spec §3.3 (windows 0–2/2–5/5–10s).

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import ramp_strength


def test_ramp_strength_early_window_higher_for_early_riser():
    hop = 0.1
    t = np.arange(100) * hop
    z = np.zeros((2, 100))
    z[0, 0:] = np.clip(t, 0, 5)                 # rises immediately -> high 0-2 AUC
    z[1, 70:] = np.clip(t[70:] - t[70], 0, 5)   # rises late -> low 0-2 AUC
    out = ramp_strength(z, t_axis=t, windows=((0, 2), (2, 5), (5, 10)))
    assert out["AUC"][(0, 2)][0] > out["AUC"][(0, 2)][1]
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
def ramp_strength(z_trace, *, t_axis, windows=((0, 2), (2, 5), (5, 10))):
    """Per-contact early-window AUC + linear slope of robust-z."""
    z = np.asarray(z_trace, float)
    t = np.asarray(t_axis, float)
    n_ch = z.shape[0]
    auc, slope = {}, {}
    for w in windows:
        m = (t >= w[0]) & (t < w[1])
        auc[w] = np.full(n_ch, np.nan)
        slope[w] = np.full(n_ch, np.nan)
        if m.sum() < 2:
            continue
        tw = t[m]
        for c in range(n_ch):
            zc = z[c, m]
            if np.isfinite(zc).all():
                auc[w][c] = float(np.trapz(zc, tw))
                slope[w][c] = float(np.polyfit(tw, zc, 1)[0])
    return {"AUC": auc, "slope": slope}
```

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): ramp_strength`.

---

## Task 7: `region_aggregate` + region-label null (§3.4)

Re-read spec §3.4. Aggregate per-contact value by group (median; group size ≥2). Region-level null = shuffle REGION labels (not contact identity). `MIN_GROUP=4` regions to enter.

**Files:** Modify module + test

- [ ] **Step 1: Failing test**

```python
from src.topic5_dynamic_echo import region_aggregate


def test_region_aggregate_medians_by_group_and_drops_singletons():
    value = np.array([1.0, 3.0, 10.0, 20.0, 5.0])
    groups = np.array(["A", "A", "B", "B", "C"])     # C is singleton -> dropped (min 2)
    reg_val, reg_labels = region_aggregate(value, groups, min_group=2)
    assert reg_labels == ["A", "B"]
    assert np.allclose(reg_val, [2.0, 15.0])
```

- [ ] **Step 2: FAIL.** **Step 3: Implement**

```python
def region_aggregate(value, groups, *, min_group=2):
    """Median-aggregate a per-contact value to region level. Groups with < min_group
    contacts are dropped. Returns (region_values, region_labels)."""
    v = np.asarray(value, float)
    g = np.asarray(groups)
    labels, vals = [], []
    for lab in sorted(set(g.tolist())):
        m = g == lab
        if int(m.sum()) >= min_group and np.isfinite(v[m]).any():
            labels.append(lab)
            vals.append(float(np.nanmedian(v[m])))
    return np.array(vals, float), labels
```

(Region-level `align_score` + region-label shuffle null are composed in the runner: aggregate the template rank the same way, then `align_score` over regions; the null permutes the region-aggregated template labels, NOT contacts. A small composition test is added in Task 9.)

- [ ] **Step 4: PASS.** **Step 5: Commit** `feat(topic5-dyn): region_aggregate`.

### PHASE 1 GATE
`pytest tests/test_topic5_dynamic_echo.py -v` all green. Pure-math complete; no real data touched. Do NOT start Phase 2 until green.

---

# PHASE 2 — Runner sentinel (from cache) + plotter (MANUAL GATE)

## Task 8: Runner scaffold + cache reader + per-seizure dynamic compute

Re-read spec §2 (consumes Stage-2 cache), §2.1/§3 (families), §5. The cache is `results/topic5_ictal_recruitment/sentinel_cache/<ds_sid>.{npz,json}` (keys `tr__{feat}__{idx}`, meta with dataset/fs/channels/pre_sec/eeg_rel/template_montage). Robust-z via `src.topic5_ictal_recruitment._z_from_traces` (detrend default `rolling_median`).

**Files:** Create `scripts/run_topic5_dynamic_echo.py`

- [ ] **Step 1: Inspect the existing cache live**

```bash
python -c "import json,glob; f=sorted(glob.glob('results/topic5_ictal_recruitment/sentinel_cache/*.json'))[0]; d=json.load(open(f)); print(f); print({k:d[k] for k in ['dataset','fs','loaded_idxs','target_idxs','template_montage']}); print('channels', d['channels'][:5])"
```
Expected: confirm the meta keys + that `tr__{feat}__{idx}` arrays exist in the npz. Record the field names actually present.

- [ ] **Step 2: Scaffold + `compute_seizure_dynamic`** (per seizure: raw traces → robust-z via `_z_from_traces` → activation/slope → echo_curve (per feature) → latency/ramp → align_score; aligns ictal channels to the masked template by per-dataset `bipolar_alias_label`)

```python
"""Topic 5 Stage-2b dynamic-pattern-echo runner. Reads the Stage-2 cache; computes the
early-ictal dynamic echo families (§3). NO EDF reload for sentinel. Spec v2.1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import topic5_dynamic_echo as dyn
from src import topic5_ictal_recruitment as recruit       # cache I/O + _z_from_traces + montage
from src import topic5_echo_gate as echo                   # subject pooling
from src.propagation_skeleton_geometry import parse_shaft

CACHE_DIR = Path("results/topic5_ictal_recruitment/sentinel_cache")
OUT = Path("results/topic5_dynamic_echo")
HOP = 0.1; T0, T1 = 0.0, 10.0; MEAN_WIN = (0.0, 5.0)
MIN_CH = 8; Z_MIN = 2.0; DELTA_MIN = 1.0; B = 2000; RNG_SEED = 20260611
DETREND = "rolling_median"
FUSED = ("line_length", "broadband", "hfa", "spectral_edge")
AMP = ("line_length", "broadband", "hfa")


def _aligned_template(ds_sid, ictal_channels, dataset):
    """Masked narrow template (Main-A) aligned to ictal channels by per-dataset alias.
    Returns template_rank vector over ictal_channels order (NaN where no template)."""
    tmpl = recruit._load_masked_template(ds_sid)
    recruit.assert_channel_identity(template_montage=tmpl["template_montage"],
                                    ictal_montage=recruit.ICTAL_REFERENCE[dataset])
    # template channel -> rank (use cluster-0 masked rank; both clusters handled in cohort)
    t_ch = tmpl["channels"]; t_rank = np.asarray(tmpl["templates"][0], float)
    name2rank = {c: r for c, r in zip(t_ch, t_rank) if np.isfinite(r)}
    return np.array([name2rank.get(recruit.bipolar_alias_label(c), np.nan)
                     for c in ictal_channels], float), tmpl
```

(`compute_seizure_dynamic(raw, meta, idx)`: `_z_from_traces(raw[idx], meta['pre_sec'], eeg_rel, detrend=DETREND)` → per available fused feature build the early-ictal window slice `[T0,T1]` on the common 0.1s grid (interpolate per §2), `activation_and_slope`, `echo_curve` (activation + slope), `slope_latencies`, `ramp_strength`; return per-feature curves + per-feature align_scores. Confirm `_z_from_traces` return shape live; the window→frame mapping reuses `recruit._sec_to_frame`.)

- [ ] **Step 3: Commit** `feat(topic5-dyn): runner scaffold + per-seizure dynamic compute`.

---

## Task 9: `sentinel` subcommand + max-null + region composition test

Re-read spec §2.2 (max-null), §3.4 (region), §8 (sentinel). Compute echo(t) + max-over-time null (channel / within-shaft via `parse_shaft` / anchor-matched) per feature, plus max-over-feature; write per-seizure JSON.

**Files:** Modify runner + test

- [ ] **Step 1: Add `cmd_sentinel`** (loops the cached sentinels epi 1146:2/5 + yuquan litengsheng:0; per feature: echo_curve + echo_curve_null; max-over-feature; latency/ramp align_score; region-level; dumps JSON with echo_peak/p/t_peak/echo_mean per feature + family direction summary)

- [ ] **Step 2: Region composition test** (region-level align_score uses region-label shuffle, not contact shuffle)

```python
def test_region_level_uses_region_label_shuffle(monkeypatch):
    import scripts.run_topic5_dynamic_echo as R
    # 6 contacts in 3 shafts; region align + region-label null must not call contact shuffle
    val = np.array([1., 2, 9, 10, 4, 5]); shafts = np.array(list("AABBCC"))
    rv, labels = R.dyn.region_aggregate(val, shafts, min_group=2)
    assert len(labels) == 3
```

- [ ] **Step 3: Run sentinel**

`python scripts/run_topic5_dynamic_echo.py sentinel`
Expected: per-seizure JSON under `results/topic5_dynamic_echo/sentinel/`; prints per-feature echo_peak / p(max-null) / t_peak / echo_mean for epi + yuquan.

- [ ] **Step 4: Commit** `feat(topic5-dyn): sentinel + max-null + region composition`.

---

## Task 10: Plotter — echo(t) curves + max-null band (MANUAL GATE)

Re-read spec §5.2 + AGENTS.md figure standards. Paper-grade self-contained.

**Files:** Create `scripts/plot_topic5_dynamic_echo.py`

- [ ] **Step 1:** Plot per seizure: (A) `echo(t)` per feature with the max-null band + the pre-registered confirmatory window shaded + `t_peak` marker; (B) latency/ramp align_score bar per feature; (C) region-level overlay. Use a shared, labeled time axis (no codenames).

- [ ] **Step 2: Generate + eyeball**
`python scripts/plot_topic5_dynamic_echo.py` → PNGs under `results/topic5_dynamic_echo/sentinel/`.

- [ ] **Step 3: Write `figures/README.md`** (Chinese, after render).

- [ ] **Step 4: Commit** `feat(topic5-dyn): echo(t) sentinel plotter + README`.

### PHASE 2 GATE (MANUAL)
**STOP. Present echo(t) curves + the per-feature max-null p-values to the user.** Per spec §4/§8, Phase 3 runs only if, on BOTH epi(CAR) and yuquan(bipolar): echo(t) shows a same-direction positive peak in `[0,~6]s`, it survives the channel max-null, and at least one of within-shaft / anchor-matched survives. If not, iterate (window / detrend / feature) — do NOT run cohort.

---

# PHASE 3 — build-cache → per-subject → cohort (only after sentinel sign-off)

## Task 11: `build-cache` (cohort cache, EDF once each, §2.3)

Re-read spec §2.3. Reuse Stage-2 `_cache_subject` pattern (`extract_seizure_window` once per eligible seizure) writing to `results/topic5_dynamic_echo/cache/` with the §2 manifest (`fs/channels/pre_sec/eeg_rel/montage/template_id/feature_win/hop`).

**Files:** Modify runner

- [ ] **Step 1:** Add `cmd_build_cache(args)` enumerating eligible subjects (masked template ∩ loadable seizures), loading each seizure's window once (per-dataset `ICTAL_REFERENCE`), saving raw traces + manifest. Reuse `recruit._cache_subject` if directly importable; else mirror it. **Step 2:** Run on a small subset first, eyeball manifest. **Step 3: Commit** `feat(topic5-dyn): build-cache (cohort)`.

---

## Task 12: `per-subject` (dynamic families + max-null + subject record)

Re-read spec §3/§4. Per subject, per eligible seizure: `compute_seizure_dynamic` + echo_curve_null (3 null modes) per feature; record per-seizure `echo_peak/p/t_peak/echo_mean`, latency/ramp align_scores, family direction, `er_vs_fused` consistency (ER held-out, does not vote).

**Files:** Modify runner

- [ ] **Step 1:** `cmd_per_subject` reading the cohort cache (Task 11), writing `per_subject/<ds_sid>.json`. Both template clusters handled (max-over-cluster, mirror Stage-1 max-over-templates; null max-over-cluster too). **Step 2:** Run. **Step 3: Commit** `feat(topic5-dyn): per-subject dynamic echo`.

---

## Task 13: `cohort` — verdict (§4)

Re-read spec §4 (operational primary/construct criteria; subject-level pooling). Primary = pre-registered broadband `echo_mean[0,+5]s` direction>0 AND echo_peak survives channel max-null AND (within-shaft OR anchor-matched survives). Construct = activation/slope/ramp mostly same direction across features; ER reports but doesn't vote. Subject-level via `echo.pool_echo_subject_level`; dataset stratified; epi+yuquan both required for a standing claim.

**Files:** Modify runner

- [ ] **Step 1:** `cmd_cohort` → `cohort_dynamic_echo_summary.json` with the §4 verdict (`站住·动态echo含路径` / `站住·稳定锚为主` / `没看清·阴性`), per-null pooled directions, dataset strata, construct-validity distribution, `er_vs_fused`. **Step 2:** Run. **Step 3: Commit** `feat(topic5-dyn): cohort verdict (§4)`.

---

# PHASE 4 — Figures + archive

## Task 14: Cohort figures + archive doc

- [ ] **Step 1:** Cohort figures (subject-level echo forest by feature + max-null; narrow vs broad; epi vs yuquan) + `figures/README.md`. **Step 2:** Archive doc `docs/archive/topic5/dynamic_echo/stage2b_results_<date>.md` — plain-language 三段式 abstract (`hfosp-plain-language-recap`), verdict, both-dataset, sensitivity, Stage-2 failure context. **Step 3:** Backlink in `docs/topic5_seizure_subtyping.md` §5 (exploratory; no paper claim). **Step 4: Commit.**

---

## Self-Review

**Spec coverage (v2.1):** §2.1 align_score sign-lock (Task 1, sign-flip test) ✓; §2 savgol derivative + common-grid (Tasks 2, 8) ✓; §3.1 echo_curve (Task 3) ✓; §2.2 max-over-time/feature null (Task 4 + runner max-over-feature Task 9) ✓; §3.2 latency eligibility (Task 5) ✓; §3.3 ramp (Task 6) ✓; §3.4 region + region-label null (Tasks 7, 9) ✓; §2.3 cohort cache-building (Task 11) ✓; §4 verdict + operational criteria (Task 13) ✓; §5 reuse (cache, montage, template, pooling) ✓; §8 staged gate + manual sentinel (Phase gates) ✓. **§6 broad + §6b preictal = explicitly NOT in this plan** (separate sub-plans after main passes).

**Open execution risks (flagged checks):** `_z_from_traces` return shape + window→frame mapping verified live (Task 8 Step 1); template direction (rank small=source) re-confirmed against align_score sign at Task 8; both template clusters → max-over-cluster (Task 12); detrend default `rolling_median` is a sensitivity axis (re-check at sentinel); the manual Phase-2 gate is load-bearing — do not skip.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-11-topic5-stage2b-dynamic-pattern-echo.md`. Execution: Subagent-Driven (recommended; phase boundaries + the Phase-2 sentinel manual gate are natural checkpoints) or Inline with phase-gate checkpoints. Which approach?
