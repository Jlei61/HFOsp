# Topic5 Contact-Similarity Geometry-Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-rung geometry-ladder adjudication of the A-line field-similarity result, isolating the smoothing and grid contributions of the field on identical inputs, plus a separate Spearman/Kendall contact-sequence sanity track.

**Architecture:** A new module `src/topic5_contact_similarity.py` provides (1) a grid-free Gaussian contact kernel proven equal to the existing `smooth_field`, (2) raw/kernel polarity-free maxAB similarity replicating the field's polarity handling, (3) a per-seizure→median-over-seizures null-fold harness that takes a pluggable per-seizure statistic, (4) a Spearman/Kendall sequence-similarity helper. A runner plugs three statistics (R1 raw / R2 same-plane kernel / R3 field) into the SAME harness so the only difference between rungs is geometry. A plot script renders three independent panels. All heavy IO reuses `scripts/run_topic5_axis_alignment.py` patterns.

**Tech Stack:** Python, numpy, scipy.stats (pearsonr/spearmanr/kendalltau), matplotlib, pytest. Reuses `src/propagation_contact_plane_readout.py`, `src/topic5_axis_alignment.py`, `src/propagation_skeleton_geometry.py`.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-06-30-topic5-contact-similarity-ladder-design.md`. Every task's requirements implicitly include these.

- `B = 1000` (= existing maxAB artifact); `seed = 20260614`.
- `MIN_CH = 6` (matched contacts; = A-line `run_topic5_axis_alignment.py:91`).
- `MIN_FINITE_PER_SZ = 6` (per-seizure finite paired contacts; = L141).
- `MIN_SHAFTS = 2`; `MIN_EFFECTIVE_SHUFFLE_N = 4` (else `INSUFFICIENT_NULL`).
- `t_a` required; `t_b` optional, needs ≥4 finite (= L114); absent t_b → single-template, null also single-template.
- `SESOI = 0.05` (equivalence margin for paired Δ).
- ladder primary correlation = **Pearson**; sequence-sanity = **Spearman + Kendall**.
- Gate is **sign-free**: `obs = max(|sim_A|, |sim_B|)` per seizure, `obs_subject = median_s`. signed corr = sidecar only.
- Null: **per draw recompute maxAB** (both A,B on the SAME shuffled vector); fold = `median_s` per draw then percentile (replicates `_p95_med`). Save full quantiles p5/p50/p95/p99.
- R2 = R3 minus grid: reuse R3's `x_norm/y_norm` + `sigma_xy` + `support`; mirror flips **eval points** y only.
- R3 recomputed in this runner at the same B/seed/statistic (NOT just read); cross-check vs existing artifact; does not change A-line main result.
- Results dir: `results/topic5_ictal_recruitment/contact_similarity/`. Subject-level only, no cross-dataset point-cloud pooling.
- Output-name contract (P1-1, matches `axis_alignment` convention): ONE namespaced dir, not one dir per activation — filenames carry the activation (`cohort_summary_{activation}.{json,csv}`), not the directory name. No `contact_similarity_hfa/`.
- Tier: sensitivity/robustness; no new cohort claim. Conclusion language gated per spec §9.

---

### Task 0: Verify input-data access (worktree reads inputs from main tree)

**Files:** No code change.

**Context:** This worktree is checked out from `origin/main`; `results/` exists as a **real directory** but the **gitignored INPUTS** — t0 feature cache, axis records (`*_t_a.json`/`*_t_b.json`), the existing maxAB artifact — live ONLY in the main working tree at `/home/honglab/leijiaxin/HFOsp/results`. Do **NOT** symlink the whole `results/` (`[ -e results ]` is already true, so a guarded symlink is a silent no-op and inputs stay missing). The runner (Task 5) reads inputs via `--input-results-root` (default `results`; pass the main-tree path in this worktree) and writes outputs into the worktree `results/` (gitignored locally).

- [ ] **Step 1: Confirm the main-tree input paths resolve**

```bash
ROOT=/home/honglab/leijiaxin/HFOsp/results
ls "$ROOT/topic5_ictal_recruitment/t0_feature_cache/" | head -2
ls "$ROOT/topic5_ictal_recruitment/axis_alignment/axis_alignment_broadband_max_ab_B1000.json"
ls "$ROOT/spatial_modulation/propagation_geometry/observation_readout/real_subjects/" | grep _t_a | head -2
```
Expected: cache `.npz`, the maxAB json, and `*_t_a.json` records all listed.

- [ ] **Step 2: Baseline tests (src is tracked/shared, importable in the worktree)**

```bash
python -m pytest tests/ -k "axis_alignment or contact_plane or skeleton_geometry" -q
```
Expected: PASS (or report pre-existing failures and ask before proceeding).

- [ ] **Step 3: Record the worktree invocation convention**

Every `run_topic5_contact_similarity.py` call in this worktree passes
`--input-results-root /home/honglab/leijiaxin/HFOsp/results`
(the default `results` is correct only in a normal main-tree checkout). No commit (no file change).

---

### Task 1: Grid-free Gaussian contact kernel (proven equal to `smooth_field`)

**Files:**
- Create: `src/topic5_contact_similarity.py`
- Test: `tests/test_topic5_contact_similarity.py`

**Interfaces:**
- Consumes: `smooth_field`, `make_plane_grid`, `make_field_record`, `R_smooth_rank` from `src.propagation_contact_plane_readout` / `src.topic5_axis_alignment` (read their exact signatures at `src/propagation_contact_plane_readout.py:230,24,439` and `src/topic5_axis_alignment.py:31`).
- Produces: `kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma) -> np.ndarray (n_eval,)`.

**Critical contract:** the per-point weight expression MUST be byte-identical to `smooth_field`'s. Do not guess `sig2`; copy the exact weight line from `smooth_field` (`src/propagation_contact_plane_readout.py:253-267`). The Step-2 cross-check test fails unless the kernel matches `smooth_field` on the grid.

- [ ] **Step 1: Write the cross-check + reduction failing tests**

```python
# tests/test_topic5_contact_similarity.py
import numpy as np
import pytest
from src.topic5_contact_similarity import kernel_smooth_at_contacts
from src.propagation_contact_plane_readout import smooth_field, make_plane_grid


def _toy_pts():
    # 3 contacts, irregular spacing on the plane
    pts = np.array([[0.0, 0.0], [0.4, 0.1], [1.0, -0.2]])
    vals = np.array([0.0, 1.0, 2.0])
    sup = np.ones(3)
    return pts, vals, sup


def test_kernel_matches_smooth_field_on_grid():
    """R2 kernel ≡ R3 field math: evaluating the contact kernel at the grid
    points reproduces smooth_field's grid (at finite, well-supported pixels)."""
    pts, vals, sup = _toy_pts()
    sigma = 0.3
    X, Y = make_plane_grid()
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])   # (N,2): column_stack of two 1D arrays is correct
    record = {"channels": [{"x_norm": float(p[0]), "y_norm": float(p[1]),
                            "support": float(s), "typical_rank": float(v)}
                           for p, s, v in zip(pts, sup, vals)]}
    field = smooth_field(record, X, Y, sigma_xy=sigma, s_thresh=0.0)  # real sig: record-first, returns {"T","S"}
    T = field["T"]
    mine = kernel_smooth_at_contacts(vals, pts, grid_pts, sup, sigma).reshape(X.shape)
    m = np.isfinite(T) & np.isfinite(mine)
    assert m.sum() > 100
    assert np.allclose(T[m], mine[m], atol=1e-9)


def test_kernel_reduces_to_self_value_as_sigma_to_zero():
    """σ→0: each eval point (=its own contact) returns its own value."""
    pts, vals, sup = _toy_pts()
    out = kernel_smooth_at_contacts(vals, pts, pts, sup, sigma=1e-4)
    assert np.allclose(out, vals, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -q`
Expected: FAIL with `ImportError`/`cannot import name 'kernel_smooth_at_contacts'`.

- [ ] **Step 3: Implement `kernel_smooth_at_contacts`**

First READ `src/propagation_contact_plane_readout.py:230-270` and copy the exact weight/normalization. The skeleton (align the weight line to the real `smooth_field`):

```python
# src/topic5_contact_similarity.py
"""Topic5 contact-level similarity ladder (R1 raw / R2 same-plane kernel),
grid-free counterparts of the field maxAB. See
docs/superpowers/specs/2026-06-30-topic5-contact-similarity-ladder-design.md."""
import numpy as np


def kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma):
    """Nadaraya-Watson Gaussian smoothing identical to smooth_field, but
    evaluated at arbitrary eval_pts instead of a grid (no grid -> no pixel
    density reweighting). Mirror = pass y-flipped eval_pts.

    values, support: (n_src,); source_pts: (n_src,2); eval_pts: (n_eval,2).
    Returns (n_eval,) with NaN where total support <= 1e-12.
    """
    v = np.asarray(values, float)
    sup = np.asarray(support, float)
    src = np.asarray(source_pts, float)
    ev = np.asarray(eval_pts, float)
    sig2 = 2.0 * float(sigma) * float(sigma)   # MUST match smooth_field's sig2 exactly
    out = np.full(ev.shape[0], np.nan)
    fin = np.isfinite(v)
    for i in range(ev.shape[0]):
        d2 = (src[:, 0] - ev[i, 0]) ** 2 + (src[:, 1] - ev[i, 1]) ** 2
        w = sup * np.exp(-d2 / sig2)
        wsum_all = w.sum()                      # support gate uses all sources (as smooth_field S)
        if wsum_all <= 1e-12:
            continue
        wf = w[fin]
        if wf.sum() > 1e-12:
            out[i] = float((wf * v[fin]).sum() / wf.sum())
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -q`
Expected: PASS. If `test_kernel_matches_smooth_field_on_grid` fails, the `sig2` or weight line differs from `smooth_field` — fix to match exactly.

- [ ] **Step 5: Commit**

```bash
git add src/topic5_contact_similarity.py tests/test_topic5_contact_similarity.py
git commit -m "feat(topic5): grid-free contact kernel equal to smooth_field"
```

---

### Task 2: Polarity-free maxAB similarity (raw + same-plane kernel, mirror-faithful)

**Files:**
- Modify: `src/topic5_contact_similarity.py`
- Test: `tests/test_topic5_contact_similarity.py`

**Interfaces:**
- Consumes: `kernel_smooth_at_contacts` (Task 1).
- Produces:
  - `contact_corr(rank, value, *, mode, source_pts, support, sigma, mirror=False) -> float` (single-template signed Pearson; `mode in {"raw","kernel"}`).
  - `polarity_free_maxab(rank_a, rank_b, value, *, mode, source_pts, support, sigma) -> float` (= `max(|abs_mirror_corr_A|, |abs_mirror_corr_B|)`, replicating `_abs_corr`+`window_maxab`).

**Replication contract (read `scripts/run_topic5_axis_alignment.py:59` `_abs_corr` (script-level, replicate locally), `src/propagation_contact_plane_readout.py:285` `corr_pair_mirror_invariant`, `scripts/run_topic5_axis_alignment.py:117-127` `window_maxab`):** per template `t`, `r_t = abs(max(c_identity, c_mirror))` for kernel mode (mirror = activation field evaluated at y-flipped eval points; rank field stays identity), `r_t = abs(pearson(rank, value))` for raw mode. `maxAB = max(r_A, r_B)`; if `t_b` absent, `maxAB = r_A`.

- [ ] **Step 1: Write failing tests**

```python
from src.topic5_contact_similarity import contact_corr, polarity_free_maxab
from scipy.stats import pearsonr


def test_raw_mode_is_plain_abs_pearson():
    rng = np.random.default_rng(0)
    rank = rng.random(8); val = 2 * rank + rng.normal(0, 0.1, 8)
    pts = rng.random((8, 2)); sup = np.ones(8)
    got = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    assert np.isclose(abs(got), abs(pearsonr(rank, val)[0]))


def test_kernel_sigma_to_zero_equals_raw():
    rng = np.random.default_rng(1)
    rank = rng.random(10); val = rng.random(10)
    pts = rng.random((10, 2)); sup = np.ones(10)
    raw = contact_corr(rank, val, mode="raw", source_pts=pts, support=sup, sigma=0.3)
    ker = contact_corr(rank, val, mode="kernel", source_pts=pts, support=sup, sigma=1e-4)
    assert np.isclose(raw, ker, atol=1e-4)


def test_maxab_takes_better_template():
    rng = np.random.default_rng(2)
    val = rng.random(12)
    rank_a = rng.random(12)              # unrelated to val
    rank_b = val + rng.normal(0, 0.01, 12)  # strongly related
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, rank_b, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert mab > 0.9   # picks template B


def test_maxab_sign_free_reverse_passes():
    """Sign-free: a perfectly reversed rank is a true positive (|corr|=1)."""
    rng = np.random.default_rng(3)
    val = rng.random(12); rank_a = -val   # reversed
    pts = rng.random((12, 2)); sup = np.ones(12)
    mab = polarity_free_maxab(rank_a, None, val, mode="raw",
                              source_pts=pts, support=sup, sigma=0.3)
    assert np.isclose(mab, 1.0, atol=1e-6)
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -k "raw_mode or sigma_to_zero or maxab" -q`
Expected: FAIL (`cannot import name 'contact_corr'`).

- [ ] **Step 3: Implement**

```python
from scipy.stats import pearsonr


def _pearson_over_contacts(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def contact_corr(rank, value, *, mode, source_pts, support, sigma, mirror=False):
    rank = np.asarray(rank, float); value = np.asarray(value, float)
    if mode == "raw":
        return _pearson_over_contacts(rank, value)
    pts = np.asarray(source_pts, float)
    f_rank = kernel_smooth_at_contacts(rank, pts, pts, support, sigma)   # identity eval
    eval_pts = pts.copy()
    if mirror:
        eval_pts[:, 1] = -eval_pts[:, 1]        # flip EVAL points y only (source unchanged)
    f_val = kernel_smooth_at_contacts(value, pts, eval_pts, support, sigma)
    return _pearson_over_contacts(f_rank, f_val)


def _abs_mirror(rank, value, *, mode, source_pts, support, sigma):
    if mode == "raw":
        c = contact_corr(rank, value, mode="raw", source_pts=source_pts,
                         support=support, sigma=sigma)
        return abs(c) if np.isfinite(c) else np.nan
    c_id = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=False)
    c_mr = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=True)
    cand = [c for c in (c_id, c_mr) if np.isfinite(c)]
    return abs(max(cand)) if cand else np.nan   # max-by-value then abs (== _abs_corr)


def polarity_free_maxab(rank_a, rank_b, value, *, mode, source_pts, support, sigma):
    r_a = _abs_mirror(rank_a, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    if rank_b is None:
        return r_a
    r_b = _abs_mirror(rank_b, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    vals = [v for v in (r_a, r_b) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topic5_contact_similarity.py tests/test_topic5_contact_similarity.py
git commit -m "feat(topic5): polarity-free maxAB contact similarity (raw+kernel, mirror-faithful)"
```

---

### Task 3: Per-seizure → median-over-seizures null fold (pluggable statistic)

**Files:**
- Modify: `src/topic5_contact_similarity.py`
- Test: `tests/test_topic5_contact_similarity.py`

**Interfaces:**
- Consumes: `within_shaft_shuffle`, `channel_shuffle`, `anchor_matched_shuffle`, `effective_shuffle_n` from `src.topic5_axis_alignment` (signatures at `src/topic5_axis_alignment.py:48,54,70,137`).
- Produces:
  - `fold_subject(per_sz_obs, per_sz_null) -> dict` with `obs_subject`, `null_q` (p5/p50/p95/p99 of `median_s` per draw), `passed` (`obs_subject > p95`). Replicates `real_med`(`run_topic5_axis_alignment.py:162`) + `_p95_med`(`:64-69`).
  - `subject_null(stat_fn, sz_value_vectors, names, *, shuffle, B, seed, anchor_by_sz=None) -> dict` — drives the per-seizure × B loop, recomputes `stat_fn` (a maxAB closure) per draw, returns `fold_subject(...)` + `effective_shuffle_n` + `INSUFFICIENT_NULL` flag.

**Folding contract (spec §5.0):** `obs_subject = median_s(stat_fn(vals_s))`; null draw `b` = `median_s(stat_fn(shuffle_b(vals_s)))`; subject null = the B medians. `stat_fn` recomputes maxAB internally (both A,B on the same shuffled vector — selection cost in every draw).

- [ ] **Step 1: Write failing tests**

```python
from src.topic5_contact_similarity import fold_subject, subject_null


def test_fold_matches_p95_med():
    # draws[sz] = [B], obs[sz]; replicate np.nanmedian(draws, axis=0) then pct95
    rng = np.random.default_rng(4)
    obs = [0.6, 0.7, 0.5]
    null = [list(rng.random(50)) for _ in range(3)]
    out = fold_subject(obs, null)
    expect_dist = np.nanmedian(np.asarray(null, float), axis=0)
    assert np.isclose(out["obs_subject"], np.median(obs))
    assert np.isclose(out["null_q"]["p95"], np.nanpercentile(expect_dist, 95))
    assert out["passed"] == bool(np.median(obs) > np.nanpercentile(expect_dist, 95))


def test_subject_null_recomputes_maxab_each_draw():
    """The null statistic must be the MAX-selected statistic, so a 2-template
    stat_fn yields a higher null upper tail than a single-template stat_fn."""
    rng = np.random.default_rng(5)
    names = [f"A{i}" for i in range(1, 7)] + [f"B{i}" for i in range(1, 7)]
    vals = {0: rng.random(12)}
    def stat_max(v):   # closure that internally takes max over 2 templates
        return max(abs(np.corrcoef(v, rng.random(12))[0, 1]),
                   abs(np.corrcoef(v, rng.random(12))[0, 1]))
    def stat_one(v):
        return abs(np.corrcoef(v, rng.random(12))[0, 1])
    from src.topic5_axis_alignment import channel_shuffle
    r_max = subject_null(stat_max, vals, names, shuffle="channel", B=200, seed=1)
    r_one = subject_null(stat_one, vals, names, shuffle="channel", B=200, seed=1)
    assert r_max["null_q"]["p95"] >= r_one["null_q"]["p95"]


def test_within_shaft_never_crosses_shaft():
    rng = np.random.default_rng(6)
    from src.topic5_axis_alignment import within_shaft_shuffle, parse_shaft
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    vals = np.arange(6.0)
    out = within_shaft_shuffle(vals, names, rng)
    # multiset within each shaft preserved
    for sh in ("A", "B"):
        idx = [i for i, n in enumerate(names) if parse_shaft(n)[0] == sh]
        assert sorted(out[idx]) == sorted(vals[idx])
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -k "fold or subject_null or never_crosses" -q`
Expected: FAIL (`cannot import name 'fold_subject'`).

- [ ] **Step 3: Implement**

```python
from src.topic5_axis_alignment import (
    within_shaft_shuffle, channel_shuffle, anchor_matched_shuffle, effective_shuffle_n,
)

_SHUFFLE = {
    "within_shaft": lambda v, names, anchor, rng: within_shaft_shuffle(v, names, rng),
    "channel": lambda v, names, anchor, rng: channel_shuffle(v, rng),
    "anchor_matched": lambda v, names, anchor, rng: anchor_matched_shuffle(v, anchor, rng),
}


def fold_subject(per_sz_obs, per_sz_null):
    obs = np.asarray(per_sz_obs, float)
    obs_subject = float(np.nanmedian(obs))
    dist = np.nanmedian(np.asarray(per_sz_null, float), axis=0)   # [B] median-over-seizures
    q = {f"p{p}": float(np.nanpercentile(dist, p)) for p in (5, 50, 95, 99)}
    return {"obs_subject": obs_subject, "null_q": q,
            "passed": bool(obs_subject > q["p95"])}


def subject_null(stat_fn, sz_value_vectors, names, *, shuffle, B, seed, anchor_by_sz=None):
    rng = np.random.default_rng(seed)
    shuf = _SHUFFLE[shuffle]
    per_sz_obs, per_sz_null = [], []
    for idx, vals in sz_value_vectors.items():
        anchor = None if anchor_by_sz is None else anchor_by_sz.get(idx)
        r = stat_fn(vals)
        if not np.isfinite(r):
            continue
        per_sz_obs.append(r)
        per_sz_null.append([stat_fn(shuf(vals, names, anchor, rng)) for _ in range(B)])
    eff = effective_shuffle_n(names, (anchor_by_sz or {}).get(next(iter(sz_value_vectors), None)),
                              "within_shaft" if shuffle == "within_shaft" else shuffle)
    if not per_sz_obs:
        return {"status": "no_resolvable_seizure"}
    out = fold_subject(per_sz_obs, per_sz_null)
    out["effective_shuffle_n"] = eff
    out["n_seizures"] = len(per_sz_obs)
    if eff is not None and eff < 4:      # MIN_EFFECTIVE_SHUFFLE_N
        out["status"] = "INSUFFICIENT_NULL"
    else:
        out["status"] = "ok"
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic5_contact_similarity.py tests/test_topic5_contact_similarity.py
git commit -m "feat(topic5): per-seizure median-fold null harness (pluggable maxAB stat)"
```

---

### Task 4: Sequence-sanity (Spearman + Kendall) track

**Files:**
- Modify: `src/topic5_contact_similarity.py`
- Test: `tests/test_topic5_contact_similarity.py`

**Interfaces:**
- Produces: `sequence_maxab(rank_a, rank_b, value, *, method) -> float` — polarity-free `max(|corr_A|,|corr_B|)`, `method in {"spearman","kendall"}`, no geometry.

- [ ] **Step 1: Write failing test**

```python
from src.topic5_contact_similarity import sequence_maxab


def test_sequence_spearman_monotone():
    val = np.array([1.0, 2, 3, 4, 5, 6])
    rank_a = np.array([6.0, 5, 4, 3, 2, 1])   # reversed monotone -> |spearman|=1
    s = sequence_maxab(rank_a, None, val, method="spearman")
    assert np.isclose(s, 1.0, atol=1e-9)


def test_sequence_kendall_runs():
    rng = np.random.default_rng(7)
    val = rng.random(10); ra = rng.random(10); rb = val.copy()
    k = sequence_maxab(ra, rb, val, method="kendall")
    assert 0.9 < k <= 1.0   # template B identical -> tau ~ 1
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -k sequence -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
from scipy.stats import spearmanr, kendalltau


def _seq_corr(rank, value, method):
    a = np.asarray(rank, float); b = np.asarray(value, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    fn = spearmanr if method == "spearman" else kendalltau
    c = fn(a[m], b[m])[0]
    return abs(float(c)) if np.isfinite(c) else np.nan


def sequence_maxab(rank_a, rank_b, value, *, method):
    r_a = _seq_corr(rank_a, value, method)
    if rank_b is None:
        return r_a
    r_b = _seq_corr(rank_b, value, method)
    vals = [v for v in (r_a, r_b) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_topic5_contact_similarity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic5_contact_similarity.py tests/test_topic5_contact_similarity.py
git commit -m "feat(topic5): sequence-sanity spearman/kendall track"
```

---

### Task 5: Runner — three-rung ladder per subject + cohort summary

**Files:**
- Create: `scripts/run_topic5_contact_similarity.py`
- Test: `tests/test_run_topic5_contact_similarity.py`

**Interfaces:**
- Consumes: all of `src.topic5_contact_similarity`; plus `make_field_record`, `matched_channels` (`src/topic5_axis_alignment.py`), `make_plane_grid`, `R_smooth_rank` (`src/propagation_contact_plane_readout.py`), and the IO pattern of `scripts/run_topic5_axis_alignment.py:_subject` (lines 72-190 — meta/cache load, eligible_idxs loop, `bact__idx` anchor).
- Produces: per-subject JSON + `cohort_summary_{activation}.{json,csv}` under `results/topic5_ictal_recruitment/contact_similarity/` (namespaced-in-one-dir; same activation-in-both-runs writes into the SAME dir, matching `axis_alignment`'s `axis_alignment_{band}_max_ab_B1000.json` convention — no per-activation subdir).

**Per-subject context (build once, reuse for all rungs):** input dirs derive from `--input-results-root` (default `results`): `CACHE_DIR=<root>/topic5_ictal_recruitment/t0_feature_cache`, `AXIS_DIR=<root>/spatial_modulation/propagation_geometry/observation_readout/real_subjects`, `MAXAB_REF=<root>/topic5_ictal_recruitment/axis_alignment` (literal relative paths from `run_topic5_axis_alignment.py:53-54`). Load `{ds_sid}_t_a.json` (+ t_b), `matched = matched_channels(...)` (require ≥6); build plane: `source_pts = np.array([[c["x_norm"], c["y_norm"]] for c in matched], float)` (shape **(n,2)** — NOT `column_stack`, which gives (2,n)), `support = np.array([c["support"] for c in matched], float)`, `sigma = R_smooth_rank(make_field_record(matched, rank_a), X, Y, None, S_THRESH)["sigma_xy"]` (frozen on t_a, reused everywhere — matches `run_topic5_axis_alignment.py:101`). Load per-seizure `bb_auc__{idx}` vectors (≥6 finite) and `bact__{idx}` anchors for `idx in meta["eligible_idxs"]`.

**Three per-seizure statistics (closures over context):**
- `R1(vals) = polarity_free_maxab(rank_a, rank_b, vals, mode="raw", source_pts, support, sigma)`
- `R2(vals) = polarity_free_maxab(..., mode="kernel", ..., sigma=sigma)`
- `R3(vals)` = field maxAB: replicate `window_maxab` exactly — `R_smooth_rank(make_field_record(matched, vals), X, Y, sigma, S_THRESH)` then `_abs_corr` vs `F_inter_a`/`F_inter_b`, max. **`_abs_corr` is replicated locally in the runner (it is script-level in `run_topic5_axis_alignment.py:59`, NOT importable); field builders reused from src. R3 must be byte-faithful to the A-line — the Task 7 cross-check enforces this (existing runner uses the same `RNG_SEED=20260614`, so R3 should reproduce the published maxAB closely).**

**σ-sweep:** R2 also at `sigma*0.5`, `sigma*2`.

- [ ] **Step 1: Write a small-fixture integration test**

```python
# tests/test_run_topic5_contact_similarity.py
import numpy as np
from scripts.run_topic5_contact_similarity import run_subject


def test_run_subject_smoke(tmp_path, monkeypatch):
    """Pick one real eligible subject (broadband). All three rungs return a
    finite obs_subject and a within-shaft null verdict; R3 recompute is finite."""
    res = run_subject("epilepsiae_958", activation="broadband", B=50, seed=20260614)
    assert res["status"] == "ok"
    for rung in ("R1", "R2", "R3"):
        assert np.isfinite(res[rung]["within_shaft"]["obs_subject"])
        assert res[rung]["within_shaft"]["status"] in ("ok", "INSUFFICIENT_NULL")
    # paired delta present
    assert "grid_delta" in res and "smooth_delta" in res


def test_negative_control_scrambled_activation_fails():
    """Bad-data regression (spec §6.2): spatially scrambled ictal activation
    must NOT pass any rung's within-shaft null. Sign-free, so a *reversed* rank
    would still pass — the failing control is spatial scramble, not reversal."""
    res = run_subject("epilepsiae_958", activation="broadband", B=200,
                      seed=20260614, negative_control=True)
    for rung in ("R1", "R2", "R3"):
        assert res[rung]["within_shaft"]["passed"] is False
```
(If `epilepsiae_958` is ineligible, substitute the first subject for which `run_subject(...)["status"]=="ok"`; discover via a 1-liner over `AXIS_DIR` `*_t_a.json`.)

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_run_topic5_contact_similarity.py -q`
Expected: FAIL (`No module named ... run_topic5_contact_similarity`).

- [ ] **Step 3: Implement runner**

Mirror `scripts/run_topic5_axis_alignment.py` for arg parsing, `AXIS_DIR`/`CACHE_DIR` globals, meta/cache loading, and the `--masked-features` path convention. The novel core (`run_subject`):

```python
# scripts/run_topic5_contact_similarity.py  (load-bearing core; IO mirrors run_topic5_axis_alignment.py)
import json, numpy as np
from pathlib import Path
from src.topic5_contact_similarity import polarity_free_maxab, sequence_maxab, subject_null
from src.topic5_axis_alignment import make_field_record, matched_channels
from src.propagation_contact_plane_readout import (
    make_plane_grid, R_smooth_rank, corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN,
)


def _abs_corr(Fi, Fj):   # local copy: run_topic5_axis_alignment._abs_corr is script-level, NOT importable
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan

SESOI = 0.05
SIGMA_SWEEP = (0.5, 1.0, 2.0)


def _ctx(ds_sid, activation):
    """Build the per-subject context: matched channels, plane pts/support, frozen
    sigma, interictal fields, and per-seizure bb_auc + bact vectors. Returns None
    if ineligible. (See run_topic5_axis_alignment.py:78-159 for the exact loaders.)"""
    # ... load t_a (+ t_b), cache npz + meta exactly as _subject does ...
    # matched, names_m, m_in_cache, rank_a (list), rank_b (list or None)
    # X, Y = make_plane_grid()
    # F_inter_a = R_smooth_rank(make_field_record(matched, rank_a), X, Y, None, S_THRESH)
    # sigma = F_inter_a["sigma_xy"]; F_inter_b = (R_smooth_rank(..., sigma, ...) if t_b else None)
    # source_pts = np.array([[c["x_norm"], c["y_norm"]] for c in matched], float)   # (n,2), NOT column_stack
    # support = np.array([c["support"] for c in matched], float)
    # sz_vals = {idx: cache[f"{activation}__{idx}"][m_in_cache] (>=6 finite)}
    # anchor = {idx: cache[f"bact__{idx}"][m_in_cache]} when present
    # return dict(...)
    ...


def _stats(ctx):
    ra, rb = ctx["rank_a"], ctx["rank_b"]
    sp, su, sg = ctx["source_pts"], ctx["support"], ctx["sigma"]
    def R1(v): return polarity_free_maxab(ra, rb, v, mode="raw", source_pts=sp, support=su, sigma=sg)
    def R2(v): return polarity_free_maxab(ra, rb, v, mode="kernel", source_pts=sp, support=su, sigma=sg)
    def R3(v):
        F = lambda vals: R_smooth_rank(make_field_record(ctx["matched"], vals), ctx["X"], ctx["Y"], sg, S_THRESH)
        r_a = _abs_corr(ctx["F_inter_a"], F(v))
        if ctx["F_inter_b"] is None:
            return r_a
        r_b = _abs_corr(ctx["F_inter_b"], F(v))
        vals = [x for x in (r_a, r_b) if np.isfinite(x)]
        return float(max(vals)) if vals else np.nan
    return {"R1": R1, "R2": R2, "R3": R3}


def run_subject(ds_sid, *, activation="broadband", B=1000, seed=20260614,
                negative_control=False):
    ctx = _ctx(ds_sid, activation)
    if ctx is None:
        return {"subject_id": ds_sid, "status": "ineligible"}
    if ctx["n_shafts"] < 2:
        return {"subject_id": ds_sid, "status": "single_shaft"}
    if negative_control:   # bad-data gate: spatially scramble each seizure's activation once
        rng = np.random.default_rng(seed)
        from src.topic5_axis_alignment import channel_shuffle
        ctx["sz_vals"] = {i: channel_shuffle(v, rng) for i, v in ctx["sz_vals"].items()}
    stats = _stats(ctx)
    out = {"subject_id": ds_sid, "dataset": ds_sid.split("_", 1)[0], "status": "ok",
           "activation": activation, "B": B, "seed": seed, "sigma_xy": ctx["sigma"],
           "n_matched_channels": len(ctx["matched"]), "n_seizures": len(ctx["sz_vals"])}
    for name, fn in stats.items():
        out[name] = {nm: subject_null(fn, ctx["sz_vals"], ctx["names_m"],
                                      shuffle=nm, B=B, seed=seed,
                                      anchor_by_sz=ctx["anchor"] if nm == "anchor_matched" else None)
                     for nm in ("within_shaft", "channel", "anchor_matched")}
    # sigma sweep on R2
    out["R2_sigma_sweep"] = {}
    for k in SIGMA_SWEEP:
        def R2k(v, kk=k): return polarity_free_maxab(ctx["rank_a"], ctx["rank_b"], v, mode="kernel",
                                                     source_pts=ctx["source_pts"], support=ctx["support"],
                                                     sigma=ctx["sigma"] * kk)
        out["R2_sigma_sweep"][f"{k}x"] = subject_null(R2k, ctx["sz_vals"], ctx["names_m"],
                                                      shuffle="within_shaft", B=B, seed=seed)
    # sequence sanity (no geometry)
    out["sequence"] = {}
    for method in ("spearman", "kendall"):
        def seq(v, mm=method): return sequence_maxab(ctx["rank_a"], ctx["rank_b"], v, method=mm)
        out["sequence"][method] = subject_null(seq, ctx["sz_vals"], ctx["names_m"],
                                               shuffle="within_shaft", B=B, seed=seed)
    # paired deltas (subject-level)
    g = lambda r: out[r]["within_shaft"]["obs_subject"]
    out["grid_delta"] = g("R3") - g("R2")     # grid contribution
    out["smooth_delta"] = g("R2") - g("R1")   # same-plane smoothing contribution
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_run_topic5_contact_similarity.py -q`
Expected: PASS (finite obs for all rungs).

- [ ] **Step 5: Implement `main()` + cohort summary + R3 cross-check**

`main()` adds args `--input-results-root` (default `results`; rebases `CACHE_DIR`/`AXIS_DIR`/`MAXAB_REF`), `--out-dir` (default `results/topic5_ictal_recruitment/contact_similarity`), `--activation`, `--B 1000`, `--seed 20260614`. It iterates the eligible cohort (same discovery as `run_topic5_axis_alignment.main`), writes `per_subject/{ds_sid}.json` and `cohort_summary_{activation}.{json,csv}` with: per-subject pass flags (R1/R2/R3 within_shaft), `grid_delta`/`smooth_delta`, cohort rows. Add the cohort equivalence read-out:

```python
deltas = np.array([s["grid_delta"] for s in subjects if s["status"] == "ok"])
ci = np.percentile([np.median(np.random.default_rng(seed + b).choice(deltas, len(deltas)))
                    for b in range(2000)], [2.5, 97.5])     # bootstrap CI of median grid_delta
summary["grid_delta_ci"] = [float(ci[0]), float(ci[1])]
summary["grid_negligible"] = bool(ci[0] > -SESOI and ci[1] < SESOI)   # equivalence (TOST-style)
```
Cross-check R3: assert per-subject `R3.within_shaft.obs_subject` ≈ existing `axis_alignment_broadband_max_ab_B1000.json` `real_median_abs_corr` within MC tolerance (`atol=0.03`); log mismatches.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_topic5_contact_similarity.py tests/test_run_topic5_contact_similarity.py
git commit -m "feat(topic5): contact-similarity ladder runner (R1/R2/R3 + sweep + sequence + cohort)"
```

---

### Task 6: Plot — three independent panels + sequence-sanity + README

**Files:**
- Create: `scripts/plot_topic5_contact_similarity.py`
- Create: `results/topic5_ictal_recruitment/contact_similarity/figures/README.md`

**Interfaces:**
- Consumes: `cohort_summary_{activation}.json` (default `<out-dir>/cohort_summary_{activation}.json`, matching the runner's naming; `--summary` overrides) + `per_subject/*.json` from Task 5.

**Panels (spec §8.3; §7 multi-panel discipline — each answers ONE question):**
- **A** per-subject grouped bars: R1/R2/R3 `obs_subject` with within-shaft `null_q.p95` as a marker/band. Q: do the three rungs agree per subject?
- **B** geometry-ladder slopegraph: one line per subject across x = {R1, R2, R3} at `obs_subject`. Q: how much does smoothing (R1→R2) and grid (R2→R3) each move it?
- **C** σ-sweep: R2 `obs_subject` vs {0.5,1,2}×σ, one line per subject. Q: bandwidth robustness?
- sequence-sanity: a small separate panel/table (Spearman & Kendall obs vs p95). Not mixed into A/B/C.

- [ ] **Step 1: Implement plot script** (matplotlib; read `docs/figure_style_guide.md` for cohort-figure conventions; paper-grade self-contained labels, no `§`/cluster-id in axes).

- [ ] **Step 2: Generate figures**

Run: `python scripts/plot_topic5_contact_similarity.py --activation broadband --out-dir results/topic5_ictal_recruitment/contact_similarity` (and again with `--activation hfa`).
Expected: PNGs written under `results/topic5_ictal_recruitment/contact_similarity/figures/`.

- [ ] **Step 3: Eyeball the figures** (render → inspect → fix; per the figure self-contained rule). Confirm panels are readable, legends shared, axes tight.

- [ ] **Step 4: Write `figures/README.md`** (Chinese, per AGENTS.md: `### filename` + 2–4 句 + `**关注点**：`, one block per figure).

- [ ] **Step 5: Commit**

```bash
git add scripts/plot_topic5_contact_similarity.py results/topic5_ictal_recruitment/contact_similarity/figures/README.md
git commit -m "feat(topic5): contact-similarity ladder figures + README"
```

---

### Task 7: Cohort run + index + verify

**Files:**
- Modify: `results/FIGURE_INDEX.md`

- [ ] **Step 1: Full cohort run**

```bash
python scripts/run_topic5_contact_similarity.py --activation broadband --B 1000
python scripts/run_topic5_contact_similarity.py --activation hfa --B 1000
```
Expected: `cohort_summary_{broadband,hfa}.{json,csv}` written into the SAME `contact_similarity/` dir (no `_hfa` subdir) + `per_subject/*.json`; log prints R3 cross-check pass count.

- [ ] **Step 2: Verify R3 cross-check + eligibility counts**

```bash
python -c "import json; s=json.load(open('results/topic5_ictal_recruitment/contact_similarity/cohort_summary_broadband.json')); print('n_ok', s['n_ok']); print('grid_ci', s['grid_delta_ci'], 'negligible', s['grid_negligible'])"
```
Expected: prints cohort n, grid-delta CI, equivalence verdict. Confirm n matches the axis_alignment maxAB cohort.

- [ ] **Step 3: Append to `results/FIGURE_INDEX.md`** the new `contact_similarity/figures/` dir (one line per figure, per the results-dir standard).

- [ ] **Step 4: Full test suite**

Run: `python -m pytest tests/test_topic5_contact_similarity.py tests/test_run_topic5_contact_similarity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add results/FIGURE_INDEX.md results/topic5_ictal_recruitment/contact_similarity/cohort_summary_broadband.json results/topic5_ictal_recruitment/contact_similarity/cohort_summary_broadband.csv results/topic5_ictal_recruitment/contact_similarity/cohort_summary_hfa.json results/topic5_ictal_recruitment/contact_similarity/cohort_summary_hfa.csv
git commit -m "feat(topic5): contact-similarity ladder cohort results + figure index"
```

---

## Self-Review

**1. Spec coverage:**
- §3.1 R1/R2/R3 → Tasks 2 (R1/R2) + 5 (R3 recompute). ✓
- §3.1 mirror/support contract → Task 2 (`_abs_mirror` mirror=eval-flip) + Task 1 (kernel ≡ smooth_field). ✓
- §3.2 R2b native-3D → **deferred (YAGNI per spec)**; not a task. Noted gap is intentional.
- §3.2 σ-sweep → Task 5 `R2_sigma_sweep`. ✓
- §3.3 sequence-sanity → Task 4 + Task 5 `sequence`. ✓
- §5.0 seizure fold → Task 3 `fold_subject`/`subject_null` (test matches `_p95_med`). ✓
- §5.2 per-draw maxAB selection → Task 3 test `recomputes_maxab_each_draw`. ✓
- §5.3 sign-free gate → Task 2 `sign_free_reverse` test; obs/p95 in Task 3. ✓
- §5.4 SESOI/equivalence → Task 5 Step 5 `grid_negligible`. ✓
- §6.2 bad-data (spatial scramble → not aligned) → Task 5 `test_negative_control_scrambled_activation_fails` + `negative_control=True` path. single-shaft → Task 5 `single_shaft`; `INSUFFICIENT_NULL` → Task 3. ✓
- §7 locked params → Global Constraints + Task 5 defaults. ✓
- §8 results dir + 3 panels + README → Tasks 5/6. ✓
- §9 conclusion language → enforced in README/summary text (Task 6/7), not code.

**2. Placeholder scan:** `_ctx` body is a structured comment pointing at exact `run_topic5_axis_alignment.py` line ranges to replicate (loaders existing, faithful-copy required) — reuse-direction, not an undefined reference; outputs fully enumerated. `S_THRESH`/`OVERLAP_MIN`/`make_field_record`/`corr_pair_mirror_invariant` imported from src; `_abs_corr` replicated locally (script-level, not importable). No "TODO/TBD".

**3. Type consistency:** `polarity_free_maxab(rank_a, rank_b, value, *, mode, source_pts, support, sigma)` used identically in Tasks 2 and 5. `subject_null(stat_fn, sz_value_vectors, names, *, shuffle, B, seed, anchor_by_sz)` consistent Tasks 3/5. `fold_subject` keys (`obs_subject`,`null_q`,`passed`) consistent. ✓

---

## Execution Handoff

Plan complete. Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute in this session with checkpoints.
