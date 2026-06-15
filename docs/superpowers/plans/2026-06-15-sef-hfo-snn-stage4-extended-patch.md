# SEF-HFO SNN Stage 4 — Extended single-patch stochastic readout (Phase 0 + Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the substrate + analysis helpers + a gated pilot for Stage 4 — one extended excitable patch spanning 4–5 virtual electrodes whose internal random nucleation produces a structured distribution of electrode-direction readouts.

**Architecture:** Reuse the now-tracked LIF E–I engine (`src/snn_engine/`) and the spontaneous runner. Phase 0 makes the L≈32 substrate practical (tail-bounded connectivity sampler + equivalence gate) and fixes the montage geometry. Phase 1 adds the `extended_patch` lesion + pure Stage-4 analysis helpers (`src/sef_hfo_stage4.py`) and runs a short pilot behind a hard-stop gate. **Phase 2/3 (ensemble + controls + analysis) are deliberately NOT in this plan — they are planned after the Phase 1 pilot gate passes**, per the spec's PILOT-FIRST hard stop.

**Tech Stack:** Python, numpy, scipy (`cKDTree`, `stats`), pytest. Engine modules imported by `sys.path.insert(0, "src/snn_engine")`. Spec: `docs/superpowers/specs/2026-06-15-sef-hfo-snn-stage4-extended-patch-stochastic-readout-design.md`.

**Scope note:** This plan stops at the Phase 1 hard-stop gate. The terminal deliverable is a go/no-go decision on the pilot + a chosen workpoint, NOT a finished Stage 4 result.

---

## File Structure

- **Modify** `src/snn_engine/connectivity_rot.py` — add opt-in `prune_radius` (tail-bounded KDTree candidate restriction) to `_sample_partners_rot` + `build_connectivity_rot`. Default `None` = current bit-identical path. (Engine edit → re-bless `engine_versions.json`.)
- **Modify** `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` — re-snapshot `connectivity_rot.py` sha256 after the edit.
- **Create** `tests/test_snn_engine_prune.py` — equivalence gate (in-degree exact, partner-distance KS, realized AR, tail-mass) for the pruned sampler.
- **Create** `src/sef_hfo_stage4.py` — pure helpers: `nucleation_centroid`, `readout_direction_distribution`, `first_contact_entropy`, `correspondence_two_stage`, `_auc`.
- **Create** `tests/test_sef_hfo_stage4.py` — TDD for every helper.
- **Modify** `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — `extended_patch` branch in `build_lesion_vth` + `--lesion extended_patch`; per-event nucleation-centroid emission into the sidecar (calls `src.sef_hfo_stage4`).
- **Create** `scripts/run_sef_hfo_snn_stage4_pilot.sh` — Phase 1 pilot driver (small grid) + gate echo.
- **Create** `docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_<date>.md` and `stage4_phase1_pilot_<date>.md` — gate records.

---

## PHASE 0 — Engine feasibility @ L≈32 + montage geometry (HARD STOP)

### Task 0.1: Tail-bounded candidate restriction in `connectivity_rot` (opt-in)

**Files:**
- Modify: `src/snn_engine/connectivity_rot.py:50-67` (`_sample_partners_rot`), `:70-117` (`build_connectivity_rot` signature + E→E call site)
- Test: `tests/test_snn_engine_prune.py`

**Contract (spec §8):** `prune_radius=None` → current behavior, BIT-IDENTICAL (same rng draws). `prune_radius=R` (mm) → restrict E candidates to those within `R` of the target via a prebuilt `cKDTree(posE)`, then run the same Efraimidis–Spirakis weighted reservoir over the local subset. This is a **tail-bounded approximation** (different rng consumption → different realization), NOT bit-identical; equivalence is distributional (Task 0.2).

- [ ] **Step 1: Write the failing test (in-degree exact + tail-mass on a small net)**

```python
# tests/test_snn_engine_prune.py
import sys, os, numpy as np
sys.path.insert(0, "src/snn_engine")
from params import Params
from connectivity import place_neurons
import connectivity_rot as cr


def _small_net(L=4.0, density=400.0, seed=1):
    p = Params(L=L, density=density, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    return p, pos, labels, NE, NI


def test_pruned_in_degree_exact_and_tailmass():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par = p.l_EE * np.sqrt(2.0); l_perp = p.l_EE / np.sqrt(2.0)
    theta = np.radians(45.0)
    rng = np.random.default_rng(0)
    # prune radius generous vs kernel scale: >= ~6*l_par contains the mass
    R = 6.0 * l_par
    # a central target
    t = int(np.argmin(np.linalg.norm(posE - posE.mean(0), axis=1)))
    cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp,
                                   theta, rng, self_local=t, prune_radius=R)
    # in-degree exact: min(C_EE, n_local_nonzero)
    assert cols.size == min(p.C_EE, (np.linalg.norm(posE - posE[t], axis=1) <= R).sum() - 1)
    # all sampled partners lie within the prune radius
    assert (np.linalg.norm(posE[cols] - posE[t], axis=1) <= R + 1e-9).all()


def test_prune_none_is_bit_identical():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par = p.l_EE * np.sqrt(2.0); l_perp = p.l_EE / np.sqrt(2.0)
    th = np.radians(45.0); t = 7
    a = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t)
    b = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t,
                                prune_radius=None)
    assert np.array_equal(np.sort(a), np.sort(b))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_snn_engine_prune.py -q`
Expected: FAIL — `_sample_partners_rot() got an unexpected keyword argument 'prune_radius'`.

- [ ] **Step 3: Implement the opt-in prune in `_sample_partners_rot`**

Add a `prune_radius=None` kwarg. When `None`, the body is unchanged (bit-identical). When set, restrict candidates to a KDTree ball BEFORE computing weights/keys:

```python
# src/snn_engine/connectivity_rot.py  (top: add)
from scipy.spatial import cKDTree

# replace _sample_partners_rot signature + body:
def _sample_partners_rot(pos_t, src_pos, C, l_par, l_perp, theta, rng,
                         self_local=None, prune_radius=None, src_tree=None):
    """Weighted reservoir sample with the ROTATED kernel. prune_radius=None ->
    full-population (bit-identical to the pre-2026-06-15 path). prune_radius=R ->
    tail-bounded: only E sources within R mm of the target are candidates
    (src_tree = prebuilt cKDTree(src_pos), built once by the caller)."""
    if prune_radius is None:
        cand = np.arange(len(src_pos))
    else:
        tree = src_tree if src_tree is not None else cKDTree(src_pos)
        cand = np.asarray(tree.query_ball_point(pos_t, prune_radius), dtype=np.int64)
        if cand.size == 0:
            return np.empty(0, dtype=np.int64)
    dz = src_pos[cand] - pos_t
    lw = _kernel_logweights_rot(dz, l_par, l_perp, theta)
    w = np.exp(lw - lw.max())
    if self_local is not None:
        w[cand == self_local] = 0.0
    nz = int(np.count_nonzero(w))
    if nz == 0:
        return np.empty(0, dtype=np.int64)
    Cc = min(C, nz)
    Ns = len(cand)
    if Cc >= Ns:
        return cand[w > 0.0]
    keys = rng.standard_exponential(Ns) / np.where(w > 0.0, w, np.inf)
    return cand[np.argpartition(keys, Cc - 1)[:Cc]]
```

Thread `prune_radius` + a prebuilt `posE` tree through `build_connectivity_rot`:

```python
# build_connectivity_rot signature: add prune_radius=None
def build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE, AR,
                           verbose=False, local_scale_EI=None,
                           w_EE_gain_core=1.0, core_mask_E=None,
                           prune_radius=None):
    ...
    posE = pos[:NE]
    _etree = cKDTree(posE) if prune_radius is not None else None
    ...
    # at the E->E call site (was line ~116):
    cols = _sample_partners_rot(pt, posE, C, l_par, l_perp, theta_EE, rng,
                                self_local=self_local, prune_radius=prune_radius,
                                src_tree=_etree)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_snn_engine_prune.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Re-bless the engine checksum (guarded edit)**

The engine file changed → its sha256 changed → `engine_versions.json` must be re-snapshot, or every guarded runner loud-fails. The change is backward-compatible (default `None` = bit-identical, proven by `test_prune_none_is_bit_identical`).

```bash
python -c "
import json, sys; sys.path.insert(0,'.')
from src.sef_hfo_snn_engine_guard import record_versions
p='results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'
rec=json.load(open(p))
rec['src/snn_engine/connectivity_rot.py']=record_versions(['src/snn_engine/connectivity_rot.py'])['src/snn_engine/connectivity_rot.py']
json.dump(rec, open(p,'w'), indent=2)
print('re-blessed connectivity_rot.py')
"
python -c "import json,sys; sys.path.insert(0,'.'); from src.sef_hfo_snn_engine_guard import assert_versions; assert_versions(json.load(open('results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'))); print('GUARD PASS')"
```

Expected: `re-blessed` then `GUARD PASS`.

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/connectivity_rot.py tests/test_snn_engine_prune.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json
git commit -m "feat(snn-engine): opt-in tail-bounded prune_radius in build_connectivity_rot (Stage 4 Phase 0)"
```

### Task 0.2: Equivalence gate driver (pruned vs naive, distributional)

**Files:**
- Test: `tests/test_snn_engine_prune.py` (add)

**Contract (spec §8 gate (1)–(5)):** on a small net, pruned vs naive must match on in-degree (exact), partner-distance distribution (KS, p>0.05 ⇒ cannot reject equality), realized E→E elongation (AR within 10%), and the analytic tail-mass beyond `R` must be < 1% of total kernel mass.

- [ ] **Step 1: Write the failing test**

```python
def test_pruned_matches_naive_distributionally():
    from scipy import stats
    p, pos, labels, NE, NI = _small_net(L=4.0, density=400.0, seed=2)
    posE = pos[:NE]
    l_par = p.l_EE*np.sqrt(2.0); l_perp = p.l_EE/np.sqrt(2.0); th=np.radians(45.0)
    R = 6.0*l_par
    from scipy.spatial import cKDTree
    tree = cKDTree(posE)
    def partner_dists(prune):
        rng = np.random.default_rng(11); ds=[]
        for t in range(0, NE, 5):           # subsample targets
            cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp,
                                           th, rng, self_local=t,
                                           prune_radius=prune, src_tree=tree)
            ds.append(np.linalg.norm(posE[cols]-posE[t], axis=1))
        return np.concatenate(ds)
    d_naive = partner_dists(None)
    d_prune = partner_dists(R)
    ks = stats.ks_2samp(d_naive, d_prune)
    assert ks.pvalue > 0.05, f"partner-distance KS rejects equality: p={ks.pvalue}"
    # analytic tail-mass beyond R for the elliptical-exponential kernel < 1%
    # (radial bound: weight <= exp(-r/l_par); mass(>R)/mass(all) <= exp(-R/l_par)*(...))
    tail = np.exp(-R / l_par)
    assert tail < 0.01, f"tail-mass bound {tail:.4f} not < 1%"
```

- [ ] **Step 2: Run to verify it fails, then passes after Task 0.1 is in**

Run: `python -m pytest tests/test_snn_engine_prune.py::test_pruned_matches_naive_distributionally -q`
Expected: PASS (Task 0.1 already implemented). If KS fails, widen `R` (e.g. `8*l_par`) and re-record the chosen R in the Phase 0 archive doc.

- [ ] **Step 3: Commit**

```bash
git add tests/test_snn_engine_prune.py
git commit -m "test(snn-engine): distributional equivalence gate for pruned sampler"
```

### Task 0.3: L≈32 build + 1 s sim feasibility timing + `oneend` dynamics smoke (gate (6))

**Files:**
- Create: `docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_2026-06-15.md` (gate record)

This task is a **measured gate, not a unit test.** No code; run + record.

- [ ] **Step 1: Time connectivity build + 1 s sim at L=32/d100 with the prune**

Run (choose `R` from Task 0.2, e.g. `8*l_par ≈ 4.3 mm`):

```bash
python -c "
import time, sys, numpy as np; sys.path.insert(0,'src/snn_engine')
from params import Params; from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
p=Params(L=32.0, density=100.0, seed=1)
rng=np.random.default_rng(1)
t0=time.time(); pos,labels,NE,NI=place_neurons(p,rng); t1=time.time()
net=build_connectivity_rot(p,pos,labels,NE,NI,rng,theta_EE=np.radians(45),AR=2.0,
                           prune_radius=8.0*p.l_EE*np.sqrt(2.0), verbose=True); t2=time.time()
print(f'N={NE+NI} place={t1-t0:.1f}s build={t2-t1:.1f}s')
import resource; print('peak RSS GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
"
```

Expected/gate: build completes in a **practical budget** (record the seconds + peak RSS; RSS must fit the 243 GB box — it will). If build is impractically slow, widen the KDTree leafsize or coarsen `R`; if it cannot be made practical, FALL BACK to L≈28 (record the decision).

- [ ] **Step 2: `oneend` dynamics smoke — pruned reproduces naive readout**

Run a short `oneend_neg` kick on a SMALL net (L=8) under naive vs pruned and confirm the recovered propagation **sign/axis agree** (the dynamics aren't broken by the tail truncation). Reuse the existing runner with `--lesion oneend_neg --L 8 --T 800` once with default (naive) and once after wiring `--prune-radius` through (Task 1.4 adds the CLI; until then, call `build_connectivity_rot` directly in a scratch script). Record: both read forward (sign=+1), `axis_err_deg` within a few degrees of each other.

- [ ] **Step 3: Write the Phase 0 gate record + commit**

Record in `stage4_phase0_feasibility_2026-06-15.md`: chosen `prune_radius` R, build/sim seconds + peak RSS at L=32, KS p-value, tail-mass bound, the oneend smoke agreement, and the **GO/NO-GO + (L, density) decision**. Commit the doc.

```bash
git add docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_2026-06-15.md
git commit -m "docs(stage4): Phase 0 engine feasibility gate record"
```

**HARD STOP:** do not start Phase 1 simulation runs until gate (1)–(6) pass and the (L, density, R) are recorded.

---

## PHASE 1 — Extended patch + analysis helpers + pilot (HARD STOP)

### Task 1.1: `nucleation_centroid` (robust ground-truth seed location)

**Files:**
- Create: `src/sef_hfo_stage4.py`
- Test: `tests/test_sef_hfo_stage4.py`

**Contract (spec §4.1):** robust centroid of the earliest-firing patch E-cells, not the single first spike. Returns `None` if too few early cells.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sef_hfo_stage4.py
import numpy as np
from src.sef_hfo_stage4 import nucleation_centroid


def test_nucleation_centroid_robust_to_one_outlier():
    # 6 patch E cells; a tight cluster at x=2 fires first, one stray at x=18 fires
    # in the same early window -> robust centroid stays near the cluster (~2), not the mean (~4.3)
    posE = np.array([[2,5],[2.1,5],[1.9,5.1],[2,4.9],[18,5],[2.05,5.0]], float)
    patch_E_idx = np.arange(6)
    nsteps = 50
    spk = np.zeros((nsteps, 6), bool)
    # active-fraction first crosses at step 10; cluster cells 0-3,5 fire 10-12, stray cell 4 fires 11
    for c in [0,1,2,3,5]: spk[10, c] = True
    spk[11, 4] = True
    axis_unit = np.array([1.0, 0.0])         # axis along x
    patch_center = np.array([2.0, 5.0])
    out = nucleation_centroid(spk, patch_E_idx, posE, t_on_idx=8,
                              tau_nuc_steps=4, axis_unit=axis_unit,
                              patch_center=patch_center, k_min=3)
    assert out is not None
    assert out["n_early_cells"] >= 5
    assert abs(out["centroid_xy"][0] - 2.0) < 0.5      # robust: stray did not drag it
    assert abs(out["s_nuc"] - 0.0) < 0.5               # near patch center along axis


def test_nucleation_centroid_too_few_returns_none():
    posE = np.array([[2,5],[2,5]], float)
    spk = np.zeros((20, 2), bool); spk[10,0] = True
    out = nucleation_centroid(spk, np.arange(2), posE, t_on_idx=8, tau_nuc_steps=4,
                              axis_unit=np.array([1.0,0.0]),
                              patch_center=np.array([2.0,5.0]), k_min=3)
    assert out is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: FAIL — `ModuleNotFoundError: src.sef_hfo_stage4`.

- [ ] **Step 3: Implement**

```python
# src/sef_hfo_stage4.py
"""Pure helpers for SEF-HFO SNN Stage 4 (extended single-patch stochastic readout).
Spec: docs/superpowers/specs/2026-06-15-sef-hfo-snn-stage4-extended-patch-stochastic-readout-design.md
"""
from __future__ import annotations
import numpy as np


def nucleation_centroid(spk, patch_E_idx, posE, t_on_idx, tau_nuc_steps,
                        axis_unit, patch_center, k_min=3):
    """Robust spatial centroid of the earliest-firing patch E-cells (spec §4.1).

    1. find the first step in [t_on_idx, ...) where any patch E-cell fires (the
       active-fraction first-crossing proxy at single-cell resolution);
    2. collect patch E-cells whose FIRST spike falls in [t_first, t_first+tau_nuc_steps);
    3. trimmed-mean (robust) centroid of their positions -> project on axis_unit
       (relative to patch_center) -> s_nuc; perpendicular component -> r_off.
    Returns None if fewer than k_min early cells (unstable centroid)."""
    sub = spk[t_on_idx:, patch_E_idx]                       # (T', npatch)
    fired = sub.any(axis=0)
    if not fired.any():
        return None
    first_step = sub.argmax(axis=0)                          # per cell, first True
    t_first = int(first_step[fired].min())
    early = fired & (first_step >= t_first) & (first_step < t_first + tau_nuc_steps)
    n_early = int(early.sum())
    if n_early < k_min:
        return None
    pts = posE[patch_E_idx][early]                           # (n_early, 2)
    # robust centroid: coordinate-wise 20% trimmed mean
    def _trim_mean(x, frac=0.2):
        x = np.sort(x); k = int(len(x) * frac)
        return x[k:len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()
    centroid = np.array([_trim_mean(pts[:, 0]), _trim_mean(pts[:, 1])])
    rel = centroid - np.asarray(patch_center, float)
    au = np.asarray(axis_unit, float); au = au / np.linalg.norm(au)
    s_nuc = float(rel @ au)
    perp = np.array([-au[1], au[0]])
    r_off = float(rel @ perp)
    return dict(centroid_xy=centroid, s_nuc=s_nuc, r_off=r_off,
                n_early_cells=n_early)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): nucleation_centroid robust seed-location helper"
```

### Task 1.2: `readout_direction_distribution` + `first_contact_entropy` (co-primary B)

**Files:**
- Modify: `src/sef_hfo_stage4.py`, `tests/test_sef_hfo_stage4.py`

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage4 import readout_direction_distribution, first_contact_entropy


def test_direction_distribution_forward_reverse_mix():
    # signs: 6 forward, 4 reverse, 2 unreadable; angles near 0 / 180
    signs = [1,1,1,1,1,1,-1,-1,-1,-1,None,None]
    angles = [2,-3,1,0,4,-1, 178,182,179,176, np.nan, np.nan]  # deg
    out = readout_direction_distribution(signs, angles, axis_angle_deg=0.0)
    assert out["n_readable"] == 10 and out["n_unreadable"] == 2
    assert abs(out["forward_frac"] - 0.6) < 1e-9
    assert out["circular_variance"] > 0.0          # dispersed (both modes present)
    assert out["near_axis_frac"] > 0.9             # all readable hug the 0/180 axis


def test_first_contact_entropy_uniform_vs_degenerate():
    # all events start at the same contact -> entropy 0; uniform over 4 -> entropy 1 (normalized)
    assert first_contact_entropy([2,2,2,2], n_contacts=4) == 0.0
    assert abs(first_contact_entropy([0,1,2,3], n_contacts=4) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: FAIL — `ImportError: cannot import name 'readout_direction_distribution'`.

- [ ] **Step 3: Implement**

```python
# append to src/sef_hfo_stage4.py
def readout_direction_distribution(signs, angles_deg, axis_angle_deg,
                                   near_axis_tol_deg=30.0):
    """Co-primary B summary of per-event readouts. signs in {+1,-1,None};
    angles_deg recovered axis angle (NaN if unreadable). Returns dispersion +
    forward/reverse balance + how tightly readable events hug the connectivity axis."""
    signs = list(signs)
    readable = [i for i, s in enumerate(signs) if s in (1, -1)]
    n_read = len(readable); n_unread = len(signs) - n_read
    fwd = sum(1 for i in readable if signs[i] == 1)
    forward_frac = (fwd / n_read) if n_read else float("nan")
    # circular variance of doubled angles (axis = mod 180): R = |mean(exp(i*2*theta))|
    th = np.radians([angles_deg[i] for i in readable]) * 2.0
    R = np.abs(np.mean(np.exp(1j * th))) if n_read else 0.0
    circ_var = float(1.0 - R)
    # near-axis fraction: angular distance to axis (mod 180) within tol
    def _axdist(a):
        d = abs((a - axis_angle_deg) % 180.0)
        return min(d, 180.0 - d)
    near = sum(1 for i in readable if _axdist(angles_deg[i]) <= near_axis_tol_deg)
    near_axis_frac = (near / n_read) if n_read else float("nan")
    return dict(n_readable=n_read, n_unreadable=n_unread,
                forward_frac=forward_frac, circular_variance=circ_var,
                near_axis_frac=near_axis_frac)


def first_contact_entropy(first_contacts, n_contacts):
    """Shannon entropy of the first-active-contact distribution, normalized to
    [0,1] by log(n_contacts). 0 = always same contact; 1 = uniform."""
    fc = np.asarray(first_contacts)
    counts = np.bincount(fc, minlength=n_contacts).astype(float)
    p = counts[counts > 0] / counts.sum()
    H = -(p * np.log(p)).sum()
    return float(H / np.log(n_contacts)) if n_contacts > 1 else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): readout direction-distribution + first-contact entropy (co-primary B)"
```

### Task 1.3: `correspondence_two_stage` + `_auc` (co-primary A, anti-blur)

**Files:**
- Modify: `src/sef_hfo_stage4.py`, `tests/test_sef_hfo_stage4.py`

**Contract (spec §4.2):** Stage 1 = does position (`s_nuc`, `r_off`) predict readability (AUC). Stage 2 = among readable, does `s_nuc` predict sign, beating a within-event shuffle null (permute the s_nuc↔sign pairing). No sklearn — AUC via the Mann–Whitney rank identity.

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage4 import correspondence_two_stage, _auc


def test_auc_separable_and_chance():
    assert _auc(np.array([0.,0,0,1,1,1]), np.array([0,0,0,1,1,1])) == 1.0
    assert abs(_auc(np.array([1.,2,3,4]), np.array([0,1,0,1])) - 0.5) < 1e-9


def test_two_stage_correspondence_real_vs_shuffle():
    rng = np.random.default_rng(0)
    n = 80
    s_nuc = rng.uniform(-1, 1, n)
    r_off = rng.uniform(-0.3, 0.3, n)
    # readable when |s_nuc| large (end-like) and |r_off| small; sign = sign(s_nuc)
    readable = (np.abs(s_nuc) > 0.3) & (np.abs(r_off) < 0.2)
    sign = np.where(s_nuc > 0, 1, -1)
    out = correspondence_two_stage(s_nuc, r_off, readable, sign,
                                   rng=np.random.default_rng(1), n_shuffle=200)
    assert out["stage1_auc_s_nuc"] > 0.7          # position predicts readability...
    assert out["stage2_auc_sign"] > 0.9           # ...and s_nuc predicts sign
    assert out["stage2_p_shuffle"] < 0.05         # beats the within-event shuffle null
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: FAIL — `cannot import name 'correspondence_two_stage'`.

- [ ] **Step 3: Implement**

```python
# append to src/sef_hfo_stage4.py
from scipy.stats import rankdata


def _auc(scores, labels):
    """ROC AUC via the Mann-Whitney U identity. labels in {0,1}."""
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(int)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def correspondence_two_stage(s_nuc, r_off, readable, sign, rng, n_shuffle=500):
    """Spec §4.2 two-stage. Stage 1: AUC of |position features| -> readable.
    Stage 2: among readable, AUC of s_nuc -> (sign==+1), with a within-event
    shuffle null (permute s_nuc within the readable set)."""
    s_nuc = np.asarray(s_nuc, float); r_off = np.asarray(r_off, float)
    readable = np.asarray(readable).astype(bool); sign = np.asarray(sign).astype(int)
    stage1_s = _auc(np.abs(s_nuc), readable)        # end-like -> readable
    stage1_r = _auc(-np.abs(r_off), readable)       # small offset -> readable
    sr = s_nuc[readable]; lab = (sign[readable] == 1).astype(int)
    obs = _auc(sr, lab)
    obs_c = max(obs, 1.0 - obs)                      # direction-agnostic strength
    null = np.empty(n_shuffle)
    for k in range(n_shuffle):
        perm = rng.permutation(lab)
        a = _auc(sr, perm); null[k] = max(a, 1.0 - a)
    p = float((null >= obs_c).mean())
    return dict(stage1_auc_s_nuc=stage1_s, stage1_auc_r_off=stage1_r,
                stage2_auc_sign=obs, stage2_p_shuffle=p, n_readable=int(readable.sum()))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS (all Stage 4 helper tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): two-stage nucleation->readout correspondence + shuffle null (co-primary A)"
```

### Task 1.4: `extended_patch` lesion + `--prune-radius` CLI in the runner

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — `build_lesion_vth` (~:108), argparse (~:195-209), the `build_connectivity_rot` call site.

**Contract (spec §2, §9):** one large low-`V_th` disk via `sample_core_field` with a large `core_r`; add to `--lesion` choices; thread `--prune-radius` into the connectivity build so L≈32 is runnable.

- [ ] **Step 1: Add the `extended_patch` branch to `build_lesion_vth`**

In `build_lesion_vth`, add (mirroring the `oneend_neg` single-`sample_core_field` pattern, but a single patch centered at the sheet center spanning `core_r`):

```python
    if lesion == "extended_patch":
        # ONE large excitable disk at the sheet center; interior low-Vth (easily
        # ignitable), exterior base (18.0). core_r set large (Phase 0 geometry: ~6-8 mm).
        vth, core_mask = sample_core_field(
            net["pos"], net["labels"] == 0, patch_center=center,
            patch_radius=core_r, rng=net["rng"],
            core_mean=core_mean, core_std=core_std, base_mean=18.0)
        return vth, core_mask
```

- [ ] **Step 2: Add `extended_patch` to the `--lesion` choices + a `--prune-radius` arg**

```python
    ap.add_argument("--lesion", choices=["oneend_neg", "oneend_pos", "twoend_deph",
                    "twoend_equal", "extended_patch", ...existing...])
    ap.add_argument("--prune-radius", type=float, default=None,
                    help="tail-bounded E->E candidate radius (mm); None=full pop (L<=24). "
                         "Set ~8*l_EE*sqrt(AR) for L>=28 (Phase 0).")
```

Thread it into the build call:

```python
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.radians(a.theta), AR=a.AR,
                                 prune_radius=a.prune_radius)
```

- [ ] **Step 3: Smoke the new lesion (small net, fast)**

Run: `python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion extended_patch --L 12 --core-r 4 --T 600 --seed 1 --out /tmp/stage4_smoke`
Expected: runs to completion; the engine guard passes; output written. (No science gate here — just that the lesion path executes.)

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
git commit -m "feat(stage4): extended_patch lesion + --prune-radius in cm spontaneous runner"
```

### Task 1.5: Emit per-event nucleation centroid into the sidecar

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — sidecar build path (calls `src.sef_hfo_stage4.nucleation_centroid`).

**Contract:** for `extended_patch` runs, every detected event record gains `nucleation_xy`, `s_nuc`, `r_off`, `n_early_cells` (or null if excluded). Reuse the existing event-windowing (`build_sidecar`'s `ev_recs`); the patch E indices come from the `core_mask` returned by `build_lesion_vth`.

- [ ] **Step 1: Import + wire**

At the runner's sidecar assembly (after events are detected and `spk`/`core_mask` are available):

```python
from src.sef_hfo_stage4 import nucleation_centroid       # noqa: E402
...
    if a.lesion == "extended_patch":
        patch_E_idx = np.where(core_mask[:NE])[0]
        axis_unit = np.array([np.cos(np.radians(a.theta)), np.sin(np.radians(a.theta))])
        posE = net["pos"][:NE]
        tau_nuc_steps = int(round(2.0 / p.dt))         # 2 ms window (pilot-tuned)
        for e in sidecar["events"]:
            nc = nucleation_centroid(spk, patch_E_idx, posE,
                                     t_on_idx=int(round(e["t_on"] / (p.dt))),
                                     tau_nuc_steps=tau_nuc_steps, axis_unit=axis_unit,
                                     patch_center=center, k_min=5)
            e["nucleation"] = None if nc is None else {
                "xy": nc["centroid_xy"].tolist(), "s_nuc": nc["s_nuc"],
                "r_off": nc["r_off"], "n_early_cells": nc["n_early_cells"]}
```

(Adapt `t_on` units to the sidecar's actual convention — confirm against `build_sidecar`.)

- [ ] **Step 2: Verify on the smoke run**

Run the Task 1.4 smoke again; open the sidecar JSON; confirm events carry a `nucleation` block with finite `s_nuc` for events that crossed, `null` for too-few-early.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
git commit -m "feat(stage4): emit per-event nucleation centroid for extended_patch runs"
```

### Task 1.6: Phase 1 pilot + hard-stop gate

**Files:**
- Create: `scripts/run_sef_hfo_snn_stage4_pilot.sh`
- Create: `docs/archive/topic4/sef_hfo/stage4_phase1_pilot_2026-06-15.md` (gate record)

**Contract (spec §8 Phase 1):** 2–3 patch sizes (`core_r`) × small (`core_mean`, `core_std`) grid × 2–3 seeds × short `T`, at the Phase-0 (L, density, R). HARD-STOPS: (a) no runaway (population rate does not lock high), (b) ≥ N events/run, (c) first-contact entropy non-degenerate (not all same contact first), (d) readable fraction not ≈0.

- [ ] **Step 1: Write the pilot driver**

```bash
# scripts/run_sef_hfo_snn_stage4_pilot.sh
set -euo pipefail
L=32; DENS=100; R=4.3          # <- from Phase 0 record
OUT=results/topic4_sef_hfo/observation_layer/stage4_pilot
for core_r in 6 8; do
  for m in 17.0 17.5; do
    for s in 1 2 3; do
      python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
        --lesion extended_patch --L $L --density $DENS --prune-radius $R \
        --core-r $core_r --core-mean $m --core-std 1.5 --nc 7 --T 4000 --seed $s \
        --out "$OUT/r${core_r}_m${m}_s${s}"
    done
  done
done
```

- [ ] **Step 2: Run the pilot**

Run: `bash scripts/run_sef_hfo_snn_stage4_pilot.sh`
Expected: 12 runs complete. (Watch the first run's printed population rate for runaway before launching all.)

- [ ] **Step 3: Evaluate the hard-stop gate**

Write a short aggregation that loads each run's sidecar + readout and computes per run: max population rate (runaway?), n_events, `first_contact_entropy`, readable fraction. Apply the gate:

```python
# inline check — gate per spec §8 Phase 1
import json, glob, numpy as np
from src.sef_hfo_stage4 import first_contact_entropy
GATE = dict(max_rate_hz=200, min_events=20, min_fc_entropy=0.3, min_readable_frac=0.1)
rows = []
for d in glob.glob("results/topic4_sef_hfo/observation_layer/stage4_pilot/*"):
    sc = json.load(open(f"{d}/sidecar.json"))     # adapt filename
    ev = sc["events"]; fcs = [e["first_contact"] for e in ev if e.get("first_contact") is not None]
    H = first_contact_entropy(fcs, n_contacts=sc["n_contacts"]) if fcs else 0.0
    readable = np.mean([1 if e.get("sign") in (1,-1) else 0 for e in ev]) if ev else 0.0
    rate_ok = sc["max_rate_hz"] < GATE["max_rate_hz"]
    rows.append((d, len(ev), H, readable, rate_ok))
    print(d, "events", len(ev), "fc_entropy %.2f"%H, "readable %.2f"%readable, "rate_ok", rate_ok)
n_pass = sum(1 for _,e,H,r,ok in rows if ok and e>=GATE["min_events"]
             and H>=GATE["min_fc_entropy"] and r>=GATE["min_readable_frac"])
print(f"GATE: {n_pass}/{len(rows)} cells pass")
```

- [ ] **Step 4: Write the gate record + decide GO/NO-GO**

Record in `stage4_phase1_pilot_2026-06-15.md`: the per-cell table, which (`core_r`, `core_mean`) workpoint(s) pass all four hard-stops, the chosen patch shape, and a plain-language §8 GO/NO-GO. **If 0 cells pass → STOP and reassess (do not proceed to Phase 2); reasons → runaway? too-local? unreadable? — each maps to a different next move.** Commit.

```bash
git add scripts/run_sef_hfo_snn_stage4_pilot.sh docs/archive/topic4/sef_hfo/stage4_phase1_pilot_2026-06-15.md
git commit -m "analysis(stage4): Phase 1 pilot + hard-stop gate record"
```

**HARD STOP — pilot gate review.** Do not write Phase 2 (ensemble + control matrix) or Phase 3 (co-primary A/B analysis) until a workpoint passes the Phase 1 gate. Those phases get their own plan, parameterized by the chosen (L, density, R, core_r, core_mean, patch shape).

---

## Phase 2 / 3 (NOT planned here — after the Phase 1 gate)

Sketch only, to be detailed once Phase 1 passes:
- **Phase 2** — at the chosen workpoint(s): long-`T` / many-seed ensemble to build the readout-direction distribution; the full control matrix (`C-point-readable`, `C-point-bottleneck`, `C-extended`, `C-homog` rate-matched via `mean_match_vth`, `C-rot` montage rotation, `C-iso` AR=1, `C-rate`); run model events through the real masked lagPat pipeline (`mask_phantom=True`) for the secondary `stable_k` check.
- **Phase 3** — co-primary A (`correspondence_two_stage`) + co-primary B (`readout_direction_distribution`) on the ensemble + controls; archive doc + paper-grade figures (one question per panel, §7 discipline); main-doc pointer only if a conclusion lands.

---

## Self-Review

- **Spec coverage:** Phase 0 (tail-bounded sampler + equivalence gate + L≈32 feasibility + montage) → Tasks 0.1–0.3. `nucleation_centroid` (§4.1) → 1.1. Co-primary B stats (§4.3) → 1.2. Co-primary A two-stage + shuffle (§4.2) → 1.3. `extended_patch` lesion (§2) → 1.4. Sidecar nucleation emission (§4.1) → 1.5. Pilot hard-stops (§8) → 1.6. Controls + co-primary analysis (§5, Phase 2/3) → deferred by design (PILOT-FIRST). Engine reuse from `src/snn_engine/` (post-refactor) reflected throughout.
- **Placeholder scan:** every code step has complete test + impl; the two non-TDD gates (0.3, 1.6) have explicit run commands + numeric thresholds. `core_r`/`prune_radius`/`tau_nuc` are pilot-tuned values carried from Phase 0, not placeholders.
- **Type consistency:** `nucleation_centroid` returns the same keys (`centroid_xy`, `s_nuc`, `r_off`, `n_early_cells`) used by Task 1.5; `correspondence_two_stage` consumes `s_nuc`/`r_off`/`readable`/`sign` produced by 1.5 + the readout; `first_contact_entropy(first_contacts, n_contacts)` signature matches its call in 1.6.
- **Risk:** `t_on`→step-index conversion in Task 1.5 must be reconciled with `build_sidecar`'s actual time convention (flagged in the step). The Phase 0 `R` must hold ≥800 E-candidates at d100 (gate (1)); if KS fails at `6*l_par`, widen to `8*l_par` and re-record.
