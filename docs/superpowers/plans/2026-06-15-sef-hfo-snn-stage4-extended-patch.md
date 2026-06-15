# SEF-HFO SNN Stage 4 — Extended single-patch stochastic readout (Phase 0 + Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the substrate + analysis helpers + a gated pilot for Stage 4 — one extended excitable patch spanning 4–5 virtual electrodes whose internal random nucleation produces a structured distribution of electrode-direction readouts.

**Architecture:** Reuse the now-tracked LIF E–I engine (`src/snn_engine/`) and the spontaneous runner. Phase 0 makes the L≈32 substrate practical (tail-bounded connectivity sampler + a full 6-check equivalence gate) and threads `--prune-radius` through the runner so the dynamics smoke runs end-to-end. Phase 1 adds the `extended_patch` lesion (single-core), **extends the universal per-event record `readout_{tag}.json::events`** (NOT the dual-core Stage 3 `build_sidecar`), adds pure Stage-4 helpers (`src/sef_hfo_stage4.py`), and runs a short pilot behind a hard-stop gate.

**Tech Stack:** Python, numpy, scipy (`cKDTree`, `stats.rankdata`, `stats.ks_2samp`), pytest. Engine modules imported by `sys.path.insert(0, "src/snn_engine")`. Spec: `docs/superpowers/specs/2026-06-15-sef-hfo-snn-stage4-extended-patch-stochastic-readout-design.md`.

**Scope note:** This plan stops at the Phase 1 hard-stop gate. Terminal deliverable = a GO/NO-GO + chosen workpoint, NOT a finished Stage 4 result. Phase 2/3 (ensemble + controls + co-primary analysis) are planned after the gate.

---

## Stage 4 event record (the data contract — read this before any task)

There is **no Stage 4 sidecar.** The runner already writes one per-run JSON, `readout_{tag}.json`, whose `events` array is the universal per-event record (`runner:289-299, 414`), built for **every** lesion. Stage 4 **extends each event dict in place**; it does NOT call the dual-core `build_sidecar` (that is gated on `len(core_masks)==2`, `runner:373`, and is irrelevant to a single patch).

Per-event fields **already present**: `t_on, t_off, event_peak_t, returned, n_part, axis_err, sign (+1/-1/None), readability, ranks{name->rank|None}`.
Per-event fields **Task 1.5 ADDS**: `first_contact` (valid-contact name with min rank), `nucleation {xy, s_nuc, r_off, n_early_cells}|null` (extended_patch only).
Run-level fields **already present** and used by the gate: `n_events`, `detector.peak` (max active fraction), `detector.true_inter_event_floor` (**p95 active fraction OUTSIDE event windows — the runaway proxy: if it does NOT stay low, the network never quiets**), `n_clean_forward/reverse`. Task 1.5 adds `config.n_valid_contacts` (`int(valid.sum())`) so first-contact entropy has its alphabet size.

---

## File Structure

- **Modify** `src/snn_engine/connectivity_rot.py` — add opt-in `prune_radius` (tail-bounded KDTree candidate restriction) to `_sample_partners_rot` + `build_connectivity_rot`. Default `None` = current bit-identical path. (Engine edit → re-bless `engine_versions.json`.)
- **Modify** `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` — re-snapshot `connectivity_rot.py` sha256 after the edit.
- **Create** `tests/test_snn_engine_prune.py` — full equivalence gate (in-degree exact, partner-distance KS, delay quantile, realized AR/covariance, tail-mass, bit-identical-None).
- **Create** `src/sef_hfo_stage4.py` — pure helpers: `nucleation_centroid`, `readout_direction_distribution`, `first_contact_entropy`, `correspondence_two_stage`, `_auc`.
- **Create** `tests/test_sef_hfo_stage4.py` — TDD for every helper.
- **Modify** `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — `extended_patch` branch in `build_lesion_vth` (returns the **4-tuple**); `--lesion extended_patch`; `--prune-radius` CLI threaded into `build_connectivity_rot`; extend the `ev_recs` loop with `first_contact` + `nucleation`; write `config.n_valid_contacts`.
- **Create** `scripts/run_sef_hfo_snn_stage4_pilot.sh` — Phase 1 pilot driver.
- **Create** `docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_2026-06-15.md`, `stage4_phase1_pilot_2026-06-15.md` — gate records.

---

## PHASE 0 — Engine feasibility @ L≈32 + montage geometry (HARD STOP)

### Task 0.1: Tail-bounded candidate restriction in `connectivity_rot` (opt-in)

**Files:**
- Modify: `src/snn_engine/connectivity_rot.py:50-67` (`_sample_partners_rot`), `:70-117` (`build_connectivity_rot`)
- Test: `tests/test_snn_engine_prune.py`

**Contract (spec §8):** `prune_radius=None` → current behavior, BIT-IDENTICAL (same rng draws). `prune_radius=R` (mm) → restrict E candidates to those within `R` via a prebuilt `cKDTree(posE)`, then the same Efraimidis–Spirakis reservoir over the local subset. Tail-bounded approximation, NOT bit-identical; equivalence is distributional (Task 0.2).

- [ ] **Step 1: Write the failing test (in-degree exact + bit-identical None)**

```python
# tests/test_snn_engine_prune.py
import sys, numpy as np
sys.path.insert(0, "src/snn_engine")
from params import Params
from connectivity import place_neurons
import connectivity_rot as cr


def _small_net(L=4.0, density=400.0, seed=1):
    p = Params(L=L, density=density, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    return p, pos, labels, NE, NI


def test_pruned_in_degree_exact_and_within_radius():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par = p.l_EE * np.sqrt(2.0); l_perp = p.l_EE / np.sqrt(2.0); th = np.radians(45.0)
    R = 8.0 * l_par                                   # see Task 0.2 tail-mass
    t = int(np.argmin(np.linalg.norm(posE - posE.mean(0), axis=1)))
    cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                   np.random.default_rng(0), self_local=t, prune_radius=R)
    n_local = int((np.linalg.norm(posE - posE[t], axis=1) <= R).sum()) - 1
    assert cols.size == min(p.C_EE, n_local)
    assert (np.linalg.norm(posE[cols] - posE[t], axis=1) <= R + 1e-9).all()


def test_prune_none_is_bit_identical():
    p, pos, labels, NE, NI = _small_net()
    posE = pos[:NE]
    l_par = p.l_EE*np.sqrt(2.0); l_perp = p.l_EE/np.sqrt(2.0); th = np.radians(45.0); t = 7
    a = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t)
    b = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                np.random.default_rng(3), self_local=t, prune_radius=None)
    assert np.array_equal(np.sort(a), np.sort(b))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_snn_engine_prune.py -q`
Expected: FAIL — `_sample_partners_rot() got an unexpected keyword argument 'prune_radius'`.

- [ ] **Step 3: Implement the opt-in prune**

```python
# src/snn_engine/connectivity_rot.py  (add import)
from scipy.spatial import cKDTree

def _sample_partners_rot(pos_t, src_pos, C, l_par, l_perp, theta, rng,
                         self_local=None, prune_radius=None, src_tree=None):
    """prune_radius=None -> full population (bit-identical pre-2026-06-15 path).
    prune_radius=R -> only E sources within R mm are candidates (src_tree =
    prebuilt cKDTree(src_pos), built once by the caller)."""
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
    Cc = min(C, nz); Ns = len(cand)
    if Cc >= Ns:
        return cand[w > 0.0]
    keys = rng.standard_exponential(Ns) / np.where(w > 0.0, w, np.inf)
    return cand[np.argpartition(keys, Cc - 1)[:Cc]]
```

Thread it through `build_connectivity_rot` (add `prune_radius=None` to the signature; build the tree once; pass it at the E→E call site `runner-internal ~line 116`):

```python
def build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE, AR,
                           verbose=False, local_scale_EI=None,
                           w_EE_gain_core=1.0, core_mask_E=None, prune_radius=None):
    ...
    posE = pos[:NE]
    _etree = cKDTree(posE) if prune_radius is not None else None
    ...
    cols = _sample_partners_rot(pt, posE, C, l_par, l_perp, theta_EE, rng,
                                self_local=self_local, prune_radius=prune_radius,
                                src_tree=_etree)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_snn_engine_prune.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Re-bless the engine checksum (guarded edit)**

```bash
python -c "
import json, sys; sys.path.insert(0,'.')
from src.sef_hfo_snn_engine_guard import record_versions
p='results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'; rec=json.load(open(p))
k='src/snn_engine/connectivity_rot.py'; rec[k]=record_versions([k])[k]
json.dump(rec, open(p,'w'), indent=2); print('re-blessed', k)
"
python -c "import json,sys; sys.path.insert(0,'.'); from src.sef_hfo_snn_engine_guard import assert_versions; assert_versions(json.load(open('results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'))); print('GUARD PASS')"
```

Expected: `re-blessed` then `GUARD PASS`. (Backward-compatible — `test_prune_none_is_bit_identical` proves default-None is unchanged.)

- [ ] **Step 6: Commit**

```bash
git add src/snn_engine/connectivity_rot.py tests/test_snn_engine_prune.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json
git commit -m "feat(snn-engine): opt-in tail-bounded prune_radius in build_connectivity_rot (Stage 4 Phase 0)"
```

### Task 0.2: Full equivalence gate — KS + delay quantile + realized AR + tail-mass

**Files:** Test: `tests/test_snn_engine_prune.py` (add)

**Contract (spec §8 gate (2)–(5)):** pruned vs naive must match distributionally on partner-distance (KS), delay quantiles (delay = `tau0 + d/v_axon`, monotone in d), and the realized E→E anisotropy (partner-offset covariance angle ≈ `theta_EE`, elongation ratio within 10%). And the analytic 2D tail-mass beyond `R` must be < 1% using the **correct** `(1+R/l_par)·exp(-R/l_par)` form (the bare `exp(-R/l_par)` underestimates it).

- [ ] **Step 1: Write the failing test**

```python
def _partner_dists(p, posE, prune, seed=11, step=5):
    from scipy.spatial import cKDTree
    l_par = p.l_EE*np.sqrt(2.0); l_perp = p.l_EE/np.sqrt(2.0); th = np.radians(45.0)
    tree = cKDTree(posE); rng = np.random.default_rng(seed); ds = []; offs = []
    for t in range(0, len(posE), step):
        cols = cr._sample_partners_rot(posE[t], posE, p.C_EE, l_par, l_perp, th,
                                       rng, self_local=t, prune_radius=prune, src_tree=tree)
        if cols.size:
            ds.append(np.linalg.norm(posE[cols]-posE[t], axis=1))
            offs.append(posE[cols]-posE[t])
    return np.concatenate(ds), np.concatenate(offs)


def _cov_axis(offs):
    C = np.cov(offs, rowvar=False); ev, evec = np.linalg.eigh(C)
    vmaj = evec[:, 1]
    ang = np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0
    return ang, float(np.sqrt(ev[1]/ev[0]))


def test_equivalence_ks_delay_ar_tailmass():
    from scipy import stats
    p, pos, labels, NE, NI = _small_net(L=4.0, density=400.0, seed=2)
    posE = pos[:NE]; l_par = p.l_EE*np.sqrt(2.0)
    R = 8.0 * l_par
    d_naive, o_naive = _partner_dists(p, posE, None)
    d_prune, o_prune = _partner_dists(p, posE, R)
    # (2) partner-distance KS cannot reject equality
    assert stats.ks_2samp(d_naive, d_prune).pvalue > 0.05
    # (3) delay quantiles match (delay = tau0 + d/v_axon -> compare d quantiles)
    qs = [0.1, 0.5, 0.9, 0.99]
    assert np.allclose(np.quantile(d_naive, qs), np.quantile(d_prune, qs), rtol=0.05)
    # (4) realized anisotropy preserved: covariance major-axis ~ theta_EE=45, ratio within 10%
    a_n, r_n = _cov_axis(o_naive); a_p, r_p = _cov_axis(o_prune)
    assert min(abs(a_p-45), 180-abs(a_p-45)) < 8.0
    assert abs(r_p - r_n) / r_n < 0.10
    # (5) 2D tail-mass bound < 1% at R=8*l_par (the WRONG bare exp(-R/l) would pass at 6*l_par too)
    tail = (1.0 + R/l_par) * np.exp(-R/l_par)
    assert tail < 0.01, f"2D tail-mass {tail:.4f} not < 1% (use R>=8*l_par)"
    # sanity: the underestimate at R=6*l_par really is > 1% (documents why we use 8)
    assert (1.0 + 6.0)*np.exp(-6.0) > 0.01
```

- [ ] **Step 2: Run to verify it passes** (Task 0.1 already implemented)

Run: `python -m pytest tests/test_snn_engine_prune.py::test_equivalence_ks_delay_ar_tailmass -q`
Expected: PASS. If KS or AR fails at `8*l_par`, widen R and **record the chosen R in the Phase 0 archive** (do NOT loosen the tolerances).

- [ ] **Step 3: Commit**

```bash
git add tests/test_snn_engine_prune.py
git commit -m "test(snn-engine): full equivalence gate (KS + delay quantile + realized AR + 2D tail-mass)"
```

### Task 0.3: `--prune-radius` CLI in the runner + L≈32 feasibility timing + `oneend` dynamics smoke (gate (1)+(6))

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (argparse + the two `build_connectivity_rot` call sites at `runner:258` and `runner:267`)
- Create: `docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_2026-06-15.md`

The `--prune-radius` CLI lives in **Phase 0** (not Phase 1) so the dynamics smoke reproduces through the runner.

- [ ] **Step 1: Add the CLI + thread it (both build sites)**

```python
    ap.add_argument("--prune-radius", type=float, default=None,
                    help="tail-bounded E->E candidate radius (mm); None=full pop (L<=24). "
                         "L>=28: set ~8*l_EE*sqrt(AR) ~= 4.3 mm (Phase 0).")
```
Add `prune_radius=a.prune_radius` to BOTH `build_connectivity_rot(...)` calls (`runner:258-260` EI-lesion branch and `runner:267` default branch).

- [ ] **Step 2: Build + 1 s sim timing at L=32/d100 (gate (1): in-degree feasibility + practical wall-time)**

```bash
python -c "
import time, sys, numpy as np; sys.path.insert(0,'src/snn_engine')
from params import Params; from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
p=Params(L=32.0, density=100.0, seed=1); rng=np.random.default_rng(1)
R=8.0*p.l_EE*np.sqrt(2.0)
t0=time.time(); pos,labels,NE,NI=place_neurons(p,rng); t1=time.time()
net=build_connectivity_rot(p,pos,labels,NE,NI,rng,theta_EE=np.radians(45),AR=2.0,
                           prune_radius=R, verbose=True); t2=time.time()
import resource
print(f'N={NE+NI} place={t1-t0:.1f}s build={t2-t1:.1f}s peakRSS_GB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} R={R:.2f}')
# gate (1): every E target must have drawn C_EE=800 (prune radius held enough candidates)
import numpy as np
ind = np.array([len(net['ampa_by_delay'])])  # presence check; see verbose synapse count
"
```
Gate: build completes in a practical budget (record seconds + peak RSS). If impractically slow → coarsen `R` or fall back to L≈28 (record the decision). Confirm the verbose AMPA synapse count ≈ `NE * C_EE` (no target starved below 800 → prune radius adequate; if starved, widen R).

- [ ] **Step 3: `oneend` dynamics smoke through the runner (gate (6))**

```bash
# naive (small L so full-pop is cheap) vs pruned at the same L -> readout sign/axis must agree
python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion oneend_neg --L 10 --T 800 --seed 1 --out /tmp/p0_naive
python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion oneend_neg --L 10 --T 800 --seed 1 --prune-radius 4.3 --out /tmp/p0_prune
python -c "
import json
a=json.load(open('/tmp/p0_naive/readout_oneend_neg_s1.json'))
b=json.load(open('/tmp/p0_prune/readout_oneend_neg_s1.json'))
fa=[e['sign'] for e in a['events'] if e['sign'] in (1,-1)]
fb=[e['sign'] for e in b['events'] if e['sign'] in (1,-1)]
print('naive fwd-frac', sum(s==1 for s in fa)/max(1,len(fa)), 'pruned', sum(s==1 for s in fb)/max(1,len(fb)))
print('naive n_events', a['n_events'], 'pruned', b['n_events'])
"
```
Gate (6): both read predominantly forward (`oneend_neg` ⇒ sign=+1), comparable event counts. (Different realizations — the prune changes the rng stream — so expect close-but-not-identical; the SIGN/axis sense must agree.)

- [ ] **Step 4: Write the Phase 0 gate record + commit**

Record in `stage4_phase0_feasibility_2026-06-15.md`: chosen `R`, build/sim seconds + peak RSS at L=32, KS p / delay-quantile / realized-AR numbers, the 2D tail-mass, the oneend smoke agreement, and the **GO/NO-GO + (L, density, R)** decision (§8 plain language).

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py docs/archive/topic4/sef_hfo/stage4_phase0_feasibility_2026-06-15.md
git commit -m "feat(stage4): --prune-radius CLI + Phase 0 engine feasibility gate record"
```

**HARD STOP:** do not start Phase 1 simulation runs until gate (1)–(6) pass and (L, density, R) are recorded.

---

## PHASE 1 — Extended patch + analysis helpers + pilot (HARD STOP)

### Task 1.1: `nucleation_centroid` (robust seed location — onset anchored on the k_min-th spike, not the first)

**Files:** Create `src/sef_hfo_stage4.py`; Test `tests/test_sef_hfo_stage4.py`

**Contract (spec §4.1):** a single early spike must NOT anchor the nucleation window. Anchor onset on the **k_min-th** earliest patch spike, take cells within ±`tau_nuc` of that, then a coordinate-wise trimmed-mean centroid.

- [ ] **Step 1: Write the failing test (temporally-isolated stray must NOT pollute)**

```python
# tests/test_sef_hfo_stage4.py
import numpy as np
from src.sef_hfo_stage4 import nucleation_centroid


def test_isolated_early_spike_does_not_pollute_centroid():
    # 6-cell cluster at x~2 fires steps 10-11; ONE stray at x=18 fires step 2 (way early)
    posE = np.array([[2,5],[2.1,5],[1.9,5.1],[2,4.9],[2.05,5.0],[1.95,5.0],[18,5]], float)
    spk = np.zeros((40, 7), bool)
    spk[10,0]=spk[10,1]=spk[11,2]=spk[10,3]=spk[11,4]=spk[10,5]=True   # cluster
    spk[2,6]=True                                                       # stray, far earlier
    out = nucleation_centroid(spk, np.arange(7), posE, t_on_idx=0, tau_nuc_steps=4,
                              axis_unit=np.array([1.0,0.0]), patch_center=np.array([2.0,5.0]),
                              k_min=5)
    assert out is not None
    assert abs(out["centroid_xy"][0] - 2.0) < 0.5      # stray (x=18) excluded by the onset window
    assert out["n_early_cells"] == 6                    # the cluster only
    assert abs(out["s_nuc"]) < 0.5


def test_too_few_early_returns_none():
    posE = np.array([[2,5],[2,5],[2,5]], float)
    spk = np.zeros((20,3), bool); spk[10,0]=spk[11,1]=True              # only 2 fire, k_min=5
    out = nucleation_centroid(spk, np.arange(3), posE, t_on_idx=0, tau_nuc_steps=4,
                              axis_unit=np.array([1.0,0.0]), patch_center=np.array([2.0,5.0]),
                              k_min=5)
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
from scipy.stats import rankdata


def _trim_mean(x, frac=0.2):
    x = np.sort(np.asarray(x, float)); k = int(len(x) * frac)
    return x[k:len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def nucleation_centroid(spk, patch_E_idx, posE, t_on_idx, tau_nuc_steps,
                        axis_unit, patch_center, k_min=5):
    """Robust ground-truth seed location (spec §4.1). The onset is anchored on the
    k_min-th earliest patch spike (NOT the single first spike), so a temporally
    isolated stray cell cannot move the window; the centroid is a coordinate-wise
    trimmed mean. Returns None if fewer than k_min patch cells fire."""
    sub = spk[t_on_idx:, patch_E_idx]                       # (T', npatch)
    fired = sub.any(axis=0)
    if int(fired.sum()) < k_min:
        return None
    first = sub.argmax(axis=0).astype(float)                # first-spike step per cell
    first[~fired] = np.inf
    onset = np.sort(first[fired])[k_min - 1]                # k_min-th earliest = robust onset
    early = fired & (first >= onset - tau_nuc_steps) & (first <= onset + tau_nuc_steps)
    n_early = int(early.sum())
    if n_early < k_min:
        return None
    pts = posE[patch_E_idx][early]
    centroid = np.array([_trim_mean(pts[:, 0]), _trim_mean(pts[:, 1])])
    rel = centroid - np.asarray(patch_center, float)
    au = np.asarray(axis_unit, float); au = au / np.linalg.norm(au)
    perp = np.array([-au[1], au[0]])
    return dict(centroid_xy=centroid, s_nuc=float(rel @ au),
                r_off=float(rel @ perp), n_early_cells=n_early)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): nucleation_centroid (k_min-anchored robust seed location)"
```

### Task 1.2: `readout_direction_distribution` (sign entropy for bidirectionality) + `first_contact_entropy`

**Files:** Modify `src/sef_hfo_stage4.py`, `tests/test_sef_hfo_stage4.py`

**Contract (spec §4.3):** the forward/reverse MIX is captured by **sign entropy** — NOT by doubled-angle circular variance, which collapses forward (0°) and reverse (180°) onto the same axis line and would read ≈0 for a 50/50 mix. The doubled-angle metric is reported separately as `axis_concentration` (how tightly the readable axes hug one axis line — high = on-axis), which is a different question.

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage4 import readout_direction_distribution, first_contact_entropy


def test_sign_entropy_captures_bidirectionality_axis_concentration_separate():
    signs = [1,1,1,1,1,1,-1,-1,-1,-1,None,None]          # 6 fwd, 4 rev, 2 unreadable
    angles = [2,-3,1,0,4,-1, 178,182,179,176, np.nan, np.nan]
    out = readout_direction_distribution(signs, angles, axis_angle_deg=0.0)
    assert out["n_readable"] == 10 and out["n_unreadable"] == 2
    assert abs(out["forward_frac"] - 0.6) < 1e-9
    assert out["sign_entropy"] > 0.9          # bidirectional mix present (H(0.6)/log2 ~ 0.971)
    assert out["axis_concentration"] > 0.9    # all readable hug the 0/180 axis line
    assert out["near_axis_frac"] > 0.9


def test_sign_entropy_zero_when_unidirectional():
    out = readout_direction_distribution([1,1,1,1], [1,2,-1,0], axis_angle_deg=0.0)
    assert out["sign_entropy"] == 0.0         # one sign -> NOT bidirectional
    assert out["axis_concentration"] > 0.9    # but still tightly on-axis


def test_first_contact_entropy_uniform_vs_degenerate():
    assert first_contact_entropy(["c2","c2","c2","c2"], n_contacts=4) == 0.0
    assert abs(first_contact_entropy(["c0","c1","c2","c3"], n_contacts=4) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: FAIL — `cannot import name 'readout_direction_distribution'`.

- [ ] **Step 3: Implement**

```python
# append to src/sef_hfo_stage4.py
def _binary_entropy_bits(pfwd):
    if pfwd in (0.0, 1.0):
        return 0.0
    return float(-(pfwd*np.log2(pfwd) + (1-pfwd)*np.log2(1-pfwd)))   # in [0,1]


def readout_direction_distribution(signs, angles_deg, axis_angle_deg,
                                   near_axis_tol_deg=30.0):
    """Co-primary B. sign_entropy = binary Shannon entropy (bits) of the
    forward/reverse split -> THE bidirectionality measure (1 = 50/50, 0 = unidirectional).
    axis_concentration = |mean exp(i*2*theta)| over readable -> how tightly the readable
    AXES hug one line (sign-folded); high = on-axis, NOT a bidirectionality claim."""
    signs = list(signs)
    readable = [i for i, s in enumerate(signs) if s in (1, -1)]
    n_read = len(readable); n_unread = len(signs) - n_read
    fwd = sum(1 for i in readable if signs[i] == 1)
    forward_frac = (fwd / n_read) if n_read else float("nan")
    sign_entropy = _binary_entropy_bits(forward_frac) if n_read else float("nan")
    th2 = np.radians([angles_deg[i] for i in readable]) * 2.0
    axis_concentration = float(np.abs(np.mean(np.exp(1j*th2)))) if n_read else float("nan")
    def _axdist(a):
        d = abs((a - axis_angle_deg) % 180.0); return min(d, 180.0 - d)
    near = sum(1 for i in readable if _axdist(angles_deg[i]) <= near_axis_tol_deg)
    near_axis_frac = (near / n_read) if n_read else float("nan")
    return dict(n_readable=n_read, n_unreadable=n_unread, forward_frac=forward_frac,
                sign_entropy=sign_entropy, axis_concentration=axis_concentration,
                near_axis_frac=near_axis_frac)


def first_contact_entropy(first_contacts, n_contacts):
    """Normalized Shannon entropy of the first-active-contact distribution (over the
    categorical contact labels). 0 = always same contact; 1 = uniform over n_contacts."""
    labels = [c for c in first_contacts if c is not None]
    if not labels or n_contacts <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    return float(H / np.log(n_contacts))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): sign-entropy direction distribution + first-contact entropy (co-primary B)"
```

### Task 1.3: `correspondence_two_stage` + `_auc` (co-primary A; None-safe, tie/one-class defined)

**Files:** Modify `src/sef_hfo_stage4.py`, `tests/test_sef_hfo_stage4.py`

**Contract (spec §4.2):** Stage 1 = position → readability AUC. Stage 2 = among readable, `s_nuc` → sign AUC, vs a within-event shuffle null. `_auc` via the Mann–Whitney rank identity with average ranks (ties handled), `nan` for one-class. `sign` may contain `None` for unreadable events — **must not crash** (only readable indices are converted).

- [ ] **Step 1: Write the failing test**

```python
from src.sef_hfo_stage4 import correspondence_two_stage, _auc


def test_auc_values_ties_and_oneclass():
    assert _auc([0,0,0,1,1,1], [0,0,0,1,1,1]) == 1.0           # perfectly separable
    assert _auc([1,2,3,4], [0,1,1,0]) == 0.5                    # genuine chance
    assert _auc([1,2,3,4], [0,1,0,1]) == 0.75                   # the corrected value
    assert _auc([5,5,5,5], [0,1,0,1]) == 0.5                    # all ties -> 0.5
    assert np.isnan(_auc([1,2,3], [1,1,1]))                     # one class -> nan


def test_two_stage_none_safe_and_beats_shuffle():
    rng = np.random.default_rng(0); n = 80
    s_nuc = rng.uniform(-1, 1, n); r_off = rng.uniform(-0.3, 0.3, n)
    readable = (np.abs(s_nuc) > 0.3) & (np.abs(r_off) < 0.2)
    sign = [(1 if s_nuc[i] > 0 else -1) if readable[i] else None for i in range(n)]   # None for unreadable
    out = correspondence_two_stage(s_nuc, r_off, readable, sign,
                                   rng=np.random.default_rng(1), n_shuffle=200)
    assert out["stage1_auc_s_nuc"] > 0.7        # end-like position predicts readability
    assert out["stage2_auc_sign"] > 0.9         # s_nuc predicts sign
    assert out["stage2_p_shuffle"] < 0.05       # beats the within-event shuffle null
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: FAIL — `cannot import name 'correspondence_two_stage'`.

- [ ] **Step 3: Implement**

```python
# append to src/sef_hfo_stage4.py
def _auc(scores, labels):
    """ROC AUC via Mann-Whitney U with average ranks (ties -> 0.5). nan if one class."""
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(int)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def correspondence_two_stage(s_nuc, r_off, readable, sign, rng, n_shuffle=500):
    """Spec §4.2 two-stage. sign may contain None (unreadable) — only readable
    indices are converted, so None never reaches astype(int)."""
    s_nuc = np.asarray(s_nuc, float); r_off = np.asarray(r_off, float)
    readable = np.asarray(readable, bool); sign = list(sign)
    stage1_s = _auc(np.abs(s_nuc), readable.astype(int))      # end-like -> readable
    stage1_r = _auc(-np.abs(r_off), readable.astype(int))     # small offset -> readable
    ridx = np.where(readable)[0]
    if ridx.size < 4:
        return dict(stage1_auc_s_nuc=stage1_s, stage1_auc_r_off=stage1_r,
                    stage2_auc_sign=float("nan"), stage2_p_shuffle=float("nan"),
                    n_readable=int(ridx.size))
    sr = s_nuc[ridx]
    lab = np.array([1 if sign[i] == 1 else 0 for i in ridx])  # readable -> sign is +1/-1, never None
    obs = _auc(sr, lab); obs_c = max(obs, 1.0 - obs)
    null = np.array([(lambda a: max(a, 1.0 - a))(_auc(sr, rng.permutation(lab)))
                     for _ in range(n_shuffle)])
    return dict(stage1_auc_s_nuc=stage1_s, stage1_auc_r_off=stage1_r,
                stage2_auc_sign=obs, stage2_p_shuffle=float((null >= obs_c).mean()),
                n_readable=int(ridx.size))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_stage4.py -q`
Expected: PASS (all Stage 4 helper tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_stage4.py tests/test_sef_hfo_stage4.py
git commit -m "feat(stage4): None-safe two-stage correspondence + shuffle null (co-primary A)"
```

### Task 1.4: `extended_patch` lesion (4-tuple return) + thread it through the runner

**Files:** Modify `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — `build_lesion_vth` (~`:108-148`) + the `--lesion` choices (~`:209`).

**Contract:** `build_lesion_vth` MUST return the 4-tuple `(vth, core_mask, foci, core_masks)` (the caller unpacks 4, `runner:268`); `sample_core_field` returns a **dict** (`cf["vth"]`, `cf["core_mask"]`). The patch is one large disk at the sheet `center`, radius `core_r`.

- [ ] **Step 1: Add the `extended_patch` branch (insert before the `twoend_deph` fallthrough)**

```python
    if lesion == "extended_patch":
        # ONE large excitable disk at the sheet centre; interior low-Vth, exterior base (18.0).
        # core_r is the patch radius (Phase 0 geometry: ~6-8 mm to span 4-5 contacts). Single
        # core -> foci/core_masks are length-1 (NO dual-core sidecar downstream).
        cf = sample_core_field(net["pos"], is_E, center, core_r, np.random.default_rng(seed + 7),
                               core_mean=core_mean, core_std=core_std, base_mean=18.0)
        return cf["vth"], cf["core_mask"], [center], [cf["core_mask"]]
```

- [ ] **Step 2: Add to the `--lesion` choices**

```python
    ap.add_argument("--lesion", choices=["oneend_neg", "oneend_pos", "twoend_deph",
                    "twoend_equal", "extended_patch", "oneend_inhib", "oneend_recur",
                    "oneend_combined"], default="twoend_equal")
```

- [ ] **Step 3: Smoke the new lesion (small net, fast)**

```bash
python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion extended_patch \
  --L 12 --core-r 4 --core-mean 17.0 --core-std 1.5 --nc 7 --T 600 --seed 1 --out /tmp/stage4_smoke
python -c "import json; d=json.load(open('/tmp/stage4_smoke/readout_extended_patch_s1.json')); print('n_events', d['n_events'], 'foci', d['config']['foci'], 'no sidecar key needed')"
```
Expected: runs to completion (guard passes, 4-tuple unpacks, single focus at centre); `readout_extended_patch_s1.json` written; **no `sidecar_*.json`** (single core).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
git commit -m "feat(stage4): extended_patch single-core lesion (4-tuple return) in cm spontaneous runner"
```

### Task 1.5: Extend `readout_{tag}.json::events` with `first_contact` + `nucleation` (NO sidecar)

**Files:** Modify `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` — the `ev_recs` loop (`:289-299`), and `config` (`:399-404`).

**Contract:** every event gains `first_contact` (the valid-contact name with the smallest rank, from `rd["ranks"]`); for `extended_patch`, every event gains `nucleation` (call `src.sef_hfo_stage4.nucleation_centroid` on `spk`). Add `config.n_valid_contacts`. This extends the **existing universal record** — no `build_sidecar`, no new file.

- [ ] **Step 1: Import + compute patch indices once (after `vth, core_mask, ...` is set, ~`:272`)**

```python
from src.sef_hfo_stage4 import nucleation_centroid       # noqa: E402 (top imports)
...
    # (after posE is defined ~line 272)
    patch_E_idx = np.where(core_mask[:NE])[0] if a.lesion == "extended_patch" else None
    tau_nuc_steps = int(round(2.0 / DT))                 # 2 ms nucleation window (pilot-tuned)
```

- [ ] **Step 2: Extend the `ev_recs` append (replace `:296-299`)**

```python
        first_contact = None
        if rd["ranks"]:
            part = {k: v for k, v in rd["ranks"].items() if v is not None}
            first_contact = min(part, key=part.get) if part else None
        nuc = None
        if patch_E_idx is not None:
            nc = nucleation_centroid(spk, patch_E_idx, posE,
                                     t_on_idx=int(round(ev["t_on"] / DT)),
                                     tau_nuc_steps=tau_nuc_steps,
                                     axis_unit=axis_unit, patch_center=center, k_min=5)
            nuc = None if nc is None else {"xy": [round(float(nc["centroid_xy"][0]), 3),
                                                  round(float(nc["centroid_xy"][1]), 3)],
                                           "s_nuc": round(nc["s_nuc"], 3),
                                           "r_off": round(nc["r_off"], 3),
                                           "n_early_cells": nc["n_early_cells"]}
        ev_recs.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                            event_peak_t=round(ep, 1), returned=bool(ev["returned"]),
                            n_part=rd["n_part"], axis_err=rd["axis_err"], sign=rd["sign"],
                            readability=rd["readability"], ranks=rd["ranks"],
                            first_contact=first_contact, nucleation=nuc))
```

- [ ] **Step 3: Add `n_valid_contacts` to `config` (`:399-404`)**

```python
                   config=dict(..., n_core=int(core_mask.sum()),
                               n_valid_contacts=int(valid.sum())),
```

- [ ] **Step 4: Verify on the smoke run**

```bash
python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion extended_patch \
  --L 12 --core-r 4 --core-mean 17.0 --core-std 1.5 --nc 7 --T 800 --seed 1 --out /tmp/stage4_smoke2
python -c "
import json; d=json.load(open('/tmp/stage4_smoke2/readout_extended_patch_s1.json'))
ev=d['events']; print('n_events', len(ev), 'n_valid_contacts', d['config']['n_valid_contacts'])
print('with nucleation:', sum(1 for e in ev if e['nucleation']), '/', len(ev))
print('first_contacts set:', sum(1 for e in ev if e['first_contact'] is not None))
print('runaway proxy (true_inter_event_floor):', d['detector']['true_inter_event_floor'])
"
```
Expected: events carry `nucleation` (finite `s_nuc` for crossing events, `null` for too-few-early) + `first_contact`; `config.n_valid_contacts` present; `detector.true_inter_event_floor` present.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py
git commit -m "feat(stage4): per-event first_contact + nucleation in readout_{tag}.json (no sidecar)"
```

### Task 1.6: Phase 1 pilot + hard-stop gate (reads the real `readout_{tag}.json`)

**Files:** Create `scripts/run_sef_hfo_snn_stage4_pilot.sh`, `docs/archive/topic4/sef_hfo/stage4_phase1_pilot_2026-06-15.md`.

**Contract (spec §8 Phase 1):** 2 patch sizes × 2 means × 3 seeds × short `T` at the Phase-0 (L, density, R). HARD-STOPS, all read from `readout_{tag}.json`: (a) **no runaway** = `detector.true_inter_event_floor` stays low (network quiets between events); (b) ≥ N events; (c) `first_contact_entropy(events[].first_contact, config.n_valid_contacts)` non-degenerate; (d) readable fraction (`sign in {±1}`) not ≈0.

- [ ] **Step 1: Write the pilot driver**

```bash
# scripts/run_sef_hfo_snn_stage4_pilot.sh
set -euo pipefail
L=32; DENS=100; R=4.3          # <- from the Phase 0 record
OUT=results/topic4_sef_hfo/observation_layer/stage4_pilot
for core_r in 6 8; do
  for m in 17.0 17.5; do
    for s in 1 2 3; do
      python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
        --lesion extended_patch --L $L --density $DENS --prune-radius $R \
        --core-r $core_r --core-mean $m --core-std 1.5 --nc 7 --T 4000 --seed $s \
        --tag "r${core_r}_m${m}_s${s}" --out "$OUT"
    done
  done
done
```

- [ ] **Step 2: Run ONE cell first (runaway watch), then the grid**

```bash
# single cell first — eyeball the printed peak / true_inter_event_floor for runaway before fanning out
python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py --lesion extended_patch --L 32 \
  --density 100 --prune-radius 4.3 --core-r 8 --core-mean 17.0 --core-std 1.5 --nc 7 --T 4000 \
  --seed 1 --tag probe --out results/topic4_sef_hfo/observation_layer/stage4_pilot
# if not runaway -> run the full grid
bash scripts/run_sef_hfo_snn_stage4_pilot.sh
```

- [ ] **Step 3: Evaluate the hard-stop gate from `readout_{tag}.json`**

```python
import json, glob, numpy as np
from src.sef_hfo_stage4 import first_contact_entropy
GATE = dict(max_inter_event_floor=0.05, min_events=20, min_fc_entropy=0.3, min_readable_frac=0.1)
rows = []
for f in glob.glob("results/topic4_sef_hfo/observation_layer/stage4_pilot/readout_*.json"):
    d = json.load(open(f)); ev = d["events"]
    fcs = [e["first_contact"] for e in ev]
    H = first_contact_entropy(fcs, d["config"]["n_valid_contacts"])
    readable = np.mean([1 if e["sign"] in (1, -1) else 0 for e in ev]) if ev else 0.0
    quiet = d["detector"]["true_inter_event_floor"]
    not_runaway = (quiet is not None) and (quiet < GATE["max_inter_event_floor"])
    ok = not_runaway and len(ev) >= GATE["min_events"] and H >= GATE["min_fc_entropy"] and readable >= GATE["min_readable_frac"]
    rows.append((d["tag"], len(ev), round(H, 2), round(readable, 2), quiet, ok))
    print(d["tag"], "ev", len(ev), "fcH %.2f" % H, "read %.2f" % readable, "quiet", quiet, "PASS" if ok else "")
print(f"GATE: {sum(r[-1] for r in rows)}/{len(rows)} cells pass")
```

- [ ] **Step 4: Write the gate record + GO/NO-GO**

Record in `stage4_phase1_pilot_2026-06-15.md`: per-cell table, which (`core_r`, `core_mean`) passes all four hard-stops, chosen patch shape, and a §8 plain-language GO/NO-GO. **If 0 pass → STOP; the failing dimension (runaway / too-local / unreadable) names the next move — do NOT proceed to Phase 2.** Commit.

```bash
git add scripts/run_sef_hfo_snn_stage4_pilot.sh docs/archive/topic4/sef_hfo/stage4_phase1_pilot_2026-06-15.md
git commit -m "analysis(stage4): Phase 1 pilot + hard-stop gate record"
```

**HARD STOP — pilot gate review.** Phase 2 (ensemble + control matrix) / Phase 3 (co-primary A/B analysis) get their own plan, parameterized by the chosen (L, density, R, core_r, core_mean, patch shape).

---

## Phase 2 / 3 (NOT planned here — after the Phase 1 gate)

- **Phase 2** — at the chosen workpoint: long-`T` / many-seed ensemble for the readout-direction distribution; the full control matrix (`C-point-readable` made readable via raised drive/evoked, `C-point-bottleneck` descriptive, `C-extended`, `C-homog` rate-matched via `mean_match_vth`, `C-rot` montage rotation, `C-iso` AR=1, `C-rate`); model events through the real masked lagPat pipeline (`mask_phantom=True`) for the secondary `stable_k` check.
- **Phase 3** — co-primary A (`correspondence_two_stage`) + co-primary B (`readout_direction_distribution` — `sign_entropy` for bidirectionality, `axis_concentration` separate) on ensemble + controls; archive doc + paper-grade figures (one question per panel, §7); main-doc pointer only if a conclusion lands.

---

## Self-Review

- **Spec coverage:** tail-bounded sampler + full 6-check equivalence (§8) → 0.1, 0.2, 0.3; L≈32 feasibility + dynamics smoke (§8) → 0.3; `nucleation_centroid` robust (§4.1) → 1.1; co-primary B sign-entropy + axis-concentration (§4.3) → 1.2; co-primary A two-stage + shuffle, None-safe (§4.2) → 1.3; `extended_patch` single-core lesion (§2) → 1.4; per-event record extension (§4.1) → 1.5; pilot hard-stops (§8) → 1.6. Controls + co-primary analysis (§5, Phase 2/3) deferred (PILOT-FIRST).
- **Data-structure correctness (review 2026-06-15):** Stage 4 uses `readout_{tag}.json::events` (universal record), NOT the dual-core `build_sidecar` (gated on `len(core_masks)==2`). `build_lesion_vth` returns the 4-tuple; `sample_core_field` returns a dict. Runaway = `detector.true_inter_event_floor`. first-contact alphabet = `config.n_valid_contacts`.
- **Placeholder scan:** every code step has complete test + impl; the two measured gates (0.3, 1.6) have run commands + numeric thresholds. `R=8*l_par`, `core_r`, `tau_nuc=2ms` are derived/pilot-tuned, not placeholders.
- **Type consistency:** `nucleation_centroid` returns `{centroid_xy, s_nuc, r_off, n_early_cells}` consumed verbatim by 1.5; `readout_direction_distribution` returns `sign_entropy`/`axis_concentration` (no misleading `circular_variance`); `correspondence_two_stage(s_nuc, r_off, readable, sign, rng, n_shuffle)` is None-safe and matches its 1.6/Phase-3 use; `first_contact_entropy(first_contacts, n_contacts)` matches the 1.6 call with `config.n_valid_contacts`.
- **Metric-validity note:** doubled-angle circular variance was REMOVED as the bidirectionality metric (it collapses forward/reverse) and replaced by `sign_entropy`; `axis_concentration` (doubled-angle) is kept only as the on-axis-tightness measure.
