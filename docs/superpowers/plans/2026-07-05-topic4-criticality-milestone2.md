# Topic 4 M3-v2.2 Approach-Criticality — Milestone 2 (T0–T5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
> **Spec of record:** `docs/superpowers/specs/2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md` (**rev2.1**). Re-read the relevant spec § before each task body (CLAUDE.md §5/§6).
> **Reference implementation (working, in-repo):** `results/topic4_criticality_m2/pilots/m2_pilots.py` (round-1: crossing localization, shape scores, nonaxis residual) + `m2_pilots_round2.py` (round-2: two-core region breakdown, nonlinear footprint). Every task productionizes a named function from these — they already run clean against the real M1 verdict points.

**Goal:** Build the **two-stage ignition/spread readout** on the M1 α₀-crossing: `linear_ignition` (where it ignites — core_localized, from the frozen-Jacobian critical mode + two-core symmetry-break confirmation) and `nonlinear_spread` (how it spreads — axial-onset→global/self-limited, from `field_rhs` footprint integration), emitted **alongside** the unchanged M1 `csd_verdict=unresolved_operating_point`.

**Architecture:** New module `src/topic4_criticality_m2.py` reuses M1 (`src/topic4_criticality.py` internals + `src/topic4_m3b_spectral_phase.py` machinery) — do NOT rebuild. The critical mode / shape scores / gain all live on the m3b rate-field `Grid(n=6)` single-core THETA_EE atlas; `subject1146` electrode geometry only feeds the M1 slow trajectory (never the eigenmode analysis). One `src/topic4_m3b_spectral_phase.py` edit: fix `field_rhs`'s missing g_K/h_G shift so its Jacobian matches `build_jacobian_dense` (JVP hard gate).

**Tech Stack:** Python, numpy, scipy (`scipy.linalg.eig`/`expm`, `scipy.stats.spearmanr`), PyYAML, pytest. SNN engine via `sys.path.insert(0, "src/snn_engine")` (inherited from M1; M2 does no SNN runs).

## Global Constraints (verbatim from spec rev2.1)

- **Two-stage, not three-way.** `csd_verdict` (M1, unchanged) + `linear_ignition` (ignition) + `nonlinear_spread` (spread) are three *different* questions. `linear_ignition=core_localized` is a NEW field — it does **NOT** replace/flip M1's `csd_verdict=unresolved_operating_point` (both coexist). The retired rev1.1 `final_verdict{axial_supported/off_axis_global_supported}` must NOT reappear.
- **Spread verdict comes from the nonlinear footprint, never from the linear mode shape.** Never write "the critical mode is axial." The linear mode answers *where it ignites* (core); axial/global is a **time-phase** in the footprint (`spread_onset` / `spread_endgame`), not a static single-mode label.
- **`nonaxis` is a sentinel / negative-control, not a spread class.** `off_axis: present` only when BOTH gates break (`off_axis_score ≥ off_axis_score_tol=0.05` AND `gain_nonaxis − max(gain_axis,gain_global) ≥ nonaxis_gain_excess_tol=0.10` AND ratio `≥ nonaxis_gain_ratio_tol=1.25`); else `absent`/`undetermined` — **never a propagation conclusion**. If `e_nonaxis gain` is reported, it MUST carry the annotation string `"nonaxis_residual = core-compactness residual in a core-localized mode, NOT sideways propagation"`.
- **Pilot results are exploratory de-risk scouts, not milestone conclusions.** The formal `core_localized` / `axial-onset` conclusion is produced by *this* pipeline (T2/T4/T5) with its registered gates — do not cite the pilot numbers as the verdict.
- **near-fold caveat:** the two-core own-crossing α₁ (~+0.19 in pilot) is a *post-fold first-positive* value, NOT a precise α₀≈0 critical shape. `near_fold_note` must say so. **symmetric-disinhibition approx:** two cores share one `q_core` scalar → two-core proves "even given an axial two-core opportunity the corridor stays dark", NOT "subject1146 dual-source slow-vars reproduced."
- **`axis_wavevector_alignment`** (=`phase_gradient_axis_score`) is **undirected** (`|F|²·cos(2(β−θ))`, 180°-symmetric) — never "signed early→late".
- **tier = `model_side_preliminary`**; single eigenvalue ≠ seizure; global runaway ≠ real seizure; never "model proves CSD exists/absent"; never "真数据" (use "actual v2.2 SIMULATION trajectory").
- **eigenmode analysis space = m3b `Grid(n=6)` / THETA_EE only** — never `subject1146` `axis_unit/axis_mask/theta_rad`.
- **results dir:** `results/topic4_criticality_m2/`. Commit **new files with explicit `git add`** (never `-am` for new files).
- **Base:** implement on **M1-merged `main`** (per user: merge PR #6 first, so this branch's science work is not tangled with the M1 merge/rebase). The plan doc lives on `topic4-criticality-m2`; the code tasks below assume M1 (`src/topic4_criticality.py`, `src/topic4_m3b_spectral_phase.py`) is present on the base.

---

## File Structure

- **Create** `config/topic4_criticality_m2.yaml` — spec §8: `basis` / `densification` / `ignition` / `two_core_confirm` / `spread` / `gain` / `perturbation`.
- **Create** `src/topic4_criticality_m2.py` — M2 module. Responsibilities: config load; basis vectors (`e_global`/`e_axis_gradient`/`e_nonaxis`); shape-score readout (`shape_scores_at`); dense α₀ localization (`localize_alpha0_crossing`); linear-ignition readout (`read_linear_ignition` + `two_core_symmetry_break`); projected gain + nonaxis sentinel (`projected_gains`, `off_axis_sentinel`); nonlinear footprint (`integrate_footprint`, `read_nonlinear_spread`); two-stage verdict (`build_ignition_spread_verdict`).
- **Modify** `src/topic4_m3b_spectral_phase.py` — `field_rhs(...)` gains g_K/h_G shift so `field_rhs(z*)≈0` at a shifted op and its finite-diff JVP matches `build_jacobian_dense` (§4.1). Blast radius: the two pilot scripts + T4 (the only `field_rhs` callers); keep the default (no shift args) byte-identical.
- **Create** `scripts/run_topic4_crit_m2.py` — CLI: `build_ignition_spread_verdict` → `results/topic4_criticality_m2/ignition_spread_verdict.json` + `STATUS.md` + figures (ignition panel / spread panel / basis sanity) + `figures/README.md`, and append `results/FIGURE_INDEX.md`. `--from-json` re-render mode (no recompute).
- **Test** `tests/test_topic4_criticality_m2.py` (unit) + `tests/test_topic4_crit_m2_integration.py` (end-to-end on M1 verdict fixture + CLI smoke).

---

## Task 0: Config + basis vectors + shape-score sanity

**Files:**
- Create: `config/topic4_criticality_m2.yaml`
- Create: `src/topic4_criticality_m2.py`
- Test: `tests/test_topic4_criticality_m2.py`

**Interfaces:**
- Consumes (from M1 / m3b): `src.topic4_criticality.load_crit_config`, `_crit_op_context(cfg)->(grid,kernels,core,b_core)`; `src.topic4_m3b_spectral_phase` (`elongation_axis_score(field,grid,theta)`, `off_axis_score(field,grid,theta)`, `phase_gradient_axis_score(field,grid,theta)`, `globality(field,grid)`, `core_overlap(field,grid,core)`, `THETA_EE`, `Grid.coords()->(X,Y)`).
- Produces: `load_m2_config(path=None)->dict`; `basis_vectors(grid, theta)->{"e_global":ndarray,"e_axis_gradient":ndarray}` (each flat, unit-norm); `nonaxis_direction(loading, grid, theta, min_norm)->(e_nonaxis|None, frac_resid, frac_global, frac_axis)`; `shape_scores_at(res, grid, kernels, core)->dict` (the 5 continuous scores + subspace metadata + `_loading`).

- [ ] **Step 1: Write `config/topic4_criticality_m2.yaml`** (verbatim spec §8 values)

```yaml
# Topic 4 M3-v2.2 criticality Milestone 2 — spec rev2.1 §8
basis:
  theta: THETA_EE            # resolved to np.pi/4 by loader
  embedding: rE_block
  nonaxis_direction_min_norm: 1.0e-3
  off_axis_score_tol: 0.05
  nonaxis_gain_excess_tol: 0.10
  nonaxis_gain_ratio_tol: 1.25
densification:
  coarse_K: 5
  crossing_width_ms_tol: 1.0
  max_bisect_hard_cap: 16
ignition:
  core_localized_overlap_thresh: 0.8
  core_localized_globality_thresh: 0.3
  overlap_sweep: [0.7, 0.8, 0.9]
  globality_sweep: [0.2, 0.3, 0.4]
  delocalized_globality_thresh: 0.5
  iso_thresh: 0.2
  corridor_lit_thresh: 0.2
two_core_confirm:
  kind: two
  radius: 0.9
  separation: 2.4
  single_core_thresh: 0.9
  corridor_dark_thresh: 0.05
spread:
  axial_onset_thresh: 0.2
  expand_active_delta: 0.1
  global_thresh: 0.5
  flood_active_thresh: 0.9
  self_limit_active_thresh: 0.1
  footprint_sample_ms: [2, 5, 10, 20, 30, 50, 75, 100, 200, 300]
  epsilon_onset_agreement: all
  epsilon_endgame_agreement: majority
gain:
  horizons_ms: [10, 25, 50, 100, 250, 500]
perturbation:
  epsilon_rel: [0.01, 0.05]
  max_time_ms: 300
  dt_ms: 0.1
  recovery_radius_rel: 0.05
  polarities: [-1, 1]
```

- [ ] **Step 2: Write the failing sanity test** (`tests/test_topic4_criticality_m2.py`) — port of pilot4; the shape scores behave on synthetic anisotropic blobs.

```python
import numpy as np
from src.topic4_criticality import load_crit_config, _crit_op_context
import src.topic4_criticality_m2 as m2

def _ctx():
    grid, kernels, core, b_core = _crit_op_context(load_crit_config())
    return grid, kernels, core, b_core

def _gauss(grid, theta, sig_par, sig_perp, ang):
    X, Y = grid.coords(); cx, cy = X.mean(), Y.mean()
    u = (X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang)
    w = -(X - cx) * np.sin(ang) + (Y - cy) * np.cos(ang)
    return np.exp(-0.5 * ((u / sig_par) ** 2 + (w / sig_perp) ** 2))

def test_shape_scores_sanity_on_synthetic_blobs():
    grid, kernels, core, _ = _ctx(); th = kernels.theta
    import src.topic4_m3b_spectral_phase as spm
    along = _gauss(grid, th, 1.6, 0.5, th)
    perp = _gauss(grid, th, 1.6, 0.5, th + np.pi / 2)
    uniform = np.ones_like(along)
    assert spm.elongation_axis_score(along, grid, th) > 0.3
    assert spm.elongation_axis_score(along, grid, th) > spm.off_axis_score(along, grid, th)
    assert spm.off_axis_score(perp, grid, th) > 0.3 and spm.elongation_axis_score(perp, grid, th) < 0
    assert spm.globality(uniform, grid) > 0.9 and abs(spm.elongation_axis_score(uniform, grid, th)) < 0.05

def test_basis_vectors_unit_norm_and_orthogonal():
    grid, kernels, _, _ = _ctx()
    b = m2.basis_vectors(grid, kernels.theta)
    assert abs(np.linalg.norm(b["e_global"]) - 1.0) < 1e-9
    assert abs(np.linalg.norm(b["e_axis_gradient"]) - 1.0) < 1e-9
    assert abs(float(b["e_global"] @ b["e_axis_gradient"])) < 1e-9   # axis grad is zero-mean

def test_load_m2_config_resolves_theta():
    cfg = m2.load_m2_config()
    assert cfg["basis"]["off_axis_score_tol"] == 0.05
    assert isinstance(cfg["basis"]["theta"], float)      # "THETA_EE" -> np.pi/4
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_topic4_criticality_m2.py -q`
Expected: FAIL (`No module named src.topic4_criticality_m2`).

- [ ] **Step 4: Write minimal `src/topic4_criticality_m2.py`** (config loader + basis + `nonaxis_direction` + `shape_scores_at`, ported from `m2_pilots.py:_shape_scores`/`pilot3`)

```python
"""Topic 4 M3-v2.2 criticality Milestone 2 — two-stage ignition/spread readout.

Productionizes the M2 de-risk pilots (results/topic4_criticality_m2/pilots/*.py).
Spec: docs/superpowers/specs/2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md (rev2.1).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
import src.topic4_m3b_spectral_phase as spm

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _REPO / "config/topic4_criticality_m2.yaml"

def load_m2_config(path=None) -> dict:
    cfg = yaml.safe_load(Path(path or _DEFAULT_CFG).read_text())
    if cfg["basis"].get("theta") == "THETA_EE":
        cfg["basis"]["theta"] = float(spm.THETA_EE)
    return cfg

def basis_vectors(grid, theta) -> dict:
    X, Y = grid.coords()
    e_global = np.ones(X.size); e_global /= np.linalg.norm(e_global)
    s = (X * np.cos(theta) + Y * np.sin(theta)).ravel(); s = s - s.mean()
    e_axis = s - (s @ e_global) * e_global
    e_axis /= (np.linalg.norm(e_axis) + 1e-300)
    return {"e_global": e_global, "e_axis_gradient": e_axis}

def nonaxis_direction(loading, grid, theta, min_norm):
    v = np.asarray(loading, float).ravel(); nv = float(np.linalg.norm(v))
    b = basis_vectors(grid, theta)
    proj_g = (v @ b["e_global"]) * b["e_global"]
    proj_a = (v @ b["e_axis_gradient"]) * b["e_axis_gradient"]
    residual = v - proj_g - proj_a
    rn = float(np.linalg.norm(residual))
    frac = lambda x: float(np.linalg.norm(x) / (nv + 1e-300))
    e_nonaxis = residual / rn if rn >= min_norm else None
    return e_nonaxis, frac(residual), frac(proj_g), frac(proj_a)

def shape_scores_at(res, grid, kernels, core) -> dict:
    idxs = spm.leading_subspace_indices(res.eigenvalues, min_sep=1e-3, imag_tol=1e-3)
    loading = spm.pair_loading(res.right, idxs, grid)
    th = kernels.theta; lead = res.eigenvalues[0]
    return {
        "axis_elongation": float(spm.elongation_axis_score(loading, grid, th)),
        "axis_wavevector_alignment": float(spm.phase_gradient_axis_score(loading, grid, th)),
        "off_axis": float(spm.off_axis_score(loading, grid, th)),
        "globality": float(spm.globality(loading, grid)),
        "core_overlap": float(spm.core_overlap(loading, grid, core)),
        "leading_subspace_dim": int(len(idxs)),
        "leading_is_complex_pair": bool(len(idxs) == 2 and abs(lead.imag) > 1e-3),
        "leading_eigenvalue_real": float(lead.real),
        "leading_eigenvalue_imag": float(lead.imag),
        "_loading": loading,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_topic4_criticality_m2.py -q`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add config/topic4_criticality_m2.yaml src/topic4_criticality_m2.py tests/test_topic4_criticality_m2.py
git commit -m "feat(topic4-crit m2): T0 config + basis vectors + shape-score sanity"
```

---

## Task 1: Dense α₀ localization

**Files:**
- Modify: `src/topic4_criticality_m2.py`
- Test: `tests/test_topic4_criticality_m2.py`

**Interfaces:**
- Consumes: M1 internals `src.topic4_criticality._fields_from_slow`, `_low_branch_at`; the M1 verdict JSON `results/topic4_criticality/trajectory_verdict.json` (`points[]` each with `qualified`, `branch_id`, `saturated`, `time_ms`, `alpha1`, `slow_inputs`). Reference: `m2_pilots.py:pilot2` (bracket → coarse scan → bisect).
- Produces: `interp_slow(a,b,frac)->dict`; `low_solve_fast(grid,kernels,core,slow,cfg_crit,prev)->(op,sat)`; `alpha1_and_eig(grid,kernels,op)->(alpha1,res,J)`; `localize_alpha0_crossing(points, grid, kernels, core, cfg_crit, m2cfg)->dict` with keys `alpha0_crossing_time_ms`, `alpha0_crossing_slow_state`, `crossing_frac`, `crossing_width_ms`, `alpha_left`, `alpha_right`, `crossing_status` (`single|multiple_alpha0_crossings|none`), `op_solve_quality_left/right`, `_crossing_op`, `_crossing_res`.

- [ ] **Step 1: Write the failing test** — localization on the real M1 verdict points returns a crossing with α straddling 0 and a sub-ms/qualified width.

```python
import json
from src.topic4_criticality import load_crit_config, _crit_op_context

def _points():
    p = _REPO / "results/topic4_criticality/trajectory_verdict.json"   # M1 deliverable
    return json.loads(p.read_text())["points"]

def test_localize_alpha0_crossing_brackets_zero():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert out["crossing_status"] in ("single", "multiple_alpha0_crossings")
    assert out["alpha_left"] < 0.0                       # last neg before crossing
    assert out["crossing_frac"] is not None
    assert 470.0 < out["alpha0_crossing_time_ms"] < 520.0  # M1 idx14->idx15 window
```
(`_REPO` import: add `from src.topic4_criticality_m2 import _REPO` or recompute in the test.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic4_criticality_m2.py::test_localize_alpha0_crossing_brackets_zero -q`
Expected: FAIL (`localize_alpha0_crossing` not defined).

- [ ] **Step 3: Implement `localize_alpha0_crossing`** (productionize `m2_pilots.py:pilot2` (a)+(b): bracket, coarse scan `coarse_K` with multi-crossing count, recursive bisection to `crossing_width_ms_tol` or `max_bisect_hard_cap`, quality two layers). Port `interp_slow`/`low_solve_fast`/`alpha1_and_eig` from `m2_pilots.py` (`_interp_slow`/`_low_solve_fast`/`_alpha1_and_eig`) verbatim into this module. Add the two-layer quality:

```python
def _op_solve_quality(op, res):
    return bool(op is not None and not op.saturated and res is not None
                and res.status == "resolved" and res.eigenvalues.size > 0)

def localize_alpha0_crossing(points, grid, kernels, core, cfg_crit, m2cfg):
    dcfg = m2cfg["densification"]
    q = [p for p in points if p.get("qualified") and p.get("branch_id") == "low_branch"]
    last_q = q[-1]
    trans = next(p for p in points if p.get("saturated") and p["time_ms"] > last_q["time_ms"])
    a, b = last_q["slow_inputs"], trans["slow_inputs"]
    # coarse scan
    fracs = list(np.linspace(0.0, 1.0, dcfg["coarse_K"]))
    prev, scan = None, []
    for fr in fracs:
        op, sat = low_solve_fast(grid, kernels, core, interp_slow(a, b, fr), cfg_crit, prev)
        if not sat and op is not None: prev = op
        a1 = (float("nan") if sat else alpha1_and_eig(grid, kernels, op)[0])
        scan.append({"frac": float(fr), "alpha1": (None if (sat or not np.isfinite(a1)) else float(a1)), "sat": sat})
    defined = [(s["frac"], s["alpha1"]) for s in scan if s["alpha1"] is not None]
    sign_changes = sum(1 for (_, x), (_, y) in zip(defined, defined[1:]) if (x < 0) != (y < 0))
    status = "multiple_alpha0_crossings" if sign_changes > 1 else ("single" if defined else "none")
    # first neg -> (>=0 or gone) sub-bracket, then bisect (see pilot2 (b)); returns best dict
    ...  # bisection loop identical to m2_pilots.pilot2 lines 201-233, using max_bisect_hard_cap + crossing_width_ms_tol
    # width in ms over the idx14->idx15 time bracket:
    span_ms = trans["time_ms"] - last_q["time_ms"]
    crossing_width_ms = (hi_fr - lo_fr) * span_ms
    return {"alpha0_crossing_time_ms": t_cross, "alpha0_crossing_slow_state": best_slow,
            "crossing_frac": best_frac, "crossing_width_ms": crossing_width_ms,
            "alpha_left": alpha_lo, "alpha_right": alpha_hi, "crossing_status": status,
            "op_solve_quality_left": ql, "op_solve_quality_right": qr,
            "_crossing_op": best_op, "_crossing_res": best_res}
```
(Fill the `...` with the exact bisection loop from `m2_pilots.py:pilot2` (b); replace the hard-coded `range(8)`/`2e-3` with `max_bisect_hard_cap` and `crossing_width_ms_tol/span_ms`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic4_criticality_m2.py::test_localize_alpha0_crossing_brackets_zero -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_criticality_m2.py tests/test_topic4_criticality_m2.py
git commit -m "feat(topic4-crit m2): T1 dense alpha0 localization (bracket+coarse+bisect+quality)"
```

---

## Task 2: linear_ignition readout + two-core symmetry-break confirmation

**Files:**
- Modify: `src/topic4_criticality_m2.py`
- Test: `tests/test_topic4_criticality_m2.py`

**Interfaces:**
- Consumes: `localize_alpha0_crossing` (T1) `_crossing_op`/`_crossing_res`; `shape_scores_at` (T0); `spm.make_core_mask(grid, kind="two", radius, separation)`. Reference: `m2_pilots_round2.py:pilotA` (`_region_masks`, `_region_breakdown`, two-core own-crossing).
- Produces: `read_linear_ignition(crossing, grid, kernels, core, cfg_crit, m2cfg, points)->dict` with keys per spec §1: `class` (`core_localized|delocalized|ambiguous`), `delocalized_subtype` (`null|corridor_lit|global_like|multi_core`), `core_overlap`, `globality`, `two_core_symmetry_break` (bool), `corridor_power`, `shape_descriptors{axis_elongation,off_axis,axis_wavevector_alignment}`, `near_fold_note`, `ignition_sensitivity`.

- [ ] **Step 1: Write the failing tests** — ignition class from the real crossing is `core_localized`; two-core confirm sets `two_core_symmetry_break=True` with dark corridor; a synthetic global loading classifies `delocalized/global_like`.

```python
def test_linear_ignition_core_localized_on_real_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config(); pts = _points()
    crossing = m2.localize_alpha0_crossing(pts, grid, kernels, core, cfg, m2cfg)
    ig = m2.read_linear_ignition(crossing, grid, kernels, core, cfg, m2cfg, pts)
    assert ig["class"] == "core_localized"
    assert ig["core_overlap"] >= m2cfg["ignition"]["core_localized_overlap_thresh"]
    assert ig["globality"] <= m2cfg["ignition"]["core_localized_globality_thresh"]
    assert ig["two_core_symmetry_break"] is True
    assert ig["corridor_power"] <= m2cfg["two_core_confirm"]["corridor_dark_thresh"]
    assert "post-fold" in ig["near_fold_note"] and "symmetric" in ig["near_fold_note"]

def test_ignition_class_delocalized_on_global_loading():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    cls, sub = m2._classify_ignition(core_overlap=0.2, globality=0.8,
                                     axis_elongation=0.0, off_axis=0.0,
                                     corridor_power=0.0, n_core_peaks=1, m2cfg=m2cfg)
    assert cls == "delocalized" and sub == "global_like"
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_topic4_criticality_m2.py -k linear_ignition -q` → FAIL.

- [ ] **Step 3: Implement `_classify_ignition` + `two_core_symmetry_break` + `read_linear_ignition`** (port region breakdown + two-core own-crossing from `m2_pilots_round2.py:pilotA`; the classifier below).

```python
def _classify_ignition(*, core_overlap, globality, axis_elongation, off_axis,
                       corridor_power, n_core_peaks, m2cfg):
    ig = m2cfg["ignition"]
    if core_overlap >= ig["core_localized_overlap_thresh"] and globality <= ig["core_localized_globality_thresh"]:
        return "core_localized", None
    if globality >= ig["delocalized_globality_thresh"]:
        if corridor_power >= ig["corridor_lit_thresh"]:
            return "delocalized", "corridor_lit"
        if abs(axis_elongation) < ig["iso_thresh"] and off_axis < ig["iso_thresh"]:
            return "delocalized", "global_like"
        if n_core_peaks >= 2:
            return "delocalized", "multi_core"
        return "delocalized", "global_like"
    return "ambiguous", None

def _ignition_sensitivity(core_overlap, globality, m2cfg):
    ig = m2cfg["ignition"]; flips = []
    for ot in ig["overlap_sweep"]:
        for gt in ig["globality_sweep"]:
            flips.append(core_overlap >= ot and globality <= gt)
    return "core_localized but threshold-sensitive" if not all(flips) and any(flips) else "stable"
```
`read_linear_ignition`: score the crossing mode (`shape_scores_at(crossing["_crossing_res"], ...)`), then run the two-core confirmation — `make_core_mask(kind="two", radius, separation)`, localize the two-core OWN crossing along the same bracket (reuse T1 machinery with `core2`), region-breakdown its loading (`coreA/coreB/corridor_axial`), set `two_core_symmetry_break = max(coreA,coreB) >= single_core_thresh and corridor_axial <= corridor_dark_thresh`, `corridor_power = corridor_axial`. `near_fold_note = "two-core crossing alpha1 is a post-fold first-positive value, not a precise alpha0~0 critical shape; two cores share one q_core (symmetric-disinhibition approximation) → proves corridor stays dark given an axial two-core opportunity, not subject1146 dual-source reproduction."`

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_topic4_criticality_m2.py -k "linear_ignition or ignition_class" -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_criticality_m2.py tests/test_topic4_criticality_m2.py
git commit -m "feat(topic4-crit m2): T2 linear_ignition + two-core symmetry-break confirm"
```

---

## Task 3: Projected gain/leak + nonaxis off_axis sentinel

**Files:**
- Modify: `src/topic4_criticality_m2.py`
- Test: `tests/test_topic4_criticality_m2.py`

**Interfaces:**
- Consumes: `basis_vectors`, `nonaxis_direction` (T0); `spm.transient_gain(matrix, b, T)`, `spm.build_jacobian_dense`. Directions embed into the full state via the rE block (`embed_rE`); readout projects the rE block back.
- Produces: `embed_rE(e, grid, kernels)->ndarray`; `projected_gains(J, grid, kernels, dirs, horizons)->dict[str,dict[T,gain]]`; `off_axis_sentinel(crossing, grid, kernels, core, m2cfg)->dict` with `off_axis` (`absent|present|undetermined`), `nonaxis_gain`, `axis_gain`, `global_gain`, `annotation`, `nonaxis_source_policy`.

- [ ] **Step 1: Write the failing test** — on the real core-localized crossing, the off_axis sentinel is `absent` and carries the mandated annotation; `present` requires BOTH gates.

```python
def test_off_axis_sentinel_absent_on_core_localized_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    crossing = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    s = m2.off_axis_sentinel(crossing, grid, kernels, core, m2cfg)
    assert s["off_axis"] == "absent"
    assert "core-compactness residual" in s["annotation"]

def test_off_axis_present_requires_both_gates():
    m2cfg = m2.load_m2_config()
    # score gate open but gain gate closed -> NOT present
    v = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.10,
                              gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v == "undetermined"
    v2 = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.40,
                               gain_axis=0.10, gain_global=0.05, m2cfg=m2cfg)
    assert v2 == "present"
    v3 = m2._off_axis_decision(off_axis_score=0.01, gain_nonaxis=0.01,
                               gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v3 == "absent"
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Implement** the three-state gate (spec §2.3, verbatim thresholds) + gains:

```python
_NONAXIS_ANNOTATION = ("nonaxis_residual = core-compactness residual in a core-localized mode, "
                       "NOT sideways propagation")

def embed_rE(e, grid, kernels):
    z = np.zeros(6 * grid.n * grid.n)     # 6 fields rE,rI,sEE,sEI,sIE,sII
    z[: grid.n * grid.n] = e
    return z

def projected_gains(J, grid, kernels, dirs, horizons):
    from scipy.linalg import expm
    out = {}
    for name, e in dirs.items():
        b = embed_rE(e, grid, kernels)
        out[name] = {int(T): float(spm.transient_gain(J, b, float(T))) for T in horizons}
    return out

def _off_axis_decision(*, off_axis_score, gain_nonaxis, gain_axis, gain_global, m2cfg):
    bcfg = m2cfg["basis"]
    denom = max(gain_axis, gain_global, 1e-300)
    score_gate = off_axis_score >= bcfg["off_axis_score_tol"]
    gain_gate = ((gain_nonaxis - max(gain_axis, gain_global)) >= bcfg["nonaxis_gain_excess_tol"]
                 and (gain_nonaxis / denom) >= bcfg["nonaxis_gain_ratio_tol"])
    if score_gate and gain_gate:   return "present"
    if not score_gate and not gain_gate: return "absent"
    return "undetermined"
```
`off_axis_sentinel`: build `e_nonaxis` from the crossing loading (`nonaxis_direction`, low-norm→`nonaxis_source_policy="unavailable_low_residual_energy"`, gain=NaN, `off_axis="absent"`); compute axis/global/nonaxis gains at the near-recovery horizon; call `_off_axis_decision`; attach `_NONAXIS_ANNOTATION`.

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_criticality_m2.py tests/test_topic4_criticality_m2.py
git commit -m "feat(topic4-crit m2): T3 projected gains + nonaxis off_axis sentinel (both-gate)"
```

---

## Task 4: field_rhs shift fix + JVP hard gate + nonlinear footprint spread

**Files:**
- Modify: `src/topic4_m3b_spectral_phase.py` (`field_rhs`)
- Modify: `src/topic4_criticality_m2.py`
- Test: `tests/test_topic4_criticality_m2.py`, `tests/test_topic4_m3b_spectral_phase.py` (JVP gate)

**Interfaces:**
- Consumes: `spm.field_rhs` (fixed), `spm.op_state_vector`, `spm.unpack_state`, `spm._SAT_RATE_KHZ`, `spm.core_perturbation_vector`. Reference: `m2_pilots_round2.py:pilotB` (`_footprint_metrics`, `_integrate_footprint`).
- Produces: `integrate_footprint(grid,kernels,op,core,theta,v,*,eps,dt,t_max,sample_ms)->dict` (`fixedpoint_residual`,`escaped_at_ms`,`trajectory[]`); `read_nonlinear_spread(crossing, points, grid, kernels, core, b_core, cfg_crit, m2cfg)->dict` per spec §1 (`onset`,`endgame`,`off_axis`,`depth_dependent`,`footprint_trajectory`,`control_minus_kick`,`epsilon_sensitivity`).

- [ ] **Step 1: Write the JVP hard-gate test FIRST** (spec §4.1) — on a SHIFTED op, finite-diff JVP of the fixed `field_rhs` matches `build_jacobian_dense`.

```python
import numpy as np, src.topic4_m3b_spectral_phase as spm
def test_field_rhs_jvp_matches_jacobian_on_shifted_op():
    grid = spm.Grid(n=6); kernels = spm.make_kernels(grid)
    core = spm.make_core_mask(grid, kind="single", radius=0.9)
    exc = spm.build_excitability_field(grid, core, mu_core=0.0)
    inh = spm.build_inhibition_field(grid, core, q_global=1.0, q_core=1.0)
    op = spm.solve_operating_point(grid, kernels, exc, inh, hG_scalar=0.5, eta_G=1.0)
    z = spm.op_state_vector(op, kernels, grid); J = spm.build_jacobian_dense(grid, kernels, op)
    assert np.linalg.norm(spm.field_rhs(z, grid, kernels, op)) < 1e-6      # z* is a fixed point
    rng = np.random.default_rng(0); errs = []
    for _ in range(6):
        v = rng.standard_normal(z.size); v /= np.linalg.norm(v)
        fd = (spm.field_rhs(z + 1e-6 * v, grid, kernels, op)
              - spm.field_rhs(z - 1e-6 * v, grid, kernels, op)) / 2e-6
        errs.append(np.linalg.norm(fd - J @ v) / (np.linalg.norm(J @ v) + 1e-300))
    assert max(errs) < 1e-4
```
(Confirm the exact `make_kernels`/`solve_operating_point` shift-arg names against the module before writing — `pilot1` uses `solve_operating_point(..., hG_scalar=H, eta_G=eta_G)` and `gK_field=gK, eta_K=eta_K`.)

- [ ] **Step 2: Run to verify it FAILS** (documents the shift-gap)

Run: `pytest tests/test_topic4_m3b_spectral_phase.py::test_field_rhs_jvp_matches_jacobian_on_shifted_op -q`
Expected: FAIL (fixed-point residual ~3e-2; JVP rel-err ~3e-2) — this is the T2.5-deferred `field_rhs` shift-gap.

- [ ] **Step 3: Fix `field_rhs`** so it applies the SAME g_K/h_G shift that `solve_operating_point`/`build_jacobian_dense` use. Read `field_rhs` (m3b:670) and `solve_operating_point` (m3b:414) to find how the shift enters `muE` (`muE -= eta_G*hG`, `-= eta_K*gK`); thread the same shift into `field_rhs`'s `muE`. Preferred: read the shift from `op` if it is stored there; else add optional `gK_field=None, hG_scalar=None, eta_K=..., eta_G=...` kwargs mirroring `solve_operating_point`. **Keep the no-shift default byte-identical** (the m3b byte-parity suite `pytest -k m3b` must stay 82 pass / 7 pre-existing artifact-absence fail).

- [ ] **Step 4: Run JVP gate + m3b parity**

Run: `pytest tests/test_topic4_m3b_spectral_phase.py -q`
Expected: JVP test PASS; parity unchanged (82 pass / 7 pre-existing fail).

- [ ] **Step 5: Write the failing spread test** — footprint onset is axial, off_axis absent, and the crossing point self-limits.

```python
def test_nonlinear_spread_axial_onset_off_axis_absent():
    cfg = load_crit_config(); grid, kernels, core, b_core = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    crossing = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    sp = m2.read_nonlinear_spread(crossing, _points(), grid, kernels, core, b_core, cfg, m2cfg)
    assert sp["onset"] in ("axial", "core_only", "global_first", "off_axis", "undetermined")
    assert sp["off_axis"] in ("absent", "present", "undetermined")
    assert sp["control_minus_kick"] is True
    # trajectory sanity: off-axis power stays ~0 across all sampled steps
    for fm in sp["footprint_trajectory"]["at_crossing"]["core_kick"]:
        assert fm["off_axis"] < 0.1
```

- [ ] **Step 6: Run to verify fail** → FAIL.

- [ ] **Step 7: Implement `integrate_footprint` + `read_nonlinear_spread`** (port `m2_pilots_round2.py:pilotB` `_integrate_footprint`/`_footprint_metrics` verbatim; add the enum verdicts + epsilon pass/fail from spec §4.3). Run ≥2 depths (`at_crossing` frac, `just_past` frac=0.75) × `epsilon_rel=[0.01,0.05]` × `polarities=[-1,1]`; per (depth,eps,pol) classify `onset`/`endgame`/`off_axis`; then:

```python
def _spread_onset(traj, m2cfg):
    sp = m2cfg["spread"]; bcfg = m2cfg["basis"]
    act = [fm["active_frac"] for fm in traj]
    rose = (max(act) - act[0]) >= sp["expand_active_delta"]
    exp = [fm for fm in traj if fm["active_frac"] > act[0] + 1e-9] or traj
    elong = np.mean([fm["elongation_axis"] for fm in exp])
    offax = np.mean([fm["off_axis"] for fm in exp])
    glob0 = traj[min(2, len(traj)-1)]["globality"]
    if offax >= bcfg["off_axis_score_tol"]:                 return "off_axis"
    if not rose:                                            return "core_only"
    if elong > sp["axial_onset_thresh"] and offax < bcfg["off_axis_score_tol"]: return "axial"
    if glob0 >= sp["global_thresh"] and elong <= sp["axial_onset_thresh"]:      return "global_first"
    return "undetermined"

def _spread_endgame(traj, escaped, m2cfg):
    sp = m2cfg["spread"]; act = [fm["active_frac"] for fm in traj]
    if act[-1] >= sp["flood_active_thresh"]:                                    return "global_flooding"
    if escaped is None and min(act[act.index(max(act)):]) <= sp["self_limit_active_thresh"]: return "self_limited"
    return "marginal"
```
Aggregate with the pass/fail rule: `onset` + `off_axis` must be identical across all 4 (eps×pol) at a given depth (`epsilon_onset_agreement=all`); `endgame` majority ≥3/4 (`epsilon_endgame_agreement=majority`); else `epsilon_sensitivity="epsilon_sensitive"` and the spread segment is undetermined (`unresolved_subreason=unresolved_nonlinear_spread`). `depth_dependent = endgame(at_crossing) != endgame(just_past)`. `control_minus_kick=True` (the integrator always subtracts the v=0 control).

- [ ] **Step 8: Run to verify pass**

Run: `pytest tests/test_topic4_criticality_m2.py -k spread -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/topic4_m3b_spectral_phase.py src/topic4_criticality_m2.py tests/test_topic4_criticality_m2.py tests/test_topic4_m3b_spectral_phase.py
git commit -m "feat(topic4-crit m2): T4 field_rhs shift fix + JVP gate + nonlinear footprint spread"
```

---

## Task 5: Two-stage verdict builder + CLI + figures

**Files:**
- Modify: `src/topic4_criticality_m2.py`
- Create: `scripts/run_topic4_crit_m2.py`
- Create: `results/topic4_criticality_m2/figures/README.md` (after figures render)
- Test: `tests/test_topic4_crit_m2_integration.py`

**Interfaces:**
- Consumes: all of T1–T4. M1 `csd_verdict` read from `results/topic4_criticality/trajectory_verdict.json` (`verdict` field = `unresolved_operating_point`).
- Produces: `build_ignition_spread_verdict(points, cfg_crit, m2cfg)->dict` (top-level spec §1: `csd_verdict`, `linear_ignition`, `nonlinear_spread`, `interpretation`, `base_gate_passed`, `unresolved_subreason`); CLI writing `ignition_spread_verdict.json` + `STATUS.md` + figures.

- [ ] **Step 1: Write the failing integration test** — the full verdict has the three coexisting blocks and does NOT resurrect the retired three-way.

```python
def test_build_verdict_two_stage_coexists_with_csd():
    cfg = load_crit_config(); m2cfg = m2.load_m2_config()
    v = m2.build_ignition_spread_verdict(_points(), cfg, m2cfg)
    assert v["csd_verdict"] == "unresolved_operating_point"          # M1 unchanged
    assert v["linear_ignition"]["class"] == "core_localized"
    assert set(("onset", "endgame", "off_axis", "depth_dependent")) <= set(v["nonlinear_spread"])
    assert "final_verdict" not in v                                  # retired rev1.1 three-way
    assert "ignition" in v["interpretation"] and "off_axis" in v["interpretation"]
```

- [ ] **Step 2: Run to verify fail** → FAIL.

- [ ] **Step 3: Implement `build_ignition_spread_verdict`** — base gates (ignition: crossing localized ∧ `op_solve_quality` clean ∧ shape scores present; spread: JVP-gate-pass ∧ `control_minus_kick` ∧ integration non-fail), then `read_linear_ignition` (T2) + `read_nonlinear_spread` (T4) + `off_axis_sentinel` (T3), then the mechanical interpretation compose (spec §5.3):

```python
def _interpretation(ig, sp):
    return (f"{ig['class']} ignition followed by {sp['onset']} transient and {sp['endgame']}; "
            f"off_axis {sp['off_axis']}")   # NEVER re-glue spread onto the linear mode
```
`csd_verdict = json.loads(M1_verdict)["verdict"]`. Set `unresolved_subreason` from whichever segment failed its gate (ignition→`ignition_not_localized`, spread→`unresolved_nonlinear_spread`); segments are decoupled (one undetermined does not block the other).

- [ ] **Step 4: Run to verify pass** → PASS.

- [ ] **Step 5: Write `scripts/run_topic4_crit_m2.py`** — loads points, `build_ignition_spread_verdict`, writes `results/topic4_criticality_m2/ignition_spread_verdict.json` + `STATUS.md` (plain-language two-stage summary, CLAUDE.md §8) + three figures: **ignition panel** (crossing mode loading heatmap + two-core region bars `coreA/coreB/corridor` + `core_overlap`/`globality`), **spread panel** (`active_frac(t)` / `off_axis(t)` / `elongation(t)` for `at_crossing` vs `just_past`, showing axial-onset→endgame + off_axis≈0), **basis sanity** (`e_axis_gradient` field + crossing loading + nonaxis residual). Add `--from-json` re-render. Then write `figures/README.md` (中文, per-figure `### filename` + `**关注点**：`) and append `results/FIGURE_INDEX.md`.

- [ ] **Step 6: CLI smoke + lazy-import test** (mirror M1's `test_verdict_cli_lazy_deps_importable`)

```python
def test_cli_smoke_writes_verdict(tmp_path):
    import subprocess, sys, json as _j
    r = subprocess.run([sys.executable, "scripts/run_topic4_crit_m2.py", "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=str(_REPO))
    assert r.returncode == 0, r.stderr
    v = _j.loads((tmp_path / "ignition_spread_verdict.json").read_text())
    assert v["csd_verdict"] == "unresolved_operating_point"
```

- [ ] **Step 7: Run full M2 suite + m3b parity**

Run: `pytest tests/test_topic4_criticality_m2.py tests/test_topic4_crit_m2_integration.py -q && pytest -k m3b -q`
Expected: all M2 PASS; m3b parity 82 pass / 7 pre-existing fail.

- [ ] **Step 8: Render figures, eyeball, then write README + commit** (figures require human visual check per AGENTS.md — render, look, fix, then finalize README)

```bash
python scripts/run_topic4_crit_m2.py --out results/topic4_criticality_m2
git add src/topic4_criticality_m2.py scripts/run_topic4_crit_m2.py tests/test_topic4_crit_m2_integration.py \
        results/topic4_criticality_m2/ignition_spread_verdict.json results/topic4_criticality_m2/STATUS.md \
        results/topic4_criticality_m2/figures results/FIGURE_INDEX.md
git commit -m "feat(topic4-crit m2): T5 two-stage verdict + CLI + ignition/spread/basis figures"
```

---

## Self-Review

**1. Spec coverage:**
- §1 two-stage verdict → T5 (`csd_verdict`+`linear_ignition`+`nonlinear_spread`+`interpretation`). ✓
- §2.2 shape scores persisted → T0 (`shape_scores_at`) + T2. §2.3 nonaxis sentinel three-state + annotation → T3. ✓
- §3.1 dense α₀ localization (coarse+bisect+quality two-layer+multi-crossing) → T1. §3.2 ignition + two-core confirm + near_fold_note → T2. §3.3 gains + nonaxis sentinel → T3. ✓
- §4.1 field_rhs shift + JVP hard gate → T4 Steps 1–4. §4.2 footprint integrator → T4 Step 7. §4.3 onset/endgame/off_axis enums + epsilon pass/fail → T4 `_spread_onset`/`_spread_endgame`/aggregation. ✓
- §5.0 base gate (two segments) → T5. §5.1 ignition class + delocalized_subtype + sensitivity → T2 `_classify_ignition`/`_ignition_sensitivity`. §5.2 spread gate → T4. §5.3 interpretation mechanical compose → T5 `_interpretation`. ✓
- §7 red lines (pilot-tier, nonaxis-sentinel, near-fold, symmetric-approx, no-三真数据) → Global Constraints + tests assert the annotation / near_fold_note strings. ✓
- §8 config → T0 yaml (all fields). §9 T0–T5 mapping → this plan's Task 0–5. ✓

**2. Placeholder scan:** The two `...` blocks (T1 bisection loop, referenced to `m2_pilots.py:pilot2` (b) with exact line range and the two params that replace the hard-codes) are pointers to a stable in-repo reference, not vague TODOs — the engineer copies the named loop and swaps `range(8)`→`max_bisect_hard_cap` and `2e-3`→`crossing_width_ms_tol/span_ms`. Every other step has complete code.

**3. Type consistency:** `localize_alpha0_crossing` returns `_crossing_op`/`_crossing_res` (T1) → consumed by `read_linear_ignition` (T2) and `off_axis_sentinel` (T3) and `read_nonlinear_spread` (T4). `shape_scores_at` key names (`axis_elongation`/`off_axis`/`globality`/`core_overlap`/`axis_wavevector_alignment`) identical across T0/T2/T4. `_off_axis_decision` params match `off_axis_sentinel` (T3). `build_ignition_spread_verdict` block names match spec §1 and the T5 test. ✓

**Verification note (before executing T1/T4):** re-confirm the live signatures of `solve_operating_point` (shift-arg names `gK_field`/`hG_scalar`/`eta_K`/`eta_G`), `make_kernels`, `core_perturbation_vector`, and `unpack_state` against `src/topic4_m3b_spectral_phase.py` on the M1-merged base — the pilots used these exact names, but re-grep at execution time (CLAUDE.md §6).
