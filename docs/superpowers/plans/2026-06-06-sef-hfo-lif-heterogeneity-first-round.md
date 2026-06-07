# SEF-HFO LIF Rate-Field First Round (Connectivity Contract + E-Threshold Heterogeneity) Implementation Plan

> **⚠ 2026-06-07 AMENDMENT — threshold distribution fix + gap-limited std lock (read before Tasks 7–9).** User review caught a science bug in the as-built Tasks 4/6: `phi_eff_vth` modeled the E threshold as an **unbounded** Gaussian. Gauss–Hermite then sampled `v_th < V_RESET` (down to −16 mV), where the Siegert formula returns a **negative** firing rate, which was silently clamped with `max(0, .)`. Effect: `phi_eff` was **non-monotonic** in `vth_mean` (a higher mean threshold could give a *higher* mean rate) — breaking `mean_match_vth`'s monotonicity premise and the whole forced chain.
>
> **Fixed (commits 155cdc7, d4a1bf3):** `lif_rate` now **raises** for `v_th < V_RESET`; `phi_eff_vth` integrates a **truncated** Gaussian on `[V_RESET, vth_mean+8·std]` via fixed `leggauss(96)` (deterministic, smooth in μ, renormalized — replaces Gauss–Hermite + clamp); `mean_match_vth` default bracket is now `(V_RESET+0.5, V_TH+17)`; `closed_loop_leading` reports a `converged` flag. **The Task 4 / Task 6 embedded code + test blocks below have been UPDATED IN PLACE to the corrected (truncated leggauss / locked-std) form; the committed source in `src/sef_hfo_heterogeneity.py` / `src/sef_hfo_lif.py` is the source of truth if they ever drift.**
>
> **Gap-limit finding (locks Tasks 7 & 9 params):** with `V_TH−V_RESET = 7 mV` at the canonical sub-reset operating point, `lif_rate` has a steep reset knee. The plan's original `vth_std_wide=4.0` puts **~49%** of the effective rate in the `[V_RESET, 13]` reset knee = reset-floor saturation, **not** collective gain. **Locked clean params: `vth_std_wide=1.5`, `vth_std_narrow=0.5`** (knee share 0.29% / ~0%; narrowing 1.5→0.5 mean-matched raises effective gain ~13% — a computed signal, reported not asserted, spec §7). **All `vth_std_wide=4.0 / vth_std_narrow=1.0` in Tasks 7 & 9 below have been updated to 1.5 / 0.5.** See spec §5.2 and `tests/test_sef_hfo_heterogeneity.py::test_locked_std_params_stay_out_of_reset_knee`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the existing canonical LIF rate field, (a) lock the connectivity "fixed-total-mass" contract and map the ρ/ℓ geometry axis (Track G — mostly already built), and (b) build the genuinely new E-threshold heterogeneity layer with the mandatory mean-matched control, then run the forced chain (narrow `Var(V_th,E)` → effective-curve slope/curvature → closed-loop margin → observables) with direction **computed, not preset** (Track E).

**Architecture:** Extend `src/sef_hfo_lif.py` (transfer takes `v_th`) and add a new `src/sef_hfo_heterogeneity.py` module (Φ_eff integration over a V_th distribution, gain/curvature at op, mean-matched control, spatial-patch integrator). Two analysis scripts: an operating-point level (non-spatial: slope/curvature/λ/margin, raw vs mean-matched) and a spatial-patch level (finite-pulse nucleation + event stats, raw vs mean-matched). Track G is a thin lock-and-sweep on existing machinery. Everything reuses the validated `mean_field` / `lif_rate` / `integrate_lif_field` / `amplitude_thresholds` / `detect_events`.

**Tech Stack:** Python, numpy, scipy (`fsolve`, `erfcx`, `brentq`, fixed Gauss–Legendre via `numpy.polynomial.legendre`), pytest. No new third-party deps.

**Spec of record:** `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` (§5.0 four contracts, §5.1 connectivity, §5.2 heterogeneity, §6 LIF/SNN, §7 acceptance). **This plan covers the LIF rate-field half only; the SNN isomorphic-validation plan is a separate follow-on** (it must match this plan's locked operating point — spec §6 / Contract 4).

**Discipline anchors (read before coding):**
- Spec §5.0 Contract 1 (fixed total E→E mass when sweeping geometry; `w_EE` is the only strength knob), Contract 2 (every heterogeneity experiment runs raw + mean-matched), Contract 3 (next-step is conditional on `joint_A_failmode`, not assumed), Contract 4 (SNN isomorphism — out of scope here, but the operating-point this plan locks IS the SNN's matching target).
- **Non-circularity (spec §7):** tests assert internal consistency (Φ_eff matches quadrature; gain matches finite-diff; mean-match restores baseline rate). Tests **must not** assert the *sign* of the heterogeneity effect — direction is a computed JSON output, and the analysis is allowed to disconfirm the literature (Rich) direction.
- CLAUDE.md §3 (surgical), §6 (implement to full contract; stubs raise `NotImplementedError`), AGENTS.md (reuse, don't reinvent).

---

## File Structure

**Modify:**
- `src/sef_hfo_lif.py` — (Task 3) add `v_th: float = V_TH` parameter to `lif_rate` (backward-compatible; all existing callers unchanged); (Task 6b) append `closed_loop_leading` (+ private `_char_det`/`_rightmost`/`_ghat`), promoting the validated 2×2 LIF closed-loop dispersion from the untracked exploratory script into the canonical module.

**Create:**
- `src/sef_hfo_heterogeneity.py` — the new heterogeneity layer. Responsibilities: (1) `phi_eff_vth` = Φ_LIF integrated over a Gaussian V_th distribution; (2) `eff_gain_curvature` = slope & curvature of Φ_eff at an operating point; (3) `mean_match_vth` = solve mean V_th so narrowing variance does not move the mean rate; (4) `hetero_lut` + `integrate_hetero_field` = spatial core/surround patch integrator reusing the field loop. One module, one responsibility (V_th-distribution effective transfer + its spatial application).
- `scripts/run_sef_hfo_conn_geometry_sweep.py` — Track G: ρ/ℓ_∥ sweep at fixed mass, reusing the 0d machinery; writes `results/topic4_sef_hfo/connectivity_geometry/geometry_sweep.json`.
- `scripts/run_sef_hfo_hetero_optpoint.py` — Track E level 1 (non-spatial): forced chain at the operating point, raw vs mean-matched; writes `results/topic4_sef_hfo/heterogeneity/optpoint.json`.
- `scripts/run_sef_hfo_hetero_patch.py` — Track E level 2 (spatial): patch finite-pulse + event stats, raw vs mean-matched; writes `results/topic4_sef_hfo/heterogeneity/patch.json`.
- `tests/test_sef_hfo_heterogeneity.py` — tests for the new module.
- `results/topic4_sef_hfo/connectivity_geometry/figures/README.md` and `results/topic4_sef_hfo/heterogeneity/figures/README.md` — per AGENTS.md results standard (Chinese, written after figures exist).

**Modify (test lock):**
- `tests/test_sef_hfo_field.py` — add one regression test locking Contract 1 (mass-invariance across ℓ/ρ/θ).

---

## Task 1: Lock Contract 1 — connectivity kernel mass-invariance regression test

The kernel is already normalized (`src/sef_hfo_field.py:47-49` does `g/g.sum()`). This task does NOT change behavior; it locks the contract with an explicit regression test so a future edit that breaks mass-invariance fails loudly.

**Files:**
- Test: `tests/test_sef_hfo_field.py` (append one test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_field.py
import numpy as np
from src.sef_hfo_field import anisotropic_gaussian

def test_ee_kernel_mass_invariant_across_geometry():
    """Contract 1 (spec §5.0): sweeping ell_par/ell_perp/theta must NOT change
    total E→E mass. Kernel is unit-sum by construction; lock it so a future
    edit that reintroduces a shape-dependent total fails here."""
    n, L = 96, 12.0
    masses = []
    for ell_par, ell_perp, theta in [
        (0.9, 0.45, 0.0), (0.9, 0.45, np.pi / 4), (0.9, 0.45, np.pi / 2),
        (0.6, 0.6, 0.0),                # isotropic
        (1.2, 0.30, np.pi / 3),         # high anisotropy
        (0.40, 0.40, 0.0),              # small isotropic
    ]:
        K = anisotropic_gaussian(n, L, ell_par, ell_perp, theta)
        masses.append(float(K.sum()))
    masses = np.array(masses)
    assert np.allclose(masses, 1.0, atol=1e-10), f"kernel mass drifted: {masses}"
```

- [ ] **Step 2: Run test to verify it passes immediately (regression lock, not TDD red)**

Run: `python -m pytest tests/test_sef_hfo_field.py::test_ee_kernel_mass_invariant_across_geometry -v`
Expected: PASS (the kernel is already unit-sum). If it FAILS, stop — the existing normalization is broken and must be investigated before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sef_hfo_field.py
git commit -m "test(sef-hfo): lock Contract 1 — E→E kernel mass-invariant across ell/rho/theta"
```

---

## Task 2: Track G geometry sweep — ρ / ℓ_∥ at fixed mass (reuse 0d machinery)

Map the connectivity geometry axis: ρ→1 should lose the directional template (anisotropy ratio → 1), small ρ over-elongates, and ℓ_∥ scales front reach. This reuses the validated finite-pulse + principal-axis tools; θ_EE rotation is already covered by the locked 0d test, so this sweep adds ρ and ℓ_∥ only.

**Files:**
- Create: `scripts/run_sef_hfo_conn_geometry_sweep.py`
- Reuse: `src.sef_hfo_lif.mean_field`, `src.sef_hfo_lif.integrate_lif_field` (return_peak_field), `scripts.sef_hfo_step0d_anisotropy_control.principal_axis` (import the existing function), `src.sef_hfo_lif.classify_response`

- [ ] **Step 1: Write the failing test (smoke, asserts ρ→1 loses axis)**

```python
# tests/test_sef_hfo_heterogeneity.py  (new file — first test goes here so the
# Track-G smoke and Track-E tests share one module; rename later if it grows)
import numpy as np
import pytest
from src.sef_hfo_lif import mean_field, integrate_lif_field
from scripts.sef_hfo_step0d_anisotropy_control import principal_axis

def _offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    """Off-center seed matching the Step-0b validated chain (= classify_response
    defaults stim_x0=-3, stim_r=2). principal_axis centers the data before the
    covariance, so the off-center displacement does not bias the measured axis."""
    from src.sef_hfo_field import _grid
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)

def test_geometry_rho_controls_directionality():
    """ρ<1 (anisotropic) gives a stronger directional template than ρ=1 (isotropic),
    which has no preferred axis (ratio ~1). Relative assertion is robust to exact
    values; mass is fixed by the unit-sum kernel."""
    op = mean_field(1.0)
    def ratio_for(ell_par, ell_perp):
        out = integrate_lif_field(op, _offcenter_pulse(), theta_EE=0.0,
                                  ell_par=ell_par, ell_perp=ell_perp,
                                  return_peak_field=True, t_max=150.0)
        _angle, ratio = principal_axis(out[2] - op["nuE"])
        return ratio
    r_aniso = ratio_for(0.9, 0.45)   # rho = 0.5
    r_iso = ratio_for(0.6, 0.6)      # rho = 1.0
    assert r_iso < 1.3                       # isotropic: no preferred axis
    assert r_aniso > r_iso + 0.3             # anisotropy creates a directional template
```

- [ ] **Step 2: Run test to verify it passes (validates reuse wiring)**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_geometry_rho_controls_directionality -v`
Expected: PASS for both params. If `principal_axis` import path is wrong, fix the import to the actual location reported by `python -c "import scripts.sef_hfo_step0d_anisotropy_control as m; print(m.principal_axis)"` before proceeding.

- [ ] **Step 3: Write the sweep script**

```python
# scripts/run_sef_hfo_conn_geometry_sweep.py
"""Track G geometry sweep (spec §5.1): ρ and ℓ_∥ at FIXED total E→E mass.
θ_EE rotation is already locked by scripts/sef_hfo_step0d_anisotropy_control.py;
this adds the ρ (directionality) and ℓ_∥ (front reach) axes."""
import json
import numpy as np
from pathlib import Path
from src.sef_hfo_lif import mean_field, integrate_lif_field, classify_response, ELL_PAR
from src.sef_hfo_field import _grid
from scripts.sef_hfo_step0d_anisotropy_control import principal_axis

OUT = Path("results/topic4_sef_hfo/connectivity_geometry/geometry_sweep.json")

def offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    """Off-center seed = Step-0b validated geometry (matches classify_response
    defaults stim_x0=-3, stim_r=2). principal_axis centers before covariance, so
    the displacement does not bias the measured elongation axis."""
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)

def run():
    op = mean_field(1.0)
    rho_rows = []
    for rho in [1.0, 0.9, 0.75, 0.5, 0.33]:
        ell_par = 0.9
        ell_perp = rho * ell_par
        out = integrate_lif_field(op, offcenter_pulse(), ell_par=ell_par,
                                  ell_perp=ell_perp, return_peak_field=True, t_max=150.0)
        ext, front, peak = out[0], out[1], out[2]
        angle, ratio = principal_axis(peak - op["nuE"])
        label, info = classify_response(ext, front)
        rho_rows.append(dict(rho=rho, ell_par=ell_par, ell_perp=ell_perp,
                             axis_angle_deg=angle, anisotropy_ratio=ratio,
                             label=label, adv_mm=info["adv_mm"], dur_ms=info["dur_ms"]))
    ell_rows = []
    for ell_par in [0.36, 0.54, 0.72, 0.9, 1.2]:
        ell_perp = 0.5 * ell_par  # hold ρ=0.5, vary scale
        out = integrate_lif_field(op, offcenter_pulse(), ell_par=ell_par,
                                  ell_perp=ell_perp, return_field=True, t_max=150.0)
        ext, front = out[0], out[1]
        label, info = classify_response(ext, front)
        ell_rows.append(dict(ell_par=ell_par, ell_perp=ell_perp, label=label,
                             adv_mm=info["adv_mm"], dur_ms=info["dur_ms"], max_ext=info["max_ext"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        note="Track G geometry sweep at fixed total E→E mass (unit-sum kernel). "
             "θ_EE rotation locked separately in step0d.",
        operating_point=dict(nuE=op["nuE"], nuI=op["nuI"]),
        rho_sweep=rho_rows, ell_sweep=ell_rows), indent=2))
    print(f"wrote {OUT}")

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run the script**

Run: `python scripts/run_sef_hfo_conn_geometry_sweep.py`
Expected: prints `wrote results/topic4_sef_hfo/connectivity_geometry/geometry_sweep.json`; the JSON shows `anisotropy_ratio` decreasing toward ~1 as `rho → 1.0`, and `adv_mm` increasing with `ell_par`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_conn_geometry_sweep.py tests/test_sef_hfo_heterogeneity.py results/topic4_sef_hfo/connectivity_geometry/geometry_sweep.json
git commit -m "feat(sef-hfo): Track G ρ/ℓ geometry sweep at fixed mass (reuses 0d machinery)"
```

---

## Task 3: Make the LIF transfer threshold-parameterizable (surgical)

`lif_rate` currently reads the module-global `V_TH`. To integrate over a V_th distribution we need it as a parameter. Backward-compatible default = `V_TH`, so every existing caller is unchanged.

**Files:**
- Modify: `src/sef_hfo_lif.py` (the `lif_rate` function only)
- Test: `tests/test_sef_hfo_heterogeneity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E

def test_lif_rate_vth_default_matches_global():
    """Adding v_th param must not change existing behavior at the default."""
    r_default = lif_rate(5.0, 4.0, TAU_ME, TREF_E)
    r_explicit = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    assert r_explicit == r_default

def test_lif_rate_vth_lower_threshold_raises_rate():
    """Lowering threshold (easier to fire) raises rate at fixed mu,sigma —
    a monotone single-neuron sanity, NOT the closed-loop direction claim."""
    r_hi = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    r_lo = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH - 2.0)
    assert r_lo > r_hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_lif_rate_vth_default_matches_global -v`
Expected: FAIL with `TypeError: lif_rate() got an unexpected keyword argument 'v_th'`.

- [ ] **Step 3: Edit `lif_rate` to accept `v_th`**

In `src/sef_hfo_lif.py`, change the signature and the `y_th` line:

```python
def lif_rate(mu: float, sigma: float, tau_m: float, tau_ref: float,
             v_th: float = V_TH) -> float:
    """Siegert formula for the LIF firing rate (kHz).

    v_th: spike threshold (mV). Default V_TH preserves all existing callers;
    pass a varying value to integrate over a threshold distribution
    (src/sef_hfo_heterogeneity.py).
    """
    y_th = (v_th - mu) / sigma
    y_r = (V_RESET - mu) / sigma
    integ, _ = quad(lambda x: erfcx(-x), y_r, y_th, limit=200)
    return 1.0 / (tau_ref + tau_m * np.sqrt(np.pi) * integ)
```

- [ ] **Step 4: Run tests to verify they pass (and nothing else broke)**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_lif_rate_vth_default_matches_global tests/test_sef_hfo_heterogeneity.py::test_lif_rate_vth_lower_threshold_raises_rate tests/test_sef_hfo_lif.py -v`
Expected: all PASS (the existing `test_sef_hfo_lif.py` confirms no regression).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_lif.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): parameterize lif_rate threshold v_th (default-preserving)"
```

---

## Task 4: `phi_eff_vth` — Φ_LIF integrated over a Gaussian V_th distribution

The effective transfer of a patch whose E cells have threshold spread `~N(vth_mean, vth_std²)`. Gauss–Hermite quadrature over V_th. This is the formal `Φ_eff = ∫ Φ_LIF(μ,σ;θ) p(θ) dθ` from the spec.

**Files:**
- Create: `src/sef_hfo_heterogeneity.py`
- Test: `tests/test_sef_hfo_heterogeneity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
from src.sef_hfo_heterogeneity import phi_eff_vth
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E

def test_phi_eff_zero_std_reduces_to_lif_rate():
    """vth_std=0 → no spread → exactly lif_rate at vth_mean."""
    r = phi_eff_vth(5.0, 4.0, TAU_ME, TREF_E, vth_mean=V_TH, vth_std=0.0)
    assert abs(r - lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)) < 1e-9

def test_phi_eff_matches_renormalized_truncated_dense_average():
    """[2026-06-07 fix] GL quadrature must match a dense trapezoid of the RENORMALIZED
    TRUNCATED integrand on the LEGAL support [V_RESET, hi] — not the old unbounded range
    vm±5σ which sampled v_th < V_RESET (circular: it validated the integral against the
    very illegal values the fix removes)."""
    import numpy as np
    from src.sef_hfo_lif import V_RESET
    mu, sig, vm, vs = 8.0, 4.0, 19.0, 1.5
    hi = vm + 8.0 * vs
    grid = np.linspace(V_RESET, hi, 20001)
    w = np.exp(-0.5 * ((grid - vm) / vs) ** 2)
    r = np.array([lif_rate(mu, sig, TAU_ME, TREF_E, v_th=v) for v in grid])
    dense = float(np.trapz(w * r, grid) / np.trapz(w, grid))
    gl = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
    assert abs(gl - dense) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_phi_eff_zero_std_reduces_to_lif_rate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sef_hfo_heterogeneity'`.

- [ ] **Step 3: Create the module with `phi_eff_vth`**

```python
# src/sef_hfo_heterogeneity.py
"""E-threshold heterogeneity layer for the SEF-HFO LIF rate field (spec §5.2).

Effective transfer of a patch whose excitatory thresholds are spread as
N(vth_mean, vth_std^2):  Φ_eff(μ,σ) = ∫ Φ_LIF(μ,σ; v) N(v; vth_mean, vth_std) dv.

DISCIPLINE: narrowing vth_std changes Φ_eff's slope/curvature AND (by Jensen)
its mean level. Every experiment must pair the raw narrowing with a
mean-matched control (mean_match_vth) — spec §5.0 Contract 2. Direction of the
effect is COMPUTED at the operating point, never assumed (spec §7).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from src.sef_hfo_lif import lif_rate, V_TH, V_RESET

# Fixed (deterministic) Gauss–Legendre nodes — smooth in mu for eff_gain_curvature.
_GL_NODES, _GL_WEIGHTS = leggauss(96)

def phi_eff_vth(mu: float, sigma: float, tau_m: float, tau_ref: float,
                vth_mean: float = V_TH, vth_std: float = 0.0) -> float:
    """[2026-06-07 fix — see top banner] Φ_LIF averaged over a TRUNCATED Gaussian
    threshold v_th ~ N(vth_mean, vth_std^2) restricted to the physical support
    v_th >= V_RESET, renormalized by the retained mass. An UNbounded Gaussian sampled
    v_th < V_RESET where the Siegert rate is NEGATIVE (the old max(0,.) clamp masked it
    and made Φ_eff non-monotonic). vth_std=0 returns exactly lif_rate at vth_mean.
    """
    if vth_std <= 0.0:
        return lif_rate(mu, sigma, tau_m, tau_ref, v_th=vth_mean)
    hi = vth_mean + 8.0 * vth_std
    half = 0.5 * (hi - V_RESET); mid = 0.5 * (hi + V_RESET)
    vths = mid + half * _GL_NODES                              # GL nodes mapped onto [V_RESET, hi]
    w_gauss = np.exp(-0.5 * ((vths - vth_mean) / vth_std) ** 2)
    rates = np.array([lif_rate(mu, sigma, tau_m, tau_ref, v_th=float(v)) for v in vths])
    num = float(np.sum(_GL_WEIGHTS * w_gauss * rates))         # the GL ``half`` Jacobian cancels
    den = float(np.sum(_GL_WEIGHTS * w_gauss))                 # renormalize by retained truncated mass
    result = num / den if den > 0.0 else float("nan")
    if not np.isfinite(result):                               # degenerate dist or extreme-threshold quad overflow
        raise ValueError(f"phi_eff_vth: non-finite rate (num={num:.3g}, den={den:.3g}) at "
                         f"vth_mean={vth_mean}, vth_std={vth_std}")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_phi_eff_zero_std_reduces_to_lif_rate tests/test_sef_hfo_heterogeneity.py::test_phi_eff_quadrature_matches_dense_average -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_heterogeneity.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): phi_eff_vth — Φ_LIF integrated over Gaussian V_th distribution"
```

---

## Task 5: `eff_gain_curvature` — slope and curvature of Φ_eff at the operating point

The middle of the forced chain: the first derivative (slope = gain) and second derivative (curvature) of the effective transfer w.r.t. input μ, at a given operating point. Direction is reported, never asserted.

**Files:**
- Modify: `src/sef_hfo_heterogeneity.py`
- Test: `tests/test_sef_hfo_heterogeneity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
from src.sef_hfo_heterogeneity import eff_gain_curvature, phi_eff_vth
from src.sef_hfo_lif import TAU_ME, TREF_E, V_TH

def test_eff_gain_matches_central_difference():
    """Reported slope/curvature must equal a finite-difference of phi_eff itself
    (consistency check — NOT a direction claim)."""
    mu, sig, vm, vs = 6.0, 4.0, V_TH, 2.0
    g = eff_gain_curvature(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs, h=1e-2)
    f = lambda m: phi_eff_vth(m, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
    slope = (f(mu + 1e-2) - f(mu - 1e-2)) / (2e-2)
    curv = (f(mu + 1e-2) - 2 * f(mu) + f(mu - 1e-2)) / (1e-2 ** 2)
    assert abs(g["slope"] - slope) < 1e-6
    assert abs(g["curvature"] - curv) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_eff_gain_matches_central_difference -v`
Expected: FAIL with `ImportError: cannot import name 'eff_gain_curvature'`.

- [ ] **Step 3: Add `eff_gain_curvature` to the module**

```python
# append to src/sef_hfo_heterogeneity.py
def eff_gain_curvature(mu: float, sigma: float, tau_m: float, tau_ref: float,
                       vth_mean: float = V_TH, vth_std: float = 0.0,
                       h: float = 1e-2) -> dict:
    """Slope (∂Φ_eff/∂μ) and curvature (∂²Φ_eff/∂μ²) at input mu, by central
    finite difference. Returns {"slope","curvature","rate"} (kHz, kHz/mV, kHz/mV²).
    The SIGN of these is the computed output of the forced chain — callers report
    it, they do not assume it (spec §7 non-circularity)."""
    f0 = phi_eff_vth(mu, sigma, tau_m, tau_ref, vth_mean, vth_std)
    fp = phi_eff_vth(mu + h, sigma, tau_m, tau_ref, vth_mean, vth_std)
    fm = phi_eff_vth(mu - h, sigma, tau_m, tau_ref, vth_mean, vth_std)
    return {"rate": f0, "slope": (fp - fm) / (2 * h), "curvature": (fp - 2 * f0 + fm) / (h * h)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_eff_gain_matches_central_difference -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_heterogeneity.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): eff_gain_curvature — slope/curvature of Φ_eff at operating point"
```

---

## Task 6: `mean_match_vth` — the mandatory mean-matched control (Contract 2)

When `vth_std` narrows, Φ_eff at fixed μ shifts (Jensen). The control solves for the mean threshold that restores the baseline rate, so the comparison isolates "distribution narrowing" from "mean operating-point moved."

**Files:**
- Modify: `src/sef_hfo_heterogeneity.py`
- Test: `tests/test_sef_hfo_heterogeneity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
from src.sef_hfo_heterogeneity import mean_match_vth, phi_eff_vth
from src.sef_hfo_lif import TAU_ME, TREF_E, V_TH

def test_mean_match_restores_baseline_rate():
    """After mean-matching, Φ_eff at the op input equals the baseline rate,
    even though vth_std changed. This is Contract 2's whole point."""
    mu, sig = 8.0, 4.0
    baseline = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=V_TH, vth_std=1.5)   # locked wide
    vm_matched = mean_match_vth(target_rate=baseline, mu=mu, sigma=sig,
                                tau_m=TAU_ME, tau_ref=TREF_E, vth_std=0.5)         # locked narrow
    got = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm_matched, vth_std=0.5)
    assert abs(got - baseline) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_mean_match_restores_baseline_rate -v`
Expected: FAIL with `ImportError: cannot import name 'mean_match_vth'`.

- [ ] **Step 3: Add `mean_match_vth`**

```python
# append to src/sef_hfo_heterogeneity.py
from scipy.optimize import brentq

def mean_match_vth(target_rate: float, mu: float, sigma: float,
                   tau_m: float, tau_ref: float, vth_std: float,
                   bracket: tuple[float, float] = (V_RESET + 0.5, V_TH + 17.0)) -> float:
    """Solve for vth_mean such that phi_eff_vth(...; vth_mean, vth_std) == target_rate
    at the operating input mu. Restores the baseline mean rate after a variance
    change (spec §5.0 Contract 2). Raises if target is unreachable in the bracket."""
    def g(vm):
        return phi_eff_vth(mu, sigma, tau_m, tau_ref, vth_mean=vm, vth_std=vth_std) - target_rate
    lo, hi = bracket
    if g(lo) * g(hi) > 0:
        raise ValueError(
            f"target_rate={target_rate:.5g} not bracketed by vth_mean∈{bracket} "
            f"at mu={mu}, sigma={sigma}, vth_std={vth_std}; widen the bracket.")
    return float(brentq(g, lo, hi, xtol=1e-8))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_mean_match_restores_baseline_rate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_heterogeneity.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): mean_match_vth — mean-matched control (Contract 2)"
```

---

## Task 6b: Promote the LIF 2×2 closed-loop dispersion to canonical `src/sef_hfo_lif.py`

The real closed-loop stability readout (spec §2C: `max_k Re λ` of the 2×2 E/I Jacobian) currently lives only in the untracked exploratory `scripts/sef_hfo_lif_dispersion_closure.py` (`char_det`/`rightmost`/`leading`), which also imports from the demoted sigmoid `src/sef_hfo_stability.py`. Promote it into canonical `src/sef_hfo_lif.py`, self-contained, so Task 7 can use the **full closed loop** (not an E→E-only proxy). Transcribed verbatim from the validated dispersion structure; locked against the framework banner value (`max Re λ≈−0.05`, k≈0) for the canonical operating point.

**Files:**
- Modify: `src/sef_hfo_lif.py` (append `closed_loop_leading` + private `_char_det`/`_rightmost`/`_ghat`)
- Test: `tests/test_sef_hfo_heterogeneity.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
def test_closed_loop_leading_canonical_op_stable():
    """Canonical white-noise operating point is robustly STABLE with the dominant
    mode near k=0 (framework banner: max Re λ≈−0.05, k=0). Locks the promotion."""
    from src.sef_hfo_lif import mean_field, lif_gains, closed_loop_leading
    op = mean_field(1.0)
    g = lif_gains(op)
    res = closed_loop_leading(g["E"], g["I"])
    assert res["re_max"] < 0.0           # stable
    assert res["re_max"] > -0.15         # ~ -0.05, not wildly off
    assert res["k_star"] < 0.3           # dominant mode near k=0 (no finite-k Hopf)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_closed_loop_leading_canonical_op_stable -v`
Expected: FAIL with `ImportError: cannot import name 'closed_loop_leading'`.

- [ ] **Step 3: Append the promoted dispersion to `src/sef_hfo_lif.py`**

```python
# append to src/sef_hfo_lif.py — closed-loop linear stability (2×2 LIF dispersion)

_CL_DELAY: float = 1.0                          # conduction delay (ms), matches step0a
_CL_K = np.linspace(0.0, 3.0, 31)               # spatial-mode grid (1/mm)

def _ghat(k: float, ell: float) -> float:
    """Gaussian kernel Fourier amplitude along one axis (k_perp=0)."""
    return np.exp(-0.5 * ell * ell * k * k)

def _char_det(lam, k, gE, gI, w_ee_mult):
    """2×2 rate-dispersion determinant at spatial mode k (conduction delay + AMPA/GABA
    synaptic low-pass). gE, gI are dimensionless gains. Verbatim structure from the
    validated LIF dispersion closure; w_ee_mult scales the E→E coupling (default 1.0)."""
    H = lambda ts: np.exp(-lam * _CL_DELAY) / (1 + lam * ts)
    WEE = (C_EE * W_EE * w_ee_mult) * _ghat(k, ELL_PAR)   # k_perp=0 ⇒ only the long axis enters
    WEI = (C_EI * W_EI) * _ghat(k, L_INH)
    WIE = (C_IE * W_IE) * _ghat(k, L_INH)
    WII = (C_II * W_II) * _ghat(k, L_INH)
    a = (1 + TAU_ME * lam) - gE * WEE * H(TAU_AMPA)
    b = gE * WEI * H(TAU_GABA)
    c = -gI * WIE * H(TAU_AMPA)
    d = (1 + TAU_MI * lam) + gI * WII * H(TAU_GABA)
    return a * d - b * c

def _rightmost(k, gE, gI, w_ee_mult):
    best = (-np.inf, 0.0)
    def fr(v):
        z = _char_det(complex(v[0], v[1]), k, gE, gI, w_ee_mult)
        return [z.real, z.imag]
    for r0 in np.linspace(-0.5, 0.3, 12):
        for i0 in np.linspace(0.0, 0.6, 12):
            s, _info, ier, _msg = fsolve(fr, [r0, i0], full_output=True)
            if (ier == 1 and -0.5 - 1e-6 <= s[0] <= 0.3 + 1e-6 and abs(s[1]) <= 0.6
                    and abs(complex(*fr(s))) < 1e-7 and s[0] > best[0]):
                best = (float(s[0]), abs(float(s[1])))
    return best

def closed_loop_leading(gE: float, gI: float, w_ee_mult: float = 1.0) -> dict:
    """Leading eigenvalue (max over spatial mode k) of the 2×2 LIF rate-dispersion
    relation, including conduction delay + AMPA/GABA synaptic kernels. gE, gI are the
    dimensionless E/I gains (∂ν/∂μ·τ_m, as returned by ``lif_gains``). This IS the
    spec §2C closed-loop stability readout — ``re_max`` < 0 ⇒ closed-loop stable.
    NOT the E→E-only proxy. Returns {k_star, re_max, omega, freq_Hz, is_hopf, regime}."""
    res = [_rightmost(float(k), gE, gI, w_ee_mult) for k in _CL_K]
    re = np.array([r[0] for r in res]); im = np.array([r[1] for r in res])
    j = int(np.argmax(re))
    return dict(k_star=float(_CL_K[j]), re_max=float(re[j]), omega=float(im[j]),
                freq_Hz=float(1000.0 * im[j] / (2 * np.pi)),
                is_hopf=bool(_CL_K[j] > 1e-3 and im[j] > 1e-3 and re[j] > re[0] + 1e-4),
                regime=("unstable" if re[j] > 1e-3 else "candidate" if re[j] > -0.02 else "stable"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_closed_loop_leading_canonical_op_stable -v`
Expected: PASS (`re_max` ≈ −0.05, `k_star` ≈ 0). If it fails, diff `_char_det` against `scripts/sef_hfo_lif_dispersion_closure.py::char_det` — the transcription must be exact.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_lif.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): promote 2×2 closed-loop dispersion (closed_loop_leading) to canonical lif module"
```

---

## Task 7: Operating-point forced-chain analysis (Track E level 1, raw vs mean-matched)

Non-spatial: at the locked operating point, narrow `Var(V_th,E)` (raw + mean-matched), compute slope/curvature of Φ_eff and the **full 2×2 E/I closed-loop stability** `max_k Re λ` (reuse `closed_loop_leading` from Task 6b, fed the effective E gain + the bare I gain), and report. Direction is the JSON output, not an assertion.

**Files:**
- Create: `scripts/run_sef_hfo_hetero_optpoint.py`
- Reuse: `src.sef_hfo_lif.{mean_field, lif_gains, closed_loop_leading, TAU_ME, TREF_E}`, `src.sef_hfo_heterogeneity.{eff_gain_curvature, mean_match_vth}`

- [ ] **Step 1: Write the failing test (the script's core helper is importable + isolates the variance effect)**

```python
# append to tests/test_sef_hfo_heterogeneity.py
def test_optpoint_baseline_sits_at_self_consistent_rest():
    """The baseline (and matched) layers must evaluate AT the self-consistent rest
    nuE — else slope/curvature are read off a non-rest input and the field would
    start off its fixed point (review fix 2026-06-06)."""
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    res = analyze_optpoint(vth_std_wide=1.5, vth_std_narrow=0.5)
    nuE = res["operating_point"]["nuE"]
    assert abs(res["baseline"]["rate"] - nuE) < 1e-6
    assert abs(res["mean_matched"]["rate"] - nuE) < 1e-6

def test_optpoint_control_isolates_variance_effect():
    """raw narrowing moves both mean-rate and shape; mean-matched holds rate at nuE.
    We assert ONLY the control invariant (Contract 2), never the sign of the shape
    change (spec §7 non-circularity)."""
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    res = analyze_optpoint(vth_std_wide=1.5, vth_std_narrow=0.5)
    assert abs(res["mean_matched"]["rate"] - res["baseline"]["rate"]) < 1e-6
    for layer in ("baseline", "raw_narrow", "mean_matched"):
        assert np.isfinite(res[layer]["slope"]) and np.isfinite(res[layer]["curvature"])
        assert np.isfinite(res[layer]["closed_loop_re_max"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_optpoint_control_isolates_variance_effect -v`
Expected: FAIL with `ModuleNotFoundError` (script not created).

- [ ] **Step 3: Write the script**

```python
# scripts/run_sef_hfo_hetero_optpoint.py
"""Track E level 1 (spec §5.2): operating-point forced chain for narrowing
Var(V_th,E). Reports slope/curvature of Φ_eff and closed-loop loop gain for
three layers — baseline(wide var), raw narrow, mean-matched narrow. DIRECTION IS
COMPUTED, NOT PRESET (spec §7): the JSON records the signs; this script asserts
nothing about them."""
import json
import numpy as np
from pathlib import Path
from src.sef_hfo_lif import mean_field, lif_gains, closed_loop_leading, TAU_ME, TREF_E
from src.sef_hfo_heterogeneity import eff_gain_curvature, mean_match_vth

OUT = Path("results/topic4_sef_hfo/heterogeneity/optpoint.json")

def _layer(muE, sE, vth_mean, vth_std, w_ee_mult, gI):
    g = eff_gain_curvature(muE, sE, TAU_ME, TREF_E, vth_mean=vth_mean, vth_std=vth_std)
    gE_eff = g["slope"] * TAU_ME                       # effective dimensionless E gain
    cl = closed_loop_leading(gE_eff, gI, w_ee_mult=w_ee_mult)   # FULL 2×2 E/I closed loop (spec §2C)
    return dict(vth_mean=vth_mean, vth_std=vth_std, rate=g["rate"],
                slope=g["slope"], curvature=g["curvature"], gE_eff=gE_eff,
                closed_loop_re_max=cl["re_max"], closed_loop_k_star=cl["k_star"],
                closed_loop_regime=cl["regime"])

def analyze_optpoint(ratio=1.0, w_ee_mult=1.0, vth_std_wide=1.5, vth_std_narrow=0.5):
    op = mean_field(ratio, w_ee_mult=w_ee_mult)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    gI = lif_gains(op)["I"]                            # I gain is homogeneous (no threshold spread)
    # Anchor every layer to the self-consistent rest nuE (review fix 2026-06-06):
    # mean_field solved nuE with bare lif_rate (vth_std=0), so the wide-var transfer
    # must be mean-matched to nuE at muE or it starts off its own fixed point.
    vm_base = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_wide)
    baseline = _layer(muE, sE, vm_base, vth_std_wide, w_ee_mult, gI)        # at nuE
    raw = _layer(muE, sE, vm_base, vth_std_narrow, w_ee_mult, gI)           # same mean, narrowed → Jensen shift
    vm_match = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_narrow)
    matched = _layer(muE, sE, vm_match, vth_std_narrow, w_ee_mult, gI)      # narrowed AND re-matched → at nuE
    return dict(operating_point=dict(muE=muE, sE=sE, nuE=nuE, gI=gI, w_ee_mult=w_ee_mult),
                baseline=baseline, raw_narrow=raw, mean_matched=matched,
                interpretation=dict(
                    note="All layers referenced to the self-consistent rest nuE "
                         "(baseline & mean_matched sit AT nuE; raw_narrow carries the Jensen "
                         "shift). closed_loop_re_max is the FULL 2×2 E/I stability (spec §2C), "
                         "not an E→E proxy. Direction computed, not preset (spec §7). "
                         "mean_matched − baseline = PURE variance effect; "
                         "raw − mean_matched = the Jensen mean shift alone.",
                    d_slope_pure=matched["slope"] - baseline["slope"],
                    d_curvature_pure=matched["curvature"] - baseline["curvature"],
                    d_closed_loop_re_max_pure=matched["closed_loop_re_max"] - baseline["closed_loop_re_max"]))

def run():
    res = analyze_optpoint()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    print(f"  pure variance Δ(closed-loop max Re λ) = {res['interpretation']['d_closed_loop_re_max_pure']:+.4f}")

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test and script**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_optpoint_baseline_sits_at_self_consistent_rest tests/test_sef_hfo_heterogeneity.py::test_optpoint_control_isolates_variance_effect -v && python scripts/run_sef_hfo_hetero_optpoint.py`
Expected: both tests PASS (baseline & matched sit at nuE; control invariant holds); script prints `wrote ...optpoint.json` and a `pure variance Δ(closed-loop max Re λ) = <signed number>`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_hetero_optpoint.py tests/test_sef_hfo_heterogeneity.py results/topic4_sef_hfo/heterogeneity/optpoint.json
git commit -m "feat(sef-hfo): Track E level-1 op-point forced chain (raw vs mean-matched)"
```

---

## Task 8: Spatial core/surround patch integrator (`integrate_hetero_field`)

Apply heterogeneity spatially: a core disk (radius `r_patch` at `x_patch`) with narrowed `Var(V_th,E)` embedded in a surround with baseline `Var(V_th,E)`. Reuses the field loop structure but with a per-region effective transfer (two LUTs + a patch mask). Keeps `integrate_lif_field` untouched (surgical).

**Files:**
- Modify: `src/sef_hfo_heterogeneity.py` (add `hetero_lut`, `integrate_hetero_field`)
- Test: `tests/test_sef_hfo_heterogeneity.py`
- Reuse: `src.sef_hfo_field.{anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid}`, constants from `src.sef_hfo_lif`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sef_hfo_heterogeneity.py
def test_hetero_field_uniform_patch_matches_homogeneous():
    """With core==surround AND vth_std=0, integrate_hetero_field must reduce to the
    homogeneous integrate_lif_field exactly (same LUTs, same loop). vth_std=0 is
    required because integrate_lif_field uses bare lif_rate (no threshold spread)."""
    from src.sef_hfo_lif import mean_field, integrate_lif_field, V_TH
    from src.sef_hfo_heterogeneity import integrate_hetero_field
    from src.sef_hfo_field import _grid
    op = mean_field(1.0)
    X, Y = _grid(96, 12.0)
    pulse = lambda t: (8.0 * ((X ** 2 + Y ** 2) <= 1.5 ** 2) if t < 30.0 else 0.0)
    ext_h, _ = integrate_lif_field(op, pulse, t_max=80.0)
    ext_g, _ = integrate_hetero_field(op, pulse, x_patch=0.0, r_patch=2.0,
                                      vth_std_core=0.0, vth_std_surround=0.0,
                                      vth_mean_core=V_TH, vth_mean_surround=V_TH, t_max=80.0)
    assert np.max(np.abs(ext_h - ext_g)) < 5e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_hetero_field_uniform_patch_matches_homogeneous -v`
Expected: FAIL with `ImportError: cannot import name 'integrate_hetero_field'`.

- [ ] **Step 3: Add `hetero_lut` + `integrate_hetero_field`**

```python
# append to src/sef_hfo_heterogeneity.py
from src.sef_hfo_field import anisotropic_gaussian, isotropic_gaussian, convolve_periodic, _grid
from src.sef_hfo_lif import (
    lif_rate as _lif_rate, W_EE as _W_EE, W_EI as _W_EI, W_IE as _W_IE, W_II as _W_II,
    C_EE as _C_EE, C_EI as _C_EI, C_IE as _C_IE, C_II as _C_II,
    TAU_ME as _TME, TAU_MI as _TMI, TREF_E as _TRE, TREF_I as _TRI,
    JX_E as _JXE, JX_I as _JXI, TAU_AMPA as _TA, TAU_GABA as _TG,
    ELL_PAR as _EP, ELL_PERP as _EPP, L_INH as _LI, DETECT as _DET,
    _DEFAULT_N as _N, _DEFAULT_L as _L,
)

def hetero_lut(sigma: float, vth_mean: float, vth_std: float,
               lo: float = -12.0, hi: float = 45.0, npts: int = 5000):
    """(mus, rates) LUT for the E effective transfer Φ_eff(μ; vth_mean,vth_std)
    at fixed sigma. np.interp over this is the per-region E transfer."""
    mus = np.linspace(lo, hi, npts)
    rates = np.array([phi_eff_vth(m, sigma, _TME, _TRE, vth_mean, vth_std) for m in mus])
    return mus, rates

def integrate_hetero_field(op, stim_fn, *, x_patch: float, r_patch: float,
                           vth_std_core: float, vth_std_surround: float,
                           vth_mean_core: float, vth_mean_surround: float,
                           dt: float = 0.25, t_max: float = 300.0,
                           theta_EE: float = 0.0, n: int = _N, L: float = _L,
                           ell_par: float = _EP, ell_perp: float = _EPP, l_inh: float = _LI,
                           return_peak_field: bool = False):
    """MINIMAL finite-pulse heterogeneity integrator (NOT a full mirror of
    integrate_lif_field). Same core Euler loop and synaptic filters, but DROPS
    recovery (b_a/tau_a), coherence (coh_len) and axis_accum — all out of this
    round's scope (finite-pulse only, no recovery, no noise). The substantive
    addition is a spatially heterogeneous E threshold distribution:
    core disk (center (x_patch,0), radius r_patch) uses (vth_mean_core, vth_std_core);
    surround uses (vth_mean_surround, vth_std_surround); I population unchanged
    (homogeneous Φ_LIF). Per-region E transfer = two LUTs blended by the patch mask.

    Contract: surround params are REQUIRED kwargs (no defaults) so a caller cannot
    silently tune the core without defining the background (spec §5.2)."""
    wee = float(op.get("w_ee_mult", 1.0)) * _W_EE
    KEE = anisotropic_gaussian(n, L, ell_par, ell_perp, theta_EE)
    KI = isotropic_gaussian(n, L, l_inh)
    X, Y = _grid(n, L)
    core = ((X - x_patch) ** 2 + Y ** 2) <= r_patch ** 2

    musC, rC = hetero_lut(op["sE"], vth_mean_core, vth_std_core)
    musS, rS = hetero_lut(op["sE"], vth_mean_surround, vth_std_surround)
    musI = np.linspace(-12.0, 45.0, 5000)
    rI_lut = np.array([_lif_rate(m, op["sI"], _TMI, _TRI) for m in musI])

    def fE(mu):
        out = np.interp(mu, musS, rS)
        out[core] = np.interp(mu[core], musC, rC)
        return out
    fI = lambda mu: np.interp(mu, musI, rI_lut)

    muxE = _TME * _JXE * op["nuext"]; muxI = _TMI * _JXI * op["nuext"]
    rE = np.full((n, n), op["nuE"]); rI = np.full((n, n), op["nuI"])
    sEE = convolve_periodic(rE, KEE).copy(); sEI = convolve_periodic(rI, KI).copy()
    sIE = convolve_periodic(rE, KI).copy(); sII = convolve_periodic(rI, KI).copy()

    nsteps = int(t_max / dt)
    ext = np.empty(nsteps); front = np.empty(nsteps); thr = op["nuE"] + _DET
    peak_field = rE.copy(); peak_ext = -1.0
    for t in range(nsteps):
        stim = stim_fn(t * dt)
        sEE += dt / _TA * (convolve_periodic(rE, KEE) - sEE)
        sEI += dt / _TG * (convolve_periodic(rI, KI) - sEI)
        sIE += dt / _TA * (convolve_periodic(rE, KI) - sIE)
        sII += dt / _TG * (convolve_periodic(rI, KI) - sII)
        muE = _TME * (_C_EE * wee * sEE - _C_EI * _W_EI * sEI) + muxE + stim
        muI = _TMI * (_C_IE * _W_IE * sIE - _C_II * _W_II * sII) + muxI
        rE = rE + dt / _TME * (-rE + fE(muE))
        rI = rI + dt / _TMI * (-rI + fI(muI))
        m = rE > thr
        ext[t] = m.mean()
        front[t] = float(X[m].max()) if m.any() else np.nan
        if ext[t] > peak_ext:
            peak_ext = ext[t]; peak_field = rE.copy()
    if return_peak_field:
        return ext, front, peak_field
    return ext, front
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_hetero_field_uniform_patch_matches_homogeneous -v`
Expected: PASS. With `vth_std=0` and `vth_mean=V_TH`, `hetero_lut` is built from `phi_eff_vth(...vth_std=0)` = `lif_rate(...v_th=V_TH)`, identical to the homogeneous `_lut`, on the same `(-12,45,5000)` μ-grid — so the two integrators are numerically identical and the difference is at solver-noise level.

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_heterogeneity.py tests/test_sef_hfo_heterogeneity.py
git commit -m "feat(sef-hfo): integrate_hetero_field — spatial core/surround V_th patch"
```

---

## Task 9: Spatial patch finite-pulse + event analysis (Track E level 2, raw vs mean-matched)

Fire a finite pulse into the field with a narrowed-`Var(V_th,E)` core; measure whether activity nucleates/concentrates at the patch and whether it self-limits; compare raw vs mean-matched. Reuses `classify_response` for the label (single finite pulse → no noise-driven `detect_events` this round). Surround `Var(V_th,E)` is a required parameter (Contract: don't tune core without surround).

**Files:**
- Create: `scripts/run_sef_hfo_hetero_patch.py`
- Reuse: `src.sef_hfo_heterogeneity.{integrate_hetero_field, mean_match_vth}`, `src.sef_hfo_lif.{mean_field, classify_response, TAU_ME, TREF_E}`

- [ ] **Step 1: Write the failing test (smoke: runs and produces raw + mean-matched layers)**

```python
# append to tests/test_sef_hfo_heterogeneity.py
def test_patch_analysis_runs_both_layers():
    from scripts.run_sef_hfo_hetero_patch import analyze_patch
    res = analyze_patch(t_max=80.0, vth_std_wide=1.5, vth_std_narrow=0.5)
    assert set(res["layers"]) >= {"baseline", "raw_narrow", "mean_matched"}
    for layer in res["layers"].values():
        assert "label" in layer and "max_ext" in layer
    # Contract: surround must be defined (the call below must raise without it)
    import pytest
    from src.sef_hfo_heterogeneity import integrate_hetero_field
    from src.sef_hfo_lif import mean_field
    with pytest.raises(TypeError):
        integrate_hetero_field(mean_field(1.0), lambda t: 0.0,
                               x_patch=0.0, r_patch=2.0, vth_std_core=0.5,
                               vth_mean_core=18.0)  # missing surround kwargs → TypeError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_patch_analysis_runs_both_layers -v`
Expected: FAIL with `ModuleNotFoundError` (script not created).

- [ ] **Step 3: Write the script**

```python
# scripts/run_sef_hfo_hetero_patch.py
"""Track E level 2 (spec §5.2): spatial patch finite-pulse + event analysis.
Narrowed Var(V_th,E) core vs baseline surround; raw vs mean-matched control.
Reports nucleation concentration, self-limit label, event stats. Direction
computed not preset (spec §7); surround is mandatory (don't tune core alone)."""
import json
import numpy as np
from pathlib import Path
from src.sef_hfo_lif import mean_field, classify_response, TAU_ME, TREF_E
from src.sef_hfo_heterogeneity import integrate_hetero_field, mean_match_vth
from src.sef_hfo_field import _grid

OUT = Path("results/topic4_sef_hfo/heterogeneity/patch.json")

def _offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)

def _one(op, x_patch, r_patch, vmc, vsc, vms, vss, t_max):
    ext, front, peak = integrate_hetero_field(
        op, _offcenter_pulse(), x_patch=x_patch, r_patch=r_patch,
        vth_mean_core=vmc, vth_std_core=vsc,
        vth_mean_surround=vms, vth_std_surround=vss,
        t_max=t_max, return_peak_field=True)
    label, info = classify_response(ext, front)
    # nucleation concentration: peak-field mass fraction inside the patch
    X, Y = _grid(96, 12.0)
    core = ((X - x_patch) ** 2 + Y ** 2) <= r_patch ** 2
    dev = np.clip(peak - op["nuE"], 0, None)
    frac_in_patch = float(dev[core].sum() / max(dev.sum(), 1e-12))
    return dict(label=label, max_ext=info["max_ext"], adv_mm=info["adv_mm"],
                dur_ms=info["dur_ms"], returned=info["returned"],
                frac_mass_in_patch=frac_in_patch)

def analyze_patch(ratio=1.0, x_patch=0.0, r_patch=2.0, vth_std_wide=1.5,
                  vth_std_narrow=0.5, t_max=200.0):
    op = mean_field(ratio)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    # Mean-match the SURROUND (the bulk that sets the rest) and the baseline core to
    # the self-consistent rest nuE, so the whole field starts at rest — else a
    # stimulus-free transient contaminates every layer (review fix 2026-06-06).
    vm_surround = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_wide)
    vm_core_matched = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vth_std_narrow)
    # baseline: uniform wide var at nuE (core == surround) → field at rest.
    base = _one(op, x_patch, r_patch, vm_surround, vth_std_wide, vm_surround, vth_std_wide, t_max)
    # raw_narrow: core variance narrowed, SAME mean as surround → carries Jensen core shift.
    raw = _one(op, x_patch, r_patch, vm_surround, vth_std_narrow, vm_surround, vth_std_wide, t_max)
    # mean_matched: core narrowed AND core mean re-solved to nuE → pure variance change.
    matched = _one(op, x_patch, r_patch, vm_core_matched, vth_std_narrow, vm_surround, vth_std_wide, t_max)
    return dict(operating_point=dict(nuE=nuE, muE=muE, sE=sE),
                patch=dict(x_patch=x_patch, r_patch=r_patch,
                           vth_std_wide=vth_std_wide, vth_std_narrow=vth_std_narrow,
                           vth_mean_surround=vm_surround, vth_mean_core_matched=vm_core_matched),
                layers=dict(baseline=base, raw_narrow=raw, mean_matched=matched),
                note="Whole field referenced to self-consistent rest nuE (surround, baseline & "
                     "matched-core sit AT nuE). Direction computed not preset (spec §7). "
                     "mean_matched vs baseline = pure variance effect on nucleation/self-limit; "
                     "raw carries the Jensen core shift.")

def run():
    res = analyze_patch()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
    for k, v in res["layers"].items():
        print(f"  {k:14s} label={v['label']:26s} frac_in_patch={v['frac_mass_in_patch']:.3f}")

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test and script**

Run: `python -m pytest tests/test_sef_hfo_heterogeneity.py::test_patch_analysis_runs_both_layers -v && python scripts/run_sef_hfo_hetero_patch.py`
Expected: test PASS; script prints three layers with labels and `frac_in_patch`. (The science verdict — whether narrowing concentrates/self-limits — is read from the JSON, not asserted.)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_sef_hfo_hetero_patch.py tests/test_sef_hfo_heterogeneity.py results/topic4_sef_hfo/heterogeneity/patch.json
git commit -m "feat(sef-hfo): Track E level-2 spatial patch finite-pulse + event analysis"
```

---

## Task 10: Results READMEs + full-suite regression

Per AGENTS.md results standard, each `figures/` dir needs a Chinese `README.md`. This round produces JSON (no figures yet), so we add a short `README.md` at the results-dir level documenting what each JSON holds and the "direction computed, not preset" caveat, and run the full sef_hfo test suite as a regression gate.

**Files:**
- Create: `results/topic4_sef_hfo/heterogeneity/README.md`, `results/topic4_sef_hfo/connectivity_geometry/README.md`

- [ ] **Step 1: Write the heterogeneity results README**

```markdown
<!-- results/topic4_sef_hfo/heterogeneity/README.md -->
# heterogeneity — E 阈值异质性首轮 (Track E)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.2

- `optpoint.json` — 非空间：在锁定工作点上收窄 `Var(V_th,E)`，比较 baseline / raw_narrow / mean_matched 三层的有效曲线斜率、曲率、**完整 2×2 E/I 闭环稳定性 `closed_loop_re_max`**（max Re λ，spec §2C，不是 E→E 自回授 proxy）。`interpretation.d_*_pure` = mean_matched 减 baseline = **纯方差效应**（已去掉 Jensen 平均移动）。
- `patch.json` — 空间：把收窄方差的 core patch 放进场、打有限脉冲，看活动是否在 patch 聚核 (`frac_mass_in_patch`) + 是否自限 (`label`)，同样三层。

**关注点**：方向是**算出来的、不是预设的**（spec §7）。看 `mean_matched` 与 `baseline` 的差才是"分布变窄本身"的效应；`raw_narrow` 混着平均工作点移动。计算**允许**得出"在这个工作点上收窄方差不压缩余量"——那是真实结果，不是 bug。
```

- [ ] **Step 2: Write the connectivity_geometry results README**

```markdown
<!-- results/topic4_sef_hfo/connectivity_geometry/README.md -->
# connectivity_geometry — 连接几何首轮 (Track G)

spec: `docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.1

- `geometry_sweep.json` — 固定总 E→E mass（unit-sum 核）下扫 `ρ=ℓ_⊥/ℓ_∥`（方向性）与 `ℓ_∥`（前沿可达距离）。`rho_sweep[*].anisotropy_ratio` 随 ρ→1 应趋于 ~1（失去方向轴）；`ell_sweep[*].adv_mm` 随 ℓ_∥ 增大。

**关注点**：θ_EE 旋转的承重判据已由 `scripts/sef_hfo_step0d_anisotropy_control.py` 锁定（DONE PASS），本目录只补 ρ / ℓ 两轴。总 mass 不随几何变（合同 1，由 `test_ee_kernel_mass_invariant_across_geometry` 锁）。
```

- [ ] **Step 3: Run the full sef_hfo regression suite**

Run: `python -m pytest tests/test_sef_hfo_lif.py tests/test_sef_hfo_field.py tests/test_sef_hfo_heterogeneity.py tests/test_sef_hfo_events.py tests/test_sef_hfo_pulse.py tests/test_sef_hfo_stability.py -v`
Expected: ALL PASS (new heterogeneity tests + no regression in the 38 existing tests).

- [ ] **Step 4: Commit**

```bash
git add results/topic4_sef_hfo/heterogeneity/README.md results/topic4_sef_hfo/connectivity_geometry/README.md
git commit -m "docs(sef-hfo): results READMEs for Track G geometry + Track E heterogeneity first round"
```

---

## Self-Review

**Spec coverage (§5.0–§7):**
- Contract 1 (fixed mass when sweeping geometry) → Task 1 (lock test) + Task 2 (sweep uses unit-sum kernel). ✓
- Contract 2 (mean-matched control) → Task 6 (`mean_match_vth`) + Tasks 7 & 9 (raw vs mean-matched layers). ✓
- Contract 3 (heterogeneity vs recovery conditional on `joint_A_failmode`) → this plan implements the heterogeneity branch; the conditional gate itself lives in the existing `run_sef_hfo_step1_joint.py` failmode analysis and is **out of this plan's scope** (noted here so the implementer does not also rebuild it). ✓ (scope-flagged)
- Contract 4 (SNN isomorphism) → separate follow-on plan; this plan locks the operating point that plan must match. ✓ (scope-flagged)
- §5.1 connectivity (θ_EE/ℓ/ρ/w_EE, "定往哪传") → θ_EE already locked (0d); ρ/ℓ in Task 2; w_EE is the existing `w_ee_mult` knob (not swept for direction). ✓
- §5.2 heterogeneity (Var(V_th,E)/x_patch/r_patch/surround, "定哪里点着") → Tasks 4–9; surround is a required kwarg (Task 8). ✓
- §6 LIF (compute gain, integrate heterogeneity) → `phi_eff_vth` (Task 4); SNN is the follow-on plan. ✓
- §7 non-circularity → every test asserts consistency/invariants, never the sign of the effect; direction is JSON output (Tasks 5, 7, 9 + both READMEs). ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step has runnable code; every run step has an exact command + expected output. ✓

**2026-06-06 user-review fixes incorporated:** (1) Task 7 & 9 baselines/surround are mean-matched to the self-consistent rest `nuE` (else the heterogeneous field starts off its fixed point and a stimulus-free transient contaminates every layer); a regression test `test_optpoint_baseline_sits_at_self_consistent_rest` locks it. (2) Task 2 uses the off-center Step-0b pulse so `classify_response`'s default `stim_x0/stim_r` are correct; the ρ-directionality test asserts a relative (anisotropic > isotropic) relationship. (3) Task 8 docstring states it is a *minimal* finite-pulse integrator (drops recovery/coherence/axis_accum — out of scope), not a full mirror. (4) Task 9 reuses `classify_response` only (no misleading `detect_events`); unused imports removed. (5) Task 7's `loop_gain` E→E proxy is replaced by the **real 2×2 E/I closed-loop** `max_k Re λ` (spec §2C): new Task 6b promotes the validated LIF dispersion (`closed_loop_leading`) from the untracked `sef_hfo_lif_dispersion_closure.py` into canonical `src/sef_hfo_lif.py`, locked against the framework banner value (`re_max≈−0.05`, k≈0); Task 7 feeds it the effective E gain + bare I gain and reports `closed_loop_re_max`. ✓

**Type/signature consistency:** `phi_eff_vth(mu,sigma,tau_m,tau_ref,vth_mean,vth_std)` used identically in Tasks 4,5,6,7,8,9. `eff_gain_curvature(...)` returns `{slope,curvature,rate}` used in Task 7. `mean_match_vth(target_rate,mu,sigma,tau_m,tau_ref,vth_std,...)` call order matches definition (Task 6) and uses (Tasks 7,9). `integrate_hetero_field(op, stim_fn, *, x_patch, r_patch, vth_std_core, vth_std_surround, vth_mean_core, vth_mean_surround, ...)` — keyword-only required args; the missing-surround `TypeError` test (Task 9) matches the no-default signature (Task 8). `principal_axis` / `classify_response` / `mean_field` / `integrate_lif_field` reused with their real signatures from the code map. ✓

**Known scope edges (flagged, not gaps):** Contract 3's failmode gate and Contract 4's SNN are deliberately separate; GABA sign-axis / slow-state / inhibitory connectivity are pre-registered (spec §8), not in this round.
