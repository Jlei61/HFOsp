# Reduced 2-D `S_L(x)+S_G` field screen — Implementation Plan (rev 2, post-review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reduced 2-D rate-field screen testing whether the **spatial rank** of inhibitory feedback (local `S_L(x)` vs global `S_G`) makes the synchronised burst-train orbit transversally unstable and yields a bounded phase-staggered attractor — conditional on a NEW dual (divisive+subtractive) pool oscillator.

**Architecture:** Phase-0 uniform 3-state mean-field (`src/topic4_zm_field_meanfield.py`) confirms an orbit exists and locks the operating point by a **minimal-intervention** rule; then the 2-D field (`src/topic4_zm_field_screen.py`: 4 arms, anisotropic `K_E`, derived `w_frac`, per-mode Floquet, streaming metrics); a pure adjudicator (`src/topic4_zm_field_verdict.py`); an orchestrator (`scripts/run_topic4_zm_field_screen.py`) doing Phase 0 → A (write-once lock) → B (**Floquet first**, nonlinear only for candidate windows, per-arm resume).

**Tech Stack:** Python, numpy, scipy, matplotlib(Agg). Reuse `slow_field.psi_recruit/pnorm_pool`, `topic4_zm_patch_screen.population_occupancy`.

**Spec:** `docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md` (rev 3, approved).

## Global Constraints

- Reduced rate field ONLY — no SNN, no H/termination, no E→E change.
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1` on every run.
- SNN-inherited pooling constants (verbatim): `r50=0.4, n_psi=2, p_pool=3, τ_μ=30, τ_S=80, S_max=1`.
- Fast-field constants (verbatim): `τ_a=10, a_max=1, u_half=0.5`. `(W0,α,β,θ)` LOCKED by Phase 0.
- `K_E`: elliptical-exponential `l∥=0.537, l⊥=0.269 mm` rotated to `θ_EE`, `K_E(0)=0`, `Σ=1`. `K_σS`: isotropic Gaussian `σ_S=2.0 mm`. `L=20 mm`, `n=32` primary / `n=64` sensitivity.
- **`w_rec=W0·q_cell`, `w_c=W0·(1−q_cell)`, `q_cell` DERIVED by quadrature** (≈0.226 at n=32, ≈0.077 at n=64). Never hand-set to 0.5.
- **4 arms**: `div_global` (β=0, the current-Z/M baseline), `dual_global`, `dual_local`, `dual_mixed` (`ε_G=0.2`).
- Gate thresholds (verbatim, never tuned to results): occupancy ≥0.80; `P95 ≥ 0.1·a_max`; `mean P_local ≥ 0.5·mean P_global`; active_area_frac ≥0.50; osc_frac (over ALL cells) ≥0.50 with `p2p/a_max ≥0.20` and ≥10 cycles; median `R_phase` <0.50 (only over times with `phase_coverage ≥0.50`); pairwise corr <0.50; **median LOCAL-cell period** ∈ `[0.5,2]×` the matched `dual_global` period; ≥3 CONSECUTIVE of 5 levels each in ≥3/4 seeds; `dual_global` control `R_phase ≥0.80`; transverse `local λ_⊥(k*)>0 & global λ_⊥<0` **excluding the DC mode**, sign stable at `dt` and `dt/2`.
- Phase 0 no-orbit → STOP. `phaseA_lock.json` is **write-once (fail closed if it exists)**.
- **Floquet FIRST, nonlinear only for candidate windows.** Per-arm checkpoint/resume; never one monolithic run.
- Language: "synchronised burst-train orbit" not "carrier"; `ξ∈[0,1]` monotone excitability, never "frozen z"; arm1-vs-2 (β creates the orbit) and arm2-vs-3/4 (spatial rank) are SEPARATE contrasts.

---

### Task 1: Phase-0 mean-field module + orbit detector (the GATE)

**Files:** Create `src/topic4_zm_field_meanfield.py`; Test `tests/test_topic4_zm_field_meanfield.py`

**Interfaces — Produces:**
`F(u,u_half=0.5)`, `psi(r,r50=0.4,n=2.0)`, `psi_prime(r,r50=0.4,n=2.0)`,
`@dataclass MFParams(W0,alpha,beta,theta,I0,tau_a=10.,tau_mu=30.,tau_S=80.,S_max=1.)`,
`simulate_meanfield(p,T=6000.,dt=0.25,r0=0.15) -> (nsteps,3) [r,mu,S]`,
`detect_orbit(traj,dt,settle=0.5) -> dict(oscillates,depth,trough,peak,period_ms,ncyc)`.
Orbit rule (locked): `oscillates` iff `ncyc>=4 and depth>0.5 and trough<0.25*peak and peak>0.1`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topic4_zm_field_meanfield.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_meanfield import (F, psi, psi_prime, MFParams, simulate_meanfield, detect_orbit)

def test_F_and_psi():
    assert F(-1.0) == 0.0 and abs(F(0.5) - 0.5) < 1e-9
    assert psi(0.0) == 0.0 and 0.0 < psi(0.4) < 1.0
    # psi_prime matches a numeric derivative
    h = 1e-6
    assert abs(psi_prime(0.5) - (psi(0.5 + h) - psi(0.5 - h)) / (2 * h)) < 1e-4

def test_dual_pool_oscillates():
    o = detect_orbit(simulate_meanfield(MFParams(W0=2., alpha=2., beta=4., theta=.5, I0=1.)), 0.25)
    assert o["oscillates"] and o["ncyc"] >= 6
    assert o["trough"] < 0.25 * o["peak"] and 100.0 < o["period_ms"] < 300.0

def test_divisive_only_beta0_has_no_orbit():
    """The CURRENT Z/M sg arm is beta_SG=0 -> purely divisive -> no synchronised orbit (Phase-0 finding)."""
    for alpha in (2.0, 8.0, 16.0):
        o = detect_orbit(simulate_meanfield(MFParams(W0=2., alpha=alpha, beta=0.0, theta=.5, I0=1.)), 0.25)
        assert not o["oscillates"]
```

- [ ] **Step 2: Run to verify FAIL** — `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py` → FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write the implementation**

```python
# src/topic4_zm_field_meanfield.py
"""Phase-0 uniform mean-field (r,mu,S) for the reduced S_L(x)+S_G field.
Fix A dual pool: alpha*S divides the recurrent term (matches SNN S_G on I_E_rec) and beta*S subtracts on the
membrane (NEW on this line -- the Z/M sg arm ran beta_SG=0). Gates the 2-D field: no orbit -> STOP.
Spec docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md rev3 §6.0."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def F(u, u_half=0.5):
    x = max(float(u), 0.0)
    return x / (u_half + x)

def psi(r, r50=0.4, n=2.0):
    x = max(float(r), 0.0) ** n
    return x / (r50 ** n + x)

def psi_prime(r, r50=0.4, n=2.0):
    r = max(float(r), 0.0)
    if r <= 0.0:
        return 0.0
    a = r50 ** n
    return n * r ** (n - 1) * a / (a + r ** n) ** 2

@dataclass
class MFParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0

def simulate_meanfield(p: MFParams, T=6000.0, dt=0.25, r0=0.15):
    n = int(round(T / dt)); r, mu, S = float(r0), 0.0, 0.0
    tr = np.empty((n, 3))
    for t in range(n):
        u = p.I0 + p.W0 * r / (1.0 + p.alpha * S) - p.beta * S - p.theta
        r = max(r + dt * (-r + F(u)) / p.tau_a, 0.0)
        mu = mu + dt * (-mu + psi(r)) / p.tau_mu
        S = S + dt * (-S + p.S_max * mu) / p.tau_S
        tr[t] = (r, mu, S)
    return tr

def detect_orbit(traj, dt, settle=0.5):
    r = np.asarray(traj)[int(len(traj) * settle):, 0]
    peak, trough, mean = float(r.max()), float(r.min()), float(r.mean())
    depth = (peak - trough) / (mean + 1e-9)
    mid = 0.5 * (peak + trough)
    cr = np.flatnonzero((r[:-1] < mid) & (r[1:] >= mid))
    period_ms = float(np.mean(np.diff(cr)) * dt) if cr.size >= 2 else float("nan")
    return dict(oscillates=bool(cr.size >= 4 and depth > 0.5 and trough < 0.25 * peak and peak > 0.1),
                depth=depth, trough=trough, peak=peak, period_ms=period_ms, ncyc=int(cr.size))
```

- [ ] **Step 4: Run to verify PASS** — same command → `3 passed`
- [ ] **Step 5: Commit** — `git add src/topic4_zm_field_meanfield.py tests/test_topic4_zm_field_meanfield.py && git commit -m "feat(topic4): Phase-0 mean-field gate (dual pool oscillates; beta=0 does not)"`

---

### Task 2: Contiguous-segment continuation + minimal-intervention lock

**Files:** Modify `src/topic4_zm_field_meanfield.py`; Test `tests/test_topic4_zm_field_meanfield.py`

**Interfaces — Produces:**
`contiguous_runs(flags) -> list[(i0,i1)]` (half-open index runs of consecutive True);
`meanfield_continuation(grid=None, I0s=None, dt=0.25, min_seg=5) -> dict(has_orbit, operating_point, segment, n_configs_with_segment)`.
`operating_point = dict(W0,alpha,beta,theta,I0)`; `segment = dict(I0_lo,I0_hi,interior_I0s,period_ms)`.
Selection (locked, spec §6.0): among configs with a single usable contiguous run of `>= min_seg` oscillatory `I0` points, sort by `(beta, abs(log2(alpha/16)), alpha, abs(W0-2), W0, abs(theta-0.5), theta)` and take the first. `interior_I0s` = the run's points minus its two boundary points.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_meanfield.py
from src.topic4_zm_field_meanfield import contiguous_runs, meanfield_continuation

def test_contiguous_runs_splits_gaps():
    assert contiguous_runs([True, True, False, True, True, True]) == [(0, 2), (3, 6)]
    assert contiguous_runs([False, False]) == []

def test_continuation_minimal_intervention_prefers_smallest_beta():
    # beta=2 and beta=8 BOTH oscillate at (W0=2, alpha=2, I0~1.0-1.5) -> must pick beta=2 (least new
    # mechanism). Unconditional: if this grid yields no orbit the test FAILS (it must not pass vacuously).
    r = meanfield_continuation(grid=dict(W0=[2], alpha=[2], beta=[2, 8], theta=[0.5]),
                               I0s=np.arange(0.8, 1.81, 0.1), min_seg=3)
    assert r["has_orbit"], "expected an orbit for both beta=2 and beta=8 in this grid"
    assert r["operating_point"]["beta"] == 2

def test_continuation_reports_interior_levels_and_segment():
    r = meanfield_continuation(min_seg=5)
    assert r["has_orbit"], "Phase-0 must find an orbit for the field to be built"
    seg = r["segment"]
    assert len(seg["interior_I0s"]) >= 3
    assert seg["I0_lo"] < min(seg["interior_I0s"]) and max(seg["interior_I0s"]) < seg["I0_hi"]

def test_continuation_beta0_grid_has_no_orbit():
    r = meanfield_continuation(grid=dict(W0=[2, 4], alpha=[2, 8, 16], beta=[0.0], theta=[0.5]), min_seg=3)
    assert not r["has_orbit"] and r["operating_point"] is None
```

- [ ] **Step 2: Run to verify FAIL** — `... -k "contiguous or continuation"` → FAIL (ImportError)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_meanfield.py
import itertools

_DEFAULT_GRID = dict(W0=[2, 3, 4, 6], alpha=[1, 2, 4], beta=[0, 1, 2, 4, 8], theta=[0.4, 0.5, 0.6])

def contiguous_runs(flags):
    """Half-open (i0,i1) index runs of consecutive True in `flags`."""
    runs, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i)); start = None
    return runs

def _selection_key(cfg):
    W0, alpha, beta, theta = cfg
    return (beta, abs(np.log2(alpha / 16.0)), alpha, abs(W0 - 2.0), W0, abs(theta - 0.5), theta)

def meanfield_continuation(grid=None, I0s=None, dt=0.25, min_seg=5):
    """Continuation over the pre-registered grid; MINIMAL-INTERVENTION selection (smallest beta, then closest
    to the SNN anchor alpha=16, then W0, then theta; deterministic lexicographic tie-break). Only a SINGLE
    contiguous oscillatory I0 run of >= min_seg points is usable; the 5 levels come from its interior."""
    grid = dict(_DEFAULT_GRID if grid is None else grid)
    I0s = np.arange(0.5, 2.01, 0.1) if I0s is None else np.asarray(I0s, float)
    cands = []
    for W0, alpha, beta, theta in itertools.product(grid["W0"], grid["alpha"], grid["beta"], grid["theta"]):
        flags, pers = [], []
        for I0 in I0s:
            o = detect_orbit(simulate_meanfield(MFParams(W0, alpha, beta, theta, float(I0)), dt=dt), dt)
            flags.append(o["oscillates"]); pers.append(o["period_ms"])
        runs = [(a, b) for a, b in contiguous_runs(flags) if b - a >= min_seg]
        if len(runs) != 1:                      # 0 -> no usable segment; >1 -> ambiguous, skip (spec §6.0)
            continue
        a, b = runs[0]
        cands.append(((W0, alpha, beta, theta), a, b, float(np.nanmedian(pers[a:b]))))
    if not cands:
        return dict(has_orbit=False, operating_point=None, segment=None, n_configs_with_segment=0)
    cfg, a, b, per = sorted(cands, key=lambda c: _selection_key(c[0]))[0]
    seg_I0 = I0s[a:b]
    interior = seg_I0[1:-1]
    mid = float(interior[len(interior) // 2])
    return dict(has_orbit=True, n_configs_with_segment=len(cands),
                operating_point=dict(W0=cfg[0], alpha=cfg[1], beta=cfg[2], theta=cfg[3], I0=mid),
                segment=dict(I0_lo=float(seg_I0[0]), I0_hi=float(seg_I0[-1]),
                             interior_I0s=[float(x) for x in interior], period_ms=per))
```

- [ ] **Step 4: Run to verify PASS** — `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py` → `7 passed` (the continuation test may take ~1-2 min; that is expected)
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): contiguous-segment continuation + minimal-intervention operating-point lock"`

---

### Task 3: Kernels + derived `q_cell`

**Files:** Create `src/topic4_zm_field_screen.py`; Test `tests/test_topic4_zm_field_screen.py`

**Interfaces — Produces:**
`elliptical_exp_kernel(n,L,l_par,l_perp,theta) -> (n,n)` (periodic offsets, `K[0,0]=0`, `Σ=1`);
`gaussian_kernel(n,L,sigma) -> (n,n)` (`Σ=1`);
`cell_mass_fraction(L,n,l_par=0.537,l_perp=0.269,theta=0.0,sub=64,span_cells=12) -> float` (= `q_cell`);
`kernel_axis_and_ar(K,L) -> (axis_rad, ar)` via the kernel-weighted covariance principal eigenvector.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topic4_zm_field_screen.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_screen import (elliptical_exp_kernel, gaussian_kernel, cell_mass_fraction,
                                        kernel_axis_and_ar)

def test_kernels_normalised_and_self_zero():
    K = elliptical_exp_kernel(32, 20.0, 0.537, 0.269, np.radians(30))
    assert abs(K.sum() - 1.0) < 1e-9 and K[0, 0] == 0.0
    assert abs(gaussian_kernel(32, 20.0, 2.0).sum() - 1.0) < 1e-9        # abs() -- a negative diff must fail

def test_kernel_axis_and_ar_recovered_at_several_rotations():
    """Covariance-eigen recovery (NOT row/col HWHM -- DX varies along axis 0, so row/col is easy to flip)."""
    for deg in (0.0, 30.0, 45.0, 75.0):
        K = elliptical_exp_kernel(64, 20.0, 0.537, 0.269, np.radians(deg))
        axis, ar = kernel_axis_and_ar(K, 20.0)
        d = np.degrees(axis) % 180.0
        assert min(abs(d - deg), 180 - abs(d - deg)) < 8.0, (deg, d)     # axis within 8 deg
        assert 1.5 < ar < 3.0, (deg, ar)                                  # AR near 2

def test_cell_mass_fraction_scales_with_resolution():
    q32 = cell_mass_fraction(20.0, 32); q64 = cell_mass_fraction(20.0, 64)
    assert 0.15 < q32 < 0.30 and 0.04 < q64 < 0.12 and q64 < q32          # finer cells hold less mass
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ImportError)

- [ ] **Step 3: Write the implementation**

```python
# src/topic4_zm_field_screen.py
"""Reduced 2-D S_L(x)+S_G rate field (Fix A dual pool) + anisotropic K_E + per-mode Floquet + streaming
metrics + 4 arms. Spec 2026-07-24 rev3. Reduced rate field only -- no SNN, no H, no E->E."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _offset_grid(n, L):
    """Periodic signed offsets (mm); offset (0,0) sits at index (0,0). DX varies along axis 0."""
    idx = (np.arange(n) + n // 2) % n - n // 2
    d = idx * (L / n)
    return np.meshgrid(d, d, indexing="ij")

def elliptical_exp_kernel(n, L, l_par, l_perp, theta):
    DX, DY = _offset_grid(n, L)
    u = DX * np.cos(theta) + DY * np.sin(theta)
    v = -DX * np.sin(theta) + DY * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    K[0, 0] = 0.0
    return K / K.sum()

def gaussian_kernel(n, L, sigma):
    DX, DY = _offset_grid(n, L)
    K = np.exp(-0.5 * (DX ** 2 + DY ** 2) / sigma ** 2)
    return K / K.sum()

def cell_mass_fraction(L, n, l_par=0.537, l_perp=0.269, theta=0.0, sub=64, span_cells=12):
    """q_cell = fraction of the CONTINUOUS K_E mass inside one lattice cell (fine quadrature). Sets the
    local/non-local recurrent split w_rec=W0*q_cell, w_c=W0*(1-q_cell) -- derived, never hand-set."""
    h = L / n
    xs = (np.arange(-span_cells * sub, span_cells * sub) + 0.5) * (h / sub)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    u = X * np.cos(theta) + Y * np.sin(theta); v = -X * np.sin(theta) + Y * np.cos(theta)
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    inside = (np.abs(X) <= h / 2) & (np.abs(Y) <= h / 2)
    return float(K[inside].sum() / K.sum())

def kernel_axis_and_ar(K, L):
    """Principal axis (rad) + aspect ratio of a kernel, via its mass-weighted spatial covariance."""
    n = K.shape[0]
    DX, DY = _offset_grid(n, L)
    w = K / K.sum()
    cxx = float((w * DX * DX).sum()); cyy = float((w * DY * DY).sum()); cxy = float((w * DX * DY).sum())
    C = np.array([[cxx, cxy], [cxy, cyy]])
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0])), float(np.sqrt(max(vals) / max(min(vals), 1e-30)))
```

- [ ] **Step 4: Run to verify PASS** — `3 passed`
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): anisotropic K_E + gaussian pool kernel + derived q_cell"`

---

### Task 4: Fix-A field step, 4 arms, matched-budget identity

**Files:** Modify `src/topic4_zm_field_screen.py`; Test `tests/test_topic4_zm_field_screen.py`

**Interfaces — Produces:**
`@dataclass FieldParams(W0,alpha,beta,theta,I0, n=32, L=20., tau_a=10., tau_mu=30., tau_S=80., S_max=1., r50=0.4, n_psi=2., p_pool=3., sigma_S=2.0, l_par=0.537, l_perp=0.269, theta_EE=0.0, eps_G=0.2, w_frac=None)` — `w_frac=None` ⇒ derived via `cell_mass_fraction`;
`ARMS = ("div_global","dual_global","dual_local","dual_mixed")`;
`simulate_field(p, arm, T, dt, seed=0, r_init=None, state_init=None, record_stride=20) -> dict(r_trace(nrec,n,n) f32, t_ms, final_state)`.
`div_global` forces `beta=0`; `dual_*` use `p.beta`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import FieldParams, simulate_field, ARMS
from src.topic4_zm_field_meanfield import simulate_meanfield, MFParams

def _P(**kw):
    return FieldParams(W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0, **kw)

def test_w_frac_is_derived_not_half():
    p = _P(n=32)
    assert p.w_frac is None
    from src.topic4_zm_field_screen import resolve_w_frac
    assert abs(resolve_w_frac(p) - 0.226) < 0.03            # derived q_cell, NOT 0.5

def test_dual_arms_identical_on_the_uniform_manifold():
    p = _P(n=16)
    outs = {a: simulate_field(p, a, T=800., dt=0.25, r_init=np.full((16, 16), 0.15))["final_state"]["r"]
            for a in ("dual_global", "dual_local", "dual_mixed")}
    assert np.allclose(outs["dual_global"], outs["dual_local"], atol=1e-9)
    assert np.allclose(outs["dual_global"], outs["dual_mixed"], atol=1e-9)
    assert np.allclose(outs["dual_global"], outs["dual_global"].mean())    # stays uniform

def test_div_global_arm_forces_beta_zero():
    p = _P(n=16)
    a = simulate_field(p, "div_global", T=1500., dt=0.25, r_init=np.full((16, 16), 0.15))
    b = simulate_field(FieldParams(W0=2., alpha=2., beta=0.0, theta=.5, I0=1., n=16), "dual_global",
                       T=1500., dt=0.25, r_init=np.full((16, 16), 0.15))
    assert np.allclose(a["final_state"]["r"], b["final_state"]["r"], atol=1e-9)

def test_uniform_field_reduces_to_meanfield():
    p = _P(n=16)
    fr = simulate_field(p, "dual_global", T=1500., dt=0.25, r_init=np.full((16, 16), 0.15), record_stride=20)
    mf = simulate_meanfield(MFParams(2., 2., 4., .5, 1.), T=1500., dt=0.25, r0=0.15)
    got = fr["r_trace"].reshape(fr["r_trace"].shape[0], -1).mean(axis=1)
    assert np.allclose(got, mf[::20, 0][:len(got)], atol=1e-6)
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ImportError: FieldParams)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "src", "snn_engine"))
from slow_field import psi_recruit, pnorm_pool   # noqa: E402  (reuse the SNN pooling nonlinearities)

ARMS = ("div_global", "dual_global", "dual_local", "dual_mixed")

def _Fsat(U, u_half=0.5):
    X = np.maximum(U, 0.0)
    return X / (u_half + X)

@dataclass
class FieldParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    n: int = 32; L: float = 20.0
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0
    r50: float = 0.4; n_psi: float = 2.0; p_pool: float = 3.0
    sigma_S: float = 2.0; l_par: float = 0.537; l_perp: float = 0.269; theta_EE: float = 0.0
    eps_G: float = 0.2
    w_frac: float | None = None          # None -> DERIVED from cell_mass_fraction (never hand-set 0.5)

def resolve_w_frac(p: FieldParams):
    return float(p.w_frac) if p.w_frac is not None else cell_mass_fraction(p.L, p.n, p.l_par, p.l_perp, p.theta_EE)

def arm_beta(p: FieldParams, arm):
    return 0.0 if arm == "div_global" else p.beta

def _S_eff(arm, SL, SG, eps_G):
    if arm in ("div_global", "dual_global"):
        return SG
    if arm == "dual_local":
        return SL
    return (1.0 - eps_G) * SL + eps_G * SG

def simulate_field(p: FieldParams, arm, T=6000.0, dt=0.25, seed=0, r_init=None, state_init=None,
                   record_stride=20):
    assert arm in ARMS, arm
    rng = np.random.default_rng(seed); n = p.n
    KE = np.fft.rfft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.rfft2(gaussian_kernel(n, p.L, p.sigma_S))
    q = resolve_w_frac(p); w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    beta = arm_beta(p, arm)
    if state_init is not None:
        r = np.array(state_init["r"], float); muL = np.array(state_init["muL"], float)
        SL = np.array(state_init["SL"], float); muG = float(state_init["muG"]); SG = float(state_init["SG"])
    else:
        r = np.array(r_init, float) if r_init is not None else np.full((n, n), 0.15)
        muL = np.zeros((n, n)); SL = np.zeros((n, n)); muG = 0.0; SG = 0.0
    r = np.maximum(r, 0.0); rec = []
    for t in range(int(round(T / dt))):
        Se = _S_eff(arm, SL, SG, p.eps_G)
        rec_E = w_rec * r + w_c * np.fft.irfft2(np.fft.rfft2(r) * KE, s=(n, n))
        u = p.I0 + rec_E / (1.0 + p.alpha * Se) - beta * Se - p.theta
        r = np.maximum(r + dt * (-r + _Fsat(u, 0.5)) / p.tau_a, 0.0)
        z = psi_recruit(r, 0.0, p.r50, p.n_psi)                       # nonlinearity FIRST (per location)
        conv = np.fft.irfft2(np.fft.rfft2(z ** p.p_pool) * KS, s=(n, n))
        A_L = np.maximum(conv, 0.0) ** (1.0 / p.p_pool)               # clamp: FFT roundoff -> NaN under ^(1/p)
        A_G = pnorm_pool(z, p.p_pool)
        muL += dt * (-muL + A_L) / p.tau_mu; SL += dt * (-SL + p.S_max * muL) / p.tau_S
        muG += dt * (-muG + A_G) / p.tau_mu; SG += dt * (-SG + p.S_max * muG) / p.tau_S
        if t % record_stride == 0:
            rec.append(r.astype(np.float32).copy())
    return dict(r_trace=np.asarray(rec), t_ms=np.arange(len(rec)) * record_stride * dt,
                final_state=dict(r=r, muL=muL, SL=SL, muG=muG, SG=SG))
```

- [ ] **Step 4: Run to verify PASS** — `7 passed`
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): Fix-A field step, 4 arms, derived w_frac, FFT-root clamp"`

---

### Task 5: Streaming metrics (local-cell period + phase coverage + fail-closed)

**Files:** Modify `src/topic4_zm_field_screen.py`; Test `tests/test_topic4_zm_field_screen.py`

**Interfaces — Produces:**
`_cycle_crossings(x) -> np.ndarray` (upward mid-line crossing indices);
`field_metrics(r_trace, dt_rec_ms, a_max=1.0, settle=0.25) -> dict(occupancy, P95, mean_P, active_area_frac, osc_frac, median_R_phase, phase_coverage_frac, mean_pair_corr, median_local_period_ms, population_period_ms)`.
`median_R_phase` uses ONLY times with `phase_coverage >= 0.5` (fraction of oscillatory cells holding a valid phase); if no such time exists → `median_R_phase = 1.0` (fail closed = looks synchronised). `median_local_period_ms` is the median over oscillatory cells (population period is DIAGNOSTIC only, may be NaN for an ideal staggered state).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import field_metrics

def _osc(n, nt, phases, amp=0.8, base=0.1, period=40):
    t = np.arange(nt); f = np.empty((nt, n, n))
    for i in range(n):
        for j in range(n):
            f[:, i, j] = base + amp * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / period + phases[i, j])))
    return f.astype(np.float32)

def test_inphase_vs_desync_metrics():
    # nt=600 (not 400): settle=0.25 drops 25%, so 600 frames leave 450 = 11.25 cycles at period=40,
    # clearing the LOCKED ncyc>=10 oscillation gate. With nt=400 only 7.5 cycles survive, no cell counts
    # as oscillatory, and BOTH inputs fail-close to R_phase=1.0 -- the test could not discriminate at all.
    n, nt = 8, 600
    mi = field_metrics(_osc(n, nt, np.zeros((n, n))), 5.0)
    md = field_metrics(_osc(n, nt, np.random.default_rng(0).uniform(0, 2 * np.pi, (n, n))), 5.0)
    assert mi["median_R_phase"] > 0.8 and md["median_R_phase"] < 0.5
    assert mi["osc_frac"] > 0.9 and mi["active_area_frac"] > 0.9

def test_local_period_survives_a_flat_population_signal():
    """The IDEAL staggered state flattens P(t) -> population period is NaN, but each cell still cycles.
    The gate must use the LOCAL period, else the best result would fail on a NaN."""
    n, nt = 8, 600
    f = _osc(n, nt, np.random.default_rng(1).uniform(0, 2 * np.pi, (n, n)))
    m = field_metrics(f, 5.0)
    assert 100.0 < m["median_local_period_ms"] < 300.0      # ~40 bins * 5 ms = 200 ms
    assert m["osc_frac"] > 0.9

def test_plateau_and_tiny_active_set_loopholes():
    n, nt = 8, 400
    plateau = np.full((nt, n, n), 0.8, np.float32)
    assert field_metrics(plateau, 5.0)["osc_frac"] < 0.1
    tiny = np.full((nt, n, n), 0.8, np.float32)
    tiny[:, 0, 0] = _osc(1, nt, np.zeros((1, 1)))[:, 0, 0]
    assert field_metrics(tiny, 5.0)["active_area_frac"] < 0.1

def test_phase_coverage_reported_and_failclosed():
    n, nt = 8, 400
    m = field_metrics(np.full((nt, n, n), 0.8, np.float32), 5.0)   # no oscillating cells at all
    assert m["phase_coverage_frac"] == 0.0 and m["median_R_phase"] == 1.0   # fail closed


def test_osc_frac_denominator_is_all_cells_not_the_active_subset():
    """LOCKED contract: osc_frac's denominator is ALL cells, never the active subset. Here 4 of 64 cells
    oscillate (with nt=600, enough frames to clear the ncyc>=10 gate) while the other 60 are flat. The
    correct denominator gives 4/64=0.0625; an active-subset denominator would give 1.0 and would let a
    tiny-active-set state pass as fully oscillatory."""
    n, nt = 8, 600
    f = np.full((nt, n, n), 0.10, np.float32)
    f[:, :2, :2] = _osc(2, nt, np.zeros((2, 2)))      # 2x2 = 4 oscillating cells, same frame budget
    m = field_metrics(f, 5.0)
    assert abs(m["osc_frac"] - 4.0 / 64.0) < 1e-9
    assert m["active_area_frac"] < 0.1                 # only 4/64 cells are active at all
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ImportError: field_metrics)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
def _cycle_crossings(x):
    """Upward mid-line crossing indices (relaxation-oscillation cycle markers)."""
    mid = 0.5 * (float(np.max(x)) + float(np.min(x)))
    x = np.asarray(x)
    return np.flatnonzero((x[:-1] < mid) & (x[1:] >= mid))

def field_metrics(r_trace, dt_rec_ms, a_max=1.0, settle=0.25):
    R = np.asarray(r_trace, float)[int(len(r_trace) * settle):]
    nt = R.shape[0]; flat = R.reshape(nt, -1); ncell = flat.shape[1]
    P = flat.mean(axis=1)
    P95 = float(np.percentile(P, 95))
    occ = float((P >= 0.2 * P95).mean()) if P95 > 1e-12 else 0.0
    amp = flat.max(axis=0) - flat.min(axis=0)
    active = amp >= 0.1 * a_max
    crossings = [(_cycle_crossings(flat[:, c]) if active[c] else np.array([], int)) for c in range(ncell)]
    ncyc = np.array([c.size for c in crossings])
    osc_cells = (ncyc >= 10) & (amp / a_max >= 0.20)
    # per-cell (LOCAL) period -- the gate metric; population period is diagnostic only
    locp = [float(np.mean(np.diff(crossings[c])) * dt_rec_ms) for c in np.flatnonzero(osc_cells)
            if crossings[c].size >= 2]
    median_local_period = float(np.median(locp)) if locp else float("nan")
    # phase per oscillatory cell, then R(t) only where coverage >= 50%
    phases = np.full((nt, ncell), np.nan)
    for c in np.flatnonzero(osc_cells):
        cr = crossings[c]
        for a, b in zip(cr[:-1], cr[1:]):
            phases[a:b, c] = 2 * np.pi * (np.arange(a, b) - a) / (b - a)
    n_osc = int(osc_cells.sum())
    Rt, cov = [], []
    for t in range(nt):
        ph = phases[t][~np.isnan(phases[t])]
        c = (ph.size / n_osc) if n_osc else 0.0
        cov.append(c)
        if n_osc and c >= 0.5 and ph.size >= 2:
            Rt.append(abs(np.mean(np.exp(1j * ph))))
    median_R = float(np.median(Rt)) if Rt else 1.0            # fail closed: no valid coverage -> "synchronised"
    act = flat[:, active]
    if act.shape[1] >= 2 and np.all(act.std(axis=0) > 0):
        C = np.corrcoef(act.T); iu = np.triu_indices(act.shape[1], 1); mpc = float(np.nanmean(C[iu]))
    else:
        mpc = 1.0
    crP = _cycle_crossings(P)
    pop_period = float(np.mean(np.diff(crP)) * dt_rec_ms) if crP.size >= 2 else float("nan")
    return dict(occupancy=occ, P95=P95, mean_P=float(P.mean()), active_area_frac=float(active.mean()),
                osc_frac=float(osc_cells.mean()), median_R_phase=median_R,
                phase_coverage_frac=float(np.mean(cov)), mean_pair_corr=mpc,
                median_local_period_ms=median_local_period, population_period_ms=pop_period)
```

- [ ] **Step 4: Run to verify PASS** — `12 passed` (7 from Tasks 3-4 + 5 new)
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): field metrics with local-cell period + phase coverage + fail-closed"`

---

### Task 6: Transverse Floquet (correct base state, arm-specific dimension, DC excluded)

**Files:** Modify `src/topic4_zm_field_screen.py`; Test `tests/test_topic4_zm_field_screen.py`

**Interfaces — Produces:**
`uniform_orbit(p, dt, T=6000., settle=0.5) -> (orbit(nper,3), period_ms)`;
`variational_jacobian(p, arm, Wk, Kk, r0, mu0, S0) -> np.ndarray` — `(1,1)` for `div_global`/`dual_global` at a non-DC mode (the global pool has NO spatial d.o.f. there), else `(3,3)`;
`transverse_floquet(p, arm, mx, my, orbit, dt) -> float` — INTEGER FFT lattice mode `(mx,my)`; raises `ValueError` on `(0,0)`;
`floquet_map(p, arm, orbit, dt, m_max=6) -> dict(modes, lam, lam_max, k_star)` — enumerates integer modes, **excludes DC**.

Base state (locked, review P0-4): `u0 = I0 + W0·r0/D − β·S0 − θ`, `D = 1+α·S0`, `F'(u0)=u_half/(u_half+u0)²` for `u0>0` else 0.
`∂ṙ_k/∂r_k = (−1 + F'·W_k/D)/τ_a` with `W_k = w_rec + w_c·K̂_E(k)`;
`∂ṙ_k/∂S_k = F'·(−α·W0·r0/D² − β)·c_S/τ_a`, `c_S = 1` (local) / `1−ε_G` (mixed) / 0 (global, non-DC);
`∂μ̇_k/∂r_k = K̂_σS(k)·Ψ'(r0)/τ_μ`; `∂Ṡ_k/∂μ_k = S_max/τ_S`.

- [ ] **Step 1: Write the failing tests** (constructed fixtures with KNOWN answers — never assume the science outcome)

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import (uniform_orbit, variational_jacobian, transverse_floquet, floquet_map)

def test_constant_orbit_recovers_the_jacobian_eigenvalue():
    """With a CONSTANT 'orbit', the monodromy is exp(J*T) -> lambda == max Re eig(J). Known answer."""
    p = _P(n=32)
    const = np.tile(np.array([[0.3, 0.2, 0.2]]), (400, 1))       # frozen (r,mu,S)
    lam = transverse_floquet(p, "dual_local", 2, 0, const, 0.25)
    J = variational_jacobian(p, "dual_local", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert abs(lam - float(np.max(np.linalg.eigvals(J).real))) < 5e-3

def _wk_kk(p, mx, my):
    from src.topic4_zm_field_screen import mode_responses
    return mode_responses(p, mx, my)

def test_global_arm_is_one_dimensional_off_dc():
    p = _P(n=32)
    J = variational_jacobian(p, "dual_global", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert J.shape == (1, 1)                                     # no spatial pool d.o.f. at k != 0
    Jl = variational_jacobian(p, "dual_local", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert Jl.shape == (3, 3)

def test_dc_mode_rejected_and_excluded_from_the_map():
    p = _P(n=32)
    const = np.tile(np.array([[0.3, 0.2, 0.2]]), (200, 1))
    try:
        transverse_floquet(p, "dual_local", 0, 0, const, 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("DC mode must be rejected")
    fm = floquet_map(p, "dual_local", const, 0.25, m_max=3)
    assert (0, 0) not in [tuple(m) for m in fm["modes"]]

def test_dt_halving_sign_margin_on_a_real_orbit():
    p = _P(n=32)
    o1, _ = uniform_orbit(p, 0.25); o2, _ = uniform_orbit(p, 0.125)
    l1 = transverse_floquet(p, "dual_local", 2, 0, o1, 0.25)
    l2 = transverse_floquet(p, "dual_local", 2, 0, o2, 0.125)
    assert np.sign(l1) == np.sign(l2) and abs(l1 - l2) < 0.5 * max(abs(l1), abs(l2), 1e-6) + 1e-3
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ImportError)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
from src.topic4_zm_field_meanfield import (simulate_meanfield, MFParams, detect_orbit, psi_prime)

def uniform_orbit(p: FieldParams, dt, T=6000.0, settle=0.5):
    mf = MFParams(p.W0, p.alpha, p.beta, p.theta, p.I0, p.tau_a, p.tau_mu, p.tau_S, p.S_max)
    tr = simulate_meanfield(mf, T=T, dt=dt)
    o = detect_orbit(tr, dt, settle)
    if not o["oscillates"]:
        raise ValueError("no uniform orbit at this operating point (Phase-0 STOP condition)")
    per = max(2, int(round(o["period_ms"] / dt)))
    tail = tr[int(len(tr) * settle):]
    return tail[:per].copy(), o["period_ms"]

def mode_responses(p: FieldParams, mx, my):
    """(W_k, Khat_sigmaS(k)) at the INTEGER FFT lattice mode (mx,my)."""
    n = p.n
    KE = np.fft.fft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.fft2(gaussian_kernel(n, p.L, p.sigma_S))
    q = resolve_w_frac(p); w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    return w_rec + w_c * float(KE[mx % n, my % n].real), float(KS[mx % n, my % n].real)

def variational_jacobian(p: FieldParams, arm, Wk, Kk, r0, mu0, S0, is_dc=False):
    beta = arm_beta(p, arm)
    D = 1.0 + p.alpha * S0
    u0 = p.I0 + p.W0 * r0 / D - beta * S0 - p.theta            # BASE state uses W0 (uniform), NOT Wk
    Fp = 0.0 if u0 <= 0 else 0.5 / (0.5 + u0) ** 2
    a_rr = (-1.0 + Fp * Wk / D) / p.tau_a
    if arm in ("div_global", "dual_global") and not is_dc:
        return np.array([[a_rr]])                              # global pool has no d.o.f. off DC -> 1-D
    c_S = 1.0 if arm == "dual_local" else (1.0 - p.eps_G) if arm == "dual_mixed" else 1.0
    a_rS = Fp * (-p.alpha * p.W0 * r0 / D ** 2 - beta) * c_S / p.tau_a
    a_mr = Kk * psi_prime(r0, p.r50, p.n_psi) / p.tau_mu
    return np.array([[a_rr, 0.0, a_rS],
                     [a_mr, -1.0 / p.tau_mu, 0.0],
                     [0.0, p.S_max / p.tau_S, -1.0 / p.tau_S]])

def transverse_floquet(p: FieldParams, arm, mx, my, orbit, dt):
    """lambda_perp at integer mode (mx,my) via the monodromy over one orbit period. DC is not transverse."""
    if (mx % p.n, my % p.n) == (0, 0):
        raise ValueError("(0,0) is the DC mode; it is not a transverse mode (its multiplier is neutral)")
    Wk, Kk = mode_responses(p, mx, my)
    dim = 1 if arm in ("div_global", "dual_global") else 3
    M = np.eye(dim)
    for r0, mu0, S0 in orbit:
        J = variational_jacobian(p, arm, Wk, Kk, r0, mu0, S0)
        M = (np.eye(dim) + dt * J) @ M
    rho = float(np.max(np.abs(np.linalg.eigvals(M))))
    return float(np.log(max(rho, 1e-300)) / (len(orbit) * dt))

def floquet_map(p: FieldParams, arm, orbit, dt, m_max=6):
    modes, lam = [], []
    for mx in range(-m_max, m_max + 1):
        for my in range(-m_max, m_max + 1):
            if (mx, my) == (0, 0):
                continue                                        # DC excluded from the transverse map
            modes.append((mx, my)); lam.append(transverse_floquet(p, arm, mx, my, orbit, dt))
    lam = np.asarray(lam); i = int(np.argmax(lam))
    return dict(modes=modes, lam=lam, lam_max=float(lam[i]), k_star=modes[i])
```

- [ ] **Step 4: Run to verify PASS** — `16 passed`
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): transverse Floquet (correct base state, arm dimension, DC excluded)"`

---

### Task 7: Full-state phase-reset helpers

**Files:** Modify `src/topic4_zm_field_screen.py`; Test `tests/test_topic4_zm_field_screen.py`

**Interfaces — Produces:** `orbit_phasepoint_state(p, orbit, phase_idx) -> dict(r,muL,SL,muG,SG)` (ALL fields uniform at that orbit phase); `add_r_perturbation(state, eps, seed, n) -> state` (zero-mean, `r` only).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import orbit_phasepoint_state, add_r_perturbation

def test_phasepoint_resets_every_field_uniformly():
    p = _P(n=8); orbit, _ = uniform_orbit(p, 0.25)
    st = orbit_phasepoint_state(p, orbit, len(orbit) // 3)
    for k in ("r", "muL", "SL"):
        assert np.allclose(st[k], st[k].flat[0])       # no leftover spatial S_L memory
    assert np.isscalar(st["muG"]) and np.isscalar(st["SG"])

def test_perturbation_is_zero_mean_and_r_only():
    p = _P(n=16); orbit, _ = uniform_orbit(p, 0.25)
    st0 = orbit_phasepoint_state(p, orbit, 5)
    st1 = add_r_perturbation({k: (v.copy() if hasattr(v, "copy") else v) for k, v in st0.items()}, 1e-4, 0, 16)
    assert abs(float((st1["r"] - st0["r"]).mean())) < 1e-12
    assert np.allclose(st1["SL"], st0["SL"]) and np.allclose(st1["muL"], st0["muL"])
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ImportError)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
def orbit_phasepoint_state(p: FieldParams, orbit, phase_idx):
    r0, mu0, S0 = orbit[int(phase_idx) % len(orbit)]
    o = np.ones((p.n, p.n))
    return dict(r=r0 * o, muL=mu0 * o, SL=S0 * o, muG=float(mu0), SG=float(S0))

def add_r_perturbation(state, eps, seed, n):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, n)); d -= d.mean()
    state["r"] = state["r"] + eps * float(np.max(state["r"])) * d
    return state
```

- [ ] **Step 4: Run to verify PASS** — `18 passed`
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): full-state phase-reset + zero-mean r-perturbation"`

---

### Task 8: Adjudicator (pure function, TDD)

**Files:** Create `src/topic4_zm_field_verdict.py`; Test `tests/test_topic4_zm_field_verdict.py`

**Interfaces — Produces:**
`TH = dict(occ=0.80, area=0.50, osc=0.50, R=0.50, corr=0.50, R_global=0.80, p_lo=0.5, p_hi=2.0, P95_min=0.1, seeds_required=3, levels_required=3)`;
`level_arm_passes(metrics_list, global_period_ms) -> (n_pass:int, detail)` — a seed passes iff ALL criteria; missing/NaN → fail closed;
`level_is_valid(global_metrics) -> bool` — the `dual_global` control must still be a synchronised oscillation (`median_R_phase >= 0.80` and `osc_frac >= 0.5`), else the level has no matched comparison;
`adjudicate_field_screen(summary, lock) -> dict(verdict, taxonomy, passing_levels, window, reasons)`.
Verdict values: `GO`, `reverse_global_unstable_local_stable`, `both_stable`, `both_unstable`, `subcritical_finite_amplitude_candidate`, `no_go`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topic4_zm_field_verdict.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_verdict import level_arm_passes, level_is_valid, adjudicate_field_screen

def M(**kw):
    d = dict(occupancy=0.9, P95=0.5, mean_P=0.3, active_area_frac=0.9, osc_frac=0.9, median_R_phase=0.2,
             phase_coverage_frac=0.9, mean_pair_corr=0.1, median_local_period_ms=200.0)
    d.update(kw); return d

def _summary(levels):
    return dict(levels=levels)

def _lvl(local_metrics, lam_local=0.05, lam_global=-0.05, glob=None):
    return dict(arms=dict(dual_local=dict(metrics=local_metrics, lambda_perp_max=lam_local),
                          dual_global=dict(metrics=[glob or M(median_R_phase=0.95)] * 4,
                                           lambda_perp_max=lam_global, period_ms=200.0)))

def test_seed_and_criteria_counting():
    n, _ = level_arm_passes([M(), M(), M(), M(occupancy=0.1)], 200.0)
    assert n == 3
    n2, _ = level_arm_passes([M(), M(), M(median_R_phase=0.95), M(osc_frac=0.1)], 200.0)
    assert n2 == 2

def test_nan_and_missing_fail_closed():
    n, _ = level_arm_passes([M(median_local_period_ms=float("nan")), M(), M(), M()], 200.0)
    assert n == 3
    n2, _ = level_arm_passes([{}, M(), M(), M()], 200.0)
    assert n2 == 3

def test_period_band_enforced():
    n, _ = level_arm_passes([M(median_local_period_ms=2000.0)] * 4, 200.0)     # 10x global -> out of band
    assert n == 0

def test_level_validity_requires_synchronised_global_control():
    assert level_is_valid(dict(median_R_phase=0.95, osc_frac=0.9))
    assert not level_is_valid(dict(median_R_phase=0.2, osc_frac=0.9))          # global desynced -> invalid
    assert not level_is_valid(dict(median_R_phase=0.95, osc_frac=0.05))        # global silent -> invalid

def test_three_consecutive_pass_gives_GO():
    lv = {str(i): _lvl([M()] * 4) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] == "GO" and len(r["passing_levels"]) >= 3

def test_non_consecutive_levels_do_not_give_GO():
    lv = {str(i): _lvl([M()] * 4) for i in (0, 2, 4)}
    lv.update({str(i): _lvl([M(occupancy=0.1)] * 4) for i in (1, 3)})
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] != "GO"

def test_two_of_four_seeds_does_not_pass():
    lv = {str(i): _lvl([M(), M(), M(occupancy=0.1), M(occupancy=0.1)]) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] != "GO"

def test_invalid_global_level_is_excluded_not_counted_as_failure():
    lv = {str(i): _lvl([M()] * 4) for i in range(5)}
    lv["2"] = _lvl([M()] * 4, glob=M(median_R_phase=0.2))          # global desynced -> level excluded
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert 2 not in [int(x) for x in r["passing_levels"]]
    assert "2" in [str(x) for x in r["reasons"].get("excluded_levels", [])]

def test_subcritical_when_nonlinear_passes_but_floquet_stable():
    lv = {str(i): _lvl([M()] * 4, lam_local=-0.05) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["verdict"] == "subcritical_finite_amplitude_candidate"

def test_lambda_below_the_noise_floor_is_indeterminate_not_a_verdict():
    """spec §6.2: |lam| under the discretisation error floor cannot resolve a sign, so it must NOT be
    reported as a stable/unstable cell and must never yield GO."""
    lv = {str(i): _lvl([M()] * 4, lam_local=5e-4, lam_global=-5e-4) for i in range(5)}
    r = adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))
    assert r["taxonomy"] == "indeterminate_below_noise_floor"
    assert r["verdict"] != "GO"


def test_taxonomy_four_cells():
    def tax(ll, lg):   # magnitudes deliberately ABOVE TH["lam_floor"]=2e-3 so signs are resolvable
        lv = {str(i): _lvl([M(occupancy=0.1)] * 4, lam_local=ll, lam_global=lg) for i in range(5)}
        return adjudicate_field_screen(_summary(lv), dict(I0_levels=[0, 1, 2, 3, 4]))["taxonomy"]
    assert tax(-0.05, -0.05) == "both_stable"
    assert tax(0.05, 0.05) == "both_unstable"
    assert tax(-0.05, 0.05) == "global_unstable_local_stable"
    assert tax(0.05, -0.05) == "global_stable_local_unstable"
```

- [ ] **Step 2: Run to verify FAIL** — FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write the implementation**

```python
# src/topic4_zm_field_verdict.py
"""Pure adjudicator for the reduced-field screen (spec 2026-07-24 rev3 §8-§9). Thresholds locked; every
missing / NaN metric fails CLOSED. The transverse taxonomy already excludes the DC mode upstream."""
from __future__ import annotations
import math

TH = dict(occ=0.80, area=0.50, osc=0.50, R=0.50, corr=0.50, R_global=0.80, p_lo=0.5, p_hi=2.0,
          P95_min=0.1, seeds_required=3, levels_required=3,
          # spec §6.2: a growth rate is only sign-resolvable ABOVE the discretisation error floor.
          # Measured Euler-vs-exact monodromy error 2e-5..1.3e-3 and dt-halving scatter ~5e-4 (local) /
          # 1.3e-3 (global) at dt=0.25 -> 2e-3 is the conservative floor. |lam| <= floor = indeterminate.
          lam_floor=2e-3)

def _num(d, k):
    v = d.get(k, None) if isinstance(d, dict) else None
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v

def _seed_passes(m, global_period_ms):
    need = dict(occupancy=lambda v: v >= TH["occ"], P95=lambda v: v >= TH["P95_min"],
                active_area_frac=lambda v: v >= TH["area"], osc_frac=lambda v: v >= TH["osc"],
                median_R_phase=lambda v: v < TH["R"], mean_pair_corr=lambda v: v < TH["corr"])
    for k, ok in need.items():
        v = _num(m, k)
        if v is None or not ok(v):
            return False, k
    per = _num(m, "median_local_period_ms")
    if per is None or global_period_ms in (None, 0):
        return False, "median_local_period_ms"
    if not (TH["p_lo"] * global_period_ms <= per <= TH["p_hi"] * global_period_ms):
        return False, "period_band"
    return True, None

def level_arm_passes(metrics_list, global_period_ms):
    n, why = 0, []
    for m in metrics_list:
        ok, k = _seed_passes(m if isinstance(m, dict) else {}, global_period_ms)
        n += int(ok); why.append(k)
    return n, dict(failed_on=why)

def level_is_valid(global_metrics):
    """The dual_global control must remain a SYNCHRONISED oscillation, else this level has no matched
    comparison (excluded, NOT counted as 'global failed to synchronise')."""
    R = _num(global_metrics or {}, "median_R_phase"); osc = _num(global_metrics or {}, "osc_frac")
    return bool(R is not None and osc is not None and R >= TH["R_global"] and osc >= TH["osc"])

def _taxonomy(lam_local, lam_global):
    def cls(v):
        if v is None or abs(v) <= TH["lam_floor"]:
            return "indet"                      # below the discretisation noise floor -> sign not resolvable
        return "unstable" if v > 0 else "stable"
    l, g = cls(lam_local), cls(lam_global)
    if "indet" in (l, g): return "indeterminate_below_noise_floor"
    if g == "unstable" and l == "unstable": return "both_unstable"
    if g == "unstable": return "global_unstable_local_stable"
    if l == "unstable": return "global_stable_local_unstable"
    return "both_stable"

def adjudicate_field_screen(summary, lock):
    levels = summary.get("levels", {})
    order = [str(x) for x in lock.get("I0_levels", sorted(levels))]
    passing, excluded, tax_votes, floquet_ok = [], [], [], []
    for key in order:
        lv = levels.get(key)
        if not lv:
            continue
        arms = lv.get("arms", {})
        g = arms.get("dual_global", {}); gm = (g.get("metrics") or [{}])[0]
        if not level_is_valid(gm):
            excluded.append(key); continue
        gper = _num(g, "period_ms")
        lam_g = _num(g, "lambda_perp_max")
        best = None
        for arm in ("dual_local", "dual_mixed"):
            a = arms.get(arm)
            if not a:
                continue
            n, _ = level_arm_passes(a.get("metrics") or [], gper)
            lam_l = _num(a, "lambda_perp_max")
            tax_votes.append(_taxonomy(lam_l, lam_g))
            if n >= TH["seeds_required"]:
                best = arm
                floquet_ok.append(lam_l is not None and lam_g is not None
                                  and lam_l > TH["lam_floor"] and lam_g < -TH["lam_floor"])
        if best:
            passing.append(key)
    # longest run of CONSECUTIVE passing levels in the locked order
    run = best_run = 0; window = []
    cur = []
    for key in order:
        if key in passing:
            cur.append(key); run += 1
            if run > best_run:
                best_run, window = run, list(cur)
        else:
            run = 0; cur = []
    taxonomy = max(set(tax_votes), key=tax_votes.count) if tax_votes else "both_stable"
    if best_run >= TH["levels_required"]:
        verdict = "GO" if any(floquet_ok) else "subcritical_finite_amplitude_candidate"
    else:
        verdict = {"global_unstable_local_stable": "reverse_global_unstable_local_stable",
                   "both_stable": "both_stable", "both_unstable": "both_unstable"}.get(taxonomy, "no_go")
    return dict(verdict=verdict, taxonomy=taxonomy, passing_levels=passing, window=window,
                reasons=dict(excluded_levels=excluded, longest_consecutive=best_run))
```

- [ ] **Step 4: Run to verify PASS** — `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_verdict.py` → `11 passed`
- [ ] **Step 5: Commit** — `git commit -m "feat(topic4): pure reduced-field adjudicator with fail-closed TDD"`

---

### Task 9: Orchestrator — Phase 0 → A (write-once lock) → B (Floquet first, resume)

**Files:** Create `scripts/run_topic4_zm_field_screen.py`

**Interfaces — Produces** (each callable independently, review P0-6):
`run_formation_arm(p, arm, seed, T, dt, orbit)` (aligned-orbit + `1e-4` perturbation → metrics + trace path);
`run_phase_reset_arm(p, arm, seed, T, dt, orbit, formed_state)` (full-state reset → re-form check);
`run_long_confirm`, `run_resolution_confirm` (`n=64`), `run_dt_confirm` (`dt/2`).
Outputs `results/topic4_sef_hfo/zm_field_screen/{phaseA_lock.json, floquet_map.json, runs/<level>_<arm>_<seed>.json, traces/<...>.npz, field_screen_summary.json}`.

- [ ] **Step 1: Write the orchestrator**

```python
#!/usr/bin/env python
"""Reduced 2-D S_L(x)+S_G field screen: Phase 0 (mean-field gate) -> Phase A (write-once lock) -> Phase B
(FLOQUET FIRST for every level; nonlinear runs ONLY for candidate windows) -> adjudicate. Per-arm resume.
Spec 2026-07-24 rev3. Reduced rate field only. --confirm-run required."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.topic4_zm_field_meanfield import meanfield_continuation, simulate_meanfield, MFParams, detect_orbit
from src.topic4_zm_field_screen import (FieldParams, simulate_field, field_metrics, uniform_orbit,
                                        floquet_map, orbit_phasepoint_state, add_r_perturbation,
                                        resolve_w_frac, ARMS)
from src.topic4_zm_field_verdict import adjudicate_field_screen

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_field_screen")
RUNS, TRACES = os.path.join(OUT, "runs"), os.path.join(OUT, "traces")
SEEDS = (0, 1, 2, 3); DT = 0.25; N = 32; EPS = 1e-4; REC_MS = 5.0

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def _git(*a):
    return subprocess.check_output(["git", "-C", _ROOT, *a], text=True).strip()

def phase0():
    r = meanfield_continuation()
    if not r["has_orbit"]:
        print("[PHASE0] no contiguous orbit segment in the grid -> STOP (field NOT built)."); sys.exit(2)
    op, seg = r["operating_point"], r["segment"]
    # dt/2 classification stability of the operating point
    o2 = detect_orbit(simulate_meanfield(MFParams(op["W0"], op["alpha"], op["beta"], op["theta"], op["I0"]),
                                         dt=DT / 2), DT / 2)
    if not o2["oscillates"]:
        print("[PHASE0] operating point not stable under dt/2 -> STOP."); sys.exit(2)
    print(f"[PHASE0] op={op} segment={seg['I0_lo']}..{seg['I0_hi']} interior={seg['interior_I0s']}")
    return op, seg

def phaseA_lock(op, seg):
    path = os.path.join(OUT, "phaseA_lock.json")
    if os.path.exists(path):                       # WRITE-ONCE: fail closed, never silently overwrite
        print(f"[PHASE A] lock already exists at {path}; reusing it (delete it manually to re-lock).")
        return json.load(open(path))
    os.makedirs(OUT, exist_ok=True)
    interior = [float(x) for x in seg["interior_I0s"]]
    if len(interior) >= 5:            # 5 levels EVENLY across the interior (spec §8 Phase A) -- taking the
        idx = np.linspace(0, len(interior) - 1, 5).round().astype(int)   # first 5 would bias toward low I0
        levels = [interior[i] for i in dict.fromkeys(idx.tolist())]
    else:
        levels = interior
    p0 = FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=op["I0"], n=N)
    lock = dict(spec_sha=_sha(os.path.join(_ROOT, "docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md")),
                operating_point=op, segment=seg, I0_levels=levels, seeds=list(SEEDS), dt=DT, grid_n=N,
                w_frac_derived=resolve_w_frac(p0), eps_perturb=EPS,
                git_sha=_git("rev-parse", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(lock, open(path, "w"), indent=2)
    print(f"[PHASE A] wrote {path} levels={levels}")
    return lock

def _params(lock, I0, n=None):
    op = lock["operating_point"]
    return FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=I0,
                       n=n or lock["grid_n"])

def _run_path(level, arm, seed, tag):
    return os.path.join(RUNS, f"{tag}_L{level}_{arm}_s{seed}.json")

def run_formation_arm(lock, I0, arm, seed, T, dt=None, n=None, tag="form"):
    """Aligned-orbit start + fixed 1e-4 zero-mean r-perturbation -> does a staggered state FORM?"""
    dt = dt or lock["dt"]; level = f"{I0:.4f}"
    path = _run_path(level, arm, seed, tag)
    if os.path.exists(path):
        return json.load(open(path))                 # resume
    p = _params(lock, I0, n)
    orbit, per = uniform_orbit(p, dt)
    st = add_r_perturbation(orbit_phasepoint_state(p, orbit, len(orbit) // 3), lock["eps_perturb"], seed, p.n)
    out = simulate_field(p, arm, T=T, dt=dt, seed=seed, state_init=st, record_stride=int(round(REC_MS / dt)))
    m = field_metrics(out["r_trace"], REC_MS)
    os.makedirs(RUNS, exist_ok=True); os.makedirs(TRACES, exist_ok=True)
    tp = os.path.join(TRACES, f"{tag}_L{level}_{arm}_s{seed}.npz")
    np.savez_compressed(tp, r_trace=out["r_trace"][::4], t_ms=out["t_ms"][::4])   # downsampled for figures
    rec = dict(level=level, arm=arm, seed=seed, tag=tag, T=T, dt=dt, n=p.n, metrics=m, period_ms=per,
               trace=os.path.basename(tp), final_state={k: (v.tolist() if hasattr(v, "tolist") else v)
                                                        for k, v in out["final_state"].items()})
    json.dump(rec, open(path, "w"), indent=2)
    return rec

def run_phase_reset_arm(lock, I0, arm, seed, T, formed_rec):
    """FULL-state reset (r,muL,SL,muG,SG) to a uniform orbit phase + the same perturbation -> does the
    staggered state RE-FORM? (An attractor test, not a leftover-inhibition-memory test.)"""
    return run_formation_arm(lock, I0, arm, seed, T, tag="reset")

def run_long_confirm(lock, I0, arm, seed):
    return run_formation_arm(lock, I0, arm, seed, T=60000.0, tag="long60s")

def run_resolution_confirm(lock, I0, arm, seed):
    return run_formation_arm(lock, I0, arm, seed, T=30000.0, n=64, tag="n64")

def run_dt_confirm(lock, I0, arm, seed):
    return run_formation_arm(lock, I0, arm, seed, T=30000.0, dt=lock["dt"] / 2, tag="dthalf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("refusing to run without --confirm-run")
    op, seg = phase0()
    lock = phaseA_lock(op, seg)
    T = 3000.0 if a.smoke else 30000.0

    # ---- Phase B.1: FLOQUET FIRST (cheap) ----
    fmap = {}
    for I0 in lock["I0_levels"]:
        p = _params(lock, I0); orbit, _ = uniform_orbit(p, lock["dt"])
        fmap[f"{I0:.4f}"] = {arm: floquet_map(p, arm, orbit, lock["dt"], m_max=4)["lam_max"] for arm in ARMS}
        print(f"[floquet I0={I0:.3f}] " + " ".join(f"{k}={v:+.4f}" for k, v in fmap[f'{I0:.4f}'].items()))
    json.dump(fmap, open(os.path.join(OUT, "floquet_map.json"), "w"), indent=2)
    targets = [I0 for I0 in lock["I0_levels"]
               if fmap[f"{I0:.4f}"]["dual_global"] < 0 and
               max(fmap[f"{I0:.4f}"]["dual_local"], fmap[f"{I0:.4f}"]["dual_mixed"]) > 0]
    print(f"[floquet] target-window levels: {targets}")
    if not targets and not a.smoke:
        print("[floquet] no target window -> writing taxonomy verdict WITHOUT the expensive nonlinear sweep.")

    # ---- Phase B.2: nonlinear ONLY for candidate levels (or all, in smoke) ----
    run_levels = lock["I0_levels"] if (a.smoke or targets) else []
    if targets:
        run_levels = targets
    summary = dict(phaseA=lock, floquet=fmap, levels={})
    for I0 in run_levels:
        key = f"{I0:.4f}"; arms = {}
        for arm in ("dual_global", "dual_local", "dual_mixed"):
            recs = [run_formation_arm(lock, I0, arm, s, T=T) for s in SEEDS]
            arms[arm] = dict(metrics=[r["metrics"] for r in recs], period_ms=recs[0]["period_ms"],
                             lambda_perp_max=fmap[key][arm])
            print(f"  [L{key} {arm}] R={np.median([r['metrics']['median_R_phase'] for r in recs]):.2f} "
                  f"occ={np.median([r['metrics']['occupancy'] for r in recs]):.2f}")
        summary["levels"][key] = dict(arms=arms)
    verdict = adjudicate_field_screen(summary, lock)
    summary["verdict"] = verdict
    json.dump(summary, open(os.path.join(OUT, "field_screen_summary.json"), "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"[VERDICT] {verdict['verdict']} | taxonomy={verdict['taxonomy']} | window={verdict['window']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run** — `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/run_topic4_zm_field_screen.py --confirm-run --smoke`
Expected: `[PHASE0] op=...`, a `phaseA_lock.json`, per-level `[floquet ...]` lines, then short nonlinear runs and a `[VERDICT] ...` line. Exit 2 at Phase 0 = the legitimate STOP (report it, do not force).

- [ ] **Step 3: Commit** — `git commit -m "feat(topic4): reduced-field orchestrator (Floquet-first, write-once lock, per-arm resume)"`

---

### Task 10: Plotter + figures README + FIGURE_INDEX

**Files:** Create `scripts/plot_topic4_zm_field_screen.py`; Create `results/topic4_sef_hfo/zm_field_screen/figures/README.md` (AFTER figures exist); Modify `results/FIGURE_INDEX.md`

- [ ] **Step 1:** Read `docs/figure_style_guide.md` (Topic 4 section) BEFORE writing any plotting code; follow it.
- [ ] **Step 2:** Write the plotter — 4 panels, one question each (CLAUDE.md §7): (A) mean-field orbit `(r̄,μ̄,S̄)` + the locked 5 levels on the `I0` segment; (B) `λ_⊥` heatmaps over integer modes, `dual_global` vs `dual_local` (mark `k*` and its angle vs `θ_EE`; DC excluded); (C) level × arm pass grid with the consecutive run highlighted; (D) `r(x)` snapshots + `R_phase(t)` for `dual_global` vs `dual_local` at the central level.
- [ ] **Step 3:** Run it; **eyeball every figure**; fix anything unreadable.
- [ ] **Step 4:** Write `figures/README.md` (Chinese; `### filename`, 2-4 sentences, final `**关注点**：`) — only after the figures exist.
- [ ] **Step 5:** Append one row to `results/FIGURE_INDEX.md`. NOTE: `results/` is gitignored — `git add` the FIGURE_INDEX row (that file IS tracked) but do NOT expect `figures/README.md` to be committable; verify with `git check-ignore`.
- [ ] **Step 6:** Commit code and docs separately.

---

### Task 11: Execute the gate + adjudicate + archive

- [ ] **Step 1:** Real Phase 0 (`--confirm-run`, no `--smoke`). No orbit → STOP, write the NO-GO archive, done.
- [ ] **Step 2:** Floquet map for all 5 levels. If no target window → write the taxonomy verdict and SKIP the expensive sweeps (cheap-first).
- [ ] **Step 3:** If a target window exists: nonlinear formation runs (30 s × 4 seeds × 3 dual arms) for those levels; then phase-reset, 60 s central, `dt/2`, `n=64` confirmations. Monitor RSS/`MemAvailable`/swap every 2-5 min; 2-4 workers max, all BLAS threads pinned to 1.
- [ ] **Step 4:** Adjudicate; generate + eyeball figures.
- [ ] **Step 5:** Write `docs/archive/topic4/sef_hfo/zm_reduced_field_screen_2026-07-25.md`: plain-language 三段式 (测了什么/怎么测/揭示了什么), the arm1-vs-2 (β creates the orbit) and arm2-vs-3/4 (spatial rank) contrasts REPORTED SEPARATELY, the taxonomy verdict, and the forbidden-claims list (§11). Do NOT write "localising the current Z/M+S_G produces a carrier".
- [ ] **Step 6:** Commit; confirm worktree clean, no residual processes, and report the resource peak.

---

## Self-Review notes (resolved inline)
- **Spec coverage:** §0 scope conditional → T1 `test_divisive_only_beta0_has_no_orbit` + T4 `div_global` arm; §1 dual pool + derived `w_frac` → T3/T4; §3 pooling order → T4; §4 kernels → T3; §5 four arms → T4/T9; §6.0 minimal-intervention + contiguity + dt/2 + write-once lock → T2/T9; §6.2 Floquet → T6; §6.3 init → T7/T9; §7 metrics → T5; §8 gate + Floquet-first → T8/T9; §9 taxonomy → T8; §12 engineering (threads, streaming, resume) → T9.
- **Placeholders:** none — every code step is complete runnable code.
- **Type consistency:** `field_metrics` keys used by `topic4_zm_field_verdict._seed_passes` match T5's return dict exactly (`occupancy, P95, active_area_frac, osc_frac, median_R_phase, mean_pair_corr, median_local_period_ms`). `floquet_map` returns `lam_max`, consumed as `lambda_perp_max` in T9's summary rows and read by the adjudicator via `_num(a,"lambda_perp_max")`.
- **Known approximation (documented, not hidden):** the mode-`k` pool linearisation uses `δA_L,k = K̂_σS(k)·Ψ'(r0)·δr_k` (exact for the p-norm at a uniform base state, since `[K∗g]^{1/p}` linearises to `K̂·Ψ'` there). The full-field perturbation growth rate remains the independent sanity check in T11 if linear and nonlinear pictures disagree.
