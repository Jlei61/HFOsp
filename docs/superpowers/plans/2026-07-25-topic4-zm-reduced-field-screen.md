# Reduced 2-D `S_L(x)+S_G` field screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reduced 2-D rate-field screen that tests whether spatially-local inhibition (`S_L(x)`) makes the synchronised burst-train orbit transversally unstable and forms a bounded phase-staggered attractor, where a matched global scalar `S_G` does not.

**Architecture:** A gating Phase-0 uniform 3-state mean-field (`src/topic4_zm_field_meanfield.py`) confirms a synchronised relaxation orbit exists under Fix-A (dual divisive+subtractive pool); only then the 2-D field (`src/topic4_zm_field_screen.py`, Fix-A + anisotropic elliptical-exp `K_E` + per-mode 3×3 Floquet + streaming metrics + 3 arms) is built; an orchestrator (`scripts/run_topic4_zm_field_screen.py`) runs Phase 0 → Phase A (lock 5 `ξ` levels → immutable `phaseA_lock.json`) → Phase B (screen 3 arms × 5 levels × 4 seeds + phase-reset + 60 s central), then adjudicates the 4-cell verdict taxonomy.

**Tech Stack:** Python, numpy, scipy (FFT convolutions, analytic signal), matplotlib (Agg). Reuse `slow_field.psi_recruit/pnorm_pool`, `sef_hfo_field.convolve_periodic/isotropic_gaussian`, `topic4_zm_patch_screen.population_occupancy`.

**Spec:** `docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md` (rev 2, approved).

## Global Constraints

- Reduced rate field ONLY — no SNN, no H/termination, no E→E change.
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1` for every run.
- SNN-inherited pooling constants (verbatim): `r50=0.4, n_psi=2, p_pool=3, τ_μ=30, τ_S=80, S_max=1`.
- Fast-field constants (verbatim): `τ_a=10, a_max=1, u_half=0.5`. `(W0,α,β,θ)` are LOCKED by Phase 0.
- Kernel (verbatim): `K_E` elliptical-exponential `l_EE=0.38 mm, AR=2 → l∥=0.537, l⊥=0.269`, `K_E(0)=0`, `Σ=1`; `K_σS` isotropic Gaussian `σ_S=2.0 mm` primary; `L=20 mm`; `n=32` primary, `n=64` central sensitivity.
- Pre-registered gate thresholds (verbatim, DO NOT tune to results): occupancy ≥ 0.80; `P95 ≥ 0.1·a_max`; `mean P_local ≥ 0.5·mean P_global`; active_area_frac ≥ 0.50; oscillatory fraction (over ALL cells) ≥ 0.50 with `p2p/a_max ≥ 0.20` and ≥ 10 cycles; median `R_phase < 0.50`; pairwise corr < 0.50; local period ∈ `[0.5, 2]×` global period; ≥ 3 CONSECUTIVE of 5 levels each in ≥ 3/4 seeds; global-only control `R_phase ≥ 0.80`; transverse `local λ_⊥(k*)>0 & global λ_⊥<0` with a `dt` vs `dt/2` sign margin.
- Phase 0 no-orbit → STOP (do NOT build the field). `phaseA_lock.json` is immutable; Phase B reads it only.
- Streaming metrics: never hold `full time × 32 × 32 × all-states`; downsample field traces to ~5 ms for figures.
- Verdict language: "synchronised burst-train orbit", never "carrier"; `ξ∈[0,1]` monotone, never "frozen z".

---

### Task 1: Phase-0 mean-field module + orbit detector (the GATE)

**Files:**
- Create: `src/topic4_zm_field_meanfield.py`
- Test: `tests/test_topic4_zm_field_meanfield.py`

**Interfaces:**
- Produces: `F(u, u_half=0.5) -> float`; `psi(r, r50=0.4, n=2.0) -> float`;
  `@dataclass MFParams(W0, alpha, beta, theta, I0, tau_a=10., tau_mu=30., tau_S=80., S_max=1.)`;
  `simulate_meanfield(p: MFParams, T=6000., dt=0.25, r0=0.15) -> np.ndarray  # (nsteps,3) = (r,mu,S)`;
  `detect_orbit(traj, dt, settle=0.5) -> dict(oscillates:bool, depth, trough, peak, period_ms, ncyc)`.
- Orbit definition (locked): `oscillates` iff `ncyc>=4 and depth>0.5 and trough<0.25*peak and peak>0.1`
  (`depth=(peak-trough)/mean`; cycles via upward mid-line crossings; `period_ms = mean(diff(crossings))*dt`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topic4_zm_field_meanfield.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_meanfield import F, psi, MFParams, simulate_meanfield, detect_orbit

def test_F_and_psi_shapes():
    assert F(-1.0) == 0.0 and abs(F(0.5) - 0.5) < 1e-9        # u_half=0.5 -> F(0.5)=0.5
    assert psi(0.0) == 0.0 and 0.0 < psi(0.4) < 1.0

def test_fixA_dual_pool_oscillates():
    # verified Phase-0 point (probe): W0=2, alpha=2, beta=4, I0=1, theta=0.5 -> relaxation orbit
    p = MFParams(W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    o = detect_orbit(simulate_meanfield(p), 0.25)
    assert o["oscillates"] and o["ncyc"] >= 6
    assert o["trough"] < 0.25 * o["peak"] and 100.0 < o["period_ms"] < 300.0

def test_divisive_only_does_not_oscillate():
    # beta=0 -> recurrent-only division -> NO orbit (the Phase-0 no-orbit STOP case)
    p = MFParams(W0=2.0, alpha=16.0, beta=0.0, theta=0.5, I0=1.0)
    assert not detect_orbit(simulate_meanfield(p), 0.25)["oscillates"]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py`
Expected: FAIL (ModuleNotFoundError: src.topic4_zm_field_meanfield)

- [ ] **Step 3: Write minimal implementation**

```python
# src/topic4_zm_field_meanfield.py
"""Phase-0 uniform mean-field (r,mu,S) for the reduced S_L(x)+S_G field (Fix A: divisive alpha*S on the
recurrent term + subtractive beta*S on the membrane). Gates the 2-D field: if no synchronised orbit exists
in the pre-registered grid, STOP (spec 2026-07-24 §6.0). Divisive-only (beta=0) does NOT oscillate."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def F(u, u_half=0.5):
    x = max(float(u), 0.0)
    return x / (u_half + x)

def psi(r, r50=0.4, n=2.0):
    x = max(float(r), 0.0) ** n
    return x / (r50 ** n + x)

@dataclass
class MFParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0

def simulate_meanfield(p: MFParams, T=6000.0, dt=0.25, r0=0.15):
    n = int(round(T / dt)); r, mu, S = r0, 0.0, 0.0
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
    oscillates = bool(cr.size >= 4 and depth > 0.5 and trough < 0.25 * peak and peak > 0.1)
    return dict(oscillates=oscillates, depth=depth, trough=trough, peak=peak,
                period_ms=period_ms, ncyc=int(cr.size))
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_meanfield.py tests/test_topic4_zm_field_meanfield.py
git commit -m "feat(topic4): Phase-0 mean-field gate for the reduced S_L+S_G field (Fix A)"
```

---

### Task 2: Phase-0 continuation + operating-point lock

**Files:**
- Modify: `src/topic4_zm_field_meanfield.py`
- Test: `tests/test_topic4_zm_field_meanfield.py`

**Interfaces:**
- Produces: `meanfield_continuation(grid=None, I0s=None, dt=0.25) -> dict(has_orbit:bool,
  operating_point:dict|None, window:dict|None, n_orbits:int)`. `operating_point` = the oscillatory
  `(W0,alpha,beta,theta,I0)` with the largest `depth` at mid-`I0`; `window` = the `[I0_lo,I0_hi]` span (at
  the locked `W0,alpha,beta,theta`) over which `oscillates` is True. Grid default = spec §6.0:
  `W0∈{2,3,4,6}, alpha∈{1,2,4}, beta∈{1,2,4,8}, theta∈{0.4,0.5,0.6}`, `I0s=np.arange(0.5,2.01,0.1)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_meanfield.py
from src.topic4_zm_field_meanfield import meanfield_continuation

def test_continuation_finds_fixA_window():
    r = meanfield_continuation()
    assert r["has_orbit"] and r["n_orbits"] > 0
    op = r["operating_point"]
    assert op["beta"] > 0 and 100.0 < r["window"]["period_ms"] < 300.0
    assert r["window"]["I0_hi"] > r["window"]["I0_lo"]      # a non-empty xi/I0 window

def test_continuation_no_orbit_grid_stops():
    # a divisive-only grid (beta fixed 0) has no orbit -> has_orbit False (the STOP)
    r = meanfield_continuation(grid=dict(W0=[2,4,8], alpha=[2,8,16], beta=[0.0], theta=[0.5]))
    assert not r["has_orbit"] and r["operating_point"] is None
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py::test_continuation_finds_fixA_window`
Expected: FAIL (AttributeError: meanfield_continuation)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_meanfield.py
import itertools

_DEFAULT_GRID = dict(W0=[2, 3, 4, 6], alpha=[1, 2, 4], beta=[1, 2, 4, 8], theta=[0.4, 0.5, 0.6])

def meanfield_continuation(grid=None, I0s=None, dt=0.25):
    grid = dict(_DEFAULT_GRID if grid is None else grid)
    I0s = np.arange(0.5, 2.01, 0.1) if I0s is None else np.asarray(I0s)
    orbits = []   # (W0,alpha,beta,theta,I0, depth, period_ms)
    for W0, alpha, beta, theta in itertools.product(grid["W0"], grid["alpha"], grid["beta"], grid["theta"]):
        for I0 in I0s:
            o = detect_orbit(simulate_meanfield(MFParams(W0, alpha, beta, theta, float(I0)), dt=dt), dt)
            if o["oscillates"]:
                orbits.append((W0, alpha, beta, theta, float(I0), o["depth"], o["period_ms"]))
    if not orbits:
        return dict(has_orbit=False, operating_point=None, window=None, n_orbits=0)
    # operating point: pick the (W0,alpha,beta,theta) config with the widest I0 window; within it, mid-I0
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for o in orbits:
        by_cfg[o[:4]].append(o)
    cfg = max(by_cfg, key=lambda c: len(by_cfg[c]))
    rows = sorted(by_cfg[cfg], key=lambda o: o[4])
    I0_lo, I0_hi = rows[0][4], rows[-1][4]
    mid = rows[len(rows) // 2]
    op = dict(W0=cfg[0], alpha=cfg[1], beta=cfg[2], theta=cfg[3], I0=mid[4])
    return dict(has_orbit=True, n_orbits=len(orbits), operating_point=op,
                window=dict(I0_lo=I0_lo, I0_hi=I0_hi, period_ms=mid[6]))
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_meanfield.py`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_meanfield.py tests/test_topic4_zm_field_meanfield.py
git commit -m "feat(topic4): Phase-0 mean-field continuation + operating-point lock"
```

---

### Task 3: Anisotropic elliptical-exponential `K_E` kernel

**Files:**
- Create: `src/topic4_zm_field_screen.py`
- Test: `tests/test_topic4_zm_field_screen.py`

**Interfaces:**
- Produces: `elliptical_exp_kernel(n, L, l_par, l_perp, theta) -> (n,n) np.ndarray` — periodic, rotated by
  `theta`, `K[center]=0`, normalised `Σ=1`. Also `gaussian_kernel(n, L, sigma) -> (n,n)` normalised `Σ=1`
  (reuse `sef_hfo_field.isotropic_gaussian` then renormalise).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_topic4_zm_field_screen.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_screen import elliptical_exp_kernel, gaussian_kernel

def test_kernel_normalised_and_self_zero():
    K = elliptical_exp_kernel(32, 20.0, 0.537, 0.269, np.radians(30))
    assert abs(K.sum() - 1.0) < 1e-9
    assert K[0, 0] == 0.0                                    # K_E(0)=0 at the (0,0) offset cell
    assert (gaussian_kernel(32, 20.0, 2.0).sum() - 1.0) < 1e-9

def test_kernel_anisotropy_along_theta():
    # along theta=0 (x-axis) the kernel decays SLOWER than perpendicular (AR=2)
    K = elliptical_exp_kernel(64, 20.0, 0.537, 0.269, 0.0)
    Kc = np.roll(np.roll(K, 32, 0), 32, 1)                  # center it for readability
    row = Kc[32, :]; col = Kc[:, 32]                        # along-x vs along-y through the center
    # width at half-max: x (parallel) wider than y (perpendicular)
    def hwhm(v): 
        v = v / v.max(); return np.sum(v >= 0.5)
    assert hwhm(row) > hwhm(col)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py`
Expected: FAIL (ImportError)

- [ ] **Step 3: Write minimal implementation**

```python
# src/topic4_zm_field_screen.py  (header + kernels)
"""Reduced 2-D S_L(x)+S_G rate field (Fix A: divisive+subtractive dual pool) + anisotropic K_E +
per-2D-mode 3x3 Floquet + streaming metrics + 3 arms (global/local/mixed). Spec 2026-07-24 rev2.
Reduced rate field only -- no SNN, no H, no E->E."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

def _offset_grid(n, L):
    # periodic signed offsets in mm on an n x n lattice, offset (0,0) at index (0,0)
    idx = (np.arange(n) + n // 2) % n - n // 2
    d = idx * (L / n)
    return np.meshgrid(d, d, indexing="ij")   # (DX, DY), DX varies along axis0

def elliptical_exp_kernel(n, L, l_par, l_perp, theta):
    DX, DY = _offset_grid(n, L)
    u = DX * np.cos(theta) + DY * np.sin(theta)     # parallel to theta
    v = -DX * np.sin(theta) + DY * np.cos(theta)    # perpendicular
    K = np.exp(-np.sqrt((u / l_par) ** 2 + (v / l_perp) ** 2))
    K[0, 0] = 0.0                                    # K_E(0)=0 (no self)
    return K / K.sum()

def gaussian_kernel(n, L, sigma):
    DX, DY = _offset_grid(n, L)
    K = np.exp(-0.5 * (DX ** 2 + DY ** 2) / sigma ** 2)
    return K / K.sum()
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_screen.py tests/test_topic4_zm_field_screen.py
git commit -m "feat(topic4): anisotropic elliptical-exp K_E + gaussian pooling kernel"
```

---

### Task 4: Fix-A field step + 3 arms + matched-budget uniform identity

**Files:**
- Modify: `src/topic4_zm_field_screen.py`
- Test: `tests/test_topic4_zm_field_screen.py`

**Interfaces:**
- Produces: `@dataclass FieldParams(n=32, L=20., W0, alpha, beta, theta, I0, w_frac=0.5,
  tau_a=10., tau_mu=30., tau_S=80., S_max=1., r50=0.4, n_psi=2., p_pool=3., sigma_S=2.0,
  l_par=0.537, l_perp=0.269, theta_EE=0.0, eps_G=0.2)`; `w_rec = W0*w_frac`, `w_c = W0*(1-w_frac)`.
- `simulate_field(p, arm, T, dt, seed, r_init=None, state_init=None, record_stride=20) ->
  dict(r_trace (nrec,n,n) float32, t_ms, final_state dict(r,muL,SL,muG,SG))`. `arm ∈ {"global","local","mixed"}`.
- Uses `psi_recruit`/`pnorm_pool` from slow_field (import), FFT convolution with precomputed `K̂_E,K̂_σS`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import FieldParams, simulate_field
from src.topic4_zm_field_meanfield import simulate_meanfield, MFParams

def test_matched_budget_uniform_manifold_identity_across_arms():
    # uniform IC + homogeneous params -> the 3 arms stay identical on the uniform manifold
    p = FieldParams(n=16, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    outs = {arm: simulate_field(p, arm, T=800.0, dt=0.25, seed=0,
                                r_init=np.full((16, 16), 0.15))["final_state"]["r"] for arm in ("global","local","mixed")}
    assert np.allclose(outs["global"], outs["local"], atol=1e-8)
    assert np.allclose(outs["global"], outs["mixed"], atol=1e-8)
    assert np.allclose(outs["global"], outs["global"].mean())   # stays spatially uniform

def test_uniform_field_matches_meanfield():
    p = FieldParams(n=16, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    fr = simulate_field(p, "global", T=1500.0, dt=0.25, seed=0, r_init=np.full((16, 16), 0.15))
    mf = simulate_meanfield(MFParams(2.0, 2.0, 4.0, 0.5, 1.0), T=1500.0, dt=0.25, r0=0.15)
    # the field's spatial-mean r-trace equals the 0-D mean-field r (uniform reduction)
    field_meanr = fr["r_trace"].reshape(fr["r_trace"].shape[0], -1).mean(axis=1)
    mf_r_at_rec = mf[::20, 0][:len(field_meanr)]
    assert np.allclose(field_meanr, mf_r_at_rec, atol=1e-6)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k matched_budget`
Expected: FAIL (ImportError: FieldParams)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
import os, sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "snn_engine"))
from slow_field import psi_recruit, pnorm_pool   # noqa: E402  (reuse SNN pooling nonlinearities)

def _Fsat(U, u_half=0.5):
    X = np.maximum(U, 0.0)
    return X / (u_half + X)

@dataclass
class FieldParams:
    W0: float; alpha: float; beta: float; theta: float; I0: float
    n: int = 32; L: float = 20.0; w_frac: float = 0.5
    tau_a: float = 10.0; tau_mu: float = 30.0; tau_S: float = 80.0; S_max: float = 1.0
    r50: float = 0.4; n_psi: float = 2.0; p_pool: float = 3.0
    sigma_S: float = 2.0; l_par: float = 0.537; l_perp: float = 0.269; theta_EE: float = 0.0
    eps_G: float = 0.2

def simulate_field(p: FieldParams, arm, T=6000.0, dt=0.25, seed=0, r_init=None, state_init=None, record_stride=20):
    rng = np.random.default_rng(seed)
    n = p.n
    KE = np.fft.rfft2(elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.rfft2(gaussian_kernel(n, p.L, p.sigma_S))
    w_rec, w_c = p.W0 * p.w_frac, p.W0 * (1.0 - p.w_frac)
    if state_init is not None:
        r = state_init["r"].copy(); muL = state_init["muL"].copy(); SL = state_init["SL"].copy()
        muG = float(state_init["muG"]); SG = float(state_init["SG"])
    else:
        r = (r_init.copy() if r_init is not None else np.full((n, n), 0.15) + 1e-3 * rng.standard_normal((n, n)))
        muL = np.zeros((n, n)); SL = np.zeros((n, n)); muG = 0.0; SG = 0.0
    r = np.maximum(r, 0.0)
    ns = int(round(T / dt)); rec = []
    for t in range(ns):
        S_eff = SG if arm == "global" else SL if arm == "local" else (1 - p.eps_G) * SL + p.eps_G * SG
        rec_E = w_rec * r + w_c * np.fft.irfft2(np.fft.rfft2(r) * KE, s=(n, n))
        u = p.I0 + rec_E / (1.0 + p.alpha * S_eff) - p.beta * S_eff - p.theta
        r = np.maximum(r + dt * (-r + _Fsat(u, 0.5)) / p.tau_a, 0.0)
        z = psi_recruit(r, 0.0, p.r50, p.n_psi)              # nonlinearity FIRST (per location)
        A_L = np.fft.irfft2(np.fft.rfft2(z ** p.p_pool) * KS, s=(n, n)) ** (1.0 / p.p_pool)   # then local pool
        A_G = pnorm_pool(z, p.p_pool)                        # then global pool
        muL += dt * (-muL + A_L) / p.tau_mu;  SL += dt * (-SL + p.S_max * muL) / p.tau_S
        muG += dt * (-muG + A_G) / p.tau_mu;  SG += dt * (-SG + p.S_max * muG) / p.tau_S
        if t % record_stride == 0:
            rec.append(r.astype(np.float32).copy())
    return dict(r_trace=np.asarray(rec), t_ms=np.arange(len(rec)) * record_stride * dt,
                final_state=dict(r=r, muL=muL, SL=SL, muG=muG, SG=SG))
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k "matched_budget or uniform_field"`
Expected: PASS (2 passed) — NOTE: for the uniform-identity test the fixed seed's `1e-3` IC noise is
overridden by `r_init`, so the field stays uniform.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_screen.py tests/test_topic4_zm_field_screen.py
git commit -m "feat(topic4): Fix-A 2D field step + 3 arms + matched-budget uniform identity"
```

---

### Task 5: Streaming metrics (occupancy / R_phase / active-area / oscillatory-fraction / period)

**Files:**
- Modify: `src/topic4_zm_field_screen.py`
- Test: `tests/test_topic4_zm_field_screen.py`

**Interfaces:**
- Produces: `field_metrics(r_trace, dt_rec_ms, a_max=1.0, settle=0.25) -> dict(occupancy, P95, mean_P,
  active_area_frac, osc_frac, median_R_phase, mean_pair_corr, period_ms)`. Phase per cell via
  **upward mid-line crossings** (relaxation-appropriate); R_phase from linear cycle-interpolated phase.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import field_metrics

def _osc_field(n, nt, phases, amp=0.8, base=0.1, period=40):
    t = np.arange(nt)
    fld = np.empty((nt, n, n))
    for i in range(n):
        for j in range(n):
            fld[:, i, j] = base + amp * (0.5 + 0.5 * np.sign(np.sin(2*np.pi*t/period + phases[i, j])))
    return fld.astype(np.float32)

def test_metrics_inphase_vs_desync():
    n, nt = 8, 400
    inph = _osc_field(n, nt, np.zeros((n, n)))
    rng = np.random.default_rng(0); desy = _osc_field(n, nt, rng.uniform(0, 2*np.pi, (n, n)))
    mi = field_metrics(inph, 5.0); md = field_metrics(desy, 5.0)
    assert mi["median_R_phase"] > 0.8 and md["median_R_phase"] < 0.5
    assert mi["osc_frac"] > 0.9 and mi["active_area_frac"] > 0.9

def test_metrics_plateau_and_tiny_active_set_loopholes():
    n, nt = 8, 400
    plateau = np.full((nt, n, n), 0.8, np.float32)           # high, but no oscillation
    assert field_metrics(plateau, 5.0)["osc_frac"] < 0.1     # osc_frac over ALL cells excludes it
    tiny = np.full((nt, n, n), 0.8, np.float32); tiny[:, 0, 0] = _osc_field(1, nt, np.zeros((1,1)))[:, 0, 0]
    assert field_metrics(tiny, 5.0)["active_area_frac"] < 0.1  # only 1/64 cells active
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k metrics`
Expected: FAIL (ImportError: field_metrics)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
sys_path_patch = None  # (no-op marker)

def _cycle_crossings(x, dt):
    """Upward mid-line crossing indices of a 1-D signal (relaxation-oscillation cycle markers)."""
    mid = 0.5 * (x.max() + x.min())
    return np.flatnonzero((x[:-1] < mid) & (x[1:] >= mid))

def field_metrics(r_trace, dt_rec_ms, a_max=1.0, settle=0.25):
    R = np.asarray(r_trace, float)[int(len(r_trace) * settle):]     # (nt, n, n)
    nt = R.shape[0]; flat = R.reshape(nt, -1)                        # (nt, ncells)
    P = flat.mean(axis=1)
    P95 = float(np.percentile(P, 95))
    occ = float((P >= 0.2 * P95).mean()) if P95 > 1e-12 else 0.0
    # per-cell amplitude / cycles / phase
    amp = flat.max(axis=0) - flat.min(axis=0)
    active = amp >= 0.1 * a_max
    active_area_frac = float(active.mean())
    ncyc = np.array([len(_cycle_crossings(flat[:, c], dt_rec_ms)) for c in range(flat.shape[1])])
    p2p = amp / a_max
    osc_cells = (ncyc >= 10) & (p2p >= 0.20)
    osc_frac = float(osc_cells.mean())                              # denominator = ALL cells
    # R_phase(t): cycle-interpolated phase per active cell, Kuramoto at each t, then median over time
    phases = np.full((nt, flat.shape[1]), np.nan)
    for c in np.flatnonzero(active):
        cr = _cycle_crossings(flat[:, c], dt_rec_ms)
        for a, b in zip(cr[:-1], cr[1:]):
            phases[a:b, c] = 2 * np.pi * (np.arange(a, b) - a) / (b - a)
    Rt = []
    for t in range(nt):
        ph = phases[t][~np.isnan(phases[t])]
        if ph.size >= 2:
            Rt.append(abs(np.mean(np.exp(1j * ph))))
    median_R = float(np.median(Rt)) if Rt else 1.0
    # pairwise correlation across active cells
    act = flat[:, active]
    if act.shape[1] >= 2 and np.all(act.std(axis=0) > 0):
        C = np.corrcoef(act.T); iu = np.triu_indices(act.shape[1], 1); mpc = float(np.nanmean(C[iu]))
    else:
        mpc = 1.0
    # dominant period from the population signal
    crP = _cycle_crossings(P, dt_rec_ms)
    period = float(np.mean(np.diff(crP)) * dt_rec_ms) if crP.size >= 2 else float("nan")
    return dict(occupancy=occ, P95=P95, mean_P=float(P.mean()), active_area_frac=active_area_frac,
                osc_frac=osc_frac, median_R_phase=median_R, mean_pair_corr=mpc, period_ms=period)
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k metrics`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_screen.py tests/test_topic4_zm_field_screen.py
git commit -m "feat(topic4): streaming field metrics (R_phase/active-area/osc-frac/energy/period)"
```

---

### Task 6: Transverse Floquet estimator (per 2-D mode, 3×3 monodromy)

**Files:**
- Modify: `src/topic4_zm_field_screen.py`
- Test: `tests/test_topic4_zm_field_screen.py`

**Interfaces:**
- Produces: `uniform_orbit(p, dt, T=6000., settle=0.5) -> (orbit (period_steps,3), period_ms)` — one clean
  `(r,mu,S)` period of the mean-field; `transverse_floquet(p, arm, kx, ky, orbit, dt) -> lambda_perp` via a
  3×3 monodromy of the mode-`k` variational system; `floquet_map(p, arm, orbit, dt, kmax) -> (KX,KY,LAM)`.
- Mode-`k` linearisation (locked): global arm's pool responds only at `k=0` (so `δS_eff=0` for `k≠0`); local
  responds via `K̂_σS(k)`; recurrent coupling via `K̂_E(k)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import uniform_orbit, transverse_floquet

def test_global_arm_all_modes_decay():
    p = FieldParams(n=32, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    orbit, per = uniform_orbit(p, 0.25)
    lam = [transverse_floquet(p, "global", kx, 0.0, orbit, 0.25) for kx in (0.5, 1.0, 2.0, 3.0)]
    assert all(l < 1e-3 for l in lam)                        # global: transverse modes do not grow

def test_floquet_sign_margin_dt_halving():
    p = FieldParams(n=32, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    o1, _ = uniform_orbit(p, 0.25); o2, _ = uniform_orbit(p, 0.125)
    l1 = transverse_floquet(p, "local", 1.0, 0.0, o1, 0.25)
    l2 = transverse_floquet(p, "local", 1.0, 0.0, o2, 0.125)
    assert np.sign(l1) == np.sign(l2)                        # sign stable under dt halving
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k "floquet or all_modes"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
def uniform_orbit(p: FieldParams, dt, T=6000.0, settle=0.5):
    from src.topic4_zm_field_meanfield import simulate_meanfield, MFParams, detect_orbit
    mf = MFParams(p.W0, p.alpha, p.beta, p.theta, p.I0, p.tau_a, p.tau_mu, p.tau_S, p.S_max)
    tr = simulate_meanfield(mf, T=T, dt=dt)
    o = detect_orbit(tr, dt, settle)
    per = int(round(o["period_ms"] / dt))
    tail = tr[int(len(tr) * settle):]
    return tail[:per].copy(), o["period_ms"]

def transverse_floquet(p: FieldParams, arm, kx, ky, orbit, dt):
    """3x3 monodromy of the mode-k variational system along one uniform orbit. State (dr,dmu,dS)."""
    n, L = p.n, p.L
    KE = np.fft.fft2(elliptical_exp_kernel(n, L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.fft2(gaussian_kernel(n, L, p.sigma_S))
    # nearest lattice-mode indices for (kx,ky) in cycles/mm -> integer mode m = round(k*L)
    mx, my = int(round(kx * L / (2*np.pi))) % n, int(round(ky * L / (2*np.pi))) % n
    kE = float(KE[mx, my].real); kS = float(KS[mx, my].real)
    is_dc = (mx == 0 and my == 0)
    w_rec, w_c = p.W0 * p.w_frac, p.W0 * (1 - p.w_frac)
    Wk = w_rec + w_c * kE                                    # recurrent coupling at this mode
    M = np.eye(3)
    for r0, mu0, S0 in orbit:
        rec = Wk * r0
        denom = 1.0 + p.alpha * S0
        u = p.I0 + rec / denom - p.beta * S0 - p.theta
        Fp = 0.0 if u <= 0 else 0.5 / (0.5 + max(u, 0.0)) ** 2   # F'(u), u_half=0.5
        dF_dr = Fp * Wk / denom
        # local pool responds at every k (kS); global pool ONLY at k=0 (is_dc). mixed: (1-eps)*local + eps*global
        pool_k = kS if arm == "local" else (0.0 if not is_dc else 1.0) if arm == "global" \
            else (1 - p.eps_G) * kS + (p.eps_G * (1.0 if is_dc else 0.0))
        dF_dS = Fp * (-rec * p.alpha / denom**2 - p.beta)
        psip = 0.0 if r0 <= 0 else (p.n_psi * (r0**(p.n_psi-1)) * p.r50**p.n_psi) / (p.r50**p.n_psi + r0**p.n_psi)**2  # d psi/dr = n r^(n-1) r50^n / (r50^n+r^n)^2
        # d(A)/dr at this mode: pool_k * psi'(r0) * (Psi(r0)^(p-1)/Psi^(p-1)) ~ approximate p-norm deriv by psi'
        dA_dr = pool_k * psip
        J = np.array([
            [(-1 + dF_dr) / p.tau_a, 0.0, dF_dS / p.tau_a],
            [dA_dr / p.tau_mu, -1.0 / p.tau_mu, 0.0],
            [0.0, p.S_max / p.tau_S, -1.0 / p.tau_S],
        ])
        M = (np.eye(3) + dt * J) @ M
    rho = np.max(np.abs(np.linalg.eigvals(M)))
    Tper = len(orbit) * dt
    return float(np.log(max(rho, 1e-300)) / Tper)

def floquet_map(p, arm, orbit, dt, kmax=4.0, nk=13):
    ks = np.linspace(-kmax, kmax, nk)
    LAM = np.array([[transverse_floquet(p, arm, kx, ky, orbit, dt) for ky in ks] for kx in ks])
    KX, KY = np.meshgrid(ks, ks, indexing="ij")
    return KX, KY, LAM
```

- [ ] **Step 4: Run to verify PASS** (adjust `_khat` stub — delete the unused `_khat` placeholder before running)

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k "floquet or all_modes"`
Expected: PASS (2 passed). If the local sign is not yet positive anywhere, that is a SCIENTIFIC result to
record in Task 8, not a test failure — the sign-margin test only checks dt-stability of whatever sign occurs.

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_screen.py tests/test_topic4_zm_field_screen.py
git commit -m "feat(topic4): per-mode 3x3 monodromy transverse Floquet estimator"
```

---

### Task 7: Phase-reset + transverse-init helpers

**Files:**
- Modify: `src/topic4_zm_field_screen.py`
- Test: `tests/test_topic4_zm_field_screen.py`

**Interfaces:**
- Produces: `orbit_phasepoint_state(p, orbit, phase_idx) -> dict(r,muL,SL,muG,SG)` (all fields uniform at the
  orbit's `(r,mu,S)` at `phase_idx`); `add_r_perturbation(state, eps, seed, n) -> state` (zero-mean noise on
  `r` only).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import orbit_phasepoint_state, add_r_perturbation

def test_phasepoint_state_uniform_all_fields():
    p = FieldParams(n=8, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    orbit, _ = uniform_orbit(p, 0.25)
    st = orbit_phasepoint_state(p, orbit, phase_idx=len(orbit)//3)
    for k in ("r", "muL", "SL"):
        assert np.allclose(st[k], st[k].flat[0])            # every field spatially uniform
    assert np.isscalar(st["muG"]) and np.isscalar(st["SG"])

def test_r_perturbation_zero_mean_r_only():
    p = FieldParams(n=16, W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0)
    orbit, _ = uniform_orbit(p, 0.25)
    st0 = orbit_phasepoint_state(p, orbit, 5)
    st1 = add_r_perturbation({k: (v.copy() if hasattr(v,"copy") else v) for k,v in st0.items()}, 1e-4, 0, 16)
    assert abs((st1["r"] - st0["r"]).mean()) < 1e-12        # zero-mean
    assert np.allclose(st1["SL"], st0["SL"])                # only r perturbed
```

- [ ] **Step 2: Run to verify FAIL**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k "phasepoint or perturbation"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Add the implementation**

```python
# append to src/topic4_zm_field_screen.py
def orbit_phasepoint_state(p: FieldParams, orbit, phase_idx):
    r0, mu0, S0 = orbit[int(phase_idx) % len(orbit)]
    o = np.ones((p.n, p.n))
    return dict(r=r0 * o, muL=mu0 * o, SL=S0 * o, muG=float(mu0), SG=float(S0))

def add_r_perturbation(state, eps, seed, n):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, n)); d -= d.mean()          # zero-mean
    amp = float(state["r"].max()) if np.ndim(state["r"]) else 1.0
    state["r"] = state["r"] + eps * amp * d
    return state
```

- [ ] **Step 4: Run to verify PASS**

Run: `OMP_NUM_THREADS=1 python -m pytest -q tests/test_topic4_zm_field_screen.py -k "phasepoint or perturbation"`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_zm_field_screen.py tests/test_topic4_zm_field_screen.py
git commit -m "feat(topic4): full-state phase-reset + zero-mean r-perturbation helpers"
```

---

### Task 8: Orchestrator — Phase 0 → A (lock) → B (screen) + provenance

**Files:**
- Create: `scripts/run_topic4_zm_field_screen.py`
- Test: manual smoke (tiny grid) — the acceptance verdict is a science output, not a unit test.

**Interfaces:**
- Consumes: everything above. Produces `results/topic4_sef_hfo/zm_field_screen/{phaseA_lock.json,
  field_screen_seed_summary.json}` + figures dir.

- [ ] **Step 1: Write the orchestrator**

```python
# scripts/run_topic4_zm_field_screen.py
#!/usr/bin/env python
"""Phase 0 (mean-field gate) -> Phase A (lock 5 xi levels -> immutable phaseA_lock.json) -> Phase B
(screen global/local/mixed x 5 levels x 4 seeds + phase-reset + transverse Floquet; central level 60s +
dt/2 + n=64). Adjudicates the 4-cell verdict (spec 2026-07-24 rev2 §8-§9). Reduced field only. --confirm-run."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys
import numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.topic4_zm_field_meanfield import meanfield_continuation
from src.topic4_zm_field_screen import (FieldParams, simulate_field, field_metrics, uniform_orbit,
                                        floquet_map, orbit_phasepoint_state, add_r_perturbation)
OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_field_screen")
SEEDS = (0, 1, 2, 3); DT = 0.25; N = 32
# locked thresholds (spec Global Constraints)
TH = dict(occ=0.80, area=0.50, osc=0.50, p2p=0.20, Rlt=0.50, corr=0.50, Rglob=0.80, p_lo=0.5, p_hi=2.0)

def _sha(path):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest(); return h

def phase0():
    r = meanfield_continuation()
    if not r["has_orbit"]:
        print("[PHASE0] NO synchronised orbit in the pre-registered grid -> STOP (do not build the field).")
        sys.exit(2)
    print(f"[PHASE0] orbit found: op={r['operating_point']} window={r['window']}")
    return r["operating_point"], r["window"]

def lock_levels(op, window):
    I0s = np.linspace(window["I0_lo"], window["I0_hi"], 5).round(4).tolist()   # 5 xi/I0 levels in the window
    lock = dict(spec_sha=_sha(os.path.join(_ROOT, "docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md")),
                operating_point=op, window=window, I0_levels=I0s, seeds=list(SEEDS), dt=DT, grid_n=N,
                git_sha=subprocess.check_output(["git","-C",_ROOT,"rev-parse","HEAD"],text=True).strip(),
                created=datetime.datetime.now().isoformat(timespec="seconds"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(lock, open(os.path.join(OUT, "phaseA_lock.json"), "w"), indent=2)
    return lock

def _params(op, I0, n=N):
    return FieldParams(n=n, W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=I0)

def screen_level(op, I0, arm, T=30000.0, dt=DT, n=N):
    passes = 0; rows = []
    p = _params(op, I0, n)
    orbit, per = uniform_orbit(p, dt)
    for seed in SEEDS:
        st = add_r_perturbation(orbit_phasepoint_state(p, orbit, phase_idx=len(orbit)//3), 1e-4, seed, n)
        out = simulate_field(p, arm, T=T, dt=dt, seed=seed, state_init=st, record_stride=int(round(5.0/dt)))
        m = field_metrics(out["r_trace"], record_ms := 5.0)
        rows.append(m)
    return rows, per

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm-run", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run: raise SystemExit("need --confirm-run")
    op, window = phase0()
    lock = lock_levels(op, window)
    # Phase B: global-only period per level (matched period reference) + Floquet + nonlinear screen
    T = 3000.0 if a.smoke else 30000.0; n = 16 if a.smoke else N
    summary = dict(phaseA=lock, levels={})
    for I0 in lock["I0_levels"]:
        p = _params(op, I0, n); orbit, _ = uniform_orbit(p, DT)
        lam = {arm: floquet_map(p, arm, orbit, DT)[2].max() for arm in ("global", "local", "mixed")}
        arms = {}
        for arm in ("global", "local", "mixed"):
            rows, per = screen_level(op, I0, arm, T=T, n=n)
            arms[arm] = dict(metrics=rows, period_ms=per, lambda_perp_max=lam[arm])
        summary["levels"][round(I0, 4)] = arms
        print(f"[level I0={I0:.3f}] lam_perp max global={lam['global']:.4f} local={lam['local']:.4f} "
              f"mixed={lam['mixed']:.4f}")
    json.dump(summary, open(os.path.join(OUT, "field_screen_seed_summary.json"), "w"), indent=2,
              default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"[done] {OUT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (tiny grid)**

Run: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/run_topic4_zm_field_screen.py --confirm-run --smoke`
Expected: prints `[PHASE0] orbit found ...` then per-level `lam_perp` lines; writes `phaseA_lock.json` + `field_screen_seed_summary.json`. If Phase 0 prints NO orbit and exits 2, that is the legitimate STOP — report it, do not force.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_topic4_zm_field_screen.py
git commit -m "feat(topic4): reduced-field screen orchestrator (Phase 0 -> A lock -> B)"
```

---

### Task 9: Adjudicator + plotter + README + FIGURE_INDEX

**Files:**
- Create: `scripts/plot_topic4_zm_field_screen.py`
- Create: `results/topic4_sef_hfo/zm_field_screen/figures/README.md` (AFTER figures exist)
- Modify: `results/FIGURE_INDEX.md`

- [ ] **Step 1: Adjudication function** (append to the runner or a small module) — apply the §8 Phase-B gate
  to `field_screen_seed_summary.json`: per level, `arm=local/mixed` passes iff all of {occ≥0.80,
  active_area_frac≥0.50, osc_frac≥0.50, median_R_phase<0.50, mean_pair_corr<0.50, period∈[0.5,2]×global,
  energy floors} in ≥3/4 seeds; GO iff ≥3 CONSECUTIVE levels pass AND `global λ_⊥<0 & local/mixed λ_⊥>0`.
  Emit the 4-cell taxonomy verdict.

- [ ] **Step 2: Plotter** — 4 panels: (A) mean-field orbit `(r,mu,S)`; (B) `λ_⊥(kx,ky)` heatmaps
  global vs local (+ most-unstable direction vs `θ_EE`); (C) per-level pass grid (arm × level, colored by
  criteria met) with the consecutive-run highlighted; (D) representative `r(x)` snapshots + `P(t)` +
  `R_phase(t)` for the global vs local arm at the central level.

- [ ] **Step 3: Write `figures/README.md`** (Chinese, `### filename` + 2-4 sentences + `**关注点**`) and append a `results/FIGURE_INDEX.md` row.

- [ ] **Step 4: Commit** (code) then (docs) separately.

---

### Task 10: Execute the gate + adjudicate + archive

- [ ] **Step 1:** Run Phase 0 for real (`--confirm-run`, no `--smoke`). If NO orbit → STOP, write the NO-GO to the archive, done. If orbit → continue.
- [ ] **Step 2:** Run the full screen (30 s primary; central level 60 s + `dt/2` + `n=64`). Monitor RAM/threads.
- [ ] **Step 3:** Adjudicate the 4-cell verdict; generate figures; eyeball them.
- [ ] **Step 4:** Write the archive section (`docs/archive/topic4/sef_hfo/zm_reduced_field_screen_<date>.md`) + update memory. If GO → the SNN migration is a SEPARATE next plan (out of scope here).
- [ ] **Step 5:** Commit results/docs; confirm worktree clean + no residual processes.

---

## Self-Review notes (resolved inline)
- **Spec coverage:** Phase 0 (T1-2), dual pool + arms (T4), pooling order (T4), anisotropic `K_E` (T3), matched budget uniform identity (T4), transverse Floquet 3×3 (T6), full-state phase-reset (T7), metrics incl. loopholes (T5), Phase-A lock + provenance (T8), 4-cell verdict (T9), run+archive (T10). `σ_S`/`n=64`/`dt/2` sensitivities live in T8/T10.
- **Floquet caveat:** `transverse_floquet` uses a nearest-lattice-mode `K̂` and a `psi'`-based `dA/dr` (a first-order estimate); the spec marks the full-field growth-rate as the independent sanity check (add in T10 if the linear and nonlinear pictures disagree). Delete the `_khat` stub in T6 before running.
