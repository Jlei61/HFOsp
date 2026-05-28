# SEF-ITP Phase 4 Stage 2 (2D Homogeneous Excitable Sheet) Implementation Plan — **v1**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a 2D homogeneous (θ≡0) Hindmarsh-Rose excitable sheet with anisotropic diffusion + spatial-OU noise, find the parameter regime where brief spatially-localized propagation events occur ("local-event" regime), and hand off a locked (D₀, D_x/D_y\*, σ\*, I\*) cell to Stage 3.

**Architecture:** Each node runs the Stage-1 HR dynamics; nodes couple only through diffusion of the x variable. Time stepping is Strang operator splitting — Crank-Nicolson half-step on the (linear, anisotropic) diffusion of x, a full RK4 step on the (local, nonlinear) HR reaction + OU noise, then a second CN diffusion half-step. The anisotropic Laplacian is a single scipy.sparse matrix factorized once per simulation and reused every step. Node events are detected with the **Stage-1b burst-envelope unit** (`detect_burst_envelopes`) per node; a population regime classifier groups co-occurring node-envelope onsets into spatial events.

**Tech Stack:** Python 3.11, numpy, scipy.sparse (+ scipy.sparse.linalg.splu, scipy.ndimage.gaussian_filter), matplotlib, pytest. Reuses `src/topic4_modeling/{hr_core,hr_config,hr_dynamics}.py` from Stage 1/1b.

**Spec source:** `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.3 §3 Stage 2 + §5.2 (aniso Laplacian) + §5.3 (spatial OU).

**Upstream hand-off (locked):**
- Per-node baseline: I\*=1.0, r\*=0.006, σ\*=0.4 (`stage1_results_2026-05-28.md`).
- Event unit = node **burst envelope**, onset = first spike start, `envelope_gap=30` (`stage1b_results_2026-05-28.md`).

**Plan boundary:** Stage 2 **smoke (80×60) only**. High-resolution replay (200×150) is a *separate* plan, written only after smoke locks a representative cell (see Exit Contract).

---

## Locked decisions (resolve forks before coding; rationale one-line each)

1. **Numerical scheme = CN-diffusion + RK4-reaction (Strang split). No explicit-Euler fallback.** HR fires fast spikes; explicit diffusion stability `dt·D·(2/Δx²+2/Δy²)<1` is violated at dt=0.05 for D₀≳1, exactly during spikes. CN avoids shrinking dt 10–40× and avoids a "smoke used a different scheme" caveat. (advisor 2026-05-28)
2. **Absolute diffusion scale D₀ is an explicit calibration target (Task 5), not a hidden constant.** Spec pins only the *ratio* {1,2,3,5}. D₀ (node²/HR-time) sets propagation speed/coupling; lock it by matching a physical-ish traveling speed before sweeping the ratio. Working in grid units (Δ=1 node), `D_x = D₀·ratio_x`, `D_y = D₀·ratio_y`.
3. **Group-event window W = envelope_gap = 30 HR units** (same merge threshold as Stage 1b — keeps single-node and multi-node temporal grouping consistent). "local-event" regime = group events with ≥3 participating nodes AND ≤20% of grid nodes per event (averaged over T).
4. **Diffusion acts on x only** (spec §5.2); y, z are purely local. OU noise η enters the reaction RHS of x.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/topic4_modeling/grid2d.py` | `Grid2DConfig`, `build_aniso_laplacian`, `factorize_diffusion`, spatial-OU field, `GridState`, `step_grid`, `simulate_grid` |
| `src/topic4_modeling/regime.py` | population event extraction (`extract_group_events`) + `classify_population_regime` |
| `src/topic4_modeling/grid_viz.py` | x(t) heatmap movie frames, node raster, population mean trace, propagation-speed map, regime map |
| `scripts/run_topic4_phase4_stage2_grid.py` | CLI: D₀ calibration + smoke regime sweep + viz + JSON outputs + exit-code contract |
| `tests/test_topic4_modeling_grid2d.py` | Laplacian / solver / OU-field / stepper unit tests |
| `tests/test_topic4_modeling_grid2d_integration.py` | `@pytest.mark.slow` empirical: silent-stays-silent, single-bump propagates-then-decays |
| `tests/test_topic4_modeling_regime.py` | population classifier on synthetic onset patterns |
| `tests/test_topic4_modeling_stage2_cli.py` | CLI help + exit-contract (synthetic no-local-event → exit 1) |
| `docs/archive/topic4/sef_itp_phase4_v1/stage2_results_<date>.md` | results archive (Task 7) |
| `results/topic4_sef_itp/phase4_modeling/stage2_2d_homogeneous/` | outputs (Task 6/7) |

**Convention notes:** tests flat in `tests/`, `test_*.py`; frozen dataclass for config; empirical regime-behavior tests go in `_integration.py` with `pytest.xfail` on boundary mismatch (Stage-1 v3 discipline — never red on an empirical boundary). Reuse `detect_burst_envelopes` per node (do NOT reinvent node event detection).

---

## Task 1: Anisotropic sparse Laplacian + factorized diffusion solver

**Files:**
- Create: `src/topic4_modeling/grid2d.py`
- Create: `tests/test_topic4_modeling_grid2d.py`

**Math (spec §5.2, grid units Δ=1):** `A = D_x·Lxx + D_y·Lyy`, where `Lxx`,`Lyy` are 1D second-difference operators with **Neumann (no-flux)** boundaries (boundary node's stencil drops the missing neighbor: diagonal −1 instead of −2 there). Flatten index `k = j*nx + i` (j=y row, i=x col).

- [ ] **Step 1: Write failing tests**

```python
"""Tests for src/topic4_modeling/grid2d.py (Laplacian + diffusion solver)."""
from __future__ import annotations
import numpy as np
import pytest


def test_aniso_laplacian_isotropic_recovers_standard():
    """D_x=D_y=1 reproduces the standard 5-point Laplacian on the interior."""
    from src.topic4_modeling.grid2d import Grid2DConfig, build_aniso_laplacian
    cfg = Grid2DConfig(nx=5, ny=4)
    A = build_aniso_laplacian(cfg, D_x=1.0, D_y=1.0).toarray()
    field = np.arange(cfg.ny * cfg.nx, dtype=float).reshape(cfg.ny, cfg.nx)
    lap = (A @ field.ravel()).reshape(cfg.ny, cfg.nx)
    # interior node (1,1): x[i+1]+x[i-1]+x[j+1]+x[j-1]-4x = standard
    i, j = 1, 1
    expected = (field[j, i+1] + field[j, i-1] + field[j+1, i] + field[j-1, i]
                - 4 * field[j, i])
    assert lap[j, i] == pytest.approx(expected)


def test_neumann_boundary_constant_field_zero_laplacian():
    """No-flux BC: a spatially constant field has zero Laplacian everywhere
    (including boundary/corner nodes)."""
    from src.topic4_modeling.grid2d import Grid2DConfig, build_aniso_laplacian
    cfg = Grid2DConfig(nx=6, ny=5)
    A = build_aniso_laplacian(cfg, D_x=2.0, D_y=3.0)
    const = np.full(cfg.ny * cfg.nx, 7.0)
    np.testing.assert_allclose(A @ const, 0.0, atol=1e-12)


def test_aniso_laplacian_anisotropy_scales_axes():
    """D_x scales the x-second-difference, D_y the y-second-difference."""
    from src.topic4_modeling.grid2d import Grid2DConfig, build_aniso_laplacian
    cfg = Grid2DConfig(nx=5, ny=5)
    Ax = build_aniso_laplacian(cfg, D_x=3.0, D_y=0.0).toarray()
    # pure-x ramp varying only along i: y-differences are 0, so only D_x term
    field = np.tile(np.arange(cfg.nx, dtype=float), (cfg.ny, 1))
    lap = (Ax @ field.ravel()).reshape(cfg.ny, cfg.nx)
    # linear ramp → interior 2nd difference 0; check a curved row instead
    field2 = np.tile((np.arange(cfg.nx, dtype=float))**2, (cfg.ny, 1))
    lap2 = (Ax @ field2.ravel()).reshape(cfg.ny, cfg.nx)
    # second difference of i^2 is 2; times D_x=3 → 6 on interior
    assert lap2[2, 2] == pytest.approx(6.0)


def test_factorized_diffusion_solver_inverts():
    """splu factorization of (I - 0.5*dt*A) solves the CN system, matching a
    dense solve on a tiny grid. Regression guard if anyone re-optimizes."""
    from src.topic4_modeling.grid2d import (Grid2DConfig, build_aniso_laplacian,
                                            factorize_diffusion)
    import scipy.sparse as sp
    cfg = Grid2DConfig(nx=4, ny=3)
    A = build_aniso_laplacian(cfg, D_x=1.0, D_y=1.5)
    dt = 0.05
    N = cfg.ny * cfg.nx
    M = sp.eye(N) - 0.25 * dt * A          # half-step CN LHS (dt/4 → see Task 3)
    lu = factorize_diffusion(A, dt)
    v = np.random.default_rng(0).standard_normal(N)
    np.testing.assert_allclose(lu.solve(v), np.linalg.solve(M.toarray(), v),
                               rtol=1e-9, atol=1e-9)
```

- [ ] **Step 2: Run, verify FAIL** — `pytest tests/test_topic4_modeling_grid2d.py -v` → ImportError.

- [ ] **Step 3: Implement Grid2DConfig + Laplacian + factorization**

```python
"""2D homogeneous excitable sheet: anisotropic diffusion + spatial-OU noise.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 2,
§5.2 (aniso Laplacian), §5.3 (spatial OU). Diffusion acts on x only; y,z local.
Numerics: Strang split — CN diffusion half-step / RK4 reaction / CN half-step.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class Grid2DConfig:
    nx: int = 80            # smoke grid (spec: 80x60)
    ny: int = 60
    dx_mm: float = 0.4      # physical node spacing (smoke)
    tau_ou: float = 10.0    # OU temporal constant (HR units)
    lambda_eta: float = 1.5  # OU spatial correlation length (grid units)


def _laplacian_1d(n: int) -> sp.csr_matrix:
    """1D second-difference with Neumann (no-flux) boundaries (grid units)."""
    main = np.full(n, -2.0)
    main[0] = main[-1] = -1.0          # no-flux: drop missing neighbor
    off = np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def build_aniso_laplacian(cfg: Grid2DConfig, D_x: float, D_y: float) -> sp.csc_matrix:
    """A = D_x·(Ix ⊗ Lxx... ) anisotropic 5-point Laplacian, Neumann BC.

    Flatten k = j*nx + i. Lxx acts within a row (x), Lyy across rows (y).
    """
    Lx = _laplacian_1d(cfg.nx)
    Ly = _laplacian_1d(cfg.ny)
    Ix = sp.identity(cfg.nx)
    Iy = sp.identity(cfg.ny)
    A = D_x * sp.kron(Iy, Lx) + D_y * sp.kron(Ly, Ix)
    return A.tocsc()


def factorize_diffusion(A: sp.spmatrix, dt: float):
    """Pre-factor the CN half-step LHS M = (I - dt/4 · A); reuse every step.

    Strang half-step of size dt/2 with CN uses (I - (dt/2)/2 · A) = (I - dt/4·A).
    """
    N = A.shape[0]
    M = sp.identity(N, format="csc") - 0.25 * dt * A
    return splu(M.tocsc())
```

- [ ] **Step 4: Run, verify PASS** (4 tests).

- [ ] **Step 5: Commit** — `git add src/topic4_modeling/grid2d.py tests/test_topic4_modeling_grid2d.py && git commit -m "feat(topic4 phase4 stage2): anisotropic sparse Laplacian + CN diffusion factorization"`

---

## Task 2: Spatial-OU noise field

**Files:** Modify `src/topic4_modeling/grid2d.py`; add tests to `tests/test_topic4_modeling_grid2d.py`.

**Math (spec §5.3):** `dη/dt = −η/τ + σ·ξ`, `ξ = Gaussian_conv(white, λ_η)`. Spatial smoothing reduces variance by Σg² (g = normalized Gaussian kernel); renormalize so per-node stationary variance = σ². Exact discrete OU in time (as Stage-1 `ou_noise`), applied per node.

- [ ] **Step 1: Write failing tests**

```python
def test_spatial_ou_correlation_length():
    """Spatial autocorrelation of a single OU field frame decays on ~lambda_eta."""
    from src.topic4_modeling.grid2d import Grid2DConfig, SpatialOUField
    cfg = Grid2DConfig(nx=64, ny=64, lambda_eta=3.0)
    ou = SpatialOUField(cfg, sigma=0.2, dt=0.05, seed=0)
    # advance to stationary, take a frame
    for _ in range(500):
        frame = ou.step()
    f = frame - frame.mean()
    # 1D autocorr along x at lag = lambda_eta should be well below 1, > at lag 0
    ac0 = (f * f).mean()
    acL = (f[:, :-3] * f[:, 3:]).mean()
    assert 0.0 < acL / ac0 < 0.9


def test_spatial_ou_stationary_variance():
    """Per-node stationary variance ≈ sigma² after spatial-norm correction."""
    from src.topic4_modeling.grid2d import Grid2DConfig, SpatialOUField
    cfg = Grid2DConfig(nx=48, ny=48, lambda_eta=1.5)
    ou = SpatialOUField(cfg, sigma=0.2, dt=0.05, seed=1)
    frames = [ou.step() for _ in range(4000)]
    arr = np.stack(frames[1000:])           # drop burn-in
    assert np.var(arr) == pytest.approx(0.2**2, rel=0.25)


def test_spatial_ou_reproducibility_and_zero_sigma():
    from src.topic4_modeling.grid2d import Grid2DConfig, SpatialOUField
    cfg = Grid2DConfig(nx=16, ny=16)
    a = [SpatialOUField(cfg, 0.2, 0.05, seed=7).step() for _ in range(5)]
    b = [SpatialOUField(cfg, 0.2, 0.05, seed=7).step() for _ in range(5)]
    np.testing.assert_array_equal(a[-1], b[-1])
    z = SpatialOUField(cfg, 0.0, 0.05, seed=7)
    np.testing.assert_array_equal(z.step(), np.zeros((cfg.ny, cfg.nx)))
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement SpatialOUField**

```python
class SpatialOUField:
    """Per-node Ornstein-Uhlenbeck noise with spatial Gaussian correlation.

    eta[t+dt] = eta[t]·decay + noise_scale · normalized_spatial_white
    where decay = exp(-dt/tau), noise_scale = sigma·sqrt(1-decay²), and the
    spatial white field is Gaussian-smoothed (sigma=lambda_eta grid units) then
    renormalized to unit variance so per-node stationary variance = sigma².
    """
    def __init__(self, cfg: Grid2DConfig, sigma: float, dt: float, seed: int):
        self.cfg = cfg
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)
        self._decay = float(np.exp(-dt / cfg.tau_ou))
        self._scale = float(sigma * np.sqrt(1.0 - self._decay**2))
        self._eta = np.zeros((cfg.ny, cfg.nx))
        # variance-reduction factor of gaussian_filter: filter a unit-variance
        # white field once and measure std, invert it (exact for the same mode).
        if sigma > 0.0:
            probe = self._rng.standard_normal((cfg.ny, cfg.nx))
            sm = gaussian_filter(probe, cfg.lambda_eta, mode="reflect")
            self._norm = 1.0 / float(sm.std())
        else:
            self._norm = 0.0

    def step(self) -> np.ndarray:
        if self.sigma == 0.0:
            return self._eta            # stays zeros
        white = self._rng.standard_normal((self.cfg.ny, self.cfg.nx))
        xi = gaussian_filter(white, self.cfg.lambda_eta, mode="reflect") * self._norm
        self._eta = self._eta * self._decay + self._scale * xi
        return self._eta
```

- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** — `feat(topic4 phase4 stage2): spatial-OU noise field`

---

## Task 3: Grid time stepper (Strang split CN + RK4)

**Files:** Modify `src/topic4_modeling/grid2d.py`; add unit test + create `tests/test_topic4_modeling_grid2d_integration.py`.

- [ ] **Step 1: Write unit + integration tests**

Unit (`tests/test_topic4_modeling_grid2d.py`):
```python
def test_simulate_grid_shapes_and_finite():
    from src.topic4_modeling.grid2d import Grid2DConfig, simulate_grid
    from src.topic4_modeling.hr_core import HRParams
    cfg = Grid2DConfig(nx=12, ny=10)
    out = simulate_grid(cfg, HRParams(), I=1.0, D0=1.0, ratio_x=1.0, ratio_y=1.0,
                        sigma=0.0, T=5.0, dt=0.05, seed=0, store_history=True)
    n = int(5.0 / 0.05)
    assert out["x_history"].shape == (n, cfg.ny, cfg.nx)
    assert np.isfinite(out["x_history"]).all()


def test_simulate_grid_constant_diffusion_invariant():
    """A spatially uniform rest field with sigma=0 stays uniform (diffusion of a
    constant is 0); no spurious spatial structure appears."""
    from src.topic4_modeling.grid2d import Grid2DConfig, simulate_grid
    from src.topic4_modeling.hr_core import HRParams
    cfg = Grid2DConfig(nx=12, ny=10)
    out = simulate_grid(cfg, HRParams(), I=1.0, D0=2.0, ratio_x=3.0, ratio_y=1.0,
                        sigma=0.0, T=20.0, dt=0.05, seed=0, store_history=True)
    last = out["x_history"][-1]
    assert last.std() < 1e-6      # remained spatially uniform
```

Integration (`tests/test_topic4_modeling_grid2d_integration.py`, slow, xfail on boundary):
```python
"""Slow empirical Stage-2 grid observations. xfail (not fail) on boundary
mismatch — report in archive, do NOT tune the model to satisfy a hard test."""
from __future__ import annotations
import numpy as np
import pytest


@pytest.mark.slow
def test_silent_grid_stays_silent_no_noise():
    from src.topic4_modeling.grid2d import Grid2DConfig, simulate_grid
    from src.topic4_modeling.hr_core import HRParams
    cfg = Grid2DConfig(nx=40, ny=30)
    out = simulate_grid(cfg, HRParams(), I=1.0, D0=1.0, ratio_x=1.0, ratio_y=1.0,
                        sigma=0.0, T=100.0, dt=0.05, seed=0, store_history=True,
                        burn_in=100.0)
    xmax = float(out["x_history"][:, :, :].max())
    if xmax >= 0.5:
        pytest.xfail(f"Expected silent grid (x.max<0.5) at sigma=0, got {xmax:.3f}")


@pytest.mark.slow
def test_single_seeded_bump_propagates_then_decays():
    """A localized supra-threshold seed launches a brief spatial excursion that
    spreads to neighbors then returns to rest (excitable wavelet, not sustained)."""
    from src.topic4_modeling.grid2d import Grid2DConfig, simulate_grid
    from src.topic4_modeling.hr_core import HRParams
    cfg = Grid2DConfig(nx=40, ny=30)
    seed_xy = (cfg.ny // 2, cfg.nx // 2)
    out = simulate_grid(cfg, HRParams(), I=1.0, D0=1.0, ratio_x=1.0, ratio_y=1.0,
                        sigma=0.0, T=120.0, dt=0.05, seed=0, store_history=True,
                        seed_bump=seed_xy, burn_in=100.0)
    xh = out["x_history"]
    # a neighbor a few nodes away should rise above threshold at some point
    jn, inb = seed_xy[0], seed_xy[1] + 3
    neighbor_fired = bool((xh[:, jn, inb] > 1.0).any())
    final_rest = bool(xh[-1].max() < 0.5)
    if not (neighbor_fired and final_rest):
        pytest.xfail(f"propagate-then-decay not observed at D0=1: "
                     f"neighbor_fired={neighbor_fired} final_rest={final_rest}")
```

- [ ] **Step 2: Run unit tests, verify FAIL.**

- [ ] **Step 3: Implement GridState + step_grid + simulate_grid**

```python
def _hr_reaction_rhs(x, y, z, p, I, eta):
    """Elementwise HR reaction (no diffusion). Arrays in, arrays out."""
    dx = y - p.a * x**3 + p.b * x**2 - z + I + eta
    dy = p.c - p.d * x**2 - y
    dz = p.r * (p.s * (x - p.x_R) - z)
    return dx, dy, dz


def _rk4_reaction(x, y, z, p, I, eta, dt):
    k1 = _hr_reaction_rhs(x, y, z, p, I, eta)
    k2 = _hr_reaction_rhs(x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], z + 0.5*dt*k1[2], p, I, eta)
    k3 = _hr_reaction_rhs(x + 0.5*dt*k2[0], y + 0.5*dt*k2[1], z + 0.5*dt*k2[2], p, I, eta)
    k4 = _hr_reaction_rhs(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2], p, I, eta)
    x2 = x + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y2 = y + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    z2 = z + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return x2, y2, z2


def simulate_grid(cfg, params, *, I, D0, ratio_x, ratio_y, sigma, T, dt, seed,
                  store_history=False, seed_bump=None, burn_in=0.0):
    """Strang split: CN-diffusion(dt/2) ∘ RK4-reaction(dt) ∘ CN-diffusion(dt/2).

    Diffusion acts on x only. Returns dict with x_history (if store_history) and
    always the final state. seed_bump=(j,i) sets one node supra-threshold at t=0
    (excitable wavelet probe, used by D0 calibration + integration test).
    """
    ny, nx, N = cfg.ny, cfg.nx, cfg.ny * cfg.nx
    A = build_aniso_laplacian(cfg, D_x=D0*ratio_x, D_y=D0*ratio_y)
    lu = factorize_diffusion(A, dt)
    rhs_half = sp.identity(N, format="csc") + 0.25 * dt * A   # CN RHS (I + dt/4·A)
    p = params

    # initial state: HR rest (Stage-1 init), optional supra-threshold seed
    x = np.full((ny, nx), -1.6); y = np.full((ny, nx), -10.0); z = np.full((ny, nx), 2.0)
    if seed_bump is not None:
        x[seed_bump] = 1.5
    ou = SpatialOUField(cfg, sigma, dt, seed)

    n_steps = int(T / dt); burn_n = int(burn_in / dt)
    hist = np.empty((n_steps, ny, nx)) if store_history else None

    def _diffuse_half(xf):
        return lu.solve(rhs_half @ xf.ravel()).reshape(ny, nx)

    for t in range(burn_n + n_steps):
        eta = ou.step()
        x = _diffuse_half(x)
        x, y, z = _rk4_reaction(x, y, z, p, I, eta, dt)
        x = _diffuse_half(x)
        if store_history and t >= burn_n:
            hist[t - burn_n] = x
    out = {"x_final": x, "y_final": y, "z_final": z, "dt": dt, "T": T}
    if store_history:
        out["x_history"] = hist
    return out
```

- [ ] **Step 4: Run unit tests, verify PASS.** Integration tests run in Task 6 smoke (slow).
- [ ] **Step 5: Commit** — `feat(topic4 phase4 stage2): grid time stepper (Strang CN+RK4)`

---

## Task 4: Population event extraction + regime classifier

**Files:** Create `src/topic4_modeling/regime.py`; create `tests/test_topic4_modeling_regime.py`.

**Contract (decision 3):** per-node burst-envelope onsets (reuse `detect_burst_envelopes`) → **group events** = onsets within window W=`envelope_gap` (30) of each other; classify the simulation:
- **silent**: ~0 node onsets.
- **scattered**: node onsets exist but do not form ≥3-node groups (no spatial co-activation).
- **local-event**: group events exist with ≥3 nodes AND mean participating-node fraction ≤ 20% of grid. ← target
- **synchronous-burst**: group events recruit > ~80% of grid simultaneously.
- **traveling-pattern**: sustained activity (active-node fraction stays high across most of T, never returns to rest).

- [ ] **Step 1: Write failing tests**

```python
"""Population regime classifier on synthetic per-node onset patterns."""
from __future__ import annotations
import numpy as np
import pytest


def _onsets_to_node_envelopes(onset_map):
    """helper: dict {(j,i): [onset_times]} → list per node for the classifier."""
    return onset_map


def test_classify_population_silent():
    from src.topic4_modeling.regime import classify_population_regime
    # no node has any onset
    node_onsets = {}
    assert classify_population_regime(node_onsets, n_nodes=1000, T=500.0,
                                      window=30.0) == "silent"


def test_classify_population_scattered():
    """Onsets exist but never ≥3 nodes within one window → scattered."""
    from src.topic4_modeling.regime import classify_population_regime
    node_onsets = {(0, 0): [10.0], (5, 5): [200.0], (9, 9): [400.0]}
    assert classify_population_regime(node_onsets, n_nodes=1000, T=500.0,
                                      window=30.0) == "scattered"


def test_classify_population_local_event():
    """A few spatial clusters of ≥3 co-active nodes, each ≤20% of grid."""
    from src.topic4_modeling.regime import classify_population_regime
    # 5 nodes fire within 30u around t=100 and again around t=300; grid=1000
    node_onsets = {(j, j): [100.0 + j, 300.0 + j] for j in range(5)}
    assert classify_population_regime(node_onsets, n_nodes=1000, T=500.0,
                                      window=30.0) == "local-event"


def test_classify_population_synchronous():
    """≥80% of grid co-active within one window → synchronous-burst."""
    from src.topic4_modeling.regime import classify_population_regime
    node_onsets = {(0, k): [100.0 + 0.01 * k] for k in range(90)}
    assert classify_population_regime(node_onsets, n_nodes=100, T=500.0,
                                      window=30.0) == "synchronous-burst"
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement regime.py**

```python
"""Stage 2/4 population event extraction + regime classifier.

A node event is a burst envelope (Stage 1b unit). A group event bundles node
onsets that fall within `window` (= envelope_gap) of each other. The regime is
classified from group-event participation statistics.
Spec: §3 Stage 2 (regime classes) + plan decision 3.
"""
from __future__ import annotations
import numpy as np

from .hr_config import BurstConfig
from .hr_dynamics import detect_burst_envelopes


def extract_node_onsets(x_history, dt, cfg: BurstConfig | None = None) -> dict:
    """Per-node burst-envelope onsets from an (n_steps, ny, nx) x history.

    Reuses detect_burst_envelopes on each node's trace (Stage 1b unit).
    Returns {(j, i): [onset_times]}.
    """
    cfg = cfg or BurstConfig()
    n, ny, nx = x_history.shape
    t = np.arange(n) * dt
    onsets: dict[tuple[int, int], list[float]] = {}
    for j in range(ny):
        for i in range(nx):
            envs = detect_burst_envelopes(x_history[:, j, i], t, cfg)
            if envs:
                onsets[(j, i)] = [e.onset for e in envs]
    return onsets


def extract_group_events(node_onsets: dict, window: float) -> list[dict]:
    """Cluster all node onsets into group events by a sliding time window.

    Sort all (time, node) onsets; greedily group those within `window` of the
    group's first onset. Returns list of {t0, nodes:set, size}.
    """
    flat = sorted((tt, node) for node, ts in node_onsets.items() for tt in ts)
    groups: list[dict] = []
    for tt, node in flat:
        if groups and (tt - groups[-1]["t0"]) <= window:
            groups[-1]["nodes"].add(node)
        else:
            groups.append({"t0": tt, "nodes": {node}})
    for g in groups:
        g["size"] = len(g["nodes"])
    return groups


def classify_population_regime(node_onsets: dict, n_nodes: int, T: float,
                               window: float,
                               local_max_frac: float = 0.20,
                               sync_min_frac: float = 0.80,
                               min_group_nodes: int = 3) -> str:
    """silent / scattered / local-event / synchronous-burst / traveling-pattern."""
    total_onsets = sum(len(v) for v in node_onsets.values())
    if total_onsets == 0:
        return "silent"
    groups = extract_group_events(node_onsets, window)
    big = [g for g in groups if g["size"] >= min_group_nodes]
    if not big:
        return "scattered"
    fracs = np.array([g["size"] / n_nodes for g in big])
    # traveling: a large fraction of nodes active across most of the recording
    active_nodes = len(node_onsets) / n_nodes
    if active_nodes > sync_min_frac and len(groups) > 0.5 * T / window:
        return "traveling-pattern"
    if fracs.max() >= sync_min_frac:
        return "synchronous-burst"
    if fracs.mean() <= local_max_frac:
        return "local-event"
    return "synchronous-burst"
```

- [ ] **Step 4: Run, verify PASS.** (test name `test_regime_classifier_population` covered by the 4 tests above — rename one to that exact name per spec §6.3.)
- [ ] **Step 5: Commit** — `feat(topic4 phase4 stage2): population event extraction + regime classifier`

---

## Task 5: D₀ absolute-scale calibration (Stage-2 "Task 0" intent)

**Files:** Modify `scripts/run_topic4_phase4_stage2_grid.py` (calibration mode); helper in `grid2d.py`.

**Why:** the regime sweep over the ratio is uninterpretable on top of an undetermined absolute D₀. Lock D₀ first by matching a physical-ish propagation speed.

- [ ] **Step 1: Add `measure_propagation_speed(out, cfg, seed_xy)` to grid2d.py**

```python
def measure_propagation_speed(out, cfg, seed_xy) -> float:
    """Speed (mm/ms-proxy) of the excitable wavelet from a single seed bump:
    first-crossing time vs radial distance, robust linear fit. Returns mm per
    HR-time-unit (caller maps HR units → ms when interpreting)."""
    xh = out["x_history"]; dt = out["dt"]
    n, ny, nx = xh.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((jj - seed_xy[0])**2 + (ii - seed_xy[1])**2) * cfg.dx_mm
    first = np.full((ny, nx), np.nan)
    crossed = (xh > 1.0)
    for j in range(ny):
        for i in range(nx):
            idx = np.argmax(crossed[:, j, i]) if crossed[:, j, i].any() else -1
            if idx > 0:
                first[j, i] = idx * dt
    m = np.isfinite(first) & (dist > 0)
    if m.sum() < 10:
        return float("nan")
    slope = np.polyfit(first[m], dist[m], 1)[0]   # mm per HR time unit
    return float(slope)
```

- [ ] **Step 2: Calibration loop in CLI `--mode calibrate`** — scan `D0 ∈ {0.25, 0.5, 1, 2, 4}` at ratio=1, I=1.0, σ=0, single center seed bump, T=120, store_history; measure speed; pick D0 whose speed lands in a target band (document the literature target ~0.05–0.1 mm/ms after HR-unit→ms mapping; if no ms mapping is justified, lock D0 by the *qualitative* criterion "wavelet reaches grid edge in ~half of T, not instantly and not stalling"). Write `d0_calibration.json` with per-D0 speed + chosen D0 + rationale.

- [ ] **Step 3: Commit** — `feat(topic4 phase4 stage2): D0 propagation-speed calibration`

---

## Task 6: Regime sweep + viz + CLI (smoke)

**Files:** Modify `scripts/run_topic4_phase4_stage2_grid.py`; create `src/topic4_modeling/grid_viz.py`; create `tests/test_topic4_modeling_stage2_cli.py`.

- [ ] **Step 1: Implement grid_viz.py** — `plot_regime_map(df)` (D_x/D_y × σ heatmap, faceted by I), `plot_x_heatmap_frames(x_history, n_frames)` (movie frames montage), `plot_node_raster(node_onsets, cfg)`, `plot_population_mean(x_history)`, `plot_speed_map(first_crossing)`. Each returns a matplotlib Figure (Agg backend). (Paper-grade self-contained per the figures memory: shared legend, tight axes, no internal codenames in axis labels.)

- [ ] **Step 2: Implement sweep + CLI** — `--mode smoke`: at locked D0, sweep `ratio ∈ {1,2,3,5}` × `σ ∈ {0.2,0.4,0.6}` × `I ∈ {1.0}` (extend if smoke shows all-silent/all-sync), per cell run `simulate_grid` (store_history for ≤5 representative cells, else stream), `extract_node_onsets`, `classify_population_regime`; build regime DataFrame; pick a local-event cell as Stage-3 baseline. Outputs: `regime_map_aniso_sweep.png`, `regime_summary.json` (config + regime counts + chosen cell + exit-contract flag), `representative_movies/`, `node_raster.png`, `population_mean.png`, `speed_map.png`. **Exit code 1** if no connected local-event regime found (spec Stage 2 exit contract); plot-failure handling mirrors Stage 1 CLI.

- [ ] **Step 3: CLI tests** (`tests/test_topic4_modeling_stage2_cli.py`): `--help` returns 0; a `--mode synthetic-nolocal` (tiny grid forced silent/sync) → exit 1 + `stage2_exit_contract_passed=false` (mirror Stage 1 exit-1 test).

- [ ] **Step 4: Run smoke** `python scripts/run_topic4_phase4_stage2_grid.py --mode calibrate` then `--mode smoke`; run slow integration tests; **eyeball every figure** before archiving (AGENTS.md figures-first).

- [ ] **Step 5: Commit** — `feat(topic4 phase4 stage2): regime sweep + viz + CLI (smoke)`

---

## Task 7: Archive results (Chinese 三段式)

**Files:** Create `docs/archive/topic4/sef_itp_phase4_v1/stage2_results_<date>.md` + `results/.../stage2_2d_homogeneous/figures/README.md`.

- [ ] **Step 1:** invoke `hfosp-plain-language-recap`; write 三段式 (测了什么/怎么测的/揭示了什么) + regime map readout + locked (D₀, ratio\*, σ\*, I\*) + the propagation-speed calibration + caveats (CN scheme, grid-unit D, classifier thresholds are operational). Figures README per AGENTS.md (### filename, 2–4 句, **关注点**：).

- [ ] **Step 2:** advisor consult on the Stage 2 verdict over-claim (spec discipline: advisor between stages). Commit archive.

---

## Exit contract (spec §3 Stage 2)

- [ ] ≥1 connected-component local-event regime exists in the (ratio, σ, I) sweep at locked D₀.
- [ ] A representative (D₀, ratio\*, σ\*, I\*) cell is locked as Stage 3 baseline.
- [ ] regime_map + regime_summary.json + ≤5 representative movies + raster + speed map written; every figure eyeballed.
- [ ] All unit tests GREEN; slow integration tests run (xfail allowed on empirical boundary).
- [ ] **Replay handoff (NOT in this plan):** if smoke locks a representative cell, write a separate `2026-..-..-topic4-phase4-stage2-replay.md` plan to re-run that cell at 200×150 and confirm the regime label doesn't drift. Do NOT run 200×150 here.

## Failure modes (pre-registered, spec §8)

| failure | action |
|---|---|
| no local-event regime in smoke | adjust λ_η or OU τ (spec §8); if still none → FHN sensitivity; if still none → stop-and-review |
| local-event regime < 5% of swept cells | regime not robust; stop-and-review (don't widen thresholds to manufacture it) |
| CN solve unstable / NaN | check Laplacian sign convention + dt; do NOT silently switch to explicit |
| sweep wall-clock-bound | profile `detect_burst_envelopes` per node first; `peak_x` (`x[window].max()`) is the cheapest cut for the population classifier (needs onsets only) |

## Self-review (writing-plans checklist)

- **Spec coverage:** §3 Stage 2 entry/impl/exit + §5.2 aniso Laplacian + §5.3 spatial OU + §6.3 grid2d test names (aniso_laplacian_isotropic_recovers_standard ✓, neumann_boundary ✓, spatial_ou_correlation_length ✓, regime_classifier_population ✓) + factorized-solver guard (added) all mapped.
- **Placeholders:** none — code shown for Laplacian/OU/stepper/classifier/speed; thresholds explicit.
- **Type consistency:** `Grid2DConfig`, `simulate_grid(... store_history, seed_bump, burn_in)`, `detect_burst_envelopes` (Stage 1b), `BurstConfig.envelope_gap` consistent across tasks.
- **Out of scope (carried from spec §9):** θ heterogeneity (Stage 5), K_long (OFF), 200×150 replay (separate plan), shaft/LFP observation (Stage 3).
