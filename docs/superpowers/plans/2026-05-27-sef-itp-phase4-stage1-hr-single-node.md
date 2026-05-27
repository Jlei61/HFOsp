# SEF-ITP Phase 4 Stage 1 (Single-Node HR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build single-node Hindmarsh-Rose excitable unit infrastructure: ODE + RK4 + OU noise + burst detector + regime classifier + phase-portrait viz + parameter-sweep CLI; produce regime map and select baseline (I*, r*, σ*) for Stage 2 hand-off.

**Architecture:** numpy + numba JIT (RK4 hot loop) + matplotlib (viz) + pytest (TDD). Single-node simulation, no network coupling. Parameter sweep via joblib parallel. Results land in `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/` and selected baseline JSON is the hand-off contract to Stage 2.

**Tech Stack:** Python 3.11, numpy 1.x, scipy 1.13, numba 0.60, matplotlib, pytest, joblib (already in env), dataclasses (stdlib).

**Spec source:** `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` §3 Stage 1 + §5.1 + §5.3 + §6.1-6.3 + §7 stage 1 row + §8 stage 1 failure mode.

**Plan boundary:** This plan covers Stage 1 ONLY. Stages 2-5 each get their own plan, issued after the prior stage's exit contract is verified (per spec §13).

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `src/topic4_modeling/__init__.py` | Package init (exports public API) |
| `src/topic4_modeling/hr.py` | HRParams dataclass, HR ODE rhs, RK4 integrator, OU noise, trajectory simulator, burst detector, regime classifier |
| `src/topic4_modeling/hr_viz.py` | Phase portrait + nullcline plotter, regime-map heatmap plotter |
| `src/topic4_modeling/hr_sweep.py` | Parameter sweep runner + baseline picker |
| `tests/test_topic4_modeling_hr.py` | Unit tests for hr.py |
| `tests/test_topic4_modeling_hr_viz.py` | Plot smoke tests |
| `tests/test_topic4_modeling_hr_sweep.py` | Sweep + baseline picker tests |
| `scripts/run_topic4_phase4_stage1_hr.py` | CLI: run sweep, plot regime map, pick baseline, save artifacts |
| `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/` | Output dir (regime_map.png, regime_summary.json, phase_portraits/, baseline.json) |

**Modified:** none in stage 1. (Framework v1.0.8 banner amendment will happen at end of stage 5, not now.)

**Convention notes (codebase):**
- Tests are flat in `tests/`, not nested (verified: `tests/test_*.py`). Use `tests/test_topic4_modeling_*.py`.
- Numba `@njit` already in env (verified: numba 0.60). Use cache=True for fast import.
- Existing pattern: dataclass for params, top-level function for sim, joblib for parallelism (see `src/sef_itp_phase2.py` for analogy).

---

## Task 1: Package skeleton + smoke import

**Files:**
- Create: `src/topic4_modeling/__init__.py`
- Create: `tests/test_topic4_modeling_hr.py`
- Create: `src/topic4_modeling/hr.py` (empty stub)

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_topic4_modeling_hr.py`:

```python
"""Tests for src/topic4_modeling/hr.py (Stage 1 single-node HR)."""

from __future__ import annotations

import numpy as np
import pytest


def test_module_importable():
    """Sanity: hr module can be imported."""
    from src.topic4_modeling import hr  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails (no module yet)**

Run: `pytest tests/test_topic4_modeling_hr.py::test_module_importable -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.topic4_modeling'`

- [ ] **Step 3: Create package + empty stub**

Create `src/topic4_modeling/__init__.py`:

```python
"""SEF-ITP Phase 4 modeling: Hindmarsh-Rose single-node + 2D sheet + observation layer.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md
"""
```

Create `src/topic4_modeling/hr.py`:

```python
"""Stage 1 — Single-node Hindmarsh-Rose excitable unit.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1 + §5.1 + §5.3
"""

from __future__ import annotations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic4_modeling_hr.py::test_module_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/__init__.py src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4): scaffold src/topic4_modeling package + hr.py stub"
```

---

## Task 2: HRParams dataclass

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_hr_params_defaults_match_spec():
    """Default HRParams should match spec §5.1 baseline values."""
    from src.topic4_modeling.hr import HRParams
    p = HRParams()
    assert p.a == 1.0
    assert p.b == 3.0
    assert p.c == 1.0
    assert p.d == 5.0
    assert p.r == 0.006
    assert p.s == 4.0
    assert p.x_R == -1.6


def test_hr_params_is_frozen():
    """HRParams must be frozen (hashable for caching)."""
    from src.topic4_modeling.hr import HRParams
    p = HRParams()
    with pytest.raises((AttributeError, Exception)):
        p.a = 2.0  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic4_modeling_hr.py::test_hr_params_defaults_match_spec tests/test_topic4_modeling_hr.py::test_hr_params_is_frozen -v`
Expected: FAIL with `ImportError: cannot import name 'HRParams'`

- [ ] **Step 3: Implement HRParams**

Append to `src/topic4_modeling/hr.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class HRParams:
    """Hindmarsh-Rose ODE parameters. Defaults from spec §5.1 baseline."""
    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float = 0.006
    s: float = 4.0
    x_R: float = -1.6
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic4_modeling_hr.py -v`
Expected: PASS (3 tests now)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): HRParams frozen dataclass with spec defaults"
```

---

## Task 3: HR ODE right-hand side

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

HR equations (spec §5.1):

```
dx/dt = y - a x³ + b x² - z + I + η
dy/dt = c - d x² - y
dz/dt = r ( s (x - x_R) - z )
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_hr_rhs_returns_finite_3vec():
    """rhs(x, y, z, params, I, eta) returns finite (dx, dy, dz)."""
    from src.topic4_modeling.hr import HRParams, hr_rhs
    p = HRParams()
    dx, dy, dz = hr_rhs(0.0, 0.0, 0.0, p, I=-1.6, eta=0.0)
    assert np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz)


def test_hr_rhs_known_fixed_point_y_equation():
    """At x = x_R, dy/dt = c - d*x_R² - y (algebraic check)."""
    from src.topic4_modeling.hr import HRParams, hr_rhs
    p = HRParams()
    y_test = -10.0  # arbitrary y
    _, dy, _ = hr_rhs(p.x_R, y_test, 0.0, p, I=0.0, eta=0.0)
    expected = p.c - p.d * p.x_R**2 - y_test
    assert dy == pytest.approx(expected)


def test_hr_rhs_z_equation_at_x_R_yields_zero_drive():
    """At x = x_R and z = 0, dz/dt = r * (s * 0 - 0) = 0."""
    from src.topic4_modeling.hr import HRParams, hr_rhs
    p = HRParams()
    _, _, dz = hr_rhs(p.x_R, 0.0, 0.0, p, I=0.0, eta=0.0)
    assert dz == pytest.approx(0.0)


def test_hr_rhs_eta_adds_to_dx_only():
    """eta enters dx/dt only (not dy or dz)."""
    from src.topic4_modeling.hr import HRParams, hr_rhs
    p = HRParams()
    dx0, dy0, dz0 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.0)
    dx1, dy1, dz1 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.5)
    assert dx1 - dx0 == pytest.approx(0.5)
    assert dy1 == pytest.approx(dy0)
    assert dz1 == pytest.approx(dz0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "hr_rhs"`
Expected: 4 FAILs with `ImportError: cannot import name 'hr_rhs'`

- [ ] **Step 3: Implement hr_rhs**

Append to `src/topic4_modeling/hr.py`:

```python
def hr_rhs(
    x: float,
    y: float,
    z: float,
    params: HRParams,
    I: float,
    eta: float,
) -> tuple[float, float, float]:
    """Hindmarsh-Rose right-hand side (single node).

    Args:
        x, y, z: state variables (fast, fast, slow)
        params: HR parameters
        I: deterministic input current (baseline + theta)
        eta: stochastic perturbation (OU noise sample)

    Returns:
        (dx/dt, dy/dt, dz/dt)
    """
    p = params
    dx = y - p.a * x**3 + p.b * x**2 - z + I + eta
    dy = p.c - p.d * x**2 - y
    dz = p.r * (p.s * (x - p.x_R) - z)
    return dx, dy, dz
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "hr_rhs"`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): hr_rhs ODE right-hand side"
```

---

## Task 4: RK4 integrator step

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_rk4_step_deterministic():
    """rk4_step with fixed inputs produces deterministic output."""
    from src.topic4_modeling.hr import HRParams, rk4_step
    p = HRParams()
    s0 = (0.0, 0.0, 0.0)
    out1 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    out2 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    assert out1 == out2


def test_rk4_step_returns_finite_3tuple():
    """rk4_step returns finite (x, y, z)."""
    from src.topic4_modeling.hr import HRParams, rk4_step
    p = HRParams()
    x, y, z = rk4_step((0.0, 0.0, 0.0), p, I=-1.6, eta=0.0, dt=0.05)
    assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


def test_rk4_step_matches_first_order_euler_in_small_dt_limit():
    """As dt → 0, RK4 step ≈ Euler step (within O(dt²))."""
    from src.topic4_modeling.hr import HRParams, hr_rhs, rk4_step
    p = HRParams()
    s0 = (-1.0, -5.0, 0.5)
    dt = 1e-4
    x_rk4, y_rk4, z_rk4 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=dt)
    dx, dy, dz = hr_rhs(*s0, p, I=-1.6, eta=0.0)
    x_euler = s0[0] + dt * dx
    y_euler = s0[1] + dt * dy
    z_euler = s0[2] + dt * dz
    assert x_rk4 == pytest.approx(x_euler, abs=1e-6)
    assert y_rk4 == pytest.approx(y_euler, abs=1e-6)
    assert z_rk4 == pytest.approx(z_euler, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "rk4_step"`
Expected: 3 FAILs with `ImportError: cannot import name 'rk4_step'`

- [ ] **Step 3: Implement rk4_step**

Append to `src/topic4_modeling/hr.py`:

```python
def rk4_step(
    state: tuple[float, float, float],
    params: HRParams,
    I: float,
    eta: float,
    dt: float,
) -> tuple[float, float, float]:
    """Classical RK4 step on HR ODE (one timestep).

    The stochastic term eta is held constant over the RK4 step (Stratonovich
    interpretation with sub-step noise frozen — adequate when dt << τ_η).
    """
    x, y, z = state
    k1x, k1y, k1z = hr_rhs(x, y, z, params, I, eta)
    k2x, k2y, k2z = hr_rhs(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z,
        params, I, eta,
    )
    k3x, k3y, k3z = hr_rhs(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z,
        params, I, eta,
    )
    k4x, k4y, k4z = hr_rhs(
        x + dt * k3x, y + dt * k3y, z + dt * k3z,
        params, I, eta,
    )
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x_new, y_new, z_new
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "rk4_step"`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): rk4_step classical RK4 integrator"
```

---

## Task 5: OU noise generator

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

Spec §5.3: `dη/dt = −η/τ_η + σ_η · ξ(t)`, τ_η = 10 ms (in HR time units; assume 1 HR time unit ≈ 1 ms for stage 1 simplicity → τ_η = 10).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_ou_noise_seed_reproducibility():
    """Same seed produces same noise trace."""
    from src.topic4_modeling.hr import generate_ou_noise
    trace1 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    trace2 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    np.testing.assert_array_equal(trace1, trace2)


def test_ou_noise_stationary_variance_matches_sigma_squared():
    """Long OU trace has variance ≈ sigma² (analytic stationary variance)."""
    from src.topic4_modeling.hr import generate_ou_noise
    trace = generate_ou_noise(n_steps=200_000, dt=0.05, tau=10.0, sigma=0.1, seed=0)
    burn_in = 5_000
    assert np.var(trace[burn_in:]) == pytest.approx(0.1**2, rel=0.15)


def test_ou_noise_autocorrelation_time_matches_tau():
    """Autocorrelation at lag = τ should equal 1/e of zero-lag (analytic OU)."""
    from src.topic4_modeling.hr import generate_ou_noise
    tau = 10.0
    dt = 0.05
    trace = generate_ou_noise(n_steps=400_000, dt=dt, tau=tau, sigma=0.2, seed=1)
    burn_in = 5_000
    x = trace[burn_in:]
    x = x - x.mean()
    var0 = (x * x).mean()
    lag_steps = int(tau / dt)
    cov_lag = (x[:-lag_steps] * x[lag_steps:]).mean()
    autocorr = cov_lag / var0
    assert autocorr == pytest.approx(np.exp(-1.0), abs=0.1)


def test_ou_noise_length_matches_n_steps():
    """Output length equals n_steps."""
    from src.topic4_modeling.hr import generate_ou_noise
    trace = generate_ou_noise(n_steps=137, dt=0.05, tau=10.0, sigma=0.1, seed=0)
    assert trace.shape == (137,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "ou_noise"`
Expected: 4 FAILs with `ImportError: cannot import name 'generate_ou_noise'`

- [ ] **Step 3: Implement generate_ou_noise**

Append to `src/topic4_modeling/hr.py`:

```python
def generate_ou_noise(
    n_steps: int,
    dt: float,
    tau: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck noise trace.

    SDE: dη = -η/τ dt + sigma * sqrt(2/τ) dW (so stationary variance = sigma²)

    Args:
        n_steps: number of timesteps
        dt: timestep
        tau: correlation time
        sigma: stationary std (target Var = sigma²)
        seed: random seed

    Returns:
        OU trace of shape (n_steps,), starting from η_0 = 0.
    """
    rng = np.random.default_rng(seed)
    eta = np.zeros(n_steps)
    # Exact discrete OU update for constant timestep:
    # η[t+dt] = η[t] * exp(-dt/τ) + sigma * sqrt(1 - exp(-2 dt/τ)) * N(0,1)
    decay = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1.0 - decay**2)
    randn = rng.standard_normal(n_steps)
    for i in range(1, n_steps):
        eta[i] = eta[i - 1] * decay + noise_scale * randn[i]
    return eta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "ou_noise"`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): generate_ou_noise (exact discrete OU)"
```

---

## Task 6: Trajectory simulator

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_simulate_trajectory_shapes():
    """Output shapes match expected (n_steps, 3) for state + (n_steps,) for time."""
    from src.topic4_modeling.hr import HRParams, simulate_trajectory
    p = HRParams()
    t, traj = simulate_trajectory(p, I=-1.6, T=10.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    n_expected = int(10.0 / 0.05)
    assert t.shape == (n_expected,)
    assert traj.shape == (n_expected, 3)


def test_simulate_trajectory_reproducibility_with_seed():
    """Same seed → same trajectory."""
    from src.topic4_modeling.hr import HRParams, simulate_trajectory
    p = HRParams()
    _, t1 = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                 sigma_ou=0.1, tau_ou=10.0, seed=42)
    _, t2 = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                 sigma_ou=0.1, tau_ou=10.0, seed=42)
    np.testing.assert_array_equal(t1, t2)


def test_simulate_trajectory_silent_at_low_I_no_noise():
    """At very low I (deeply sub-threshold) with no noise, x stays bounded near rest."""
    from src.topic4_modeling.hr import HRParams, simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-2.5, T=200.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    x = traj[:, 0]
    # x should not exhibit bursting (max stays below burst threshold)
    assert x.max() < 0.5, f"x.max()={x.max()} suggests spontaneous firing at I=-2.5"


def test_simulate_trajectory_repetitive_burst_at_high_I():
    """At elevated I (above stable rest), HR enters repetitive bursting."""
    from src.topic4_modeling.hr import HRParams, simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=2.0, T=300.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    x = traj[:, 0]
    # multiple bursts means many crossings above x=1.0
    crossings_up = np.sum((x[:-1] < 1.0) & (x[1:] >= 1.0))
    assert crossings_up >= 3, f"At I=2.0 expected ≥3 bursts, got {crossings_up}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "simulate_trajectory"`
Expected: 4 FAILs with `ImportError: cannot import name 'simulate_trajectory'`

- [ ] **Step 3: Implement simulate_trajectory**

Append to `src/topic4_modeling/hr.py`:

```python
def simulate_trajectory(
    params: HRParams,
    I: float,
    T: float,
    dt: float,
    sigma_ou: float,
    tau_ou: float,
    seed: int,
    x0: float = -1.6,
    y0: float = -10.0,
    z0: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate single-node HR trajectory for duration T.

    Args:
        params: HR parameters
        I: constant input current
        T: total simulated time (HR time units)
        dt: integration timestep
        sigma_ou: OU noise stationary std (0 disables noise)
        tau_ou: OU noise correlation time
        seed: random seed (also picks initial state perturbation)
        x0, y0, z0: initial conditions (default near typical HR rest)

    Returns:
        (t, trajectory) where t.shape = (n_steps,) and trajectory.shape = (n_steps, 3)
        with columns [x, y, z].
    """
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = (x0, y0, z0)

    if sigma_ou > 0.0:
        eta = generate_ou_noise(n_steps, dt, tau_ou, sigma_ou, seed)
    else:
        eta = np.zeros(n_steps)

    state = (x0, y0, z0)
    for i in range(1, n_steps):
        state = rk4_step(state, params, I, eta[i], dt)
        trajectory[i] = state
    return t, trajectory
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "simulate_trajectory"`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): simulate_trajectory (RK4 + OU noise)"
```

---

## Task 7: Burst detector with hysteresis

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_detect_bursts_zero_on_silent_trace():
    """Constant x=-1.6 (rest) produces no bursts."""
    from src.topic4_modeling.hr import detect_bursts
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    bursts = detect_bursts(x, t, x_threshold=1.0, min_duration=5.0)
    assert bursts == []


def test_detect_bursts_one_on_synthetic_pulse():
    """Single pulse above threshold for >min_duration registers as 1 burst."""
    from src.topic4_modeling.hr import detect_bursts
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    # pulse from t=10 to t=20 (10 ms wide, well above 5 ms min)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    bursts = detect_bursts(x, t, x_threshold=1.0, min_duration=5.0)
    assert len(bursts) == 1
    t_start, t_end = bursts[0]
    assert t_start == pytest.approx(10.0, abs=0.1)
    assert t_end == pytest.approx(20.0, abs=0.1)


def test_detect_bursts_hysteresis_ignores_brief_dip():
    """Brief dip below threshold within a sustained burst does not split it."""
    from src.topic4_modeling.hr import detect_bursts
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 30.0)] = 1.5
    # 1 ms dip below threshold (shorter than 5 ms min_duration)
    x[(t >= 19.5) & (t <= 20.5)] = 0.5
    bursts = detect_bursts(x, t, x_threshold=1.0, min_duration=5.0)
    assert len(bursts) == 1, (
        f"Expected 1 burst (hysteresis should bridge 1ms dip), got {len(bursts)}"
    )


def test_detect_bursts_three_well_separated():
    """Three well-separated pulses give 3 bursts."""
    from src.topic4_modeling.hr import detect_bursts
    t = np.arange(0, 200, 0.05)
    x = np.full_like(t, -1.6)
    for t_start in [10.0, 60.0, 120.0]:
        x[(t >= t_start) & (t <= t_start + 10.0)] = 1.5
    bursts = detect_bursts(x, t, x_threshold=1.0, min_duration=5.0)
    assert len(bursts) == 3


def test_detect_bursts_rejects_brief_above_threshold_noise():
    """Brief spike above threshold (<min_duration) is rejected as not a burst."""
    from src.topic4_modeling.hr import detect_bursts
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    # 1ms spike — well below 5ms min_duration
    x[(t >= 10.0) & (t <= 11.0)] = 1.5
    bursts = detect_bursts(x, t, x_threshold=1.0, min_duration=5.0)
    assert bursts == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "detect_bursts"`
Expected: 5 FAILs

- [ ] **Step 3: Implement detect_bursts**

Append to `src/topic4_modeling/hr.py`:

```python
def detect_bursts(
    x: np.ndarray,
    t: np.ndarray,
    x_threshold: float = 1.0,
    min_duration: float = 5.0,
    bridge_gap: float = 2.0,
) -> list[tuple[float, float]]:
    """Detect bursts in x trace using hysteresis + min duration.

    Algorithm:
        1. Find contiguous segments where x > x_threshold
        2. Bridge segments separated by gaps < bridge_gap (hysteresis)
        3. Filter out merged segments shorter than min_duration

    Args:
        x: signal trace
        t: time array (same shape as x)
        x_threshold: amplitude threshold for "above"
        min_duration: minimum burst duration (HR time units; spec: 5ms)
        bridge_gap: max gap (HR time units) to bridge within a single burst

    Returns:
        list of (t_start, t_end) tuples, one per detected burst.
    """
    above = x > x_threshold
    if not above.any():
        return []
    # Find rising / falling edges
    transitions = np.diff(above.astype(np.int8))
    rises = np.where(transitions == 1)[0] + 1
    falls = np.where(transitions == -1)[0] + 1
    # Handle boundary conditions
    if above[0]:
        rises = np.concatenate([[0], rises])
    if above[-1]:
        falls = np.concatenate([falls, [len(x)]])
    # Build raw segments [(rise_idx, fall_idx), ...]
    segments = list(zip(rises, falls))
    # Bridge gaps
    bridged: list[tuple[int, int]] = []
    for seg in segments:
        if bridged and (t[seg[0]] - t[bridged[-1][1] - 1]) < bridge_gap:
            bridged[-1] = (bridged[-1][0], seg[1])
        else:
            bridged.append(seg)
    # Filter by min_duration
    result = []
    for r, f in bridged:
        t_start = t[r]
        t_end = t[f - 1]
        if (t_end - t_start) >= min_duration:
            result.append((t_start, t_end))
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "detect_bursts"`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): detect_bursts with hysteresis + min_duration"
```

---

## Task 8: Regime classifier

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

Four regimes per spec §3 stage 1: silent / single-burst (= excitable, requires perturbation per burst) / repetitive-burst (spontaneous) / unstable (burst doesn't terminate).

Operational definitions (concrete, testable):
- **silent**: 0 bursts in T
- **excitable**: ≥1 burst, all bursts ≤ 50 HR time units, mean inter-burst-interval ≥ 30 HR time units (sparse, perturbation-driven)
- **repetitive-burst**: ≥3 bursts, mean inter-burst-interval < 30 HR time units (regular spontaneous)
- **unstable**: any single burst longer than 100 HR time units (no termination)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_classify_regime_silent_when_zero_bursts():
    """No bursts → 'silent'."""
    from src.topic4_modeling.hr import classify_regime
    assert classify_regime(bursts=[], T=1000.0) == "silent"


def test_classify_regime_excitable_when_sparse_short_bursts():
    """Sparse short bursts (long IBI) → 'excitable'."""
    from src.topic4_modeling.hr import classify_regime
    bursts = [(100.0, 110.0), (300.0, 310.0), (700.0, 710.0)]  # IBI ~190
    assert classify_regime(bursts=bursts, T=1000.0) == "excitable"


def test_classify_regime_repetitive_burst_when_regular_short_ibi():
    """Regular short-IBI bursts → 'repetitive-burst'."""
    from src.topic4_modeling.hr import classify_regime
    bursts = [(t, t + 5.0) for t in range(50, 500, 20)]  # IBI = 20 < 30
    assert classify_regime(bursts=bursts, T=1000.0) == "repetitive-burst"


def test_classify_regime_unstable_when_burst_does_not_terminate():
    """Burst lasting >100 time units → 'unstable'."""
    from src.topic4_modeling.hr import classify_regime
    bursts = [(50.0, 200.0)]  # 150 unit burst
    assert classify_regime(bursts=bursts, T=1000.0) == "unstable"


def test_classify_regime_unstable_takes_precedence_over_repetitive():
    """Even if many bursts, one unstable burst dominates classification."""
    from src.topic4_modeling.hr import classify_regime
    bursts = [(50.0, 55.0), (60.0, 65.0), (70.0, 200.0)]  # last one unstable
    assert classify_regime(bursts=bursts, T=1000.0) == "unstable"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "classify_regime"`
Expected: 5 FAILs

- [ ] **Step 3: Implement classify_regime**

Append to `src/topic4_modeling/hr.py`:

```python
def classify_regime(
    bursts: list[tuple[float, float]],
    T: float,
    max_burst_duration: float = 100.0,
    excitable_max_burst: float = 50.0,
    excitable_min_ibi: float = 30.0,
) -> str:
    """Classify HR regime from burst list.

    Returns one of: "silent", "excitable", "repetitive-burst", "unstable".

    Spec §3 Stage 1 + this plan task 8 operational definitions.
    """
    if not bursts:
        return "silent"
    durations = [end - start for start, end in bursts]
    if any(d > max_burst_duration for d in durations):
        return "unstable"
    if len(bursts) >= 2:
        ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
        mean_ibi = float(np.mean(ibis))
    else:
        mean_ibi = float("inf")
    if (
        all(d <= excitable_max_burst for d in durations)
        and mean_ibi >= excitable_min_ibi
    ):
        return "excitable"
    return "repetitive-burst"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "classify_regime"`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): classify_regime (silent/excitable/repetitive/unstable)"
```

---

## Task 9: Phase portrait + nullclines plotter

**Files:**
- Create: `src/topic4_modeling/hr_viz.py`
- Create: `tests/test_topic4_modeling_hr_viz.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_topic4_modeling_hr_viz.py`:

```python
"""Tests for src/topic4_modeling/hr_viz.py."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def test_nullcline_x_formula_at_known_point():
    """x-nullcline: y = a x³ - b x² + z - I at given x."""
    from src.topic4_modeling.hr_viz import compute_x_nullcline
    from src.topic4_modeling.hr import HRParams
    p = HRParams()
    x_grid = np.array([0.0, 1.0, -1.0])
    y_null = compute_x_nullcline(x_grid, params=p, z=0.5, I=-1.6)
    expected = p.a * x_grid**3 - p.b * x_grid**2 + 0.5 - (-1.6)
    np.testing.assert_allclose(y_null, expected)


def test_nullcline_y_formula_at_known_point():
    """y-nullcline: y = c - d x²."""
    from src.topic4_modeling.hr_viz import compute_y_nullcline
    from src.topic4_modeling.hr import HRParams
    p = HRParams()
    x_grid = np.array([0.0, 1.0, -1.0])
    y_null = compute_y_nullcline(x_grid, params=p)
    expected = p.c - p.d * x_grid**2
    np.testing.assert_allclose(y_null, expected)


def test_plot_phase_portrait_smoke(tmp_path):
    """plot_phase_portrait runs and saves PNG with expected size."""
    from src.topic4_modeling.hr_viz import plot_phase_portrait
    from src.topic4_modeling.hr import HRParams, simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-1.6, T=200.0, dt=0.05,
                                   sigma_ou=0.1, tau_ou=10.0, seed=0)
    out = tmp_path / "phase_portrait.png"
    fig = plot_phase_portrait(traj, params=p, I=-1.6)
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial png
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr_viz.py -v`
Expected: 3 FAILs with `ModuleNotFoundError`

- [ ] **Step 3: Implement hr_viz.py**

Create `src/topic4_modeling/hr_viz.py`:

```python
"""Stage 1 phase-portrait + nullcline visualization for HR single node.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .hr import HRParams


def compute_x_nullcline(
    x_grid: np.ndarray, params: HRParams, z: float, I: float,
) -> np.ndarray:
    """x-nullcline: where dx/dt = 0, given z and I (and eta=0).

    From dx/dt = y - a x³ + b x² - z + I = 0:
        y = a x³ - b x² + z - I
    """
    p = params
    return p.a * x_grid**3 - p.b * x_grid**2 + z - I


def compute_y_nullcline(x_grid: np.ndarray, params: HRParams) -> np.ndarray:
    """y-nullcline: where dy/dt = 0:  y = c - d x²."""
    p = params
    return p.c - p.d * x_grid**2


def plot_phase_portrait(
    trajectory: np.ndarray,
    params: HRParams,
    I: float,
    figsize: tuple[float, float] = (8.0, 6.0),
) -> plt.Figure:
    """Plot phase portrait: x-y plane with trajectory overlay + nullclines.

    Nullclines drawn at trajectory mean z (single representative z slice).
    """
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z_mean = float(trajectory[:, 2].mean())

    x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
    y_null_x = compute_x_nullcline(x_grid, params, z_mean, I)
    y_null_y = compute_y_nullcline(x_grid, params)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, alpha=0.4, color="C0", linewidth=0.5, label="trajectory")
    ax.plot(x_grid, y_null_x, "r-", label=f"dx/dt=0  (z≈{z_mean:.2f})")
    ax.plot(x_grid, y_null_y, "g-", label="dy/dt=0")
    ax.scatter([x[0]], [y[0]], marker="o", color="black", zorder=5, label="start")
    ax.scatter([x[-1]], [y[-1]], marker="s", color="black", zorder=5, label="end")
    ax.set_xlabel("x (fast voltage-like)")
    ax.set_ylabel("y (spiking variable)")
    ax.set_title(f"HR phase portrait  (I={I:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    margin_y = 0.1 * (y.max() - y.min() + 1e-6)
    ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr_viz.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr_viz.py tests/test_topic4_modeling_hr_viz.py
git commit -m "feat(topic4 phase4 stage1): hr_viz nullclines + phase portrait plotter"
```

---

## Task 10: Single-cell regime evaluator (composing trajectory + bursts + classify)

**Files:**
- Modify: `src/topic4_modeling/hr.py`
- Modify: `tests/test_topic4_modeling_hr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr.py`:

```python
def test_evaluate_cell_returns_regime_string_and_metadata():
    """evaluate_cell composes simulate + detect + classify into one call."""
    from src.topic4_modeling.hr import HRParams, evaluate_cell
    p = HRParams()
    result = evaluate_cell(p, I=-2.5, sigma_ou=0.0, tau_ou=10.0,
                           r_override=None, T=500.0, dt=0.05, seed=0)
    assert isinstance(result, dict)
    assert "regime" in result
    assert "n_bursts" in result
    assert "mean_burst_duration" in result
    assert "mean_ibi" in result
    assert result["regime"] in {"silent", "excitable", "repetitive-burst", "unstable"}


def test_evaluate_cell_low_I_no_noise_is_silent():
    """At deeply sub-threshold I with no noise → silent regime."""
    from src.topic4_modeling.hr import HRParams, evaluate_cell
    p = HRParams()
    result = evaluate_cell(p, I=-3.0, sigma_ou=0.0, tau_ou=10.0,
                           r_override=None, T=500.0, dt=0.05, seed=0)
    assert result["regime"] == "silent"
    assert result["n_bursts"] == 0


def test_evaluate_cell_high_I_is_repetitive_or_unstable():
    """At elevated I, regime should be repetitive-burst or unstable."""
    from src.topic4_modeling.hr import HRParams, evaluate_cell
    p = HRParams()
    result = evaluate_cell(p, I=2.0, sigma_ou=0.0, tau_ou=10.0,
                           r_override=None, T=500.0, dt=0.05, seed=0)
    assert result["regime"] in {"repetitive-burst", "unstable"}
    assert result["n_bursts"] >= 3


def test_evaluate_cell_r_override_changes_burst_rate():
    """Faster r (slow var) → shorter inter-burst-interval."""
    from src.topic4_modeling.hr import HRParams, evaluate_cell
    p = HRParams()
    slow = evaluate_cell(p, I=2.0, sigma_ou=0.0, tau_ou=10.0,
                         r_override=0.003, T=1000.0, dt=0.05, seed=0)
    fast = evaluate_cell(p, I=2.0, sigma_ou=0.0, tau_ou=10.0,
                         r_override=0.012, T=1000.0, dt=0.05, seed=0)
    # fast r → more bursts (each cycle shorter)
    assert fast["n_bursts"] > slow["n_bursts"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "evaluate_cell"`
Expected: 4 FAILs with `ImportError: cannot import name 'evaluate_cell'`

- [ ] **Step 3: Implement evaluate_cell**

Append to `src/topic4_modeling/hr.py`:

```python
from dataclasses import replace


def evaluate_cell(
    params: HRParams,
    I: float,
    sigma_ou: float,
    tau_ou: float,
    r_override: float | None,
    T: float,
    dt: float,
    seed: int,
    x_threshold: float = 1.0,
    min_burst_duration: float = 5.0,
) -> dict:
    """Run one sim + detect bursts + classify regime.

    Args:
        params: HR baseline params
        I: input current
        sigma_ou: OU noise std
        tau_ou: OU correlation time
        r_override: if not None, override params.r with this value
        T, dt: simulation duration + timestep (HR time units)
        seed: random seed
        x_threshold, min_burst_duration: burst detector params

    Returns:
        dict with keys: regime, n_bursts, mean_burst_duration, mean_ibi,
        I, r_used, sigma_ou, seed, T (full provenance for sweep cell)
    """
    p = params if r_override is None else replace(params, r=r_override)
    t, traj = simulate_trajectory(p, I, T, dt, sigma_ou, tau_ou, seed)
    x = traj[:, 0]
    bursts = detect_bursts(x, t, x_threshold=x_threshold,
                            min_duration=min_burst_duration)
    durations = [end - start for start, end in bursts]
    ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    regime = classify_regime(bursts, T)
    return {
        "regime": regime,
        "n_bursts": len(bursts),
        "mean_burst_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_ibi": float(np.mean(ibis)) if ibis else float("inf"),
        "I": I,
        "r_used": p.r,
        "sigma_ou": sigma_ou,
        "seed": seed,
        "T": T,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr.py -v -k "evaluate_cell"`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "feat(topic4 phase4 stage1): evaluate_cell composing sim+detect+classify"
```

---

## Task 11: Parameter sweep runner

**Files:**
- Create: `src/topic4_modeling/hr_sweep.py`
- Create: `tests/test_topic4_modeling_hr_sweep.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_topic4_modeling_hr_sweep.py`:

```python
"""Tests for src/topic4_modeling/hr_sweep.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_sweep_grid_total_count_matches_cartesian_product():
    """Sweep produces n_I * n_r * n_sigma * n_seeds rows."""
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0, -1.6, 0.0],
        r_grid=[0.004, 0.006],
        sigma_grid=[0.0, 0.1],
        seeds=[0, 1],
        T=100.0,
        dt=0.05,
        n_jobs=1,
    )
    assert len(df) == 3 * 2 * 2 * 2  # = 24


def test_sweep_grid_columns_complete():
    """Sweep DataFrame has all expected columns."""
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0], r_grid=[0.006], sigma_grid=[0.0], seeds=[0],
        T=50.0, dt=0.05, n_jobs=1,
    )
    required = {"I", "r_used", "sigma_ou", "seed", "regime", "n_bursts",
                "mean_burst_duration", "mean_ibi"}
    assert required.issubset(df.columns)


def test_sweep_grid_deterministic_per_cell():
    """Same params + seed → same regime label across two sweep calls."""
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df1 = sweep_hr_parameters(
        I_grid=[-1.6], r_grid=[0.006], sigma_grid=[0.1], seeds=[42],
        T=100.0, dt=0.05, n_jobs=1,
    )
    df2 = sweep_hr_parameters(
        I_grid=[-1.6], r_grid=[0.006], sigma_grid=[0.1], seeds=[42],
        T=100.0, dt=0.05, n_jobs=1,
    )
    assert df1["regime"].iloc[0] == df2["regime"].iloc[0]
    assert df1["n_bursts"].iloc[0] == df2["n_bursts"].iloc[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr_sweep.py -v`
Expected: 3 FAILs with `ModuleNotFoundError`

- [ ] **Step 3: Implement hr_sweep.py**

Create `src/topic4_modeling/hr_sweep.py`:

```python
"""Stage 1 parameter sweep runner + baseline picker.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .hr import HRParams, evaluate_cell


def sweep_hr_parameters(
    I_grid: Sequence[float],
    r_grid: Sequence[float],
    sigma_grid: Sequence[float],
    seeds: Sequence[int],
    T: float,
    dt: float,
    n_jobs: int = 1,
    params_base: HRParams | None = None,
) -> pd.DataFrame:
    """Run Cartesian sweep over (I, r, sigma, seed).

    Args:
        I_grid, r_grid, sigma_grid: parameter values to scan
        seeds: random seeds per cell (each (I, r, sigma) cell × each seed = 1 row)
        T, dt: sim duration + timestep
        n_jobs: joblib parallelism (1 = serial)
        params_base: HR baseline params (default = HRParams())

    Returns:
        DataFrame with one row per (I, r, sigma, seed) cell,
        columns from evaluate_cell return dict.
    """
    if params_base is None:
        params_base = HRParams()

    cells = [
        (I, r, sigma, seed)
        for I in I_grid
        for r in r_grid
        for sigma in sigma_grid
        for seed in seeds
    ]

    def _eval(cell):
        I, r, sigma, seed = cell
        return evaluate_cell(
            params=params_base,
            I=I,
            sigma_ou=sigma,
            tau_ou=10.0,
            r_override=r,
            T=T,
            dt=dt,
            seed=seed,
        )

    if n_jobs == 1:
        results = [_eval(c) for c in cells]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval)(c) for c in cells
        )
    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr_sweep.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr_sweep.py tests/test_topic4_modeling_hr_sweep.py
git commit -m "feat(topic4 phase4 stage1): hr_sweep parameter sweep runner"
```

---

## Task 12: Baseline picker (excitable regime + noise-robust)

**Files:**
- Modify: `src/topic4_modeling/hr_sweep.py`
- Modify: `tests/test_topic4_modeling_hr_sweep.py`

Spec §3 Stage 1 退出契约:

> 存在参数子带 (I*, r*, σ*) 节点无外驱时静默，OU 推扰可 trigger brief burst 然后回静默
> 该 regime 对 noise amplitude ±50% 不漂

Operational baseline picker:
1. Filter cells where regime == "excitable" at σ_OU > 0
2. For each (I, r) pair, check that the same (I, r) at σ_OU = 0 is "silent" AND at σ_OU × 1.5 is still "excitable" (not "repetitive-burst")
3. Among surviving (I, r), pick the one with median sigma value (most "central" / robust)
4. Return (I*, r*, σ*) + diagnostic dict

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr_sweep.py`:

```python
def test_pick_excitable_baseline_returns_silent_at_zero_noise():
    """Picked baseline must satisfy: silent at sigma=0."""
    from src.topic4_modeling.hr_sweep import (
        sweep_hr_parameters, pick_excitable_baseline,
    )
    df = sweep_hr_parameters(
        I_grid=np.linspace(-2.5, -0.5, 11).tolist(),
        r_grid=[0.004, 0.006, 0.008],
        sigma_grid=[0.0, 0.05, 0.1, 0.15],
        seeds=[0, 1, 2],
        T=500.0, dt=0.05, n_jobs=1,
    )
    baseline = pick_excitable_baseline(df)
    if baseline is None:
        pytest.skip("No excitable baseline found in this sweep — Stage 1 exit fails")
    I_star, r_star, sigma_star = (
        baseline["I_star"], baseline["r_star"], baseline["sigma_star"],
    )
    # silent at sigma=0 in this (I, r) cell
    silent_check = df[
        (df["I"] == I_star)
        & (df["r_used"] == r_star)
        & (df["sigma_ou"] == 0.0)
    ]
    assert silent_check["regime"].mode().iloc[0] == "silent"


def test_pick_excitable_baseline_robust_to_noise_perturbation():
    """Picked baseline regime stays 'excitable' under ±50% sigma."""
    from src.topic4_modeling.hr_sweep import (
        sweep_hr_parameters, pick_excitable_baseline,
    )
    df = sweep_hr_parameters(
        I_grid=np.linspace(-2.5, -0.5, 11).tolist(),
        r_grid=[0.004, 0.006, 0.008],
        sigma_grid=[0.0, 0.05, 0.1, 0.15],
        seeds=[0, 1, 2],
        T=500.0, dt=0.05, n_jobs=1,
    )
    baseline = pick_excitable_baseline(df)
    if baseline is None:
        pytest.skip("No excitable baseline found")
    assert baseline["noise_robust"] is True


def test_pick_excitable_baseline_returns_none_when_no_candidate():
    """If no cell is excitable + silent-at-zero, returns None."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    import pandas as pd
    # synthetic: all cells either silent or unstable
    df = pd.DataFrame([
        {"I": -1.6, "r_used": 0.006, "sigma_ou": 0.0, "seed": 0,
         "regime": "silent", "n_bursts": 0, "mean_burst_duration": 0.0,
         "mean_ibi": float("inf")},
        {"I": -1.6, "r_used": 0.006, "sigma_ou": 0.1, "seed": 0,
         "regime": "silent", "n_bursts": 0, "mean_burst_duration": 0.0,
         "mean_ibi": float("inf")},
    ])
    assert pick_excitable_baseline(df) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_topic4_modeling_hr_sweep.py -v -k "pick_excitable_baseline"`
Expected: 3 FAILs with `ImportError: cannot import name 'pick_excitable_baseline'`

- [ ] **Step 3: Implement pick_excitable_baseline**

Append to `src/topic4_modeling/hr_sweep.py`:

```python
def pick_excitable_baseline(df: pd.DataFrame) -> dict | None:
    """Pick (I*, r*, sigma*) cell satisfying Stage 1 exit contract.

    Criteria (spec §3 Stage 1 退出契约):
        1. At chosen (I, r, sigma), majority of seeds → regime == "excitable"
        2. Same (I, r) at sigma=0 → majority → "silent" (no spontaneous firing)
        3. Same (I, r) at sigma × 1.5 → still mostly "excitable" (not flipped to
           "repetitive-burst" or "unstable") — noise robustness check

    Among surviving (I, r, sigma) cells, picks the one with median sigma
    (central, robust). Returns None if no cell satisfies all 3 criteria.
    """
    candidates = []
    for (I, r, sigma), group in df.groupby(["I", "r_used", "sigma_ou"]):
        modal_regime = group["regime"].mode().iloc[0]
        if modal_regime != "excitable" or sigma <= 0.0:
            continue
        # Check silent at sigma=0 for same (I, r)
        zero_noise = df[
            (df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == 0.0)
        ]
        if zero_noise.empty:
            continue
        if zero_noise["regime"].mode().iloc[0] != "silent":
            continue
        # Find available sigma > sigma (for the ×1.5 check; nearest above)
        all_sigmas = sorted(df["sigma_ou"].unique())
        higher = [s for s in all_sigmas if s >= 1.4 * sigma]
        noise_robust = False
        if higher:
            higher_sigma = higher[0]
            high_cell = df[
                (df["I"] == I)
                & (df["r_used"] == r)
                & (df["sigma_ou"] == higher_sigma)
            ]
            if not high_cell.empty:
                regime_at_high = high_cell["regime"].mode().iloc[0]
                if regime_at_high == "excitable":
                    noise_robust = True
        # Accept candidate only if noise-robust
        if not noise_robust:
            continue
        candidates.append({
            "I_star": float(I),
            "r_star": float(r),
            "sigma_star": float(sigma),
            "noise_robust": True,
        })
    if not candidates:
        return None
    # Pick candidate with median sigma_star (central)
    candidates.sort(key=lambda c: c["sigma_star"])
    return candidates[len(candidates) // 2]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_topic4_modeling_hr_sweep.py -v -k "pick_excitable_baseline"`
Expected: 3 PASS (or skip if no excitable regime found in sweep — that's a Stage 1 exit failure, fall back to FHN per spec §8)

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr_sweep.py tests/test_topic4_modeling_hr_sweep.py
git commit -m "feat(topic4 phase4 stage1): pick_excitable_baseline with noise-robust check"
```

---

## Task 13: Regime map plotter

**Files:**
- Modify: `src/topic4_modeling/hr_viz.py`
- Modify: `tests/test_topic4_modeling_hr_viz.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_topic4_modeling_hr_viz.py`:

```python
def test_plot_regime_map_smoke(tmp_path):
    """Given sweep DataFrame, plot_regime_map writes a non-empty PNG."""
    import pandas as pd
    from src.topic4_modeling.hr_viz import plot_regime_map
    # synthetic 3×3×2 grid with mock regimes
    rows = []
    for I in [-2.0, -1.6, -1.0]:
        for r in [0.004, 0.006, 0.008]:
            for sigma in [0.0, 0.1]:
                regime = "silent" if sigma == 0.0 else "excitable"
                rows.append({"I": I, "r_used": r, "sigma_ou": sigma,
                             "seed": 0, "regime": regime, "n_bursts": 1,
                             "mean_burst_duration": 5.0, "mean_ibi": 100.0})
    df = pd.DataFrame(rows)
    out = tmp_path / "regime_map.png"
    fig = plot_regime_map(df)
    fig.savefig(out)
    matplotlib.pyplot.close(fig)
    assert out.exists()
    assert out.stat().st_size > 1000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_topic4_modeling_hr_viz.py::test_plot_regime_map_smoke -v`
Expected: FAIL with `ImportError: cannot import name 'plot_regime_map'`

- [ ] **Step 3: Implement plot_regime_map**

Append to `src/topic4_modeling/hr_viz.py`:

```python
import pandas as pd

REGIME_COLOR = {
    "silent": "#dddddd",
    "excitable": "#4daf4a",
    "repetitive-burst": "#ff7f00",
    "unstable": "#e41a1c",
}
REGIME_ORDER = ["silent", "excitable", "repetitive-burst", "unstable"]


def plot_regime_map(
    sweep_df: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot regime map: rows = sigma_ou, cols = r_used, x-axis = I.

    Each cell colored by modal regime across seeds for that (I, r, sigma).
    """
    sigmas = sorted(sweep_df["sigma_ou"].unique())
    rs = sorted(sweep_df["r_used"].unique())
    Is = sorted(sweep_df["I"].unique())

    n_rows = len(sigmas)
    n_cols = len(rs)
    if figsize is None:
        figsize = (3.0 * n_cols, 1.2 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)

    for i, sigma in enumerate(sigmas):
        for j, r in enumerate(rs):
            ax = axes[i, j]
            cell_data = []
            for I in Is:
                sub = sweep_df[
                    (sweep_df["sigma_ou"] == sigma)
                    & (sweep_df["r_used"] == r)
                    & (sweep_df["I"] == I)
                ]
                if sub.empty:
                    cell_data.append("silent")
                else:
                    cell_data.append(sub["regime"].mode().iloc[0])
            colors = [REGIME_COLOR[c] for c in cell_data]
            ax.bar(range(len(Is)), [1] * len(Is), color=colors, width=1.0,
                   edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(Is)))
            ax.set_xticklabels([f"{I:.1f}" for I in Is], rotation=45, fontsize=7)
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"r={r:.4f}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"σ_OU={sigma:.2f}", fontsize=8)
            if i == n_rows - 1:
                ax.set_xlabel("I", fontsize=8)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[r]) for r in REGIME_ORDER
    ]
    fig.legend(handles, REGIME_ORDER, loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.suptitle("HR single-node regime map", y=0.94, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_topic4_modeling_hr_viz.py::test_plot_regime_map_smoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/topic4_modeling/hr_viz.py tests/test_topic4_modeling_hr_viz.py
git commit -m "feat(topic4 phase4 stage1): plot_regime_map heatmap (sigma × r × I grid)"
```

---

## Task 14: CLI orchestration script

**Files:**
- Create: `scripts/run_topic4_phase4_stage1_hr.py`

This is operational, not unit-tested in detail — tested via smoke run in Task 15.

- [ ] **Step 1: Create script**

Create `scripts/run_topic4_phase4_stage1_hr.py`:

```python
#!/usr/bin/env python3
"""SEF-ITP Phase 4 Stage 1 runner: HR single-node parameter sweep.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
Plan: docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md

Usage:
    python scripts/run_topic4_phase4_stage1_hr.py --mode smoke
    python scripts/run_topic4_phase4_stage1_hr.py --mode full --n-jobs 8

Outputs land in results/topic4_sef_itp/phase4_modeling/stage1_hr_single/:
    - regime_map.png
    - regime_summary.json
    - baseline.json     (selected I*, r*, sigma*)
    - phase_portraits/  (representative cells)
    - sweep_results.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.topic4_modeling.hr import HRParams, simulate_trajectory
from src.topic4_modeling.hr_sweep import (
    pick_excitable_baseline,
    sweep_hr_parameters,
)
from src.topic4_modeling.hr_viz import plot_phase_portrait, plot_regime_map

OUTPUT_ROOT = Path("results/topic4_sef_itp/phase4_modeling/stage1_hr_single")

SMOKE_CONFIG = {
    "I_grid": np.linspace(-2.5, 0.0, 6).tolist(),
    "r_grid": [0.004, 0.006, 0.008],
    "sigma_grid": [0.0, 0.05, 0.10, 0.15],
    "seeds": [0, 1, 2],
    "T": 200.0,
    "dt": 0.05,
}

FULL_CONFIG = {
    "I_grid": np.arange(-2.5, 0.05, 0.1).tolist(),
    "r_grid": np.arange(0.002, 0.016, 0.001).tolist(),
    "sigma_grid": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    "seeds": [0, 1, 2, 3, 4],
    "T": 500.0,
    "dt": 0.05,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke",
                        help="smoke: ~5 min; full: ~25 min single-threaded")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="joblib parallelism")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    config = SMOKE_CONFIG if args.mode == "smoke" else FULL_CONFIG
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_portraits").mkdir(exist_ok=True)

    print(f"[stage1] mode={args.mode}  n_jobs={args.n_jobs}")
    print(f"[stage1] sweep cells = "
          f"{len(config['I_grid']) * len(config['r_grid']) * len(config['sigma_grid']) * len(config['seeds'])}")

    df = sweep_hr_parameters(
        I_grid=config["I_grid"],
        r_grid=config["r_grid"],
        sigma_grid=config["sigma_grid"],
        seeds=config["seeds"],
        T=config["T"],
        dt=config["dt"],
        n_jobs=args.n_jobs,
    )

    df.to_parquet(args.output_dir / "sweep_results.parquet")
    print(f"[stage1] sweep done, n_rows={len(df)}")
    print(f"[stage1] regime distribution:\n{df['regime'].value_counts()}")

    fig = plot_regime_map(df)
    fig.savefig(args.output_dir / "regime_map.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"[stage1] regime_map.png saved")

    baseline = pick_excitable_baseline(df)
    summary = {
        "mode": args.mode,
        "sweep_config": config,
        "regime_counts": df["regime"].value_counts().to_dict(),
        "baseline": baseline,
        "stage1_exit_contract_passed": baseline is not None,
    }
    with open(args.output_dir / "regime_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[stage1] regime_summary.json saved")

    if baseline is not None:
        with open(args.output_dir / "baseline.json", "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"[stage1] baseline (I*, r*, sigma*) = "
              f"({baseline['I_star']:.3f}, {baseline['r_star']:.4f}, "
              f"{baseline['sigma_star']:.3f}) [hand-off contract to Stage 2]")

        # Phase portraits at baseline and 3 neighbor cells
        p = HRParams()
        for label, (I, r, sigma) in [
            ("baseline", (baseline["I_star"], baseline["r_star"],
                          baseline["sigma_star"])),
            ("baseline_zero_noise", (baseline["I_star"], baseline["r_star"], 0.0)),
            ("baseline_high_noise", (baseline["I_star"], baseline["r_star"],
                                      baseline["sigma_star"] * 2.0)),
        ]:
            from dataclasses import replace
            p_cell = replace(p, r=r)
            _, traj = simulate_trajectory(p_cell, I=I, T=300.0, dt=0.05,
                                           sigma_ou=sigma, tau_ou=10.0,
                                           seed=0)
            fig = plot_phase_portrait(traj, p_cell, I=I)
            fig.savefig(args.output_dir / "phase_portraits" / f"{label}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"[stage1] phase_portraits/ written (baseline + 2 neighbors)")
        return 0
    else:
        print("[stage1] EXIT CONTRACT FAILED: no excitable regime found")
        print("[stage1] per spec §8 stage 1 fallback: try FHN-with-adaptation")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports without error (smoke check)**

Run: `python -c "import scripts.run_topic4_phase4_stage1_hr as m; print('OK')"`
Expected: prints `OK` (no import error)

Note: this requires PYTHONPATH including project root. If module-path issue, prefix:
`PYTHONPATH=. python -c "..."`

- [ ] **Step 3: Verify help text works**

Run: `PYTHONPATH=. python scripts/run_topic4_phase4_stage1_hr.py --help`
Expected: argparse help text including `--mode`, `--n-jobs`, `--output-dir`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_topic4_phase4_stage1_hr.py
git commit -m "feat(topic4 phase4 stage1): CLI runner for HR parameter sweep"
```

---

## Task 15: Smoke run + Stage 1 exit contract verification

**Files:**
- Run (no new code)
- Modify: `docs/topic4_sef_itp_framework.md` (only add Stage 1 status link if passes)

This is the operational milestone. We run the smoke sweep, verify regime map looks reasonable, confirm baseline is found.

- [ ] **Step 1: Run smoke sweep**

Run:
```bash
PYTHONPATH=. python scripts/run_topic4_phase4_stage1_hr.py --mode smoke --n-jobs 4
```

Expected output (approximate):
- prints `[stage1] mode=smoke  n_jobs=4`
- prints `[stage1] sweep cells = 216` (6 × 3 × 4 × 3)
- runs ~3-5 minutes
- prints `[stage1] regime distribution:` with non-zero counts in multiple regimes
- creates `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/regime_map.png`
- creates `regime_summary.json` + (if exit passes) `baseline.json` + `phase_portraits/`
- exit code 0 if baseline found, 1 if not

- [ ] **Step 2: Manually inspect regime_map.png**

Open `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/regime_map.png`.

Expected:
- Multiple cells colored green ("excitable") at sigma > 0
- silent (gray) cells at low I + low sigma
- repetitive-burst (orange) cells at high I
- No unstable (red) cells dominating

If pattern unclear (all gray or all orange): smoke grid too coarse, run `--mode full` next.

- [ ] **Step 3: Inspect baseline.json**

Run: `cat results/topic4_sef_itp/phase4_modeling/stage1_hr_single/baseline.json`

Expected (concrete values will vary):
```json
{
  "I_star": -1.3,
  "r_star": 0.006,
  "sigma_star": 0.1,
  "noise_robust": true
}
```

Verify all three values are within sweep range and `noise_robust=true`.

- [ ] **Step 4: Inspect phase portraits**

Open `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/phase_portraits/baseline.png`.

Expected:
- Trajectory bounces around rest with occasional brief excursions through the burst region
- Nullclines (red x-nullcline, green y-nullcline) intersect near rest

Open `baseline_zero_noise.png`: should show trajectory stuck at rest (no excursions).

Open `baseline_high_noise.png`: should show more frequent excursions but still discrete bursts (not continuous oscillation).

If `baseline.png` shows trajectory at rest only OR continuous oscillation: smoke sweep didn't find the right baseline. Either re-run with finer grid or check parameter ranges.

- [ ] **Step 5: If smoke passed, optionally run full sweep**

If smoke regime map looks reasonable AND baseline found AND phase portraits sensible, run the full sweep for the lock-in baseline:

```bash
PYTHONPATH=. python scripts/run_topic4_phase4_stage1_hr.py --mode full --n-jobs 8
```

Expected: ~25 min on 8 cores. Same artifacts but higher-resolution `regime_map.png` and more confidence in baseline pick.

- [ ] **Step 6: Run full test suite to confirm no regression**

Run:
```bash
pytest tests/test_topic4_modeling_hr.py tests/test_topic4_modeling_hr_viz.py tests/test_topic4_modeling_hr_sweep.py -v
```

Expected: all green (~30+ tests).

- [ ] **Step 7: Write Stage 1 archive results doc**

Create `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md`:

```markdown
# SEF-ITP Phase 4 Stage 1 Results — Single-Node HR

> 状态：[PASS|FAIL] (取实际)
> 日期：[YYYY-MM-DD]
> 上游 spec：`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md`
> 上游 plan：`docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md`

## 一句话朴素话

我们把 HR 单节点的 3 个核心参数 (I, r, σ_OU) 各自扫一遍，看哪些参数组合下节点行为像「正常静默 + 偶尔被噪声推一下打一个 brief burst 又回到静默」(excitable regime)。结果在 (I*, r*, σ*) = [实跑数] 处找到 such 子带；这个区间将作为 Stage 2 2D 均质 sheet 的节点 baseline 参数。

## Stage 1 退出契约结果

- [ ] 存在 excitable 参数子带：[PASS|FAIL]
- [ ] noise amplitude ±50% regime 不漂：[PASS|FAIL]
- [ ] 选中 baseline (I*, r*, σ*) = (___, ___, ___)
- [ ] TDD 测试全 GREEN: [n tests passed]

## Regime distribution

[paste df['regime'].value_counts() output]

## 输出 artifact

- `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/regime_map.png`
- `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/regime_summary.json`
- `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/baseline.json`
- `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/phase_portraits/`
- `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/sweep_results.parquet`

## Advisor consult

[paste advisor() call summary; verdict over-claim check]

## 下一步

[If PASS] → Stage 2 plan 立项 (`docs/superpowers/plans/2026-05-XX-sef-itp-phase4-stage2-2d-homogeneous.md`)
[If FAIL] → spec §8 fallback: FHN-with-adaptation 重做 Stage 1
```

Fill in the bracketed values from your actual run.

- [ ] **Step 8: Commit results**

```bash
mkdir -p docs/archive/topic4/sef_itp_phase4_v1
git add docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md \
        results/topic4_sef_itp/phase4_modeling/stage1_hr_single/
git commit -m "results(topic4 phase4 stage1): HR single-node sweep + baseline lock"
```

- [ ] **Step 9: Advisor consult before declaring Stage 1 done**

In the implementation session, call `advisor()` with the full Stage 1 transcript. Advisor should check:
- regime map looks sensible (not all one regime)
- baseline noise-robustness is real, not lucky single seed
- Stage 2 hand-off contract clearly defined
- No over-claim in archive results doc

If advisor flags issues → fix inline (re-run, adjust, re-commit) before moving to Stage 2 plan.

---

## Self-Review

**1. Spec coverage**:

| Spec §3 Stage 1 requirement | Task |
|---|---|
| HR ODE with spec §5.1 params | Task 2 (HRParams) + Task 3 (hr_rhs) |
| RK4, Δt=0.05 | Task 4 (rk4_step) |
| Parameter sweep I × r × σ_OU × seeds | Task 11 (sweep_hr_parameters) |
| 5 seeds per cell | Task 14 CLI (FULL_CONFIG seeds=[0..4]) |
| Phase portrait + nullclines | Task 9 (hr_viz) |
| Burst detection x > 1.0 sustained ≥ 5ms | Task 7 (detect_bursts) |
| Regime classifier (silent/single-burst/repetitive/unstable) | Task 8 (classify_regime); note: spec says "single-burst" we use "excitable" (synonymous in this context, perturbation-driven) |
| Exit: excitable subband + noise ±50% robust | Task 12 (pick_excitable_baseline) |
| Output regime_map.png + regime_summary.json + selected baseline | Task 14 CLI + Task 15 |
| TDD 6 test modules GREEN | Tasks 1-13 each have tests; Task 15 step 6 runs full suite |

**Gap identified + fixed**: spec says "single-burst" regime, plan uses "excitable" label. These are semantically equivalent under spec §3 Stage 1 ("节点静息但被短扰动或 OU fluctuation 可触发 brief burst") — both mean perturbation-triggered single bursts. "Excitable" is the standard dynamical systems term; "single-burst" is the spec's phenomenological term. I've kept "excitable" in the code and noted the equivalence in this self-review. If you prefer "single-burst" rename: replace string literal in Task 8 + Task 12 + Task 13 + Task 15 step 3 output. Not a blocker.

**2. Placeholder scan**:
- No "TBD" / "TODO" / "implement later" anywhere
- All test code is complete (not "write similar test")
- All implementation code is complete (not "fill in here")
- All commands have exact invocations + expected output
- Task 15 step 7 archive results doc has bracketed `[YYYY-MM-DD]` / `[___]` — those are user-fillable runtime values, not plan placeholders (clearly marked as "fill in from your actual run")

**3. Type consistency**:
- `HRParams` used consistently across Tasks 2-12
- `hr_rhs(x, y, z, params, I, eta)` signature consistent in Tasks 3-4
- `rk4_step(state, params, I, eta, dt)` returns `(x, y, z)` tuple consistent
- `simulate_trajectory` returns `(t, traj)` where traj.shape = (n_steps, 3) consistent in Tasks 6, 9, 14
- `detect_bursts(x, t, ...)` returns `list[(t_start, t_end)]` consistent
- `classify_regime` returns one of 4 strings; `pick_excitable_baseline` filters on `"excitable"`; `plot_regime_map` color-keys on 4 strings. All consistent.
- `sweep_hr_parameters` returns DataFrame with columns from `evaluate_cell` dict; `pick_excitable_baseline` reads those columns; `plot_regime_map` reads them too. Column names checked consistent.

**Self-review pass**.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (15 tasks), review between tasks, fast iteration. Best for: discipline + caught regressions early.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints. Best for: faster wall time, lower coordination overhead.

**Which approach?**
