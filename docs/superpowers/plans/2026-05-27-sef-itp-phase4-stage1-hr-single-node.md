# SEF-ITP Phase 4 Stage 1 (Single-Node HR) Implementation Plan — **v3**

> **v3 supersedes v2 (commit 16a85aa)**, second user-return strict catch 2026-05-27. v2 had 3 must-fix bugs + 3 small notes:
>
> 1. Task 4 baseline picker only checked sigma=0 silent + sigma*1.4 still excitable; **missing the ±50% lower-side check** (sigma*0.5 still excitable). Spec §3 Stage 1 says "noise ±50% 不漂"; v2 picker would accept noise-threshold-edge baselines. v3 fixes picker to enforce BOTH lower AND upper neighbor.
> 2. Task 5 CLI `try/except` around regime_map plot silently swallowed failures + exit 0. v3: synthetic-allsilent skips plot; smoke/full → plot failure flips `stage1_exit_contract_passed=false` + exit 1.
> 3. Task 3 dynamics tests "I=2.0 produces ≥3 bursts" / "r_override changes burst rate" / "silent at deeply subthreshold" are empirical not algebraic — they invite tuning model to satisfy tests. v3 moves them to integration tests `@pytest.mark.slow`, keeps only algebraic / signature / reproducibility tests in unit suite.
> 4. Plan commit messages dropped: no more Co-Authored-By footer (matches repo convention: recent commits bc73135 / 0563525 etc don't have it). Existing 5 committed footers (ead2200..16a85aa) left as-is (no force-push).
> 5. Task 0 step 1 uses `rg` (ripgrep 14.1 available) instead of `grep | head`.
> 6. Framework doc + v1 stub still pending in tree — Task 0 lands amendment first, Task 1 deletes stub.
>
> v2 git history preserved at commit 16a85aa. v1 at commit 857d916.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build single-node Hindmarsh-Rose excitable infrastructure (HR + RK4 numba JIT + OU noise + burst detector + regime classifier + plot + CLI) and produce Stage 1 regime map + baseline (I*, r*, σ*) as hand-off contract to Stage 2 — **after** framework v1.0.7 → v1.0.8 banner amendment commits (Task 0, prerequisite).

**Architecture:** Real numba `@njit(cache=True)` on hot ODE / RK4 / OU loops; numpy elsewhere; matplotlib for viz; pytest for tests with strict TDD on algebraic invariants; joblib for sweep parallelism. JSON (not parquet) for sweep outputs. Configurable thresholds in a frozen dataclass to avoid magic numbers. Baseline picker tested via (a) synthetic DataFrame unit tests that NEVER skip + (b) CLI exit-code 1 + archive FAIL marker when no baseline found.

**Tech Stack:** Python 3.11, numpy, scipy, numba 0.60 (verified), matplotlib, pytest, joblib, dataclasses, json (stdlib). NO parquet / pyarrow / cython.

**Spec source:** `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2 (post-user-return).

**Plan boundary:** Stage 1 ONLY. Stages 2-5 each get their own plan after prior stage's exit contract verified.

---

## What's locked in v3 (vs v2)

| Issue | v2 | **v3** |
|---|---|---|
| Baseline picker noise-robustness | only upper-side check (`sigma * 1.4`) | **both lower (`σ * 0.5`) AND upper (`σ * 1.5`)** neighbors must be excitable; rules out threshold-edge baselines |
| CLI regime_map plot failure handling | silently caught, exit 0 even on plot fail | smoke/full: plot failure → `stage1_exit_contract_passed=false` + exit 1; synthetic-allsilent: skip plot legitimately |
| Empirical dynamics tests | mixed into unit suite | moved to integration suite `@pytest.mark.slow`; unit suite contains only algebraic / signature / reproducibility invariants |
| Commit message footer | Co-Authored-By present | **dropped** (matches repo convention: recent commits don't have it) |
| Task 0 step 1 tooling | `grep ... | head` | `rg` (ripgrep available) |

## What's locked from v1→v2 (still in v3)

| Issue | v1 | **v3** |
|---|---|---|
| Framework amendment timing | "to Stage 5 后再改" | **Task 0 first**, before any code |
| numba | claimed but Python loops | real `@njit(cache=True)` on `hr_rhs_jit`, `rk4_step_jit`, `_trajectory_jit`, `_ou_loop_jit` |
| Commits | 15 per task, per-step | **7 commits total** (one per Task, no per-step) |
| Baseline picker test | `pytest.skip` if no baseline | synthetic always-run + real xfail + CLI exit-1 contract |
| Cluster L3 layer | "swap_class > 30% events" | per-sim label + cohort across-seeds fraction (spec v0.2) |
| Output format | parquet (no pyarrow guard) | JSON |
| L4 principal-curve | hard pass gate | descriptive classifier (spec v0.2) |
| 3-proxy verdict | loose "only引P1" | strict 三 proxy 一致 (spec v0.2) |
| Cell B framing | "C-only test (H-C)" | "anatomical-axis-substrate sanity" (spec v0.2) |

---

## File Structure

**Created:**

| File | Responsibility | Module owner |
|---|---|---|
| `src/topic4_modeling/__init__.py` | Package init (already exists from v1 scaffold commit 30858df) | Task 1 update (re-export) |
| `src/topic4_modeling/hr_core.py` | HRParams + hr_rhs + numba RK4 step | Task 1 |
| `src/topic4_modeling/ou_noise.py` | numba OU noise generator | Task 2 |
| `src/topic4_modeling/hr_config.py` | Frozen dataclass for burst / regime thresholds (avoid magic numbers) | Task 3 |
| `src/topic4_modeling/hr_dynamics.py` | simulate_trajectory + detect_bursts + classify_regime | Task 3 |
| `src/topic4_modeling/hr_sweep.py` | evaluate_cell + sweep_hr_parameters + pick_excitable_baseline | Task 4 |
| `src/topic4_modeling/hr_viz.py` | nullclines + phase portrait + regime map plots | Task 5 |
| `scripts/run_topic4_phase4_stage1_hr.py` | CLI orchestration | Task 5 |
| `tests/test_topic4_modeling_hr_core.py` | Task 1 tests |
| `tests/test_topic4_modeling_ou_noise.py` | Task 2 tests |
| `tests/test_topic4_modeling_hr_dynamics.py` | Task 3 tests |
| `tests/test_topic4_modeling_hr_sweep.py` | Task 4 tests (synthetic always run + real-sweep xfail) |
| `tests/test_topic4_modeling_hr_viz.py` | Task 5 plot smoke tests |
| `tests/test_topic4_modeling_hr_cli.py` | Task 5 CLI exit-contract test |
| `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md` | Stage 1 results archive (Task 6) |
| `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/` | Output dir (Task 5/6) |

**Modified:**
- `docs/topic4_sef_itp_framework.md` v1.0.7 → v1.0.8 (Task 0)
- `src/topic4_modeling/hr.py` from v1 stub (commit 30858df) → either delete (Task 1) or repurpose as re-export shim. **Decision: delete** — `hr_core.py` replaces it, name was too generic.
- `tests/test_topic4_modeling_hr.py` from v1 stub → **delete** (Task 1; replaced by `test_topic4_modeling_hr_core.py`)

**Existing scaffold from v1 (commit 30858df)**: `src/topic4_modeling/__init__.py` is kept (still valid as package init).

**Convention notes (codebase):**
- Tests flat in `tests/`, `test_*.py` naming
- Module names snake_case
- numba `@njit(cache=True)` requires file path (not REPL) — works fine in src files (verified)
- Existing pattern: dataclass for params, top-level functions for sim, joblib for parallelism

---

## Task 0 (PREREQUISITE — NO CODE): Framework v1.0.7 → v1.0.8 banner amendment

**Why this is Task 0:** spec source-of-truth contradiction. v1.0.7 §6.5 字面 bans HR; writing HR code first leaves the framework doc out of sync. v1.0.8 banner amendment locks the scope changes the design spec (`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md`) already records. **No HR code written before Task 0 commits.**

**Files:**
- Modify: `docs/topic4_sef_itp_framework.md`

- [ ] **Step 1: Read current §0 banner + §6.5 in framework**

Use Read tool (or `rg -n 'FHN|HR|Phase 4|v1.0.7|prerequisite' docs/topic4_sef_itp_framework.md`)
to locate the lines that contradict the spec (HR ban + Phase 4 prerequisite).

- [ ] **Step 2: Add v1.0.8 banner amendment block at top of framework doc**

Insert immediately after the existing `> 状态：` banner. Use exact wording:

```markdown
> **v1.0.8 banner amendment 2026-05-27 (user-return Phase 4 spec ratified)**：scope = Phase 4 modeling track 启动条件 + 节点动力学选择 + shaft 控制 + cluster 判据 + aniso D 进 stage 2 + smoke-first 网格 + 3-proxy sensitivity + adaptive gate + event extraction sensitivity sweep。修订 §6.5 三处：
> 1. **Phase 4 prerequisite**：v1.0.7 "Phase 1+2 H6+H1+H3 PASS + Phase 3 NULL" → v1.0.8 "framework structural prerequisite met (phantom-rank 修复 done + Phase 1 runner 落地 done) 即可启动 Phase 4 modeling track；model 输出作 mechanism exploration 层，**不替代** cohort verdict"
> 2. **节点动力学**：v1.0.7 "FHN，不加 Hindmarsh-Rose / Epileptor" → v1.0.8 "**HR (Hindmarsh-Rose) 主**，FHN / 简化 excitable unit 作 sensitivity；不模拟 HFO carrier 80–250Hz"
> 3. **Shaft 采样 + cluster 判据**：v1.0.7 未规定 → v1.0.8 "**强制 shaft 控制**：aligned / orthogonal / offset / angled / random / jittered (≥6 几何) + isotropic θ negative control；**禁止单用 KMeans k=2 作机制成功**；必须并列 split-half + odd-even + forward/reverse Spearman + rank-displacement swap_sweep + principal-curve audit；**isotropic θ + random shaft 也不能通过这套**；3 proxy (P1=x, P2=dx/dt, P3=detrend envelope) 全 PASS 才进 framework mechanism support 栏（取严格版）；L4 principal-curve audit 是 descriptive classifier 不是 hard gate"
>
> 完整 v1 spec 见 `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2。本 banner block 是 v1.0.7 doc 的 surgical 修订，§6.5 内容 v1.0.7 prose 不删除，下方 § 6.5 文末加 "→ v1.0.8 banner amendment 修订上面 3 项" cross-link。
```

- [ ] **Step 3: Add cross-link at end of §6.5 (where v1.0.7 prose lives)**

Locate the line in §6.5 ending with "`results/topic4_sef_itp/phase4_fhn_toy/`，含 figures + sanity JSON + plain-language README" (approximately line 805 in current file).

Append immediately after:

```markdown

> **→ v1.0.8 banner amendment (top of file) 修订上面 3 项**：(1) prerequisite 放宽 (2) HR 主、FHN sensitivity (3) shaft 控制 + cluster 判据 + 3-proxy sensitivity。v1.0.7 prose 保留作 audit trail。
```

- [ ] **Step 4: Verify framework doc parses (markdown sanity)**

Run: `python -c "open('docs/topic4_sef_itp_framework.md').read().count('v1.0.8 banner amendment 2026-05-27')"`
Expected: prints `1` (the banner block exists exactly once)

Run: `rg -c 'v1.0.8 banner amendment \(top of file\)' docs/topic4_sef_itp_framework.md`
Expected: `1` (cross-link exists once)

- [ ] **Step 5: Commit Task 0**

```bash
git add docs/topic4_sef_itp_framework.md
git commit -m "$(cat <<'EOF'
docs(topic4 framework): v1.0.8 banner amendment for Phase 4 modeling track

Surgical 3-point modification to §6.5:
1. Phase 4 prerequisite relaxed from "Phase 1+2+3 PASS" to "structural
   prerequisite met (phantom-rank fix done + Phase 1 runner done)";
   model outputs are mechanism exploration only, not cohort verdict.
2. Node dynamics changed from FHN-only to HR primary + FHN sensitivity.
3. Shaft sampling discipline + cluster judgement criteria added:
   >=6 shaft geometries + negative control; multi-criterion verdict
   (split-half + odd-even + forward/reverse Spearman + rank-displacement
   swap_sweep + L4 principal-curve descriptive); 3-proxy (x / dx/dt /
   detrend envelope) consistency required for framework mechanism
   support claim (strict).

v1.0.7 §6.5 prose preserved as audit trail; banner block at top of file
records authorization scope. Full spec at:
docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md v0.2

Source-of-truth fix: this commit MUST precede any Phase 4 HR
implementation code; otherwise framework doc contradicts the spec.
EOF
)"
```

**Task 0 exit criterion**: framework v1.0.7 → v1.0.8 banner committed. No HR code in this commit.

---

## Task 1: HR core (HRParams + hr_rhs + numba RK4)

**Files:**
- Delete: `src/topic4_modeling/hr.py` (v1 stub, name too generic; replaced by `hr_core.py`)
- Delete: `tests/test_topic4_modeling_hr.py` (v1 stub; replaced)
- Create: `src/topic4_modeling/hr_core.py`
- Create: `tests/test_topic4_modeling_hr_core.py`

**Math (spec §5.1)**:
```
dx/dt = y − a x³ + b x² − z + I + η
dy/dt = c − d x² − y
dz/dt = r ( s (x − x_R) − z )
```

- [ ] **Step 1: Delete v1 stubs**

```bash
git rm src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
```

- [ ] **Step 2: Write failing algebraic tests**

Create `tests/test_topic4_modeling_hr_core.py`:

```python
"""Tests for src/topic4_modeling/hr_core.py (HRParams + hr_rhs + RK4)."""

from __future__ import annotations

import numpy as np
import pytest


# ── HRParams ─────────────────────────────────────────────────────────────

def test_hr_params_defaults_match_spec():
    """Default HRParams match spec §5.1 baseline."""
    from src.topic4_modeling.hr_core import HRParams
    p = HRParams()
    assert p.a == 1.0 and p.b == 3.0 and p.c == 1.0 and p.d == 5.0
    assert p.r == 0.006 and p.s == 4.0 and p.x_R == -1.6


def test_hr_params_is_frozen():
    """HRParams is frozen (hashable for caching, immutable for safety)."""
    from src.topic4_modeling.hr_core import HRParams
    p = HRParams()
    with pytest.raises((AttributeError, Exception)):
        p.a = 2.0  # type: ignore[misc]


# ── hr_rhs algebraic invariants ──────────────────────────────────────────

def test_hr_rhs_returns_finite_3vec():
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    dx, dy, dz = hr_rhs(0.0, 0.0, 0.0, p, I=-1.6, eta=0.0)
    assert np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz)


def test_hr_rhs_y_eq_at_known_point():
    """At arbitrary x, y: dy/dt = c - d*x² - y (algebraic identity)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    x_test, y_test = 0.5, -3.0
    _, dy, _ = hr_rhs(x_test, y_test, 0.0, p, I=0.0, eta=0.0)
    expected = p.c - p.d * x_test**2 - y_test
    assert dy == pytest.approx(expected)


def test_hr_rhs_z_eq_at_x_R_zero_z_yields_zero():
    """At x = x_R and z = 0: dz/dt = r * (s * 0 - 0) = 0."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    _, _, dz = hr_rhs(p.x_R, 0.0, 0.0, p, I=0.0, eta=0.0)
    assert dz == pytest.approx(0.0)


def test_hr_rhs_eta_linear_only_in_dx():
    """eta enters dx/dt linearly, doesn't enter dy or dz."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    dx0, dy0, dz0 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.0)
    dx1, dy1, dz1 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.5)
    assert dx1 - dx0 == pytest.approx(0.5)
    assert dy1 == pytest.approx(dy0)
    assert dz1 == pytest.approx(dz0)


# ── RK4 step ─────────────────────────────────────────────────────────────

def test_rk4_step_deterministic_no_noise():
    """RK4 step with eta=0 is fully deterministic."""
    from src.topic4_modeling.hr_core import HRParams, rk4_step
    p = HRParams()
    s0 = (0.0, 0.0, 0.0)
    out1 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    out2 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    assert out1 == out2


def test_rk4_step_returns_finite_3tuple():
    from src.topic4_modeling.hr_core import HRParams, rk4_step
    p = HRParams()
    x, y, z = rk4_step((0.0, 0.0, 0.0), p, I=-1.6, eta=0.0, dt=0.05)
    assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


def test_rk4_step_matches_euler_at_tiny_dt():
    """As dt → 0, RK4 step ≈ Euler step within O(dt²)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs, rk4_step
    p = HRParams()
    s0 = (-1.0, -5.0, 0.5)
    dt = 1e-4
    x_rk4, y_rk4, z_rk4 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=dt)
    dx, dy, dz = hr_rhs(*s0, p, I=-1.6, eta=0.0)
    np.testing.assert_allclose(
        [x_rk4, y_rk4, z_rk4],
        [s0[0] + dt * dx, s0[1] + dt * dy, s0[2] + dt * dz],
        atol=1e-6,
    )


# ── numba JIT smoke ──────────────────────────────────────────────────────

def test_hr_rhs_jit_matches_python_version():
    """Numba-JIT-compiled hr_rhs gives same answer as Python ref (if exposed)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs, hr_rhs_jit
    p = HRParams()
    for x in [-1.5, 0.0, 1.0]:
        for y in [-5.0, 0.0, 2.0]:
            ref = hr_rhs(x, y, 0.5, p, I=-1.0, eta=0.1)
            # JIT version takes tuple of params (frozen dataclass → tuple unpack)
            jit_out = hr_rhs_jit(x, y, 0.5,
                                 p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
                                 I=-1.0, eta=0.1)
            np.testing.assert_allclose(ref, jit_out, rtol=1e-10)


def test_rk4_step_jit_matches_python_version():
    """JIT RK4 matches Python RK4 within 1e-10."""
    from src.topic4_modeling.hr_core import HRParams, rk4_step, rk4_step_jit
    p = HRParams()
    s0 = (-1.0, -5.0, 0.5)
    ref = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    jit_out = rk4_step_jit(s0[0], s0[1], s0[2],
                            p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
                            I=-1.6, eta=0.0, dt=0.05)
    np.testing.assert_allclose(ref, jit_out, rtol=1e-10)
```

- [ ] **Step 3: Verify tests fail**

Run: `pytest tests/test_topic4_modeling_hr_core.py -v`
Expected: 9 FAILs (`ModuleNotFoundError` or `ImportError`).

- [ ] **Step 4: Implement hr_core.py**

Create `src/topic4_modeling/hr_core.py`:

```python
"""Hindmarsh-Rose ODE core: params + rhs + RK4 step.

Provides both pure-Python reference and numba @njit JIT versions.
Hot loops (sweep / trajectory) call the _jit variants; tests verify
JIT and reference agree to numerical tolerance.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §5.1
"""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit


# ── Parameters (immutable, hashable) ──────────────────────────────────────

@dataclass(frozen=True)
class HRParams:
    """Hindmarsh-Rose parameters. Defaults from spec §5.1 baseline."""
    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float = 0.006
    s: float = 4.0
    x_R: float = -1.6


# ── Pure-Python reference (used by tests for ground-truth check) ─────────

def hr_rhs(
    x: float, y: float, z: float,
    params: HRParams,
    I: float, eta: float,
) -> tuple[float, float, float]:
    """HR ODE right-hand side (reference)."""
    p = params
    dx = y - p.a * x**3 + p.b * x**2 - z + I + eta
    dy = p.c - p.d * x**2 - y
    dz = p.r * (p.s * (x - p.x_R) - z)
    return dx, dy, dz


def rk4_step(
    state: tuple[float, float, float],
    params: HRParams,
    I: float, eta: float, dt: float,
) -> tuple[float, float, float]:
    """Classical RK4 step (reference). eta held constant over the step."""
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


# ── numba JIT variants (called from hot loops) ───────────────────────────
#
# numba does not accept dataclass directly; params expanded to scalars.
# Cache=True bakes JIT to disk so first-call cost amortized across runs.

@njit(cache=True, fastmath=True)
def hr_rhs_jit(
    x, y, z,
    a, b, c, d, r, s, x_R,
    I, eta,
):
    """HR ODE rhs (numba JIT). Same math as hr_rhs."""
    dx = y - a * x**3 + b * x**2 - z + I + eta
    dy = c - d * x**2 - y
    dz = r * (s * (x - x_R) - z)
    return dx, dy, dz


@njit(cache=True, fastmath=True)
def rk4_step_jit(
    x, y, z,
    a, b, c, d, r, s, x_R,
    I, eta, dt,
):
    """RK4 step (numba JIT)."""
    k1x, k1y, k1z = hr_rhs_jit(x, y, z, a, b, c, d, r, s, x_R, I, eta)
    k2x, k2y, k2z = hr_rhs_jit(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    k3x, k3y, k3z = hr_rhs_jit(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    k4x, k4y, k4z = hr_rhs_jit(
        x + dt * k3x, y + dt * k3y, z + dt * k3z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x_new, y_new, z_new
```

- [ ] **Step 5: Run tests, verify all 9 pass**

Run: `pytest tests/test_topic4_modeling_hr_core.py -v`
Expected: 9 PASS.

Note: first JIT call ~3-5s for numba compile + cache write. Subsequent runs use disk cache (~50ms). One-time pain.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/topic4_modeling/hr_core.py tests/test_topic4_modeling_hr_core.py
git add -u src/topic4_modeling/hr.py tests/test_topic4_modeling_hr.py
git commit -m "$(cat <<'EOF'
feat(topic4 phase4 stage1): HR core (params + ODE rhs + numba RK4)

Replaces v1 stub (hr.py / test_topic4_modeling_hr.py — deleted).
hr_core.py provides:
- HRParams frozen dataclass (spec §5.1 defaults)
- hr_rhs Python reference for algebraic tests
- hr_rhs_jit / rk4_step_jit numba @njit(cache=True, fastmath=True)
  for hot loops (sweep + trajectory)

Tests verify: param defaults + frozen, rhs finiteness + y/z/eta
algebraic invariants, RK4 determinism + Euler equivalence at tiny dt,
JIT-vs-Python agreement at 1e-10 rtol.

Numba first-call compile ~3-5s, then cached. Files: 9 tests passing.
EOF
)"
```

**Task 1 exit criterion**: hr_core.py with both Python ref + numba JIT; 9 tests green; v1 stubs deleted.

---

## Task 2: OU noise generator (numba)

**Files:**
- Create: `src/topic4_modeling/ou_noise.py`
- Create: `tests/test_topic4_modeling_ou_noise.py`

Spec §5.3: `dη/dt = −η/τ_η + σ_η · ξ(t)`, exact discrete OU update.

- [ ] **Step 1: Write tests**

Create `tests/test_topic4_modeling_ou_noise.py`:

```python
"""Tests for src/topic4_modeling/ou_noise.py."""

from __future__ import annotations

import numpy as np
import pytest


def test_ou_noise_seed_reproducibility():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    t1 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    t2 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    np.testing.assert_array_equal(t1, t2)


def test_ou_noise_different_seeds_diverge():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    t1 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    t2 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=43)
    assert not np.array_equal(t1, t2)


def test_ou_noise_stationary_variance():
    """Long sample variance ≈ sigma²."""
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(
        n_steps=200_000, dt=0.05, tau=10.0, sigma=0.1, seed=0
    )
    burn_in = 5000
    assert np.var(trace[burn_in:]) == pytest.approx(0.1**2, rel=0.15)


def test_ou_noise_autocorrelation_at_lag_tau_is_inv_e():
    """Autocorrelation at lag = tau ≈ 1/e."""
    from src.topic4_modeling.ou_noise import generate_ou_noise
    tau, dt = 10.0, 0.05
    trace = generate_ou_noise(n_steps=400_000, dt=dt, tau=tau, sigma=0.2, seed=1)
    burn_in = 5000
    x = trace[burn_in:] - trace[burn_in:].mean()
    var0 = (x * x).mean()
    lag = int(tau / dt)
    autocorr = (x[:-lag] * x[lag:]).mean() / var0
    assert autocorr == pytest.approx(np.exp(-1.0), abs=0.05)


def test_ou_noise_length_exact():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(n_steps=137, dt=0.05, tau=10.0, sigma=0.1, seed=0)
    assert trace.shape == (137,)


def test_ou_noise_zero_sigma_returns_zero_trace():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(n_steps=100, dt=0.05, tau=10.0, sigma=0.0, seed=0)
    np.testing.assert_array_equal(trace, np.zeros(100))
```

- [ ] **Step 2: Run, verify failures**

`pytest tests/test_topic4_modeling_ou_noise.py -v` → 6 FAILs

- [ ] **Step 3: Implement ou_noise.py**

Create `src/topic4_modeling/ou_noise.py`:

```python
"""Ornstein-Uhlenbeck noise generator (numba JIT).

Exact discrete OU update:
    η[t+dt] = η[t] · exp(-dt/τ) + sigma · sqrt(1 - exp(-2 dt/τ)) · N(0,1)

so that stationary variance = sigma² regardless of dt.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §5.3
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _ou_loop_jit(n_steps: int, decay: float, noise_scale: float,
                  randn: np.ndarray) -> np.ndarray:
    """numba hot loop: applies discrete OU update n_steps times."""
    eta = np.zeros(n_steps)
    for i in range(1, n_steps):
        eta[i] = eta[i - 1] * decay + noise_scale * randn[i]
    return eta


def generate_ou_noise(
    n_steps: int,
    dt: float,
    tau: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Generate OU noise trace with stationary variance sigma².

    Returns trace of shape (n_steps,) starting from eta_0 = 0.
    """
    if sigma == 0.0:
        return np.zeros(n_steps)
    rng = np.random.default_rng(seed)
    decay = float(np.exp(-dt / tau))
    noise_scale = float(sigma * np.sqrt(1.0 - decay**2))
    randn = rng.standard_normal(n_steps)
    return _ou_loop_jit(n_steps, decay, noise_scale, randn)
```

- [ ] **Step 4: Run, verify 6 pass**

`pytest tests/test_topic4_modeling_ou_noise.py -v` → 6 PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/topic4_modeling/ou_noise.py tests/test_topic4_modeling_ou_noise.py
git commit -m "$(cat <<'EOF'
feat(topic4 phase4 stage1): OU noise generator with numba JIT loop

Exact discrete OU update so stationary variance equals sigma^2
independent of dt. RNG via numpy default_rng(seed); randn array
pre-generated outside JIT loop (numba cannot use default_rng directly
without explicit typing). Hot per-step update is inside @njit loop.

Tests: seed reproducibility, different seeds diverge, stationary
variance within 15% of target, autocorrelation at lag tau ~= 1/e
within abs 0.05, length exactness, zero-sigma fast-path.
EOF
)"
```

**Task 2 exit criterion**: ou_noise.py + 6 tests green; numba JIT hot loop in `_ou_loop_jit`.

---

## Task 3: Trajectory + burst detector + regime classifier (config-driven)

**Files:**
- Create: `src/topic4_modeling/hr_config.py`
- Create: `src/topic4_modeling/hr_dynamics.py`
- Create: `tests/test_topic4_modeling_hr_dynamics.py` (unit tests — algebraic / signature / reproducibility only)
- Create: `tests/test_topic4_modeling_hr_dynamics_integration.py` (`@pytest.mark.slow` empirical tests)

Centralized thresholds (no magic numbers) in `hr_config.py` per user-return critique.

**Test discipline (v3 user-return strict catch)**: empirical regime-behavior tests ("at I=2.0 produces ≥3 bursts", "fast r produces more bursts than slow r", "at I=-3.0 stays silent") are NOT algebraic invariants — they describe what HR happens to do at a specific parameter regime, which invites the implementer to tune the model until tests pass. These go in `_integration.py` as `@pytest.mark.slow` so they're explicitly empirical observations (run during Task 6 smoke), not unit-test hard gates.

- [ ] **Step 1: Write tests**

Create `tests/test_topic4_modeling_hr_dynamics.py`:

```python
"""Tests for src/topic4_modeling/hr_dynamics.py + hr_config.py."""

from __future__ import annotations

import numpy as np
import pytest


# ── hr_config ────────────────────────────────────────────────────────────

def test_burst_thresholds_defaults():
    """BurstConfig has defaults matching spec §3 stage 1."""
    from src.topic4_modeling.hr_config import BurstConfig
    c = BurstConfig()
    assert c.x_threshold == 1.0
    assert c.min_burst_duration == 5.0
    assert c.bridge_gap == 2.0


def test_regime_thresholds_defaults():
    """RegimeConfig has defaults matching spec §3 stage 1."""
    from src.topic4_modeling.hr_config import RegimeConfig
    c = RegimeConfig()
    assert c.max_burst_duration == 100.0
    assert c.excitable_max_burst == 50.0
    assert c.excitable_min_ibi == 30.0


# ── simulate_trajectory ──────────────────────────────────────────────────

def test_simulate_trajectory_shapes():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    t, traj = simulate_trajectory(p, I=-1.6, T=10.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    n = int(10.0 / 0.05)
    assert t.shape == (n,)
    assert traj.shape == (n, 3)


def test_simulate_trajectory_reproducibility():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, a = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                sigma_ou=0.1, tau_ou=10.0, seed=42)
    _, b = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                sigma_ou=0.1, tau_ou=10.0, seed=42)
    np.testing.assert_array_equal(a, b)


# (Empirical regime-behavior tests "silent at I=-3.0" / "≥3 bursts at I=2.0"
#  moved to tests/test_topic4_modeling_hr_dynamics_integration.py per
#  v3 user-return critique — not algebraic, would invite model tuning.)


# ── detect_bursts ────────────────────────────────────────────────────────

def test_detect_bursts_zero_on_silent_trace():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    assert detect_bursts(x, t, BurstConfig()) == []


def test_detect_bursts_one_pulse():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    bursts = detect_bursts(x, t, BurstConfig())
    assert len(bursts) == 1
    assert bursts[0][0] == pytest.approx(10.0, abs=0.1)


def test_detect_bursts_hysteresis_bridges_short_dip():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 30.0)] = 1.5
    x[(t >= 19.5) & (t <= 20.5)] = 0.5  # 1ms dip < bridge_gap=2ms
    assert len(detect_bursts(x, t, BurstConfig())) == 1


def test_detect_bursts_three_separated():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 200, 0.05)
    x = np.full_like(t, -1.6)
    for t0 in [10.0, 60.0, 120.0]:
        x[(t >= t0) & (t <= t0 + 10.0)] = 1.5
    assert len(detect_bursts(x, t, BurstConfig())) == 3


def test_detect_bursts_rejects_short_noise_spike():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 11.0)] = 1.5  # 1ms < min_burst_duration=5ms
    assert detect_bursts(x, t, BurstConfig()) == []


# ── classify_regime ─────────────────────────────────────────────────────

def test_classify_regime_silent():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    assert classify_regime([], T=1000.0, cfg=RegimeConfig()) == "silent"


def test_classify_regime_excitable_sparse_short():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(100.0, 110.0), (300.0, 310.0), (700.0, 710.0)]
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "excitable"


def test_classify_regime_repetitive_burst():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(t, t + 5.0) for t in range(50, 500, 20)]  # IBI=15<30
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "repetitive-burst"


def test_classify_regime_unstable_long_burst():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    assert classify_regime([(50.0, 200.0)], T=1000.0, cfg=RegimeConfig()) == "unstable"


def test_classify_regime_unstable_takes_precedence():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(50.0, 55.0), (60.0, 65.0), (70.0, 200.0)]
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "unstable"
```

- [ ] **Step 2a: Write integration test file (empirical, slow)**

Create `tests/test_topic4_modeling_hr_dynamics_integration.py`:

```python
"""Integration tests for hr_dynamics — empirical regime-behavior observations.

These are NOT algebraic invariants. They describe what HR happens to do at
specific parameter regimes. They run during Stage 1 smoke (Task 6) as
empirical observations to be reported in archive, not as TDD unit gates.

If these xfail in Task 6, the implementation isn't wrong — HR's regime
boundaries genuinely depend on parameters and we want to discover, not
enforce, them.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_hr_silent_at_deeply_subthreshold():
    """Empirical observation: at I=-3.0 with no noise, HR rests (x.max < 0.5)."""
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-3.0, T=200.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    assert traj[:, 0].max() < 0.5, (
        f"At I=-3.0 expected silent, got x.max={traj[:, 0].max():.3f}. "
        "If unexpected, HR parameter regime differs from prior assumption — "
        "report in Stage 1 archive, do not adjust to satisfy test."
    )


@pytest.mark.slow
def test_hr_repetitive_at_high_I():
    """Empirical observation: at I=2.0, HR enters spontaneous bursting."""
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, traj = simulate_trajectory(p, I=2.0, T=300.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    x = traj[:, 0]
    ups = int(np.sum((x[:-1] < 1.0) & (x[1:] >= 1.0)))
    assert ups >= 3, (
        f"At I=2.0 expected >=3 bursts, got {ups}. If unexpected, HR regime "
        "boundary differs from prior assumption — report in Stage 1 archive."
    )


@pytest.mark.slow
def test_hr_higher_r_yields_more_bursts():
    """Empirical observation: larger r (slow var rate) → faster bursting cycle."""
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_sweep import evaluate_cell
    slow = evaluate_cell(HRParams(), I=2.0, sigma_ou=0.0, tau_ou=10.0,
                          r_override=0.003, T=1000.0, dt=0.05, seed=0)
    fast = evaluate_cell(HRParams(), I=2.0, sigma_ou=0.0, tau_ou=10.0,
                          r_override=0.012, T=1000.0, dt=0.05, seed=0)
    assert fast["n_bursts"] > slow["n_bursts"], (
        f"Expected fast r (0.012) → more bursts than slow r (0.003); "
        f"got fast={fast['n_bursts']}, slow={slow['n_bursts']}. "
        "Report in Stage 1 archive if regime boundary differs."
    )
```

Note: the `evaluate_cell` test will import from `hr_sweep` (Task 4) — it stays as XFAIL until Task 4 lands `hr_sweep.py`. That's fine for a slow integration suite.

- [ ] **Step 2: Run unit tests, verify all fail**

`pytest tests/test_topic4_modeling_hr_dynamics.py -v` → all FAIL (modules don't exist).

The integration tests in `_integration.py` are skipped by `-m "not slow"` default discipline and won't run during TDD.

- [ ] **Step 3: Implement hr_config.py**

Create `src/topic4_modeling/hr_config.py`:

```python
"""Centralized thresholds for HR burst detection + regime classification.

Centralizing here avoids magic numbers in hr_dynamics.py and lets the
sweep / CLI override per experiment.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BurstConfig:
    """Burst detection thresholds (HR time units; spec: 1 HR unit ≈ 1 ms)."""
    x_threshold: float = 1.0       # x crossing to call "above"
    min_burst_duration: float = 5.0  # below this is rejected as noise spike
    bridge_gap: float = 2.0          # gaps shorter than this don't split a burst


@dataclass(frozen=True)
class RegimeConfig:
    """Regime classification thresholds (HR time units)."""
    max_burst_duration: float = 100.0  # exceeding this = "unstable"
    excitable_max_burst: float = 50.0  # all bursts shorter than this = excitable-compatible
    excitable_min_ibi: float = 30.0    # IBI shorter than this = repetitive-burst
```

- [ ] **Step 4: Implement hr_dynamics.py**

Create `src/topic4_modeling/hr_dynamics.py`:

```python
"""Trajectory simulation + burst detection + regime classification.

simulate_trajectory uses the numba JIT kernels from hr_core + ou_noise.
detect_bursts and classify_regime stay numpy / pure Python (not hot loops).

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .hr_config import BurstConfig, RegimeConfig
from .hr_core import HRParams, rk4_step_jit
from .ou_noise import generate_ou_noise


@njit(cache=True, fastmath=True)
def _trajectory_jit(
    n_steps: int, dt: float,
    x0: float, y0: float, z0: float,
    a: float, b: float, c: float, d: float, r: float, s: float, x_R: float,
    I: float,
    eta_trace: np.ndarray,
) -> np.ndarray:
    """numba hot loop: run RK4 trajectory using pre-computed eta trace."""
    traj = np.empty((n_steps, 3))
    traj[0, 0] = x0
    traj[0, 1] = y0
    traj[0, 2] = z0
    x, y, z = x0, y0, z0
    for i in range(1, n_steps):
        x, y, z = rk4_step_jit(x, y, z, a, b, c, d, r, s, x_R,
                                I, eta_trace[i], dt)
        traj[i, 0] = x
        traj[i, 1] = y
        traj[i, 2] = z
    return traj


def simulate_trajectory(
    params: HRParams,
    I: float, T: float, dt: float,
    sigma_ou: float, tau_ou: float, seed: int,
    x0: float = -1.6, y0: float = -10.0, z0: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HR single-node trajectory for duration T, return (t, traj).

    Uses numba JIT kernels under the hood (~50ms per simulation for
    T=500, dt=0.05 = 10000 steps after first compile).
    """
    n_steps = int(T / dt)
    t = np.arange(n_steps) * dt
    eta = generate_ou_noise(n_steps, dt, tau_ou, sigma_ou, seed)
    p = params
    traj = _trajectory_jit(
        n_steps, dt, x0, y0, z0,
        p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
        I, eta,
    )
    return t, traj


def detect_bursts(
    x: np.ndarray, t: np.ndarray, cfg: BurstConfig,
) -> list[tuple[float, float]]:
    """Detect bursts in x trace via hysteresis + min-duration filter.

    Algorithm:
        1. Find above-threshold contiguous segments
        2. Bridge segments separated by gaps shorter than cfg.bridge_gap
        3. Filter by minimum duration cfg.min_burst_duration
    """
    above = x > cfg.x_threshold
    if not above.any():
        return []
    trans = np.diff(above.astype(np.int8))
    rises = np.where(trans == 1)[0] + 1
    falls = np.where(trans == -1)[0] + 1
    if above[0]:
        rises = np.concatenate([[0], rises])
    if above[-1]:
        falls = np.concatenate([falls, [len(x)]])
    segments = list(zip(rises, falls))
    # Bridge close-together segments
    bridged: list[tuple[int, int]] = []
    for seg in segments:
        if bridged and (t[seg[0]] - t[bridged[-1][1] - 1]) < cfg.bridge_gap:
            bridged[-1] = (bridged[-1][0], seg[1])
        else:
            bridged.append(seg)
    # Filter by min duration
    out: list[tuple[float, float]] = []
    for r, f in bridged:
        t_start = float(t[r])
        t_end = float(t[f - 1])
        if (t_end - t_start) >= cfg.min_burst_duration:
            out.append((t_start, t_end))
    return out


def classify_regime(
    bursts: list[tuple[float, float]],
    T: float,
    cfg: RegimeConfig,
) -> str:
    """Classify regime from burst list: silent / excitable / repetitive-burst / unstable.

    Operational definitions (spec §3 stage 1 + plan task 3):
        - silent: 0 bursts
        - unstable: any burst longer than cfg.max_burst_duration (takes precedence)
        - excitable: all bursts ≤ cfg.excitable_max_burst AND mean IBI ≥ cfg.excitable_min_ibi
        - repetitive-burst: otherwise (short IBI = spontaneous regular firing)
    """
    if not bursts:
        return "silent"
    durations = [end - start for start, end in bursts]
    if any(d > cfg.max_burst_duration for d in durations):
        return "unstable"
    if len(bursts) >= 2:
        ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
        mean_ibi = float(np.mean(ibis))
    else:
        mean_ibi = float("inf")
    if (
        all(d <= cfg.excitable_max_burst for d in durations)
        and mean_ibi >= cfg.excitable_min_ibi
    ):
        return "excitable"
    return "repetitive-burst"
```

- [ ] **Step 5: Run all Task 3 unit tests, verify pass**

`pytest tests/test_topic4_modeling_hr_dynamics.py -v` → 12 PASS (unit suite; reduced from 14 after moving 2 empirical tests to integration; third empirical test was in evaluate_cell already moved).

Also run all prior tests to confirm no regression:
`pytest tests/test_topic4_modeling_hr_core.py tests/test_topic4_modeling_ou_noise.py tests/test_topic4_modeling_hr_dynamics.py -v` → 27 total PASS (9 + 6 + 12).

Verify integration suite is gated (should be skipped by `-m "not slow"`):
`pytest tests/test_topic4_modeling_hr_dynamics_integration.py -v -m "not slow"` → expected "no tests ran in ..." or "deselected 3 items".

- [ ] **Step 6: Commit Task 3**

```bash
git add src/topic4_modeling/hr_config.py src/topic4_modeling/hr_dynamics.py \
        tests/test_topic4_modeling_hr_dynamics.py \
        tests/test_topic4_modeling_hr_dynamics_integration.py
git commit -m "$(cat <<'EOF'
feat(topic4 phase4 stage1): trajectory + burst + regime (config-driven)

hr_config.py: BurstConfig + RegimeConfig frozen dataclasses centralize
thresholds (avoid magic numbers; sweep/CLI can override).

hr_dynamics.py:
- simulate_trajectory wraps numba JIT _trajectory_jit (RK4 + pre-computed
  OU eta trace). One simulation T=500 dt=0.05 (10k steps) ~50ms after
  first compile.
- detect_bursts: hysteresis + min-duration; numpy operations.
- classify_regime: silent / excitable / repetitive-burst / unstable;
  unstable takes precedence; excitable = sparse short bursts only.

Test discipline (v3): only algebraic / signature / reproducibility
tests in unit suite (12 tests). Empirical regime-behavior observations
(silent at deeply-subthreshold I, repetitive at high I, fast r yields
more bursts than slow r) moved to _integration.py @pytest.mark.slow —
they describe what HR does at specific param regimes, not what the code
must enforce. They run during Task 6 smoke and surface unexpected
boundaries instead of being TDD hard gates.

Full unit suite: 27/27 green (9 hr_core + 6 ou_noise + 12 hr_dynamics).
EOF
)"
```

**Task 3 exit criterion**: simulate_trajectory + detect_bursts + classify_regime green; integration tests gated; no regression in prior tests.

---

## Task 4: Sweep + baseline picker (synthetic always-run + real-no-baseline → exit 1)

**Files:**
- Create: `src/topic4_modeling/hr_sweep.py`
- Create: `tests/test_topic4_modeling_hr_sweep.py`

**KEY FIX vs v1**: baseline picker has split test strategy:
- **Synthetic DataFrame unit tests** (always run, NEVER skip) verify picker logic correctness
- **Real-sweep integration test** uses `pytest.xfail` (not skip) so "no baseline" doesn't silently masquerade as green
- **CLI exit-contract test** (Task 5) verifies CLI returns code 1 when baseline=None

- [ ] **Step 1: Write tests**

Create `tests/test_topic4_modeling_hr_sweep.py`:

```python
"""Tests for src/topic4_modeling/hr_sweep.py.

Two test layers (per user-return strict catch):
  - Synthetic DataFrame unit tests for pick_excitable_baseline (always run)
  - Real fast smoke sweep (~30s) that xfail-flags if no excitable regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── evaluate_cell smoke ──────────────────────────────────────────────────

def test_evaluate_cell_returns_dict_with_regime_and_metadata():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_sweep import evaluate_cell
    result = evaluate_cell(HRParams(), I=-2.5, sigma_ou=0.0, tau_ou=10.0,
                            r_override=None, T=100.0, dt=0.05, seed=0)
    assert isinstance(result, dict)
    assert "regime" in result
    assert "n_bursts" in result
    assert result["regime"] in {"silent", "excitable", "repetitive-burst", "unstable"}


# (Empirical "r_override changes burst rate" test moved to
#  tests/test_topic4_modeling_hr_dynamics_integration.py per v3
#  user-return critique — not algebraic, would invite model tuning.)


# ── sweep_hr_parameters ──────────────────────────────────────────────────

def test_sweep_total_count_matches_cartesian_product():
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0, -1.6, 0.0],
        r_grid=[0.004, 0.006],
        sigma_grid=[0.0, 0.1],
        seeds=[0, 1],
        T=50.0, dt=0.05, n_jobs=1,
    )
    assert len(df) == 3 * 2 * 2 * 2


def test_sweep_columns_complete():
    from src.topic4_modeling.hr_sweep import sweep_hr_parameters
    df = sweep_hr_parameters(
        I_grid=[-2.0], r_grid=[0.006], sigma_grid=[0.0], seeds=[0],
        T=50.0, dt=0.05, n_jobs=1,
    )
    required = {"I", "r_used", "sigma_ou", "seed", "regime", "n_bursts"}
    assert required.issubset(df.columns)


def test_sweep_deterministic_per_cell():
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


# ── pick_excitable_baseline SYNTHETIC unit tests (always run, never skip) ─

def _row(I, r, sigma, seed, regime, n_bursts=1):
    return {"I": I, "r_used": r, "sigma_ou": sigma, "seed": seed,
            "regime": regime, "n_bursts": n_bursts,
            "mean_burst_duration": 5.0, "mean_ibi": 100.0}


def test_picker_returns_candidate_with_full_noise_window():
    """Synthetic: candidate σ has BOTH lower (~0.5σ) and upper (~1.5σ)
    neighbors that are excitable, sigma=0 silent → returns candidate."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    # Candidate σ=0.10: needs lower in [0.04, 0.06] and upper in [0.14, 0.16]
    # Grid: [0, 0.05, 0.10, 0.15] (smoke config style)
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))   # lower neighbor
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))   # candidate
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))   # upper neighbor
    # Unrelated cells
    for seed in [0, 1, 2]:
        rows.append(_row(-2.5, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(2.0, 0.006, 0.10, seed, "repetitive-burst", 10))
    df = pd.DataFrame(rows)
    baseline = pick_excitable_baseline(df)
    assert baseline is not None
    assert baseline["I_star"] == -1.3 and baseline["r_star"] == 0.006
    assert baseline["sigma_star"] == 0.10
    assert baseline["noise_robust"] is True


def test_picker_returns_none_when_no_excitable_anywhere():
    """Synthetic: all silent → returns None."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for I in [-2.5, -1.6, 0.0]:
        for sigma in [0.0, 0.05, 0.10, 0.15]:
            rows.append(_row(I, 0.006, sigma, 0, "silent", 0))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_zero_noise_not_silent():
    """Synthetic: zero-noise NOT silent → reject (candidate is not the noise-driven excitable regime)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "repetitive-burst", 10))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_upper_sigma_flips_regime():
    """Synthetic: upper neighbor flips to repetitive-burst → reject (upper-edge fragile)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
        rows.append(_row(-1.3, 0.006, 0.15, seed, "repetitive-burst", 10))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_lower_sigma_is_silent():
    """v3 NEW (user-return strict catch): lower neighbor silent → reject (lower-edge fragile).

    v2 picker missed this case — without lower-side check, a candidate σ
    sitting right at the noise threshold (where any smaller σ flips to
    silent) would pass. v3 enforces both lower AND upper noise-robustness.
    """
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "silent", 0))     # lower neighbor SILENT → fragile
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))     # candidate
        rows.append(_row(-1.3, 0.006, 0.15, seed, "excitable"))
    df = pd.DataFrame(rows)
    assert pick_excitable_baseline(df) is None


def test_picker_rejects_when_no_lower_neighbor_in_grid():
    """Candidate σ=0.05 has no lower neighbor in [0.02, 0.03] → reject."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    for seed in [0, 1, 2]:
        rows.append(_row(-1.3, 0.006, 0.0, seed, "silent", 0))
        rows.append(_row(-1.3, 0.006, 0.05, seed, "excitable"))   # candidate has no in-grid lower
        rows.append(_row(-1.3, 0.006, 0.10, seed, "excitable"))
    df = pd.DataFrame(rows)
    # sigma=0.05 has no neighbor in [0.02, 0.03] (next-lower would be 0
    # which is silent by separate criterion) → fragile lower-side → reject.
    # sigma=0.10 has no upper in [0.14, 0.16] in this small grid → reject.
    # → no candidate qualifies
    assert pick_excitable_baseline(df) is None


def test_picker_picks_median_sigma_when_multiple_candidates():
    """Synthetic: 2 candidates (σ=0.10 and σ=0.20) both satisfy full noise window → picker picks median (sorted+median = σ=0.20 by len//2)."""
    from src.topic4_modeling.hr_sweep import pick_excitable_baseline
    rows = []
    # Grid: [0, 0.05, 0.10, 0.15, 0.20, 0.30]
    # σ=0.10: lower=0.05 ∈[0.04,0.06]✓, upper=0.15 ∈[0.14,0.16]✓ → candidate
    # σ=0.15: lower=? need [0.06,0.09], none in grid → fail
    # σ=0.20: lower=0.10 ∈[0.08,0.12]✓, upper=0.30 ∈[0.28,0.32]✓ → candidate
    # σ=0.30: upper=? need [0.42,0.48], none → fail
    grid_sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    for seed in [0, 1, 2]:
        for sigma in grid_sigmas:
            regime = "silent" if sigma == 0.0 else "excitable"
            rows.append(_row(-1.3, 0.006, sigma, seed, regime,
                              0 if regime == "silent" else 1))
    df = pd.DataFrame(rows)
    baseline = pick_excitable_baseline(df)
    assert baseline is not None
    # Two candidates [0.10, 0.20]; sorted; median by len//2 = index 1 = 0.20
    assert baseline["sigma_star"] in {0.10, 0.20}, (
        f"Expected 0.10 or 0.20, got {baseline['sigma_star']}"
    )


# ── Real fast smoke sweep integration (xfail, NOT skip) ─────────────────

@pytest.mark.slow
def test_real_smoke_sweep_finds_excitable_region():
    """Real (small, ~20s) sweep — xfail if no excitable found.

    Using xfail (not skip) so this test counts as failure if it
    unexpectedly passes (XPASS) or unexpectedly fails (XFAIL marker
    visible in pytest output). Skip would silently hide a missing
    exit contract.
    """
    from src.topic4_modeling.hr_sweep import (
        sweep_hr_parameters, pick_excitable_baseline,
    )
    df = sweep_hr_parameters(
        I_grid=np.linspace(-2.0, -1.0, 6).tolist(),
        r_grid=[0.006, 0.008],
        sigma_grid=[0.0, 0.1, 0.15],
        seeds=[0, 1, 2],
        T=200.0, dt=0.05, n_jobs=1,
    )
    baseline = pick_excitable_baseline(df)
    if baseline is None:
        pytest.xfail(
            "No excitable baseline in small smoke sweep. "
            "Either parameter ranges need adjustment or HR doesn't have "
            "excitable regime — see spec §8 stage 1 fallback (FHN)."
        )
    # If baseline found, verify exit contract
    assert baseline["noise_robust"] is True
```

- [ ] **Step 2: Verify failures**

`pytest tests/test_topic4_modeling_hr_sweep.py -v -k "not slow"` → 10 FAILs (modules don't exist; slow test gated by mark)

- [ ] **Step 3: Implement hr_sweep.py**

Create `src/topic4_modeling/hr_sweep.py`:

```python
"""Stage 1 parameter sweep + baseline picker.

evaluate_cell composes sim+detect+classify into one call.
sweep_hr_parameters runs Cartesian product over (I, r, sigma, seed) with
joblib parallel.
pick_excitable_baseline applies Stage 1 exit-contract criteria.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .hr_config import BurstConfig, RegimeConfig
from .hr_core import HRParams
from .hr_dynamics import classify_regime, detect_bursts, simulate_trajectory


def evaluate_cell(
    params: HRParams,
    I: float, sigma_ou: float, tau_ou: float,
    r_override: float | None,
    T: float, dt: float, seed: int,
    burst_cfg: BurstConfig | None = None,
    regime_cfg: RegimeConfig | None = None,
) -> dict:
    """Run one cell: sim + detect bursts + classify regime."""
    if burst_cfg is None:
        burst_cfg = BurstConfig()
    if regime_cfg is None:
        regime_cfg = RegimeConfig()
    p = params if r_override is None else replace(params, r=r_override)
    t, traj = simulate_trajectory(p, I, T, dt, sigma_ou, tau_ou, seed)
    bursts = detect_bursts(traj[:, 0], t, burst_cfg)
    durations = [e - s for s, e in bursts]
    ibis = [bursts[i + 1][0] - bursts[i][1] for i in range(len(bursts) - 1)]
    regime = classify_regime(bursts, T, regime_cfg)
    return {
        "regime": regime,
        "n_bursts": len(bursts),
        "mean_burst_duration": float(np.mean(durations)) if durations else 0.0,
        "mean_ibi": float(np.mean(ibis)) if ibis else float("inf"),
        "I": I, "r_used": p.r, "sigma_ou": sigma_ou, "seed": seed, "T": T,
    }


def sweep_hr_parameters(
    I_grid: Sequence[float],
    r_grid: Sequence[float],
    sigma_grid: Sequence[float],
    seeds: Sequence[int],
    T: float, dt: float, n_jobs: int = 1,
    params_base: HRParams | None = None,
) -> pd.DataFrame:
    """Cartesian sweep. Returns DataFrame with one row per (I, r, sigma, seed)."""
    if params_base is None:
        params_base = HRParams()
    cells = [
        (I, r, sigma, seed)
        for I in I_grid for r in r_grid for sigma in sigma_grid for seed in seeds
    ]
    def _eval(cell):
        I, r, sigma, seed = cell
        return evaluate_cell(params_base, I=I, sigma_ou=sigma, tau_ou=10.0,
                              r_override=r, T=T, dt=dt, seed=seed)
    if n_jobs == 1:
        results = [_eval(c) for c in cells]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval)(c) for c in cells
        )
    return pd.DataFrame(results)


def pick_excitable_baseline(df: pd.DataFrame) -> dict | None:
    """Pick Stage 1 baseline (I*, r*, sigma*) per spec §3 stage 1 exit contract.

    Criteria (spec §3 Stage 1: regime stable to ±50% noise perturbation):
        (a) modal regime at (I, r, sigma) == "excitable", sigma > 0
        (b) same (I, r) at sigma=0 modal regime == "silent"
        (c) **lower** sigma in [0.4 * sigma, 0.6 * sigma] (closest to 0.5σ):
            modal regime still "excitable" (NOT silent, NOT repetitive-burst)
        (d) **upper** sigma in [1.4 * sigma, 1.6 * sigma] (closest to 1.5σ):
            modal regime still "excitable" (NOT repetitive-burst, NOT unstable)

    A candidate's grid must contain BOTH lower (in window) AND upper (in
    window) neighbors AND both must be modal excitable. This enforces the
    spec's "±50% noise robust" requirement and rules out threshold-edge
    baselines (v3 fix: v2 only checked upper-side, missed lower).

    Among surviving candidates, picks the one with median sigma_star
    (sorted, len//2 index).

    Returns dict {I_star, r_star, sigma_star, noise_robust=True,
    lower_sigma, upper_sigma} or None.
    """
    all_sigmas = sorted(df["sigma_ou"].unique())
    candidates = []
    for (I, r, sigma), group in df.groupby(["I", "r_used", "sigma_ou"]):
        if sigma <= 0.0:
            continue
        if group["regime"].mode().iloc[0] != "excitable":
            continue
        # (b) zero-noise silent
        zn = df[(df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == 0.0)]
        if zn.empty or zn["regime"].mode().iloc[0] != "silent":
            continue
        # (c) lower-side robustness: need sigma' in [0.4σ, 0.6σ]
        lower_window = [s for s in all_sigmas if 0.4 * sigma <= s <= 0.6 * sigma]
        if not lower_window:
            continue
        # Closest to 0.5σ (take the one with min |s - 0.5σ|)
        lo_sigma = min(lower_window, key=lambda s: abs(s - 0.5 * sigma))
        lo_cell = df[
            (df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == lo_sigma)
        ]
        if lo_cell.empty or lo_cell["regime"].mode().iloc[0] != "excitable":
            continue
        # (d) upper-side robustness: need sigma' in [1.4σ, 1.6σ]
        upper_window = [s for s in all_sigmas if 1.4 * sigma <= s <= 1.6 * sigma]
        if not upper_window:
            continue
        hi_sigma = min(upper_window, key=lambda s: abs(s - 1.5 * sigma))
        hi_cell = df[
            (df["I"] == I) & (df["r_used"] == r) & (df["sigma_ou"] == hi_sigma)
        ]
        if hi_cell.empty or hi_cell["regime"].mode().iloc[0] != "excitable":
            continue
        candidates.append({
            "I_star": float(I), "r_star": float(r), "sigma_star": float(sigma),
            "noise_robust": True,
            "lower_sigma": float(lo_sigma),
            "upper_sigma": float(hi_sigma),
        })
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["sigma_star"])
    return candidates[len(candidates) // 2]
```

- [ ] **Step 4: Run tests (exclude slow), verify pass**

`pytest tests/test_topic4_modeling_hr_sweep.py -v -k "not slow"` → 11 PASS (7 picker synthetic + 3 sweep determinism + 1 evaluate_cell smoke).

Optionally run slow integration:
`pytest tests/test_topic4_modeling_hr_sweep.py -v -m slow` → 1 PASS or 1 XFAIL with informative reason.

- [ ] **Step 5: Verify full suite no regression**

`pytest tests/test_topic4_modeling_hr_core.py tests/test_topic4_modeling_ou_noise.py tests/test_topic4_modeling_hr_dynamics.py tests/test_topic4_modeling_hr_sweep.py -v -k "not slow"`
Expected: 38 PASS (9 + 6 + 12 + 11).

- [ ] **Step 6: Commit Task 4**

```bash
git add src/topic4_modeling/hr_sweep.py tests/test_topic4_modeling_hr_sweep.py
git commit -m "$(cat <<'EOF'
feat(topic4 phase4 stage1): sweep + excitable-baseline picker

evaluate_cell composes sim+detect+classify (one row per cell call).
sweep_hr_parameters runs Cartesian product over (I, r, sigma, seed)
with joblib parallel; returns DataFrame.
pick_excitable_baseline applies 4-criterion Stage 1 exit contract
(v3: lower-side check added per user-return strict catch — v2 only
checked upper, would accept threshold-edge baselines):
  (a) modal regime "excitable" at sigma > 0
  (b) same (I, r) at sigma=0 modal "silent"
  (c) lower neighbor in [0.4σ, 0.6σ] (closest to 0.5σ): modal "excitable"
  (d) upper neighbor in [1.4σ, 1.6σ] (closest to 1.5σ): modal "excitable"
Picks median sigma_star among surviving candidates.

Test strategy (v3 user-return strict catch — v1 used pytest.skip which
silently bypassed the most important exit contract):
  - 6 synthetic DataFrame unit tests for picker logic (ALWAYS run):
    full-window pass, all-silent → None, zero-noise-not-silent reject,
    upper-flip reject, lower-silent reject (NEW), no-in-grid-lower
    reject (NEW), median pick across multiple candidates
  - 2 sweep determinism + column tests
  - 2 evaluate_cell signature / dict-shape tests
  - 1 real-smoke integration test (~20s, marked @slow) using
    pytest.xfail (NOT skip) so missing baseline does not masquerade as
    green. CLI exit-code 1 enforcement in Task 5.

Total Task 4 tests: 11 non-slow + 1 slow integration. Full suite 38/38
non-slow green (9 hr_core + 6 ou_noise + 12 hr_dynamics + 11 sweep).

Empirical "r_override changes burst rate" test moved to
test_topic4_modeling_hr_dynamics_integration.py (v3 user-return: not
algebraic, would invite model tuning).
EOF
)"
```

**Task 4 exit criterion**: picker validates both lower AND upper noise neighbors; synthetic tests always run; real sweep uses xfail not skip; full suite green.

---

## Task 5: Plotting + CLI (JSON output, exit-code 1 on no baseline)

**Files:**
- Create: `src/topic4_modeling/hr_viz.py`
- Create: `scripts/run_topic4_phase4_stage1_hr.py`
- Create: `tests/test_topic4_modeling_hr_viz.py`
- Create: `tests/test_topic4_modeling_hr_cli.py`

Output format: **JSON** for all artifacts (sweep_results.json, regime_summary.json, baseline.json). No parquet (per user catch — pyarrow availability is incidental, JSON is canonical).

- [ ] **Step 1: Write hr_viz tests**

Create `tests/test_topic4_modeling_hr_viz.py`:

```python
"""Tests for src/topic4_modeling/hr_viz.py."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def test_nullcline_x_formula():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_viz import compute_x_nullcline
    p = HRParams()
    x = np.array([0.0, 1.0, -1.0])
    out = compute_x_nullcline(x, p, z=0.5, I=-1.6)
    expected = p.a * x**3 - p.b * x**2 + 0.5 - (-1.6)
    np.testing.assert_allclose(out, expected)


def test_nullcline_y_formula():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_viz import compute_y_nullcline
    p = HRParams()
    x = np.array([0.0, 1.0, -1.0])
    out = compute_y_nullcline(x, p)
    np.testing.assert_allclose(out, p.c - p.d * x**2)


def test_plot_phase_portrait_smoke(tmp_path):
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    from src.topic4_modeling.hr_viz import plot_phase_portrait
    p = HRParams()
    _, traj = simulate_trajectory(p, I=-1.6, T=100.0, dt=0.05,
                                   sigma_ou=0.1, tau_ou=10.0, seed=0)
    fig = plot_phase_portrait(traj, p, I=-1.6)
    out = tmp_path / "phase.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_regime_map_smoke(tmp_path):
    from src.topic4_modeling.hr_viz import plot_regime_map
    rows = []
    for I in [-2.0, -1.6, -1.0]:
        for r in [0.004, 0.006]:
            for sigma in [0.0, 0.1]:
                regime = "silent" if sigma == 0.0 else "excitable"
                rows.append({"I": I, "r_used": r, "sigma_ou": sigma,
                             "seed": 0, "regime": regime, "n_bursts": 1,
                             "mean_burst_duration": 5.0, "mean_ibi": 100.0})
    df = pd.DataFrame(rows)
    fig = plot_regime_map(df)
    out = tmp_path / "regime.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 1000
```

- [ ] **Step 2: Write hr_cli test (exit-code contract)**

Create `tests/test_topic4_modeling_hr_cli.py`:

```python
"""Tests for scripts/run_topic4_phase4_stage1_hr.py exit-code contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_topic4_phase4_stage1_hr.py", *args],
        cwd=str(cwd),
        env={"PYTHONPATH": str(cwd), **dict(__import__("os").environ)},
        capture_output=True, text=True, timeout=600,
    )


def test_cli_help_works():
    """CLI --help returns 0 and mentions key flags."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(["--help"], repo_root)
    assert proc.returncode == 0
    assert "--mode" in proc.stdout
    assert "--output-dir" in proc.stdout


@pytest.mark.slow
def test_cli_no_baseline_exits_one(tmp_path):
    """CLI exits 1 when sweep produces no excitable baseline.

    Trigger by feeding a sweep grid that's all silent (very deep I).
    Outputs regime_summary.json with stage1_exit_contract_passed=false.
    """
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "stage1_no_baseline"
    proc = _run_cli(
        ["--mode", "synthetic-allsilent",
         "--output-dir", str(output_dir)],
        repo_root,
    )
    assert proc.returncode == 1, (
        f"Expected exit 1, got {proc.returncode}; stdout={proc.stdout}"
    )
    summary_path = output_dir / "regime_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["stage1_exit_contract_passed"] is False
    assert summary["baseline"] is None
```

- [ ] **Step 3: Verify both tests fail**

`pytest tests/test_topic4_modeling_hr_viz.py tests/test_topic4_modeling_hr_cli.py -v -k "not slow"` → 5 FAILs

- [ ] **Step 4: Implement hr_viz.py**

Create `src/topic4_modeling/hr_viz.py`:

```python
"""Phase portrait + nullcline + regime-map visualization.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hr_core import HRParams


# ── Nullclines (closed-form) ─────────────────────────────────────────────

def compute_x_nullcline(x_grid: np.ndarray, params: HRParams,
                         z: float, I: float) -> np.ndarray:
    """y where dx/dt = 0: y = a x³ - b x² + z - I (with eta=0)."""
    p = params
    return p.a * x_grid**3 - p.b * x_grid**2 + z - I


def compute_y_nullcline(x_grid: np.ndarray, params: HRParams) -> np.ndarray:
    """y where dy/dt = 0: y = c - d x²."""
    p = params
    return p.c - p.d * x_grid**2


# ── Phase portrait ───────────────────────────────────────────────────────

def plot_phase_portrait(trajectory: np.ndarray, params: HRParams,
                         I: float, figsize=(8.0, 6.0)) -> plt.Figure:
    """Plot x-y phase plane with trajectory + nullclines at mean(z)."""
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z_mean = float(trajectory[:, 2].mean())
    x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
    y_xnull = compute_x_nullcline(x_grid, params, z_mean, I)
    y_ynull = compute_y_nullcline(x_grid, params)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, alpha=0.4, color="C0", linewidth=0.5, label="trajectory")
    ax.plot(x_grid, y_xnull, "r-", label=f"dx/dt=0  (z≈{z_mean:.2f})")
    ax.plot(x_grid, y_ynull, "g-", label="dy/dt=0")
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


# ── Regime map heatmap ───────────────────────────────────────────────────

REGIME_COLOR = {
    "silent": "#dddddd",
    "excitable": "#4daf4a",
    "repetitive-burst": "#ff7f00",
    "unstable": "#e41a1c",
}
REGIME_ORDER = ["silent", "excitable", "repetitive-burst", "unstable"]


def plot_regime_map(sweep_df: pd.DataFrame, figsize=None) -> plt.Figure:
    """Subplot grid: rows=sigma_ou, cols=r_used, x-axis=I, colored by modal regime."""
    sigmas = sorted(sweep_df["sigma_ou"].unique())
    rs = sorted(sweep_df["r_used"].unique())
    Is = sorted(sweep_df["I"].unique())
    n_rows, n_cols = len(sigmas), len(rs)
    if figsize is None:
        figsize = (3.0 * n_cols, 1.2 * n_rows + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                              squeeze=False, sharex=True, sharey=True)
    for i, sigma in enumerate(sigmas):
        for j, r in enumerate(rs):
            ax = axes[i, j]
            cell_data = []
            for I in Is:
                sub = sweep_df[(sweep_df["sigma_ou"] == sigma)
                                & (sweep_df["r_used"] == r)
                                & (sweep_df["I"] == I)]
                cell_data.append(sub["regime"].mode().iloc[0] if not sub.empty else "silent")
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
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[r]) for r in REGIME_ORDER]
    fig.legend(handles, REGIME_ORDER, loc="upper center",
                ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize=9)
    fig.suptitle("HR single-node regime map", y=0.94, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig
```

- [ ] **Step 5: Implement CLI**

Create `scripts/run_topic4_phase4_stage1_hr.py`:

```python
#!/usr/bin/env python3
"""SEF-ITP Phase 4 Stage 1 CLI: HR single-node parameter sweep.

Modes:
  --mode smoke      Small sweep, ~5 min single-threaded; default.
  --mode full       Full sweep, ~25 min single-threaded; lock-in baseline.
  --mode synthetic-allsilent
                    Artificial test mode: tiny grid guaranteed to find no
                    excitable regime. Used by test_topic4_modeling_hr_cli
                    to verify exit-code 1 contract.

Outputs (JSON, in --output-dir):
  - sweep_results.json    : list[dict] of all sweep rows
  - regime_summary.json   : config, regime counts, baseline, exit-contract flag
  - baseline.json         : (I*, r*, sigma*) — only if baseline found
  - regime_map.png        : 3-axis heatmap
  - phase_portraits/      : baseline + 2 neighbor phase portraits

Exit codes:
  0  Stage 1 exit contract met (baseline found, noise-robust)
  1  Exit contract failed (no excitable baseline)
  2  Argparse/usage error (default argparse behavior)

Spec:  docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §3 Stage 1
Plan:  docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.topic4_modeling.hr_core import HRParams
from src.topic4_modeling.hr_dynamics import simulate_trajectory
from src.topic4_modeling.hr_sweep import (
    pick_excitable_baseline,
    sweep_hr_parameters,
)
from src.topic4_modeling.hr_viz import plot_phase_portrait, plot_regime_map

DEFAULT_OUTPUT = Path("results/topic4_sef_itp/phase4_modeling/stage1_hr_single")

CONFIGS: dict[str, dict] = {
    "smoke": {
        "I_grid": np.linspace(-2.5, 0.0, 6).tolist(),
        "r_grid": [0.004, 0.006, 0.008],
        "sigma_grid": [0.0, 0.05, 0.10, 0.15],
        "seeds": [0, 1, 2],
        "T": 200.0, "dt": 0.05,
    },
    "full": {
        "I_grid": np.arange(-2.5, 0.05, 0.1).tolist(),
        "r_grid": np.arange(0.002, 0.016, 0.001).tolist(),
        "sigma_grid": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
        "seeds": [0, 1, 2, 3, 4],
        "T": 500.0, "dt": 0.05,
    },
    # Test-only: deep-subthreshold + zero noise = guaranteed all silent
    "synthetic-allsilent": {
        "I_grid": [-3.5, -3.0],
        "r_grid": [0.006],
        "sigma_grid": [0.0, 0.05],
        "seeds": [0, 1],
        "T": 50.0, "dt": 0.05,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=list(CONFIGS), default="smoke")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cfg = CONFIGS[args.mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_portraits").mkdir(exist_ok=True)

    n_cells = (len(cfg["I_grid"]) * len(cfg["r_grid"])
                * len(cfg["sigma_grid"]) * len(cfg["seeds"]))
    print(f"[stage1] mode={args.mode}  n_jobs={args.n_jobs}  n_cells={n_cells}")

    df = sweep_hr_parameters(
        I_grid=cfg["I_grid"], r_grid=cfg["r_grid"],
        sigma_grid=cfg["sigma_grid"], seeds=cfg["seeds"],
        T=cfg["T"], dt=cfg["dt"], n_jobs=args.n_jobs,
    )
    print(f"[stage1] sweep done, n_rows={len(df)}")
    print(f"[stage1] regimes:\n{df['regime'].value_counts().to_string()}")

    # JSON sweep results (no parquet)
    (args.output_dir / "sweep_results.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2, default=str)
    )

    # Regime map. Plot failure handling (v3 fix):
    #   - synthetic-allsilent: tiny grid may legitimately not plot → log + continue
    #   - smoke / full: plot failure is an exit-contract violation → flag + exit 1
    plot_failed = False
    plot_error: str | None = None
    try:
        fig = plot_regime_map(df)
        fig.savefig(args.output_dir / "regime_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[stage1] regime_map.png saved")
    except Exception as e:  # noqa: BLE001 — intentional broad catch + exit-1 escalation
        if args.mode == "synthetic-allsilent":
            print(f"[stage1] regime_map skipped (synthetic mode): {e}")
        else:
            plot_failed = True
            plot_error = repr(e)
            print(f"[stage1] regime_map FAILED ({args.mode} mode): {e}")

    baseline = pick_excitable_baseline(df)
    contract_passed = (baseline is not None) and (not plot_failed)
    summary = {
        "mode": args.mode,
        "sweep_config": {k: v if not hasattr(v, "tolist") else v.tolist()
                          for k, v in cfg.items()},
        "regime_counts": df["regime"].value_counts().to_dict(),
        "baseline": baseline,
        "regime_map_failed": plot_failed,
        "regime_map_error": plot_error,
        "stage1_exit_contract_passed": contract_passed,
    }
    (args.output_dir / "regime_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("[stage1] regime_summary.json saved")

    if baseline is None:
        print("[stage1] EXIT CONTRACT FAILED: no excitable baseline found")
        print("[stage1] per spec §8 stage 1 fallback: try FHN-with-adaptation")
        return 1
    if plot_failed:
        print("[stage1] EXIT CONTRACT FAILED: regime_map plot failed in non-synthetic mode")
        print(f"[stage1] error: {plot_error}")
        return 1

    # Baseline + phase portraits
    (args.output_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2)
    )
    print(f"[stage1] baseline (I*, r*, σ*) = "
           f"({baseline['I_star']:.3f}, {baseline['r_star']:.4f}, "
           f"{baseline['sigma_star']:.3f})  [hand-off to Stage 2]")

    p = HRParams()
    for label, (I, r, sigma) in [
        ("baseline", (baseline["I_star"], baseline["r_star"], baseline["sigma_star"])),
        ("baseline_zero_noise", (baseline["I_star"], baseline["r_star"], 0.0)),
        ("baseline_high_noise",
         (baseline["I_star"], baseline["r_star"], baseline["sigma_star"] * 2.0)),
    ]:
        p_cell = replace(p, r=r)
        _, traj = simulate_trajectory(p_cell, I=I, T=300.0, dt=0.05,
                                       sigma_ou=sigma, tau_ou=10.0, seed=0)
        fig = plot_phase_portrait(traj, p_cell, I=I)
        fig.savefig(args.output_dir / "phase_portraits" / f"{label}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("[stage1] phase_portraits/ written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run tests, verify pass**

```bash
pytest tests/test_topic4_modeling_hr_viz.py -v
pytest tests/test_topic4_modeling_hr_cli.py -v -k "not slow"
```

Expected: 4 + 1 = 5 PASS.

Optionally run slow CLI exit-code test:
`pytest tests/test_topic4_modeling_hr_cli.py -v -k "slow"` → 1 PASS (~1 min for subprocess + sweep).

Full suite no regression check:
`pytest tests/test_topic4_modeling_hr_core.py tests/test_topic4_modeling_ou_noise.py tests/test_topic4_modeling_hr_dynamics.py tests/test_topic4_modeling_hr_sweep.py tests/test_topic4_modeling_hr_viz.py tests/test_topic4_modeling_hr_cli.py -v -k "not slow"`
Expected: 43 PASS (38 prior + 5 viz/cli).

- [ ] **Step 7: Commit Task 5**

```bash
git add src/topic4_modeling/hr_viz.py scripts/run_topic4_phase4_stage1_hr.py \
        tests/test_topic4_modeling_hr_viz.py tests/test_topic4_modeling_hr_cli.py
git commit -m "$(cat <<'EOF'
feat(topic4 phase4 stage1): plotting + CLI with JSON output + exit-code 1 on no baseline

hr_viz.py: compute_x_nullcline / compute_y_nullcline (closed form),
plot_phase_portrait (x-y plane + nullclines + start/end markers),
plot_regime_map (sigma × r subplot grid colored by modal regime).

scripts/run_topic4_phase4_stage1_hr.py: argparse CLI with three modes
(smoke / full / synthetic-allsilent for testing). All outputs in JSON
(no parquet — per user-return strict catch: pyarrow availability is
incidental, JSON is canonical):
  - sweep_results.json (per-row dicts)
  - regime_summary.json (config + counts + baseline + exit-contract flag)
  - baseline.json (only if exit contract passes)
  - regime_map.png
  - phase_portraits/baseline*.png

Exit-code contract (v3 strict; user-return strict catch):
  0 = baseline found AND regime_map plot ok (Stage 2 hand-off ready)
  1 = no excitable baseline (spec §8 fallback: try FHN), OR
      regime_map plot failed in smoke/full mode (artifact missing)
  2 = argparse error
Synthetic-allsilent mode: tiny grid may legitimately skip plot; only no-
baseline triggers exit 1 in that mode. regime_summary.json always
records both `regime_map_failed` and `stage1_exit_contract_passed` flags.

Test test_cli_no_baseline_exits_one verifies exit 1 via subprocess on
synthetic-allsilent mode.

Tests added: 4 hr_viz smoke + 1 CLI help + 1 CLI exit-1 (slow).
Full suite: 43/43 non-slow green (38 prior + 5 viz/cli).
EOF
)"
```

**Task 5 exit criterion**: hr_viz + CLI green; JSON outputs; CLI subprocess test verifies exit 1 on no baseline.

---

## Task 6: Stage 1 smoke run + archive + advisor

**Files:**
- Run (no new code)
- Create: `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md`

- [ ] **Step 1: Run smoke mode**

```bash
PYTHONPATH=. python scripts/run_topic4_phase4_stage1_hr.py --mode smoke --n-jobs 4
```

Expected:
- Prints sweep cell count + regime distribution
- ~3-5 minutes on 4 cores after numba first-compile (~5s extra)
- Writes outputs in `results/topic4_sef_itp/phase4_modeling/stage1_hr_single/`
- Exit code 0 (baseline found) or 1 (per spec §8 fallback)

- [ ] **Step 2: Manually inspect outputs**

Open in image viewer:
- `regime_map.png`: should show silent (gray) at low-I+low-noise corner, excitable (green) cells at moderate I+noise, repetitive-burst (orange) at high I. If all gray or all orange: grid too coarse, run `--mode full`.
- `phase_portraits/baseline.png`: trajectory with occasional bursts; nullclines visible
- `phase_portraits/baseline_zero_noise.png`: trajectory stuck at rest (no excursions)
- `phase_portraits/baseline_high_noise.png`: more frequent bursts than baseline

Inspect JSON:
- `cat results/topic4_sef_itp/phase4_modeling/stage1_hr_single/baseline.json`
- `cat results/topic4_sef_itp/phase4_modeling/stage1_hr_single/regime_summary.json | head -30`

If smoke regime map sensible AND baseline found AND portraits look right → proceed to Step 3.

If exit=1 or visual sanity fails → fall back to spec §8 (FHN-with-adaptation as Stage 1 plan v3). Do NOT P-hack the sweep grid.

- [ ] **Step 3: Optionally run full mode for lock-in baseline**

```bash
PYTHONPATH=. python scripts/run_topic4_phase4_stage1_hr.py --mode full --n-jobs 8
```

Expected: ~25 min on 8 cores. Higher-resolution regime map + more confident baseline pick.

- [ ] **Step 4: Run full test suite — confirm zero regression**

```bash
pytest tests/test_topic4_modeling_hr_core.py \
       tests/test_topic4_modeling_ou_noise.py \
       tests/test_topic4_modeling_hr_dynamics.py \
       tests/test_topic4_modeling_hr_sweep.py \
       tests/test_topic4_modeling_hr_viz.py \
       tests/test_topic4_modeling_hr_cli.py \
       -v
```

Expected: 43 non-slow PASS + 5 slow integration tests
(3 hr_dynamics_integration + 1 sweep_xfail + 1 cli_exit_1; each xfail
or PASS depending on whether the empirical regime boundary is met).

- [ ] **Step 5: Write archive results doc**

Create `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md` filling in actual run numbers:

```markdown
# SEF-ITP Phase 4 Stage 1 Results — Single-Node HR

> 状态：[PASS|FAIL] — 取决于实际 exit code
> 日期：[YYYY-MM-DD]
> 上游 spec：`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2
> 上游 plan：`docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md` v2
> Framework banner：v1.0.7 → v1.0.8 (commit [SHA])

## 一句话朴素话

我们把 HR 单节点的 3 个核心参数 (I, r, σ_OU) 扫遍后, 找一个参数组合让节点
在 "正常静默 + 偶尔被噪声扰动打一个 brief burst 又回到静默" 状态 (excitable regime),
作为后续 2D 网格中每个节点的基础动力学参数. 实跑找到 (I*, r*, σ*) = (___, ___, ___).

## Stage 1 退出契约

- [ ] 存在 excitable 参数子带：[PASS|FAIL]
- [ ] noise amplitude ±50% regime 不漂：[PASS|FAIL]
- [ ] 选中 baseline (I*, r*, σ*) = (___, ___, ___)
- [ ] TDD 测试全 GREEN：[N tests passed]
- [ ] CLI exit code: [0|1]

## Regime distribution

[paste df['regime'].value_counts() output]

## Artifacts

- regime_map.png
- regime_summary.json
- baseline.json (if PASS)
- phase_portraits/{baseline, baseline_zero_noise, baseline_high_noise}.png
- sweep_results.json

## Advisor consult

[paste advisor() consult result; over-claim check]

## 下一步

[PASS] → Stage 2 plan 立项 (`docs/superpowers/plans/2026-05-XX-sef-itp-phase4-stage2-2d-homogeneous.md`)
[FAIL] → spec §8 fallback: FHN-with-adaptation Stage 1 plan v3
```

Fill in bracketed values from your actual run.

- [ ] **Step 6: Advisor consult**

Call `advisor()` in your implementation session. Advisor should verify:
- regime map sensible, not all one regime
- baseline noise-robustness real (not lucky single seed)
- archive results doc honestly reflects what was found
- Stage 2 hand-off clear

If advisor flags issues → fix inline (re-run, adjust) before committing Stage 1 done.

- [ ] **Step 7: Commit Task 6**

```bash
mkdir -p docs/archive/topic4/sef_itp_phase4_v1
git add docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-27.md \
        results/topic4_sef_itp/phase4_modeling/stage1_hr_single/
git commit -m "$(cat <<'EOF'
results(topic4 phase4 stage1): HR single-node sweep + baseline lock

Stage 1 [PASS|FAIL — fill in].

Smoke run on 4 cores, ~5 min. Regime map shows expected pattern
(silent low-I, excitable middle, repetitive high-I). Baseline picker
found (I*, r*, σ*) = ([fill in]) with both-side noise-robust check.

Full unit suite: 43/43 non-slow GREEN; slow integration tests:
[hr_silent_at_deeply_subthreshold: PASS|XFAIL],
[hr_repetitive_at_high_I: PASS|XFAIL],
[hr_higher_r_yields_more_bursts: PASS|XFAIL],
[smoke_real_sweep: PASS|XFAIL],
[cli_exit_1: PASS].

Advisor consult: [paste verdict].

Hand-off to Stage 2: baseline.json contains the (I*, r*, σ*) tuple plus
the verified ±50% noise window (lower_sigma, upper_sigma) for use as
per-node baseline in the 2D homogeneous sheet.
EOF
)"
```

**Task 6 exit criterion**: archive doc written with real numbers, advisor consult done, results committed.

---

## Self-Review

**1. Spec coverage**:

| Spec §3 Stage 1 requirement | Task |
|---|---|
| Framework v1.0.7 ban on HR resolved before code | Task 0 |
| HR ODE (spec §5.1 params) | Task 1 (hr_core: HRParams + hr_rhs + JIT) |
| RK4 integrator dt=0.05 | Task 1 |
| OU noise generator (spec §5.3) | Task 2 |
| Parameter sweep over (I, r, σ, seed) | Task 4 (sweep_hr_parameters) |
| Phase portrait + nullclines | Task 5 (hr_viz) |
| Burst detector with hysteresis + min_duration | Task 3 (detect_bursts; thresholds in hr_config) |
| Regime classifier (silent/excitable/repetitive/unstable) | Task 3 (classify_regime; thresholds in hr_config) |
| Excitable subband baseline + noise-robust ±50% (lower AND upper) | Task 4 (pick_excitable_baseline — v3 fix: BOTH neighbors) |
| Output regime_map + summary JSON + selected baseline | Task 5 CLI; Task 6 actual run |
| TDD all-green | Tasks 1-5 each have green tests; Task 6 full-suite verification |
| Stage 1 exit contract: real "no baseline" = FAIL not skip | Task 4 (synthetic always-run + real-xfail) + Task 5 (CLI exit 1) |
| Stage 1 exit contract: regime_map.png must be produced | Task 5 CLI (v3 fix: plot failure → exit 1 in smoke/full mode) |
| Unit suite contains only algebraic invariants | Task 3 (empirical regime tests moved to integration `@slow` — v3 user-return) |

**2. Placeholder scan**: All code blocks complete. The only bracketed strings (`[fill in]`, `[YYYY-MM-DD]`, `[PASS|FAIL]`) are in Task 6 step 5/7 archive results doc — these are clearly marked as "fill in from actual run", not plan placeholders.

**3. Type consistency**:
- `HRParams` from `hr_core` used everywhere consistently
- `BurstConfig` / `RegimeConfig` from `hr_config` used in `hr_dynamics` + `hr_sweep` consistently
- JIT and Python `hr_rhs` / `rk4_step` have matching numerical output (verified by Task 1 tests)
- `sweep_hr_parameters` returns DataFrame; `pick_excitable_baseline` consumes same columns; `plot_regime_map` consumes same columns
- `evaluate_cell` returns dict with `regime`, `n_bursts`, etc. — matches DataFrame column expectations in Task 4 + 5

Self-review pass.

---

## Execution Handoff

Plan v3 complete. Next: invoke `superpowers:subagent-driven-development` (already loaded in the session) and dispatch Task 0 first (framework v1.0.8 banner — no code, just doc edit), then Tasks 1-6 in sequence with two-stage review per task.

The v1 stub commit (30858df) leaves `src/topic4_modeling/__init__.py` already created (still valid as package init) — Task 1 step 1 deletes the v1 `hr.py` + `tests/test_topic4_modeling_hr.py` stubs and replaces with `hr_core.py` + `tests/test_topic4_modeling_hr_core.py`.

**v3 vs v2 surgical diff summary** (so the implementer subagent doesn't re-read v2 mistakes):
- Task 4 picker enforces lower (`σ * 0.5`) AND upper (`σ * 1.5`) neighbor robustness; not just upper. Added 2 synthetic tests for lower-side rejection cases (NEW v3).
- Task 5 CLI on `regime_map` plot failure: smoke/full → exit 1 + `stage1_exit_contract_passed=false`; only synthetic-allsilent skips. Old v2 try/except silently exit 0 — fixed.
- Task 3 unit suite shrinks from 14 → 12 unit tests (2 empirical regime-behavior tests moved to `tests/test_topic4_modeling_hr_dynamics_integration.py` `@pytest.mark.slow`); Task 4 unit suite shrinks evaluate_cell from 2 → 1 (r_override empirical moved to same integration file). Plan total: 38 unit + 11 sweep + 4 viz + 1 cli help = **43 non-slow PASS expected**, plus 3 + 1 + 1 = **5 slow integration tests** run in Task 6 smoke.
- Commit messages no longer include Co-Authored-By footer (matches repo convention).
- Task 0 step 1 uses `rg` not `grep | head`.
