# M3A-A2 Abbott local+global resource — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dynamic regional inhibitory-resource slow variable (`RegionalResource`) on the SNN slow-variable path and run a staged pilot testing whether use-dependent disinhibition can autonomously carry the Stage-3 two-focus core across the A1b interictal→seizure-like boundary and back.

**Architecture:** Add a new off-by-default slow object `RegionalResource` to `src/snn_engine/slow_vars.py` (a per-region generalization of the existing `z`), driven each step by the existing `simulate_kick` slow loop (no `kick_probe.py` edit). Compute `rho(t)` + per-event R-class in a new isolated `src/sef_hfo_a2.py` (reusing `classify_event` + `spatial_bins` from `src`, never importing the M3B-edited `run_m3_kick_calibration.py`). Extend the existing `run_sef_hfo_snn_cm_spontaneous_readout.py` runner with `--a2-*` flags, and add `scripts/analyze_a2_pilot.py` mirroring the A1c/A1b analyzers. Then run Task-0 → Task-0b → Task-1 with hard stops.

**Tech Stack:** Python 3, NumPy, pytest. SNN engine in `src/snn_engine/` (load via `sys.path.insert(0, "src/snn_engine")`). No new dependencies.

## Global Constraints

(Every task's requirements implicitly include this section. Values copied verbatim from the spec `docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_dynamic_slowvars_spec_2026-06-25.md`.)

- **Off-by-default byte parity (HARD):** `RegionalResource(q=1, k_use=0)` must be bit-identical to `slow=None`. `simulate_kick` slow membrane update (`kick_probe.py:270`) equals `membrane_step` default (`:84-85`); at `q=1`, `apply_currents` returns `I_E - 1.0·I_I ≡ I_E - I_I`. Parity test is **self-contained, no fixture** (do NOT touch `tests/fixtures/a1c_parity_baseline.pkl` — M3B owns it).
- **Zero edits to the 6 watched engine files** (`src/snn_engine/{kick_probe,params,model,connectivity,connectivity_rot,lfp}.py`) → `assert_versions` (T8) stays green, **no re-bless**. `slow_vars.py` is NOT watched.
- **Zero M3B-shared files:** A2 must NOT edit `kick_probe.py`, `run_m3_kick_calibration.py`, `m3_b1_validation_recap*`, or the a1c parity fixture, and must NOT import from `run_m3_kick_calibration.py`.
- **`q_global` is a TRUE global multiplier** (scales inhibition on ALL E incl. core); `q_core`/`q_L`/`q_R` is the core-extra factor (§2.1).
- **`q_min` floor = 0.25, `tau_rec` = 5000 ms, `tau_a` = 100 ms** (PLACEHOLDER — any conclusion is contingent on these uncalibrated values; report "在这组量级下").
- **Substrate (A1b calibrated, MUST pass explicitly, not runner defaults):** `--L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 --sep-frac 0.7 --drive 0.6 --lesion twoend_equal`. Runner defaults (`core-mean 17.0, core-std 1.5, sep-frac 0.6`) are WRONG for A2.
- **`rho` boundary is NOT hardcoded 1.35/1.86** — it is the Task-0b frozen-q calibrated value `B` per anchor.
- **R-class via `classify_event`** (`src/sef_hfo_mu_basin.py`, `DEFAULT_CAPS`: R95_CAP=6.0, FAR_CAP=0.5, ACT_FLOOR=1e-3, FRONT_THRESH=0.5). R4a = sustained+front (seizure-like bridge); R4b = tonic runaway.
- **Mutual exclusion (HARD):** `RegionalResource` (a2-mode≠off) requires `--slow-var none`, no `--shunt-gaba`, `--feedback-gain 0`. Construction/runner raises otherwise.
- **Results root:** `results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/` with `figures/README.md` (Chinese, AGENTS.md standard). Per-event/per-run JSON in subdirs.
- **PILOT-FIRST hard stop:** stop after Task-0 + Task-0b (Stage A), then after Task-1 (core-only). Task-2/3 + controls are gated on user decision and NOT in this plan's executable scope.
- **Reporting discipline (§4.4):** screen-level "看起来像/不像/没看清". Forbidden: "证明发作机制 / Abbott 成立 / 慢变量导致终止". A2 may only claim "use-dependent disinhibition drives onset/excursion; recovery is emergent".

---

## Task 0 (prereq): Create the A2 worktree

**REQUIRED SUB-SKILL:** Use superpowers:using-git-worktrees.

- [ ] **Step 1: Create worktree off the hub HEAD**

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m3
git worktree add -b topic4-m3a-a2 ../topic4-m3a-a2 HEAD
cd ../topic4-m3a-a2
```

- [ ] **Step 2: Verify engine guard is green and the spec is present**

```bash
python -c "import sys; sys.path.insert(0,'src/snn_engine'); import json; from src.sef_hfo_snn_engine_guard import assert_versions; assert_versions(json.load(open('results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'))); print('T8 GREEN')"
ls docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_dynamic_slowvars_spec_2026-06-25.md
```
Expected: `T8 GREEN` and the spec path listed. All subsequent paths are relative to this worktree.

---

## Task 1: `RegionalResource` config + `apply_currents` (core_only / two_tank) + parity

**Files:**
- Modify: `src/snn_engine/slow_vars.py` (append `RegionalResourceConfig` + `RegionalResource`; do not change `SlowVars`)
- Test: `tests/test_a2_regional_resource.py`

**Interfaces:**
- Consumes: `simulate_kick(slow=...)` already calls `slow.apply_currents(I_E, I_I, labels)`, `slow.threshold(base_vth)`, `slow.step(spk, labels, dt)` (verified `kick_probe.py:256/261/285`).
- Produces: `RegionalResource(N, V_th0, core_mask_E, cfg, NE=, left_core_E=, right_core_E=)` with attrs `q_core, q_global, q_L, q_R, is_E, trace_core, trace_global` and methods `apply_currents(I_E, I_I, labels=None)->ndarray`, `threshold(V_th_base)->V_th_base`, `step(spk, labels, dt)`. `RegionalResourceConfig(mode, k_use, tau_rec, tau_a, q_min, frozen, q_core_init, q_global_init)`.

- [ ] **Step 1: Write the failing parity + partition test**

```python
# tests/test_a2_regional_resource.py
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from slow_vars import RegionalResource, RegionalResourceConfig  # noqa: E402

def _mk(N=10, NE=8, mode="two_tank", **kw):
    core_mask_E = np.zeros(N, bool); core_mask_E[:3] = True   # first 3 E cells are core
    cfg = RegionalResourceConfig(mode=mode, **kw)
    return RegionalResource(N, 18.0, core_mask_E, cfg, NE=NE), core_mask_E

def test_full_tank_is_parity():
    rr, _ = _mk(k_use=0.0)
    I_E = np.arange(10, dtype=float) + 1.0
    I_I = np.arange(10, dtype=float) * 0.5
    out = rr.apply_currents(I_E, I_I, labels=None)
    assert np.array_equal(out, I_E - I_I)   # q=1 everywhere -> exact

def test_q_global_is_true_global_multiplier():
    rr, core = _mk(); rr.q_global = 0.5; rr.q_core = 1.0
    I_E = np.ones(10); I_I = np.ones(10)
    out = rr.apply_currents(I_E, I_I, None)
    # E cells (first 8) all scaled by 0.5 (core AND surround); I cells (8,9) unscaled
    assert np.allclose(out[:8], 1.0 - 0.5 * 1.0)
    assert np.allclose(out[8:], 1.0 - 1.0)

def test_q_core_is_core_extra():
    rr, core = _mk(); rr.q_global = 1.0; rr.q_core = 0.5
    I_E = np.ones(10); I_I = np.ones(10)
    out = rr.apply_currents(I_E, I_I, None)
    assert np.allclose(out[:3], 1.0 - 0.5)    # core E: q_global*q_core = 0.5
    assert np.allclose(out[3:8], 1.0 - 1.0)   # surround E: q_global = 1.0
    assert np.allclose(out[8:], 1.0 - 1.0)    # I cells unscaled
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_a2_regional_resource.py -v`
Expected: FAIL with `ImportError: cannot import name 'RegionalResource'`.

- [ ] **Step 3: Implement config + class skeleton + `apply_currents` + `threshold`**

Append to `src/snn_engine/slow_vars.py`:

```python
@dataclass
class RegionalResourceConfig:
    """A2 Abbott-LG dynamic regional inhibitory resource (off-by-default; PLACEHOLDER params)."""
    mode: str = "two_tank"        # 'core_only' | 'two_tank' | 'per_core'
    k_use: float = 0.0            # depletion rate (0 + init=1 -> frozen-full -> byte parity)
    tau_rec: float = 5000.0       # ms recovery toward full
    tau_a: float = 100.0          # ms activity EMA
    q_min: float = 0.25
    frozen: bool = False          # Task-0b: hold q at init, no depletion/recovery
    q_core_init: float = 1.0
    q_global_init: float = 1.0


class RegionalResource:
    """Per-region inhibitory 'fuel tank' that scales the inhibition E cells receive.
    q_global scales ALL E (A1b global_ei_scale axis); q_core/q_L/q_R is the core-extra factor.
    Mirrors the SlowVars z-path (I_net = I_E - scale*I_I) generalized to regions.
    """

    def __init__(self, N, V_th0, core_mask_E, cfg: RegionalResourceConfig | None = None,
                 NE=None, left_core_E=None, right_core_E=None):
        self.cfg = cfg or RegionalResourceConfig()
        self.N = int(N)
        core = np.asarray(core_mask_E, bool)
        self.NE = int(NE) if NE is not None else int(core.size)   # # of E cells (E occupy [:NE])
        self.is_E = np.arange(self.N) < self.NE                   # E-cell mask (no `labels` dependency)
        self.core_E_idx = np.flatnonzero(core)                    # core E indices (all < NE)
        self.left_idx = np.flatnonzero(np.asarray(left_core_E, bool)) if left_core_E is not None else None
        self.right_idx = np.flatnonzero(np.asarray(right_core_E, bool)) if right_core_E is not None else None
        self.q_core = float(self.cfg.q_core_init); self.q_global = float(self.cfg.q_global_init)
        self.q_L = float(self.cfg.q_core_init); self.q_R = float(self.cfg.q_core_init)
        self._ema_core = self._ema_global = self._ema_L = self._ema_R = 0.0
        self._I_I_last = np.zeros(N); self._alpha_a = None        # alpha set lazily in step() once dt known
        self.trace_core = []; self.trace_global = []              # per-step q samples (a2_trace)
        self.trace_a_core = []; self.trace_a_global = []          # [P1-3] per-step EMA activity (auditable a_bar)
        self.trace_L = []; self.trace_R = []                      # per_core q (Task-3); stay 1.0 otherwise

    def apply_currents(self, I_E, I_I, labels=None):
        self._I_I_last = I_I
        scale = np.ones(self.N, dtype=float)
        scale[self.is_E] = self.q_global                          # global multiplier on ALL E
        if self.cfg.mode == "per_core" and self.left_idx is not None:
            scale[self.left_idx] *= self.q_L; scale[self.right_idx] *= self.q_R
        else:
            scale[self.core_E_idx] *= self.q_core                 # core-extra factor
        return I_E - scale * I_I                                  # I cells: scale=1 -> I_E - I_I

    def threshold(self, V_th_base):
        return V_th_base                                # A2 does not touch threshold (heterogeneous core via V_th_per_neuron)
```

> NOTE: the E-cell mask is `arange(N) < NE` (E cells occupy `[:NE]` in this engine). `RegionalResource` **ignores** the `labels` arg `simulate_kick` passes (spec §3.3: region info lives in the object, not in `labels`), so there is no `labels`-content dependency to verify. The runner passes `NE` at construction (Task 6).

- [ ] **Step 4: Run to verify partition + parity pass**

Run: `python -m pytest tests/test_a2_regional_resource.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/slow_vars.py tests/test_a2_regional_resource.py
git commit -m "feat(M3A-A2): RegionalResource config + apply_currents (q_global global, q_core core-extra) + parity tests"
```

---

## Task 2: `RegionalResource.step` (activity EMA + depletion ODE) + exactness/bounded/no-NaN

**Files:**
- Modify: `src/snn_engine/slow_vars.py` (add `step` + `_ode_step`)
- Test: `tests/test_a2_regional_resource.py` (append)

**Interfaces:**
- Produces: `step(spk, labels, dt)` updates `q_core`/`q_global` (two_tank) per the ODE `q ← clip(q + dt[(1-q)/tau_rec - k_use·ā·q], q_min, 1)`, with `ā` the τ_a-EMA of region E firing fraction. `frozen=True` → no-op.

- [ ] **Step 1: Write failing ODE-exactness + bounded + frozen tests**

```python
def test_ode_exact_one_step():
    rr, _ = _mk(mode="core_only", k_use=0.002, tau_rec=5000.0, tau_a=100.0)
    N, NE = 10, 8; dt = 0.1
    spk = np.zeros(N, bool); spk[:3] = True            # all 3 core E spike -> a_core=1.0
    rr.step(spk, labels=["E"]*8 + ["I"]*2, dt=dt)
    alpha = 1.0 - np.exp(-dt / 100.0)
    ema = 0.0 + alpha * (1.0 - 0.0)                     # a_core fraction = 3/3 = 1.0
    q_exp = 1.0 + dt * ((1.0 - 1.0) / 5000.0 - 0.002 * ema * 1.0)
    assert abs(rr.q_core - q_exp) < 1e-12
    assert rr.q_global == 1.0                           # core_only: q_global frozen at 1

def test_bounded_floor():
    rr, _ = _mk(mode="core_only", k_use=10.0, q_min=0.25, tau_rec=5000.0, tau_a=1.0)
    spk = np.zeros(10, bool); spk[:3] = True
    for _ in range(2000):
        rr.step(spk, ["E"]*8 + ["I"]*2, 0.1)
    assert 0.25 <= rr.q_core <= 1.0

def test_frozen_is_noop():
    rr, _ = _mk(mode="core_only", k_use=5.0, frozen=True, q_core_init=0.6)
    spk = np.ones(10, bool)
    for _ in range(100):
        rr.step(spk, ["E"]*8 + ["I"]*2, 0.1)
    assert rr.q_core == 0.6 and rr.q_global == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_a2_regional_resource.py -k "ode or bounded or frozen" -v`
Expected: FAIL (`step` missing or wrong).

- [ ] **Step 3: Implement `step` + `_ode_step`**

Add to `RegionalResource`:

```python
    def _ode_step(self, q, a_ema, dt):
        q = q + dt * ((1.0 - q) / self.cfg.tau_rec - self.cfg.k_use * a_ema * q)
        return float(min(1.0, max(self.cfg.q_min, q)))

    def step(self, spk, labels, dt):
        if self._alpha_a is None:
            self._alpha_a = 1.0 - np.exp(-dt / self.cfg.tau_a)
        spk = np.asarray(spk, bool); e_mask = self.is_E; a = self._alpha_a
        def reg_frac(idx):
            return float(spk[idx].mean()) if idx.size else 0.0
        # [P1-2/P1-3] activity EMAs are ALWAYS updated (they drive q, give a_bar, AND make frozen runs
        # auditable) — independent of frozen. core/global always; per-core when present.
        self._ema_core += a * (reg_frac(self.core_E_idx) - self._ema_core)
        self._ema_global += a * (float(spk[e_mask].mean()) - self._ema_global)
        if self.left_idx is not None:
            self._ema_L += a * (reg_frac(self.left_idx) - self._ema_L)
            self._ema_R += a * (reg_frac(self.right_idx) - self._ema_R)
        if not self.cfg.frozen:                          # [P0-2] frozen HOLDS q; dynamic updates it
            if self.cfg.mode == "per_core" and self.left_idx is not None:
                self.q_L = self._ode_step(self.q_L, self._ema_L, dt)
                self.q_R = self._ode_step(self.q_R, self._ema_R, dt)
                self.q_global = self._ode_step(self.q_global, self._ema_global, dt)
            else:
                self.q_core = self._ode_step(self.q_core, self._ema_core, dt)
                if self.cfg.mode == "two_tank":
                    self.q_global = self._ode_step(self.q_global, self._ema_global, dt)
        # [P0-2/P1-3] ALWAYS trace q + activity (frozen Task-0b runs are auditable)
        self.trace_core.append(self.q_core); self.trace_global.append(self.q_global)
        self.trace_a_core.append(self._ema_core); self.trace_a_global.append(self._ema_global)
        self.trace_L.append(self.q_L); self.trace_R.append(self.q_R)
```

- [ ] **Step 4: Run to verify pass + add no-NaN over a long synthetic run**

```python
def test_no_nan_long():
    rr, _ = _mk(mode="two_tank", k_use=0.003)
    rng = np.random.default_rng(0)
    for _ in range(5000):
        spk = rng.random(10) < 0.1
        rr.step(spk, ["E"]*8 + ["I"]*2, 0.1)
        assert np.isfinite(rr.q_core) and np.isfinite(rr.q_global)
```
Run: `python -m pytest tests/test_a2_regional_resource.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/snn_engine/slow_vars.py tests/test_a2_regional_resource.py
git commit -m "feat(M3A-A2): RegionalResource.step depletion ODE + EMA + exactness/bounded/frozen/no-nan tests"
```

---

## Task 3: per-core depletion isolation (q_L/q_R) test

**Files:** Test: `tests/test_a2_regional_resource.py` (append). No new impl (per_core already in Task 2).

- [ ] **Step 1: Write the failing isolation test**

```python
def test_per_core_isolation():
    N, NE = 12, 10
    core_mask_E = np.zeros(N, bool); core_mask_E[:4] = True   # cores = E cells 0..3
    left = np.zeros(N, bool); left[:2] = True                 # left core = E 0,1
    right = np.zeros(N, bool); right[2:4] = True              # right core = E 2,3
    cfg = RegionalResourceConfig(mode="per_core", k_use=0.05, tau_a=1.0, tau_rec=1e9)
    rr = RegionalResource(N, 18.0, core_mask_E, cfg, NE=NE, left_core_E=left, right_core_E=right)
    spk = np.zeros(N, bool); spk[:2] = True                   # ONLY left core fires
    for _ in range(500):
        rr.step(spk, ["E"]*10 + ["I"]*2, 0.1)
    assert rr.q_L < 0.99       # left depletes
    assert rr.q_R == 1.0       # right untouched (no right activity)
    assert rr.q_global < 1.0   # global sees the (left) activity
```

- [ ] **Step 2: Run to verify it fails** (if per_core wiring has a bug) or passes.

Run: `python -m pytest tests/test_a2_regional_resource.py::test_per_core_isolation -v`
Expected: PASS if Task 2 per_core branch is correct; if FAIL, fix the per_core branch in `step`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_a2_regional_resource.py
git commit -m "test(M3A-A2): per-core q_L/q_R depletion isolation (left activity does not deplete q_R)"
```

---

## Task 4: `build_regional_resource` builder + k_use-from-target + mutual-exclusion guards

**Files:**
- Create: `src/sef_hfo_a2.py` (builder + guards; R-class/rho added in Task 5)
- Test: `tests/test_a2_builder.py`

**Interfaces:**
- Produces: `build_regional_resource(N, V_th0, core_mask, NE, *, mode, q_target=None, k_use=None, tau_rec=5000.0, tau_a=100.0, q_min=0.25, a_bar=None, frozen=False, frozen_q_core=1.0, frozen_q_global=1.0, foci_masks=None) -> RegionalResource`. `k_use_from_target(q_target, a_bar, tau_rec) -> float`. `assert_a2_exclusive(slow_var, shunt_gaba, feedback_gain)`.

- [ ] **Step 1: Write failing builder/guard tests**

```python
# tests/test_a2_builder.py
import sys, os, numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from src.sef_hfo_a2 import build_regional_resource, k_use_from_target, assert_a2_exclusive

def test_k_use_from_target():
    # q* = 1/(1 + k_use*a*tau_rec)  =>  k_use = (1/q_target - 1)/(a*tau_rec)
    k = k_use_from_target(q_target=0.74, a_bar=0.05, tau_rec=5000.0)
    assert abs(k - (1.0/0.74 - 1.0) / (0.05 * 5000.0)) < 1e-12

def test_builder_derives_k_use_from_target():
    cm = np.zeros(12, bool); cm[:4] = True
    rr = build_regional_resource(12, 18.0, cm, NE=10, mode="core_only",
                                 q_target=0.74, a_bar=0.05, tau_rec=5000.0)
    assert abs(rr.cfg.k_use - k_use_from_target(0.74, 0.05, 5000.0)) < 1e-12

def test_mutual_exclusion_raises():
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="z", shunt_gaba=False, feedback_gain=0.0)
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="none", shunt_gaba=True, feedback_gain=0.0)
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="none", shunt_gaba=False, feedback_gain=8.0)
    assert_a2_exclusive(slow_var="none", shunt_gaba=False, feedback_gain=0.0)  # ok

def test_builder_requires_target_or_k_use():
    cm = np.zeros(12, bool); cm[:4] = True
    with pytest.raises(ValueError):
        build_regional_resource(12, 18.0, cm, NE=10, mode="core_only")  # neither q_target nor k_use
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_a2_builder.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `src/sef_hfo_a2.py` (builder section)**

```python
"""M3A-A2 Abbott-LG helpers: builder, k_use derivation, mutual-exclusion guard,
rho coordinate, per-event R-class, bout detection. Isolated from M3B-edited scripts.
"""
from __future__ import annotations
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "snn_engine"))
from slow_vars import RegionalResource, RegionalResourceConfig  # noqa: E402


def k_use_from_target(q_target, a_bar, tau_rec):
    """Invert the ODE fixed point q* = 1/(1 + k_use*a_bar*tau_rec) for k_use.
    a_bar = baseline core E firing fraction (Task-0). NOTE: this sets the RESTING operating point;
    bout activity overshoots beyond q* (positive feedback) — expected (spec §7 #14)."""
    if a_bar <= 0 or tau_rec <= 0:
        raise ValueError("a_bar and tau_rec must be > 0 to derive k_use")
    return (1.0 / float(q_target) - 1.0) / (float(a_bar) * float(tau_rec))


def assert_a2_exclusive(slow_var, shunt_gaba, feedback_gain):
    """A2 RegionalResource is mutually exclusive with the frozen slow-var path, the GABA shunt
    path, and A1c feedback (spec §5.4)."""
    if str(slow_var) != "none":
        raise ValueError("A2 (a2-mode != off) requires --slow-var none")
    if shunt_gaba:
        raise ValueError("A2 is incompatible with --shunt-gaba")
    if float(feedback_gain) > 0.0:
        raise ValueError("A2 is incompatible with --feedback-gain > 0 (A1c)")


def build_regional_resource(N, V_th0, core_mask, NE, *, mode, q_target=None, k_use=None,
                            tau_rec=5000.0, tau_a=100.0, q_min=0.25, a_bar=None,
                            frozen=False, frozen_q_core=1.0, frozen_q_global=1.0, foci_masks=None):
    """Construct a RegionalResource. core_mask is the length-N bool over all cells (E core True).
    For per_core, foci_masks = [left_core_mask, right_core_mask] (each length-N bool)."""
    if not frozen and k_use is None:
        if q_target is None or a_bar is None:
            raise ValueError("dynamic build needs k_use OR (q_target AND a_bar)")
        k_use = k_use_from_target(q_target, a_bar, tau_rec)
    cfg = RegionalResourceConfig(mode=mode, k_use=float(k_use or 0.0), tau_rec=tau_rec, tau_a=tau_a,
                                 q_min=q_min, frozen=frozen,
                                 q_core_init=float(frozen_q_core), q_global_init=float(frozen_q_global))
    left = foci_masks[0] if (mode == "per_core" and foci_masks is not None) else None
    right = foci_masks[1] if (mode == "per_core" and foci_masks is not None) else None
    return RegionalResource(N, V_th0, core_mask, cfg, NE=NE, left_core_E=left, right_core_E=right)  # [P0-1] NE!
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_a2_builder.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_a2.py tests/test_a2_builder.py
git commit -m "feat(M3A-A2): build_regional_resource + k_use-from-q_target + mutual-exclusion guards"
```

---

## Task 5: `rho` + per-event R-class + bout detection in `src/sef_hfo_a2.py`

**Files:**
- Modify: `src/sef_hfo_a2.py` (add `compute_rho`, `event_rclass`, `detect_bouts`)
- Test: `tests/test_a2_rho_rclass.py`

**Interfaces:**
- Produces: `compute_rho(q_core, q_global, lgr_static)->float`; `event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, bin_w, t_on, t_off, dt, foci=None, src_window_ms=10.0, far_radius=DEFAULT_FAR)->(R_class, metrics_dict, n_act, src_bin)` ([P1-4] per-event source = early-activity peak bin, tie→nearest focus); `detect_bouts(rho_bin, B)->list[(i0,i1)]`. **[P1] R-class is CANONICAL**: reuses `event_props` (normalized `peak_active` fraction + `returned`) + `classify_event` from `src/sef_hfo_mu_basin.py`, and faithfully reimplements `_bin_spike_counts_in_window`/`_spatial_extent` (run_m3_kick_calibration.py:241/375 — NOT imported; M3B edits that file) + a 50 ms front window. `spatial_bins` (src/topic4_propagation_operator.py:26) returns a **dict** `{bin_of_cell, bin_centers}`.

- [ ] **Step 1: Write failing rho + R-class + bout tests**

```python
# tests/test_a2_rho_rclass.py
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from src.sef_hfo_a2 import compute_rho, event_rclass, detect_bouts

def test_rho_full_tank_equals_lgr():
    assert abs(compute_rho(1.0, 1.0, 1.16) - 1.16) < 1e-12
    assert abs(compute_rho(0.5, 1.0, 1.0) - 2.0) < 1e-12       # q_core 0.5 doubles rho
    assert abs(compute_rho(0.5, 0.5, 1.0) - 4.0) < 1e-12

def test_detect_bouts_above_boundary():
    rho = np.array([1.0,1.0,1.4,1.5,1.4,1.0,1.0,1.5,1.5,1.0])
    bouts = detect_bouts(rho, B=1.35)
    assert bouts == [(2, 4), (7, 8)]

def test_rclass_active_peak_is_fraction_and_returned_local_not_R4():
    # [P1] active_peak must be an active FRACTION (event_props), not a raw spike count; a returned,
    # source-localized event must classify R2/R3 (not R4). bin_centers is 2D (spatial_bins contract).
    nb = 4; n_bins = nb * nb
    bin_centers = np.array([[j, i] for i in range(nb) for j in range(nb)], float)   # 2D, row-major
    bin_of_cell = np.repeat(np.arange(n_bins), 2)             # 32 E cells, 2 per bin
    NEc = bin_of_cell.size; dt, bin_w, nsteps = 0.1, 5.0, 400  # 40 ms record
    spk = np.zeros((nsteps, NEc), bool); fire_bin = 5
    spk[50:150, bin_of_cell == fire_bin] = True              # bin 5 fires 5-15 ms, then quiet
    nbw = int(bin_w / dt); ntb = nsteps // nbw
    af = spk[:ntb * nbw].reshape(ntb, nbw, NEc).mean(axis=(1, 2))   # active FRACTION per bin
    rcls, m, n_act, src = event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, bin_w,
                                       t_on=5.0, t_off=15.0, dt=dt, foci=[bin_centers[fire_bin]])
    assert src == fire_bin                                    # [P1-4] source = early-activity peak bin
    assert 0.0 <= m["active_peak"] <= 1.0                     # FRACTION (~2/32), NOT a raw count
    assert m["returned"] is True and m["runaway"] is False    # ends at 15 ms << 40 ms record
    assert m["far_ea"] == 0.0                                  # all mass in the source bin
    assert rcls in ("R2", "R3")
    assert set(m) == {"event_detected","returned","runaway","r95_ea","far_ea",
                      "active_peak","sustained_front_score"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_a2_rho_rclass.py -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `compute_rho`, `event_rclass`, `detect_bouts`**

Add to `src/sef_hfo_a2.py`:

```python
sys.path.insert(0, os.path.dirname(__file__))
from sef_hfo_mu_basin import classify_event, DEFAULT_CAPS, event_props  # noqa: E402

DEFAULT_FAR = 6.0   # far-radius (mm) for far_ea, matches A1 spatial-extent default


def compute_rho(q_core, q_global, lgr_static):
    """Dynamic image of A1b's local_global_ratio along the inhibition axis (spec §2.5)."""
    return float(lgr_static) / (float(q_core) * float(q_global))


def _bin_spike_counts(spk, bin_of_cell, n_bins, lo_step, hi_step):
    """Total E spikes per bin in [lo_step, hi_step) timesteps. EXACT mirror of
    run_m3_kick_calibration._bin_spike_counts_in_window:241 (NOT imported — M3B edits that script)."""
    per_cell = np.asarray(spk[lo_step:hi_step], bool).sum(axis=0).astype(float)        # (NE,)
    return np.bincount(np.asarray(bin_of_cell, int), weights=per_cell, minlength=n_bins)


def _spatial_extent(net_bins, bin_centers, src_bin, far_radius):
    """(n_activated, r95_mm, far_field_frac) over NON-source bins. EXACT mirror of
    run_m3_kick_calibration._spatial_extent:375-432; the source bin is excluded from all three.
    bin_centers is 2D [n_bins, 2] (spatial_bins contract)."""
    bc = np.asarray(bin_centers, float)
    radii = np.linalg.norm(bc - bc[src_bin], axis=1)
    non_source = np.ones(len(net_bins), bool); non_source[src_bin] = False
    nonsrc = net_bins[non_source]
    floor = max(2.0, 0.05 * float(nonsrc.max() if nonsrc.size else 0.0))
    activated = non_source & (net_bins > floor)
    n_act = int(activated.sum())
    if n_act > 0:
        order = np.argsort(radii[activated]); r_s = radii[activated][order]; w_s = net_bins[activated][order]
        cw = np.cumsum(w_s)
        r95 = float(np.interp(0.95 * cw[-1], cw, r_s)) if cw[-1] > 0 else float(np.percentile(radii[activated], 95))
    else:
        r95 = 0.0
    total_ns = float(net_bins[non_source].sum())
    far = float(net_bins[non_source & (radii > far_radius)].sum()) / total_ns if total_ns > 0 else 0.0
    return n_act, r95, far


def event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, bin_w, t_on, t_off, dt,
                 foci=None, src_window_ms=10.0, far_radius=DEFAULT_FAR):
    """[P1] CANONICAL per-event R-class. [P1-4] source bin = the EARLY-activity peak bin (first
    ~src_window_ms after t_on), NOT the geometric center — so a focus-local event is not misread as
    spreading; ties break to the nearest focus. peak_active via event_props (FRACTION); front via a
    real 50ms window; spatial extent = exact A1 _spatial_extent. `af` = runner active-fraction trace
    (bin_w bins); `spk` = raw (n_steps, NE) bool; `foci` = [[x,y],...] focus centers (tie-break).
    Returns (R_class, metrics_dict, n_activated, src_bin)."""
    s, e = int(t_on / bin_w), int(t_off / bin_w)
    ep = event_props(af, (s, e), bin_w, len(af))                      # peak_active = max FRACTION; returned
    lo, hi = int(round(t_on / dt)), int(round(t_off / dt))
    # [P1-4] source bin from the early window's per-bin spike peak (tie -> nearest focus)
    early_hi = max(int(round(min(t_off, t_on + src_window_ms) / dt)), lo + 1)
    early = _bin_spike_counts(spk, bin_of_cell, n_bins, lo, early_hi)
    cand = np.flatnonzero(early == early.max())
    if foci is not None and len(cand) > 1:
        bc = np.asarray(bin_centers, float); fc = np.asarray(foci, float)
        src_bin = int(cand[int(np.argmin([min(np.linalg.norm(bc[b] - f) for f in fc) for b in cand]))])
    else:
        src_bin = int(cand[0])
    bins = _bin_spike_counts(spk, bin_of_cell, n_bins, lo, hi)
    n_act, r95, far = _spatial_extent(bins, bin_centers, src_bin, far_radius)
    tail_lo = max(t_on, t_off - 50.0)                                 # last 50 ms (STABLE window)
    tlo, thi = int(round(tail_lo / dt)), int(round(t_off / dt))
    tail_bins = _bin_spike_counts(spk, bin_of_cell, n_bins, tlo, thi)
    front_score = 1.0 - int(np.sum(tail_bins > 0)) / n_bins
    m = {"event_detected": True, "returned": bool(ep["returned"]), "runaway": bool(ep["sustained"]),
         "r95_ea": float(r95), "far_ea": float(far), "active_peak": float(ep["peak_active"]),
         "sustained_front_score": float(front_score)}
    return classify_event(m, DEFAULT_CAPS), m, n_act, src_bin


def detect_bouts(rho_bin, B):
    """Maximal contiguous index ranges where rho_bin >= B (the seizure-like band entry)."""
    above = np.asarray(rho_bin, float) >= float(B)
    bouts = []; i = 0; n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            bouts.append((i, j)); i = j + 1
        else:
            i += 1
    return bouts
```

> [P1] NOTE: `peak_active` comes from `event_props` (the normalized active-FRACTION peak the A1 producer feeds `classify_event`), the front uses a real `[t_off-50, t_off]` window, and `_spatial_extent`/`_bin_spike_counts` are byte-faithful copies of the A1 helpers (run_m3_kick_calibration.py:241/375) — so A2's R-class is the SAME instrument as A1's; only the spatial helpers are copied (not imported) to avoid the M3B-edited script. `event_rclass` consumes the runner's `af` trace + raw `spk` + `spatial_bins` dict + ms `t_on/t_off` (Task 6).

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_a2_rho_rclass.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sef_hfo_a2.py tests/test_a2_rho_rclass.py
git commit -m "feat(M3A-A2): compute_rho + event_rclass (reuse classify_event) + detect_bouts"
```

---

## Task 6: Runner CLI extension (`--a2-*`) + a2 readout block + a2_trace + per-event R_class

**Files:**
- Modify: `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` (add `--a2-*` args near the existing `--slow-var` block ~line 487; add the A2 branch in the slow-construction region ~line 603; add a2 readout emission near `_activity`/`readout_{tag}.json` ~lines 658/980)
- Test: `tests/test_a2_runner_smoke.py`

**Interfaces:**
- Consumes: `build_regional_resource`, `assert_a2_exclusive`, `compute_rho`, `event_rclass`, `detect_bouts` (Tasks 4-5); existing `build_lesion_vth` (returns `(vth, core_mask, foci_xy, core_masks_per_focus)`), `simulate_kick(..., slow=, V_th_per_neuron=vth)`, `spatial_bins` (`src.topic4_propagation_operator`), `local_global_ratio` (`src.sef_hfo_a1b`), `a1b_weight_lesion`.
- Produces: `readout_{tag}.json` with new `a2` block + per-event `R_class`/metrics; `a2_trace_{tag}.npz` with `q_core_bin,q_global_bin,rho_bin,a_core_bin,a_global_bin,rate_E_hz`.

- [ ] **Step 1: Confirm the runner-side contract (one-time check before wiring Steps 5-7)**

`RegionalResource` does NOT use `labels` (it uses `is_E = arange(N) < NE`), so there is no labels check. Read `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py` around the `simulate_kick` call (lines ~600-640) and the event loop (~730-763) to confirm the variable names the A2 branch references: `NE`, `center`, `axis_unit`, `half`, `posE`, the `res['E_spk_bool']` orientation (`(T, NE)` vs `(NE, T)`), and the `events` element shape (`(b0,b1)` bin indices vs `t_on/t_off` ms). Confirm `spatial_bins`' return contract:
```bash
python - <<'PY'
import inspect
from src.topic4_propagation_operator import spatial_bins
print(inspect.signature(spatial_bins))
PY
```
These contracts are **pinned (confirmed against the live runner 2026-06-26)**: `spatial_bins(posE, n_bins_per_axis)` returns `dict{bin_of_cell, bin_centers}`; events carry ms `t_on/t_off`; `spk` is `(n_steps, NE)`; `summary` is built at runner:937. Step 1 is now just a quick re-confirm before wiring Steps 5-7.

- [ ] **Step 2: Write the failing runner smoke + off-parity test**

```python
# tests/test_a2_runner_smoke.py
import subprocess, sys, os, json, glob, numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
COMMON = ["--L","20","--density","100","--theta","45","--core-mean","17.5","--core-std","1.0",
          "--core-r","1.5","--sep-frac","0.7","--drive","0.6","--lesion","twoend_equal",
          "--T","800","--seed","1"]

def _run(extra, out):
    cmd = [sys.executable, "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
           *COMMON, "--out", out, "--tag", "smoke"] + extra
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, timeout=600)

def test_a2_off_is_parity_with_no_a2(tmp_path):
    a, b = str(tmp_path/"off"), str(tmp_path/"none")
    _run(["--a2-mode","off"], a)
    _run([], b)
    ra = json.load(open(glob.glob(a+"/readout_*.json")[0]))
    rb = json.load(open(glob.glob(b+"/readout_*.json")[0]))
    assert ra["activity"] == rb["activity"]                  # a2-mode off changes nothing

def test_a2_core_only_emits_a2_block(tmp_path):
    out = str(tmp_path/"co")
    _run(["--a2-mode","core_only","--a2-k-use","0.003","--a2-tau-rec","5000",
          "--a2-tau-a","100","--a2-q-min","0.25","--dump-a2-trace"], out)
    r = json.load(open(glob.glob(out+"/readout_*.json")[0]))
    assert r["a2"]["mode"] == "core_only"
    assert "rho_static" in r["a2"] and "seizure_boundary" in r["a2"]
    npz = np.load(glob.glob(out+"/a2_trace_*.npz")[0])
    assert {"q_core_bin","q_global_bin","rho_bin","rate_E_hz"} <= set(npz.files)
    if r["events"]:
        assert "R_class" in r["events"][0]
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_a2_runner_smoke.py -v`
Expected: FAIL (`--a2-mode` unrecognized).

- [ ] **Step 4: Add the `--a2-*` argparse block** (after the `--slow-var`/`--slow-level` lines ~487)

```python
ap.add_argument('--a2-mode', choices=['off', 'core_only', 'two_tank', 'per_core'], default='off')
ap.add_argument('--a2-k-use', type=float, default=None)
ap.add_argument('--a2-q-target', type=float, default=None)
ap.add_argument('--a2-a-bar', type=float, default=None)      # baseline core E firing frac (Task-0)
ap.add_argument('--a2-tau-rec', type=float, default=5000.0)
ap.add_argument('--a2-tau-a', type=float, default=100.0)
ap.add_argument('--a2-q-min', type=float, default=0.25)
ap.add_argument('--a2-frozen', action='store_true')          # Task-0b: hold q at product, no depletion
ap.add_argument('--a2-frozen-qcore', type=float, default=1.0)
ap.add_argument('--a2-frozen-qglobal', type=float, default=1.0)
ap.add_argument('--a2-boundary', type=float, default=None)   # Task-0b-calibrated rho seizure boundary B
ap.add_argument('--dump-a2-trace', action='store_true')
```

- [ ] **Step 5: Build the RegionalResource in the slow-construction region** (replace/extend the `if a.slow_var != 'none':` block ~603 so A2 takes precedence and is mutually exclusive)

```python
slow = None
if a.a2_mode != 'off':
    from src.sef_hfo_a2 import build_regional_resource, assert_a2_exclusive
    assert_a2_exclusive(a.slow_var, a.shunt_gaba, a.feedback_gain)
    # core_mask, core_masks_per_focus from build_lesion_vth return (vth, core_mask, foci_xy, core_masks)
    _foci = core_masks if a.a2_mode == 'per_core' else None
    slow = build_regional_resource(
        NE + NI, p.V_th, core_mask, NE, mode=a.a2_mode,
        q_target=a.a2_q_target, k_use=a.a2_k_use, a_bar=a.a2_a_bar,
        tau_rec=a.a2_tau_rec, tau_a=a.a2_tau_a, q_min=a.a2_q_min,
        frozen=a.a2_frozen, frozen_q_core=a.a2_frozen_qcore, frozen_q_global=a.a2_frozen_qglobal,
        foci_masks=_foci)
elif a.slow_var != 'none':
    from src.sef_hfo_slowvars_quasistatic import build_frozen_slowvars
    _slow_kw = {'z': {'z': a.slow_level}, 'phi': {'phi_offset': a.slow_level, 'vth_field': vth},
                'gK': {'gK': a.slow_level}}[a.slow_var]
    slow = build_frozen_slowvars(NE + NI, p.V_th, **_slow_kw)
```

> `build_lesion_vth(...)` is already unpacked as `vth, core_mask, foci_xy, core_masks` (the runner uses `core_mask` at :640 and `core_masks` at :912). **The `simulate_kick(...)` call (runner:608-613) must dump the currents for the [P1-3] magnitude readout** — change its `dump_drive=(a.dump_drive or a.dump_fb)` to `dump_drive=(a.dump_drive or a.dump_fb or a.a2_mode != 'off')` so `res['I_I_peak']`/`res['I_E_peak']` exist (Step 6).

- [ ] **Step 6: Emit the `a2` block into the EXISTING `summary` (runner:937) + [P1-3] magnitude readout + `a2_trace`**

`RegionalResource` appends `(q_core, q_global)` each `step` (Tasks 1-2; **frozen Task-0b runs append the constant value too** [P0-2]). [P0-3] The runner builds `summary` at **line 937** (already including `config['local_global_ratio']` at :948) and dumps it at **line 980** — insert the a2 block **between** them (right before `json.dump`). [P1-boundary] `B` is passed via `--a2-boundary` **only after Task-0b locks it**; do NOT hardcode 1.35 — band stats are computed only when `B` is given (frozen Task-0b runs pass no `B`).

```python
# insert just before `json.dump(summary, ...)` at runner:980
if a.a2_mode != 'off':
    qc = np.asarray(slow.trace_core, float); qg = np.asarray(slow.trace_global, float)
    ac = np.asarray(slow.trace_a_core, float); ag = np.asarray(slow.trace_a_global, float)  # [P1-3] EMA activity
    lgr = summary['config']['local_global_ratio']          # the verified existing field (runner:948)
    rho_t = lgr / (qc * qg)                                 # full step-resolution rho(t)
    a2 = dict(mode=a.a2_mode, k_use=round(float(slow.cfg.k_use), 8), q_target=a.a2_q_target,
              tau_rec=a.a2_tau_rec, tau_a=a.a2_tau_a, q_min=a.a2_q_min, frozen=bool(a.a2_frozen),
              rho_static=round(float(lgr), 4), seizure_boundary=a.a2_boundary,   # None until Task-0b locks
              q_core_min=round(float(qc.min()), 4), q_global_min=round(float(qg.min()), 4),
              q_core_end=round(float(qc[-1]), 4), q_global_end=round(float(qg[-1]), 4),
              rho_peak=round(float(rho_t.max()), 4), rho_p95=round(float(np.percentile(rho_t, 95)), 4),
              # [P1-2] DIRECT a_bar (mean per-step E active fraction) — NOT Hz×duty; feeds Task-1 k_use derivation
              a_core_mean=round(float(ac.mean()), 6), a_global_mean=round(float(ag.mean()), 6))
    if a.a2_boundary is not None:                           # band stats ONLY once B is calibrated
        B = float(a.a2_boundary)
        a2['n_boundary_crossings'] = len(detect_bouts(rho_t, B))
        a2['frac_time_seizure_band'] = round(float(np.mean(rho_t >= B)), 4)
    # [P1-3] magnitude sanity: is the inhibition q scales actually LOAD-BEARING on core E? (q is
    # dimensionless so no unit error, but if |I_I| << |I_E| the scaling is inert -> an R-stay is an artifact)
    if 'I_I_peak' in res:
        _IIe = np.abs(res['I_I_peak'][:NE]); _IEe = np.abs(res['I_E_peak'][:NE]); _c = core_mask[:NE]
        a2.update(I_I_core_median=round(float(np.median(_IIe[_c])), 4),
                  I_E_core_median=round(float(np.median(_IEe[_c])), 4),
                  I_I_surround_median=round(float(np.median(_IIe[~_c])), 4),
                  I_I_over_I_E_core=round(float(np.median(_IIe[_c]) / max(np.median(_IEe[_c]), 1e-9)), 4))
    summary['a2'] = a2
    if a.dump_a2_trace:
        _rate_hz = np.asarray(res['rate_E'], float)
        np.savez_compressed(os.path.join(out_dir, f'a2_trace_{tag}.npz'),
                            q_core_bin=qc.astype(np.float32), q_global_bin=qg.astype(np.float32),
                            a_core_bin=ac.astype(np.float32), a_global_bin=ag.astype(np.float32),  # [P1-3]
                            rho_bin=rho_t.astype(np.float32), rate_E_hz=_rate_hz.astype(np.float32))
```
(`detect_bouts` runs on the full-resolution `rho_t`; bouts are contiguous index ranges, so the count is the boundary-crossing count. No `local_global_ratio` import is needed — read `summary['config']['local_global_ratio']`.)

- [ ] **Step 7: Add per-event `R_class`** in the EXISTING `for ev in events:` loop (runner:740-763)

[P0-3] Events are dicts with **`t_on`/`t_off` in ms** + `returned` (runner:740-741); `spk` is **(n_steps, NE)** (runner:643); `af`/`bin_w` come from `active_fraction` (runner:616). Compute the spatial bins ONCE before the loop, then call `event_rclass` per event:

```python
# BEFORE the `for ev in events:` loop (posE/foci already exist):
if a.a2_mode != 'off':
    from src.topic4_propagation_operator import spatial_bins
    from src.sef_hfo_a2 import event_rclass
    _sb = spatial_bins(posE, a.n_bins_per_axis)            # dict{bin_of_cell, bin_centers}; default 5x5
    _boc = _sb['bin_of_cell']; _bc = _sb['bin_centers']; _nbins = len(_bc)
# INSIDE the loop, right after `ev_recs.append(dict(...))`:  [P1-4] src bin is PER-EVENT (early activity)
if a.a2_mode != 'off':
    rcls, m, _, _src = event_rclass(af, spk, _boc, _nbins, _bc, bin_w, ev['t_on'], ev['t_off'], DT,
                                    foci=foci)            # foci = build_lesion_vth's [neg_xy, pos_xy]
    ev_recs[-1]['R_class'] = rcls
    ev_recs[-1].update(r95_ea=round(m['r95_ea'], 3), far_ea=round(m['far_ea'], 3),
                       active_peak=round(m['active_peak'], 5),
                       sustained_front_score=round(m['sustained_front_score'], 3), src_bin=int(_src))
```
> [P0-3] `spatial_bins(posE, a.n_bins_per_axis)` returns a **dict** and takes `n_bins_per_axis` (existing runner arg, default 5). `ev['t_on']/ev['t_off']` are **ms**. [P1-4] `event_rclass` derives the source bin per event from the early window (`foci` = the two focus centers for tie-break) — no fixed center src bin.

- [ ] **Step 8: Run smoke + off-parity tests**

Run: `python -m pytest tests/test_a2_runner_smoke.py tests/test_a2_regional_resource.py tests/test_a2_builder.py tests/test_a2_rho_rclass.py -v`
Expected: PASS (all). Also re-run the engine guard:
```bash
python -c "import sys; sys.path.insert(0,'src/snn_engine'); import json; from src.sef_hfo_snn_engine_guard import assert_versions; assert_versions(json.load(open('results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json'))); print('T8 GREEN')"
```
Expected: `T8 GREEN` (no watched engine file edited).

- [ ] **Step 9: Add the heterogeneous-core-live test** (invariant #11)

```python
# tests/test_a2_runner_smoke.py (append)
def test_core_is_heterogeneous_under_a2(tmp_path):
    out = str(tmp_path/"hc")
    _run(["--a2-mode","core_only","--a2-k-use","0.0","--dump-fullfield"], out)
    # full-tank k_use=0 must still show CORE events (nonzero), i.e. the heterogeneous vth is live;
    # if V_th_per_neuron were dropped, the homogeneous substrate would be ~silent (A1 failure).
    r = json.load(open(glob.glob(out+"/readout_*.json")[0]))
    assert r["activity"]["core_E_rate_mean_hz"] > r["activity"]["surround_E_rate_mean_hz"]
```
Run: `python -m pytest tests/test_a2_runner_smoke.py::test_core_is_heterogeneous_under_a2 -v`
Expected: PASS (core hotter than surround → core field is live).

- [ ] **Step 10: Commit**

```bash
git add scripts/run_sef_hfo_snn_cm_spontaneous_readout.py src/snn_engine/slow_vars.py tests/test_a2_runner_smoke.py
git commit -m "feat(M3A-A2): runner --a2-* CLI + a2 readout block + a2_trace + per-event R_class + heterogeneous-core test"
```

---

## Task 7: `scripts/analyze_a2_pilot.py` — per-run verdict + bout gate + status JSON

**Files:**
- Create: `scripts/analyze_a2_pilot.py`
- Test: `tests/test_a2_analyzer.py`

**Interfaces:**
- Produces: `_run_verdict(readout)->str` in {R-stay, R-excursion, R-runaway}; writes `status_a2_pilot.json`. Reuses A1c `TAIL_GATE=1.5`/`IGNITE_PEAK=0.05` + A1b `_state` thresholds.

- [ ] **Step 1: Write the failing verdict test** (synthetic readouts)

```python
# tests/test_a2_analyzer.py
import importlib.util, os
spec = importlib.util.spec_from_file_location("aza", os.path.join(
    os.path.dirname(__file__), "..", "scripts", "analyze_a2_pilot.py"))
aza = importlib.util.module_from_spec(spec); spec.loader.exec_module(aza)

def _ro(rho_peak, tail, max_R, frac_seiz, returned_after):
    return {"a2": {"seizure_boundary": 1.35, "rho_peak": rho_peak,
                   "frac_time_seizure_band": frac_seiz, "q_core_end": (0.9 if returned_after else 0.3)},
            "activity": {"tail_to_baseline_ratio": tail, "active_E_fraction_peak": 0.2,
                         "peak_E_rate_hz": 10.0},
            "events": [{"R_class": max_R}]}

def test_stay_when_below_boundary():
    assert aza._run_verdict(_ro(1.1, 1.0, "R3", 0.0, True)) == "R-stay"

def test_excursion_when_crosses_R4a_and_returns():
    assert aza._run_verdict(_ro(1.5, 1.2, "R4a", 0.2, True)) == "R-excursion"

def test_runaway_when_crosses_but_not_returns():
    assert aza._run_verdict(_ro(1.7, 3.0, "R4b", 0.6, False)) == "R-runaway"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_a2_analyzer.py -v`
Expected: FAIL (`analyze_a2_pilot.py` missing).

- [ ] **Step 3: Implement `scripts/analyze_a2_pilot.py`**

```python
"""M3A-A2 pilot analyzer — per-run R-stay/R-excursion/R-runaway verdict + bout R-class gate.
Mirrors analyze_a1c_pilot.py / analyze_m3a_a1b.py. Reads readout_*.json; writes status_a2_pilot.json.
"""
import os, sys, json, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAIL_GATE = 1.5          # A1c absolute tail
IGNITE_PEAK = 0.05       # A1c ignition
SEIZURE_RCLASSES = ("R4a", "R3")   # R4a primary; R3->R4a-like allowed (spec §4.3.2)


def _run_verdict(r):
    """R-stay / R-excursion / R-runaway from one readout (spec §4.2 + §4.3.2 R-class gate)."""
    a2 = r.get("a2", {}); act = r.get("activity", {})
    B = a2.get("seizure_boundary", 1.35)
    crossed = a2.get("rho_peak", 0.0) >= B
    ignited = (act.get("active_E_fraction_peak", 0.0) >= IGNITE_PEAK) or (act.get("peak_E_rate_hz", 0.0) >= 3.0)
    returned = act.get("tail_to_baseline_ratio", 1e9) <= TAIL_GATE
    # [P1-boundary] recovery = tanks refilled ABOVE the seizure-entry product (lgr/B), derived from the
    # Task-0b-locked B + this anchor's lgr — NOT a hardcoded 0.74.
    lgr = a2.get("rho_static", 1.0); seizure_product = lgr / B if B else 1.0
    q_recovered = (a2.get("q_core_end", 0.0) * a2.get("q_global_end", 1.0)) > seizure_product
    max_R = "R0"
    order = ["R0", "R1", "R2", "R3", "R4a", "R4b"]
    for ev in r.get("events", []):
        if order.index(ev.get("R_class", "R0")) > order.index(max_R):
            max_R = ev["R_class"]
    if not crossed:
        return "R-stay"
    seizure_pheno = (max_R in SEIZURE_RCLASSES) and ignited
    if seizure_pheno and returned and q_recovered:
        return "R-excursion"
    return "R-runaway"


def main(base):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "readout_*.json"))):
        r = json.load(open(f))
        rows.append({"tag": os.path.basename(f)[8:-5], "verdict": _run_verdict(r),
                     "rho_peak": r.get("a2", {}).get("rho_peak"),
                     "tail": r.get("activity", {}).get("tail_to_baseline_ratio"),
                     "boundary": r.get("a2", {}).get("seizure_boundary")})
    from collections import Counter
    status = {"base": os.path.relpath(base, ROOT),
              "tier": "MECHANISM-SCREEN (NOT seizure-mechanism validation)",
              "tail_gate": TAIL_GATE, "n_runs": len(rows), "per_run": rows,
              "verdict_counts": dict(Counter(x["verdict"] for x in rows)),
              "caveat": "PLACEHOLDER params; rho boundary = Task-0b frozen-q calibrated; recovery is emergent."}
    json.dump(status, open(os.path.join(base, "status_a2_pilot.json"), "w"), indent=1)
    print(json.dumps(status["verdict_counts"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg"))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_a2_analyzer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_a2_pilot.py tests/test_a2_analyzer.py
git commit -m "feat(M3A-A2): analyze_a2_pilot — R-stay/excursion/runaway verdict + bout R-class gate"
```

---

## Task 8: Task-0 RUN — anchor baseline + `a_bar` (3 candidates, pick primary)

**Files:** Results: `results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0/`. No code.

- [ ] **Step 1: Run the 3 candidate anchors at full tank (k_use=0, 3 seeds each)**

```bash
OUT=results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0
for cell in "l0_g1.0:1.0:1.0:1.0" "l1_g1.3:0.85:1.15:1.3" "l2_g1.6:0.70:1.30:1.6"; do
  IFS=: read name cei cee gei <<< "$cell"
  for s in 1 2 3; do
    python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
      --L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 \
      --sep-frac 0.7 --drive 0.6 --lesion twoend_equal --T 20000 --seed $s \
      --core-ei-scale $cei --core-ee-gain $cee --global-ei-scale $gei \
      --a2-mode core_only --a2-k-use 0.0 --dump-a2-trace --dump-fullfield \
      --out $OUT --tag ${name}_k0_s${s}
  done
done
```

- [ ] **Step 2: Pick the primary anchor (clean + closest-to-boundary)**

```bash
python - <<'PY'
import glob, json, numpy as np
rows = {}
for f in glob.glob("results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0/readout_*.json"):
    r = json.load(open(f)); a = r["activity"]; a2 = r.get("a2", {}); tag = f.split("readout_")[1][:-5]
    cell = tag.rsplit("_k0_s",1)[0]
    rows.setdefault(cell, []).append((a["tail_to_baseline_ratio"], a2.get("a_core_mean", float("nan")),
                                      len(r["events"])))   # [P1-2] a_core_mean = the DIRECT a_bar
LGR = {"l0_g1.0":1.00, "l1_g1.3":1.04, "l2_g1.6":1.16}   # closest-to-1.35 wins among clean
for cell, v in sorted(rows.items()):
    v = np.array(v); clean = (np.median(v[:,0]) <= 1.5) and (np.median(v[:,2]) > 0)
    print(f"{cell}: tail_med={np.median(v[:,0]):.2f} evt_med={np.median(v[:,2]):.0f} "
          f"a_bar={np.median(v[:,1]):.5f} lgr={LGR[cell]} clean={clean}")
print("PICK: clean candidate with lgr closest to 1.35; l0_g1.0 is the attribution reference (always run).")
PY
```
**HARD STOP — record:** which anchor(s) are clean (absolute tail ≤ 1.5, nonzero events); the primary = clean + lgr closest to 1.35; the baseline `a_bar` = `readout['a2']['a_core_mean']` from the k_use=0 run ([P1-2] the DIRECT mean per-step core-E active fraction that `k_use_from_target` inverts — NOT Hz×duty; the k_use=0 run emits the a2 block, so it is read straight off disk). If NO candidate is clean → STOP and report (anchor problem, do not proceed to dynamics).

- [ ] **Step 3: Commit the Task-0 status note**

```bash
git add results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0
git commit -m "run(M3A-A2): Task-0 anchor baseline — primary anchor + a_bar locked"
```

---

## Task 9: Task-0b RUN — frozen-q boundary calibration (lock `B`)

**Files:** Results: `.../a2_abbott_lg/task0b/`. No code (uses `--a2-frozen`).

- [ ] **Step 1: Run frozen-q at the 3 target products on the PRIMARY anchor** (example: primary `l2_g1.6`, lgr=1.16 → products {1.0, 1.16/1.35=0.86, 1.16/1.86=0.62}; for core_only, q_core = product, q_global=1)

```bash
OUT=results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0b
# substitute --core-ei-scale/--core-ee-gain/--global-ei-scale + lgr for the locked primary anchor
for prod in 1.00 0.86 0.62; do
  for s in 1 2; do
    python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
      --L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 \
      --sep-frac 0.7 --drive 0.6 --lesion twoend_equal --T 20000 --seed $s \
      --core-ei-scale 0.70 --core-ee-gain 1.30 --global-ei-scale 1.6 \
      --a2-mode core_only --a2-frozen --a2-frozen-qcore $prod --a2-frozen-qglobal 1.0 \
      --dump-a2-trace --out $OUT --tag prod${prod}_s${s}
  done
done
```

- [ ] **Step 2: Check each product reproduces interictal / seizure-like / runaway** (apply the A1b `_state` thresholds via the analyzer's activity + sidecar)

```bash
python - <<'PY'
import glob, json, numpy as np
for f in sorted(glob.glob("results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0b/readout_*.json")):
    r = json.load(open(f)); a = r["activity"]
    print(f.split("readout_")[1][:-5], "tail=%.2f gr=%.2f coreR=%.0f maxR=%s" % (
        a["tail_to_baseline_ratio"], a["global_E_rate_mean_hz"], a["core_E_rate_mean_hz"],
        max([e.get("R_class","R0") for e in r["events"]] or ["R0"])))
print("EXPECT: prod1.00=interictal, prod0.86=seizure-like (R4a/collision), prod0.62=runaway.")
PY
```
**HARD STOP — lock `B`:** if the products reproduce the A1b ladder → `B = 1.35` (sealing the A1b boundary for this anchor). If NOT → set `B` to the rho at which the frozen-q runs actually flip to seizure-like phenotype (the A2-intrinsic boundary), and use that `--a2-boundary B` for Task-1. Record the decision.

- [ ] **Step 3: Commit**

```bash
git add results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task0b
git commit -m "run(M3A-A2): Task-0b frozen-q boundary — B locked for primary anchor"
```

---

## Task 10: Task-1 RUN — core-only dynamic ladder + analyze + HARD STOP + figures

**Files:** Results: `.../a2_abbott_lg/task1/` + `figures/`. No code (uses Tasks 6-7).

- [ ] **Step 1: Derive the k_use ladder from `a_bar` + `B`, run core-only × 3 seeds**

[P1-boundary] The `q_target` ladder is **derived from the Task-0b-locked `B` + the anchor's `lgr`**, NOT hardcoded: `rho_target ∈ {0.85·B, B, 1.2·B}` → `q_target = lgr/rho_target` (core_only ⇒ `q_global=1` ⇒ `q_core=q_target`). Fill `B` from Task-0b, `LGR`/`ABAR`/the structural knobs from Task-0; the loop computes the ladder:

```bash
OUT=results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task1
B=<from_task0b>; LGR=<anchor_lgr_from_task0>; ABAR=<core_a_bar_from_task0>
CEI=0.70; CEE=1.30; GEI=1.6                 # locked primary anchor's knobs (example l2_g1.6)
for f in 0.85 1.0 1.2; do
  QT=$(python -c "print(round($LGR/($f*$B), 4))")     # q_target = lgr / (f*B)
  for s in 1 2 3; do
    python scripts/run_sef_hfo_snn_cm_spontaneous_readout.py \
      --L 20 --density 100 --theta 45 --core-mean 17.5 --core-std 1.0 --core-r 1.5 \
      --sep-frac 0.7 --drive 0.6 --lesion twoend_equal --T 20000 --seed $s \
      --core-ei-scale $CEI --core-ee-gain $CEE --global-ei-scale $GEI \
      --a2-mode core_only --a2-q-target $QT --a2-a-bar $ABAR --a2-tau-rec 5000 \
      --a2-tau-a 100 --a2-q-min 0.25 --a2-boundary $B --dump-a2-trace \
      --out $OUT --tag f${f}_s${s}
  done
done
python scripts/analyze_a2_pilot.py $OUT
```

- [ ] **Step 2: Render the 4 pilot figures + Chinese README**

Write `scripts/plot_a2_pilot.py` (mirror existing `plot_*` style) producing into `$OUT/figures/`:
`(1) rho_trace.png` (rho(t) + q_core/q_global + global/core/surround rate, one representative seed),
`(2) state_timeline.png` (rho bands interictal/seizure/runaway over time),
`(3) event_phenotype_scatter.png` (per-event q_core_pre vs r95_ea/sustained_front_score, colored by R_class),
`(4) kuse_ladder_summary.png` (R-stay/excursion/runaway counts per q_target × seed).
Then write `$OUT/figures/README.md` (Chinese, per AGENTS.md: `### filename` + 2-4 句 + `**关注点**：`).

```bash
python scripts/plot_a2_pilot.py results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task1
```

- [ ] **Step 3: HARD STOP — write the pilot recap**

Read `status_a2_pilot.json` + eyeball the 4 figures. Write `docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_pilot_recap_<DATE>.md` using the plain-language 三段式 (invoke `hfosp-plain-language-recap`): 测了什么 / 怎么测的 / 揭示什么 = which of R-stay/R-excursion/R-runaway dominated, whether it's an activity artifact (rho crossed but phenotype/R-class didn't → §4.3.2 anti-rule), whether `rho` crossed `B`, per-seed sign-consistency. Apply the reporting discipline (§4.4): recovery is emergent; no "termination" claim. **Then STOP** — Task-2 (two-tank), Task-3 (per-core), and controls C1-C4 are gated on the user's read of this recap (spec §9.4).

- [ ] **Step 4: Commit**

```bash
git add results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg/task1 scripts/plot_a2_pilot.py docs/archive/topic4/sef_hfo/m3a_a2_abbott_lg_pilot_recap_*.md
git commit -m "run(M3A-A2): Task-1 core-only dynamic pilot + figures + recap — HARD STOP for user review"
```

---

## Deferred (NOT in this plan's executable scope — gated on Task-1 recap, spec §9)

- **Task-2** two-tank dynamic (`--a2-mode two_tank`), only if Task-1 shows a trend.
- **Task-3** per-core confirm (`--a2-mode per_core`), only if R-excursion / sync rise appears — rules out shared-`q_core` false synchrony.
- **Task-4** event-accumulation analysis (`rho_pre → next-bout class` hazard; `Δq_event → next IEI`).
- **Controls C1-C4** (k_use=0 baseline / frozen-q matched-static / replay-q time-shuffle / core-only-vs-global-only).
- **Failure branches** (R-runaway → add `g_K`/local feedback; R-stay → stronger k_use / near-boundary anchor; rate-only → null; A2b dynamic e_GABA fallback).

---

## Self-Review (run before handing off)

**Spec coverage:** §2 mechanism → Tasks 1-2 (apply_currents + step); §2.1 q_global global + §2.4 → Task 1; per-core §2.1/#3 → Task 3; rho §2.5 → Task 5; substrate §3.1 + heterogeneous-core §3.3/#11 → Task 6 (Steps 1,9); anchor §3.2 → Task 8; frozen-q boundary §4.1/#1 → Task 9; R-class gate §4.3.2/#6 → Tasks 5,7; onset/offset §4.3.3/#5 → Task 7 verdict (no q-leads-decay gate); engine parity §5.2/#1 → Task 1; mutual-exclusion §5.4/#7 → Task 4; readout §6 → Task 6; invariants §7 (#1-15) → Tasks 1-7; k_use-from-target #4 → Task 4; pilot order §9 → Tasks 8-10; controls/failure §9.2-9.3 → Deferred section. **Gap check:** invariant #10 (substrate-param guard) — add to Task 6 Step 8 (assert the run echoes `core_mean=17.5` into provenance, not the default 17.0); fold into the smoke test.

**Type consistency:** `RegionalResource(N, V_th0, core_mask_E, cfg, NE=, left_core_E=, right_core_E=)` — `NE=` passed by every construction (Tasks 1,2,3 tests + Task 4 builder [P0-1]). `build_regional_resource(...)` keyword set identical in Task 4 def and Task 6 call. `event_rclass(af, spk, bin_of_cell, n_bins, bin_centers, src_bin, bin_w, t_on, t_off, dt, far_radius)` — same in Task 5 def and Task 6 call (returns `(R_class, metrics, n_act)`); `spatial_bins` returns a dict; events are ms. `_run_verdict` reads `a2.seizure_boundary/rho_peak/q_core_end` + `activity.tail_to_baseline_ratio/active_E_fraction_peak/peak_E_rate_hz` + `events[].R_class` — all produced by Task 6. `compute_rho(q_core,q_global,lgr_static)` — same Task 5 def + Task 6 call.

**Open implementation detail (resolve in Task 6 Step 1, not a placeholder):** the exact `labels` content and `events` element keys (`b0/b1` vs `t_on/t_off`) and `spatial_bins` signature must be confirmed against the live runner before Steps 5-7; the plan states the fallback (use `is_E = arange(N) < NE`; convert ms→bins) so the implementer is not blocked.
