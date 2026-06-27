# M3A-v2 Spatial Slow-Variable Field — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two SPATIAL slow-variable fields — inhibitory-resource `q_I(x,t)` and fatigue/recovery `g_K(x,t)` — to the existing anisotropic E-I SNN, plus the source-space four-state event classifier and proxy phase plane, so that "axis fatigues while off-axis permissiveness rises → axis-breaking" becomes *representable and detectable* (which the v1 two-scalar `q_core/q_global` model structurally cannot carry).

**Architecture:** A new `slow` object (`SpatialSlowField`) implements the existing `simulate_kick` slow protocol (`apply_currents` / `threshold` / `step`), so the SNN engine is unchanged. The fields live on an `n_grid×n_grid` lattice over the `L×L` sheet, drain/build from the local smoothed firing rate (convolution kernels reused from `src/sef_hfo_field.py`), and couple back to each neuron at its position. Event readout + the four-state classifier + the proxy phase plane live in a separate pure module (`src/topic4_m3a_v2_phenotype.py`) that reuses the source-space onset gradient (`src/sef_hfo_snn_metrics.py::onset_axis`).

**Tech Stack:** Python, NumPy, pytest. SNN engine `src/snn_engine/` (loaded via `sys.path.insert`). FFT periodic convolution from `src/sef_hfo_field.py`.

## Global Constraints

- **Spec of record:** `docs/snn_core_model_equations.md §B5` (formulas, variable definitions, four-state table, red lines). Re-read the relevant `§B5.x` before writing each function body (CLAUDE.md §6).
- **This is a mechanism SCREEN, not a seizure validation.** `ictal_like_candidate` is a *detection label*. Never write "proved seizure mechanism / Abbott holds / passed interictal→ictal". Whether axis-breaking actually occurs is empirical and is deferred to ablation (see Deferred).
- **Methodological lock (§4 / §B5.6):** the axis score MUST be computed from the **source-space per-cell onset gradient** (`onset_axis`). Never contact-space direction, collision, or spike-cloud elongation.
- **Off-by-default byte parity:** `k_q=0, k_K=0, q_init=1` ⇒ `q_I≡1, g_K≡0` ⇒ the engine is bit-identical to `slow=None`. This is a hard regression gate (`test_offparity_byte_identical_to_slow_none`).
- **Stubs raise `NotImplementedError`** until implemented; never return plausible placeholders (CLAUDE.md §6).
- **Minimal v2 = `q_I + g_K`.** `D_EE(x,t)` and the ablation battery are explicitly out of scope this round (Deferred).
- **Acceptance gates encode the conclusion, not existence** (memory `feedback_acceptance_gate_encode_conclusion`): each load-bearing claim has a numeric gate + a bad-data regression. The classifier fails closed to `INSUFFICIENT`.

## Status (2026-06-28, review-fixed)

Red-TDD scaffold is **written and committed-pending on a dedicated branch** (see "Branch & commit discipline" below):
- Stubs: `src/snn_engine/slow_field.py`, `src/topic4_m3a_v2_phenotype.py` (all logic raises `NotImplementedError`; config defaults are the locked spec values).
- Tests: `tests/test_m3a_v2_spatial_slowvars.py` — 40 fast (red) + 1 `@slow` parity (red). Verified: `pytest … -m "not slow"` ⇒ `40 failed` (all `NotImplementedError`), no collection errors.

Each task below turns a named group of those red tests green. **Do not weaken a test to pass it** — if a test looks wrong, stop and reconcile against `§B5`.

The 2026-06-28 review added 7 tests + tightened 4 contracts (g_K bounded build uses `k_K`; ictal-like requires `R_area >= area_large`; proxy `Y = P_global`; `eta_I` weighting pinned via `aq_drive`); they are folded into Tasks 5/9/10 below.

### Branch & commit discipline (P0 — MANDATORY for every worker)

- **Work on the dedicated branch `codex/topic4-m3a-v2-spatial-field`** (created off `main`). NEVER commit on `main`.
- **NEVER `git add -A` / `git add .` / `git commit -a`.** The working tree holds unrelated untracked files (e.g. `docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md`, `docs/superpowers/plans/2026-06-28-topic5-ictal-field-dynamics.md`) that MUST NOT be staged.
- **Commit only with explicit paths.** The entire allowed file set for this work is exactly: `docs/snn_core_model_equations.md`, `docs/topic4_m3_stage.md`, `docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md`, `src/snn_engine/slow_field.py`, `src/topic4_m3a_v2_phenotype.py`, `tests/test_m3a_v2_spatial_slowvars.py`.
- Each task commits ONLY the single source file it edits (Tasks 1–5 → `src/snn_engine/slow_field.py`; Tasks 6–10 → `src/topic4_m3a_v2_phenotype.py`), via `git commit -m "…" -- <that file>`.

## Convention lock (read before Task 3/4)

- **Grid index ↔ field:** image convention `field[iy, ix]` where `ix = clip(floor(x/L·n_grid), 0, n_grid-1)`, `iy = clip(floor(y/L·n_grid), 0, n_grid-1)`, `x = pos[:,0]`, `y = pos[:,1]`. `firing_rate_field` and `sample_field_at` MUST use this same convention (tests `test_sample_field_at_recovers_grid_values`, `test_firing_rate_field_single_spike_peaks_near_its_position` pin it).
- **E/I split:** E neurons occupy indices `[:nE]`, I neurons `[nE:]` (same as `RegionalResource.is_E`). `step` splits `spk[:nE]` / `spk[nE:]`; `apply_currents` couples fields to E cells only.
- **Reuse:** `from src.sef_hfo_field import isotropic_gaussian, convolve_periodic`. `isotropic_gaussian(n, L, sigma)` returns a normalized `(n,n)` kernel; `convolve_periodic(field, kernel)` is FFT periodic convolution.

## File Structure

| File | Responsibility | This round |
| --- | --- | --- |
| `src/snn_engine/slow_field.py` | `SpatialSlowFieldConfig` (+`validate`), `saturation`, `aq_drive`, `firing_rate_field`, `sample_field_at`, `SpatialSlowField` (field dynamics; `slow` protocol) | Tasks 1–5 |
| `src/topic4_m3a_v2_phenotype.py` | per-event metrics, four-state `classify_event`, proxy phase plane | Tasks 6–10 |
| `tests/test_m3a_v2_spatial_slowvars.py` | red contracts (already written) | gate per task |
| `docs/snn_core_model_equations.md §B5` | spec of record | reference only |

---

### Task 1: Config structural invariants

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`SpatialSlowFieldConfig.validate`)
- Test: `tests/test_m3a_v2_spatial_slowvars.py::test_config_validate_*`

**Interfaces:**
- Produces: `SpatialSlowFieldConfig.validate() -> None` (raises `ValueError`).

- [ ] **Step 1 — Tests already written** (`test_config_validate_accepts_locked_defaults`, `…_rejects_sigma_q_not_greater_than_sigma_K`, `…_rejects_eta_I_below_eta_E`).
- [ ] **Step 2 — Run, verify red:** `pytest tests/test_m3a_v2_spatial_slowvars.py -k config -q` → 3 fail (`NotImplementedError`).
- [ ] **Step 3 — Implement `validate`:**

```python
def validate(self) -> None:
    if not (self.sigma_q > self.sigma_K):
        raise ValueError(f"sigma_q ({self.sigma_q}) must be > sigma_K ({self.sigma_K}) "
                         "(wide disinhibition footprint, narrow fatigue footprint; §B5.3)")
    if self.eta_I < self.eta_E:
        raise ValueError(f"eta_I ({self.eta_I}) must be >= eta_E ({self.eta_E}) (§B5.2)")
    if not (0.0 < self.q_min <= 1.0):
        raise ValueError(f"q_min must be in (0, 1], got {self.q_min}")
    if self.gK_max < 0.0:
        raise ValueError(f"gK_max must be >= 0, got {self.gK_max}")
    if self.n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {self.n_grid}")
```

- [ ] **Step 4 — Run, verify green:** `pytest tests/test_m3a_v2_spatial_slowvars.py -k config -q` → 3 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): config structural invariants (sigma_q>sigma_K, eta_I>=eta_E)" -- src/snn_engine/slow_field.py`

---

### Task 2: Saturation `f(a)`

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`saturation`)
- Test: `…::test_saturation_*`

**Interfaces:**
- Produces: `saturation(a, a0, a50) -> ndarray|float`, Hill-like, elementwise.

- [ ] **Step 1 — Tests already written** (zero at/below onset; half at `a0+a50`; → 1 and monotone).
- [ ] **Step 2 — Verify red:** `pytest …-k saturation -q` → 3 fail.
- [ ] **Step 3 — Implement:**

```python
def saturation(a, a0, a50):
    x = np.maximum(np.asarray(a, dtype=float) - a0, 0.0)   # [a - a0]_+
    return x / (a50 + x)
```

- [ ] **Step 4 — Verify green:** `pytest …-k saturation -q` → 3 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): Hill saturation f(a)=[a-a0]_+/(a50+[a-a0]_+)" -- src/snn_engine/slow_field.py`

---

### Task 3: Firing-rate field + sampling

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`firing_rate_field`, `sample_field_at`)
- Test: `…::test_firing_rate_field_*`, `…::test_sample_field_at_recovers_grid_values`

**Interfaces:**
- Consumes: `src.sef_hfo_field.{isotropic_gaussian, convolve_periodic}`.
- Produces: `firing_rate_field(spk_bool, pos, L, n_grid, sigma) -> (n_grid,n_grid)`; `sample_field_at(field, pos, L, n_grid) -> (n,)`. Both in the `field[iy, ix]` convention.

- [ ] **Step 1 — Tests already written.**
- [ ] **Step 2 — Verify red:** `pytest …-k "firing_rate_field or sample_field_at" -q` → 3 fail.
- [ ] **Step 3 — Implement** (module-top import: `from src.sef_hfo_field import isotropic_gaussian, convolve_periodic`):

```python
def _grid_index(pos, L, n_grid):
    ix = np.clip((np.asarray(pos)[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((np.asarray(pos)[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return ix, iy

def firing_rate_field(spk_bool, pos, L, n_grid, sigma):
    counts = np.zeros((n_grid, n_grid))
    spk_bool = np.asarray(spk_bool, bool)
    if spk_bool.any():
        ix, iy = _grid_index(pos[spk_bool], L, n_grid)
        np.add.at(counts, (iy, ix), 1.0)                       # field[iy, ix]
    return convolve_periodic(counts, isotropic_gaussian(n_grid, L, sigma))

def sample_field_at(field, pos, L, n_grid):
    ix, iy = _grid_index(pos, L, n_grid)
    return np.asarray(field)[iy, ix]
```

- [ ] **Step 4 — Verify green:** `pytest …-k "firing_rate_field or sample_field_at" -q` → 3 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): spike->rate field (bin+isotropic conv) + nearest-grid sampling" -- src/snn_engine/slow_field.py`

---

### Task 4: `SpatialSlowField.__init__` + `apply_currents`

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`SpatialSlowField.__init__`, `apply_currents`)
- Test: `…::test_apply_currents_*`

**Interfaces:**
- Consumes: `firing_rate_field`/`sample_field_at` conventions (Task 3), `isotropic_gaussian` (precompute `K_q`, `K_K`).
- Produces: `SpatialSlowField(N, V_th0, posE, posI, L, core_mask_E=None, cfg=None)`; attributes `q_I`, `g_K` (`(n_grid,n_grid)`), `cfg`, `nE`; `apply_currents(I_E, I_I, labels=None) -> I_net`.

- [ ] **Step 1 — Tests already written** (`…_off_is_IE_minus_II`, `…_uniform_qI_matches_scalar_regionalresource`, `…_gK_subtracts_on_E_only`).
- [ ] **Step 2 — Verify red:** `pytest …-k apply_currents -q` → 3 fail.
- [ ] **Step 3 — Implement** (`apply_currents` samples the fields DIRECTLY via precomputed E-cell grid indices, so externally setting `fld.g_K[:]=…` is reflected — `test_apply_currents_gK_subtracts_on_E_only` requires this; no per-neuron cache):

```python
def __init__(self, N, V_th0, posE, posI, L, core_mask_E=None, cfg=None):
    self.cfg = cfg or SpatialSlowFieldConfig()
    self.cfg.validate()
    self.N = int(N); self.nE = int(np.asarray(posE).shape[0]); self.L = float(L)
    self.posE = np.asarray(posE, float); self.posI = np.asarray(posI, float)
    n = self.cfg.n_grid
    self.q_I = np.full((n, n), self.cfg.q_init, dtype=float)
    self.g_K = np.zeros((n, n), dtype=float)
    self.rE = np.zeros((n, n)); self.rI = np.zeros((n, n))         # EMA rate fields
    self._Kq = isotropic_gaussian(n, L, self.cfg.sigma_q)
    self._Kk = isotropic_gaussian(n, L, self.cfg.sigma_K)
    self._ixE, self._iyE = _grid_index(self.posE, L, n)            # fixed E->grid map
    self._alpha_a = None
    self.trace_qI_mean = []; self.trace_gK_mean = []

def apply_currents(self, I_E, I_I, labels=None):
    qI_E = self.q_I[self._iyE, self._ixE]                          # (nE,)
    gK_E = self.g_K[self._iyE, self._ixE]
    out = np.asarray(I_E, float) - np.asarray(I_I, float)          # I cells: I_E - I_I
    nE = self.nE
    out[:nE] = I_E[:nE] - qI_E * I_I[:nE] - self.cfg.eta_K * gK_E  # E cells
    return out
```

- [ ] **Step 4 — Verify green:** `pytest …-k apply_currents -q` → 3 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): SpatialSlowField init + apply_currents (parity, v1-reduction, gK on E)" -- src/snn_engine/slow_field.py`

---

### Task 5: `SpatialSlowField.step` (field dynamics) + byte parity

**Files:**
- Modify: `src/snn_engine/slow_field.py` (`SpatialSlowField.step`)
- Test: `…::test_step_off_holds_fields`, `…::test_qI_depletes_with_activity_then_refills`, `…::test_qI_depletes_locally_not_globally`, `…::test_qI_bounded_floor`, `…::test_gK_builds_on_E_activity_and_decays_bounded`, `…::test_kernel_q_footprint_wider_than_kernel_K`, `…::test_step_no_nan_long_random`, `…::test_offparity_byte_identical_to_slow_none` (`@slow`)

**Interfaces:**
- Consumes: `firing_rate_field`, `saturation`, `convolve_periodic`, `self._Kq/_Kk`.
- Produces: `step(spk, labels, dt) -> None`; mutates `self.q_I`, `self.g_K`, `self.rE`, `self.rI`.

- [ ] **Step 1 — Tests already written** (sign, SPATIAL locality, bounds, kernel-width footprint, no-NaN, byte parity; + `aq_drive` weighted drive, `qI_depletes_from_inhibitory_activity`, `gK_zero_kK`/`gK_larger_kK`/`gK_bounded_ceiling`).
- [ ] **Step 2 — Verify red:** `pytest …-k "step or qI or gK or kernel or offparity or aq_drive" -q` (+`-m "not slow"` to skip parity) → fail.
- [ ] **Step 3 — Implement** (gate the ODEs on `k_q!=0` / `k_K!=0` so the off path leaves `q_I≡1, g_K≡0` exactly — the parity contract; also implement the module-level `aq_drive` helper that `step` calls, so the `eta_I·r_I` weighting is a named, unit-pinned contract; `firing_rate_field`/`saturation` already imported in-module):

```python
def aq_drive(rE, rI, eta_E, eta_I):                              # §B5.2 weighted depletion drive
    return eta_E * np.asarray(rE, float) + eta_I * np.asarray(rI, float)

def step(self, spk, labels, dt):
    cfg = self.cfg
    spk = np.asarray(spk, bool)
    rE_inst = firing_rate_field(spk[:self.nE], self.posE, self.L, cfg.n_grid, cfg.sigma_r)
    rI_inst = firing_rate_field(spk[self.nE:], self.posI, self.L, cfg.n_grid, cfg.sigma_r)
    if self._alpha_a is None:
        self._alpha_a = 1.0 - np.exp(-dt / cfg.tau_a)
    a = self._alpha_a
    self.rE += a * (rE_inst - self.rE)                            # EMA (§B5.1)
    self.rI += a * (rI_inst - self.rI)
    if cfg.use_qI and cfg.k_q != 0.0:                            # §B5.2 (depletion ~ k_q*f*q_I)
        a_q = convolve_periodic(aq_drive(self.rE, self.rI, cfg.eta_E, cfg.eta_I), self._Kq)
        fq = saturation(a_q, cfg.a0_q, cfg.a50_q)
        self.q_I += dt * ((1.0 - self.q_I) / cfg.tau_q - cfg.k_q * fq * self.q_I)
        np.clip(self.q_I, cfg.q_min, 1.0, out=self.q_I)
    if cfg.use_gK and cfg.k_K != 0.0:                            # §B5.3 BOUNDED build (k_K is the knob)
        a_K = convolve_periodic(self.rE, self._Kk)
        fk = saturation(a_K, cfg.a0_K, cfg.a50_K)
        self.g_K += dt * (-self.g_K / cfg.tau_K + cfg.k_K * fk * (cfg.gK_max - self.g_K))
        np.clip(self.g_K, 0.0, cfg.gK_max, out=self.g_K)
    self.trace_qI_mean.append(float(self.q_I.mean()))
    self.trace_gK_mean.append(float(self.g_K.mean()))
```

- [ ] **Step 4 — Verify green:** `pytest …-k "step or qI or gK or kernel or aq_drive" -m "not slow" -q` → pass; then `pytest …::test_offparity_byte_identical_to_slow_none -q` → pass.
- [ ] **Step 5 — Re-bless the engine** (slow_field couples through `simulate_kick`; record its hash): update `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` per the existing bless procedure if the parity test required any engine touch (it should NOT — verify `simulate_kick` is unchanged).
- [ ] **Step 6 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): SpatialSlowField.step (q_I depletion + g_K bounded build, aq_drive, spatial locality, byte parity)" -- src/snn_engine/slow_field.py`

---

### Task 6: `recruitment_area` + `axis_score` (methodological lock)

**Files:**
- Modify: `src/topic4_m3a_v2_phenotype.py` (`recruitment_area`, `axis_score`)
- Test: `…::test_recruitment_area_fraction_above_threshold`, `…::test_axis_score_source_space_onset_gradient`, `…::test_axis_score_nan_when_too_few_onsets`

**Interfaces:**
- Consumes: `src.sef_hfo_snn_metrics.onset_axis` (already imported in-module).
- Produces: `recruitment_area(A, theta_A) -> float`; `axis_score(posE, onset, u_axis, min_n=20) -> float` (NaN if axis undefined).

- [ ] **Step 1 — Tests already written.**
- [ ] **Step 2 — Verify red:** `pytest …-k "recruitment_area or axis_score" -q` → 3 fail.
- [ ] **Step 3 — Implement** (`axis_score` delegates direction to `onset_axis` — the source-space instrument; NaN propagates to `INSUFFICIENT` downstream):

```python
def recruitment_area(A, theta_A):
    A = np.asarray(A, float)
    return float((A > theta_A).mean())

def axis_score(posE, onset, u_axis, min_n=20):
    v = onset_axis(posE, onset, min_n=min_n)                      # source-space onset gradient
    if v is None:
        return float("nan")
    u = np.asarray(u_axis, float); u = u / np.linalg.norm(u)
    return float(abs(np.dot(v, u)))
```

- [ ] **Step 4 — Verify green:** `pytest …-k "recruitment_area or axis_score" -q` → 3 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): recruitment_area + source-space axis_score (onset-gradient lock)" -- src/topic4_m3a_v2_phenotype.py`

---

### Task 7: `offaxis_fraction` + `participation_ratio`

**Files:**
- Modify: `src/topic4_m3a_v2_phenotype.py` (`offaxis_fraction`, `participation_ratio`)
- Test: `…::test_offaxis_fraction_on_axis_vs_off_axis`, `…::test_participation_ratio_bounds`

**Interfaces:**
- Produces: `offaxis_fraction(A, grid_xy, center, u_axis, corridor_halfwidth) -> float`; `participation_ratio(A) -> float`.

- [ ] **Step 1 — Tests already written.**
- [ ] **Step 2 — Verify red:** `pytest …-k "offaxis or participation" -q` → 2 fail.
- [ ] **Step 3 — Implement** (`grid_xy[...,0]=x, [...,1]=y`; perpendicular distance uses `u_perp ⊥ u_axis`):

```python
def offaxis_fraction(A, grid_xy, center, u_axis, corridor_halfwidth):
    A = np.asarray(A, float); g = np.asarray(grid_xy, float)
    u = np.asarray(u_axis, float); u = u / np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    d = (g - np.asarray(center, float))
    perp = np.abs(d[..., 0] * u_perp[0] + d[..., 1] * u_perp[1])
    tot = A.sum()
    if tot <= 0:
        return float("nan")
    return float(A[perp > corridor_halfwidth].sum() / tot)

def participation_ratio(A):
    A = np.asarray(A, float)
    s1 = A.sum(); s2 = (A * A).sum()
    if s2 <= 0:
        return float("nan")
    return float(s1 * s1 / (A.size * s2))
```

- [ ] **Step 4 — Verify green:** `pytest …-k "offaxis or participation" -q` → 2 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): off-axis fraction + participation ratio (globality)" -- src/topic4_m3a_v2_phenotype.py`

---

### Task 8: `event_recovery`

**Files:**
- Modify: `src/topic4_m3a_v2_phenotype.py` (`event_recovery`)
- Test: `…::test_event_recovery_returned_vs_runaway`

**Interfaces:**
- Produces: `event_recovery(rate, dt, t_post0, baseline, sigma_base, m=1.5, t_return=120.0) -> bool`.

- [ ] **Step 1 — Test already written.**
- [ ] **Step 2 — Verify red:** `pytest …-k event_recovery -q` → 1 fail.
- [ ] **Step 3 — Implement:**

```python
def event_recovery(rate, dt, t_post0, baseline, sigma_base, m=1.5, t_return=120.0):
    rate = np.asarray(rate, float)
    i0 = int(round(t_post0 / dt)); i1 = int(round((t_post0 + t_return) / dt))
    seg = rate[i0:i1]
    if seg.size == 0:
        return False
    return bool(seg.mean() <= baseline + m * sigma_base)
```

- [ ] **Step 4 — Verify green:** `pytest …-k event_recovery -q` → 1 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): event recovery (returned vs runaway)" -- src/topic4_m3a_v2_phenotype.py`

---

### Task 9: `classify_event` (four-state gate structure)

**Files:**
- Modify: `src/topic4_m3a_v2_phenotype.py` (`classify_event`)
- Test: `…::test_classify_*` (7 tests, incl. the `…_large_axial_is_NOT_ictal_like` AND `…_small_offaxis_is_NOT_ictal_like` size boundaries and `…_insufficient_fails_closed`)

**Interfaces:**
- Consumes: `PhenotypeGates` (defaults are calibration); a `metrics` dict with keys `n_onsets, R_area, S_axis, F_offaxis, G_PR, recovery`.
- Produces: `classify_event(metrics, gates=None) -> str` ∈ `{interictal_axial, expanded_axial, ictal_like_candidate, runaway, INSUFFICIENT}`.

- [ ] **Step 1 — Tests already written.** The locked ORDER is the science contract: fail-closed first, runaway second, ictal-like (LARGE + axis-breaking) third, interictal fourth, else expanded. `area_large` is the SIZE gate (P1-b): a small off-axis blip is never ictal-like.
- [ ] **Step 2 — Verify red:** `pytest …-k classify -q` → 7 fail.
- [ ] **Step 3 — Implement** (the gate STRUCTURE is the contract; thresholds come from `gates`):

```python
def classify_event(metrics, gates: PhenotypeGates | None = None):
    g = gates or PhenotypeGates()
    s_axis = metrics["S_axis"]
    # 1. fail-closed: insufficient evidence -> never a state, never ictal-like
    if metrics["n_onsets"] < g.min_onsets or s_axis != s_axis:   # s_axis != s_axis == isnan
        return "INSUFFICIENT"
    # 2. did not return to baseline
    if not metrics["recovery"]:
        return "runaway"
    # 3. ictal-like: LARGE recruitment AND axis dominance dropped AND off-axis/low-k rose.
    #    SIZE (R_area >= area_large) is NECESSARY -- a small off-axis blip is never ictal-like.
    #    recovery is already guaranteed by gate 2.
    if (metrics["R_area"] >= g.area_large and s_axis < g.axis_broken
            and (metrics["F_offaxis"] >= g.offaxis_high or metrics["G_PR"] >= g.gpr_high)):
        return "ictal_like_candidate"
    # 4. small + axis-dominant
    if metrics["R_area"] < g.area_small and s_axis >= g.axis_high:
        return "interictal_axial"
    # 5. large axis-dominant, or otherwise-unclassified recovered event (size alone is NOT ictal-like)
    return "expanded_axial"
```

- [ ] **Step 4 — Verify green:** `pytest …-k classify -q` → 7 pass.
- [ ] **Step 5 — Commit (explicit path, P0):** `git commit -m "feat(m3a-v2): four-state classifier (fail-closed; size gate -> size!=ictal-like)" -- src/topic4_m3a_v2_phenotype.py`

---

### Task 10: Proxy phase plane

**Files:**
- Modify: `src/topic4_m3a_v2_phenotype.py` (`region_pressure`, `proxy_phase_point`)
- Test: `…::test_region_pressure_formula`, `…::test_proxy_phase_point_axis_dominant_has_positive_X` (sign lock), `…::test_proxy_phase_point_Y_is_global_pressure` (Y = P_global, P1-c)

**Interfaces:**
- Produces: `region_pressure(q_I_region, g_K_region, lgr, beta_K, eps=1e-9) -> float`; `proxy_phase_point(field, region_masks, lgr, beta_K) -> (X, Y)`, with `X = P_axis - P_offaxis`, `Y = P_global`.

- [ ] **Step 1 — Tests already written.** Sign lock (P1-c): `q_I↓`=disinhibited=higher pressure, so axis-dominant (axis more disinhibited) ⇒ `X>0`; `Y` is the GLOBAL pressure (not P_offaxis) to match spectral `Y=α_global`.
- [ ] **Step 2 — Verify red:** `pytest …-k "region_pressure or proxy_phase" -q` → 3 fail.
- [ ] **Step 3 — Implement** (`§B5.7`; `region_masks` keys: `axis`, `offaxis`, `global` — the `global` mask is USED, not a dead arg):

```python
def region_pressure(q_I_region, g_K_region, lgr, beta_K, eps=1e-9):
    q = np.asarray(q_I_region, float); gk = np.asarray(g_K_region, float)
    return float(np.log(lgr) - np.mean(np.log(q + eps)) - beta_K * np.mean(gk))

def proxy_phase_point(field, region_masks, lgr, beta_K):
    def P(name):
        m = region_masks[name]
        return region_pressure(field.q_I[m], field.g_K[m], lgr, beta_K)
    P_axis, P_off, P_global = P("axis"), P("offaxis"), P("global")
    return (P_axis - P_off, P_global)            # X = axis dominance (>0 axis leads), Y = global pressure
```

- [ ] **Step 4 — Verify green:** `pytest …-k "region_pressure or proxy_phase" -q` → 3 pass.
- [ ] **Step 5 — Full suite + commit (explicit path, P0):** `pytest tests/test_m3a_v2_spatial_slowvars.py -q` (41 pass: 40 fast + 1 `@slow`) → `git commit -m "feat(m3a-v2): proxy phase plane (region pressure, X=axis-dominance, Y=global)" -- src/topic4_m3a_v2_phenotype.py`

---

## Acceptance Gates (cross-cutting; must hold at end)

| Gate | Test | Encodes |
| --- | --- | --- |
| **Byte parity** | `test_offparity_byte_identical_to_slow_none` | off ⇒ bit-identical to `slow=None` |
| **v1 reduction** | `test_apply_currents_uniform_qI_matches_scalar_regionalresource` | uniform field == scalar `RegionalResource` |
| **Spatial locality** | `test_qI_depletes_locally_not_globally` | the field carries spatial history (the whole point of v2) |
| **Kernel width** | `test_kernel_q_footprint_wider_than_kernel_K` | σ_q > σ_K behaviorally (disinhibition wide, fatigue narrow) |
| **Methodological lock** | `test_axis_score_source_space_onset_gradient` | axis from source-space onset gradient |
| **Fail-closed** | `test_classify_insufficient_fails_closed` | bad data → `INSUFFICIENT`, never `ictal_like` |
| **Size ≠ ictal-like** | `test_classify_large_axial_is_NOT_ictal_like` + `test_classify_small_offaxis_is_NOT_ictal_like` | the key scientific boundary, BOTH directions (large-axial→not ictal, small-offaxis→not ictal) |
| **k_K is a strength knob** (P1-a) | `test_gK_larger_kK_builds_more` | g_K bounded build genuinely scales with `k_K` (not just on/off) |
| **Inhibitory-use depletion** (P1-d) | `test_aq_drive_weights_inhibition_at_least_excitation` + `test_qI_depletes_from_inhibitory_activity` | `q_I` depletes with `eta_I·r_I` (≥ `eta_E·r_E`); step cannot silently drop `r_I` |
| **Phase-plane Y = global** (P1-c) | `test_proxy_phase_point_Y_is_global_pressure` + `…_axis_dominant_has_positive_X` | `Y=P_global` (overlay with spectral `α_global`); sign lock `X>0` for axis-dominant |

Run-all gate: `pytest tests/test_m3a_v2_spatial_slowvars.py -q` ⇒ 41 pass (40 fast + 1 slow), and `pytest -q -m "not slow"` (full repo) stays green (no regression in `test_a2_regional_resource`, `test_m3a_quasistatic_slowvars`, `test_snn_shunting`).

## Deferred (explicitly out of scope this round — do NOT build)

- **`D_EE(x,t)` E→E depression (§B5.4):** formula is in the spec; no implementation, no TDD. Open only if `q_I+g_K` cannot drop `S_axis` (calibration finding), with `D_min≈0.5–0.8`.
- **Ablation battery (A/B/C/D):** the experiment runner that *proves the mechanism* (A=no slow-vars; B=q_I only; C=q_I+g_K main; D=+D_EE). This is what licenses any mechanism claim. Until it runs, v2 delivers the **detector + carrier** only; any pilot run is descriptive screen, not a claim.
- **Calibration / pilot:** choosing `(k_q, q_min, sigma_q, k_K, eta_K, tau_*)` so the model expresses the four phenotypes on the Stage-3 two-core substrate. The risk (spec §B5.8 / user §10): `q_I` depletes most ON the axis → may only amplify axis, not break it. If only `expanded_axial` appears, that is a *balance* finding, not a model failure. Tune order: widen `K_q` → lower `q_min` → strengthen `g_K` → (only then) open `D_EE`.
- **Spectral phase plane + B-line overlay (§B5.7):** reuse `topic4_m3b_spectral_phase.py`; gated behind the `sef_hfo_m3_interface.py` D1/D2/D3 contract. The spatial-field interface extension is itself deferred (contract §9).

## Self-Review (writing-plans)

- **Spec coverage:** §B5.0 motivation→file header/Goal; §B5.1 rate field→Task 3; §B5.2 q_I→Tasks 4–5; §B5.3 g_K + σ_q>σ_K→Tasks 1,5; §B5.4 D_EE→Deferred; §B5.5 membrane parity→Task 5; §B5.6 metrics+four-state→Tasks 6–9; §B5.7 phase plane→Task 10; §B5.8 red lines→Global Constraints + Deferred. No §B5 clause is unassigned.
- **Type consistency:** `SpatialSlowField(N, V_th0, posE, posI, L, core_mask_E=None, cfg=None)`, `apply_currents(I_E, I_I, labels=None)`, `step(spk, labels, dt)`, `aq_drive(rE, rI, eta_E, eta_I)`, `classify_event(metrics, gates=None)`, `proxy_phase_point(field, region_masks, lgr, beta_K)` are identical across stubs, tests, and this plan.
- **No placeholders:** every implementation step shows complete code matching the already-red test it greens.
- **2026-06-28 review fixes pinned by tests (not just prose):** (P1-a) `k_K` bounded build → `test_gK_larger_kK_builds_more` / `_zero_kK_does_not_build` / `_bounded_ceiling`; (P1-b) `area_large` size gate → `test_classify_small_offaxis_is_NOT_ictal_like`; (P1-c) `Y=P_global` + sign lock → `test_proxy_phase_point_Y_is_global_pressure` / `_axis_dominant_has_positive_X`; (P1-d) `eta_I·r_I` weighting → `test_aq_drive_weights_inhibition_at_least_excitation` / `test_qI_depletes_from_inhibitory_activity`. (P0) all commits use explicit paths on `codex/topic4-m3a-v2-spatial-field`.
