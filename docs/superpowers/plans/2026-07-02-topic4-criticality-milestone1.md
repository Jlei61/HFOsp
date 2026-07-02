# Topic 4 M3-v2.2 Approach-Criticality — Milestone 1 (T0–T3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
> **Spec of record:** `docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md` (**rev2.1**). Re-read the relevant spec § before each task body (CLAUDE.md §5/§6).

**Goal:** Build the branch-aware frozen-Jacobian spectral atlas of the M3-v2.2 quiet→runaway approach and emit a pre-registered three-way critical-slowing-down verdict (`smooth_CSD` / `hard_jump_no_CSD` / `unresolved_operating_point`), with the M3-v2.2 trajectory overlaid.

**Architecture:** Reuse the M3B frozen-Jacobian machinery (`src/topic4_m3b_spectral_phase.py`) — extend, don't rebuild. T1 wires the v2.2 continuous sim into the M3A→M3B interface exporter; T2 rebuilds the phase grid in normalized coords so the overlay is contract-legal; T3a adds the operating-point quality gate, branch-aware operating points, complex-pair / next-distinct eigen-metrics, non-normality, and the verdict. Model-side only — no phase2, no SNN perturbation (that is Milestone 2).

**Tech Stack:** Python, numpy, scipy (`scipy.linalg.eig`, `scipy.sparse.linalg.expm_multiply`), PyYAML, pytest. SNN engine imported via `sys.path.insert(0, "src/snn_engine")`.

## Global Constraints (verbatim from spec)

- **主指标 `α₁` = continuous-time frozen-Jacobian leading real-part eigenvalue, per-ms** (`J` confirmed continuous). Discrete operator → `α=log(ρ)/dt`. `τ=−1/α₁` **only for α₁<0**.
- **不预设 `α₁→0`**: three pre-registered verdicts; `runaway` is a saturation classification (`classify_mode` line 954), α₁ may stay negative.
- **CSD read only on quality-gated quasi-static points**; unqualified → `trajectory_not_linearizable` (≠ `hard_jump_no_CSD`).
- **branch-aware**: CSD reads `low_branch`/`approach_branch` only.
- **complex conjugate pair = one invariant 2D subspace**; spectral gap = next-**distinct** real part.
- **overlay only when interface gate passes** (fail-closed); atlas with h_G projected is named `conditional_2d_atlas_at_phase_recovery=...`.
- **results dir**: `results/topic4_criticality/` (reference old `results/topic4_sef_hfo/m3b_spectral_phase_map/` for provenance only).
- **execute only against rev2.1**; T4 (correspondence) gated on topic5 phase2 — **not in this plan**.

---

## File Structure

- Create `config/topic4_criticality.yaml` — all thresholds/terminology (spec §18).
- Create `src/sef_hfo_transition_sim.py` — factor the v2.2 continuous sim out of the figure script into a reusable `run_transition(cfg)`.
- Create `src/topic4_criticality.py` — Milestone-1 pure functions: quality gate, verdict, non-normality, branch clustering, complex-pair/next-distinct helpers that wrap/extend `topic4_m3b_spectral_phase`.
- Modify `src/topic4_m3b_spectral_phase.py` — add `init=` to `solve_operating_point`; extend `rate_eigenpairs`-derived metrics (complex-pair loading, next-distinct gap, pair controllability); `next_distinct_gap`.
- Modify `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py` — import the factored sim (behavior byte-parity).
- Create `scripts/run_topic4_crit_export.py` (T1), `scripts/run_topic4_crit_atlas.py` (T2/T3a).
- Tests: `tests/test_topic4_criticality.py` (pure fns), `tests/test_topic4_crit_integration.py` (`@pytest.mark.integration`).

---

## Task 0: Config + terminology lock + module skeleton

**Files:** Create `config/topic4_criticality.yaml`, `src/topic4_criticality.py`; Test `tests/test_topic4_criticality.py`.
**Interfaces — Produces:** `load_crit_config(path=None) -> dict`.

- [ ] **Step 1: Write config** (spec §18 verbatim; fill the `...` with initial values)

```yaml
# config/topic4_criticality.yaml
operator: {type: continuous_jacobian, dt_ms: null, alpha_units: per_ms, tau_units: ms}
quality_gate:
  residual_rms_tol: 1.0e-4
  rate_mismatch_rel_tol: 0.10
  slow_mismatch_rel_tol: 0.05
  adiabatic_index_tol: 0.20
  alpha_drift_index_tol: 0.20
  min_qualified_points: 5
  min_qualified_fraction: 0.30
verdict:
  alpha_near_zero_tol: 0.02          # per-ms; |alpha1| below this counts as "reached ~0"
  alpha_margin_hard: 0.05            # per-ms; last stable alpha1 must be below -this for hard_jump
  jump_window_ms: 100.0
  smooth_min_tau_growth_ratio: 2.0
  smooth_min_alpha_spearman: 0.6
  unresolved_if_branch_ambiguous: true
branching:
  solve_inits: [low_rate, previous_point, high_rate, random_small]
  branch_cluster_rate_tol: 0.20      # relative rate distance to merge two solutions into one branch
  selected_branch: approach_low_branch
mode:
  complex_pair_policy: invariant_subspace_loading
  spectral_gap_policy: next_distinct_real_part
  next_distinct_min_sep: 1.0e-3      # |Δalpha| below this = same distinct level (pair member)
finite_time_gain: {horizons_ms: [10, 25, 50, 100, 250, 500], norm: weighted_l2, report_numerical_abscissa: true}
slow_to_ratefield:
  q_I: {target: E_inhibition, entry: inh_q_scales_WEI, uniform: false, sign_test: required}
  g_K: {target: E_current, entry: 'muE -= eta_K*g_K(x)', uniform: false, eta_K: 1.0, sign_test: required}
  h_G: {target: E_current, entry: 'muE -= eta_G*h_G', uniform: true, eta_G: 1.0, sign_test: required}
slow_sensitivity: {finite_difference: central, step_fraction_qI: 0.05, step_fraction_gK: 0.05, step_fraction_hG: 0.05, require_both_sides_qualified: true}
virtual_seeg: {use_topic5_estimator_code: true, channel_sets: [source_all_nodes, virtual_all_contacts, matched_10ch], same_windows_as_topic5: true, same_surrogates_as_topic5: true}
tier: model_side_ground_truth
```

- [ ] **Step 2: Write failing loader test**

```python
# tests/test_topic4_criticality.py
from src.topic4_criticality import load_crit_config
def test_config_locks_continuous_operator_and_verdicts():
    c = load_crit_config()
    assert c["operator"]["type"] == "continuous_jacobian"
    assert c["operator"]["alpha_units"] == "per_ms"
    assert c["mode"]["spectral_gap_policy"] == "next_distinct_real_part"
    assert set(c["slow_to_ratefield"]) == {"q_I", "g_K", "h_G"}
    assert c["branching"]["solve_inits"][0] == "low_rate"
```

- [ ] **Step 3: Run test to verify it fails** — `pytest tests/test_topic4_criticality.py::test_config_locks_continuous_operator_and_verdicts -v` → FAIL (ImportError).
- [ ] **Step 4: Implement loader**

```python
# src/topic4_criticality.py
import os, yaml
_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "topic4_criticality.yaml")
def load_crit_config(path=None):
    with open(path or _DEFAULT) as f:
        return yaml.safe_load(f)
```

- [ ] **Step 5: Run test to verify it passes** — same command → PASS.
- [ ] **Step 6: Commit** — `git add config/topic4_criticality.yaml src/topic4_criticality.py tests/test_topic4_criticality.py && git commit -m "feat(topic4-crit): T0 config + terminology lock + module skeleton"`

---

## Task 1: Factor v2.2 continuous sim to src + interface export (two-layer fail-closed)

**Files:** Create `src/sef_hfo_transition_sim.py`, `scripts/run_topic4_crit_export.py`; Modify `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py`; Test `tests/test_topic4_crit_integration.py`.
**Interfaces:**
- Consumes: `src.sef_hfo_m3a_export.build_handoff_from_sim(sim, events, dt_ms, *, mapping_id, gk_enabled, hG_enabled)`, `write_handoff_artifacts(out_dir, *, landmark_rows, mapping, ranges, summary, event_metrics)`; `src.sef_hfo_m3_interface.audit_m3a_interface / compute_overlay_verdict`.
- Produces: `run_transition(cfg) -> dict` (keys `rate_E`, `E_spk_bool`, `times`, `trace_qI_*`, `slow`, `events`); `export_v2_2_handoff(out_dir, cfg) -> overlay_verdict:str`.

- [ ] **Step 1: Read `plot_fig_m3a_v2_2_hG_runaway_transition_gif.py:255-410` (`_simulate_continuous`) and re-read spec §8 P1.** Confirm the sim returns `rate_E`/`E_spk_bool`/`trace_qI_*` unconditionally (agent-verified) and the `run_a2_axisbreak_sweep.py:122-131` exporter call pattern.

- [ ] **Step 2: Write failing byte-parity test for the factoring**

```python
# tests/test_topic4_crit_integration.py
import numpy as np, pytest
@pytest.mark.integration
def test_run_transition_byte_parity_with_figure_sim():
    from src.sef_hfo_transition_sim import run_transition, DEFAULT_CFG
    import scripts.paper_figures.plot_fig_m3a_v2_2_hG_runaway_transition_gif as G
    cfg = DEFAULT_CFG(layout="subject1146", top="qI")     # the 757.5 ms trajectory
    res = run_transition(cfg)
    S = G._build(G._cfg_from(cfg)); ref = G._simulate_continuous(S, G._cfg_from(cfg))
    assert np.array_equal(res["E_spk_bool"], ref["E_spk_bool"])   # factoring changed nothing
```

- [ ] **Step 3: Run to verify fail** — `pytest tests/test_topic4_crit_integration.py::test_run_transition_byte_parity_with_figure_sim -v` → FAIL (ImportError).
- [ ] **Step 4: Move the sim body into `src/sef_hfo_transition_sim.py::run_transition(cfg)`** (cut from the figure script, keep identical numerics), add `DEFAULT_CFG(layout, top)`. In the figure script, replace the inlined body with `from src.sef_hfo_transition_sim import run_transition` and call it. Do NOT touch RNG order.
- [ ] **Step 5: Run to verify pass** — same command → PASS (byte-identical).

- [ ] **Step 6: Write failing two-layer fail-closed export test** (spec §8, P1-4)

```python
@pytest.mark.integration
def test_export_fixture_passes_but_real_gate_is_fail_closed(tmp_path):
    from src.sef_hfo_transition_sim import DEFAULT_CFG
    from src.topic4_criticality import export_v2_2_handoff, export_fixture_handoff
    # (a) fixture MUST reach phase_map_trajectory
    assert export_fixture_handoff(tmp_path / "fix") == "phase_map_trajectory"
    # (b) real v2.2 artifact: whatever the verdict, it is one of the enum and never silently upgraded
    v = export_v2_2_handoff(tmp_path / "real", DEFAULT_CFG(layout="subject1146", top="qI"))
    assert v in {"phase_map_trajectory", "mechanism_candidate_only", "refused"}
    # if refused/candidate, a blocking reason file exists
    if v != "phase_map_trajectory":
        assert (tmp_path / "real" / "m3a_interface_audit.json").exists()
```

- [ ] **Step 7: Run to verify fail** → FAIL (ImportError).
- [ ] **Step 8: Implement `export_fixture_handoff` and `export_v2_2_handoff`** in `src/topic4_criticality.py`:

```python
def export_v2_2_handoff(out_dir, cfg):
    import os; os.makedirs(out_dir, exist_ok=True)
    from src.sef_hfo_transition_sim import run_transition
    from src.sef_hfo_m3a_export import build_handoff_from_sim, write_handoff_artifacts
    from src.sef_hfo_m3_interface import audit_m3a_interface  # fail-closed gate
    res = run_transition(cfg)
    h = build_handoff_from_sim(res["slow"], res["events"], cfg["dt_ms"],
                               mapping_id="m3a_v2_2_approach",
                               gk_enabled=cfg["use_gK"], hG_enabled=cfg["use_hG"])
    write_handoff_artifacts(out_dir, **h)
    audit = audit_m3a_interface(out_dir)     # writes m3a_interface_audit.json, returns dict w/ overlay_verdict
    return audit["overlay_verdict"]          # never relaxed; real gate may refuse
```

`export_fixture_handoff`: build a hand-crafted landmark/mapping that is guaranteed sign-cal + range valid (so it returns `phase_map_trajectory`) — proves the machinery, isolates real-data refusal as science not adapter bug.

- [ ] **Step 9: Run to verify pass** → PASS.
- [ ] **Step 10: Commit** — `git add src/sef_hfo_transition_sim.py scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py scripts/run_topic4_crit_export.py src/topic4_criticality.py tests/test_topic4_crit_integration.py && git commit -m "feat(topic4-crit): T1 factor v2.2 sim to src + interface export (two-layer fail-closed)"`

---

## Task 2: Normalized phase grid (conditional_2d_atlas, contract D1)

**Files:** Modify `scripts/run_topic4_crit_atlas.py`; Test integration.
**Interfaces — Produces:** `build_normalized_atlas(mapping, ranges, grid_spec) -> dict` writing `finite_jacobian_grid.json` with `m3a_overlay_consumable=True`, `axes_built_from_slow_to_rate_mapping_id == mapping_id`.

- [ ] **Step 1: Re-read spec §2 P2 + §8.** The existing `build_m3b_spectral_outputs.py` builds a raw-knob atlas (`mu_core×q_global`, `m3a_overlay_consumable=False`). We rebuild axes as normalized `phase_x_core × phase_y_global ∈ [0,1]` from the T1 mapping/ranges.

- [ ] **Step 2: Write failing test**

```python
@pytest.mark.integration
def test_normalized_atlas_is_overlay_consumable(tmp_path):
    from src.topic4_criticality import build_normalized_atlas
    import json
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    mapping, ranges = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    out = build_normalized_atlas(mapping, ranges, grid_spec={"n": 6, "L": 5.0}, out_dir=tmp_path)
    meta = json.loads((tmp_path / "finite_jacobian_grid.json").read_text())
    assert meta["m3a_overlay_consumable"] is True
    assert meta["axes_built_from_slow_to_rate_mapping_id"] == "m3a_v2_2_approach"
    assert meta["x_axis"].startswith("phase_x_core") and meta["y_axis"].startswith("phase_y_global")
    # h_G projected -> honest naming
    assert meta["atlas_name"].startswith("conditional_2d_atlas_at_phase_recovery")
```

- [ ] **Step 3: Run to verify fail** → FAIL.
- [ ] **Step 4: Implement `build_normalized_atlas`** — map each normalized `(phase_x_core, phase_y_global)` grid node back to `(mu_core, q_global)` via the mapping's inverse transform, call `spm.solve_operating_point` + `spm.build_phase_map` at those, write `finite_jacobian_grid.json` with the provenance fields (spec §8) and `atlas_name="conditional_2d_atlas_at_phase_recovery=<value>"`.
- [ ] **Step 5: Run to verify pass** → PASS.
- [ ] **Step 6: Commit** — `git commit -am "feat(topic4-crit): T2 normalized phase grid (overlay-consumable, conditional_2d_atlas)"`

---

## Task 3a: Branch-aware atlas + eigen-metrics + non-normality + quality gate + verdict

Decomposed into 3a-1…3a-5. Each sub-task is an independently testable deliverable.

### Task 3a-1: Operating-point quality gate (pure functions)

**Files:** Modify `src/topic4_criticality.py`; Test `tests/test_topic4_criticality.py`.
**Interfaces — Produces:** `rate_mismatch_rel(rate_sim, z_star)->float`, `adiabatic_index(slow_speed, alpha1, slow_scale)->float`, `qualify_point(op_fields, cfg)->(bool, reason:str)`.

- [ ] **Step 1: Write failing test** (spec §3)

```python
import numpy as np
from src.topic4_criticality import rate_mismatch_rel, adiabatic_index, qualify_point, load_crit_config
def test_quality_gate_normalized_and_adiabatic():
    z = np.array([2.0, 2.0, 2.0, 2.0]); sim = z + 0.1
    assert abs(rate_mismatch_rel(sim, z) - 0.05) < 1e-9          # 0.1/2.0
    assert adiabatic_index(slow_speed=1.0, alpha1=-2.0, slow_scale=1.0) == 0.5   # 1*(1/2)/1
    c = load_crit_config()
    ok, why = qualify_point({"converged": True, "saturated": False, "residual_rms": 1e-6,
                             "rate_mismatch_rel": 0.01, "slow_mismatch_rel": 0.01,
                             "adiabatic_index": 0.05}, c)
    assert ok and why == "qualified"
    bad, why2 = qualify_point({"converged": True, "saturated": False, "residual_rms": 1e-6,
                              "rate_mismatch_rel": 0.01, "slow_mismatch_rel": 0.01,
                              "adiabatic_index": 0.9}, c)
    assert (not bad) and why2 == "not_quasistatic"
```

- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement**

```python
def rate_mismatch_rel(rate_sim, z_star, eps=1e-9):
    import numpy as np
    a = np.asarray(rate_sim, float).ravel(); b = np.asarray(z_star, float).ravel()
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    return rms / (float(np.median(np.abs(b))) + eps)
def adiabatic_index(slow_speed, alpha1, slow_scale, eps=1e-9):
    tau_fast = (-1.0 / alpha1) if alpha1 < 0 else float("inf")
    return float(slow_speed) * tau_fast / (float(slow_scale) + eps)
def qualify_point(f, cfg):
    g = cfg["quality_gate"]
    if not f["converged"]:            return (False, "nonconverged")
    if f["saturated"]:                return (False, "saturated")
    if f["residual_rms"] >= g["residual_rms_tol"]:        return (False, "high_residual")
    if f["rate_mismatch_rel"] >= g["rate_mismatch_rel_tol"]: return (False, "rate_mismatch")
    if f["slow_mismatch_rel"] >= g["slow_mismatch_rel_tol"]: return (False, "slow_mismatch")
    if f["adiabatic_index"] >= g["adiabatic_index_tol"]:  return (False, "not_quasistatic")
    return (True, "qualified")
```

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic4-crit): T3a-1 operating-point quality gate (normalized RMS + adiabatic)"`

### Task 3a-2: `solve_operating_point` init arg + branch protocol

**Files:** Modify `src/topic4_m3b_spectral_phase.py:403` (add `init=`); `src/topic4_criticality.py` (`solve_branches`); Test both.
**Interfaces — Produces:** `solve_operating_point(..., init: dict|None=None)`; `solve_branches(grid, kernels, exc, inh, cfg, prev=None) -> list[Branch]` with `branch_id∈{low,high,saturated,ambiguous}`.

- [ ] **Step 1: Write failing test for `init=`**

```python
def test_solve_operating_point_accepts_init_and_reaches_distinct_branches():
    import numpy as np, sys; sys.path.insert(0, "src")
    from src.topic4_m3b_spectral_phase import solve_operating_point, Grid, build_kernels, make_core_mask, ExcitabilityField, InhibitionField
    g = Grid(n=6, L=5.0); k = build_kernels(g, ell_perp=0.6); core = make_core_mask(g, "single", 0.9)
    exc = ExcitabilityField.from_core(core, mu_core=0.9); inh = InhibitionField.uniform(g, q=0.94)
    lo = solve_operating_point(g, k, exc, inh, init={"rE": 1e-3, "rI": 1e-3})
    hi = solve_operating_point(g, k, exc, inh, init={"rE": 5.0, "rI": 5.0})
    assert lo.rE.mean() != hi.rE.mean() or hi.saturated   # init can reach a different branch
```

- [ ] **Step 2: Run to verify fail** → FAIL (unexpected kwarg `init`).
- [ ] **Step 3: Add `init=None` to `solve_operating_point`** — before line 427's `try:` block, if `init` given seed `rE=full(n,n, init["rE"])`, `rI=full(...)` instead of the `mean_field` guess. Everything else unchanged (keeps the `mean_field` default when `init is None` → byte-parity for existing callers).
- [ ] **Step 4: Run to verify pass** → PASS. Also run existing M3B tests to confirm no regression: `pytest tests/ -k m3b -q`.
- [ ] **Step 5: Write failing branch-protocol test** (spec §5)

```python
from src.topic4_criticality import solve_branches, load_crit_config
def test_solve_branches_labels_low_high_saturated():
    import sys; sys.path.insert(0, "src")
    from src.topic4_m3b_spectral_phase import Grid, build_kernels, make_core_mask, ExcitabilityField, InhibitionField
    g = Grid(n=6, L=5.0); k = build_kernels(g, ell_perp=0.6); core = make_core_mask(g, "single", 0.9)
    exc = ExcitabilityField.from_core(core, mu_core=0.9); inh = InhibitionField.uniform(g, q=0.94)
    brs = solve_branches(g, k, exc, inh, load_crit_config())
    ids = {b.branch_id for b in brs}
    assert "low_branch" in ids or "saturated_branch" in ids
    assert all(hasattr(b, "branch_alpha1") and hasattr(b, "branch_selected_reason") for b in brs)
```

- [ ] **Step 6: Run to verify fail** → FAIL.
- [ ] **Step 7: Implement `solve_branches`** — solve from each of `cfg["branching"]["solve_inits"]` (`low_rate`→small, `high_rate`→large, `previous_point`→`prev.rE`, `random_small`→jittered small using a fixed seed derived from grid index, no `Math.random`), cluster the resulting `op.rE.mean()` by `branch_cluster_rate_tol` (relative), label `saturated_branch` if `op.saturated`, else `low_branch`/`high_branch` by relative rate, `ambiguous_branch` if a cluster mixes. Compute `branch_alpha1` via `rate_eigenpairs` on each branch's `build_jacobian_dense`. Return dataclass list.
- [ ] **Step 8: Run to verify pass** → PASS.
- [ ] **Step 9: Commit** — `git commit -am "feat(topic4-crit): T3a-2 solve_operating_point init arg + branch-aware protocol"`

### Task 3a-3: Eigen-metrics — complex-pair loading, next-distinct gap, pair controllability

**Files:** Modify `src/topic4_m3b_spectral_phase.py` (add `next_distinct_gap`, `pair_loading`); `src/topic4_criticality.py`; Test.
**Interfaces — Produces:** `next_distinct_gap(eigenvalues, min_sep)->float`; `pair_loading(R, idx_pair, grid)->ndarray` (non-negative, complex-pair energy); `pair_controllability(L, idx_pair, b_core)->float`.

- [ ] **Step 1: Write failing test** (spec §6; the current `spectral_gap`=`e[0].real−e[1].real` zeros out for a conjugate pair)

```python
import numpy as np
from src.topic4_m3b_spectral_phase import next_distinct_gap
def test_next_distinct_gap_skips_conjugate_pair():
    # leading conjugate pair at -0.1 ± 3i, next real level at -0.5
    ev = np.array([-0.1+3j, -0.1-3j, -0.5+0j])
    # naive gap would be 0 (pair members share Re); next-distinct must be -0.1 - (-0.5) = 0.4
    assert abs(next_distinct_gap(ev, min_sep=1e-3) - 0.4) < 1e-9
```

- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement `next_distinct_gap`**

```python
def next_distinct_gap(eigenvalues, min_sep=1e-3):
    import numpy as np
    re = np.sort(np.real(np.asarray(eigenvalues)))[::-1]
    a1 = re[0]
    for r in re[1:]:
        if abs(a1 - r) > min_sep:
            return float(a1 - r)
    return float("inf")
```

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Write failing pair-loading + controllability test**

```python
from src.topic4_m3b_spectral_phase import pair_loading
def test_pair_loading_is_nonneg_and_combines_conjugate_members():
    import numpy as np
    from src.topic4_m3b_spectral_phase import Grid
    g = Grid(n=2, L=1.0); N = g.size
    v1 = np.zeros(6*N, complex); v1[0] = 1+1j; v2 = np.conj(v1)
    R = np.column_stack([v1, v2])
    load = pair_loading(R, (0, 1), g)     # rE-field loading of the invariant 2D subspace
    assert load.shape == (g.n, g.n) and np.all(load >= 0)
    assert abs(load.flat[0] - np.sqrt(abs(v1[0])**2 + abs(v2[0])**2)) < 1e-9
```

- [ ] **Step 6: Run to verify fail** → FAIL.
- [ ] **Step 7: Implement `pair_loading`** (rE-field slice, `sqrt(|R[:,i]|²+|R[:,j]|²)`, reshape to grid) and `pair_controllability(L, idx_pair, b_core)` = `sqrt(|ψ_i^H b|²+|ψ_j^H b|²)`. Feed `core_overlap`/`globality` the non-negative `pair_loading` instead of a signed single eigenvector.
- [ ] **Step 8: Run to verify pass** → PASS.
- [ ] **Step 9: Commit** — `git commit -am "feat(topic4-crit): T3a-3 complex-pair loading + next-distinct gap + pair controllability"`

### Task 3a-4: Non-normality (numerical abscissa; finite-time gain over horizons)

**Files:** Modify `src/topic4_criticality.py`; Test.
**Interfaces — Produces:** `numerical_abscissa(J)->float`; `finite_time_gain_curve(J, b_core, horizons_ms)->dict` (reuses `spm.finite_time_gain`), `transient_amplification_present(curve, alpha1)->bool`.

- [ ] **Step 1: Write failing test** (spec §7; non-normal matrix: Re(λ)<0 but transient gain>1)

```python
import numpy as np
from src.topic4_criticality import numerical_abscissa, transient_amplification_present
def test_numerical_abscissa_positive_for_nonnormal_stable():
    J = np.array([[-1.0, 10.0], [0.0, -2.0]])       # stable eigenvalues, strongly non-normal
    assert numerical_abscissa(J) > 0                # max eig((J+J^T)/2) > 0 => transient growth
```

- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement**

```python
def numerical_abscissa(J):
    import numpy as np
    Jm = np.asarray(J, float); S = 0.5 * (Jm + Jm.T)
    return float(np.max(np.linalg.eigvalsh(S)))
def transient_amplification_present(curve, alpha1, gain_thresh=1.5):
    return bool(max(curve.values()) > gain_thresh)   # G_max over horizons exceeds ~1.5
```

`finite_time_gain_curve`: loop `spm.finite_time_gain(J, grid, core, T)` over `horizons_ms`, return `{T: G_T}`, plus `G_max`/`T_at_G_max`.

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat(topic4-crit): T3a-4 non-normality (numerical abscissa + finite-time-gain curve)"`

### Task 3a-5: Verdict + report quantities + atlas assembly + Figure 1

**Files:** Modify `src/topic4_criticality.py`, `scripts/run_topic4_crit_atlas.py`; Test both.
**Interfaces — Produces:** `classify_trajectory(points, cfg) -> dict` with `verdict∈{smooth_CSD,hard_jump_no_CSD,unresolved_operating_point}` + report fields.

- [ ] **Step 1: Write failing test** (spec §1/§4; naming: closest-to-zero = max of negative α₁; τ only α<0)

```python
import numpy as np
from src.topic4_criticality import classify_trajectory, load_crit_config
def test_verdict_smooth_vs_hard_vs_unresolved():
    c = load_crit_config()
    # smooth: qualified low-branch alpha1 climbs -0.5 -> -0.01 (reaches ~0), tau grows
    smooth = [{"alpha1": a, "qualified": True, "branch_id": "low_branch"} for a in np.linspace(-0.5, -0.01, 8)]
    r = classify_trajectory(smooth + [{"alpha1": None, "qualified": False, "saturated": True, "branch_id":"saturated_branch"}], c)
    assert r["verdict"] == "smooth_CSD"
    assert r["alpha1_closest_to_zero_pre_onset"] == max(p["alpha1"] for p in smooth)   # NOT min
    # hard: last qualified low-branch alpha1 stays < -alpha_margin_hard, then saturates within window
    hard = [{"alpha1": a, "qualified": True, "branch_id": "low_branch"} for a in np.linspace(-0.6, -0.2, 8)]
    r2 = classify_trajectory(hard + [{"alpha1": None, "qualified": False, "saturated": True, "branch_id":"saturated_branch"}], c)
    assert r2["verdict"] == "hard_jump_no_CSD"
    assert r2["jump_distance_to_alpha0"] == abs(hard[-1]["alpha1"])
    # unresolved: too few qualified points
    r3 = classify_trajectory([{"alpha1": None, "qualified": False, "branch_id":"ambiguous_branch"}], c)
    assert r3["verdict"] == "unresolved_operating_point"
```

- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement `classify_trajectory`** — filter qualified low-branch points; if `< min_qualified_points` → `unresolved_operating_point`. Else `q=[p["alpha1"] for qualified low-branch]`; `alpha1_closest_to_zero_pre_onset=max(q)`; `last_stable_alpha1=q[-1]`; `jump_distance_to_alpha0=abs(q[-1])`. `smooth_CSD` iff `max(q) >= -alpha_near_zero_tol` AND Spearman(q, index) `>= smooth_min_alpha_spearman` AND tau-growth ratio `>= smooth_min_tau_growth_ratio`. `hard_jump_no_CSD` iff `last_stable_alpha1 < -alpha_margin_hard` AND `max(q) < -alpha_near_zero_tol` AND a saturated point follows within `jump_window_ms`. Else `unresolved_operating_point`. Emit `tau_ms=-1/α₁` (α₁<0) else `NaN`; `instability_growth_time=1/α₁` (α₁>0).

- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Integration — assemble atlas + verdict + Figure 1** in `scripts/run_topic4_crit_atlas.py`: build normalized atlas (T2), `solve_branches` per node (T3a-2), eigen-metrics with `next_distinct_gap`/`pair_loading` (T3a-3), non-normality (T3a-4), quality gate (T3a-1) per trajectory point, overlay the T1 `phase_trajectory.csv` **only if its `overlay_verdict==phase_map_trajectory`**, run `classify_trajectory`. Write `results/topic4_criticality/{finite_jacobian_grid.json, trajectory_verdict.json, STATUS.md}` + `figures/` (α₁=0 contour on `qualified_low_branch` mask only + mode-class map + v2.2 overlay) + `figures/README.md` (中文). Integration test asserts `trajectory_verdict.json` has `verdict`, `alpha1_closest_to_zero_pre_onset`, `operator_type=="continuous_jacobian"`.

- [ ] **Step 6: Run** — `pytest tests/test_topic4_criticality.py tests/test_topic4_crit_integration.py -q` all pass; eyeball Figure 1.
- [ ] **Step 7: Commit** — `git commit -am "feat(topic4-crit): T3a-5 three-way verdict + atlas assembly + Figure 1 (branch-aware, alpha1=0 on qualified mask)"`

---

## Milestone-1 Hard QC

- [ ] `trajectory_verdict.json` carries `operator_type=continuous_jacobian`, `alpha1_per_ms`, `tau_ms` (NaN where α₁≥0), verdict ∈ the 3 enum.
- [ ] `α₁=0` contour drawn ONLY on `qualified_low_branch` mask (not saturated/nonconverged).
- [ ] verdict NOT preset to CSD; `hard_jump_no_CSD` requires branch-continuation (qualified low-branch margin) — a run that only saturates without qualified low-branch points is `unresolved`, not hard_jump.
- [ ] overlay drawn only when `overlay_verdict==phase_map_trajectory`; else STATUS.md records `mechanism_candidate_only`/`refused` + reason.
- [ ] existing M3B tests still green (`pytest tests/ -k m3b -q`); `solve_operating_point(init=None)` byte-parity preserved.

---

## Follow-on plans (NOT in this plan)

- **Milestone 2** (`2026-07-…-topic4-criticality-milestone2.md`): **T3b** two-layer spot-check (T3b-rate rate-field nonlinear perturbation `z*+ε·Re(v)` primary; T3b-snn input-projected E-current, direction-consistency) + **T3c** slow-var attribution (lock `g_K`/`h_G` into `_moments()` per §18 `slow_to_ratefield`; `∂α₁/∂{q_I,g_K,h_G}` central diff + trajectory contribution) + **T3d** controls (no-core / isotropic AR=1 / shuffled-core / branch-control / ramp-rate). No phase2.
- **Milestone 3** (gated on topic5 phase2): **T4** 3×3 correspondence + virtual-SEEG proxy (reuse topic5 estimator code path, matched_10ch, `mode_observability`) + Topic5 interface prediction vector (axis/nonaxis/global).

---

## Self-Review

1. **Spec coverage (Milestone 1 scope = T0,T1,T2,T3a)**: T0 config §18 → Task 0; T1 export + fail-closed §8/§2 → Task 1; T2 normalized grid §2/§8 → Task 2; quality gate §3 → 3a-1; branch-aware §5 → 3a-2; complex-pair/next-distinct §6 → 3a-3; non-normality §7 → 3a-4; verdict/report/naming §1/§4 + α₁=0 mask §10 → 3a-5. T3b/c/d(§9/§11/§12)+T4(§13/§14/§15) → follow-on plans (declared). **Covered.**
2. **Placeholder scan**: config `...` removed (initial values given); every code step has real test+impl. Integration steps that assemble existing helpers name the exact functions. **OK** (branch-clustering/atlas-assembly bodies describe concrete operations over named helpers; refined during TDD).
3. **Type consistency**: `load_crit_config` (T0) keys used consistently in 3a-1/3a-5; `solve_operating_point(init=)` (3a-2) matches `solve_branches`; `next_distinct_gap`/`pair_loading` (3a-3) names match 3a-5 usage; `overlay_verdict` string (T1) matches 3a-5 gate. **OK.**
