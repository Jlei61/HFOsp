# Topic 4 M3-v2.2 Approach-Criticality — Milestone 1 (T0–T3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).
> **Spec of record:** `docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md` (rev2.1). **Plan status: post-review rev1** (folds the 26-point plan review; §27 10 blocking edits all applied).
> Re-read the relevant spec § before each task body (CLAUDE.md §5/§6).

**Goal:** Build the branch-aware frozen-Jacobian **verdict on the real M3-v2.2 slow trajectory (including `h_G(t)`)** and emit a pre-registered three-way critical-slowing-down verdict. **Output framing (locked): "branch-aware frozen-Jacobian PRELIMINARY verdict, pending Milestone-2 perturbation spot-check / slow-var attribution / controls."** Never "model proves CSD exists/absent."

**Architecture:** Extend M3B machinery (`src/topic4_m3b_spectral_phase.py`), don't rebuild — `rate_eigenpairs` already gives biorthonormal left/right + residuals. **Verdict is computed from the actual 3-D operating point at each real trajectory point `(q_I(t), g_K(t), h_G(t))`**, NOT by sampling the 2-D atlas (the 2-D atlas is visualization/context only). Model-side only — no phase2, no SNN perturbation (Milestone 2).

**Tech Stack:** Python, numpy, scipy (`scipy.linalg.eig`, `scipy.linalg.expm`), PyYAML, pytest. SNN engine via `sys.path.insert(0, "src/snn_engine")`.

## Global Constraints (verbatim from spec rev2.1)

- `α₁` = **continuous-time** frozen-Jacobian leading real-part eigenvalue, **per-ms** (`J` confirmed continuous). `τ=−1/α₁` only for α₁<0; `instability_growth_time=1/α₁` for α₁>0.
- **不预设 `α₁→0`**: three pre-registered verdicts; `runaway`=saturation (`classify_mode` line 954), α₁ may stay negative.
- **verdict from real 3-D trajectory (incl `h_G(t)`)**, not the 2-D conditional atlas.
- **CSD read only on quality-gated quasi-static points** (converged ∧ ¬saturated ∧ residual ∧ rate-mismatch ∧ slow-mismatch ∧ adiabatic ∧ alpha-drift); else `trajectory_not_linearizable`/`not_quasistatic`.
- **branch-aware** (CSD reads low/approach branch only, field-level clustering); **hard_jump requires branch-continuation bisection**.
- **complex conjugate pair = one invariant 2-D subspace**; spectral gap = next-**distinct** real part; left-vector projection normalized (`left_mode_input_projection`, not "controllability" until standardized).
- **finite-time gain: directional vs operator norm named separately** (Milestone-1 = directional only).
- **overlay only when interface gate passes** (fail-closed); atlas with `h_G` projected named `conditional_2d_atlas_at_phase_recovery=...`.
- **results dir**: `results/topic4_criticality/`. Commit **new files with explicit `git add`** (never `-am` for new files).
- execute only against spec rev2.1; T4 gated on topic5 phase2 — not this plan.

---

## File Structure

- Create `config/topic4_criticality.yaml` (spec §18 + review threshold_sweep / rate floors / atlas condition).
- Create `src/sef_hfo_transition_sim.py` — `run_transition(cfg)` factored from the figure script + `sim_dict_for_handoff(res)` adapter (maps v2.2 field traces → the `trace_core/trace_global/trace_gk/spk/posE` keys `build_handoff_from_sim` reads).
- Create `src/topic4_criticality.py` — config loader, quality gate, branch clustering, verdict, non-normality, `evaluate_actual_trajectory_points`, export wrappers.
- Modify `src/topic4_m3b_spectral_phase.py` — `solve_operating_point(init=...)` (scalar|array); **`g_K`/`h_G` entry into `_moments()`** (slow_to_ratefield, P1-1); `next_distinct_gap`, `leading_subspace_indices`, `pair_loading` (via `mode_e_field`), `left_mode_input_projection`.
- Modify `scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py` — import factored `run_transition` (golden-fixture parity).
- Create `scripts/run_topic4_crit_export.py`, `scripts/run_topic4_crit_atlas.py` (both with CLI smoke tests).
- Fixtures `tests/fixtures/topic4_m3v2_2_transition_golden.{npz,json}`; Tests `tests/test_topic4_criticality.py`, `tests/test_topic4_crit_integration.py`.

---

## Task 0: Config + terminology lock

**Files:** Create `config/topic4_criticality.yaml`, `src/topic4_criticality.py`; Test `tests/test_topic4_criticality.py`.
**Produces:** `load_crit_config(path=None)->dict`.

- [ ] **Step 1: Write config** (spec §18 + review #2/#4/#8/#14/#22 additions)

```yaml
# config/topic4_criticality.yaml
operator: {type: continuous_jacobian, dt_ms: null, alpha_units: per_ms, tau_units: ms}
quality_gate:
  residual_rms_tol: 1.0e-4
  rate_scale_floor: 0.05            # #8 quiet-branch absolute floor (kHz)
  rate_mismatch_abs_tol: 0.05
  rate_mismatch_rel_tol: 0.10
  slow_mismatch_rel_tol: 0.05
  adiabatic_index_tol: 0.20
  alpha_drift_index_tol: 0.20       # #7 now actually enforced
  min_qualified_points: 5
  min_qualified_fraction: 0.30      # #19 now actually enforced
verdict:
  threshold_policy: initial_sensitivity_required     # #4
  alpha_near_zero_tol_per_ms: 0.002                   # primary (τ≈500 ms); NOT final
  alpha_margin_hard_per_ms: 0.01
  threshold_sweep:                                    # #4 report verdict stability
    alpha_near_zero_tol_per_ms: [0.001, 0.002, 0.005, 0.01, 0.02]
    alpha_margin_hard_per_ms: [0.005, 0.01, 0.02, 0.05]
  jump_window_ms: 100.0
  smooth_min_tau_growth_ratio: 2.0
  smooth_min_alpha_spearman: 0.6
  unresolved_if_branch_ambiguous: true               # #20
  branch_continuation_n_bisect: 8                     # #2/#3
branching:
  solve_inits: [low_rate, previous_point, high_rate, random_small]
  branch_cluster_field_tol: 0.20     # #9 field-level relative distance to merge branches
  selected_branch: approach_low_branch
mode:
  complex_pair_policy: invariant_subspace_loading
  spectral_gap_policy: next_distinct_real_part
  next_distinct_min_sep_per_ms: 1.0e-3
  imag_tol_per_ms: 1.0e-3            # #12 |Im| above this = complex leading pair
finite_time_gain: {horizons_ms: [10, 25, 50, 100, 250, 500], mode: directional_core, report_numerical_abscissa: true, gain_thresh: 1.5}
slow_to_ratefield:                   # #P1-1: how each slow var enters the rate field (lock + sign test)
  q_I: {target: E_inhibition, entry: inh_q_scales_WEI, uniform: false, sign_test: required}   # already in solve_operating_point
  g_K: {target: E_current, entry: 'muE -= eta_K*g_K(x)', uniform: false, eta_K: 1.0, sign_test: required}
  h_G: {target: E_current, entry: 'muE -= eta_G*h_G',    uniform: true,  eta_G: 1.0, sign_test: required}
slow_sensitivity: {finite_difference: central, step_fraction_qI: 0.05, step_fraction_gK: 0.05, step_fraction_hG: 0.05, require_both_sides_qualified: true}
atlas:
  normalized_grid_n: 31
  grid_L: 5.0
  phase_recovery_condition: {policy: trajectory_median, fixed_value: null}   # #22
virtual_seeg: {use_topic5_estimator_code: true, channel_sets: [source_all_nodes, virtual_all_contacts, matched_10ch], same_windows_as_topic5: true}
tier: model_side_ground_truth_preliminary
```

- [ ] **Step 2: Failing loader test**

```python
from src.topic4_criticality import load_crit_config
def test_config_locks_units_verdicts_and_review_additions():
    c = load_crit_config()
    assert c["operator"]["alpha_units"] == "per_ms"
    assert c["verdict"]["alpha_near_zero_tol_per_ms"] == 0.002          # #4 per-ms, low default
    assert "threshold_sweep" in c["verdict"]                             # #4
    assert c["quality_gate"]["rate_scale_floor"] > 0                     # #8
    assert c["branching"]["branch_cluster_field_tol"] > 0                # #9
    assert set(c["slow_to_ratefield"]) == {"q_I", "g_K", "h_G"}          # P1-1
    assert c["finite_time_gain"]["mode"] == "directional_core"          # #15
```

- [ ] **Step 3: Run → FAIL. Step 4: Implement loader** (`yaml.safe_load` of the default path). **Step 5: Run → PASS.**
- [ ] **Step 6: Commit** — `git add config/topic4_criticality.yaml src/topic4_criticality.py tests/test_topic4_criticality.py && git commit -m "feat(topic4-crit): T0 config + terminology lock (units, threshold-sweep, floors, slow_to_ratefield)"`

---

## Task 1: Factor v2.2 sim to src + golden-fixture parity + interface export (fail-closed)

**Files:** Create `src/sef_hfo_transition_sim.py`, `scripts/run_topic4_crit_export.py`, `tests/fixtures/topic4_m3v2_2_transition_golden.{npz,json}`; Modify the figure script; Test integration.
**Produces:** `run_transition(cfg)->dict` (keys `rate_E,E_spk_bool,times,trace_qI_mean,trace_qI_min,trace_gK,trace_hG,spk,posE,events`); `sim_dict_for_handoff(res)->dict`; `default_transition_config(layout,top)->dict` (**dict, one type only — #5**); `export_v2_2_handoff(out_dir, cfg)->str`; `export_fixture_handoff(out_dir)->str`.

- [ ] **Step 1: Capture golden fixture BEFORE refactor (#3).** Run the *current* figure sim once and save checksums of ALL outputs:

```python
# one-off, in a scratch script, committed as the fixture:
import numpy as np, hashlib, json
import scripts.paper_figures.plot_fig_m3a_v2_2_hG_runaway_transition_gif as G
S = G._build(G._cfg_subject1146_qI()); ref = G._simulate_continuous(S, G._cfg_subject1146_qI())
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
np.savez("tests/fixtures/topic4_m3v2_2_transition_golden.npz",
         rate_E=ref["rate_E"], trace_qI_mean=ref["trace_qI_mean"])
json.dump({"E_spk_bool_hash": h(ref["E_spk_bool"]), "rate_E_hash": h(ref["rate_E"]),
           "trace_qI_mean_hash": h(ref["trace_qI_mean"]), "trace_qI_min_hash": h(ref["trace_qI_min"]),
           "trace_gK_hash": h(ref["trace_gK"]), "trace_hG_hash": h(ref.get("trace_hG", np.array([]))),
           "n_steps": int(len(ref["times"])), "events": ref["events"]},
          open("tests/fixtures/topic4_m3v2_2_transition_golden.json", "w"))
```

- [ ] **Step 2: Failing golden-parity test (#3 — all outputs, not just spikes)**

```python
import numpy as np, json, hashlib, pytest
def _h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
@pytest.mark.integration
def test_run_transition_matches_golden_fixture():
    from src.sef_hfo_transition_sim import run_transition, default_transition_config
    g = json.load(open("tests/fixtures/topic4_m3v2_2_transition_golden.json"))
    res = run_transition(default_transition_config(layout="subject1146", top="qI"))
    for key in ["E_spk_bool", "rate_E", "trace_qI_mean", "trace_qI_min", "trace_gK"]:
        assert _h(res[key]) == g[f"{key}_hash"], f"{key} changed by factoring"
    assert res["events"] == g["events"]
```

- [ ] **Step 3: Run → FAIL. Step 4:** move `_simulate_continuous` body verbatim into `run_transition(cfg)` (dict cfg), have the figure script import it; do NOT touch RNG order. **Step 5: Run → PASS** (all output hashes match).

- [ ] **Step 6: Failing two-layer fail-closed export test (#P1-4), full-sim-dict signature (#6/#9)**

```python
@pytest.mark.integration
def test_export_fixture_passes_and_real_is_fail_closed(tmp_path):
    from src.topic4_criticality import export_fixture_handoff, export_v2_2_handoff
    from src.sef_hfo_transition_sim import default_transition_config
    assert export_fixture_handoff(tmp_path / "fix") == "phase_map_trajectory"     # machinery proven
    v = export_v2_2_handoff(tmp_path / "real", default_transition_config("subject1146", "qI"))
    assert v in {"phase_map_trajectory", "mechanism_candidate_only", "refused"}   # never silently upgraded
    if v != "phase_map_trajectory":
        assert (tmp_path / "real" / "m3a_interface_audit.json").exists()          # blocking reason written
```

- [ ] **Step 7: Run → FAIL. Step 8: Implement** the adapter + export (pass the **full sim dict**, adapting v2.2 trace keys):

```python
def sim_dict_for_handoff(res):
    # build_handoff_from_sim reads trace_core/trace_global/trace_gk/spk/posE (v1 tank names).
    # v2.2 field carrier: map q_I mean -> global tank proxy, q_I core-region -> core, g_K -> gk.
    return {"trace_core": res["trace_qI_core"], "trace_global": res["trace_qI_mean"],
            "trace_gk": res["trace_gK"], "spk": res["spk"], "posE": res["posE"]}
def export_v2_2_handoff(out_dir, cfg):
    import os; os.makedirs(out_dir, exist_ok=True)
    from src.sef_hfo_transition_sim import run_transition, sim_dict_for_handoff
    from src.sef_hfo_m3a_export import build_handoff_from_sim, write_handoff_artifacts
    from src.sef_hfo_m3_interface import audit_m3a_interface
    res = run_transition(cfg)
    h = build_handoff_from_sim(sim_dict_for_handoff(res), res["events"], cfg["dt_ms"],
                               mapping_id="m3a_v2_2_approach", gk_enabled=cfg["use_gK"])
    write_handoff_artifacts(out_dir, **h)
    return audit_m3a_interface(out_dir)["overlay_verdict"]     # fail-closed; may refuse
```

`export_fixture_handoff`: hand-built sign-cal-valid mapping guaranteed to pass → isolates real-data refusal as science, not adapter bug.

- [ ] **Step 9: Run → PASS. Step 10: Commit** — `git add src/sef_hfo_transition_sim.py scripts/run_topic4_crit_export.py tests/fixtures/topic4_m3v2_2_transition_golden.npz tests/fixtures/topic4_m3v2_2_transition_golden.json tests/test_topic4_crit_integration.py && git add scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py src/topic4_criticality.py && git commit -m "feat(topic4-crit): T1 factor v2.2 sim + golden-fixture parity + full-sim-dict interface export (fail-closed)"`

---

## Task 2: Conditional 2-D atlas (VISUALIZATION/CONTEXT ONLY — not verdict source)

**Files:** Modify `scripts/run_topic4_crit_atlas.py`; Test integration.
**Produces:** `build_conditional_atlas(mapping, ranges, cfg, out_dir)->dict` writing `finite_jacobian_grid.json` with `m3a_overlay_consumable=True`, `atlas_name="conditional_2d_atlas_at_phase_recovery=<policy>:<value>"`, `verdict_source="actual_trajectory_not_atlas"`.

- [ ] **Step 1: Failing test (#1 label + #22 condition value)**

```python
@pytest.mark.integration
def test_atlas_is_conditional_and_not_verdict_source(tmp_path):
    from src.topic4_criticality import build_conditional_atlas, load_crit_config
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    import json
    m, r = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    build_conditional_atlas(m, r, load_crit_config(), out_dir=tmp_path)
    meta = json.loads((tmp_path / "finite_jacobian_grid.json").read_text())
    assert meta["m3a_overlay_consumable"] is True
    assert meta["atlas_name"].startswith("conditional_2d_atlas_at_phase_recovery=")
    assert meta["verdict_source"] == "actual_trajectory_not_atlas"          # #1 guard
    assert meta["axes_built_from_slow_to_rate_mapping_id"] == "m3a_v2_2_approach"
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** — normalized `phase_x_core×phase_y_global` grid at fixed `phase_recovery` (policy=`trajectory_median` → compute from the T1 trajectory, record the value in `atlas_name`); solve op + phase map at nodes; write meta with `verdict_source` guard. **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** — `git add scripts/run_topic4_crit_atlas.py tests/test_topic4_crit_integration.py && git commit -m "feat(topic4-crit): T2 conditional 2D atlas (viz-only; verdict_source=actual_trajectory guard)"`

---

## Task 2.5: Lock + wire `slow_to_ratefield` (g_K/h_G into the operating point) — prereq for the 3-D verdict

> **Why here (not Milestone 2):** blocking-#1 requires the verdict from the real `(q_I,g_K,h_G)`. To solve the op there, `g_K`/`h_G` must enter `_moments()`. (The `∂α₁/∂slow` finite-diff attribution stays Milestone 2.)

**Files:** Modify `src/topic4_m3b_spectral_phase.py` (`_moments`/`solve_operating_point`, `build_jacobian_dense` — the `g_K`/`h_G` terms are constants w.r.t. the fast state so they shift `muE` only, not the Jacobian blocks); Test.
**Produces:** `solve_operating_point(..., gK_field=None, hG_scalar=0.0, eta_K=1.0, eta_G=1.0)`; sign-test helper `slow_to_ratefield_sign_ok(cfg)->dict`.

- [ ] **Step 1: Failing sign-test (#P1-1: raising g_K/h_G lowers E drive → lowers α₁)**

```python
def test_slow_to_ratefield_signs_lower_excitability():
    import numpy as np, sys; sys.path.insert(0, "src")
    from src.topic4_m3b_spectral_phase import solve_operating_point, build_jacobian_dense, rate_eigenpairs, Grid, build_kernels, make_core_mask, ExcitabilityField, InhibitionField
    g = Grid(n=6, L=5.0); k = build_kernels(g, ell_perp=0.6); core = make_core_mask(g, "single", 0.9)
    exc = ExcitabilityField.from_core(core, mu_core=1.0); inh = InhibitionField.uniform(g, q=0.94)
    a0 = rate_eigenpairs(build_jacobian_dense(g, k, solve_operating_point(g, k, exc, inh)), g).eigenvalues[0].real
    hi = solve_operating_point(g, k, exc, inh, hG_scalar=2.0, eta_G=1.0)     # more global recovery current
    a1 = rate_eigenpairs(build_jacobian_dense(g, k, hi), g).eigenvalues[0].real
    assert a1 <= a0 + 1e-9        # h_G suppresses E -> alpha1 not higher
    assert hi.rE.mean() <= solve_operating_point(g, k, exc, inh).rE.mean() + 1e-9
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** — in `_moments`, add `muE -= eta_G*hG_scalar` (global uniform) and `muE -= eta_K*gK_field` (per-cell); default `hG_scalar=0, gK_field=None` → **byte-parity with existing callers**. The `g_K`/`h_G` currents are additive constants to `muE`, so `build_jacobian_dense` blocks are unchanged (they enter only via the `gE=dΦ/dμ` gain at the shifted op). **Step 4: Run → PASS + existing M3B tests green** (`pytest tests/ -k m3b -q`).
- [ ] **Step 5: Commit** — `git add src/topic4_m3b_spectral_phase.py tests/test_topic4_criticality.py && git commit -m "feat(topic4-crit): T2.5 wire g_K/h_G into operating point (slow_to_ratefield, sign-tested, byte-parity default)"`

---

## Task 3a-1: Operating-point quality gate (pure fns — #5/#7/#8)

**Produces:** `rate_mismatch(rate_sim, z_star, rate_scale_floor)->(abs,rel)`; `adiabatic_index(slow_speed, alpha1, slow_scale)->float`; `qualify_point(fields, cfg)->(bool, reason)`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic4_criticality import rate_mismatch, adiabatic_index, qualify_point, load_crit_config
def test_quality_gate_all_review_fixes():
    c = load_crit_config()
    a, r = rate_mismatch(np.array([2.1,2.1]), np.array([2.0,2.0]), 0.05); assert abs(r-0.05)<1e-9
    aq, rq = rate_mismatch(np.array([0.06,0.0]), np.array([0.0,0.0]), 0.05)     # #8 quiet branch: floor prevents blow-up
    assert rq < 1.0
    assert adiabatic_index(1.0, -2.0, 1.0) == 0.5
    ok, why = qualify_point({"converged":True,"saturated":False,"residual_rms":1e-6,"rate_mismatch_abs":0.01,
        "rate_mismatch_rel":0.01,"slow_mismatch_rel":0.01,"adiabatic_index":0.05,"alpha_drift_index":0.05}, c)
    assert ok and why=="qualified"
    bad, why2 = qualify_point({"converged":True,"saturated":False,"residual_rms":1e-6,"rate_mismatch_abs":0.01,
        "rate_mismatch_rel":0.01,"slow_mismatch_rel":0.01,"adiabatic_index":0.05,"alpha_drift_index":0.9}, c)
    assert (not bad) and why2=="alpha_drift_too_fast"                               # #7 enforced
    miss, why3 = qualify_point({"converged":True,"saturated":False,"residual_rms":1e-6}, c)
    assert (not miss) and why3.startswith("missing_")                              # #5 fail-closed
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
def rate_mismatch(rate_sim, z_star, rate_scale_floor):
    import numpy as np
    a=np.asarray(rate_sim,float).ravel(); b=np.asarray(z_star,float).ravel()
    rms=float(np.sqrt(np.mean((a-b)**2))); scale=max(float(np.median(np.abs(b))), float(rate_scale_floor))
    return rms, rms/scale
def adiabatic_index(slow_speed, alpha1, slow_scale, eps=1e-9):
    tf=(-1.0/alpha1) if alpha1<0 else float("inf"); return float(slow_speed)*tf/(float(slow_scale)+eps)
_REQ=["converged","saturated","residual_rms","rate_mismatch_abs","rate_mismatch_rel",
      "slow_mismatch_rel","adiabatic_index","alpha_drift_index"]
def qualify_point(f, cfg):
    g=cfg["quality_gate"]
    for k in _REQ:
        if k not in f or f[k] is None: return (False, f"missing_{k}")     # #5 fail-closed
    if not f["converged"]: return (False,"nonconverged")
    if f["saturated"]: return (False,"saturated")
    if f["residual_rms"]>=g["residual_rms_tol"]: return (False,"high_residual")
    if f["rate_mismatch_abs"]>=g["rate_mismatch_abs_tol"] and f["rate_mismatch_rel"]>=g["rate_mismatch_rel_tol"]:
        return (False,"rate_mismatch")                                    # #8 both abs AND rel
    if f["slow_mismatch_rel"]>=g["slow_mismatch_rel_tol"]: return (False,"slow_mismatch")
    if f["adiabatic_index"]>=g["adiabatic_index_tol"]: return (False,"not_quasistatic")
    if f["alpha_drift_index"]>=g["alpha_drift_index_tol"]: return (False,"alpha_drift_too_fast")  # #7
    return (True,"qualified")
```

- [ ] **Step 4: Run → PASS. Step 5: Commit** — `git commit -am "feat(topic4-crit): T3a-1 quality gate (abs+rel floor, adiabatic, alpha-drift, missing-field fail-closed)"`

---

## Task 3a-2: `solve_operating_point(init=)` (scalar|array — #10) + branch protocol (field distance #9, deterministic seed #11)

**Produces:** `solve_operating_point(..., init=None)`; `solve_branches(grid, kernels, exc, inh, cfg, *, prev=None, seed_key=None)->list[Branch]` (fields `branch_id, branch_rate_mean, branch_field_distance_to_low, branch_alpha1, branch_residual, branch_selected_reason`).

- [ ] **Step 1: Failing `init=` test (scalar and array)**

```python
def test_init_accepts_scalar_and_array():
    import numpy as np, sys; sys.path.insert(0,"src")
    from src.topic4_m3b_spectral_phase import solve_operating_point, Grid, build_kernels, make_core_mask, ExcitabilityField, InhibitionField
    g=Grid(n=6,L=5.0); k=build_kernels(g,ell_perp=0.6); core=make_core_mask(g,"single",0.9)
    exc=ExcitabilityField.from_core(core,mu_core=0.9); inh=InhibitionField.uniform(g,q=0.94)
    lo=solve_operating_point(g,k,exc,inh,init={"rE":1e-3,"rI":1e-3})       # scalar
    prev=solve_operating_point(g,k,exc,inh,init={"rE":lo.rE,"rI":lo.rI})   # array (#10)
    assert prev.rE.shape==(g.n,g.n)
```

- [ ] **Step 2: Run → FAIL. Step 3:** add `init=None`; `_init_field(x, shape)` = `np.full(shape,float(x))` if scalar else `np.asarray(x,float)` (assert shape). Seed `rE/rI` from init when given, else `mean_field` (byte-parity default). **Step 4: Run → PASS + M3B regression green.**
- [ ] **Step 5: Failing branch test (#9 field distance, #11 deterministic)**

```python
from src.topic4_criticality import solve_branches, load_crit_config
def test_branches_labeled_by_field_distance_deterministic():
    import sys; sys.path.insert(0,"src")
    from src.topic4_m3b_spectral_phase import Grid, build_kernels, make_core_mask, ExcitabilityField, InhibitionField
    g=Grid(n=6,L=5.0); k=build_kernels(g,ell_perp=0.6); core=make_core_mask(g,"single",0.9)
    exc=ExcitabilityField.from_core(core,mu_core=0.9); inh=InhibitionField.uniform(g,q=0.94)
    b1=solve_branches(g,k,exc,inh,load_crit_config(),seed_key=(2,3))
    b2=solve_branches(g,k,exc,inh,load_crit_config(),seed_key=(2,3))       # #11 same seed_key -> identical
    assert [x.branch_id for x in b1]==[x.branch_id for x in b2]
    assert all(hasattr(x,"branch_field_distance_to_low") for x in b1)      # #9 field-level
```

- [ ] **Step 6: Run → FAIL. Step 7: Implement `solve_branches`** — solve from each `cfg["branching"]["solve_inits"]` (`random_small` uses `np.random.default_rng(abs(hash(seed_key)) % 2**32)` — deterministic, no bare `Math.random`); cluster by **field distance** `rms(a.rE-b.rE)/max(floor, median|a.rE|, median|b.rE|)` (#9); label `saturated_branch` if `op.saturated`, else low/high by rate, `ambiguous_branch` if a cluster mixes; `branch_alpha1 = rate_eigenpairs(build_jacobian_dense(...)).eigenvalues[0].real`. **Step 8: Run → PASS. Step 9: Commit** — `git commit -am "feat(topic4-crit): T3a-2 solve_operating_point init(scalar|array) + field-distance branch protocol (deterministic)"`

---

## Task 3a-3: Eigen-metrics — leading subspace, next-distinct gap, pair loading, left projection (#12/#13/#14)

**Produces (in `topic4_m3b_spectral_phase`):** `leading_subspace_indices(eigvals, min_sep, imag_tol)->tuple`; `next_distinct_gap(eigvals, min_sep)->float`; `pair_loading(R, idx, grid)->ndarray` (via `mode_e_field`); `left_mode_input_projection(L, R, idx, b_core)->float` (biorthonormalized).

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from src.topic4_m3b_spectral_phase import next_distinct_gap, leading_subspace_indices, pair_loading
def test_next_distinct_gap_and_leading_subspace():
    ev=np.array([-0.1+3j,-0.1-3j,-0.5+0j])
    assert abs(next_distinct_gap(ev, min_sep=1e-3)-0.4)<1e-9              # skips conjugate member (#5/#12)
    assert set(leading_subspace_indices(ev, min_sep=1e-3, imag_tol=1e-3))=={0,1}   # #12 conj pair
def test_pair_loading_uses_state_helper_nonneg():
    from src.topic4_m3b_spectral_phase import Grid
    g=Grid(n=2,L=1.0); N=g.size
    v1=np.zeros(6*N,complex); v1[0]=1+1j; R=np.column_stack([v1,np.conj(v1)])
    load=pair_loading(R,(0,1),g)                                          # #13 via mode_e_field, not hardcode
    assert load.shape==(g.n,g.n) and np.all(load>=0)
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
def next_distinct_gap(eigenvalues, min_sep=1e-3):
    import numpy as np
    re=np.sort(np.real(np.asarray(eigenvalues)))[::-1]; a1=re[0]
    for r in re[1:]:
        if abs(a1-r)>min_sep: return float(a1-r)
    return float("inf")
def leading_subspace_indices(eigenvalues, min_sep=1e-3, imag_tol=1e-3):
    import numpy as np
    ev=np.asarray(eigenvalues); i=int(np.argmax(ev.real))
    if abs(ev[i].imag)>imag_tol:
        j=int(np.argmin(np.abs(ev-np.conj(ev[i]))))                      # conjugate partner
        return (i,j)
    return tuple(int(x) for x in np.where(np.abs(ev.real-ev[i].real)<=min_sep)[0])
def pair_loading(R, idx, grid):
    import numpy as np
    from src.topic4_m3b_spectral_phase import mode_e_field                # #13 STATE-aware helper
    acc=np.zeros((grid.n,grid.n));
    for k in idx: acc=acc+np.abs(mode_e_field(R[:,k], grid))**2
    return np.sqrt(acc)
def left_mode_input_projection(L, R, idx, b_core):
    import numpy as np
    acc=0.0
    for k in idx:
        c=np.vdot(L[:,k],R[:,k])                                          # #14 biorthonormalize
        psi=L[:,k]/np.conj(c) if abs(c)>1e-300 else L[:,k]/ (np.linalg.norm(L[:,k])+1e-300)
        acc+=abs(np.vdot(psi,b_core))**2
    return float(np.sqrt(acc))
```

Artifact key = `left_mode_input_projection` (NOT "controllability" — #14). **Step 4: Run → PASS. Step 5: Commit** — `git commit -am "feat(topic4-crit): T3a-3 leading-subspace + next-distinct gap + pair loading(mode_e_field) + normalized left projection"`

---

## Task 3a-4: Non-normality (#15/#16/#17)

**Produces:** `numerical_abscissa(J)->float`; `directional_finite_time_gain_curve(J, b, horizons_ms)->dict`; `transient_amplification_present(curve, alpha1, thresh)->bool`.

- [ ] **Step 1: Failing test**

```python
import numpy as np
from src.topic4_criticality import numerical_abscissa, transient_amplification_present
def test_nonnormality_review_fixes():
    J=np.array([[-1.,10.],[0.,-2.]]); assert numerical_abscissa(J)>0        # #16
    assert transient_amplification_present({"10":3.0}, alpha1=-0.5)          # stable + gain -> True
    assert not transient_amplification_present({"10":3.0}, alpha1=0.2)       # #17 alpha>=0 -> modal growth, not transient
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement**

```python
def numerical_abscissa(J):
    import numpy as np
    Jm=np.asarray(J); S=0.5*(Jm+Jm.conj().T)                              # #16 conj().T, keep complex-safe
    return float(np.max(np.linalg.eigvalsh(S).real))
def directional_finite_time_gain_curve(J, b, horizons_ms):
    import numpy as np; from scipy.linalg import expm
    b=np.asarray(b,complex); nb=np.linalg.norm(b)+1e-300
    return {str(int(T)): float(np.linalg.norm(expm(np.asarray(J)*T)@b)/nb) for T in horizons_ms}  # #15 directional
def transient_amplification_present(curve, alpha1, gain_thresh=1.5):
    if alpha1>=0: return False                                            # #17 modal growth, not stable transient
    return max(curve.values())>gain_thresh
```

Artifact records `operator_gain_computed=false` (#15) + `numerical_abscissa` + `Gmax_directional_core`/`T_at_Gmax`. **Step 4: Run → PASS. Step 5: Commit** — `git commit -am "feat(topic4-crit): T3a-4 non-normality (numerical abscissa conj-safe, directional gain, alpha>=0 guard)"`

---

## Task 3a-5: Real 3-D trajectory eval + branch continuation + verdict + Figure 1 (#1/#2/#3/#8/#18/#19/#20)

**Produces:** `evaluate_actual_trajectory_points(sim, mapping, cfg)->list[TrajectoryPoint]`; `check_low_branch_continuation_between(pt_a, pt_b, cfg)->dict`; `classify_trajectory(points, cfg)->dict`.

- [ ] **Step 1: Failing verdict tests (naming #3.1, jump-window #18, fraction #19, ambiguity #20, continuation #2)**

```python
import numpy as np
from src.topic4_criticality import classify_trajectory, load_crit_config
def _pts(alphas, t0=0, dt=10, branch="low_branch"):
    return [{"time_ms":t0+i*dt,"alpha1":a,"qualified":True,"branch_id":branch,"branch_continuation_checked":True}
            for i,a in enumerate(alphas)]
def test_verdicts():
    c=load_crit_config()
    smooth=_pts(np.linspace(-0.5,-0.001,8))+[{"time_ms":80,"alpha1":None,"qualified":False,"saturated":True,"branch_id":"saturated_branch"}]
    r=classify_trajectory(smooth,c); assert r["verdict"]=="smooth_CSD"
    assert r["alpha1_closest_to_zero_pre_onset"]==max(p["alpha1"] for p in smooth if p["qualified"])   # #3.1 max not min
    hard=_pts(np.linspace(-0.6,-0.2,8))+[{"time_ms":85,"alpha1":None,"qualified":False,"saturated":True,"branch_id":"saturated_branch","branch_continuation_checked":True,"continuation_status":"low_branch_remains_far_from_alpha0_until_jump"}]
    r2=classify_trajectory(hard,c); assert r2["verdict"]=="hard_jump_no_CSD" and r2["jump_distance_to_alpha0"]==abs(hard[7]["alpha1"])
def test_hard_requires_continuation_and_window_and_fraction_and_ambiguity():
    c=load_crit_config()
    # #2 no continuation -> unresolved
    noc=_pts(np.linspace(-0.6,-0.2,8)); noc[-1]["branch_continuation_checked"]=False
    noc+= [{"time_ms":85,"alpha1":None,"qualified":False,"saturated":True,"branch_id":"saturated_branch"}]
    assert classify_trajectory(noc,c)["verdict"]=="unresolved_operating_point"
    # #18 saturation outside window -> unresolved
    late=_pts(np.linspace(-0.6,-0.2,8))+[{"time_ms":10000,"alpha1":None,"qualified":False,"saturated":True,"branch_id":"saturated_branch","branch_continuation_checked":True}]
    assert classify_trajectory(late,c)["verdict"]=="unresolved_operating_point"
    # #19 too few qualified fraction -> unresolved
    many=_pts(np.linspace(-0.5,-0.001,5))+[{"time_ms":100+i,"alpha1":None,"qualified":False,"branch_id":"low_branch"} for i in range(95)]
    assert classify_trajectory(many,c)["verdict"]=="unresolved_operating_point"
    # #20 ambiguous near transition -> unresolved
    amb=_pts(np.linspace(-0.5,-0.001,8)); amb[-1]["branch_id"]="ambiguous_branch"
    assert classify_trajectory(amb,c)["verdict"]=="unresolved_operating_point"
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement `classify_trajectory`** — qualified low-branch points `q`; if `len(q)<min_qualified_points` OR `len(q)/len(points)<min_qualified_fraction` → `unresolved`. If any `ambiguous_branch` within `jump_window_ms` of the last qualified point → `unresolved` (#20). `alpha1_closest_to_zero_pre_onset=max(a for a in q)`; `last_stable_alpha1=q[-1]`; `jump_distance_to_alpha0=abs(q[-1])`. **smooth_CSD**: `max(q)>=-alpha_near_zero_tol` ∧ Spearman(q,idx)>=`smooth_min_alpha_spearman` ∧ tau-growth>=`smooth_min_tau_growth_ratio`. **hard_jump_no_CSD**: `last_stable_alpha1<-alpha_margin_hard` ∧ `max(q)<-alpha_near_zero_tol` ∧ a saturated point follows within `jump_window_ms` ∧ `branch_continuation_checked` AND `continuation_status ∈ {low_branch_disappears_before_alpha0, low_branch_remains_far_from_alpha0_until_jump}` (#2/#3). Else `unresolved`. Emit `tau_ms` (α₁<0 only), `instability_growth_time_ms` (α₁>0). Also emit `threshold_sensitivity` by re-running the verdict over `threshold_sweep` (#4).

- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Implement `evaluate_actual_trajectory_points` (#1 — the 3-D verdict source) + `check_low_branch_continuation_between` (#2) + assembly.** For each real trajectory sample `(q_I(t), g_K(t), h_G(t))` from `run_transition`: build `exc/inh` + `gK_field`/`hG_scalar` (T2.5), `solve_branches` (warm-start `prev`), pick `low_branch`, `build_jacobian_dense`, `rate_eigenpairs`, `leading_subspace_indices`→`pair_loading`/`next_distinct_gap`/`left_mode_input_projection`, `numerical_abscissa`+directional gain, compute quality-gate fields (`rate_mismatch` vs `run_transition` rate, `slow_mismatch`, `adiabatic_index` from `dslow/dt`, `alpha_drift_index` from `dα₁/dt`), `qualify_point`. Between last-qualified and first-saturated, `check_low_branch_continuation_between` bisects (`n_bisect`) the slow state, re-solves low-branch, sets `continuation_status`. Write `results/topic4_criticality/{trajectory_verdict.json, STATUS.md}` (verdict + report fields + `operator_type=continuous_jacobian` + threshold_sensitivity + `verdict_source=actual_trajectory`), overlay onto the T2 atlas **only if** T1 `overlay_verdict==phase_map_trajectory`, Figure 1 (`α₁=0` contour on `qualified_low_branch` mask only + mode-class + overlay), `figures/README.md` (中文). Integration test asserts `trajectory_verdict.json.verdict_source=="actual_trajectory"` and verdict ∈ 3 enum.

- [ ] **Step 6: Run** — `pytest tests/test_topic4_criticality.py tests/test_topic4_crit_integration.py -q` pass; eyeball Figure 1.
- [ ] **Step 7: Commit** — `git add scripts/run_topic4_crit_atlas.py src/topic4_criticality.py tests/test_topic4_criticality.py tests/test_topic4_crit_integration.py && git commit -m "feat(topic4-crit): T3a-5 real 3D trajectory verdict + branch-continuation + Figure 1 (verdict NOT from 2D atlas)"`

---

## Task 3a-6: CLI smoke tests (#24)

- [ ] **Step 1: Failing subprocess smoke test**

```python
@pytest.mark.integration
def test_cli_scripts_run_and_write_artifacts(tmp_path):
    import subprocess, sys
    subprocess.run([sys.executable,"scripts/run_topic4_crit_export.py","--config","config/topic4_criticality.yaml","--out",str(tmp_path/"export")], check=True)
    subprocess.run([sys.executable,"scripts/run_topic4_crit_atlas.py","--config","config/topic4_criticality.yaml","--handoff",str(tmp_path/"export"),"--out",str(tmp_path/"atlas")], check=True)
    assert (tmp_path/"atlas"/"trajectory_verdict.json").exists() and (tmp_path/"atlas"/"STATUS.md").exists()
```

- [ ] **Step 2–4:** fail → add `argparse` mains calling the library fns → pass. **Step 5: Commit** — `git add scripts/run_topic4_crit_export.py scripts/run_topic4_crit_atlas.py tests/test_topic4_crit_integration.py && git commit -m "feat(topic4-crit): T3a-6 CLI entrypoints + smoke tests"`

---

## Milestone-1 Hard QC (review §25)

- [ ] `trajectory_verdict.json` computed from the **actual slow trajectory incl `h_G(t)`**, NOT by sampling the 2-D atlas (`verdict_source==actual_trajectory`; atlas `verdict_source` guard present).
- [ ] carries `operator_type=continuous_jacobian`, `alpha_units=per_ms`, `alpha1_per_ms`, `tau_ms` only where α₁<0, `instability_growth_time_ms` only where α₁>0, verdict ∈ {smooth_CSD, hard_jump_no_CSD, unresolved_operating_point}.
- [ ] verdict reported with `threshold_sensitivity` table (alpha_near_zero + alpha_margin_hard sweep).
- [ ] `α₁=0` contour drawn ONLY on `qualified_low_branch` mask (never saturated/nonconverged/ambiguous).
- [ ] `hard_jump_no_CSD` requires: enough qualified low-branch points, `last_stable_alpha1 < -alpha_margin_hard`, saturated transition within `jump_window_ms`, AND explicit branch-continuation showing no skipped low-branch α₁≈0 — else `unresolved`.
- [ ] `unresolved_if_branch_ambiguous` enforced near transition; `min_qualified_fraction` enforced.
- [ ] overlay drawn only when `overlay_verdict==phase_map_trajectory`; else STATUS.md records `mechanism_candidate_only`/`refused` + reason.
- [ ] finite-time-gain labelled `directional_gain` (not `||exp(JT)||₂`); `operator_gain_computed=false`.
- [ ] `left_mode_input_projection` named as such (not "controllability").
- [ ] existing M3B tests green; `solve_operating_point(init=None, hG_scalar=0, gK_field=None)` byte-parity preserved (`pytest tests/ -k m3b -q`).
- [ ] all CLI scripts run from a clean checkout and write expected artifacts.
- [ ] **STATUS.md output framing = "branch-aware frozen-Jacobian PRELIMINARY verdict, pending Milestone-2 spot-check / attribution / controls."** No "model proves CSD exists/absent."

---

## Follow-on plans (NOT here)

- **Milestone 2** (no phase2): T3b two-layer spot-check (T3b-rate rate-field nonlinear `z*+ε·Re(v)` primary; T3b-snn input-projected E-current, direction-consistency); T3c `∂α₁/∂{q_I,g_K,h_G}` central-diff + trajectory contribution (uses the T2.5 wiring); T3d controls (no-core / isotropic AR=1 / shuffled-core / branch-control / ramp-rate).
- **Milestone 3** (gated on topic5 phase2): T4 3×3 correspondence + virtual-SEEG proxy (reuse topic5 estimator, matched_10ch, `mode_observability`) + Topic5 interface prediction vector.

---

## Self-Review

1. **Blocking-edit coverage (review §27)**: #1 3-D verdict→3a-5 `evaluate_actual_trajectory_points` + atlas `verdict_source` guard + T2.5 wiring; #2 continuation→3a-5 `check_low_branch_continuation_between`; #3 golden fixture→T1 Step1-2; #4 threshold sweep→T0+3a-5; #5/#7/#8 gate→3a-1; #6/#9-sig→T1 full-sim-dict+adapter; #9 field cluster→3a-2; #10 init scalar|array→3a-2; #11 seed→3a-2; #12 leading_subspace→3a-3; #13 pair_loading helper→3a-3; #14 left projection normalized+rename→3a-3; #15 directional-vs-operator→3a-4; #16 conj abscissa→3a-4; #17 α≥0 guard→3a-4; #18/#19/#20 verdict tests→3a-5; #24 CLI smoke→3a-6; #25 QC→Hard QC; commit explicit `git add`→all tasks. **Covered.**
2. **Placeholder scan**: config values concrete; every code step has real test+impl. Assembly steps (3a-5 Step 5) name exact functions from earlier tasks. **OK.**
3. **Type consistency**: cfg is dict throughout (#5 resolved — `default_transition_config` returns dict, `load_crit_config` returns dict); `build_handoff_from_sim(sim_dict,...)` matches verified signature (#6/#9); `rate_mismatch` returns (abs,rel) consumed by `qualify_point` fields; `solve_operating_point(init=)`/`solve_branches` names match 3a-5. **OK.**
