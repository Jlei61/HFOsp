# M3A-A2 producer handoff — key paths + state (for the M3 series)

> Status: HANDOFF / REFERENCE, 2026-06-28.
> Scope: the M3A-A2 **producer side** of the M3A ↔ M3B-R2 interface — everything that emits the
> canonical handoff artifacts a calibrated phase trajectory needs.
> Canonical contract: `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md`.
> M3B-R2 consumer plan: `docs/superpowers/plans/2026-06-27-sef-hfo-m3b-spectral-phase-map-plan.md`.
> **The producer code lives on branch `topic4-m3a-a2`** (commit range `2f31090..18fe0de`), NOT on this
> branch. This doc is the cross-branch map.

## 0. Plain language (what this preparation is)

We built the *shipping end* of the model→map handoff. The model line (M3A) runs the spiking sheet and
produces slow-variable traces (inhibition-fuel `q` depleting, sAHP `g_K`, threshold `phi`, …). The map
line (M3B-R2) wants to draw those as a trajectory through a spectral phase map. This work makes the
model run **automatically emit a full set of contract-valid handoff files**: it translates the slow
variables into normalized phase coordinates, labels every spontaneous event with a regime class, and
self-audits. Every piece is locked with synthetic-data tests and verified on a real engine run. The
engineering is complete and honestly labelled; the only thing left is the **science** (does the slow
state actually move the event phenotype — `gate_A`), which is an experiment, not a wiring step.

## 1. The handoff gate in one picture

```
RegionalResource q(t) traces  +  detected events (t_on/t_off, source-space metrics)
   └─ sample_event_landmarks            pre/onset/peak/end/post_* snapshots (peak = real activity peak)
        └─ evaluate_phase_coord (SHARED) phase_x_core / phase_y_global / phase_recovery  ∈ [0,1]
        └─ assemble_event_metrics        per-event classify_event dict (SOURCE space) + absolute tail
             └─ build_phase_trajectory_rows / build_event_phase_samples
                  └─ build_self_audit → audit_m3a_interface   (4-condition fail-closed overlay gate)
```

Overlay is drawn only when `overlay_verdict == phase_map_trajectory`, i.e. all four conditions hold:
`cond1` mapping sign-calibrated · `cond2` axes/ranges provenance match · `cond3` rows valid + ≤5%
out-of-range · `cond4` `gate_A == PASS` AND `rate_matched_control == passed`. **cond4 is a science
outcome of the real sweep, not a wiring step** — so the wired pipeline is correctly `refused` until an
experiment shows phenotype movement.

## 2. Key paths

### Shared contract (canonical — imported by BOTH M3A producer and M3B consumer)
- `src/sef_hfo_m3_interface.py` — schema validators, the deterministic `evaluate_phase_coord` transform,
  `mapping_sign_tests_passed`, the overlay gate `audit_m3a_interface` / `compute_overlay_verdict` /
  `build_slow_trajectory_overlay`, the min-columns join.
- `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md` — D1–D7 locked decisions, §7
  test partition, §8 blocker→fix map.
- `tests/test_sef_hfo_m3_interface.py` — 46 contract-layer tests (GREEN without the SNN).

### M3A-A2 producer modules (branch `topic4-m3a-a2`)
- `src/sef_hfo_m3a_export.py` — the M3A exporter:
  `default_precalib_mapping_and_ranges`, `build_phase_trajectory_rows`, `assemble_event_metrics`,
  `build_event_phase_samples`, `build_self_audit`, `build_handoff_from_sim`, `write_handoff_artifacts`.
- `src/sef_hfo_m3a_calibration.py` — sign calibration:
  `evaluate_engine_sign_test`, `apply_calibration`, `measure_q_firing_responses`,
  `calibrate_axisbreak_mapping`.
- `src/sef_hfo_a2.py` — added `sample_event_landmarks`, `tail_to_baseline_absolute`, `event_peak_ms`.

### Runners / CLIs (branch `topic4-m3a-a2`)
- `scripts/calibrate_a2_mapping.py` — one-time sign calibration → writes a calibration dir.
- `scripts/run_a2_axisbreak_sweep.py` — the dynamic sweep; emits a handoff per grid point; `--calibration-dir`.
- `scripts/plot_a2p_synchronous_burst_figure.py` — `simulate_a2(frozen_q=…)` + full `trace_core/global/gk`
  + `core_mask` returns + `read_events`.

### Reused canonical producers (NOT re-invented — §6 discipline)
- `src/sef_hfo_mu_basin.py::classify_event`, `::event_props` — the R0–R4a/R4b regime label.
- `run_m3_static_mu_spontaneous.py::_event_spatial` — source-space r95 / far / sustained-front-score.
- `src/topic4_propagation_operator.py::spatial_bins` — source-space binning from `posE`.

### Producer tests (branch `topic4-m3a-a2`)
- `tests/test_m3a_export.py` (~26) · `tests/test_m3a_calibration.py` (6) ·
  `tests/test_m3a_tail_and_peak.py` (5) · `tests/test_m3a_dynamic_recorder.py` (6).

## 3. Emitted artifacts

### Per sweep grid point → `<out>/handoff_<tag>/`
- `slow_to_rate_mapping.json` · `phase_coord_ranges.json` · `phase_trajectory.csv` ·
  `event_phase_samples.csv` · `dynamic_slowvars_summary.json` · `m3a_interface_audit.json` · `STATUS.md`.
- The summary carries a non-contract `provenance` block: `handoff_kind`
  (`pre_calibration_scaffold` | `calibrated_handoff`), `mapping_calibration`, `calibration_caveat`,
  `peak_landmark_source`, `tail_to_baseline_definition`, `two_core_reduction`, `deferred_artifacts`,
  `undefined_science_decisions`, `expected_overlay_verdict`.

### Per calibration run → `<cal_dir>/`
- `calibrated_mapping.json` · `phase_coord_ranges.json` · `calibration_report.json`
  (`calibration_kind: "sign_only"`, engine sha, q sweep, per-axis sign tests).

## 4. Locked decisions (contract §1 D5–D7, user 2026-06-27)
- **D5 `two_core_reduction = "mean_q"`** — equal cores collapse to `phase_x_core` by averaging
  `q_core_L`/`q_core_R` (symmetric; `source_core`/`min_q` deferred and raise).
- **D6 absolute `tail_to_baseline`** — `mean(event tail) / mean(fixed 5–50 ms quiet baseline)`,
  returned if ≤ 1.5. Denominator is the fixed baseline, never the event's own peak.
  Helper `src/sef_hfo_a2.tail_to_baseline_absolute`. This is the `return_to_baseline` COLUMN;
  `R_class` uses `event_props.returned` (sustained-ness) — two distinct questions, allowed to differ.
- **D7 peak landmark** — real activity-fraction peak inside the event window
  (`src/sef_hfo_a2.event_peak_ms`); window midpoint only as a recorded fallback.

## 5. How to run

```bash
# 1. calibrate the mapping ONCE for a substrate (sign calibration: direction only)
python scripts/calibrate_a2_mapping.py [sim args] --out CAL_DIR

# 2. dynamic sweep; each grid point writes a calibrated handoff referencing CAL_DIR
python scripts/run_a2_axisbreak_sweep.py [sim args] --tag <tag> --out OUT --calibration-dir CAL_DIR
#    (omit --calibration-dir -> fail-closed pre-calibration scaffold, cond1=false)
```

Verified end-to-end (small substrate L=8/T=200): calibration → both axes `slope_sign=-1, passed`;
sweep `--calibration-dir` → handoff `handoff_kind=calibrated_handoff`, `undefined_science_decisions=[]`,
audit `cond1=True`, `cond4=False`, `overlay_verdict=refused` (refused only because gate_A is pending).

## 6. State + what is NOT done
- **Done (engineering):** all 6 contract artifacts emitted per grid point; R_class wired; the mapping is
  sign-calibratable against the engine (`cond1` real); STATUS/provenance label everything honestly.
- **NOT done (science):** `cond4` — a real two-tank / axis-break sweep showing the slow state moves the
  event phenotype beyond rate-only heating (`gate_A == PASS` + a rate-matched control). May or may not
  pass; it is exploration.
- **NOT done (optional):** **P1-2 response-curve calibration.** The present calibration is *sign only*
  (direction locked); the transform `a/b/input_min/input_max` are normalized placeholders, so phase-map
  *distances/magnitudes* are not yet quantitatively meaningful. Add a fitted rate-response curve only if
  M3B needs to interpret coordinate distances.

## 7. Commit chain (branch `topic4-m3a-a2`, `2f31090..18fe0de`)
firewall TDD → reconcile to canonical `sef_hfo_m3_interface` (retire the duplicate) → exporter + landmark
recorder → runner wiring (fail-closed) → decisions A/B/C → lock A/B/C in the contract (§9→§1) → real
event_phase_samples builder → R_class runner feed → calibration sign logic → calibration engine-feed →
calibrated mapping into the sweep + sign-only honesty.
