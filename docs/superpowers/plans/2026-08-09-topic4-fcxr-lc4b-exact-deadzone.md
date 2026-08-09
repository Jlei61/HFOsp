# FCXR-LC4b exact-dead-zone cooperative terminator — IMPLEMENTATION PLAN

Design of record:
`docs/superpowers/specs/2026-08-09-topic4-fcxr-lc4b-exact-deadzone-design.md`

## Task 1 — D0 locked analytic candidate

- Add a pure dead-zone activation/candidate builder reading the frozen `per_cell.npz` arrays.
- Persist the artifact hash, three source extrema, `m0`, `K_excess`, n, kinetics, ictal activation
  and force-matched `g_max` in `candidate_lock.json` before changing the engine.
- Add tests for clean-gap failure, exact interictal zero, force matching and no outcome-dependent
  parameter choice.

## Task 2 — off-by-default mechanism implementation

- Add optional `m_hill_deadzone` to non-blessed `src/snn_engine/mz_slow_vars.py`.
- Compute `u=max(m-m0,0)` before the existing Hill curve; unset keeps the literal old expression.
- Add unit tests for below/at/above threshold, `deadzone=0` equivalence, invalid configuration,
  deterministic snapshots and absent-field backward compatibility.
- Re-run LC3/LC4/slow-variable regression and lock the mechanism-module hash; six blessed files
  must remain byte-identical.

## Task 3 — D1 paired baseline

- Reuse the committed LC4 actuator-off control only after hash and inert-path preflight.
- Run the single dead-zone candidate for 12 s under `setsid nohup`, one worker.
- Score the unchanged LC4 functional clauses plus exact-zero actuator and byte-identical
  population-rate/active-fraction traces.
- Persist run JSON/NPZ, `baseline_verdict.json`, resources and sentinels.  Stop on any failure.

## Task 4 — D2 frozen-D onset

- Reuse the committed actuator-off D10 positive control after provenance validation.
- Reuse D1 as Dhealthy; run candidate D10, then conditionally D30, then conditionally D50.
- Persist whole-record lifecycle labels and `onset_surface_verdict.json`.
- Stop immediately at first departure (pass) or after D50 with no departure (no-go).

## Task 5 — D3 lifecycle and exact-D confirmation

- Reuse the unchanged LC4 F2 adjudicator with the locked dead-zone candidate and a new result root.
- Run one 70 s no-kick dynamic trajectory, single worker.
- Only a nominally eligible result launches the 12 s exact-state actual-D-frozen continuation.
- Persist the event ledger, regional D/H/X/a paths, current-based activity readout, numerical
  diagnostics and exact final state.

## Task 6 — closeout

- Plot only stages that actually ran.  The final diagnostic must distinguish baseline identity,
  frozen-D entry, offset and statistical return; never draw placeholder curves for gated stages.
- Add Chinese `figures/README.md`, STATUS, manifest and archive with allowed/forbidden language.
- Run focused regression, blessed/mechanism hash checks, `git diff --check`, process/swap audit and
  commit each completed scientific gate.
