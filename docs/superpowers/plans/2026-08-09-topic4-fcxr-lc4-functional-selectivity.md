# FCXR-LC4 functional selectivity — IMPLEMENTATION PLAN

Design of record: `docs/superpowers/specs/2026-08-09-topic4-fcxr-lc4-functional-selectivity-design.md`

## Task 1 — pure contracts and tests

- Add `src/topic4_fcxr_lc4_gate.py` for force-matched candidate construction, event summaries,
  baseline adjudication and onset-surface adjudication.
- Add synthetic tests locking the n=4 ictal-current anchor, candidate preference, every baseline
  clause and the F1 positive-control logic.

## Task 2 — F0 paired baseline runner

- Add `scripts/run_topic4_fcxr_lc4_gate.py --stage baseline --confirm-run`.
- Run control, n6 and n8 on frozen `D_healthy`, same seed, 12 s, one worker.
- Persist per-arm JSON/NPZ, `baseline_verdict.json`, resource log and sentinels.
- Stop if no candidate passes.

## Task 3 — F1 onset surface

- Reuse the same runner with `--stage onset`.
- Run selected candidate on `D_healthy/D10/D30/D50`, plus actuator-off D10.
- Persist whole-window lifecycle labels and `onset_surface_verdict.json`.
- Stop if the positive control fails or no candidate row departs through D50.

## Task 4 — F2 lifecycle

- Reuse the same candidate without further tuning.
- Run one 70 s no-kick dynamic-Z/H/X trajectory under `setsid nohup`, single worker.
- Persist event ledger, regional slow-variable trajectories, numerical diagnostics and the actual
  post-offset D field required for the 12 s frozen recovery confirmation.
- Do not call the result complete before the frozen actual-D check and >=8 s returning-event gate.

## Task 5 — closeout

- Produce one diagnostic figure with baseline ratios, frozen-D onset bracket, population activity
  and the `(D, actuator)` slow path.  Add Chinese `figures/README.md` after visual inspection.
- Archive exact results, tests, hashes, resources and allowed/forbidden wording.
- Run focused LC4/LC3/slow-variable regression and `git diff --check`; commit each scientific gate.
