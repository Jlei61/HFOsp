# FCXR-LC4c entry-offset alignment — IMPLEMENTATION PLAN

Design of record:
`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4c-entry-offset-alignment-design.md`

## Task 1 — C0 candidate lock

- Read the frozen LC3 `theta=1.1` entry row and LC4b nominal/trace artifacts.
- Verify hashes, no-kick provenance, 11 s onset, 29 pre-onset events, numerical safety and
  `a_mean_max=0.2963244915008545`.
- Derive `g_m_max=I_target/a_mean_max` and persist the single candidate before runner changes.
- Add pure tests for artifact mismatch, dose identity, exact interictal zero and no hidden sweep.

## Task 2 — candidate plumbing

- Extend the non-blessed LC4 lifecycle adapter to accept an optional candidate
  `theta_h_lc2`; absence must preserve the LC4b path.
- Add a new result-root wrapper and a pure C1 adjudicator.
- Tests must cover `[8,15] s` entry, early/no-entry stops, first-4-s zero current, numerical and
  refractory gates, plus nominal/frozen-D delegation to the unchanged LC4 adjudicator.
- Re-run LC3/LC4/slow-variable regression and verify six blessed hashes.

## Task 3 — C1 15 s entry

- Lock source hashes and resource baseline.
- Launch one `setsid nohup` 40k worker with PID/flock/sentinels and a 3 h wall guard.
- Persist event ledger, rate/AF/current traces, regional D/H/X/y snapshots and C1 verdict.
- Stop without C2 on any failed clause.

## Task 4 — C2 nominal and conditional exact-D

- On C1 pass, launch a fresh 70 s no-kick trajectory under `setsid nohup`, one worker, 7 h guard.
- Run the unchanged LC4 nominal lifecycle adjudicator.
- Only nominal eligibility launches the exact-state actual-D-frozen 12 s confirmation.
- Do not launch seed3/unseen noise in this sprint.

## Task 5 — closeout

- Plot only completed stages: C1 entry timing, C2 carrier/offset/return, and the slow path; no
  placeholder for gated confirmation.
- Add Chinese `figures/README.md`, STATUS, manifest and archive.
- Run focused regression, blessed/mechanism hash checks, `git diff --check`, process/swap audit,
  then commit the scientific gate.

## Execution deviation ledger

- **2026-08-10 C1 instrument failure (no scientific verdict):** the first locked 15 s run
  completed its simulation but failed during post-run adjudication because the C1 unit-test
  fixture invented `t_on_ms`, while the canonical LC3/LC4 event producer and all established
  consumers use `t_on` in milliseconds.  C2 was not submitted.  The failed sentinels/logs are
  retained under `lc4c_entry_offset_alignment/superseded/c1_event_schema_failure_2026-08-10/`.
  The repair changes only the adjudicator field name and its fixture, adds a bad-schema
  regression, and permits one exact-protocol C1 rerun; candidate values, seeds, duration and all
  gates remain frozen.
