# FCXR-LC4d offset-latency alignment — IMPLEMENTATION PLAN

Design of record:
`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4d-offset-latency-alignment-design.md`

## Task 1 — L0 analytic lock

- Read and hash the executed LC4c candidate, entry verdict, nominal verdict and trace.
- Verify 11 s onset, 66 s offset, no intervention, numerical safety and 10 ms trace spacing.
- Read `a_mean` at exactly 15 s and derive the sole candidate `g_m_max=I_target/a_15s`.
- Recheck exact interictal dead-zone zero and first-4-s zero current.
- Persist `candidate_lock.json` before runner execution; tests reject off-by-one time samples, source drift and hidden candidate lists.

## Task 2 — L1 adjudicator and runner

- Add a pure 18 s entry/offset adjudicator covering onset, pre-events, zero prefix, 1–5 s duration, autonomous offset, 2 s relapse guard, post-rate suppression, numerical and refractory gates.
- Reuse the existing LC4 simulation adapter; only result root, candidate provider and screen duration change.
- Persist rate/AF/current traces, D/H/X/y regional snapshots, event ledger, exact clause verdict, resource record and sentinels.
- Unit tests must include good path, early/no entry, sub-1-s pseudo-offset, >5-s/record-end carrier, relapse, non-suppressed post rate, current leakage and bad event schema.

## Task 3 — detached conditional chain

- Launch L1 through `setsid nohup`, one worker, 4 h guard.
- Only L1 pass launches fresh 70 s nominal under a new detached session and 7 h guard.
- Only nominal eligibility launches exact-D confirmation under a new detached session and 3 h guard.
- Source digest is checked before every submission; +256 MiB swap blocks later stages and +512 MiB terminates only the newest task-owned run.

## Task 4 — closeout

- Plot only completed stages; do not draw placeholders for gated stages.
- Write Chinese `figures/README.md`, STATUS, manifest and archive.
- Run LC3/LC4/slow-variable regressions, blessed/mechanism hash checks, JSON/hash audit, `git diff --check`, residual-process and swap audit.
- Commit the scientific gate.  Do not launch seed3/unseen noise in this sprint.
