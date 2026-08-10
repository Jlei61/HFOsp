# FCXR-LC4e spatially shared terminator — IMPLEMENTATION PLAN

Design of record:
`docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4e-spatially-shared-terminator-design.md`

## Task 1 — lock the one-variable architecture change

- Hash the executed LC4d candidate, latency result, trace, runner and mechanism module.
- Derive the LC4e candidate by copying LC4d and adding only `m_hill_spatial_mode=shared`.
- Verify the archived local onset (11 s), first nonzero current (11.83 s), numerical safety and off-axis
  persistence.
- Write `candidate_lock.json` before any 40k run.  No candidate list is allowed.

## Task 2 — TDD and engine adapter

- Add an off-by-default enum `m_hill_spatial_mode in {local,shared}` to the non-blessed slow-variable
  module.
- Factor the cooperative E-current into one tested helper used by the full-conductance path.
- Prove local byte parity, shared uniformity, exact population-mean dose matching, dead-zone zero,
  I-cell exclusion and invalid-mode rejection.
- Thread the candidate field through the existing LC4 lifecycle adapter.  Omitted field must remain
  `local`.

## Task 3 — E1 detached 18 s screen

- Launch the one shared arm with `setsid nohup`, one worker and the 4 h wall guard.
- Persist rate/AF/current/a-mean traces, D/H/X/y regional snapshots, event ledger, resource log and
  sentinels.
- Compare against the archived LC4d local arm read-only.  The adjudicator must separate prefix parity,
  entry, carrier duration, offset, protection, spatial residual and numerical clauses.
- Stop immediately on any non-positive E1 verdict; do not run a gain rescue.

## Task 4 — conditional E2 lifecycle

- Only `SPATIALLY_SHARED_OFFSET_CANDIDATE` launches the unchanged 70 s nominal gate in a new detached
  session.
- Only nominal eligibility launches the unchanged exact-final-D 12 s continuation in another detached
  session.
- Every later submission rechecks source hashes, MemAvailable and swap delta.

## Task 5 — closeout

- Plot only stages that ran.  The E1 figure must show local versus shared rate/current and regional
  carrier traces at matched mean dose.
- Write Chinese `figures/README.md`, STATUS, manifest and an archive with explicit claim boundaries.
- Run targeted tests, LC3/LC4/slow-variable regression, blessed/mechanism hash audit, JSON/hash audit,
  `git diff --check`, residual-process and swap checks.
- Commit the result.  Do not launch seed3/unseen noise or morphology tuning in this sprint.

