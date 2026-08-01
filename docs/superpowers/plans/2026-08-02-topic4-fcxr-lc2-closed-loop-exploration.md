# FCXR-LC2 closed-loop mechanism exploration — IMPLEMENTATION PLAN

Date: 2026-08-02  
Design: `docs/superpowers/specs/2026-08-02-topic4-fcxr-lc2-closed-loop-exploration-design.md`

## 0. Execution graph

```text
E0 lock + R1 acceptance check
 -> E1 offline R1 segmentation/support/Pareto
 -> E2 frozen-X field TDD + parity
 -> E3 complete 40k high-init screen
 -> E4 finalist A/B/C/D forks + neighbour/noise replication
 -> if geometry exists: E5 dynamic Z/H/X pilot
 -> E6 figures + archive + commit
```

Only engineering corruption, OOM safety or the 10 h budget stops the registered E3 grid early. Scientific
negative points are recorded and the grid continues.

## 1. E0 — lock and preflight

1. Verify HEAD contains R1 acceptance commit `d7b40906`.
2. Hash four R1 trace NPZs and canonical HEO1/HEO2 LFP artifacts.
3. Record candidate rules, grids, noise IDs, D values, LC1 X loads and engine hashes in
   `execution_lock.json` before reading E1 outputs.
4. Check sibling 40k tasks, MemAvailable and swap; do not launch simulation in E0/E1.

## 2. E1 — offline characterization

Implement pure functions and synthetic tests for:

- trimmed bout/gap segmentation;
- 20/50 ms rolling rate/current-energy diagnostics;
- h-independent recruited-drive support;
- 50 ms latch score and empirical target threshold;
- active/full/support duty and bridge ratio;
- deterministic Pareto/non-dominated candidate selection.

Run the locked 25 tau × 4 false-latch targets on existing traces. Persist every row, not only selected
candidates. Plot all-cell vs support and active-bout vs full-window panels. Gate is completion and data
integrity, not separability.

## 3. E2 — frozen-X vertical slice

Add `x_relay_frozen_E` to `MZSlowVarsConfig`:

- legal only with `use_x=True` and shape `(NE,)`, finite, in `[0,1]`;
- initializes `x_relay` and `ee_relay_send`;
- sensor y may still be recorded, but step must not evolve the frozen X field;
- E→E scatter uses the frozen per-source values; E→I remains unchanged;
- default absent path remains byte-identical;
- snapshot/restart preserves the field.

Tests: invalid field, frozen invariance, source-level E→E-only force, off parity, deterministic 500 ms smoke.
Re-run LC2/MZ/RC1/LC1 regression and blessed hashes.

## 4. E3 — 40k screen

Create one manifest row per candidate/k/rho combination before launch. Each run:

- connection seed 1; development noise stream 401;
- RC1 recurrent-only conductance, `g_sat=21.6`, `coop_A=0`, `M=0`, `X=off`;
- frozen D=0.15 on the locked `p_i` field;
- `h_init_E=2*theta`; T=1000 ms; no kick; early runaway guard allowed only as a numerical/performance
  guard and must be reported;
- save scalar traces and compact per-cell tail summaries, not N×T matrices.

First run measures RSS and wall time.  It measured 6.793 GiB while MemAvailable remained 182.5 GiB on
an 80-CPU host.  The spec resource formula therefore unlocks four E3 workers while retaining at least
96 GiB; E4 stays at two until separately timed.  The launcher writes per-cell outputs and an aggregate
sentinel, and a resumed launcher skips completed cells.  Submission is breadth-first by rho, k, then H
candidate so a 10 h resource terminal preserves cross-candidate coverage; canonical manifest rows and
indices are unchanged.

Classify every row as `decay_low`, `screen_survivor`, `saturated_tonic`, `numerical_failure`, or
`unresolved_1s`. Do not call a survivor a basin.

## 5. E4 — frozen forks

The original ceiling is six ordered survivors.  Measured E3 wall time fixes this execution to the best
survivor plus one adjacent point (two finalists total) so that direct basin/X forks remain inside the
10 h sprint instead of spending the entire budget on the screen.  Run A-low/A-high/B/C/D1/D2 for those
two points. A/B/C duration is
`min(5000,max(2000,5*tau_H)) ms`; D1/D2 duration is
`min(8000,max(3000,4*tau_H)) ms`, so the termination probe can include its locked post-offset window.
An insufficient post-offset observation is unresolved, not negative. Use matched RNG reset. For candidates that satisfy A/B/C/D, run:

Before launch, lock the finalist ordering proxy to `tail_gH_mean/max(tail_gA_mean,1e-12)`, insert one
available one-step rho/k neighbour immediately after the top point, and lock the fork labels to the
spec's final-window `low_like`/`high_like` definitions.  These rules must be unit-tested before the
manifest is created.

- one adjacent rho step (prefer lower; otherwise higher);
- noise stream 402, frozen before launch and not used to tune parameters.

Store the complete fork matrix and classify the failure location: healthy false basin, no high basin,
ceiling branch, X load insufficient, isolated point, or replicated window.

## 6. E5 — dynamic pilot

Unlocked only by `H_X_FROZEN_GEOMETRY_CANDIDATE`. Use the existing LC1 Z/X equations without changing
their forms. This execution uses only the exact q75 Z calibration plus the accepted q50 termination-
authority X kinetics listed in the spec. The previously contemplated intermediate hazard is deferred:
its mapping was not numerically locked before the sprint, so it cannot be invented after seeing q75.

Run nominal X-on and matched X-off for 20–30 s, one worker, detached. If onset occurs, record the first
onset snapshot for matched causal comparison. A lifecycle candidate must be kick-free and satisfy the
observable sequence in the design; otherwise issue the specific entry/offset/recovery-negative label.

## 7. E6 — delivery

Required figures, when input exists:

1. `r1_sensor_characterization.png`;
2. `r1_sensor_pareto.png`;
3. `h_loop_screen.png`;
4. `frozen_fork_map.png`;
5. `dynamic_pilot.png` only when E5 ran;
6. `failure_taxonomy.png`.

Write/update `figures/README.md` after rendering. Archive to
`docs/archive/topic4/sef_hfo/fcxr_lc2_closed_loop_exploration_2026-08-02.md` and commit logical batches.
Final report must state stage reached, complete candidate counts, frozen geometry/X/dynamic results,
tests/hashes, RSS/swap/workers, nohup/sentinels, artifacts, commits, allowed claims and the single next
mechanistic recommendation.
