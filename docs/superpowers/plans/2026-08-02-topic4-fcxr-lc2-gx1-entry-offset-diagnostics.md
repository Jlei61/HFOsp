# FCXR-LC2-GX1 entry/offset diagnostics — IMPLEMENTATION PLAN

Date: 2026-08-02  
Design: `docs/superpowers/specs/2026-08-02-topic4-fcxr-lc2-gx1-entry-offset-diagnostics-design.md`

## 0. Execution graph

```text
G0 closeout wording + execution lock
 -> G1 pure contracts/TDD + 36-row S1 manifest
 -> G2 40k selectivity strip (12 points x 3 arms)
 -> G3 strip aggregation / optional two-row noise402 confirmation
 -> G4 four-arm maximal-X probe
 -> G5 figures / verdict / archive / conditional next spec
 -> STOP (no dynamic lifecycle)
```

## 1. G0 — closeout correction and lock

1. Regenerate LC2 `candidate_verdict.json`, `STATUS.md` and failure taxonomy with the canonical component
   wording:

   ```text
   bounded high-state generation positive
   susceptibility-selective onset negative
   X amplitude control positive
   X offset state-transition authority negative
   ```

2. Hash the locked R1 candidate table, P-field, baseline workpoint contract, LC2 frozen map and six
   blessed engine files.
3. Write the complete S1/X1 grids, seed rules, durations, thresholds and resource policy to
   `execution_lock.json` before new simulation.
4. Confirm no sibling process is touched and measure current MemAvailable/swap.

## 2. G1 — runner and pure contracts

Add one non-engine runner, `scripts/run_topic4_fcxr_lc2_gx1.py`, reusing the validated LC2 frozen fork
vertical slice. It must expose:

```text
lock
strip-manifest
strip-one --index N --confirm-run
strip-all --workers {1,2} --confirm-run
strip-aggregate
x-manifest
x-one --index N --confirm-run
x-all --workers {1,2} --confirm-run
finalize
```

Pure tests must pin:

- exact 12-point / 36-arm strip manifest;
- H1/H6 constants and below-old-minimum rho values;
- empirical workpoint labels rather than raw 20 Hz labels;
- adjacency rule for a natural window;
- deterministic anchor tie-break;
- four X availabilities and `max(5000,8*tau)` duration;
- X verdict logic, including `x=0` structural semantics;
- no dynamic command and explicit `M/K/A/ELR=False` fields;
- engine hashes and off-by-default parity inherited from LC2.

## 3. G2 — selectivity strip execution

### 3.1 Manifest

Write all 36 rows before launch. Submission order is breadth-first by rho, theta scale, family, arm so a
resource terminal does not spend the stage on one family. Canonical indices never change.

Each worker:

- builds the connection-seed-1 RC1 substrate once;
- resets noise RNG to 401 for every matched arm;
- uses `D={0,0.15}`, `X availability=1`, `M=0`, no kick;
- runs 4000 ms at `dt=0.05 ms`;
- records compact rate/H/gA/gH/X traces, numerical safety and workpoint metrics;
- writes one deterministic cell JSON and DONE sentinel.

### 3.2 Launch

Run the first row alone to remeasure RSS. If the two-worker gate passes, detach the remaining stage with
`setsid nohup` and two workers. Add an exact-PID watchdog with 8 h wall limit, 96 GiB reserve and swap
guards. Resume skips only cells whose contract version and source hash match.

### 3.3 Aggregate

Aggregate all 36 cells and label each of 12 points. Produce adjacency components in the `(rho,theta)`
grid separately for H1 and H6. Do not pool families to manufacture adjacency.

If a window exists, lock its anchor and one neighbour; run only those two parameter points at noise402
with the same three arms. A failed confirmation changes the label to `DEVELOPMENT_ONLY_NOT_REPLICATED`.
If no development point passes, noise402 remains unopened.

## 4. G4 — maximal-X authority

### 4.1 Anchor and manifest

After the strip aggregate, create—not before—one four-row X manifest using the deterministic anchor rule.
If S1 has no window, use archived `H6_k05_r10`. The manifest records why that anchor was selected.

Each row has `D=0.15`, `H(0)=2theta`, no kick, noise401 and availability `{1,0.5,0.1,0}`. Use identical
initial conditions and RNG reset. Run `max(5000,8*tau)` and require the final `max(2000,3*tau)` low
window.

The first implementation may start all four arms from the same analytically specified high H field, as
in E4. A snapshot prefix is optional; if used, every branch must consume the same saved state and RNG
state, and the no-intervention prefix must reproduce byte-identically.

### 4.2 Verdict

Use the spec labels without rewording after results. Report time-to-first sustained low window, final H,
rolling occupancy and rate reduction at every x. `x=0` returning low proves theoretical path
reachability, not physiological validity.

## 5. G5 — delivery

Generate:

1. `selectivity_strip.png`: categorical three-arm outcome plus healthy/susceptible rate contrast;
2. `x_authority.png`: matched 300 ms rolling-rate and H traces for all four x levels;
3. `failure_logic.png`: S1 × X1 decision table with the observed cell highlighted.

Write Chinese `figures/README.md` only after visual inspection. Finalize `candidate_verdict.json`,
`STATUS.md`, and archive to:

```text
docs/archive/topic4/sef_hfo/fcxr_lc2_gx1_entry_offset_diagnostics_2026-08-02.md
```

The archive must state:

- exact stage reached and all row counts;
- S1 natural-window verdict and any noise402 result;
- X theoretical-authority verdict;
- what structure is and is not justified next;
- numerical safety, tests, blessed hashes, RSS/swap/workers and nohup/sentinels;
- no dynamic lifecycle and no morphology claim.

## 6. Stop rules

- Scientific negatives never stop the registered S1 grid.
- Numerical corruption invalidates only the affected matched point; repeated corruption stops G2.
- OOM guard stops submission without touching sibling jobs.
- Missing or drifted upstream artifact blocks launch loudly.
- G4 cannot start before a complete S1 aggregate.
- No result in GX1 authorizes dynamic Z/H/X. The next structural experiment needs a separate lock even if
  both diagnoses are positive for a repair.

## 7. Expected budget

Existing 3.16 s forks cost about 15 min each. Thirty-six 4 s branches at two workers are expected to take
about 5–6 h; the four X branches add roughly 45–75 min. Engineering, aggregation and figures fit within
one 8 h detached execution when the machine remains above the resource reserve.
