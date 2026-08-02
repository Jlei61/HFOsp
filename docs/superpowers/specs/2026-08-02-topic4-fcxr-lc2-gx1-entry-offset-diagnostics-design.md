# FCXR-LC2-GX1 — susceptibility-selective entry and maximal-X offset diagnostics

Status: **DESIGN LOCK**  
Date: 2026-08-02  
Upstream closeout: `docs/archive/topic4/sef_hfo/fcxr_lc2_closed_loop_exploration_2026-08-02.md`

## 1. Scientific question

LC2-Core established a finite, non-clipped H-supported high state. It did not establish control of that
state:

```text
bounded high-state generation                 = positive
Z susceptibility-selective onset control      = negative
X amplitude control at tested LC1 loads       = positive
X offset state-transition authority           = negative
```

GX1 asks two narrower questions before changing the equation:

1. Does the **existing** H equation contain a weak-gain/high-threshold region in which healthy and
   susceptible low starts remain interictal but a susceptible high start remains finite-high?
2. Can theoretical maximal presynaptic relay shutdown remove an already established H high state, or
   does the H actuator bypass the X execution path strongly enough to preserve it?

This is a frozen-geometry diagnosis. It does not test a dynamic lifecycle.

## 2. Locked scope

Allowed:

- current RC1 recurrent-only conductance and smooth saturation;
- current post-X local H state and additive H actuator, unchanged;
- frozen local depletion field at `D={0,0.15}`;
- frozen source-level relay availability;
- connection seed 1, development noise 401;
- confirmation noise 402 only after a development candidate exists;
- empirical Stage-D workpoint classifier.

Locked out:

- dynamic Z, dynamic X and lifecycle acceptance;
- `M/K/A/ELR`, new E→E edges, kick, reset or parameter step;
- explicit `S_D(D)`, changed H equation or shared-path H implementation during GX1;
- hidden seeds and E1146 morphology claims.

Six blessed engine files must remain byte-identical. GX1 should require no change to
`src/snn_engine/mz_slow_vars.py`.

## 3. Diagnostic S1 — natural selectivity strip

### 3.1 Candidate families and grid

Use two already locked R1 families:

| family | role | tau_H (ms) | theta_H,base | false latch |
|---|---|---:|---:|---:|
| H1 | fastest zero-false-latch member | 522.0314431365 | 1.2594716549 | 0 |
| H6 | fastest remaining Pareto member | 632.4555320 | 1.1122742295 | 2/9 |

Do not select between them from GX1 results. For each family use:

```text
theta_scale  = {1.00, 1.25}
rho_H/g_sat  = {0.025, 0.050, 0.075}
k_H/theta_H  = 0.05
g_sat        = 21.6
```

This gives 12 parameter points, all below the previous minimum `rho/g_sat=0.10`; half also move theta
upward. The strip is fixed before any new simulation.

### 3.2 Three matched arms per point

| arm | D | H initial state | required outcome |
|---|---:|---:|---|
| healthy_low | 0 | 0 | `INTERICTAL_WORKPOINT` |
| susceptible_low | 0.15 | 0 | `INTERICTAL_WORKPOINT` |
| susceptible_high | 0.15 | `2*theta` | `FINITE_HIGH_FIXED` or `FINITE_HIGH_ORBIT` |

All arms use the same fresh RNG reset and run for 4000 ms. The final 2000 ms is classified with the
accepted seed1 `roll_hi=9.7382291667 Hz` and 300 ms rolling workpoint definition. A high outcome must also
be finite, have zero hard clip, no early numerical guard, refractory-ceiling fraction <5%, and H slope no
more negative than 5% of its tail level per second.

### 3.3 Strip verdicts

```text
NATURAL_SELECTIVITY_WINDOW_CANDIDATE
ISOLATED_SELECTIVITY_POINT
NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP
SELECTIVITY_STRIP_NUMERICAL_FAILURE
```

A candidate point passes all three arms. A **window** requires an adjacent passing point along rho or
theta within the same family. A single point is isolated and cannot block an explicit-D hypothesis, but
also cannot authorize it as necessary.

If a window exists, the next mechanism must retain the current H equation; adding explicit `S_D(D)` is
not authorized. Confirm only the window anchor and one adjacent point on noise 402.

If no point passes and all 36 trajectories are valid, explicit D-dependent H coupling becomes justified
as a new testable hypothesis, not as a fitted repair.

## 4. Diagnostic X1 — theoretical maximal X authority

### 4.1 Anchor

If S1 finds a natural window, use its deterministic anchor: lower rho, then higher theta, then H1 before
H6. Otherwise use the already canonical `H6_k05_r10` susceptible-high configuration. Establish one
matched susceptible high state at `D=0.15`, `H(0)=2theta`.

### 4.2 Frozen relay arms

From the same high-state definition, run:

```text
x_relay availability = {1.0, 0.5, 0.1, 0.0}
```

`x=0` is a causal structural probe, not a physiological parameter. Each arm uses the exact presynaptic
E→E relay field already TDD'd in LC2. Run for `max(5000 ms, 8*tau_H)`. The required low-exposure window is
the final `max(2000 ms, 3*tau_H)`.

The current architecture is recorded explicitly:

```text
gA_postX -> H source
u_pre_sat = gA_postX + rho*S_H(H)
```

Thus X suppresses the fast input and future H source, but does not instantaneously multiply the existing
H actuator. X1 determines whether that indirect path can nevertheless remove the state within eight H
time constants.

### 4.3 X verdicts

```text
X_PATH_REACHABLE_RANGE_INSUFFICIENT
H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN
X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH
X_AUTHORITY_UNRESOLVED
```

- If `x=0` returns to `INTERICTAL_WORKPOINT` for the required window but the archived physiological
  availability levels `0.872/0.786` do not, the path is reachable and the observed dynamic range is
  insufficient.
- If `x=0` remains finite-high after eight tau, the current H actuator bypasses the relay path at maximal
  shutdown.
- If a nonzero experimental arm returns low, report the smallest tested availability that does so; do not
  call it physiological.
- Numerical failure or insufficient post-offset exposure is unresolved.

## 5. Conditional next-structure logic

GX1 itself does not implement a new structure. It only decides which hypothesis is legal next:

| S1 | X1 | Authorized next hypothesis |
|---|---|---|
| natural window | path reachable | keep H equation; calibrate X dynamic range only |
| natural window | maximal bypass | shared-path X/H coupling only |
| no window | path reachable | smooth local D-dependent H gain only; X range separately |
| no window | maximal bypass | causal 2x2: D gate × shared-path X/H coupling |
| unresolved | any | repair measurement; no structural claim |

Candidate entry mechanism, only if authorized:

```text
S_D(D_i) = logistic((D_i-D_H)/k_D)
gH_i     = rho_H*S_D(D_i)*S_H(h_i)
```

This is a new falsifiable mechanism assumption. It cannot use a core mask, seizure label or time switch.

Candidate shared-path mechanism must ensure X controls the entire H-supported recurrent path. A later
implementation may use a multiplicative post-relay recurrent gain,

```text
u_pre_sat = gA_postX * (1 + rho_tilde*S_D(D_i)*S_H(h_i)),
```

or a source-resolved slow E→E scatter. A target-level scalar that merely imitates X is not acceptable.

## 6. Future 2x2 accounting correction

The current architecture is already an accepted archived negative and need not be rerun. Three new
architectures would each require **five**, not four, scientific arms:

1. healthy-low;
2. healthy-high;
3. susceptible-low;
4. susceptible-high;
5. susceptible-high + physiological X.

At two noise streams, the correct total is `3 architectures x 5 arms x 2 = 30` short branches. Shared
prefix snapshots may reduce compute but do not reduce the scientific arm count. This structural 2x2 is
not authorized by the GX1 execution plan; it requires a post-GX1 lock.

## 7. Resources and detachment

- First 4 s run remeasures RSS/wall time. Existing evidence is 6.8–7.2 GiB per 40k fork.
- GX1 uses at most two 40k workers. Start worker 2 only when
  `MemAvailable >= 96 GiB + 2*1.35*RSS_single` and swap is stable.
- OMP/OpenBLAS/MKL/NUMEXPR = 1.
- Swap +256 MiB stops new submission; +512 MiB and rising terminates only the newest GX1 worker.
- Every long stage uses `setsid nohup`, exact launcher PID, stage flock, RUNNING/DONE/FAILED sentinel,
  resource log and wall guard. Never use `pgrep -f`; never touch sibling processes or worktrees.
- S1 completes all 36 registered development trajectories unless engineering corruption, OOM safety or
  an 8 h stage wall guard fires. A scientific negative does not stop the strip.
- X1 runs only after S1 aggregate exists. No dynamic lifecycle is launched.

## 8. Required outputs

```text
results/topic4_sef_hfo/fcxr_lc2_core/gx1_entry_offset_diagnostics/
  execution_lock.json
  selectivity_strip_manifest.json
  selectivity_strip.json
  x_authority_manifest.json
  x_authority_map.json
  candidate_verdict.json
  STATUS.md
  resource_log.jsonl
  figures/selectivity_strip.png
  figures/x_authority.png
  figures/failure_logic.png
  figures/README.md
```

Final archive must preserve every registered negative point and distinguish component positives from
state-transition authority.
