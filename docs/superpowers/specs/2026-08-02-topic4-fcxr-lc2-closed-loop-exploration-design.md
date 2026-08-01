# FCXR-LC2 closed-loop mechanism exploration — DESIGN LOCK

Date: 2026-08-02  
Scope: 8–10 h development exploration; connection seed 1; no phenotype claim

## 1. Scientific question

Test one mechanism chain, without adding rescue mechanisms:

```text
post-X local recurrent-drive persistence H
  -> bounded low/high basin geometry on RC1
  -> existing X relay load removes the high basin
  -> dynamic Z/H/X can enter, leave, and return to the IED-producing neighbourhood without a kick.
```

`H` is a local exponential moving average in recurrent-conductance units, not a cumulative termination
load. `X` remains the cumulative/persistence-linked termination coordinate. `M/K/A/ELR`, new E→E edges,
global seizure labels, hard reset and parameter steps are locked out.

This sprint does not require 3–8 Hz, broadband, E1146 morphology, final spatial propagation, hidden-seed
confirmation or a paper-ready lifecycle figure.

## 2. Accepted upstream state

R0 and R1 implementation are accepted at commit `d7b40906`. Canonical R1 labels are:

```text
R1_IMPLEMENTATION_ACCEPTED
R1_SUSTAINED_CONTROL_SEPARABLE
R1_ACTIVE_BOUT_REANALYSIS_REQUIRED
R1_LONG_REST_GAP_NOT_BRIDGED
R1_CLOSED_LOOP_H_GEOMETRY_UNTESTED
R2_CLOSED_LOOP_EXPLORATION_AUTHORIZED
```

The raw `H_SENSOR_NOT_SEPARABLE` value is retained only as a strict full-window / long-gap diagnostic.
It has no authority to stop closed-loop exploration.

## 3. R1 characterization, not classification gate

### 3.1 Locked segmentation

Use the already stored 1 ms `gA_raw` replays. No new 40k simulation is allowed for this stage.

- HEO1 active control: `[1000,3500] ms`.
- HEO2 bouts come from the frozen event detector. Trim 50 ms from both ends; the first bout additionally
  starts no earlier than 1500 ms (adaptation is enabled at 1000 ms).
- HEO2 long gap is the interval between the first returned bout and the second returned bout, trimmed by
  50 ms at each edge.
- The gap is called `rest_like` only when its 50 ms population-rate median is below the RC1 inter-event
  q95 and its 20 ms current-vSEEG RMS median is within 3 dB of HEO2's `[0,150] ms` low reference on at
  least 12/15 contacts. Otherwise it remains `silent_gap_unresolved`.

### 3.2 Recruited-drive support

The support mask must not use `h` or its threshold. On the fixed 4096-cell sample, a cell belongs to the
HEO2 recruited-drive support when its median `gA_raw` over the trimmed active bouts exceeds that same
cell's RC1 baseline temporal q99. Report support fraction and spatial 4×4 block occupancy. If support is
below 2%, support-level results are `UNRESOLVED_SUPPORT`, not silently relaxed.

### 3.3 False latch and duty

Use `tau_H = logspace(20,2000,25) ms`. For each returning baseline IED, define

```text
a(t) = spatial Q99 of h(t)
latch_score(event) = largest threshold sustained by a(t) for 50 ms after event offset and before next onset.
```

For target false-latch fractions `{0, 0.05, 0.10, 0.25}`, choose the smallest empirical threshold whose
observed fraction of `latch_score > theta` is no larger than the target. With only nine baseline events,
report the discrete resolution; 0.05 and 0.10 may collapse to the same threshold.

For each `(tau,theta)` report:

- observed event-level false-latch fraction and reset time;
- HEO1 all-cell and support duty;
- HEO2 active-bout all-cell and support duty;
- HEO2 full-window duty;
- long-gap end/start persistence ratio and above-threshold support;
- all-cell and support summaries separately.

Confidence intervals are within-trajectory block intervals only: resample IED events; for HEO states
resample 50 ms time blocks and 4×4 spatial blocks. Never describe cells or time samples as independent
seeds.

### 3.4 Six deterministic candidate roles

Choose unique candidates from the non-dominated Pareto set, in this order; ties use smaller `tau`, then
smaller `theta`:

1. fastest member with observed false latch 0;
2. largest HEO1 support duty with false latch ≤0.10;
3. largest HEO2-active support duty with false latch ≤0.10;
4. largest `min(HEO1 duty, HEO2-active duty)` with false latch ≤0.10;
5. largest gap persistence with false latch ≤0.25;
6. smallest-tau remaining Pareto member with false latch ≤0.25.

If a role duplicates an earlier candidate, skip it and fill from the Pareto set lexicographically by
false latch, negative minimum active duty, negative gap persistence, tau. Fewer than six valid unique
candidates is allowed and must be reported.

## 4. Direct H-loop screen

Do not turn a surrogate into a scientific verdict. Use the original 40k RC1 network for the screen.

For each R1 candidate, test:

```text
k_H/theta_H = {0.05, 0.10, 0.20}
rho_H/g_sat  = {0.10, 0.20, 0.35, 0.50, 0.70}
```

At frozen susceptible depletion `D=0.15`, start `h_E=2*theta` uniformly as an upper-bound basin probe;
`X=off`, `M=off`, `coop_A=0`. Run 1 s with the same connection and noise seed. A screening candidate
must be finite, have zero hard clip, tail rate ≥20 Hz, refractory-ceiling fraction <5%, and non-negative
tail H slope within numerical tolerance. Tail is the final 250 ms; the exact slope tolerance is
`slope_per_s >= -0.05*max(tail_h_mean,theta)`, i.e. no more than 5% decay per second. Refractory ceiling
is per-cell tail rate ≥80% of `1000/tau_ref_E`. A 1 s positive is only `screen_survivor`, never a basin
claim.

Complete all registered combinations unless parity/snapshot fails, numerical state is corrupted, OOM
safety fires, or the 10 h exploration budget is reached. A negative point does not stop the grid.

## 5. Frozen basin and X forks

Take at most six screen survivors using the following ordering, not a weighted score:

1. finite sustained tail;
2. lower ceiling fraction;
3. lower `rho`;
4. higher tail local-gain proxy;
5. smaller baseline H carryover;
6. an adjacent `(theta or rho)` survivor.

The local-gain ordering proxy is fixed as
`tail_gH_mean/max(tail_gA_mean,1e-12)`.  After the best sorted survivor is selected, its first available
one-grid neighbour in `rho` (then `k`) is inserted before the remaining sorted points.  This ordering is
developmental prioritisation only; it is not a scientific score.

Run matched fresh-noise forks (same seed/RNG reset).  A/B/C use
`min(5 s,max(2 s,5*tau_H))`.  Because the D arms must observe offset *and* a post-offset low interval,
they use `min(8 s,max(3 s,4*tau_H))`; this longer D-only cap is fixed before any fork outcome is read.

| arm | depletion | h init | X relay availability | question |
|---|---|---|---|---|
| A-low | healthy D=0 | low | 1 | accepted low state |
| A-high | healthy D=0 | 2theta | 1 | healthy state rejects high init |
| B | susceptible D=0.15 | low | 1 | susceptible low basin |
| C | susceptible D=0.15 | 2theta | 1 | susceptible finite high basin |
| D1 | same as C | high | mean depletion 0.128 | observed LC1 load can remove high |
| D2 | same as C | high | mean depletion 0.214 | stronger observed LC1 load can remove high |

The D arms are developmental uniform-load upper bounds derived from the accepted LC1 `D_X` values; they
are not spatial-X confirmation. Implement them as a frozen per-E relay field through the same E→E scatter
path. No additive X current is allowed.

Fork state labels use the final 1 s unless the D post-offset window below is longer.  `high_like` requires
mean rate >=20 Hz, rate occupancy above 20 Hz >=0.25, mean H >=theta, and the same 5%/s H-slope tolerance
as the screen.  `low_like` requires mean rate <20 Hz, high-rate occupancy <0.20, and mean H <theta.
The deliberate 0.20--0.25 occupancy gap is `unresolved`, not silently assigned.  Any non-finite value,
hard clip, or early numerical guard invalidates the complete matched fork set.

`H_BASIN_CANDIDATE` requires A-low/A-high/B to end in the same low statistical state, C to remain in a
distinct finite non-ceiling state after burn-in, and at least D1 or D2 to return to the matched low state
for `max(1 s,3*tau_H)`.  If the observed offset leaves less than that much recorded time, the D arm is
`unresolved_post_offset_window`, never a pass or a fail. One adjacent theta/rho point and a second noise stream are required for
`H_X_FROZEN_GEOMETRY_REPLICATED`.

Record rate, H/gH/gA, X, clip/tau ratio, single-cell rate/ISI summary, ceiling fraction, pairwise sample
correlation, fine-bin PSD and core/axis/off-axis recruitment. Morphology mismatch is diagnostic, not a
stop before X is tested.

## 6. Dynamic Z/H/X pilot

Only frozen-geometry candidates unlock this stage. Keep `M=0`.

For the two best replicated frozen candidates, run 20–30 s:

1. nominal dynamic Z/H/X;
2. matched X-off;
3. a second noise stream if resources permit.

No kick, hard reset or parameter step is allowed in the accepted lifecycle trajectory. The pilot asks:

- ≥8 s initial sparse returning IEDs;
- spontaneous entry into a bounded H-supported high state;
- X-on autonomous offset while matched X-off lasts substantially longer or reaches cap;
- post-offset suppression followed by returning sparse irregular IEDs.

Recovery means return to the pre-event statistical neighbourhood (event rate, IEI, duration,
participation and direction), not exact slow-variable reset. Narrowband/common-mode high activity is
allowed at Core candidate tier and deferred to later `M_i` morphology work.

## 7. Labels and claim boundary

Allowed outputs include:

```text
SENSOR_CHARACTERIZATION_COMPLETED
H_LOOP_SCREEN_NEGATIVE
H_BASIN_CANDIDATE
H_X_FROZEN_GEOMETRY_CANDIDATE
H_X_FROZEN_GEOMETRY_REPLICATED
X_OBSERVED_LOAD_INSUFFICIENT
ZXH_ENTRY_MISS
ZXH_OFFSET_POSITIVE_RECOVERY_NEGATIVE
ZXH_CORE_LIFECYCLE_CANDIDATE
```

Even the strongest label does not establish E1146 phenotype, broadband morphology, final spatial
recruitment or cohort replication.

## 8. Engineering and resources

- Six blessed engine files remain byte-identical; only non-blessed `mz_slow_vars.py` may gain an
  off-by-default frozen-X field.
- New paths require TDD, off-by-default byte parity, exact frozen-X scatter effect and deterministic
  restart tests.
- T>=20 s: strictly one worker.  The first short 40k run must measure RSS.  Subsequent E3 screen workers
  are capped at four and additionally by
  `floor((MemAvailable_start-96 GiB)/single_run_peak_RSS)`; E4 forks remain capped at two because their
  longer state matrices have not yet been timed.  This resource-only amendment was locked after the
  first E3 smoke measured 6.793 GiB with 182.5 GiB still available on an 80-CPU host; it does not depend
  on the scientific label of that row.
- Start a second worker only if `MemAvailable >= 96 GiB + 2*1.35*RSS_single` and swap is stable.
- OMP/OpenBLAS/MKL/NUMEXPR = 1. Swap +256 MiB stops submission; +512 MiB and rising terminates only the
  newest task-owned worker. Never touch sibling processes/worktrees.
- Runs >10 min use `setsid nohup`, stage flock, exact launcher PID, RUNNING/DONE/FAILED sentinel,
  resource log and wall guard. Wait by exact PID/sentinel, never `pgrep -f`.
- No hidden confirmation seeds are opened in this development sprint.

## 9. Required outputs

```text
results/topic4_sef_hfo/fcxr_lc2_core/closed_loop_exploration/
  execution_lock.json
  r1_resegmentation_summary.json
  r1_sensor_pareto.csv
  r1_sensor_support_map.npz
  h_loop_screen.json
  frozen_fork_map.json
  dynamic_pilot.json                 # only if unlocked
  candidate_verdict.json
  STATUS.md
  figures/*.png
  figures/README.md
```

Every generated figure must have real inputs and Chinese README text. Final archive records the complete
mechanism map, including all negative registered candidates; it must not collapse the sprint to one label.
