# FCXR-LC4c entry-offset alignment — DESIGN LOCK

Date: 2026-08-10

Status: **DESIGN LOCK — one conditional candidate authorised**

## 1. Scientific question

LC4b established three facts in one lineage: an exact low-load dead zone preserves the paired
interictal trajectory exactly; the D/Z entry surface remains reachable; but the 70 s closed
trajectory enters after only 5 s and never autonomously offsets. LC4c asks the smallest next
question:

> If the already measured H entry threshold is moved to the existing 11 s cumulative-entry
> anchor, and the exact-dead-zone executor is rescaled only enough to deliver its previously
> matched termination current in the actually observed closed high state, does one no-kick
> trajectory complete entry, bounded activity, autonomous offset and returning-IED recovery?

This is an entry/offset alignment experiment, not a parameter sweep. It changes no sensor,
spatial edge, Z/X equation, dead-zone location, Hill exponent or time constant.

## 2. Two independently anchored repairs

### 2.1 Entry coordinate

The previously executed LC3 entry grid contains one row with the registered `tau_z=5000 ms` and
only `theta_h_lc2` raised by 10%:

```text
source: stage1_entry_window/cell_tauz5000_theta110.json
theta_h_lc2 = 1.7317735254764568
onset = 11000 ms
returning events before onset = 29
no kick/reset/parameter step; relay frozen at 1
```

The registered row at `theta_scale=1.0` entered at 5 s. LC4c therefore locks the existing 1.1
row; it does not interpolate or scan theta. This changes where cumulative D/Z/H crosses entry,
not the D/Z wear equation itself.

### 2.2 Offset dose

LC4b locked `I_target=44.8619393917937`, but the closed 70 s high state reached only
`a_mean_max=0.2963244915008545`. The sole dose repair is the analytic transfer

```text
g_m_max = I_target / a_mean_max
        = 151.3946389128093
        = 1.7059758374055478 * g_m_max_LC4b.
```

No safety factor is added. This is a one-time correction for the observed state-distribution
shift between the frozen calibration artifact and the actual closed high state. Because the
dead zone made activation exactly zero in the paired Dhealthy baseline, this dose change is
structurally invisible there; that fact is still checked by provenance and bad-data regression.

## 3. Frozen candidate

Keep from LC4b:

- `m0=46.83235549926758`, `K_excess=19.869522094726562`, `n=4`;
- `tau_m=1000 ms`, `tau_a_on=100 ms`, `tau_a_off=10000 ms`;
- RC1 recurrent conductance/saturation, patient-specific E-to-E axis and two cores;
- Z/H/X equations and all parameters other than `theta_h_lc2`;
- lifecycle adjudicator, LC1 reference band and exact-final-D confirmation;
- connection seed 1, noise seed 401 for this development candidate.

Change only:

```text
theta_h_lc2 = 1.7317735254764568
g_m_max = 151.3946389128093
```

## 4. C0 lock and regression gate

Before simulation:

- both source artifacts and sha256 values are recorded;
- the 1.1 entry row is finite, zero-clip, no-kick and onset is in `[8,15] s`;
- LC4b nominal artifact reports no offset and the exact `a_mean_max` above;
- the analytic dose identity holds to `1e-12`;
- candidate activation remains exactly zero on the frozen interictal load artifact;
- unset candidate fields preserve the LC4b code path;
- six blessed engine hashes and the slow-variable mechanism hash are locked.

Failure is `ENTRY_OFFSET_REPAIR_NOT_IDENTIFIABLE`.

## 5. C1 15 s dynamic-entry gate

Run one fresh 15 s seed1/noise401 trajectory from rest with dynamic Z/H/X and the full LC4c
candidate active at t=0. No kick, reset or parameter step.

Pass only if:

- first qualifying high bout begins in `[8,15] s`;
- at least 3 self-terminating events precede onset;
- the first 8 s are free of a qualifying ictal bout;
- the trajectory is finite, zero-clip and not at the refractory ceiling;
- the executed current is exactly zero over the fixed first 4 s (bad-data regression against an
  executor that leaks before cumulative entry).

If onset is early, absent, or numerical safety fails, stop. Do not adjust theta or dose.

## 6. C2 70 s continuous lifecycle

Only C1 pass launches one fresh 70 s trajectory from rest with the identical locked candidate.
It is not a continuation of the 15 s run. Reuse the LC4b adjudicator unchanged:

- at least 8 s and 3 returning IEDs before spontaneous onset;
- bounded non-refractory high bout lasting 1--5 s;
- autonomous offset and at least 2 s without relapse, with postictal rate below pre-onset mean;
- the fixed final 8 s are interictal and contain at least 3 events matching the frozen reference
  rate, duration and participation bands.

Only nominal eligibility may launch the exact-final-state, actual-spatial-D-frozen 12 s
continuation. After 2 s burn-in, the remaining 10 s must again be interictal and reference-like.

## 7. Stop rules and claims

- C1 failure closes this one aligned candidate; no grid or rescue.
- C2 entry without offset is `OFFSET_DOSE_REPAIR_INSUFFICIENT`.
- Offset followed by relapse is `POSTICTAL_PROTECTION_INSUFFICIENT`.
- Low activity without frozen-reference events is `DISTRIBUTIONAL_RETURN_FAILED`.
- A single seed1 success is only a candidate lifecycle; robust lifecycle requires a later
  connection/noise confirmation lock.
- The old high-state forks may justify the dose anchor but never count as a lifecycle leg.

## 8. Resources and detached execution

- Every 40k stage uses exactly one worker and pinned BLAS/OpenMP threads.
- Every simulation uses `setsid nohup`, its own PID/session, stage flock, source lock and
  RUNNING/DONE/FAILED/STOP sentinels.
- Do not submit below 128 GiB MemAvailable; stop new submission at swap delta +256 MiB; terminate
  only the newest task-owned run at +512 MiB.
- C1 wall guard: 3 h. C2 wall guard: 7 h, raised from 6 h because the measured LC4b 70 s run took
  5 h 44 min. Exact-D guard: 3 h.
- Never touch sibling processes/worktrees.
- Results root:
  `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4c_entry_offset_alignment/`.
