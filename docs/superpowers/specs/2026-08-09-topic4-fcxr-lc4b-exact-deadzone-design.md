# FCXR-LC4b exact-dead-zone cooperative terminator — DESIGN LOCK

Date: 2026-08-09

Status: **DESIGN LOCK — conditional execution authorised**

## 1. Scientific question

LC4 showed that making a smooth Hill actuator arbitrarily small in the interictal state does not
preserve the event-producing neighbourhood: n=6 reduced returning-event rate and n=8 increased it,
despite executed leakage below 0.005% of recurrent excitation.  LC4b asks the narrow next question:

> Does replacing merely-small interictal activation with an exact low-load dead zone preserve the
> paired interictal trajectory, while retaining the D/Z onset surface and the already observed
> slow-off termination/recovery authority?

This is not an n sweep and not a new sensor.  The cell still reads only its own linear spike load.
The new parameter is an executor threshold: below it the high-threshold outward channel is exactly
closed.  It is a reduced high-threshold conductance, not a claim about one identified ion channel.

## 2. Frozen lineage and evidence

Keep unchanged:

- RC1 recurrent E-to-E conductance, recurrent-only saturation and patient-specific axis;
- H1 local carrier point, D/Z/H/X dynamics, core geometry, tonic drive and event bar;
- `tau_m=1000 ms`, `tau_a_on=100 ms`, `tau_a_off=10000 ms`;
- the n=4 high-state current target `I_target=44.8619393917937`;
- all LC4 F0 functional bands and F2 lifecycle/recovery gates.

The sole calibration artifact is
`results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/percell_separation/per_cell.npz`, sha256
`81d2c97a16eaf69753951dcd91f55153c7102d5d7ca4c0f75e12f9b24208a33d`.  For `tau_m=1000 ms` it gives:

```text
max(interictal peak load) = 35.99027633666992
min(ictal settled load)   = 57.674434661865234
median(ictal load)        = 66.70187759399414
```

The clean extreme gap is positive before any new simulation.

## 3. The one locked candidate

Define

```text
m0 = (max_interictal + min_ictal) / 2 = 46.83235549926758
u_i = max(m_i - m0, 0)
K_excess = median_ictal - m0 = 19.869522094726562
a_inf(m_i) = u_i^4 / (K_excess^4 + u_i^4)
tau_a da_i/dt = a_inf(m_i) - a_i
I_out,i = g_max a_i
```

Use n=4 because that curve already demonstrated termination authority; n=6/n=8 are closed by the
LC4 F0 result.  On the frozen calibration artifact the interictal activation is exactly zero, the
ictal mean activation is `0.5055224225319433`, and half the ictal cells are at or above half
activation.  Force matching gives

```text
g_max = I_target / mean(a_ictal) = 88.74371816605014.
```

No threshold, exponent, time constant or dose may be changed after execution starts.

## 4. D0 analytic and engineering gate

Before 40k execution require:

- calibration artifact hash matches;
- `max_interictal < m0 < min_ictal`;
- activation is bit-exact zero for every stored interictal cell;
- ictal mean activation is at least 0.20 and `g_max` is finite;
- `m_hill_deadzone=None` is byte-identical to the existing LC4 path;
- `m_hill_deadzone=0` is numerically identical to the old Hill equation;
- invalid deadzone/K combinations fail loudly;
- six blessed engine hashes remain unchanged; mechanism-module hash is locked separately.

Failure is `DEADZONE_NOT_IDENTIFIABLE`; stop without 40k simulation.

## 5. D1 paired functional baseline

Run one 12 s, seed1/noise401, no-kick candidate at frozen `D_healthy`, relay=1.  The actuator-off
control is the committed LC4 control from the same substrate and seed.  Reuse is allowed only if:

- its JSON/NPZ hashes match the LC4 closeout commit;
- all dynamics sources outside the off-by-default deadzone path match;
- tests prove the new field is inert when unset.

The candidate must satisfy every LC4 F0 functional band and numerical clause.  In addition:

- `max(a_i)` and maximum delivered current must remain **exactly zero** throughout D1;
- the saved population-rate and active-fraction traces must be byte-identical to the committed
  control traces.

If either exact-zero or exact-trace identity fails, report `DEADZONE_BASELINE_NOT_INERT`; do not
reinterpret a small difference as acceptable.  If event statistics fail despite exact identity,
the instrument is invalid.

## 6. D2 frozen-D onset surface

The committed actuator-off `D10` quiet-watch row (departure at 7 s in a 12 s record) is the positive
control if its provenance hashes match.  D1 supplies the stable `D_healthy` row.  Run the dead-zone
candidate sequentially at `D10`, then only if needed `D30`, then only if needed `D50`; each is 12 s
from a fresh low state, relay=1, no kick.

Pass when at least one candidate row through D50 develops a whole-record sustained bout while
`D_healthy` remains stable.  Stop at the first departing field.  Failure is
`DEADZONE_ONSET_SURFACE_UNREACHABLE`; do not extend time, alter Z or lower the dead zone.

## 7. D3 one continuous lifecycle

Only D1+D2 pass may launch one 70 s seed1/noise401 trajectory from rest with dynamic Z/H/X and the
dead-zone actuator active from t=0.  No kick, reset, parameter step or saved-high-state fork.

Reuse the locked LC4 F2 adjudicator unchanged.  Acceptance requires, in the same trajectory:

- at least 8 s and 3 returning IEDs before spontaneous onset;
- a bounded non-refractory high bout lasting 1–5 s;
- autonomous offset and at least 2 s without relapse, with postictal rate below pre-onset mean;
- refractory-ceiling fraction <=1%;
- the pre-fixed final 8 s are interictal and contain at least 3 returning IEDs whose rate,
  duration and participation lie inside the frozen LC1 reference;
- from the exact final state, freeze the actual spatial D and continue 12 s; after 2 s burn-in the
  remaining 10 s again contain at least 3 reference-like returning IEDs and no ictal bout.

A nominal pass before exact-D continuation is only `D3_NOMINAL_ELIGIBLE_FOR_CONFIRM`.

## 8. Stop rules and claims

- D1 fail closes this exact-dead-zone candidate; no rescue threshold.
- D2 fail means the terminator protects baseline by moving entry out of the tested D range.
- D3 fail is reported at entry, carrier, offset, postictal protection or distributional return;
  do not collapse them into one `lifecycle failed` label.
- A single seed1 success is a **candidate lifecycle**, not robust lifecycle; seed3/unseen-noise
  confirmation requires a later lock.
- Do not call the frozen high-state fork a lifecycle or use it to fill a missing D3 leg.

## 9. Resources and provenance

- T>=20 s: exactly one 40k worker; T<20 s: one worker in this sprint.
- Pin OMP/MKL/OpenBLAS/NUMEXPR to 1.
- Do not submit below 128 GiB MemAvailable; stop new submission at swap delta +256 MiB; terminate
  only this task's newest run at +512 MiB.
- Every long stage uses `setsid nohup`, PID, stage flock, RUNNING/DONE/FAILED/STOP sentinels and a
  source-hash lock.  Never touch sibling processes/worktrees.
- Results root:
  `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4b_deadzone_lifecycle/`.
