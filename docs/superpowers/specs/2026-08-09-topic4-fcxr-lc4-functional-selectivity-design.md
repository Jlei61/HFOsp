# FCXR-LC4 functional selectivity and lifecycle — DESIGN LOCK

Date: 2026-08-09
Status: **DESIGN LOCK — execution authorised**

## 1. The one question

Can the cooperative per-cell outward-current actuator be made steep enough to preserve the
interictal event-producing neighbourhood, while leaving the dynamic `D = 1 - Z` path able to cross
an onset surface and then using the already observed slow-off authority to stop the high state and
protect recovery?

This sprint is not a broad parameter sweep.  It separates three failure modes in order:

1. **baseline leakage** — the actuator suppresses ordinary returning IEDs;
2. **entry displacement** — baseline survives, but the actuator moves the frozen-D onset surface
   outside the range reached by repeated IEDs;
3. **lifecycle failure** — baseline and entry both survive, but the high state does not terminate or
   returning IED statistics do not recover.

## 2. Frozen upstream assets

- RC1 recurrent E-to-E conductance with smooth saturation;
- H1 local cooperative carrier point and patient-specific E-to-E axis;
- dynamic Z/H/X equations and the LC3 event ledger;
- the 12 s frozen-field quiet-watch result (`D_healthy` stable; `D10` first departs at 7 s);
- LC4 fork result: n=4 slow-off stops the established high state and lets mean E-cell D fall from
  0.436 to 0.0017; this is termination authority, not a lifecycle;
- LC1 baseline event bar and event distributions.

No connection, tonic drive, H point, Z calibration, X calibration, core mask or event threshold may
change in this sprint.

## 3. Candidate family

The load remains the existing linear per-cell spike filter (`tau_m = 1000 ms`).  Only the channel
activation curve changes:

```text
a_inf(m) = m^n / (K^n + m^n)
tau_a da/dt = a_inf(m) - a
I_out = g_max a
```

Locked candidates are `n in {6, 8}`.  `K` remains the measured mid-gap value.  The ictal current is
force-matched to the executed n=4 arm:

```text
I_target = g_max(n=4) * mean_a_ictal(n=4)
g_max(n) = I_target / mean_a_ictal(n)
```

Thus n changes curve shape, not the ictal dose.  `tau_a_on = 100 ms` and
`tau_a_off = 10000 ms` remain fixed.  If both candidates pass, n=6 is selected because it is the
less singular transfer curve.

## 4. Stage F0 — paired functional baseline

Each arm starts from the same fresh fast state with the same connection/noise seed, H active,
`D_healthy = 0` frozen and relay availability frozen at 1.  Run 12 s and score after a 2 s burn-in.
The paired control is the identical configuration with the cooperative actuator off.

A candidate passes only if all hold:

- numerical safety and no sustained high bout;
- at least 3 returning events after burn-in;
- event-rate ratio to control in `[0.80, 1.25]`;
- IEI-CV ratio in `[2/3, 1.5]`;
- median duration ratio in `[0.75, 4/3]`;
- median participation ratio in `[0.80, 1.25]`;
- maximum delivered outward current below `0.1%` of the locked recurrent-excitation scale.

The last clause is a leakage diagnostic; the functional event clauses carry the verdict.

If neither n passes, stop this Hill family.  The next design may add a true dead zone, but this
sprint may not invent it.

## 5. Stage F1 — frozen-D onset surface

Only the selected F0 candidate is run from the low state at
`D_healthy, D10, D30, D50`, relay frozen at 1, for 12 s.  One actuator-off `D10` row is rerun as the
positive-control anchor.  The whole record is scanned for a sustained bout; tail-only labels cannot
hide a departure.

F1 passes only if:

- actuator-off `D10` departs;
- actuator-on `D_healthy` does not depart;
- at least one of actuator-on `D10/D30/D50` departs.

The first departing field is the mechanism-conditioned onset surface bracket.  Absence through
`D50` means `ONSET_SURFACE_UNREACHABLE_IN_TESTED_RANGE`; do not extend the observation window or
increase Z strength in this sprint.

## 6. Stage F2 — one continuous lifecycle

Only after F0 and F1 pass, run one 70 s trajectory from rest with dynamic Z/H/X and the selected
actuator active from t=0.  No kick, reset, parameter step or saved-high-state fork is allowed.

Acceptance requires all of the following in the same trajectory:

- at least 3 returning IEDs before spontaneous onset;
- a bounded high state lasting 1–5 s, not runaway or a refractory-ceiling plateau;
- autonomous offset;
- postictal suppression;
- the actual post-offset D field, when frozen at the first candidate recovery time, remains low for
  12 s (the scalar `D=0.047` guide alone is not sufficient);
- at least 8 s of post-offset observation containing returning IEDs whose event rate, duration and
  participation return to the frozen LC1 reference distribution.

Failure is reported at the furthest completed leg.  A 70 s no-onset run is not rescued by extending
time unless D is still increasing and the registered wall-time window, not an equilibrium, is the
only unresolved quantity; such an extension needs a new lock.

## 7. Resources and provenance

- T >= 20 s: exactly one 40k worker; T < 20 s: at most two, but this execution defaults to one.
- OMP/MKL/OpenBLAS/NUMEXPR threads fixed to 1.
- Do not submit if MemAvailable < 128 GiB.  Stop new submission at swap delta +256 MiB; abort this
  task's newest run at +512 MiB.  Never touch sibling processes or worktrees.
- Every long run uses `setsid nohup`, a PID file, RUNNING/DONE/FAILED sentinel and stage lock.
- Results live under `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4_lifecycle_gate/`.

## 8. Claim boundary

F0 proves functional baseline selectivity only.  F1 proves a frozen-D onset surface only.  The LC4
fork proves termination authority only.  Only F2 plus the frozen actual-D recovery check may be
called a candidate complete lifecycle, and confirmation seeds remain a later sprint.
