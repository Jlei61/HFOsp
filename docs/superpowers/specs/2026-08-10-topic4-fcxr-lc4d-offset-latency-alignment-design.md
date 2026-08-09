# FCXR-LC4d offset-latency alignment — DESIGN LOCK

Date: 2026-08-10

Status: **DESIGN LOCK — one analytic candidate authorised**

## 1. Scientific question

LC4c produced the first fresh, no-kick trajectory in this lineage with both cumulative entry and autonomous offset: onset at 11 s after 29 returning events, offset at 66 s.  It nevertheless failed the lifecycle contract because the high bout lasted 55 s rather than 1–5 s and left only four post-offset seconds.

LC4d asks one narrow question:

> If the exact-dead-zone executor is rescaled so that the already validated termination target current becomes reachable by the fourth second after the observed onset, while the D/Z/H entry coordinate and all time constants remain frozen, does a fresh trajectory retain cumulative entry, terminate within 1–5 s, and leave enough time for returning-IED recovery?

This is not a parameter grid.  It does not change the sensor, dead zone, Hill exponent, Z/H/X equations, patient-specific spatial axis, noise, connection seed or lifecycle gate.

## 2. Locked analytic repair

Authoritative LC4c evidence:

```text
onset = 11000 ms
offset = 66000 ms
I_target = 44.8619393917937
a_mean(onset + 4000 ms) = a_mean(15000 ms)
                                = 0.06110576540231705
```

The registered 1–5 s carrier bound leaves one second between `onset+4 s` and the latest accepted offset.  The only new value is therefore

```text
g_m_max = I_target / a_mean(onset + 4000 ms)
        = 734.1686843528613.
```

No safety factor, interpolation or neighbouring value is authorised.  The repair is deliberately tied to a time-domain failure (late offset), not to the maximum activation anywhere in the 70 s record.

## 3. Frozen candidate

Keep from LC4c:

- `theta_h_lc2=1.7317735254764568`;
- `deadzone=46.83235549926758`, `K=19.869522094726562`, `n=4`;
- `tau_adp=1000 ms`, `tau_a_on=100 ms`, `tau_a_off=10000 ms`;
- all RC1, Z, H, X, E/I, geometry, seed and classifier settings;
- exact low-load dead zone and frozen returning-IED reference bands.

Change only:

```text
g_m_max = 734.1686843528613.
```

Because dead-zone activation is exactly zero on the frozen interictal load artifact, changing `g_m_max` must remain exactly invisible there.  This structural statement is rechecked; it is not inferred from a small current.

## 4. L0 provenance and bad-data gate

Before simulation:

- lock sha256 for LC4c candidate, entry verdict, nominal verdict and nominal trace;
- verify LC4c was no-kick/no-reset/no-step, finite, zero clip, onset 11 s and offset 66 s;
- verify the activation trace uses 10 ms spacing and contains the exact `t=15000 ms` sample;
- verify the analytic identity `g_m_max*a_15s=I_target` to `1e-12`;
- verify candidate activation on the frozen interictal load artifact is exactly zero;
- verify the first 4 s of the executed LC4c trace are exactly zero current;
- lock six blessed engine hashes and the non-blessed slow-variable mechanism hash.

Failure is `OFFSET_LATENCY_REPAIR_NOT_IDENTIFIABLE` and no 40k run is authorised.

## 5. L1 18 s fresh entry/offset screen

Run one fresh seed1/noise401 trajectory from rest for 18 s with dynamic Z/H/X and the full LC4d candidate active at t=0.  No kick, reset, state fork or parameter step.

Pass only if all hold:

- a qualifying high bout begins in `[8,15] s`;
- at least 3 self-terminating returning events precede onset;
- the first 8 s contain no qualifying high bout;
- executed current is exactly zero over the fixed first 4 s;
- the high bout lasts 1–5 s and ends inside the record;
- at least 2 s after offset are observed with no ICTAL relapse;
- the first 2 s post-offset mean rate is below the pre-onset mean;
- finite, zero clip and refractory-ceiling fraction `<=0.01`.

The 18 s screen is a cost-control gate, not a lifecycle claim.  It cannot establish returning-IED recovery.

## 6. L2 70 s nominal and conditional exact-D confirmation

Only L1 pass launches one new 70 s fresh-from-rest trajectory with the identical locked candidate.  Reuse the unchanged LC4 nominal adjudicator:

- at least 8 s and 3 returning IEDs before spontaneous onset;
- bounded non-refractory high bout lasting 1–5 s;
- autonomous offset, 2 s relapse guard and postictal rate below pre-onset mean;
- fixed final 8 s entirely after the guard, interictal, with at least 3 events matching frozen rate, duration and participation bands.

Only nominal eligibility launches the unchanged exact-final-state, actual-spatial-D-frozen 12 s confirmation.  After 2 s burn-in, the remaining 10 s must be low/interictal and reference-like.

## 7. Stop rules and claim boundary

- L1 no qualifying bout: `TERMINATOR_PREVENTS_QUALIFYING_ENTRY`.
- L1 bout longer than 5 s or running to record end: `OFFSET_LATENCY_REPAIR_INSUFFICIENT`.
- L1 bout shorter than 1 s is not a qualified carrier and cannot pass by disappearing quickly.
- L1 offset followed by relapse or no rate suppression: `SHORT_POSTICTAL_PROTECTION_INSUFFICIENT`.
- L2 late/absent offset: stop; no dose escalation in this sprint.
- L2 offset without frozen-reference return: `DISTRIBUTIONAL_RETURN_FAILED`.
- A single seed1 success is only a candidate lifecycle.  Robustness requires a later preregistered connection/noise confirmation.
- No result may be called an E1146-like ictal morphology without the separate morphology gate.

## 8. Resources and detached execution

- Every 40k stage uses one worker and one pinned BLAS/OpenMP thread.
- Every simulation uses `setsid nohup`, independent PID/SID, stage flock, source lock and RUNNING/DONE/FAILED/STOP sentinels.
- Do not submit below 128 GiB MemAvailable; block later submission at swap delta +256 MiB; terminate only the newest task-owned run at +512 MiB.
- L1 wall guard: 4 h.  L2 nominal: 7 h.  Exact-D: 3 h.
- Never touch sibling processes or worktrees.
- Result root: `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4d_offset_latency_alignment/`.
