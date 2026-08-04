# FCXR-LC3 E4 — first no-kick trajectory (noise 401)

Date: 2026-08-04
Artifact: `results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/dynamic_reconnaissance/recon_noise401.json`
Source lock: `93816262` · wall 30083 s (8.4 h) · `n = 1` seed, seeds 405/406 in flight

## Abstract

A block of tissue carrying the wear a 24 s interictal recording had accumulated was
started from rest and left alone for 45 s. Nothing was injected: no kick, no reset, no
parameter step. It produced twelve ordinary interictal events, sped them up, enlarged
the last two, and then went into a sustained discharge at 5 s that never stopped for
the remaining 40 s.

Entry works and is cumulative rather than explosive. The carrier is bounded and is not
a refractory artefact. Termination does not happen. And the frozen geometry that was
supposed to describe this system turns out not to cover the region the trajectory
actually visits.

## 1. Entry — twelve events, accelerating, then a crossing

The event detector runs on a frozen bar taken from an accepted quiet reference record
(34 returning events, duration 8-22 ms, participation 0.045-0.080 with p90 = 0.0717,
event rate 0.086-3.15 /s). Against that reference:

| | events 1-10 | events 11-12 |
|---|---|---|
| participation | 0.046-0.062 — **10/10 inside the reference range** | **0.0921 / 0.0947** — both above the reference p90 |
| duration | 9-21 ms | 14 ms |

Inter-event interval fell from 456 ms to 266 ms across the twelve, i.e. a local event
rate of 2.19/s rising to 3.76/s, which **passes the reference upper edge of 3.15/s**.
The bout began in the window starting at 5.0 s.

Registered entry class: **`CUMULATIVE`** (12 returning events; the bar is 3).

This is not the startup transient it could have been. The trajectory starts cold from
rest, so an early bout is a transient candidate by default — but a transient ramps
monotonically from the beginning, and here the first ten events are statistically
indistinguishable from the quiet reference and only the last two depart from it.

Caveat that must travel with this: the pre-onset train sits at the **top** of the
reference event-rate band from its first event (2.19/s against the accepted canonical
low state's 0.9/s). This is a hot interictal regime, and 5 s is a short run-up.
"Twelve events pushed it across" and "a hot start climbed for five seconds" are not yet
separable — the per-event dose that would separate them was not recorded by this run.

## 2. Carrier — bounded, structured, not an artefact

- 40 s sustained, from 5 s to the end of the record;
- 454 events inside the bout, **every one of them self-terminating**, at 11.7 Hz;
- participation climbs 0.05 -> 0.43, duration 15-34 ms;
- **refractory-ceiling fraction 0.0000** against a 0.05 veto — not a tonic plateau;
- finite throughout, zero conductance clipping, minimum effective time constant
  0.275 ms.

Lifecycle label `ICTAL_LIKE_BOUNDED`, verdict `RECON_HIGH_WITHOUT_OFFSET`, with the
classifier's own reason: *"ictal bout runs to the end of the record; no autonomous
termination observed"*.

## 3. Termination — does not happen, and not for the expected reason

`X` behaves causally: it starts at 1.0, depletes only after onset, and reaches a mean
of 0.394. But it is neither slow nor still moving:

- `tau_x_down = 500 ms`, i.e. 80 time constants inside the bout;
- it has settled at its own set point, not en route to the `x_min = 0.1` floor;
- the sensor mean never reaches its gate — 59.4 Hz against `y_gate = 76.64 Hz` — so the
  Hill drive never saturates. Depletion happens at all only because the Hill is
  evaluated per cell and part of the population crosses the gate.

**Depth is the bottleneck, not speed.** Recovery time never enters: `X` never turns
around, its minimum falling at 41.5 s.

## 4. The frozen map does not contain this trajectory

| | frozen 102-row map | this trajectory at 44.75 s |
|---|---|---|
| mean `D` covered / reached | 0 .. **0.097** | **0.663** |
| core vs off-axis `D` | 2.6x enriched | **1.00x** — core_A 0.6631, core_B 0.6629, axial 0.6622, off_axis 0.6630 |
| `a_X` return boundary | high branch dies at `a_X <= 0.65` | **100% of cells below 0.65**, bout persists |

Two things follow.

**The map's domain does not contain the trajectory.** Its wear axis was sampled from
interictal quantiles; a 40 s seizure drives wear 6.8x beyond the largest of them. The
map is not wrong here, it is silent — an earlier draft of this note said "the map
predicted wrong", which overstates what a map can be blamed for outside its domain.

**The seizure erases the spatial structure it started from.** Before onset the wear was
2.8x enriched on the core/corridor; by 44.75 s all four regions sit at 0.663 within
±0.5%. The slow-vector probes had already measured uniform accumulation diluting the
corridor; this is the same effect at full strength. `X` depletion is likewise nearly
uniform (core 0.34-0.35, off-axis 0.39), not core-concentrated.

## 5. What this licenses, and what it does not

Licensed:

- entry into a bounded sustained state occurs without any injected drive, preceded by
  twelve returning events whose statistics match the frozen quiet reference until the
  last two;
- the carrier is bounded and free of the registered bad-data signatures;
- no autonomous offset within 40 s;
- the frozen `D x a_X` map cannot be used to predict this trajectory's termination.

Not licensed:

- anything about robustness — `n = 1` seed;
- any statement that the twelve events *caused* the crossing; the per-event dose and
  the per-event regional slow state were not recorded (the run stores the rate
  decimated to 10 ms against 8-19 ms events, and keeps only four of ~180 full-field
  snapshots). The event ledger added on 2026-08-04 records both, and needs a re-run;
- any claim that `X` cannot terminate a seizure — only that it does not here, at a `D`
  the map never covered. The 2x2 arbitration probe is what decides that.

## 6. Recoverability check

The full-resolution slow traces are **not** recoverable after the fact:
`save_prepared_checkpoint` compacts every `trace_*` list to empty before pickling, so
the landmark checkpoints carry the exact dynamical state but no history. This was
tested, not assumed. Per-event slow state therefore requires the instrumented re-run.

## 7. Links

- Design of record: `docs/superpowers/specs/2026-08-04-topic4-fcxr-lc3-event-driven-pivot-design.md`
- Ledger plan: `docs/superpowers/plans/2026-08-04-topic4-fcxr-lc3-event-ledger.md`
- Frozen geometry appendix: `docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md`
- Entry reader: `scripts/analyze_topic4_fcxr_lc3_entry.py`
- Arbitration probe: `scripts/run_topic4_fcxr_lc3_dxprobe.py`
