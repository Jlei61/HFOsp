# FCXR-LC3 — what sustains the non-terminating bout, and can the relay be made to stop it

Date: 2026-08-04
Seed state for every arm: `dynamic_reconnaissance/exact_landmarks/noise401_step895001.pkl`
(the byte-parity-verified state 44.75 s into the first no-kick trajectory)
Artifacts: `dx_arbitration_probe/dx_arbitration.json`, `hill_placement_sweep/hill_sweep.json`

## Abstract

A block of worn tissue, left alone, ignited by itself and then discharged for forty
seconds without stopping. Two questions follow: what holds it up, and can the brake it
already has be made to release it.

The first has a clean answer. Wear and relay availability each hold it up on their own
— drop either one and the discharge stops — so the bout needs both together. The
second does not. Moving the relay's response curve does deepen the relay and does weaken
the discharge, but it stalls the tissue in a middle state around forty spikes a second
instead of returning it to its own quiet rhythm, because the very activity that empties
the relay disappears as the relay empties.

One measurement recurs and is worth stating plainly: **holding the relay at a depth is
not the same as letting it descend to that depth.** Held at 0.35 the discharge stops
cleanly; descending on its own to 0.26 it does not.

## 1. The 2x2: both factors are individually sufficient

Wear frozen at what the trajectory reached (0.663) or at the largest level the frozen
102-row map ever covered (0.097); relay frozen at the observed field (mean 0.3945) or
at its 0.10 floor.

| | relay = observed 0.3945 | relay = 0.100 floor |
|---|---|---|
| **wear 0.663 (observed)** | **persists**, 71.2 Hz | terminates, 0.39 Hz |
| wear 0.097 (map maximum) | terminates, 0.52 Hz | terminates, 0.24 Hz |

One of four cells sustains. The control reproducing the bout is what licenses reading
the other three at all.

This also validates the frozen map inside its own domain: it says the high branch dies
below `a_X = 0.65`, and at wear 0.097 with a relay mean of 0.3945 it does. The map is
not unreliable, it is silent outside the wear range it sampled — interictal quantiles
topping out at 0.097, against the 0.663 a forty-second discharge produces.

## 2. The relay depth that terminates, and by how much the trajectory missed

Uniform relay clamps at the observed wear:

| relay | outcome | mean rate |
|---|---|---|
| 0.3950 | persists | 72.12 Hz |
| **0.3800** | **terminates** | 8.73 Hz |
| 0.3500 | terminates | **3.07 Hz** |
| 0.1000 | terminates | 0.39 Hz |

The threshold is between 0.380 and 0.395; the trajectory's relay settled at 0.3945, so
it missed self-termination by under 4%. Both no-kick seeds settled at the same 0.388
floor, so the miss replicates.

The flip is all-or-none: 0.045 of relay depth takes the discharge from 72 Hz to 3 Hz,
occupancy 1.000 to 0.000, the local feedback variable from 1.83 to 0.004.

**Retracted 2026-08-04 — the rates in that table are window artefacts.** An earlier
version of this section read them as showing that 0.350 "lands at 3.07 Hz, right at the
canonical 2.81 Hz, so there is a window where the relay both stops the discharge and
returns the tissue to its own interictal rate." That is wrong. Those means are taken
over the whole 5 s window, which begins with the collapse out of the discharge, and the
collapse dominates them.

Measured properly — 20 s with the first 2 s excluded as collapse, not interictal
activity — the shallowest terminating clamp is silent:

| clamp | mean rate over the 5 s window (collapse included) | post-collapse rate over 18 s | returning events |
|---|---|---|---|
| 0.380 | 8.73 Hz | **0.093 Hz** | **0 in 18 s** |

Zero events is outside the frozen reference band of 0.086-3.15 events/s, and 0.093 Hz
sits with the geometry map's quenched columns (0.04-0.06 Hz), not with the canonical
quiet state's 2.81 Hz. All four clamps agree: 0.380, 0.350, 0.300 and the 0.100 floor
each produce zero returning events across 18 s.

**Second retraction, same day.** An intermediate reading of those four arms concluded
that "at this wear the tissue discharges above 0.395 and is silent below 0.380, with no
interictal branch in between", making the Stage-2 failure structural. That is also
withdrawn, and for a reason worth carrying: **the clamp is what produces the silence.**

The 12 s Hill arms have the relay free, and it does not stay down. It empties to
0.26-0.28, the discharge collapses, and then — activity having fallen below the gate —
it refills to **0.59-0.60**, well above the 0.3945 the trajectory sat at:

| gate | relay start | minimum | final | tier at 12 s | rate |
|---|---|---|---|---|---|
| 76.64 | 0.3945 | 0.3902 | 0.3936 | high branch | 72.9 Hz |
| 72.00 | 0.3945 | 0.2795 | **0.5960** | elevated event train | 40.7 Hz |
| 68.00 | 0.3945 | 0.2716 | **0.5959** | elevated event train | 36.5 Hz |
| 64.00 | 0.3945 | 0.2692 | **0.6043** | elevated event train | 32.8 Hz |
| 60.00 | 0.3945 | 0.2620 | **0.5884** | elevated event train | 32.2 Hz |

So the discharge does terminate and the relay does recover; what is left is an elevated
event train, not silence and not the interictal rate. A clamped relay cannot refill, so
the clamped arms measure the clamp rather than the tissue.

Corrected picture:

| condition | outcome at this wear |
|---|---|
| relay clamped (artificial) | discharge, or silence |
| relay free, wear pinned | discharge at the registered gate; **32-40 Hz elevated train** once the gate moves |
| relay free, wear free | being measured |

The remaining gap — 32-40 Hz against a canonical 2.81 Hz — is plausibly the pinned wear
itself, which is exactly what the free-wear arms test. Wear relaxes back whenever the
inhibitory sensor falls below threshold, on a 5 s constant, so pinning it at the 0.663 a
discharge produced may be the whole reason nothing returns to baseline.

**The methodological lesson, twice in one day:** a constraint imposed to isolate a
variable becomes part of the result. Averaging across a window that opens with a state
transition measures the transition; clamping a variable that would otherwise recover
measures the clamp.

Replacing the real relay field (mean 0.3945, spread 0.10-0.63) with a uniform clamp of
the same mean did not change the outcome. That is not proof that spread never matters —
both sit on the persisting side.

## 3. The Hill sweep: moving the curve does not close the gap

Wear pinned at 0.663, relay left free, only the placement of its response curve varied.
Adjudicated from `resolved_label` in three tiers, never from the runner's binary field.

| gate | tier | relay minimum | mean rate |
|---|---|---|---|
| 76.64 (registered) | high branch | 0.378-0.390 | 66-73 Hz |
| 75.00 | high branch | 0.3831 | 68 Hz |
| 74.00 | high branch | **0.3790** | 66 Hz |
| 72.00 | unresolved | 0.272-0.280 | 45-51 Hz |
| 68.00 | unresolved | 0.269-0.273 | 41-44 Hz |
| 64.00 | decayed transient | 0.262-0.269 | 38-41 Hz |
| 60.00 | decayed transient | 0.258-0.263 | 36-38 Hz |

**Zero of fifteen arms reached the interictal tier.**

Three things this establishes.

**The relay saturates.** From gate 72 to gate 60 — a further 17% — the minimum moves
only 0.2795 to 0.2629. Emptying the relay quiets the tissue, quieting the tissue removes
the sensor drive that empties the relay, and the two settle against each other.

**The half-activation is a weak knob and the gate a strong one.** Cutting the
half-activation 40% moved the relay 0.013; cutting the gate 6% moved it 0.118. Once the
gate moves, the half-activation barely registers.

**A transient dip below the clamped threshold is not enough.** Gate 74 dips to 0.3790 —
below the 0.380 that terminates when *held* — and the bout persists regardless. The
clamped bracket does not transfer to a descending trajectory.

## 4. Claim boundary

Licensed: the arbitration and the bracket, on one real late-bout state; the sweep's
finding that moving the Hill deepens the relay and weakens the discharge without
reaching the interictal tier inside the registered 5 s window.

Not licensed:

- any lifecycle claim — every arm here is frozen or half-frozen, seeded from one state
  of one noise seed;
- any parameter acceptance — a terminating clamp is a mechanism demonstration;
- reading `n_terminating` from `hill_sweep.json`. It says 12; the true count of arms
  reaching interictal is 0. The field conflates "not a high branch" with "terminated",
  folding six unresolved and six decayed arms into the count. Use `resolved_label`;
- reading `mechanism_consistent` from the per-arm records. It compares an instantaneous
  relay minimum against a threshold measured under a sustained clamp, which §3 shows are
  different quantities.

## 5. Open

Whether the unresolved and decayed arms terminate slowly or hold a stable smouldering
state. Their local feedback variable collapses at 0.52-0.75 per second with the crossing
projected 1.2-1.7 s past the registered window, which argues for slow termination; the
relay and rate saturation argues for a stable middle state. A 12 s window is running.

The extrapolation deserves distrust on its own terms: the control arm projects a
crossing at 30.8 s, and the real 45 s trajectory never terminated at all.

## 6. Method notes worth carrying

- Every arm uses the geometry map's screen/extend protocol and classifier, so all
  outcomes here sit on the same scale as the 102 map rows.
- `freeze_dynamic_state` is separate from `replace_frozen_fields` on purpose: the latter
  swaps fields on an already-frozen state and leaves the config alone when the frozen
  fields are `None`, which is exactly the dynamic case, so it would set the arrays and
  let them evolve again.
- Peak memory is duration-driven at 0.596 GiB per simulated second of E spike bools. The
  geometry map's fixed 2.0 worst-case factor is calibrated for a 5 s extension and
  under-budgets a longer window by the ratio of the two; the sweep derives its own.
- A process pool holds each spawned worker's substrate (~6.5 GiB) even while idle, so the
  pool width bounds residency, not just submission.
