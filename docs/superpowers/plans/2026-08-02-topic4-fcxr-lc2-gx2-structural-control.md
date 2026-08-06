# FCXR-LC2-GX2 structural-control 2×2 — IMPLEMENTATION PLAN

Status: **SUPERSEDED CONTINGENCY — DO NOT EXECUTE**

Date: 2026-08-02
Design: `docs/superpowers/specs/2026-08-02-topic4-fcxr-lc2-gx2-structural-control-design.md`

## 0. Execution graph

```text
X0 GX1 authorization + artifact hashes
 -> X1 pure equations/TDD + force match
 -> X2 one 40k smoke
 -> X3 30-arm frozen architecture matrix
 -> X4 entry/offset aggregation and figures
 -> STOP for review
```

No dynamic Z/H/X command exists in this plan.

## Current authorization state

GX1 is complete, and its canonical decision is:

```text
NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP
+ X_PATH_REACHABLE_RANGE_INSUFFICIENT
-> LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE
```

Accordingly, **X0 currently fails by design** and X1--X4 must not run. The current H path is reachable
under strong experimental relay depletion, so adding a shared X/H path would answer a failure that was
not observed. Reachable does **not** mean already sufficient: the returning arms are `0.1` and `0.0`,
while the archived `0.872/0.786` loads at the same anchor stay high, which is why the X dynamic range is
carried forward as an unresolved fixed-`D` observation. This plan remains only as the required
conditional 2×2 contingency. The post-review executable route is LC3: first map the unchanged equation's
coupled `D-a_X` state plane and spatial instability. It does not authorize local D-gated entry geometry,
X retuning, or this matrix.

## 1. X0 — authorization

Require a future signed superseding verdict (not the current GX1 verdict) to contain:

```text
authorized_next_hypothesis = CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH
```

Hash GX1 verdict, GX1 X map, canonical frozen fork, P-field, baseline contract and six blessed engine
files. Any mismatch blocks execution. Write all architecture/arm/noise rows before code that runs 40k.

## 2. X1 — TDD and force match

Add off-by-default fields for `h_d_gate_lc2` and `h_shared_x_path_lc2`. Tests must prove:

1. both off reproduces current LC2 byte-for-byte;
2. `G_D(0.15)=1`, `G_D(0)≈0.0498`, monotonicity and finite input checks;
3. D_only reduces exactly to the locked additive H expression at D=0.15;
4. shared_only and combined give exactly zero recurrent H path when post-X recurrent input is zero;
5. no target-side X surrogate or new connectivity is introduced;
6. source timing remains causal (`t-` H and current post-X conductance);
7. five-arm × three-architecture × two-noise manifest has exactly 30 unique rows;
8. empirical workpoint/high-state classifiers and two-second offset floor remain unchanged.

Compute `gA_ref` only from the predeclared archived susceptible-high/no-X fork, then lock
`a_H=rho_H/gA_ref` in `force_match.json`. Do not inspect GX2 outcomes before locking it.

## 3. X2 — smoke and resources

Run `combined/healthy_low/noise401` for the full registered duration. Require finite output, no clip,
valid workpoint metrics and measured RSS. A scientific high result is allowed and does not stop; an
engineering failure blocks X3.

Before two workers, require the spec memory inequality and stable swap. Launch X3 with `setsid nohup`,
exact session leader, 9 h wall guard, stage flock, per-row DONE and aggregate FAILED/DONE sentinels.

## 4. X3 — frozen matrix

Run breadth-first by architecture then scientific arm so resource interruption does not leave only one
architecture observed. Each matched arm resets the same noise RNG and substrate state. Durations:

- non-X arms: at least `max(4000 ms,6 tau_H)`;
- X arm: at least `max(5000 ms,8 tau_H)`;
- required low window: `max(2000 ms,3 tau_H)`.

Complete all 30 registered rows unless resource/numerical safety stops the stage. Never unlock dynamic
slow variables because an intermediate architecture looks positive.

## 5. X4 — aggregation

For each architecture and noise, emit all five labels, tail rate/H, ceiling/clip, local gain, and offset
window. Aggregate entry and offset separately before the joint label.

Required figures:

1. `architecture_2x2.png`: archived old cell plus three new cells, with entry and offset gates shown
   independently;
2. `entry_offset_traces.png`: the five matched combined-arm traces for both noises;
3. Chinese `figures/README.md` written after visual inspection.

Archive the exact 30/30 result count, tests, hashes, resource peaks, sentinels and claim boundary. Stop
for review even if the joint frozen candidate passes.

## 6. Outcome routing

- joint combined pass -> write a new dynamic-lifecycle design candidate; do not run it;
- entry pass / offset fail -> X dynamic-range or load-sensor redesign only;
- offset pass / entry fail -> revisit the susceptibility coordinate, not M/K/A;
- all negative -> close local scalar H and consider source-resolved slow E→E state;
- unresolved -> measurement repair only.
