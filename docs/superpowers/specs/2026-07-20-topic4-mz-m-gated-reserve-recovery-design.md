# Topic 4 MZ M-gated inhibitory-capacity recovery（locked R3 design v1.0）

Date: 2026-07-20

Branch: `codex/topic4-mz-divisive-lifecycle`

Status: locked after R2 and before the canonical R3 run. Numerical estimates used to choose the registered axis are design pilots, not accepted results.

## 1. Why R3 exists

R2 established a structural conflict in a single constant recovery timescale.
Preictal event accumulation needs slow recovery, whereas the postictal latch
must restore effective inhibitory capacity to `q=.885` within `120 s`. Only
the isolated `tau_r=80 s` node passed all three q-hold cells, so coupled R2 is
closed.

R3 asks one narrower question: can the already existing dimensionless M state
separate those two recovery regimes without changing recurrent excitation or
adding another current?

The registered prediction is deliberately bounded. M is zero before entry,
so R3 cannot repair the `70 s` high-q entry-margin failure or the `120/160 s`
sparse-schedule failures. It can only turn the R2 upper handoff failures at
`90/100 s` into passes and thereby create the exact three-node corridor
`80/90/100 s`.

## 2. Locked equation and units

Replace only the recovery coefficient in the R1/R2 reserve equation:

\[
r_{rec}(m)=\frac{1-m}{\tau_{slow}}+\frac{m}{\tau_{fast}},
\]

\[
\dot q=r_{rec}(m)(q_0-q)
-\frac{U}{\tau_D}(q-q_{res}).
\]

The existing M state is dimensionless and satisfies `0<=m<=1`. The additive
current coordinate remains

\[
A=A_{max}m,\qquad A_{max}=1.6\ {\rm mV}.
\]

Code must use dimensionless `m` in `r_rec`. A value read from an artifact in
mV must first be divided by `Amax`; mV-valued A must never be inserted directly
into a recovery rate.

No threshold, Hill exponent, coupling gain, or additional state is introduced.

## 3. Nullcline and local stability prediction

For frozen mean use `Ubar` and frozen m, define

\[
a(m)=r_{rec}(m),\qquad b=\bar U/\tau_D.
\]

The unique q-nullcline is

\[
q^*(m,\bar U)=
\frac{a(m)q_0+bq_{res}}{a(m)+b}.
\]

For `tau_fast<tau_slow`,

\[
\frac{\partial q^*}{\partial m}=
\left(\frac1{\tau_{fast}}-\frac1{\tau_{slow}}\right)
\frac{b(q_0-q_{res})}{[a(m)+b]^2}>0.
\]

At fixed occupancy the q eigenvalue is

\[
\lambda_q=-[a(m)+b]<0.
\]

The q-M slow block is triangular both while the active latch raises M and
after reset releases it; its diagonal terms remain negative. R3 therefore
does not preregister a Hopf, torus, second q fixed point, or autonomous smooth
limit cycle. Its intended role is a hybrid relaxation-path correction: the
fast subsystem supplies the bounded CCO inner orbit, additive M crosses the
fast exit fold, and frozen post-exit M accelerates q recovery before the
state-defined latch reset.

## 4. Independent scope

R3 may read the accepted P3/R0b/R2 artifacts and may regenerate a fixed-q
occupancy/M ramp sensor. It must not modify:

- E-E weights, kernels, delays, conductance, or `rec_sat_g`;
- relay variables `y/x`;
- the P3 geometry or fixed bath-resource mask;
- M build/release constants (`225 ms`, `12 s`) or latch thresholds;
- the event schedule, periodic CCO sensor, q-hold axis, q-reserve mapping, or
  R0b fast folds.

The fixed bath mask remains an imposed boundary and cannot support an
emergent containment claim.

## 5. Registered axes and upstream parity

The R3 continuation is fixed as

```text
tau_slow = [70, 80, 90, 100, 120, 160] s
q_hold   = [.8400, .8425, .8450]
tau_fast primary     = 20 s
tau_fast sensitivity = [15, 25] s
```

For every `(tau_slow,q_hold)` cell, reuse the exact R2 `q_res/tau_D` mapping.
No parameter is refit after M gating is introduced.

Before entry, set `m=0`. The R3 producer must hash-lock and reproduce the R2
event, periodic, and schedule classifications. In particular:

- `70 s` remains an entry-margin failure because `q_hold=.845` fails;
- `80/90/100 s` retain event-6-first entry and the registered schedule;
- `120/160 s` retain the sparse-schedule failure;
- the complete phase x source-dt periodic orbit remains the R2 orbit.

Any change to those labels is a fail-closed implementation error, not a
candidate rescue.

## 6. Fixed-q M-ramp sensor

The path oracle must not jump M from zero to its endpoint. Regenerate the
accepted R0b fixed-q P3 ramp for all

```text
3 q_hold x 4 phase x 2 dt = 24 records.
```

Each record uses an established bounded CCO checkpoint, `m=0`, active regional
latch, `enable_z=false`, the unchanged 225-ms occupancy-gated M law, and no
external pulse. Save at least `m(t)`, regional `U(t)`, occupancy, regional fast
rate, support/bound/finite flags, and return times.

The sensor-generation gate requires:

- the exact 24-cell Cartesian product with no duplicates;
- source CCO and ramp labels match R0b at base/half dt;
- no support, bound, or nonfinite failure;
- M begins at zero and is nondecreasing until it freezes;
- both regional rates enter the registered low region and remain there for at
  least `50 ms`;
- fixed q error is at numerical precision;
- final A agrees with the accepted R0b ramp table within `.002 mV`.

This is still a frozen-q sensor, not a q-coupled lifecycle simulation.

## 7. Four-segment scalar/path oracle

### 7.1 Pre-entry parity

Use the exact R2 event/periodic/schedule records with `m=0`. No R3-specific
parameter may alter this segment.

### 7.2 Ictal M-ramp replay

Start `q=q_hold,m=0` at each fixed-q CCO checkpoint. Replay the measured
`U(t),m(t)` sensor and integrate the R3 q equation with exact piecewise-linear
or sufficiently fine (`<=1 ms`) reporting steps until the sensor reaches a
50-ms sustained regional low state.

Require:

- q stays finite, monotone only as dictated by the replay, and within the
  observed R0b fold-interpolation domain;
- maximum `|q-q_hold| <= .00125` during this short feed-forward segment;
- the additive coordinate at low-state entry exceeds the interpolated R0b
  low fold at the dynamic q by at least `.020 mV`;
- the dynamic q remains below the entry fold during the active CCO portion;
- all primary and sensitivity arms use the same measured sensor.

The small q-excursion gate is what makes a frozen-sensor replay a legitimate
necessary-condition test; failure means a coupled run is required before any
claim and R3 remains closed.

### 7.3 Protected low-state recovery

After sensor-defined exit, set `U=0` and retain the implemented latch semantics:
M is frozen while the latch remains active. Advance q with the constant
`r_rec(m_exit)` until both q reaches `.885` and the conservative persistence
bound

\[
t_{p,off}=.75s\log(1/.03)
\]

has elapsed. Require reset within `120 s` and require

\[
A_{exit}\ge A_{fold}(q)+.020\ {\rm mV}

throughout the portion of the path within the registered fold domain.

### 7.4 Post-reset release

After the state-defined reset,

\[
m(t)=m_{reset}\exp[-(t-t_{reset})/12s].

Integrate q with the resulting time-varying `r_rec(m)`. Require q to remain
finite and nondecreasing, A to reach `.020 mV`, and the final state to remain
on the geometrically protected low side. Report q and time at A release.

## 8. Causal controls

The oracle must keep four mechanism labels separate.

1. `additive on / recovery gate off`: set `tau_fast=tau_slow`; its analytic
   handoff must reproduce the R2 constant-recovery result.
2. `additive off / recovery gate on`: use the CCO nullcline upper bound for
   `m=1`; q must remain below the entry fold, so recovery gating alone does not
   count as an exit mechanism.
3. `additive on / recovery gate on`: the registered R3 mechanism.
4. `slow off`: upstream returning-event behavior is inherited and unchanged;
   R3 cannot relabel it as seizure prevention.

The canonical report must state that additive M supplies fast exit and M-gated
q recovery supplies timely reset. They are not interchangeable causal effects.

## 9. Acceptance and stop rules

R3 scalar/path support requires all of the following:

1. all upstream hashes and R2 provenance pass;
2. the complete 24-cell M-ramp sensor passes its numerical/biophysical gates;
3. pre-entry event, periodic, and schedule parity is exact;
4. at primary `tau_fast=20 s`, every q-hold/phase/dt path passes at
   `tau_slow=80,90,100 s`;
5. the same three-node corridor passes fixed, no-refit `tau_fast=15/25 s`;
6. `70 s` remains rejected by entry robustness and `120/160 s` remain rejected
   by schedule selectivity; no failure boundary is relaxed;
7. gate-off reproduces R2, and gate-only cannot cross the fast exit boundary;
8. the accepted component contains exactly the registered `80/90/100 s`
   corridor and has at least three consecutive nodes.

If supported, unlock only a short regional P3 state-fork at
`tau_slow=[80,90,100] s`, all three q-hold values, and base/half dt. That arm
must still test at least four pulse-free returns, clean finite exit, true latch
reset, same-basin recovery, and early/late retrigger.

Stop and proceed to a biologically separated two-pool inhibitory-resource
design if any of the following occurs:

- only one tau-slow node passes;
- only `tau_fast=15 s` passes;
- the M ramp must be endpoint-jumped or refit;
- q moves by more than `.00125` during the frozen-sensor segment;
- recovery gating alone is treated as termination;
- a coupled pilot later shortens activity below four returns;
- any E-E/conductance/relay change is required.

No R3 scalar result unlocks continuous field, full SNN, spatial containment,
wavefront annihilation, or the three downstream workflows.

## 10. Resource and artifact contract

Use one process and one BLAS thread. Run base and half dt sequentially. Peak
RSS must remain below `1.5 GiB`; estimate trace bytes before allocation and
stop above `256 MiB` per sensor batch. Write CSV/NPZ and strict JSON
(`allow_nan=False`) before plotting. The figure directory must contain a
Chinese `README.md`.

The primary figure should retain the established 2x3 mechanism layout:

- A: nullcline shift with M;
- B: inherited entry/schedule corridor;
- C: measured M-ramp and q excursion;
- D: tau-slow x q-hold path acceptance;
- E: protected reset/release trajectories;
- F: gate verdict and claim boundary.
