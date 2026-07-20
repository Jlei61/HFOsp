# Topic 4 MZ inhibitory-reserve recovery-timescale corridor (locked design v1.0)

Date: 2026-07-20

Branch: `codex/topic4-mz-divisive-lifecycle`

Status: locked before the canonical R2 run. A read-only pilot exposed the candidate interval; the pilot is not an accepted result.

## 1. Why this node exists

R0b established a fixed-q exit corridor on `q=.835-.845`. R1 then fixed
`tau_r=20 s`, solved `q_res` and `tau_d` from the bounded CCO and locked event
endpoint, and found that the fifth event crossed the entry fold before the
sixth event. Exact periodic-q analysis showed that the same law can stably
hold the CCO, so the failure is in the event map rather than the CCO
nullcline.

The locked event intervals are irregular. In particular, event 5 to event 6
has the longest gap (`3.384 s`). With `tau_r=20 s`, recovery during this gap is
large enough that event 5 is deeper than event 6. A read-only pilot showed
that slowing recovery can reverse this ordering without adding another
current, threshold, E-E term, or state variable. This is a direct correction
to the previous design assumption and must be tested before a more complex
eligibility mechanism.

## 2. Unchanged equation and phase-plane effect

The equation remains

\[
\dot q={q_0-q\over\tau_r}
-{(q-q_{res})U\over\tau_d}.
\]

For frozen mean use `Ubar`, the q-nullcline is

\[
q^*={q_0/\tau_r+q_{res}\bar U/\tau_d
\over 1/\tau_r+\bar U/\tau_d}.
\]

Changing `tau_r` does not create a second q fixed point or a Hopf bifurcation.
After `q_res` is remapped, the desired CCO fixed point remains `q_hold`, and
the q-direction multiplier must remain contractive. What changes is the
finite event-to-event map: a longer `tau_r` reduces recovery during the last
long gap, allowing the sixth event to be the first entry event.

The possible full lifecycle is therefore a relaxation loop, not a new
two-variable Hopf claim: the fast subsystem supplies the low branch and
bounded CCO; q crosses the entry fold; additive M crosses the exit fold; after
exit, q recovery and M decay must hand the trajectory back to the low branch.

## 3. Independent scope

This node changes only the inhibitory-reserve recovery timescale and remaps
its two scalar parameters. It does not change:

- E-E weights, kernels, delays, conductance, or `rec_sat_g`;
- relay variables `y/x`;
- the P3 mass-balanced geometry;
- the additive M equation or `tau_m_down=12 s`;
- the fixed bath-resource mask used by the current regional oracle.

It is independent of `codex/topic4-mz-conductance` and cannot establish
emergent spatial containment.

## 4. Pilot-informed registered axis

The canonical one-dimensional continuation is fixed as

```text
tau_r = [20, 40, 50, 60, 70, 80, 90, 100, 120, 160] s
q_hold = [.8400, .8425, .8450]
preferred tau_r = 80 s
preferred q_hold = .8425
```

The axis includes the original 20-s no-go, the predicted lower entry boundary,
the predicted upper postictal-handoff boundary, and two stress nodes. It is a
mechanism-discovery continuation, not parameter identification.

For every `(tau_r,q_hold)` cell:

1. use the hash-locked complete CCO sensor and locked six-event sensor;
2. solve `q_res` from the exact periodic CCO mean at the primary
   `(phase=0,dt=.125 ms)` record;
3. solve `tau_d` only from the unchanged final target `q=.855`;
4. do not tune either parameter against the pre-last gate or handoff gate;
5. reject nonphysical, nonunique, or nonmonotone roots.

## 5. Entry and hold gates

A cell passes entry only if exact ZOH integration gives:

- no fold crossing before event 6;
- `min(q before event 6) >= q_entry + .00125`;
- event 6 crosses `q_entry=.8558315843088748`;
- the post-event minimum reaches the unchanged `.855` target;
- base and half scalar subdivision labels agree.

For every mapped cell, all four phases and both source dt values must also
satisfy:

- `q_min>=.8325`, `q_max<=.8500`;
- absolute periodic-mean error `<=.00125`;
- q-direction per-return multiplier `<.90`;
- exact periodic closure error `<=1e-10`.

At the preregistered primary `tau_r=80 s,q_hold=.8425`, keep its solved
`q_res,tau_d` fixed and evaluate `tau_r=72/88 s`. Both sensitivity arms must:

- retain event 6 as the first fold crossing;
- retain the pre-last `+.00125` margin;
- reach at least `q_entry-.0005` after event 6;
- keep the periodic orbit inside the same q bounds with mean error `<=.00125`.

The fixed-parameter sensitivity is not required to hit the central fitted
`.855` endpoint exactly; requiring the calibration equality after a parameter
perturbation would be a mathematically inappropriate robustness gate.

The primary q-hold node also replays the already locked schedule probes at
every tau node. Accepted tau nodes require base/half agreement, isolated and
six-event sparse (`3.4 s`) schedules to remain outside the entry fold, the
six-event dense (`1.2 s`) schedule to enter, and held-out schedules never to
enter before event 4. Held-out entry/no-entry labels are reported rather than
forced to contain an artificial mixture.

`q_res` remains a parameter floor, not a safety boundary. Safety is judged
from the actual periodic trajectory.

## 6. Postictal q-M handoff predictor

This node adds a fail-closed analytic predictor; it is not yet a coupled-fast
proof. The existing autonomous implementation is a state-defined hybrid latch,
not an immediate M decay. While the regional latch remains active and joint
occupancy is low, `dm/dt=0`; the additive exit variable is frozen. The latch
resets only when regional rates are low, persistence is off, and `q>=.885`.

The predictor must therefore preserve the implemented two-stage path.

### 6.1 Protected recovery before latch reset

After the confirmed frozen-q M ramp reaches the low branch, set `U=0` and use

\[
q(t)=q_0-(q_0-q_{start})e^{-t/\tau_r},
\qquad A(t)=A_0.
\]

For each q-hold node:

- `q_start` is the minimum exact periodic q, which is more conservative than
  starting at its mean;
- `A_0` is the minimum final additive value across the accepted R0b smooth
  ramps at that q;
- `A_fold(q)` is the piecewise-linear interpolation of the hash-locked R0b
  low-root folds, augmented by `(q_entry,0)`;
- until the state-defined reset threshold `q_safe=.885`, require
  `A(t) >= A_fold(q(t)) + .025 mV` at 1-ms reporting resolution;
- use the conservative persistence bound
  `t_poff=.75 s * log(1/.03)` from `p_start=1`, and set
  `t_reset=max(t_qsafe,t_poff)`;
- require `t_reset<=120 s` for an accepted cell. This horizon is inherited from
  the existing Topic 4 recovery/retrigger protocol, not selected from the R2
  outcome;
- also report the zero-margin result and times to recover above `q_entry`,
  `.885`, and `.895`.

### 6.2 M release after latch reset

After reset,

\[
A(t)=A_0e^{-(t-t_{reset})/12s},
\]

while q continues to recover. At reset, q is already at least `.885`, which is
`.02917` above the A=0 entry fold. Report time to `A<=.02 mV` and q at that
time, and require q never to decrease. This stage is geometrically protected
by the restored low branch; it must not be evaluated as if M had decayed
during the pre-reset interval.

The `.025 mV` margin, `.885` reset threshold, `.75 s` persistence timescale,
`.03` off threshold, and `12 s` M decay are inherited from the existing
implementation. Passing this predictor only licenses a short coupled
state-fork, because interpolation between frozen folds and assumed low
occupancy are not a full dynamic proof.

## 7. Mechanism-level acceptance and stop rules

R2 is supported only if:

1. every registered cell resolves without a numeric error and every found root
   is unique, monotone, and physical;
2. at least three consecutive `tau_r` nodes pass both entry and handoff for all
   three q-hold nodes;
3. the accepted component contains the preregistered `80 s` node;
4. all accepted cells pass the complete periodic-q oracle and base/half event
   replay, and the primary q-hold schedule contract passes;
5. no thresholded-eligibility parameter is used.

If supported, unlock only a short P3 regional coupled arm at
`tau_r=[60,70,80] s`, with unchanged M parameters, base/half dt, and the fixed
bath mask explicitly labelled non-emergent. If not supported, do not tune a
new tau axis; proceed to a genuinely two-pool inhibitory resource design.

No R2 outcome by itself unlocks continuous field, full SNN, spatial
containment, autonomous reset/retrigger, or the three downstream workflows.

## 8. Resource contract

The scalar continuation must run in one process and one BLAS thread, with peak
RSS below 1 GiB. Intermediate CSV/NPZ files must be written before plotting,
strict JSON must use `allow_nan=False`, and every figure directory must include
a Chinese `README.md`.
