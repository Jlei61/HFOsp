# Topic 4 MZ thresholded inhibitory-eligibility design

Date: 2026-07-20

Branch: `codex/topic4-mz-divisive-lifecycle`

## 1. Locked problem

R0b established a fixed-q safe exit corridor on `q=.835-.845`. The first reserve law,

\[
\dot q={q_0-q\over \tau_r}-{(q-q_{res})U\over \tau_d},
\]

can hold the bounded CCO at `q_hold=.840/.8425/.845`, but the locked six-event calibration crosses the entry fold after event 5, not event 6. This is a clean entry-ordering no-go, not a numerical failure.

Two tempting repairs are closed before further simulation:

1. `tau_H dH=U-H` with depletion proportional to `H` leaves the averaged q-nullcline unchanged and does not repair the locked ordering over `tau_H=1 ms-10^7 ms`.
2. depletion proportional to `U H` only repairs the ordering with `tau_d<=4.5 ms`, which destroys the registered slow-variable interpretation.

The remaining arm must therefore change when event-driven depletion is eligible, while preserving the R0 fast geometry and the original M exit arm.

## 2. Independent mechanism

The registered scalar law is

\[
\tau_H\dot H=U-H,
\]

\[
g_H(H)={1\over2}\left[1+\tanh\left({H-\theta_H\over w_H}\right)\right],
\]

\[
\dot q={q_0-q\over\tau_r}
-g_H(H){(q-q_{res})U\over\tau_d}.
\]

Interpretation:

- `H` is a slow eligibility trace of regional inhibitory use, not another inhibitory current;
- below threshold, `q` only recovers toward `.90`;
- above threshold, event use can deplete q;
- when `U=0`, depletion is exactly zero even if `H` remains elevated, so post-event recovery remains possible;
- on established CCO, `H` is above threshold and the R0 q-hold geometry is retained.

This line does not change E-E weights, kernels, delays, recurrent conductance, `rec_sat_g`, relay variables, or membrane conductance. It is independent of `codex/topic4-mz-conductance`.

## 3. Parameter contract

Geometry independently fixes:

- `q_hold=.8425`, the registered node nearest the midpoint of `.830` and entry fold `.8558315843`;
- `q0=.90`, `tau_r=20 s`;
- gate width `w_H=.002`;
- locked event seed/onsets and final target `q=.855`.

Only a 3x3 discovery grid is allowed:

\[
\tau_H=[5,10,15]\;s,
\qquad
\theta_H=[.015,.020,.025].
\]

For each cell:

1. solve `q_res` from the exact periodic CCO sensor so the periodic mean is `q_hold`;
2. solve `tau_d` from the locked event final target;
3. do not tune either value against the pre-last entry gate;
4. reject multiple/non-monotone roots rather than choosing a favorable one.

The center `(10 s,.020)` is the preregistered representative; it is not selected after the screen.

## 4. Cheap scalar discovery gates

Every grid cell is evaluated with base/half scalar integration and `theta_H +/- .001` sensitivity.

A cell is discovery-safe only if:

- `tau_d>=100 ms` and `0<q_res<q_hold`;
- exact periodic CCO gives `q_min>=.8325`, `q_max<=.850`, and the q-direction per-cycle multiplier `<.9`; the deliberately slow H multiplier is reported separately and is not misused as a fast-attractor gate;
- before the last locked event, `q>=q_entry+.00125`;
- the last event reaches the unchanged `.855` target;
- base/half labels and both threshold-sensitivity labels agree;
- after a scripted `60 s` `U=0` recovery, both base/half integrations satisfy `q>=.895` and `H<=.001`; directional movement alone is insufficient.

The mechanism-level discovery gate additionally requires at least two edge-adjacent safe grid cells. A single isolated cell is classified as fine tuning.

A monotone scan with no endpoint root inside the registered physical `tau_d=[25,1000] ms` domain is a resolved failed cell, not a global numerical unresolved result. Only an actual evaluation/refinement error makes the mechanism status numerically unresolved. A no-root peripheral cell does not block an otherwise valid primary center plus edge-adjacent safe component.

## 5. Schedule probes

The following probes are fixed before this implementation is run:

- isolated: one sensor event;
- dense: six equal-dose events separated by 1.2 s;
- sparse: six equal-dose events separated by 3.4 s;
- held-out shifted-exponential schedules with seeds `20260721-20260723`, stored as exact onset arrays in config.

Required qualitative contract:

- isolated and sparse do not enter;
- dense enters;
- held-out schedules are reported without parameter changes, have base/half agreement, never enter before event 4, and include at least one entry and one no-entry schedule.

Because the 3x3 grid was informed by a pilot diagnostic, this entire node remains mechanism discovery. It cannot establish parameter identification or a seizure lifecycle.

## 6. Stop and unlock rules

- If fewer than two adjacent cells pass, close thresholded eligibility.
- If only millisecond `tau_d` values pass, close it as a slow-variable mechanism.
- If schedule probes fail, do not change seeds, target, grid, or gate width in this node.
- A passing scalar screen unlocks only one short coupled arm at the registered center plus its two theta neighbors.
- It does not unlock full autonomous runs, M retuning, retrigger, field containment, SNN migration, or any E-E modification.

## 7. Resource contract

The screen is scalar and must run in one process with one BLAS thread. It consumes the already generated fixed-event and cycle sensor artifacts; no SNN or large spatial grid is run. Expected peak memory is below 1 GiB.
