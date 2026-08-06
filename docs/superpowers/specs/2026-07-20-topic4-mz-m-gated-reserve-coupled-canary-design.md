# Topic 4 MZ M-gated reserve coupled P3 canary（conditional locked design v1.0）

Date: 2026-07-20

Branch: `codex/topic4-mz-divisive-lifecycle`

Status: conditionally locked before the R3 scalar result. This node may run
only if R3 reports its exact registered `80/90/100 s` corridor as supported.

## 1. Purpose and predicted failure

R3 scalar/path support is only a necessary condition. It tests entry parity,
the fixed-q M ramp, and post-exit recovery, but it does not test whether the
same M ramp terminates the coupled activity too early.

The existing base-dt regional trace gives a preregistered timing sentinel:

```text
pulse-free returns 1/2: 11.988 / 12.547 s
latch set:             12.612625 s
pulse-free returns 3/4: 13.036996 / 13.889377 s
```

In the old 225-ms arm, A reaches the R0b q=.840 fold near `13.079 s`, only
about `42 ms` after return 3 and about `810 ms` before return 4. Holding q near
the R0b corridor lowers the required A compared with the old continuing-Z
depletion path. Therefore the explicit prediction is that the coupled R3
canary may cleanly exit after only three returns.

This is a hard falsification gate. If the canary has fewer than four
pulse-free regional returns, record a clean no-go and do not tune M or run the
remaining grid.

## 2. Minimal backward-compatible equation change

Retain the existing `10P+2` state layout and use the current `z` slot as q.
Extend `RegionalSlowParameters` with default-off reserve fields:

```text
q_reserve: float | None = None
tau_z_fast_recovery_ms: float | None = None
enable_m_gated_z_recovery: bool = false
```

When `q_reserve is None`, execute the old equation exactly:

\[
\dot z=(z_0-z)/\tau_r-Uz/\tau_D.
\]

Only registered R3 arms use

\[
r_{rec}(m_R)=(1-m_R)/\tau_{slow}+m_R/\tau_{fast},
\]

\[
\dot q=r_{rec}(m_R)(q_0-q)-U(q-q_{res})/\tau_D.
\]

`m_R` is the same area-weighted core/annulus dimensionless M coordinate used
by the pooled regional effector. Bath remains at q=.90 with zero depletion.

Validation must enforce `0<q_res<q0`, `0<tau_fast<tau_slow`, dimensionless
`0<=m<=1`, and paired presence of all reserve fields. Existing tests must show
that default arms preserve the old RHS/integration semantics. The integrator
must return `final_latch_state` explicitly.

No E-E, conductance, kernel, delay, relay, state-layout, geometry, M-timescale,
or latch-threshold change is permitted.

## 3. Conditional arms and execution order

If R3 scalar is supported, the full registered arm would be

```text
tau_slow = [80, 90, 100] s
q_hold   = [.8400, .8425, .8450]
dt       = [.125, .0625] ms
tau_fast = 20 s
```

Every cell reuses the corresponding R2 `q_res/tau_D`, without refitting.

Execution is cheap-first:

1. center canary only: `tau_slow=90 s,q_hold=.8425,dt=.125 ms`;
2. only if it passes every Segment-A gate, run all 9 base-dt arms;
3. only if base dt passes, run the same 9 half-dt arms;
4. base/half dt are never concurrent.

## 4. Segment A: real coupled onset and termination

Start from the original low state and replay the unchanged six background
events. Integrate fast q/p/M dynamics together for at most `20 s`.

The canary passes only if:

- events 1-5 do not cross the regional entry fold;
- event 6 is the first entry event;
- M remains zero and the latch remains off before entry;
- core and annulus each have at least four paired pulse-free returns after the
  final external event;
- a sustained low state begins only after the fourth return and lasts at least
  `250 ms`;
- exit is finite and clean: no transfer-support, state-bound, or nonfinite
  failure;
- the latch is still active at clean exit;
- bath remains only an imposed fixed-mask diagnostic.

If the center canary has fewer than four returns, use status

```text
R3_COUPLED_CLEAN_NO_GO_EARLY_M_EXIT
```

and stop before the other 17 paths. No `tau_m_up`, `p_on`, Amax, or q mapping
adjustment is allowed.

## 5. Segments B-D, only after canary duration passes

### B. Protected recovery

At the clean low checkpoint require regional occupancy and U to be zero,
latch active, and M frozen. Advance q/p analytically to the reset neighborhood.
At 25/50/75% of the recovery interval, run short full-fast sentinels and
require stable low activity, zero returns, and zero support violations.

### C. True latch reset

Resume full integration shortly before the predicted reset. The existing
state machine, not the runner, must create exactly one true-to-false latch
transition after all three conditions hold:

```text
rE_fast <= .005 kHz
q >= .885
p <= .03
```

Reset must occur within `120 s` of exit.

### D. Natural release and same basin

After reset, let M decay with the unchanged `12 s` time constant. Use analytic
q/M advances only inside certified zero-U low intervals and verify short
full-fast sentinels at `A=.10,.02,.002 mV`. Then run a final no-pulse segment.

Require LLL, no section return, regional rates below 5 Hz, final fast RHS below
`1e-8/ms`, and decreasing distance to the original interictal root. Slow
variables may not be manually reset to `.9/0/0`.

## 6. Retrigger and causal controls

Only if Segments A-D pass:

- early retrigger copies the protected-low, latch-active checkpoint and uses
  the identical six-event train; it must not create a pulse-free CCO;
- late retrigger copies the naturally recovered same-basin checkpoint and
  uses the identical train; bounded ictal entry must recover and cannot occur
  before event 4.

Center, base, and half-dt controls:

1. additive on / recovery gate off (`tau_fast=tau_slow`): additive exit may
   remain, but timely reset must revert to the R2 boundary;
2. additive off / recovery gate on (`Amax=0` in the prepared fast scaffold):
   recovery gating alone must not terminate the bounded CCO.

## 7. Stop rules and claim boundary

Stop immediately for fewer than four canary returns, pre-entry M leakage,
event-5 entry, support escape, dt-label disagreement, missing true reset,
unstable analytic-bridge sentinels, failed same-basin recovery, failed
early/late retrigger, or a gate-only exit.

No failure may be rescued by changing E-E, conductance, relay, bath mask,
M timing, persistence thresholds, or the R2 mapping. A clean failure routes to
a biologically separated two-pool inhibitory-resource design.

Even a full pass would establish only a regional, event-triggered hybrid
lifecycle on a fixed bath mask. It would not establish zero-input spontaneous
seizures, Hopf/torus dynamics, continuous spatial recruitment, wavefront
annihilation, or a full-SNN lifecycle.

## 8. Resource contract

Use one process and one BLAS thread. Estimate trace bytes before allocation;
cap each Segment-A trace at `64 MiB` and RSS at `1.5 GiB`. Save full outcome
tables but only representative NPZ traces. The canary is the resource gate:
if its measured extrapolation exceeds `45 min` for the full registered arm,
stop and report rather than launching the grid.
