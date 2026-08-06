# Topic 4 MZ actual-entry-aligned lifecycle closure（R4 locked design v1.0）

Date: 2026-07-21

Branch: `codex/topic4-mz-divisive-lifecycle`

Status: locked after the R3 coupled center canary and before any R4 execution.

## 1. Scientific question and non-overlap boundary

The R3 scalar/path oracle showed that M-gated recovery can separate slow
preictal memory from faster postictal reset. Closing `q -> fast activity ->
inhibitory use -> q`, however, moved the first fold crossing from the
preregistered event 6 to event 5. The formal R3 canary therefore remains

```text
R3_COUPLED_CLEAN_NO_GO_PREMATURE_EVENT5_ENTRY
```

That no-go must not be overwritten. It also revealed a separate phenotype:
after event 5, and after excluding the trigger-response cluster, core and
annulus produced four paired autonomous returns before event 6; M then produced
a finite fast exit and suppressed the within-train event-6 challenge.

R4 asks only whether this already observed actual-entry phenotype can close its
outer recovery loop:

```text
event-triggered entry -> bounded recurrent bursting -> finite fast exit
-> q reset -> latch release -> M decay -> same low basin
-> refractory early train / recovered late train
```

R4 does **not** recalibrate `q_res/tau_D`, change the six-event schedule, or
repair event ordering. It does not modify `W_EE`, E-to-E kernels/delays,
recurrent saturation, conductance membrane equations, presynaptic relay
variables, geometry, bath mask, M timescales, or latch thresholds. The parallel
conductance/E-to-E line remains scientifically and mechanically independent.

## 2. Why R4 is admissible without q remapping

Event-5 versus event-6 entry is a susceptibility-calibration result, not the
definition of a lifecycle. The current center trace already contains delayed
cumulative entry, four pulse-free autonomous recurrences, finite low-state
return, and a protected-low checkpoint. Reset, M release, same-basin return,
and recovered susceptibility are orthogonal questions that can be tested from
that checkpoint without fitting any parameter to the newly observed trace.

Recalibrating q first would be post-hoc tuning to restore the old event index
and would not answer whether the slow outer loop closes. A successful R4 would
only unlock a future fully coupled q-map recalibration; it would not validate
the existing scalar corridor in the coupled system.

## 3. Locked Segment A phenotype

Hash-lock the R3 coupled config, summary, and trace. State continuation must use
the saved double-precision `final_state` and `final_latch_state`; reconstructed
plot traces are not state sources.

The source phenotype is accepted for R4 only if all of the following are
recomputed from the saved artifact:

1. events 1--4 remain above the regional entry fold;
2. event 5 is the first fold-crossing event;
3. crossings in the trigger-associated response cluster are excluded through
   `event_5_end + 100 ms`;
4. before event 6, core and annulus have at least four one-to-one paired upward
   crossings of the 20-Hz section, with each pair separated by at most `20 ms`;
5. the fourth paired time is the later of the two regional crossings;
6. after the fourth return, both regional fast traces cross downward through
   20 Hz, then `rE` and `rE_fast` remain below 5 Hz for at least `250 ms`;
7. event 6 creates no 20-Hz upward crossing and remains a **within-train
   immediate post-clonic challenge**, not an early-recovery probe;
8. no transfer-support, state-bound, or nonfinite failure occurs.

The registered source values are diagnostic sentinels, not tolerances to fit:

```text
first fold crossing       7620 ms
paired returns            8833.603, 9544.708, 10177.268, 10811.904 ms
joint last down-crossing  10890.699 ms
sustained all-low onset   10922 ms
latch set                 10233.375 ms
```

Any source-artifact mismatch stops R4 before integration.

## 4. Segment B: protected recovery from the 20-s checkpoint

The locked checkpoint must independently satisfy:

```text
latch = on
regional U = 0
regional occupancy = 0
regional rE and rE_fast < 5 Hz
p <= .03
q < .885
M > 0
numeric state finite and supported
```

While the latch is active and certified `U=occupancy=0`, the slow coordinates
have the exact bridge

\[
m(t)=m_0,
\qquad p(t)=p_0e^{-t/\tau_p},
\]

\[
q(t)=q_0-(q_0-q_{20})e^{-r_{rec}(m_0)t},
\]

\[
r_{rec}(m)={1-m\over\tau_{slow}}+{m\over\tau_{fast}}.
\]

Advance only to `q=.885-epsilon`, with `epsilon=5e-4`. At 25%, 50%, and
75% of the predicted bridge duration, interrupt the analytic bridge with a
`500 ms` full fast--slow, no-pulse sentinel. Each sentinel must retain the low
branch, active latch, zero returns, zero use/occupancy after its settling
margin, and no numeric failure. The analytic state passed into each sentinel
must be the state produced by the previous sentinel, not a fresh copy of the
20-s checkpoint.

Analytic bridging is permitted only between successful zero-use sentinels and
must not allocate a millisecond trace over the skipped interval.

## 5. Segment C: state-machine latch reset

From just below `q=.885`, resume full integration with no pulses. The existing
state machine must create exactly one true-to-false transition, without runner
mutation of the latch or manual reset of q, p, or M. At the transition:

```text
core/annulus rE_fast <= .005 kHz
core/annulus q >= .885
core/annulus p <= .03
```

Reset must occur within `120 s` of the Segment-A termination time. There may be
no reset chatter, latch re-set, section return, support failure, bound failure,
or nonfinite state.

## 6. Segment D: natural M release and same-basin return

After reset and only inside certified zero-use intervals,

\[
m(t)=m_Re^{-t/\tau_M},
\]

and q follows

\[
q_0-q(t)=(q_0-q_R)\exp\left[
-{t\over\tau_s}
-\left({1\over\tau_f}-{1\over\tau_s}\right)
m_R\tau_M(1-e^{-t/\tau_M})
\right].
\]

Run `500 ms` full-fast sentinels at `A=.10,.02,.002 mV`. If `A=.002 mV` is
reached before `q=.899`, continue the analytic zero-use bridge until q reaches
`.899`, then run a final `4 s` no-pulse full integration.

The same-basin checkpoint requires:

- latch off, q at least `.899`, A at most `.002 mV`, and p at most `.001`;
- LLL label, no section return, and regional rates below 5 Hz;
- fast-vector-field norm below `1e-8/ms` at the final state;
- distance of the fast coordinates to the original interictal root decreases
  over the final sentinel;
- q is nondecreasing and M nonincreasing across every analytic/full segment;
- no support, bound, or nonfinite failure.

The slow state may approach its root asymptotically; it may not be assigned by
the runner to `.9/0/0`.

## 7. Common-classifier early and late retrigger

Use two state forks and the exact same relative challenge:

```text
onsets   [1000, 3122, 5044, 6321, 7531, 10915] ms
duration 20 ms
amplitude 3 mV
profile  [1,0,0]
```

- **Early fork**: copy the original 20-s protected-low checkpoint with its
  latch still active and M still elevated.
- **Late fork**: copy the naturally recovered same-basin checkpoint with latch
  off; do not recreate the initial state manually.

Both forks use one classifier. An `actual_entry_lifecycle_candidate` exists
only if, after excluding a trigger response cluster, at least four one-to-one
paired autonomous returns occur, followed by a finite 250-ms low exit.

Acceptance requires:

- early fork: finite and supported, but no lifecycle candidate;
- late fork: events 1--4 remain above the fold, event 5 is first entry, at
  least four paired autonomous returns occur, event 6 produces no section
  crossing, and finite low exit recurs;
- the early result cannot pass merely because it ran away or left transfer
  support;
- early and late use identical thresholds, pairing, response exclusion, and
  numeric gates.

## 8. Numerical replication, stop rules, and resources

Run the entire center closure first at `dt=.125 ms`. Only a full base-dt pass
unlocks an exact `.0625 ms` replicate. The two dt labels must agree on source
phenotype, reset, same basin, early suppression, late recovered lifecycle, and
all stop categories. Continuous crossing times may differ within `20 ms`.

Stop immediately for source mismatch, nonzero protected use/occupancy,
sentinel return, missing or chattering reset, M reactivation during release,
same-basin failure, early lifecycle recurrence, absent/early/runaway late
lifecycle, dt-label disagreement, or any support/bound/nonfinite failure. No
parameter may be changed after a stop.

Use one process and one BLAS thread. Base and half dt run sequentially. Cap each
saved trace at `64 MiB`, peak RSS at `1.5 GiB`, and total base-plus-half wall
time at `20 min`. Analytic gaps store endpoints only. Save full gate tables and
representative NPZ traces; figures require PNG, PDF, and a Chinese README.

## 9. Verdict and downstream boundary

The only supported status is

```text
R4_ACTUAL_ENTRY_REGIONAL_HYBRID_LIFECYCLE_CENTER_SUPPORTED
```

if every base/half-dt gate passes. Any scientifically resolved failure is a
named clean no-go; numeric/resource ambiguity is unresolved, never no-go.

Even a full pass establishes only a center-point, regional,
event-triggered hybrid relaxation lifecycle under an imposed fixed bath mask.
It does not establish zero-input spontaneous seizure onset, a smooth Hopf,
SNIC, torus, or limit cycle of the full slow system, robust q-parameter
corridor, continuous spatial wavefront containment, wavefront annihilation, or
full-SNN seizure dynamics. It unlocks only:

1. fully coupled q-map recalibration and local robustness at this unchanged
   mechanism;
2. if that corridor survives, a coarse continuous spatial field test;
3. only after spatial entry/containment/recovery passes, full SNN transfer and
   the three downstream workflow readouts.
