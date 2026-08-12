# Topic 4 rev10-D: local adaptation and dynamic route accessibility

## 1. Scientific question

rev10-R2 found no shared mode-A solution in a finite library of static,
observation-invariant continuous E-to-E edge redistributions. The next question
is narrower than fitting another spatial field:

> Can activity history on one frozen Node/connectivity substrate transiently
> suppress the route used by the last event and make the alternate patient-like
> propagation mode accessible within the same network?

This is an exploratory interictal-repertoire experiment. It does not test an
ictal onset/carrier/termination/recovery cycle.

## 2. Frozen substrate

- Node threshold field: rev10-SA V6.2 `v62_density_t050`.
- E-to-E edge map: exact no-op; topology, delays, AMPA/GABA weights and incoming
  budgets are unchanged.
- Network and noise construction, virtual contacts, common absolute detector,
  shaft-aware patient target and returned-only scoring are inherited unchanged
  from rev10-R2.1.
- No Gaussian component, contact coordinate, shaft label, patient event or
  patient mode label enters the dynamic mechanism.

## 3. Dynamic mechanism

For each excitatory neuron, define a non-negative adaptation current `a_i`:

```text
a_i(t + dt) = exp(-dt / tau_a) a_i(t) + q_a S_i(t)
I_net,i(t)  = I_E,i(t) - I_I,i(t) - a_i(t)
```

`S_i(t)` is the E-neuron spike indicator. The frozen heterogeneous threshold
vector is passed through exactly; adaptation does not rewrite `Vtheta`.
Inhibitory neurons receive no adaptation current.

This mechanism is spatial only through endogenous firing history. It has no
learned or hand-placed spatial basis.

## 4. Controls

Three arm types are required.

1. `off`: exact rev10-R2 Node/no-edge baseline with no slow object.
2. `local`: each E neuron receives its own `a_i`.
3. `global`: one scalar state receives `q_a * n_spikes_E / N_E` per step and is
   applied uniformly to all E neurons.

For an imposed identical spike train, local and global arms have the same
population-mean adaptation current at every step. Their difference is whether
the state retains neuron identity. The global arm therefore controls for a
generic activity-dependent brake without local route memory.

## 5. Frozen exploratory grid

The no-op rev10-R2 returned-event onset gaps have a median near `0.5 s`, with
event durations mainly `31-101 ms`. Before reading any rev10-D network, freeze:

```text
tau_a in {250, 750, 2000} ms
q_a   in {0.10, 0.25, 0.50} mV-equivalent per E spike
mode  in {local, global}
```

Together with `off`, this gives 19 candidates. The grid brackets event duration,
typical inter-event interval and a slower carry-over regime. It is a mechanism
screen, not a continuous optimizer.

## 6. Networks and scoring

- Canary/fit networks: `1081-1083`, common random numbers across candidates.
- A later confirmation, only after a candidate is frozen: `1091-1093`.
- Network seed is the independent unit; pooled event counts are descriptive.
- Only `event_returned=True` events enter shaft-aware A/B support and shape
  scores.
- KMeans is descriptive and cannot replace supervised A/B support.

The existing equal-network objective remains unchanged. Additional mechanism
readouts are local/global adaptation mean, spatial standard deviation and
maximum over time.

## 7. Exploratory interpretation

The canary is considered informative for local route memory when a local arm:

- produces returned, joint, in-distribution A and B events in at least `2/3`
  networks;
- preserves mode B in at least `2/3` networks;
- exceeds both its same-parameter global control and exact-off baseline in
  same-network A/B support; and
- does not produce runaway activity.

These are interpretation rules, not a large blocker stack. If no local arm
meets them, report `LOCAL_ADAPTATION_ROUTE_ACCESS_NOT_OBSERVED` and do not tune
`beta` or compare optimizers. If local and global improve equally, the result is
a generic activity-brake effect, not evidence for route-specific memory.

## 8. Confirmation and figures

If an informative local candidate exists, freeze it before reading `1091-1093`,
then compare `off`, the selected `local` arm and its matched `global` control.
The confirmation output must use the accepted Fig.4 pair:

1. direct same-network A/B propagation and model-current readout;
2. returned-only KMeans heatmap/profile with supervised support shown separately.

Without a new patient-blind unit, all patient comparisons remain development
only.

## 9. D1 outcome

All `19 x 3 = 57` workers completed on seeds 1081-1083 without runaway. Every
arm, including exact off, had formal mode-A support in `0/3` networks. Exact
off yielded 22 returned events; increasing adaptation generally reduced event
yield, reaching 2-5 returned events in the strongest slow arms. The diagnostic
best was the weakest global brake (`tau=250 ms`, `q=0.10 mV`) at `11.327`, only
`0.021` below exact off (`11.348`). No local arm exceeded both its matched
global control and off in same-network A/B support.

The frozen verdict is `REV10D_LOCAL_ADAPTATION_ROUTE_ACCESS_NOT_OBSERVED`.
This rules out the tested negative-feedback grid as a mode-A access mechanism;
it does not justify optimizer comparison or a weaker interpolation sweep. A
next canary must be able to increase route accessibility, not only suppress
recently active neurons.
