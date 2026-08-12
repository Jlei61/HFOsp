# Topic 4 rev10-D2: continuous inhibitory-resource accessibility

## Question

The D1 adaptation canary only removed excitability and produced mode A in
`0/3` networks for every arm. D2 asks whether an activity-dependent, continuous
local reduction of inhibitory efficacy can transiently open an otherwise
inaccessible route on the same frozen Node/connectivity substrate.

## Mechanism

Use the existing q-only inhibitory-resource equation on a uniform sheet grid:

```text
tau_r dr_E/I / dt = spike_field_E/I - r_E/I
f_q(x,t) = sat(K_sigma * (eta_E r_E + eta_I r_I))
dq_I(x,t)/dt = (1-q_I)/tau_q - k_q f_q q_I
I_net,E = I_E - q_I(x_i,t) I_I
```

`q_I=1` is the exact off state. Lower `q_I` is local disinhibition. The field is
driven only by simulated E/I spikes and neuron positions. It has no contact,
shaft, patient-label, Gaussian-component or discrete-core input.

## Paired control

- `local`: evolve q on the 32 x 32 sheet and sample it at each E neuron.
- `global`: use the spatial mean of the same `f_q` field to evolve one scalar q
  applied to every E neuron.
- `off`: no slow object.

Local and global receive the same mean depletion drive for an identical spike
field. Their difference is whether inhibitory history retains spatial identity.

## Frozen canary

Fix `tau_q=750 ms`, `sigma_q=1.5 mm`, `sigma_r=0.5 mm`, `tau_r=100 ms`,
`q_min=0.5`, `eta_E=0.3`, `eta_I=1.0`, and scan only:

```text
k_q in {0.01, 0.03, 0.10} ms^-1
mode in {local, global}
```

The initial D2 preflight at `k_q=0.01/ms` reached `q_min=0.5` and entered
runaway at `2.20 s`; this is `D2_DOSE_PARAMETERIZATION_FAIL`, not a mechanism
result. D2.1 therefore freezes a lower bracket before new networks:

```text
k_q in {0.0005, 0.0015, 0.004} ms^-1
mode in {local, global}
```

The slow field updates every `1 ms` while the SNN remains at `0.1 ms`. A
deterministic reference test requires the resulting q field to stay within
`2e-4` of the 0.1-ms slow-field update under the same spike history.

Together with exact off this is 7 candidates x 3 fresh networks (`1141-1143`).
The common detector, returned-only shaft-aware objective and frozen direction
classifier remain unchanged.

## Interpretation

An exploratory local-access signal requires a local arm to produce returned,
joint, in-distribution A and B in at least `2/3` networks, preserve B in at
least `2/3`, exceed both its matched global control and off in same-network A/B
support, and avoid runaway. This is not patient-blind validation or an ictal
lifecycle test.

## D2.1 outcome and D2.2 boundary extension

D2.1 completed `7 x 3 = 21` workers without runaway. Exact off and both lower
local doses produced mode A in `0/3` networks. The upper local boundary
`k_q=0.004/ms` produced one returned, joint, in-distribution A event on seed
1141 and mode B on all three networks; matched global and exact off produced A
in `0/3`. Its equal-network score was `10.245` versus `11.417` for off, but
same-network A/B support was only `1/3`, so the formal verdict remained
`REV10D2_LOCAL_INHIBITORY_RESOURCE_ACCESS_NOT_OBSERVED`.

Because the only local-specific A appeared at the upper tested boundary while
`q_min` remained `0.861-0.909` with no runaway, one final fixed dose extension
is eligible. D2.2 freezes, before fresh networks 1171-1173:

```text
k_q in {0.0055, 0.0070, 0.0085} ms^-1
mode in {local, global}
```

The same gate and controls apply. No interpolation or optimizer follows if
D2.2 fails.
