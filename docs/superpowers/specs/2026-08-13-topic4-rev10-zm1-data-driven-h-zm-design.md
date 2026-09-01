# Topic 4 rev10-ZM1 data-driven h + Z/M consistency design

## Scientific question

The current data-driven SNN uses a learned continuous node field `h`, a frozen
current-based spatial E/I LIF network, and a local spatial OU accessibility
process. Its fast system is shared with the MZ line, but its slow protocol is
off. ZM1 asks whether adding the already frozen excitatory-cell Z/M equations
to this learned substrate preserves, improves, or destroys the interictal
TA/TB repertoire measured by the canonical Fig.4 readout.

This is a mechanism-transfer experiment. It does not re-fit `h`, Z, or M; it
does not claim seizure generation, recovery, or patient-blind generalization.

## Frozen equations

For E cells only:

```text
tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k)
I_net,i = I_i^E - z_i I_i^I - eta_m m_i
```

I cells retain `I_E - I_I`. The learned heterogeneous threshold vector is
passed through unchanged:

```text
V_th,i = V_th,0 - h_i d_i
```

The transferred candidate is `zA_q75_tz5000__mA0p001_tau500`:

```text
I_th_EI = 95.19851312666987
tau_z = 5000 ms
tau_adp = 500 ms
eta_m = 0.007451594355587098
```

These values were frozen on the earlier hand-placed MZ substrate. Their use
here is deliberately a transfer test, not a recalibration. Realized threshold
occupancy and Z/M trajectories must therefore be reported.

## Paired arms

Each fresh network seed is run with common random numbers in two arms:

1. `h_spou_slow_off`: learned `h` + frozen local OU + `slow=None`.
2. `h_spou_zm_transfer`: the same `h`, network, OU seed, and detector + active
   Z/M.

Topology, weights, delays, external-drive law, simulation duration, contact
montage, event detector, returned-event rule, TA/TB classifier, and Fig.4C
masked-rank KMeans contract are identical. Edge coefficients remain exactly
zero. No Z-only or M-only arm is added in this first exploratory supplement.

## Readouts

The existing equal-network summary is retained: detected and returned events,
same-network TA/TB support, OOD, detector occupancy, patient prototype matrix,
and canonical masked-rank KMeans. ZM1 additionally records mean/minimum Z,
mean/maximum M, mean adaptation current, the fraction of E cells above the
transferred Z threshold, and operational runaway/return.

The result is descriptive. A numerical improvement does not validate a
seizure lifecycle; a failure closes only direct transfer of this frozen Z/M
candidate onto this frozen learned substrate.

## Engineering contract

- Both-off Z/M must be byte-identical to `slow=None` under common random
  numbers.
- Z/M acts on E cells only and does not alter the learned threshold field.
- State is integrated at 0.1 ms; audit traces are sampled every 1 ms.
- Workers run through `systemd-run --user -> nohup`, one numerical thread each,
  with a measured-RSS sentinel, cgroup memory bounds, atomic outputs, status
  sentinels, and desktop notification on completion.
