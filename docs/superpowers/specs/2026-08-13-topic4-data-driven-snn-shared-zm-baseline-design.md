# Topic 4 shared data-driven SNN baseline v1

## Purpose

All future data-driven node-field searches should run on the same mechanistic
engine family used by the Z/M line:

```text
V_th,i = V_th,0 - h_i d_i
tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k)
I_net,i = I_i^E - z_i I_i^I - eta_m m_i
```

The shared runtime also retains the frozen local spatial OU process. This
prevents the free-field and slow-variable lines from silently using different
SNN equations.

## Baseline identity

The canonical contract is:

```text
config/topic4_data_driven_snn_baseline_zm_v1.json
baseline_id = data_driven_snn_h_spatial_ou_zm_reference_v1
```

Consumers must hash-lock this file and select one runtime explicitly:

- `active_z_plus_m`: fixed transferred Z/M reference while optimizing `h`;
- `paired_slow_off`: exact slow-off comparator with the same `h`, network, and
  OU realization.

There is deliberately no implicit default.

## Scientific boundary

`active_z_plus_m` is a mechanism reference, not an accepted stable baseline.
On the current warm field, every tested `tau_adp` in 500-2000 ms ran away on
3/3 fresh 20 s networks. Therefore:

- a new free-field search may ask whether changing `h` stabilizes the fixed
  Z/M reference and recovers the patient modes;
- it may not describe Z/M as previously validated or stable;
- every candidate requires at least 20 s simulation and late runaway remains
  invalid;
- a positive result still requires natural KMeans and the two canonical
  Figure 4 panels, not only supervised labels.

## Compatibility policy

The running D6.3 replication remains an immutable slow-off historical
experiment because it was frozen before this baseline existed and explicitly
forbids slow variables. The updated baseline applies to the next free-field
round. This preserves D6.3 provenance and supplies a direct old-base versus
Z/M-base comparison rather than rewriting an experiment in flight.
