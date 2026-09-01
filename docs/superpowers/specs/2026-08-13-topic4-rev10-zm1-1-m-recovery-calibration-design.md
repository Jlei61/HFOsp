# Topic 4 rev10-ZM1.1: learned h + spatial OU + Z/M recovery calibration

## Scientific question

The direct transfer of `tau_adp=500 ms` engaged both Z and M and improved the
supervised mode shapes, but 3/6 networks entered runaway after 6.4-7.6 s. This
round asks one narrow question: can a longer M recovery time preserve the two
interictal propagation modes without late runaway on the frozen data-driven
substrate?

This is a development calibration, not a patient-blind test, a seizure
lifecycle experiment, or evidence that the learned field is a unique core.

## Frozen equations and substrate

For excitatory neurons,

```text
tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k)
I_net,i = I_i^E - z_i I_i^I - eta_m m_i
V_th,i = V_th,0 - h_i d_i
```

The learned `h`, spatial OU process, topology, delays, detector, contact
readout, `I_th_EI`, `tau_z`, and `eta_m` remain frozen. Static edge
coefficients remain exactly zero. The only varied quantity is:

```text
tau_adp in {500, 1000, 1500, 2000} ms
```

The values bracket the previous transition: 500 ms was unstable, 1000 ms was
mixed on the earlier substrate, and 2000 ms was bounded. The 1500 ms point
localizes the transition without opening a second parameter.

## Network split and duration

- fit: seeds 1361-1363
- selection: seeds 1371-1374
- confirmation: seeds 1381-1386
- duration: 20 s in every phase

No seed appears in more than one phase. Twenty seconds is required because the
earlier `tau_adp=1000 ms` candidate could fail only at 15.4 s; an 8 s screen
would misclassify delayed runaway as stability.

## Advancement rule

There is no composite selection score. Runaway is the only safety exclusion.
Among safe, evaluable candidates, report the Pareto surface over:

- maximize natural-KMeans signed TA/TB geometry margin;
- maximize natural-KMeans direction purity;
- minimize worst supervised mode-shape distance;
- minimize OOD fraction.

At most two fit candidates advance. Independent selection networks choose one
candidate lexicographically in the order above. Confirmation never changes the
candidate.

## Confirmation and figure contract

Confirmation must regenerate the two frozen data-driven SNN figures:

1. continuous `h` landscape, Model TA/TB spatial modes, and a same-network
   all-contact 30-80 Hz readout;
2. Figure 1E-style masked-rank KMeans heatmap, rank distribution, MTA/MTB vs
   patient TA/TB profiles, and the 2x2 cluster-patient matrix.

Scientific acceptance reports, without replacing them by one score:

- no runaway;
- same-network TA and TB availability;
- returned events;
- natural KMeans evaluability and patient-matched purity benchmark;
- positive diagonal and negative crossed patient geometry;
- nonzero Z and M dynamic participation.

Engineering completion or a visually valid figure does not imply scientific
acceptance.
