# Topic 4 rev10-D3: source-specific dynamic E->E accessibility

## Question

Static continuous edge redistribution, local nodal adaptation and local
inhibitory-resource modulation did not restore returned patient-supported mode
A in a majority of networks. D3 asks a narrower connectivity question: can a
source-specific dynamic state on existing E->E edges make the alternate route
accessible after recent activity?

This is not a new spatial field fit. The accepted continuous Node field,
topology, weights, delays, detector and shaft-aware patient-training score stay
frozen. All static edge coefficients remain exactly zero.

## Mechanism

For each presynaptic E neuron `s`, retain an availability state

```text
dx_s/dt = (1 - x_s) / tau_D
x_s(t+) = x_s(t-) (1 - U)  when source s spikes
W_EE(t,s) = x_s(t) W_EE(s)
```

Only E->E edges are scaled. E->I, I->E, I->I, topology and delays are
unchanged. The mechanism uses only simulated presynaptic spikes and contains no
contact, shaft, patient-label, Gaussian-component, core-mask or field input.

## Mean-matched control

Both arms retain the same latent per-source `x_s` dynamics under their own
spike history.

- `local`: apply `x_s` to source `s`.
- `global`: apply `mean_s(x_s)` to every E source.
- `off`: preserve the exact pre-STD path.

For an identical spike history, local and global have exactly the same latent
mean resource. Their only application-level difference is source identity.
This avoids interpreting a generic reduction of recurrent gain as route memory.

## Frozen canary

Use a small established-engine-range grid:

```text
U in {0.08, 0.20}
tau_D in {500, 1500} ms
mode in {local, global}
```

Together with exact off this is 9 candidates x 3 fresh network seeds
`1201-1203`. Simulations remain 8 s, use the common absolute detector, score
returned events only and weight network seeds equally. Long execution must use
`systemd-run --user -> nohup`, a measured-RSS sentinel, one numeric thread per
worker, 180 s controller waits and completion notification.

## Exploratory decision

A local-specific route-access signal requires one local candidate to:

1. avoid runaway in all three networks;
2. produce returned, joint, in-distribution A and B in at least 2/3 networks;
3. retain B in at least 2/3 networks;
4. exceed both matched global and exact off in same-network A/B support.

The continuous objective and availability traces remain diagnostic; they do not
replace the same-network support criterion. A positive canary permits a frozen
fresh-network confirmation and the accepted Fig.4 direct-readout plus KMeans
pair. A negative canary closes this minimal dynamic-edge family without opening
beta, changing K, fitting contacts or comparing optimizers.

## Claim boundary

This is a three-network development canary using a patient-training target. It
cannot establish patient generalization, a recovered clinical waveform, a
patient core, or an ictal lifecycle mechanism.
