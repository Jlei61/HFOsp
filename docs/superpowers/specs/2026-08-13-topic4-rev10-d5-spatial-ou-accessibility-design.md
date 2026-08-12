# Topic 4 rev10-D5: translation-invariant continuous spatial accessibility

## Motivation

D4.1 shows that the frozen Node scaffold carries both A and B routes when synchronously initiated across fresh networks, but it does not show that ordinary fluctuations can enter both route basins. The A route source found by D4 lies outside the learned Node field, so placing another core there would encode the observed answer. D5 instead asks whether a continuous, observation-invariant local afferent fluctuation can make both routes spontaneously accessible.

## Mechanism

On a periodic two-dimensional grid, a zero-spatial-mean field follows the exact discrete OU update

`z(t+dt_u) = exp(-dt_u/tau) z(t) + sqrt(1-exp(-2dt_u/tau)) sigma K_ell * epsilon(t)`.

`K_ell` is an isotropic Gaussian spatial kernel. The grid field is bilinearly interpolated to every E neuron and enters only the external E afferent rate:

`nu_E,i(t) = max(0, nu_ext(t) + z_i(t))`.

It does not change thresholds, recurrent weights, topology, delays, I input, contacts, or patient targets. The field uses no fixed centers and no finite K components. Its distribution is translation invariant; a particular realization may have transient maxima anywhere on the sheet.

## Exact marginal control

For every local candidate, a `permuted` candidate uses the identical OU innovations and identical per-update E-neuron value multiset, then applies one frozen random neuron permutation. Thus local versus permuted differs in spatial adjacency, not amplitude distribution, temporal spectrum, or population mean.

The canary scans `sigma={0.5,1.0}/ms`, `ell={0.38,0.76} mm`, `tau=20 ms`, plus exact off. The length scales are one and two times the intrinsic E-to-E connection scale, not electrode spacing or D4 source geometry.

## Exploratory decision

- `SPATIAL_LOCALITY_ACCESS_OBSERVED`: a local arm has no runaway, produces clean A and B in at least two of three networks, and exceeds both exact off and its matched permuted arm in same-network dual-mode support.
- `NONLOCAL_MARGINAL_ACCESS_OBSERVED`: local and permuted arms both restore access, so spatial coherence is not specifically supported.
- Otherwise: no access in the tested continuous fluctuation family.

This is a development canary, not patient generalization. Only a surviving candidate may enter fresh-network confirmation and the required Fig.4 pair. KMeans stability is never substituted for supervised patient-mode support.
