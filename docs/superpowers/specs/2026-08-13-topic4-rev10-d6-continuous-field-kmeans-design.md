# Topic 4 rev10-D6 continuous-field KMeans sensitivity design

## Scientific question

D5.2 showed that a frozen continuous Node field plus a stationary local spatial
OU drive can produce returned, same-network A/B events with the correct signed
patient-prototype geometry. D5.4 showed that the natural KMeans separation does
not replicate the patient benchmark. The residual is asymmetric: model A is
diffuse and model B recruits too few contacts.

D6 asks whether smooth degrees of freedom of the same whole-sheet field can
improve that residual while the accessibility process is frozen. It does not
test a new core count, an edge mechanism, `beta`, or an optimizer family.

## Field representation

The warm field is `v62_density_t050`, represented by one `18 x 18` uniform
tensor cubic B-spline log field. D6 adds one low-frequency whole-sheet Fourier
direction at a time:

```text
s_z(x) = s_warm(x) + a phi_m(x)
q_z(x) = exp(s_z(x))
h_z(x) = fixed-mass projection(q_z)
```

All real sine/cosine phases with isotropic harmonic radius at most two are
included. Every direction is tested at `+/-0.4` and `+/-0.8` log-RMS. The
Fourier directions are projected into the same uniform spline basis on a
uniform sheet grid. Candidate generation receives no contact coordinates,
shaft identity, patient events, labels, source locations, component identity,
or peak count. Spline coefficients are numerical coordinates, not cores.

## Frozen dynamics

All candidates use the D5.2 local spatial OU law:

```text
sigma = 0.10 / ms
tau = 20 ms
ell = 0.38 mm
```

The field mass, `d_i`, topology, delays, detector, duration, and seeds shared
across candidates are frozen. Edge coefficients are exact zero. `beta`,
adaptation, depression, and inhibitory resources remain closed.

## Readout and exploratory score

The primary Fig.4 feature is masked within-event normalized contact rank.
KMeans is balanced by network and supervised A/B mode before bootstrapping.
The patient consistency term is the signed Spearman margin
`min(r_AA, r_BB, -r_AB, -r_BA)`. A separate mode-specific recruitment term
penalizes the larger absolute error in recruited-contact median, because D5.4
showed a specific mode-B under-recruitment residual.

```text
J = (1 - purity)
    + 0.125 (1 - signed_margin)
    + 0.25 worst_mode_recruitment_error
    + 0.10 OOD
    + 0.05 detector_occupancy
```

This is an exploratory ranking, not a pass/fail gate. A candidate is merely
unevaluable if fewer than two networks each provide three formal clean events
per mode or if it runs away. Patient matched q05 is reported but not used to
discard candidates.

## Interpretation boundary

- Improvement would identify a continuous-field direction worth fresh-network
  selection; it would not establish optimizer convergence or patient blind
  generalization.
- No improvement would close this local sensitivity radius only. It would not
  prove that all continuous fields or the SNN mechanism family are incapable.
- Fig.4 remains unaccepted until the selected field produces same-network A/B
  direct waveforms and reaches the frozen KMeans/patient-consistency contract
  on untouched confirmation networks.
