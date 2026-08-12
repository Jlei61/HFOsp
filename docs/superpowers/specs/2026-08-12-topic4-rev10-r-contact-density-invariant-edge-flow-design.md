# Topic 4 rev10-R: contact-density-invariant graph edge-flow residual

## 1. Scientific question

rev10-SA established a narrow result: a patient-trained, observation-invariant
continuous Node field robustly supports joint-shaft patient mode B across fresh
networks, but does not provide shared mode-A capacity. rev10-R asks:

> With the Node field frozen, can a low-rank continuous redistribution of the
> existing E-to-E graph restore the missing mode-A route while preserving mode B?

This is an exploratory route-capacity experiment. It is not another field fit,
not patient-blind generalization, and not evidence for a unique biological core.

## 2. What stays frozen

- Node threshold field and heterogeneity:

```text
Vtheta_i = Vtheta_0 - h_i d_i
```

- Neuron positions, E/I labels, topology, delay-bin assignment, E-to-I edges,
  GABA edges, external drive, detector, readout, and shaft-aware patient target.
- Development Node anchor `t=0.050`; `t=0.025/0.075` are field sensitivities.
  The anchor was chosen after reading V6.2 and is not an independently selected
  winner. Seeds 1041-1043 are consequently development history and may not be
  reused as rev10-R selection or confirmation networks.
- No contact, shaft, onset, patient label, Gaussian component, or field peak may
  enter the edge basis construction.

## 3. Contact-density-invariant edge coordinates

### R1 graph-spectral preflight outcome

The graph-spectral construction below was implemented and audited on fit
networks 1051-1054 before any SNN candidate screen. Retained singular values
were tightly compressed (`0.953-0.992`); the four-column Chebyshev response
design had condition number `3.3e6-4.2e6`, sampled feature singular-value ratios
approximately `1:0.12:9e-4:1.5e-7`, and maximum column correlation about
`0.9995`. Thus the nominal four-parameter family had only about two usable
directions. R1 is retained as a parameterization-negative graph-only audit and
must not be described as optimizer or mechanism failure. No R1 SNN candidates
were launched.

### R2 primary spatial-vector family

The next admissible family uses fixed physical sheet coordinates shared across
network realizations. Let `m_ts=(x_t+x_s)/2`, `d_ts=x_s-x_t`, and let `B(m)` be
the complete degree-two polynomial basis on the uniformly normalized 20 mm
sheet. A continuous vector field `v_c(m)` has six scalar coefficients per
component (12 total), and

```text
ell_ts(c) = v_c(m_ts) dot d_ts / l_EE
W'_ts^(delta) = S_t * softmax_(s,delta)
                [log W_ts^(delta) + ell_ts(c)]
```

This is one continuous route field, not multiple cores. Swapping source and
target changes the sign of `ell` while leaving the midpoint unchanged. The
basis is uniform over the sheet and does not use contacts, shafts, patient
events, Node `h`, or observed field peaks. Before SNN screening, all existing
E-to-E edges on every fit graph must show effective feature rank at least 10
at relative covariance eigenvalue `1e-6`, with condition number at most `1e6`.
Candidate directions are whitened using the equal-network mean full-edge
feature covariance, then amplitude-bounded by exact per-graph feature maxima.

R2 graph-only capacity passed on all four fit networks before patient scoring:
effective rank was `12/12` in every network, covariance condition number was
`30.9-31.6`, and the weakest relative covariance eigenvalue remained
`0.0316-0.0323`. The 12-dimensional spatial-vector library is therefore
eligible for the exploratory SNN screen. This is a representation-capacity
result only; it does not show that any coefficient field restores mode A.

Let `W_ts` be the frozen E-to-E weight from source `s` to target `t`, summed over
the fixed delay labels for basis construction. Because rows are targets and
columns are sources, a row-normalized matrix is an incoming-source distribution,
not a forward Markov transition matrix. The directed basis therefore uses the
two-sided normalized sparse operator:

```text
d_in(t)  = sum_s W_ts
d_out(s) = sum_t W_ts
A = diag(d_in)^(-1/2) W diag(d_out)^(-1/2)
A approximately U Sigma V^T
```

Drop the leading degree-dominated singular pair, retain the next `R` left/right
coordinates, and scale every coordinate to unit RMS over neurons. The basis is
computed from the frozen simulated graph only. Sign is frozen by a deterministic
largest-magnitude-entry convention. The rank-`R` subspaces and singular values
are hashed; near-degenerate subspaces are compared by projector rather than by
individual vector identity.

For every existing E-to-E edge and fixed delay `delta`, the primary shared
cross-network family is a continuous spectral filter.  Four Chebyshev
coefficients `c_0..c_3` define `f_c` on normalized singular value, and:

```text
xi_r = 2 sigma_r / sigma_1 - 1
f_c(sigma_r) = sum_q c_q T_q(xi_r)
ell_ts(c) = sum_r f_c(sigma_r) u_r(t) v_r(s)
S_t = sum_(s,delta) W_ts^(delta)
W'_ts^(delta) = S_t * softmax_(s,delta)
                [log W_ts^(delta) + ell_ts(c)]
```

This interaction is nonseparable after target normalization and can alter
directional source allocation. The product `u_r(t)v_r(s)` is invariant to the
joint sign flip of an SVD pair. If a singular block is exactly degenerate,
`f_c` assigns the same response to every vector in that block, so the summed
field is invariant to arbitrary rotations within it. Near-degenerate modes
receive nearby responses. The same numerical `c` therefore has a graph-spectral
meaning across independently sampled networks without aligning neuron-specific
singular vectors.

A full `R x R Gamma` is not primary. Although the *set* `U Gamma V^T` is
subspace-rotation invariant, a fixed numerical `Gamma` is not invariant when a
near-degenerate SVD block rotates between network seeds. Cross-mode terms may
be opened only after a graph-only alignment audit freezes a common coordinate
transport; they cannot be used to rescue the first screen post hoc. The primary
rank is `R=4`; `R=2/6` are numerical resolution sensitivities, not biological
mode counts. This remains one low-rank graph-flow field, not a collection of
cores.

`contact-density-invariant` has a narrow meaning: no extra basis element is
placed where contacts are dense and no shaft path determines its support. It
does not mean patient-observation-free. The frozen E-to-E scaffold already
inherits the rank-derived patient propagation axis, and the patient-training
objective selects `c`; both facts must remain explicit in every artifact.

## 4. SNN connection

The modified graph enters only the recurrent excitatory current:

```text
tau_m dV_t/dt = -V_t + I^E_t - I^I_t
I^E_t(t) is driven by sum_(s,delta) W'_ts^(delta) spike_s(t-delta)
```

Node changes pre-spike accessibility through `Vtheta_i`; edge flow acts only
after presynaptic activity exists. The experiment therefore tests whether the
remaining failure is post-initiation routing rather than field ignition.

## 5. Minimal structural contract

Only implementation-invalid outcomes stop a run:

- finite nonnegative weights;
- exact topology and delay labels;
- incoming E budget error at most `1e-9` for every nonzero E target;
- E-to-I and GABA hashes unchanged;
- `c=0` exact no-op.

Edge ratio, KL, ESS, source outgoing influence, and weighted-delay change are
continuous diagnostics. Exploratory bounds start at edge ratio `[0.5,2]`; a
secondary `[0.25,4]` range is opened only if the narrow range changes the
mode-A route without destabilization.

### R2.1 dose correction

The first R2 sentinel exposed a dose-parameterization failure before the full
screen: scaling every direction by the single most extreme edge produced only
`0.009-0.013` full-edge logit RMS, and the sentinel's sampled 1st-99th edge
ratio was only `0.972-1.027`. This run is `R2_DOSE_PARAMETERIZATION_FAIL`, not
an edge-mechanism result. R2.1 retains the same 12-dimensional continuous
sheet field and Sobol directions. Each direction is scaled by the equal-network
full-edge second moment to raw logit RMS `0.15`, then clipped edge-wise at
`0.95 log(2)/2` before target normalization. The observation-free feature
sample predicts clipped RMS `0.123-0.142` with only `3.5%-4.9%` clipped edges;
the ratio guarantee remains `[0.518,1.932]`. No patient event, contact, shaft,
Node field, component, or peak informed this correction.

## 6. Objective and independent unit

The independent unit is the network seed, not pooled events. For mode
`k in {A,B}` and network `r`, report:

```text
n_joint_ID(k,r)
joint_ID_rate(k,r)
OOD_rate(k,r)
D_rec(k,r), D_prec_II(k,r), D_prec_IS(k,r), D_prec_SS(k,r)
D_profile(k,r), D_cloud(k,r)
```

Candidate ranking uses equal network weights and protects the weak mode:

```text
J_route = LSE_tau(D_A, D_B)
          + lambda_s * LSE_tau(L_support,A, L_support,B)
          + lambda_o * mean_network(OOD)
```

`L_support,k` is a continuous deficit in patient-supported joint-event rate;
zero-event networks remain valid observations. Pooled KMeans, mean correlation,
or total joint fraction cannot compensate for missing mode A. Same-network A/B
coexistence is reported explicitly but is not multiplied into many blockers.
Only detector events with `event_returned=True` enter the interictal mode score,
KMeans panel, or direct representative-event readout. All detector events and
the excluded non-returned count remain descriptive outputs. This is a
zero-simulation censoring rule on saved event sidecars, not a rerun or candidate
adaptation.

## 7. Exploration sequence

1. Implement basis/no-op/normalization tests and a zero-SNN structure sidecar.
2. Run a small symmetric Sobol library in the four-dimensional Chebyshev
   coefficient space on common fit networks 1051-1054. Include `+c/-c` pairs
   and `c=0`; do not start with CMA-ES. The initial
   budget is 32 nonzero fields plus one no-op, with one Node baseline reused per
   network.
3. Freeze at most six diverse Pareto fields before selection networks
   1061-1063. Retain fields with different A/B tradeoffs, not only the scalar
   minimum.
4. Freeze at most three fields before confirmation networks 1071-1073, with the
   common detector and mode-conditioned joint+ID readout.
5. Compare Node with Node+Edge per network. Edge-only is descriptive because
   rev9 showed it is not an ignition substrate.

Only after a known-good shared mode-A solution exists may Sobol, local search,
and CMA-ES be compared at equal SNN evaluation budget. Until then, optimizer
failure remains unresolved rather than assumed.

## 8. Interpretation

- A improves and B is preserved across networks: directional edge flow is a
  candidate relay mechanism; proceed to a frozen four-arm confirmation.
- Event yield rises but mode shapes do not: edge is a conditional amplifier,
  not the missing route mechanism.
- Different `c` is needed per network: finite shared route capacity remains
  unobserved; do not increase rank without a residual-specific reason.
- Ranks 2/4/6 all fail with adequate structural movement: static redistribution
  is insufficient; the next candidate is a dynamic state or slow-variable
  mechanism, not more field peaks.

`beta` remains closed until the residual is specifically a radial response
width or effective-delay-scale error. Static substrate results remain separate
from onset/carrier/termination/recovery claims about the ictal lifecycle.
