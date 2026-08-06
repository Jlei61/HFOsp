# Topic 4 Stage 0F: smooth-transfer discrete-variational certificate (LOCKED)

**Version:** 1.0  
**Date:** 2026-07-20  
**Role:** numerical certificate repair after Stage 0E. This node does not alter
the model, search biological parameters, or open Stage 1/spatial simulation.

## 1. Question and boundary

Stage 0E resolved a period-1 Poincare fixed point at both locked parameter
points, reproduced it at `dt/2`, and obtained contraction from two independent
nonlinear return batteries. Its only failed gate was the lack of an epsilon
platform for finite-return Jacobians computed through a piecewise-bilinear
transfer table. Stage 0F asks only:

> Does the same orbit remain after replacing the derivative-hostile bilinear
> interpolation by an orbit-local smooth interpolation of the same exact
> Siegert table, and do two independent derivatives of the resulting *discrete
> event map* agree on transverse stability?

The two fixed points are `z=0.85, alpha_G in {15,16}`. The Stage 0C nine-state
equations, `w_ee_mult=1.1`, drive ratio `1.0`, upward `S_G=0.15` section,
forward-Euler base/half steps (`0.125/0.0625 ms`), and Stage 0E state scales are
immutable. A positive result certifies only a homogeneous frozen-`z` fast orbit.
It does not establish a slow entry/exit path, spatial recruitment, physiological
validity, or a full SNN seizure lifecycle.

## 2. Locked inputs

| Input | SHA256 |
|---|---|
| Stage 0E summary | `b478adc6b261eb6712c3c757777ba9441208a8dc2c159d2829e8450f66f4bca0` |
| extra-fine Siegert table | `dd40a7b82e1ca5ca4a6fcf514b1e0c721242502e3806133295e4c4411bd4e703` |
| Stage 0E module | `5c3ef24ffdc1f4d64962a3b3dc80dac2ab55d4c77ecfdca89ba6daf9dfb3c877` |
| Stage 0C module | `25b6538007d6974b44aa500f4f05929fa6f1d9ed48c6692d8acb1798180235ca` |
| transfer module | `48ab839f6039134bfab22968d6deaad25011fbc654c206363341eef1bb1bc7ed` |
| alpha 15 shooting iterates | `0da44a5999a57a8f53e244ab23e29fe38a0dd9bedce36d7efdae900296b0a120` |
| alpha 15 base/half traces | `eec8886a0dab52761e68eab0e01d85f53b3bcfd5056a2c73e9e22f2db510d700` / `cd304b6fcf3f6bf3d12dfc7fdcd64889923f743c01b0cf19be7619eec449fa0e` |
| alpha 15 outcome | `643991c3d63b8f861b81e4d87cb4040ff3abb8d06e419b927d4a352836e4660c` |
| alpha 16 shooting iterates | `d3130ea6db6b65f912e2489aed3ecb5b86dbc42d79ff3edccfba0188194f7dc3` |
| alpha 16 base/half traces | `d1bc0553f85f12b0d425be10c809824e799eb71c94fd043546e0532be9dab531` / `467f28402272334e98746ebd3c117fc83f24c86d6c7942f6fd6e6adde7291e02` |
| alpha 16 outcome | `1ff24defa3684195be1fd934ee57f46bed59c5f6fe3306642637bbd7e61506d6` |

Any pre/post-execution mismatch is
`STAGE0F_ENGINEERING_OR_PROVENANCE_FAIL`, not a scientific result.

## 3. Smooth transfer and exact parity

Fit a tensor-product interpolating cubic spline (`kx=ky=3`, smoothing `s=0`)
to the unchanged extra-fine **log Siegert integral** table on the locked local
domain

```text
mu in [-160, 80] mV
sigma in [3, 20] mV.
```

No table value is refit, regularized, clipped, or extrapolated. Every nominal
orbit state must remain inside this local domain. The rate is reconstructed from
the interpolated log integral with the unchanged membrane and refractory time
constants.

At 512 phase-uniform states from each independently shot smooth orbit and each
time step, audit both E and I transfer calls against direct log-domain Siegert
quadrature. Direct derivatives are obtained from the moving-boundary identities

\[
\partial_\mu I=(f(a)-f(b))/\sigma,\qquad
\partial_\sigma I=(a f(a)-b f(b))/\sigma,
\]

where `f(u)=erfcx(-u)`, `a=(V_reset-mu)/sigma`, and
`b=(V_th-mu)/sigma`. Signed log arithmetic is required.

Locked parity gates, separately for E and I and pooled across both time steps:

- rate maximum absolute error `<=5e-5 kHz`;
- rate maximum relative error with denominator floor `1e-4 kHz` `<=5e-3`;
- each derivative maximum absolute error `<=5e-5 kHz/mV`;
- each derivative maximum relative error with denominator floor
  `1e-7 kHz/mV` `<=0.05`.

Failure closes the point; a stable multiplier cannot rescue transfer mismatch.

## 4. Orbit reconstruction and LUT-orbit parity

Use the final Stage 0E base/half shooting states only as deterministic seeds.
At each time step re-shoot the smooth event map by repeated Poincare returns for
at most 20 iterations. Require:

- scaled section residual `<=1e-8`;
- two-cycle aligned smooth-orbit residual `<=2e-4`;
- period coefficient of variation over the last four returns `<=1e-3`;
- all crossings upward, all states finite, no rate `>=100 Hz`, no natural-bound
  or local-domain violation.

Compare each smooth orbit with the corresponding Stage 0E bilinear-LUT orbit
after 256-bin phase normalization, using the frozen Stage 0E scales:

- period difference `<=1 ms`;
- full-state aligned waveform residual `<=0.03`.

The independently shot smooth base/half orbits must also differ by at most
`max(1 ms, 0.5%)` in period and `0.03` in aligned waveform residual.

## 5. Two locked derivative constructions

Both constructions differentiate the same forward-Euler trajectory and the
same linearly event-located Poincare crossing. They do not finite-difference the
whole return map.

### 5.1 Chain-rule variational map

Construct the full `9 x 9` continuous RHS Jacobian analytically from:

- exact derivatives of the Stage 0C moment algebra;
- cubic-spline derivatives of log Siegert integral and the resulting rate;
- the analytic recruitment-sensor derivative;
- exact linear synapse and pool derivatives.

Propagate `A_{n+1}=(I+dt J_f(x_n))A_n` from the eight scaled section basis
vectors.

### 5.2 Centered-RHS variational map

Construct the same RHS Jacobian by central state differences at two locked
scale-relative steps:

```text
h/scale = [1e-5, 3e-6]
absolute floor = 1e-9.
```

Propagate both matrices along the unchanged nominal trajectory. No perturbed
trajectory may determine the event time.

### 5.3 Event sensitivity

If `x_n` and `x_{n+1}` bracket the upward section, differentiate the exact
linear crossing interpolation, including the derivative of the crossing
fraction. The returned section row must be zero to `1e-10`; crossing
transversality must exceed `1e-4 /ms`.

At each time step require:

- all three Poincare matrices finite;
- centered-RHS ladder normalized Frobenius difference `<=0.05`;
- chain versus each centered matrix normalized Frobenius difference `<=0.05`;
- spectral-radius range over the three matrices `<=0.02`.

Normalized Frobenius denominators use
`max(||A||_F, ||B||_F, 1e-8)`; the floor must be reported. This prevents both a
near-zero matrix and roundoff from being hidden by an undefined relative error.

## 6. Floquet and dt certificate

The eight eigenvalues of each normalized section Jacobian are the non-trivial
discrete Floquet multipliers. Let `rho_max` be the maximum radius over both time
steps and both derivative constructions, `delta_method` the largest same-dt
radius spread, and `delta_dt` the largest base/half difference after matching
the same construction. Accept transverse stability only when

\[
\rho_{max}<1,\qquad
1-\rho_{max}\ge\max(0.05,3\delta_{method},3\delta_{dt}),
\]

and every transfer, orbit, derivative-consistency, event, physical, and
provenance gate passes. Multiplier matching is reported by minimum-cost complex
assignment but is diagnostic; the fail-closed radius/matrix gates are primary.

## 7. Outcomes and interpretation

Per point:

- `stable_periodic_orbit_derivative_certified`;
- `periodic_orbit_derivative_unresolved`;
- `smooth_orbit_not_lut_equivalent`;
- `numerical_failure`.

Overall:

- `STAGE0F_DERIVATIVE_CERTIFIED_ALPHA15_AND_ALPHA16`;
- `STAGE0F_DERIVATIVE_CERTIFIED_ALPHA15_ONLY`;
- `STAGE0F_DERIVATIVE_CERTIFIED_ALPHA16_ONLY`;
- `STAGE0F_NO_DERIVATIVE_CERTIFICATE_AT_LOCKED_POINTS`;
- `STAGE0F_NUMERICAL_UNRESOLVED`;
- `STAGE0F_ENGINEERING_OR_PROVENANCE_FAIL`.

Even the strongest verdict keeps `stage1_open=false` and `space_open=false`.
It would establish that delayed divisive feedback can support a locally stable
fast rhythm in this reduced frozen system; it would not establish entry,
termination, spatial containment, or physiological realism.

## 8. Engineering and artifacts

- single process; BLAS threads `1`; peak RSS `<4 GiB`;
- no biological parameter sweep, stochastic run, slow variable, or spatial
  coupling;
- new files/output only:
  - `config/topic4_spatial_slowfast_stage0f.yaml`;
  - `src/topic4_spatial_slowfast_stage0f.py`;
  - `scripts/run_topic4_spatial_slowfast_stage0f.py`;
  - `tests/test_topic4_spatial_slowfast_stage0f.py`;
  - `results/topic4_sef_hfo/spatial_slowfast_topology/stage0f_smooth_transfer_variational_certificate/`.

Required artifacts: execution lock; aggregate and per-point JSON/CSV; smooth
base/half cycle traces; exact transfer-parity rows; all three Poincare matrices
and multiplier sets per time step; derivative-consistency table; `STATUS.md`;
diagnostic PNG/PDF and Chinese `figures/README.md`.
