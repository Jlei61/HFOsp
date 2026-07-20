# Topic 4 Stage 0E: Poincare shooting and Floquet audit (LOCKED)

**Version:** 1.0
**Date:** 2026-07-20
**Role:** post-discovery numerical-topology audit. It does not amend or reinterpret
Stage 0C, Stage 0D v1, or Stage 0D v1.1.

## 1. Question and claim boundary

Stage 0D used fixed-duration rate summaries whose slope and single-bin spectral
ratio depend on the starting phase and window length. Stage 0E asks a narrower,
topology-native question:

> At frozen `z=0.85`, and only at `alpha_G=15` and `alpha_G=16`, does the
> unchanged nine-dimensional fast system contain a numerically resolved,
> transversely stable periodic orbit?

This is not a parameter search. The two points, section, integration schedule,
perturbations, finite-difference scales, and acceptance rules below are locked
before Stage 0E implementation and execution. No Stage 0C/0D result, source, or
document may be edited or overwritten. A positive result remains a homogeneous,
frozen-`z`, deterministic fast-system statement; Stage 1, slow lifecycle,
spatial coupling, noise, and SNN simulation remain closed.

## 2. Immutable model and inputs

Stage 0E reuses without modification:

- the Stage 0C nine-state dynamic-divisor equations;
- `w_ee_mult=1.1`, external-drive ratio `1.0`, all pool constants, and the
  extra-fine no-clip/no-extrapolation Siegert transfer;
- forward Euler, checked independently at `dt=0.125 ms` and `dt=0.0625 ms`;
- the Stage 0D phase-source `phase_050` full state only as a deterministic
  shooting seed. It is not counted as evidence.

Locked inputs:

| Input | SHA256 |
|---|---|
| `stage0d_local_basin_replication/phase_source_traces.npz` | `c819aa83b926b53d771a4124d18c818aadc217ee21f57d873b1e7b897afb6397` |
| `stage0d_local_basin_replication/phase_source.json` | `f676c6f331cbaa1401ff348e014ded724aba22d4e6000c3ff50ffa9ea1e3cec9` |
| `stage0c_transfer_support_audit_v1_1/extended_transfer_extra_fine.npz` | `dd40a7b82e1ca5ca4a6fcf514b1e0c721242502e3806133295e4c4411bd4e703` |
| `src/topic4_spatial_slowfast_stage0c.py` | `25b6538007d6974b44aa500f4f05929fa6f1d9ed48c6692d8acb1798180235ca` |
| `src/topic4_spatial_slowfast_stage0c_transfer.py` | `48ab839f6039134bfab22968d6deaad25011fbc654c206363341eef1bb1bc7ed` |

Any mismatch is `STAGE0E_ENGINEERING_OR_PROVENANCE_FAIL`, not a scientific
result. The authoritative runner must write an execution lock before integration
and verify the same hashes after integration.

## 3. Poincare section and high-resolution scout

The fixed section is

\[
h(x)=S_G-0.15=0, \qquad \dot S_G>0.
\]

`S_G` is used because it is smooth and crosses this interior level once during
the known relaxation-like cycle. A return is accepted only after the trajectory
has first moved below the section and then crosses upward. Crossing time and the
full nine-state crossing vector are linearly interpolated between adjacent Euler
states. Allowed return time is 300--1200 ms. No `rE` peak or FFT quantity enters
the section or shooting calculation.

At each fixed parameter point, start from the immutable `phase_050` seed and run
a 12 s scout at `dt=0.125 ms`, saving every Euler state. Require at least 12 clean
upward returns. Record every-cycle period, full section state, and a 256-bin
phase-normalized full-state waveform. State scales are

\[
s_j=\max(\operatorname{peak-to-peak}_j,10^{-3}),
\]

computed from the last two complete base-`dt` cycles and then reused for every
residual, perturbation, epsilon, and `dt/2` comparison at that parameter point.

## 4. Fixed-point shooting and cycle residuals

Starting from the last scout crossing, perform at most 20 Poincare fixed-point
shooting iterations, `x_{k+1}=P(x_k)`, with `S_G` fixed by the section. Record

\[
r_k=\|P(x_k)-x_k\|_{\infty,s}.
\]

The shooting candidate must have:

- final scaled section residual `<=1e-6`;
- the final three residuals non-increasing, with no unresolved state/support
  audit;
- last-four period coefficient of variation `<=1e-3`;
- period-aligned full-state residual between the final two cycles `<=2e-3`.

The final section state is then integrated for two complete returns and checked
again. Failure is unresolved/no-go; a spectral peak cannot rescue it.

## 5. Transverse Poincare Jacobian and Floquet multipliers

Delete the fixed `S_G` coordinate and represent the section by eight coordinates
`y`. For the three locked relative perturbation scales

```text
epsilon = [1e-3, 3e-4, 1e-4]
```

compute the central finite-difference Jacobian

\[
J_{:,j}(\epsilon)=
\frac{P(y+\epsilon s_j e_j)-P(y-\epsilon s_j e_j)}
     {2\epsilon s_j}.
\]

All 16 perturbed returns are integrated together but audited separately. Any
failed return invalidates that epsilon. The eight eigenvalues of `J` are the
non-trivial Floquet multipliers; the neutral time-shift multiplier is removed by
the Poincare section. Full Jacobians and complex multipliers must be saved.

For each `dt`, require:

- all three epsilon levels valid;
- spectral-radius range across epsilon `<=0.03`;
- normalized Frobenius difference from `1e-3 -> 3e-4` and
  `3e-4 -> 1e-4` each `<=0.10`;
- the smaller-epsilon difference cannot exceed `1.25` times the larger-epsilon
  difference plus `1e-3`.

Repeat shooting and the complete three-epsilon Jacobian at `dt=0.0625 ms`.
Require base-vs-`dt/2` period difference `<=max(1 ms,0.5%)`, phase-aligned
waveform residual `<=0.03`, and spectral-radius difference `<=0.05`.

Transverse stability is accepted only when

\[
\rho_{\max}<1,\qquad
1-\rho_{\max}\ge
\max(0.05,3\Delta\rho_{\epsilon},3\Delta\rho_{dt}),
\]

where `rho_max` is the largest radius across both time steps and all epsilon
levels. Thus a multiplier merely below one or inside numerical uncertainty is
fail-closed unresolved. If a robust full Floquet calculation is unavailable,
Stage 0E is unresolved; FFT power ratio is forbidden as a substitute.

## 6. Phase restarts and non-collinear perturbations

From the base-`dt` shooting cycle, interpolate states at phases
`[0,0.25,0.50,0.75]`. At every phase run:

1. an unchanged phase restart;
2. two fixed non-collinear multiplicative fast directions on
   `[rE,rI,sEE,sEI,sIE,sII]`:
   `[+,-,+,-,-,+]` and `[+,+,-,-,+,-]`;
3. two fixed non-collinear multiplicative pool directions on
   `[rE_fast,mu_G,S_G]`: `[+,-,0]` and `[-,0,+]`.

Nonzero entries change the corresponding coordinate by 3%. No coordinate is
clipped. Each history must provide eight clean section returns. Distances are
scaled full-state distances from the shooting fixed point at each return.

- all four unperturbed phase restarts must finish within `5e-4`;
- for each of the fast and pool families, all eight histories must be valid,
  have negative log-distance slope, end closer than their first return, have
  family-median final/first ratio `<=0.70`, and maximum final distance `<=0.05`.

Both independent perturbation families must pass. Phase anchors do not count as
perturbation evidence.

## 7. Physical and numerical audit

For each accepted orbit and every perturbation/Floquet return, audit every Euler
state for finite values, non-negativity, natural state bounds, transfer support,
refractory occupancy, and `rE>=100 Hz`. No state, moment, divisor, or transfer
input may be clipped or extrapolated.

For the final orbit report at minimum:

- period and period variability;
- section and period-aligned full-state residuals;
- all phase/perturbation return-distance series;
- all Jacobians, multipliers, spectral radii, epsilon and `dt/2` sensitivities;
- `muE`, `muI`, `sigmaE`, and `sigmaI` minima/maxima;
- E-rate peak and fraction of orbit time above 80 Hz and 100 Hz.

Any `rE>=100 Hz`, support violation, nonfinite state, or natural-bound violation
prevents stable-orbit acceptance.

## 8. Locked outcomes

A point is `stable_periodic_orbit` only if shooting, cycle residual, both
perturbation families, full Floquet margin, physical audits, and `dt/2` agreement
all pass. Otherwise it is `periodic_orbit_numerically_unresolved`,
`no_periodic_orbit`, or `numerical_failure`, with the failed gate recorded.

Overall verdicts are:

- `STAGE0E_STABLE_PERIODIC_ORBITS_ALPHA15_AND_ALPHA16`;
- `STAGE0E_STABLE_PERIODIC_ORBIT_ALPHA15_ONLY`;
- `STAGE0E_STABLE_PERIODIC_ORBIT_ALPHA16_ONLY`;
- `STAGE0E_NO_STABLE_PERIODIC_ORBIT_AT_LOCKED_POINTS`;
- `STAGE0E_NUMERICAL_UNRESOLVED`;
- `STAGE0E_ENGINEERING_OR_PROVENANCE_FAIL`.

No Stage0E verdict opens Stage 1 automatically.

## 9. Engineering and artifacts

- single process, BLAS threads = 1, peak RSS `<4 GiB`;
- cheap-first: scout and shooting must pass before Floquet and perturbation
  batteries run for that point;
- new code and outputs only:
  - `config/topic4_spatial_slowfast_stage0e.yaml`;
  - `src/topic4_spatial_slowfast_stage0e.py`;
  - `scripts/run_topic4_spatial_slowfast_stage0e.py`;
  - `tests/test_topic4_spatial_slowfast_stage0e.py`;
  - `results/topic4_sef_hfo/spatial_slowfast_topology/stage0e_poincare_floquet_audit/`.

Required outputs: summary JSON, per-point JSON/CSV, cycle/crossing tables,
Poincare Jacobian and multiplier JSON/NPZ, perturbation-return JSON/CSV, compact
full-state traces, `STATUS.md`, and a diagnostic figure with Chinese
`figures/README.md`.
