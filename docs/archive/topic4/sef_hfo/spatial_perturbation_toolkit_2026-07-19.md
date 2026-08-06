# Topic 4 spatial perturbation toolbox — frozen contract (2026-07-19)

## Purpose

This toolbox tests how a spatial dynamical system responds to perturbations as its state approaches a
transition. It does **not** define the biological slow variables or manufacture a Jacobian. A new model
must provide its own state trajectory, tangent operator `J`, input space `B`, output readout `C`, spatial
coordinates and registered source/sink geometry.

Stable pure API: `src/spatial_perturbation_toolkit.py`. Topic-4 MZ/M3B adapter:
`src/topic4_state_conditioned_susceptibility.py`. Gated artifact runner:
`scripts/run_topic4_state_conditioned_susceptibility.py`.

## Four objects that must stay separate

1. **Instantaneous eigenmode of J**: asymptotic stability, oscillation frequency and spatial loading.
2. **Finite-time optimal input V1**: input that maximizes `||C exp(JT) B v||` for declared B/C/T.
3. **Finite-time optimal output U1**: output produced by V1; not an eigenvector.
4. **Fixed-kick response**: `C exp(Jt)b_fixed`; the only one that answers what one pre-declared
   perturbation actually does across states.

`sigma1(T)` is an operator envelope because its input is re-optimized at every state/T. It must not be
labelled fixed-kick gain. A kymograph shows space-time recruitment; it does not prove a continuous
wavefront without an arrival-time/distance analysis.

## Stable API

| Function | Contract |
|---|---|
| `finite_time_operator_svd` | SVD of caller-declared `C exp(JT) B`; returns σ1, V1 coordinates, U1 and singular spectrum |
| `operator_gain_envelope` | σ1 over T with the same B/C spaces |
| `linear_response_timecourse` | one fixed input b propagated over time; never re-optimizes b |
| `response_gain_curve` | norm of the fixed-input output |
| `region_response_curve` | fixed-mask RMS or mean-absolute regional response |
| `cumulative_response_ratio` | stable cumulative sink/source energy ratio; avoids division spikes at source zero crossings |
| `axis_kymograph` | band-averaged `|response|` with array contract `(time, position)` |
| `first_arrival_times` | threshold-defined recruitment latency; unhit positions stay NaN |
| `fit_arrival_time_distance` | directed source→sink latency slope, velocity proxy, R²; fails closed when too few positions cross |
| `normalized_field_overlap` | adjacent non-negative mode-loading overlap to expose mode switches |

Topic adapter functions `leading_mode_snapshot`, `make_localized_kick`, `fixed_kick_readouts`,
`state_operator`, probe dictionaries and MZ `z_bar→q` mapping remain model-specific. A future model should
replace this adapter, not edit the pure toolbox.

## Required analysis order for a new model

1. Capture actual state timestamps and freeze onset/runoff definition before reading responses.
2. Validate each operating point and Jacobian; unresolved states remain blank.
3. Plot leading Re/Im plus invariant-subspace loading over actual time. Report mode overlap/switches.
4. Compute V1/U1/σ1 only after B/C/T are explicit.
5. Apply one fixed perturbation across states; report total response, regional response and spatial maps.
6. Add axis-time kymograph **and** arrival-time vs distance with threshold sensitivity.
7. Run scaffold/state controls, grid/domain convergence and at least one nonlinear small-amplitude check.
8. If the model is oscillatory, linearize along the actual trajectory or limit cycle; a frozen fixed-point
   Jacobian is not a Floquet analysis.

## Acceptance and claim boundary

- Shape/order/unit tests pass; t=0 response equals the supplied input/readout.
- SVD σ1 is at least as large as any probe in the same declared input space.
- Fixed-kick and operator-envelope curves are separate artifacts and captions.
- Arrival fits require at least four source→sink positions; otherwise result is `eligible=false`.
- Threshold, band, masks, B/C spaces, grid/domain and physical coordinate conversion are saved.
- Eigenvalue Re/Im support stability/frequency claims; eigenvectors support spatial-mode claims; neither
  alone proves propagation.
- Current MZ result is a **z-only SNN trajectory-derived frozen-q M3B rate-field susceptibility** result:
  not old qI/gK, and not a direct perturbation of the full MZ spiking network.

## Tests

`tests/test_spatial_perturbation_toolkit.py` covers analytic decay, declared B/C SVD, fixed-mask response,
cumulative ratio stability, synthetic wavefront slope recovery, fail-closed insufficient reach and mode
overlap. `tests/test_topic4_state_conditioned_susceptibility.py` covers the Topic-4 adapter and real
operator contracts.
