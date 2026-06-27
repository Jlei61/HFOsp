# M3B-next spectral phase-map implementation plan

> Status: PLAN, 2026-06-27.
> Design spec: `docs/superpowers/specs/2026-06-27-sef-hfo-m3b-spectral-phase-map-design.md`.
> Scope: build a Brunel-inspired mean-field / finite-Jacobian phase map for the current SEF-HFO
> SNN family, then validate it against rate-field and SNN events.
> Working name: **M3B-R2 spectral mechanism validation of the spontaneous / slow SNN bridge**.

## 0. Hard Boundary

- This is **M3B-next**, not the completed Round-1 bridge.
- M3B-next produces a frozen-state spectral phase map; it does not by itself prove spontaneous
  seizure transition.
- M3A provides slow trajectories. M3B overlays those trajectories after the phase map exists.
- Start with the canonical LIF-derived rate field in `src/sef_hfo_lif.py`; do not revive the
  demoted sigmoid field as the primary object.
- Homogeneous `k`-spectrum is a sanity check; finite core-heterogeneous eigenmodes are the main object.
- The Jacobian is for a coarse-grained E/I rate-synapse field, **not** for microscopic spike/reset
  LIF states.
- Non-normal transient growth and left-eigenvector core controllability are primary metrics, not
  optional diagnostics.
- All "seizure-like" claims must still split R4a structured recruitment from R4b tonic runaway.

## Implementation Order: TDD First

The scientific story is still:

```text
homogeneous sanity -> finite-core spectrum -> phase map -> rate-field/SNN validation
-> SEEG bridge -> M3A overlay
```

The implementation order is deliberately different:

```text
contracts -> state/gain/op-point -> metrics/JVP/eigen -> tiny map
-> controls -> validation -> readout -> M3A overlay
```

This prevents a failure mode where a beautiful phase map is produced before the gain signs,
metric definitions, or bridge schema are actually trustworthy.

## TDD-0: Claim Boundary and Artifact Contract

Goal: freeze outputs and forbidden claims before any eigenvalue exists.

- [ ] Add `STATUS.md` under `results/topic4_sef_hfo/m3b_spectral_phase_map/` with:
  - current M3B Round-1 bridge verdict;
  - new M3B-R2 objective;
  - explicit M3A handoff requirement;
  - allowed verdict categories.
- [ ] Declare artifact paths and minimal schemas for JSON/CSV outputs.
- [ ] Tests:
  - `test_status_exists_and_mentions_round1_m3b_next_m3a`
  - `test_artifact_paths_are_declared`
  - `test_no_forbidden_claims_in_status`

## TDD-1: Grid / Kernel / State-Vector Contract

Goal: implement the pure data model before dynamics.

- [ ] Implement `src/topic4_m3b_spectral_phase.py`.
- [ ] Define dataclasses for grid, kernels, core masks, excitability fields, global drive /
  disinhibition, frozen slow-state fields.
- [ ] Primary atlas geometry is **single core**. Two-core geometry is supported for validation /
  subject-facing readout only, not for the first main atlas.
- [ ] Lock state vector:

```text
z = [rE, rI, sEE, sEI, sIE, sII]
```

- [ ] Tests:
  - `test_pack_unpack_state_roundtrip`
  - `test_state_size_equals_6_times_grid_size`
  - `test_no_core_mask_empty_or_uniform`
  - `test_single_core_mask_area_reasonable`
  - `test_two_core_mask_centers_on_ee_axis`
  - `test_ar1_kernel_is_isotropic`
  - `test_ar2_kernel_major_axis_is_45deg`

## TDD-2: LIF Transfer and Local Gain

Goal: make `Phi_LIF`, `dPhi/dmu`, drive scaling, and `q` scaling numerically trustworthy before
any Jacobian is built.

- [ ] Wrap / reuse `src.sef_hfo_lif.lif_rate` and local gain calculation.
- [ ] Implement explicit effective inhibition / drive scaling helpers for `q_global` and `q_core`.
- [ ] If M3A-A1 exists, load `slow_to_rate_mapping.json` and assert the local helpers use the same
  sign and valid ranges.
- [ ] Tests:
  - `test_phi_lif_monotonic_in_mu`
  - `test_dphi_dmu_matches_finite_difference`
  - `test_gain_finite_under_low_and_high_drive`
  - `test_gain_nonnegative_in_valid_regime`
  - `test_q_global_reduces_effective_inhibition_in_expected_direction`
  - `test_q_core_affects_core_e_cells_only`
  - `test_m3a_mapping_signs_match_rate_helpers`

## TDD-3: Operating Point Solver

Goal: define exactly what state is being linearized.

- [ ] Primary source: deterministic rate-field integration-to-steady.
- [ ] Secondary sources: SNN pre-event baseline average and M3A frozen slow-state samples.
- [ ] Store `operating_point_source`, rates, residuals, core/surround summary, convergence status,
  and saturation/runaway flags.
- [ ] Failed points are `unresolved`, never stable.
- [ ] Tests:
  - `test_no_core_operating_point_is_spatially_uniform`
  - `test_core_excitability_raises_core_rE_or_lowers_threshold_effectively`
  - `test_operating_point_residual_below_tol_when_converged`
  - `test_bad_params_return_unresolved_not_stable`
  - `test_operating_point_source_is_recorded`
  - `test_high_rate_saturation_is_flagged_not_axial`

## TDD-4: Homogeneous Brunel-Style Sanity

Goal: test the numerical dispersion structure after gains and operating points are locked.

- [ ] Re-run / promote a clean homogeneous dispersion runner:
  - input: `ratio`, `w_ee_mult`, `g`, E/I gain scalars;
  - output: `lambda(k)`, `k_star`, `freq_Hz`, `regime`, convergence flags.
- [ ] Add documented 2D `k=(kx,ky)` support if needed.
- [ ] Generate `homogeneous_dispersion.json`.
- [ ] Tests:
  - `test_lambda_k_returns_finite_values_on_small_k_grid`
  - `test_lambda_k_symmetry_k_and_minus_k`
  - `test_ar1_dispersion_is_rotation_consistent_approximately`
  - `test_ar2_dispersion_prefers_expected_axis_when_anisotropy_present`
  - `test_homogeneous_dispersion_json_schema`
- [ ] Gate:
  - numerical unresolved / root-search failure = **stop and fix**;
  - numerically stable but no finite-k tendency = background negative / calibration caveat;
  - lack of a Brunel-like finite-k tendency alone does **not** block finite-core maps.

## TDD-5: Jacobian Builder / JVP

Goal: prove the linear operator is correct before solving eigenmodes.

- [ ] Start without explicit axonal delays.
- [ ] Include AMPA/GABA synaptic state dynamics.
- [ ] Use local diagonal LIF gains `dPhi/dmu`.
- [ ] Keep microscopic SNN spike/reset state out of `J`.
- [ ] Provide dense debug mode for tiny grids and matrix-free `LinearOperator` for real grids.
- [ ] Tests:
  - `test_dense_jacobian_shape`
  - `test_linear_operator_matvec_matches_dense`
  - `test_jvp_matches_finite_difference_tiny_grid`
  - `test_synaptic_blocks_have_expected_time_constant_signs`
  - `test_inhibitory_blocks_have_expected_negative_effect_on_rE`
  - `test_no_core_jacobian_eigs_match_homogeneous_dispersion_samples`
  - `test_core_excitability_increases_growth_of_core_overlap_mode`

## TDD-6: Eigenpair Extraction With Left/Right Modes

Goal: lock eigensolver residuals, sorting, conjugate handling, and non-normal diagnostics.

- [ ] Use `scipy.sparse.linalg.eigs` / `LinearOperator` for leading eigenpairs.
- [ ] Compute right eigenvectors and left / adjoint eigenvectors.
- [ ] Normalize left/right pairs and record biorthogonality diagnostics.
- [ ] Define constants: `eig_residual_tol`, `biorthogonality_tol`, `max_condition_warning`.
- [ ] Tests:
  - `test_right_eigen_residual_norm_small`
  - `test_left_eigen_residual_norm_small`
  - `test_complex_conjugate_pairs_are_handled`
  - `test_modes_sorted_by_real_part`
  - `test_left_right_biorthogonality_after_normalization`
  - `test_unstable_or_failed_eigs_mark_unresolved`

## TDD-7: Synthetic Mode Metrics and Classifier

Goal: define metric semantics before reading real eigenmode plots.

- [ ] Implement metrics:
  - `growth`, `frequency_hz`, `spectral_gap`;
  - `core_overlap`;
  - `elongation_axis_score`;
  - `phase_gradient_axis_score`;
  - `globality`;
  - `off_axis_score`;
  - `core_controllability = |psi_m^T b_core|`;
  - `finite_time_gain = ||exp(J*T)b_core|| / ||b_core||`.
- [ ] Tests:
  - `test_core_localized_mode_has_high_core_overlap_low_globality`
  - `test_axial_elongated_ridge_has_high_elongation_axis_score`
  - `test_phase_gradient_wave_has_expected_phase_gradient_axis_score`
  - `test_global_low_k_mode_has_high_participation_ratio`
  - `test_off_axis_mode_is_not_called_axial`
  - `test_axis_score_handles_90deg_wavevector_vs_ridge_ambiguity`
  - `test_non_normal_toy_has_negative_alpha_but_high_finite_time_gain`
  - `test_core_controllability_uses_left_not_right_eigenvector`

## TDD-8: Single-Point Finite-Core Golden Cases

Goal: debug representative points before scanning a map.

- [ ] Primary golden cases use **single core**:
  - low core excitability + high inhibition -> stable/local;
  - moderate core excitability + normal inhibition -> axial/core-controllable;
  - high global disinhibition -> mixed/global candidate;
  - very high drive -> runaway/high-rate saturation candidate.
- [ ] Tests:
  - `test_low_excitability_point_is_stable_or_local`
  - `test_increasing_core_excitability_raises_core_overlap_or_growth`
  - `test_increasing_global_disinhibition_raises_globality_or_low_k_energy`
  - `test_high_rate_saturation_is_flagged_runaway_not_axial`

## TDD-9: Pilot Phase Map

Goal: scan only after local pieces work.

- [ ] Run a 3x3 smoke map first.
- [ ] Expand to a coarse 7x7x2 map only after the smoke map is stable.
- [ ] Output:
  - `finite_jacobian_grid.json`;
  - `mode_metrics.csv`;
  - phase-map preview figures.
- [ ] Tests:
  - `test_phase_map_3x3_runs_to_completion`
  - `test_all_grid_points_have_status`
  - `test_unresolved_fraction_below_threshold`
  - `test_mode_metrics_csv_has_required_columns`
  - `test_phase_map_has_nontrivial_variation_in_alpha_or_mode_class`

## TDD-10: Controls / Ablations

Goal: prove the phase-map structure is scaffold/core-specific.

- [ ] Run no-core homogeneous background.
- [ ] Run isotropic E->E (`AR=1`).
- [ ] Run off-axis core placement.
- [ ] Run shuffled core thresholds.
- [ ] Compare frozen slow-state samples vs baseline.
- [ ] Run weak/no recovery (`phi/gK` off where available).
- [ ] Write `control_summary.json`.
- [ ] Tests:
  - `test_no_core_does_not_reproduce_core_localized_story`
  - `test_ar1_weakens_or_removes_45deg_axial_preference`
  - `test_off_axis_core_rotates_or_weakens_axis_score_as_expected`
  - `test_shuffled_core_thresholds_do_not_create_same_clean_axis_consistently`
  - `test_controls_summary_contains_all_required_controls`

## TDD-11: Rate-Field Dynamic Spot Checks

Goal: verify linear predictions against nonlinear rate-field events.

- [ ] Pick stable/local, axial, mixed/preictal-like, and global/R4b-risk points.
- [ ] Run finite-pulse and noise-driven `integrate_lif_field` checks.
- [ ] Write `ratefield_spotcheck_summary.json`.
- [ ] Tests:
  - `test_axial_spectral_point_produces_axial_ratefield_response`
  - `test_global_spectral_point_produces_higher_active_fraction`
  - `test_stable_point_returns_to_baseline_after_pulse`
  - `test_runaway_risk_point_is_flagged_if_no_return`

## TDD-12: SNN Frozen-State Spot Checks

Goal: verify phase-map predictions in actual SNN pilots, still without full exploration.

- [ ] Map selected phase-map parameters to SNN controls with documented transforms.
- [ ] Run short frozen-state SNN pilots.
- [ ] Classify R0/R1/R2/R3/R4a/R4b, return-to-baseline, early recruitment axis, duration,
  and active mass.
- [ ] Write `snn_spotcheck_summary.json`.
- [ ] Tests:
  - `test_snn_param_mapping_is_documented_for_each_point`
  - `test_snn_runs_emit_classification_fields`
  - `test_axial_point_not_systematically_r4b`
  - `test_global_risk_point_has_higher_active_mass_than_axial_point`
  - `test_systematic_spectrum_snn_mismatch_triggers_stop_status`

## TDD-13: Mode / Event Projection Into M3B Readout Space

Goal: reconnect the spectral mechanism to Round-1 bridge machinery.

- [ ] First run mock-mode schema smoke tests; do not wait until real modes are available.
- [ ] Convert representative `phi_m^E(x)` to virtual-SEEG-compatible model records.
- [ ] Convert representative SNN events to event-level model records from envelopes/ranks.
- [ ] Run `compare_model_to_cohort`, geometry nulls, and ictal-early placement where appropriate.
- [ ] Write `mode_readout_projection.json`.
- [ ] Tests:
  - `test_mock_mode_to_virtual_seeg_record_schema`
  - `test_real_mode_projection_has_required_scalars_and_channels`
  - `test_snn_event_projection_uses_same_masked_readout_conventions`
  - `test_compare_model_to_cohort_runs_without_schema_adapter_hacks`
  - `test_geometry_null_failure_forces_placement_only_verdict`
- [ ] Claim ladder:
  - spectrum only = theoretical appendix;
  - spectrum + SNN event match = model-side spontaneous mechanism;
  - spectrum + SNN event match + readout/null pass = M3B bridge support.

## TDD-14: M3A Slow-Trajectory Overlay

> **2026-06-27 — the interface contract + its contract-layer TDD are already implemented.** The
> shared module `src/sef_hfo_m3_interface.py` (canonical doc
> `docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md`, tests
> `tests/test_sef_hfo_m3_interface.py`, 48 passing) hosts the overlay gate:
> `audit_m3a_interface()` → `m3a_interface_audit.json` (four required condition booleans +
> `overlay_verdict`) and `build_slow_trajectory_overlay()` which REFUSES (returns no rows) unless
> `overlay_verdict == phase_map_trajectory`. TDD-14 below is thereby split:
> - **CONTRACT-LAYER (done):** missing-trace / missing-mapping / present-but-failed-each-condition
>   refusals; strict `cond4`; `cond2` id+transform identity; ≤5 % out-of-range gate; overlay schema;
>   `phenotype-positive-without-mapping → mechanism_candidate_only`; structural no-claim. The old
>   `test_no_slow_trajectory_claim_without_m3a_handoff` is re-bound to STRUCTURED fields (overlay rows
>   == 0 AND verdict != phase_map_trajectory), not a STATUS.md prose grep.
> - **RUNNER-LAYER (deferred to the M3A-A2 worktree, NOT faked here):** the engine-sign check
>   (`test_m3a_mapping_signs_match_rate_helpers`), the phenotype-movement determination, and
>   `tail_to_baseline` "absolute" computation. These must emit booleans/enums that the contract-layer
>   tests then assert on. See the contract doc §7 partition.

Only start after M3A produces usable slow-state traces.

- [ ] Read M3A `slow_to_rate_mapping.json`, `phase_trajectory.csv`, `event_phase_samples.csv`,
  event classes, and pre/onset/peak/end/post samples.
- [ ] Audit that the phase map was built over the same coordinate definitions/ranges.
- [ ] Test relation to `alpha_1=0`, axial/global crossing, spectral-gap collapse, and finite-time
  gain threshold.
- [ ] Write `m3a_interface_audit.json`.
- [ ] Write `slow_trajectory_overlay.csv`.
- [ ] Tests:
  - `test_m3a_overlay_refuses_missing_slow_traces`
  - `test_m3a_overlay_refuses_missing_slow_to_rate_mapping`
  - `test_slow_state_to_phase_coords_transform_is_documented`
  - `test_phase_trajectory_samples_are_in_map_or_flagged`
  - `test_overlay_csv_contains_event_stage_and_phase_coords`
  - `test_no_slow_trajectory_claim_without_m3a_handoff`

## TDD-15: Figures / Verdict / Claim Audit

Required figures:

- [ ] `figures/homogeneous_dispersion.png`.
- [ ] `figures/example_modes.png`.
- [ ] `figures/phase_map_mode_class.png`.
- [ ] `figures/phase_map_gap_gain.png`.
- [ ] `figures/non_normal_gain_controllability.png`.
- [ ] `figures/mode_readout_projection.png`.
- [ ] `figures/snn_spotcheck_grid.png`.
- [ ] `figures/slow_trajectory_overlay.png` if M3A is available.
- [ ] Chinese `figures/README.md`.
- [ ] Tests:
  - `test_required_artifacts_exist`
  - `test_required_figures_exist_or_are_marked_na`
  - `test_verdict_category_is_one_of_allowed_values`
  - `test_full_bridge_requires_phase_map_snn_m3a_readout_null_pass`
  - `test_no_forbidden_claims_in_status_and_readme`

Verdict categories:

- **SPM-PASS full bridge**: finite phase map predicts SNN spot checks, M3A trajectory moves through
  the predicted regions, and mode/event readout passes the relevant M3B placement/null tests.
- **SPM-PASS spontaneous mechanism**: phase map predicts frozen-state SNN spot checks, but readout
  bridge or M3A trajectory is incomplete.
- **SPM-PASS frozen map**: phase map is coherent and controls pass, but no validated SNN/M3A
  trajectory exists yet.
- **SPM-BOUNDED negative**: map is numerically valid but does not show an axial-to-global/mixed
  transition in plausible ranges.
- **SPM-MODEL mismatch**: spectral predictions do not match rate-field/SNN dynamics.
- **SPM-UNRESOLVED**: operating point or eigenmode computation is too unstable to interpret.

## Stop Rules

Stop and write a recap before expanding if:

- homogeneous dispersion is unresolved or inconsistent with the current LIF operating point;
- finite-Jacobian modes are mostly numerical artifacts;
- the phase map requires parameter ranges outside the SNN's valid regime;
- left/right biorthogonality or finite-time gain is numerically unstable;
- controls reproduce the same result, making the scaffold/core interpretation non-specific;
- SNN spot checks show only R4b tonic runaway;
- mode/event readout fails geometry nulls and the draft tries to call it a patient bridge;
- M3A handoff is absent but the draft starts making slow-trajectory claims.
