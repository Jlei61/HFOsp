# Plotting sources

These are local copies of the numerical painters behind the current paper-ready panels. No original-repository imports, image restoration, or stored paper results are used. Figure constructors return a Figure; data-dependent sizes and labels are supplied by the notebook adapters.

- `fig1.py`: `results/paper-ready-figure/fig1/source/code_snapshot/scripts/plot_interictal_propagation.py` — `_viridis_with_lightgray_bad, _mask_phantom_cells, _plot_rank_heatmap, _plot_cluster_rank_fig4`.
- `fig2_rose.py`: `scripts/paper_figures/plot_interictal_spatial_information_gain.py` — `_histogram_proportions, _rose_max, draw_combined_probability_rose`.
- `fig4_search.py`: `results/paper-ready-figure/fig4/source/plot_core_count_optimization_progress.py` — `mean_std_trajectory, draw`.
- `fig4_errors.py`: `results/paper-ready-figure/fig4/source/fig4_parameter_response_assets.py` — `error_curves`.
- `fig4_similarity.py`: `results/paper-ready-figure/fig4/source/build_fig4_template_recovery_by_mode.py` — `draw_modes`.
- `fig4_matching.py`: `scripts/paper_figures/build_fig4_geometry_patient_comparison.py` — `rank_panel, matrix_panel`.
- `fig2_dynamic.py`: `scripts/plot_topic5_interictal_event_envelope_field.py` — `_static_frame_relative_values, _panel, _readout`.

- `fig1_spectrum.py`: accepted Fig1 snapshot `scripts/paper_figures/restore_fig1_legacy_spectrum.py` — `draw_spectrum`; channel count and labels are parameters instead of Y1 constants. `plot_fig1_single_hfo_schematic.py` — the mean-spectrogram and bin-edge functions; `build_fig1_three_column_review.py` — the numerical HFO painter, supplied with newly detected snippets.
- `fig4_circuit.py`: Fig4 `source/local_circuit_native_source.py` — `_draw_integrated_mechanism` and its local drawing helpers.
- `fig5.py`: `scripts/paper_figures/build_fig5_fig4f_historical_zm.py` — `marks, draw_slow, draw_a, draw_native_maps, clip_trajectory_segments, draw_zm_trajectory`. These implement the **2026-10-11 six-panel candidate**, not the older top-level unfiltered Fig5. Display times follow the newly measured entry, retaining the reference offsets. Runtime settings are from the frozen `native_metadata.json` and `display_selection.json`: 80–1000 Hz, P99.5, gain 0.65, 20-ms maps, and the recorded Z/M colors.
- `contact_display.py`: `src/snn_contact_display.py` — fixed contact ordering, axis and within-shaft centroid lines. The movie adapter follows `build_fig4_ef_gifs.py` and the accepted `build_supplementary_video2_model_propagation.py` paired rows; it renders new spikes instead of restoring frozen PNGs.

Adjacent analysis copies supply the original physical-mm smoother (`plot_contact_plane_static.py`), dynamic-frame selection (`plot_topic5_interictal_event_envelope_field.py`), baseline envelope normalization (`src/topic5_interictal_event_field.py`), and Fig1 spectrogram kernels (`fig1_spectrogram_utils.py`).

The adapters in `../fig1.py`, `../fig2.py`, `../fig4.py` and `../fig5.py` pass freshly computed arrays. They change canvas sizes, contact count, subject label and example times to fit the Y1 and E1146 demos; they do not import the original repository. Cohort summaries require multiple patients. Fig1 anatomy requires separate inputs. Fig5 clinical panels use the bundled raw E1146 excerpts; full cohort figures require the remaining patients.

- `onset_field.py`: `scripts/paper_figures/plot_fig3b_interictal_ictal_shared_field.py` — `_normalize_minmax, _draw_field`; `_event_field` and the shared-plane display convention from the Fig2 dynamic-field producer. The transverse sign follows the dominant anatomical component; limits use the original 35-mm minimum span and 12% padding. Fig5 uses this same Blues spatial painter, with min-max display and a separate actual-z colorbar view.

- `fig4_waveform.py`: `scripts/paper_figures/build_fig4_geometry_patient_comparison.py` — `waveform_panel`; only the artifact-saving tail is replaced by returning the Figure. Pair selection, filter, common scale, two event spans and axes are unchanged.

- `fig5_statistic.py`: `build_fig5g_full_ramp_energy.py::draw_stat`, `build_fig5_single_panel_energy_similarity.py::swarm`, and `plot_fig3_field_concordance_cohort_stat.py::_add_violin_box_points`. The one-patient adapter changes only the patient-count label.
- `analysis/paper_clinical.py`: pure timing/phase functions from `src/topic5_energy_timing.py`, `review_fig5_broadband_phase.py`, `audit_fig5_phase_alignment.py`, `analyze_fig5_ramp_windows.py`, `review_fig5_final_recruitment.py`, and `proportional_rest.py`. Filesystem loading/saving is replaced by raw-array arguments; the clinical phase, qualification and quiet-window calculations are retained. `analysis/clinical_fields.py` applies these to new spectra and currents, with the original full-contact-before-smoothing null and repeated event selection.

Fig4E is an explicitly reduced scatter demo, as requested by the author. It shows evaluated coordinates and recomputed J_joint; the 418-condition density painter is not invoked. Fig1 uses Y1; Fig2 and the model examples use E1146.

- `modeling/representative_events.py`: `plot_topic4_compact_english_acceptance.py::mode_medoid` from the archived Fig4 source. The original shaft-balanced rank/participation rule is reapplied to newly simulated events for Video2.
