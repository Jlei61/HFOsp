"""
HFOsp — interictal HFO propagation analysis library.

Module map (src/)
=================

DATA LAYER
  preprocessing              EDF / Epilepsiae .data+.head loaders, re-referencing,
                             notch filter, seizure detector
  epilepsiae_dataset         Epilepsiae inventory, block-SQL contract, sync manifest
  yuquan_dataset             Yuquan 24h record loader (mirrors Epilepsiae API)
  seeg_coord_loader          3D SEEG coordinate loader v2 (7 hard invariants;
                             Yuquan → fs_native_ras_mm, Epilepsiae → mni152_1mm)
  atlas_loading              Layer A v2.3 per-subject JSON + onset-matrix loader

DETECTION & CORE PIPELINE
  hfo_detector               BQK HFO detector → legacy-compatible _gpu.npz writer
  group_event_analysis       Pack detections → 500 ms group-event windows, lagPat
                             matrices, centroid / lag-rank (legacy-aligned)
  lagpat_rank_audit          ⚠ Topic 0 phantom-rank fix: build_masked_kmeans_features()
                             is required before any KMeans step (see docs/topic0_*)

TOPIC 1 — Within-Event Dynamics: Propagation + Synchrony
  interictal_propagation     KMeans cluster stereotypy (adaptive k-scan, split-half /
                             odd-even), template anchoring, rate-state coupling
                             [PR-2 / PR-2.5 / PR-6 / PR-7 core; use_masked_features=True]
  rank_displacement          Signed rank displacement + swap-k metrics [PR-6 supplement]
  template_anatomical_anchoring  PR-6 endpoint anatomical anchoring (statistical layer)
  template_temporal_pairing  PR-7 template antagonistic temporal pairing
  cluster_geometry           PCA + UMAP embeddings of cluster feature matrices
  interictal_synchrony       PR-4 event-level synchrony metrics (lagPat-based)
  interictal_synchrony_aggregation  PR-5 / PR-5b aggregation: seizure-interval annotation,
                                    Epilepsiae + Yuquan
  interictal_synchrony_analysis    PR-6 interval-first synchrony statistics + visualization
  topic1_topic5_bridge       Q1 / Q1b / Q3 bridge between Topic 1 templates and
                             Topic 5 seizure subtypes

TOPIC 2 — Between-Event Dynamics: Periodicity
  event_periodicity          IEI + population-event PSD; specparam; surrogate tools
                             [CLI: scripts/run_event_periodicity.py]

TOPIC 3 — Spatial SOZ Modulation
  ictal_er_rank              PR-T3-1 v2.1 Layer A: ictal ER-rank (Page-Hinkley CUSUM)
  ictal_onset_extraction     PR-6-A ictal onset extraction primitives (sliding-window ER)
  data_driven_soz            PR-T3-1 v1.1 — ⚠ OBSOLETE (superseded 2026-05-03),
                             kept as audit trail only

TOPIC 4 — SEF-HFO Modeling  (docs/topic4_sef_itp_framework.md)
  topic4_attractor_diagnostics   Topic 1→4 bridge: PCA / principal-curve / λ₂ H3 probe;
                                 build_rank_feature_matrix [mask_phantom=True required]
  sef_itp_phase1             H1 source/sink compactness + H6 spatial geometry + H2 ingest
  sef_itp_phase2             H3 mark-independence + H4 rate/geometry instability
  sef_itp_phase3             H5 peri-ictal spatial recruitment
  sef_itp_phase3_trajectory  H5 v2 continuous 24h trajectory + alternative endpoints
  sef_itp_direction_axis     H2b direction-axis disambiguation (archive supplementary only)
  sef_hfo_field              Step 0 rate field — scaffold (data-anchoring pending)
  sef_hfo_stability          Step 0a delayed linear stability + dispersion map
  sef_hfo_pulse              Step 0b finite-pulse response + wavefront classifier

TOPIC 5 — Seizure Subtyping  (exploratory)
  ictal_seizure_clustering   PR-1 per-subject seizure clustering on ER-onset dissimilarity
  ictal_zer_features         PR-1 Step 2 z-ER binned tensor features
  ictal_seizure_plotting     PR-1 MDS embedding + subtype color palette rendering

SHARED UTILITIES
  visualization              Multi-channel SEEG waveforms + HFO overlays (topic-agnostic)
  network_analysis           HFO co-activation network — direction-first pipeline (exploratory)
  plot_style                 Shared publication-quality style (Nature/Science conventions)
  atomic_io                  Atomic JSON write helpers for long-running cohort drivers

Import note: ``import src`` is lightweight — heavy modules load only on first attribute
access via PEP 562 ``__getattr__``.  See _LAZY_EXPORTS below for the lazy API surface.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

__version__ = "0.1.0"


# Lazy-exported symbols (module, attr)
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # epilepsiae_dataset public API
    "EpilepsiaePaths": ("src.epilepsiae_dataset", "EpilepsiaePaths"),
    "EpilepsiaeInventory": ("src.epilepsiae_dataset", "EpilepsiaeInventory"),
    "EpilepsiaeTimeConfig": ("src.epilepsiae_dataset", "EpilepsiaeTimeConfig"),
    "resolve_epilepsiae_timezone": ("src.epilepsiae_dataset", "resolve_epilepsiae_timezone"),
    "survey_epilepsiae_dataset": ("src.epilepsiae_dataset", "survey_epilepsiae_dataset"),
    "build_epilepsiae_sync_subject_manifest": (
        "src.epilepsiae_dataset",
        "build_epilepsiae_sync_subject_manifest",
    ),
    "load_epilepsiae_sync_subject_manifest": (
        "src.epilepsiae_dataset",
        "load_epilepsiae_sync_subject_manifest",
    ),
    "save_epilepsiae_inventory": ("src.epilepsiae_dataset", "save_epilepsiae_inventory"),
    "save_epilepsiae_sync_subject_manifest": (
        "src.epilepsiae_dataset",
        "save_epilepsiae_sync_subject_manifest",
    ),
    # group_event_analysis public API
    "EventWindow": ("src.group_event_analysis", "EventWindow"),
    "GroupEventAnalyzer": ("src.group_event_analysis", "GroupEventAnalyzer"),
    "LagMatrices": ("src.group_event_analysis", "LagMatrices"),
    "MNERawOnDemandLoader": ("src.group_event_analysis", "MNERawOnDemandLoader"),
    "build_windows_from_detections": ("src.group_event_analysis", "build_windows_from_detections"),
    "build_windows_from_packed_times": ("src.group_event_analysis", "build_windows_from_packed_times"),
    "bqk_detect_and_compare_windows_to_packed": ("src.group_event_analysis", "bqk_detect_and_compare_windows_to_packed"),
    "compare_to_reference_lagpat": ("src.group_event_analysis", "compare_to_reference_lagpat"),
    "compare_window_sets": ("src.group_event_analysis", "compare_window_sets"),
    "compare_channel_sets": ("src.group_event_analysis", "compare_channel_sets"),
    "compute_dense_rank": ("src.group_event_analysis", "compute_dense_rank"),
    "compute_centroid_matrix": ("src.group_event_analysis", "compute_centroid_matrix"),
    "filter_windows_by_min_channels": ("src.group_event_analysis", "filter_windows_by_min_channels"),
    "gpu_detections_and_compare_windows_to_packed": ("src.group_event_analysis", "gpu_detections_and_compare_windows_to_packed"),
    "select_core_channels_by_event_count": ("src.group_event_analysis", "select_core_channels_by_event_count"),
    "legacy_refine_channels_from_detections": ("src.group_event_analysis", "legacy_refine_channels_from_detections"),
    "legacy_refine_core_channels": ("src.group_event_analysis", "legacy_refine_core_channels"),
    "lag_rank_from_centroids": ("src.group_event_analysis", "lag_rank_from_centroids"),
    "validate_packedtimes_centroid_lagrank_against_lagpat": (
        "src.group_event_analysis",
        "validate_packedtimes_centroid_lagrank_against_lagpat",
    ),
    "precompute_envelope_cache": ("src.group_event_analysis", "precompute_envelope_cache"),
    "load_envelope_cache": ("src.group_event_analysis", "load_envelope_cache"),
    "compute_centroid_matrix_from_envelope_cache": ("src.group_event_analysis", "compute_centroid_matrix_from_envelope_cache"),
    # network_analysis public API
    "NetworkResult": ("src.network_analysis", "NetworkResult"),
    "build_hfo_network": ("src.network_analysis", "build_hfo_network"),
    "build_broad_graph": ("src.network_analysis", "build_broad_graph"),
    "compute_pairwise_association": ("src.network_analysis", "compute_pairwise_association"),
    "validate_propagation_velocity": ("src.network_analysis", "validate_propagation_velocity"),
    "inject_direction": ("src.network_analysis", "inject_direction"),
    "compare_ictal_interictal_networks": ("src.network_analysis", "compare_ictal_interictal_networks"),
    "compute_network_metrics": ("src.network_analysis", "compute_network_metrics"),
    "plot_network_topology_2d": ("src.network_analysis", "plot_network_topology_2d"),
    "plot_broad_graph_comparison": ("src.network_analysis", "plot_broad_graph_comparison"),
    "plot_delta_outflow": ("src.network_analysis", "plot_delta_outflow"),
    "plot_diff_adjacency_heatmap": ("src.network_analysis", "plot_diff_adjacency_heatmap"),
    "surrogate_significance_test": ("src.network_analysis", "surrogate_significance_test"),
    "compute_stability_weights": ("src.network_analysis", "compute_stability_weights"),
    "save_network_result": ("src.network_analysis", "save_network_result"),
    "load_network_result": ("src.network_analysis", "load_network_result"),
    "plot_outflow_bar_chart": ("src.network_analysis", "plot_outflow_bar_chart"),
    "plot_adjacency_heatmap": ("src.network_analysis", "plot_adjacency_heatmap"),
    "plot_edge_direction_summary": ("src.network_analysis", "plot_edge_direction_summary"),
    # interictal_synchrony public API
    "InterictalSynchronyResult": ("src.interictal_synchrony", "InterictalSynchronyResult"),
    "select_core_penumbra_mask": ("src.interictal_synchrony", "select_core_penumbra_mask"),
    "compute_event_synchrony_metrics": ("src.interictal_synchrony", "compute_event_synchrony_metrics"),
    "compute_adjacent_jaccard": ("src.interictal_synchrony", "compute_adjacent_jaccard"),
    "compute_interictal_synchrony": ("src.interictal_synchrony", "compute_interictal_synchrony"),
    "build_interictal_synchrony": ("src.interictal_synchrony", "build_interictal_synchrony"),
    "load_legacy_lagpat_group_analysis": (
        "src.interictal_synchrony",
        "load_legacy_lagpat_group_analysis",
    ),
    "build_interictal_synchrony_from_legacy_lagpat": (
        "src.interictal_synchrony",
        "build_interictal_synchrony_from_legacy_lagpat",
    ),
    "run_epilepsiae_interictal_synchrony_from_manifest": (
        "src.interictal_synchrony",
        "run_epilepsiae_interictal_synchrony_from_manifest",
    ),
    "run_yuquan_interictal_synchrony": (
        "src.interictal_synchrony",
        "run_yuquan_interictal_synchrony",
    ),
    "save_interictal_synchrony_summary": (
        "src.interictal_synchrony",
        "save_interictal_synchrony_summary",
    ),
    "save_interictal_synchrony_result": ("src.interictal_synchrony", "save_interictal_synchrony_result"),
    "load_interictal_synchrony_result": ("src.interictal_synchrony", "load_interictal_synchrony_result"),
    # interictal_synchrony_aggregation public API
    "EpilepsiaeSyncAggregationConfig": (
        "src.interictal_synchrony_aggregation",
        "EpilepsiaeSyncAggregationConfig",
    ),
    "YuquanSyncAggregationConfig": (
        "src.interictal_synchrony_aggregation",
        "YuquanSyncAggregationConfig",
    ),
    "build_epilepsiae_seizure_intervals": (
        "src.interictal_synchrony_aggregation",
        "build_epilepsiae_seizure_intervals",
    ),
    "build_yuquan_seizure_inventory": (
        "src.interictal_synchrony_aggregation",
        "build_yuquan_seizure_inventory",
    ),
    "build_yuquan_seizure_intervals": (
        "src.interictal_synchrony_aggregation",
        "build_yuquan_seizure_intervals",
    ),
    "annotate_epilepsiae_sync_events": (
        "src.interictal_synchrony_aggregation",
        "annotate_epilepsiae_sync_events",
    ),
    "annotate_yuquan_sync_events": (
        "src.interictal_synchrony_aggregation",
        "annotate_yuquan_sync_events",
    ),
    "aggregate_epilepsiae_sync_rows": (
        "src.interictal_synchrony_aggregation",
        "aggregate_epilepsiae_sync_rows",
    ),
    "aggregate_yuquan_sync_rows": (
        "src.interictal_synchrony_aggregation",
        "aggregate_yuquan_sync_rows",
    ),
    "run_epilepsiae_sync_aggregation": (
        "src.interictal_synchrony_aggregation",
        "run_epilepsiae_sync_aggregation",
    ),
    "run_yuquan_sync_aggregation": (
        "src.interictal_synchrony_aggregation",
        "run_yuquan_sync_aggregation",
    ),
    # interictal_synchrony_analysis (PR6) public API
    "load_event_rows": (
        "src.interictal_synchrony_analysis",
        "load_event_rows",
    ),
    "assign_fixed_window_positions": (
        "src.interictal_synchrony_analysis",
        "assign_fixed_window_positions",
    ),
    "compute_normalized_trajectory": (
        "src.interictal_synchrony_analysis",
        "compute_normalized_trajectory",
    ),
    "paired_window_test": (
        "src.interictal_synchrony_analysis",
        "paired_window_test",
    ),
    "trajectory_trend_test": (
        "src.interictal_synchrony_analysis",
        "trajectory_trend_test",
    ),
    "within_interval_trend_test": (
        "src.interictal_synchrony_analysis",
        "within_interval_trend_test",
    ),
    "run_pr6_analysis": (
        "src.interictal_synchrony_analysis",
        "run_pr6_analysis",
    ),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> Any:
    """
    Lazy attribute access for heavy modules.

    Example:
      from src import precompute_envelope_cache
    """
    if name in _LAZY_EXPORTS:
        mod_name, attr = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'src' has no attribute '{name}'")
