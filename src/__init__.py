"""
Keep package import lightweight.

This repo is used both as:
  - a library (importing plotting utilities, cached-result analysis, etc.)
  - a pipeline runner (needs heavier deps like `mne`)

Importing `src` should NOT eagerly import heavy dependencies. We provide lazy access to the
group-event analysis API via PEP 562 `__getattr__`.
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
    "build_epilepsiae_seizure_intervals": (
        "src.interictal_synchrony_aggregation",
        "build_epilepsiae_seizure_intervals",
    ),
    "annotate_epilepsiae_sync_blocks": (
        "src.interictal_synchrony_aggregation",
        "annotate_epilepsiae_sync_blocks",
    ),
    "aggregate_epilepsiae_sync_rows": (
        "src.interictal_synchrony_aggregation",
        "aggregate_epilepsiae_sync_rows",
    ),
    "run_epilepsiae_sync_aggregation": (
        "src.interictal_synchrony_aggregation",
        "run_epilepsiae_sync_aggregation",
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
