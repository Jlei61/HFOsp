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
    "build_skeleton": ("src.network_analysis", "build_skeleton"),
    "inject_direction": ("src.network_analysis", "inject_direction"),
    "compute_network_metrics": ("src.network_analysis", "compute_network_metrics"),
    "plot_network_topology_2d": ("src.network_analysis", "plot_network_topology_2d"),
    "select_network_nodes": ("src.network_analysis", "select_network_nodes"),
    "surrogate_significance_test": ("src.network_analysis", "surrogate_significance_test"),
    "compute_stability_weights": ("src.network_analysis", "compute_stability_weights"),
    "save_network_result": ("src.network_analysis", "save_network_result"),
    "load_network_result": ("src.network_analysis", "load_network_result"),
    "plot_outflow_bar_chart": ("src.network_analysis", "plot_outflow_bar_chart"),
    "plot_adjacency_heatmap": ("src.network_analysis", "plot_adjacency_heatmap"),
    "plot_edge_direction_summary": ("src.network_analysis", "plot_edge_direction_summary"),
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
