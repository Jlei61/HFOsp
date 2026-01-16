# HFOsp - High Frequency Oscillation Analysis Package
# For Yuquan 24h SEEG Dataset

__version__ = "0.1.0"

# Public API (keep imports lightweight)
from .group_event_analysis import (  # noqa: F401
    EventWindow,
    GroupEventAnalyzer,
    LagMatrices,
    MNERawOnDemandLoader,
    build_windows_from_detections,
    build_windows_from_packed_times,
    bqk_detect_and_compare_windows_to_packed,
    compare_to_reference_lagpat,
    compare_window_sets,
    compare_channel_sets,
    compute_dense_rank,
    compute_centroid_matrix,
    filter_windows_by_min_channels,
    gpu_detections_and_compare_windows_to_packed,
    select_core_channels_by_event_count,
    lag_rank_from_centroids,
    validate_packedtimes_centroid_lagrank_against_lagpat,
    precompute_envelope_cache,
    load_envelope_cache,
    compute_centroid_matrix_from_envelope_cache,
)
