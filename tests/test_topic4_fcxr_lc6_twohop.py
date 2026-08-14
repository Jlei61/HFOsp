import numpy as np
from scipy import sparse

from src.topic4_fcxr_lc6_surround import EToIGraph
from src.topic4_fcxr_lc6_twohop import (
    coarse_two_hop_operator,
    sample_two_hop_latencies,
    spatial_bins,
    summarize_two_hop_operator,
)


def _graphs():
    e2i = EToIGraph(
        np.array([[0, 1], [2, 3]], np.int32),
        np.ones((2, 2)),
        np.array([[2, 4], [3, 5]], np.int32),
    )
    i2e = EToIGraph(
        np.array([[0], [0], [1], [1]], np.int32),
        np.full((4, 1), 2.0),
        np.array([[6], [7], [8], [9]], np.int32),
    )
    return e2i, i2e


def test_coarse_operator_is_target_bin_by_source_bin_and_uses_magnitude():
    pos_e = np.array([[.1, .1], [.2, .1], [1.1, .1], [1.2, .1]])
    bins = spatial_bins(pos_e, sheet_size_mm=2.0, n_bins_axis=2)
    e2i, i2e = _graphs()
    operator = coarse_two_hop_operator(e2i, i2e, bins, n_e=4, n_i=2)
    assert operator.shape == (4, 4)
    assert np.isclose(operator.sum(), 16.0)
    # Each of two E targets behind I0 receives both source-bin edges at weight 2.
    assert operator[bins.cell_bin[0], bins.cell_bin[0]] == 8.0


def test_two_hop_summary_separates_center_and_surround():
    pos_e = np.array([[.1, .1], [.2, .1], [1.1, .1], [1.2, .1]])
    bins = spatial_bins(pos_e, sheet_size_mm=2.0, n_bins_axis=2)
    operator = sparse.csr_matrix(
        ([8.0, 3.0, 3.0, 8.0], ([0, 0, 2, 2], [0, 2, 0, 2])),
        shape=(4, 4),
    )
    summary = summarize_two_hop_operator(
        operator, bins, [1, 0], ee_sigma_parallel_mm=.5,
        ee_sigma_perpendicular_mm=.5, edge_margin_mm=.2,
    )
    assert summary["center_mass"] > 0.0
    assert summary["surround_mass"] > 0.0
    assert np.isfinite(summary["q_parallel_two_hop"])


def test_path_latency_uses_sum_of_two_engine_delay_steps():
    e2i, i2e = _graphs()
    audit = sample_two_hop_latencies(
        e2i, i2e, n_e=4, n_i=2, engine_dt_ms=.1,
        n_paths=2000, audit_seed=77,
    )
    assert .8 <= audit["median_ms"] <= 1.4
    assert audit["q95_ms"] <= 1.4000001
    assert audit["n_paths"] == 2000
