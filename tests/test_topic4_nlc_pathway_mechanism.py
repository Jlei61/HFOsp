from __future__ import annotations

import numpy as np

from scripts.paper_figures.plot_fig4c_nlc_pathway_ablation import (
    _metric_arrays,
)
from src.topic4_nlc_pathway_mechanism import (
    ARM_IDS,
    bootstrap_mean,
    event_aligned_pathway_readout,
    factorial_bootstrap,
    network_mode_endpoints,
    paired_bootstrap,
)


def test_fig4c_displays_mode_shares_and_kmeans_alignment_as_percentages():
    rows = {arm: {} for arm in ARM_IDS}
    for index, arm in enumerate(ARM_IDS):
        rows[arm]["7"] = {
            "TA_like_count": 1 + index,
            "TB_like_count": 3 - index if index < 3 else 1,
            "natural_alignment": 0.5 + 0.1 * index,
            "ood_fraction_returned": 0.4 - 0.1 * index,
        }
    metrics = _metric_arrays(rows, [7])
    assert list(metrics) == [
        "Mode 1 share (%)", "Mode 2 share (%)", "KMeans match (%)", "OOD (%)",
    ]
    np.testing.assert_allclose(
        metrics["Mode 1 share (%)"] + metrics["Mode 2 share (%)"], 100.0,
    )
    np.testing.assert_allclose(metrics["KMeans match (%)"][0], [50, 60, 70, 80])
    np.testing.assert_allclose(metrics["OOD (%)"][0], [40, 30, 20, 10])


def test_mode_endpoints_count_absent_mode_as_zero():
    assignments = {
        "labels": np.array([0, 0, 1]),
        "clean": np.array([True, True, False]),
        "returned": np.array([True, True, True]),
        "ood": np.array([False, False, True]),
    }
    row = network_mode_endpoints(assignments, duration_ms=20000.0)
    assert row["TA_like_count"] == 2
    assert row["TB_like_count"] == 0
    assert row["TA_like_rate_hz"] == 0.1
    assert row["TB_like_fraction"] == 0.0
    assert row["ood_fraction_returned"] == 1 / 3


def test_paired_and_factorial_bootstrap_preserve_pairing():
    node = np.array([1.0, 2.0, 3.0])
    ee = node + 1.0
    etoi = node + 2.0
    joint = node + 3.5
    paired = paired_bootstrap(ee, node, draws=128, seed=4)
    assert paired["mean_delta"] == 1.0
    assert paired["q05"] == paired["q95"] == 1.0
    interaction = factorial_bootstrap(
        node, ee, etoi, joint, draws=128, seed=4,
    )
    assert interaction["mean"] == 0.5
    mean = bootstrap_mean(node, draws=128, seed=4)
    assert mean["n_networks"] == 3


def test_event_aligned_readout_baseline_corrects_each_event():
    time = np.arange(0.0, 100.0, 1.0)
    trace = np.zeros_like(time)
    trace[40:50] = 2.0
    result = event_aligned_pathway_readout(
        time,
        {"population_rate_I_hz": trace},
        np.array([40.0]),
        np.array([1]),
        np.array([True]),
        event_window_ms=(-20.0, 40.0),
        baseline_window_ms=(-20.0, -5.0),
        summary_windows_ms={"ignition": (0.0, 10.0)},
        trace_dt_ms=1.0,
    )
    assert result["modes"]["TA-like"]["n_events"] == 0
    tb = result["modes"]["TB-like"]
    assert tb["n_events"] == 1
    assert tb["traces"]["population_rate_I_hz"]["windows"]["ignition"] == 2.0
