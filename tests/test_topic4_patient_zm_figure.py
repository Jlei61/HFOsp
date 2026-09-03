import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.topic4_patient_zm_figure import draw_critical_manifold_trajectory


def _fixture():
    projection = {
        "manifold": {
            "eta_m": 0.02,
            "tau_m_ms": 12.5,
            "q_fold": 0.89,
            "mean_rate_e_hz_at_fold": 125.0,
        },
        "runs": [{"seed": 1842, "scientific_onset_ms": 400.0}],
    }
    time = np.arange(0.0, 1001.0, 10.0)
    rate = 40.0 + 350.0 / (1.0 + np.exp(-(time - 400.0) / 25.0))
    m = 12.5 * rate / 1000.0
    arrays = {
        "manifold_low_q": np.asarray([0.78, 1.0]),
        "manifold_low_rate_e_hz": np.asarray([0.03, 0.02]),
        "manifold_returned_q": np.asarray([0.83, 0.89]),
        "manifold_returned_rate_e_hz": np.asarray([8.0, 125.0]),
        "manifold_high_q": np.asarray([0.78, 0.89]),
        "manifold_high_rate_e_hz": np.asarray([370.0, 125.0]),
        "seed1842_time_ms": time,
        "seed1842_q_core": 1.0 - 0.225 * np.minimum(time / 500.0, 1.0),
        "seed1842_q_mean": 1.0 - 0.225 * np.minimum(time / 650.0, 1.0),
        "seed1842_M": m,
        "seed1842_rate_E_20ms_hz": rate,
    }
    return projection, arrays


def test_critical_manifold_panel_keeps_spatial_q_summaries_distinct():
    projection, arrays = _fixture()
    fig, ax = plt.subplots()
    metadata = draw_critical_manifold_trajectory(
        ax, projection, arrays, seed=1842,
        add_rate_colorbar=False, show_legend=True)
    labels = [line.get_label() for line in ax.lines]
    assert r"SNN $q_{core}$" in labels
    assert r"SNN $q_{mean}$" in labels
    assert metadata["high_branch_stability"] == (
        "delay-unstable in the audited closure")
    assert metadata["fold"]["q"] == 0.89
    plt.close(fig)


def test_critical_manifold_colorbar_stays_vector_for_tight_pdf_export():
    projection, arrays = _fixture()
    fig, ax = plt.subplots()
    draw_critical_manifold_trajectory(
        ax, projection, arrays, seed=1842,
        add_rate_colorbar=True, show_legend=False)
    colorbar_axis = fig.axes[-1]
    assert colorbar_axis is not ax
    assert all(not collection.get_rasterized()
               for collection in colorbar_axis.collections)
    plt.close(fig)
