import numpy as np

from src.early_recruitment_readout import (
    arrival_field,
    compare_arrival_to_energy,
    early_energy_field,
    permutation_null,
    positive_excess,
    register_source_grid_to_subject_sheet,
)

# NOTE: ported from codex/topic4-early-readout together with
# src/early_recruitment_readout.py (8h-prompt §0/§5 "port the minimal common
# module and its test"). The upstream file also carried one test of an
# out-of-scope M2 integrator hook (`test_m2_integrator_frame_hook`); it is
# dropped here because the MZ bridge does not use `src.topic4_criticality_m2`.
# The 8 tests below exercise the generic readout library the bridge reuses.


def _travelling_wave():
    times = np.arange(0.0, 11.0, 1.0)
    centers = np.array([1.0, 3.0, 5.0, 7.0])
    frames = np.column_stack([np.exp(-0.5 * ((times - c) / 1.0) ** 2) for c in centers])
    return times, frames


def test_positive_excess_drops_suppression_and_checks_shape():
    kick = np.array([[2.0, 0.0], [1.0, 3.0]])
    ctrl = np.array([[1.0, 1.0], [2.0, 1.0]])
    np.testing.assert_allclose(positive_excess(kick, ctrl), [[1.0, 0.0], [0.0, 2.0]])


def test_arrival_field_recovers_wave_order_and_masks_weak_tail():
    times, wave = _travelling_wave()
    weak = 0.01 * wave[:, :1]
    field = arrival_field(np.column_stack([wave, weak]), times,
                          peak_fraction=0.5, participation_fraction=0.1)
    assert np.all(np.diff(field.arrival_ms[:4]) > 0)
    assert field.participating[:4].all()
    assert not field.participating[4]
    assert np.isnan(field.arrival_ms[4])


def test_fixed_window_energy_fails_closed_on_escape_or_incomplete_window():
    times, wave = _travelling_wave()
    good = early_energy_field(wave, times, (2.0, 8.0))
    assert good.status == "eligible" and np.all(np.isfinite(good.energy))
    escaped = early_energy_field(wave, times, (2.0, 8.0), escape_at_ms=7.0)
    assert escaped.status == "ineligible_escape_before_window_end"
    assert np.isnan(escaped.energy).all()
    incomplete = early_energy_field(wave, times, (2.0, 12.0))
    assert incomplete.status == "ineligible_incomplete_window"


def test_earlier_locations_hotter_has_positive_earliness_stat_and_perfect_topk():
    arrival = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    energy = np.array([9.0, 7.0, 5.0, 1.0, 100.0])
    out = compare_arrival_to_energy(arrival, energy, top_k=2)
    assert out["status"] == "eligible"
    assert np.isclose(out["arrival_energy_spearman"], -1.0)
    assert np.isclose(out["earliness_energy_spearman"], 1.0)
    assert out["field_cosine"] > 0.98
    assert np.isclose(out["top_k_overlap"], 1.0)


def test_comparison_reports_insufficient_and_degenerate_separately():
    few = compare_arrival_to_energy([1, np.nan, 3], [3, 2, np.nan], min_points=3)
    assert few["status"] == "insufficient_support"
    flat = compare_arrival_to_energy([1, 2, 3], [1, 1, 1], min_points=3)
    assert flat["status"] == "degenerate_field"


def test_permutation_null_is_reproducible_and_group_constrained():
    arrival = np.arange(8.0)
    energy = np.arange(8.0)[::-1]
    groups = np.repeat(["A", "B"], 4)
    a = permutation_null(arrival, energy, groups=groups, n_permutations=200, seed=7)
    b = permutation_null(arrival, energy, groups=groups, n_permutations=200, seed=7)
    assert a == b
    assert a["observed"] == 1.0
    assert a["effective_shuffle_n"] == 8
    assert a["method"] == "exact"
    assert a["n_unique_possible"] == 24 * 24
    assert 0.0 < a["p_one_sided"] <= 1.0


def test_subject_sheet_registration_maps_source_and_axis_anchor_exactly():
    theta = np.pi / 4.0
    u = np.array([np.cos(theta), np.sin(theta)])
    source = np.array([3.5, 8.5])
    sink = np.array([16.5, 8.5])
    points = np.vstack([np.zeros(2), 2.5 * u, np.array([0.5, -0.5])])
    mapped, transform = register_source_grid_to_subject_sheet(
        points,
        model_axis_theta_rad=theta,
        subject_source_xy=source,
        subject_sink_xy=sink,
        model_axis_anchor_mm=2.5,
    )
    np.testing.assert_allclose(mapped[0], source, atol=1e-12)
    np.testing.assert_allclose(mapped[1], sink, atol=1e-12)
    assert np.isclose(transform["scale"], np.linalg.norm(sink - source) / 2.5)


def test_subject_sheet_registration_rejects_degenerate_geometry():
    with np.testing.assert_raises(ValueError):
        register_source_grid_to_subject_sheet(
            np.zeros((2, 2)),
            model_axis_theta_rad=0.0,
            subject_source_xy=[1.0, 1.0],
            subject_sink_xy=[1.0, 1.0],
            model_axis_anchor_mm=1.0,
        )
