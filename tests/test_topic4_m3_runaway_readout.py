import numpy as np

from scripts.paper_figures.plot_fig_m3a_v2_1_qigk_runaway_transition_gif import (
    ProtocolConfig,
    _effective_core_radii,
)
from scripts.paper_figures.plot_fig_topic4_early_recruitment_readout import (
    FIELD_SIGMA_MM,
    LATENCY_CMAP,
    TRACE_BAND_HZ,
    _load_mode_pair,
    _normalize_minmax,
    _pre_runaway_burst_trace,
    _shared_template_plane,
)
from scripts.run_topic4_m3_runaway_readout import (
    _neuron_template_latency,
    _ordinal_contact_rank,
    _runaway_window,
    _template_latency,
)


def test_runaway_window_stops_before_next_external_pulse():
    start, end, next_pulse = _runaway_window(
        1109.8, np.array([940.0, 1075.0, 1210.0]), 1499.9, cap_ms=100.0)
    assert start == 1109.8
    assert end == 1209.8
    assert next_pulse == 1210.0


def test_template_latency_requires_contact_in_every_reference_event():
    times = np.arange(0.0, 200.0, 1.0)
    z = np.zeros((len(times), 3), float)
    # Two tempB events: contacts 0/1 are readable in both; contact 2 only once.
    z[20, 0] = 1.0
    z[30, 1] = 1.0
    z[40, 2] = 1.0
    z[120, 0] = 1.0
    z[130, 1] = 1.0
    rows = [
        {"source": "tempB", "t0": 0.0, "qualifies_local": True},
        {"source": "tempB", "t0": 100.0, "qualifies_local": True},
    ]
    latency, stack, event_t0, support = _template_latency(times, z, rows, "tempB")
    np.testing.assert_allclose(latency[:2], [20.0, 30.0])
    assert np.isnan(latency[2])
    assert stack.shape == (2, 3)
    np.testing.assert_array_equal(event_t0, [0.0, 100.0])
    np.testing.assert_array_equal(support, [2, 2, 1])


def test_single_event_contact_rank_is_ordinal_not_milliseconds():
    latency_ms = np.array([35.0, 4.0, np.nan, 18.0])
    rank = _ordinal_contact_rank(latency_ms)
    np.testing.assert_array_equal(rank[[0, 1, 3]], [3.0, 1.0, 2.0])
    assert np.isnan(rank[2])
    assert np.nanmax(rank) == 3.0


def test_subject_core_scale_defaults_to_circle_and_can_become_ellipse():
    substrate = {"layout": {"core_r": 0.75}}
    circular = ProtocolConfig(core_radius_scale=3.0)
    np.testing.assert_allclose(_effective_core_radii(substrate, circular), [2.25, 2.25])
    ellipse = ProtocolConfig(core_radius_scale=3.0, core_transverse_scale=5.0)
    np.testing.assert_allclose(_effective_core_radii(substrate, ellipse), [2.25, 3.75])


def test_neuron_latency_uses_real_first_spikes_and_repeated_event_support():
    times = np.arange(0.0, 200.0, 1.0)
    spikes = np.zeros((times.size, 3), bool)
    spikes[10, 0] = True
    spikes[20, 1] = True
    spikes[30, 2] = True
    spikes[115, 0] = True
    spikes[125, 1] = True
    rows = [
        {"source": "tempB", "t0": 0.0, "qualifies_local": True},
        {"source": "tempB", "t0": 100.0, "qualifies_local": True},
    ]
    latency, stack, event_t0, support = _neuron_template_latency(
        times, spikes, rows, "tempB"
    )
    np.testing.assert_allclose(latency[:2], [12.5, 22.5])
    assert np.isnan(latency[2])
    assert stack.shape == (2, 3)
    np.testing.assert_array_equal(event_t0, [0.0, 100.0])
    np.testing.assert_array_equal(support, [2, 2, 1])


def test_model_field_reuses_fig3b_display_contract():
    assert LATENCY_CMAP == "viridis"
    assert FIELD_SIGMA_MM == 3.0
    values = _normalize_minmax(np.array([2.0, 4.0, np.nan, 6.0]))
    np.testing.assert_allclose(values[[0, 1, 3]], [0.0, 0.5, 1.0])
    assert np.isnan(values[2])


def test_trace_display_exposes_signed_hfo_band_cycles():
    times = np.arange(0.0, 500.0, 0.5)
    target = np.sin(2.0 * np.pi * 50.0 * times / 1000.0)
    slow = 2.0 + 0.3 * np.sin(2.0 * np.pi * 3.0 * times / 1000.0)
    arrays = {"times_ms": times, "lfp_trace": (slow + target)[:, None]}
    shown = _pre_runaway_burst_trace(arrays, onset_ms=400.0)[:, 0]
    assert TRACE_BAND_HZ == (30.0, 80.0)
    assert shown.min() < 0.0 < shown.max()
    assert np.corrcoef(shown[100:-100], target[100:-100])[0, 1] > 0.98


def test_registered_plane_preserves_complete_e1146_layout_without_tb_mirroring():
    geometry = {
        "contacts": np.array([[18.0, 8.0], [10.0, 8.0], [2.0, 8.0]]),
        "reg": np.asarray(
            {"axis_unit": [1.0, 0.0], "center": [10.0, 8.0]}, dtype=object
        ),
        "L": np.asarray(20.0),
    }
    points, xlim, ylim, axis, _, _ = _shared_template_plane(geometry, "tempB")
    np.testing.assert_allclose(axis, [1.0, 0.0])
    assert points[0, 0] > points[1, 0] > points[2, 0]
    assert xlim == (-10.0, 10.0)
    assert ylim == (-10.0, 10.0)


def test_mode_pair_uses_same_three_seed_cohort_and_registered_states():
    labels = ["baseline_1000ms", "pre_onset_100ms"]
    arrays = {}
    per_seed = {}
    for seed, scale in [(1, 1.0), (3, 2.0), (4, 3.0)]:
        arrays[f"{seed}__resolved"] = np.array([True, True])
        arrays[f"{seed}__fields"] = np.stack(
            [np.full((2, 2), scale), np.full((2, 2), 10.0 * scale)]
        )
        per_seed[str(seed)] = {
            "records": [
                {"label": labels[0], "op_status": "resolved", "axis_score": 0.1,
                 "globality": 0.9, "time_to_runoff_ms": -4000.0},
                {"label": labels[1], "op_status": "resolved", "axis_score": 0.8,
                 "globality": 0.2, "time_to_runoff_ms": -100.0},
            ]
        }
    summary = {"labels": labels, "seeds": [1, 3, 4], "per_seed": per_seed}
    fields, metrics, seeds = _load_mode_pair(arrays, summary)
    np.testing.assert_allclose(fields["baseline"], 2.0)
    np.testing.assert_allclose(fields["pre_onset"], 20.0)
    assert seeds == [1, 3, 4]
    assert np.isclose(metrics["baseline"]["globality_mean"], 0.9)
    assert np.isclose(metrics["pre_onset"]["axis_score_mean"], 0.8)


def test_mode_pair_fails_closed_when_one_seed_is_unresolved():
    labels = ["baseline_1000ms", "pre_onset_100ms"]
    arrays = {
        "1__resolved": np.array([True, False]),
        "1__fields": np.ones((2, 2, 2)),
    }
    summary = {
        "labels": labels,
        "seeds": [1],
        "per_seed": {"1": {"records": [
            {"label": labels[0], "op_status": "resolved", "axis_score": 0.1,
             "globality": 0.9, "time_to_runoff_ms": -4000.0},
            {"label": labels[1], "op_status": "unresolved", "axis_score": None,
             "globality": None, "time_to_runoff_ms": -100.0},
        ]}},
    }
    with np.testing.assert_raises(ValueError):
        _load_mode_pair(arrays, summary)
