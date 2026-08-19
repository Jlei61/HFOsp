"""Perturbation geometry, descendant-only metrics, and counterfactual splices."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.snn_engine.checkpoint import digest  # noqa: E402
from src.topic4_zm_perturbation import (  # noqa: E402
    hotspot_compactness, in_window_ignition, response_metrics, select_packet,
    splice_checkpoint)


def _flat():
    envelope = np.zeros((15, 200), np.float32)
    return dict(dt_ms=0.1, envelope_probe=envelope, envelope_sham=envelope,
                envelope_dt_ms=2.0, inject_step=0, split_ms=50.0, window_ms=200.0)


def test_packet_is_the_nearest_cells_and_respects_the_radius():
    rng = np.random.default_rng(0)
    positions = rng.random((32000, 2)) * 20.0     # the real substrate's density
    mask = select_packet(positions, np.array([10.0, 10.0]), n_cells=64, radius_mm=1.2)
    assert mask.sum() == 64
    chosen = positions[mask]
    assert np.all(np.linalg.norm(chosen - 10.0, axis=1) <= 1.2)
    others = positions[~mask]
    inside = others[np.linalg.norm(others - 10.0, axis=1) <= 1.2]
    assert np.all(np.linalg.norm(chosen - 10.0, axis=1).max()
                  <= np.linalg.norm(inside - 10.0, axis=1).min() + 1e-12)


def test_packet_raises_when_the_disk_is_too_sparse():
    positions = np.array([[10.0, 10.0], [10.1, 10.0]])
    with pytest.raises(ValueError, match="insufficient"):
        select_packet(positions, np.array([10.0, 10.0]), n_cells=64, radius_mm=1.0)


def test_identical_probe_and_sham_give_exactly_zero_susceptibility():
    n_steps, n_e = 2000, 500
    spikes = np.zeros((n_steps, n_e), bool)
    spikes[::7, ::3] = True
    packet = np.zeros(n_e, bool); packet[:32] = True
    out = response_metrics({"E_spk_bool": spikes}, {"E_spk_bool": spikes},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 0.0
    assert out["excess_spikes_early"] == 0.0
    assert out["excess_spikes_late"] == 0.0
    assert out["contact_excess_energy"] == 0.0


def test_injected_spikes_alone_produce_exactly_zero_susceptibility():
    """The decisive regression: a packet that fires once and propagates to
    nothing must score 0, not `n_cells`."""
    n_steps, n_e, n_cells = 2000, 500, 256
    sham = np.zeros((n_steps, n_e), bool)
    probe = sham.copy()
    packet = np.zeros(n_e, bool)
    packet[:n_cells] = True
    probe[0, packet] = True                     # the injection, and nothing else
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 0.0
    assert out["excess_spikes_early"] == 0.0


def test_descendant_spikes_are_still_counted():
    n_steps, n_e, n_cells = 2000, 500, 32
    sham = np.zeros((n_steps, n_e), bool)
    probe = sham.copy()
    packet = np.zeros(n_e, bool)
    packet[:n_cells] = True
    probe[0, packet] = True                     # injection, removed
    probe[40, 400:410] = True                   # 10 descendants at 4 ms (early)
    probe[900, 400:405] = True                  # 5 descendants at 90 ms (late)
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=np.zeros((n_e, 2)), packet_mask=packet,
                           packet_xy=np.zeros(2), **_flat())
    assert out["susceptibility"] == 15.0
    assert out["excess_spikes_early"] == 10.0   # the 4 ms group
    assert out["excess_spikes_late"] == 5.0     # the 90 ms group


def test_susceptibility_is_the_sum_of_its_two_parts():
    rng = np.random.default_rng(2)
    n_steps, n_e = 2000, 300
    sham = rng.random((n_steps, n_e)) < 0.001
    probe = sham | (rng.random((n_steps, n_e)) < 0.002)
    packet = np.zeros(n_e, bool); packet[:16] = True
    out = response_metrics({"E_spk_bool": probe}, {"E_spk_bool": sham},
                           positions_e=rng.random((n_e, 2)) * 5.0,
                           packet_mask=packet, packet_xy=np.array([2.5, 2.5]),
                           **_flat())
    assert np.isclose(out["susceptibility"],
                      out["excess_spikes_early"] + out["excess_spikes_late"])
    assert 0.0 < out["r90_mm"] <= np.sqrt(2) * 5.0


def _active(pattern_ms, *, total_ms=200.0, dt=1.0, level=0.06):
    """Active-fraction series in 1 ms bins with the given ON spans."""
    n = int(round(total_ms / dt))
    series = np.zeros(n)
    for lo, hi in pattern_ms:
        series[int(lo / dt):int(hi / dt)] = level
    return series


def test_ignition_requires_the_event_to_be_absent_from_the_sham():
    """The network is above the common detector 41 % of the time, so an event
    in the probe branch is only evidence if the sham branch lacks it."""
    span = [(50.0, 90.0)]
    shared = in_window_ignition(_active(span), _active(span), active_dt_ms=1.0,
                                detector_threshold=0.02, inject_ms=0.0,
                                window_ms=200.0)
    assert shared["n_probe_events"] == 1 and shared["n_sham_events"] == 1
    assert shared["probe_attributable_event_200ms"] is False
    assert shared["e1_evaluable"] is True

    only = in_window_ignition(_active(span), _active([]), active_dt_ms=1.0,
                              detector_threshold=0.02, inject_ms=0.0,
                              window_ms=200.0)
    assert only["probe_attributable_event_200ms"] is True
    assert only["e1_evaluable"] is False


def test_a_brief_excursion_is_recorded_but_is_not_a_detector_event():
    """MIN_DUR_MS is 8 ms in the frozen detector. A 3 ms blip must not be able
    to invalidate an E1 site -- over-detecting ignition would strip good sites
    out of the map and could manufacture NO_SUBEVENT_PROBE_REGIME."""
    out = in_window_ignition(_active([(50.0, 53.0)]), _active([]), active_dt_ms=1.0,
                             detector_threshold=0.02, inject_ms=0.0, window_ms=200.0)
    assert out["n_probe_events"] == 0
    assert out["probe_attributable_event_200ms"] is False
    assert out["brief_threshold_excursion"] is True
    assert out["e1_evaluable"] is True


def test_events_closer_than_the_merge_gap_are_one_event():
    """MERGE_GAP_MS is 12 ms; two spans 5 ms apart are one event, not two."""
    out = in_window_ignition(_active([(40.0, 60.0), (65.0, 85.0)]), _active([]),
                             active_dt_ms=1.0, detector_threshold=0.02,
                             inject_ms=0.0, window_ms=200.0)
    assert out["n_probe_events"] == 1


def test_model_ictal_flag_uses_the_frozen_120hz_100ms_criterion():
    rate = np.concatenate([np.full(500, 10.0), np.full(2000, 300.0)])
    out = in_window_ignition(_active([]), _active([]), active_dt_ms=1.0,
                             detector_threshold=0.02, inject_ms=0.0,
                             window_ms=200.0, probe_rate_hz=rate, dt_ms=0.1)
    assert out["reached_model_ictal_200ms"] is True
    assert out["e1_evaluable"] is False
    quiet = in_window_ignition(_active([]), _active([]), active_dt_ms=1.0,
                               detector_threshold=0.02, inject_ms=0.0,
                               window_ms=200.0,
                               probe_rate_hz=np.full(2500, 10.0), dt_ms=0.1)
    assert quiet["reached_model_ictal_200ms"] is False


def test_splice_only_touches_the_named_slow_variable():
    pre = {"slow": {"z": np.full(4, 0.2), "m": np.full(4, 5.0)},
           "V": np.arange(4.0), "external_drive": {"next_step": 7}}
    base = {"slow": {"z": np.full(4, 0.9), "m": np.full(4, 0.1)},
            "V": np.zeros(4), "external_drive": {"next_step": 1}}
    out = splice_checkpoint(pre, base, mode="reset_z")
    assert np.allclose(out["slow"]["z"], 0.9)     # taken from baseline
    assert np.allclose(out["slow"]["m"], 5.0)     # kept from pre-ictal
    assert np.allclose(out["V"], np.arange(4.0))  # fast state untouched
    assert out["external_drive"]["next_step"] == 7
    assert np.allclose(pre["slow"]["z"], 0.2)     # donor not mutated
    assert out["off_manifold"] is True

    both = splice_checkpoint(pre, base, mode="reset_zm")
    assert np.allclose(both["slow"]["z"], 0.9) and np.allclose(both["slow"]["m"], 0.1)

    sufficiency = splice_checkpoint(pre, base, mode="slow_only")
    assert np.allclose(sufficiency["V"], np.zeros(4))
    assert np.allclose(sufficiency["slow"]["z"], 0.2)
    assert sufficiency["external_drive"]["next_step"] == 1

    native = splice_checkpoint(pre, base, mode="native_pre_ictal")
    assert native["off_manifold"] is False


def _digest_state(tag, rng):
    return {
        "schema": "topic4_snn_checkpoint_v1", "step": 20000 + tag,
        "absolute_time_ms": 2000.0 + tag,
        "V": rng.random(8), "ref": rng.integers(0, 4, 8).astype(np.int32),
        "s_E": rng.random(8), "I_E": rng.random(8),
        "s_I": rng.random(8), "I_I": rng.random(8),
        "ring_sE": rng.random((5, 8)), "ring_sI": rng.random((5, 8)),
        "xi": float(rng.random()), "rng_state": {"bit_generator": "PCG64", "n": tag},
        "ras_keep": np.array([0, 2, 4]), "es_ema": 3.0 + tag, "es_run": tag,
        "track_rec": False, "s_E_rec": None, "I_E_rec": None,
        "slow": {"kind": "MZSlowVars", "z": rng.random(8), "m": rng.random(8),
                 "I_I_last": rng.random(8), "step_index": 100 + tag,
                 "acc_n": 0, "acc_seen": 0, "acc_D": None, "acc_A": None},
        "external_drive": {"field_state": rng.random((4, 4)), "cached": rng.random(8),
                           "next_step": 30 + tag, "last_step": 29 + tag,
                           "rng_state": {"bit_generator": "PCG64", "n": 99 + tag}},
    }


def test_splice_leaves_every_non_slow_field_bit_identical():
    """Splice integrity is a bit-level property, not an assumption: apart from
    the named z/m arrays, the host's fast state, OU state, both RNG states, the
    delay rings and the time index must be untouched."""
    rng = np.random.default_rng(11)
    pre, base = _digest_state(1, rng), _digest_state(2, rng)
    for mode, host in (("reset_z", pre), ("reset_m", pre), ("reset_zm", pre),
                       ("slow_only", base)):
        out = splice_checkpoint(pre, base, mode=mode)
        host_no_slow = copy.deepcopy(host); host_no_slow["slow"] = None
        out_no_slow = copy.deepcopy(out); out_no_slow["slow"] = None
        out_no_slow.pop("off_manifold", None)
        out_no_slow.pop("splice_mode", None)
        assert digest(out_no_slow) == digest(host_no_slow), mode
        assert out["slow"]["step_index"] == host["slow"]["step_index"], mode
        assert np.array_equal(out["slow"]["I_I_last"], host["slow"]["I_I_last"]), mode


def test_hotspot_compactness_detects_a_planted_cluster():
    xy = np.stack(np.meshgrid(np.linspace(3, 17, 7), np.linspace(3, 17, 7)),
                  axis=-1).reshape(-1, 2)
    values = np.zeros(len(xy))
    corner = np.linalg.norm(xy - np.array([4.0, 4.0]), axis=1)
    values[np.argsort(corner)[:10]] = 100.0
    out = hotspot_compactness(xy, values, quantile=0.8, n_null=500, seed=1,
                              tail="high")
    assert out["p_value"] < 0.01
    assert out["observed_mean_pairwise_mm"] < out["null_mean_pairwise_mm"]


def _grid_with_corner_cluster(planted):
    """Corner cluster at `planted`; everything else ranked by distance from the
    sheet centre, so the OPPOSITE tail is the scattered perimeter. Constant
    filler would tie 39 of 49 sites at the cut and make the wrong-tail arm
    select almost the whole grid, which is degenerate rather than a control."""
    xy = np.stack(np.meshgrid(np.linspace(3, 17, 7), np.linspace(3, 17, 7)),
                  axis=-1).reshape(-1, 2)
    corner = np.linalg.norm(xy - np.array([4.0, 4.0]), axis=1)
    cluster = np.argsort(corner)[:10]
    centre = np.linalg.norm(xy - np.array([10.0, 10.0]), axis=1)
    values = centre / centre.max()          # 0 at centre, 1 at the perimeter
    values[cluster] = planted
    return xy, values


def test_a_clustered_DECREASE_is_found_by_the_low_tail_and_missed_by_the_high():
    """The endpoint is pre-ictal minus baseline and the round registers that it
    can fall. A clustered fall lives in the LOW tail; taking the high tail
    regardless clusters the LEAST affected sites and reports nothing."""
    xy, values = _grid_with_corner_cluster(-100.0)
    low = hotspot_compactness(xy, values, quantile=0.8, n_null=500, seed=1,
                              tail="low")
    high = hotspot_compactness(xy, values, quantile=0.8, n_null=500, seed=1,
                               tail="high")
    assert low["p_value"] < 0.01
    assert low["observed_mean_pairwise_mm"] < low["null_mean_pairwise_mm"]
    assert high["p_value"] > 0.05, "the wrong tail must not find the cluster"


def test_the_tail_is_required_so_no_caller_silently_keeps_the_old_path():
    xy, values = _grid_with_corner_cluster(100.0)
    with pytest.raises(TypeError):
        hotspot_compactness(xy, values, quantile=0.8, n_null=10, seed=1)
    with pytest.raises(ValueError, match="registered direction"):
        hotspot_compactness(xy, values, quantile=0.8, n_null=10, seed=1,
                            tail="whichever_is_smaller")


def test_both_tails_select_the_same_number_of_sites():
    """Otherwise the two directions would not be comparable: a quantile applied
    to the wrong side silently changes the hotspot size along with the tail."""
    xy, values = _grid_with_corner_cluster(100.0)
    high = hotspot_compactness(xy, values, quantile=0.8, n_null=10, seed=1,
                               tail="high")
    low = hotspot_compactness(xy, values, quantile=0.8, n_null=10, seed=1,
                              tail="low")
    assert high["n_hotspot_sites"] == low["n_hotspot_sites"]


def test_top_ladder_rung_is_realizable_at_the_frozen_radius():
    """32000 E over 400 mm^2 means a 1.0 mm disk holds ~251 cells -- the 256
    rung would be unrealizable. The frozen radius must clear the top rung."""
    import json
    config = json.loads((ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json").read_text())
    radius = float(config["perturbation"]["packet_radius_mm"])
    top = max(config["perturbation"]["dose_ladder_cells"])
    density = 32000 / (20.0 * 20.0)
    assert density * np.pi * radius ** 2 > top * 1.2
