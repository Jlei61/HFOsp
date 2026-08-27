import csv
import json

import numpy as np

import scripts.aggregate_topic4_rev13_node_zero_sum_canary as aggregate
from scripts.aggregate_topic4_rev13_node_zero_sum_canary import (
    _load_model_internal_arrays,
    _project_worker_payload,
    build_paired_rows,
    canonicalize_cluster_labels,
    controller_diagnostic,
    directional_k2_formal,
    fragmentation_audit,
    model_internal_decision,
    occupancy_matched_transition_null,
    overlap_connected_episode_audit,
    paired_record,
    whole_sheet_onset_map_kmeans_diagnostic,
    write_outputs,
)


def _bimodal(n=36, magnitude=3.0):
    pattern = np.array([-magnitude, -0.9 * magnitude,
                        0.9 * magnitude, magnitude])
    return np.resize(pattern, n).astype(np.float64)


def _unimodal(n=36, seed=1):
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def _run(candidate_id, *, seed=2311, displacement=None, runaway=False,
         compound=0.1, fragmented=False, event_t_on_ms=None):
    values = np.asarray(
        _bimodal() if displacement is None else displacement,
        dtype=np.float64,
    )
    root_ids = np.arange(len(values), dtype=np.int64)
    if fragmented and len(values) >= 2:
        root_ids[1] = root_ids[0]
    times = np.asarray(
        np.arange(len(values), dtype=np.float64) * 100.0
        if event_t_on_ms is None else event_t_on_ms,
        dtype=np.float64,
    )
    assert times.shape == values.shape
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "n_returned_evaluable_causal_families": int(len(values)),
        "n_isolated_returned_evaluable_causal_families": int(len(values)),
        "compound_detector_fragment_fraction": compound,
        "runaway": runaway,
        "substrate_identity": {
            "positions_E_sha256": "positions",
            "delta_vtheta_sha256": "node",
            "node_mapping_sha256": "mapping",
        },
        "whole_sheet_onset_map_kmeans_diagnostic": {
            "status": "OK",
            "formal_acceptance_role": "TOPOLOGY_VISUALIZATION_DIAGNOSTIC_ONLY",
        },
        "_directional_displacements_mm": values,
        "_event_t_on_ms": times,
        "_directed_root_ids": root_ids,
        "_fragment_counts": np.ones(len(values), dtype=np.int64),
        "controller": {
            "zero_sum_ok": True,
            "widespread_saturation": False,
            "widespread_static_sign_flip": False,
        },
    }


def _manifest():
    candidates = [
        "exact_off", "zero_sum_c010", "zero_sum_c020", "zero_sum_c040",
        "raise_only_c020", "spatial_shift_c020",
    ]
    return {
        "candidates": [{"candidate_id": value} for value in candidates],
        "search": {
            "canary_network_seeds": [2311],
            "fit_network_seeds": [2312, 2313],
        },
    }


def _matched_c020(active=None, *, off=None, raise_only=None, shuffled=None):
    return paired_record(
        _run("zero_sum_c020") if active is None else active,
        _run("exact_off", displacement=_unimodal(seed=11)) if off is None else off,
        matched_controls=[
            _run("raise_only_c020", displacement=_unimodal(seed=12))
            if raise_only is None else raise_only,
            _run("spatial_shift_c020", displacement=_unimodal(seed=13))
            if shuffled is None else shuffled,
        ],
        expected_matched_control_ids=[
            "raise_only_c020", "spatial_shift_c020",
        ],
    )


def test_paired_record_rejects_cross_arm_substrate_drift():
    active = _run("zero_sum_c020")
    off = _run("exact_off", displacement=_unimodal(seed=11))
    off["substrate_identity"] = dict(off["substrate_identity"])
    off["substrate_identity"]["delta_vtheta_sha256"] = "changed"
    with np.testing.assert_raises_regex(RuntimeError, "frozen Node substrate"):
        paired_record(active, off)


def test_cluster_direction_summary_is_invariant_to_kmeans_label_swap():
    displacement = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    canonical, medians = canonicalize_cluster_labels(labels, displacement)
    swapped, swapped_medians = canonicalize_cluster_labels(
        1 - labels, displacement,
    )
    assert np.array_equal(canonical, swapped)
    assert medians == swapped_medians
    assert medians[0] < 0.0 < medians[1]


def test_formal_directional_k2_freezes_all_positive_criteria():
    values = _bimodal()
    result = directional_k2_formal(
        values, np.arange(len(values)) * 100.0, seed=19,
    )
    assert result["status"] == "OK"
    assert result["heldout_k2_minus_k1_loglik_per_event"] > 0.0
    assert result["cluster_median_axis_displacement_mm"][0] < 0.0
    assert result["cluster_median_axis_displacement_mm"][1] > 0.0
    assert result["minimum_within_cluster_direction_consistency"] >= 0.70
    assert result["minority_fraction"] >= 0.20
    assert result["temporal_blocks_with_each_sign"] == {
        "negative": 3, "positive": 3,
    }
    assert result["both_signs_in_at_least_two_of_three_blocks"] is True
    assert len(result["heldout_fold_deltas"]) == 3
    assert result["temporal_block_definition"] == "three_equal_duration_blocks"
    assert result["preferred_event_count_reached"] is True


def test_formal_directional_k2_rejects_sign_confined_to_one_time_block():
    displacement = np.r_[
        np.tile([-3.0, -2.8], 12),
        np.tile([2.8, 3.0], 6),
    ]
    result = directional_k2_formal(
        displacement, np.arange(len(displacement)) * 100.0, seed=21,
    )
    assert result["heldout_k2_minus_k1_loglik_per_event"] < 0.0
    assert result["same_network_opposite_directions"] is True
    assert result["temporal_blocks_with_each_sign"] == {
        "negative": 2, "positive": 1,
    }
    assert result["both_signs_in_at_least_two_of_three_blocks"] is False


def test_formal_k2_uses_equal_duration_not_equal_event_count_blocks():
    values = np.r_[np.tile([-3.0, 3.0], 15), [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0]]
    times = np.r_[np.linspace(0.0, 100.0, 30), [400.0, 410.0, 800.0, 810.0, 900.0, 910.0]]
    result = directional_k2_formal(values, times, seed=22)
    assert result["status"] == "OK"
    assert [
        row["negative"] + row["positive"]
        for row in result["temporal_block_sign_counts"]
    ] == [30, 2, 4]


def test_formal_k2_insufficient_support_is_not_evaluable():
    values = np.r_[np.full(18, -3.0), np.full(5, 3.0)]
    result = directional_k2_formal(
        values, np.arange(len(values)) * 100.0, seed=23,
    )
    assert result["status"] == "NOT_EVALUABLE"
    assert set(result["evaluability_reasons"]) == {
        "fewer_than_24_isolated_families",
        "fewer_than_6_families_in_one_direction",
    }
    assert result["events_per_direction"] == {"negative": 18, "positive": 5}


def test_event_counts_are_matched_across_active_off_and_controls():
    paired = _matched_c020(
        active=_run("zero_sum_c020", displacement=_bimodal(60)),
        off=_run("exact_off", displacement=_unimodal(36, seed=31)),
        raise_only=_run("raise_only_c020", displacement=_unimodal(48, seed=32)),
        shuffled=_run(
            "spatial_shift_c020", displacement=_unimodal(42, seed=33),
        ),
    )
    assert paired["matched_event_count_per_arm"] == 36
    assert paired["directional_k2_formal"]["n_events"] == 36
    assert {
        record["n_events"]
        for record in paired["matched_control_directional_k2_formal"].values()
    } == {36}


def test_candidate_must_beat_off_and_every_same_coefficient_control():
    passed = _matched_c020()
    assert passed["checks"]["heldout_k2_minus_k1_positive"] is True
    assert passed["checks"]["directional_k2_above_paired_off"] is True
    assert passed["checks"][
        "directional_k2_above_same_coefficient_controls"
    ] is True
    assert passed["model_internal_network_pass"] is True

    stronger_control = _run(
        "spatial_shift_c020", displacement=_bimodal(magnitude=4.0),
    )
    failed = _matched_c020(shuffled=stronger_control)
    assert failed["checks"][
        "directional_k2_above_same_coefficient_controls"
    ] is False
    assert failed["model_internal_network_pass"] is False


def test_same_network_direction_and_consistency_are_formal_checks(monkeypatch):
    original = aggregate._matched_directional_record

    def fake(record, *, n_events, seed):
        formal, indices = original(record, n_events=n_events, seed=seed)
        if record["candidate_id"] == "zero_sum_c020":
            formal["same_network_opposite_directions"] = False
            formal["minimum_within_cluster_direction_consistency"] = 0.69
        return formal, indices

    monkeypatch.setattr(aggregate, "_matched_directional_record", fake)
    paired = _matched_c020()
    assert paired["checks"]["same_network_opposite_directions"] is False
    assert paired["checks"][
        "within_cluster_direction_consistency_at_least_70_percent"
    ] is False
    assert paired["model_internal_network_pass"] is False


def test_minority_and_temporal_recurrence_are_formal_checks(monkeypatch):
    original = aggregate._matched_directional_record

    def fake(record, *, n_events, seed):
        formal, indices = original(record, n_events=n_events, seed=seed)
        if record["candidate_id"] == "zero_sum_c020":
            formal["minority_fraction"] = 0.19
            formal["both_signs_in_at_least_two_of_three_blocks"] = False
        return formal, indices

    monkeypatch.setattr(aggregate, "_matched_directional_record", fake)
    paired = _matched_c020()
    assert paired["checks"]["minority_at_least_20_percent"] is False
    assert paired["checks"][
        "both_signs_in_at_least_two_of_three_temporal_blocks"
    ] is False
    assert paired["model_internal_network_pass"] is False


def test_c010_and_c040_without_matched_controls_require_extension():
    rows = [
        _run("exact_off", displacement=_unimodal(seed=40)),
        _run("zero_sum_c010"),
        _run("zero_sum_c020"),
        _run("zero_sum_c040"),
        _run("raise_only_c020", displacement=_unimodal(seed=41)),
        _run("spatial_shift_c020", displacement=_unimodal(seed=42)),
    ]
    paired = build_paired_rows(
        rows,
        expected_candidate_ids=[row["candidate_id"] for row in rows],
    )
    by_candidate = {row["candidate_id"]: row for row in paired}
    for candidate_id in ("zero_sum_c010", "zero_sum_c040"):
        assert by_candidate[candidate_id]["status"] == (
            "MATCHED_CONTROL_EXTENSION_REQUIRED"
        )
        assert by_candidate[candidate_id]["model_internal_network_pass"] is False
    assert by_candidate["zero_sum_c020"]["matched_control_status"] == (
        "MATCHED_CONTROLS_COMPLETE"
    )


def test_onset_map_kmeans_is_diagnostic_only_and_has_no_formal_density():
    maps = np.full((12, 3, 3), np.nan)
    for event in range(12):
        if event % 2:
            maps[event] = np.arange(9).reshape(3, 3)
        else:
            maps[event] = np.arange(8, -1, -1).reshape(3, 3)
    result = whole_sheet_onset_map_kmeans_diagnostic(
        maps, axis_unit=np.array([1.0, 0.0]), bin_mm=1.0, seed=8,
    )
    assert result["status"] == "OK"
    assert result["formal_acceptance_role"] == (
        "TOPOLOGY_VISUALIZATION_DIAGNOSTIC_ONLY"
    )
    assert "heldout_gmm_k2_minus_k1_loglik_per_event" not in result


def test_occupancy_matched_null_rejects_forced_alternation():
    labels = np.tile([0, 1], 24)
    audit = occupancy_matched_transition_null(labels, draws=2048, seed=19)
    assert audit["status"] == "OK"
    assert audit["observed_same_mode_probability"] == 0.0
    assert audit["anti_persistent"] is True
    assert audit["observed_same_mode_probability"] < audit[
        "occupancy_matched_null_q05"
    ]


def test_spatial_shift_uses_support_weighted_controller_diagnostics():
    result = controller_diagnostic({
        "node_accessibility": {
            "mode": "spatial_shift",
            "diagnostics": {
                "maximum_zero_sum_error_mV": 1e-12,
                "maximum_support_weighted_saturation_fraction": 0.1,
                "maximum_support_weighted_static_modulation_sign_flip_fraction": 0.2,
                "maximum_saturation_fraction": 0.9,
                "maximum_static_modulation_sign_flip_fraction": 0.9,
            },
        },
    })
    assert result["zero_sum_required"] is True
    assert result["zero_sum_ok"] is True
    assert result["widespread_saturation"] is False
    assert result["widespread_static_sign_flip"] is False


def test_overlap_connected_episode_excludes_transitive_historical_style_chain():
    isolated, audit = overlap_connected_episode_audit(
        event_t_on_ms=np.array([0.0, 80.0, 170.0, 400.0, 700.0]),
        event_t_off_ms=np.array([100.0, 190.0, 250.0, 450.0, 760.0]),
        displacements=np.array([-3.0, 3.0, -2.0, 2.0, -1.0]),
    )
    assert isolated.tolist() == [False, False, False, True, True]
    assert audit["n_overlap_pairs"] == 2
    assert audit["n_overlap_components"] == 1
    assert audit["n_excluded_families"] == 3
    assert audit["overlap_pair_direction_counts"] == {
        "same": 0,
        "opposite": 2,
        "zero_involved": 0,
    }
    assert audit["overlap_components"][0]["contains_opposite_directions"] is True


def test_touching_family_boundaries_are_not_overlap():
    isolated, audit = overlap_connected_episode_audit(
        event_t_on_ms=np.array([0.0, 100.0]),
        event_t_off_ms=np.array([100.0, 200.0]),
        displacements=np.array([-1.0, 1.0]),
    )
    assert isolated.tolist() == [True, True]
    assert audit["n_overlap_pairs"] == 0


def test_repeated_root_fragmentation_is_diagnostic_only():
    audit = fragmentation_audit(
        labels=np.array([0, 1, 0, 1]),
        root_ids=np.array([10, 10, 11, 12]),
        fragment_counts=np.array([1, 1, 2, 1]),
    )
    assert audit["formal_acceptance_role"] == "DIAGNOSTIC_ONLY"
    assert audit["repeated_root_crosses_k2_clusters"] is True
    assert audit["cross_cluster_duplicated_root_ids"] == [10]


def test_repeated_root_diagnostic_does_not_change_formal_verdict(monkeypatch):
    original = aggregate.fragmentation_audit

    def forced_fragmentation(labels, root_ids, fragment_counts):
        result = original(labels, root_ids, fragment_counts)
        result["repeated_root_crosses_k2_clusters"] = True
        return result

    monkeypatch.setattr(aggregate, "fragmentation_audit", forced_fragmentation)
    paired = _matched_c020()
    assert "not_family_fragmentation" not in paired["checks"]
    assert paired["fragmentation"]["repeated_root_crosses_k2_clusters"] is True
    assert paired["model_internal_network_pass"] is True


def test_paired_insufficient_is_not_evaluable_not_scientific_fail():
    short = _bimodal(20)
    paired = _matched_c020(
        active=_run("zero_sum_c020", displacement=short),
        off=_run("exact_off", displacement=short),
        raise_only=_run("raise_only_c020", displacement=short),
        shuffled=_run("spatial_shift_c020", displacement=short),
    )
    assert paired["status"] == "NOT_EVALUABLE"
    assert paired["formal_comparison_evaluable"] is False
    assert paired["failure_reasons"] == []
    assert paired["model_internal_network_pass"] is False


def test_model_internal_loaders_project_hard_allowlists(tmp_path):
    payload = {
        "status": "ok",
        "candidate_id": "x",
        "seed": 1,
        "node_accessibility": {
            "mode": "zero_sum",
            "diagnostics": {},
            "patient_runtime_inputs_used": False,
            "contact_rank": [1, 2],
            "patient_target": [3, 4],
        },
        "contact_waveform": [1, 2, 3],
        "patient_prototype": [4, 5, 6],
    }
    projected = _project_worker_payload(payload)
    serialized = json.dumps(projected).lower()
    assert "contact" not in serialized
    assert "patient_target" not in serialized
    assert projected["node_accessibility"]["patient_runtime_inputs_used"] is False

    arrays_path = tmp_path / "worker.npz"
    np.savez_compressed(
        arrays_path,
        event_returned=np.array([True]),
        event_t_off_ms=np.array([100.0]),
        event_trigger_t_on_ms=np.array([20.0]),
        event_root_count=np.array([1]),
        contact_envelope=np.ones((2, 3)),
        patient_rank=np.ones((2, 3)),
    )
    loaded = _load_model_internal_arrays(arrays_path)
    assert set(loaded) == {
        "event_returned", "event_t_off_ms", "event_trigger_t_on_ms",
        "event_root_count",
    }


def test_decision_and_writers_emit_paired_formal_outputs(tmp_path):
    per_run = [
        _run("exact_off", displacement=_unimodal(seed=51)),
        _run("zero_sum_c020"),
        _run("raise_only_c020", displacement=_unimodal(seed=52)),
        _run("spatial_shift_c020", displacement=_unimodal(seed=53)),
    ]
    paired = build_paired_rows(
        per_run,
        expected_candidate_ids=[row["candidate_id"] for row in _manifest()["candidates"]],
    )
    decision = model_internal_decision(per_run, paired, _manifest())
    assert decision["status"] == "REV13_MODEL_INTERNAL_AGGREGATE_INCOMPLETE"
    assert decision["patient_labels_or_prototypes_loaded"] is False
    assert decision["rules"]["formal_endpoint"] == (
        "signed_causal_displacement_1d"
    )

    paths = write_outputs(tmp_path, per_run, paired, decision)
    assert set(paths) == {
        "per_run_json", "per_run_csv", "paired_json", "paired_csv",
        "decision_json", "decision_csv",
    }
    saved = json.loads((tmp_path / "analysis/model_internal_decision.json").read_text())
    assert saved["patient_labels_or_prototypes_loaded"] is False
    per_run_saved = json.loads(
        (tmp_path / "aggregate/model_internal_per_run.json").read_text()
    )
    assert all(
        not key.startswith("_") for row in per_run_saved["rows"] for key in row
    )
    with (tmp_path / "aggregate/model_internal_paired.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["matched_event_count_per_arm"] == "36"
    with (tmp_path / "analysis/model_internal_decision.csv").open() as handle:
        decision_rows = list(csv.DictReader(handle))
    assert {row["candidate_id"] for row in decision_rows} == {
        "zero_sum_c010", "zero_sum_c020", "zero_sum_c040",
    }
