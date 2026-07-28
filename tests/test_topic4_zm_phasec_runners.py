import json
import os

import numpy as np
import pytest

import scripts.run_topic4_zm_phasec_cell as CELL
import scripts.run_topic4_zm_phasec0_parallel as COORD


def test_phasec_matrix_includes_exact_zero_and_is_unique():
    rows = COORD.tasks(("identity", "gain"))
    assert len(rows) == 18 + 135
    assert len({row["key"] for row in rows}) == len(rows)
    zero = [
        row for row in rows
        if row["kind"] == "gain"
        and row["expected"]["delta_mV"] == 0.0
        and row["expected"]["sign"] == 0
    ]
    assert len(zero) == 3 * 3 * 3
    assert all("--delta-mV" in row["cmd"] for row in zero)


def test_smoke_namespace_cannot_match_a_production_part_or_summary_input():
    common = {
        "resolution": "dt",
        "seed": 1,
        "state_tag": "bounded_mid__rising",
        "replicate": "noise_replay",
    }
    smoke = type("Args", (), {**common, "smoke": True})()
    production = type("Args", (), {**common, "smoke": False})()
    smoke_root = CELL._c0_cell_root(smoke, "identity")
    production_root = CELL._c0_cell_root(production, "identity")
    coordinator_path = COORD._identity_output(
        1, "bounded_mid__rising", "noise_replay", resolution="dt"
    )
    assert f"{os.sep}smoke{os.sep}" in smoke_root
    assert f"{os.sep}parts{os.sep}" not in smoke_root
    assert production_root == os.path.dirname(coordinator_path)
    assert smoke_root != production_root


def test_fixed_panels_are_manifest_sourced_and_fail_closed():
    ctx = {
        "S": {"NE": 8},
        "core": np.asarray([True, True, True, True, False, False, False, False]),
    }
    row = {
        "activity_independent": True,
        "analysis_panel_E_ids": [0, 1, 4, 5],
        "analysis_panel_n_core": 2,
        "analysis_panel_n_surround": 2,
        "pairwise_panel_E_ids": [0, 4],
        "pairwise_panel_n_core": 1,
        "pairwise_panel_n_surround": 1,
    }
    row["panel_sha256"] = CELL._object_sha(row)
    fixed, analysis, pairs = CELL._validate_fixed_panels(
        {"fixed_panels": row}, 1, ctx
    )
    assert fixed is row
    np.testing.assert_array_equal(analysis, [0, 1, 4, 5])
    np.testing.assert_array_equal(pairs, [0, 4])

    bad = dict(row, pairwise_panel_E_ids=[0, 0])
    bad["panel_sha256"] = CELL._object_sha({
        key: value for key, value in bad.items() if key != "panel_sha256"
    })
    with pytest.raises(RuntimeError, match="panel is invalid"):
        CELL._validate_fixed_panels({"fixed_panels": bad}, 1, ctx)
    with pytest.raises(RuntimeError, match="not activity-independent"):
        CELL._validate_fixed_panels(
            {"fixed_panels": dict(row, activity_independent=False)}, 1, ctx
        )


def test_atomic_publishers_never_overwrite(tmp_path):
    json_path = tmp_path / "part.json"
    CELL._publish_json_once(str(json_path), {"value": 1})
    with pytest.raises(FileExistsError):
        CELL._publish_json_once(str(json_path), {"value": 2})
    assert json.loads(json_path.read_text()) == {"value": 1}

    npz_path = tmp_path / "part.npz"
    CELL._publish_npz_once(str(npz_path), value=np.asarray([1]))
    with pytest.raises(FileExistsError):
        CELL._publish_npz_once(str(npz_path), value=np.asarray([2]))
    with np.load(npz_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z["value"], [1])


def test_content_addressed_npz_orphan_never_blocks_exact_resume(tmp_path):
    arrays = {"value": np.arange(8, dtype=np.int32)}
    first, first_sha = CELL._publish_content_addressed_npz(
        str(tmp_path), "observables", arrays
    )
    # Simulate a crash before JSON publication by calling the producer again.
    second, second_sha = CELL._publish_content_addressed_npz(
        str(tmp_path), "observables", arrays
    )
    assert os.path.exists(first)
    assert os.path.exists(second)
    assert CELL._sha(first) == first_sha
    assert CELL._sha(second) == second_sha
    assert os.path.basename(first).startswith("observables.")
    assert os.path.basename(second).startswith("observables.")


def test_sustained_gate_is_not_single_bin():
    assert CELL._first_sustained(
        [False, True, True, False, True, True, True], 3
    ) == 4
    assert CELL._first_sustained([True, False, True], 2) is None


def test_resource_cap_never_forces_one_worker(monkeypatch):
    args = type("Args", (), {
        "reserve_cpus": 8,
        "reserve_gb": 96.0,
        "worker_rss_gb": 8.0,
        "max_workers": 16,
    })()
    monkeypatch.setattr(COORD, "mem_available_gb", lambda: 100.0)
    monkeypatch.setattr(COORD.os, "cpu_count", lambda: 64)
    assert COORD._resource_cap(args) == 0
    monkeypatch.setattr(COORD, "mem_available_gb", lambda: 224.0)
    assert COORD._resource_cap(args) == 12


def test_terminal_validator_requires_observable_hash_and_gain_blocks(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(COORD, "ROOT", str(tmp_path))
    observables = tmp_path / "obs.npz"
    np.savez_compressed(
        observables,
        hierarchical_schema=np.asarray(
            "zm_phasec_hierarchical_stats_v1_2026-07-28"
        ),
        pairwise_null_draws=np.asarray(100),
        rho80_active_core_by_block_window=np.zeros((16, 6)),
        block_isi_cv2_by_panel_neuron=np.zeros((16, 4)),
        block_refractory_isi_fraction_by_panel_neuron=np.zeros((16, 4)),
        pair_corr_by_block_and_pair=np.zeros((16, 6)),
        pair_null_median_by_block_and_draw=np.zeros((16, 3, 100)),
        pair_null_stratum_names=np.asarray(
            ("core_core", "core_surround", "surround_surround")
        ),
        active_area_fraction_by_block_window=np.zeros((16, 20)),
        spatial_grid_n_occupied_E=np.asarray(256),
        spatial_grid_all_E_bins_occupied=np.asarray(True),
        spatial_active_floor_hz=np.asarray(5.0),
        spatial_area_denominator=np.asarray(
            "anatomy_occupied_E_grid_bins"
        ),
        analysis_panel_E_ids=np.arange(4),
        pairwise_panel_E_ids=np.arange(4),
    )
    common = {
        "manifest_sha256": "m",
        "status": "complete",
        "carrier_gates": {
            "runaway": False,
            "whole_sheet_plateau": False,
            "empirical_rest_dwell": False,
        },
        "runtime_provenance": {
            "manifest_sha256": "m",
            "producer_sha256": {"producer.py": "p"},
        },
    }
    identity_path = tmp_path / "identity.json"
    identity = {
        **common,
        "schema": "zm_phasec_identity_cell_v1",
        "state_hash": "state",
        "state_file_sha256": "state-file",
        "noise_bank_sha": "noise",
        "config_sha": "config",
        "observables_path": os.path.relpath(observables, tmp_path),
        "observables_sha256": COORD._sha(observables),
    }
    identity_path.write_text(json.dumps(identity))
    task = {
        "kind": "identity",
        "expected": {
            "schema": "zm_phasec_identity_cell_v1",
            "manifest_sha256": "m",
            "state_hash": "state",
            "state_file_sha256": "state-file",
            "noise_bank_sha": "noise",
            "config_sha": "config",
        },
    }
    valid, reason, _ = COORD.validate_terminal_output(
        str(identity_path), task
    )
    assert valid, reason
    identity["observables_sha256"] = "bad"
    identity_path.write_text(json.dumps(identity))
    assert COORD.validate_terminal_output(
        str(identity_path), task
    )[:2] == (False, "observables_sha_mismatch")
    identity["observables_sha256"] = COORD._sha(observables)
    identity["state_file_sha256"] = "mutated"
    identity_path.write_text(json.dumps(identity))
    valid, reason, _ = COORD.validate_terminal_output(
        str(identity_path), task
    )
    assert not valid
    assert reason == "identity_mismatch:state_file_sha256"

    gain_path = tmp_path / "gain.json"
    gain = {
        **common,
        "schema": "zm_phasec_gain_cell_v1",
        "core_rate_500ms_hz": [1.0],
        "gain_plateau_gate_pass": True,
    }
    gain_path.write_text(json.dumps(gain))
    gain_task = {
        "kind": "gain",
        "expected": {
            "schema": "zm_phasec_gain_cell_v1",
            "manifest_sha256": "m",
        },
    }
    assert COORD.validate_terminal_output(
        str(gain_path), gain_task
    )[:2] == (False, "invalid_gain_block_observables")
