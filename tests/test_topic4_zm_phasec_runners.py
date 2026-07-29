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

    c1_smoke = CELL._c1_base_relative_path(
        "dt", 1, "primary_convex", "cell", "rising", "noise_replay",
        smoke=True,
    )
    c1_production = CELL._c1_base_relative_path(
        "dt", 1, "primary_convex", "cell", "rising", "noise_replay"
    )
    assert f"{os.sep}smoke{os.sep}" in c1_smoke
    assert f"{os.sep}parts{os.sep}" not in c1_smoke
    assert f"{os.sep}parts{os.sep}" in c1_production
    assert c1_smoke != c1_production


def _smoke_observable_arrays():
    n_block = 2
    n_time = 8
    analysis_ids = np.arange(4)
    pairwise_ids = np.arange(4)
    return {
        "hierarchical_schema": np.asarray(
            CELL.PCM.HIERARCHICAL_STATS_VERSION
        ),
        "rho80_active_core_by_block_window": np.zeros((n_block, 6)),
        "block_isi_cv2_by_panel_neuron": np.zeros((n_block, 4)),
        "block_refractory_isi_numerator_by_stratum": np.zeros(
            (n_block, 2), dtype=np.int64
        ),
        "block_refractory_isi_denominator_by_stratum": np.ones(
            (n_block, 2), dtype=np.int64
        ),
        "refractory_isi_stratum_names": np.asarray(("core", "surround")),
        "pair_corr_by_block_and_pair": np.zeros((n_block, 6)),
        "pair_null_median_by_block_and_draw": np.zeros(
            (n_block, 3, 100)
        ),
        "pair_null_stratum_names": np.asarray(
            ("core_core", "core_surround", "surround_surround")
        ),
        "active_area_fraction_by_block_window": np.zeros((n_block, 20)),
        "spatial_grid_n_occupied_E": np.asarray(16),
        "spatial_area_denominator": np.asarray(
            "anatomy_occupied_E_grid_bins"
        ),
        "analysis_panel_E_ids": analysis_ids,
        "pairwise_panel_E_ids": pairwise_ids,
        "block_ms": np.asarray(500.0),
        "ceiling_window_ms": np.asarray(250.0),
        "ceiling_stride_ms": np.asarray(50.0),
        "active_area_window_ms": np.asarray(25.0),
        "pairwise_bin_ms": np.asarray(5.0),
        "pairwise_null_draws": np.asarray(100),
        "spatial_grid_n": np.asarray(16),
        "raw_sample_time_ms": np.arange(n_time, dtype=float),
        "effective_sample_time_ms": np.arange(n_time, dtype=float),
        "raw_raw_ampa_core_mean_mV": np.ones(n_time),
        "raw_raw_gaba_core_mean_mV": np.ones(n_time),
        "effective_effective_excitation_core_mean_mV": np.ones(n_time),
        "effective_effective_inhibition_z_core_mean_mV": np.ones(n_time),
        "effective_adaptation_m_core_mean_mV": np.ones(n_time),
        "effective_effective_outward_total_core_mean_mV": np.ones(n_time),
        "E_rate_grid": np.zeros((n_time, 16, 16)),
        "I_rate_grid": np.zeros((n_time, 16, 16)),
        "global_E_rate_hz": np.zeros(n_time),
        "global_I_rate_hz": np.zeros(n_time),
        "fine_bin_ms": np.asarray(2.0),
    }


def test_smoke_requires_and_accepts_complete_hierarchical_schema():
    complete = _smoke_observable_arrays()
    kwargs = {
        "analysis_ids": np.arange(4),
        "pairwise_ids": np.arange(4),
        "thresholds": {
            "time_block_ms": 500.0,
            "sliding_rate_window_ms": 250.0,
            "sliding_rate_window_stride_ms": 50.0,
            "active_area_window_ms": 25.0,
            "pairwise_bin_ms": 5.0,
            "pairwise_shift_null_draws": 100,
            "spatial_grid_n": 16,
        },
    }
    old_skipped_bootstrap = {
        key: value for key, value in complete.items()
        if key not in CELL.HIERARCHICAL_ARRAY_FIELDS
        and key not in {
            "pair_null_stratum_names", "refractory_isi_stratum_names"
        }
    }
    assert not CELL._smoke_observables_complete(
        old_skipped_bootstrap, **kwargs
    )
    assert CELL._smoke_observables_complete(complete, **kwargs)
    old_schema = dict(
        complete,
        hierarchical_schema=np.asarray(
            "zm_phasec_hierarchical_stats_v1_2026-07-28"
        ),
    )
    assert not CELL._smoke_observables_complete(old_schema, **kwargs)
    bad_null = dict(
        complete,
        pair_null_median_by_block_and_draw=np.zeros((2, 100)),
    )
    assert not CELL._smoke_observables_complete(bad_null, **kwargs)
    c1 = dict(complete)
    c1.update({
        "source_rate_hz": np.zeros(8),
        "rest_mask": np.zeros(8, bool),
        "active_area_fraction": np.zeros(8),
        "kymograph": np.zeros((8, 4)),
        "axis_positions": np.arange(4.0),
        "bin_ms": np.asarray(2.0),
    })
    assert CELL._smoke_observables_complete(c1, c1=True, **kwargs)

    one_block = {
        key: (value[:1] if isinstance(value, np.ndarray) and value.ndim >= 1
              and value.shape[0] == 2 else value)
        for key, value in complete.items()
    }
    assert not CELL._smoke_observables_complete(one_block, **kwargs)
    missing_current = dict(complete)
    missing_current.pop("effective_adaptation_m_core_mean_mV")
    assert not CELL._smoke_observables_complete(missing_current, **kwargs)
    negative_outward = dict(
        complete,
        effective_effective_outward_total_core_mean_mV=-np.ones(8),
    )
    assert not CELL._smoke_observables_complete(negative_outward, **kwargs)
    wrong_panel = dict(complete, analysis_panel_E_ids=np.asarray([0, 1, 2]))
    assert not CELL._smoke_observables_complete(wrong_panel, **kwargs)
    wrong_metadata = dict(complete, ceiling_stride_ms=np.asarray(25.0))
    assert not CELL._smoke_observables_complete(wrong_metadata, **kwargs)
    bad_grid = dict(complete, I_rate_grid=np.zeros((7, 16, 16)))
    assert not CELL._smoke_observables_complete(bad_grid, **kwargs)
    bad_c1 = dict(c1, kymograph=np.zeros((7, 4)))
    assert not CELL._smoke_observables_complete(
        bad_c1, c1=True, **kwargs
    )


def test_smoke_and_production_share_hierarchical_builder(monkeypatch):
    expected = _smoke_observable_arrays()
    observed = {}

    def fake_builder(e, dt, tau_ref, **kwargs):
        observed.update(
            n_time=len(e),
            dt=dt,
            tau_ref=tau_ref,
            kwargs=kwargs,
        )
        return expected

    monkeypatch.setattr(CELL.PCM, "phasec_bootstrap_units", fake_builder)
    ctx = {
        "dt": 0.5,
        "core": np.asarray([True, True, False, False]),
        "S": {
            "p": type("P", (), {"tau_ref_E": 2.0})(),
            "posE": np.zeros((4, 2)),
            "L": 20.0,
        },
    }
    locks = {
        "analysis_ids": np.arange(4),
        "pairwise_ids": np.arange(4),
    }
    manifest = {
        "thresholds": {
            "time_block_ms": 500.0,
            "sliding_rate_window_ms": 250.0,
            "sliding_rate_window_stride_ms": 50.0,
            "active_area_window_ms": 25.0,
            "pairwise_bin_ms": 5.0,
            "pairwise_shift_null_draws": 100,
            "local_active_floor_hz": 5.0,
            "spatial_grid_n": 16,
        }
    }
    out = CELL._build_hierarchical_observables(
        np.zeros((2000, 4), bool),
        ctx,
        locks,
        manifest,
        17,
        technical_complete=True,
    )
    assert out is expected
    assert observed["n_time"] == 2000
    assert observed["kwargs"]["pairwise_null_seed"] == 17
    assert observed["kwargs"]["pairwise_n_null"] == 100
    assert observed["kwargs"]["ceiling_stride_ms"] == 50.0
    assert observed["kwargs"]["active_area_window_ms"] == 25.0
    assert CELL._build_hierarchical_observables(
        np.zeros((2000, 4), bool),
        ctx,
        locks,
        manifest,
        17,
        technical_complete=False,
    ) is None


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


def test_swap_guard_ignores_small_shared_host_jitter_but_fails_at_limit(
    monkeypatch,
):
    baseline = 800_000
    monkeypatch.setattr(COORD, "swap_used_kb", lambda: baseline + 2 * 1024)
    assert COORD._swap_growth_exceeded(baseline, 64.0) is False
    monkeypatch.setattr(
        COORD, "swap_used_kb", lambda: baseline + 64 * 1024 + 1
    )
    assert COORD._swap_growth_exceeded(baseline, 64.0) is True


def test_production_rss_floor_is_not_below_measured_cells():
    assert COORD.MIN_MEASURED_WORKER_RSS_GB == {
        "identity": 7.23,
        "gain": 6.90,
    }


def test_terminal_validator_requires_observable_hash_and_gain_blocks(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(COORD, "ROOT", str(tmp_path))
    observables = tmp_path / "obs.npz"
    np.savez_compressed(
        observables,
        hierarchical_schema=np.asarray(
            CELL.PCM.HIERARCHICAL_STATS_VERSION
        ),
        pairwise_null_draws=np.asarray(100),
        rho80_active_core_by_block_window=np.zeros((16, 6)),
        block_isi_cv2_by_panel_neuron=np.zeros((16, 4)),
        block_refractory_isi_numerator_by_stratum=np.zeros(
            (16, 2), dtype=np.int64
        ),
        block_refractory_isi_denominator_by_stratum=np.ones(
            (16, 2), dtype=np.int64
        ),
        refractory_isi_stratum_names=np.asarray(("core", "surround")),
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
                "self_vm_swap_kb_at_publish": 0,
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
    with np.load(observables, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["hierarchical_schema"] = np.asarray(
        "zm_phasec_hierarchical_stats_v1_2026-07-28"
    )
    np.savez_compressed(observables, **arrays)
    identity["observables_sha256"] = COORD._sha(observables)
    identity_path.write_text(json.dumps(identity))
    assert COORD.validate_terminal_output(
        str(identity_path), task
    )[:2] == (False, "hierarchical_schema_mismatch")
    arrays["hierarchical_schema"] = np.asarray(
        CELL.PCM.HIERARCHICAL_STATS_VERSION
    )
    np.savez_compressed(observables, **arrays)
    identity["observables_sha256"] = COORD._sha(observables)
    identity_path.write_text(json.dumps(identity))
    task["coordinator_run_id"] = "run-1"
    task["coordinator_launch_token"] = "token-1"
    identity["runtime_provenance"].update({
        "coordinator_run_id": "run-1",
        "coordinator_launch_token": "token-1",
    })
    identity_path.write_text(json.dumps(identity))
    assert COORD.validate_terminal_output(
        str(identity_path), task
    )[:2] == (True, "valid_terminal")
    identity["runtime_provenance"]["coordinator_launch_token"] = "wrong"
    identity_path.write_text(json.dumps(identity))
    assert COORD.validate_terminal_output(
        str(identity_path), task
    )[:2] == (False, "runtime_coordinator_identity_mismatch")
    task.pop("coordinator_run_id")
    task.pop("coordinator_launch_token")
    identity["runtime_provenance"].pop("coordinator_run_id")
    identity["runtime_provenance"].pop("coordinator_launch_token")
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
