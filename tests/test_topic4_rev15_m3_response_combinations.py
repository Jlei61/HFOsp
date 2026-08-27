from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import freeze_topic4_rev15_m3_response_combinations as freezer
from scripts import aggregate_topic4_rev15_m3_response_combinations as aggregate
from scripts import launch_topic4_rev15_m3_response_combinations as launcher
from scripts import monitor_topic4_rev15_m3_response_combinations as monitor
from scripts import run_topic4_rev15_m3_response_combination_worker as worker


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG = ROOT / "config/topic4_rev15_m3_response_combinations.json"


def _config():
    return json.loads(CONFIG.read_text())


def _build():
    return freezer.build_candidates(_config(), ARTIFACT_ROOT)


def test_combination_manifest_inventory_and_mechanisms_are_frozen():
    candidates, _, _ = _build()
    assert len(candidates) == 15
    assert sum(row["selection_eligible"] for row in candidates) == 14
    assert candidates[0]["candidate_id"] == "exact_off"
    assert [row["candidate_id"] for row in candidates[1:3]] == [
        "m3_c09_m_r08", "m3_c07_p_r08",
    ]
    assert _config()["pathways"] == {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off",
        "Z_M": "off",
    }


def test_registered_response_directions_and_rms_shell_are_exact():
    candidates, audit, _ = _build()
    assert audit["harmful_B_component_removed"] is True
    assert audit["dense_predicted_first_order_B_change"] > 0.0
    assert abs(audit["bprotected_predicted_first_order_B_change"]) < 1e-14
    assert audit["sparse4_source_candidates"] == [
        "m3_c09_m_r08", "m3_c21_m_r08",
        "m3_c00_p_r08", "m3_c24_p_r08",
    ]
    assert audit["sparse8_source_candidates"][-4:] == [
        "m3_c13_m_r08", "m3_c10_m_r08",
        "m3_c07_p_r08", "m3_c18_m_r08",
    ]
    combinations = [
        row for row in candidates if row["candidate_id"].startswith("combo_")
    ]
    assert len(combinations) == 12
    assert sorted({
        row["fourier_coordinate"]["target_centered_surface_rms"]
        for row in combinations
    }) == [0.6, 0.8, 1.0]
    hashes = [row["fourier_coordinate"]["coefficients_sha256"] for row in candidates[1:]]
    assert len(hashes) == len(set(hashes))


def test_gradient_hashes_and_candidate_generation_are_deterministic():
    first, first_audit, _ = _build()
    second, second_audit, _ = _build()
    assert first_audit["g_A_sha256"] == second_audit["g_A_sha256"]
    assert first_audit["g_B_sha256"] == second_audit["g_B_sha256"]
    assert [
        row.get("fourier_coordinate", {}).get("coefficients_sha256")
        if row.get("fourier_coordinate") else None for row in first
    ] == [
        row.get("fourier_coordinate", {}).get("coefficients_sha256")
        if row.get("fourier_coordinate") else None for row in second
    ]


def test_worker_wrapper_uses_combination_freezer_and_status():
    previous = (
        worker.base.freezer,
        worker.base.EXPECTED_PATHWAYS,
        worker.base.WORKER_STATUS,
        worker.base.PREPARE_STATUS,
    )
    try:
        worker.configure_base()
        assert worker.base.freezer is freezer
        assert worker.base.WORKER_STATUS == worker.WORKER_STATUS
        assert worker.base.EXPECTED_PATHWAYS == freezer.EXPECTED_PATHWAYS
    finally:
        (
            worker.base.freezer,
            worker.base.EXPECTED_PATHWAYS,
            worker.base.WORKER_STATUS,
            worker.base.PREPARE_STATUS,
        ) = previous


def test_monitor_wrapper_uses_30_run_fresh_seed_contract():
    names = (
        "freezer", "WORKER", "WORKER_STATUS", "DEFAULT_UNIT_PREFIX",
        "CONTROLLER_SCHEMA", "_validate_unit_prefix", "QUEUE_EMERGENCY_STATUS",
        "QUEUE_COMPLETE_STATUS", "QUEUE_DRAINING_STATUS", "QUEUE_FAILED_STATUS",
        "QUEUE_WAIT_STATUS", "QUEUE_RUNNING_STATUS",
    )
    previous = {name: getattr(monitor.base, name) for name in names}
    try:
        monitor.configure_base()
        assert monitor.base.WORKER.name == (
            "run_topic4_rev15_m3_response_combination_worker.py"
        )
        assert monitor.base.WORKER_STATUS == worker.WORKER_STATUS
        assert _config()["search"]["canary_network_seeds"] == [2332, 2333]
        assert _config()["search"]["active_network_seeds"] == [2332, 2333]
        assert 15 * 2 == 30
        assert _config()["resources"]["maximum_workers"] == 10
    finally:
        for name, value in previous.items():
            setattr(monitor.base, name, value)


def test_launcher_command_uses_systemd_nohup_and_numeric_thread_limits(tmp_path):
    unit, command = launcher._command(
        config_path=CONFIG,
        expected_commit="a" * 40,
        artifact_root=ARTIFACT_ROOT,
        unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
        worker_cap=9,
        log_path=tmp_path / "controller.log",
    )
    assert unit.startswith("codex-t4-r15-m3combo-controller-")
    assert command[0:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert "--setenv=OMP_NUM_THREADS=1" in command


def test_prepare_payload_is_complete_without_running_snn():
    payload = freezer.build_manifest_payload(
        CONFIG,
        artifact_root=ARTIFACT_ROOT,
        provenance={"formal_ready": False},
        status=freezer.PREPARE_STATUS,
    )
    assert payload["status"] == freezer.PREPARE_STATUS
    assert len(payload["candidates"]) == 15
    assert payload["direction_audit"]["patient_heldout_used"] is False
    assert np.isfinite(payload["direction_audit"]["g_A"]).all()


def test_rev12_compatibility_config_declares_both_fresh_seeds():
    config = _config()
    manifest = freezer.build_manifest_payload(
        CONFIG, artifact_root=ARTIFACT_ROOT,
        provenance={"formal_ready": False}, status=freezer.PREPARE_STATUS,
    )
    compatibility, _ = worker.base._compatibility_config(
        config, manifest, artifact_root=ARTIFACT_ROOT,
    )
    allowed = {
        int(seed) for key in (
            "canary_network_seeds", "fit_network_seeds",
            "selection_network_seeds", "confirmation_network_seeds",
        ) for seed in compatibility["search"].get(key, [])
    }
    assert {2332, 2333}.issubset(allowed)


def test_replication_summary_requires_fresh_paired_improvement_and_support():
    def row(candidate_id, seed, a, b, support_a=7.0, support_b=8.0):
        return {
            "candidate_id": candidate_id, "seed": seed,
            "mode_0_mean": a, "mode_1_mean": b, "j14": a + b,
            "mode_0_effective_events": support_a,
            "mode_1_effective_events": support_b,
            "family": "dense_a", "target_rms": 0.8,
        }

    rows = [
        row("exact_off", 2332, 1.5, 1.2),
        row("exact_off", 2333, 1.6, 1.3),
        row("combo_dense_a_r08", 2332, 1.2, 1.25),
        row("combo_dense_a_r08", 2333, 1.3, 1.35),
        row("m3_c09_m_r08", 2332, 1.4, 1.25),
        row("m3_c09_m_r08", 2333, 1.7, 1.35),
    ]
    manifest = {
        "candidates": [
            {"candidate_id": "exact_off"},
            {"candidate_id": "combo_dense_a_r08"},
            {"candidate_id": "m3_c09_m_r08"},
        ],
        "m3_design": {"single_control_ids": ["m3_c09_m_r08"]},
    }
    atlas_payload = {"per_candidate": [
        row("exact_off", 2331, 1.55, 1.25),
        row("m3_c09_m_r08", 2331, 1.20, 1.26),
    ]}
    summaries = aggregate._summaries(rows, manifest, atlas_payload)
    combo = next(row for row in summaries if row["candidate_id"] == "combo_dense_a_r08")
    single = next(row for row in summaries if row["candidate_id"] == "m3_c09_m_r08")
    assert combo["usable_two_mode_anchor"] is True
    assert combo["fresh_A_improvement_count"] == 2
    assert single["evaluation_network_count"] == 3
    assert single["A_improvement_count"] == 2
    assert single["usable_two_mode_anchor"] is True
