from __future__ import annotations

import json

import numpy as np

from scripts import prepare_topic4_rev15_m3_robust_candidate_config as prepare
from scripts import aggregate_topic4_rev15_m3_robust_candidates as aggregate
from scripts import freeze_topic4_rev15_m3_robust_candidates as freezer
from scripts import launch_topic4_rev15_m3_robust_candidates as launcher
from scripts import monitor_topic4_rev15_m3_robust_candidates as monitor
from scripts import run_topic4_rev15_m3_robust_candidate_worker as worker
from scripts import wait_topic4_rev15_m3_response_then_prepare_robust as waiter


def _aggregate():
    direction = np.zeros(28, dtype=float)
    direction[0] = 1.0
    other = np.zeros(28, dtype=float)
    other[1] = 1.0
    return {
        "status": "COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "figure_used": False, "EE_EtoI_ZM": "off",
        },
        "robust_directions": {
            "mean_a": {"feasible": True, "direction": direction.tolist()},
            "maximin_bprotected": {"feasible_positive_margin": True, "direction": other.tolist()},
            "maximin_supportprotected": {"feasible_positive_margin": False, "direction": None},
            "consensus_sparse": {"feasible": True, "direction": direction.tolist()},
        },
    }


def test_feasible_directions_follow_frozen_family_order():
    directions = prepare.feasible_directions(_aggregate())
    assert list(directions) == [
        "mean_a", "maximin_bprotected", "consensus_sparse",
    ]


def test_blueprint_includes_all_rms_and_deduplicates_equal_directions():
    rows, audit = prepare.candidate_blueprint(_aggregate(), n_per_axis=32)
    assert audit["feasible_direction_ids"] == [
        "mean_a", "maximin_bprotected", "consensus_sparse",
    ]
    assert len(rows) == 6
    assert audit["candidate_count_including_exact"] == 7
    assert {row["target_rms"] for row in rows} == {0.4, 0.6, 0.8}
    assert {row["family"] for row in rows} == {
        "mean_a", "maximin_bprotected",
    }
    assert len(audit["deduplicated"]) == 3


def test_forbidden_selection_inputs_fail_closed():
    payload = _aggregate()
    payload["ranking_contract"]["figure_used"] = True
    try:
        prepare.feasible_directions(payload)
    except RuntimeError as error:
        assert "forbidden boundary" in str(error)
    else:
        raise AssertionError("figure-informed aggregate was accepted")


def test_nonunit_direction_fails_closed():
    payload = _aggregate()
    payload["robust_directions"]["mean_a"]["direction"][0] = 2.0
    try:
        prepare.feasible_directions(payload)
    except RuntimeError as error:
        assert "unit norm" in str(error)
    else:
        raise AssertionError("nonunit robust direction was accepted")


def test_worker_and_monitor_wrappers_keep_node_only_contract():
    previous_worker = (
        worker.base.freezer, worker.base.EXPECTED_PATHWAYS,
        worker.base.WORKER_STATUS, worker.base.PREPARE_STATUS,
    )
    names = (
        "freezer", "WORKER", "WORKER_STATUS", "DEFAULT_UNIT_PREFIX",
        "CONTROLLER_SCHEMA", "_validate_unit_prefix",
    )
    previous_monitor = {name: getattr(monitor.base, name) for name in names}
    try:
        worker.configure_base()
        monitor.configure_base()
        assert worker.base.freezer is freezer
        assert worker.base.EXPECTED_PATHWAYS == {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        }
        assert monitor.base.WORKER.name == (
            "run_topic4_rev15_m3_robust_candidate_worker.py"
        )
        assert monitor.base.WORKER_STATUS == worker.WORKER_STATUS
    finally:
        (
            worker.base.freezer, worker.base.EXPECTED_PATHWAYS,
            worker.base.WORKER_STATUS, worker.base.PREPARE_STATUS,
        ) = previous_worker
        for name, value in previous_monitor.items():
            setattr(monitor.base, name, value)


def test_launcher_uses_systemd_nohup_and_one_numeric_thread(tmp_path):
    unit, command = launcher._command(
        config_path=prepare.DEFAULT_OUTPUT,
        expected_commit="a" * 40, artifact_root=prepare.ARTIFACT_ROOT,
        unit_prefix=monitor.DEFAULT_UNIT_PREFIX, worker_cap=9,
        log_path=tmp_path / "controller.log",
    )
    assert unit.startswith("codex-t4-r15-m3robust-controller-")
    assert command[:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert "--setenv=OMP_NUM_THREADS=1" in command


def test_summary_requires_three_of_three_and_two_mode_support():
    def row(candidate_id, seed, a, b, support_a=7.0, support_b=8.0):
        return {
            "candidate_id": candidate_id, "seed": seed,
            "mode_0_mean": a, "mode_1_mean": b, "j14": a + b,
            "mode_0_effective_events": support_a,
            "mode_1_effective_events": support_b,
            "family": "mean_a", "target_rms": 0.4,
        }

    rows = []
    for seed, a, b in ((2341, 1.5, 1.2), (2342, 1.6, 1.3), (2343, 1.4, 1.1)):
        rows.append(row("exact_off", seed, a, b))
        rows.append(row("good", seed, a - 0.2, b * 1.05))
        rows.append(row("fails_one_B", seed, a - 0.3, b * (1.2 if seed == 2343 else 1.0)))
        rows.append(row("low_support", seed, a - 0.1, b, support_a=5.0))
    manifest = {
        "candidates": [
            {"candidate_id": name}
            for name in ("exact_off", "good", "fails_one_B", "low_support")
        ],
        "selection": {
            "fresh_A_improvement_required_networks": 3,
            "fresh_B_protection_required_networks": 3,
            "B_protection_ratio": 1.10,
            "equal_network_effective_support_minimum_per_mode": 6.0,
        },
    }
    summaries = aggregate._summaries(rows, manifest)
    assert summaries[0]["candidate_id"] == "good"
    assert summaries[0]["usable_two_mode_anchor"] is True
    assert next(row for row in summaries if row["candidate_id"] == "fails_one_B")[
        "usable_two_mode_anchor"
    ] is False
    assert next(row for row in summaries if row["candidate_id"] == "low_support")[
        "usable_two_mode_anchor"
    ] is False


def test_waiter_requires_complete_validated_response_tensor():
    complete = {
        "status": "COMPLETE",
        "inventory": {
            "complete_cartesian_product": True,
            "expected_runs": 174, "present_validated": 174,
            "missing": [], "invalid_artifact": [],
            "by_source": {
                "seed2331_raw_workers": {
                    "complete_cartesian_product": True,
                    "expected_runs": 58, "present_validated": 58,
                    "missing": [], "invalid_artifact": [],
                },
                "seeds2332_2333_raw_workers": {
                    "complete_cartesian_product": True,
                    "expected_runs": 116, "present_validated": 116,
                    "missing": [], "invalid_artifact": [],
                },
            },
        },
        "response_tensor": {"metrics": []},
        "robust_directions": {"mean_a": {}},
    }
    assert waiter.classify(complete) == "complete"
    incomplete = json.loads(json.dumps(complete))
    incomplete["inventory"]["by_source"][
        "seeds2332_2333_raw_workers"
    ]["present_validated"] = 115
    assert waiter.classify(incomplete) == "failed"
    wrong_total = json.loads(json.dumps(complete))
    wrong_total["inventory"]["present_validated"] = 116
    assert waiter.classify(wrong_total) == "failed"
    assert waiter.classify({"status": "INCOMPLETE"}) == "wait"
