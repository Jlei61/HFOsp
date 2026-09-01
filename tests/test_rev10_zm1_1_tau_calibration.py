"""Contracts for the one-dimensional rev10-ZM1.1 M recovery calibration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.freeze_topic4_rev10_zm1_1_tau_library as freezer
import scripts.run_topic4_rev10_zm1_1_tau_phase_controller as controller
import scripts.finalize_topic4_rev10_zm1_1_tau_phase as finalizer
from scripts.audit_topic4_rev10_zm1_1_tau_phase import (
    _dominates,
    _eligible,
    _pareto,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_zm1_1_tau_fit.json"


def test_fit_config_varies_only_tau_and_separates_network_phases():
    config = json.loads(CONFIG.read_text())
    assert config["scientific_role"] == (
        "development_only_data_driven_h_zm_tau_adp_calibration"
    )
    assert config["mz_calibration"]["varied_parameter_only"] == "tau_adp"
    assert config["mz_calibration"]["candidate_tau_adp_ms"] == [
        500.0, 1000.0, 1500.0, 2000.0,
    ]
    search = config["search"]
    fit, selection, confirmation = map(set, (
        search["fit_network_seeds"], search["selection_network_seeds"],
        search["confirmation_network_seeds"],
    ))
    assert fit.isdisjoint(selection | confirmation)
    assert selection.isdisjoint(confirmation)
    assert search["simulation"]["duration_ms"] == 20000.0
    for record in config["inputs"].values():
        digest = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert digest == record["sha256"]


def test_fit_freezer_keeps_h_ou_z_eta_fixed(monkeypatch):
    monkeypatch.setattr(freezer, "_runtime_provenance", lambda commit: {
        "runtime_modules_dirty": False,
        "runtime_modules_match_expected_commit": True,
    })

    def fake_check_output(command, **kwargs):
        del kwargs
        if command[:3] == ["git", "rev-parse", "test-commit"]:
            return "a" * 40 + "\n"
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(freezer.subprocess, "check_output", fake_check_output)
    manifest = freezer.build_manifest(CONFIG, "test-commit")
    control, *active = manifest["candidate_set"]["candidates"]
    assert control["candidate_id"] == freezer.CONTROL_ID
    assert [row["mz"]["tau_adp"] for row in active] == [
        500.0, 1000.0, 1500.0, 2000.0,
    ]
    assert len({tuple(row["coefficients"]) for row in active}) == 1
    assert len({json.dumps(row["spatial_ou"], sort_keys=True) for row in active}) == 1
    for key in ("I_th_EI", "tau_z", "eta_m"):
        assert len({row["mz"][key] for row in active}) == 1


def _row(candidate, *, margin, purity, shape, ood, runaway=0):
    return {
        "candidate_id": candidate,
        "n_runaway_networks": runaway,
        "kmeans_status": "EVALUABLE",
        "zm_dynamically_engaged": True,
        "kmeans_signed_geometry_margin": margin,
        "kmeans_direction_purity": purity,
        "worst_supervised_shape_distance": shape,
        "mean_ood_fraction": ood,
    }


def test_pareto_selection_has_no_composite_score_and_excludes_runaway():
    best = _row("best", margin=0.4, purity=0.8, shape=1.0, ood=0.2)
    dominated = _row("dominated", margin=0.3, purity=0.7, shape=1.2, ood=0.3)
    unsafe = _row(
        "unsafe", margin=0.9, purity=0.9, shape=0.5, ood=0.1, runaway=1,
    )
    assert _eligible(best)
    assert _dominates(best, dominated)
    assert [row["candidate_id"] for row in _pareto([
        best, dominated, unsafe,
    ])] == ["best"]


def test_phase_controller_can_import_repo_atomic_writer():
    from src.topic4_core_field_runner import atomic_write_json

    assert controller.ROOT == ROOT
    assert callable(atomic_write_json)
    assert finalizer.DECISION_BY_PHASE["fit"] == "tau_adp_fit_decision.json"
