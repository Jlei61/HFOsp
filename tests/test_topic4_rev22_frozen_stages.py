from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stage = _module(
    "aggregate_topic4_rev22_frozen_stages",
    ROOT / "scripts/aggregate_topic4_rev22_frozen_stages.py",
)
fit_test = _module(
    "test_topic4_rev22_fit_aggregate_helpers",
    ROOT / "tests/test_topic4_rev22_fit_aggregate.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict, *, sidecar: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha(path)
    if sidecar:
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _candidate(candidate_id: str, family: str) -> dict:
    row = fit_test._candidate(candidate_id)
    row["family_membership"] = [family]
    return row


def _context(tmp_path: Path) -> dict:
    base = fit_test._make_contracts(tmp_path / "base", candidate_ids=("ref", "full", "locked"))
    qualification = json.loads(base["qualification_path"].read_text())
    objective_commit = qualification["git_commit"]

    response_hash = "response-design-hash"
    seed_path = tmp_path / "seed_manifest.json"
    seeds = {
        "schema_id": "topic4_rev22_dci_seed_manifest_v1",
        "response_design_manifest_sha256": response_hash,
        "fit": {"units": base["units"]},
        "qualification": {"units": [
            {"topology_seed": 201 + i, "dynamics_seed": 201 + i} for i in range(6)
        ]},
        "confirmation": {"units": [
            {"topology_seed": 301 + i, "dynamics_seed": 301 + i} for i in range(12)
        ]},
    }
    seed_hash = _write(seed_path, seeds, sidecar=True)
    manifest_path = tmp_path / "execution_candidate_manifest.json"
    manifest = {
        "schema_id": "topic4_rev22_dci_execution_candidate_manifest_v1",
        "git_commit": objective_commit,
        "candidate_count": 3,
        "response_design_manifest_sha256": response_hash,
        "seed_manifest_sha256": seed_hash,
        "candidates": [
            _candidate("ref", "M0000"),
            _candidate("full", "M1111"),
            _candidate("locked", "M0111"),
        ],
    }
    manifest_hash = _write(manifest_path, manifest)
    response_fit_path = tmp_path / "response_fit.json"
    response_fit = {
        "schema_id": "topic4_rev22_dci_response_fit_v1",
        "status": "RESPONSE_FIT_COMPLETE",
        "training_only": True,
        "git_commit": objective_commit,
        "identifiable_components": list(stage.COMPONENTS),
        "input_hashes": {"seed_manifest": {"path": str(seed_path), "sha256": seed_hash}},
        "proposals": {
            "M0000": {"frozen_points": [{"origin": "gp", "execution_candidate_id": "ref"}],
                      "gp": {"predicted_excess": {k: 0.0 for k in stage.COMPONENTS},
                             "predicted_sd": {k: 0.1 for k in stage.COMPONENTS}}},
            "M1111": {"frozen_points": [{"origin": "gp", "execution_candidate_id": "full"}],
                      "gp": {"predicted_excess": {k: 0.2 for k in stage.COMPONENTS},
                             "predicted_sd": {k: 0.1 for k in stage.COMPONENTS}}},
            "M0111": {"frozen_points": [{"origin": "tree", "execution_candidate_id": "locked"}],
                      "tree": {"predicted_excess": {k: 0.3 for k in stage.COMPONENTS},
                               "predicted_sd": {k: 0.2 for k in stage.COMPONENTS}}},
        },
    }
    response_fit_hash = _write(response_fit_path, response_fit)
    frozen_path = tmp_path / "frozen_candidates.json"
    _write(frozen_path, {
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        "status": "REV22_CANDIDATES_FROZEN",
        "branch": "PRIMARY_4D_BRANCH",
        "candidate_ids": ["ref", "full", "locked"],
        "mask_to_candidates": {"M0000": ["ref"], "M1111": ["full"], "M0111": ["locked"]},
        "response_design_manifest_sha256": response_hash,
        "seed_manifest_sha256": seed_hash,
        "execution_candidate_manifest_sha256": manifest_hash,
        "input_hashes": {"response_fit": {"path": str(response_fit_path.resolve()),
                                           "sha256": response_fit_hash}},
    })
    return {
        **base, "response_hash": response_hash, "seed_hash": seed_hash,
        "seed_path": seed_path, "seeds": seeds, "manifest_path": manifest_path,
        "response_fit_path": response_fit_path, "frozen_path": frozen_path,
        "qualification_worker_dir": tmp_path / "qualification/workers",
        "confirmation_worker_dir": tmp_path / "confirmation/workers",
        "out": tmp_path / "aggregate",
    }


def _worker(ctx: dict, phase: str, candidate_id: str, topology: int, *,
            n_events: int = 14, runaway: bool = False, nonfinite: bool = False) -> None:
    old = ctx["worker_dir"]
    ctx["worker_dir"] = ctx[f"{phase}_worker_dir"]
    try:
        fit_test._write_worker(
            ctx, candidate_id, topology, n_events=n_events, runaway=runaway,
            nonfinite=nonfinite, response_hash=ctx["response_hash"],
        )
    finally:
        ctx["worker_dir"] = old


def _populate(ctx: dict) -> None:
    for phase in stage.PHASES:
        for candidate in ("ref", "full", "locked"):
            for unit in ctx["seeds"][phase]["units"]:
                _worker(ctx, phase, candidate, unit["topology_seed"],
                        n_events=12 + unit["topology_seed"] % 4)


def _run(ctx: dict, **overrides) -> dict:
    kwargs = dict(
        frozen_candidates_path=ctx["frozen_path"], response_fit_path=ctx["response_fit_path"],
        candidate_manifest_path=ctx["manifest_path"], seed_manifest_path=ctx["seed_path"],
        objective_qualification_path=ctx["qualification_path"],
        qualification_worker_dir=ctx["qualification_worker_dir"],
        confirmation_worker_dir=ctx["confirmation_worker_dir"], output_dir=ctx["out"],
        bootstrap_draws=4, bootstrap_seed=7,
        ancestry_checker=lambda _a, _b, _root: True,
    )
    kwargs.update(overrides)
    return stage.aggregate_frozen_stages(**kwargs)


def test_complete_grids_pool_events_and_emit_prediction_and_paired_intervals(tmp_path):
    ctx = _context(tmp_path)
    _populate(ctx)
    payload = _run(ctx)

    assert payload["status"] == "FROZEN_STAGE_AGGREGATE_COMPLETE"
    assert payload["inventory"]["expected_unit_count"] == 3 * (6 + 12)
    assert payload["phases"]["qualification"]["primary_estimable_candidates"] == 3
    assert payload["phases"]["confirmation"]["primary_estimable_candidates"] == 3
    candidate = payload["phases"]["confirmation"]["candidates"]["full"]
    assert candidate["pooled"]["n_events"] == sum(candidate["n_returned_families_by_unit"])
    assert set(candidate["bootstrap_90"]) == set(stage.COMPONENTS)
    predictions = payload["qualification_prediction_audit"]
    assert {row["candidate_id"] for row in predictions} == {"ref", "full", "locked"}
    assert all(row["observed_standardized_Z"] is not None for row in predictions)
    assert len(payload["confirmation_contrasts"]["candidate_vs_reference"]) == 2
    assert len(payload["confirmation_contrasts"]["full_vs_locked"]) == 1
    contrast = payload["confirmation_contrasts"]["full_vs_locked"][0]
    assert set(contrast["point_estimate"]) == set(stage.COMPONENTS)
    for component in stage.COMPONENTS:
        expected = (
            payload["phases"]["confirmation"]["candidates"]["locked"]["pooled"]
            ["components"][component]
            - payload["phases"]["confirmation"]["candidates"]["full"]["pooled"]
            ["components"][component]
        )
        assert contrast["point_estimate"][component] == pytest.approx(expected)
    assert (ctx["out"] / "frozen_stage_aggregate.json").is_file()
    assert (ctx["out"] / "frozen_stage_feasibility.csv").is_file()


def test_every_bootstrap_draw_rescores_repoolled_event_tables(tmp_path, monkeypatch):
    ctx = _context(tmp_path)
    _populate(ctx)
    calls = 0
    original = stage._score_tables

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(stage, "_score_tables", counted)
    _run(ctx, bootstrap_draws=3)
    # Six point estimates + 54 leave-one-topology-out rescored pools + 18 candidate
    # bootstraps + three paired contrasts, each rescoring both members for all three draws.
    assert calls == 6 + 54 + 18 + 18


@pytest.mark.parametrize("failure", ["missing", "runaway", "nonfinite", "pooled_low_yield"])
def test_expected_unit_failures_are_fail_closed_and_retained(tmp_path, failure):
    ctx = _context(tmp_path)
    _populate(ctx)
    topology = ctx["seeds"]["qualification"]["units"][0]["topology_seed"]
    root = ctx["qualification_worker_dir"]
    stem = f"full_seed_{topology}"
    if failure == "missing":
        (root / f"{stem}.json").unlink()
        (root / f"{stem}.npz").unlink()
    else:
        if failure == "pooled_low_yield":
            for unit in ctx["seeds"]["qualification"]["units"]:
                _worker(ctx, "qualification", "full", unit["topology_seed"], n_events=8)
        else:
            _worker(ctx, "qualification", "full", topology,
                n_events=14,
                runaway=failure == "runaway", nonfinite=failure == "nonfinite")

    payload = _run(ctx)
    full = payload["phases"]["qualification"]["candidates"]["full"]
    assert full["primary_estimable"] is False
    assert full["pooled"] is None
    assert any(row["candidate_id"] == "full" and row["observed_standardized_Z"] is None
               for row in payload["qualification_prediction_audit"])
    reasons = "|".join(full["failure_reasons"])
    expected = {
        "missing": "MISSING_WORKER_JSON", "runaway": "RUNAWAY",
        "nonfinite": "NPZ_INVALID", "pooled_low_yield": "LOW_RETURNED_FAMILY_YIELD",
    }
    assert expected[failure] in reasons


def test_one_low_yield_unit_is_retained_when_stage_pool_and_loo_are_estimable(tmp_path):
    ctx = _context(tmp_path)
    _populate(ctx)
    topology = ctx["seeds"]["qualification"]["units"][0]["topology_seed"]
    _worker(ctx, "qualification", "full", topology, n_events=8)

    payload = _run(ctx)

    full = payload["phases"]["qualification"]["candidates"]["full"]
    assert full["yield_estimable_units"] == 5
    assert full["primary_estimable"] is True
    assert full["pooled"]["n_events"] >= 72
    assert len(full["leave_one_topology_out"]) == 6
    assert min(row["n_events"] for row in full["leave_one_topology_out"]) >= 60


def test_bootstrap_marks_support_instability_without_imputation(monkeypatch):
    calls = 0

    def alternating_score(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls % 2 == 0:
            raise RuntimeError("synthetic low support")
        return {"components": {component: 1.0 for component in stage.COMPONENTS}}

    monkeypatch.setattr(stage, "_score_tables", alternating_score)
    result = stage._bootstrap_candidate(
        [object(), object()], training={}, objective=None, draws=10, seed=1,
    )
    assert all(row["status"] == "BOOTSTRAP_SUPPORT_UNSTABLE" for row in result.values())
    assert all(row["median"] is None for row in result.values())
    assert all(row["valid_draws"] == 5 and row["invalid_draws"] == 5
               for row in result.values())


def test_out_of_interval_qualification_result_is_reported_not_deleted(tmp_path):
    ctx = _context(tmp_path)
    _populate(ctx)
    response = json.loads(ctx["response_fit_path"].read_text())
    response["proposals"]["M1111"]["gp"]["predicted_excess"] = {
        component: 999.0 for component in stage.COMPONENTS
    }
    response_hash = _write(ctx["response_fit_path"], response)
    frozen = json.loads(ctx["frozen_path"].read_text())
    frozen["input_hashes"]["response_fit"]["sha256"] = response_hash
    _write(ctx["frozen_path"], frozen)

    payload = _run(ctx)
    full = next(row for row in payload["qualification_prediction_audit"]
                if row["candidate_id"] == "full")
    assert not any(full["covered_90"].values())
    assert payload["phases"]["confirmation"]["candidates"]["full"]["primary_estimable"] is True


def test_confirmation_failure_disables_all_dependent_paired_contrasts(tmp_path):
    ctx = _context(tmp_path)
    _populate(ctx)
    topology = ctx["seeds"]["confirmation"]["units"][0]["topology_seed"]
    root = ctx["confirmation_worker_dir"]
    (root / f"full_seed_{topology}.json").unlink()
    (root / f"full_seed_{topology}.npz").unlink()

    payload = _run(ctx)
    candidate_contrast = next(
        row for row in payload["confirmation_contrasts"]["candidate_vs_reference"]
        if row["left"] == "full"
    )
    locked_contrast = payload["confirmation_contrasts"]["full_vs_locked"][0]
    assert candidate_contrast["status"] == "TRAINING_COMPONENT_NOT_ESTIMABLE"
    assert locked_contrast["status"] == "TRAINING_COMPONENT_NOT_ESTIMABLE"
    assert candidate_contrast["paired_90"] is None
    assert locked_contrast["paired_90"] is None


def test_training_only_path_guard_rejects_validation_inputs(tmp_path):
    ctx = _context(tmp_path)
    forbidden = tmp_path / "patient_heldout_endpoint.json"
    _write(forbidden, {})
    with pytest.raises(RuntimeError, match="training-only boundary"):
        _run(ctx, response_fit_path=forbidden)
