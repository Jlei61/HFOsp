from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.topic4_rev22_interictal_objective import embedding_features, fit_training_embedding
from src.topic4_shaft_aware import build_event_features, fit_patient_embedding
from src.topic4_shaft_aware_direction import fit_direction_classifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/aggregate_topic4_rev22_validation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_topic4_rev22_validation", SCRIPT)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validation)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return _sha(path)


def _contract() -> dict:
    contacts = [
        {"contact_index": 0, "shaft_id": "ICL", "within_shaft_order_by_shared_axis": 0},
        {"contact_index": 1, "shaft_id": "ICL", "within_shaft_order_by_shared_axis": 1},
        {"contact_index": 2, "shaft_id": "SCL", "within_shaft_order_by_shared_axis": 0},
        {"contact_index": 3, "shaft_id": "SCL", "within_shaft_order_by_shared_axis": 1},
    ]
    pairs = [
        {"i": 0, "j": 1, "pair_class": "ICL-ICL"},
        {"i": 2, "j": 3, "pair_class": "SCL-SCL"},
        *[{"i": i, "j": j, "pair_class": "ICL-SCL"} for i in (0, 1) for j in (2, 3)],
    ]
    return {"contacts": contacts, "pairs": pairs}


def _onsets(n: int, *, shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(42 + int(shift * 10))
    rows = []
    for index in range(n):
        if index % 2:
            row = np.asarray([0.0, 4.0, 9.0, 13.0])
        else:
            row = np.asarray([13.0, 9.0, 4.0, 0.0])
        rows.append(row + shift + rng.normal(0, 0.3, 4))
    return np.asarray(rows)


def _ranks(onsets: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(onsets, axis=1), axis=1).astype(float)


def _candidate_row(candidate_id: str, family: str) -> dict:
    return {
        "candidate_id": candidate_id, "family_membership": [family],
        "mechanisms": {"Z_M": "off", "g_EE": 0.5, "g_EtoI": 1.0,
                       "ellipse_angle_deg": 45.0, "ellipse_aspect_ratio": 2.0},
        "node_field": {"field_sha256": "field-hash"},
        "physical": {"g_LEE": 0.5, "g_LEI": 1.0, "theta_FT_deg": 0.0, "AR_FT": 2.0},
        "unit_cube": {"g_LEE": 0.5, "g_LEI": 0.5, "theta_FT_deg": 0.5, "AR_FT": 0.5},
        "block": "fixture",
    }


def _make_inputs(tmp_path: Path, *, candidate_ids=("full", "locked"),
                 fit_candidate_ids=("fit0",)) -> dict:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    train_ms = _onsets(96)
    blocks = np.repeat(np.arange(8), 12)
    groups = {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])}

    new_features = embedding_features(train_ms, groups)
    new_embedding = fit_training_embedding(
        new_features, seed=1, max_components=8, reference_n=96, n_directions=8,
    )
    training_contract = tmp_path / "patient_training_contract_v1.npz"
    np.savez(
        training_contract, feature_center=new_embedding["center"],
        feature_scale=new_embedding["scale"], pca_components=new_embedding["components"],
        sw_directions=new_embedding["directions"], reference_z=new_embedding["reference_z"],
        contact_names=names, patient_train_onsets_ms=train_ms,
    )

    old_features = build_event_features(train_ms / 1000.0, groups)["features"]
    old_embedding = fit_patient_embedding(
        old_features, seed=2, max_components=8, reference_n=96, n_directions=8,
    )
    labels = np.arange(len(train_ms)) % 2
    classifier = fit_direction_classifier(
        train_ms / 1000.0, labels, blocks, groups=groups, embedding=old_embedding,
        n_splits=4,
    )
    target = tmp_path / "patient_training_target.npz"
    np.savez(
        target, feature_center=old_embedding["center"], feature_scale=old_embedding["scale"],
        pca_components=old_embedding["components"], sw_directions=old_embedding["directions"],
        global_reference_z=old_embedding["reference_z"], contact_names=names,
        patient_train_ranks=_ranks(train_ms), patient_train_old_labels=labels,
    )
    heldout = tmp_path / "patient_heldout_endpoint.npz"
    held_ms = _onsets(72, shift=0.1)
    np.savez(heldout, contact_names=names, heldout_onsets=held_ms / 1000.0,
             heldout_ranks=_ranks(held_ms), heldout_old_labels=np.arange(72) % 2,
             heldout_block_ids=np.repeat(np.arange(6), 12))
    contact = tmp_path / "contact_contract.json"
    _json(contact, _contract())
    classifier_path = tmp_path / "classifier_manifest.json"
    classifier_json = dict(classifier)
    for key, value in list(classifier_json.items()):
        if isinstance(value, np.ndarray):
            classifier_json[key] = value.tolist()
    _json(classifier_path, {"direction_classifier": classifier_json})

    manifest = tmp_path / "execution_candidate_manifest.json"
    manifest_payload = {
        "schema_id": "topic4_rev22_dci_execution_candidate_manifest_v1",
        "candidates": [_candidate_row(cid, "M1111" if cid == "full" else "M0111")
                       for cid in candidate_ids]
    }
    manifest_hash = _json(manifest, manifest_payload)
    response_design = tmp_path / "response_design_manifest.json"
    response_payload = {
        "schema_id": "topic4_rev22_dci_response_design_manifest_v1",
        "candidate_count": len(fit_candidate_ids),
        "candidates": [_candidate_row(cid, "M1111") for cid in fit_candidate_ids],
    }
    response_hash = _json(response_design, response_payload)
    frozen = tmp_path / "frozen_candidates.json"
    seeds = tmp_path / "seed_manifest.json"
    seed_payload = {
        "schema_id": "topic4_rev22_dci_seed_manifest_v1",
        "response_design_manifest_sha256": response_hash,
        "fit": {"units": [
            {"topology_seed": 11 + i, "dynamics_seed": 11 + i} for i in range(4)
        ]},
        "qualification": {"units": [
            {"topology_seed": 101 + i, "dynamics_seed": 201 + i} for i in range(6)
        ]},
        "confirmation": {"units": [
            {"topology_seed": 301 + i, "dynamics_seed": 401 + i} for i in range(12)
        ]},
    }
    seed_hash = _json(seeds, seed_payload)
    _json(frozen, {
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        "candidate_ids": list(candidate_ids), "branch": "primary",
        "response_design_manifest_sha256": response_hash,
        "execution_candidate_manifest_sha256": manifest_hash,
        "seed_manifest_sha256": seed_hash,
        "mask_to_candidates": {"M1111": ["full"], "M0111": ["locked"]},
    })
    return {
        "training_contract": training_contract, "target": target, "heldout": heldout,
        "contact": contact, "classifier": classifier_path, "manifest": manifest,
        "response_design": response_design, "frozen": frozen, "seeds": seeds,
        "seed_payload": seed_payload, "fit_candidate_ids": list(fit_candidate_ids),
        "seed_hash": seed_hash, "response_hash": response_hash,
        "git_commit": git_commit,
        "fit": tmp_path / "fit/workers",
        "qualification": tmp_path / "qualification/workers",
        "confirmation": tmp_path / "confirmation/workers", "out": tmp_path / "validation",
    }


def _worker(ctx: dict, phase: str, candidate: str, topology: int, dynamics: int,
            *, n_events: int = 16, runaway: bool = False, nonfinite: bool = False) -> None:
    root = ctx[phase]
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{candidate}_topo_{topology}_dyn_{dynamics}"
    npz = root / f"{stem}.npz"
    onsets = _onsets(n_events, shift=(topology % 3) * 0.1)
    active = np.asarray([0.1, 0.2])
    if nonfinite:
        active[0] = np.inf
    np.savez(
        npz, contact_names=np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"]),
        onsets=onsets, ranks=_ranks(onsets), event_returned=np.ones(n_events, bool),
        topology_seed=np.asarray(topology), dynamics_seed=np.asarray(dynamics),
        active_fraction=active, contact_envelope=np.zeros((4, 4)),
        mechanism_parameters=np.ones(6),
    )
    _json(root / f"{stem}.json", {
        "status": "REV12ND_NODE_WORKER_COMPLETE", "candidate_id": candidate,
        "field_sha256": "field-hash",
        "topology_seed": topology, "dynamics_seed": dynamics,
        "response_design_manifest_sha256": ctx["response_hash"],
        "seed_manifest_sha256": ctx["seed_hash"],
        "simulation": {"duration_ms": 20000.0,
                       "runaway_early_stop_ms": 9000 if runaway else None},
        "mechanism_freeze": {"Z_M": "off", "g_EE": 0.5, "g_EtoI": 1.0,
                             "ellipse_angle_deg": 45.0, "ellipse_aspect_ratio": 2.0},
        "arrays": {"path": str(npz), "sha256": _sha(npz)},
        "provenance": {"git_commit": ctx["git_commit"],
                       "expected_git_commit": ctx["git_commit"],
                       "runtime_modules_match_expected_commit": 1,
                       "runtime_modules_dirty": 0, "config_sha256": "config-hash",
                       "config_sha256_at_expected_commit": "config-hash"},
    })


def _populate(ctx: dict, *, failure: tuple[str, str, int] | None = None,
              low_yield: bool = False) -> None:
    for candidate in ctx["fit_candidate_ids"]:
        for unit in ctx["seed_payload"]["fit"]["units"]:
            _worker(ctx, "fit", candidate, unit["topology_seed"], unit["dynamics_seed"])
    for phase in ("qualification", "confirmation"):
        for candidate in ("full", "locked"):
            for unit in ctx["seed_payload"][phase]["units"]:
                topology = unit["topology_seed"]
                if failure == (phase, candidate, topology) and failure[0] == "missing":
                    continue
                _worker(
                    ctx, phase, candidate, topology, unit["dynamics_seed"],
                    n_events=10 if low_yield and candidate == "full" else 16,
                    runaway=failure == ("runaway", candidate, topology),
                    nonfinite=failure == ("nonfinite", candidate, topology),
                )


def _run(ctx: dict, **kwargs) -> dict:
    return validation.aggregate_validation(
        frozen_candidates_path=ctx["frozen"], candidate_manifest_path=ctx["manifest"],
        response_design_path=ctx["response_design"], seed_manifest_path=ctx["seeds"],
        fit_worker_dir=ctx["fit"], qualification_worker_dir=ctx["qualification"],
        confirmation_worker_dir=ctx["confirmation"], training_contract_path=ctx["training_contract"],
        patient_training_target_path=ctx["target"], heldout_path=ctx["heldout"],
        contact_contract_path=ctx["contact"], classifier_manifest_path=ctx["classifier"],
        output_dir=ctx["out"], n_cov=12, r_cov=5.0, floor_draws=4,
        bootstrap_draws=2, recall_subsamples=4, c2st_resamples=2,
        fit_floor_draws=2, fit_recall_subsamples=2, fit_c2st_resamples=2,
        expected_fit_candidates=len(ctx["fit_candidate_ids"]),
        n_pair_min=2, kmeans_seed=10, **kwargs,
    )


def test_complete_grid_emits_all_task11_endpoints_and_selection_blind_language(tmp_path):
    ctx = _make_inputs(tmp_path, fit_candidate_ids=("fit0", "fit1", "fit2"))
    _populate(ctx)
    payload = _run(ctx)

    assert payload["status"] == "VALIDATION_AGGREGATE_COMPLETE"
    assert payload["snn_simulation_run"] is False
    assert payload["patient_ictal_input_read"] is False
    assert payload["terminology"]["kmeans_and_ood"] == "selection-blind"
    assert len(payload["unit_inventory"]) == 2 * (6 + 12)
    fit = payload["fit_descriptive"]
    assert fit["candidate_count"] == 3
    assert fit["candidate_ids"] == ["fit0", "fit1", "fit2"]
    assert fit["frozen_candidate_ids_unchanged"] == ["full", "locked"]
    assert fit["descriptive_only"] is True and fit["cannot_select"] is True
    assert fit["selection_effect"] == "NONE"
    assert len(fit["unit_inventory"]) == 3 * 4
    assert all(row["primary_status"] == "OK" for row in fit["candidates"])
    assert all(row["selection_rank"] is None for row in fit["candidates"])
    full = next(row for row in payload["phases"]["confirmation"]
                if row["candidate_id"] == "full")
    assert full["primary_status"] == "OK"
    assert set(full["primary_endpoints"]) == set(validation.PRIMARY)
    assert all(value is not None for value in full["primary_endpoints"].values())
    assert full["heldout_timing"]["raw_ms"] is not None
    assert full["secondary"]["D_cloud_composite"] is not None
    assert full["secondary"]["c2st"]["status"] in ("OK", validation.C2ST_NOT_ESTIMABLE)
    assert full["secondary"]["c2st"]["formal_endpoint"] == "separability"
    assert full["secondary"]["c2st"]["raw_auc_role"] == "diagnostic_only"
    assert full["clipping"]["pooled"]["event_contact_fraction"] == 0.0
    assert full["mode_proportions"]["networks_evaluable"] == 12
    assert full["mode_proportions"]["network_dual_mode_fraction"] is not None
    assert full["kmeans_ood"]["grouped_k2_vs_k1"]["group_separated"] is True
    assert full["kmeans_ood"]["network_identity_audit"]["nmi"] is not None
    patient_benchmark = payload["patient_recording_block_benchmark"]
    assert patient_benchmark["grouped_k2_vs_k1"]["group_separated"] is True
    assert patient_benchmark["grouped_alignment_benchmark"]["group_separated"] is True
    assert payload["paired_pareto_contrasts"]
    assert (ctx["out"] / "validation_aggregate.json").is_file()


@pytest.mark.parametrize("failure_kind", ["missing", "runaway", "nonfinite"])
def test_any_bad_expected_unit_fail_closes_all_six_candidate_endpoints(tmp_path, failure_kind):
    ctx = _make_inputs(tmp_path)
    _populate(ctx)
    unit = ctx["seed_payload"]["confirmation"]["units"][3]
    topology = unit["topology_seed"]
    if failure_kind == "missing":
        stem = f"full_topo_{topology}_dyn_{unit['dynamics_seed']}"
        (ctx["confirmation"] / f"{stem}.json").unlink()
        (ctx["confirmation"] / f"{stem}.npz").unlink()
    else:
        _worker(ctx, "confirmation", "full", topology, unit["dynamics_seed"],
                runaway=failure_kind == "runaway", nonfinite=failure_kind == "nonfinite")

    payload = _run(ctx)
    full = next(row for row in payload["phases"]["confirmation"]
                if row["candidate_id"] == "full")
    assert len(full["units"]) == 12
    assert full["primary_status"] == validation.PRIMARY_NOT_ESTIMABLE
    assert all(value is None for value in full["primary_endpoints"].values())
    assert full["sidecars_only_due_to_incomplete_grid"] is True
    contrast = payload["paired_pareto_contrasts"][0]
    assert contrast["status"] == validation.PRIMARY_NOT_ESTIMABLE


def test_low_yield_recall_is_not_replaced_by_zero(tmp_path):
    ctx = _make_inputs(tmp_path)
    _populate(ctx, low_yield=True)
    payload = _run(ctx)
    full = next(row for row in payload["phases"]["confirmation"]
                if row["candidate_id"] == "full")
    assert full["endpoint_status"]["recall"] == "NOT_ESTIMABLE_LOW_YIELD"
    assert full["conditional_survivor_endpoints"]["recall"] is None
    assert full["primary_status"] == validation.PRIMARY_PARTIALLY_ESTIMABLE
    assert full["primary_endpoints"]["recall"] is None
    assert all(full["primary_endpoints"][name] is not None
               for name in validation.PRIMARY if name != "recall")
    assert all(row["recall"] is None for row in full["recall_units"])


def test_failed_fit_unit_retained_and_fit_candidate_fail_closed_without_refreeze(tmp_path):
    ctx = _make_inputs(tmp_path, fit_candidate_ids=("fit0", "fit1"))
    _populate(ctx)
    unit = ctx["seed_payload"]["fit"]["units"][2]
    stem = f"fit1_topo_{unit['topology_seed']}_dyn_{unit['dynamics_seed']}"
    (ctx["fit"] / f"{stem}.json").unlink()
    (ctx["fit"] / f"{stem}.npz").unlink()

    payload = _run(ctx)
    rows = {row["candidate_id"]: row for row in payload["fit_descriptive"]["candidates"]}
    assert rows["fit0"]["primary_status"] == "OK"
    assert rows["fit1"]["primary_status"] == validation.PRIMARY_NOT_ESTIMABLE
    assert len(rows["fit1"]["units"]) == 4
    assert all(value is None for value in rows["fit1"]["primary_endpoints"].values())
    assert payload["fit_descriptive"]["frozen_candidate_ids_unchanged"] == ["full", "locked"]
    assert payload["paired_pareto_contrasts"][0]["full_candidate_id"] == "full"


def test_patient_ictal_path_is_rejected_before_read(tmp_path):
    path = tmp_path / "patient_ictal_target.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="patient-ictal boundary"):
        validation._read_json(path, "forbidden")


def test_formal_response_design_inventory_requires_and_keeps_all_96_candidates():
    candidates = [_candidate_row(f"dci_{index:03d}", "M1111") for index in range(96)]
    ids, index = validation._fit_candidate_index({
        "schema_id": "topic4_rev22_dci_response_design_manifest_v1",
        "candidate_count": 96, "candidates": candidates,
    }, expected_count=96)
    assert len(ids) == 96
    assert ids == [f"dci_{value:03d}" for value in range(96)]
    assert set(index) == set(ids)


def test_formal_bootstrap_repools_and_rescores_every_topology_draw(monkeypatch):
    calls = []

    def fake_score(units, _context, *, seed, recall_subsamples):
        calls.append((tuple(unit["topology_seed"] for unit in units), seed, recall_subsamples))
        mean = float(np.mean([unit["value"] for unit in units]))
        return {"D_support": mean, "D_order": mean, "D_time_ms": mean,
                "recall": 1.0 - mean, "kmeans_alignment": 1.0 - mean, "ood": mean}

    monkeypatch.setattr(validation, "_primary_vector_from_units", fake_score)
    full = [{"topology_seed": i, "primary_eligible": True, "value": 0.1 + i * 0.01}
            for i in range(4)]
    locked = [{"topology_seed": i, "primary_eligible": True, "value": 0.2 + i * 0.01}
              for i in range(4)]
    result = validation.paired_nonlinear_bootstrap(
        full, locked, {}, draws=7, seed=3, recall_subsamples=2,
    )
    assert result["bootstrap_method"].startswith("paired_topology_resample_then_repool")
    assert result["status"] == "PARETO_SUPPORTED"
    assert all(row["delta"] > 0 for row in result["endpoints"].values())
    assert len(calls) == 2 * (1 + 7)
    assert all(len(topologies) == 4 for topologies, _, _ in calls)


def test_bootstrap_retains_stable_endpoints_when_recall_support_is_incomplete(monkeypatch):
    calls = 0

    def fake_score(units, _context, *, seed, recall_subsamples):
        nonlocal calls
        calls += 1
        mean = float(np.mean([unit["value"] for unit in units]))
        recall = None if calls in {1, 2, 5, 8} else 1.0 - mean
        return {"D_support": mean, "D_order": mean, "D_time_ms": mean,
                "recall": recall, "kmeans_alignment": 1.0 - mean, "ood": mean}

    monkeypatch.setattr(validation, "_primary_vector_from_units", fake_score)
    full = [{"topology_seed": i, "primary_eligible": True, "value": 0.1 + i * 0.01}
            for i in range(4)]
    locked = [{"topology_seed": i, "primary_eligible": True, "value": 0.2 + i * 0.01}
              for i in range(4)]
    result = validation.paired_nonlinear_bootstrap(
        full, locked, {}, draws=10, seed=3, recall_subsamples=2,
    )
    assert result["status"] == "NON_IDENTIFIABLE_AT_CURRENT_SEEDS"
    assert result["reason"] == "ENDPOINT_SUPPORT_UNSTABLE"
    assert result["endpoints"]["recall"]["status"] == validation.BOOTSTRAP_SUPPORT_UNSTABLE
    assert result["endpoints"]["recall"]["delta"] is None
    assert all(result["endpoints"][name]["status"] == "OK"
               for name in validation.PRIMARY if name != "recall")
