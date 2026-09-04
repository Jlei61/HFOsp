from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/aggregate_topic4_rev22_fit.py"
SPEC = importlib.util.spec_from_file_location("aggregate_topic4_rev22_fit", SCRIPT)
aggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(aggregate)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict, *, sidecar: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = _sha256(path)
    if sidecar:
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _patient_onsets(n_events: int = 48) -> np.ndarray:
    rng = np.random.default_rng(20260903)
    rows = []
    for event in range(n_events):
        base = float(event % 7)
        if event % 2:
            row = np.asarray([base + 0.0, base + 3.0, base + 8.0, base + 11.0])
        else:
            row = np.asarray([base + 3.0, base + 0.0, base + 11.0, base + 8.0])
        row += rng.normal(0.0, 0.9, size=4)
        missing = rng.random(4) < np.asarray([0.04, 0.10, 0.12, 0.18])
        if missing.sum() > 1:
            missing[np.flatnonzero(missing)[1:]] = False
        row[missing] = np.nan
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _write_training_contract(path: Path) -> str:
    objective = aggregate._load_training_objective()
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    groups, _ = aggregate._groups_and_pairs(names)
    onsets = _patient_onsets()
    features = objective.embedding_features(onsets, groups)
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    components = np.eye(features.shape[1], dtype=float)
    reference_z = ((features - center) / scale) @ components.T
    directions = np.eye(features.shape[1], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        feature_center=center,
        feature_scale=scale,
        pca_components=components,
        sw_directions=directions,
        reference_z=reference_z,
        reference_indices=np.arange(len(onsets)),
        contact_names=names,
        patient_train_onsets_ms=onsets,
        patient_train_block_ids=np.repeat(np.arange(8), 6),
    )
    return _sha256(path)


def _candidate(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "block": "full4d",
        "physical": {"g_LEE": 0.5, "g_LEI": 1.0, "theta_FT_deg": 45.0, "AR_FT": 2.0},
        "family_membership": ["M1111"],
        "mechanisms": {
            "Z_M": "off", "g_EE": 0.5, "g_EtoI": 1.0,
            "ellipse_angle_deg": 45.0, "ellipse_aspect_ratio": 2.0,
        },
        "node_field": {"field_sha256": "field-hash"},
    }


def _make_contracts(tmp_path: Path, candidate_ids=("c0",), *, n_units: int = 4) -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    response_path = tmp_path / "response_design_manifest.json"
    response = {
        "schema_id": "topic4_rev22_dci_response_design_manifest_v1",
        "candidate_count": len(candidate_ids),
        "git_commit": commit,
        "candidates": [_candidate(candidate_id) for candidate_id in candidate_ids],
    }
    response_hash = _write_json(response_path, response, sidecar=True)
    units = [
        {"topology_seed": 101 + index, "dynamics_seed": 101 + index, "seed_mode": "legacy"}
        for index in range(n_units)
    ]
    seed_path = tmp_path / "seed_manifest.json"
    seed_hash = _write_json(seed_path, {
        "schema_id": "topic4_rev22_dci_seed_manifest_v1",
        "response_design_manifest_sha256": response_hash,
        "fit": {"units": units},
    }, sidecar=True)
    contract_path = tmp_path / "patient_training_contract_v1.npz"
    contract_hash = _write_training_contract(contract_path)
    qualification_path = tmp_path / "objective_qualification.json"
    _write_json(qualification_path, {
        "schema_id": "topic4_rev22_dci_objective_qualification_v2",
        "status": "OBJECTIVE_QUALIFIED",
        "smoke": False,
        "git_commit": commit,
        "components": list(aggregate.COMPONENTS),
        "forbidden_inputs_loaded": False,
        "floors": {
            "draws": 12,
            "seed": 20260903,
            "kind": "block_split_count_matched; recruitment-thinned for D_order/D_lag",
        },
        "patient_training_contract_npz": str(contract_path),
        "patient_training_contract_sha256": contract_hash,
    })
    return {
        "commit": commit,
        "response_path": response_path,
        "response_hash": response_hash,
        "seed_path": seed_path,
        "seed_hash": seed_hash,
        "qualification_path": qualification_path,
        "contract_path": contract_path,
        "units": units,
        "worker_dir": tmp_path / "workers",
        "output_dir": tmp_path / "aggregate",
    }


def _worker_onsets(n_events: int, shift: float = 0.0) -> np.ndarray:
    values = _patient_onsets(n_events).copy()
    values[:, 1] += shift
    return values


def _write_worker(ctx: dict, candidate_id: str, topology_seed: int, *,
                  n_events: int = 12, runaway: bool = False,
                  nonfinite: bool = False, response_hash: str | None = None,
                  drop_scl: bool = False) -> None:
    worker_dir = ctx["worker_dir"]
    worker_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{candidate_id}_seed_{topology_seed}"
    npz_path = worker_dir / f"{stem}.npz"
    onsets = _worker_onsets(n_events, shift=(topology_seed % 4) * 0.2)
    if drop_scl:
        onsets[:, 2:] = np.nan
    active = np.asarray([0.1, 0.2], float)
    if nonfinite:
        active[0] = np.inf
    np.savez(
        npz_path,
        contact_names=np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"]),
        onsets=onsets,
        event_returned=np.ones(n_events, dtype=bool),
        topology_seed=np.asarray(topology_seed, np.int64),
        dynamics_seed=np.asarray(topology_seed, np.int64),
        active_fraction=active,
        contact_envelope=np.zeros((4, 3), dtype=float),
        mechanism_parameters=np.asarray([1.0, 1.0, 0.5, 1.0, 45.0, 2.0]),
    )
    npz_hash = _sha256(npz_path)
    config_hash = "config-hash"
    payload = {
        "status": "REV22_DCI_FIT_WORKER_COMPLETE",
        "candidate_id": candidate_id,
        "field_sha256": "field-hash",
        "topology_seed": topology_seed,
        "dynamics_seed": topology_seed,
        "simulation": {
            "duration_ms": 20000.0,
            "runaway_early_stop_ms": 8000.0 if runaway else None,
        },
        "mechanism_freeze": {
            "Z_M": "off", "g_EE": 0.5, "g_EtoI": 1.0,
            "ellipse_angle_deg": 45.0, "ellipse_aspect_ratio": 2.0,
        },
        "response_design_manifest_sha256": response_hash or ctx["response_hash"],
        "seed_manifest_sha256": ctx["seed_hash"],
        "arrays": {"path": str(npz_path), "sha256": npz_hash},
        "provenance": {
            "git_commit": ctx["commit"],
            "expected_git_commit": ctx["commit"],
            "runtime_modules_match_expected_commit": 1,
            "runtime_modules_dirty": 0,
            "config_sha256": config_hash,
            "config_sha256_at_expected_commit": config_hash,
        },
    }
    _write_json(worker_dir / f"{stem}.json", payload)


_ABSENT = object()


def _set_provenance(ctx: dict, stem: str, **fields) -> None:
    path = ctx["worker_dir"] / f"{stem}.json"
    payload = json.loads(path.read_text())
    for key, value in fields.items():
        if value is _ABSENT:
            payload["provenance"].pop(key, None)
        else:
            payload["provenance"][key] = value
    _write_json(path, payload)


def _run(ctx: dict, *, ancestry_checker=lambda _a, _b, _root: True) -> dict:
    return aggregate.aggregate_fit(
        response_design_path=ctx["response_path"],
        seed_manifest_path=ctx["seed_path"],
        objective_qualification_path=ctx["qualification_path"],
        worker_dir=ctx["worker_dir"],
        output_dir=ctx["output_dir"],
        ancestry_checker=ancestry_checker,
    )


def test_complete_four_unit_candidate_enters_pooled_event_surfaces(tmp_path):
    ctx = _make_contracts(tmp_path)
    for unit in ctx["units"]:
        _write_worker(ctx, "c0", unit["topology_seed"], n_events=12 + unit["topology_seed"] % 3)

    payload = _run(ctx)

    assert payload["status"] == "FIT_AGGREGATE_COMPLETE"
    assert payload["inventory"]["expected_unit_count"] == 4
    assert payload["inventory"]["continuous_surface_candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["continuous_surface_eligible"] is True
    assert len(candidate["leave_one_topology_out"]) == 4
    assert len(payload["continuous_component_surface"]) == 4
    objective = aggregate._load_training_objective()
    training = aggregate._load_training_contract(
        ctx["qualification_path"],
        json.loads(ctx["qualification_path"].read_text()),
        _sha256(ctx["contract_path"]),
    )
    training["reference"] = objective.patient_reference(
        training["onsets_ms"], training["groups"], training["pairs"], training["embedding"],
    )
    pooled_onsets = np.concatenate([
        _worker_onsets(12 + unit["topology_seed"] % 3, shift=(unit["topology_seed"] % 4) * 0.2)
        for unit in ctx["units"]
    ])
    expected = objective.component_vector(
        pooled_onsets, training["reference"], training["groups"], training["pairs"],
        training["embedding"], composite=False,
    )
    assert candidate["pooled_candidate"]["method"].startswith("concatenate_four_topology")
    assert candidate["pooled_candidate"]["n_pooled_events"] == len(pooled_onsets)
    for component in aggregate.COMPONENTS:
        assert candidate["pooled_candidate"]["components"][component] == pytest.approx(
            expected[component]["value"]
        )
        assert candidate["jackknife_sd"][component] >= 0.0
        floor = candidate["floor"][component]
        assert floor["q95"] > floor["q50"]
        expected_z = objective.standardized_excess(
            expected[component]["value"], floor,
        )
        assert candidate["standardized_Z"][component] == pytest.approx(expected_z)
        assert candidate["normalized_excess_E"][component] == pytest.approx(max(0.0, expected_z))
        assert candidate["standardized_jackknife_sd"][component] == pytest.approx(
            candidate["jackknife_sd"][component] /
            (floor["q95"] - floor["q50"] + 1e-9)
        )
    surface = {row["component"]: row for row in payload["continuous_component_surface"]}
    for component in aggregate.COMPONENTS:
        assert surface[component]["raw_D"] == pytest.approx(expected[component]["value"])
        assert surface[component]["standardized_Z"] == pytest.approx(
            candidate["standardized_Z"][component]
        )
        assert surface[component]["normalized_excess_E"] == pytest.approx(
            candidate["normalized_excess_E"][component]
        )
    assert (ctx["output_dir"] / "fit_aggregate.json").is_file()
    assert (ctx["output_dir"] / "fit_candidate_components.csv").is_file()
    assert (ctx["output_dir"] / "fit_unit_inventory.csv").is_file()


def test_one_low_yield_unit_does_not_discard_estimable_pooled_candidate(tmp_path):
    ctx = _make_contracts(tmp_path)
    low_seed = ctx["units"][0]["topology_seed"]
    for unit in ctx["units"]:
        _write_worker(
            ctx,
            "c0",
            unit["topology_seed"],
            n_events=8 if unit["topology_seed"] == low_seed else 16,
        )

    payload = _run(ctx)

    candidate = payload["candidates"][0]
    low = next(row for row in candidate["units"] if row["topology_seed"] == low_seed)
    assert low["event_yield_estimable"] is False
    assert "LOW_RETURNED_FAMILY_YIELD" in low["failure_reasons"]
    assert candidate["pooled_candidate"]["n_pooled_events"] == 56
    assert min(
        56 - next(
            row["n_returned_families"] for row in candidate["units"]
            if row["topology_seed"] == loo["omitted_topology_seed"]
        )
        for loo in candidate["leave_one_topology_out"]
    ) >= aggregate.MIN_LOO_RETURNED_FAMILIES
    assert candidate["continuous_surface_eligible"] is True


@pytest.mark.parametrize("failure", ["missing", "runaway", "low_yield", "nonfinite", "bad_hash"])
def test_failed_unit_is_retained_and_never_enters_surface(tmp_path, failure):
    ctx = _make_contracts(tmp_path)
    failed_seed = ctx["units"][2]["topology_seed"]
    for unit in ctx["units"]:
        seed = unit["topology_seed"]
        if failure == "missing" and seed == failed_seed:
            continue
        _write_worker(
            ctx, "c0", seed,
            n_events=8 if failure == "low_yield" and seed == failed_seed else 12,
            runaway=failure == "runaway" and seed == failed_seed,
            nonfinite=failure == "nonfinite" and seed == failed_seed,
            response_hash="wrong" if failure == "bad_hash" and seed == failed_seed else None,
        )

    payload = _run(ctx)

    candidate = payload["candidates"][0]
    assert len(candidate["units"]) == 4
    assert candidate["continuous_surface_eligible"] is False
    assert candidate["pooled_candidate"] is None
    assert payload["continuous_component_surface"] == []
    failed = next(row for row in candidate["units"] if row["topology_seed"] == failed_seed)
    assert failed["feasibility"] is False
    if failure == "missing":
        assert failed["inventory_status"] == "MISSING"
        assert payload["status"] == "FIT_ARTIFACT_GRID_INCOMPLETE"
    elif failure == "runaway":
        assert "RUNAWAY" in failed["failure_reasons"]
        assert failed["artifact_integrity"] is True
    elif failure == "low_yield":
        assert "LOW_RETURNED_FAMILY_YIELD" in failed["failure_reasons"]
    elif failure == "nonfinite":
        assert any(reason.startswith("NPZ_INVALID:") for reason in failed["failure_reasons"])
    else:
        assert "RESPONSE_DESIGN_HASH_MISMATCH" in failed["failure_reasons"]


def test_seed_manifest_requires_exact_four_unit_cartesian_product(tmp_path):
    ctx = _make_contracts(tmp_path, n_units=3)
    with pytest.raises(RuntimeError, match="exactly four unique fit units"):
        _run(ctx)


def test_pooled_low_joint_support_blocks_every_continuous_component_surface(tmp_path):
    ctx = _make_contracts(tmp_path)
    for unit in ctx["units"]:
        _write_worker(
            ctx, "c0", unit["topology_seed"], n_events=12,
            drop_scl=True,
        )

    payload = _run(ctx)

    candidate = payload["candidates"][0]
    assert all(row["artifact_integrity"] is True for row in candidate["units"])
    assert all(row["safe"] is True for row in candidate["units"])
    assert all(row["event_yield_estimable"] is True for row in candidate["units"])
    assert candidate["continuous_surface_eligible"] is False
    assert candidate["pooled_candidate"] is None
    assert any(
        reason.startswith("POOLED_OR_LOO_NOT_ESTIMABLE:")
        for reason in candidate["candidate_failure_reasons"]
    )
    assert payload["continuous_component_surface"] == []


def test_commit_ancestry_failure_is_fail_closed_and_retained(tmp_path):
    ctx = _make_contracts(tmp_path)
    for unit in ctx["units"]:
        _write_worker(ctx, "c0", unit["topology_seed"])

    payload = _run(ctx, ancestry_checker=lambda _a, _b, _root: False)

    assert len(payload["candidates"][0]["units"]) == 4
    assert payload["candidates"][0]["continuous_surface_eligible"] is False
    assert all(
        "DESIGN_COMMIT_NOT_ANCESTOR" in row["failure_reasons"]
        for row in payload["candidates"][0]["units"]
    )


def test_training_boundary_rejects_forbidden_contract_before_open(tmp_path):
    ctx = _make_contracts(tmp_path)
    forbidden = tmp_path / "patient_heldout_contract.npz"
    forbidden.write_bytes(ctx["contract_path"].read_bytes())
    qualification = json.loads(ctx["qualification_path"].read_text())
    qualification["patient_training_contract_npz"] = str(forbidden)
    qualification["patient_training_contract_sha256"] = _sha256(forbidden)
    _write_json(ctx["qualification_path"], qualification)

    with pytest.raises(RuntimeError, match="training-only boundary rejected"):
        _run(ctx)


def test_training_objective_loader_does_not_import_validation_cluster_dependency(monkeypatch):
    aggregate._OBJECTIVE_MODULE = None
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "sklearn.cluster" or name.startswith("sklearn.cluster."):
            raise AssertionError("validation clustering dependency was imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    objective = aggregate._load_training_objective()
    assert callable(objective.component_vector)


def test_candidate_specific_floor_is_deterministic_and_recruitment_thinned(tmp_path):
    ctx = _make_contracts(tmp_path, candidate_ids=("full", "censored"))
    for unit in ctx["units"]:
        _write_worker(ctx, "full", unit["topology_seed"], n_events=16)
        _write_worker(ctx, "censored", unit["topology_seed"], n_events=16)
        stem = f"censored_seed_{unit['topology_seed']}"
        npz_path = ctx["worker_dir"] / f"{stem}.npz"
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
        arrays["onsets"] = np.asarray(arrays["onsets"], float)
        arrays["onsets"][::2, 3] = np.nan
        np.savez(npz_path, **arrays)
        payload_path = ctx["worker_dir"] / f"{stem}.json"
        payload = json.loads(payload_path.read_text())
        payload["arrays"]["sha256"] = _sha256(npz_path)
        _write_json(payload_path, payload)

    first = _run(ctx)
    first_json = (ctx["output_dir"] / "fit_aggregate.json").read_bytes()
    second = _run(ctx)
    second_json = (ctx["output_dir"] / "fit_aggregate.json").read_bytes()

    assert first_json == second_json
    rows = {row["candidate_id"]: row for row in first["candidates"]}
    # Equal pooled counts share unconditional floors, while recruitment-dependent
    # conditional floors are computed under each candidate's own censoring profile.
    for component in aggregate.UNCONDITIONAL_COMPONENTS:
        assert rows["full"]["floor"][component] == rows["censored"]["floor"][component]
    assert any(
        rows["full"]["floor"][component] != rows["censored"]["floor"][component]
        for component in aggregate.CONDITIONAL_COMPONENTS
    )


@pytest.mark.parametrize(
    "flags, expect_valid",
    [
        ({"runtime_modules_match_expected_commit": 1, "runtime_modules_dirty": 0}, True),
        ({"runtime_modules_match_expected_commit": True, "runtime_modules_dirty": False}, True),
        ({"runtime_modules_match_expected_commit": 0, "runtime_modules_dirty": 0}, False),
        ({"runtime_modules_match_expected_commit": 1, "runtime_modules_dirty": 1}, False),
        ({"runtime_modules_match_expected_commit": _ABSENT, "runtime_modules_dirty": 0}, False),
        ({"runtime_modules_match_expected_commit": 1, "runtime_modules_dirty": _ABSENT}, False),
        ({"runtime_modules_match_expected_commit": "yes", "runtime_modules_dirty": 0}, False),
        ({"runtime_modules_match_expected_commit": 2, "runtime_modules_dirty": 0}, False),
        ({"runtime_modules_match_expected_commit": 1, "runtime_modules_dirty": -1}, False),
    ],
)
def test_provenance_flags_accept_worker_ints_and_fail_closed(tmp_path, flags, expect_valid):
    ctx = _make_contracts(tmp_path)
    for unit in ctx["units"]:
        _write_worker(ctx, "c0", unit["topology_seed"], n_events=14)
    stem = f"c0_seed_{ctx['units'][0]['topology_seed']}"
    _set_provenance(ctx, stem, **flags)

    payload = _run(ctx)
    first = ctx["units"][0]
    unit = next(
        row for row in payload["candidates"][0]["units"]
        if row["topology_seed"] == first["topology_seed"]
        and row["dynamics_seed"] == first["dynamics_seed"]
    )
    assert unit["artifact_integrity"] is expect_valid
    if expect_valid:
        assert unit["inventory_status"] == "PRESENT_VALIDATED"
    else:
        assert unit["inventory_status"] == "INVALID_ARTIFACT"
        assert any(reason.startswith("RUNTIME_MODULES_") for reason in unit["failure_reasons"])
