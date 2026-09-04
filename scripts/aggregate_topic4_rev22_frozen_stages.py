#!/usr/bin/env python3
"""Aggregate rev22 Task 9/10 frozen stages using training inputs only.

The producer inventories the exact frozen-candidate by topology-unit grids for
qualification and confirmation.  It reuses the fit-stage artifact/provenance
validator and the frozen four-component objective.  Candidate estimates and every
bootstrap replicate concatenate the selected topology event tables before nonlinear
components are recomputed.

No held-out endpoint, KMeans/OOD artifact, or patient ictal input is read here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE = ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_interictal_identifiability"
DEFAULT_FROZEN = STAGE / "response_fit/frozen_candidates.json"
DEFAULT_RESPONSE_FIT = STAGE / "response_fit/response_fit.json"
DEFAULT_MANIFEST = STAGE / "response_design/execution_candidate_manifest.json"
DEFAULT_SEEDS = STAGE / "response_design/seed_manifest.json"
DEFAULT_OBJECTIVE = STAGE / "objective_qualification/objective_qualification.json"
DEFAULT_OUT = STAGE / "frozen_stage_aggregate"
PHASES = ("qualification", "confirmation")
EXPECTED_UNITS = {"qualification": 6, "confirmation": 12}
COMPONENTS = ("D_support", "D_order", "D_lag", "D_cover")
FORBIDDEN_MARKERS = (
    "heldout", "held_out", "held-out", "kmeans", "ood", "out_of_distribution",
    "patient_ictal", "seizure", "fig3", "fig5", "early_ictal",
)


def _load_fit_module():
    spec = importlib.util.spec_from_file_location(
        "aggregate_topic4_rev22_fit_reused",
        ROOT / "scripts/aggregate_topic4_rev22_fit.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load aggregate_topic4_rev22_fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIT = _load_fit_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _guard(path: Path, role: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    normalized = str(resolved).lower().replace("interictal", "")
    if "ictal" in normalized or any(marker in normalized for marker in FORBIDDEN_MARKERS):
        raise RuntimeError(f"training-only boundary rejected {role}: {resolved}")
    return resolved


def _read(path: Path, role: str) -> tuple[dict, str]:
    path = _guard(path, role)
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    return json.loads(path.read_text(encoding="utf-8")), _sha256(path)


def _phase_units(seed_manifest: Mapping[str, Any], phase: str) -> list[dict]:
    rows = list((seed_manifest.get(phase) or {}).get("units") or [])
    expected = EXPECTED_UNITS[phase]
    keys = [(int(row["topology_seed"]), int(row["dynamics_seed"])) for row in rows]
    if len(keys) != expected or len(set(keys)) != expected:
        raise RuntimeError(f"{phase} must contain exactly {expected} unique units")
    if len({topology for topology, _ in keys}) != expected:
        raise RuntimeError(f"{phase} topology seeds must be distinct")
    return [{"topology_seed": topology, "dynamics_seed": dynamics}
            for topology, dynamics in keys]


def _candidate_index(manifest: Mapping[str, Any], frozen: Mapping[str, Any]) -> tuple[list[str], dict]:
    rows = {str(row["candidate_id"]): row for row in manifest.get("candidates", [])}
    ids = [str(value) for value in frozen.get("candidate_ids", [])]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("frozen candidate ids are empty or duplicated")
    missing = sorted(set(ids) - set(rows))
    if missing:
        raise RuntimeError(f"frozen candidates absent from execution manifest: {missing}")
    return ids, rows


def _verify_bindings(
    *, frozen: Mapping[str, Any], response_fit_path: Path,
    response_fit_hash: str, manifest_hash: str, seed_hash: str,
    response_fit: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if frozen.get("status") != "REV22_CANDIDATES_FROZEN":
        raise RuntimeError("Task 8 candidate freeze is not execution-bound")
    expected = {
        "execution_candidate_manifest_sha256": manifest_hash,
        "seed_manifest_sha256": seed_hash,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen candidate binding mismatch: {key}")
    response_record = (frozen.get("input_hashes") or {}).get("response_fit") or {}
    if response_record.get("sha256") != response_fit_hash:
        raise RuntimeError("frozen candidate response-fit hash mismatch")
    if Path(str(response_record.get("path", ""))).expanduser().resolve() != response_fit_path:
        raise RuntimeError("frozen candidate response-fit path mismatch")
    if manifest.get("seed_manifest_sha256") != seed_hash:
        raise RuntimeError("execution manifest seed binding mismatch")
    if manifest.get("response_design_manifest_sha256") != frozen.get(
        "response_design_manifest_sha256"
    ):
        raise RuntimeError("execution manifest response-design binding mismatch")
    if response_fit.get("training_only") is not True or response_fit.get("status") != "RESPONSE_FIT_COMPLETE":
        raise RuntimeError("response fit is not a completed training-only artifact")
    if response_fit.get("input_hashes", {}).get("seed_manifest", {}).get("sha256") != seed_hash:
        raise RuntimeError("response fit seed binding mismatch")


def _score_tables(
    tables: Sequence[np.ndarray], *, training: Mapping[str, Any], objective: Any,
    minimum_events: int = 0,
) -> dict:
    if not tables:
        raise ValueError("at least one topology event table is required")
    combined = np.concatenate([np.asarray(table, float) for table in tables], axis=0)
    if len(combined) < int(minimum_events):
        raise RuntimeError(
            f"pooled event yield {len(combined)} is below the registered minimum "
            f"{minimum_events}"
        )
    vector = objective.component_vector(
        combined, training["reference"], training["groups"], training["pairs"],
        training["embedding"], composite=False,
    )
    values = {}
    for component in COMPONENTS:
        if vector[component]["status"] != objective.STATUS_OK:
            raise RuntimeError(f"pooled {component} is not estimable")
        values[component] = float(vector[component]["value"])
    return {"n_events": int(len(combined)), "components": values,
            "recruitment_profile": objective.recruitment_profile(combined)}


def _patient_floors(
    score: Mapping[str, Any], *, candidate_id: str, block_views: Any,
    objective: Any, floor_draws: int, floor_seed: int,
) -> dict:
    n_events = int(score["n_events"])
    requests = [
        {"key": f"n{n_events}", "n": n_events,
         "components": FIT.UNCONDITIONAL_COMPONENTS, "thin_profile": None},
        {"key": f"candidate::{candidate_id}", "n": n_events,
         "components": FIT.CONDITIONAL_COMPONENTS,
         "thin_profile": score["recruitment_profile"]},
    ]
    floors = objective.block_split_floors(
        block_views, requests, draws=floor_draws, seed=floor_seed,
        n_pair_min=int(objective.N_PAIR_MIN), cover_quantile=0.90,
    )
    standardized = {}
    records = {}
    for component in COMPONENTS:
        key = (f"candidate::{candidate_id}" if component in FIT.CONDITIONAL_COMPONENTS
               else f"n{n_events}")
        floor = (floors.get(key) or {}).get(component)
        if not floor or floor.get("q50") is None or floor.get("q95") is None:
            raise RuntimeError(f"{component} patient floor is not estimable")
        if not np.isfinite(float(floor["q50"])) or not np.isfinite(float(floor["q95"])):
            raise RuntimeError(f"{component} patient floor is non-finite")
        if float(floor["q95"]) <= float(floor["q50"]):
            raise RuntimeError(f"{component} patient floor has zero width")
        records[component] = floor
        standardized[component] = float(objective.standardized_excess(
            score["components"][component], floor,
        ))
    return {"floor": records, "standardized_Z": standardized}


def _bootstrap_candidate(
    tables: Sequence[np.ndarray], *, training: Mapping[str, Any], objective: Any,
    draws: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    replicates = {component: [] for component in COMPONENTS}
    n_units = len(tables)
    invalid = 0
    for _ in range(int(draws)):
        indices = rng.integers(0, n_units, size=n_units)
        try:
            values = _score_tables(
                [tables[index] for index in indices], training=training, objective=objective,
                minimum_events=FIT.MIN_RETURNED_FAMILIES * n_units,
            )["components"]
        except RuntimeError:
            invalid += 1
            continue
        for component in COMPONENTS:
            replicates[component].append(values[component])
    valid_fraction = (int(draws) - invalid) / int(draws)
    output = {}
    for component, values in replicates.items():
        stable = bool(values and valid_fraction >= 0.80)
        output[component] = {
            "status": "OK" if stable else "BOOTSTRAP_SUPPORT_UNSTABLE",
            "median": float(np.median(values)) if stable else None,
            "q05": float(np.quantile(values, 0.05)) if stable else None,
            "q95": float(np.quantile(values, 0.95)) if stable else None,
            "valid_draws": len(values), "invalid_draws": invalid,
            "valid_fraction": float(valid_fraction),
        }
    return output


def _paired_bootstrap(
    left_tables: Sequence[np.ndarray], right_tables: Sequence[np.ndarray], *,
    training: Mapping[str, Any], objective: Any, draws: int, seed: int,
) -> dict:
    if len(left_tables) != len(right_tables) or not left_tables:
        raise ValueError("paired bootstrap requires aligned topology tables")
    rng = np.random.default_rng(seed)
    replicates = {component: [] for component in COMPONENTS}
    invalid = 0
    for _ in range(int(draws)):
        indices = rng.integers(0, len(left_tables), size=len(left_tables))
        try:
            left = _score_tables(
                [left_tables[index] for index in indices], training=training,
                objective=objective,
                minimum_events=FIT.MIN_RETURNED_FAMILIES * len(left_tables),
            )["components"]
            right = _score_tables(
                [right_tables[index] for index in indices], training=training,
                objective=objective,
                minimum_events=FIT.MIN_RETURNED_FAMILIES * len(right_tables),
            )["components"]
        except RuntimeError:
            invalid += 1
            continue
        for component in COMPONENTS:
            # Positive means the left candidate has lower training distance.
            replicates[component].append(right[component] - left[component])
    valid_fraction = (int(draws) - invalid) / int(draws)
    output = {}
    for component, values in replicates.items():
        stable = bool(values and valid_fraction >= 0.80)
        output[component] = {
            "status": "OK" if stable else "BOOTSTRAP_SUPPORT_UNSTABLE",
            "median": float(np.median(values)) if stable else None,
            "q05": float(np.quantile(values, 0.05)) if stable else None,
            "q95": float(np.quantile(values, 0.95)) if stable else None,
            "valid_draws": len(values), "invalid_draws": invalid,
            "valid_fraction": float(valid_fraction),
        }
    return output


def _prediction_rows(response_fit: Mapping[str, Any], frozen: Mapping[str, Any]) -> list[dict]:
    rows = []
    proposals = response_fit.get("proposals") or {}
    for mask, candidate_ids in sorted((frozen.get("mask_to_candidates") or {}).items()):
        proposal = proposals.get(mask) or {}
        points = proposal.get("frozen_points") or []
        by_id = {str(point.get("execution_candidate_id")): point for point in points}
        for candidate_id in candidate_ids or []:
            candidate_id = str(candidate_id)
            point = by_id.get(candidate_id)
            if point is None:
                rows.append({"mask": mask, "candidate_id": candidate_id,
                             "status": "PREDICTION_NOT_FOUND"})
                continue
            origin = str(point.get("origin", ""))
            prediction = proposal.get(origin) if origin in ("gp", "tree") else None
            if not isinstance(prediction, Mapping):
                rows.append({"mask": mask, "candidate_id": candidate_id, "origin": origin,
                             "status": "NO_SURFACE_PREDICTION_FOR_FALLBACK"})
                continue
            rows.append({
                "mask": mask, "candidate_id": candidate_id, "origin": origin,
                "status": "OK", "predicted_excess": prediction.get("predicted_excess") or {},
                "predicted_sd": prediction.get("predicted_sd") or {},
            })
    return rows


def _qualification_prediction_audit(
    candidate_rows: Mapping[str, Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
) -> list[dict]:
    output = []
    for prediction in predictions:
        row = dict(prediction)
        candidate = candidate_rows.get(str(row["candidate_id"]))
        if row.get("status") != "OK" or not candidate or not candidate.get("primary_estimable"):
            row.update(observed_standardized_Z=None, residual=None, interval_90=None,
                       covered_90=None)
            output.append(row)
            continue
        observed = candidate["standardized_Z"]
        residual, intervals, covered = {}, {}, {}
        for component in COMPONENTS:
            mean = row["predicted_excess"].get(component)
            sd = row["predicted_sd"].get(component)
            if mean is None or sd is None:
                residual[component] = intervals[component] = covered[component] = None
                continue
            lower = max(0.0, float(mean) - 1.645 * float(sd))
            upper = float(mean) + 1.645 * float(sd)
            value = float(observed[component])
            residual[component] = value - float(mean)
            intervals[component] = {"lower": lower, "upper": upper}
            covered[component] = bool(lower <= value <= upper)
        row.update(observed_standardized_Z=observed, residual=residual,
                   interval_90=intervals, covered_90=covered)
        output.append(row)
    return output


def _reference_and_locked_pairs(frozen: Mapping[str, Any]) -> tuple[list[str], list[tuple]]:
    masks = frozen.get("mask_to_candidates") or {}
    reference = [str(value) for value in masks.get("M0000", []) if value is not None]
    if len(reference) != 1:
        raise RuntimeError("frozen stages require exactly one M0000 reference candidate")
    branch = str(frozen.get("branch", ""))
    if "PRIMARY" in branch.upper() or "4D" in branch.upper():
        full_mask, locked_masks = "M1111", ("M0111", "M1011", "M1101", "M1110")
    else:
        full_mask, locked_masks = "M1100", ("M1000", "M0100")
    pairs = []
    for full in masks.get(full_mask, []) or []:
        for locked_mask in locked_masks:
            for locked in masks.get(locked_mask, []) or []:
                pairs.append((str(full), str(locked), full_mask, locked_mask))
    return reference, pairs


def aggregate_frozen_stages(
    *, frozen_candidates_path: Path, response_fit_path: Path,
    candidate_manifest_path: Path, seed_manifest_path: Path,
    objective_qualification_path: Path, qualification_worker_dir: Path,
    confirmation_worker_dir: Path, output_dir: Path,
    bootstrap_draws: int = 2000, bootstrap_seed: int = 20260911,
    repo_root: Path = ROOT,
    ancestry_checker: Callable[[str, str, Path], bool] = FIT._git_is_ancestor,
) -> dict:
    if int(bootstrap_draws) < 1:
        raise ValueError("bootstrap_draws must be positive")
    paths = {
        "frozen_candidates": frozen_candidates_path,
        "response_fit": response_fit_path,
        "candidate_manifest": candidate_manifest_path,
        "seed_manifest": seed_manifest_path,
        "objective_qualification": objective_qualification_path,
        "qualification_worker_dir": qualification_worker_dir,
        "confirmation_worker_dir": confirmation_worker_dir,
    }
    for role, path in paths.items():
        paths[role] = _guard(path, role)

    frozen, frozen_hash = _read(paths["frozen_candidates"], "frozen candidates")
    response_fit, response_fit_hash = _read(paths["response_fit"], "response fit")
    manifest, manifest_hash = _read(paths["candidate_manifest"], "execution candidate manifest")
    seeds, seed_hash = _read(paths["seed_manifest"], "seed manifest")
    qualification, qualification_hash = _read(
        paths["objective_qualification"], "objective qualification",
    )
    _verify_bindings(
        frozen=frozen,
        response_fit_path=paths["response_fit"], response_fit_hash=response_fit_hash,
        manifest_hash=manifest_hash, seed_hash=seed_hash,
        response_fit=response_fit, manifest=manifest,
    )
    if qualification.get("status") != "OBJECTIVE_QUALIFIED" or qualification.get(
        "forbidden_inputs_loaded"
    ) is not False:
        raise RuntimeError("formal training objective is not safely qualified")
    if tuple(qualification.get("components") or ()) != COMPONENTS:
        raise RuntimeError("training component contract drifted")

    candidate_ids, candidate_index = _candidate_index(manifest, frozen)
    objective = FIT._load_training_objective()
    training = FIT._load_training_contract(
        paths["objective_qualification"], qualification,
        str(qualification["patient_training_contract_sha256"]),
    )
    training["reference"] = objective.patient_reference(
        training["onsets_ms"], training["groups"], training["pairs"], training["embedding"],
    )
    block_views = objective.PatientBlockViews(
        training["onsets_ms"], training["block_ids"], training["groups"], training["pairs"],
        training["embedding"], lag_cap_ms=float(objective.LAG_CAP_MS),
    )
    floor_contract = qualification.get("floors") or {}
    floor_draws, floor_seed = int(floor_contract.get("draws", 0)), int(floor_contract.get("seed", -1))
    if floor_draws < 1 or floor_seed < 0:
        raise RuntimeError("invalid frozen patient-floor contract")

    design_hash = str(manifest.get("response_design_manifest_sha256") or "")
    response_commit = str(manifest.get("git_commit") or "")
    objective_commit = str(qualification.get("git_commit") or "")
    phase_results, all_unit_rows = {}, []
    tables_by_phase: dict[str, dict[str, list[np.ndarray]]] = {}
    worker_hashes = {}
    for phase in PHASES:
        units = _phase_units(seeds, phase)
        worker_dir = paths[f"{phase}_worker_dir"]
        candidates = {}
        tables_by_phase[phase] = {}
        for candidate_id in candidate_ids:
            candidate = candidate_index[candidate_id]
            rows, tables = [], []
            for unit in units:
                topology, dynamics = unit["topology_seed"], unit["dynamics_seed"]
                row, onsets = FIT._validate_and_score_unit(
                    candidate=candidate, topology_seed=topology, dynamics_seed=dynamics,
                    worker_dir=worker_dir, response_hash=design_hash, seed_hash=seed_hash,
                    response_commit=response_commit, objective_commit=objective_commit,
                    training=training, objective=objective, ancestry_checker=ancestry_checker,
                    repo_root=repo_root,
                )
                row = {**row, "phase": phase}
                rows.append(row)
                all_unit_rows.append(row)
                key = f"{phase}::{candidate_id}::{topology}::{dynamics}"
                worker_hashes[key] = row.get("input_hashes") or None
                if onsets is not None:
                    tables.append(onsets)
            expected = EXPECTED_UNITS[phase]
            all_safe = bool(
                len(rows) == expected
                and len(tables) == expected
                and all(row["artifact_integrity"] and row["safe"] for row in rows)
            )
            record = {
                "candidate_id": candidate_id,
                "family_membership": candidate.get("family_membership") or [],
                "expected_units": expected,
                "artifact_integrity_units": int(sum(row["artifact_integrity"] for row in rows)),
                "safe_units": int(sum(row["safe"] for row in rows)),
                "yield_estimable_units": int(sum(row["event_yield_estimable"] for row in rows)),
                "conditional_estimable_units": int(sum(row["conditional_estimable"] for row in rows)),
                "primary_estimable": False,
                "n_returned_families_by_unit": [int(row["n_returned_families"]) for row in rows],
                "failure_reasons": sorted({reason for row in rows for reason in row["failure_reasons"]}),
                "pooled": None, "floor": None, "standardized_Z": None,
                "bootstrap_90": None, "leave_one_topology_out": [], "units": rows,
            }
            if all_safe:
                try:
                    pooled = _score_tables(
                        tables, training=training, objective=objective,
                        minimum_events=FIT.MIN_RETURNED_FAMILIES * expected,
                    )
                    leave_out = []
                    for omitted, unit in enumerate(units):
                        score = _score_tables(
                            [table for index, table in enumerate(tables) if index != omitted],
                            training=training, objective=objective,
                            minimum_events=FIT.MIN_RETURNED_FAMILIES * (expected - 1),
                        )
                        leave_out.append({
                            "omitted_topology_seed": int(unit["topology_seed"]), **score,
                        })
                except RuntimeError as error:
                    record["failure_reasons"].append(
                        f"POOLED_OR_LOO_NOT_ESTIMABLE:{error}"
                    )
                else:
                    normalization = _patient_floors(
                        pooled, candidate_id=candidate_id, block_views=block_views,
                        objective=objective, floor_draws=floor_draws, floor_seed=floor_seed,
                    )
                    record.update(
                        primary_estimable=True,
                        pooled={"method": "pool_topology_events_then_recompute", **pooled},
                        floor=normalization["floor"],
                        standardized_Z=normalization["standardized_Z"],
                        bootstrap_90=_bootstrap_candidate(
                            tables, training=training, objective=objective,
                            draws=bootstrap_draws,
                            seed=bootstrap_seed + (0 if phase == "qualification" else 10000),
                        ),
                        leave_one_topology_out=leave_out,
                    )
                    tables_by_phase[phase][candidate_id] = tables
            else:
                record["failure_reasons"].append("MISSING_UNSAFE_OR_NONFINITE_STAGE_UNIT")
            candidates[candidate_id] = record
        phase_results[phase] = {
            "expected_units_per_candidate": EXPECTED_UNITS[phase],
            "candidate_count": len(candidate_ids),
            "expected_unit_count": len(candidate_ids) * EXPECTED_UNITS[phase],
            "primary_estimable_candidates": int(sum(row["primary_estimable"] for row in candidates.values())),
            "candidates": candidates,
        }

    predictions = _prediction_rows(response_fit, frozen)
    qualification_prediction = _qualification_prediction_audit(
        phase_results["qualification"]["candidates"], predictions,
    )
    reference, locked_pairs = _reference_and_locked_pairs(frozen)
    reference_id = reference[0]
    confirmation_candidates = phase_results["confirmation"]["candidates"]

    def contrast(left: str, right: str, label: str, extra: Mapping[str, Any]) -> dict:
        left_row, right_row = confirmation_candidates[left], confirmation_candidates[right]
        if not left_row["primary_estimable"] or not right_row["primary_estimable"]:
            return {"contrast": label, "left": left, "right": right,
                    "status": "TRAINING_COMPONENT_NOT_ESTIMABLE", **dict(extra),
                    "paired_90": None}
        point_estimate = {
            component: float(
                right_row["pooled"]["components"][component]
                - left_row["pooled"]["components"][component]
            )
            for component in COMPONENTS
        }
        return {
            "contrast": label, "left": left, "right": right, "status": "OK", **dict(extra),
            "orientation": "positive means left has lower training distance",
            "point_estimate": point_estimate,
            "paired_90": _paired_bootstrap(
                tables_by_phase["confirmation"][left], tables_by_phase["confirmation"][right],
                training=training, objective=objective, draws=bootstrap_draws,
                seed=bootstrap_seed + 20000,
            ),
        }

    candidate_vs_reference = [
        contrast(candidate_id, reference_id, "candidate_vs_M0000", {})
        for candidate_id in candidate_ids if candidate_id != reference_id
    ]
    full_vs_locked = [
        contrast(full, locked, "full_vs_locked", {"full_mask": full_mask,
                                                   "locked_mask": locked_mask})
        for full, locked, full_mask, locked_mask in locked_pairs
    ]

    expected_total = len(candidate_ids) * sum(EXPECTED_UNITS.values())
    complete_grid = len(all_unit_rows) == expected_total and all(
        row["artifact_integrity"] for row in all_unit_rows
    )
    output_dir = Path(output_dir).expanduser().resolve()
    payload = {
        "schema_id": "topic4_rev22_dci_training_only_frozen_stages_v1",
        "status": "FROZEN_STAGE_AGGREGATE_COMPLETE" if complete_grid else "FROZEN_STAGE_ARTIFACT_GRID_INCOMPLETE",
        "scientific_role": "training_only_task9_task10_stage_aggregate",
        "training_only": True, "forbidden_inputs_loaded": False,
        "component_contract": list(COMPONENTS),
        "candidate_statistic": "topology event tables pooled before every nonlinear component calculation",
        "bootstrap_contract": {
            "unit": "paired topology seed", "draws": int(bootstrap_draws),
            "seed": int(bootstrap_seed), "interval": "percentile_90",
            "recompute_after_every_topology_resample": True,
            "minimum_valid_fraction": 0.80,
            "nonestimable_resamples_reported_not_imputed": True,
        },
        "eligibility_contract": {
            "per_unit_low_yield_sidecar_threshold": int(FIT.MIN_RETURNED_FAMILIES),
            "qualification_min_pooled": int(FIT.MIN_RETURNED_FAMILIES * EXPECTED_UNITS["qualification"]),
            "qualification_min_leave_one_out": int(FIT.MIN_RETURNED_FAMILIES * (EXPECTED_UNITS["qualification"] - 1)),
            "confirmation_min_pooled": int(FIT.MIN_RETURNED_FAMILIES * EXPECTED_UNITS["confirmation"]),
            "confirmation_min_leave_one_out": int(FIT.MIN_RETURNED_FAMILIES * (EXPECTED_UNITS["confirmation"] - 1)),
            "missing_runaway_nonfinite_fail_closed": True,
            "failed_units_retained_in_feasibility_sidecar": True,
        },
        "inventory": {
            "frozen_candidate_count": len(candidate_ids), "expected_unit_count": expected_total,
            "observed_inventory_rows": len(all_unit_rows),
            "artifact_integrity_count": int(sum(row["artifact_integrity"] for row in all_unit_rows)),
            "safe_count": int(sum(row["safe"] for row in all_unit_rows)),
            "yield_estimable_count": int(sum(row["event_yield_estimable"] for row in all_unit_rows)),
            "primary_estimable_unit_count": int(sum(row["feasibility"] for row in all_unit_rows)),
        },
        "phases": phase_results,
        "qualification_prediction_audit": qualification_prediction,
        "confirmation_contrasts": {
            "reference_candidate_id": reference_id,
            "candidate_vs_reference": candidate_vs_reference,
            "full_vs_locked": full_vs_locked,
        },
        "input_hashes": {
            "frozen_candidates": {"path": str(paths["frozen_candidates"]), "sha256": frozen_hash},
            "response_fit": {"path": str(paths["response_fit"]), "sha256": response_fit_hash},
            "candidate_manifest": {"path": str(paths["candidate_manifest"]), "sha256": manifest_hash},
            "seed_manifest": {"path": str(paths["seed_manifest"]), "sha256": seed_hash},
            "objective_qualification": {"path": str(paths["objective_qualification"]),
                                        "sha256": qualification_hash},
            "patient_training_contract": {"path": str(training["path"]),
                                           "sha256": str(qualification["patient_training_contract_sha256"])},
            "workers": worker_hashes,
        },
        "claim_boundary": (
            "Training-only Task 9/10 aggregation. Qualification prediction errors never delete "
            "a frozen candidate. No held-out, KMeans, OOD or patient ictal artifact is read."
        ),
    }
    _atomic_json(output_dir / "frozen_stage_aggregate.json", payload)
    fields = [
        "phase", "candidate_id", "topology_seed", "dynamics_seed", "inventory_status",
        "artifact_integrity", "safe", "event_yield_estimable", "conditional_estimable",
        "feasibility", "n_returned_families", "failure_reasons",
    ]
    _atomic_csv(output_dir / "frozen_stage_feasibility.csv", [
        {**row, "failure_reasons": "|".join(row["failure_reasons"])} for row in all_unit_rows
    ], fields)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-candidates", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--response-fit", type=Path, default=DEFAULT_RESPONSE_FIT)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--objective-qualification", type=Path, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--qualification-worker-dir", type=Path, default=STAGE / "qualification/workers")
    parser.add_argument("--confirmation-worker-dir", type=Path, default=STAGE / "confirmation/workers")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    args = parser.parse_args()
    payload = aggregate_frozen_stages(
        frozen_candidates_path=args.frozen_candidates, response_fit_path=args.response_fit,
        candidate_manifest_path=args.candidate_manifest, seed_manifest_path=args.seed_manifest,
        objective_qualification_path=args.objective_qualification,
        qualification_worker_dir=args.qualification_worker_dir,
        confirmation_worker_dir=args.confirmation_worker_dir, output_dir=args.out_dir,
        bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": payload["status"], **payload["inventory"],
                      "output": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
