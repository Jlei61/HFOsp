#!/usr/bin/env python3
"""Fit descriptive-only rev22 Task 11 validation response surfaces.

This producer consumes the frozen 96-point descriptive validation aggregate.  It
never proposes, ranks, selects, or rewrites a candidate and never runs an SNN.  The
surfaces exist only to draw conditional one- and two-dimensional validation views
around the already-frozen full-model coordinate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.topic4_rev22_response_surface import (  # noqa: E402
    PARAMETER_ORDER,
    fit_component_gp,
    pooled_shrinkage_noise,
)


STAGE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability"
)
DEFAULT_VALIDATION = STAGE / "validation/validation_aggregate.json"
DEFAULT_DESIGN = STAGE / "response_design/response_design_manifest.json"
DEFAULT_FROZEN = STAGE / "response_fit/frozen_candidates.json"
DEFAULT_OUTPUT = STAGE / "validation/validation_response_surface.json"

VALIDATION_SCHEMA = "topic4_rev22_dci_validation_aggregate_v1"
DESIGN_SCHEMA = "topic4_rev22_dci_response_design_manifest_v1"
FROZEN_SCHEMA = "topic4_rev22_dci_frozen_candidates_v1"
OUTPUT_SCHEMA = "topic4_rev22_dci_validation_response_surface_v1"
EXPECTED_POINTS = 96
ENDPOINTS = (
    "D_support", "D_order", "D_time_ms", "recall", "kmeans_alignment", "ood",
)
HIGHER_IS_BETTER = frozenset(("recall", "kmeans_alignment"))
Z90 = 1.6448536269514722


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path, role: str, schema: str) -> tuple[dict, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {role}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_id") != schema:
        raise RuntimeError(f"unexpected {role} schema: {payload.get('schema_id')}")
    return payload, _sha256(path)


def _recorded_hash(payload: Mapping, key: str) -> str | None:
    value = (payload.get("input_hashes") or {}).get(key)
    if isinstance(value, Mapping):
        value = value.get("sha256")
    return str(value) if value else None


def _require_hash(recorded: str | None, observed: str, label: str) -> None:
    if recorded != observed:
        raise RuntimeError(f"broken frozen hash binding: {label}")


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _design_rows(design: Mapping) -> tuple[list[dict], dict[str, list[float]], dict]:
    rows = design.get("candidates")
    if not isinstance(rows, list) or len(rows) != EXPECTED_POINTS:
        raise RuntimeError("response design must contain exactly 96 candidates")
    if int(design.get("candidate_count", -1)) != EXPECTED_POINTS:
        raise RuntimeError("response-design candidate_count is not 96")
    ids = [str(row.get("candidate_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("response-design candidate ids are duplicated")
    bounds = design.get("bounds") or {}
    domain = {}
    for name in PARAMETER_ORDER:
        values = bounds.get(name)
        if not isinstance(values, Sequence) or len(values) != 2:
            raise RuntimeError(f"response design lacks bounds for {name}")
        lo, hi = map(float, values)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise RuntimeError(f"invalid response-design bounds for {name}")
        domain[name] = [lo, hi]
    physical = {}
    for row in rows:
        block = row.get("physical") or {}
        vector = [_finite(block.get(name)) for name in PARAMETER_ORDER]
        if any(value is None for value in vector):
            raise RuntimeError(f"candidate {row.get('candidate_id')} lacks physical coordinates")
        physical[str(row["candidate_id"])] = [float(value) for value in vector]
    return rows, physical, domain


def _fit_descriptive_rows(validation: Mapping) -> list[dict]:
    block = validation.get("fit_descriptive")
    rows = block.get("candidates") if isinstance(block, Mapping) else block
    if not isinstance(rows, list) or len(rows) != EXPECTED_POINTS:
        raise RuntimeError("validation aggregate must contain all 96 fit_descriptive candidates")
    return rows


def _yield(row: Mapping) -> int:
    secondary = row.get("secondary") or {}
    value = secondary.get("yield_total", row.get("yield_total", 0))
    return int(value) if _finite(value) is not None else 0


def _unit_values(row: Mapping, endpoint: str) -> list[float]:
    values = []
    for unit in row.get("unit_endpoints") or []:
        value = _finite(unit.get(endpoint))
        if value is not None:
            values.append(value)
    return values


def _unit_value_map(row: Mapping, endpoint: str) -> dict[int, float]:
    values = {}
    for unit in row.get("unit_endpoints") or []:
        topology_seed = unit.get("topology_seed")
        value = _finite(unit.get(endpoint))
        if topology_seed is not None and value is not None:
            values[int(topology_seed)] = float(value)
    return values


def _candidate_table(validation: Mapping, design: Mapping) -> tuple[list[dict], dict]:
    design_rows, physical, domain = _design_rows(design)
    rows = _fit_descriptive_rows(validation)
    by_id = {str(row.get("candidate_id")): row for row in rows}
    if len(by_id) != EXPECTED_POINTS or set(by_id) != set(physical):
        raise RuntimeError("fit_descriptive candidate grid differs from the frozen response design")
    table = []
    for design_row in design_rows:  # preserve the original design order; never rank candidates
        candidate_id = str(design_row["candidate_id"])
        row = by_id[candidate_id]
        endpoints = row.get("primary_endpoints") or {}
        status = str(row.get("primary_status", "MISSING_STATUS"))
        values = {name: _finite(endpoints.get(name)) for name in ENDPOINTS}
        not_estimable = {name: values[name] is None for name in ENDPOINTS}
        table.append({
            "candidate_id": candidate_id,
            "physical": dict(zip(PARAMETER_ORDER, physical[candidate_id])),
            "primary_status": status,
            "endpoints": values,
            "unit_values": {name: _unit_values(row, name) for name in ENDPOINTS},
            "unit_value_maps": {name: _unit_value_map(row, name) for name in ENDPOINTS},
            "heldout_count_matched_floors": row.get("heldout_count_matched_floors") or {},
            "yield_total": _yield(row),
            "not_estimable": not_estimable,
        })
    return table, domain


def _reference_id(design: Mapping) -> str:
    rows = design.get("candidates") or []
    references = [str(row.get("candidate_id")) for row in rows if row.get("is_reference")]
    if len(references) != 1:
        raise RuntimeError("response design must identify exactly one M0000 reference")
    return references[0]


def _display_benchmarks(validation: Mapping, reference: Mapping) -> dict[str, dict]:
    floors = reference.get("heldout_count_matched_floors") or {}
    patient = validation.get("patient_recording_block_benchmark") or {}
    floor_keys = {"D_support": "D_support", "D_order": "D_order", "D_time_ms": "D_lag"}
    targets: dict[str, tuple[float | None, str]] = {}
    for endpoint, floor_key in floor_keys.items():
        target = _finite((floors.get(floor_key) or {}).get("q95"))
        targets[endpoint] = (target, "patient split-block q95")
    targets["recall"] = (1.0, "ideal fixed-budget recall")
    targets["kmeans_alignment"] = (
        _finite(patient.get("balanced_alignment")), "patient held-out natural KMeans alignment",
    )
    targets["ood"] = (0.0, "ideal zero OOD")

    output = {}
    for endpoint in ENDPOINTS:
        reference_value = _finite(reference["endpoints"].get(endpoint))
        target_value, target_role = targets[endpoint]
        if reference_value is None or target_value is None:
            raise RuntimeError(f"display benchmark is not estimable for {endpoint}")
        direction = 1.0 if endpoint in HIGHER_IS_BETTER else -1.0
        gap = direction * (target_value - reference_value)
        if not np.isfinite(gap) or gap <= 1e-12:
            raise RuntimeError(
                f"display benchmark does not improve on M0000 for {endpoint}: "
                f"reference={reference_value}, target={target_value}"
            )
        output[endpoint] = {
            "reference_raw": float(reference_value),
            "benchmark_raw": float(target_value),
            "benchmark_role": target_role,
            "oriented_gap_raw": float(gap),
            "score_definition": "oriented(candidate-reference)/oriented(benchmark-reference)",
            "reference_score": 0.0,
            "benchmark_score": 1.0,
        }
    return output


def _add_display_scores(table: list[dict], reference_id: str,
                        benchmarks: Mapping[str, Mapping]) -> None:
    reference = next((row for row in table if row["candidate_id"] == reference_id), None)
    if reference is None:
        raise RuntimeError("M0000 reference is absent from the 96-point response design")
    for row in table:
        row["display_score"] = {}
        row["display_unit_values"] = {}
        for endpoint in ENDPOINTS:
            direction = 1.0 if endpoint in HIGHER_IS_BETTER else -1.0
            gap = float(benchmarks[endpoint]["oriented_gap_raw"])
            value = row["endpoints"][endpoint]
            reference_value = reference["endpoints"][endpoint]
            row["display_score"][endpoint] = (
                None if value is None or reference_value is None
                else direction * (float(value) - float(reference_value)) / gap
            )
            candidate_units = row["unit_value_maps"][endpoint]
            reference_units = reference["unit_value_maps"][endpoint]
            shared = sorted(set(candidate_units) & set(reference_units))
            row["display_unit_values"][endpoint] = [
                direction * (candidate_units[seed] - reference_units[seed]) / gap
                for seed in shared
            ]


def _full_coordinate(frozen: Mapping, physical: Mapping[str, Sequence[float]]) -> tuple[str, list[float]]:
    branch = str(frozen.get("branch", ""))
    mask = "M1100" if "fallback" in branch.lower() else "M1111"
    ids = [str(value) for value in (frozen.get("mask_to_candidates") or {}).get(mask, []) if value]
    if len(ids) != 1:
        raise RuntimeError(f"frozen full-model family {mask} must contain exactly one coordinate")
    candidate_id = ids[0]
    if candidate_id in physical:
        return candidate_id, [float(value) for value in physical[candidate_id]]
    coordinate_maps = (
        frozen.get("candidate_parameters"), frozen.get("physical_by_candidate"),
    )
    for mapping in coordinate_maps:
        if isinstance(mapping, Mapping) and candidate_id in mapping:
            record = mapping[candidate_id]
            vector = [_finite((record or {}).get(name)) for name in PARAMETER_ORDER]
            if all(value is not None for value in vector):
                return candidate_id, [float(value) for value in vector]
    direct = frozen.get("full_model_coordinate")
    if isinstance(direct, Mapping):
        vector = [_finite(direct.get(name)) for name in PARAMETER_ORDER]
        if all(value is not None for value in vector):
            return candidate_id, [float(value) for value in vector]
    raise RuntimeError("frozen full-model proposal lacks a bound physical coordinate")


def _noise_for_endpoint(table: Sequence[Mapping], endpoint: str) -> tuple[np.ndarray, dict]:
    sd_map: dict[str, float | None] = {}
    n_map: dict[str, int] = {}
    for row in table:
        values = np.asarray(row["display_unit_values"][endpoint], float)
        candidate_id = str(row["candidate_id"])
        n_map[candidate_id] = int(len(values))
        sd_map[candidate_id] = (
            float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else None
        )
    usable_n = {key: max(2, value) for key, value in n_map.items() if sd_map[key] is not None}
    shrink = pooled_shrinkage_noise(
        {key: sd_map[key] for key in usable_n}, usable_n,
    )
    variance = np.asarray([
        shrink["variance"].get(str(row["candidate_id"]), np.nan) for row in table
    ], float)
    return variance, {
        "method": "pooled_shrinkage_of_paired_topology_unit_improvement_sem",
        "prior_weight": 8.0,
        "s_pool_sq": shrink.get("s_pool_sq"),
        "n_candidates_with_noise": shrink.get("n_finite"),
    }


def _cv_diagnostics(X: np.ndarray, y: np.ndarray, noise: np.ndarray, domain: Mapping,
                    *, seed: int, folds: int, full_surface) -> dict:
    keep = np.isfinite(y) & np.isfinite(noise)
    Xk, yk, nk = X[keep], y[keep], noise[keep]
    n_splits = min(int(folds), len(yk))
    if n_splits < 3:
        return {"status": "NOT_ESTIMABLE", "n": int(len(yk)), "adequate": False}
    pred = np.empty(len(yk), float)
    sd = np.empty(len(yk), float)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for train, test in splitter.split(Xk):
        fold = fit_component_gp(
            Xk[train], yk[train], nk[train], domain, seed,
            n_restarts=0, kernel=full_surface.gp.kernel_, optimizer=None,
        )
        pred[test], sd[test] = fold.predict(Xk[test])
    residual = pred - yk
    rho = spearmanr(pred, yk).statistic if len(yk) >= 3 else np.nan
    observed_range = float(np.ptp(yk))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    coverage = float(np.mean(np.abs(residual) <= Z90 * np.sqrt(sd ** 2 + nk)))
    checks = {
        "spearman_at_least_0_5": bool(np.isfinite(rho) and rho >= 0.5),
        "rmse_at_most_half_observed_range": bool(observed_range > 0 and rmse <= 0.5 * observed_range),
        "coverage_90_at_least_0_6": bool(coverage >= 0.6),
    }
    return {
        "status": "OK", "n": int(len(yk)), "folds": int(n_splits),
        "rmse": rmse, "spearman": None if not np.isfinite(rho) else float(rho),
        "coverage_90": coverage, "observed_range": observed_range,
        "checks": checks, "adequate": bool(all(checks.values())),
        "consequence": "diagnostic_only_no_selection_or_optimization",
    }


def _prediction(surface, X: np.ndarray) -> dict:
    mean, sd = surface.predict(X)
    return {
        "mean": mean, "lo90": mean - Z90 * sd, "hi90": mean + Z90 * sd,
    }


def _conditional_slice(surface, center: Sequence[float], domain: Mapping,
                       parameter_index: int, points: int) -> dict:
    name = PARAMETER_ORDER[parameter_index]
    axis = np.linspace(*domain[name], int(points))
    X = np.tile(np.asarray(center, float), (len(axis), 1))
    X[:, parameter_index] = axis
    return {"axis": axis, **_prediction(surface, X)}


def _conditional_plane(surface, center: Sequence[float], domain: Mapping,
                       indices: tuple[int, int], points: int) -> dict:
    names = (PARAMETER_ORDER[indices[0]], PARAMETER_ORDER[indices[1]])
    axis0 = np.linspace(*domain[names[0]], int(points))
    axis1 = np.linspace(*domain[names[1]], int(points))
    grid0, grid1 = np.meshgrid(axis0, axis1, indexing="ij")
    X = np.tile(np.asarray(center, float), (grid0.size, 1))
    X[:, indices[0]] = grid0.ravel()
    X[:, indices[1]] = grid1.ravel()
    predicted = _prediction(surface, X)
    return {
        "parameters": list(names), "axis_0": axis0, "axis_1": axis1,
        "mean": predicted["mean"].reshape(grid0.shape),
        "lo90": predicted["lo90"].reshape(grid0.shape),
        "hi90": predicted["hi90"].reshape(grid0.shape),
    }


def fit_validation_response(*, validation_path: Path, design_path: Path,
                            frozen_path: Path, output_path: Path,
                            seed: int = 20260912, n_restarts: int = 2,
                            cv_folds: int = 8, slice_points: int = 101,
                            plane_points: int = 41) -> dict:
    validation, validation_hash = _read_json(validation_path, "validation aggregate", VALIDATION_SCHEMA)
    design, design_hash = _read_json(design_path, "response design", DESIGN_SCHEMA)
    frozen, frozen_hash = _read_json(frozen_path, "frozen candidates", FROZEN_SCHEMA)
    if validation.get("status") != "VALIDATION_AGGREGATE_COMPLETE":
        raise RuntimeError("validation aggregate is incomplete")
    if validation.get("snn_simulation_run") is not False:
        raise RuntimeError("validation aggregate does not preserve the no-simulation boundary")
    _require_hash(_recorded_hash(validation, "response_design"), design_hash,
                  "validation -> response design")
    _require_hash(_recorded_hash(validation, "frozen_candidates"), frozen_hash,
                  "validation -> frozen candidates")
    _require_hash(str(frozen.get("response_design_manifest_sha256") or ""), design_hash,
                  "frozen candidates -> response design")

    table, domain = _candidate_table(validation, design)
    reference_id = _reference_id(design)
    reference_row = next(row for row in table if row["candidate_id"] == reference_id)
    benchmarks = _display_benchmarks(validation, reference_row)
    _add_display_scores(table, reference_id, benchmarks)
    physical = {row["candidate_id"]: list(row["physical"].values()) for row in table}
    full_id, center = _full_coordinate(frozen, physical)
    X = np.asarray([physical[row["candidate_id"]] for row in table], float)
    surfaces = {}
    noise_by_endpoint: dict[str, np.ndarray] = {}
    for index, endpoint in enumerate(ENDPOINTS):
        y = np.asarray([
            np.nan if row["display_score"][endpoint] is None else row["display_score"][endpoint]
            for row in table
        ], float)
        noise, noise_record = _noise_for_endpoint(table, endpoint)
        noise_by_endpoint[endpoint] = noise
        usable = np.isfinite(y) & np.isfinite(noise)
        endpoint_record: dict[str, Any] = {
            "direction": "higher_is_better",
            "display_quantity": "fraction_of_M0000_to_benchmark_gap_closed",
            "benchmark": benchmarks[endpoint],
            "n_estimable": int(usable.sum()), "noise": noise_record,
        }
        if usable.sum() < 3:
            endpoint_record.update(
                status="NOT_ESTIMABLE", cv={"status": "NOT_ESTIMABLE", "adequate": False},
                conditional_slices=None, conditional_planes=None,
            )
            surfaces[endpoint] = endpoint_record
            continue
        surface = fit_component_gp(
            X[usable], y[usable], noise[usable], domain, seed + index,
            n_restarts=int(n_restarts),
        )
        endpoint_record.update(
            status="OK", kernel=surface.kernel_str,
            fit_warnings=list(surface.fit_warnings),
            cv=_cv_diagnostics(
                X, y, noise, domain, seed=seed + 100 + index,
                folds=cv_folds, full_surface=surface,
            ),
            conditional_slices={
                name: _conditional_slice(surface, center, domain, dim, slice_points)
                for dim, name in enumerate(PARAMETER_ORDER)
            },
            conditional_planes={
                "g_LEE_x_g_LEI": _conditional_plane(surface, center, domain, (0, 1), plane_points),
                "theta_x_AR": _conditional_plane(surface, center, domain, (2, 3), plane_points),
            },
        )
        surfaces[endpoint] = endpoint_record

    original_points = []
    for row in table:
        record = {key: value for key, value in row.items()
                  if key not in {"unit_values", "unit_value_maps", "display_unit_values",
                                 "heldout_count_matched_floors"}}
        record["observation_variance"] = {}
        for endpoint in ENDPOINTS:
            idx = len(original_points)
            variance = noise_by_endpoint[endpoint][idx]
            record["observation_variance"][endpoint] = (
                None if not np.isfinite(variance) else float(variance)
            )
        original_points.append(record)

    payload = {
        "schema_id": OUTPUT_SCHEMA,
        "status": "DESCRIPTIVE_VALIDATION_RESPONSE_COMPLETE",
        "descriptive_only": True,
        "cannot_select": True,
        "selection_permitted": False,
        "candidate_order_preserved": True,
        "snn_simulation_run": False,
        "task8_freeze_modified": False,
        "endpoint_count": len(ENDPOINTS),
        "design_point_count": len(table),
        "parameter_order": list(PARAMETER_ORDER),
        "domain": domain,
        "display_contract": {
            "quantity": "fraction_of_M0000_to_benchmark_gap_closed",
            "higher_is_better": True,
            "reference_score": 0.0,
            "benchmark_score": 1.0,
            "reference_candidate_id": reference_id,
            "benchmarks": benchmarks,
            "raw_values_retained_in": "original_design_points[].endpoints",
        },
        "frozen_full_model": {"candidate_id": full_id, "coordinate": dict(zip(PARAMETER_ORDER, center))},
        "original_design_points": original_points,
        "surfaces": surfaces,
        "input_hashes": {
            "validation_aggregate": {"path": str(Path(validation_path).resolve()), "sha256": validation_hash},
            "response_design": {"path": str(Path(design_path).resolve()), "sha256": design_hash},
            "frozen_candidates": {"path": str(Path(frozen_path).resolve()), "sha256": frozen_hash},
        },
        "claim_boundary": (
            "Post-freeze, selection-blind validation surfaces for visualization only. "
            "CV adequacy cannot trigger optimization, candidate selection, ranking, or Task 8 updates."
        ),
    }
    safe_payload = _json_safe(payload)
    _atomic_json(Path(output_path), safe_payload)
    return safe_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-aggregate", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--response-design-manifest", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--frozen-candidates", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--n-restarts", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=8)
    parser.add_argument("--slice-points", type=int, default=101)
    parser.add_argument("--plane-points", type=int, default=41)
    args = parser.parse_args()
    payload = fit_validation_response(
        validation_path=args.validation_aggregate,
        design_path=args.response_design_manifest,
        frozen_path=args.frozen_candidates,
        output_path=args.output,
        seed=args.seed, n_restarts=args.n_restarts, cv_folds=args.cv_folds,
        slice_points=args.slice_points, plane_points=args.plane_points,
    )
    print(json.dumps({
        "status": payload["status"], "descriptive_only": payload["descriptive_only"],
        "cannot_select": payload["cannot_select"], "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
