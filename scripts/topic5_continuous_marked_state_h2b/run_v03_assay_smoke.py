#!/usr/bin/env python3
"""Run the seven-world H2b v0.3 semi-synthetic assay smoke."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.v03_assay import (  # noqa: E402
    ALPHA_GRID,
    WORLDS,
    AssayTemplate,
    build_template,
    run_replicate,
    wilson_interval,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    assert_frozen_contract_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (  # noqa: E402
    assert_frozen_exploration_policy_matches,
)


PRODUCER_SCRIPT = Path(__file__).resolve()
ASSAY_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_assay.py"
_TEMPLATE: AssayTemplate | None = None


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _initialise_worker(template: AssayTemplate) -> None:
    global _TEMPLATE
    _TEMPLATE = template
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def _worker(task: tuple[str, int, int, float]) -> dict[str, Any]:
    if _TEMPLATE is None:
        raise RuntimeError("assay worker template was not initialised")
    world, seed, initial_k, effect_scale = task
    return run_replicate(
        _TEMPLATE, world, seed, initial_k=initial_k, effect_scale=effect_scale,
    )


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _supported_seizure_onsets(path: Path) -> np.ndarray:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("primary_30min_supported", "")).lower() not in {
                "true", "1",
            }:
                continue
            values[str(row["seizure_id"])] = float(row["onset_time"])
    return np.asarray(sorted(values.values()), dtype=np.float64)


def _load_template(
    v02: Path, v03: Path, subject: str, seed: int,
) -> tuple[AssayTemplate, dict]:
    root = v03 / "instrument/by_cell" / subject / f"seed_{seed}"
    manifest_path = root / "instrument_manifest.json"
    cache_path = root / "interictal_d_state_trace.npz"
    supported_seizure_path = v02 / "risk_sets" / subject / "seizures.csv"
    manifest = _json(manifest_path)
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("seizure_risk_outcome_read") is not False
        or manifest.get("source", {}).get("checkpoint", {}).get("state_frozen")
        is not True
    ):
        raise ValueError("assay template is not a frozen interictal-only trace")
    if manifest.get("trace_sha256") != sha256_file(cache_path):
        raise ValueError("assay template trace SHA256 drift")
    design_path = Path(manifest["source"]["design_path"])
    embedding_path = Path(manifest["source"]["embedding_path"])
    if manifest["source"]["design_sha256"] != sha256_file(design_path):
        raise ValueError("assay design SHA256 drift")
    if manifest["source"]["embedding_sha256"] != sha256_file(embedding_path):
        raise ValueError("assay observation embedding SHA256 drift")
    onsets = _supported_seizure_onsets(supported_seizure_path)
    with (
        np.load(cache_path, allow_pickle=False) as data,
        np.load(design_path, allow_pickle=False) as design,
    ):
        design_keys = {
            (int(session), float(time)): index
            for index, (session, time) in enumerate(zip(
                design["anchor_session"], design["anchor_time"],
            ))
        }
        trace_keys = list(zip(data["anchor_session"], data["anchor_time"]))
        if len(design_keys) != len(design["anchor_time"]):
            raise ValueError("assay design anchor key is not unique")
        try:
            row = np.asarray([
                design_keys[(int(session), float(time))]
                for session, time in trace_keys
            ], dtype=np.int64)
        except KeyError as error:
            raise ValueError("interictal trace/design anchor mismatch") from error
        observation = np.load(embedding_path, allow_pickle=False)
        template = build_template(
            time_epoch=np.asarray(data["anchor_time"], dtype=np.float64),
            segment=np.asarray(data["anchor_session"], dtype=np.int64),
            deterministic_history=np.asarray(
                design["anchor_history"], dtype=np.float64,
            )[row],
            current_observation=np.asarray(observation, dtype=np.float64)[row],
            persistent_decoder=np.asarray(
                data["persistent_decoder"], dtype=np.float64,
            ),
            memoryless_decoder=np.asarray(
                data["memoryless_decoder"], dtype=np.float64,
            ),
            n_seizures=len(onsets), observed_seizure_onsets=onsets,
        )
    return template, {
        "subject": subject, "seed": int(seed),
        "interictal_decoder_trace": str(cache_path),
        "interictal_decoder_trace_sha256": sha256_file(cache_path),
        "instrument_manifest": str(manifest_path),
        "instrument_manifest_sha256": sha256_file(manifest_path),
        "design": str(design_path), "design_sha256": sha256_file(design_path),
        "observation_embedding": str(embedding_path),
        "observation_embedding_sha256": sha256_file(embedding_path),
        "seizure_crosswalk": str(v02 / "manifests/seizure_crosswalk.csv"),
        "seizure_crosswalk_sha256": sha256_file(
            v02 / "manifests/seizure_crosswalk.csv"
        ),
        "supported_seizure_table": str(supported_seizure_path),
        "supported_seizure_table_sha256": sha256_file(supported_seizure_path),
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    transfer = result["transfer"]
    geometry = result["geometry"]
    return {
        "world": result["world"], "seed": result["seed"],
        "initial_k": result["initial_k"],
        "n_simulated_seizures": result["n_simulated_seizures"],
        "n_positive_grid_rows": result["n_positive_grid_rows"],
        "status": transfer["status"],
        "n_oof_seizures": transfer.get("n_oof_seizures", 0),
        "T_relative_logloss_improvement": transfer.get(
            "T_relative_logloss_improvement"
        ),
        "M_relative_logloss_improvement": transfer.get(
            "M_relative_logloss_improvement"
        ),
        "persistent_vs_memoryless_relative_improvement": transfer.get(
            "persistent_vs_memoryless_relative_improvement"
        ),
        "lag_degradation": transfer.get("lag_degradation"),
        "geometry_winner": geometry.get("winning_family"),
        "geometry_basin_score": geometry.get("scores", {}).get("basin_gating"),
        "geometry_approach_score": geometry.get("scores", {}).get(
            "directed_approach"
        ),
        "geometry_abrupt_score": geometry.get("scores", {}).get(
            "abrupt_transition"
        ),
    }


def _run_tasks(
    tasks: list[tuple[str, int, int, float]], template: AssayTemplate, workers: int,
) -> list[dict[str, Any]]:
    rows = []
    with ProcessPoolExecutor(
        max_workers=int(workers), initializer=_initialise_worker,
        initargs=(template,),
    ) as pool:
        future = {pool.submit(_worker, task): task for task in tasks}
        for item in as_completed(future):
            rows.append(_compact(item.result()))
    return sorted(rows, key=lambda row: (
        row["world"], int(row["initial_k"]), int(row["seed"]),
    ))


def _rate(rows: list[dict], key: str) -> dict[str, Any]:
    estimable = [row for row in rows if row["status"] == "COMPLETE"]
    success = sum(bool(row[key]) for row in estimable)
    lower, upper = wilson_interval(success, len(estimable))
    return {
        "successes": int(success), "total": len(estimable),
        "rate": float(success / len(estimable)) if estimable else None,
        "wilson_95_lower": lower, "wilson_95_upper": upper,
    }


def _threshold(rows: list[dict], key: str) -> float:
    values = np.asarray([
        row[key] for row in rows
        if row["status"] == "COMPLETE" and row.get(key) is not None
    ], dtype=np.float64)
    if not len(values):
        return float("inf")
    return float(np.quantile(values, 0.95, method="higher"))


def _detect(rows: list[dict], key: str, threshold: float) -> list[dict]:
    result = []
    for row in rows:
        value = row.get(key)
        copy = dict(row)
        copy[f"{key}_detected"] = bool(
            row["status"] == "COMPLETE" and value is not None
            and float(value) > float(threshold)
        )
        result.append(copy)
    return result


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--template-subject", default="epilepsiae_1125")
    parser.add_argument("--template-seed", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--effect-scale", type=float, default=1.75)
    args = parser.parse_args()
    v02, root = args.v0_2_root.resolve(), args.result_root.resolve()
    assert_frozen_contract_matches(_json(root / "analysis_contract.json"))
    assert_frozen_exploration_policy_matches(_json(root / "exploration_policy.json"))
    template, provenance = _load_template(
        v02, root, str(args.template_subject), int(args.template_seed),
    )
    available = _mem_available_bytes()
    memory_workers = max(1, int(0.70 * available // int(0.25 * 1024 ** 3)))
    workers = max(1, min(
        int(args.cpu_workers), memory_workers, max(1, (os.cpu_count() or 1) // 2),
    ))
    effect = float(args.effect_scale)
    n = int(args.replicates)
    k_tasks = [
        (world, 10_000 + seed, initial_k, effect)
        for initial_k in (2, 3, 4, 5)
        for world in ("null", "persistent_state") for seed in range(n)
    ]
    k_rows = _run_tasks(k_tasks, template, workers)
    k_summary = []
    for initial_k in (2, 3, 4, 5):
        null_rows = [row for row in k_rows
                     if row["initial_k"] == initial_k and row["world"] == "null"]
        persistent_rows = [row for row in k_rows if row["initial_k"] == initial_k
                           and row["world"] == "persistent_state"]
        null_rows = sorted(null_rows, key=lambda row: int(row["seed"]))
        split = max(1, len(null_rows) // 2)
        calibration, null_evaluation = null_rows[:split], null_rows[split:]
        t_threshold = _threshold(calibration, "T_relative_logloss_improvement")
        m_threshold = _threshold(calibration, "M_relative_logloss_improvement")
        lag_threshold = _threshold(calibration, "lag_degradation")
        null_evaluation = _detect(
            null_evaluation, "T_relative_logloss_improvement", t_threshold,
        )
        persistent_rows = _detect(
            persistent_rows, "T_relative_logloss_improvement", t_threshold,
        )
        false_positive = _rate(
            null_evaluation, "T_relative_logloss_improvement_detected",
        )
        power = _rate(
            persistent_rows, "T_relative_logloss_improvement_detected",
        )
        k_summary.append({
            "initial_k": initial_k, "null_false_positive": false_positive,
            "persistent_power": power,
            "null_calibration_replicates": len(calibration),
            "null_evaluation_replicates": len(null_evaluation),
            "thresholds": {
                "T_relative_logloss_improvement": t_threshold,
                "M_relative_logloss_improvement": m_threshold,
                "lag_degradation": lag_threshold,
            },
        })
    admissible = [row for row in k_summary
                  if (row["null_false_positive"]["rate"] or 1.0) <= 0.10]
    pool = admissible or k_summary
    selected = sorted(
        pool,
        key=lambda row: (
            -(row["persistent_power"]["rate"] or -1.0),
            row["null_false_positive"]["rate"] or 1.0,
            row["initial_k"],
        ),
    )[0]["initial_k"]
    selected_summary = next(row for row in k_summary
                            if row["initial_k"] == selected)
    thresholds = selected_summary["thresholds"]
    main_tasks = [
        (world, 100_000 + 10_000 * WORLDS.index(world) + seed,
         int(selected), effect)
        for world in WORLDS for seed in range(n)
    ]
    main_rows = _run_tasks(main_tasks, template, workers)
    for key, threshold in thresholds.items():
        main_rows = _detect(main_rows, key, float(threshold))
    world_summary = []
    for world in WORLDS:
        rows = [row for row in main_rows if row["world"] == world]
        transfer = _rate(rows, "T_relative_logloss_improvement_detected")
        memory = _rate(rows, "M_relative_logloss_improvement_detected")
        lag = _rate(rows, "lag_degradation_detected")
        expected_geometry = world if world in {
            "basin_gating", "directed_approach", "abrupt_transition",
        } else None
        geometry_success = sum(
            row["geometry_winner"] == expected_geometry for row in rows
        ) if expected_geometry else 0
        geometry_lower, geometry_upper = wilson_interval(geometry_success, len(rows))
        world_summary.append({
            "world": world, "transfer_detection": transfer,
            "persistent_memory_detection": memory,
            "lag_degradation_detection": lag,
            "geometry_expected_family": expected_geometry,
            "geometry_recovery": {
                "successes": geometry_success, "total": len(rows),
                "rate": geometry_success / len(rows) if rows else None,
                "wilson_95_lower": geometry_lower,
                "wilson_95_upper": geometry_upper,
            },
        })
    by_world = {row["world"]: row for row in world_summary}
    nuisance_false_positive = max(
        float(by_world[world]["transfer_detection"]["rate"] or 0.0)
        for world in ("null", "observation_only", "clock_confounded")
    )
    persistent_power = float(
        by_world["persistent_state"]["transfer_detection"]["rate"] or 0.0
    )
    minimum_geometry = min(
        float(by_world[world]["geometry_recovery"]["rate"] or 0.0)
        for world in ("basin_gating", "directed_approach", "abrupt_transition")
    )
    smoke_track_status = {
        "transfer": (
            "SMOKE_SENSITIVE" if nuisance_false_positive <= 0.10
            and persistent_power >= 0.70 else "ASSAY_NOT_SENSITIVE"
        ),
        "geometry": (
            "SMOKE_SENSITIVE" if minimum_geometry >= 0.75
            else "ASSAY_NOT_SENSITIVE"
        ),
        "maximum_null_observation_clock_false_positive": nuisance_false_positive,
        "persistent_world_power": persistent_power,
        "minimum_geometry_family_recovery": minimum_geometry,
        "not_final_acceptance": True,
    }
    output = root / "assay"
    created = utc_now()
    frozen_config = {
        "status": (
            "FROZEN_AFTER_100_REPLICATE_SMOKE" if n == 100
            else "DEVELOPMENT_DRY_RUN_NOT_FROZEN"
        ),
        "created_utc": created, "template": provenance,
        "n_template_rows": len(template.time_epoch),
        "n_template_segments": int(len(np.unique(template.segment))),
        "n_template_seizures": int(template.n_seizures),
        "selected_initial_k": int(selected), "initial_k_candidates": [2, 3, 4, 5],
        "initial_k_selection_rule": (
            "among null false-positive <=0.10 choose greatest persistent power, "
            "then lowest false-positive, then smallest K"
        ),
        "effect_scale": effect, "alpha_grid": list(ALPHA_GRID),
        "selected_null_calibrated_thresholds": thresholds,
        "minimum_relevant_relative_logloss_improvement": 0.05,
        "horizon_minutes": float(template.horizon_minutes),
        "worlds": list(WORLDS), "smoke_replicates_per_world": n,
        "final_replicates_per_world": 1000,
        "empirical_interictal_coverage_clock_and_state_autocorrelation_preserved": True,
        "real_supported_seizure_count_and_clock_distribution_preserved": True,
        "additional_control_subsampling": False,
        "producer_script_sha256": sha256_file(PRODUCER_SCRIPT),
        "assay_module_sha256": sha256_file(ASSAY_MODULE),
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    summary = {
        "status": (
            "COMPLETE_SMOKE_NOT_FINAL_ASSAY_ACCEPTANCE" if n == 100
            else "COMPLETE_DEVELOPMENT_DRY_RUN"
        ),
        "created_utc": created, "selected_initial_k": int(selected),
        "k_selection": k_summary, "worlds": world_summary,
        "smoke_track_status": smoke_track_status,
        "n_k_selection_replicates": len(k_rows),
        "n_main_replicates": len(main_rows), "cpu_workers": workers,
        "mem_available_bytes_at_start": available,
        "per_worker_memory_budget_bytes": int(0.25 * 1024 ** 3),
        "interpretation": (
            "100-replicate implementation and sensitivity smoke only; type-I/power "
            "acceptance requires the frozen 1000-replicate run"
        ),
        "negative_result_is_not_global_blocker": True,
        "real_seizure_probe_outcome_fitted": False,
        "real_seizure_structure_used_for_simulation": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(output / "frozen_assay_config.json", frozen_config)
    atomic_json(output / "type1_power_summary_smoke.json", summary)
    _atomic_jsonl(output / "replicate_manifest_smoke.jsonl", k_rows + main_rows)
    atomic_csv(output / "mechanism_recovery_smoke.csv", main_rows)
    atomic_json(root / "reports/scientific_route_audit_A2.json", {
        "status": "COMPLETE_SMOKE" if n == 100 else "DEVELOPMENT_DRY_RUN",
        "created_utc": created,
        "core_question": (
            "can the planned instrument recover cross-task state and distinct "
            "basin/approach/abrupt worlds on real coverage and state autocorrelation"
        ),
        "answer_source": str(output / "type1_power_summary_smoke.json"),
        "route_drift": False, "real_seizure_probe_outcome_fitted": False,
        "real_seizure_structure_used_for_simulation": True,
        "negative_result_is_not_global_blocker": True,
    })
    print(f"COMPLETE selected_K={selected} rows={len(k_rows) + len(main_rows)} workers={workers}")


if __name__ == "__main__":
    main()
