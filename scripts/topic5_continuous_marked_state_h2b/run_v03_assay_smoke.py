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


def _count_supported_seizures(path: Path) -> int:
    values = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("primary_30min_supported", "")).lower() not in {
                "true", "1",
            }:
                continue
            values.add(str(row["seizure_id"]))
    return len(values)


def _load_template(v02: Path, subject: str, seed: int) -> tuple[AssayTemplate, dict]:
    root = v02 / "state_cache" / subject / f"seed_{seed}"
    manifest_path = root / "states.manifest.json"
    cache_path = root / "states.npz"
    supported_seizure_path = v02 / "risk_sets" / subject / "seizures.csv"
    manifest = _json(manifest_path)
    if manifest.get("all_parameters_frozen") is not True:
        raise ValueError("assay template state was not frozen")
    if manifest.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("assay template state cache SHA256 drift")
    with np.load(cache_path, allow_pickle=False) as data:
        available = np.asarray(data["observation_available"], dtype=bool)
        template = build_template(
            time_epoch=np.asarray(data["anchor_time_epoch"], dtype=np.float64)[available],
            segment=np.asarray(data["coverage_segment_index"], dtype=np.int64)[available],
            deterministic_history=np.asarray(
                data["deterministic_history"], dtype=np.float64,
            )[available],
            persistent_state=np.asarray(data["persistent_state"], dtype=np.float64)[available],
            memoryless_state=np.asarray(
                data["memoryless_observation_code"], dtype=np.float64,
            )[available],
            n_seizures=_count_supported_seizures(supported_seizure_path),
        )
    return template, {
        "subject": subject, "seed": int(seed),
        "state_cache": str(cache_path), "state_cache_sha256": sha256_file(cache_path),
        "state_manifest": str(manifest_path),
        "state_manifest_sha256": sha256_file(manifest_path),
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
        "relative_logloss_improvement": transfer.get(
            "relative_logloss_improvement"
        ),
        "T_detected_at_5_percent": transfer.get("T_detected_at_5_percent", False),
        "M_detected_at_5_percent": transfer.get("M_detected_at_5_percent", False),
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
        v02, str(args.template_subject), int(args.template_seed),
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
        false_positive = _rate(null_rows, "T_detected_at_5_percent")
        power = _rate(persistent_rows, "T_detected_at_5_percent")
        k_summary.append({
            "initial_k": initial_k, "null_false_positive": false_positive,
            "persistent_power": power,
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
    main_tasks = [
        (world, 100_000 + 10_000 * WORLDS.index(world) + seed,
         int(selected), effect)
        for world in WORLDS for seed in range(n)
    ]
    main_rows = _run_tasks(main_tasks, template, workers)
    world_summary = []
    for world in WORLDS:
        rows = [row for row in main_rows if row["world"] == world]
        transfer = _rate(rows, "T_detected_at_5_percent")
        expected_geometry = world if world in {
            "basin_gating", "directed_approach", "abrupt_transition",
        } else None
        geometry_success = sum(
            row["geometry_winner"] == expected_geometry for row in rows
        ) if expected_geometry else 0
        geometry_lower, geometry_upper = wilson_interval(geometry_success, len(rows))
        world_summary.append({
            "world": world, "transfer_detection": transfer,
            "geometry_expected_family": expected_geometry,
            "geometry_recovery": {
                "successes": geometry_success, "total": len(rows),
                "rate": geometry_success / len(rows) if rows else None,
                "wilson_95_lower": geometry_lower,
                "wilson_95_upper": geometry_upper,
            },
        })
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
        "minimum_relevant_relative_logloss_improvement": 0.05,
        "horizon_minutes": float(template.horizon_minutes),
        "worlds": list(WORLDS), "smoke_replicates_per_world": n,
        "final_replicates_per_world": 1000,
        "empirical_query_and_control_sampling_preserved": True,
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
        "n_k_selection_replicates": len(k_rows),
        "n_main_replicates": len(main_rows), "cpu_workers": workers,
        "mem_available_bytes_at_start": available,
        "per_worker_memory_budget_bytes": int(0.25 * 1024 ** 3),
        "interpretation": (
            "100-replicate implementation and sensitivity smoke only; type-I/power "
            "acceptance requires the frozen 1000-replicate run"
        ),
        "negative_result_is_not_global_blocker": True,
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
        "route_drift": False, "real_seizure_outcome_fitted": False,
        "negative_result_is_not_global_blocker": True,
    })
    print(f"COMPLETE selected_K={selected} rows={len(k_rows) + len(main_rows)} workers={workers}")


if __name__ == "__main__":
    main()
