#!/usr/bin/env python3
"""Calibrate the v0.4 heterogeneous-route estimator on real coverage."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    V0_4_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.v03_hazard import (  # noqa: E402
    HazardDesign,
    build_hazard_design,
)
from src.topic5_continuous_marked_state_h2b.v04_assay import (  # noqa: E402
    apply_synthetic_postictal_exclusion,
    inject_slow_state,
    sample_synthetic_onsets,
    zscore,
)
from src.topic5_continuous_marked_state_h2b.v04_heterogeneous import (  # noqa: E402
    circular_shift_state_within_segment,
    prequential_heterogeneous_hazard,
)


PRODUCER = Path(__file__).resolve()
ASSAY_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v04_assay.py"
ESTIMATOR_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v04_heterogeneous.py"
WORLDS = (
    "null", "observation_only", "persistent_single_route",
    "persistent_two_route", "clock_confounded",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapped_seizures(path: Path) -> int:
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                int(row["segment_id"])
                float(row["onset_time"])
            except (TypeError, ValueError):
                continue
            count += 1
    return count


def _pc1(values: np.ndarray) -> np.ndarray:
    matrix = zscore(np.asarray(values, dtype=np.float64))
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    if not len(vt):
        return np.zeros(len(matrix), dtype=np.float64)
    loading = np.array(vt[0], copy=True)
    pivot = int(np.argmax(np.abs(loading)))
    if loading[pivot] < 0:
        loading *= -1.0
    return zscore((matrix @ loading).reshape(-1, 1))[:, 0]


def _load_templates(
    result_root: Path,
    v02_root: Path,
    *,
    minimum_mapped_seizures: int = 10,
) -> list[dict[str, Any]]:
    inventory = _json(result_root / "manifests/source_cells.json")
    by_subject: dict[str, list[dict]] = {}
    for cell in inventory["cells"]:
        by_subject.setdefault(cell["subject"], []).append(cell)
    templates = []
    for subject, cells in sorted(by_subject.items()):
        seizure_path = v02_root / "risk_sets" / subject / "seizures.csv"
        mapped = _mapped_seizures(seizure_path)
        if mapped < int(minimum_mapped_seizures):
            continue
        cell = min(cells, key=lambda row: int(row["seed"]))
        cache_path = Path(cell["state_cache"])
        if sha256_file(cache_path) != cell["state_cache_sha256"]:
            raise ValueError(f"assay template cache SHA256 drift: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            design = build_hazard_design(
                time_epoch=data["anchor_time_epoch"],
                segment=data["coverage_segment_index"],
                history=data["deterministic_history"],
                current_observation=data["current_explicit_summary"],
                persistent_state=data["persistent_state"],
                memoryless_state=data["memoryless_observation_code"],
                observation_available=data["observation_available"],
                onset_time=[], onset_segment=[], spacing_seconds=300.0,
            )
        injected, slow = inject_slow_state(design, amplitude=2.5)
        design = HazardDesign(**{**design.__dict__, "persistent_state": injected})
        templates.append({
            "subject": subject,
            "seed": int(cell["seed"]),
            "mapped_seizures": mapped,
            "design": design,
            "slow": slow,
            "observation_axis": _pc1(design.current_observation),
            "clock_axis": np.sin(
                2.0 * np.pi * ((design.time_epoch % 86400.0) / 86400.0)
            ),
            "state_cache": str(cache_path),
            "state_cache_sha256": cell["state_cache_sha256"],
        })
    if len(templates) < 2:
        raise ValueError("fewer than two support-rich assay templates")
    return templates


def _one_replicate(
    template: dict[str, Any],
    *,
    world: str,
    phase: str,
    replicate: int,
    random_seed: int,
) -> dict[str, Any]:
    design: HazardDesign = template["design"]
    slow = np.asarray(template["slow"], dtype=np.float64)
    rng = np.random.default_rng(int(random_seed))
    if world == "null":
        score = np.zeros(len(design.time_epoch), dtype=np.float64)
        balance = None
        strength = 0.0
    elif world == "observation_only":
        score = template["observation_axis"]
        balance = None
        strength = 3.0
    elif world == "persistent_single_route":
        score = slow[:, 0]
        balance = None
        strength = 6.0
    elif world == "persistent_two_route":
        score = np.abs(slow[:, 0])
        balance = slow[:, 0] >= 0.0
        strength = 6.0
    elif world == "clock_confounded":
        score = template["clock_axis"]
        balance = None
        strength = 3.0
    else:
        raise ValueError(f"unknown assay world: {world}")
    try:
        onset, onset_group, _ = sample_synthetic_onsets(
            design, score, rng=rng, n_seizures=10,
            horizon_minutes=30.0, minimum_separation_minutes=180.0,
            balance=balance, strength=strength,
        )
        synthetic, take = apply_synthetic_postictal_exclusion(
            design, onset, onset_group, postictal_minutes=120.0,
        )
        wrong = circular_shift_state_within_segment(
            synthetic, synthetic.persistent_state, 0.5,
        )
        result = prequential_heterogeneous_hazard(
            synthetic, horizon_minutes=30.0,
            wrong_time_state=wrong,
        )
    except ValueError as error:
        return {
            "phase": phase, "world": world, "replicate": int(replicate),
            "random_seed": int(random_seed), "template_subject": template["subject"],
            "template_seed": template["seed"], "status": "NOT_ESTIMABLE",
            "reason": str(error),
        }
    row = {
        "phase": phase, "world": world, "replicate": int(replicate),
        "random_seed": int(random_seed), "template_subject": template["subject"],
        "template_seed": template["seed"], "status": result["status"],
        "n_rows_after_synthetic_exclusion": int(len(take)),
        "n_supported_seizures": int(result.get("n_supported_seizures", 0)),
        "n_oof_seizures": int(result.get("n_oof_seizures", 0)),
        "n_two_route_folds": int(result.get("n_two_route_folds", 0)),
        "reason": "",
    }
    for name, value in result.get("equal_seizure_weight_effects", {}).items():
        row[name] = float(value) if value is not None else None
    return row


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def run_assay(
    *,
    result_root: Path,
    v02_root: Path,
    workers: int,
    calibration_replicates: int,
    evaluation_replicates: int,
) -> dict:
    if _json(result_root / "analysis_contract.json") != _json(
        REPO / "config/topic5_continuous_marked_state_h2b_v0_4.json"
    ):
        raise ValueError("v0.4 frozen analysis contract drift")
    templates = _load_templates(result_root, v02_root)
    tasks = []
    for replicate in range(int(calibration_replicates)):
        tasks.append((
            templates[replicate % len(templates)], "null", "calibration",
            replicate, 100000 + replicate,
        ))
    for world_index, world in enumerate(WORLDS):
        for replicate in range(int(evaluation_replicates)):
            tasks.append((
                templates[replicate % len(templates)], world, "evaluation",
                replicate, 200000 + world_index * 1000 + replicate,
            ))
    started = time.monotonic()
    rows = []
    # Each replicate spends most of its time in Python-level conditional-risk
    # objectives.  A thread pool cannot use the requested cores because of the
    # GIL; independent processes provide real worker concurrency while BLAS is
    # separately pinned to one thread by the launcher.
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(
            _one_replicate, template,
            world=world, phase=phase, replicate=replicate, random_seed=seed,
        ) for template, world, phase, replicate, seed in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (
        row["phase"], row["world"], row["replicate"], row["template_subject"],
    ))
    metrics = (
        "route_state_minus_observation",
        "route_state_minus_memoryless",
        "two_route_minus_single_axis_state",
        "correct_minus_wrong_time",
    )
    calibration = [row for row in rows
                   if row["phase"] == "calibration" and row["status"] == "COMPLETE_DEVELOPMENT"]
    if len(calibration) != int(calibration_replicates):
        raise ValueError(
            f"calibration incomplete: {len(calibration)}/{calibration_replicates}"
        )
    calibration_values = {
        metric: [row[metric] for row in calibration if row.get(metric) is not None]
        for metric in metrics
    }
    thresholds = {
        metric: (
            float(np.quantile(values, 0.05, method="lower")) if values else None
        )
        for metric, values in calibration_values.items()
    }
    summaries = {}
    evaluation_values = {
        (world, metric): np.asarray([
            row[metric] for row in rows
            if row["phase"] == "evaluation" and row["world"] == world
            and row["status"] == "COMPLETE_DEVELOPMENT"
            and row.get(metric) is not None
        ], dtype=np.float64)
        for world in WORLDS for metric in metrics
    }
    for world in WORLDS:
        selected = [row for row in rows if row["phase"] == "evaluation"
                    and row["world"] == world
                    and row["status"] == "COMPLETE_DEVELOPMENT"]
        world_summary = {
            "n_complete": len(selected),
            "n_requested": int(evaluation_replicates),
        }
        for metric in metrics:
            values = evaluation_values[(world, metric)]
            null_values = evaluation_values[("null", metric)]
            threshold = thresholds[metric]
            successes = int(np.sum(values < threshold)) if threshold is not None else 0
            low, high = _wilson(successes, len(values))
            favourable = int(np.sum(values < 0.0))
            world_summary[metric] = {
                "median": float(np.median(values)) if len(values) else None,
                "threshold_from_independent_null_calibration": threshold,
                "n_calibration_estimable": len(calibration_values[metric]),
                "n_evaluation_estimable": len(values),
                "n_below_threshold": successes,
                "rate_below_threshold": float(successes / len(values)) if len(values) else None,
                "wilson_95": [low, high],
                "n_favourable_below_zero": favourable,
                "rate_favourable_below_zero": (
                    float(favourable / len(values)) if len(values) else None
                ),
                "median_minus_evaluation_null_median": (
                    float(np.median(values) - np.median(null_values))
                    if len(values) and len(null_values) else None
                ),
                "pairwise_probability_below_evaluation_null": (
                    float(np.mean(values[:, None] < null_values[None, :]))
                    if len(values) and len(null_values) else None
                ),
            }
        summaries[world] = world_summary
    primary = "route_state_minus_observation"
    memory = "route_state_minus_memoryless"
    heterogeneity = "two_route_minus_single_axis_state"
    def rate(world: str, metric: str) -> float:
        value = summaries[world][metric]["rate_below_threshold"]
        return float(value) if value is not None else float("nan")

    def directional_rate(world: str, metric: str) -> float:
        value = summaries[world][metric]["rate_favourable_below_zero"]
        return float(value) if value is not None else float("nan")

    calibration_checks = {
        "null_primary_type1_le_0_10": (
            rate("null", primary) <= 0.10
        ),
        "observation_only_primary_false_increment_le_0_10": (
            rate("observation_only", primary) <= 0.10
        ),
        "clock_confounded_primary_false_increment_le_0_10": (
            rate("clock_confounded", primary) <= 0.10
        ),
        "null_primary_directional_rate_le_0_65": (
            directional_rate("null", primary) <= 0.65
        ),
        "observation_only_primary_directional_rate_le_0_65": (
            directional_rate("observation_only", primary) <= 0.65
        ),
        "clock_confounded_primary_directional_rate_le_0_65": (
            directional_rate("clock_confounded", primary) <= 0.65
        ),
    }
    strict_power_checks = {
        "single_route_primary_power_ge_0_80": (
            rate("persistent_single_route", primary) >= 0.80
        ),
        "single_route_memory_power_ge_0_80": (
            rate("persistent_single_route", memory) >= 0.80
        ),
        "two_route_primary_power_ge_0_80": (
            rate("persistent_two_route", primary) >= 0.80
        ),
        "two_route_memory_power_ge_0_80": (
            rate("persistent_two_route", memory) >= 0.80
        ),
        "two_route_heterogeneity_power_ge_0_80": (
            rate("persistent_two_route", heterogeneity) >= 0.80
        ),
    }
    directional_checks = {
        "single_route_primary_direction_ge_0_70": (
            directional_rate("persistent_single_route", primary) >= 0.70
        ),
        "single_route_memory_direction_ge_0_70": (
            directional_rate("persistent_single_route", memory) >= 0.70
        ),
        "single_route_time_specificity_direction_ge_0_70": (
            directional_rate("persistent_single_route", "correct_minus_wrong_time") >= 0.70
        ),
        "two_route_primary_direction_ge_0_70": (
            directional_rate("persistent_two_route", primary) >= 0.70
        ),
        "two_route_memory_direction_ge_0_70": (
            directional_rate("persistent_two_route", memory) >= 0.70
        ),
        "two_route_heterogeneity_direction_ge_0_70": (
            directional_rate("persistent_two_route", heterogeneity) >= 0.70
        ),
        "two_route_time_specificity_direction_ge_0_70": (
            directional_rate("persistent_two_route", "correct_minus_wrong_time") >= 0.70
        ),
    }
    if all(calibration_checks.values()) and all(strict_power_checks.values()):
        status = "PASS_ASSAY_HIGH_SINGLE_REPLICATE_POWER"
    elif all(calibration_checks.values()) and all(directional_checks.values()):
        status = "PASS_DIRECTIONAL_ASSAY_LOW_SINGLE_REPLICATE_POWER"
    else:
        status = "ASSAY_NOT_DIRECTIONALLY_SENSITIVE_NO_BIOLOGICAL_NEGATIVE"
    checks = {
        **calibration_checks,
        **directional_checks,
        **strict_power_checks,
    }
    csv_path = result_root / "assay/replicates.csv"
    atomic_csv(csv_path, rows)
    payload = {
        "status": status,
        "revision": "h2b_v0_4_real_coverage_heterogeneous_assay_v5_directional_gate",
        "created_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "workers": int(workers),
        "worker_backend": "process_pool_one_blas_thread_per_worker",
        "calibration_replicates": int(calibration_replicates),
        "evaluation_replicates_per_world": int(evaluation_replicates),
        "worlds": list(WORLDS),
        "thresholds": thresholds,
        "summaries": summaries,
        "checks": checks,
        "calibration_checks": calibration_checks,
        "directional_recovery_checks": directional_checks,
        "strict_single_replicate_power_checks": strict_power_checks,
        "interpretation_boundary": (
            "directional pass permits interpretation of concordant positive development effects; "
            "unless strict power also passes, null or adverse real-data effects cannot be called "
            "a biological negative"
        ),
        "assay_controls_interpretation_not_execution": True,
        "templates": [{key: value for key, value in template.items()
                       if key not in {"design", "slow", "observation_axis", "clock_axis"}}
                      for template in templates],
        "template_selection": (
            "all audited subjects with at least 10 mapped development seizures; "
            "lowest available seed; selection ignores H2b effects"
        ),
        "synthetic_state": (
            "one deterministic slow coordinate with two distant route ends injected "
            "into the existing fixed-width state; all other dimensions are structural "
            "zeros; reset at real coverage gaps; no seizure label enters the trajectory"
        ),
        "synthetic_postictal_exclusion_minutes": 120,
        "synthetic_minimum_interseizure_separation_minutes": 180,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "source": {
            "replicates": str(csv_path),
            "replicates_sha256": sha256_file(csv_path),
            "producer_sha256": sha256_file(PRODUCER),
            "assay_module_sha256": sha256_file(ASSAY_MODULE),
            "estimator_module_sha256": sha256_file(ESTIMATOR_MODULE),
        },
    }
    atomic_json(result_root / "assay/summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration-replicates", type=int, default=100)
    parser.add_argument("--evaluation-replicates", type=int, default=100)
    args = parser.parse_args()
    result = run_assay(
        result_root=args.result_root.resolve(),
        v02_root=args.v0_2_root.resolve(),
        workers=int(args.workers),
        calibration_replicates=int(args.calibration_replicates),
        evaluation_replicates=int(args.evaluation_replicates),
    )
    print(result["status"], result["elapsed_seconds"])


if __name__ == "__main__":
    main()
