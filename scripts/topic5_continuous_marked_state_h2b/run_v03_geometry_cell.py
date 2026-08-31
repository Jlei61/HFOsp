#!/usr/bin/env python3
"""Run one frozen-cell OOS geometry analysis for H2b v0.3 A6."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    load_frozen_r16_checkpoint,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    assert_frozen_contract_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (  # noqa: E402
    assert_frozen_exploration_policy_matches,
)
from src.topic5_continuous_marked_state_h2b.v03_geometry import (  # noqa: E402
    evaluate_oos_geometry_fold_full_grid,
)
from src.topic5_continuous_marked_state_h2b.v03_instrument import (  # noqa: E402
    decoder_output,
)


PRODUCER = Path(__file__).resolve()
GEOMETRY_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_geometry.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _onsets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times, groups = [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                group = int(row["segment_id"])
            except (TypeError, ValueError):
                continue
            times.append(float(row["onset_time"]))
            groups.append(group)
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(groups, dtype=np.int64)[order],
    )


def _decode(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
    values = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(state), 1024):
            tensor = torch.as_tensor(
                np.asarray(state[start:start + 1024], dtype=np.float32),
                dtype=dtype, device="cpu",
            )
            values.append(decoder_output(model, tensor).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float64, copy=False)


def _a1_patient(root: Path, subject: str) -> dict:
    summary = _json(root / "qualification/instrument_smoke_summary.json")
    rows = [row for row in summary["patient_rows"] if row["subject"] == subject]
    return rows[0] if len(rows) == 1 else {
        "subject": subject, "state_qualified": False,
        "exploration_stratum": "A1_NOT_AVAILABLE",
    }


def run(subject: str, seed: int, *, v02_root: Path, result_root: Path,
        allow_diagnostic_exploration: bool = False) -> dict:
    assert_frozen_contract_matches(_json(result_root / "analysis_contract.json"))
    assert_frozen_exploration_policy_matches(_json(result_root / "exploration_policy.json"))
    a1 = _a1_patient(result_root, subject)
    if a1.get("state_qualified") is not True and not allow_diagnostic_exploration:
        raise ValueError(f"{subject}: A6 not released because A1 failed")
    assay_path = result_root / "assay" / (
        "type1_power_summary_smoke.json" if allow_diagnostic_exploration
        else "type1_power_summary.json"
    )
    if not assay_path.is_file():
        raise ValueError("A6 selected assay receipt is missing")
    assay = _json(assay_path)
    if not allow_diagnostic_exploration and (
        assay.get("status") != "PASS_FINAL_ASSAY_ACCEPTANCE"
        or assay.get("claim_bearing_route_released") is not True
    ):
        raise ValueError("A6 not released because A2 assay did not pass")
    initial_k = int(assay["selected_initial_k"])

    instrument_root = result_root / "instrument/by_cell" / subject / f"seed_{seed}"
    instrument_path = instrument_root / "instrument_manifest.json"
    instrument = _json(instrument_path)
    checkpoint = instrument["source"]["checkpoint"]
    if (
        instrument.get("status") != "COMPLETE"
        or instrument.get("seizure_risk_outcome_read") is not False
        or checkpoint.get("state_frozen") is not True
    ):
        raise ValueError("A6 source is not a complete frozen interictal instrument")
    model, provenance = load_frozen_r16_checkpoint(
        checkpoint["checkpoint"],
        expected_sha256=checkpoint["checkpoint_sha256"],
        expected_subject=subject, expected_seed=int(seed),
        require_stable_result=False, require_complete_result=True, device="cpu",
    )
    cache_root = result_root / "full_grid/state_cache" / subject / f"seed_{seed}"
    cache_path = cache_root / "states.npz"
    cache_manifest_path = cache_root / "states.manifest.json"
    cache_manifest = _json(cache_manifest_path)
    if cache_manifest.get("cache_sha256") != sha256_file(cache_path):
        raise ValueError("risk-query state cache SHA256 drift")
    if cache_manifest.get("all_parameters_frozen") is not True:
        raise ValueError("risk-query state cache is not frozen")
    if cache_manifest.get("full_recorded_five_minute_grid") is not True:
        raise ValueError(
            "A6 requires a full recorded-coverage trajectory; v0.2 seizure-"
            "support query caches are not admissible"
        )
    with np.load(cache_path, allow_pickle=False) as cache:
        available = np.asarray(cache["observation_available"], dtype=bool)
        risk_time = np.asarray(cache["anchor_time_epoch"], dtype=np.float64)[available]
        risk_segment = np.asarray(
            cache["coverage_segment_index"], dtype=np.int64,
        )[available]
        risk_state = np.asarray(cache["persistent_state"], dtype=np.float32)[available]
    order = np.argsort(risk_time, kind="stable")
    risk_time, risk_segment, risk_state = (
        risk_time[order], risk_segment[order], risk_state[order]
    )
    risk_decoder = _decode(model, risk_state)
    seizure_path = v02_root / "risk_sets" / subject / "seizures.csv"
    onset_time, onset_segment = _onsets(seizure_path)
    by_lookback: dict[str, list[dict]] = {}
    for lookback in (15.0, 30.0, 60.0):
        by_lookback[f"{lookback:g}"] = [
            evaluate_oos_geometry_fold_full_grid(
                grid_time=risk_time, grid_segment=risk_segment,
                grid_decoder=risk_decoder, onset_time=onset_time,
                onset_segment=onset_segment, heldout_position=position,
                lookback_minutes=lookback, grid_spacing_seconds=300.0,
            )
            for position in range(initial_k, len(onset_time))
        ]
    primary = by_lookback["30"]
    complete = [row for row in primary if row["status"] == "COMPLETE_EXPLORATORY"]
    payload = {
        "status": "COMPLETE_EXPLORATORY" if complete else "NOT_ESTIMABLE",
        "revision": "h2b_v0_3_oos_geometry_cell_v3",
        "created_utc": utc_now(), "subject": subject, "seed": int(seed),
        "initial_k": initial_k, "primary_lookback_minutes": 30.0,
        "by_lookback_minutes": by_lookback,
        "n_primary_complete_folds": len(complete),
        "n_primary_attempted_folds": len(primary),
        "primary_not_estimable_reasons": {
            reason: sum(row.get("reason") == reason for row in primary)
            for reason in sorted({row.get("reason") for row in primary if row.get("reason")})
        },
        "A1_patient_stratum": a1,
        "A2_geometry_smoke_scope": (
            "family-statistic smoke only; not a calibration of this full OOS pipeline"
        ),
        "analysis_scope": "full_recorded_development_grid_exploratory",
        "control_source": "same frozen full grid strictly before outer cutoff",
        "matching": "patient-internal circular time-of-day nearest controls",
        "common_extraction_domain": True,
        "old_event_anchor_to_regular_grid_abrupt_scores_invalidated": True,
        "sleep_matching_available": False,
        "marble_status": "NOT_RUN_SIMPLE_OOS_GEOMETRY_FIRST",
        "umap_used_for_evidence": False,
        "source": {
            "instrument_manifest": str(instrument_path),
            "instrument_manifest_sha256": sha256_file(instrument_path),
            "checkpoint": provenance,
            "state_cache": str(cache_path),
            "state_cache_sha256": sha256_file(cache_path),
            "state_cache_manifest": str(cache_manifest_path),
            "state_cache_manifest_sha256": sha256_file(cache_manifest_path),
            "seizure_table": str(seizure_path),
            "seizure_table_sha256": sha256_file(seizure_path),
            "assay_summary": str(assay_path),
            "assay_summary_sha256": sha256_file(assay_path),
            "producer_sha256": sha256_file(PRODUCER),
            "geometry_module_sha256": sha256_file(GEOMETRY_MODULE),
        },
        "real_seizure_outcome_read": True, "development_only": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
        "claim_status": (
            "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_FULL_GRID"
            if allow_diagnostic_exploration else
            "CLAIM_ROUTE_RELEASED_DEVELOPMENT_ONLY"
        ),
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
    }
    output = result_root / "geometry/by_cell" / subject / f"seed_{seed}"
    atomic_json(output / "result.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--allow-diagnostic-exploration", action="store_true")
    args = parser.parse_args()
    result = run(
        str(args.subject), int(args.seed), v02_root=args.v0_2_root.resolve(),
        result_root=args.result_root.resolve(),
        allow_diagnostic_exploration=bool(args.allow_diagnostic_exploration),
    )
    print(result["status"], result["subject"], result["seed"],
          result["n_primary_complete_folds"])


if __name__ == "__main__":
    main()
