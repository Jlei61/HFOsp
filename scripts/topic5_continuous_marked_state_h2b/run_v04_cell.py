#!/usr/bin/env python3
"""Run one audited v0.4 heterogeneous-route development cell."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    V0_4_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    load_frozen_r16_checkpoint,
)
from src.topic5_continuous_marked_state_h2b.v03_geometry import (  # noqa: E402
    evaluate_oos_geometry_fold_full_grid,
)
from src.topic5_continuous_marked_state_h2b.v03_hazard import (  # noqa: E402
    build_hazard_design,
)
from src.topic5_continuous_marked_state_h2b.v03_instrument import (  # noqa: E402
    decoder_output,
)
from src.topic5_continuous_marked_state_h2b.v04_heterogeneous import (  # noqa: E402
    circular_shift_state_within_segment,
    evaluate_oos_route_geometry_fold,
    prequential_heterogeneous_hazard,
)


PRODUCER = Path(__file__).resolve()
MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v04_heterogeneous.py"
CONTRACT_SOURCE = REPO / "config/topic5_continuous_marked_state_h2b_v0_4.json"
LEADS = (5, 15, 30, 60, 120)
SHIFT_FRACTIONS = tuple(value / 10.0 for value in range(1, 10))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _onsets(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    times, segments = [], []
    total = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total += 1
            try:
                segment = int(row["segment_id"])
                onset = float(row["onset_time"])
            except (TypeError, ValueError):
                continue
            times.append(onset)
            segments.append(segment)
    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(segments, dtype=np.int64)[order],
        total,
    )


def _wrong_time(
    data: Any, source_index: np.ndarray, memoryless: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    values = np.asarray(data["wrong_time_donor_state"], dtype=np.float64)[source_index]
    valid = np.asarray(data["wrong_time_valid"], dtype=bool)[source_index]
    result = np.array(memoryless, copy=True)
    valid_row = np.zeros(len(source_index), dtype=bool)
    count = 0
    for row in range(len(source_index)):
        donors = np.flatnonzero(valid[row])
        if len(donors):
            chosen = values[row, donors[0]]
            if not np.isfinite(chosen).all():
                raise ValueError("valid wrong-time donor contains non-finite state")
            result[row] = chosen
            valid_row[row] = True
            count += 1
    return result, count, valid_row


def _decode(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
    values = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(state), 1024):
            tensor = torch.as_tensor(
                np.asarray(state[start:start + 1024], dtype=np.float32),
                dtype=dtype,
                device="cpu",
            )
            values.append(decoder_output(model, tensor).cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float64, copy=False)


def _source_cell(result_root: Path, subject: str, seed: int) -> dict:
    inventory = _json(result_root / "manifests/source_cells.json")
    rows = [row for row in inventory["cells"]
            if row["subject"] == subject and int(row["seed"]) == int(seed)]
    if len(rows) != 1:
        raise ValueError(f"source cell is not unique: {subject}/seed_{seed}")
    return rows[0]


def _state_stratum(v03_root: Path, subject: str) -> dict:
    path = v03_root / "qualification/instrument_smoke_summary.json"
    if not path.is_file():
        return {"subject": subject, "exploration_stratum": "A1_UNAVAILABLE"}
    rows = [row for row in _json(path)["patient_rows"] if row["subject"] == subject]
    return rows[0] if len(rows) == 1 else {
        "subject": subject, "exploration_stratum": "A1_NOT_UNIQUE",
    }


def run(
    subject: str,
    seed: int,
    *,
    v02_root: Path,
    v03_root: Path,
    result_root: Path,
) -> dict:
    if _json(result_root / "analysis_contract.json") != _json(CONTRACT_SOURCE):
        raise ValueError("v0.4 frozen analysis contract drift")
    source = _source_cell(result_root, subject, seed)
    for path_key, hash_key in (
        ("state_manifest", "state_manifest_sha256"),
        ("state_cache", "state_cache_sha256"),
        ("instrument_manifest", "instrument_manifest_sha256"),
        ("checkpoint", "checkpoint_sha256"),
    ):
        if sha256_file(source[path_key]) != source[hash_key]:
            raise ValueError(f"source SHA256 drift: {source[path_key]}")
    state_manifest = _json(Path(source["state_manifest"]))
    instrument = _json(Path(source["instrument_manifest"]))
    checkpoint_receipt = instrument["source"]["checkpoint"]
    if checkpoint_receipt.get("state_frozen") is not True:
        raise ValueError("instrument checkpoint is not frozen")
    if instrument.get("seizure_risk_outcome_read") is not False:
        raise ValueError("instrument qualification read seizure risk outcome")
    model, checkpoint_provenance = load_frozen_r16_checkpoint(
        source["checkpoint"],
        expected_sha256=source["checkpoint_sha256"],
        expected_subject=subject,
        expected_seed=int(seed),
        require_stable_result=False,
        require_complete_result=True,
        device="cpu",
    )
    torch.set_num_threads(1)

    seizure_path = v02_root / "risk_sets" / subject / "seizures.csv"
    onset_time, onset_segment, total_seizures = _onsets(seizure_path)
    with np.load(source["state_cache"], allow_pickle=False) as data:
        design = build_hazard_design(
            time_epoch=data["anchor_time_epoch"],
            segment=data["coverage_segment_index"],
            history=data["deterministic_history"],
            current_observation=data["current_explicit_summary"],
            persistent_state=data["persistent_state"],
            memoryless_state=data["memoryless_observation_code"],
            observation_available=data["observation_available"],
            onset_time=onset_time,
            onset_segment=onset_segment,
            spacing_seconds=300.0,
        )
        wrong_state, wrong_count, wrong_valid = _wrong_time(
            data, design.source_index, design.memoryless_state,
        )
    by_lead = {
        str(lead): prequential_heterogeneous_hazard(
            design,
            horizon_minutes=float(lead),
            wrong_time_state=wrong_state,
            wrong_time_valid=wrong_valid,
        )
        for lead in LEADS
    }
    shift_null = []
    for fraction in SHIFT_FRACTIONS:
        shifted = circular_shift_state_within_segment(
            design, design.persistent_state, fraction,
        )
        result = prequential_heterogeneous_hazard(
            design,
            horizon_minutes=30.0,
            persistent_override=shifted,
            wrong_time_state=wrong_state,
            wrong_time_valid=wrong_valid,
        )
        shift_null.append({"fraction": fraction, "result": result})

    available = np.ones(len(design.time_epoch), dtype=bool)
    decoder = _decode(model, design.persistent_state[available])
    geometry: dict[str, dict[str, list[dict]]] = {}
    for lookback in (15.0, 30.0, 60.0):
        route_rows, single_rows = [], []
        for position in range(2, len(onset_time)):
            route_rows.append(evaluate_oos_route_geometry_fold(
                grid_time=design.time_epoch,
                grid_segment=design.segment,
                grid_decoder=decoder,
                onset_time=onset_time,
                onset_segment=onset_segment,
                heldout_position=position,
                lookback_minutes=lookback,
                grid_spacing_seconds=300.0,
            ))
            single_rows.append(evaluate_oos_geometry_fold_full_grid(
                grid_time=design.time_epoch,
                grid_segment=design.segment,
                grid_decoder=decoder,
                onset_time=onset_time,
                onset_segment=onset_segment,
                heldout_position=position,
                lookback_minutes=lookback,
                grid_spacing_seconds=300.0,
            ))
        geometry[f"{lookback:g}"] = {
            "heterogeneous_route": route_rows,
            "single_route": single_rows,
        }

    primary = by_lead["30"]
    payload = {
        "status": (
            "COMPLETE_DEVELOPMENT" if primary["status"] == "COMPLETE_DEVELOPMENT"
            else "NOT_ESTIMABLE_PRIMARY_LEAD"
        ),
        "revision": "h2b_v0_4_heterogeneous_route_cell_v6_direct_history_contrast_conditional_risk_sets",
        "created_utc": utc_now(),
        "subject": subject,
        "seed": int(seed),
        "n_grid_rows": len(design.time_epoch),
        "n_mapped_seizures": len(onset_time),
        "n_inventory_seizures": total_seizures,
        "primary_lead_minutes": 30,
        "by_lead_minutes": by_lead,
        "primary": primary,
        "coverage_segment_circular_shift_null": shift_null,
        "geometry_by_lookback_minutes": geometry,
        "state_stratum_nonblocking": _state_stratum(v03_root, subject),
        "matched_wrong_time": {
            "n_rows_with_donor": wrong_count,
            "donor_fraction": float(wrong_count / max(len(design.time_epoch), 1)),
            "invalid_donor_policy": "exclude fold from correct-time versus wrong-time contrast only",
        },
        "source": {
            **source,
            "checkpoint_provenance": checkpoint_provenance,
            "state_manifest_revision": state_manifest.get("state_extraction_revision"),
            "seizure_table": str(seizure_path),
            "seizure_table_sha256": sha256_file(seizure_path),
            "analysis_contract": str(result_root / "analysis_contract.json"),
            "analysis_contract_sha256": sha256_file(result_root / "analysis_contract.json"),
            "producer_sha256": sha256_file(PRODUCER),
            "heterogeneous_module_sha256": sha256_file(MODULE),
        },
        "route_policy": {
            "maximum_routes": 2,
            "minimum_prior_seizures_for_two_routes": 4,
            "minimum_seizures_per_route": 2,
            "minimum_route_separation_bandwidth": 1.0,
            "heldout_seizure_defines_route": False,
            "single_route_direct_comparator": True,
            "risk_set_controls_per_seizure": 5,
            "primary_metric": "conditional_risk_set_log_loss",
            "support_rich_initial_training_fraction": 0.60,
        },
        "state_model_updated": False,
        "seizure_gradient_enters_state": False,
        "real_seizure_outcome_read": True,
        "development_only": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "physical_clock_run": False,
        "paper_ready_figures_modified": False,
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
    }
    output = result_root / "per_cell" / subject / f"seed_{seed}" / "result.json"
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--v0-3-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    args = parser.parse_args()
    result = run(
        str(args.subject), int(args.seed),
        v02_root=args.v0_2_root.resolve(),
        v03_root=args.v0_3_root.resolve(),
        result_root=args.result_root.resolve(),
    )
    print(result["status"], result["subject"], result["seed"])


if __name__ == "__main__":
    main()
