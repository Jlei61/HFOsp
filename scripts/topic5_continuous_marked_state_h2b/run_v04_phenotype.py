#!/usr/bin/env python3
"""Frozen-state transfer to a pre-existing continuous ictal phenotype target."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from run_v03_continuous_phenotype import (  # noqa: E402
    TARGET_SOURCE,
    TARGETS,
    _decode,
    _mapped_targets,
    _nearest_lead_row,
)
from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    V0_4_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.phenotype_transfer import (  # noqa: E402
    run_phenotype_table,
)
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    load_frozen_r16_checkpoint,
)


PRODUCER = Path(__file__).resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split(count: int, index: int) -> tuple[str, str]:
    if count >= 10:
        train_stop = max(2, int(math.floor(0.60 * count)))
        select_stop = max(train_stop + 1, int(math.floor(0.80 * count)))
        select_stop = min(select_stop, count - 1)
        split = "TRAIN" if index < train_stop else "SELECT" if index < select_stop else "TEST"
        return split, "primary_chronological"
    if count >= 5:
        return "TRAIN", "sensitivity_loso"
    if count >= 2:
        return "TRAIN", "descriptive_case_series"
    return "TRAIN", "not_estimable"


def run(*, v02_root: Path, result_root: Path, target_source: Path,
        lead_minutes: float = 30.0) -> dict:
    contract = _json(result_root / "analysis_contract.json")
    if contract.get("schema_revision") != "h2b_v0_4_heterogeneous_seizure_entry_routes_v10":
        raise ValueError("v0.4 phenotype requires the final v10 contract")
    inventory_path = result_root / "manifests/source_cells.json"
    inventory = _json(inventory_path)
    cells = inventory["cells"]
    if len(cells) != 46:
        raise ValueError("phenotype requires all 46 exact audited source cells")
    if not target_source.is_file():
        raise FileNotFoundError(target_source)
    subjects = {str(cell["subject"]) for cell in cells}
    targets, attrition = _mapped_targets(
        v02=v02_root, target_source=target_source, subjects=subjects,
    )
    target_hash = sha256_file(target_source)
    table_rows: list[dict] = []
    cell_rows: list[dict] = []
    for cell in cells:
        subject, seed = str(cell["subject"]), int(cell["seed"])
        cache_path = Path(cell["state_cache"])
        manifest_path = Path(cell["state_manifest"])
        if sha256_file(cache_path) != cell["state_cache_sha256"]:
            raise ValueError(f"phenotype cache SHA256 drift: {cache_path}")
        if sha256_file(manifest_path) != cell["state_manifest_sha256"]:
            raise ValueError(f"phenotype manifest SHA256 drift: {manifest_path}")
        manifest = _json(manifest_path)
        if manifest.get("all_parameters_frozen") is not True:
            raise ValueError(f"phenotype source is not frozen: {subject}/seed_{seed}")
        model, _ = load_frozen_r16_checkpoint(
            cell["checkpoint"], expected_sha256=cell["checkpoint_sha256"],
            expected_subject=subject, expected_seed=seed,
            require_stable_result=False, require_complete_result=True, device="cpu",
        )
        with np.load(cache_path, allow_pickle=False) as data:
            available = np.asarray(data["observation_available"], dtype=bool)
            time = np.asarray(data["anchor_time_epoch"], dtype=np.float64)[available]
            segment = np.asarray(
                data["coverage_segment_index"], dtype=np.int64,
            )[available]
            explicit = np.asarray(
                data["current_explicit_summary"], dtype=np.float64,
            )[available]
            persistent = np.asarray(data["persistent_state"], dtype=np.float32)[available]
            memoryless = np.asarray(
                data["memoryless_observation_code"], dtype=np.float32,
            )[available]
        persistent_decoder = _decode(model, persistent)
        memoryless_decoder = _decode(model, memoryless)
        mapped: list[tuple[dict, int]] = []
        for target in targets.get(subject, []):
            row = _nearest_lead_row(
                time, segment, onset=target["onset_time"],
                label=target["segment_id"], lead_minutes=float(lead_minutes),
            )
            if row is not None:
                mapped.append((target, int(row)))
        for index, (target, row) in enumerate(mapped):
            split, tier = _split(len(mapped), index)
            previous = mapped[index - 1][0]["onset_time"] if index else None
            feature = {
                "baseline__tod_sin": math.sin(
                    2.0 * math.pi * target["onset_time"] / 86400.0
                ),
                "baseline__tod_cos": math.cos(
                    2.0 * math.pi * target["onset_time"] / 86400.0
                ),
                "baseline__seizure_rank_fraction": float(
                    index / max(len(mapped) - 1, 1)
                ),
                "baseline__log_since_previous_seizure": (
                    math.log1p((target["onset_time"] - previous) / 60.0)
                    if previous is not None else 0.0
                ),
            }
            for column, value in enumerate(explicit[row]):
                feature[f"observation__explicit_{column}"] = float(value)
            for column, value in enumerate(memoryless_decoder[row]):
                feature[f"observation__memoryless_decoder_{column}"] = float(value)
            for column, value in enumerate(
                persistent_decoder[row] - memoryless_decoder[row]
            ):
                feature[f"state__persistent_minus_memoryless_decoder_{column}"] = float(value)
            common = {
                "patient_id": subject, "seed": seed,
                "seizure_id": target["seizure_id"], "split": split,
                "evaluation_tier": tier, "target_kind": "continuous",
                "target_frozen": True,
                "target_provenance": "preexisting_R3_parent_anchor_event_v1",
                "target_source_sha256": target_hash,
                **feature,
            }
            for name, column in TARGETS.items():
                table_rows.append({
                    **common, "target_name": name,
                    "target_value": float(target[column]),
                })
        cell_rows.append({
            "subject": subject, "seed": seed,
            "n_frozen_targets_in_development_prefix": len(targets.get(subject, [])),
            "n_targets_with_30min_full_grid_anchor": len(mapped),
            "state_cache": str(cache_path),
            "state_cache_sha256": cell["state_cache_sha256"],
        })
    table = pd.DataFrame(table_rows)
    feature_columns = [
        column for column in table.columns
        if column.startswith(("baseline__", "observation__", "state__"))
    ]
    if not feature_columns or not np.isfinite(
        table[feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("phenotype generated features are missing or non-finite")
    schema_padding = int(table[feature_columns].isna().sum().sum())
    table.loc[:, feature_columns] = table[feature_columns].fillna(0.0)
    result = run_phenotype_table(
        table, regularization_grid=(1.0, 10.0, 100.0),
    )
    output = result_root / "phenotype_continuous"
    atomic_csv(output / "probe_input_table.csv", table.to_dict(orient="records"))
    atomic_csv(output / "per_seed_metrics.csv", result.per_seed.to_dict(orient="records"))
    atomic_csv(
        output / "patient_median_metrics.csv",
        result.patient_medians.to_dict(orient="records"),
    )
    atomic_csv(output / "cell_availability.csv", cell_rows)
    atomic_csv(output / "target_attrition.csv", attrition)
    patient_rows = result.patient_medians.to_dict(orient="records")
    estimable = [
        row for row in patient_rows
        if row.get("state_minus_observation_loss") is not None
        and np.isfinite(float(row["state_minus_observation_loss"]))
    ]
    payload = {
        "status": result.audit.get("status"),
        "revision": "h2b_v0_4_frozen_continuous_phenotype_v1",
        "created_utc": utc_now(), "lead_minutes": float(lead_minutes),
        "target_source": str(target_source), "target_source_sha256": target_hash,
        "target_reclustered": False, "target_thresholded_after_state": False,
        "target_index_mapping": "chronological_development_prefix_only",
        "n_source_cells": len(cells), "n_source_subjects": len(subjects),
        "n_probe_rows": len(table), "n_attrition_rows": len(attrition),
        "n_estimable_patient_target_rows": len(estimable),
        "n_favourable_state_minus_observation": int(sum(
            float(row["state_minus_observation_loss"]) < 0.0 for row in estimable
        )),
        "median_state_minus_observation_loss": (
            float(np.median([
                float(row["state_minus_observation_loss"]) for row in estimable
            ])) if estimable else None
        ),
        "route_specific_phenotype_status": "NOT_ESTIMABLE_TARGET_SUPPORT_TOO_SPARSE",
        "phenotype_audit": result.audit,
        "patient_rows": patient_rows, "cell_availability": cell_rows,
        "cross_patient_schema_padding_zero_only": True,
        "n_cross_patient_schema_padding_cells": schema_padding,
        "patient_is_inference_unit": True, "seed_is_not_patient_replicate": True,
        "negative_result_biological_interpretation_allowed": False,
        "development_only": True, "formal_test_partition_opened": False,
        "sealed_opened": False, "h3_or_t2_run": False,
        "source": {
            "inventory": str(inventory_path),
            "inventory_sha256": sha256_file(inventory_path),
            "producer_sha256": sha256_file(PRODUCER),
        },
    }
    atomic_json(output / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    parser.add_argument("--target-source", type=Path, default=TARGET_SOURCE)
    parser.add_argument("--lead-minutes", type=float, default=30.0)
    args = parser.parse_args()
    payload = run(
        v02_root=args.v0_2_root.resolve(), result_root=args.result_root.resolve(),
        target_source=args.target_source.resolve(), lead_minutes=args.lead_minutes,
    )
    print(payload["status"], payload["n_probe_rows"])


if __name__ == "__main__":
    main()
