#!/usr/bin/env python3
"""Explore frozen interictal-state transfer to a frozen continuous ictal target.

The target is the pre-existing per-seizure R3 IED--ictal field-concordance
score (and its frozen null margin).  No target is derived, thresholded or
reclustered after reading state.  Only seizures in the development-prefix
tables are mapped; later source indices are retained as explicit attrition.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

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
from src.topic5_continuous_marked_state_h2b.phenotype_transfer import (  # noqa: E402
    run_phenotype_table,
)
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    load_frozen_r16_checkpoint,
)
from src.topic5_continuous_marked_state_h2b.v03_instrument import (  # noqa: E402
    decoder_output,
)

TARGET_SOURCE = Path("/home/honglab/leijiaxin/HFOsp") / (
    "results/topic5_ictal_recruitment/field_concordance_grid_method_sensitivity/"
    "n161_frozen_per_model/parent_anchor_event.csv"
)
TARGETS = {
    "ied_ictal_reuse_observed": "r3_observed",
    "ied_ictal_reuse_margin": "r3_margin",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _decode(model: torch.nn.Module, state: np.ndarray) -> np.ndarray:
    output = []
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(state), 1024):
            value = torch.as_tensor(
                np.asarray(state[start:start + 1024], dtype=np.float32),
                dtype=dtype, device="cpu",
            )
            output.append(decoder_output(model, value).cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float64, copy=False)


def _split(n: int, index: int) -> tuple[str, str]:
    if n >= 8:
        train_stop = max(3, int(math.floor(0.60 * n)))
        select_stop = max(train_stop + 1, int(math.floor(0.80 * n)))
        select_stop = min(select_stop, n - 1)
        split = "TRAIN" if index < train_stop else "SELECT" if index < select_stop else "TEST"
        return split, "primary_chronological"
    if n >= 4:
        return "TRAIN", "sensitivity_loso"
    if n >= 2:
        return "TRAIN", "descriptive_case_series"
    return "TRAIN", "not_estimable"


def _nearest_lead_row(
    time: np.ndarray, segment: np.ndarray, *, onset: float, label: int,
    lead_minutes: float,
) -> int | None:
    target = float(onset) - float(lead_minutes) * 60.0
    rows = np.flatnonzero(segment == int(label))
    if not len(rows):
        return None
    local = int(rows[np.argmin(np.abs(time[rows] - target))])
    if abs(float(time[local]) - target) > 300.0 + 1e-9:
        return None
    if float(time[local]) > float(onset):
        raise ValueError("phenotype feature anchor is after seizure onset")
    return local


def _mapped_targets(
    *, v02: Path, target_source: Path, subjects: set[str],
) -> tuple[dict[str, list[dict]], list[dict]]:
    source = pd.read_csv(target_source)
    source["r3_margin"] = source["r3_observed"] - source["r3_null_median"]
    mapped: dict[str, list[dict]] = {}
    attrition: list[dict] = []
    for subject in sorted(subjects):
        seizures_path = v02 / "risk_sets" / subject / "seizures.csv"
        if not seizures_path.is_file():
            continue
        seizures = pd.read_csv(seizures_path).sort_values(
            "onset_time", kind="mergesort",
        ).reset_index(drop=True)
        rows = source[source["subject"].astype(str) == subject].sort_values(
            "seizure_idx", kind="mergesort",
        )
        selected = []
        for row in rows.to_dict(orient="records"):
            index = int(row["seizure_idx"])
            if index >= len(seizures):
                attrition.append({
                    "subject": subject, "seizure_idx": index,
                    "reason": "outside_development_seizure_prefix",
                })
                continue
            seizure = seizures.iloc[index]
            try:
                label = int(seizure["segment_id"])
            except (TypeError, ValueError):
                attrition.append({
                    "subject": subject, "seizure_idx": index,
                    "reason": "development_seizure_has_no_coverage_segment",
                })
                continue
            selected.append({
                "subject": subject, "seizure_idx": index,
                "seizure_id": str(seizure["seizure_id"]),
                "onset_time": float(seizure["onset_time"]),
                "segment_id": label,
                "r3_observed": float(row["r3_observed"]),
                "r3_margin": float(row["r3_margin"]),
            })
        mapped[subject] = selected
    return mapped, attrition


def run(*, v02: Path, root: Path, target_source: Path,
        lead_minutes: float = 30.0) -> dict:
    if not target_source.is_file():
        raise FileNotFoundError(target_source)
    caches = sorted((root / "full_grid/state_cache").glob(
        "*/seed_*/states.manifest.json"
    ))
    if not caches:
        raise FileNotFoundError("full-grid state caches are not complete")
    subjects = {path.parents[1].name for path in caches}
    targets, attrition = _mapped_targets(
        v02=v02, target_source=target_source, subjects=subjects,
    )
    table_rows: list[dict] = []
    cell_rows: list[dict] = []
    for manifest_path in caches:
        subject = manifest_path.parents[1].name
        seed = int(manifest_path.parent.name.replace("seed_", ""))
        manifest = _json(manifest_path)
        cache_path = manifest_path.parent / "states.npz"
        if (
            manifest.get("full_recorded_five_minute_grid") is not True
            or manifest.get("cache_sha256") != sha256_file(cache_path)
            or manifest.get("all_parameters_frozen") is not True
        ):
            raise ValueError(f"invalid full-grid cell {subject}/seed_{seed}")
        model, _ = load_frozen_r16_checkpoint(
            manifest["checkpoint"], expected_sha256=manifest["checkpoint_sha256"],
            expected_subject=subject, expected_seed=seed,
            require_stable_result=False, require_complete_result=True, device="cpu",
        )
        with np.load(cache_path, allow_pickle=False) as data:
            available = np.asarray(data["observation_available"], dtype=bool)
            time = np.asarray(data["anchor_time_epoch"], dtype=np.float64)[available]
            segment = np.asarray(data["coverage_segment_index"], dtype=np.int64)[available]
            explicit = np.asarray(data["current_explicit_summary"], dtype=np.float64)[available]
            persistent_state = np.asarray(data["persistent_state"], dtype=np.float32)[available]
            memoryless_state = np.asarray(
                data["memoryless_observation_code"], dtype=np.float32,
            )[available]
        persistent_decoder = _decode(model, persistent_state)
        memoryless_decoder = _decode(model, memoryless_state)
        mapped = []
        for target in targets.get(subject, []):
            row = _nearest_lead_row(
                time, segment, onset=target["onset_time"],
                label=target["segment_id"], lead_minutes=float(lead_minutes),
            )
            if row is None:
                continue
            mapped.append((target, row))
        for index, (target, row) in enumerate(mapped):
            split, tier = _split(len(mapped), index)
            previous_onset = mapped[index - 1][0]["onset_time"] if index else None
            baseline = {
                "baseline__tod_sin": math.sin(2.0 * math.pi * target["onset_time"] / 86400.0),
                "baseline__tod_cos": math.cos(2.0 * math.pi * target["onset_time"] / 86400.0),
                "baseline__seizure_rank_fraction": float(index / max(len(mapped) - 1, 1)),
                "baseline__log_since_previous_seizure": (
                    math.log1p((target["onset_time"] - previous_onset) / 60.0)
                    if previous_onset is not None else 0.0
                ),
            }
            feature = dict(baseline)
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
                "target_source_sha256": sha256_file(target_source),
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
            "state_cache": str(cache_path), "state_cache_sha256": sha256_file(cache_path),
        })
    table = pd.DataFrame(table_rows)
    result = run_phenotype_table(table)
    output = root / "phenotype_continuous"
    atomic_csv(output / "probe_input_table.csv", table.to_dict(orient="records"))
    atomic_csv(output / "per_seed_metrics.csv", result.per_seed.to_dict(orient="records"))
    atomic_csv(
        output / "patient_median_metrics.csv",
        result.patient_medians.to_dict(orient="records"),
    )
    atomic_csv(output / "cell_availability.csv", cell_rows)
    atomic_csv(output / "target_attrition.csv", attrition)
    payload = {
        "status": result.audit.get("status"),
        "revision": "h2b_v0_3_continuous_phenotype_v1",
        "created_utc": utc_now(), "lead_minutes": float(lead_minutes),
        "target_source": str(target_source),
        "target_source_sha256": sha256_file(target_source),
        "target_reclustered": False, "target_thresholded_after_state": False,
        "target_index_mapping": "chronological_prefix_only; later indices excluded",
        "n_full_grid_cells": len(caches), "n_subjects": len(subjects),
        "n_probe_rows": len(table), "n_attrition_rows": len(attrition),
        "phenotype_audit": result.audit,
        "patient_rows": result.patient_medians.to_dict(orient="records"),
        "cell_availability": cell_rows,
        "patient_is_inference_unit": True, "seed_is_not_patient_replicate": True,
        "negative_result_biological_interpretation_allowed": False,
        "claim_status": "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE",
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--target-source", type=Path, default=TARGET_SOURCE)
    parser.add_argument("--lead-minutes", type=float, default=30.0)
    args = parser.parse_args()
    payload = run(
        v02=args.v0_2_root.resolve(), root=args.result_root.resolve(),
        target_source=args.target_source.resolve(), lead_minutes=args.lead_minutes,
    )
    print(payload["status"], payload["n_full_grid_cells"], payload["n_probe_rows"])


if __name__ == "__main__":
    main()
