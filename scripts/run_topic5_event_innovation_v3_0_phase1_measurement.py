#!/usr/bin/env python3
"""Measure rank/precedence state reliability for Topic 5 v3.0.

This stage remains upstream of event-innovation testing.  It fits only a
train-only stable backbone and diagnostic low-rank bases, then measures whether
future rank/precedence windows have reliable dynamic variation after contact
main effects are removed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_topic5_event_innovation_v3_0_phase0 import (  # noqa: E402
    _load_subject,
    _sequences_for_splits,
    array_sha256,
    build_source_segments,
    chronological_split_indices,
    sha256,
)
from src.topic5_event_innovation_data import (  # noqa: E402
    ContinuitySequence,
    assign_continuity_units,
    build_single_event_anchor_splits,
    resolve_single_event_anchor,
)
from src.topic5_event_innovation_v3_0 import (  # noqa: E402
    fit_rank_state_basis,
    masked_rank_reconstruction_error,
    rank_field_windows,
    rolling_past_rank_fields,
    split_window_precedence_reliability,
    split_window_rank_reliability,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def train_contact_backbone(
    ranks: np.ndarray,
    participation: np.ndarray,
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rank = np.asarray(ranks, dtype=float)[train_indices]
    mask = np.asarray(participation, dtype=bool)[train_indices] & np.isfinite(rank)
    support = mask.sum(axis=0)
    total = np.where(mask, rank, 0.0).sum(axis=0)
    backbone = np.full(rank.shape[1], np.nan, dtype=float)
    np.divide(total, support, out=backbone, where=support > 0)
    if np.any(~np.isfinite(backbone)):
        raise ValueError("every contact requires train-event support")
    return backbone, support


def unit_balanced_dense_fields(
    ranks: np.ndarray,
    participation: np.ndarray,
    sequences: Sequence[ContinuitySequence],
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense past fields and weights with equal continuity-unit mass."""

    sequence_indices = [np.asarray(item.event_indices, dtype=np.int64) for item in sequences]
    fields, supports = rolling_past_rank_fields(
        ranks,
        participation,
        sequence_indices,
        start_offset=0,
        stop_offset=int(window),
    )
    selected_parts = []
    weight_parts = []
    for indices in sequence_indices:
        selected = indices[int(window) :]
        if not len(selected):
            continue
        selected_parts.append(selected)
        weight_parts.append(np.full(len(selected), 1.0 / len(selected), dtype=float))
    if not selected_parts:
        raise ValueError("no training continuity unit supports a dense state window")
    selected = np.concatenate(selected_parts)
    weight = np.concatenate(weight_parts)
    valid = np.any(supports[selected] > 0, axis=1)
    return fields[selected][valid], supports[selected][valid], weight[valid]


def formal_post_windows(anchors, sequences: Sequence[ContinuitySequence]) -> list[np.ndarray]:
    return [
        resolve_single_event_anchor(anchors, row, sequences)[2]
        for row in range(len(anchors))
    ]


def _dimension_candidates(config: Mapping[str, Any], n_contacts: int, n_rows: int) -> list[tuple[str, int]]:
    maximum = min(int(n_contacts), int(n_rows))
    output: list[tuple[str, int]] = []
    for raw in config["dimensions"]:
        dimension = maximum if str(raw) == "full" else int(raw)
        dimension = min(dimension, maximum)
        label = "full" if str(raw) == "full" else str(int(raw))
        if dimension >= 1 and all(existing != dimension for _, existing in output):
            output.append((label, dimension))
    if all(dimension != maximum for _, dimension in output):
        output.append(("full", maximum))
    return output


def _basis_diagnostic(
    train_fields: np.ndarray,
    train_support: np.ndarray,
    train_weight: np.ndarray,
    validation_fields: np.ndarray,
    validation_support: np.ndarray,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for label, dimension in _dimension_candidates(
        config, train_fields.shape[1], len(train_fields)
    ):
        basis = fit_rank_state_basis(
            train_fields, dimension, sample_weight=train_weight
        )
        validation_reconstruction = basis.inverse(basis.transform(validation_fields))
        validation_error = masked_rank_reconstruction_error(
            validation_fields, validation_reconstruction, validation_support
        )
        validation_backbone = np.broadcast_to(basis.backbone, validation_fields.shape)
        validation_backbone_error = masked_rank_reconstruction_error(
            validation_fields, validation_backbone, validation_support
        )
        rows.append(
            {
                "label": label,
                "dimension": dimension,
                "validation_mse": validation_error,
                "validation_variance_explained": 1.0 - validation_error / max(validation_backbone_error, 1e-12),
                "singular_values": basis.singular_values.tolist(),
            }
        )
    return rows


def _prepare(subject: str, config: Mapping[str, Any], phase0_root: Path):
    phase0_path = phase0_root / "per_subject" / f"{subject}.json"
    frozen = json.loads(phase0_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "PHASE0_PATIENT_PASS":
        raise RuntimeError(f"{subject}: Phase0 patient artifact is not accepted")
    raw = _load_subject(subject, config)
    train80 = np.flatnonzero(np.asarray(raw["event_split"]) == 0)
    segments, _ = build_source_segments(
        subject, np.asarray(raw["source_id"]), np.asarray(raw["record_name"]), config
    )
    continuity = config["continuity"]
    decisions = assign_continuity_units(
        segments,
        maximum_gap_seconds=float(continuity["maximum_gap_seconds"]),
        maximum_overlap_seconds=float(continuity["maximum_overlap_seconds"]),
    )
    minimum = (
        int(config["primary_pre_events"])
        + min(map(int, config["cumulative_events"]))
        + int(config["primary_horizon"])
    )
    split_indices = chronological_split_indices(
        train80, config["split_fractions"], minimum_events=minimum
    )
    hashes = {key: array_sha256(value) for key, value in split_indices.items()}
    if hashes != frozen["split_index_sha256"]:
        raise RuntimeError(f"{subject}: Phase1 split does not reproduce frozen Phase0")
    sequences = _sequences_for_splits(raw, decisions, split_indices)
    return raw, split_indices, sequences, phase0_path


def run_subject(subject: str, config: Mapping[str, Any], phase0_root: Path, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw, split_indices, sequences, phase0_path = _prepare(subject, config, phase0_root)
    horizon = int(config["primary_horizon"])
    pre = int(config["primary_pre_events"])
    anchors = build_single_event_anchor_splits(
        sequences, pre_events=pre, horizon=horizon
    )
    windows = {
        "validation": formal_post_windows(anchors.validation, sequences["validation"])
    }
    fields: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    for split in ("validation",):
        fields[split], supports[split] = rank_field_windows(
            raw["rank"], raw["participation"], windows[split]
        )
        if not len(fields[split]):
            raise RuntimeError(f"{subject}: no formal {split} rank fields")
    train_fields, train_support, train_weight = unit_balanced_dense_fields(
        raw["rank"], raw["participation"], sequences["train"], window=horizon
    )
    backbone, contact_support = train_contact_backbone(
        raw["rank"], raw["participation"], split_indices["train"]
    )
    reliability = {
        "validation": {
            "rank": asdict(
                split_window_rank_reliability(
                    raw["rank"],
                    raw["participation"],
                    windows["validation"],
                    contact_backbone=backbone,
                )
            ),
            "precedence": asdict(
                split_window_precedence_reliability(
                    raw["rank"],
                    raw["participation"],
                    windows["validation"],
                    contact_backbone=backbone,
                )
            ),
        }
    }
    dimension = _basis_diagnostic(
        train_fields,
        train_support,
        train_weight,
        fields["validation"],
        supports["validation"],
        config,
    )
    result = {
        "contract": str(config["contract"]),
        "status": "STATE_MEASUREMENT_COMPLETE_INNOVATION_PENDING",
        "subject": subject,
        "one_step_is_one_complete_event": True,
        "primary_pre_events": pre,
        "primary_horizon": horizon,
        "n_train_dense_fields": len(train_fields),
        "n_validation_formal_windows": len(windows["validation"]),
        "contact_train_support_min": int(np.min(contact_support)),
        "contact_train_support_median": float(np.median(contact_support)),
        "reliability": reliability,
        "dimension_diagnostic": dimension,
        "dimension_selection_status": "DEFERRED_TO_OBSERVER_VALIDATION",
        "phase0_path": str(phase0_path),
        "phase0_sha256": sha256(phase0_path),
        "old_heldout20_entered_into_analysis": False,
        "human_test_outcomes_read": False,
        "event_innovation_test_run": False,
        "within_event_next_rank_model_fit": False,
        "forbidden_inputs_read": False,
    }
    atomic_write_json(output / "per_subject" / f"{subject}.json", _jsonable(result))
    diagnostic_rows = [
        {"subject": subject, **{key: value for key, value in row.items() if key != "singular_values"}}
        for row in dimension
    ]
    return result, diagnostic_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_root = ROOT / str(config["output_root"])
    phase0_state = json.loads((phase0_root / "anchor_contract.json").read_text(encoding="utf-8"))
    if phase0_state.get("status") != "PHASE0_COMPLETE" or phase0_state.get("n_pass") != 34:
        raise SystemExit("full 34-patient Phase0 contract is not complete")
    output = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / (args.output_dir or Path(str(config["phase1_output_root"])))
    )
    cohort = list(map(str, phase0_state["subjects"]))
    subjects = cohort if not args.subjects else list(map(str, args.subjects))
    if sorted(set(subjects) - set(cohort)):
        raise SystemExit("requested subject is outside frozen Phase0")
    full_cohort = subjects == cohort
    results = []
    diagnostic_rows = []
    failures = []
    for subject in subjects:
        try:
            result, rows = run_subject(subject, config, phase0_root, output)
        except Exception as exc:
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(subject, "FAIL", exc, flush=True)
            continue
        results.append(result)
        diagnostic_rows.extend(rows)
        print(subject, "PASS", flush=True)
    if diagnostic_rows:
        _atomic_csv(output / "dimension_diagnostic.csv", pd.DataFrame(diagnostic_rows))
    if failures:
        _atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    state_status = (
        "STATE_MEASUREMENT_FAIL_CLOSED"
        if failures
        else "STATE_MEASUREMENT_COMPLETE_INNOVATION_PENDING"
        if full_cohort
        else "STATE_MEASUREMENT_PARTIAL_AUDIT"
    )
    state = {
        "contract": str(config["contract"]),
        "status": state_status,
        "cohort_scope": "full_34" if full_cohort else "explicit_partial_audit",
        "n_requested": len(subjects),
        "n_pass": len(results),
        "n_failed": len(failures),
        "patients": results,
        "failures": failures,
        "phase0_state_path": str(phase0_root / "anchor_contract.json"),
        "phase0_state_sha256": sha256(phase0_root / "anchor_contract.json"),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "math_module_sha256": sha256(ROOT / "src/topic5_event_innovation_v3_0.py"),
        "old_heldout20_entered_into_analysis": False,
        "human_test_outcomes_read": False,
        "event_innovation_test_run": False,
        "within_event_next_rank_model_fit": False,
        "forbidden_inputs_read": False,
    }
    atomic_write_json(output / "state_reliability.json", _jsonable(state))
    print(json.dumps({"status": state_status, "n_pass": len(results), "n_failed": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
