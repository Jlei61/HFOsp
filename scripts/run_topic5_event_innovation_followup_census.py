#!/usr/bin/env python3
"""Descriptive follow-up census for the frozen Topic 5 V3.0 event-innovation line.

Three read-only questions, none of which can change the frozen V3.0 evidence
level:

* **coverage** — what separates the 17 innovation-valid patients from the 9
  short-history and 8 unresolved-residual ones;
* **cross-recording structure** — does any patient have two well-populated
  continuity units separated by a real silence, i.e. is a future
  across-recording contract even feasible;
* **detectability floor** — with the observed patient-to-patient scatter and
  n=17, how large a true median effect would the frozen cohort rule have caught.

The frozen human-test runner, helpers, config and acceptance script are never
imported for fitting and never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_event_innovation_followup import (  # noqa: E402
    detectability_floor,
    source_gap_census,
    source_spans,
)

DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
DEFAULT_OUTPUT = ROOT / "results/topic5_event_innovation_followup"
HUMAN_ROOT = (
    ROOT / "results/topic5_event_innovation_impulse_response/v3_0/human_exploratory"
)

TRAIN80_SPLIT_CODE = 0
OLD_HELDOUT20_SPLIT_CODE = 1

ROUTE_METRICS = {
    "local": (
        "local_propagation_gain",
        "local_true_minus_matched",
        "local_future_minus_past",
    ),
    "cumulative": (
        "cumulative_propagation_gain",
        "cumulative_true_minus_matched",
        "cumulative_alignment",
    ),
}
PRIMARY_METRIC = {
    "local": "local_propagation_gain",
    "cumulative": "cumulative_propagation_gain",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_patient_streams(subject: str, config: dict) -> dict:
    """Read one patient's frozen event stream and its source mapping."""

    dataset = np.load(
        ROOT / str(config["dataset_root"]) / "per_subject" / f"{subject}.npz",
        allow_pickle=True,
    )
    mapping_path = ROOT / str(config["source_mapping_root"]) / f"{subject}.npz"
    record_name = None
    if mapping_path.exists():
        mapping = np.load(mapping_path, allow_pickle=True)
        if "event_source_record_name" in mapping.files:
            record_name = np.asarray(mapping["event_source_record_name"])
    split = np.asarray(dataset["event_split"])
    analysis = split == TRAIN80_SPLIT_CODE
    # `event_source_index` is a per-event row pointer, not a segment label, so the
    # recording name from the mapping artifact is the only usable grouping key.
    if record_name is None:
        raise RuntimeError(
            f"{subject}: no event_source_record_name; segment structure is unknowable"
        )
    return {
        "abs_time": np.asarray(dataset["event_abs_time"], dtype=float),
        "source_index": record_name,
        "record_name": record_name,
        "analysis_mask": analysis,
        "n_events_total": int(split.size),
        "n_old_heldout20": int(np.sum(split == OLD_HELDOUT20_SPLIT_CODE)),
        "n_contacts": int(np.asarray(dataset["contact_names"]).size),
    }


def census_one_patient(
    subject: str,
    config: dict,
    status: str,
    gap_thresholds_seconds: dict[str, float],
    min_events_per_side: int,
) -> tuple[dict, dict, list[dict]]:
    streams = load_patient_streams(subject, config)
    mask = streams["analysis_mask"]
    times = streams["abs_time"][mask]
    sources = streams["source_index"][mask]
    records = (
        streams["record_name"][mask] if streams["record_name"] is not None else None
    )
    spans = source_spans(times, sources, records)

    duration = float(times.max() - times.min()) if times.size else 0.0
    coverage = {
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "innovation_status": status,
        "innovation_valid": status == "INNOVATION_VALID",
        "n_events_analysis": int(mask.sum()),
        "n_events_total": streams["n_events_total"],
        "n_old_heldout20": streams["n_old_heldout20"],
        "n_contacts": streams["n_contacts"],
        "n_sources": len(spans),
        "median_events_per_source": float(
            np.median([row["n_events"] for row in spans])
        )
        if spans
        else None,
        "max_events_per_source": max((row["n_events"] for row in spans), default=None),
        "recorded_span_hours": duration / 3600.0,
        "events_per_hour": (mask.sum() / (duration / 3600.0)) if duration > 0 else None,
    }

    cross = {"subject": subject, "dataset": coverage["dataset"], "innovation_status": status}
    for label, seconds in gap_thresholds_seconds.items():
        summary = source_gap_census(
            spans, min_gap_seconds=seconds, min_events_per_side=min_events_per_side
        )
        if "n_sources" not in cross:
            cross.update(
                {
                    "n_sources": summary["n_sources"],
                    "n_events_total": summary["n_events_total"],
                    "median_events_per_source": summary["median_events_per_source"],
                    "max_events_per_source": summary["max_events_per_source"],
                    "n_consecutive_gaps": summary["n_consecutive_gaps"],
                    "max_gap_hours": (
                        summary["max_gap_seconds"] / 3600.0
                        if summary["max_gap_seconds"] is not None
                        else None
                    ),
                    "median_gap_hours": (
                        summary["median_gap_seconds"] / 3600.0
                        if summary["median_gap_seconds"] is not None
                        else None
                    ),
                    "total_span_hours": (
                        summary["total_span_seconds"] / 3600.0
                        if summary["total_span_seconds"] is not None
                        else None
                    ),
                }
            )
        cross[f"n_qualifying_gaps_{label}"] = summary["n_qualifying_consecutive_gaps"]
        cross[f"eligible_{label}"] = summary["cross_gap_eligible"]
    return coverage, cross, spans


def build_floor(table: pd.DataFrame, n_draws: int, seed: int, k_grid) -> dict:
    """Detectability floor per route, expressed in both raw and scatter units."""

    floors: dict[str, dict] = {}
    for route, metrics in ROUTE_METRICS.items():
        eligible = table[table[f"{route}_eligible"]]
        route_block: dict[str, dict] = {}
        for metric in metrics:
            effects = eligible[metric].to_numpy(dtype=float)
            residual_sd = float(np.std(effects - np.median(effects), ddof=1))
            deltas = [float(k) * residual_sd for k in k_grid]
            plain = detectability_floor(
                effects,
                deltas=deltas,
                n_draws=n_draws,
                seed=seed,
                alpha=0.05,
                smooth=False,
            )
            smooth = detectability_floor(
                effects,
                deltas=deltas,
                n_draws=n_draws,
                seed=seed + 1,
                alpha=0.05,
                smooth=True,
            )
            for record in (plain, smooth):
                for row, k in zip(record["curve"], k_grid):
                    row["delta_in_residual_sd"] = float(k)
                record["delta80_in_residual_sd"] = (
                    record["delta80"] / residual_sd
                    if record["delta80"] is not None and residual_sd > 0
                    else None
                )
                record["observed_median_in_residual_sd"] = (
                    record["observed_median"] / residual_sd if residual_sd > 0 else None
                )
            route_block[metric] = {
                "is_primary": metric == PRIMARY_METRIC[route],
                "bootstrap": plain,
                "kernel_smoothed": smooth,
            }
        floors[route] = route_block
    return floors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-events-per-side", type=int, default=100)
    parser.add_argument("--floor-draws", type=int, default=4000)
    parser.add_argument("--floor-seed", type=int, default=20260804)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    innovation_path = (
        ROOT / str(config["innovation_output_root"]) / "innovation_validity.json"
    )
    innovation = json.loads(innovation_path.read_text(encoding="utf-8"))
    statuses = {row["subject"]: row["status"] for row in innovation["patients"]}

    summary_path = HUMAN_ROOT / "patient_summary.csv"
    human = pd.read_csv(summary_path)

    gap_thresholds = {"1h": 3600.0, "6h": 21600.0, "12h": 43200.0, "24h": 86400.0}

    coverage_rows, cross_rows, span_rows = [], [], []
    for subject in sorted(statuses):
        coverage, cross, spans = census_one_patient(
            subject,
            config,
            statuses[subject],
            gap_thresholds,
            args.min_events_per_side,
        )
        coverage_rows.append(coverage)
        cross_rows.append(cross)
        for span in spans:
            span_rows.append({"subject": subject, **span})

    coverage_table = pd.DataFrame(coverage_rows)
    cross_table = pd.DataFrame(cross_rows)
    span_table = pd.DataFrame(span_rows)

    merged = coverage_table.merge(
        human[
            [
                "subject",
                "local_eligible",
                "cumulative_eligible",
                "local_test_anchors",
                "cumulative_test_anchors",
                *ROUTE_METRICS["local"],
                *ROUTE_METRICS["cumulative"],
            ]
        ],
        on="subject",
        how="left",
    )

    k_grid = [round(0.1 * step, 2) for step in range(0, 21)]
    floors = build_floor(merged, args.floor_draws, args.floor_seed, k_grid)

    by_status = {
        status: {
            "n": int((coverage_table.innovation_status == status).sum()),
            "median_events_analysis": float(
                coverage_table.loc[
                    coverage_table.innovation_status == status, "n_events_analysis"
                ].median()
            ),
            "median_max_events_per_source": float(
                coverage_table.loc[
                    coverage_table.innovation_status == status, "max_events_per_source"
                ].median()
            ),
            "median_recorded_span_hours": float(
                coverage_table.loc[
                    coverage_table.innovation_status == status, "recorded_span_hours"
                ].median()
            ),
            "median_n_contacts": float(
                coverage_table.loc[
                    coverage_table.innovation_status == status, "n_contacts"
                ].median()
            ),
            "median_n_sources": float(
                coverage_table.loc[
                    coverage_table.innovation_status == status, "n_sources"
                ].median()
            ),
        }
        for status in sorted(coverage_table.innovation_status.unique())
    }

    feasibility = {
        label: {
            "n_eligible_patients": int(cross_table[f"eligible_{label}"].sum()),
            "n_eligible_innovation_valid": int(
                (
                    cross_table[f"eligible_{label}"]
                    & (cross_table.innovation_status == "INNOVATION_VALID")
                ).sum()
            ),
            "by_dataset": {
                dataset: int(
                    cross_table.loc[
                        cross_table.dataset == dataset, f"eligible_{label}"
                    ].sum()
                )
                for dataset in sorted(cross_table.dataset.unique())
            },
        }
        for label in gap_thresholds
    }

    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "coverage_census.csv", index=False)
    cross_table.to_csv(args.output / "cross_recording_census.csv", index=False)
    span_table.to_csv(args.output / "source_spans.csv", index=False)
    atomic_json(args.output / "detectability_floor.json", jsonable(floors))
    state = {
        "contract": "topic5_event_innovation_followup_census",
        "status": "FOLLOWUP_CENSUS_COMPLETE",
        "descriptive_only_cannot_change_frozen_evidence_level": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "n_patients": int(len(coverage_table)),
        "min_events_per_side": int(args.min_events_per_side),
        "gap_thresholds_seconds": gap_thresholds,
        "coverage_by_innovation_status": by_status,
        "cross_recording_feasibility": feasibility,
        "detectability_floor_delta_grid_in_residual_sd": k_grid,
        "inputs_sha256": {
            "config": sha256(args.config),
            "innovation_validity": sha256(innovation_path),
            "patient_summary": sha256(summary_path),
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    atomic_json(args.output / "CENSUS_STATE.json", jsonable(state))
    print(
        json.dumps(
            {
                "status": state["status"],
                "coverage_by_innovation_status": by_status,
                "cross_recording_feasibility": feasibility,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
