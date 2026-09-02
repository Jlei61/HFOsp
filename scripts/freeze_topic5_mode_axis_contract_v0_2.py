#!/usr/bin/env python3
"""Freeze future-field identifiability tiers before Topic 5.2 Pass 1."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    classify_future_field_axis,
    rank_matrix_to_event_fields,
    sha256_file,
)


PARENT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OUT = ROOT / "results/topic5_latent_propagation_landscape_v0_2"
MAPPING = PARENT / "TRAIN_MODE_TO_AB_MAPPING.csv"
SPEC = ROOT / "docs/superpowers/specs/2026-08-14-topic5-latent-propagation-landscape-v0-2-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-14-topic5-latent-propagation-landscape-v0-2.md"


def centered_norm(vector: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(vector)
    if int(finite.sum()) < 2:
        return float("nan"), int(finite.sum())
    centered = vector[finite] - float(np.mean(vector[finite]))
    return float(np.linalg.norm(centered)), int(finite.sum())


def build_rows() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING)
    if len(mapping) != 42 or mapping["fit_id"].nunique() != 42:
        raise RuntimeError("parent mode mapping must contain 42 unique fits")
    rows: list[dict[str, object]] = []
    for item in mapping.itertuples(index=False):
        classification = classify_future_field_axis(item.scope, item.mode0, item.mode1)
        events_path = PARENT / "cache" / item.fit_id / "events.npz"
        modes_path = PARENT / "cache" / item.fit_id / "train_only_modes.npz"
        with np.load(events_path, allow_pickle=False) as events:
            ranks = np.asarray(events["ranks"])
            split = np.asarray(events["split"])
            full_train_mode = np.asarray(events["full_train_mode"])
        with np.load(modes_path, allow_pickle=False) as modes:
            train_counts_file = np.asarray(modes["train_counts"], dtype=int)
        _, recurrence = rank_matrix_to_event_fields(ranks)
        train = split == 0
        means: list[np.ndarray] = []
        counts: list[int] = []
        for mode_index in (0, 1):
            use = train & (full_train_mode == mode_index)
            counts.append(int(use.sum()))
            if np.any(use):
                with np.errstate(invalid="ignore"):
                    means.append(np.nanmean(recurrence[use], axis=0))
            else:
                means.append(np.full(recurrence.shape[1], np.nan, dtype=float))
        positive = classification["positive_mode"]
        negative = classification["negative_mode"]
        if positive is None or negative is None or min(counts) == 0:
            norm, n_common = float("nan"), 0
            status = "FIELD_AXIS_NOT_IDENTIFIABLE"
            reason = (
                "AXIS_TRAIN_MODE_MISSING"
                if min(counts) == 0
                else str(classification["reason"])
            )
        else:
            norm, n_common = centered_norm(means[int(positive)] - means[int(negative)])
            status = str(classification["tier"])
            reason = str(classification["reason"])
            if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps * max(1, n_common):
                status = "FIELD_AXIS_NOT_IDENTIFIABLE"
                reason = "NUMERICALLY_DEGENERATE_START_REMOVED_CONTRAST"
        rows.append({
            "subject": str(item.subject),
            "fit_id": str(item.fit_id),
            "scope": str(item.scope),
            "mode0_parent_label": str(item.mode0),
            "mode1_parent_label": str(item.mode1),
            "mapping_source": str(item.mapping_source),
            **classification,
            "status": status,
            "status_reason": reason,
            "n_axis_train_mode0": counts[0],
            "n_axis_train_mode1": counts[1],
            "train_counts_file_mode0": int(train_counts_file[0]),
            "train_counts_file_mode1": int(train_counts_file[1]),
            "n_common_contacts": n_common,
            "start_removed_contrast_norm": norm,
            "events_sha256": sha256_file(events_path),
            "train_modes_sha256": sha256_file(modes_path),
            "target_values_read": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    frame = build_rows()
    status_counts = frame["status"].value_counts().to_dict()
    if status_counts.get("CANONICAL_AB_SHARED", 0) != 14:
        raise RuntimeError(f"expected 14 canonical shared fits, got {status_counts}")
    own = frame["scope"].isin(["own_a", "own_b"])
    if int(own.sum()) != 28 or int((own & frame["canonical_ab"]).sum()) != 0:
        raise RuntimeError("expected 28 non-canonical own fits")
    unidentified = frame["status"].eq("FIELD_AXIS_NOT_IDENTIFIABLE")
    if not frame.loc[unidentified, "scope"].isin(["own_a", "own_b"]).all():
        raise RuntimeError(f"a shared canonical axis became unidentified: {status_counts}")
    if not (frame["n_axis_train_mode0"] == frame["train_counts_file_mode0"]).all():
        raise RuntimeError("mode0 train counts disagree with frozen train_only_modes")
    if not (frame["n_axis_train_mode1"] == frame["train_counts_file_mode1"]).all():
        raise RuntimeError("mode1 train counts disagree with frozen train_only_modes")
    payload = {
        "contract": "topic5_mode_axis_identifiability_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "n_patients": int(frame["subject"].nunique()),
        "n_fits": int(len(frame)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "canonical_ab_patients": sorted(frame.loc[frame["canonical_ab"], "subject"].unique()),
        "contract_amendment": {
            "reason": "parent own_a/own_b mappings map both within-fit train modes to one geometry label",
            "shared_rule": "canonical A/B hidden axis is permitted only within the same shared fit/node space",
            "own_rule": "own fits use a generic mode1-minus-mode0 axis and cannot carry canonical A/B claims",
            "cross_node_hidden_axis": "FORBIDDEN",
            "contact_space_patient_aggregation_after_response_generation": "PERMITTED",
            "statistical_stop_added": False,
        },
        "spec_sha256": sha256_file(SPEC),
        "plan_sha256": sha256_file(PLAN),
        "mapping_sha256": sha256_file(MAPPING),
        "target_values_read": False,
    }
    if args.write:
        atomic_write_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv", frame)
        atomic_write_json(OUT / "EXECUTION_CONTRACT_AMENDMENT_MODE_IDENTIFIABILITY.json", payload)
    print(json.dumps({**payload, "written": bool(args.write)}, indent=2))


if __name__ == "__main__":
    main()
