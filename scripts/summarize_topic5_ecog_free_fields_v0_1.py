#!/usr/bin/env python3
"""Summarize held-out free-generation fields for the four pre-fixed ECoG models."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    for subject in ("958", "1084"):
        for path in sorted((ROOT / "training" / subject).glob("*/field_metrics.json")):
            result = json.loads(path.read_text())
            rows.append({**result, "path": str(path)})
    if len(rows) != 24:
        raise RuntimeError(f"need 24 pre-fixed field units, found {len(rows)}")
    family_rows = []
    for subject in ("958", "1084"):
        for family in ("TRUE_GRID", "WRONG_GRID", "DEGREE_RANDOM", "SUFFIX_SHUFFLED"):
            selected = [row for row in rows if row["subject"] == subject and row["family"] == family]
            if len(selected) != 3:
                raise RuntimeError(f"{subject} {family}: need 3 seeds")
            family_rows.append({
                "subject": subject,
                "family": family,
                "representative_graph": selected[0]["graph_id"],
                "full_field_spearman_median_seed": float(np.median([
                    row["full_field_spearman"] for row in selected
                ])),
                "start_removed_field_spearman_median_seed": float(np.median([
                    row["start_removed_field_spearman"] for row in selected
                ])),
                "generated_participant_count_median_seed": float(np.median([
                    row["generated_participant_count_median"] for row in selected
                ])),
                "observed_participant_count": selected[0]["observed_participant_count_median"],
            })
    write_csv(ROOT / "summary/FREE_FIELD_UNIT_RESULTS.csv", rows)
    write_csv(ROOT / "summary/FREE_FIELD_FAMILY_RESULTS.csv", family_rows)
    payload = {
        "schema": "topic5_ecog_free_field_summary_v0.1",
        "n_units": len(rows),
        "complete": True,
        "family_results": family_rows,
        "boundary": (
            "These fields show whether a trained model can freely regenerate a held-out spatial pattern; "
            "they do not identify an anatomical pathway or replace the graph-null and lesion tests."
        ),
    }
    (ROOT / "summary/FREE_FIELD_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
