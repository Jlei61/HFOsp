#!/usr/bin/env python3
"""Build outcome-independent five-minute development grids for H2b v0.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from src.topic5_continuous_marked_state_h2b.state_extraction import (  # noqa: E402
    InferenceRawAnchorReader,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable  # noqa: E402
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design  # noqa: E402


DEFAULT_SUBJECTS = (
    "epilepsiae_1073", "epilepsiae_1077", "epilepsiae_1125",
    "epilepsiae_253", "epilepsiae_442", "epilepsiae_548",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _downsample(
    time_epoch: np.ndarray, segment: np.ndarray, *, spacing_seconds: float,
) -> np.ndarray:
    selected: list[int] = []
    for label in np.unique(segment):
        rows = np.flatnonzero(segment == label)
        rows = rows[np.argsort(time_epoch[rows], kind="stable")]
        cursor = -np.inf
        for row in rows:
            if float(time_epoch[row]) >= cursor - 1e-9:
                selected.append(int(row))
                cursor = float(time_epoch[row]) + float(spacing_seconds)
    return np.asarray(sorted(selected, key=lambda row: time_epoch[row]), dtype=np.int64)


def build_subject(subject: str, *, v02_root: Path, result_root: Path) -> dict:
    input_manifest_path = v02_root / "risk_sets" / subject / "input_manifest.json"
    input_manifest = _json(input_manifest_path)
    coverage_path = Path(input_manifest["coverage_path"])
    design_path = Path(input_manifest["design_path"])
    if input_manifest["coverage_sha256"] != sha256_file(coverage_path):
        raise ValueError(f"{subject}: coverage SHA256 drift")
    if input_manifest["design_sha256"] != sha256_file(design_path):
        raise ValueError(f"{subject}: design SHA256 drift")
    coverage = CoverageTable.load(coverage_path)
    design = load_full_design(design_path)
    reader = InferenceRawAnchorReader(
        subject, design.event_time,
        source_repo_root=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    time_epoch, segment, continuity, _, _ = reader.inference_anchor_inventory(
        coverage,
    )
    development = time_epoch < float(coverage.dev_end_epoch)
    time_epoch, segment, continuity = (
        value[development] for value in (time_epoch, segment, continuity)
    )
    take = _downsample(time_epoch, segment, spacing_seconds=300.0)
    time_epoch, segment, continuity = (
        value[take] for value in (time_epoch, segment, continuity)
    )
    rows = [{
        "anchor_time_epoch": float(time),
        "coverage_segment_index": int(group),
        "continuity_session": int(session),
        "query_role": "full_recorded_development_grid",
        "query_id": f"{subject}_full_{index:07d}",
    } for index, (time, group, session) in enumerate(zip(
        time_epoch, segment, continuity,
    ))]
    output = result_root / "full_grid/queries" / f"{subject}.csv"
    atomic_csv(output, rows)
    payload = {
        "status": "COMPLETE", "revision": "h2b_v0_3_full_grid_query_v1",
        "created_utc": utc_now(), "subject": subject,
        "n_queries": len(rows), "n_coverage_segments": int(len(np.unique(segment))),
        "spacing_seconds": 300.0,
        "maximum_anchor_epoch": float(np.max(time_epoch)) if len(time_epoch) else None,
        "development_end_epoch": float(coverage.dev_end_epoch),
        "all_anchors_before_development_end": bool(
            len(time_epoch) == 0 or np.max(time_epoch) < float(coverage.dev_end_epoch)
        ),
        "query_role_is_outcome_independent": True,
        "seizure_table_read": False,
        "query_path": str(output), "query_sha256": sha256_file(output),
        "source": {
            "input_manifest": str(input_manifest_path),
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "coverage": str(coverage_path),
            "coverage_sha256": sha256_file(coverage_path),
            "design": str(design_path), "design_sha256": sha256_file(design_path),
            "producer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(output.with_suffix(".manifest.json"), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    args = parser.parse_args()
    rows = [build_subject(
        subject, v02_root=args.v0_2_root.resolve(),
        result_root=args.result_root.resolve(),
    ) for subject in args.subjects]
    atomic_json(args.result_root.resolve() / "full_grid/query_build_summary.json", {
        "status": "COMPLETE", "created_utc": utc_now(), "subjects": rows,
        "n_subjects": len(rows), "formal_test_partition_opened": False,
        "sealed_opened": False, "h3_or_t2_run": False,
    })
    print(f"COMPLETE subjects={len(rows)} queries={sum(row['n_queries'] for row in rows)}")


if __name__ == "__main__":
    main()
