#!/usr/bin/env python3
"""Freeze E384 phenotype-target availability without inventing a new endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load cuda_env's compatible C++ runtime before pandas extensions.
import torch as _torch  # noqa: F401
import numpy as np
import pandas as pd

from src.topic5_continuous_marked_state_h2b.contract import (
    RESULT_ROOT, atomic_csv, atomic_json, sha256_file, utc_now,
)


SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
LOCKED_BOUNDARY = Path(
    "docs/archive/topic5/epi_prssm_h2b_h3_revision_technical_2026-08-20.md"
)
RECRUITMENT_JSON = Path(
    "results/topic5_ictal_recruitment/ictal_field_long_cache/epilepsiae_384.json"
)
RECRUITMENT_NPZ = RECRUITMENT_JSON.with_suffix(".npz")


def run(source_repo: Path, result_root: Path) -> dict:
    source = Path(source_repo).resolve()
    root = Path(result_root).resolve()
    query = pd.read_csv(
        root / "risk_sets/e384_state_queries.csv",
        dtype={"case_seizure_id": str},
    )
    case = query[
        query["case_seizure_id"].notna()
        & (query["case_seizure_id"].astype(str).str.len() > 0)
        & (query["case_lead_minutes"].astype(float) == 30.0)
    ][["query_id", "case_seizure_id"]].copy()
    if case["case_seizure_id"].nunique() != 4:
        raise ValueError("E384 frozen primary support must contain four seizures")

    boundary = source / LOCKED_BOUNDARY
    cache_json = source / RECRUITMENT_JSON
    cache_npz = source / RECRUITMENT_NPZ
    cache = json.loads(cache_json.read_text(encoding="utf-8"))
    cached_ids = {
        str(row["seizure_id"]) for row in (cache.get("seizure") or {}).values()
    }
    primary_ids = sorted(case["case_seizure_id"].astype(str).unique())
    cached_primary = sorted(set(primary_ids).intersection(cached_ids))
    missing_primary = sorted(set(primary_ids).difference(cached_ids))

    rows: list[dict] = []
    for seed in (1, 3, 4):
        features = pd.read_csv(
            root / f"per_subject/epilepsiae_384/seed_{seed}_anchor_features.csv"
        )
        selected = case.merge(
            features, left_on="query_id", right_on="anchor_id",
            how="left", validate="one_to_one",
        )
        if selected["anchor_time"].isna().any():
            raise ValueError(f"seed {seed}: primary phenotype anchors are missing")
        for record in selected.to_dict(orient="records"):
            common = {
                "patient_id": "epilepsiae_384",
                "seed": int(seed),
                "seizure_id": str(record["case_seizure_id_x"]),
                "split": "NOT_ESTIMABLE",
                "evaluation_tier": "not_estimable",
                "target_frozen": True,
                "target_provenance": (
                    "locked blind-onset contract; registry 0/71; no scalar target"
                ),
                "target_source_sha256": sha256_file(boundary),
            }
            for name, value in record.items():
                if name.startswith("history__"):
                    common[f"baseline__{name.removeprefix('history__')}"] = value
                elif name.startswith("observation__"):
                    common[name] = value
                elif name.startswith("state__") or name.startswith("wrong_time__"):
                    common[name] = value
            for target_name, target_kind in (
                ("preexisting_seizure_subtype", "classification"),
                ("blind_early_recruitment_extent", "continuous"),
            ):
                rows.append({
                    **common,
                    "target_name": target_name,
                    "target_kind": target_kind,
                    "target_value": np.nan,
                })

    table_path = root / "risk_sets/e384_phenotype_targets.csv"
    atomic_csv(table_path, rows)
    payload = {
        "status": "NOT_ESTIMABLE_FROZEN_TARGET_UNAVAILABLE",
        "created_utc": utc_now(),
        "subject": "epilepsiae_384",
        "primary_lead_minutes": 30,
        "primary_seizure_ids": primary_ids,
        "n_primary_seizures": len(primary_ids),
        "preexisting_seizure_subtype_available": False,
        "blind_early_recruitment_target_available": False,
        "blind_onset_contact_registry_support": "0/71",
        "locked_boundary": str(boundary),
        "locked_boundary_sha256": sha256_file(boundary),
        "signal_cache_is_not_a_frozen_scalar_target": True,
        "signal_cache_primary_seizure_ids": cached_primary,
        "signal_cache_missing_primary_seizure_ids": missing_primary,
        "signal_cache_json": str(cache_json),
        "signal_cache_json_sha256": sha256_file(cache_json),
        "signal_cache_npz": str(cache_npz),
        "signal_cache_npz_sha256": sha256_file(cache_npz),
        "target_reclustered": False,
        "replacement_target_invented": False,
        "forbidden_substitutes_used": [],
        "reason": (
            "The locked blind-onset contract has no adjudicated onset contacts. "
            "The existing ictal cache contains per-channel signal arrays, not a "
            "pre-frozen seizure-level extent target; defining one after seeing H2b "
            "states would violate the cross-task contract."
        ),
        "probe_input": str(table_path),
        "probe_input_sha256": sha256_file(table_path),
        "n_probe_rows": len(rows),
    }
    output = root / "reports/e384_phenotype_availability.json"
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, default=SOURCE_REPO)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_repo_root, args.result_root), indent=2))


if __name__ == "__main__":
    main()
