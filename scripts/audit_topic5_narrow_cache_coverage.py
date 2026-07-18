#!/usr/bin/env python3
"""Audit narrow ictal-cache coverage against the masked rank-displacement cohort."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_t0_eligibility import _usable_record_channels


DEFAULT_RANK_ROOT = (
    ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
)
DEFAULT_GEOMETRY_ROOT = (
    ROOT
    / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
)
DEFAULT_CACHE_ROOT = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
DEFAULT_SENSITIVITY_AUDIT = (
    ROOT
    / "results/topic5_ictal_recruitment/t0_eligibility_audit_narrow_cache_sensitivity.csv"
)
DEFAULT_OUTPUT = DEFAULT_CACHE_ROOT.parent / "narrow_cache_coverage_audit.csv"


def _inventory_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for dataset in ("epilepsiae", "yuquan"):
        candidates = (
            ROOT / "results/dataset_inventory" / f"{dataset}_seizure_inventory.csv",
            ROOT / "results" / f"{dataset}_seizure_inventory.csv",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                subject = str(row.get("subject", "")).strip()
                if subject:
                    key = (dataset, subject)
                    counts[key] = counts.get(key, 0) + 1
    return counts


def _sensitivity_subjects(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = pd.read_csv(path)
    eligible = rows["narrow_cache_eligible"].astype(str).str.lower().isin(
        {"true", "1", "yes", "t"}
    )
    return set(rows.loc[eligible, "subject_id"].astype(str))


def run(
    rank_root: Path,
    geometry_root: Path,
    cache_root: Path,
    sensitivity_audit: Path,
    output: Path,
) -> Path:
    inventory = _inventory_counts()
    sensitivity_subjects = _sensitivity_subjects(sensitivity_audit)
    rows: list[dict[str, object]] = []

    for rank_path in sorted(rank_root.glob("*.json")):
        subject = rank_path.stem
        dataset, subject_id = subject.split("_", 1)
        geometry_path = geometry_root / f"{subject}_t_a.json"
        geometry_channels = (
            _usable_record_channels(geometry_path) if geometry_path.exists() else None
        )
        cache_path = cache_root / f"{subject}.json"
        cache_meta = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        cache_present = cache_path.exists() and (cache_root / f"{subject}.npz").exists()
        n_inventory = inventory.get((dataset, subject_id), 0)

        if cache_present:
            cache_tier = (
                "narrow_sensitivity_min6_overlap"
                if subject in sensitivity_subjects
                else "primary_existing"
            )
            missing_reason = ""
        elif not geometry_path.exists():
            cache_tier = "missing"
            missing_reason = "no_narrow_geometry_record"
        elif geometry_channels is None:
            cache_tier = "missing"
            missing_reason = "narrow_geometry_not_usable"
        elif n_inventory == 0:
            cache_tier = "missing"
            missing_reason = "no_seizure_inventory_for_subject"
        else:
            cache_tier = "missing"
            missing_reason = "usable_geometry_and_inventory_but_cache_missing"

        rows.append(
            {
                "subject": subject,
                "dataset": dataset,
                "rank_displacement_present": True,
                "narrow_geometry_present": geometry_path.exists(),
                "narrow_geometry_usable": geometry_channels is not None,
                "n_narrow_geometry_channels": len(geometry_channels or []),
                "n_inventory_seizures": n_inventory,
                "cache_present": cache_present,
                "cache_tier": cache_tier,
                "n_cached_seizures": len(cache_meta.get("seizure_idxs", [])),
                "n_cache_drops": len(cache_meta.get("drops", [])),
                "cache_drop_reasons": ";".join(
                    str(drop.get("reason", "")) for drop in cache_meta.get("drops", [])
                ),
                "missing_reason": missing_reason,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    cached = frame[frame["cache_present"]]
    missing = frame[~frame["cache_present"]]
    sensitivity = cached[
        cached["cache_tier"].eq("narrow_sensitivity_min6_overlap")
    ]
    summary = {
        "contract": (
            "coverage denominator is the masked rank-displacement cohort; narrow means "
            "the primary-template observation-readout record. Sensitivity additions waive "
            "only the >=80% montage fraction and remain excluded from primary inference."
        ),
        "n_rank_displacement_subjects": int(len(frame)),
        "n_narrow_geometry_present": int(frame["narrow_geometry_present"].sum()),
        "n_narrow_geometry_usable": int(frame["narrow_geometry_usable"].sum()),
        "n_cache_subjects": int(len(cached)),
        "n_primary_existing_cache_subjects": int(
            cached["cache_tier"].eq("primary_existing").sum()
        ),
        "n_narrow_sensitivity_cache_subjects": int(len(sensitivity)),
        "n_cached_seizures": int(cached["n_cached_seizures"].sum()),
        "n_narrow_sensitivity_cached_seizures": int(
            sensitivity["n_cached_seizures"].sum()
        ),
        "n_missing_cache_subjects": int(len(missing)),
        "missing_reason_counts": dict(Counter(missing["missing_reason"])),
        "sensitivity_subjects": sensitivity[
            ["subject", "n_cached_seizures", "n_cache_drops", "cache_drop_reasons"]
        ].to_dict(orient="records"),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-root", type=Path, default=DEFAULT_RANK_ROOT)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--sensitivity-audit", type=Path, default=DEFAULT_SENSITIVITY_AUDIT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        run(
            args.rank_root.resolve(),
            args.geometry_root.resolve(),
            args.cache_root.resolve(),
            args.sensitivity_audit.resolve(),
            args.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
