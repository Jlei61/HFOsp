#!/usr/bin/env python3
"""Merge independently generated confound-map shards and audit exact joins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
SHARDS = OUT / "confound_map_shards"
FAST = OUT / "confound_map_fast"
DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    fixed = [
        SHARDS / "shard0/broad/phase1_confound_maps.json",
        SHARDS / "shard2/broad/phase1_confound_maps.json",
    ]
    fast = sorted(FAST.glob("epilepsiae_*/broad/phase1_confound_maps.json"))
    if len(fast) == 5:
        paths = [*fixed, *fast]
    else:
        paths = [
            *fixed,
            SHARDS / "shard1/broad/phase1_confound_maps.json",
        ]
    if any(not path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise RuntimeError(f"confound map inputs incomplete: {missing}")
    merged = {}
    provenance = {}
    for path in paths:
        payload = json.loads(path.read_text())
        overlap = set(merged).intersection(payload)
        if overlap:
            raise RuntimeError(f"duplicate subjects across shards: {overlap}")
        merged.update(payload)
        provenance[str(path.relative_to(ROOT))] = sha256(path)
    if len(merged) != 16:
        raise RuntimeError(f"expected 16 confound subjects, found {len(merged)}")
    destination = OUT / "confound_maps/phase1_confound_maps.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
    )
    rows = []
    for subject, record in sorted(merged.items()):
        with np.load(
            DATASET / "per_subject" / f"{subject}.npz",
            allow_pickle=False,
        ) as data:
            names = np.asarray(data["contact_names"]).astype(str)
        row = {"subject": subject, "n_model_contacts": len(names)}
        for key in (
            "hfo_rate",
            "baseline_band_power",
            "broadband_1_250",
            "shaft_position",
            "soz",
        ):
            mapping = record.get(key, {})
            exact = sum(name in mapping for name in names)
            row[f"{key}_map_size"] = len(mapping)
            row[f"{key}_exact_joined"] = exact
            row[f"{key}_all_model_contacts"] = bool(exact == len(names))
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("subject")
    frame.to_csv(OUT / "confound_maps/exact_join_audit.csv", index=False)
    summary = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "baseline_power_and_contact_confound_map_merge",
        "status": "COMPLETE",
        "n_patients": len(frame),
        "source_shards": provenance,
        "merged_sha256": sha256(destination),
        "coverage": {
            key: {
                "any_exact_join": int(
                    np.count_nonzero(frame[f"{key}_exact_joined"] >= 1)
                ),
                "at_least_six_exact": int(
                    np.count_nonzero(frame[f"{key}_exact_joined"] >= 6)
                ),
                "all_model_contacts": int(
                    frame[f"{key}_all_model_contacts"].sum()
                ),
            }
            for key in (
                "hfo_rate",
                "baseline_band_power",
                "broadband_1_250",
                "shaft_position",
                "soz",
            )
        },
        "target_values_read": False,
    }
    (OUT / "BASELINE_POWER_CONFOUND_AUDIT.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
