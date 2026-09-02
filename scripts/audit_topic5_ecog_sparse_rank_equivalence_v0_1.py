#!/usr/bin/env python3
"""Freeze full-block versus padded-window ECoG rank equivalence evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/sparse_validation"
    ))
    args = parser.parse_args()
    contracts = (
        ("958", "95800102_0000"),
        ("1084", "108400102_0005"),
        ("1084", "108401102_0026"),
    )
    records = []
    for subject, stem in contracts:
        full_path = args.root / "full" / subject / "per_block" / f"{stem}.npz"
        # The E958 full-block smoke is the original formal cache; it predates
        # the sparse-validation folder but uses the identical signal contract.
        if subject == "958" and not full_path.exists():
            full_path = Path(
                "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache/958/per_block"
            ) / f"{stem}.npz"
        sparse_path = args.root / "sparse" / subject / "per_block" / f"{stem}.npz"
        with np.load(full_path, allow_pickle=False) as full, np.load(sparse_path, allow_pickle=False) as sparse:
            participation = np.asarray(full["participation"], dtype=bool)
            difference = np.abs(
                np.asarray(full["lag_sec"])[participation]
                - np.asarray(sparse["lag_sec"])[participation]
            )
            record = {
                "subject": subject,
                "block_stem": stem,
                "n_events": int(len(full["ranks"])),
                "n_participations": int(participation.sum()),
                "participation_exact": bool(np.array_equal(full["participation"], sparse["participation"])),
                "rank_matrix_exact": bool(np.array_equal(full["ranks"], sparse["ranks"])),
                "event_rows_exact_fraction": float(np.mean(np.all(full["ranks"] == sparse["ranks"], axis=1))),
                "lag_absolute_p99_ms": float(np.quantile(difference, 0.99) * 1000.0),
                "lag_absolute_max_ms": float(np.max(difference) * 1000.0),
                "full_path": str(full_path),
                "sparse_path": str(sparse_path),
            }
            records.append(record)
    payload = {
        "schema": "topic5_ecog_sparse_rank_equivalence_v0.1",
        "all_participation_exact": all(row["participation_exact"] for row in records),
        "all_rank_matrices_exact": all(row["rank_matrix_exact"] for row in records),
        "records": records,
    }
    if not payload["all_participation_exact"] or not payload["all_rank_matrices_exact"]:
        raise RuntimeError(json.dumps(payload, indent=2))
    output = args.root / "SPARSE_READ_EQUIVALENCE_AUDIT.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
