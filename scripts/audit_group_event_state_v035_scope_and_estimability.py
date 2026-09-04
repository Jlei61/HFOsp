#!/usr/bin/env python3
"""Build the v0.3.5 scope, split, decoder and physical-time estimability audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    CORE_HORIZONS_SECONDS, DATASET_ROOT, DECODER_ROOT, INPUT_ROOT, OUTPUT_ROOT,
    V035_SUBJECTS, atomic_json, initialise_scope_manifest, update_scope_manifest,
)
from src.topic5_group_event_state.v035.dynamic_rate import load_rate_data  # noqa: E402
from src.topic5_group_event_state.v035.contracts import RateTrainConfig  # noqa: E402


FITS = {
    "epilepsiae_253": "epilepsiae_253__own_a", "epilepsiae_1146": "epilepsiae_1146__shared",
    "epilepsiae_548": "epilepsiae_548__shared", "epilepsiae_583": "epilepsiae_583__shared",
    "epilepsiae_922": "epilepsiae_922__own_a",
    "epilepsiae_1096": "epilepsiae_1096__own_a",
    "epilepsiae_384": "epilepsiae_384__shared",
    "epilepsiae_1125": "epilepsiae_1125__own_a",
}
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def main() -> None:
    report_dir = OUTPUT_ROOT / "audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    scope = OUTPUT_ROOT / "scope_manifest.json"
    if not scope.exists():
        initialise_scope_manifest(scope)
    rows = []
    provenance = {}
    cfg = RateTrainConfig()
    for subject in V035_SUBJECTS:
        manifest_path = INPUT_ROOT / subject / "manifest_v3.json"
        dataset_path = DATASET_ROOT / subject / "index.json"
        row = {"subject": subject, "input_manifest": manifest_path.is_file(),
               "full_event_dataset": dataset_path.is_file(), "decoder_fit": FITS.get(subject, "")}
        try:
            data = load_rate_data(subject, cfg)
            for phase in ("FIT", "INNER", "SELECTION"):
                mask = data.phase == phase
                for j, h in enumerate(CORE_HORIZONS_SECONDS):
                    row[f"eligible_{phase.lower()}_{int(h/60)}min"] = int(data.target_valid[mask, j].sum())
                    # Correlation-length accounting: within each real coverage
                    # segment, count disjoint target-length bins represented by
                    # eligible fixed-grid anchors.
                    n_independent = 0
                    for seg in np.unique(data.segment[mask]):
                        t = np.sort(data.anchor_time[mask & (data.segment == seg) & data.target_valid[:, j]])
                        if t.size:
                            n_independent += max(1, int(np.floor((t[-1] - t[0]) / h)) + 1)
                    row[f"independent_{phase.lower()}_{int(h/60)}min"] = n_independent
            row["status"] = "OK"
            provenance[subject] = dict(data.provenance)
        except Exception as exc:
            row["status"] = f"ERROR:{type(exc).__name__}:{exc}"
        fit = FITS.get(subject)
        decoder_units = []
        if fit:
            for seed in range(3):
                unit = DECODER_ROOT / "formal_units" / fit / ARM / f"seed{seed}"
                if (unit / "DONE.json").is_file() and (unit / "weights.pt").is_file():
                    decoder_units.append(str(unit))
        row["n_decoder_seeds"] = len(decoder_units)
        row["decoder_available"] = len(decoder_units) == 3
        rows.append(row)
        provenance.setdefault(subject, {})["decoder_units"] = decoder_units
    fields = sorted({key for row in rows for key in row})
    csv_path = report_dir / "timescale_estimability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    json_path = report_dir / "split_and_decoder_provenance.json"
    atomic_json(json_path, {
        "format": "group_event_state_v0_3_5_scope_estimability_v1", "rows": rows,
        "provenance": provenance, "development_targets_read": False,
        "sealed_partition_opened": False, "seizure_outcomes_read": False,
    })
    update_scope_manifest(scope, "W0", "COMPLETE", [str(csv_path), str(json_path)])
    print(json.dumps({"rows": rows, "csv": str(csv_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
