#!/usr/bin/env python3
"""Freeze prespecified patient-level sensitivities for the v0.5 primary interaction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyse_topic5_multiscale_interictal_v0_5 import interaction  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def evaluate(rows: pd.DataFrame, seed: int) -> dict:
    result = interaction(rows.J_lat_exceedance_burden, rows.gain_nats, seed=seed)
    result["subjects"] = sorted(rows.subject.astype(str).tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("primary sensitivities must be frozen before target unseal")
    source = out / "INTERICTAL_V0_5_SUMMARY.json"
    summary = json.loads(source.read_text())
    rows = pd.DataFrame(summary["primary_rows"])
    rows = rows.loc[rows.distal_inferential_eligible.astype(bool)].copy()
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    min_contacts = census.groupby("subject").n_joint_contacts.min()
    highest_J = str(rows.sort_values("J_lat_exceedance_burden").iloc[-1].subject)
    subsets = {
        "all_prespecified_primary_patients": rows,
        "exclude_6_7_contact_patients": rows.loc[
            rows.subject.map(min_contacts).astype(int) >= 8
        ],
        "exclude_highest_J_patient": rows.loc[rows.subject != highest_J],
        "two_dimensional_geometry_only": rows.loc[rows.geometry_2d.astype(bool)],
    }
    payload = {
        "contract": "topic5_v0_5_primary_interaction_fixed_sensitivity_addendum",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "estimand": "Spearman(J_lat_exceedance_burden, L2m_NLL_minus_L3_NLL_distal)",
        "highest_J_patient_removed": highest_J,
        "analyses": {
            label: evaluate(frame, 2026081410 + index)
            for index, (label, frame) in enumerate(subsets.items())
        },
        "source_hashes": {
            "INTERICTAL_V0_5_SUMMARY.json": sha256_file(source),
            "FULL_PARENT_FIT_CENSUS.csv": sha256_file(out / "FULL_PARENT_FIT_CENSUS.csv"),
        },
    }
    destination = out / "INTERICTAL_PRIMARY_SENSITIVITY_ADDENDUM.json"
    write_json(destination, payload)
    write_json(out / "INTERICTAL_PRIMARY_SENSITIVITY_PREFREEZE_MANIFEST.json", {
        "status": "PASS", "target_values_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "addendum_sha256": sha256_file(destination),
        "source_hashes": payload["source_hashes"],
    })


if __name__ == "__main__":
    main()
