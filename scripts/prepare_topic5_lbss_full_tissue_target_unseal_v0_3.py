#!/usr/bin/env python3
"""Authorize one immutable Figure 3 target read after every v0.3 field freeze."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    required = (
        "INTERICTAL_ANALYSIS_COMPLETE.json",
        "MODEL_FIELDS_FROZEN.json",
        "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATED_FIELD_MANIFEST.json",
        "ATTENUATION_COMPLETE.json",
        "EARLY_ICTAL_METADATA_INVENTORY.json",
        "EARLY_ICTAL_METADATA_AUDIT_COMPLETE.json",
    )
    for name in required:
        path = out / name
        if not path.exists():
            raise RuntimeError(f"cannot authorize target access before {name}")
        payload = json.loads(path.read_text())
        if payload.get("target_values_read") is not False:
            raise RuntimeError(f"target-free manifest is not sealed: {name}")
    metadata = json.loads((out / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    if metadata["actual_spatial_join_patients"] != 12 or metadata["actual_spatial_join_seizures"] != 141:
        raise RuntimeError("spatial Figure 3 denominator changed after metadata freeze")
    scorer = Path(__file__).resolve().parent / "score_topic5_lbss_full_tissue_early_ictal_v0_3.py"
    if not scorer.exists():
        raise FileNotFoundError(scorer)
    authorization = {
        "contract": "topic5_lbss_full_tissue_target_unseal_v0_3",
        "authorized": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read_before_authorization": False,
        "target_role": "frozen external benchmark only",
        "target_known_to_project": True,
        "figure3_parent_denominator": {"patients": 17, "seizures": 167},
        "spatial_model_exact_join": {"patients": 12, "seizures": 141},
        "frozen_hashes": {name: sha256(out / name) for name in required},
        "intact_field_manifest_sha256": sha256(out / "MODEL_FIELD_MANIFEST.csv"),
        "attenuated_field_manifest_sha256": sha256(out / "ATTENUATED_FIELD_MANIFEST.csv"),
        "metadata_inventory_csv_sha256": sha256(out / "EARLY_ICTAL_METADATA_INVENTORY.csv"),
        "scorer_sha256": sha256(scorer),
    }
    (out / "TARGET_UNSEAL_AUTHORIZATION.json").write_text(
        json.dumps(authorization, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
