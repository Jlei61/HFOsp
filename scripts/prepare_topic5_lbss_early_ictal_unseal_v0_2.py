#!/usr/bin/env python3
"""Freeze LBSS downstream manifests and authorize the single target-value read."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OLD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-rnn-motif-cross-state-v0-4/"
    "results/topic5_rnn_motif_cross_state_benchmark_v0_4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    args = parser.parse_args()
    out = args.out_root.resolve()
    required = (
        "MODEL_FIELDS_FROZEN.json", "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATED_FIELD_MANIFEST.json", "ATTENUATION_COMPLETE.json",
    )
    payloads = {}
    for name in required:
        path = out / name
        if not path.exists():
            raise RuntimeError(f"cannot authorize target access before {name}")
        payloads[name] = json.loads(path.read_text())
        if payloads[name].get("target_values_read") is not False:
            raise RuntimeError(f"target-free manifest is not sealed: {name}")
    source_metadata = OLD_ROOT / "EARLY_ICTAL_METADATA_INVENTORY.json"
    metadata = json.loads(source_metadata.read_text())
    if metadata.get("target_values_read") is not False:
        raise RuntimeError("source metadata inventory unexpectedly read target values")
    metadata.update({
        "contract": "topic5_lbss_early_ictal_metadata_inventory_v0_2",
        "source_inventory": str(source_metadata),
        "source_inventory_sha256": sha256(source_metadata),
        "target_values_read": False,
    })
    metadata_path = out / "EARLY_ICTAL_METADATA_INVENTORY.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    scorer = Path(__file__).resolve().parent / "score_topic5_lbss_early_ictal_v0_2.py"
    authorization = {
        "contract": "topic5_lbss_target_unseal_authorization_v0_2",
        "authorized": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read_before_authorization": False,
        "target_role": "frozen external benchmark only",
        "target_known_to_project": True,
        "n_primary_patients": int(metadata["actual_primary_join_n"]),
        "n_primary_seizures": int(metadata["n_primary_seizures"]),
        "frozen_hashes": {name: sha256(out / name) for name in required},
        "intact_field_manifest_sha256": sha256(out / "MODEL_FIELD_MANIFEST.csv"),
        "attenuated_field_manifest_sha256": sha256(out / "ATTENUATED_FIELD_MANIFEST.csv"),
        "metadata_inventory_sha256": sha256(metadata_path),
        "scorer_sha256": sha256(scorer),
    }
    (out / "TARGET_UNSEAL_AUTHORIZATION.json").write_text(
        json.dumps(authorization, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
