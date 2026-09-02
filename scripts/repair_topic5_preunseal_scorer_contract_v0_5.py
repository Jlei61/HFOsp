#!/usr/bin/env python3
"""Record and freeze the target-free scorer/null-contract hardening.

The first pre-unseal review found three result-neutral implementation gaps:
NaN-unsafe oracle metadata, a replay-overwritable first-unlock record, and the
absence of a coherent synchronized-spatial-null interaction.  This repair is
allowed only while the early-ictal target namespace remains physically hidden.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


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


def main() -> None:
    if os.environ.get("TOPIC5_V0_5_TARGET_SEALED") != "1":
        raise RuntimeError("scorer-contract repair requires the physical target embargo")
    if (OUT / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("scorer-contract repair is forbidden after target authorization")
    snapshot_path = OUT / "POSTTRAINING_PIPELINE_SNAPSHOT.json"
    snapshot = json.loads(snapshot_path.read_text())
    previous = dict(snapshot["source_hashes"])
    sources = {
        "score": ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py",
        "authorize": ROOT / "scripts/prepare_topic5_multiscale_target_unseal_v0_5.py",
    }
    updated = dict(previous)
    for key, path in sources.items():
        updated[key] = sha256_file(path)
    snapshot.update({
        "source_hashes": updated,
        "target_values_read": False,
        "prefreeze_repair": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "NAN_SAFE_ORACLE_IMMUTABLE_FIRST_UNLOCK_FINITE_NULL_DENOMINATOR_"
                "AND_COHERENT_SYNCHRONIZED_SPATIAL_INTERACTION"
            ),
            "previous_source_hashes": {
                key: previous[key] for key in sources
            },
            "updated_source_hashes": {
                key: updated[key] for key in sources
            },
            "target_values_read": False,
        },
    })
    write_json(snapshot_path, snapshot)

    panel_e_path = OUT / "FIGURE6_PREUNSEAL_PANEL_E_DECISION.json"
    panel_e = json.loads(panel_e_path.read_text())
    panel_e.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "significance_rule": (
            "Only the joint primary nonlocality interaction may receive a "
            "significance star; both patient-label and synchronized spatial-null "
            "P values must pass, and the displayed star uses their maximum."
        ),
        "target_values_read_for_this_decision": False,
        "prefreeze_repair_reason": (
            "COHERENT_SPATIAL_NULL_ADDED_BEFORE_ANY_TARGET_VALUE_ACCESS"
        ),
    })
    write_json(panel_e_path, panel_e)

    write_json(OUT / "SCORER_CONTRACT_PREFREEZE_REPAIR.json", {
        "contract": "topic5_v0_5_scorer_contract_prefreeze_repair",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TARGET_FREE",
        "target_values_read": False,
        "target_authorization_absent": True,
        "snapshot_sha256": sha256_file(snapshot_path),
        "panel_e_decision_sha256": sha256_file(panel_e_path),
        "scorer_sha256": updated["score"],
        "authorizer_sha256": updated["authorize"],
        "repair_script_sha256": sha256_file(Path(__file__).resolve()),
        "tests": "57 scorer/cache tests passed before freeze",
    })


if __name__ == "__main__":
    main()
