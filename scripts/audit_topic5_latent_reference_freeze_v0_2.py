#!/usr/bin/env python3
"""Audit the sealed response-blind Topic 5.2 Pass 2 reference contract."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, parse_bool, sha256_file  # noqa: E402
from src.topic5_latent_perturbation_v0_2 import DOSES, PHASE_TARGETS  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    CONTROL_NAMES,
    FREEZE_REVISION,
    REFERENCE,
    reference_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    frozen_manifest = pd.read_csv(REFERENCE / "REFERENCE_STATE_MANIFEST.csv")
    seal = json.loads((REFERENCE / "REFERENCE_FREEZE_SEAL.json").read_text())
    failures: list[dict[str, object]] = []
    counts = {
        "audited_cells": 0, "reference_states": 0,
        "primary_progress_supported_states": 0,
        "primary_field_supported_states": 0,
        "axis_branch_fail_node_bounds": 0,
        "axis_branch_fail_conditional_knn": 0,
        "axis_branch_fail_manifold_residual": 0,
        "high_u_chords": 0, "small_u_chords": 0,
        "high_u_primary_supported": 0, "small_u_primary_supported": 0,
        "recovered_failure_records": 0,
    }
    if not seal.get("sealed") or seal.get("freeze_revision") != FREEZE_REVISION:
        failures.append({"cell": "SEAL", "error": "seal status/revision mismatch"})
    if seal.get("manifest_sha256") != sha256_file(REFERENCE / "REFERENCE_STATE_MANIFEST.csv"):
        failures.append({"cell": "SEAL", "error": "manifest content hash mismatch"})
    if seal.get("response_values_read_before_freeze") is not False:
        failures.append({"cell": "SEAL", "error": "response read before freeze"})

    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = reference_dir(row)
        key = f"{item.fit_id}/{item.public_arm}/seed{item.seed}"
        try:
            done = json.loads((target / "DONE.json").read_text())
            profile = json.loads((target / "profile.json").read_text())
            for filename, field in (
                ("reference_contract.npz", "contract_sha256"),
                ("chords.csv", "chords_sha256"),
                ("reference_manifest.csv", "manifest_sha256"),
                ("profile.json", "profile_sha256"),
            ):
                if done.get(field) != sha256_file(target / filename):
                    raise RuntimeError(f"{filename} content hash mismatch")
            if not done.get("ok") or done.get("freeze_revision") != FREEZE_REVISION:
                raise RuntimeError("DONE status/revision mismatch")
            if profile.get("analysis_revision") != ANALYSIS_REVISION or profile.get("freeze_revision") != FREEZE_REVISION:
                raise RuntimeError("profile analysis/freeze revision mismatch")
            if any(parse_bool(value) for value in (
                item.target_values_read, done.get("target_values_read"),
                profile.get("target_values_read"),
            )):
                raise RuntimeError("target marker is not false")
            if not profile.get("model_hash_unchanged") or not profile.get("decoder_hash_unchanged"):
                raise RuntimeError("frozen parameter hash changed")
            local_manifest = pd.read_csv(target / "reference_manifest.csv")
            with np.load(target / "reference_contract.npz", allow_pickle=False) as source:
                n_ref = len(source["hidden"])
                if source["axis_support_checks"].shape != (n_ref, 2, len(DOSES), 2, 3):
                    raise RuntimeError("axis support tensor shape drift")
                if source["control_support_checks"].shape != (n_ref, len(CONTROL_NAMES), 2, 3):
                    raise RuntimeError("control support tensor shape drift")
                if set(np.round(np.unique(source["phase_target"]), 8)) != set(PHASE_TARGETS):
                    raise RuntimeError("reference phase target drift")
                if not np.isfinite(source["hidden"]).all() or not np.isfinite(source["recruited"]).all():
                    raise RuntimeError("nonfinite frozen q")
                checks = source["axis_support_checks"]
                counts["primary_progress_supported_states"] += int(checks[:, 0, 1].all(axis=(1, 2)).sum())
                counts["primary_field_supported_states"] += int(checks[:, 1, 1].all(axis=(1, 2)).sum())
                counts["axis_branch_fail_node_bounds"] += int((checks[..., 0] == 0).sum())
                counts["axis_branch_fail_conditional_knn"] += int((checks[..., 1] == 0).sum())
                counts["axis_branch_fail_manifold_residual"] += int((checks[..., 2] == 0).sum())
            if (
                len(local_manifest) != n_ref
                or local_manifest["reference_index"].duplicated().any()
                or local_manifest[["q_replay_key", "phase_target"]].duplicated().any()
            ):
                raise RuntimeError("local reference manifest denominator/uniqueness drift")
            chords = pd.read_csv(target / "chords.csv")
            for family, count_key, support_key in (
                ("HIGH_U", "high_u_chords", "high_u_primary_supported"),
                ("SMALL_U", "small_u_chords", "small_u_primary_supported"),
            ):
                part = chords[chords["family"].eq(family)]
                counts[count_key] += int(len(part))
                counts[support_key] += int(part["support_eta_0.50"].map(parse_bool).sum())
            counts["reference_states"] += int(n_ref)
            counts["audited_cells"] += 1
            counts["recovered_failure_records"] += int((target / "RECOVERED_FAILURE.json").is_file())
        except Exception as error:
            failures.append({"cell": key, "error_type": type(error).__name__, "error": str(error)})
    if (
        len(frozen_manifest) != counts["reference_states"]
        or frozen_manifest[["fit_id", "public_arm", "seed", "reference_index"]].duplicated().any()
        or frozen_manifest[["q_replay_key", "phase_target"]].duplicated().any()
    ):
        failures.append({"cell": "GLOBAL_MANIFEST", "error": "global denominator/uniqueness drift"})
    if frozen_manifest["target_values_read"].map(parse_bool).any():
        failures.append({"cell": "GLOBAL_MANIFEST", "error": "target marker true"})
    payload = {
        "contract": "topic5_pass2_reference_freeze_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "freeze_revision": FREEZE_REVISION,
        "status": "PASS" if not failures and counts["audited_cells"] == 630 else "FAIL",
        "scheduled_cells": 630, **counts,
        "failure_count": len(failures), "failures_first20": failures[:20],
        "response_values_read_before_freeze": False, "target_values_read": False,
    }
    atomic_write_json(REFERENCE / "REFERENCE_FREEZE_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
