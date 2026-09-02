#!/usr/bin/env python3
"""Audit all frozen Topic 5.2 axis/control/chord response cells."""
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
from src.topic5_latent_perturbation_v0_2 import DOSES  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import (  # noqa: E402
    CONTROL_NAMES, FREEZE_REVISION, reference_dir,
)
from scripts.run_topic5_axis_perturbations_v0_2 import (  # noqa: E402
    HORIZON, PERTURB, PERTURB_REVISION, response_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    failures: list[dict[str, object]] = []
    counts = {
        "audited_cells": 0, "reference_states": 0,
        "primary_progress_supported_states": 0,
        "primary_field_supported_states": 0,
        "axis_open_finite_state_tau_records": 0,
        "axis_closed_common_risk_records": 0,
        "control_open_finite_state_tau_records": 0,
        "control_closed_common_risk_records": 0,
        "chord_open_finite_pair_tau_records": 0,
        "chord_closed_common_risk_records": 0,
    }
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        key = f"{item.fit_id}/{item.public_arm}/seed{item.seed}"
        target = response_dir(row)
        try:
            metrics_path = target / "metrics.json"
            done = json.loads((target / "DONE.json").read_text())
            metrics = json.loads(metrics_path.read_text())
            if not done.get("ok") or metrics.get("status") != "PASS":
                raise RuntimeError("response/DONE status is not PASS")
            if metrics.get("analysis_revision") != ANALYSIS_REVISION:
                raise RuntimeError("analysis revision mismatch")
            if metrics.get("freeze_revision") != FREEZE_REVISION or done.get("perturbation_revision") != PERTURB_REVISION:
                raise RuntimeError("freeze/perturbation revision mismatch")
            for filename, field in (
                ("axis_responses.npz", "axis_sha256"),
                ("control_responses.npz", "controls_sha256"),
                ("chord_responses.npz", "chords_sha256"),
                ("metrics.json", "metrics_sha256"),
            ):
                if done.get(field) != sha256_file(target / filename):
                    raise RuntimeError(f"{filename} content hash mismatch")
            frozen = reference_dir(row) / "reference_contract.npz"
            if metrics.get("reference_contract_sha256") != sha256_file(frozen):
                raise RuntimeError("frozen reference content hash mismatch")
            if any(parse_bool(value) for value in (
                item.target_values_read, metrics.get("target_values_read"), done.get("target_values_read")
            )):
                raise RuntimeError("target marker is not false")
            if not metrics.get("model_hash_unchanged") or not metrics.get("decoder_hash_unchanged"):
                raise RuntimeError("frozen parameter hash changed")
            with np.load(frozen, allow_pickle=False) as q:
                n_ref = len(q["hidden"])
                expected_progress = int(q["axis_support_checks"][:, 0, 1].all(axis=(1, 2)).sum())
                expected_field = int(q["axis_support_checks"][:, 1, 1].all(axis=(1, 2)).sum())
            if metrics["n_reference_states"] != n_ref:
                raise RuntimeError("reference-state denominator drift")
            if metrics["primary_progress_supported_states"] != expected_progress or metrics["primary_field_supported_states"] != expected_field:
                raise RuntimeError("N0 primary support denominator drift")
            with np.load(target / "axis_responses.npz", allow_pickle=False) as axis:
                if axis["open_scores"].shape != (n_ref, 2, len(DOSES), HORIZON + 1, 2):
                    raise RuntimeError("axis response tensor shape drift")
                open_valid = axis["open_valid"].astype(bool)
                closed_risk = axis["closed_risk"].astype(bool)
                if not np.isfinite(axis["open_scores"])[open_valid[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite valid open-loop axis score")
                if not np.isfinite(axis["open_contact_response"])[open_valid[..., None].repeat(int(item.n_contacts), axis=-1)].all():
                    raise RuntimeError("nonfinite valid open-loop contact response")
                if not np.isfinite(axis["closed_scores"])[closed_risk[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite closed-loop common-risk axis score")
                counts["axis_open_finite_state_tau_records"] += int(open_valid.sum())
                counts["axis_closed_common_risk_records"] += int(closed_risk.sum())
            with np.load(target / "control_responses.npz", allow_pickle=False) as control:
                if control["open_scores"].shape != (n_ref, len(CONTROL_NAMES), HORIZON + 1, 2):
                    raise RuntimeError("control response tensor shape drift")
                open_valid = control["open_valid"].astype(bool)
                closed_risk = control["closed_risk"].astype(bool)
                if not np.isfinite(control["open_scores"])[open_valid[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite valid control response")
                if not np.isfinite(control["closed_scores"])[closed_risk[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite common-risk control response")
                counts["control_open_finite_state_tau_records"] += int(open_valid.sum())
                counts["control_closed_common_risk_records"] += int(closed_risk.sum())
            with np.load(target / "chord_responses.npz", allow_pickle=False) as chord:
                if chord["open_scores"].shape[1:] != (len(DOSES), HORIZON + 1, 2):
                    raise RuntimeError("chord response tensor shape drift")
                open_valid = chord["open_valid"].astype(bool)
                closed_risk = chord["closed_risk"].astype(bool)
                if not np.isfinite(chord["open_scores"])[open_valid[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite valid chord response")
                if not np.isfinite(chord["closed_scores"])[closed_risk[..., None].repeat(2, axis=-1)].all():
                    raise RuntimeError("nonfinite common-risk chord response")
                counts["chord_open_finite_pair_tau_records"] += int(open_valid.sum())
                counts["chord_closed_common_risk_records"] += int(closed_risk.sum())
            counts["reference_states"] += n_ref
            counts["primary_progress_supported_states"] += expected_progress
            counts["primary_field_supported_states"] += expected_field
            counts["audited_cells"] += 1
        except Exception as error:
            failures.append({"cell": key, "error_type": type(error).__name__, "error": str(error)})
    payload = {
        "contract": "topic5_axis_perturbation_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION, "freeze_revision": FREEZE_REVISION,
        "perturbation_revision": PERTURB_REVISION,
        "status": "PASS" if not failures and counts["audited_cells"] == 630 else "FAIL",
        "scheduled_cells": 630, **counts,
        "failure_count": len(failures), "failures_first20": failures[:20],
        "invalid_branch_policy": "N0_FAIL_CLOSED_NOT_CLIPPED_NOT_RESCALED",
        "target_values_read": False,
    }
    atomic_write_json(PERTURB / "PERTURBATION_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
