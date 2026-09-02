#!/usr/bin/env python3
"""Audit the response-blind patch freeze and, when present, patch responses."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402
from src.topic5_latent_perturbation_v0_2 import DOSES  # noqa: E402
from scripts.freeze_topic5_latent_reference_states_v0_2 import reference_dir  # noqa: E402
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import (  # noqa: E402
    PATCH, PATCH_FREEZE_REVISION, patch_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402


def audit_freeze(manifest: pd.DataFrame) -> dict[str, object]:
    failures = []; supported = total = 0
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict()); target = patch_dir(row); reasons = []
        try:
            done = json.loads((target / "DONE.json").read_text()); metrics = json.loads((target / "metrics.json").read_text())
            if metrics.get("patch_freeze_revision") != PATCH_FREEZE_REVISION: reasons.append("revision")
            if metrics.get("status") != "PASS": reasons.append("status")
            if not metrics.get("model_hash_unchanged") or not metrics.get("decoder_hash_unchanged"): reasons.append("parameter_hash")
            if metrics.get("response_values_read_before_freeze") is not False or metrics.get("target_values_read") is not False: reasons.append("leak")
            if done.get("patch_contract_sha256") != sha256_file(target / "patch_contract.npz"): reasons.append("contract_hash")
            if metrics.get("reference_contract_sha256") != sha256_file(reference_dir(row) / "reference_contract.npz"): reasons.append("reference_hash")
            with np.load(target / "patch_contract.npz", allow_pickle=False) as source:
                q = {name: np.asarray(source[name]) for name in source.files}
            n_ref, n_node = int(metrics["n_reference_states"]), int(metrics["n_patch_centers"])
            if q["patch_directions"].shape != (n_node, int(item.n_nodes)): reasons.append("direction_shape")
            if q["patch_local_sd"].shape != (n_ref, n_node): reasons.append("sd_shape")
            if q["support_checks"].shape != (n_ref, n_node, len(DOSES), 2, 3): reasons.append("support_shape")
            if not np.allclose(np.linalg.norm(q["patch_directions"], axis=1), 1.0, atol=2e-5): reasons.append("unit_norm")
            if not np.isin(q["support_checks"], [0, 1]).all(): reasons.append("support_nonbinary")
            if not np.allclose(q["doses"], DOSES): reasons.append("doses")
            width = float(q["patch_width_mm"][0]); spacing = float(q["local_node_spacing_mm"][0])
            if not np.isclose(width, 2.0 * spacing, rtol=2e-6): reasons.append("width")
            exact = np.exp(-(
                np.linalg.norm(q["node_xy_mm"][:, None] - q["node_xy_mm"][None, :], axis=-1) ** 2
            ) / (2.0 * width ** 2))
            exact /= np.linalg.norm(exact, axis=1, keepdims=True)
            if not np.allclose(exact, q["patch_directions"], atol=3e-5): reasons.append("gaussian")
            primary = q["support_checks"][:, :, 1].all(axis=(2, 3))
            supported += int(primary.sum()); total += int(primary.size)
            if int(primary.sum()) != int(metrics["primary_supported_state_centers"]): reasons.append("denominator")
        except Exception as error: reasons.append(f"{type(error).__name__}:{error}")
        if reasons: failures.append({"cell_key": f"{item.fit_id}/{item.public_arm}/seed{item.seed}", "reasons": sorted(set(reasons))})
    payload = {
        "contract": "topic5_spatial_patch_freeze_audit_v0_2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "patch_freeze_revision": PATCH_FREEZE_REVISION,
        "status": "PASS" if len(manifest) == 630 and not failures else "FAIL",
        "audited_cells": int(len(manifest)), "primary_supported_state_centers": supported,
        "primary_total_state_centers": total, "failure_count": len(failures), "failures_first20": failures[:20],
        "n0_checks": ["node bounds", "conditional kNN validation q95", "conditional-manifold residual q95"],
        "intervention_policy": "NO_CLIP_NO_RESCALE_NO_REPLACEMENT", "target_values_read": False,
    }
    atomic_write_json(PATCH / "PATCH_FREEZE_AUDIT.json", payload); return payload


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv"); payload = audit_freeze(manifest)
    print(json.dumps(payload, indent=2));
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
