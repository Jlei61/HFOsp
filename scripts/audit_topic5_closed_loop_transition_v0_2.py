#!/usr/bin/env python3
"""Audit the frozen-decoder closed-loop transition archive used by C2."""
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
from scripts.freeze_topic5_latent_reference_states_v0_2 import reference_dir  # noqa: E402
from scripts.run_topic5_closed_loop_transition_v0_2 import (  # noqa: E402
    TRANSITION, TRANSITION_REVISION, transition_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    failures: list[dict[str, object]] = []
    totals = {
        "reference_states": 0, "teacher_forced_valid_transitions": 0,
        "closed_loop_valid_transitions": 0, "joint_valid_transitions": 0,
    }
    recovered_sets = 0
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        key = f"{item.fit_id}/{item.public_arm}/seed{item.seed}"
        target = transition_dir(row)
        reasons: list[str] = []
        try:
            metrics = json.loads((target / "metrics.json").read_text())
            done = json.loads((target / "DONE.json").read_text())
            reference = reference_dir(row) / "reference_contract.npz"
            if metrics.get("status") != "PASS": reasons.append("metrics_status")
            if metrics.get("transition_revision") != TRANSITION_REVISION: reasons.append("revision")
            if metrics.get("target_values_read") is not False: reasons.append("target_values_read")
            if not metrics.get("model_hash_unchanged"): reasons.append("model_hash")
            if not metrics.get("decoder_hash_unchanged"): reasons.append("decoder_hash")
            if metrics.get("reference_contract_sha256") != sha256_file(reference):
                reasons.append("reference_hash")
            if done.get("transition_sha256") != sha256_file(target / "transition.npz"):
                reasons.append("transition_hash")
            with np.load(reference, allow_pickle=False) as source:
                recruited0 = np.asarray(source["recruited"], dtype=bool)
            with np.load(target / "transition.npz", allow_pickle=False) as source:
                arrays = {name: np.asarray(source[name]) for name in source.files}
            n = len(recruited0)
            expected = {
                "teacher_forced_delta_z": (n, 3, 2),
                "closed_loop_delta_z": (n, 3, 2),
                "teacher_forced_delta_manifold_distance": (n, 3),
                "closed_loop_delta_manifold_distance": (n, 3),
                "teacher_forced_valid": (n, 3), "closed_loop_valid": (n, 3),
                "generated_sets": (n, 3, int(item.n_contacts)),
            }
            for name, shape in expected.items():
                if arrays[name].shape != shape: reasons.append(f"shape:{name}")
            tf = arrays["teacher_forced_valid"].astype(bool)
            cl = arrays["closed_loop_valid"].astype(bool)
            if not np.isfinite(arrays["teacher_forced_delta_z"][tf]).all(): reasons.append("tf_nonfinite")
            if not np.isfinite(arrays["closed_loop_delta_z"][cl]).all(): reasons.append("cl_nonfinite")
            if not np.isfinite(arrays["teacher_forced_delta_manifold_distance"][tf]).all():
                reasons.append("tf_distance_nonfinite")
            if not np.isfinite(arrays["closed_loop_delta_manifold_distance"][cl]).all():
                reasons.append("cl_distance_nonfinite")
            generated = arrays["generated_sets"]
            if not np.isin(generated, [0, 1]).all(): reasons.append("generated_nonbinary")
            recruited = recruited0.copy()
            for tau in range(3):
                active = cl[:, tau]
                if np.any(generated[active, tau].sum(axis=1) < 1): reasons.append(f"empty_generated_tau{tau+1}")
                if np.any(generated[active, tau].astype(bool) & recruited[active]):
                    reasons.append(f"repeated_generated_tau{tau+1}")
                recruited |= generated[:, tau].astype(bool)
                recovered_sets += int(active.sum())
            totals["reference_states"] += n
            totals["teacher_forced_valid_transitions"] += int(tf.sum())
            totals["closed_loop_valid_transitions"] += int(cl.sum())
            totals["joint_valid_transitions"] += int((tf & cl).sum())
        except Exception as error:  # fail closed and preserve cell identity
            reasons.append(f"{type(error).__name__}:{error}")
        if reasons:
            failures.append({"cell_key": key, "reasons": sorted(set(reasons))})
    payload = {
        "contract": "topic5_closed_loop_transition_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transition_revision": TRANSITION_REVISION,
        "status": "PASS" if len(manifest) == 630 and not failures else "FAIL",
        "scheduled_cells": 630, "audited_cells": int(len(manifest)),
        **totals, "generated_active_sets_audited": recovered_sets,
        "failure_count": len(failures), "failures_first20": failures[:20],
        "checks": [
            "cell/revision/hash completeness", "parameter hashes unchanged",
            "reference archive unchanged", "valid-mask finite contract",
            "generated sets binary/nonempty/no-repeat", "target remains sealed",
        ],
        "target_values_read": False,
    }
    atomic_write_json(TRANSITION / "CLOSED_LOOP_TRANSITION_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
