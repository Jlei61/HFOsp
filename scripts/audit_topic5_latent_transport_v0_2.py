#!/usr/bin/env python3
"""Audit all frozen Topic 5.2 teacher-forced transport cells."""
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
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT, SYSTEM  # noqa: E402
from scripts.run_topic5_latent_transport_v0_2 import (  # noqa: E402
    PHASE_TARGETS,
    TRANSPORT,
    TRANSPORT_REVISION,
    transport_dir,
)


REQUIRED_FINITE = (
    "progress_transport_cosine",
    "progress_gain",
    "normal_gain_median",
    "transverse_contraction",
    "progress_gain_minus_normal",
    "distance_to_progress_curve",
    "next_distance_to_progress_curve",
    "event_to_curve_convergence",
    "distance_to_PF_manifold",
    "next_distance_to_PF_manifold",
    "event_to_PF_manifold_convergence",
)


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    samples = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    failures: list[dict[str, object]] = []
    counters = {
        "scheduled_cells": int(len(manifest)),
        "audited_cells": 0,
        "audited_reference_states": 0,
        "canonical_cells": 0,
        "within_fit_cells": 0,
        "negative_PF_convergence_rows_retained": 0,
    }
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        key = f"{item.fit_id}/{item.public_arm}/seed{item.seed}"
        target = transport_dir(row)
        try:
            metrics_path = target / "metrics.json"
            frame_path = target / "transport.csv"
            done_path = target / "DONE.json"
            for path in (metrics_path, frame_path, done_path):
                if not path.is_file():
                    raise RuntimeError(f"missing {path.name}")
            metrics = json.loads(metrics_path.read_text())
            done = json.loads(done_path.read_text())
            if metrics.get("status") != "PASS" or not done.get("ok"):
                raise RuntimeError("cell/DONE status is not PASS")
            if metrics.get("analysis_revision") != ANALYSIS_REVISION:
                raise RuntimeError("Pass 1 analysis revision drift")
            if metrics.get("transport_revision") != TRANSPORT_REVISION:
                raise RuntimeError("transport revision drift")
            if done.get("metrics_sha256") != sha256_file(metrics_path):
                raise RuntimeError("metrics content hash mismatch")
            if done.get("transport_sha256") != sha256_file(frame_path):
                raise RuntimeError("transport content hash mismatch")
            if any(parse_bool(value) for value in (
                item.target_values_read,
                metrics.get("target_values_read", False),
                done.get("target_values_read", False),
            )):
                raise RuntimeError("target value read before unsealing")
            if not metrics.get("model_hash_unchanged") or not metrics.get("decoder_hash_unchanged"):
                raise RuntimeError("frozen parameter hash changed")
            conditional = (
                SYSTEM / "per_cell" / str(item.fit_id) / str(item.public_arm)
                / f"seed{int(item.seed)}" / "conditional_manifold_arrays.npz"
            )
            if metrics.get("conditional_manifold_sha256") != sha256_file(conditional):
                raise RuntimeError("conditional manifold hash mismatch")
            frame = pd.read_csv(frame_path)
            expected_events = int(samples[
                samples["fit_id"].eq(item.fit_id)
                & samples["pass2_reference_event"].map(parse_bool)
            ]["event_array_index"].nunique())
            if int(metrics["n_reference_events"]) != expected_events:
                raise RuntimeError("reference-event denominator drift")
            if len(frame) != expected_events * len(PHASE_TARGETS):
                raise RuntimeError("reference-state denominator drift")
            if set(np.round(frame["phase_target"], 8)) != set(PHASE_TARGETS):
                raise RuntimeError("phase target drift")
            if not np.isfinite(frame[list(REQUIRED_FINITE)].to_numpy(float)).all():
                raise RuntimeError("nonfinite required transport endpoint")
            field_valid = frame["field_transport_cosine"].notna()
            if not np.isfinite(frame.loc[field_valid, [
                "field_transport_cosine", "field_gain", "field_gain_minus_normal"
            ]].to_numpy(float)).all():
                raise RuntimeError("nonfinite eligible future-field endpoint")
            if not frame["finite"].map(parse_bool).all():
                raise RuntimeError("finite flag false")
            if frame["target_values_read"].map(parse_bool).any():
                raise RuntimeError("per-state target value read")
            canonical = frame["canonical_ab"].map(parse_bool)
            if canonical.nunique() != 1:
                raise RuntimeError("canonical tier changes within cell")
            counters["canonical_cells" if canonical.iloc[0] else "within_fit_cells"] += 1
            counters["negative_PF_convergence_rows_retained"] += int(
                (frame["event_to_PF_manifold_convergence"] < 0).sum()
            )
            counters["audited_cells"] += 1
            counters["audited_reference_states"] += int(len(frame))
        except Exception as error:
            failures.append({"cell": key, "error_type": type(error).__name__, "error": str(error)})

    aggregate = pd.read_csv(TRANSPORT / "TRANSPORT_CELL_PHASE_SUMMARY.csv")
    aggregate_ok = bool(
        len(aggregate) == len(manifest) * len(PHASE_TARGETS)
        and aggregate[["fit_id", "public_arm", "seed", "phase_target"]].duplicated().sum() == 0
    )
    if not aggregate_ok:
        failures.append({"cell": "AGGREGATE", "error_type": "ContractError", "error": "aggregate denominator or uniqueness drift"})
    payload = {
        "contract": "topic5_latent_transport_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "transport_revision": TRANSPORT_REVISION,
        "status": "PASS" if not failures and counters["audited_cells"] == 630 else "FAIL",
        **counters,
        "aggregate_rows": int(len(aggregate)),
        "aggregate_ok": aggregate_ok,
        "failure_count": int(len(failures)),
        "failures_first20": failures[:20],
        "negative_metric_policy": "RETAINED_NOT_CLIPPED",
        "closed_loop_consistency": "PENDING_PASS2",
        "target_values_read": False,
    }
    atomic_write_json(TRANSPORT / "TRANSPORT_AUDIT.json", payload)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
