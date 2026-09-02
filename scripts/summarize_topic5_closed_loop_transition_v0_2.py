#!/usr/bin/env python3
"""Summarize teacher-forced/closed-loop latent transition agreement for C2."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json  # noqa: E402
from scripts.run_topic5_closed_loop_transition_v0_2 import (  # noqa: E402
    TRANSITION, TRANSITION_REVISION, transition_dir,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402
from scripts.summarize_topic5_latent_geometry_v0_2 import one_sided_summary  # noqa: E402

REAL_ARMS = ("L0", "L1", "L2m", "L3")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    use = np.isfinite(left) & np.isfinite(right)
    if int(use.sum()) < 5 or np.ptp(left[use]) <= 1e-12 or np.ptp(right[use]) <= 1e-12:
        return float("nan")
    return float(spearmanr(left[use], right[use]).statistic)


def main() -> None:
    audit = json.loads((TRANSITION / "CLOSED_LOOP_TRANSITION_AUDIT.json").read_text())
    if audit.get("status") != "PASS": raise RuntimeError("closed-loop transition audit must pass")
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv")[["fit_id", "canonical_ab"]]
    manifest = manifest.merge(eligibility, on="fit_id", how="left", validate="many_to_one")
    if manifest["canonical_ab"].isna().any():
        raise RuntimeError("missing mode-axis eligibility")
    cell_rows: list[dict[str, object]] = []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        with np.load(transition_dir(row) / "transition.npz", allow_pickle=False) as source:
            tf = np.asarray(source["teacher_forced_delta_z"], float)
            cl = np.asarray(source["closed_loop_delta_z"], float)
            tf_distance = np.asarray(source["teacher_forced_delta_manifold_distance"], float)
            cl_distance = np.asarray(source["closed_loop_delta_manifold_distance"], float)
            valid = np.asarray(source["teacher_forced_valid"], bool) & np.asarray(source["closed_loop_valid"], bool)
        record: dict[str, object] = {
            "patient": item.patient, "fit_id": item.fit_id,
            "geometry_view": item.geometry_view, "public_arm": item.public_arm,
            "seed": int(item.seed), "canonical_ab": bool(item.canonical_ab),
            "n_joint_transitions": int(valid.sum()), "target_values_read": False,
        }
        for coordinate, index in (("progress", 0), ("field", 1)):
            record[f"{coordinate}_tf_cl_spearman"] = safe_spearman(tf[..., index][valid], cl[..., index][valid])
            denominator = np.linalg.norm(tf[..., index][valid]) * np.linalg.norm(cl[..., index][valid])
            record[f"{coordinate}_tf_cl_cosine"] = (
                float(np.dot(tf[..., index][valid], cl[..., index][valid]) / denominator)
                if denominator > 1e-12 else float("nan")
            )
        record["teacher_forced_manifold_convergence"] = float(-np.nanmedian(tf_distance[valid]))
        record["closed_loop_manifold_convergence"] = float(-np.nanmedian(cl_distance[valid]))
        cell_rows.append(record)
    cells = pd.DataFrame(cell_rows)
    metrics = [
        "progress_tf_cl_spearman", "field_tf_cl_spearman",
        "progress_tf_cl_cosine", "field_tf_cl_cosine",
        "teacher_forced_manifold_convergence", "closed_loop_manifold_convergence",
    ]
    patient_frames = []
    summaries: dict[str, object] = {}
    for canonical, tier in ((False, "generic_all_identifiable"), (True, "canonical_ab_shared")):
        selected = cells[cells["canonical_ab"]].copy() if canonical else cells.copy()
        seed = selected.groupby(
            ["patient", "fit_id", "geometry_view", "public_arm", "seed"], as_index=False
        )[metrics].median()
        fit = seed.groupby(["patient", "fit_id", "geometry_view", "public_arm"], as_index=False)[metrics].median()
        arm = fit.groupby(["patient", "public_arm"], as_index=False)[metrics].median()
        rows = []
        for patient, group in arm.groupby("patient"):
            indexed = group.set_index("public_arm")
            real = indexed.loc[list(REAL_ARMS), metrics].median(axis=0)
            rows.append({"tier": tier, "patient": patient, **{metric: float(real[metric]) for metric in metrics}})
        frame = pd.DataFrame(rows); patient_frames.append(frame)
        endpoints = {}
        for metric in metrics:
            endpoints[metric] = one_sided_summary(frame[metric].to_numpy(float), 520200 + len(endpoints))
        summaries[tier] = {
            "n_patients": int(frame.patient.nunique()), "endpoints": endpoints,
            "closed_loop_transition_role": "C2_TRAJECTORY_SENSITIVITY_NOT_A_RESCUE_GATE",
        }
    payload = {
        "contract": "topic5_closed_loop_transition_summary_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transition_revision": TRANSITION_REVISION, "status": "COMPLETE",
        "tiers": summaries,
        "interpretation": (
            "Positive teacher-forced/closed-loop agreement supports trajectory robustness. "
            "Positive convergence requires a decrease in distance to the train conditional manifold."
        ),
        "target_values_read": False,
    }
    atomic_write_csv(TRANSITION / "CLOSED_LOOP_TRANSITION_CELL_EFFECTS.csv", cells)
    atomic_write_csv(TRANSITION / "C2_CLOSED_LOOP_PATIENT_EFFECTS.csv", pd.concat(patient_frames, ignore_index=True))
    atomic_write_json(TRANSITION / "CLOSED_LOOP_TRANSITION_SUMMARY.json", payload)

    main_summary_path = OUT / "dynamical_transport" / "DYNAMICAL_TRANSPORT_SUMMARY.json"
    main_summary = json.loads(main_summary_path.read_text())
    for tier, values in main_summary["tiers"].items():
        values["closed_loop_transition"] = summaries[tier]
        values["C2_status"] = "UNSUPPORTED" if values["teacher_forced_C2_status"] != "SUPPORTED" else "SUPPORTED_WITH_CLOSED_LOOP_SENSITIVITY"
    main_summary["status"] = "DYNAMICAL_TRANSPORT_COMPLETE"
    main_summary["closed_loop_transition_revision"] = TRANSITION_REVISION
    main_summary["closed_loop_transition_audit"] = "PASS"
    atomic_write_json(main_summary_path, main_summary)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
