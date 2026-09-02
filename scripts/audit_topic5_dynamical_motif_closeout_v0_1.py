#!/usr/bin/env python3
"""Machine closeout audit for the Topic 5.2 dynamical motif run.

Checks the five engineering stop conditions against the artefacts that actually
exist, counts the denominators every claim will quote, and refuses to call the
run complete while any of them is unmet.  Scientific nulls are never failures
here; only provenance, leakage, equivalence, decoder and numerical faults are.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check(name: str, ok: bool, detail: object) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--frame", default="GEOMETRY_ONLY_PCA2")
    args = parser.parse_args()
    root, checks, errors = args.out_root, [], []

    # 1 provenance and split identity
    split_audit = json.loads((root / "SPLIT_PROVENANCE_AUDIT.json").read_text())
    checks.append(check(
        "model_unseen_equals_parent_heldout",
        split_audit["all_model_unseen_equal_parent_heldout"],
        {"n_patients": split_audit["n_patients"],
         "total_model_unseen": split_audit["total_model_unseen"],
         "total_rank_ineligible": split_audit["total_rank_ineligible"]}))
    views = pd.read_csv(root / "PARENT_VIEW_CENSUS.csv")
    checks.append(check(
        "dual_views_share_one_event_set",
        bool(views[["identical_events_raw", "identical_events", "identical_contacts"]].all().all()),
        {"n_subjects": int(len(views))}))
    geometry = pd.read_csv(root / "GEOMETRY_ONLY_FIT_CENSUS.csv")
    checks.append(check(
        "geometry_frame_valid",
        bool(geometry.valid.all()) and len(geometry) == 28,
        {"n_fits": int(len(geometry)),
         "one_dimensional": geometry[geometry.geometry_class ==
                                     "DEGENERATE_ONE_DIMENSIONAL"].subject.tolist()}))

    # 2 leakage
    leak = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_topic5_dynamical_motif_rnn_v0_1.py",
         "-q", "-k", "carries_no_template_or_seizure_dependency or "
                     "direction_gate_is_zero_at_the_first_rank or "
                     "freeze_direction_scale_uses_only"],
        cwd=str(ROOT), capture_output=True, text=True)
    checks.append(check("no_template_future_or_seizure_leakage",
                        leak.returncode == 0, leak.stdout.strip().splitlines()[-1:]))

    # 3 nested zero-equivalence + 4 decoder/replay + 5 numerical health
    contract = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/test_topic5_dynamical_motif_rnn_v0_1.py", "-q"],
        cwd=str(ROOT), capture_output=True, text=True)
    checks.append(check("contract_test_suite", contract.returncode == 0,
                        contract.stdout.strip().splitlines()[-1:]))

    # unit census
    manifest = pd.read_csv(root / "FORMAL_UNIT_MANIFEST.csv")
    states, seconds, flags = [], [], []
    for _, row in manifest.iterrows():
        directory = (root / args.tag / str(row.frame) / str(row.unit_id)
                     / str(row.model_id) / f"seed{int(row.seed_index)}")
        done = directory / "DONE.json"
        if done.exists() and (directory / "checkpoint.pt").exists():
            states.append("DONE")
            payload = json.loads(done.read_text())
            seconds.append(float(payload.get("seconds", np.nan)))
            metrics = json.loads((directory / "metrics.json").read_text())
            flags.append({"time_limited": bool(metrics.get("time_limited")),
                          "nonfinite": not np.isfinite(
                              float(metrics.get("best_validation_score", np.nan)))})
        elif (directory / "FAILED.json").exists():
            states.append("FAILED")
            seconds.append(np.nan)
            flags.append({})
        else:
            states.append("PENDING")
            seconds.append(np.nan)
            flags.append({})
    manifest["state"] = states
    manifest["seconds"] = seconds
    done_count = int((manifest.state == "DONE").sum())
    checks.append(check("all_formal_units_terminal",
                        bool((manifest.state != "PENDING").all()),
                        manifest.state.value_counts().to_dict()))
    checks.append(check("no_nonfinite_unit",
                        not any(f.get("nonfinite") for f in flags if f),
                        {"n_nonfinite": sum(1 for f in flags if f.get("nonfinite"))}))

    # FINITE_NONRETURNING is a statement about the finite-horizon state gain of the
    # fitted dynamics, not about the weight matrix; read it from the measurement
    # that actually computes it rather than from a weight-norm proxy.
    horizon = {"measured_units": 0, "returns_after_peak": 0, "finite_nonreturning": 0}
    peaks = []
    for path in (root / args.tag / args.frame).glob("*/*/seed0/counterfactual_summary.json"):
        record = json.loads(path.read_text()).get("finite_horizon_gain") or {}
        if not record.get("per_horizon"):
            continue
        horizon["measured_units"] += 1
        horizon["returns_after_peak"] += int(bool(record.get("returns_after_peak")))
        horizon["finite_nonreturning"] += int(bool(record.get("finite_nonreturning")))
        if record.get("peak_horizon"):
            peaks.append(float(record["peak_horizon"]))
    horizon["median_peak_horizon"] = float(np.median(peaks)) if peaks else None

    evidence_path = root / "EVIDENCE_MATRIX.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
    payload = {
        "contract": "topic5_dynamical_motif_closeout_audit_v0_1",
        "tag": args.tag, "frame": args.frame,
        "checks": checks,
        "errors": [c["check"] for c in checks if not c["ok"]],
        "unit_census": {
            "planned": int(len(manifest)),
            "done": done_count,
            "failed": int((manifest.state == "FAILED").sum()),
            "pending": int((manifest.state == "PENDING").sum()),
            "time_limited": sum(1 for f in flags if f.get("time_limited")),
            "gpu_hours": float(np.nansum(manifest.seconds) / 3600.0),
            "median_unit_seconds": float(np.nanmedian(manifest.seconds))
            if done_count else None,
            "max_unit_seconds": float(np.nanmax(manifest.seconds)) if done_count else None,
        },
        "denominators": {
            "patients": int(split_audit["n_patients"]),
            "train_events": split_audit["total_train"],
            "calibration_events": split_audit["total_calibration"],
            "development_test_events": split_audit["total_development_test"],
            "model_unseen_events": split_audit["total_model_unseen"],
        },
        "finite_horizon_gain": horizon,
        "evidence_matrix_present": evidence_path.exists(),
        "goals_reported": sorted((evidence.get("goals") or {}).keys()),
        "verdict": "PASS" if not [c for c in checks if not c["ok"]] else "INCOMPLETE",
    }
    (root / "CLOSEOUT_AUDIT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({"verdict": payload["verdict"],
                      "errors": payload["errors"],
                      "unit_census": payload["unit_census"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
