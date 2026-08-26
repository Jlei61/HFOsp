#!/usr/bin/env python3
"""Aggregate Bridge runs without pooling events across patients."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from src.topic5_continuous_marked_state import contract


def main() -> None:
    rows = []
    for path in sorted((contract.RESULT_ROOT / "bridge/runs").glob("*.json")):
        row = json.loads(path.read_text())
        if (row.get("contract") != contract.REVISION
                or row.get("fit_revision") != contract.FIT_REVISION):
            continue
        rows.append(row)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["arm"])].append(row)
    per_subject = []
    for subject in contract.PILOT_SUBJECTS:
        arm_values = {}
        for arm in ("b0_history", "b1_spectral", "b2_raw", "b3_both"):
            found = grouped.get((subject, arm), [])
            if not found:
                continue
            arm_values[arm] = {
                key: float(np.median([r["validation"][key] for r in found]))
                for key in ("joint_nll", "timing_nll", "mark_nll",
                            "participation_nll", "rank_nll", "stop_nll")
            }
            arm_values[arm]["n_fits"] = len(found)
            arm_values[arm]["fit_metric_range"] = {
                key: float(np.ptp([r["validation"][key] for r in found]))
                for key in ("joint_nll", "timing_nll", "mark_nll")
            }
        if "b0_history" in arm_values:
            base = arm_values["b0_history"]
            base_rows = {int(r["seed"]): r for r in grouped[(subject, "b0_history")]}
            for arm, values in arm_values.items():
                values["delta_joint_vs_b0"] = values["joint_nll"] - base["joint_nll"]
                values["delta_timing_vs_b0"] = values["timing_nll"] - base["timing_nll"]
                values["delta_mark_vs_b0"] = values["mark_nll"] - base["mark_nll"]
                paired = [r for r in grouped.get((subject, arm), []) if int(r["seed"]) in base_rows]
                values["n_favourable_fit_replicates_joint_vs_b0"] = int(sum(
                    r["validation"]["joint_nll"]
                    < base_rows[int(r["seed"])]["validation"]["joint_nll"]
                    for r in paired
                ))
                values["n_favourable_fit_replicates_mark_vs_b0"] = int(sum(
                    r["validation"]["mark_nll"]
                    < base_rows[int(r["seed"])]["validation"]["mark_nll"]
                    for r in paired
                ))
                values["n_favourable_fit_replicates_timing_vs_b0"] = int(sum(
                    r["validation"]["timing_nll"]
                    < base_rows[int(r["seed"])]["validation"]["timing_nll"]
                    for r in paired
                ))
        complete_arms = all(a in arm_values for a in (
            "b0_history", "b1_spectral", "b2_raw", "b3_both"
        ))
        if complete_arms:
            b0, b1, b2, b3 = (arm_values[a]["joint_nll"] for a in
                               ("b0_history", "b1_spectral", "b2_raw", "b3_both"))
            tol = 1e-3
            spectral = b1 < b0 - tol
            raw = b2 < b0 - tol
            combined = b3 < b0 - tol
            raw_beyond = combined and b3 < b1 - tol
            spectral_beyond_raw = combined and b3 < b2 - tol
            if combined and raw_beyond and spectral_beyond_raw and (spectral or raw):
                classification = "complementary"
            elif combined and raw_beyond and spectral_beyond_raw:
                classification = "combined_only_weak"
            elif combined and raw_beyond:
                classification = "raw_beyond_spectral"
            elif spectral:
                classification = "spectral_only"
            elif raw:
                classification = "raw_only_E0"
            else:
                classification = "no_increment"
        else:
            classification = "incomplete"
        per_subject.append({"subject": subject, "classification": classification,
                            "arms": arm_values})
    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "n_runs": len(rows),
        "n_subjects_any": sum(bool(x["arms"]) for x in per_subject),
        "n_subjects_complete": sum(x["classification"] != "incomplete" for x in per_subject),
        "per_subject": per_subject,
        "sealed_opened": False,
        "fit_replicate_note": (
            "The zero-initialised full-batch convex LBFGS fit is deterministic. "
            "Repeated seed labels are numerical reproducibility checks and are "
            "not independent evidence; the patient is the scientific unit."
        ),
        "claim_boundary": "patient-level Bridge-E0; raw-derived fixed features are not a final raw encoder",
    }
    path = contract.RESULT_ROOT / "bridge/BRIDGE_E0_SUMMARY.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps({k: output[k] for k in ("n_runs", "n_subjects_any", "n_subjects_complete")}))


if __name__ == "__main__":
    main()
