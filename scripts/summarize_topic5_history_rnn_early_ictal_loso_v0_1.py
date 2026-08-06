#!/usr/bin/env python3
"""Patient-first G2/G3 summary for the strict early-ictal LOSO bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEVELOPMENT_TARGET = "epilepsiae_1146"


def _p(values) -> float:
    values = np.asarray(values, float)
    if not len(values) or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="greater", method="auto").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    rows = []
    pairing_rows = []
    for done_path in sorted(root.glob("*/DONE.json")):
        done = json.loads(done_path.read_text())
        if not bool(done.get("target_values_read", False)):
            raise RuntimeError(f"G2 fold did not record target unlock: {done_path}")
        subject = done["heldout_subject"]
        metrics = pd.read_csv(done_path.parent / "heldout_seizure_metrics.csv")
        pivot = metrics.pivot(index="seizure_id", columns="model", values="spearman_rho")
        rows.append(
            {
                "subject": subject,
                "n_seizures": int(len(pivot)),
                "rho_M0": float(pivot.M0.mean()),
                "rho_M1": float(pivot.M1.mean()),
                "rho_M2": float(pivot.M2.mean()),
                "rho_increment_M2_minus_M1": float((pivot.M2 - pivot.M1).mean()),
                "rho_increment_M1_minus_M0": float((pivot.M1 - pivot.M0).mean()),
            }
        )
        wrong_path = done_path.parent / "heldout_wrong_state_pairing.csv"
        wrong = pd.read_csv(wrong_path)
        if not wrong.empty:
            correct = pivot.M2.rename("correct_rho").reset_index()
            wrong_mean = wrong.groupby("seizure_id", as_index=False).wrong_pair_rho.mean()
            paired = correct.merge(wrong_mean, on="seizure_id", validate="one_to_one")
            pairing_rows.append(
                {
                    "subject": subject,
                    "n_states": int(len(paired)),
                    "correct_rho": float(paired.correct_rho.mean()),
                    "wrong_rho": float(paired.wrong_pair_rho.mean()),
                    "correct_minus_wrong": float(
                        (paired.correct_rho - paired.wrong_pair_rho).mean()
                    ),
                }
            )
    patient = pd.DataFrame(rows).sort_values("subject")
    if len(patient) != 16:
        raise RuntimeError(f"G2 incomplete: {len(patient)}/16")
    patient.to_csv(root / "g2_patient_metrics.csv", index=False)
    primary = patient.loc[patient.subject != DEVELOPMENT_TARGET]
    increment = primary.rho_increment_M2_minus_M1.to_numpy(float)
    g2_pass = float(np.median(increment)) > 0 and _p(increment) < 0.05
    pairing = pd.DataFrame(pairing_rows).sort_values("subject")
    pairing.to_csv(root / "g3_patient_pairing_metrics.csv", index=False)
    g0_subject = pd.read_csv(
        args.g0_root.resolve() / "subject_causal_history_inventory.csv"
    )
    expected_pairing = set(
        g0_subject.loc[g0_subject.g3_pairing_eligible, "subject"].astype(str)
    )
    if set(pairing.subject.astype(str)) != expected_pairing:
        raise RuntimeError("G3 pairing denominator drifted")
    pairing_effect = pairing.correct_minus_wrong.to_numpy(float)
    g3_pass = bool(
        g2_pass
        and float(np.median(pairing_effect)) > 0
        and _p(pairing_effect) < 0.05
    )
    result = {
        "status": (
            "G3_PASS_STATE_CONDITIONED_EARLY_ICTAL_FIELD"
            if g3_pass
            else (
                "G2_PASS_G3_FAIL_HISTORY_BRIDGE_NOT_SEIZURE_SPECIFIC"
                if g2_pass
                else "G2_FAIL_NO_INCREMENTAL_EARLY_ICTAL_BRIDGE"
            )
        ),
        "contract": "topic5_history_rnn_early_ictal_field_v0_1_g2_g3",
        "target_values_read": True,
        "g2": {
            "primary_cohort": "15 development-excluded strict clinical-onset patients",
            "n_primary_patients": int(len(primary)),
            "median_rho_increment_M2_minus_M1": float(np.median(increment)),
            "one_sided_wilcoxon_p": _p(increment),
            "n_positive": int(np.sum(increment > 0)),
            "all_16_supportive_median": float(
                patient.rho_increment_M2_minus_M1.median()
            ),
            "pass": bool(g2_pass),
        },
        "g3": {
            "status": "TESTED" if g2_pass else "LOCKED_NOT_INTERPRETED",
            "n_pairing_patients": int(len(pairing)),
            "median_correct_minus_wrong_rho": float(np.median(pairing_effect)),
            "one_sided_wilcoxon_p": _p(pairing_effect),
            "n_positive": int(np.sum(pairing_effect > 0)),
            "pass": bool(g3_pass),
        },
    }
    (root / "G2_G3_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
