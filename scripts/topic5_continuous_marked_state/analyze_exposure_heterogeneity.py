#!/usr/bin/env python3
"""Describe, but do not select, patient-specific exposure time scales."""
from __future__ import annotations

from collections import Counter
import json
import os

import numpy as np
from scipy.stats import spearmanr

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION


TAUS = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 360.0)
KINDS = ("load", "participation")
REVISION = "h3_s0_patient_time_scale_heterogeneity_v1"


def main() -> None:
    root = contract.RESULT_ROOT / "exposure_screen"
    all_rows = []
    for path in sorted(root.glob("*__tau*m.json")):
        row = json.loads(path.read_text())
        if (
            row.get("contract") == contract.REVISION
            and row.get("fit_revision") == contract.FIT_REVISION
            and row.get("exposure_revision") == EXPOSURE_REVISION
            and row.get("sealed_opened") is False
        ):
            all_rows.append(row)
    rows = [
        row for row in all_rows
        if any(np.isclose(float(row["tau_minutes"]), tau) for tau in TAUS)
    ]
    subjects = sorted({row["subject"] for row in rows})
    expected = len(subjects) * len(KINDS) * len(TAUS)
    if len(subjects) != 34 or len(rows) != expected:
        raise ValueError(
            f"incomplete exposure grid: {len(subjects)} subjects, "
            f"{len(rows)}/{expected} runs"
        )
    indexed = {
        (row["subject"], row.get("exposure_kind", "load"), float(row["tau_minutes"])): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("duplicate subject/kind/tau exposure cells")

    per_patient = []
    for subject in subjects:
        first = indexed[(subject, "load", TAUS[0])]
        patient = {
            "subject": subject,
            "dataset": subject.split("_", 1)[0],
            "n_validation": int(first["n_validation"]),
            "support_group": (
                "n_validation_ge_1000" if int(first["n_validation"]) >= 1000
                else "n_validation_lt_1000"
            ),
            "by_exposure_kind": {},
        }
        for kind in KINDS:
            delta_placebo = {
                tau: float(indexed[(subject, kind, tau)]["contrasts"]["mark_nll"]
                           ["real_minus_placebo"])
                for tau in TAUS
            }
            delta_history = {
                tau: float(indexed[(subject, kind, tau)]["contrasts"]["mark_nll"]
                           ["real_minus_history"])
                for tau in TAUS
            }
            best_tau = min(TAUS, key=lambda tau: delta_placebo[tau])
            favourable = [
                tau for tau in TAUS
                if delta_placebo[tau] < 0 and delta_history[tau] < 0
            ]
            patient["by_exposure_kind"][kind] = {
                "best_tau_by_mark_real_minus_placebo": best_tau,
                "best_mark_real_minus_placebo": delta_placebo[best_tau],
                "best_tau_also_beats_history": delta_history[best_tau] < 0,
                "favourable_taus_both_history_and_placebo": favourable,
                "n_favourable_taus": len(favourable),
            }
        per_patient.append(patient)

    cohort = {}
    for kind in KINDS:
        cohort[kind] = {}
        for group in ("all", "n_validation_ge_1000", "n_validation_lt_1000"):
            found = [
                row for row in per_patient
                if group == "all" or row["support_group"] == group
            ]
            best = [
                row["by_exposure_kind"][kind]
                ["best_tau_by_mark_real_minus_placebo"] for row in found
            ]
            widths = [
                row["by_exposure_kind"][kind]["n_favourable_taus"]
                for row in found
            ]
            cohort[kind][group] = {
                "n_patients": len(found),
                "best_tau_counts": {
                    str(int(tau)): int(count)
                    for tau, count in sorted(Counter(best).items())
                },
                "median_n_favourable_taus": float(np.median(widths)),
                "n_patients_no_favourable_tau": int(np.sum(np.asarray(widths) == 0)),
            }

    cross_kind = {}
    for tau in TAUS:
        load = [
            indexed[(subject, "load", tau)]["contrasts"]["mark_nll"]
            ["real_minus_placebo"] for subject in subjects
        ]
        participation = [
            indexed[(subject, "participation", tau)]["contrasts"]["mark_nll"]
            ["real_minus_placebo"] for subject in subjects
        ]
        result = spearmanr(load, participation)
        cross_kind[str(int(tau))] = {
            "spearman_rho": float(result.statistic),
            "two_sided_p_descriptive": float(result.pvalue),
        }

    leave_one_patient = {}
    for kind in KINDS:
        leave_one_patient[kind] = {}
        for tau in TAUS:
            leave_one_patient[kind][str(int(tau))] = {}
            for endpoint in (
                "mark_nll", "participation_nll", "rank_nll", "stop_nll"
            ):
                values = np.asarray([
                    indexed[(subject, kind, tau)]["contrasts"][endpoint]
                    ["real_minus_placebo"] for subject in subjects
                ], dtype=float)
                loo = np.asarray([
                    np.median(np.delete(values, index))
                    for index in range(len(values))
                ])
                leave_one_patient[kind][str(int(tau))][endpoint] = {
                    "full_median": float(np.median(values)),
                    "leave_one_patient_median_min": float(loo.min()),
                    "leave_one_patient_median_max": float(loo.max()),
                }

    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "exposure_revision": EXPOSURE_REVISION,
        "heterogeneity_revision": REVISION,
        "n_source_runs": len(all_rows),
        "n_runs": len(rows),
        "n_patients": len(subjects),
        "taus_minutes": list(TAUS),
        "per_patient": per_patient,
        "cohort": cohort,
        "cross_kind_mark_delta_correlation": cross_kind,
        "leave_one_patient_median_ranges": leave_one_patient,
        "sealed_opened": False,
        "claim_boundary": (
            "Post-hoc descriptive heterogeneity analysis. Best tau is a noisy "
            "within-patient minimum and must not be treated as an identified "
            "patient-specific physiological constant or used for model selection."
        ),
    }
    path = root / "TIME_SCALE_HETEROGENEITY.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({
        "path": str(path),
        "n_runs": len(rows),
        "n_patients": len(subjects),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
