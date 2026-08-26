#!/usr/bin/env python3
"""Ask whether human clock contrasts emerge where the clocks are separable."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import binomtest, spearmanr

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.multiplicity import annotate_family


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    nonzero = array[array != 0]
    return {
        "median_physical_minus_event_count": float(np.median(array)),
        "n_physical_better": int(np.sum(array < 0)),
        "n_patients": int(len(array)),
        "two_sided_exact_sign_p_unadjusted": (
            float(binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5).pvalue)
            if len(nonzero) else None
        ),
    }


def main() -> None:
    root = contract.RESULT_ROOT / "exposure_clock_control"
    human = json.loads((root / "PHYSICAL_VS_EVENT_COUNT_CLOCK.json").read_text())
    synthetic = json.loads((root / "CLOCK_IDENTIFIABILITY_SYNTHETIC.json").read_text())
    correlations = {
        row["subject"]: float(row["validation_exposure_correlation"])
        for row in synthetic["per_subject"]
    }
    ordered = sorted(correlations, key=correlations.get)
    lower_correlation = set(ordered[:len(ordered) // 2])
    higher_correlation = set(ordered[len(ordered) // 2:])
    threshold = float((correlations[ordered[16]] + correlations[ordered[17]]) / 2)

    rows = []
    for cell in human["cells"]:
        for endpoint in ("joint_nll", "mark_nll", "stop_nll", "rank_nll"):
            patient = cell["endpoints"][endpoint]["patient_values"]
            all_values = {
                subject: float(row["physical_minus_count_delta_vs_placebo"])
                for subject, row in patient.items()
            }
            rho, _ = spearmanr(
                [all_values[subject] for subject in ordered],
                [correlations[subject] for subject in ordered],
            )
            rows.append({
                "exposure_kind": cell["exposure_kind"],
                "tau_minutes": cell["tau_minutes"],
                "endpoint": endpoint,
                "spearman_rho_delta_with_clock_correlation": float(rho),
                "more_separable_lower_correlation": _summary([
                    all_values[subject] for subject in ordered
                    if subject in lower_correlation
                ]),
                "less_separable_higher_correlation": _summary([
                    all_values[subject] for subject in ordered
                    if subject in higher_correlation
                ]),
            })

    recovery = []
    for truth in ("physical_time", "event_count"):
        values = {}
        for subject in synthetic["per_subject"]:
            row = next(
                row for row in subject["rows"]
                if row["truth"] == truth and row["beta"] == 0.1
            )
            values[subject["subject"]] = float(
                row["median_physical_minus_event_count_mse"]
            )
        recovery.append({
            "truth": truth,
            "beta": 0.1,
            "more_separable_lower_correlation": _summary([
                values[subject] for subject in ordered
                if subject in lower_correlation
            ]),
            "less_separable_higher_correlation": _summary([
                values[subject] for subject in ordered
                if subject in higher_correlation
            ]),
        })

    strata_family = []
    for row in rows:
        for name in ("more_separable_lower_correlation",
                     "less_separable_higher_correlation"):
            summary = row.get(name)
            if isinstance(summary, dict) and \
                    "two_sided_exact_sign_p_unadjusted" in summary:
                strata_family.append((
                    (row["exposure_kind"], str(row["tau_minutes"]),
                     row["endpoint"], name),
                    summary,
                ))
    strata_multiplicity = annotate_family(
        strata_family, family_name="clock_separability_strata")

    output = {
        "multiplicity": strata_multiplicity,
        "contract": contract.REVISION,
        "analysis_revision": "clock_separability_patient_strata_v1",
        "clock_correlation_threshold": threshold,
        "n_more_separable": len(lower_correlation),
        "n_less_separable": len(higher_correlation),
        "more_separable_subjects": sorted(lower_correlation),
        "less_separable_subjects": sorted(higher_correlation),
        "human_rows": rows,
        "synthetic_recovery_beta_0_1": recovery,
        "sealed_opened": False,
        "claim_boundary": (
            "Post-hoc observability sensitivity. It checks whether a physical-clock "
            "signal is concentrated where the two clocks are empirically separable; "
            "it is not a patient-selection rule."
        ),
    }
    path = root / "CLOCK_SEPARABILITY_STRATA.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({"path": str(path), "n_rows": len(rows)}))


if __name__ == "__main__":
    main()
