#!/usr/bin/env python3
"""Engineering and leakage acceptance for the completed v0.4 formal run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_COUNTS = {
    "M1_M3_common_frozen_recurrent": 30,
    "M1_frozen_recurrent_continuation": 30,
    "M3_joint": 30,
    "M2_time_aware_nonrecurrent": 60,
}
TRUE_MODELS = {
    "M0_STATIC_AB",
    "M1_FROZEN_HISTORY_HEAD",
    "M2_TIME_AWARE_NONRECURRENT",
    "M3_JOINT_RNN",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "INPUT_MANIFEST.json").read_text())
    subjects = list(manifest["cohort"]["primary_subjects"])
    seeds = [11, 29, 47]
    rows = []
    violations = []
    # The half-life column is intentionally absent for the head-only stages, so a
    # bare ``NaN`` there is a json.dumps artefact, not a numerical failure.  The
    # auditor only counts it -- it must never rewrite the artefacts it audits,
    # otherwise the "no NaN" check becomes self-fulfilling and unreproducible.
    inapplicable_half_life_markers = 0
    for progress_log in sorted((root / "logs").glob("train_epilepsiae_*_seed*.log")):
        text = progress_log.read_text()
        marker = '"history_half_life_hours": NaN'
        inapplicable_half_life_markers += text.count(marker)
        if "NaN" in text.replace(marker, ""):
            violations.append(f"unexpected NaN in progress log: {progress_log.name}")
    m0_by_subject: dict[str, pd.DataFrame] = {}
    for seed in seeds:
        for subject in subjects:
            directory = root / "per_subject" / f"seed_{seed}" / subject
            done_path = directory / "DONE.json"
            failed_path = directory / "FAILED.json"
            if failed_path.exists():
                violations.append(f"failed unit: seed={seed}, subject={subject}")
            if not done_path.exists():
                violations.append(f"missing DONE: seed={seed}, subject={subject}")
                continue
            done = json.loads(done_path.read_text())
            cache_index = json.loads(
                (root / "cache" / f"outer_{subject}" / "INDEX.json").read_text()
            )
            if done.get("heldout_target_used_for_training") is not False:
                violations.append(f"heldout target visible: seed={seed}, subject={subject}")
            if done.get("n_train_patients") != 14:
                violations.append(f"wrong outer train denominator: seed={seed}, subject={subject}")
            if done.get("outer_fold_encoder_sha256") != cache_index["outer_fold_shared_encoder"]["event_checkpoint_sha256"]:
                violations.append(f"encoder hash mismatch: seed={seed}, subject={subject}")
            if len({entry["encoder_checkpoint_sha256"] for entry in cache_index["entries"]}) != 1:
                violations.append(f"mixed encoder coordinates: outer={subject}")
            training = pd.read_csv(directory / "training_log.csv")
            counts = training.stage.value_counts().to_dict()
            if counts != EXPECTED_STAGE_COUNTS:
                violations.append(
                    f"epoch/stage count mismatch: seed={seed}, subject={subject}, {counts}"
                )
            numeric_columns = ["loss", "soft_maxab", "gradient_norm", "gain_a", "gain_b"]
            if not np.all(np.isfinite(training[numeric_columns].to_numpy(float))):
                violations.append(f"non-finite training metric: seed={seed}, subject={subject}")
            predictions = pd.read_csv(directory / "heldout_candidate_predictions.csv.gz")
            diagnostics_path = directory / "heldout_residual_diagnostics.csv.gz"
            if not diagnostics_path.exists():
                violations.append(f"missing state/residual diagnostics: seed={seed}, subject={subject}")
            prediction_numeric = predictions[
                ["prediction_a", "prediction_b", "target_1_45", "target_1_150"]
            ].to_numpy(float)
            if not np.all(np.isfinite(prediction_numeric)):
                violations.append(f"non-finite heldout prediction: seed={seed}, subject={subject}")
            true_models = set(predictions.loc[predictions.draw == -1, "model"])
            if not TRUE_MODELS.issubset(true_models):
                violations.append(f"missing true model output: seed={seed}, subject={subject}")
            order = predictions.loc[predictions.model == "M3_ORDER_SHUFFLE_FULL_HISTORY"]
            if order.draw.nunique() != 32:
                violations.append(f"order shuffle not 32 draws: seed={seed}, subject={subject}")
            n_seizures = int(done["n_heldout_seizures"])
            swap = predictions.loc[predictions.model == "M3_WITHIN_PATIENT_HISTORY_SWAP"]
            expected_swap_donors = max(n_seizures - 1, 0)
            if n_seizures >= 2 and (
                swap.empty
                or swap.groupby("seizure_id").donor_seizure_id.nunique().min()
                != expected_swap_donors
            ):
                violations.append(f"history swap denominator mismatch: seed={seed}, subject={subject}")
            deviation = done["initial_output_deviation_from_static"]
            if deviation["max_l2_difference"] > 0.01 or deviation["max_angle_degrees"] > 1.0:
                violations.append(f"initial residual not near static: seed={seed}, subject={subject}")
            gains = np.asarray(list(done["final_gains"].values()), dtype=float)
            if not np.all(np.isfinite(gains)) or np.any(gains < 0) or np.any(gains > 1):
                violations.append(f"gain outside [0,1]: seed={seed}, subject={subject}")
            if not math.isfinite(float(done["final_m3_half_life_hours"])):
                violations.append(f"non-finite M3 half-life: seed={seed}, subject={subject}")
            m0 = (
                predictions.loc[(predictions.draw == -1) & (predictions.model == "M0_STATIC_AB")]
                .sort_values(["seizure_id", "contact"])
                .reset_index(drop=True)
            )
            compare_columns = [
                "subject", "seizure_id", "contact", "prediction_a", "prediction_b",
                "target_1_45", "target_1_150",
            ]
            if subject in m0_by_subject:
                try:
                    pd.testing.assert_frame_equal(
                        m0_by_subject[subject][compare_columns],
                        m0[compare_columns],
                        check_exact=True,
                    )
                except AssertionError:
                    violations.append(f"M0 or targets differ across seeds: subject={subject}")
            else:
                m0_by_subject[subject] = m0
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "n_train_patients": done["n_train_patients"],
                    "n_train_seizures": done["n_train_seizures"],
                    "n_heldout_seizures": n_seizures,
                    "training_rows": len(training),
                    "order_shuffle_draws": int(order.draw.nunique()),
                    "peak_gpu_memory_mb": done["peak_gpu_memory_mb"],
                    "elapsed_seconds": done["elapsed_seconds"],
                    "final_m3_half_life_hours": done["final_m3_half_life_hours"],
                    "max_initial_angle_degrees": deviation["max_angle_degrees"],
                }
            )
    table = pd.DataFrame(rows)
    acceptance = {
        "status": "ACCEPTED" if not violations and len(table) == 45 else "REJECTED",
        "contract": "topic5_history_conditioned_field_refinement_v0_4",
        "scope": "engineering_correctness_leakage_numerical_only_not_scientific_effect",
        "formal_units_complete": int(len(table)),
        "formal_units_expected": 45,
        "failed_units": len(list((root / "per_subject").glob("seed_*/epilepsiae_*/FAILED.json"))),
        "violations": violations,
        "checks": {
            "same_outer_encoder_coordinate": not any("encoder" in item for item in violations),
            "heldout_target_sealed": not any("target visible" in item for item in violations),
            "fixed_training_budget": not any("epoch/stage" in item for item in violations),
            "finite_training_and_prediction_metrics": not any("non-finite" in item for item in violations),
            "three_seed_static_and_target_identity": not any("differ across seeds" in item for item in violations),
            "full_history_shuffle_and_swap_denominators": not any(
                "shuffle" in item or "swap" in item for item in violations
            ),
            "initial_static_anchor": not any("initial residual" in item for item in violations),
            "state_and_raw_residual_diagnostics": not any(
                "state/residual diagnostics" in item for item in violations
            ),
        },
        "resource_summary": {
            "peak_gpu_memory_mb_max": float(table.peak_gpu_memory_mb.max()) if len(table) else None,
            "elapsed_seconds_median": float(table.elapsed_seconds.median()) if len(table) else None,
            "elapsed_seconds_total_units": float(table.elapsed_seconds.sum()) if len(table) else None,
        },
        "inapplicable_half_life_markers_counted_not_rewritten": inapplicable_half_life_markers,
        "note": "Scientific effect sizes are not acceptance gates and are evaluated only in the formal summary.",
    }
    table.to_csv(root / "ACCEPTANCE_UNIT_TABLE.csv", index=False)
    (root / "ACCEPTANCE.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if acceptance["status"] != "ACCEPTED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
