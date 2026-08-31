#!/usr/bin/env python3
"""Aggregate cross-seed A1 diagnostics without issuing qualification yet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.v03_seed_stability import (  # noqa: E402
    cross_seed_stability,
    load_cell_manifest,
    load_trace,
)
from src.topic5_continuous_marked_state_h2b.v03_exploration_policy import (  # noqa: E402
    assert_frozen_exploration_policy_matches,
)


PRODUCER_SCRIPT = Path(__file__).resolve()
SEED_STABILITY_MODULE = (
    REPO / "src/topic5_continuous_marked_state_h2b/v03_seed_stability.py"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--max-anchors", type=int, default=256)
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()
    root = args.result_root.resolve()
    policy_path = root / "exploration_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert_frozen_exploration_policy_matches(policy)
    patient_rows = []
    pair_rows = []
    cell_rows = []
    source_hashes = {}
    for subject in map(str, args.subjects):
        manifests = sorted((root / "instrument/by_cell" / subject).glob(
            "seed_*/instrument_manifest.json"
        ))
        if len(manifests) < 1:
            raise FileNotFoundError(f"no instrument cells for {subject}")
        cells = [load_cell_manifest(path) for path in manifests]
        for path, cell in zip(manifests, cells):
            if cell.get("status") != "COMPLETE" or cell.get("seizure_risk_outcome_read") is not False:
                raise ValueError(f"inadmissible instrument cell: {path}")
            if cell.get("revision") != "h2b_v0_3_interictal_instrument_cell_v4":
                raise ValueError(f"superseded instrument cell: {path}")
            source_hashes[str(path)] = sha256_file(path)
            diagnostics = cell["diagnostics"]
            cell_rows.append({
                "subject": subject,
                "seed": int(cell["seed"]),
                "effective_rank": diagnostics["Q1_noncollapse"]["effective_rank"],
                "top_pc_share": diagnostics["Q1_noncollapse"]["top_pc_share"],
                "persistent_minus_memoryless_joint_nll": diagnostics[
                    "Q2_cross_window_information"
                ]["persistent_minus_memoryless_joint_nll_per_event"],
                "generator_fraction_of_decoder_motion": diagnostics[
                    "Q3_generator_contribution"
                ]["generator_fraction_of_decoder_motion"],
                "analytic_tau_minutes": diagnostics["Q4_time_constant"][
                    "analytic_generator_slowest_mode_minutes"
                ],
                "empirical_tau_minutes": diagnostics["Q4_time_constant"][
                    "empirical_decoder_tau_minutes"
                ],
                "empirical_tau_right_censored": diagnostics["Q4_time_constant"][
                    "empirical_tau_right_censored"
                ],
                "preliminary_Q1_pass": diagnostics["Q1_noncollapse"][
                    "null_calibrated_pass"
                ],
                "Q2_direction_favourable": diagnostics[
                    "Q2_cross_window_information"
                ]["direction_favourable"],
                "preliminary_Q3_pass": diagnostics["Q3_generator_contribution"][
                    "open_loop_reset_pass"
                ],
                "open_loop_predictive_horizon_minutes": diagnostics[
                    "Q3_generator_contribution"
                ]["open_loop_interictal_prediction"]["predictive_horizon_minutes"],
                "preliminary_Q4_identifiable": diagnostics["Q4_time_constant"][
                    "preliminary_absolute_threshold_pass"
                ],
                "core_Q1_Q4_seed_pass": bool(
                    diagnostics["Q1_noncollapse"]["null_calibrated_pass"]
                    and diagnostics["Q2_cross_window_information"][
                        "direction_favourable"
                    ]
                    and diagnostics["Q3_generator_contribution"][
                        "open_loop_reset_pass"
                    ]
                    and diagnostics["Q4_time_constant"][
                        "preliminary_absolute_threshold_pass"
                    ]
                ),
                "Q6_status": diagnostics["Q6_not_only_clock"]["status"],
                "Q6_median_relative_improvement": diagnostics[
                    "Q6_not_only_clock"
                ].get("median_relative_improvement"),
                "Q6_pass": diagnostics["Q6_not_only_clock"].get("pass", False),
                "scalar_axis_noncollapse": bool(
                    int(diagnostics.get("active_decoder_dimensions", 0)) >= 1
                    and float(diagnostics["Q1_noncollapse"].get("effective_rank") or 0.0)
                    >= 1.0
                    and float(diagnostics["Q1_noncollapse"].get(
                        "median_persistent_memoryless_decoder_distance"
                    ) or 0.0) > 1e-6
                    and diagnostics["Q1_noncollapse"]["temporal_shuffled_null"].get(
                        "temporally_smoother_than_shuffled", False
                    )
                    and diagnostics["Q1_noncollapse"].get(
                        "reset_not_dominant_or_not_estimable", False
                    )
                ),
            })
        seeds = [int(cell["seed"]) for cell in cells]
        traces = [load_trace(cell["trace_path"]) for cell in cells]
        stability = cross_seed_stability(
            subject, seeds, traces,
            max_anchors=int(args.max_anchors),
            n_permutations=int(args.n_permutations),
        )
        for row in stability["pair_rows"]:
            pair_rows.append({"subject": subject, **row})
        patient_cells = [row for row in cell_rows if row["subject"] == subject]
        n_strict_without_q6 = sum(bool(row["core_Q1_Q4_seed_pass"])
                                  for row in patient_cells)
        n_strict_with_q6 = sum(bool(
            row["core_Q1_Q4_seed_pass"] and row["Q6_pass"]
        ) for row in patient_cells)
        n_filter = sum(bool(
            row["preliminary_Q1_pass"] and row["Q2_direction_favourable"]
            and row["preliminary_Q4_identifiable"] and row["Q6_pass"]
        ) for row in patient_cells)
        n_scalar = sum(bool(
            row["scalar_axis_noncollapse"] and row["Q2_direction_favourable"]
            and row["preliminary_Q4_identifiable"]
        ) for row in patient_cells)
        q5 = bool(stability["preliminary_Q5_pass"])
        state_qualified = bool(q5 and n_strict_with_q6 >= 3)
        if state_qualified:
            stratum = "clock_residual_persistent_state_candidate"
        elif q5 and n_strict_without_q6 >= 3:
            stratum = "persistent_generator_candidate_q6_weak_or_unavailable"
        elif q5 and n_filter >= 3:
            stratum = "observation_filter_candidate"
        elif n_scalar >= 3:
            stratum = "scalar_slow_axis_candidate"
        else:
            stratum = "collapsed_or_unusable_for_persistent_claim"
        patient_rows.append({
            "subject": subject,
            "n_seeds": len(seeds),
            "n_Q1_preliminary_pass": sum(row["preliminary_Q1_pass"] for row in cell_rows
                                          if row["subject"] == subject),
            "n_Q2_direction_favourable": sum(row["Q2_direction_favourable"] for row in cell_rows
                                               if row["subject"] == subject),
            "n_Q3_preliminary_pass": sum(row["preliminary_Q3_pass"] for row in cell_rows
                                          if row["subject"] == subject),
            "n_Q4_identifiable": sum(row["preliminary_Q4_identifiable"] for row in cell_rows
                                      if row["subject"] == subject),
            "n_core_Q1_Q4_seed_pass": sum(row["core_Q1_Q4_seed_pass"] for row in cell_rows
                                            if row["subject"] == subject),
            "median_decoder_distance_correlation": stability[
                "median_decoder_distance_correlation"
            ],
            "median_decoder_linear_cka": stability["median_decoder_linear_cka"],
            "median_latent_procrustes_similarity": stability[
                "median_latent_procrustes_similarity"
            ],
            "fraction_seed_pairs_above_null": stability[
                "fraction_pairs_above_seed_permuted_null"
            ],
            "preliminary_Q5_pass": stability["preliminary_Q5_pass"],
            "core_Q1_Q5_pass": bool(
                n_strict_without_q6 >= 3 and q5
            ),
            "n_Q6_pass": sum(row["Q6_pass"] for row in patient_cells),
            "n_joint_Q1_to_Q6_pass": int(n_strict_with_q6),
            "n_scalar_axis_candidate": int(n_scalar),
            "state_qualified": state_qualified,
            "qualification_status": (
                "STATE_QUALIFIED_DEVELOPMENT" if state_qualified
                else "NOT_STATE_QUALIFIED_DEVELOPMENT"
            ),
            "exploration_stratum": stratum,
        })
    payload = {
        "status": "COMPLETE_GRADED_DIAGNOSTIC",
        "revision": "h2b_v0_3_interictal_instrument_aggregate_v3",
        "supersedes_revision": "h2b_v0_3_interictal_instrument_smoke_aggregate_v2",
        "created_utc": utc_now(),
        "subjects": list(map(str, args.subjects)),
        "n_cells": len(cell_rows),
        "n_patients": len(patient_rows),
        "patient_rows": patient_rows,
        "source_manifest_sha256": source_hashes,
        "producer_script_sha256": sha256_file(PRODUCER_SCRIPT),
        "seed_stability_module_sha256": sha256_file(SEED_STABILITY_MODULE),
        "exploration_policy_receipt_sha256": sha256_file(policy_path),
        "seizure_risk_outcome_read": False,
        "state_qualified_population_released": True,
        "remaining_evidence": None,
        "grading_rule": (
            "at least three seeds jointly pass Q1,Q2,Q3,Q4,Q6 and patient Q5; "
            "weaker scalar/filter/generator strata remain available for exploration"
        ),
        "claim_specific_policy": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    output = root / "qualification"
    atomic_json(output / "instrument_smoke_summary.json", payload)
    atomic_json(output / "all_frozen_manifest.json", {
        "status": "COMPLETE", "created_utc": payload["created_utc"],
        "population": "all_frozen", "n_patients": len(patient_rows),
        "n_cells": len(cell_rows), "patients": patient_rows,
        "source_summary_revision": payload["revision"],
        "seizure_risk_outcome_read": False,
    })
    qualified_subjects = [row["subject"] for row in patient_rows
                          if row["state_qualified"]]
    atomic_json(output / "state_qualified_manifest.json", {
        "status": "COMPLETE", "created_utc": payload["created_utc"],
        "population": "state_qualified", "n_patients": len(qualified_subjects),
        "subjects": qualified_subjects,
        "patients": [row for row in patient_rows if row["state_qualified"]],
        "source_summary_revision": payload["revision"],
        "qualification_is_claim_specific_not_global_gate": True,
        "seizure_risk_outcome_read": False,
    })
    atomic_csv(output / "instrument_smoke_per_cell.csv", cell_rows)
    atomic_csv(output / "instrument_smoke_per_patient.csv", patient_rows)
    atomic_csv(output / "instrument_smoke_seed_pairs.csv", pair_rows)
    print(f"COMPLETE patients={len(patient_rows)} cells={len(cell_rows)}")


if __name__ == "__main__":
    main()
