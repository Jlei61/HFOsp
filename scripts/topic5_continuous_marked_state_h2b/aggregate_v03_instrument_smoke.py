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
            if cell.get("revision") != "h2b_v0_3_interictal_instrument_cell_v3":
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
            "state_qualified": False,
            "qualification_status": "PENDING_Q6_AND_FINAL_PATIENT_RULE",
        })
    payload = {
        "status": "COMPLETE_SMOKE_DIAGNOSTIC",
        "revision": "h2b_v0_3_interictal_instrument_smoke_aggregate_v2",
        "supersedes_revision": "h2b_v0_3_interictal_instrument_smoke_aggregate_v1",
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
        "state_qualified_population_released": False,
        "remaining_evidence": "Q6 nuisance and final patient-level qualification rule",
        "claim_specific_policy": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    output = root / "qualification"
    atomic_json(output / "instrument_smoke_summary.json", payload)
    atomic_csv(output / "instrument_smoke_per_cell.csv", cell_rows)
    atomic_csv(output / "instrument_smoke_per_patient.csv", patient_rows)
    atomic_csv(output / "instrument_smoke_seed_pairs.csv", pair_rows)
    print(f"COMPLETE patients={len(patient_rows)} cells={len(cell_rows)}")


if __name__ == "__main__":
    main()
