#!/usr/bin/env python3
"""Apply the predeclared v0.5 claim hierarchy after locked scoring.

This script never trains, scores a target, or selects an endpoint.  It maps
already frozen patient-level summaries to bounded machine-readable claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
PREFREEZE = "FINAL_CLAIM_ADJUDICATOR_PREFREEZE_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def freeze_contract(out: Path) -> None:
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("claim adjudication must be frozen before target authorization")
    write_json(out / PREFREEZE, {
        "contract": "topic5_multiscale_claim_adjudication_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "alpha": 0.05,
        "primary_target_free_rule": "rho_greater_than_0_and_one_sided_permutation_p_less_than_0.05",
        "suffix_rule": "median_greater_than_0_and_one_sided_patient_p_less_than_0.05",
        "early_interaction_rule": (
            "rho_greater_than_0_and_both_patient_label_and_synchronized_"
            "spatial_null_p_less_than_0.05"
        ),
        "D1_rule": "median_margin_greater_than_0_and_one_sided_patient_p_less_than_0.05",
        "D2_rule": "seed_removed_direct_and_L3_added_attenuation_AUC_form_one_Holm_corrected_family;_any_positive_adjusted_p_less_than_0.05",
        "next_extension_rule": "E1_if_target_free_or_cross_state_nonlocality_interaction;_E2_if_D1_without_D2_and_target_free_mode_flow_direction_is_stable;_otherwise_E3",
        "target_values_read": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--freeze-contract", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if args.freeze_contract:
        freeze_contract(out)
        return
    prefreeze = json.loads((out / PREFREEZE).read_text())
    if prefreeze.get("target_values_read") is not False:
        raise RuntimeError("claim adjudication prefreeze manifest is not target-free")
    if prefreeze.get("script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("claim adjudicator changed after its pre-unseal freeze")
    if not (out / "EARLY_ICTAL_SCORING_COMPLETE.json").exists():
        raise RuntimeError("claim adjudication requires locked early-ictal scoring")
    interictal = json.loads((out / "INTERICTAL_V0_5_SUMMARY.json").read_text())
    early = json.loads((out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json").read_text())
    detectability = json.loads((
        out / "functional_shortcut_detectability/FUNCTIONAL_DETECTABILITY_SUMMARY.json"
    ).read_text())
    mode_flow = json.loads((
        out / "mechanism/MODE_FLOW_ATTENUATION_SUMMARY.json"
    ).read_text())
    comparisons = interictal["comparisons"]
    primary_ii = comparisons["primary_nonlocality_interaction_all"]
    suffix_all = comparisons["L3_vs_suffix_all"]
    suffix_distal = comparisons["L3_vs_suffix_distal"]
    primary_ei = early["primary_interaction"]
    d1 = early["D1_L3_full_margin_gt_zero"]
    d2 = early["D2_L3_minus_L2m_seed_removed_signed_oracle"]
    d2_atten = early["D2_L3_added_attenuation_auc_seed_removed_gt_zero"]

    ii_supported = bool(
        primary_ii["spearman_rho"] > 0
        and primary_ii["permutation_p_greater"] < .05
    )
    suffix_supported = bool(
        suffix_all["median"] > 0 and suffix_all["wilcoxon_p_greater"] < .05
    )
    suffix_distal_supported = bool(
        suffix_distal["median"] > 0 and suffix_distal["wilcoxon_p_greater"] < .05
    )
    ei_supported = bool(
        primary_ei.get("spearman_rho", 0) > 0
        and primary_ei.get("joint_primary_p_greater", 1) < .05
    )
    d1_supported = bool(d1["median"] > 0 and d1["wilcoxon_p_greater"] < .05)
    d2_tests = {
        "seed_removed_direct": (float(d2["median"]), float(d2["wilcoxon_p_greater"])),
        "seed_removed_attenuation_auc": (
            float(d2_atten["median"]), float(d2_atten["wilcoxon_p_greater"])
        ),
    }
    ordered = sorted(d2_tests, key=lambda key: d2_tests[key][1])
    d2_holm = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * d2_tests[key][1])
        running = max(running, adjusted)
        d2_holm[key] = running
    d2_supported = any(
        median > 0 and d2_holm[key] < .05
        for key, (median, _raw_p) in d2_tests.items()
    )
    detectability_limited = bool(
        detectability.get("status") != "PASS_ALL_GEOMETRIES"
    )
    mode_flow_result = mode_flow["same_minus_cross_distal_selectivity"]
    mode_flow_supported = bool(
        mode_flow_result.get("median", 0) > 0
        and mode_flow_result.get("wilcoxon_p_greater", 1) < .05
    )
    payload = {
        "contract": "topic5_multiscale_claim_adjudication_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_LOCKED_INTERNAL_FOLLOWUP",
        "target_role": "LOCKED_INTERNAL_MECHANISTIC_FOLLOWUP_NOT_INDEPENDENT_CONFIRMATION",
        "claims": {
            "PRIMARY_TARGET_FREE_NONLOCALITY_INTERACTION": {
                "supported": ii_supported,
                "rho": primary_ii["spearman_rho"],
                "p_greater": primary_ii["permutation_p_greater"],
                "safe_claim": (
                    "Task-selected nonlocal benefit increases with cross-fitted patient nonlocality."
                    if ii_supported else
                    "No stable patient-level coupling was established between cross-fitted nonlocality and task-selected nonlocal benefit."
                ),
            },
            "KEY_SUFFIX_INFORMATION": {
                "supported_all_transitions": suffix_supported,
                "supported_distal_specificity": suffix_distal_supported,
                "safe_claim": (
                    "Real prefix-suffix association improves overall held-out interictal prediction."
                    if suffix_supported else
                    "Real prefix-suffix association did not improve overall held-out interictal prediction."
                ) + (
                    " The increment is distal-specific."
                    if suffix_distal_supported else
                    " The increment is not distal-specific."
                ),
            },
            "D1_CROSS_STATE_FIELD_CORRESPONDENCE": {
                "supported": d1_supported,
                "median_null_relative_margin": d1["median"],
                "p_greater": d1["wilcoxon_p_greater"],
                "safe_claim": (
                    "Frozen L3 interictal fields show positive signed correspondence with early-ictal broadband energy relative to synchronized all-contact shuffle."
                    if d1_supported else
                    "Frozen L3 interictal fields do not establish cohort-level correspondence with early-ictal broadband energy relative to synchronized all-contact shuffle."
                ),
            },
            "PRIMARY_EARLY_NONLOCALITY_INTERACTION": {
                "supported": ei_supported,
                "rho": primary_ei.get("spearman_rho"),
                "patient_label_p_greater": primary_ei.get("permutation_p_greater"),
                "spatial_null_p_greater": primary_ei.get(
                    "spatial_null", {}
                ).get("spatial_null_p_greater"),
                "joint_primary_p_greater": primary_ei.get("joint_primary_p_greater"),
                "safe_claim": (
                    "The L3-L2m cross-state increment increases with patient nonlocality."
                    if ei_supported else
                    "No stable patient-level coupling was established between nonlocality and the L3-L2m cross-state increment."
                ),
            },
            "D2_SHORTCUT_SPECIFIC_CROSS_STATE_CONTRIBUTION": {
                "supported": d2_supported,
                "seed_removed_direct": d2,
                "seed_removed_attenuation_auc": d2_atten,
                "holm_p_greater_within_D2_family": d2_holm,
                "safe_claim": (
                    "Task-selected nonlocal shortcuts contribute specifically to cross-state field correspondence."
                    if d2_supported else
                    "Cross-state field correspondence is not attributable specifically to task-selected nonlocal shortcuts."
                ),
            },
        },
        "interpretation_boundary": {
            "effective_scaffold_not_connectome": True,
            "broadband_field_not_arrival_or_recruitment_order": True,
            "event_lag_raw_not_axonal_delay": True,
            "functional_detectability_limited": detectability_limited,
            "no_exact_edge_identity_claim": True,
            "target_free_mode_flow_direction_supported": mode_flow_supported,
        },
        "next_extension_rule": (
            "E1_DUAL_SCALE_LATENCY" if ii_supported or ei_supported else
            "E2_STATE_DEPENDENT_GAIN"
            if d1_supported and not d2_supported and mode_flow_supported else
            "E3_SMOOTH_SUSCEPTIBILITY"
        ),
    }
    destination = out / "FINAL_CLAIM_ADJUDICATION.json"
    write_json(destination, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
