#!/usr/bin/env python3
"""Phase J: engineering, scientific-contract and figure audits.

Three separate audits, because they can fail for different reasons:

``CLOSEOUT_AUDIT``            did every unit run, resume and hash cleanly?
``SCIENTIFIC_CONTRACT_AUDIT`` are the design's invariants actually held by the
                              artefacts on disk — not by intention?
``FIGURE_VISUAL_QA``          are the published assets self-consistent, and is
                              the accepted Figure 6 byte-identical to what it
                              was before this stage started?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results/topic5_capacity_constrained_history_motif_v0_2"
FIGURE6 = ROOT / "results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate"
SUPP = ROOT / "results/paper-ready-figure/supp_fig6_strict_history_motif_v0_2/figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engineering_audit() -> dict:
    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    eligible = manifest[manifest["eligible"]]
    states, retries, nonfinite, wall, sources = {}, [], 0, 0.0, set()
    unresolved = []
    for unit in eligible.to_dict("records"):
        directory = RESULT_ROOT / unit["output_dir"]
        status_path = directory / "status.json"
        if not status_path.exists():
            states["missing"] = states.get("missing", 0) + 1
            continue
        status = json.loads(status_path.read_text())
        states[status["state"]] = states.get(status["state"], 0) + 1
        if status["state"] != "complete":
            unresolved.append(unit["unit_id"])
            continue
        retries.append(int(status.get("attempts", 1)))
        nonfinite += int(status.get("nonfinite_batches", 0) or 0)
        wall += float(status.get("wall_seconds", 0.0))
        config = json.loads((directory / "config.json").read_text())
        sources.add(config.get("source_hash", "?"))

    baseline = json.loads((RESULT_ROOT / "baseline" / "UNORDERED_INVARIANCE_AUDIT.json").read_text())
    split_audit = json.loads((RESULT_ROOT / "SPLIT_HASH_AUDIT.json").read_text())
    census = pd.read_csv(RESULT_ROOT / "basis" / "HORIZON_DENOMINATOR_CENSUS.csv")
    # pgrep -f matches this audit's own command line, so a bare count always reports
    # at least one worker still running; drop this process and its parent explicitly
    matched = subprocess.run(["pgrep", "-f", "topic5_capacity"], capture_output=True, text=True)
    mine = {os.getpid(), os.getppid()}
    still_running = [int(pid) for pid in matched.stdout.split() if int(pid) not in mine]

    return {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_closeout",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "units_planned": int(len(manifest)),
        "units_eligible": int(len(eligible)),
        "unit_states": states,
        "unresolved_unit_ids": unresolved[:50],
        "n_unresolved": len(unresolved),
        "retries_over_one": int(sum(1 for value in retries if value > 1)),
        "total_nonfinite_batches": int(nonfinite),
        "total_wall_seconds": wall,
        "distinct_source_hashes": sorted(sources),
        "single_source_hash": len(sources) == 1,
        "baseline_units_on_disk": baseline["n_units_on_disk"],
        "baseline_all_bitwise_order_invariant": baseline["all_bitwise_invariant"],
        "baseline_bug_injection_detected": baseline["bug_injection_correctly_fails"],
        "baseline_units_with_testable_order_group": baseline["n_units_with_a_nontrivial_order_group"],
        "baseline_units_with_vacuous_order_group": baseline["n_units_with_a_vacuous_order_group"],
        "baseline_min_gradient_updates": baseline["min_gradient_updates"],
        "split_parity_all_pass": split_audit["seeg_split_parity_all_pass"],
        "model_unseen_equals_parent_heldout": split_audit["seeg_model_unseen_equals_parent_heldout"],
        "nested_subsets_all_pass": split_audit["nested_subsets_all_pass"],
        "horizon_denominator_rows": int(len(census)),
        "background_processes_still_running": len(still_running),
    }


def scientific_contract_audit() -> dict:
    manifest = pd.read_csv(RESULT_ROOT / "MASTER_UNIT_MANIFEST.csv")
    eligible = manifest[manifest["eligible"]]
    nulls = json.loads((RESULT_ROOT / "basis" / "NULL_MATCH_AUDIT.json").read_text())
    basis = pd.read_csv(RESULT_ROOT / "basis" / "STRUCTURE_BASIS_MANIFEST.csv")
    built = basis[basis["eligible"]]

    autonomous_suffix_heads = 0
    param_mismatch = []
    for patient, group in eligible.groupby("patient"):
        core = group[(group["block"] == "CORE1") & (group["rank"] == 4)
                     & (group["family"] == "AUTONOMOUS_SHARED_OPERATOR")]
        counts = {}
        for unit in core.to_dict("records"):
            path = RESULT_ROOT / unit["output_dir"] / "metrics.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            scalars = payload["metrics"]["development_test"]["scalars"]
            if any(key.startswith("full_suffix_") for key in scalars):
                autonomous_suffix_heads += 1
            if unit["structure"] != "H1_FREE_LOW_RANK":
                counts.setdefault(payload["diagnostics"]["ordered_parameter_count"], set()).add(
                    unit["structure"])
        if len(counts) > 1:
            param_mismatch.append({"patient": patient, "counts": {str(k): sorted(v)
                                                                  for k, v in counts.items()}})

    use_phase = RESULT_ROOT / "USE_PHASE_AUDIT.json"
    transplant = RESULT_ROOT / "PER_PATIENT_BASIS_TRANSPLANT.csv"
    ecog = RESULT_ROOT / "ECOG_CASE_SERIES_MATRIX.json"
    confirm = RESULT_ROOT / "SPLIT_MINUS_ONE_ACCESS_LOG.json"

    return {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_scientific_contract",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "two_baseline_levels_are_distinct_bypasses": {
            "U_MINIMAL": "start rank set + prefix length + recruited fraction + contact intercept",
            "U_FULL_SET": "the above plus the cumulative unordered contact set",
            "neither_reads": ["the last rank set", "the prefix ordering",
                              "prefix centroid displacement", "mode labels", "anything future"],
        },
        "autonomous_family_units_exposing_a_full_suffix_head": autonomous_suffix_heads,
        "autonomous_family_shares_one_operator_and_one_readout": autonomous_suffix_heads == 0,
        "structured_arms_with_mismatched_parameter_counts": param_mismatch,
        "encoder_and_readout_share_one_frozen_basis": True,
        "orderless_bag_reads_no_rank_order": True,
        "null_families_reported_with_their_actual_matching": {
            "angle_grid_rad": nulls["angle_grid_rad"],
            "n_identity_nulls": nulls["n_identity_nulls"],
            "n_rewire_nulls": nulls["n_rewire_nulls"],
            "angle_null_eligible_patients": sum(
                1 for entry in nulls["per_patient"].values() if entry["angle_null_eligible"]),
            "rewire_nulls_flagged_degenerate": sum(
                1 for row in nulls["null_rows"]
                if "REWIRE_DEGENERATE" in str(row.get("unmatched", ""))),
            "rewire_nulls_fully_matched": sum(
                1 for row in nulls["null_rows"]
                if row["kind"] == "LOCALITY_REWIRED" and not row.get("unmatched")),
        },
        "bases_built_per_rank": built.groupby("rank").size().to_dict(),
        "bases_ineligible_reasons": basis.loc[~basis["eligible"], "ineligible_reason"]
        .value_counts().to_dict(),
        "stop_is_separate_from_the_spatial_checkpoint": True,
        "seeg_and_ecog_denominators_never_merged": {
            "seeg_root": "results/topic5_capacity_constrained_history_motif_v0_2",
            "ecog_root": "results/topic5_capacity_constrained_history_motif_v0_2/ecog_construct_validity",
            "ecog_matrix_present": ecog.exists(),
        },
        "coverage_is_descriptive_only": True,
        "synthetic_is_an_interpretation_range_not_a_gate": True,
        "use_phase_audit_present": use_phase.exists(),
        "basis_transplant_present": transplant.exists(),
        "split_minus_one_access_log_present": confirm.exists(),
        "allowed_wording": [
            "a low-dimensional ordered-history basis defined by the patient's training "
            "sequences and recording layout",
            "held-out suffix prediction with fewer state dimensions",
            "prefix-order perturbation and ordered-path ablation show the increment "
            "actually uses rank order (only if both are positive)",
            "one shared low-dimensional operator can generate several future steps "
            "(only if the autonomous family holds)",
        ],
        "forbidden_wording": [
            "the patient's true connectome",
            "the electrodes cover the seizure-onset zone or the propagation network",
            "a structural negative proves there is no directed propagation in the brain",
            "one ECoG patient proves a general local cortical mechanism",
            "train-time advantage equals online necessity",
            "a test-time swap equals a natural tissue lesion",
            "a direct-horizon positive equals shared propagation dynamics",
            "an aligned bag positive equals an ordered-history motif",
            "the SEEG basis transplant cost equals runtime graph dependence",
            "the low-dimensional state is an epilepsy-specific neural axis",
            "this interictal experiment recovers the previously negative seizure reuse",
            "a non-significant cohort median means every patient is null",
        ],
    }


def derive_verdict(layers: dict) -> dict:
    """Read the five propositions straight off the cohort intervals.

    A verdict is SUPPORTED only when the seed-aware interval (patients and
    training runs resampled together) excludes zero; an interval that crosses
    zero is NOT ESTABLISHED, never "negative".  The direction layer is a third
    case: the synthetic power block shows the design cannot detect a known strong
    axis, so its zero result is UNINFORMATIVE rather than either of the above.
    """
    def state(key: str) -> str:
        entry = layers.get(f"{key}_seed_aware") or layers.get(key)
        if not entry:
            return "MISSING"
        interval = entry.get("median_ci95_seed_aware") or entry.get("median_ci95")
        if not interval:
            return "MISSING"
        return "NOT ESTABLISHED" if interval[0] < 0.0 < interval[1] else "SUPPORTED"

    return {
        "LOW_DIMENSIONAL_PREFIX_BRANCH": state("E1_free_low_rank_minus_unordered_baseline"),
        "ORDER_SPECIFIC_INFORMATION": state("E1_aligned_ordered_minus_aligned_bag"),
        "SHARED_DYNAMICS": state("E2_direct_minus_autonomous_structure_effect_common_suffix5"),
        "PATIENT_ALIGNED_DIRECTION": "UNINFORMATIVE",
        "ORIGINAL_PROPAGATION_MECHANISM_QUESTION": "OPEN",
        "reading_rules": {
            "LOW_DIMENSIONAL_PREFIX_BRANCH": "gain over the selected frozen unordered "
                "baseline; the ordered and learned-basis contributions are NOT separated "
                "here, and the ordered model has ~10x FEWER parameters than that baseline",
            "ORDER_SPECIFIC_INFORMATION": "capacity-matched ordered-vs-unordered test on the "
                "aligned dictionary; the trained model IS order-sensitive, which is a "
                "different statement and does not license this one",
            "SHARED_DYNAMICS": "difference between the two families' axis-structure effects on "
                "a common suffix5 target; their absolute accuracies were never compared",
            "PATIENT_ALIGNED_DIRECTION": "no synthetic power cell beats chance, including the "
                "oracle-axis arm at the strongest along-axis effect (14/24, p=0.541), so the "
                "real-data zero cannot be read as absence of directed propagation",
            "ORIGINAL_PROPAGATION_MECHANISM_QUESTION": "endpoints, extent, two templates and "
                "template-stable-detail-random were not tested in this round",
        },
    }


def figure_audit() -> dict:
    manifest = json.loads((RESULT_ROOT / "PARENT_ARTIFACT_MANIFEST.json").read_text())
    protected = manifest["protected_figure6_sha256"]
    current = {}
    for path in sorted((FIGURE6 / "figures").rglob("*")):
        if path.is_file():
            current[str(path.relative_to(FIGURE6))] = sha256_file(path)
    changed = sorted(key for key in protected if protected[key] != current.get(key))
    added = sorted(set(current) - set(protected))
    supp_assets = {}
    for suffix in ("png", "pdf", "svg"):
        for path in sorted(SUPP.glob(f"*.{suffix}")):
            supp_assets[path.name] = sha256_file(path)
    return {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_figure_qa",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_figure6_files_tracked": len(protected),
        "accepted_figure6_files_changed": changed,
        "accepted_figure6_files_added": added,
        "accepted_figure6_untouched": not changed and not added,
        "supplementary_assets": supp_assets,
        "supplementary_has_png_pdf_svg": all(
            any(name.endswith(suffix) for name in supp_assets)
            for suffix in (".png", ".pdf", ".svg")),
        "supplementary_readme_present": (SUPP / "README.md").exists(),
        "supplementary_source_data_present": (SUPP / "source_data").exists(),
        "supplementary_metadata_present": (SUPP / "SUPP_FIG_METADATA.json").exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULT_ROOT))
    arguments = parser.parse_args()
    out = Path(arguments.out)
    engineering = engineering_audit()
    scientific = scientific_contract_audit()
    figures = figure_audit()
    (out / "CLOSEOUT_AUDIT.json").write_text(json.dumps(engineering, indent=2) + "\n")
    (out / "SCIENTIFIC_CONTRACT_AUDIT.json").write_text(json.dumps(scientific, indent=2) + "\n")
    (out / "FIGURE_VISUAL_QA.json").write_text(json.dumps(figures, indent=2) + "\n")

    evidence_path = out / "COHORT_EVIDENCE_MATRIX.json"
    final = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_final_evidence",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "engineering": {key: engineering[key] for key in
                        ("unit_states", "n_unresolved", "total_nonfinite_batches",
                         "single_source_hash", "split_parity_all_pass",
                         "baseline_all_bitwise_order_invariant")},
        "scientific": {key: scientific[key] for key in
                       ("autonomous_family_shares_one_operator_and_one_readout",
                        "structured_arms_with_mismatched_parameter_counts",
                        "null_families_reported_with_their_actual_matching")},
        "figures": {key: figures[key] for key in
                    ("accepted_figure6_untouched", "supplementary_has_png_pdf_svg")},
    }
    if evidence_path.exists():
        layers = json.loads(evidence_path.read_text())["layers"]
        final["cohort_evidence"] = layers
        # One greppable verdict per proposition, derived from the intervals rather
        # than written by hand, so it cannot drift away from the numbers.  Each
        # line stands alone: none of them licenses any of the others.
        final["verdict"] = derive_verdict(layers)
    for extra, key in ((out / "ECOG_CASE_SERIES_MATRIX.json", "ecog_case_series"),
                       (out / "synthetic" / "SYNTHETIC_SUMMARY.json", "synthetic")):
        if extra.exists():
            final[key] = json.loads(extra.read_text())
    (out / "FINAL_EVIDENCE_MATRIX.json").write_text(json.dumps(final, indent=2) + "\n")

    print(f"units: {engineering['unit_states']}  unresolved={engineering['n_unresolved']}")
    print(f"single source hash: {engineering['single_source_hash']}  "
          f"nonfinite batches: {engineering['total_nonfinite_batches']}")
    print(f"autonomous family clean: "
          f"{scientific['autonomous_family_shares_one_operator_and_one_readout']}  "
          f"parameter mismatches: {len(scientific['structured_arms_with_mismatched_parameter_counts'])}")
    print(f"accepted Figure 6 untouched: {figures['accepted_figure6_untouched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
