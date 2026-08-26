#!/usr/bin/env python3
"""Regenerate the long total-effect machine audit from the artefacts.

The 2026-08-26 audit was hand-written; every hash in it happened to be correct,
but nothing re-derived it, so a later edit to any tracked file would leave a
silently stale audit.  This script rebuilds it from disk and records the
post-review correction state alongside the original engineering counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_long_total import LONG_TOTAL_REVISION


WINDOWS = ("event_count_10000", "physical_6h")
SEEDS = (0, 1, 2)
SUBJECT = "yuquan_zhangjiaqi"
TRACKED = {
    "contract": "docs/archive/topic5/continuous_marked_state_t2_long_total_effect_contract_2026-08-26.md",
    "post_review_corrections": "docs/archive/topic5/continuous_marked_state_t2_long_total_post_review_corrections_2026-08-26.md",
    "long_total_module": "src/topic5_continuous_marked_state_r1/t2_long_total.py",
    "human_runner": "scripts/topic5_continuous_marked_state_r1/run_t2_long_total_human.py",
    "synthetic_runner": "scripts/topic5_continuous_marked_state_r1/run_t2_long_total_synthetic.py",
    "aggregator": "scripts/topic5_continuous_marked_state_r1/aggregate_t2_long_total.py",
    "post_review_auditor": "scripts/topic5_continuous_marked_state_r1/audit_t2_long_total_post_review.py",
    "plain_report": "results/epi_prssm/continuous_marked_state/r1/final_reports/r1_2b_r1_3_t2_long_total_plain_2026-08-26.md",
    "technical_report": "results/epi_prssm/continuous_marked_state/r1/final_reports/r1_2b_r1_3_t2_long_total_technical_2026-08-26.md",
    "handoff": "results/epi_prssm/continuous_marked_state/r1/CURRENT_HANDOFF.md",
}


def _tests(log: Path) -> dict:
    text = log.read_text().strip().splitlines()[-1] if log.exists() else ""
    passed = failed = warnings = None
    for token, name in (("passed", "passed"), ("failed", "failed"),
                        ("warning", "warnings")):
        for index, word in enumerate(text.replace(",", " ").split()):
            if word.startswith(token) and index:
                value = text.replace(",", " ").split()[index - 1]
                if value.isdigit():
                    if name == "passed":
                        passed = int(value)
                    elif name == "failed":
                        failed = int(value)
                    else:
                        warnings = int(value)
    return {
        "summary_line": text,
        "passed": passed,
        "failed": failed if failed is not None else 0,
        "warnings": warnings,
        "log": str(log),
        "log_sha256": contract.sha256_file(log) if log.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=contract.RESULT_ROOT / "t2_long_total_effect")
    parser.add_argument("--output", type=Path,
                        default=contract.RESULT_ROOT / "final_reports"
                        / "r1_2b_r1_3_t2_long_total_machine_audit.json")
    args = parser.parse_args()
    human = {}
    for window in WINDOWS:
        for seed in SEEDS:
            path = args.root / "human" / SUBJECT / window / f"seed_{seed}/result.json"
            human[f"{window}/seed_{seed}"] = json.loads(path.read_text())
    t1 = {
        seed: json.loads((
            args.root / "t1_r1_3/fits" / SUBJECT / f"explicit_seed_{seed}/result.json"
        ).read_text())
        for seed in SEEDS
    }
    synthetic = json.loads((args.root / "synthetic/recovery.json").read_text())
    summary = json.loads((args.root / "reports/summary.json").read_text())
    post_review = json.loads((args.root / "reports/post_review_audit.json").read_text())
    payload = {
        "status": "COMPLETE",
        "revision": LONG_TOTAL_REVISION,
        "subject": SUBJECT,
        "engineering_completion": {
            "r1_3_t1_fits_complete": sum(
                value.get("status") == "COMPLETE" for value in t1.values()
            ),
            "r1_3_t1_fits_expected": len(SEEDS),
            "human_long_fits_complete": sum(
                value.get("status") == "COMPLETE" for value in human.values()
            ),
            "human_long_fits_expected": len(WINDOWS) * len(SEEDS),
            "synthetic_status": synthetic["status"],
            "all_sealed_flags_false": all(
                value.get("sealed_opened") is False
                for value in list(human.values()) + list(t1.values()) + [synthetic]
            ),
            "all_formal_test_partition_flags_false": all(
                value.get("formal_test_partition_opened") is False
                for value in human.values()
            ),
        },
        "scientific_admissibility": {
            "t1_selected_epoch_zero_seeds": sum(
                value["fit_trace"]["selected_total_epoch"] == 0
                for value in t1.values()
            ),
            "t1_seeds": len(SEEDS),
            "t1_state_model_entirely_at_initialisation": post_review[
                "t1_state_model"]["state_model_entirely_at_initialisation"],
            "decoder_rank": sorted({
                value["decoder_readout"]["rank"] for value in human.values()
            }),
            **{
                f"{window}_admissible_seeds": summary["windows"][window][
                    "decoder_space_evidence_vector"]["admissible_seeds"]
                for window in WINDOWS
            },
            "human_scientific_status": sorted({
                summary["windows"][window]["decoder_space_evidence_vector"][
                    "scientific_status"] for window in WINDOWS
            }),
            "structural_zero_is_not_h3_negative": True,
            "distinct_seed_payloads": {
                window: summary["windows"][window]["seed_independence"][
                    "distinct_seed_payloads"] for window in WINDOWS
            },
            "synthetic_acceptance_all_true": all(synthetic["acceptance"].values()),
            "synthetic_acceptance": synthetic["acceptance"],
        },
        "denominators": {
            window: {
                **{
                    key: human[f"{window}/seed_0"]["denominators"][key]
                    for key in (
                        "train_windows", "validation_windows",
                        "validation_next_event_pairs",
                        "windows_cross_unrecorded_gap",
                    )
                },
                "windows_cross_unrecorded_gap_recomputed": post_review["windows"][
                    window]["windows_cross_unrecorded_gap_computed"],
                "validation_endpoint_span_hours": post_review["windows"][window][
                    "endpoint_support"]["validation"]["endpoint_span_hours"],
                "validation_effective_independent_windows": post_review["windows"][
                    window]["endpoint_support"]["validation"][
                    "effective_independent_windows"],
                "generator_time_constant_minutes": post_review["windows"][window][
                    "effective_exposure_time_scale"][
                    "slowest_mode_time_constant_minutes"],
                "median_effective_weighted_events": post_review["windows"][window][
                    "effective_exposure_time_scale"][
                    "median_effective_weighted_events"],
            }
            for window in WINDOWS
        },
        "post_review": {
            "date": "2026-08-26",
            "record": TRACKED["post_review_corrections"],
            "human_results_predate_the_fixes": sorted({
                value.get("revision") for value in human.values()
            }),
            "corrections": [
                "real_minus_no_edge demoted: the exposure arms own a free "
                "state-space intercept, worth -445 on an exposure-free target",
                "nominal window is not the tested time scale: the generator "
                "time constant is 54.1 min, not 6 h",
                "window counts are not a sample size: 1.8 / 2.4 effective "
                "independent validation windows",
                "short-scale T2-S1 '0/2' was a structural zero; the correct "
                "denominator is 0/0",
            ],
            "stale_artefacts_requiring_a_rerun_before_citation": [
                "t2_s1_long_scale/human/**/result.json (placebo donor exclusion "
                "now covers validation targets)",
            ],
        },
        "tests": _tests(args.root / "logs/final_pytest.log"),
        "hashes": {
            name: contract.sha256_file(contract.REPO_ROOT / path)
            for name, path in TRACKED.items()
        },
        "result_hashes": {
            "synthetic_result": contract.sha256_file(args.root / "synthetic/recovery.json"),
            "human_summary": contract.sha256_file(args.root / "reports/summary.json"),
            "post_review_audit": contract.sha256_file(
                args.root / "reports/post_review_audit.json"),
            "per_seed_csv": contract.sha256_file(
                args.root / "reports/per_seed_summary.csv"),
        },
        "claim_boundary": (
            "development instrument result; one high-event-count patient; no "
            "causal, cohort, seizure or formal-test claim"
        ),
    }
    contract.atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
