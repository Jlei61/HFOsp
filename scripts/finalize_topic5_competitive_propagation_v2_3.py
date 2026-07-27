#!/usr/bin/env python3
"""Fail-closed reproducibility and stop-rule audit for Topic-5 v2.3."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
FORMAL = BASE / "formal"
TEST_LOG = (
    ROOT
    / "results/run_logs/"
    "topic5_rnn_closeout_v2_2_v2_3_pytest_final_2026-07-27.log"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    input_status = load(BASE / "input_audit/INPUT_AUDIT_STATUS.json")
    development = load(BASE / "development/DEVELOPMENT_FREEZE.json")
    development_launcher = load(BASE / "development/LAUNCHER_STATE.json")
    formal_launcher = load(FORMAL / "LAUNCHER_STATE.json")
    markov = load(FORMAL / "MARKOV_BENCHMARK_STATE.json")
    watcher = load(FORMAL / "WATCHER_STATE.json")
    gates = load(FORMAL / "FORMAL_GATE_STATUS.json")
    run_inventory = pd.read_csv(FORMAL / "formal_run_inventory.csv")
    patient_metrics = pd.read_csv(FORMAL / "patient_model_metrics.csv")
    claims = pd.read_csv(FORMAL / "claim_comparisons.csv")
    markov_table = pd.read_csv(FORMAL / "markov_benchmarks.csv")
    figure = BASE / "figures/competitive_propagation_rnn_formal.png"
    pdf = BASE / "figures/competitive_propagation_rnn_formal.pdf"
    readme = BASE / "figures/README.md"
    report = (
        ROOT
        / "docs/archive/topic5/"
        "symmetric_axis_competitive_propagation_rnn_v2_3_result_2026-07-27.md"
    )
    paper = (
        ROOT
        / "docs/paper-draft/"
        "figure6_competitive_propagation_rnn_bounded_result.md"
    )
    with Image.open(figure) as image:
        dimensions = list(image.size)

    checks = {
        "input_audit_pass": input_status.get("status") == "PASS",
        "denominator_34_subjects": input_status.get("n_subjects") == 34,
        "denominator_864163_events": (
            input_status.get("n_events_total") == 864_163
        ),
        "tied_event_exclusion_25": (
            input_status.get("n_non_source_tied_events_excluded") == 25
        ),
        "development_frozen": development.get("status") == "FROZEN",
        "development_36_of_36": (
            development_launcher.get("status") == "COMPLETE"
            and development_launcher.get("n_tasks_finished") == 36
            and development_launcher.get("n_tasks_failed") == 0
        ),
        "formal_66_of_66": (
            formal_launcher.get("status") == "COMPLETE"
            and formal_launcher.get("n_tasks_finished") == 66
            and formal_launcher.get("n_tasks_failed") == 0
        ),
        "formal_330_model_fits": len(run_inventory) == 330,
        "formal_22_patients_3_seeds": (
            run_inventory.subject.nunique() == 22
            and run_inventory.seed.nunique() == 3
        ),
        "markov_22_patients_66_rows": (
            markov.get("status") == "COMPLETE"
            and len(markov_table) == 66
        ),
        "watcher_completed_analysis": watcher.get("status") == "COMPLETE",
        "formal_gate_complete": gates.get("status") == "COMPLETE",
        "patient_model_table_22_by_8": len(patient_metrics) == 22 * 8,
        "claim_table_7_rows": len(claims) == 7,
        "heldout_not_used_for_selection": not gates.get(
            "heldout_used_for_training_or_epoch_selection"
        ),
        "target_values_read_false": not any(
            bool(value)
            for value in (
                input_status.get("target_values_read"),
                development.get("early_ictal_target_values_read"),
                formal_launcher.get("target_values_read"),
                markov.get("target_values_read"),
                watcher.get("target_values_read"),
                gates.get("target_values_read"),
                run_inventory.target_values_read.any(),
                markov_table.target_values_read.any(),
            )
        ),
        "latent_state_analysis_locked": not gates.get(
            "latent_state_analysis_allowed"
        ),
        "no_latent_state_output_created": not any(
            FORMAL.glob("latent_state*")
        ),
        "no_early_ictal_transfer_output_created": not (
            BASE / "early_ictal_transfer"
        ).exists(),
        "figure_png_pdf_readme_exist": (
            figure.is_file() and pdf.is_file() and readme.is_file()
        ),
        "figure_dimensions_2960_by_2140": dimensions == [2960, 2140],
        "report_and_paper_draft_exist": report.is_file() and paper.is_file(),
        "final_tests_50_passed": (
            TEST_LOG.is_file()
            and "50 passed" in TEST_LOG.read_text(encoding="utf-8")
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise SystemExit(f"v2.3 final audit failed: {failed}")

    essential = (
        BASE / "input_audit/INPUT_AUDIT_STATUS.json",
        BASE / "development/DEVELOPMENT_FREEZE.json",
        FORMAL / "LAUNCHER_STATE.json",
        FORMAL / "markov_benchmarks.csv",
        FORMAL / "formal_run_inventory.csv",
        FORMAL / "patient_model_metrics.csv",
        FORMAL / "claim_comparisons.csv",
        FORMAL / "benefit_recovery.csv",
        FORMAL / "FORMAL_GATE_STATUS.json",
        figure,
        pdf,
        readme,
        report,
        paper,
        ROOT / "src/topic5_competitive_propagation_v2_3.py",
        ROOT / "scripts/train_topic5_competitive_propagation_formal_v2_3.py",
        ROOT / "scripts/analyze_topic5_competitive_propagation_formal_v2_3.py",
        ROOT / "scripts/plot_topic5_competitive_propagation_v2_3.py",
        TEST_LOG,
    )
    payload = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "status": "PASS",
        "scientific_stop_status": {
            "claim_A_predictive_adequacy": gates[
                "claim_A_predictive_adequacy"
            ],
            "claim_B_history_state_necessary": gates[
                "claim_B_history_state_necessary"
            ],
            "claim_C_matched_axis_increment": gates[
                "claim_C_matched_axis_increment"
            ],
            "claim_D_source_conditioned_direction": gates[
                "claim_D_source_conditioned_direction"
            ],
            "latent_state_analysis": "LOCKED_NOT_RUN",
            "early_ictal_transfer": gates["early_ictal_transfer"],
        },
        "checks": checks,
        "manual_visual_qa": {
            "status": "PASS",
            "checked_png": str(figure.relative_to(ROOT)),
            "panel_titles_not_overlapping": True,
            "all_six_panels_have_scientific_roles": True,
            "gate_failure_is_visible": True,
            "no_internal_patient_identifiers_on_canvas": True,
        },
        "target_values_read": False,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in essential
        },
    }
    atomic_json(BASE / "FINAL_REPRODUCIBILITY_MANIFEST.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
