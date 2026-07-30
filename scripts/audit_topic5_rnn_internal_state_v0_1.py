#!/usr/bin/env python3
"""Final completeness and numerical reproduction audit for v0.1."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
RUNS = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    extraction_status = sorted(
        (BASE / "interictal/cells").glob("seed_*/**/CELL_STATUS.json")
    )
    analysis_status = sorted(
        (BASE / "interictal/per_subject").glob("*/ANALYSIS_STATUS.json")
    )
    perturbation_status = sorted(
        (BASE / "interictal/perturbation_cells").glob(
            "seed_*/**/CELL_STATUS.json"
        )
    )
    random_status = sorted(
        (BASE / "interictal/random_subspace_cells").glob(
            "seed_*/**/CELL_STATUS.json"
        )
    )
    collections = {
        "extraction_cells": (extraction_status, 102),
        "subject_analyses": (analysis_status, 34),
        "perturbation_cells": (perturbation_status, 102),
        "random_subspace_cells": (random_status, 102),
    }
    completeness = {}
    for name, (paths, expected) in collections.items():
        statuses = [json.loads(path.read_text()) for path in paths]
        complete = sum(payload.get("status") == "COMPLETE" for payload in statuses)
        target_blind = sum(
            not payload.get("target_values_read", False)
            and not payload.get("early_ictal_arrays_deserialized", False)
            for payload in statuses
        )
        completeness[name] = {
            "expected": expected,
            "found": len(paths),
            "complete": complete,
            "target_blind_certified": target_blind,
        }
        if len(paths) != expected or complete != expected:
            raise RuntimeError(f"{name}: incomplete")

    rows = []
    for path in extraction_status:
        payload = json.loads(path.read_text())
        with np.load(ROOT / payload["dataset"], allow_pickle=False) as data:
            heldout_total = int(np.sum(data["event_split"] == 1))
        heldout_selected = int(payload["split_events"]["heldout20"])
        old = pd.read_csv(
            RUNS
            / payload["seed_dir"]
            / payload["subject"]
            / "heldout_metrics.csv"
        ).set_index("control")
        for metric in payload["metrics"]:
            if metric["metric"] != "pca_inventory":
                continue
            control = metric["control"]
            old_nll = float(old.loc[control, "heldout_event_nll"])
            new_nll = float(metric["original_heldout_event_nll"])
            rows.append(
                {
                    "subject": payload["subject"],
                    "seed_dir": payload["seed_dir"],
                    "control": control,
                    "old_heldout_event_nll": old_nll,
                    "recomputed_heldout_event_nll": new_nll,
                    "absolute_difference": abs(old_nll - new_nll),
                    "heldout_total_events": heldout_total,
                    "heldout_selected_events": heldout_selected,
                    "same_event_denominator": (
                        heldout_total == heldout_selected
                    ),
                }
            )
    nll = pd.DataFrame(rows)
    nll.to_csv(BASE / "reproduction_nll_audit.csv", index=False)
    same_denominator = nll.loc[nll.same_event_denominator]
    maximum_exact_difference = float(
        same_denominator.absolute_difference.max()
    )
    maximum_sampled_difference = float(
        nll.loc[~nll.same_event_denominator, "absolute_difference"].max()
    )
    if maximum_exact_difference > 1.0e-6:
        raise RuntimeError("frozen heldout NLL did not reproduce")

    target = json.loads((BASE / "EARLY_ICTAL_READBACK_SUMMARY.json").read_text())
    strict = target["strict_clinical_onset_cohort"]
    if strict["n_patients"] != 16 or strict["n_seizures"] != 106:
        raise RuntimeError("strict clinical-onset denominator drifted")
    if strict["yuquan_eeg_onset_in_primary"]:
        raise RuntimeError("Yuquan EEG onset leaked into primary")
    audit = {
        "contract": "topic5_rnn_internal_state_reduction_v0_1",
        "status": "PASS",
        "completeness": completeness,
        "nll_reproduction": {
            "n_cells_times_controls": int(len(nll)),
            "same_denominator_rows": int(len(same_denominator)),
            "deterministically_sampled_rows": int(
                np.sum(~nll.same_event_denominator)
            ),
            "maximum_absolute_difference_same_denominator": (
                maximum_exact_difference
            ),
            "maximum_absolute_difference_sampled_denominator": (
                maximum_sampled_difference
            ),
            "tolerance": 1.0e-6,
            "interpretation": (
                "Exact reproduction is required only when the frozen run and "
                "state extraction score the same heldout events. Larger "
                "patients use a deterministic 2048-event extraction sample; "
                "their difference is reported but is not numerical drift."
            ),
        },
        "strict_early_ictal": strict,
        "figures_readme_exists": (
            ROOT
            / "results/paper-ready-figure/"
            "fig6_rnn_internal_state_reduction/figures/README.md"
        ).exists(),
    }
    atomic_json(BASE / "REPRODUCTION_AUDIT.json", audit)
    atomic_json(
        BASE / "FINAL_STATUS.json",
        {
            "status": "COMPLETE",
            "audit": audit,
            "interictal_summary": (
                "results/topic5_rnn_internal_state_reduction/"
                "INTERICTAL_SUMMARY.json"
            ),
            "sensitivity_summary": (
                "results/topic5_rnn_internal_state_reduction/"
                "INTERICTAL_SENSITIVITY_SUMMARY.json"
            ),
            "eventfirst_field_summary": (
                "results/topic5_rnn_internal_state_reduction/"
                "INTERICTAL_EVENTFIRST_FIELD_SUMMARY.json"
            ),
            "early_ictal_summary": (
                "results/topic5_rnn_internal_state_reduction/"
                "EARLY_ICTAL_READBACK_SUMMARY.json"
            ),
            "paper_ready_figure": (
                "results/paper-ready-figure/"
                "fig6_rnn_internal_state_reduction/figures/"
                "fig6_rnn_internal_state_reduction.png"
            ),
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
