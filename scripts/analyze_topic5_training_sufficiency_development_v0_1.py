#!/usr/bin/env python3
"""Analyse the non-LOSO development phases (B1 budget, B2 optimizer, B3 parity).

Every cohort number is patient-first: seeds are averaged inside a patient and
only then aggregated across patients, so event-rich patients cannot dominate.
The primary endpoint is the validation contact-choice NLL in nats/decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_training_sufficiency import (  # noqa: E402
    aggregate_patient_metric,
    plateau_verdict,
)

PRIMARY = "validation_contact_choice_nll"


def _load_cells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    config_rows = []
    for done in sorted(root.rglob("DONE.json")):
        cell = done.parent
        summary = json.loads((cell / "run_summary.json").read_text())
        configuration = summary["configuration"]
        frame = pd.read_csv(cell / "cycle_patient_metrics.csv")
        frame["cell"] = str(cell.relative_to(root))
        for key, value in configuration.items():
            frame[f"cfg_{key}"] = value
        metric_rows.append(frame)
        for entry in summary["per_cycle"]:
            config_rows.append(
                {
                    "cell": str(cell.relative_to(root)),
                    **{f"cfg_{key}": value for key, value in configuration.items()},
                    **entry,
                    "runtime_seconds": summary["resources"]["runtime_seconds"],
                    "gpu_peak_allocated_bytes": summary["resources"][
                        "gpu_peak_allocated_bytes"
                    ],
                }
            )
    if not metric_rows:
        raise RuntimeError(f"no completed development cells under {root}")
    return pd.concat(metric_rows, ignore_index=True), pd.DataFrame(config_rows)


def _config_key(frame: pd.DataFrame) -> pd.Series:
    return (
        "u"
        + frame.cfg_updates_per_patient.astype(str)
        + "_h"
        + frame.cfg_hidden_size.astype(str)
        + "_lr"
        + frame.cfg_learning_rate.map(lambda value: f"{value:g}")
        + "_"
        + frame.cfg_optimizer.astype(str)
        + "_wd"
        + frame.cfg_weight_decay.map(lambda value: f"{value:g}")
        + "_b"
        + frame.cfg_batch_size.astype(str)
        + "_"
        + frame.cfg_objective.astype(str)
    )


def _summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["config_key"] = _config_key(metrics)
    rows = []
    for (config, cycle), group in metrics.groupby(["config_key", "coverage_cycle"]):
        records = group.to_dict("records")
        primary = aggregate_patient_metric(records, value_key=PRIMARY)
        total = aggregate_patient_metric(records, value_key="validation_total_nll")
        stop = aggregate_patient_metric(
            records, value_key="validation_stop_contribution_nll"
        )
        gap = aggregate_patient_metric(
            records, value_key="train_validation_gap_contact_choice"
        )
        # seed variance: spread of the patient-median across seeds
        per_seed = [
            float(
                np.median(
                    seed_group.groupby("subject")[PRIMARY].mean().to_numpy(float)
                )
            )
            for _, seed_group in group.groupby("seed")
        ]
        rows.append(
            {
                "config_key": config,
                "coverage_cycle": int(cycle),
                "n_patients": primary["n_patients"],
                "n_seeds": int(group.seed.nunique()),
                "patient_median_contact_choice_nll": primary["median"],
                "patient_mean_contact_choice_nll": primary["mean"],
                "patient_sd_contact_choice_nll": primary["sd"],
                "patient_se_contact_choice_nll": (
                    primary["sd"] / np.sqrt(primary["n_patients"])
                    if primary["n_patients"]
                    else float("nan")
                ),
                "patient_median_total_nll": total["median"],
                "patient_median_stop_nll": stop["median"],
                "patient_median_train_validation_gap": gap["median"],
                "seed_patient_median_sd": (
                    float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
                ),
                "seed_patient_median_values": json.dumps(per_seed),
                "cfg_updates_per_patient": int(group.cfg_updates_per_patient.iloc[0]),
                "cfg_hidden_size": int(group.cfg_hidden_size.iloc[0]),
                "cfg_learning_rate": float(group.cfg_learning_rate.iloc[0]),
                "cfg_optimizer": str(group.cfg_optimizer.iloc[0]),
                "cfg_weight_decay": float(group.cfg_weight_decay.iloc[0]),
                "cfg_batch_size": int(group.cfg_batch_size.iloc[0]),
                "cfg_objective": str(group.cfg_objective.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["config_key", "coverage_cycle"])


def _patient_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """One value per (config, cycle, patient): seeds averaged inside a patient."""
    return (
        metrics.groupby(["config_key", "coverage_cycle", "subject", "dataset"])[PRIMARY]
        .mean()
        .reset_index()
        .rename(columns={PRIMARY: "value"})
    )


def _plateau_table(patients: pd.DataFrame) -> pd.DataFrame:
    """Plateau on the patient-median of the *paired* per-patient improvement.

    Budgets are nested and evaluated on the same 34 patients, so the honest
    improvement is computed per patient and only then aggregated.
    """
    rows = []
    for config, group in patients.groupby("config_key"):
        wide = group.pivot_table(
            index="subject", columns="coverage_cycle", values="value"
        )
        cycles = sorted(wide.columns)
        improvements = []
        detail = []
        for previous, current in zip(cycles, cycles[1:]):
            delta = (wide[previous] - wide[current]).dropna()
            improvements.append(float(np.median(delta)))
            detail.append(
                {
                    "from_cycle": int(previous),
                    "to_cycle": int(current),
                    "patient_median_improvement": float(np.median(delta)),
                    "n_patients_improved": int(np.sum(delta > 0)),
                    "n_patients": int(delta.size),
                }
            )
        # ``plateau_verdict`` reads a level series, so rebuild pseudo-levels
        # whose consecutive differences are the paired patient-median gains
        verdict = plateau_verdict(
            [0.0] + list(np.cumsum([-value for value in improvements]))
        )
        rows.append(
            {
                "config_key": config,
                "max_cycle": int(max(cycles)),
                "plateau_reached": verdict["plateau_reached"],
                "plateau_at_cycle": verdict["plateau_at_cycle"],
                "final_improvement": improvements[-1] if improvements else float("nan"),
                "improvements": json.dumps(improvements),
                "improvement_detail": json.dumps(detail),
                "threshold": verdict["threshold"],
            }
        )
    return pd.DataFrame(rows).sort_values("config_key")


def _paired_difference(patients: pd.DataFrame, left: tuple, right: tuple) -> dict:
    """Paired per-patient difference between two (config, cycle) arms."""
    def _arm(arm):
        subset = patients[
            (patients.config_key == arm[0]) & (patients.coverage_cycle == arm[1])
        ]
        return subset.set_index("subject").value

    a, b = _arm(left), _arm(right)
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 3:
        return {"n_patients": len(shared)}
    delta = (a.loc[shared] - b.loc[shared]).to_numpy(float)
    return {
        "n_patients": int(len(shared)),
        "mean_difference": float(np.mean(delta)),
        "median_difference": float(np.median(delta)),
        "se_difference": float(np.std(delta, ddof=1) / np.sqrt(len(shared))),
        "n_worse_than_reference": int(np.sum(delta > 0)),
    }


def _select(patients: pd.DataFrame, summary: pd.DataFrame, budgets: pd.DataFrame) -> dict:
    """Paired one-standard-error rule, then the cheapest optimizer budget.

    Nested budgets share the same patients, so the uncertainty that matters is
    the standard error of the *paired* difference, not the between-patient
    spread of the level.
    """
    merged = summary.merge(budgets, on=["config_key", "coverage_cycle"], how="left")
    best = merged.loc[merged.patient_median_contact_choice_nll.idxmin()]
    best_arm = (str(best.config_key), int(best.coverage_cycle))
    rows = []
    for _, row in merged.iterrows():
        arm = (str(row.config_key), int(row.coverage_cycle))
        paired = _paired_difference(patients, arm, best_arm)
        rows.append(
            {
                **row.to_dict(),
                **{f"paired_{key}": value for key, value in paired.items()},
            }
        )
    table = pd.DataFrame(rows)
    table["within_one_standard_error_of_best"] = (
        table.paired_mean_difference <= table.paired_se_difference
    )
    eligible = table[table.within_one_standard_error_of_best].sort_values(
        ["optimizer_steps_total", "patient_median_contact_choice_nll"]
    )
    selected = eligible.iloc[0]
    return {
        "best_mean_config": {
            "config_key": best_arm[0],
            "coverage_cycle": best_arm[1],
            "patient_median_contact_choice_nll": float(
                best.patient_median_contact_choice_nll
            ),
        },
        "n_within_one_standard_error": int(len(eligible)),
        "arms": table[
            [
                "config_key",
                "coverage_cycle",
                "patient_median_contact_choice_nll",
                "optimizer_steps_total",
                "paired_mean_difference",
                "paired_se_difference",
                "paired_n_worse_than_reference",
                "within_one_standard_error_of_best",
            ]
        ].to_dict("records"),
        "selected": {
            "config_key": str(selected.config_key),
            "coverage_cycle": int(selected.coverage_cycle),
            "updates_per_patient": int(selected.cfg_updates_per_patient),
            "hidden_size": int(selected.cfg_hidden_size),
            "learning_rate": float(selected.cfg_learning_rate),
            "optimizer": str(selected.cfg_optimizer),
            "weight_decay": float(selected.cfg_weight_decay),
            "batch_size": int(selected.cfg_batch_size),
            "objective": str(selected.cfg_objective),
            "optimizer_steps_total": int(selected.optimizer_steps_total),
            "patient_median_contact_choice_nll": float(
                selected.patient_median_contact_choice_nll
            ),
            "seed_patient_median_sd": float(selected.seed_patient_median_sd),
            "paired_mean_difference_vs_best": float(selected.paired_mean_difference),
            "paired_se_difference_vs_best": float(selected.paired_se_difference),
        },
        "selection_rule": (
            "patient-median validation contact-choice NLL; an arm is eligible "
            "when its paired mean difference against the best arm is within one "
            "standard error of that paired difference; the cheapest eligible "
            "optimizer budget wins"
        ),
    }


def _chunk_parity(root: Path) -> dict:
    cells = {}
    for done in sorted(root.rglob("DONE.json")):
        cell = done.parent
        summary = json.loads((cell / "run_summary.json").read_text())
        cells[int(summary["configuration"]["batch_size"])] = cell
    if set(cells) != {512, 1024}:
        raise RuntimeError(f"chunk parity needs batch sizes 512 and 1024, got {sorted(cells)}")
    left = torch.load(cells[512] / "development_checkpoint.pt", map_location="cpu", weights_only=False)
    right = torch.load(cells[1024] / "development_checkpoint.pt", map_location="cpu", weights_only=False)
    final = max(left["cycle_states"])
    deltas = {}
    for key, value in left["cycle_states"][final].items():
        other = right["cycle_states"][final][key]
        deltas[key] = float(torch.max(torch.abs(value - other)))
    offset_delta = max(
        float(torch.max(torch.abs(value - right["cycle_offsets"][final][subject])))
        for subject, value in left["cycle_offsets"][final].items()
    )
    metrics = {}
    for batch, cell in cells.items():
        frame = pd.read_csv(cell / "cycle_patient_metrics.csv")
        frame = frame[frame.coverage_cycle == final]
        metrics[batch] = float(
            np.median(frame.groupby("subject")[PRIMARY].mean().to_numpy(float))
        )
    steps = {
        batch: int(json.loads((cell / "run_summary.json").read_text())["n_optimizer_steps"])
        for batch, cell in cells.items()
    }
    chunks = {
        batch: int(json.loads((cell / "run_summary.json").read_text())["n_backward_chunks"])
        for batch, cell in cells.items()
    }
    tolerance = 1e-4
    return {
        "final_cycle": int(final),
        "max_absolute_parameter_difference": max(deltas.values()),
        "max_absolute_offset_difference": offset_delta,
        "per_parameter_max_difference": deltas,
        "patient_median_contact_choice_nll": metrics,
        "validation_nll_difference": abs(metrics[512] - metrics[1024]),
        "optimizer_steps": steps,
        "backward_chunks": chunks,
        "update_boundaries_identical": steps[512] == steps[1024],
        "chunking_actually_differed": chunks[512] > chunks[1024],
        "tolerance": tolerance,
        "parity_pass": bool(
            steps[512] == steps[1024]
            and chunks[512] > chunks[1024]
            and max(deltas.values()) < tolerance
            and abs(metrics[512] - metrics[1024]) < tolerance
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("b1", "b1x", "b2", "b3"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/topic5_rnn_training_sufficiency_v0_1/analysis",
    )
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "b3":
        parity = _chunk_parity(root)
        (out / "b3_chunk_parity.json").write_text(
            json.dumps(parity, indent=2) + "\n"
        )
        print(json.dumps(parity, indent=2))
        return

    metrics, per_cycle = _load_cells(root)
    metrics["config_key"] = _config_key(metrics)
    per_cycle["config_key"] = _config_key(per_cycle)
    summary = _summarise(metrics)
    patients = _patient_table(metrics)
    plateau = _plateau_table(patients)
    budgets = (
        per_cycle.groupby(["config_key", "coverage_cycle"])
        .agg(
            optimizer_steps_in_cycle=("optimizer_steps_in_cycle", "mean"),
            gradient_clip_fraction=("gradient_clip_fraction", "mean"),
            parameter_update_norm_median=("parameter_update_norm_median", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    budgets["optimizer_steps_total"] = budgets.groupby("config_key")[
        "optimizer_steps_in_cycle"
    ].cumsum().round().astype(int)

    metrics.to_csv(out / f"{args.phase}_cycle_patient_metrics.csv", index=False)
    patients.to_csv(out / f"{args.phase}_patient_values.csv", index=False)
    summary.merge(budgets, on=["config_key", "coverage_cycle"]).to_csv(
        out / f"{args.phase}_config_cycle_summary.csv", index=False
    )
    plateau.to_csv(out / f"{args.phase}_plateau.csv", index=False)

    # B1/B1x select the training budget, so every coverage cycle is a candidate.
    # B2 compares learning rate and optimizer at an already frozen budget, so it
    # must not be able to smuggle a cheaper budget back in through the
    # one-standard-error rule.
    select_at_final_cycle = args.phase == "b2"
    sensitivity = None
    if select_at_final_cycle:
        final = int(summary.coverage_cycle.max())
        summary = summary[summary.coverage_cycle == final]
        patients = patients[patients.coverage_cycle == final]
        budgets = budgets[budgets.coverage_cycle == final]
        # AdamW is the primary optimizer and Adam is a single sensitivity arm,
        # so Adam is reported but never selected
        adamw_keys = set(summary.loc[summary.cfg_optimizer == "adamw", "config_key"])
        sensitivity = (
            summary[summary.cfg_optimizer != "adamw"][
                [
                    "config_key",
                    "cfg_learning_rate",
                    "cfg_optimizer",
                    "cfg_weight_decay",
                    "patient_median_contact_choice_nll",
                    "seed_patient_median_sd",
                ]
            ].to_dict("records")
        )
        summary = summary[summary.config_key.isin(adamw_keys)]
        patients = patients[patients.config_key.isin(adamw_keys)]
        budgets = budgets[budgets.config_key.isin(adamw_keys)]
    selection = _select(patients, summary, budgets)
    selection["selection_restricted_to_final_cycle"] = bool(select_at_final_cycle)
    if sensitivity is not None:
        selection["optimizer_sensitivity_not_eligible_for_selection"] = sensitivity
    selection["phase"] = args.phase
    selection["n_configs"] = int(summary.config_key.nunique())
    selection["plateau"] = plateau.to_dict("records")
    selection["data_read"] = (
        "train80 inner training and inner validation only; outer heldout20 sealed"
    )
    selection["ictal_target_read"] = False
    (out / f"{args.phase}_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
