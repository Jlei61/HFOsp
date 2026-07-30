#!/usr/bin/env python3
"""Aggregate Stage-A folds at seed-within-patient then patient level."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_interictal_operator import (  # noqa: E402
    fit_empirical_template_baseline,
    pairwise_rank_concordance,
    prefix_targets,
)

FORMAL_GATE_PATIENTS = 13
FORMAL_GATE_SEEDS = 3


def _set_nll(scores: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    scores = np.maximum(np.asarray(scores, float), 1e-12)
    numerator = float(np.sum(scores[np.asarray(target, bool)]))
    denominator = float(np.sum(scores[np.asarray(valid, bool)]))
    return (
        float(-np.log(numerator / denominator))
        if numerator > 0 and denominator > 0
        else np.nan
    )


def empirical_template_metrics(dataset_dir: Path, subject: str) -> dict:
    path = dataset_dir / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as z:
        groups = np.asarray(z["event_group_ids"], int)
        split = np.asarray(z["event_split"], int)
    baseline = fit_empirical_template_baseline(groups[split == 0])
    nll, concordance = [], []
    for event in groups[split == 1]:
        n_groups = int(np.max(event[event >= 0]) + 1)
        for tau in range(1, n_groups + 1):
            target = prefix_targets(event, tau)
            score, utility = baseline.scores(event, tau)
            if not target["terminal"]:
                nll.append(
                    _set_nll(score, target["next_set"], ~target["recruited"])
                )
            value = pairwise_rank_concordance(utility, target["suffix_group"])
            if np.isfinite(value):
                concordance.append(value)
    return {
        "empirical_template_next_set_nll": float(np.mean(nll)),
        "empirical_template_suffix_concordance": float(np.mean(concordance)),
    }


def _load_run(run_dir: Path, dataset_dir: Path) -> dict:
    done_path = run_dir / "DONE.json"
    metrics_path = run_dir / "heldout_metrics.csv"
    manifest_path = run_dir / "run_manifest.json"
    if not (done_path.exists() and metrics_path.exists() and manifest_path.exists()):
        raise FileNotFoundError(f"incomplete Stage-A run: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    if bool(manifest.get("ictal_target_opened", True)):
        raise RuntimeError(f"ictal-target leakage flag in {run_dir}")
    frame = pd.read_csv(metrics_path)
    by_control = {
        str(row.control): row._asdict() for row in frame.itertuples(index=False)
    }
    required = {
        "true_order_core",
        "rank_shuffle_core",
        "unordered_deepsets",
        "matched_feedforward_contact_query",
    }
    missing = sorted(required - set(by_control))
    if missing:
        raise RuntimeError(f"{run_dir}: missing controls {missing}")
    true = by_control["true_order_core"]
    subject = str(true["subject"])
    empirical = empirical_template_metrics(dataset_dir, subject)

    static_nll = {
        "support": float(true["support_next_set_nll"]),
        "markov": float(true["markov_next_set_nll"]),
        "empirical_template": empirical["empirical_template_next_set_nll"],
        "unordered_deepsets": float(
            by_control["unordered_deepsets"]["model_next_set_nll"]
        ),
        "matched_feedforward": float(
            by_control["matched_feedforward_contact_query"]["model_next_set_nll"]
        ),
    }
    static_concordance = {
        "support": float(true["support_suffix_concordance"]),
        "markov": float(true["markov_suffix_concordance"]),
        "empirical_template": empirical[
            "empirical_template_suffix_concordance"
        ],
        "unordered_deepsets": float(
            by_control["unordered_deepsets"]["model_suffix_concordance"]
        ),
        "matched_feedforward": float(
            by_control["matched_feedforward_contact_query"][
                "model_suffix_concordance"
            ]
        ),
    }
    strongest_nll_name = min(static_nll, key=static_nll.get)
    strongest_concordance_name = max(
        static_concordance, key=static_concordance.get
    )
    model_nll = float(true["model_next_set_nll"])
    model_concordance = float(true["model_suffix_concordance"])
    shuffle_nll = float(by_control["rank_shuffle_core"]["model_next_set_nll"])
    shuffle_concordance = float(
        by_control["rank_shuffle_core"]["model_suffix_concordance"]
    )
    return {
        "run_dir": str(run_dir),
        "subject": subject,
        "seed": int(manifest["seed"]),
        "hidden_size": int(manifest["model_kwargs"]["hidden_size"]),
        "model_next_set_nll": model_nll,
        "model_suffix_concordance": model_concordance,
        "strongest_static_nll_name": strongest_nll_name,
        "strongest_static_next_set_nll": static_nll[strongest_nll_name],
        "strongest_static_concordance_name": strongest_concordance_name,
        "strongest_static_suffix_concordance": static_concordance[
            strongest_concordance_name
        ],
        "rank_shuffle_next_set_nll": shuffle_nll,
        "rank_shuffle_suffix_concordance": shuffle_concordance,
        "next_gain_vs_static": static_nll[strongest_nll_name] - model_nll,
        "suffix_gain_vs_static": model_concordance
        - static_concordance[strongest_concordance_name],
        "next_gain_vs_rank_shuffle": shuffle_nll - model_nll,
        "suffix_gain_vs_rank_shuffle": model_concordance - shuffle_concordance,
        **{f"{key}_next_set_nll": value for key, value in static_nll.items()},
        **{
            f"{key}_suffix_concordance": value
            for key, value in static_concordance.items()
        },
    }


def _bootstrap_ci(
    values: np.ndarray, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    sampled = values[rng.integers(0, values.size, size=(int(draws), values.size))]
    statistic = np.median(sampled, axis=1)
    return tuple(np.quantile(statistic, [0.025, 0.975]).astype(float))


def _wilcoxon_greater(values: Iterable[float]) -> float:
    values = np.asarray(list(values), float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.allclose(values, 0):
        return np.nan
    return float(wilcoxon(values, alternative="greater").pvalue)


def _coverage_status(
    subject: pd.DataFrame,
    failures: list[dict],
    expected_patients: int,
    expected_seeds: int,
) -> tuple[bool, bool]:
    """Separate invocation completeness from the frozen formal-gate contract."""
    n_patients = int(subject.subject.nunique())
    seed_floor = int(subject.n_seeds.min())
    requested_coverage_met = bool(
        n_patients == int(expected_patients)
        and seed_floor >= int(expected_seeds)
        and not failures
    )
    formal_gate_eligible = bool(
        n_patients == FORMAL_GATE_PATIENTS
        and seed_floor >= FORMAL_GATE_SEEDS
        and not failures
    )
    return requested_coverage_met, formal_gate_eligible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_operator_static_readout/dataset_v0_3",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--expected-patients", type=int, default=13)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help=(
            "Aggregate one preselected hidden size only. If multiple hidden "
            "sizes are supplied without this flag, fail closed."
        ),
    )
    args = parser.parse_args()
    dataset_dir = (
        args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [path if path.is_absolute() else ROOT / path for path in args.runs]
    rows, failures = [], []
    for run_dir in run_dirs:
        try:
            rows.append(_load_run(run_dir, dataset_dir))
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "reason": f"{type(exc).__name__}:{exc}"})
    cell = pd.DataFrame(rows)
    if cell.empty:
        raise RuntimeError(f"no complete run cell; failures={failures}")
    if args.hidden_size is not None:
        cell = cell[cell.hidden_size == int(args.hidden_size)].copy()
        if cell.empty:
            raise RuntimeError(
                f"no complete run uses hidden_size={int(args.hidden_size)}"
            )
    hidden_sizes = sorted(cell.hidden_size.unique().astype(int).tolist())
    if len(hidden_sizes) != 1:
        raise RuntimeError(
            "Stage-A gate cannot mix hidden sizes. Select hidden size from "
            f"target-free inner validation first; found {hidden_sizes}"
        )
    duplicate = cell.duplicated(
        subset=["subject", "seed", "hidden_size"], keep=False
    )
    if duplicate.any():
        duplicated_cells = (
            cell.loc[duplicate, ["subject", "seed", "hidden_size"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "duplicate subject-seed-hidden run cells are not allowed: "
            f"{duplicated_cells}"
        )
    cell.to_csv(out_dir / "stage_a_cell_metrics.csv", index=False)
    metrics = [
        "next_gain_vs_static",
        "suffix_gain_vs_static",
        "next_gain_vs_rank_shuffle",
        "suffix_gain_vs_rank_shuffle",
    ]
    subject = (
        cell.groupby("subject", as_index=False)[metrics]
        .median()
        .sort_values("subject")
    )
    seed_counts = cell.groupby("subject").seed.nunique()
    subject["n_seeds"] = subject.subject.map(seed_counts).astype(int)
    subject.to_csv(out_dir / "stage_a_subject_metrics.csv", index=False)

    rng = np.random.default_rng(20260724)
    summary_metrics = {}
    all_ci_positive = True
    for metric in metrics:
        values = subject[metric].to_numpy(float)
        low, high = _bootstrap_ci(values, args.bootstrap_draws, rng)
        summary_metrics[metric] = {
            "patient_median": float(np.nanmedian(values)),
            "bootstrap_95ci": [low, high],
            "wilcoxon_greater_p": _wilcoxon_greater(values),
            "n_patients": int(np.sum(np.isfinite(values))),
        }
        all_ci_positive &= bool(np.isfinite(low) and low > 0)
    requested_coverage_met, formal_eligible = _coverage_status(
        subject=subject,
        failures=failures,
        expected_patients=int(args.expected_patients),
        expected_seeds=int(args.expected_seeds),
    )
    verdict = {
        "n_run_cells": int(len(cell)),
        "n_patients": int(subject.subject.nunique()),
        "seed_count_min": int(subject.n_seeds.min()),
        "expected_patients": int(args.expected_patients),
        "expected_seeds": int(args.expected_seeds),
        "requested_coverage_met": requested_coverage_met,
        "formal_contract_patients": FORMAL_GATE_PATIENTS,
        "formal_contract_seeds": FORMAL_GATE_SEEDS,
        "hidden_size": int(hidden_sizes[0]),
        "formal_gate_eligible": formal_eligible,
        "metrics": summary_metrics,
        "event_dynamics_gate_pass": bool(formal_eligible and all_ci_positive),
        "scientific_status": (
            "pass"
            if formal_eligible and all_ci_positive
            else "fail"
            if formal_eligible
            else "pilot_only_not_formal"
        ),
        "failed_or_incomplete_runs": failures,
        "ictal_target_opened": False,
    }
    (out_dir / "stage_a_gate_summary.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
