#!/usr/bin/env python3
"""Aggregate held-out next-contact performance for Figure 6 panel C.

The primary endpoint excludes STOP and cardinality.  Every patient is the
unit of inference; seeds are reduced within patient before cohort statistics.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_patient_specific_rnn_bridge import chronological_60_20_20  # noqa: E402
from src.topic5_shared_scaffold_rnn import (  # noqa: E402
    batched_exact_conditional_k_subset_log_probability,
    estimate_node_hazard_bias,
)


MODELS = ("static", "ordinary_gru", "structured", "structured_rank_shuffle")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


@torch.no_grad()
def static_contact_metrics(
    groups: np.ndarray,
    counts: np.ndarray,
    indices: np.ndarray,
    bias: np.ndarray,
    *,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Evaluate the fit60 conditional-hazard prior on matched test decisions."""

    total_nll = 0.0
    n_continue = 0
    top1_hits = 0
    for start in range(0, len(indices), int(batch_size)):
        selected = np.asarray(indices[start : start + int(batch_size)], dtype=np.int64)
        event = torch.as_tensor(groups[selected], dtype=torch.long)
        event_count = torch.as_tensor(counts[selected], dtype=torch.long)
        logits = torch.as_tensor(bias, dtype=torch.float32).expand(len(selected), -1)
        seen = torch.zeros_like(event, dtype=torch.bool)
        for step in range(int(event_count.max().item()) - 1):
            seen |= event == step
            active = event_count > step + 1
            target = event == step + 1
            eligible = ~seen
            log_probability = batched_exact_conditional_k_subset_log_probability(
                node_logits=logits,
                eligible=eligible,
                next_set=target,
                active=active,
            )
            total_nll += float((-log_probability).sum().item())
            rows = torch.where(active)[0]
            if rows.numel():
                masked_logits = logits.masked_fill(~eligible, -torch.inf)
                predicted = torch.argmax(masked_logits, dim=1)
                top1_hits += int(target[rows, predicted[rows]].sum().item())
                n_continue += int(rows.numel())
    if not n_continue:
        raise RuntimeError("static test split has no continuation decisions")
    return {
        "contact_nll_per_continue_decision": total_nll / n_continue,
        "top1_next_contact_accuracy": top1_hits / n_continue,
        "top1_hits": top1_hits,
        "n_continue_decisions": n_continue,
        "n_events": int(len(indices)),
    }


def _bootstrap_median(values: np.ndarray, *, seed: int, n_boot: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(values, size=(int(n_boot), len(values)), replace=True)
    medians = np.median(sampled, axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def paired_summary(
    patient: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    metric: str,
    lower_is_better: bool,
    seed: int,
    tie_atol: float = 1.0e-9,
) -> dict[str, Any]:
    wide = patient.pivot(index="subject", columns="model", values=metric)
    if model_a not in wide or model_b not in wide:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "status": "NOT_YET_AVAILABLE",
            "n": 0,
        }
    pivot = wide.dropna(subset=[model_a, model_b])
    if pivot.empty:
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "status": "NOT_YET_AVAILABLE",
            "n": 0,
        }
    # Positive means model_a is better, independent of metric direction.
    delta = (
        pivot[model_b].to_numpy(float) - pivot[model_a].to_numpy(float)
        if lower_is_better
        else pivot[model_a].to_numpy(float) - pivot[model_b].to_numpy(float)
    )
    delta[np.abs(delta) <= float(tie_atol)] = 0.0
    nonzero = delta[delta != 0]
    if len(nonzero):
        try:
            p_two_sided = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
            p_greater = float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
        except ValueError:
            p_two_sided = p_greater = 1.0
    else:
        p_two_sided = p_greater = 1.0
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "status": "COMPLETE",
        "positive_means": f"{model_a}_better",
        "n": int(len(delta)),
        "subjects": pivot.index.astype(str).tolist(),
        "delta": delta.tolist(),
        "median_delta": float(np.median(delta)),
        "bootstrap_95ci": _bootstrap_median(delta, seed=seed),
        "wilcoxon_two_sided_p": p_two_sided,
        "wilcoxon_greater_p": p_greater,
        "n_positive": int(np.count_nonzero(delta > 0)),
        "n_negative": int(np.count_nonzero(delta < 0)),
        "n_tied": int(np.count_nonzero(delta == 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text())
    dataset_root = Path(config["dataset_artifact_root"]).resolve() / config["dataset_root"]
    output = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    records = load_records(dataset_root)
    expected_seeds = list(map(int, config["training"]["seeds"]))
    rows: list[dict[str, Any]] = []
    for done_path in sorted((output / "per_subject").glob("*/*/seed_*/DONE.json")):
        done = json.loads(done_path.read_text())
        if done.get("status") != "COMPLETE" or done.get("smoke"):
            continue
        rows.append(
            {
                "subject": str(done["subject"]),
                "model": str(done["model"]),
                "seed": int(done["seed"]),
                "dataset": str(records[str(done["subject"])].dataset),
                "n_contacts": int(done["n_contacts"]),
                "n_test_events": int(done["n_events"]["test20"]),
                "contact_nll": float(done["test"]["contact_nll_per_continue_decision"]),
                "top1_accuracy": float(done["test"]["top1_next_contact_accuracy"]),
                "cardinality_nll": float(done["test"]["cardinality_nll_per_continue_decision"]),
                "stop_nll": float(done["test"]["stop_nll_per_decision"]),
                "best_cycle": int(done["best_cycle"]),
                "runtime_seconds": float(done["runtime_seconds"]),
                "peak_gpu_memory_mb": float(done["peak_gpu_memory_mb"]),
            }
        )
    seed_frame = pd.DataFrame(rows)
    expected_units = len(records) * len(config["models"]["names"]) * len(expected_seeds)
    if len(seed_frame) != expected_units and not args.allow_incomplete:
        raise RuntimeError(f"formal training incomplete: {len(seed_frame)}/{expected_units}")

    present_subjects = sorted(set(seed_frame.subject)) if len(seed_frame) else []
    for subject in present_subjects:
        record = records[subject]
        fit60, _, test20 = chronological_60_20_20(record)
        hazard = estimate_node_hazard_bias(
            np.asarray(record.group_ids)[fit60],
            pseudocount=float(config["training"]["hazard_pseudocount"]),
        )
        metrics = static_contact_metrics(
            np.asarray(record.group_ids),
            np.asarray(record.group_count),
            test20,
            np.asarray(hazard["bias"], dtype=np.float32),
        )
        rows.append(
            {
                "subject": subject,
                "model": "static",
                "seed": -1,
                "dataset": str(record.dataset),
                "n_contacts": int(len(record.contact_names)),
                "n_test_events": int(len(test20)),
                "contact_nll": float(metrics["contact_nll_per_continue_decision"]),
                "top1_accuracy": float(metrics["top1_next_contact_accuracy"]),
                "cardinality_nll": np.nan,
                "stop_nll": np.nan,
                "best_cycle": 0,
                "runtime_seconds": 0.0,
                "peak_gpu_memory_mb": 0.0,
            }
        )
    full_seed_frame = pd.DataFrame(rows).sort_values(["subject", "model", "seed"])
    patient = (
        full_seed_frame.groupby(["subject", "dataset", "model"], as_index=False)
        .median(numeric_only=True)
        .sort_values(["subject", "model"])
    )
    output.mkdir(parents=True, exist_ok=True)
    full_seed_frame.to_csv(output / "interictal_seed_metrics.csv", index=False)
    patient.to_csv(output / "interictal_patient_metrics.csv", index=False)

    # The three patients used for the learning-rate audit are development
    # data.  The 31 remaining patients never influenced any frozen choice, so
    # they carry the confirmation; all 34 are reported as cohort description.
    development = sorted(map(str, config["development_lr_audit"]["subjects"]))
    confirmation = patient[~patient.subject.isin(development)]

    def _comparisons(frame: pd.DataFrame, seed_base: int) -> dict[str, Any]:
        built: dict[str, Any] = {}
        for metric, lower in (("contact_nll", True), ("top1_accuracy", False)):
            for index, comparator in enumerate(
                ("ordinary_gru", "static", "structured_rank_shuffle")
            ):
                built[f"structured_vs_{comparator}__{metric}"] = paired_summary(
                    frame,
                    model_a="structured",
                    model_b=comparator,
                    metric=metric,
                    lower_is_better=lower,
                    seed=seed_base + index + (0 if metric == "contact_nll" else 100),
                )
        return built

    comparisons = _comparisons(patient, 22_000)
    confirmation_comparisons = _comparisons(confirmation, 23_000)
    summary = {
        "contract": config["contract"],
        "primary_endpoint": "test20 contact NLL conditioned on continue and observed cardinality",
        "target_values_read": False,
        "n_subjects": int(patient.subject.nunique()),
        "n_seed_units": int(len(seed_frame)),
        "expected_seed_units": int(expected_units),
        "complete": bool(len(seed_frame) == expected_units),
        "models": list(MODELS),
        "patient_medians": {
            model: {
                "contact_nll": _finite_or_none(group.contact_nll.median()),
                "top1_accuracy": _finite_or_none(group.top1_accuracy.median()),
            }
            for model, group in patient.groupby("model")
        },
        "comparisons": comparisons,
        "development_subjects": development,
        "n_confirmation_subjects": int(confirmation.subject.nunique()),
        "confirmation_comparisons": confirmation_comparisons,
    }
    atomic_json(output / "interictal_cohort_statistics.json", summary)
    print(json.dumps({"status": "COMPLETE", **{k: summary[k] for k in ("n_subjects", "n_seed_units", "complete")}}))


if __name__ == "__main__":
    main()
