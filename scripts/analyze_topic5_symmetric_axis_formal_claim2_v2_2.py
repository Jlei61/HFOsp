#!/usr/bin/env python3
"""Patient-first formal inference for v2.2 Claim 2."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _one_sided_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.all(values == 0):
        return 1.0
    return float(
        wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )


def _bh_fdr(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def _bootstrap_median_ci(values: np.ndarray, seed: int = 20260726) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def _collect(subjects: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    expected_set = set(subjects)
    root = BASE / "formal/claim2_runs"
    for subject in subjects:
        for seed in (17, 29, 43):
            run = root / subject / f"seed_{seed}"
            metrics_path = run / "metrics.json"
            if not metrics_path.is_file() or not (run / "COMPLETE").is_file():
                missing.append(f"{subject}/seed_{seed}")
                continue
            record = json.loads(metrics_path.read_text(encoding="utf-8"))
            training = set(record["shared_training_subjects"])
            if (
                len(training) != 21
                or subject in training
                or training != expected_set - {subject}
            ):
                raise RuntimeError(f"{subject}/seed_{seed}: LOSO leakage")
            if not record.get("full_control_bias_identical"):
                raise RuntimeError(f"{subject}/seed_{seed}: node-bias mismatch")
            full = record["models"]["full"]["heldout_fit"]
            isotropic = record["models"]["local_isotropic"]["heldout_fit"]
            full_metric = full["metrics"]["heldout20"]
            iso_metric = isotropic["metrics"]["heldout20"]
            if not full_metric["finite"] or not iso_metric["finite"]:
                raise RuntimeError(f"{subject}/seed_{seed}: non-finite metric")
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "n_heldout_events": int(full_metric["n_events"]),
                    "full_next_nll": float(full_metric["next_nll"]),
                    "isotropic_next_nll": float(iso_metric["next_nll"]),
                    "next_benefit": float(
                        iso_metric["next_nll"] - full_metric["next_nll"]
                    ),
                    "full_future_nll": float(full_metric["future_nll"]),
                    "isotropic_future_nll": float(iso_metric["future_nll"]),
                    "future_benefit": float(
                        iso_metric["future_nll"] - full_metric["future_nll"]
                    ),
                    "axis_x": float(full["parameters"]["axis"][0]),
                    "axis_y": float(full["parameters"]["axis"][1]),
                    "axis_z": float(full["parameters"]["axis"][2]),
                    "gamma": float(full["parameters"]["gamma"]),
                    "gain": float(full["parameters"]["gain"]),
                    "anisotropy_ratio": float(
                        full["parameters"]["anisotropy_ratio"]
                    ),
                    "rho_p": float(full["parameters"]["rho_p"]),
                    "node_bias_sha256": str(record["node_bias_sha256"]),
                    "target_values_read": bool(record["target_values_read"]),
                }
            )
    return pd.DataFrame(rows), missing


def _plot(patient: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    endpoints = [
        ("seed_median_next_benefit", "Next-set NLL"),
        ("seed_median_future_benefit", "Future-order NLL"),
    ]
    colors = ["#4477AA", "#228833"]
    for index, (column, label) in enumerate(endpoints):
        values = patient[column].to_numpy()
        jitter = np.linspace(-0.10, 0.10, len(values))
        axes[0].scatter(
            np.full(len(values), index) + jitter,
            values,
            s=25,
            color=colors[index],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        median = float(np.median(values))
        axes[0].plot(
            [index - 0.24, index + 0.24],
            [median, median],
            color="black",
            lw=1.8,
        )
    axes[0].axhline(0, color="#777777", ls="--", lw=0.9)
    axes[0].set_xticks([0, 1], [label for _, label in endpoints])
    axes[0].set_ylabel("Heldout benefit (isotropic − full)")
    axes[0].set_title("Patient-first axis increment")

    order = np.argsort(patient.seed_median_future_benefit.to_numpy())
    sorted_patient = patient.iloc[order]
    colors_bar = np.where(
        sorted_patient.seed_median_future_benefit >= 0,
        "#228833",
        "#CC6677",
    )
    axes[1].barh(
        np.arange(len(sorted_patient)),
        sorted_patient.seed_median_future_benefit,
        color=colors_bar,
    )
    axes[1].axvline(0, color="#777777", lw=0.9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Future-order NLL benefit")
    axes[1].set_title("22 LOSO patients")
    fig.tight_layout()
    fig.savefig(
        figures / "claim2_full_vs_isotropic.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### claim2_full_vs_isotropic.png\n\n"
        "左图展示 22 位 physical-axis formal 患者中，对称轴模型相对局部各向同性"
        "模型的 heldout next-set 与 future-order NLL 改善；每点先在患者内取 3 个"
        " seed 的中位数。右图按患者排序展示 future-order 改善方向。\n\n"
        "**关注点**：患者中位数是否大于 0、改善患者是否过半，以及效应是否由少数"
        "患者驱动。\n",
        encoding="utf-8",
    )


def main() -> None:
    physical_lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = list(map(str, physical_lock["subjects"]))
    if len(subjects) != 22:
        raise SystemExit("physical-axis lock does not contain 22 patients")
    rows, missing = _collect(subjects)
    analysis = BASE / "formal/analysis"
    if missing:
        atomic_json(
            analysis / "CLAIM2_STATUS.json",
            {
                "status": "pending",
                "complete_runs": 66 - len(missing),
                "expected_runs": 66,
                "missing": missing,
                "target_values_read": False,
            },
        )
        print(f"Claim 2 pending: {66-len(missing)}/66")
        return
    if len(rows) != 66 or rows.target_values_read.any():
        raise RuntimeError("formal grid incomplete or target seal violated")
    rows.to_csv(analysis / "claim2_seed_metrics.csv", index=False)
    patient = (
        rows.groupby("subject", as_index=False)
        .agg(
            n_heldout_events=("n_heldout_events", "first"),
            seed_median_full_next_nll=("full_next_nll", "median"),
            seed_median_isotropic_next_nll=("isotropic_next_nll", "median"),
            seed_median_next_benefit=("next_benefit", "median"),
            seed_median_full_future_nll=("full_future_nll", "median"),
            seed_median_isotropic_future_nll=("isotropic_future_nll", "median"),
            seed_median_future_benefit=("future_benefit", "median"),
            seed_median_gamma=("gamma", "median"),
            seed_median_gain=("gain", "median"),
            seed_median_anisotropy_ratio=("anisotropy_ratio", "median"),
            seed_median_rho_p=("rho_p", "median"),
        )
    )
    patient.to_csv(analysis / "claim2_patient_metrics.csv", index=False)

    summaries = []
    pvalues = []
    for endpoint, column in (
        ("next_set", "seed_median_next_benefit"),
        ("future_first_arrival", "seed_median_future_benefit"),
    ):
        values = patient[column].to_numpy()
        pvalue = _one_sided_wilcoxon(values)
        pvalues.append(pvalue)
        ci_low, ci_high = _bootstrap_median_ci(values)
        summaries.append(
            {
                "endpoint": endpoint,
                "n_patients": len(values),
                "median_benefit": float(np.median(values)),
                "median_ci95_low": ci_low,
                "median_ci95_high": ci_high,
                "n_positive": int(np.sum(values > 0)),
                "fraction_positive": float(np.mean(values > 0)),
                "wilcoxon_one_sided_p": pvalue,
            }
        )
    qvalues = _bh_fdr(pvalues)
    for summary, qvalue in zip(summaries, qvalues):
        summary["bh_fdr_q"] = qvalue
        summary["pass"] = bool(
            summary["median_benefit"] > 0
            and summary["n_positive"] > summary["n_patients"] / 2
            and qvalue < 0.05
        )
    endpoint_table = pd.DataFrame(summaries)
    endpoint_table.to_csv(analysis / "claim2_axis_increment.csv", index=False)
    status = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        "claim2_next": "PASS" if summaries[0]["pass"] else "FAIL",
        "claim2_future": "PASS" if summaries[1]["pass"] else "FAIL",
        "endpoints": summaries,
        "formal_cohort_n": 22,
        "seed_aggregation": "median within patient",
        "cohort_test": "one-sided patient-level Wilcoxon with BH-FDR across 2 endpoints",
        "target_values_read": False,
        "next_stage_allowed": bool(all(summary["pass"] for summary in summaries)),
    }
    atomic_json(analysis / "CLAIM2_STATUS.json", status)
    _plot(patient, analysis / "figures")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
