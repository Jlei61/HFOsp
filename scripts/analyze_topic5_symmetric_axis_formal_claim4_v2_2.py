#!/usr/bin/env python3
"""Patient-first inference for formal v2.2 shared-scaffold Claim 4."""
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
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


def _bootstrap_median_ci(
    values: np.ndarray, seed: int = 20260726
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def _collect(
    subjects: list[str], seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rows = []
    inventory = []
    missing = []
    root = BASE / "formal/claim4_shared_scaffold_runs"
    for subject in subjects:
        for seed in seeds:
            run = root / subject / f"seed_{seed}"
            path = run / "metrics.json"
            if not path.is_file() or not (run / "COMPLETE").is_file():
                missing.append(f"{subject}/seed_{seed}")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("target_values_read") is not False:
                raise RuntimeError(f"{subject}/seed_{seed}: target read")
            base = {
                "subject": subject,
                "seed": seed,
                "analysis_status": str(record["status"]),
                **{
                    key: int(value)
                    for key, value in record["source_side_counts"].items()
                },
                "q25": float(record["q25"]),
                "q75": float(record["q75"]),
                "node_bias_sha256": str(record["node_bias_sha256"]),
            }
            inventory.append(base)
            if record["status"] != "complete":
                continue
            rows.append(
                {
                    **base,
                    "left_axis_benefit": float(record["left_axis_benefit"]),
                    "right_axis_benefit": float(record["right_axis_benefit"]),
                    "delta_two": float(record["delta_two"]),
                    "delta_axis": float(record["delta_axis"]),
                    "M": float(record["M"]),
                    "cross_left_to_right_isotropic_benefit": float(
                        record["cross_left_to_right_isotropic_benefit"]
                    ),
                    "cross_right_to_left_isotropic_benefit": float(
                        record["cross_right_to_left_isotropic_benefit"]
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(inventory), missing


def _plot(patient: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    left = patient["seed_median_left_axis_benefit"].to_numpy()
    right = patient["seed_median_right_axis_benefit"].to_numpy()
    margin = patient["seed_median_M"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for index, (values, label, color) in enumerate(
        [(left, "Source-left", "#4477AA"), (right, "Source-right", "#EE7733")]
    ):
        jitter = np.linspace(-0.10, 0.10, len(values))
        axes[0].scatter(
            index + jitter,
            values,
            s=27,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0].plot(
            [index - 0.23, index + 0.23],
            [np.median(values), np.median(values)],
            color="black",
            lw=1.7,
        )
    axes[0].axhline(0, color="#777777", ls="--", lw=0.9)
    axes[0].set_xticks([0, 1], ["Source-left", "Source-right"])
    axes[0].set_ylabel("Isotropic − shared-W NLL")
    axes[0].set_title("Cross-direction generalization")

    order = np.argsort(margin)
    ordered = margin[order]
    axes[1].barh(
        np.arange(len(ordered)),
        ordered,
        color=np.where(ordered < 0, "#228833", "#CC6677"),
    )
    axes[1].axvline(0, color="#777777", lw=0.9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("M = two-W gain − 0.1 × axis gain")
    axes[1].set_title("Direction-specific operator margin")
    fig.tight_layout()
    fig.savefig(
        figures / "claim4_shared_scaffold.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### claim4_shared_scaffold.png\n\n"
        "左图分别展示同一个冻结 scaffold 在 train80 Q25/Q75 定义的 heldout "
        "source-left 和 source-right 事件上，相对局部各向同性模型的 NLL 改善。"
        "右图检验分别拟合两侧传播强度后，其增益是否小于轴向结构增益的 10%。\n\n"
        "**关注点**：左右两侧是否都为正，以及 M 的患者 bootstrap 95% 上界是否"
        "低于 0。\n",
        encoding="utf-8",
    )


def main() -> None:
    claim3 = json.loads(
        (BASE / "formal/analysis/CLAIM3_STATUS.json").read_text(encoding="utf-8")
    )
    if not claim3.get("next_stage_allowed"):
        raise SystemExit("Claim 4 remains locked by Claim 3")
    lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = list(map(str, lock["subjects"]))
    seeds = list(map(int, lock["seeds"]))
    rows, inventory, missing = _collect(subjects, seeds)
    analysis = BASE / "formal/analysis"
    if missing:
        atomic_json(
            analysis / "CLAIM4_STATUS.json",
            {
                "status": "pending",
                "complete_runs": len(subjects) * len(seeds) - len(missing),
                "expected_runs": len(subjects) * len(seeds),
                "missing": missing,
                "target_values_read": False,
            },
        )
        print(
            f"Claim 4 pending: {len(subjects)*len(seeds)-len(missing)}/"
            f"{len(subjects)*len(seeds)}"
        )
        return
    inventory.to_csv(analysis / "claim4_source_side_inventory.csv", index=False)
    eligible_seed_counts = rows.groupby("subject").size()
    eligible_subjects = eligible_seed_counts[eligible_seed_counts == len(seeds)].index
    eligible = rows[rows.subject.isin(eligible_subjects)].copy()
    eligible.to_csv(analysis / "claim4_seed_metrics.csv", index=False)
    if len(eligible_subjects) < len(subjects) / 2:
        status = {
            "contract": "topic5_symmetric_axis_propagation_state_rnn",
            "version": "2.2",
            "status": "not_estimable",
            "claim4_shared_scaffold": "NOT_ESTIMABLE",
            "n_eligible_patients": int(len(eligible_subjects)),
            "required_minimum": int(np.ceil(len(subjects) / 2)),
            "next_stage_allowed": False,
            "target_values_read": False,
        }
        atomic_json(analysis / "CLAIM4_STATUS.json", status)
        print(json.dumps(status, indent=2))
        return

    patient = (
        eligible.groupby("subject", as_index=False)
        .agg(
            seed_median_left_axis_benefit=("left_axis_benefit", "median"),
            seed_median_right_axis_benefit=("right_axis_benefit", "median"),
            seed_median_delta_two=("delta_two", "median"),
            seed_median_delta_axis=("delta_axis", "median"),
            seed_median_M=("M", "median"),
            seed_median_cross_left_to_right_benefit=(
                "cross_left_to_right_isotropic_benefit",
                "median",
            ),
            seed_median_cross_right_to_left_benefit=(
                "cross_right_to_left_isotropic_benefit",
                "median",
            ),
        )
    )
    patient.to_csv(analysis / "claim4_shared_scaffold.csv", index=False)
    side_summaries = []
    pvalues = []
    for side, column in (
        ("source_left", "seed_median_left_axis_benefit"),
        ("source_right", "seed_median_right_axis_benefit"),
    ):
        values = patient[column].to_numpy()
        pvalue = _one_sided_wilcoxon(values)
        pvalues.append(pvalue)
        ci_low, ci_high = _bootstrap_median_ci(values)
        side_summaries.append(
            {
                "side": side,
                "n_patients": len(values),
                "median_benefit": float(np.median(values)),
                "median_ci95_low": ci_low,
                "median_ci95_high": ci_high,
                "n_positive": int(np.sum(values > 0)),
                "fraction_positive": float(np.mean(values > 0)),
                "wilcoxon_one_sided_p": pvalue,
            }
        )
    for summary, qvalue in zip(side_summaries, _bh_fdr(pvalues)):
        summary["bh_fdr_q"] = qvalue
        summary["pass"] = bool(
            summary["median_benefit"] > 0
            and summary["n_positive"] > summary["n_patients"] / 2
            and qvalue < 0.05
        )
    pd.DataFrame(side_summaries).to_csv(
        analysis / "claim4_side_summary.csv", index=False
    )

    margin = patient["seed_median_M"].to_numpy()
    margin_ci_low, margin_ci_high = _bootstrap_median_ci(margin)
    twow_pass = bool(margin_ci_high < 0)
    passed = bool(all(item["pass"] for item in side_summaries) and twow_pass)
    status = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        "claim4_shared_scaffold": "PASS" if passed else "FAIL",
        "n_eligible_patients": len(patient),
        "required_minimum": int(np.ceil(len(subjects) / 2)),
        "side_tests": side_summaries,
        "twoW_noninferiority": {
            "median_M": float(np.median(margin)),
            "bootstrap_ci95_low": margin_ci_low,
            "bootstrap_ci95_high": margin_ci_high,
            "pass": twow_pass,
        },
        "cross_side_transfer_role": "secondary_nonblocking",
        "next_stage_allowed": passed,
        "target_values_read": False,
    }
    atomic_json(analysis / "CLAIM4_STATUS.json", status)
    _plot(patient, analysis / "figures_claim4")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
