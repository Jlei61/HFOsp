#!/usr/bin/env python3
"""Patient-first inference for the formal v2.2 random-axis specificity gate."""
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
CHUNKS = tuple((start, start + 32) for start in range(0, 256, 32))


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


def _bootstrap_median_ci(
    values: np.ndarray, seed: int = 20260726
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def _collect(
    subjects: list[str], seeds: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    missing = []
    root = BASE / "formal/claim3_random_axis_runs"
    for subject in subjects:
        for seed in seeds:
            observed: dict[int, tuple[float, float]] = {}
            bias_hashes = set()
            learned_values = set()
            for start, stop in CHUNKS:
                chunk = f"chunk_{start:03d}_{stop - 1:03d}"
                run = root / subject / f"seed_{seed}" / chunk
                metrics_path = run / "metrics.json"
                if not metrics_path.is_file() or not (run / "COMPLETE").is_file():
                    missing.append(f"{subject}/seed_{seed}/{chunk}")
                    continue
                record = json.loads(metrics_path.read_text(encoding="utf-8"))
                if record.get("target_values_read") is not False:
                    raise RuntimeError(f"{subject}/seed_{seed}/{chunk}: target read")
                indices = list(map(int, record["direction_indices"]))
                random_nll = list(map(float, record["heldout20_random_next_nll"]))
                delta = list(map(float, record["random_minus_learned"]))
                if not (len(indices) == len(random_nll) == len(delta) == stop - start):
                    raise RuntimeError(f"{subject}/seed_{seed}/{chunk}: shape drift")
                if indices != list(range(start, stop)):
                    raise RuntimeError(f"{subject}/seed_{seed}/{chunk}: index drift")
                for index, nll, benefit in zip(indices, random_nll, delta):
                    if index in observed:
                        raise RuntimeError(
                            f"{subject}/seed_{seed}: duplicate direction {index}"
                        )
                    observed[index] = (nll, benefit)
                bias_hashes.add(str(record["node_bias_sha256"]))
                learned_values.add(float(record["heldout20_learned_next_nll"]))
            if len(observed) != 256:
                continue
            if len(bias_hashes) != 1 or len(learned_values) != 1:
                raise RuntimeError(f"{subject}/seed_{seed}: control contract drift")
            ordered = np.asarray([observed[index] for index in range(256)])
            learned = next(iter(learned_values))
            rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "learned_next_nll": learned,
                    "median_random_next_nll": float(np.median(ordered[:, 0])),
                    "delta_random_minus_learned": float(
                        np.median(ordered[:, 0]) - learned
                    ),
                    "fraction_random_worse_than_learned": float(
                        np.mean(ordered[:, 0] > learned)
                    ),
                    "random_nll_q025": float(np.quantile(ordered[:, 0], 0.025)),
                    "random_nll_q975": float(np.quantile(ordered[:, 0], 0.975)),
                    "node_bias_sha256": next(iter(bias_hashes)),
                }
            )
    return pd.DataFrame(rows), missing


def _plot(patient: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    values = patient["seed_median_delta_random_minus_learned"].to_numpy()
    order = np.argsort(values)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    jitter = np.linspace(-0.12, 0.12, len(values))
    axes[0].scatter(
        jitter,
        values,
        s=29,
        color=np.where(values > 0, "#4477AA", "#CC6677"),
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].plot(
        [-0.24, 0.24],
        [np.median(values), np.median(values)],
        color="black",
        lw=1.8,
    )
    axes[0].axhline(0, color="#777777", ls="--", lw=0.9)
    axes[0].set_xticks([])
    axes[0].set_ylabel("Random-axis NLL − learned-axis NLL")
    axes[0].set_title("Patient-first specificity")

    ordered = values[order]
    axes[1].barh(
        np.arange(len(ordered)),
        ordered,
        color=np.where(ordered > 0, "#4477AA", "#CC6677"),
    )
    axes[1].axvline(0, color="#777777", lw=0.9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("NLL difference")
    axes[1].set_title("22 physical-axis patients")
    fig.tight_layout()
    fig.savefig(
        figures / "claim3_random_axis_specificity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    (figures / "README.md").write_text(
        "### claim3_random_axis_specificity.png\n\n"
        "每位患者先在每个 seed 内比较 learned axis 与 256 个固定随机方向的 heldout "
        "event-first normalized next-set NLL，再在 3 个 seed 间取中位数。正值表示"
        "随机方向的预测更差，即 learned axis 具有方向特异性。\n\n"
        "**关注点**：患者中位数是否大于 0、正值患者是否过半，以及效应是否由少数"
        "患者驱动。\n",
        encoding="utf-8",
    )


def main() -> None:
    claim2 = json.loads(
        (BASE / "formal/analysis/CLAIM2_STATUS.json").read_text(encoding="utf-8")
    )
    if not claim2.get("next_stage_allowed"):
        raise SystemExit("Claim 3 remains locked by Claim 2")
    lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = list(map(str, lock["subjects"]))
    seeds = list(map(int, lock["seeds"]))
    rows, missing = _collect(subjects, seeds)
    analysis = BASE / "formal/analysis"
    expected_chunks = len(subjects) * len(seeds) * len(CHUNKS)
    if missing:
        atomic_json(
            analysis / "CLAIM3_STATUS.json",
            {
                "status": "pending",
                "complete_chunks": expected_chunks - len(missing),
                "expected_chunks": expected_chunks,
                "missing": missing,
                "target_values_read": False,
            },
        )
        print(
            f"Claim 3 pending: {expected_chunks-len(missing)}/{expected_chunks}"
        )
        return
    if len(rows) != len(subjects) * len(seeds):
        raise RuntimeError("Claim-3 seed grid is not complete")
    rows.to_csv(analysis / "claim3_random_axis_seed_metrics.csv", index=False)
    patient = (
        rows.groupby("subject", as_index=False)
        .agg(
            seed_median_learned_next_nll=("learned_next_nll", "median"),
            seed_median_random_next_nll=("median_random_next_nll", "median"),
            seed_median_delta_random_minus_learned=(
                "delta_random_minus_learned",
                "median",
            ),
            seed_median_fraction_random_worse=(
                "fraction_random_worse_than_learned",
                "median",
            ),
        )
    )
    patient.to_csv(
        analysis / "claim3_random_axis_specificity.csv", index=False
    )
    values = patient["seed_median_delta_random_minus_learned"].to_numpy()
    pvalue = _one_sided_wilcoxon(values)
    ci_low, ci_high = _bootstrap_median_ci(values)
    n_positive = int(np.sum(values > 0))
    passed = bool(
        np.median(values) > 0
        and n_positive > len(values) / 2
        and pvalue < 0.05
    )
    status = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        "claim3_random_axis": "PASS" if passed else "FAIL",
        "n_patients": len(values),
        "directions_per_patient_per_seed": 256,
        "seed_aggregation": "median within patient",
        "median_delta_random_minus_learned": float(np.median(values)),
        "median_ci95_low": ci_low,
        "median_ci95_high": ci_high,
        "n_positive": n_positive,
        "fraction_positive": float(np.mean(values > 0)),
        "wilcoxon_one_sided_p": pvalue,
        "next_stage_allowed": passed,
        "target_values_read": False,
    }
    atomic_json(analysis / "CLAIM3_STATUS.json", status)
    _plot(patient, analysis / "figures_claim3")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
