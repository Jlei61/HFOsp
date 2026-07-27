#!/usr/bin/env python3
"""Aggregate v2.2 development runs and freeze the objective when eligible."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SIMPLICITY_ORDER = {
    "next_only": 0,
    "next_plus_rollout_h3": 1,
    "next_plus_rollout_h5": 2,
}
HORIZON = {
    "next_only": 0,
    "next_plus_rollout_h3": 3,
    "next_plus_rollout_h5": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def _collect(
    root: Path, subjects: list[str], objectives: list[str], seeds: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for subject in subjects:
        for objective in objectives:
            for seed in seeds:
                run_dir = root / subject / objective / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                complete_path = run_dir / "COMPLETE"
                key = f"{subject}/{objective}/seed_{seed}"
                if not metrics_path.is_file() or not complete_path.is_file():
                    missing.append(key)
                    continue
                record = json.loads(metrics_path.read_text(encoding="utf-8"))
                if record.get("status") != "complete" or record.get("smoke"):
                    missing.append(key)
                    continue
                if not record.get("full_control_bias_identical"):
                    raise RuntimeError(f"{key}: full/control node bias differs")
                full = record["models"]["full"]
                isotropic = record["models"]["local_isotropic"]
                for partition in ("fit60", "validation20", "confirmation20"):
                    full_metric = full["metrics"][partition]
                    iso_metric = isotropic["metrics"][partition]
                    rows.append(
                        {
                            "subject": subject,
                            "objective": objective,
                            "H_train": HORIZON[objective],
                            "seed": seed,
                            "partition": partition,
                            "n_events": int(full_metric["n_events"]),
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
                            "full_best_epoch": int(full["best_epoch"]),
                            "isotropic_best_epoch": int(isotropic["best_epoch"]),
                            "node_bias_sha256": str(record["node_bias_sha256"]),
                            "peak_rss_gb": float(record["resource"]["peak_rss_gb"]),
                            "peak_cuda_allocated_gb": float(
                                record["resource"]["peak_cuda_allocated_gb"]
                            ),
                            "finite": bool(
                                full_metric["finite"] and iso_metric["finite"]
                            ),
                            "metrics_sha256": sha256(metrics_path),
                        }
                    )
    return pd.DataFrame(rows), missing


def _objective_table(inventory: pd.DataFrame) -> pd.DataFrame:
    validation = inventory[inventory.partition == "validation20"].copy()
    per_patient = (
        validation.groupby(["subject", "objective"], as_index=False)
        .agg(
            patient_seed_median_future_benefit=("future_benefit", "median"),
            patient_seed_median_next_benefit=("next_benefit", "median"),
            patient_seed_median_isotropic_future_nll=(
                "isotropic_future_nll",
                "median",
            ),
        )
    )
    rows = []
    for objective, frame in per_patient.groupby("objective"):
        rows.append(
            {
                "objective": objective,
                "H_train": HORIZON[objective],
                "n_patients": len(frame),
                "patient_median_future_benefit": float(
                    np.median(frame.patient_seed_median_future_benefit)
                ),
                "patient_median_next_benefit": float(
                    np.median(frame.patient_seed_median_next_benefit)
                ),
                "patient_median_isotropic_future_nll": float(
                    np.median(frame.patient_seed_median_isotropic_future_nll)
                ),
                "next_nonworse": bool(
                    np.median(frame.patient_seed_median_next_benefit) >= 0.0
                ),
                "per_patient_future_benefit": json.dumps(
                    dict(
                        zip(
                            frame.subject,
                            frame.patient_seed_median_future_benefit,
                        )
                    ),
                    sort_keys=True,
                ),
                "per_patient_next_benefit": json.dumps(
                    dict(
                        zip(
                            frame.subject,
                            frame.patient_seed_median_next_benefit,
                        )
                    ),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("H_train").reset_index(drop=True)


def _select_objective(table: pd.DataFrame) -> dict[str, Any]:
    eligible = table[table.next_nonworse].copy()
    if eligible.empty:
        return {
            "status": "development_instability",
            "selected_objective": None,
            "reason": "full next-set NLL is worse than local-isotropic for every objective",
        }
    best_score = float(eligible.patient_median_future_benefit.max())
    reference_scale = float(
        eligible.loc[
            eligible.patient_median_future_benefit.idxmax(),
            "patient_median_isotropic_future_nll",
        ]
    )
    tie_tolerance = 0.005 * abs(reference_scale)
    eligible["within_tie"] = (
        best_score - eligible.patient_median_future_benefit
    ) <= tie_tolerance
    selected_row = eligible[eligible.within_tie].sort_values("H_train").iloc[0]
    return {
        "status": "selected_pending_confirmation",
        "selected_objective": str(selected_row.objective),
        "H_train": int(selected_row.H_train),
        "best_future_benefit": best_score,
        "selected_future_benefit": float(
            selected_row.patient_median_future_benefit
        ),
        "selected_next_benefit": float(selected_row.patient_median_next_benefit),
        "tie_tolerance_absolute_nll": tie_tolerance,
        "tie_rule": (
            "0.5% of development patient-median local-isotropic future NLL; "
            "within tie choose next_only, then h3, then h5"
        ),
    }


def _confirmation(
    inventory: pd.DataFrame, selected: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = inventory[
        (inventory.partition == "confirmation20")
        & (inventory.objective == selected)
    ].copy()
    patient = (
        frame.groupby("subject", as_index=False)
        .agg(
            seed_median_future_benefit=("future_benefit", "median"),
            seed_median_next_benefit=("next_benefit", "median"),
            seed_median_full_future_nll=("full_future_nll", "median"),
            seed_median_isotropic_future_nll=("isotropic_future_nll", "median"),
            seed_median_full_next_nll=("full_next_nll", "median"),
            seed_median_isotropic_next_nll=("isotropic_next_nll", "median"),
        )
    )
    summary = {
        "n_patients": len(patient),
        "patient_median_future_benefit": float(
            np.median(patient.seed_median_future_benefit)
        ),
        "patient_median_next_benefit": float(
            np.median(patient.seed_median_next_benefit)
        ),
        "future_direction_not_reversed": bool(
            np.median(patient.seed_median_future_benefit) >= 0.0
        ),
        "next_direction_not_reversed": bool(
            np.median(patient.seed_median_next_benefit) >= 0.0
        ),
        "confirmation_used_for_model_changes": False,
    }
    summary["pass"] = bool(
        summary["future_direction_not_reversed"]
        and summary["next_direction_not_reversed"]
    )
    return patient, summary


def _plot(
    inventory: pd.DataFrame,
    objective_table: pd.DataFrame,
    confirmation: pd.DataFrame,
    selected: str,
    figures: Path,
) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    validation = inventory[inventory.partition == "validation20"]
    patient = (
        validation.groupby(["subject", "objective"], as_index=False)
        .agg(future_benefit=("future_benefit", "median"))
    )
    objectives = objective_table.objective.tolist()
    labels = ["next", "next + H3", "next + H5"]
    colors = ["#4477AA", "#66CCEE", "#228833"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for index, objective in enumerate(objectives):
        values = patient.loc[
            patient.objective == objective, "future_benefit"
        ].to_numpy()
        axes[0].scatter(
            np.full_like(values, index, dtype=float),
            values,
            s=35,
            color=colors[index],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        axes[0].plot(
            [index - 0.22, index + 0.22],
            [np.median(values)] * 2,
            color="black",
            lw=1.6,
        )
    axes[0].axhline(0, color="#777777", lw=0.9, ls="--")
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Future NLL benefit\n(isotropic − full)")
    axes[0].set_title("Validation objective selection")

    y = np.arange(len(confirmation))
    axes[1].barh(
        y,
        confirmation.seed_median_future_benefit,
        color=np.where(
            confirmation.seed_median_future_benefit >= 0,
            "#228833",
            "#CC6677",
        ),
    )
    axes[1].axvline(0, color="#777777", lw=0.9)
    axes[1].set_yticks(y, confirmation.subject.str.replace("_", " "))
    axes[1].set_xlabel("Future NLL benefit\n(isotropic − full)")
    axes[1].set_title(f"Frozen confirmation: {selected}")
    fig.tight_layout()
    output = figures / "development_objective_and_confirmation.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### development_objective_and_confirmation.png\n\n"
        "左图比较三种预注册训练目标在三位开发患者 validation20 上的轴向模型相对"
        "局部各向同性模型的 future-order NLL 增益；点为患者，横线为患者中位数。"
        "右图只展示按预注册规则冻结后的目标在未参与选择的 confirmation20 上是否"
        "保持同方向。\n\n"
        "**关注点**：开发选择是否依赖单一患者，以及 confirmation 是否发生方向反转。\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_symmetric_axis_propagation_state_v2_2.yaml",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    subjects = list(map(str, cfg["cohort"]["development"]))
    objectives = list(map(str, cfg["model"]["objectives"]))
    seeds = list(map(int, cfg["optimizer"]["seeds"]))
    output = ROOT / cfg["outputs"]["root"] / "development"
    inventory, missing = _collect(output / "runs", subjects, objectives, seeds)
    expected = len(subjects) * len(objectives) * len(seeds)
    if missing:
        atomic_json(
            output / "DEVELOPMENT_STATUS.json",
            {
                "status": "pending",
                "expected_runs": expected,
                "complete_runs": expected - len(missing),
                "missing": missing,
                "ictal_target_values_read": False,
            },
        )
        print(f"development pending: {expected - len(missing)}/{expected}")
        return
    if len(inventory) != expected * 3 or not inventory.finite.all():
        raise RuntimeError("development inventory is incomplete or non-finite")
    inventory.to_csv(output / "run_inventory.csv", index=False)
    objective_table = _objective_table(inventory)
    objective_table.to_csv(output / "objective_comparison.csv", index=False)
    selection = _select_objective(objective_table)
    if selection["selected_objective"] is None:
        atomic_json(
            output / "DEVELOPMENT_STATUS.json",
            {
                **selection,
                "expected_runs": expected,
                "complete_runs": expected,
                "ictal_target_values_read": False,
            },
        )
        print(json.dumps(selection, indent=2))
        return
    confirmation, confirmation_summary = _confirmation(
        inventory, selection["selected_objective"]
    )
    confirmation.to_csv(output / "confirmation_metrics.csv", index=False)
    _plot(
        inventory,
        objective_table,
        confirmation,
        selection["selected_objective"],
        output / "figures",
    )
    lock_status = (
        "pass" if confirmation_summary["pass"] else "development_instability"
    )
    payload = {
        "contract": cfg["contract"]["name"],
        "version": cfg["contract"]["version"],
        "status": lock_status,
        "selected_objective": selection["selected_objective"],
        "H_train": selection["H_train"],
        "H_eval": "remaining eligible contacts at each true prefix",
        "H_transfer": "N_contacts - N_clinical_onset_source_contacts",
        "selection": selection,
        "confirmation": confirmation_summary,
        "shared_parameters": ["anisotropy_ratio", "rho_p", "c0", "c_p", "c_n"],
        "patient_parameters": ["axis_u", "gamma", "gain"],
        "optimizer": cfg["optimizer"],
        "aggregation": (
            "eligible-contact normalization -> within-event decision mean -> "
            "patient-seed event mean -> seed median -> patient inference"
        ),
        "two_w_margin_fraction": cfg["statistics"]["two_w_margin_fraction"],
        "development_subjects": subjects,
        "input_config_hashes": {
            "config": sha256(config_path),
            "core": sha256(
                ROOT / "src/topic5_symmetric_axis_propagation_state_v2_2.py"
            ),
            "trainer": sha256(
                ROOT
                / "scripts/train_topic5_symmetric_axis_propagation_state_v2_2.py"
            ),
            "analyzer": sha256(Path(__file__)),
        },
        "git_commit": git_commit(),
        "confirmation_used_for_model_changes": False,
        "ictal_target_values_read": False,
    }
    filename = (
        "DEVELOPMENT_LOCK.json"
        if lock_status == "pass"
        else "DEVELOPMENT_STATUS.json"
    )
    atomic_json(output / filename, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
