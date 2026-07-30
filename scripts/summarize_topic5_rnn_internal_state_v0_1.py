#!/usr/bin/env python3
"""Aggregate target-blind internal-state evidence with patient-first statistics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray, seed: int, alternative: str = "two-sided") -> dict:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"n": 0}
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(data, size=(20_000, len(data)), replace=True)
    p = (
        1.0
        if np.allclose(data, 0.0)
        else float(wilcoxon(data, alternative=alternative).pvalue)
    )
    return {
        "n": int(len(data)),
        "median": float(np.median(data)),
        "bootstrap_ci95": np.quantile(
            np.median(sampled, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(data > 0)),
        "wilcoxon_p": p,
        "alternative": alternative,
    }


def collect_pca() -> pd.DataFrame:
    rows = []
    for path in sorted(
        (BASE / "interictal/cells").glob("seed_*/**/CELL_STATUS.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete extraction cell: {path}")
        for metric in payload["metrics"]:
            rows.append(
                {
                    "subject": payload["subject"],
                    "seed_dir": payload["seed_dir"],
                    **metric,
                }
            )
    frame = pd.DataFrame(rows)
    if frame[["subject", "seed_dir"]].drop_duplicates().shape[0] != 102:
        raise RuntimeError("PCA extraction does not contain 102 cells")
    return frame


def collect_subject_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = []
    probes = []
    for directory in sorted((BASE / "interictal/per_subject").glob("*")):
        if not directory.is_dir():
            continue
        status = json.loads(
            (directory / "ANALYSIS_STATUS.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete subject analysis: {directory}")
        stability.append(pd.read_csv(directory / "stability_metrics.csv"))
        probes.append(pd.read_csv(directory / "probe_metrics.csv"))
    if len(stability) != 34:
        raise RuntimeError(f"expected 34 subject analyses, found {len(stability)}")
    return pd.concat(stability, ignore_index=True), pd.concat(
        probes, ignore_index=True
    )


def collect_perturbations() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = []
    direction = []
    contact = []
    for directory in sorted(
        (BASE / "interictal/perturbation_cells").glob("seed_*/**")
    ):
        if not directory.is_dir() or not (directory / "CELL_STATUS.json").exists():
            continue
        status = json.loads(
            (directory / "CELL_STATUS.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete perturbation cell: {directory}")
        order.append(pd.read_csv(directory / "order_perturbation_metrics.csv"))
        direction.append(
            pd.read_csv(directory / "direction_perturbation_metrics.csv")
        )
        contact.append(pd.read_csv(directory / "direction_contact_fields.csv"))
    if len(order) != 102:
        raise RuntimeError(f"expected 102 perturbation cells, found {len(order)}")
    return (
        pd.concat(order, ignore_index=True),
        pd.concat(direction, ignore_index=True),
        pd.concat(contact, ignore_index=True),
    )


def contact_field_stability(contact: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = contact.loc[
        np.isclose(contact.amplitude_sd, 0.5) & (contact.event_half == "all")
    ].copy()
    keys = [
        "subject",
        "control",
        "direction_type",
        "direction_index",
        "amplitude_sd",
    ]
    for key, group in selected.groupby(keys):
        wide = group.pivot(
            index="contact_name", columns="seed_dir", values="probability_contrast"
        ).dropna()
        correlations = []
        for left_index, left in enumerate(SEED_DIRS):
            for right in SEED_DIRS[left_index + 1 :]:
                correlations.append(
                    float(spearmanr(wide[left], wide[right]).statistic)
                )
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_contacts": int(len(wide)),
                "median_pairwise_seed_spearman": float(np.nanmedian(correlations)),
                "minimum_pairwise_seed_spearman": float(np.nanmin(correlations)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not (BASE / "EXTRACTION_DONE.json").exists():
        raise SystemExit("hidden-state extraction is incomplete")
    if not (BASE / "SUBJECT_ANALYSIS_DONE.json").exists():
        raise SystemExit("subject analysis is incomplete")
    if not (BASE / "PERTURBATION_ANALYSIS_DONE.json").exists():
        raise SystemExit("perturbation analysis is incomplete")

    pca = collect_pca()
    stability, probes = collect_subject_tables()
    order, direction, contact = collect_perturbations()
    field_stability = contact_field_stability(contact)
    pca.to_csv(BASE / "interictal_pca_cell_metrics.csv", index=False)
    stability.to_csv(BASE / "interictal_stability_metrics.csv", index=False)
    probes.to_csv(BASE / "interictal_probe_metrics.csv", index=False)
    order.to_csv(BASE / "interictal_order_perturbation_metrics.csv", index=False)
    direction.to_csv(
        BASE / "interictal_direction_perturbation_metrics.csv", index=False
    )
    contact.to_csv(BASE / "interictal_direction_contact_fields.csv", index=False)
    field_stability.to_csv(
        BASE / "interictal_direction_field_seed_stability.csv", index=False
    )

    cohort: dict[str, dict] = {}
    inventory = pca.loc[pca.metric == "pca_inventory"].copy()
    collapsed_inventory = (
        inventory.groupby(["subject", "control"], as_index=False)
        .agg(
            effective_rank=("value", "median"),
            k80=("k80", "median"),
            k90=("k90", "median"),
            k95=("k95", "median"),
        )
    )
    for control, group in collapsed_inventory.groupby("control"):
        for metric in ("effective_rank", "k80", "k90", "k95"):
            cohort[f"{control}__{metric}"] = summary(
                group[metric].to_numpy(float), 2026072800 + len(cohort)
            )

    reconstruction = pca.loc[pca.metric == "pca_reconstruction"].copy()
    collapsed_reconstruction = (
        reconstruction.groupby(["subject", "control", "k"], as_index=False)
        .agg(
            variance_fidelity=("value", "median"),
            nll_loss=("nll_loss", "median"),
        )
    )
    for (control, k), group in collapsed_reconstruction.groupby(["control", "k"]):
        cohort[f"{control}__pca_k{k}_variance_fidelity"] = summary(
            group.variance_fidelity.to_numpy(float), 2026072900 + len(cohort)
        )
        cohort[f"{control}__pca_k{k}_nll_loss"] = summary(
            group.nll_loss.to_numpy(float),
            2026072900 + len(cohort),
            alternative="greater",
        )

    collapsed_stability = (
        stability.groupby(
            ["subject", "control", "comparison", "k"], as_index=False
        )
        .value.median()
    )
    for key, group in collapsed_stability.groupby(
        ["control", "comparison", "k"]
    ):
        control, comparison, k = key
        cohort[f"{control}__{comparison}__k{k}"] = summary(
            group.value.to_numpy(float), 2026073000 + len(cohort)
        )

    probe_collapsed = (
        probes.groupby(
            ["subject", "task", "feature", "k", "metric"], as_index=False
        )
        .value.median()
    )
    probe_benefit_rows = []
    for (task, metric), task_group in probe_collapsed.groupby(["task", "metric"]):
        baseline = task_group.loc[
            task_group.feature == "unordered", ["subject", "value"]
        ].rename(columns={"value": "baseline"})
        for (feature, k), group in task_group.loc[
            task_group.feature.str.startswith("unordered_plus")
        ].groupby(["feature", "k"]):
            merged = baseline.merge(
                group[["subject", "value"]], on="subject", how="inner"
            )
            merged["benefit"] = merged["baseline"] - merged["value"]
            for row in merged.itertuples():
                probe_benefit_rows.append(
                    {
                        "subject": row.subject,
                        "task": task,
                        "metric": metric,
                        "feature": feature,
                        "k": int(k),
                        "benefit_over_unordered": float(row.benefit),
                    }
                )
            cohort[f"{task}__{feature}__k{k}__benefit_over_unordered"] = summary(
                merged.benefit.to_numpy(float),
                2026073100 + len(cohort),
                alternative="greater",
            )
    probe_benefits = pd.DataFrame(probe_benefit_rows)
    probe_benefits.to_csv(
        BASE / "interictal_probe_patient_benefits.csv", index=False
    )

    if len(probe_benefits):
        wide = probe_benefits.pivot_table(
            index=["subject", "task", "metric", "k"],
            columns="feature",
            values="benefit_over_unordered",
        ).reset_index()
        full_name = "unordered_plus_full_hidden"
        shuffle_name = "unordered_plus_rank_shuffle_hidden"
        common = wide.dropna(subset=[full_name, shuffle_name]).copy()
        common["full_minus_rank_shuffle_increment"] = (
            common[full_name] - common[shuffle_name]
        )
        common.to_csv(
            BASE / "interictal_probe_full_vs_rank_shuffle.csv", index=False
        )
        for key, group in common.groupby(["task", "metric", "k"]):
            task, metric, k = key
            cohort[
                f"{task}__k{k}__full_minus_rank_shuffle_increment"
            ] = summary(
                group.full_minus_rank_shuffle_increment.to_numpy(float),
                2026073200 + len(cohort),
                alternative="greater",
            )

    order_collapsed = (
        order.groupby(
            [
                "subject",
                "control",
                "order_perturbation",
                "prefix_bin",
                "metric",
            ],
            as_index=False,
        )
        .value.median()
    )
    for key, group in order_collapsed.groupby(
        ["control", "order_perturbation", "prefix_bin", "metric"]
    ):
        control, perturbation, prefix_bin, metric = key
        cohort[
            f"{control}__order_{perturbation}__{prefix_bin}__{metric}"
        ] = summary(
            group.value.to_numpy(float),
            2026073300 + len(cohort),
            alternative="greater" if metric in {"nll_loss", "js_divergence"} else "two-sided",
        )

    direction_collapsed = (
        direction.groupby(
            [
                "subject",
                "control",
                "direction_type",
                "direction_index",
                "amplitude_sd",
            ],
            as_index=False,
        )
        .agg(
            mean_js_plus=("mean_js_plus", "median"),
            mean_js_minus=("mean_js_minus", "median"),
            stop_probability_plus_minus=(
                "stop_probability_plus_minus",
                "median",
            ),
        )
    )
    for key, group in direction_collapsed.groupby(
        ["control", "direction_type", "direction_index", "amplitude_sd"]
    ):
        control, direction_type, direction_index, amplitude = key
        cohort[
            f"{control}__{direction_type}{direction_index}__amp{amplitude}__mean_js"
        ] = summary(
            0.5
            * (group.mean_js_plus.to_numpy(float) + group.mean_js_minus.to_numpy(float)),
            2026073400 + len(cohort),
            alternative="greater",
        )
    for key, group in field_stability.groupby(
        ["control", "direction_type", "direction_index"]
    ):
        control, direction_type, direction_index = key
        cohort[
            f"{control}__{direction_type}{direction_index}__field_seed_spearman"
        ] = summary(
            group.median_pairwise_seed_spearman.to_numpy(float),
            2026073500 + len(cohort),
            alternative="greater",
        )

    atomic_json(
        BASE / "INTERICTAL_SUMMARY.json",
        {
            "contract": "topic5_rnn_internal_state_reduction_v0_1",
            "status": "COMPLETE",
            "n_subjects": 34,
            "n_seeds": 3,
            "patient_first_seed_collapse": "median",
            "cohort_metrics": cohort,
            "target_values_read": False,
            "early_ictal_arrays_deserialized": False,
        },
    )
    freeze = {
        "contract": "topic5_rnn_internal_state_reduction_v0_1_interictal_freeze",
        "status": "FROZEN",
        "n_subjects": 34,
        "n_seeds": 3,
        "directions": {
            "pca": "top two train60 variance directions",
            "output_coupled": (
                "top two right singular vectors of centered contact decoder loading"
            ),
            "sign": (
                "positive correlation with train80 interictal participation; "
                "no ictal target used"
            ),
            "primary_amplitude_sd": 0.5,
        },
        "fixed_target_readouts_after_freeze": [
            "participation",
            "endpoint_joint_mass",
        ],
        "omnibus_sensitivity": "five frozen fields with reselection inside null",
        "summary_sha256": sha256(BASE / "INTERICTAL_SUMMARY.json"),
        "target_values_read": False,
        "early_ictal_arrays_deserialized": False,
    }
    atomic_json(BASE / "INTERICTAL_FREEZE.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
