#!/usr/bin/env python3
"""Patient-level interictal analysis for full-tissue LBSS v0.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
OLD_OUT = Path("results/topic5_lbss_rnn_v0_2")
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
LABELS = {
    "L0_LOCAL_ONLY": "Local",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL": "+ local",
    "L2_LOCAL_PLUS_RANDOM_LR": "+ random",
    "L3_LOCAL_PLUS_LEARNED_LR": "+ selected",
    "C_L3_ORDER_SHUFFLED": "Shuffle",
}
COLORS = {
    "L0_LOCAL_ONLY": "#777d82",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL": "#6c96a6",
    "L2_LOCAL_PLUS_RANDOM_LR": "#9b8b72",
    "L3_LOCAL_PLUS_LEARNED_LR": "#b84b4b",
    "C_L3_ORDER_SHUFFLED": "#b8b8b8",
}


def paired_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    tolerance = 1e-9
    nonzero = values[np.abs(values) > tolerance]
    if not len(nonzero):
        two = greater = 1.0
    else:
        two = float(wilcoxon(nonzero, alternative="two-sided", method="auto").pvalue)
        greater = float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else float("nan"),
        "n_positive": int(np.sum(values > tolerance)),
        "n_negative": int(np.sum(values < -tolerance)),
        "n_tied": int(np.sum(np.abs(values) <= tolerance)),
        "wilcoxon_p_two_sided": two,
        "wilcoxon_p_greater": greater,
    }


def holm(values: list[float]) -> list[float]:
    raw = np.asarray(values, float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(raw) - rank) * raw[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def load_units(out: Path) -> pd.DataFrame:
    rows = []
    paths = sorted((out / "per_fit").glob("*/*/seed*/metrics.json"))
    if len(paths) != 465:
        raise RuntimeError(f"expected 465 formal units, found {len(paths)}")
    for path in paths:
        value = json.loads(path.read_text())
        if value.get("target_values_read") is not False:
            raise RuntimeError(f"target contamination marker: {path}")
        if not value.get("best_checkpoint_eligible") or not value.get("converged"):
            raise RuntimeError(f"ineligible formal unit: {path}")
        bins = value["distance_bins"]
        rows.append({
            "fit_id": value["fit_id"],
            "subject": value["subject"],
            "scope": value["scope"],
            "arm": value["arm"],
            "seed": int(value["seed"]),
            "test_contact_nll": value["test"]["contact_nll"],
            "test_top1": value["test"]["top1"],
            "local_contact_nll": bins["local"]["contact_nll"],
            "intermediate_contact_nll": bins["intermediate"]["contact_nll"],
            "distal_contact_nll": bins["distal"]["contact_nll"],
            "local_n": bins["local"]["n"],
            "intermediate_n": bins["intermediate"]["n"],
            "distal_n": bins["distal"]["n"],
            "rollout_spearman": value["rollout"]["seed_removed_spearman_median"],
            "rollout_length_ratio": value["rollout"]["length_ratio_median"],
            "n_nodes": value["n_nodes"],
            "n_epochs": value["n_epochs"],
            "seconds": value["seconds"],
        })
    return pd.DataFrame(rows)


def aggregate_patient(units: pd.DataFrame, old_root: Path) -> pd.DataFrame:
    metrics = [
        "test_contact_nll", "test_top1", "local_contact_nll",
        "intermediate_contact_nll", "distal_contact_nll", "local_n",
        "intermediate_n", "distal_n", "rollout_spearman",
        "rollout_length_ratio", "n_nodes", "n_epochs", "seconds",
    ]
    fit = units.groupby(["subject", "fit_id", "scope", "arm"], sort=False)[metrics].median().reset_index()
    patient = fit.groupby(["subject", "arm"], sort=False)[metrics].mean().reset_index()
    old = pd.read_csv(old_root / "interictal_per_patient.csv")
    no_rec = old.groupby("subject", sort=False).no_rec_contact_nll.first()
    patient["no_rec_contact_nll"] = patient.subject.map(no_rec)
    if patient.no_rec_contact_nll.isna().any():
        raise RuntimeError("v0.2 no-recurrence comparator does not cover the v0.3 cohort")
    return patient


def contrasts(patient: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pivot = patient.pivot(index="subject", columns="arm")
    rows = []
    definitions = {
        "L0_vs_no_rec_all": (
            pivot["no_rec_contact_nll"]["L0_LOCAL_ONLY"]
            - pivot["test_contact_nll"]["L0_LOCAL_ONLY"]
        ),
        "L3_vs_L0_all": pivot["test_contact_nll"]["L0_LOCAL_ONLY"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_all": pivot["test_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2_all": pivot["test_contact_nll"]["L2_LOCAL_PLUS_RANDOM_LR"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_shuffle_all": pivot["test_contact_nll"]["C_L3_ORDER_SHUFFLED"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L0_distal": pivot["distal_contact_nll"]["L0_LOCAL_ONLY"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_distal": pivot["distal_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2_distal": pivot["distal_contact_nll"]["L2_LOCAL_PLUS_RANDOM_LR"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_shuffle_distal": pivot["distal_contact_nll"]["C_L3_ORDER_SHUFFLED"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
    }
    summary = {}
    for label, series in definitions.items():
        summary[label] = paired_summary(series.to_numpy())
        for subject, value in series.items():
            rows.append({"subject": subject, "contrast": label, "gain_nats": value})
    for family in (
        ("L3_vs_L0_all", "L3_vs_L1_all", "L3_vs_L2_all"),
        ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2_distal"),
    ):
        adjusted = holm([summary[key]["wilcoxon_p_greater"] for key in family])
        for key, value in zip(family, adjusted):
            summary[key]["holm_p_greater_within_claim"] = value
    return pd.DataFrame(rows), summary


def geometry_sensitivity(patient: pd.DataFrame, old_root: Path) -> tuple[pd.DataFrame, dict]:
    old = pd.read_csv(old_root / "interictal_per_patient.csv")
    new_pivot = patient.pivot(index="subject", columns="arm")
    old_pivot = old.pivot(index="subject", columns="arm")
    subjects = new_pivot.index.intersection(old_pivot.index)
    rows = []
    summaries = {}
    for endpoint in ("test_contact_nll", "distal_contact_nll"):
        for comparator in ("L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L2_LOCAL_PLUS_RANDOM_LR"):
            label = f"geometry_change_L3_vs_{comparator}_{endpoint}"
            new_gain = new_pivot[endpoint][comparator].loc[subjects] - new_pivot[endpoint]["L3_LOCAL_PLUS_LEARNED_LR"].loc[subjects]
            old_gain = old_pivot[endpoint][comparator].loc[subjects] - old_pivot[endpoint]["L3_LOCAL_PLUS_LEARNED_LR"].loc[subjects]
            delta = new_gain - old_gain
            summaries[label] = paired_summary(delta.to_numpy())
            for subject in subjects:
                rows.append({
                    "subject": subject, "endpoint": endpoint,
                    "comparator": comparator, "v0_2_gain": old_gain[subject],
                    "v0_3_gain": new_gain[subject], "v0_3_minus_v0_2": delta[subject],
                })
    return pd.DataFrame(rows), summaries


def plot(patient: pd.DataFrame, contrast_rows: pd.DataFrame, sensitivity: pd.DataFrame, out: Path) -> None:
    figure_dir = out / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), gridspec_kw={"wspace": 0.48})
    pivot = patient.pivot(index="subject", columns="arm")

    axis = axes[0]
    order = list(ARMS)
    for subject in pivot.index:
        values = [pivot["test_contact_nll"][arm][subject] for arm in order]
        axis.plot(np.arange(len(order)), values, color="#c7cbcd", lw=0.55, alpha=0.7)
    medians = [pivot["test_contact_nll"][arm].median() for arm in order]
    axis.plot(np.arange(len(order)), medians, color="#161616", lw=2.0)
    axis.scatter(np.arange(len(order)), medians, c=[COLORS[arm] for arm in order], s=30, zorder=3)
    axis.set_xticks(np.arange(len(order)), [LABELS[arm] for arm in order], rotation=30, ha="right")
    axis.set_ylabel("Heldout contact NLL")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    selected = [
        "L3_vs_L0_all", "L3_vs_L1_all", "L3_vs_L2_all",
        "L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2_distal",
    ]
    positions = np.arange(len(selected))
    for position, label in zip(positions, selected):
        values = contrast_rows.loc[contrast_rows.contrast == label, "gain_nats"].dropna().to_numpy()
        jitter = np.linspace(-0.13, 0.13, len(values))
        axis.scatter(position + jitter, values, s=13, color="#9ea5a8", alpha=0.72)
        axis.plot([position - 0.20, position + 0.20], [np.median(values)] * 2, color="#171717", lw=2.0)
    axis.axhline(0, color="#555555", lw=0.8, ls="--")
    axis.set_xticks(positions, ["L0", "L1", "L2", "L0", "L1", "L2"])
    axis.set_xlabel("All transitions             Distal")
    axis.set_ylabel("Selected-shortcut gain (nats)")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[2]
    subset = sensitivity[
        (sensitivity.endpoint == "distal_contact_nll")
        & (sensitivity.comparator == "L0_LOCAL_ONLY")
    ]
    for row in subset.itertuples():
        axis.plot([0, 1], [row.v0_2_gain, row.v0_3_gain], color="#c4c8ca", lw=0.7)
    axis.scatter(np.zeros(len(subset)), subset.v0_2_gain, s=17, color="#858b90")
    axis.scatter(np.ones(len(subset)), subset.v0_3_gain, s=17, color="#b84b4b")
    axis.plot([0, 1], [subset.v0_2_gain.median(), subset.v0_3_gain.median()], color="#171717", lw=2.0)
    axis.axhline(0, color="#555555", lw=0.8, ls="--")
    axis.set_xticks([0, 1], ["Contact-dilated", "Full-tissue"])
    axis.set_ylabel("Distal gain: selected − local")
    axis.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("ABC", axes):
        axis.text(-0.18, 1.06, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"stage_d_full_tissue_interictal.{suffix}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (figure_dir / "README.md").open("a") as stream:
        stream.write(
            "\n### stage_d_full_tissue_interictal.png\n\n"
            "A 为 21 位患者五个 matched arms 的 heldout contact NLL；B 分开显示全部 transitions 和 distal transitions 上 task-selected nonlocal 的患者级增量；"
            "C 比较 contact-dilated 与 full-tissue domain 对 selected-vs-local distal contrast 的影响。\n\n"
            "**关注点**：只有 L3 同时超过 local-only、等容量 extra-local 和固定 random nonlocal，才支持 selective shortcut。\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    old = args.old_root.resolve()
    marker = json.loads((out / "FORMAL_TRAINING_COMPLETE.json").read_text())
    if marker.get("complete") != 465 or marker.get("unresolved") != 0:
        raise RuntimeError("formal training is not complete")
    units = load_units(out)
    units.to_csv(out / "interictal_per_fit_seed.csv", index=False)
    patient = aggregate_patient(units, old)
    patient.to_csv(out / "interictal_per_patient.csv", index=False)
    contrast_rows, summaries = contrasts(patient)
    contrast_rows.to_csv(out / "interictal_patient_contrasts.csv", index=False)
    sensitivity, geometry = geometry_sensitivity(patient, old)
    sensitivity.to_csv(out / "latent_domain_sensitivity.csv", index=False)
    summary = {
        "contract": "topic5_lbss_full_tissue_interictal_v0_3",
        "n_patients": int(patient.subject.nunique()),
        "n_fits": int(units.fit_id.nunique()),
        "n_units": len(units),
        "comparisons": summaries,
        "geometry_sensitivity": geometry,
        "target_values_read": False,
    }
    (out / "INTERICTAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(patient, contrast_rows, sensitivity, out)
    (out / "INTERICTAL_ANALYSIS_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "n_units": len(units), "n_patients": summary["n_patients"],
        "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
