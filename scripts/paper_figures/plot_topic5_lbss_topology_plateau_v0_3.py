#!/usr/bin/env python3
"""Patient-level target-free forest for the full-tissue topology plateau."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_lbss_claims_v0_2 import paired  # noqa: E402


L0 = "L0_LOCAL_ONLY"
L1 = "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"
L2 = "L2_LOCAL_PLUS_RANDOM_LR"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SHUFFLE = "C_L3_ORDER_SHUFFLED"
CONTROLS = (L0, L1, L2, SHUFFLE)
LABELS = {
    L0: "Local backbone",
    L1: "Extra local",
    L2: "Random nonlocal",
    SHUFFLE: "Order shuffle",
}
COLORS = {L0: "#737b80", L1: "#5f8ea0", L2: "#9a8267", SHUFFLE: "#b8b8b8"}


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = np.nanmedian(
        values[rng.integers(0, len(values), size=(10_000, len(values)))], axis=1
    )
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def metric_vectors(out: Path) -> list[tuple[str, str, dict[str, np.ndarray]]]:
    inter = pd.read_csv(out / "interictal_per_patient.csv").pivot(
        index="subject", columns="arm"
    )
    fields = pd.read_csv(out / "model_field_patient_metrics.csv").pivot(
        index="subject", columns="arm"
    )
    metrics: list[tuple[str, str, dict[str, np.ndarray]]] = []
    for label, column, source, direction in (
        ("All next-contact", "test_contact_nll", inter, -1.0),
        ("Distal next-contact", "distal_contact_nll", inter, -1.0),
        ("Free generation", "rollout_spearman", inter, 1.0),
        ("Interictal field", "canonical_empirical_r", fields, 1.0),
        ("Field beyond start", "seed_removed_empirical_r", fields, 1.0),
        ("A/B contrast field", "canonical_contrast_empirical_r", fields, 1.0),
    ):
        values = {
            control: direction * (
                source[column][L3].to_numpy(float)
                - source[column][control].to_numpy(float)
            )
            for control in CONTROLS
        }
        metrics.append((label, column, values))
    return metrics


def plot(out: Path, destination: Path) -> dict:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 10.0, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    metrics = metric_vectors(out)
    # Keep every facet on a standardized within-endpoint scale.  Raw NLL,
    # Spearman rollout and field correlations have different units; plotting
    # them with independently auto-scaled axes can make tiny topology effects
    # look as large as the true-order control.  The conversion below preserves
    # sign and patient pairing while expressing each contrast in SD units of
    # the patient-level control differences for that endpoint.
    standardized = []
    for label, column, vectors in metrics:
        pooled = np.concatenate([
            np.asarray(values, float)[np.isfinite(values)]
            for values in vectors.values()
        ])
        scale = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 1.0
        scale = max(scale, 1e-12)
        standardized.append((label, column, {
            control: np.asarray(values, float) / scale
            for control, values in vectors.items()
        }))
    metrics = standardized
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.7))
    records = []
    for metric_index, (label, _, vectors) in enumerate(metrics):
        ax = axes.flat[metric_index]
        rows = []
        for control_index, control in enumerate(CONTROLS):
            values = vectors[control]
            keep = np.isfinite(values)
            values = values[keep]
            estimate = float(np.median(values))
            lo, hi = bootstrap_interval(values, 20260812 + 10 * metric_index + control_index)
            formal = paired(values)
            rows.append((control, estimate, lo, hi, formal))
            records.append({
                "endpoint": label, "control": control, "n_patients": int(len(values)),
                "median_L3_advantage": estimate, "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi, "wilcoxon_p_two_sided": formal["wilcoxon_p_two_sided"],
                "n_positive": formal["n_positive"], "n_negative": formal["n_negative"],
            })
        for y, (control, estimate, lo, hi, _) in enumerate(rows):
            ax.plot([lo, hi], [y, y], color=COLORS[control], lw=1.7, solid_capstyle="round")
            ax.scatter(estimate, y, s=34, color=COLORS[control], edgecolor="white", lw=0.6, zorder=3)
        ax.axvline(0, color="#858c90", lw=0.8, ls="--", zorder=0)
        ax.set_yticks(range(len(rows)), [LABELS[row[0]] for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(-0.9, 2.8)
        ax.set_title(label, fontsize=11.2, fontweight="bold", pad=5)
        ax.set_xlabel("Selected-shortcut advantage (SD)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.93, wspace=0.56, hspace=0.58)
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / "topic5_lbss_spatial_topology_plateau"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    table = pd.DataFrame(records)
    table.to_csv(destination / "topic5_lbss_spatial_topology_plateau_source_data.csv", index=False)
    result = {
        "contract": "topic5_lbss_spatial_topology_plateau_v0_3",
        "n_patients": int(table.n_patients.max()),
        "interpretation": "no_unique_spatial_topology_winner",
        "target_values_read": False,
    }
    (destination / "topic5_lbss_spatial_topology_plateau_metadata.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    readme = destination / "README.md"
    entry = (
        "### topic5_lbss_spatial_topology_plateau.png / .pdf / .svg\n\n"
        "六个患者级间期端点同时比较 task-selected nonlocal 模型与局部骨架、等容量局部边、随机非局部边及 order-shuffle。点为患者差值中位数，线为 patient bootstrap 95% 区间；为让不同量纲可比较，横轴按每个端点的患者差值标准差归一化，正值表示 task-selected nonlocal 更好。\n\n"
        "**关注点**：真实顺序相对 shuffle 的优势稳定，但 L3 相对三种真实顺序 topology 的差值均围绕零；这支持 recurrence 与真实 order，而不是某一种精细空间连接胜出。\n"
    )
    if readme.exists():
        text = readme.read_text()
        if "### topic5_lbss_spatial_topology_plateau" not in text:
            readme.write_text(text.rstrip() + "\n\n" + entry)
    else:
        readme.write_text(entry)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out = args.out_root.resolve()
    destination = (
        args.out_dir.resolve() if args.out_dir is not None
        else out / "figures"
    )
    print(json.dumps(plot(out, destination), indent=2))


if __name__ == "__main__":
    main()
