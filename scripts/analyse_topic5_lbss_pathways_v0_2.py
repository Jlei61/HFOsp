#!/usr/bin/env python3
"""Target-free coarse pathway and effective-influence analysis for LBSS v0.2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_topic5_rnn_motif_influence_v0_4 import (  # noqa: E402
    edge_influence_rnn,
    hidden_before,
    prefix_inventory,
    teacher_jacobian,
)
from src.topic5_lbss_analysis_v0_2 import endpoint_density, instantiate_lbss  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


ARMS = (
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
SNAPSHOTS = (
    "SNAPSHOT_INIT", "SNAPSHOT_AFTER_WARMUP", "SNAPSHOT_REWIRE_1_3",
    "SNAPSHOT_REWIRE_2_3", "SNAPSHOT_MASK_FREEZE", "SNAPSHOT_FINAL",
)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 3 or np.std(a[use]) == 0 or np.std(b[use]) == 0:
        return float("nan")
    value = spearmanr(a[use], b[use]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def summarize_unit(out: Path, metrics_path: Path, device: torch.device, max_prefixes: int) -> dict:
    model, _, metrics, plane, events, _ = instantiate_lbss(out, metrics_path, device)
    keep = events["split"] >= 0
    ranks = events["ranks"][keep]
    split = events["split"][keep]
    tensors = build_event_tensors(ranks)
    prefixes = prefix_inventory(tensors, split, max_prefixes)
    teacher_rows, edge_rows = [], []
    for event, step in prefixes:
        x = tensors["x"][event].to(device)
        available = tensors["available"][event, step].to(device)
        h_before = hidden_before(model, x, step)
        current = x[step]
        teacher_rows.append(teacher_jacobian(model, h_before, current, available))
        edge_rows.append(edge_influence_rnn(model, h_before, current, available))
    teacher = np.nanmean(np.stack(teacher_rows), axis=0)
    effective_edge = np.nanmean(np.stack(edge_rows), axis=0)
    graph = np.load(metrics_path.parent / "graph.npz", allow_pickle=False)
    strength = graph["strength"]
    added = graph["added_mask"].astype(bool)
    local = graph["local_mask"].astype(bool)
    endpoint = endpoint_density(strength, added, plane["H"])
    effective_endpoint = endpoint_density(effective_edge, added, plane["H"])
    exposure = graph["exposure_count"].astype(float)
    exposure_endpoint = endpoint_density(exposure, exposure > 0, plane["H"])
    trajectory = {}
    for label in SNAPSHOTS:
        path = metrics_path.parent / "snapshots" / f"{label}.npz"
        if not path.exists():
            continue
        snapshot = np.load(path, allow_pickle=False)
        mask = snapshot["added_mask"].astype(bool)
        density = endpoint_density(snapshot["strength"], mask, plane["H"])
        trajectory[label] = np.r_[density["source_contact"], density["target_contact"]]
    final_pattern = np.r_[endpoint["source_contact"], endpoint["target_contact"]]
    trajectory_to_final = {
        label: safe_corr(pattern, final_pattern) for label, pattern in trajectory.items()
    }
    destination = out / "pathway_analysis" / "per_fit_seed" / metrics["fit_id"] / metrics["arm"] / f"seed{metrics['seed']}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        contacts_xy_mm=plane["contacts_xy_mm"],
        nodes_xy_mm=plane["nodes_xy_mm"],
        teacher_contact_jacobian=teacher,
        effective_edge=effective_edge,
        **{f"endpoint_{key}": value for key, value in endpoint.items()},
        **{f"effective_endpoint_{key}": value for key, value in effective_endpoint.items()},
        **{f"exposure_endpoint_{key}": value for key, value in exposure_endpoint.items()},
    )
    active_strength = strength * (local | added)
    return {
        "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["scope"],
        "arm": metrics["arm"], "seed": metrics["seed"], "n_prefixes": len(prefixes),
        "added_edges": int(added.sum()),
        "added_mean_length_mm": float(plane["D_mm"][added].mean()),
        "added_weight_fraction": float((strength * added).sum() / max((active_strength).sum(), 1e-12)),
        "added_effective_fraction": float((effective_edge * added).sum() / max(effective_edge[local | added].sum(), 1e-12)),
        "added_effective_reach_mm": float(
            (effective_edge * added * plane["D_mm"]).sum() / max((effective_edge * added).sum(), 1e-12)
        ),
        "source_vs_exposure_r": safe_corr(endpoint["source_contact"], exposure_endpoint["source_contact"]),
        "target_vs_exposure_r": safe_corr(endpoint["target_contact"], exposure_endpoint["target_contact"]),
        "exact_edge_survival_fraction_secondary": float(
            (graph["added_mask"].astype(bool) & graph["initial_added_mask"].astype(bool)).sum()
            / max(1, added.sum())
        ),
        **{f"trajectory_to_final_{key}": value for key, value in trajectory_to_final.items()},
        "path": str(destination),
        "target_values_read": False,
    }


def aggregate(rows: pd.DataFrame, out: Path) -> pd.DataFrame:
    value_columns = [column for column in rows.columns if column not in {
        "subject", "fit_id", "scope", "arm", "seed", "path", "target_values_read"
    }]
    fit = rows.groupby(["subject", "fit_id", "arm"], sort=False)[value_columns].median().reset_index()
    patient = fit.groupby(["subject", "arm"], sort=False)[value_columns].mean().reset_index()
    patient.to_csv(out / "pathway_analysis" / "pathway_per_patient.csv", index=False)

    # Aggregate coarse source/target and effective contact patterns seed->fit->patient.
    pattern_rows = []
    for (subject, arm), group in rows.groupby(["subject", "arm"], sort=False):
        fit_patterns = []
        for fit_id, fit_group in group.groupby("fit_id", sort=False):
            arrays = [np.load(path, allow_pickle=False) for path in fit_group.path]
            endpoint = np.nanmedian(np.stack([
                np.r_[item["endpoint_source_contact"], item["endpoint_target_contact"]] for item in arrays
            ]), axis=0)
            effective = np.nanmedian(np.stack([
                np.r_[item["effective_endpoint_source_contact"], item["effective_endpoint_target_contact"]]
                for item in arrays
            ]), axis=0)
            exposure = np.nanmedian(np.stack([
                np.r_[item["exposure_endpoint_source_contact"], item["exposure_endpoint_target_contact"]]
                for item in arrays
            ]), axis=0)
            fit_patterns.append((endpoint, effective, exposure))
        endpoint = np.mean(np.stack([item[0] for item in fit_patterns]), axis=0)
        effective = np.mean(np.stack([item[1] for item in fit_patterns]), axis=0)
        exposure = np.mean(np.stack([item[2] for item in fit_patterns]), axis=0)
        destination = out / "pathway_analysis" / "per_patient" / subject / f"{arm}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            endpoint_pattern=endpoint,
            effective_pattern=effective,
            proposal_exposure_pattern=exposure,
        )
        pattern_rows.append({"subject": subject, "arm": arm, "path": str(destination)})
    pattern = pd.DataFrame(pattern_rows)
    comparisons = []
    for subject in sorted(pattern.subject.unique()):
        paths = {row.arm: row.path for row in pattern[pattern.subject == subject].itertuples()}
        true = np.load(paths["L3_LOCAL_PLUS_LEARNED_LR"], allow_pickle=False)
        shuffle = np.load(paths["C_L3_ORDER_SHUFFLED"], allow_pickle=False)
        endpoint_dissimilarity = 1.0 - safe_corr(true["endpoint_pattern"], shuffle["endpoint_pattern"])
        effective_dissimilarity = 1.0 - safe_corr(true["effective_pattern"], shuffle["effective_pattern"])
        proposal_dissimilarity = 1.0 - safe_corr(
            true["proposal_exposure_pattern"], shuffle["proposal_exposure_pattern"]
        )
        comparisons.append({
            "subject": subject,
            "endpoint_true_shuffle_r": safe_corr(true["endpoint_pattern"], shuffle["endpoint_pattern"]),
            "effective_true_shuffle_r": safe_corr(true["effective_pattern"], shuffle["effective_pattern"]),
            "proposal_exposure_true_shuffle_r": safe_corr(
                true["proposal_exposure_pattern"], shuffle["proposal_exposure_pattern"]
            ),
            "endpoint_dissimilarity_beyond_proposal": endpoint_dissimilarity - proposal_dissimilarity,
            "effective_dissimilarity_beyond_proposal": effective_dissimilarity - proposal_dissimilarity,
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "pathway_analysis" / "true_vs_shuffle_patient_patterns.csv", index=False)
    return patient.merge(comparison, on="subject", how="left")


def plot_pathways(patient: pd.DataFrame, out: Path, representative: str) -> None:
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    true_path = out / "pathway_analysis" / "per_patient" / representative / "L3_LOCAL_PLUS_LEARNED_LR.npz"
    shuffle_path = out / "pathway_analysis" / "per_patient" / representative / "C_L3_ORDER_SHUFFLED.npz"
    true = np.load(true_path, allow_pickle=False)
    shuffle = np.load(shuffle_path, allow_pickle=False)
    # Coordinates come from the representative's single shared fit.
    fit_id = f"{representative}__shared"
    plane = np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False)
    n_contacts = plane["contacts_xy_mm"].shape[0]
    xy = plane["contacts_xy_mm"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.0))
    for ax, payload, title in zip(axes[:2], (true, shuffle), ("True order", "Order shuffle")):
        value = payload["effective_pattern"]
        source, target = value[:n_contacts], value[n_contacts:]
        ax.scatter(xy[:, 0], xy[:, 1], s=25 + 450 * source, color="#2f6fa3", alpha=0.75)
        ax.scatter(xy[:, 0], xy[:, 1], s=12 + 300 * target, facecolor="none", edgecolor="#c83e32", lw=1.0)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Propagation axis (mm)")
        ax.set_ylabel("Transverse axis (mm)")
    ax = axes[2]
    ax.scatter(np.zeros(len(patient)), patient["endpoint_dissimilarity_beyond_proposal"], s=18,
               color="#66727c", alpha=0.75)
    ax.scatter(np.ones(len(patient)), patient["effective_dissimilarity_beyond_proposal"], s=18,
               color="#c83e32", alpha=0.75)
    ax.scatter([0, 1], [np.nanmedian(patient["endpoint_dissimilarity_beyond_proposal"]),
                        np.nanmedian(patient["effective_dissimilarity_beyond_proposal"])],
               s=48, color="#202020", zorder=3)
    ax.set_xticks([0, 1], ["Endpoints", "Influence"])
    ax.axhline(0, color="#777777", lw=0.7, ls="--")
    ax.set_ylabel("Dissimilarity beyond proposal")
    for label, ax in zip("ABC", axes):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"stage_e_target_free_pathway_formation.{suffix}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### stage_e_target_free_pathway_formation.png\n\n"
        "A、B 在预先指定代表患者中显示真实顺序和打乱顺序训练后，新增边的 contact-space source（蓝）与 target（红圈）有效影响分布。"
        "C 在患者层显示真实顺序与打乱顺序的空间差异超过 candidate-proposal exposure 差异的部分。\n\n"
        "**关注点**：承重对象是粗空间有效影响，不是跨 seed 完全相同的二元边。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-prefixes", type=int, default=64)
    parser.add_argument("--representative", default="epilepsiae_1084")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if not (out / "MODEL_FIELDS_FROZEN.json").exists():
        raise RuntimeError("intact fields must be frozen before pathway analysis")
    rows = []
    for metrics_path in sorted((out / "per_fit").glob("*/*/seed*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics["arm"] in ARMS:
            rows.append(summarize_unit(out, metrics_path, torch.device(args.device), args.max_prefixes))
    frame = pd.DataFrame(rows)
    if len(frame) != 31 * len(ARMS) * 3:
        raise RuntimeError(f"expected {31 * len(ARMS) * 3} pathway units, observed {len(frame)}")
    root = out / "pathway_analysis"
    root.mkdir(exist_ok=True)
    frame.to_csv(root / "pathway_per_fit_seed.csv", index=False)
    patient = aggregate(frame, out)
    patient.to_csv(root / "pathway_patient_summary.csv", index=False)
    plot_pathways(patient, out, args.representative)
    (out / "PATHWAY_ANALYSIS_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "n_units": len(frame), "n_patients": int(patient.subject.nunique()),
        "primary_object": "coarse_contact_space_effective_influence",
        "exact_edge_identity_secondary": True,
        "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
