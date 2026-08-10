#!/usr/bin/env python3
"""Paper-ready six-panel summary of the completed LBSS-RNN v0.2 experiment."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.plot_contact_plane_static import _smooth_rank_field_mm  # noqa: E402
from scripts.score_topic5_rnn_motif_early_ictal_v0_4 import build_scorer  # noqa: E402
from scripts.summarize_topic5_lbss_claims_v0_2 import attenuation_damage_auc  # noqa: E402
from src.topic5_gradient_grid_field import score_event_detail_single  # noqa: E402


RED = "#B2182B"
BLUE = "#2166AC"
GRAY = "#9aa2a9"
DARK = "#263238"
L3 = "L3_LOCAL_PLUS_LEARNED_LR"
ARMS = (
    "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR", L3,
)
OLD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-rnn-motif-cross-state-v0-4/"
    "results/topic5_rnn_motif_cross_state_benchmark_v0_4"
)


def panel_letter(ax, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def normalized_event_matrix(rows: list[np.ndarray], n_contacts: int) -> np.ndarray:
    matrix = np.full((n_contacts, len(rows)), np.nan)
    for column, rank in enumerate(rows):
        finite = rank >= 0
        if finite.any():
            top = max(1.0, float(rank[finite].max()))
            matrix[finite, column] = rank[finite] / top
    return matrix


def generated_rank(sequence: list[list[int]], n_contacts: int) -> np.ndarray:
    rank = np.full(n_contacts, -1, int)
    for index, contacts in enumerate(sequence):
        rank[np.asarray(contacts, int)] = index
    return rank


def draw_graph(ax, out: Path, fit_id: str) -> None:
    plane = np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False)
    graph = np.load(out / "per_fit" / fit_id / L3 / "seed0" / "graph.npz", allow_pickle=False)
    xy = plane["nodes_xy_mm"]
    contacts = plane["contacts_xy_mm"]
    local = graph["local_mask"].astype(bool)
    added = graph["added_mask"].astype(bool)
    strength = graph["strength"]
    # Draw each symmetric local pair once; direction remains in the learned weights.
    for target, source in np.argwhere(local & np.triu(np.ones_like(local, bool), 1)):
        ax.plot(xy[[source, target], 0], xy[[source, target], 1], color="#cbd1d5",
                lw=0.35, alpha=0.38, zorder=1)
    selected = np.argwhere(added)
    order = np.argsort(strength[added])[::-1]
    for rank, index in enumerate(order):
        target, source = selected[index]
        patch = FancyArrowPatch(
            xy[source], xy[target], arrowstyle="-|>", mutation_scale=4.5,
            connectionstyle=f"arc3,rad={0.08 if rank % 2 == 0 else -0.08}",
            color=RED, lw=1.0 if rank < 8 else 0.45,
            alpha=0.85 if rank < 8 else 0.24, zorder=2,
        )
        ax.add_patch(patch)
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#7d878e", alpha=0.55, zorder=3)
    ax.scatter(contacts[:, 0], contacts[:, 1], s=26, facecolor="white", edgecolor=DARK,
               linewidth=0.8, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Local backbone + selected shortcuts", fontsize=11.5, pad=5)
    for spine in ax.spines.values(): spine.set_visible(False)


def draw_event_reproduction(fig, spec, out: Path, fit_id: str) -> None:
    sub = spec.subgridspec(2, 2, wspace=0.08, hspace=0.13)
    axes = np.asarray([[fig.add_subplot(sub[i, j]) for j in range(2)] for i in range(2)])
    events = np.load(out / "cache" / fit_id / "events.npz", allow_pickle=False)
    provenance = json.loads((out / "cache" / fit_id / "provenance.json").read_text())
    keep = events["split"] >= 0
    ranks, split, modes = events["ranks"][keep], events["split"][keep], events["mode"][keep]
    test = np.flatnonzero(split == 2)
    with gzip.open(out / "per_fit" / fit_id / L3 / "seed0" / "heldout_rollouts.json.gz", "rt") as stream:
        rollouts = json.load(stream)
    by_index = {int(row["kept_event_index"]): row for row in rollouts}
    empirical = json.loads((Path(json.loads((OLD_ROOT / "INPUT_MANIFEST.json").read_text())
                                      ["input_roots"]["field"]) /
                            f"{provenance['subject']}.json").read_text())["interictal_field"]
    empirical_names = [str(value) for value in empirical["contact_order"]]
    take = np.asarray([empirical_names.index(str(value)) for value in provenance["contacts"]], int)
    order_a = np.argsort(np.asarray(empirical["rank_a"], float)[take], kind="stable")
    order_b = np.argsort(np.asarray(empirical["rank_b"], float)[take], kind="stable")
    for row_index, (template, contact_order, color) in enumerate((("A", order_a, RED), ("B", order_b, BLUE))):
        chosen = [int(index) for index in test
                  if str(provenance["mode_to_template"].get(str(int(modes[index])), "")).upper() == template][:30]
        observed = [ranks[index] for index in chosen]
        model = [generated_rank(by_index[index]["generated_rank_sets"], ranks.shape[1]) for index in chosen]
        for column, payload in enumerate((observed, model)):
            matrix = normalized_event_matrix(payload, ranks.shape[1])[contact_order]
            cmap = mpl.colormaps["viridis"].copy(); cmap.set_bad("#e7e7e7")
            axes[row_index, column].imshow(matrix, aspect="auto", interpolation="nearest",
                                           cmap=cmap, vmin=0, vmax=1, origin="upper")
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values(): spine.set_visible(False)
        axes[row_index, 0].set_ylabel(f"T{template}", color=color, rotation=0, labelpad=12,
                                     fontsize=11.5, fontweight="bold", va="center")
    axes[0, 0].set_title("Observed", fontsize=11, pad=4)
    axes[0, 1].set_title("Generated", fontsize=11, pad=4)
    panel_letter(axes[0, 0], "B")


def dot_summary(ax, values: list[np.ndarray], labels: list[str], colors: list[str], ylabel: str) -> None:
    for index, data in enumerate(values):
        data = np.asarray(data, float); data = data[np.isfinite(data)]
        jitter = np.linspace(-0.10, 0.10, len(data)) if len(data) else np.asarray([])
        ax.scatter(index + jitter, data, s=13, color=colors[index], alpha=0.65, edgecolor="none")
        if len(data): ax.plot([index - 0.17, index + 0.17], [np.median(data)] * 2, color=DARK, lw=2.0)
    ax.axhline(0, color="#7a7a7a", lw=0.7, ls="--")
    ax.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def draw_interictal(fig, spec, out: Path) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.45)
    axes = [fig.add_subplot(sub[0, i]) for i in range(2)]
    patient = pd.read_csv(out / "interictal_per_patient.csv")
    pivot = patient.pivot(index="subject", columns="arm")
    refs = ("L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", "L2_LOCAL_PLUS_RANDOM_LR", "C_L3_ORDER_SHUFFLED")
    labels = ("Local", "Extra", "Random", "Shuffle")
    colors = ("#8395a7", "#8aa85b", "#a970b5", "#b7b7b7")
    all_gain = [pivot["test_contact_nll"][ref] - pivot["test_contact_nll"][L3] for ref in refs]
    distal_gain = [pivot["distal_contact_nll"][ref] - pivot["distal_contact_nll"][L3] for ref in refs]
    dot_summary(axes[0], all_gain, list(labels), list(colors), "All-step NLL gain")
    dot_summary(axes[1], distal_gain, list(labels), list(colors), "Distal NLL gain")
    panel_letter(axes[0], "C")


def draw_pathways(fig, spec, out: Path, subject: str) -> None:
    sub = spec.subgridspec(1, 3, width_ratios=(1, 1, 0.8), wspace=0.28)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    fit_id = f"{subject}__shared"
    plane = np.load(out / "cache" / fit_id / "plane.npz", allow_pickle=False)
    xy = plane["contacts_xy_mm"]
    for ax, arm, title in zip(axes[:2], (L3, "C_L3_ORDER_SHUFFLED"), ("True order", "Order shuffle")):
        payload = np.load(out / "pathway_analysis" / "per_patient" / subject / f"{arm}.npz", allow_pickle=False)
        pattern = payload["effective_pattern"]; n = len(xy)
        source, target = pattern[:n], pattern[n:]
        ax.scatter(xy[:, 0], xy[:, 1], s=18 + 480 * source, color=BLUE, alpha=0.72)
        ax.scatter(xy[:, 0], xy[:, 1], s=10 + 320 * target, facecolor="none", edgecolor=RED, lw=1.0)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
    comparison = pd.read_csv(out / "pathway_analysis" / "true_vs_shuffle_patient_patterns.csv")
    values = [comparison.endpoint_dissimilarity_beyond_proposal,
              comparison.effective_dissimilarity_beyond_proposal]
    dot_summary(axes[2], values, ["Endpoints", "Influence"], [GRAY, RED],
                "Dissimilarity beyond proposal")
    panel_letter(axes[0], "D")


def smooth_field(points_mm: np.ndarray, values: np.ndarray, support: np.ndarray,
                 sigma_mm: float = 6.0):
    padding = 5.0
    xlim = (float(points_mm[:, 0].min() - padding), float(points_mm[:, 0].max() + padding))
    ylim = (float(points_mm[:, 1].min() - padding), float(points_mm[:, 1].max() + padding))
    return _smooth_rank_field_mm(points_mm[:, 0], points_mm[:, 1], values, support,
                                 xlim, ylim, sigma_mm)


def draw_cross_state_maps(fig, spec, out: Path, subject: str) -> dict:
    sub = spec.subgridspec(1, 3, wspace=0.12)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
    old_manifest = json.loads((OLD_ROOT / "MODEL_FIELD_MANIFEST.json").read_text())
    record = json.loads(Path(old_manifest["patient_geometry"][subject]["empirical_record"]).read_text())
    field = record["interictal_field"]; order = [str(value) for value in field["contact_order"]]
    with np.load(out / "model_fields" / "intact" / "per_patient" / subject / f"{L3}.npz", allow_pickle=False) as data:
        names = data["contacts"].astype(str).tolist()
        lookup_a = dict(zip(names, np.asarray(data["A_canonical_full"], float)))
        lookup_b = dict(zip(names, np.asarray(data["B_canonical_full"], float)))
        model_a = np.asarray([lookup_a[name] for name in order], float)
        model_b = np.asarray([lookup_b[name] for name in order], float)
    target_root = Path(json.loads((out / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())["target_cache_root"])
    target_path = sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))[0]
    with np.load(target_path, allow_pickle=False) as data:
        target_lookup = dict(zip(data["contact_names"].astype(str).tolist(), np.asarray(data["target_1_150"], float)))
    energy = np.asarray([target_lookup.get(name, np.nan) for name in order], float)
    finite = np.isfinite(energy) & np.isfinite(field["earliness_a"]) & np.isfinite(field["earliness_b"])
    detail = score_event_detail_single(build_scorer(record, model_a, model_b, finite), energy)
    template = str(detail["best_template"])
    if template not in {"A", "B"}:
        raise RuntimeError(f"no maxAB template available for Figure 6 representative: {subject}")
    model_earliness = model_a if template == "A" else model_b
    empirical_earliness = np.asarray(field[f"earliness_{template.lower()}"], float)
    plane = field["planes"][f"own_{template.lower()}"]
    points = np.asarray(plane["points"], float) * float(plane["scale_mm"])
    support = np.asarray(field[f"support_{template.lower()}"], float)
    if detail[f"mirror_{template.lower()}"] == "mirror":
        points = points.copy(); points[:, 1] *= -1.0
    timing_payloads = (
        (axes[0], 1.0 - empirical_earliness, f"Data T{template}"),
        (axes[1], 1.0 - model_earliness, f"RNN T{template}"),
    )
    timing_image = None
    for ax, values, title in timing_payloads:
        X, Y, T, _, _ = smooth_field(points, values, support)
        timing_image = ax.imshow(T, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
                  aspect="equal", cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
        ax.scatter(points[:, 0], points[:, 1], c=values, cmap="viridis", vmin=0, vmax=1,
                   s=22, edgecolor="white", lw=0.7)
        ax.set_title(title, fontsize=11, color=RED if template == "A" else BLUE)
        ax.set_xticks([]); ax.set_yticks([])
    e_min, e_max = float(np.nanmin(energy)), float(np.nanmax(energy))
    X, Y, T, _, _ = smooth_field(points, energy, support)
    energy_image = axes[2].imshow(T, origin="lower", extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="equal", cmap="Blues", vmin=e_min, vmax=e_max, interpolation="bilinear")
    axes[2].scatter(points[:, 0], points[:, 1], c=energy, cmap="Blues", vmin=e_min, vmax=e_max,
                    s=22, edgecolor="white", lw=0.7)
    axes[2].set_title("Early ictal", fontsize=11)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for ax in axes:
        ax.set_xlabel("Propagation axis", fontsize=9)
    timing_bar = fig.colorbar(timing_image, ax=axes[:2], fraction=0.025, pad=0.015)
    timing_bar.set_ticks([0, 1], ["early", "late"]); timing_bar.ax.tick_params(labelsize=8)
    energy_bar = fig.colorbar(energy_image, ax=axes[2], fraction=0.045, pad=0.025)
    energy_bar.set_ticks([e_min, e_max]); energy_bar.ax.tick_params(labelsize=8)
    energy_bar.ax.set_title("power\n(z)", fontsize=8, pad=2)
    panel_letter(axes[0], "E")
    return {
        "subject": subject, "target_path": str(target_path), "target_key": "target_1_150",
        "maxab_template": template, "maxab_abs_r": detail["maxab"],
        "maxab_signed_r": detail[f"signed_{template.lower()}"],
        "maxab_mirror": detail[f"mirror_{template.lower()}"],
    }


def draw_early_statistics(fig, spec, out: Path) -> None:
    sub = spec.subgridspec(1, 2, wspace=0.42)
    axes = [fig.add_subplot(sub[0, i]) for i in range(2)]
    patient = pd.read_csv(out / "early_ictal" / "early_ictal_per_patient_condition.csv")
    primary = patient[(patient.primary) & (patient.endpoint == "canonical_full") & (patient.family == "intact")]
    pivot = primary.pivot(index="subject", columns="arm", values="all_contact_margin")
    for _, row in pivot.iterrows():
        axes[0].plot(range(4), row[list(ARMS)], color="#c0c5c9", lw=0.6, alpha=0.7)
    axes[0].scatter(range(4), [pivot[arm].median() for arm in ARMS], s=42,
                    color=["#8395a7", "#8aa85b", "#a970b5", RED], zorder=3)
    axes[0].axhline(0, color="#777777", lw=0.7, ls="--")
    axes[0].set_xticks(range(4), ["Local", "Extra", "Random", "Selected"], rotation=28, ha="right")
    axes[0].set_ylabel("Early-ictal margin")

    names = ("L1_ADDED", "L2_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL")
    auc = attenuation_damage_auc(patient, "seed_removed")
    values = [auc[auc.target == name].damage_auc.to_numpy(float) for name in names]
    dot_summary(axes[1], values, ["Extra", "Random", "Selected", "Local"],
                ["#8395a7", "#a970b5", RED, BLUE], "Concordance damage AUC")
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    panel_letter(axes[0], "F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--representative", default="epilepsiae_1084")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if not (out / "EARLY_ICTAL_SCORING_COMPLETE.json").exists():
        raise RuntimeError("complete early-ictal scoring before final figure")
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 11.0,
        "xtick.labelsize": 9.3, "ytick.labelsize": 9.3, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(12.0, 7.7))
    grid = fig.add_gridspec(2, 3, width_ratios=(0.83, 1.18, 1.12),
                           height_ratios=(1.0, 1.0), wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0]); draw_graph(ax_a, out, f"{args.representative}__shared"); panel_letter(ax_a, "A")
    draw_event_reproduction(fig, grid[0, 1], out, f"{args.representative}__shared")
    draw_interictal(fig, grid[0, 2], out)
    draw_pathways(fig, grid[1, 0], out, args.representative)
    metadata = draw_cross_state_maps(fig, grid[1, 1], out, args.representative)
    draw_early_statistics(fig, grid[1, 2], out)
    destination = out / "figures"; destination.mkdir(exist_ok=True)
    stem = destination / "topic5_figure6_lbss_rnn"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    (destination / "FIGURE6_METADATA.json").write_text(json.dumps({
        "contract": "topic5_figure6_lbss_rnn_v0_2", "representative": args.representative,
        "panels": {
            "A": "real patient geometry and LBSS recurrent mask",
            "B": "heldout observed versus same-start free-generated A/B rank events",
            "C": "patient-first all-step and distal interictal contrasts",
            "D": "target-free true-order versus shuffle coarse effective pathway",
            "E": "frozen RNN field versus clinical-onset 0-10 s 1-150 Hz broadband energy",
            "F": "patient-first cross-state margins and attenuation AUC",
        }, **metadata,
    }, indent=2) + "\n")
    readme_text = (
        "### topic5_figure6_lbss_rnn.png\n\n"
        "A 显示固定局部 backbone 与少量 task-selected nonlocal shortcuts；B 对照留出间期事件与只给第一 rank 后的自由生成。"
        "C 为 21 位患者的总体与远端间期增益；D 比较真实顺序和顺序打乱形成的粗空间有效影响；E 以预先指定病例并列冻结 RNN 场和 clinical onset 后 0–10 秒、1–150 Hz broadband energy；F 给出 10 位患者的跨状态统计与 attenuation。\n\n"
        "**关注点**：间期结果与 early-ictal 结果分开读；只有 selected nonlocal arm 超过 matched controls，且其 attenuation 特异损害远端传播或跨状态一致性，才支持 selective-shortcut contribution。\n"
    )
    readme = destination / "README.md"
    with readme.open("a") as stream: stream.write("\n" + readme_text)
    repo_root = out.parents[1]
    paper = repo_root / "results/paper-ready-figure/fig6_lbss_rnn/figures"
    paper.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        shutil.copy2(stem.with_suffix(f".{suffix}"), paper / stem.with_suffix(f".{suffix}").name)
    (paper / "README.md").write_text(readme_text)


if __name__ == "__main__":
    main()
