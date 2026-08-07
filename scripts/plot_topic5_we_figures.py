"""The six-panel WE-SLP-RNN v0.3 main figure.

Panel discipline follows docs/figure_style_guide.md: ordered quantities are
viridis, signed differences are a red-blue diverging map centred on zero, axes
are tight, one shared legend per quantity, and no internal identifier reaches a
reader-facing label.  Cross-patient comparisons are drawn as within-patient
paired differences because the absolute level of the prediction score tracks
contact count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_we_graph_analysis import module_of_each_node  # noqa: E402
from src.topic5_wiring_economy_rnn import (  # noqa: E402
    WEConfig,
    WEModel,
    build_event_tensors,
    rollout,
)

OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3"
ARM_LABEL = {
    "STATIC_CONTACT": "Static contact rate",
    "DENSE_TISSUE": "Dense tissue network",
    "RANDOM_SET": "Sparse, uniform rewiring",
    "SPATIAL_SET": "Sparse, wiring economy",
    "RANDOM_SET_COST": "Uniform growth + length cost",
    "SPATIAL_SET_NOCOST": "Near growth, no length cost",
}
ARM_COLOUR = {
    "STATIC_CONTACT": "#9e9e9e", "DENSE_TISSUE": "#4a6fa5",
    "RANDOM_SET": "#d98b3a", "SPATIAL_SET": "#2e7d5b",
    "RANDOM_SET_COST": "#c7a76c", "SPATIAL_SET_NOCOST": "#7fb39a",
}
MODE_LABEL = {0: "Mode A", 1: "Mode B"}


def load_analysis(out_root: Path, cell: str) -> Dict[str, Any]:
    analysis = out_root / "analysis"
    return {name: json.loads((analysis / f"{name}_{cell}.json").read_text())
            for name in ("pareto", "topology", "function", "lesion", "tendency")}


# ------------------------------------------------------------------ panel A --
def panel_model(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(3)
    nodes = rng.uniform([0.08, 0.16], [0.92, 0.80], size=(34, 2))
    d = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    keep = []
    for i in range(len(nodes)):
        for j in np.argsort(d[i])[:2]:
            keep.append((i, int(j)))
    for i, j in keep:
        ax.plot(*zip(nodes[i], nodes[j]), color="#2e7d5b", lw=0.7, alpha=0.55, zorder=1)
    for i, j in [(2, 27), (9, 31)]:
        ax.plot(*zip(nodes[i], nodes[j]), color="#b0413e", lw=1.1, alpha=0.85,
                zorder=2, ls=(0, (3, 1.5)))
    ax.scatter(*nodes.T, s=26, c="#37474f", zorder=3, linewidths=0)

    contacts = np.array([[0.18, 0.10], [0.40, 0.10], [0.62, 0.10], [0.84, 0.10]])
    ax.scatter(*contacts.T, s=95, marker="s", facecolor="white",
               edgecolor="#1a1a1a", linewidths=1.3, zorder=4)
    for c in contacts:
        near = np.argsort(np.linalg.norm(nodes - c, axis=1))[:5]
        for i in near:
            ax.plot([c[0], nodes[i, 0]], [c[1], nodes[i, 1]],
                    color="#1a1a1a", lw=0.45, alpha=0.32, zorder=0)
    ax.text(0.5, 0.02, "contacts read a local neighbourhood of tissue",
            ha="center", va="bottom", fontsize=7.2)
    ax.text(0.5, 0.955, "fixed number of recurrent edges; weakest pruned,\n"
                        "nearby pairs preferentially regrown, long edges kept if useful",
            ha="center", va="top", fontsize=7.2)
    ax.legend(handles=[
        Line2D([], [], color="#2e7d5b", lw=1.2, label="short recurrent edge"),
        Line2D([], [], color="#b0413e", lw=1.2, ls=(0, (3, 1.5)), label="surviving long edge"),
        Line2D([], [], color="#1a1a1a", lw=0.6, alpha=0.4, label="observation operator"),
    ], loc="lower right", fontsize=6.2, frameon=False, borderpad=0.1)
    ax.set_title("a  Tissue network under a wiring budget", loc="left", fontsize=9, fontweight="bold")


# ------------------------------------------------------------------ panel B --
def replay(out_root: Path, fit_id: str, arm: str = "SPATIAL_SET", seed: int = 0,
           cell: str = "rnn", n_events: int = 400) -> Dict[str, Any] | None:
    """Reload a trained unit and regenerate its held-out events from real starts."""
    unit = out_root / "per_subject" / fit_id / f"{arm}_{cell}" / f"seed{seed}"
    if not (unit / "weights.pt").exists():
        return None
    provenance = json.loads((out_root / "cache" / fit_id / "provenance.json").read_text())
    plane = np.load(out_root / "cache" / fit_id / "plane.npz")
    events = np.load(out_root / "cache" / fit_id / "events.npz")
    ranks, split, mode = events["ranks"], events["split"], events["mode"]
    keep = split >= 0
    tensors = build_event_tensors(ranks[keep])
    kept_ranks, part, kept_mode = ranks[keep], split[keep], mode[keep]

    model = WEModel(WEConfig(
        arm=arm, cell=cell, n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]), seed=seed,
        observation_operator=plane["H"], node_distance_mm=plane["D_mm"]))
    model.load_state_dict(torch.load(unit / "weights.pt", map_location="cpu"))
    model.eval()

    out: Dict[str, Any] = {"contacts": provenance["contacts"],
                           "n_contacts": provenance["n_contacts"]}
    test = np.flatnonzero(part == 2)
    steps = int(tensors["valid"].shape[1])
    for m in (0, 1):
        pick = test[kept_mode[test] == m][:n_events]
        if pick.size < 10:
            continue
        observed = np.full((pick.size, provenance["n_contacts"]), np.nan)
        for r, i in enumerate(pick):
            row = kept_ranks[i].astype(float)
            row[row < 0] = np.nan
            observed[r] = row / np.nanmax(row) if np.nanmax(row) > 0 else row
        starts = [np.flatnonzero(kept_ranks[i] == 0) for i in pick]
        generated_seqs = rollout(model, starts, provenance["n_contacts"], steps,
                                 torch.device("cpu"))
        generated = np.full_like(observed, np.nan)
        for r, seq in enumerate(generated_seqs):
            flat = [c for s in seq for c in s]
            for position, contact in enumerate(flat):
                generated[r, contact] = position / max(1, len(flat) - 1)
        out[m] = {"observed": np.nanmean(observed, 0), "generated": np.nanmean(generated, 0),
                  "n": int(pick.size)}
    return out


def panel_replay(axes, replayed: Dict[str, Any], fit_id: str) -> None:
    contacts = replayed["contacts"]
    order = np.argsort(np.nan_to_num(replayed[0]["observed"], nan=2.0)) \
        if 0 in replayed else np.arange(len(contacts))
    for column, (m, ax_pair) in enumerate(zip((0, 1), (axes[:2], axes[2:]))):
        for ax, key, title in zip(ax_pair, ("observed", "generated"),
                                  (f"{MODE_LABEL[m]} observed", f"{MODE_LABEL[m]} generated")):
            if m not in replayed:
                ax.set_axis_off()
                continue
            values = replayed[m][key][order]
            ax.imshow(values[None, :], cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_yticks([])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([contacts[i] for i in order], rotation=90, fontsize=4.6)
            if key == "observed":
                ax.set_xticklabels([])
            ax.set_title(title, fontsize=7)
    axes[0].set_ylabel(fit_id.split("__")[0].replace("_", " "), fontsize=6.5)


# ------------------------------------------------------------------ panel C --
def panel_pareto(ax_scatter, ax_delta, pareto: Dict[str, Any]) -> None:
    per_fit = pareto["per_fit"]
    for arm in ("DENSE_TISSUE", "RANDOM_SET", "SPATIAL_SET"):
        if arm not in per_fit["c_wiring"]:
            continue
        fits = sorted(set(per_fit["c_wiring"][arm]) & set(per_fit["test_next_bce"][arm]))
        x = [per_fit["mean_edge_len_mm"][arm][f] * per_fit["test_next_bce"][arm][f] * 0 +
             per_fit["mean_edge_len_mm"][arm][f] for f in fits]
        y = [per_fit["test_next_bce"][arm][f] for f in fits]
        ax_scatter.scatter(x, y, s=17, color=ARM_COLOUR[arm], label=ARM_LABEL[arm],
                           alpha=0.85, linewidths=0)
    ax_scatter.set_xlabel("mean recurrent edge length (mm)", fontsize=7.5)
    ax_scatter.set_ylabel("held-out next-rank loss", fontsize=7.5)
    ax_scatter.tick_params(labelsize=6.5)
    ax_scatter.legend(fontsize=6, frameon=False, loc="upper left")
    ax_scatter.set_title("c  Prediction against wiring", loc="left", fontsize=9, fontweight="bold")

    keys = [("SPATIAL_SET__vs__RANDOM_SET", "wiring economy\n− uniform sparse"),
            ("SPATIAL_SET__vs__DENSE_TISSUE", "wiring economy\n− dense"),
            ("SPATIAL_SET__vs__STATIC_CONTACT", "wiring economy\n− static rate")]
    available = [(k, lab) for k, lab in keys if k in pareto.get("contrasts", {})]
    for i, (key, label) in enumerate(available):
        deltas = list(pareto["contrasts"][key]["delta_by_subject"].values())
        ax_delta.scatter(np.full(len(deltas), i) + np.random.default_rng(i).normal(0, 0.055, len(deltas)),
                         deltas, s=13, color="#37474f", alpha=0.7, linewidths=0, zorder=3)
        ax_delta.plot([i - 0.26, i + 0.26], [np.median(deltas)] * 2, color="#b0413e", lw=2, zorder=4)
        p = pareto["contrasts"][key]["p"]
        ax_delta.text(i, ax_delta.get_ylim()[1], f"p={p:.3f}", ha="center", va="bottom", fontsize=6)
    ax_delta.axhline(0, color="#1a1a1a", lw=0.8, zorder=1)
    ax_delta.set_xticks(range(len(available)))
    ax_delta.set_xticklabels([lab for _, lab in available], fontsize=6)
    ax_delta.set_ylabel("paired difference in loss\n(negative = better)", fontsize=7)
    ax_delta.tick_params(labelsize=6.5)


# ------------------------------------------------------------------ panel D --
def panel_topology(axes, out_root: Path, fit_id: str, seed: int = 0) -> None:
    unit = out_root / "per_subject" / fit_id / "SPATIAL_SET_rnn" / f"seed{seed}"
    plane = np.load(out_root / "cache" / fit_id / "plane.npz")
    g = np.load(unit / "graph.npz")
    nodes = plane["nodes_xy_mm"]
    initial, final, d = g["initial_mask"] > 0, g["mask"] > 0, g["D_mm"]
    membership, communities = module_of_each_node(final, seed=seed)
    palette = plt.get_cmap("tab10")

    for ax, (mask, title) in zip(axes, ((initial, "proposed at start"),
                                        (final, "kept after training"),
                                        (final, "modules"))):
        for i, j in zip(*np.nonzero(mask)):
            long = d[i, j] > 10.0
            ax.plot(*zip(nodes[i], nodes[j]),
                    color="#b0413e" if long else "#2e7d5b",
                    lw=0.9 if long else 0.45, alpha=0.75 if long else 0.4, zorder=1)
        colours = ["#37474f"] * len(nodes) if title != "modules" else \
            [palette(membership[i] % 10) for i in range(len(nodes))]
        ax.scatter(*nodes.T, s=13, c=colours, zorder=3, linewidths=0)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].legend(handles=[
        Line2D([], [], color="#2e7d5b", lw=1.0, label="edge ≤ 10 mm"),
        Line2D([], [], color="#b0413e", lw=1.2, label="edge > 10 mm"),
    ], fontsize=5.6, frameon=False, loc="lower left")
    axes[0].set_title("d  What the task kept", loc="left", fontsize=9, fontweight="bold")


# ------------------------------------------------------------------ panel E --
def panel_tendency(ax_len, ax_topo, tendency: Dict[str, Any], topology: Dict[str, Any]) -> None:
    rows = tendency["per_unit"]
    survived = [r["mean_len_survived_mm"] for r in rows]
    lost = [r["mean_len_lost_mm"] for r in rows]
    for i, (values, label) in enumerate(((lost, "pruned"), (survived, "kept"))):
        ax_len.scatter(np.full(len(values), i) + np.random.default_rng(i).normal(0, 0.05, len(values)),
                       values, s=11, color="#37474f", alpha=0.6, linewidths=0, zorder=3)
        ax_len.plot([i - 0.25, i + 0.25], [np.nanmedian(values)] * 2, color="#b0413e", lw=2, zorder=4)
    for a, b in zip(lost, survived):
        ax_len.plot([0, 1], [a, b], color="#9e9e9e", lw=0.4, alpha=0.5, zorder=2)
    ax_len.set_xticks([0, 1])
    ax_len.set_xticklabels(["pruned", "kept"], fontsize=7)
    ax_len.set_ylabel("edge length (mm)", fontsize=7.5)
    ax_len.tick_params(labelsize=6.5)
    p = tendency["gates"]["survived_shorter_than_lost"].get("p", float("nan"))
    ax_len.set_title(f"e  Which proposals survive  (p={p:.3f})", loc="left",
                     fontsize=9, fontweight="bold")

    gates = topology["gates"]
    keys = [("modularity_q", "modularity"), ("clustering", "clustering"),
            ("long_edge_fraction", "long-edge\nfraction")]
    refs = [("growth_prior_C1", "vs untrained\nsame rule", "#8d6e63"),
            ("task_free_dynamics_C2", "vs task-free\nrewiring", "#5c6bc0"),
            ("length_preserving_rewire_C3", "vs length-matched\nreshuffle", "#26a69a")]
    width = 0.26
    for k, (key, label) in enumerate(keys):
        for r, (ref, ref_label, colour) in enumerate(refs):
            block = gates.get(key, {}).get(ref)
            if not block or "median_delta" not in block:
                continue
            ax_topo.bar(k + (r - 1) * width, block["median_delta"], width * 0.9,
                        color=colour, alpha=0.9,
                        label=ref_label if k == 0 else None)
            if block.get("p", 1.0) < 0.05:
                ax_topo.text(k + (r - 1) * width, block["median_delta"], "*",
                             ha="center", va="bottom" if block["median_delta"] > 0 else "top",
                             fontsize=9)
    ax_topo.axhline(0, color="#1a1a1a", lw=0.8)
    ax_topo.set_xticks(range(len(keys)))
    ax_topo.set_xticklabels([lab for _, lab in keys], fontsize=6.5)
    ax_topo.set_ylabel("learned − reference\n(patient median)", fontsize=7)
    ax_topo.tick_params(labelsize=6.5)
    ax_topo.legend(fontsize=5.8, frameon=False, ncol=1, loc="best")


# ------------------------------------------------------------------ panel F --
def panel_lesion(ax, lesion: Dict[str, Any]) -> None:
    rows = lesion["per_unit"]
    if not rows:
        ax.set_axis_off()
        return
    module = [r["module_delta_next_bce"] for r in rows]
    patch = [r["matched_patch_delta_next_bce"] for r in rows]
    for i, (values, label) in enumerate(((module, "module"), (patch, "matched patch"))):
        ax.scatter(np.full(len(values), i) + np.random.default_rng(i + 5).normal(0, 0.05, len(values)),
                   values, s=13, color="#37474f", alpha=0.65, linewidths=0, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.nanmedian(values)] * 2, color="#b0413e", lw=2, zorder=4)
    for a, b in zip(module, patch):
        ax.plot([0, 1], [a, b], color="#9e9e9e", lw=0.4, alpha=0.5, zorder=2)
    ax.axhline(0, color="#1a1a1a", lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["one whole module", "size- and cut-matched\ncontiguous patch"], fontsize=6.5)
    ax.set_ylabel("increase in next-rank loss\nafter cutting, without retraining", fontsize=7)
    ax.tick_params(labelsize=6.5)
    gate = lesion["gates"].get("module_vs_matched_contiguous_patch", {})
    p = gate.get("p", float("nan"))
    ax.set_title(f"f  Is a module necessary?  (p={p:.3f})", loc="left",
                 fontsize=9, fontweight="bold")


# ------------------------------------------------------------------- main ---
def pick_representative(pareto: Dict[str, Any], out_root: Path) -> str:
    """Median-sized shared-plane patient, so the topology panel is typical."""
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    shared = [r for r in manifest["fits"] if r["scope"] == "shared"]
    shared.sort(key=lambda r: r["n_nodes"])
    return shared[len(shared) // 2]["fit_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--cell", default="rnn")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    a = load_analysis(out_root, args.cell)
    representative = pick_representative(a["pareto"], out_root)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7})
    fig = plt.figure(figsize=(11.0, 10.4))
    outer = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28,
                              height_ratios=[1.0, 1.0, 0.95])

    panel_model(fig.add_subplot(outer[0, 0]))

    b = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 1], hspace=0.55)
    replayed = replay(out_root, representative)
    axes_b = [fig.add_subplot(b[i]) for i in range(4)]
    if replayed:
        panel_replay(axes_b, replayed, representative)
    axes_b[0].set_title("b  Same-start free generation, held-out events", loc="left",
                        fontsize=9, fontweight="bold", pad=12)

    c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.45)
    panel_pareto(fig.add_subplot(c[0]), fig.add_subplot(c[1]), a["pareto"])

    d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 1], wspace=0.12)
    panel_topology([fig.add_subplot(d[i]) for i in range(3)], out_root, representative)

    e = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, 0], wspace=0.5)
    panel_tendency(fig.add_subplot(e[0]), fig.add_subplot(e[1]), a["tendency"], a["topology"])

    panel_lesion(fig.add_subplot(outer[2, 1]), a["lesion"])

    figures = out_root / "figures"
    figures.mkdir(exist_ok=True)
    stem = figures / f"topic5_wiring_economy_rnn_main_{args.cell}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    (figures / f"{stem.name}_metadata.json").write_text(json.dumps({
        "cell": args.cell, "representative_fit": representative,
        "n_patients": a["pareto"].get("contrasts", {}).get(
            "SPATIAL_SET__vs__RANDOM_SET", {}).get("n"),
    }, indent=2))
    print(f"wrote {stem}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
