"""The main WE-SLP-RNN v0.3 figure.

Panel discipline follows docs/figure_style_guide.md: ordered quantities are
viridis, axes are tight, one shared legend per quantity, and no internal
identifier reaches a reader-facing label.  Cross-patient comparisons are drawn as
within-patient paired differences, because the absolute level of the prediction
score tracks contact count at rho = -0.96.

The lead panel is the sufficiency ladder (c): given only the first contact of a
held-out event, does the network regenerate the rest of that event's propagation
order?  It is drawn against the intrinsic reproducibility of the patient's own
events, because a generator scored on one noisy event can only reach the square
root of the event-to-event reliability however good it is.

The control ladder (e) asks the separate, harder question of whether the *shape*
of the graph is the task's doing.  Every topology number beats the untrained
graph the same growth rule produces, and a reader who saw only that would
conclude the task organised the network, so all three references are shown side
by side including the one that removes the conclusion.

There is no separate "which proposals survive" panel: pruned-versus-kept edge
length and the long-edge row of the ladder answer the same question, and the
ladder answers it against all three references instead of one.
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
from matplotlib.patches import Patch

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
    "DENSE_TISSUE": "all-to-all tissue network",
    "RANDOM_SET": "sparse, uniform rewiring",
    "SPATIAL_SET": "sparse, wiring economy",
}
ARM_COLOUR = {"DENSE_TISSUE": "#4a6fa5", "RANDOM_SET": "#d98b3a", "SPATIAL_SET": "#2e7d5b"}
SHORT, LONG = "#2e7d5b", "#b0413e"


def p_text(p: float) -> str:
    if not np.isfinite(p):
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def load_analysis(out_root: Path, cell: str) -> Dict[str, Any]:
    analysis = out_root / "analysis"
    out = {name: json.loads((analysis / f"{name}_{cell}.json").read_text())
           for name in ("pareto", "topology", "function", "lesion", "tendency")}
    path = analysis / f"sufficiency_{cell}.json"
    out["sufficiency"] = json.loads(path.read_text()) if path.exists() else None
    return out


SUFFICIENCY_ROWS = [
    ("SPATIAL_SET", "sparse, wiring economy", "#2e7d5b"),
    ("RANDOM_SET", "sparse, uniform rewiring", "#d98b3a"),
    ("DENSE_TISSUE", "all-to-all tissue", "#4a6fa5"),
    ("SPATIAL_SET_shuffled", "same training,\norder destroyed", "#5c6bc0"),
    ("STATIC_CONTACT", "no recurrence", "#9e9e9e"),
]


def panel_sufficiency(ax, suff: Dict[str, Any]) -> None:
    """Cohort same-start generation against its floors and its noise ceiling."""
    ceiling = np.array([suff["noise_ceiling_sqrt_reliability"][k]
                        for k in sorted(suff["noise_ceiling_sqrt_reliability"])])
    rng = np.random.default_rng(2)
    for i, (arm, label, colour) in enumerate(SUFFICIENCY_ROWS):
        values = np.array(list(suff["arms"][arm]["without_start"].values()))
        ax.scatter(values, np.full(len(values), i) + rng.normal(0, 0.07, len(values)),
                   s=13, color=colour, alpha=0.65, linewidths=0, zorder=3)
        ax.plot([np.median(values)] * 2, [i - 0.28, i + 0.28], color=colour, lw=2.6, zorder=4)
    band = (np.percentile(ceiling, 25), np.percentile(ceiling, 75))
    ax.axvspan(band[0], band[1], color="#b0413e", alpha=0.10, zorder=0)
    ax.axvline(np.median(ceiling), color=LONG, lw=1.4, ls=(0, (4, 2)), zorder=1)
    ax.text(np.median(ceiling), 1.75,
            "  how well one event of this\n  patient predicts another\n"
            "  (attenuation-corrected)",
            color=LONG, fontsize=6.2, va="center", ha="left")
    ax.set_yticks(range(len(SUFFICIENCY_ROWS)))
    ax.set_yticklabels([lab for _, lab, _ in SUFFICIENCY_ROWS], fontsize=7)
    ax.set_ylim(-0.6, len(SUFFICIENCY_ROWS) - 0.4)
    ax.set_xlabel("agreement with the real propagation order\n"
                  "(seeded contact removed, one point per patient)", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.axvline(0, color="#1a1a1a", lw=0.7, zorder=1)
    c = suff["contrasts"]
    ax.text(0.015, 0.03,
            f"vs no recurrence  {c['spatial_vs_no_recurrence']['median_delta']:+.2f}, "
            f"{c['spatial_vs_no_recurrence']['n_higher']}/{c['spatial_vs_no_recurrence']['n']}, "
            f"{p_text(c['spatial_vs_no_recurrence']['p'])}\n"
            f"vs order destroyed  {c['spatial_vs_order_destroyed']['median_delta']:+.2f}, "
            f"{c['spatial_vs_order_destroyed']['n_higher']}/{c['spatial_vs_order_destroyed']['n']}, "
            f"{p_text(c['spatial_vs_order_destroyed']['p'])}\n"
            f"vs the ceiling  {c['spatial_vs_noise_ceiling']['median_delta']:+.2f}, "
            f"{c['spatial_vs_noise_ceiling']['n_higher']}/{c['spatial_vs_noise_ceiling']['n']}, "
            f"{p_text(c['spatial_vs_noise_ceiling']['p'])}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2)


# --------------------------------------------------------------- panel a ----
def panel_model(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(3)
    nodes = rng.uniform([0.07, 0.30], [0.93, 0.86], size=(34, 2))
    d = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    for i in range(len(nodes)):
        for j in np.argsort(d[i])[:2]:
            ax.plot(*zip(nodes[i], nodes[int(j)]), color=SHORT, lw=0.7, alpha=0.55, zorder=1)
    for i, j in [(2, 27), (9, 31)]:
        ax.plot(*zip(nodes[i], nodes[j]), color=LONG, lw=1.2, alpha=0.9, zorder=2,
                ls=(0, (3, 1.5)))
    ax.scatter(*nodes.T, s=24, c="#37474f", zorder=3, linewidths=0)

    contacts = np.array([[0.16, 0.20], [0.39, 0.20], [0.62, 0.20], [0.85, 0.20]])
    for c in contacts:
        for i in np.argsort(np.linalg.norm(nodes - c, axis=1))[:5]:
            ax.plot([c[0], nodes[i, 0]], [c[1], nodes[i, 1]],
                    color="#1a1a1a", lw=0.45, alpha=0.28, zorder=0)
    ax.scatter(*contacts.T, s=90, marker="s", facecolor="white",
               edgecolor="#1a1a1a", linewidths=1.3, zorder=4)
    ax.text(0.5, 0.115, "contacts read a local patch of tissue; nothing crosses the plane\n"
                        "except through the recurrent edges",
            ha="center", va="top", fontsize=7.4)
    ax.text(0.5, 0.985, "a fixed number of recurrent edges: each epoch the weakest are cut and\n"
                        "the same number regrown, nearby pairs first, long ones kept if useful",
            ha="center", va="top", fontsize=7.4)
    ax.legend(handles=[Line2D([], [], color=SHORT, lw=1.3, label="short edge"),
                       Line2D([], [], color=LONG, lw=1.3, ls=(0, (3, 1.5)), label="surviving long edge"),
                       Line2D([], [], color="#1a1a1a", lw=0.6, alpha=0.4, label="read-out")],
              loc="upper left", fontsize=6.4, frameon=False, bbox_to_anchor=(-0.02, 0.90),
              handlelength=1.4, labelspacing=0.35)


# --------------------------------------------------------------- panel b ----
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
    model.load_state_dict(torch.load(unit / "weights.pt", map_location="cpu", weights_only=True))
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
            top = np.nanmax(row)
            observed[r] = row / top if top and top > 0 else row
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


def panel_replay(axes, replayed: Dict[str, Any]) -> None:
    contacts = replayed["contacts"]
    reference = replayed.get(0, replayed.get(1))
    order = np.argsort(np.nan_to_num(reference["observed"], nan=2.0))
    rows = [(0, "observed"), (0, "generated"), (1, "observed"), (1, "generated")]
    labels = ["Mode A\nobserved", "Mode A\ngenerated", "Mode B\nobserved", "Mode B\ngenerated"]
    image = None
    for ax, (m, key), label in zip(axes, rows, labels):
        if m not in replayed:
            ax.set_axis_off()
            continue
        image = ax.imshow(replayed[m][key][order][None, :], cmap="viridis",
                          vmin=0, vmax=1, aspect="auto")
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=6.4, rotation=0, ha="right", va="center", labelpad=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([contacts[i] for i in order] if label.endswith("generated")
                           and m == 1 else [], rotation=90, fontsize=5.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    if image is not None:
        bar = axes[0].figure.colorbar(image, ax=list(axes), fraction=0.05, pad=0.015,
                                      aspect=28)
        bar.set_ticks([0, 1])
        bar.set_ticklabels(["first", "last"])
        bar.ax.tick_params(labelsize=6)


# --------------------------------------------------------------- panel c ----
def panel_pareto(ax_scatter, ax_delta, pareto: Dict[str, Any]) -> None:
    per_fit = pareto["per_fit"]
    for arm in ("DENSE_TISSUE", "RANDOM_SET", "SPATIAL_SET"):
        fits = sorted(set(per_fit["mean_edge_len_mm"][arm]) & set(per_fit["test_next_bce"][arm]))
        ax_scatter.scatter([per_fit["mean_edge_len_mm"][arm][f] for f in fits],
                           [per_fit["test_next_bce"][arm][f] for f in fits],
                           s=15, color=ARM_COLOUR[arm], label=ARM_LABEL[arm],
                           alpha=0.8, linewidths=0)
    ax_scatter.set_xlabel("mean recurrent edge length (mm)", fontsize=7.5)
    ax_scatter.set_ylabel("held-out next-rank loss", fontsize=7.5)
    ax_scatter.tick_params(labelsize=6.5)
    ax_scatter.legend(fontsize=6, frameon=False, loc="upper left", handletextpad=0.4,
                      labelspacing=0.3, borderpad=0.2)

    # The no-recurrence contrast lives in panel c; repeating it here would be the
    # same question drawn twice.
    keys = [("SPATIAL_SET__vs__RANDOM_SET", "vs uniform\nsparse"),
            ("SPATIAL_SET__vs__DENSE_TISSUE", "vs all-\nto-all")]
    rng = np.random.default_rng(0)
    for i, (key, label) in enumerate(keys):
        block = pareto["contrasts"][key]
        deltas = list(block["delta_by_subject"].values())
        ax_delta.scatter(np.full(len(deltas), i) + rng.normal(0, 0.055, len(deltas)),
                         deltas, s=12, color="#37474f", alpha=0.7, linewidths=0, zorder=3)
        ax_delta.plot([i - 0.27, i + 0.27], [np.median(deltas)] * 2, color=LONG, lw=2.2, zorder=4)
    ax_delta.axhline(0, color="#1a1a1a", lw=0.8, zorder=1)
    lo, hi = ax_delta.get_ylim()
    ax_delta.set_ylim(lo, hi + 0.30 * (hi - lo))
    for i, (key, _) in enumerate(keys):
        block = pareto["contrasts"][key]
        ax_delta.text(i, hi + 0.02 * (hi - lo),
                      f"{p_text(block['p'])}\n{block['n_negative']}/{block['n']}",
                      ha="center", va="bottom", fontsize=5.8)
    ax_delta.set_xticks(range(len(keys)))
    ax_delta.set_xticklabels([lab for _, lab in keys], fontsize=6.4)
    ax_delta.set_xlim(-0.6, len(keys) - 0.4)
    ax_delta.set_ylabel("wiring economy − reference", fontsize=7)
    ax_delta.tick_params(labelsize=6.5)
    budget = pareto["sparse_vs_dense_budget"]
    ax_delta.set_xlabel(
        f"the sparse arms spend {budget['edge_count_ratio_median']:.0%} of the edges\n"
        f"and {budget['total_wiring_length_ratio_median']:.0%} of the total wiring",
        fontsize=6.6)


# --------------------------------------------------------------- panel d ----
def panel_topology(axes, out_root: Path, fit_id: str, seed: int = 0) -> None:
    unit = out_root / "per_subject" / fit_id / "SPATIAL_SET_rnn" / f"seed{seed}"
    plane = np.load(out_root / "cache" / fit_id / "plane.npz")
    g = np.load(unit / "graph.npz")
    nodes = plane["nodes_xy_mm"]
    initial, final, d = g["initial_mask"] > 0, g["mask"] > 0, g["D_mm"]
    membership, _ = module_of_each_node(final, seed=seed)
    palette = plt.get_cmap("tab10")

    for ax, (mask, title, colour_by_module) in zip(
            axes, ((initial, "proposed", False),
                   (final, "kept", False),
                   (final, "modules", True))):
        for i, j in zip(*np.nonzero(mask)):
            long = d[i, j] > 10.0
            ax.plot(*zip(nodes[i], nodes[j]), color=LONG if long else SHORT,
                    lw=0.55 if long else 0.4, alpha=0.5 if long else 0.35, zorder=1)
        colours = ([palette(membership[i] % 10) for i in range(len(nodes))]
                   if colour_by_module else ["#37474f"] * len(nodes))
        ax.scatter(*nodes.T, s=11, c=colours, zorder=3, linewidths=0)
        ax.set_title(title, fontsize=6.8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    axes[0].legend(handles=[Line2D([], [], color=SHORT, lw=1.0, label="≤ 10 mm"),
                            Line2D([], [], color=LONG, lw=1.0, label="> 10 mm")],
                   fontsize=5.6, frameon=False, loc="lower left", handlelength=1.2)


# --------------------------------------------------------------- panel e ----
LADDER_KEYS = [("modularity_q", "modularity"), ("clustering", "clustering"),
               ("small_worldness", "small-\nworldness"),
               ("long_edge_fraction", "long-edge\nfraction"),
               ("participation_mean", "participation")]
LADDER_REFS = [("growth_prior_C1", "same growth rule,\nnever trained", "#8d6e63"),
               ("task_free_dynamics_C2", "same training,\norder destroyed", "#5c6bc0"),
               ("length_preserving_rewire_C3", "same edge lengths,\nrewired", "#26a69a")]


def panel_ladder(ax, topology: Dict[str, Any]) -> None:
    """Learned topology minus each reference, in units of the learned value."""
    gates = topology["gates"]
    width = 0.26
    for k, (key, _) in enumerate(LADDER_KEYS):
        learned = gates[key]["learned_median"]
        for r, (ref, ref_label, colour) in enumerate(LADDER_REFS):
            block = gates.get(key, {}).get(ref)
            if not block or "median_delta" not in block:
                continue
            value = block["median_delta"] / abs(learned) if learned else np.nan
            x = k + (r - 1) * width
            ax.bar(x, value, width * 0.88, color=colour, alpha=0.92,
                   label=ref_label if k == 0 else None, zorder=2)
            if block.get("p", 1.0) < 0.05:
                ax.text(x, value + (0.012 if value >= 0 else -0.012), "*", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=9, zorder=3)
    ax.axhline(0, color="#1a1a1a", lw=0.9, zorder=1)
    ax.set_xticks(range(len(LADDER_KEYS)))
    ax.set_xticklabels([lab for _, lab in LADDER_KEYS], fontsize=6.6)
    ax.set_xlim(-0.6, len(LADDER_KEYS) - 0.4)
    ax.set_ylabel("learned − reference,\nrelative to the learned value", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.1, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.19), columnspacing=1.4, handlelength=1.1)
    ax.text(0.995, 0.965, "* patient-paired p < 0.05  (n = 21)", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.2)


# --------------------------------------------------------------- panel f ----
def panel_lesion(ax, lesion: Dict[str, Any]) -> None:
    rows = lesion["per_unit"]
    module = [r["module_delta_next_bce"] for r in rows]
    patch = [r["matched_patch_delta_next_bce"] for r in rows]
    rng = np.random.default_rng(5)
    for i, values in enumerate((module, patch)):
        ax.scatter(np.full(len(values), i) + rng.normal(0, 0.05, len(values)), values,
                   s=11, color="#37474f", alpha=0.6, linewidths=0, zorder=3)
        ax.plot([i - 0.26, i + 0.26], [np.nanmedian(values)] * 2, color=LONG, lw=2.2, zorder=4)
    for a, b in zip(module, patch):
        ax.plot([0, 1], [a, b], color="#9e9e9e", lw=0.35, alpha=0.45, zorder=2)
    ax.axhline(0, color="#1a1a1a", lw=0.8, zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["one whole module", "matched\ncontiguous patch"], fontsize=6.8)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("rise in next-rank loss after cutting,\nwithout retraining", fontsize=7)
    ax.tick_params(labelsize=6.5)
    gate = lesion["gates"].get("module_vs_matched_contiguous_patch", {})
    ax.text(0.5, 0.97, f"the matched patch costs more in "
                       f"{gate.get('n_positive', 0)}/{gate.get('n', 0)} patients, "
                       f"{p_text(gate.get('p', float('nan')))}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.4)


# ------------------------------------------------------------------ main ----
def pick_representative(out_root: Path) -> str:
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    shared = sorted((r for r in manifest["fits"] if r["scope"] == "shared"),
                    key=lambda r: r["n_nodes"])
    return shared[len(shared) // 2]["fit_id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--cell", default="rnn")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    a = load_analysis(out_root, args.cell)
    representative = pick_representative(out_root)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7})
    fig = plt.figure(figsize=(11.8, 11.6))
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40,
                              height_ratios=[1.00, 1.05, 0.92])

    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
    ax_a = fig.add_subplot(top[0])
    panel_model(ax_a)
    ax_a.set_title("a  A recurrent network on the patient's own tissue plane",
                   loc="left", fontsize=9.5, fontweight="bold", pad=6)

    b = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=top[1], hspace=0.30)
    axes_b = [fig.add_subplot(b[i]) for i in range(4)]
    replayed = replay(out_root, representative)
    if replayed:
        panel_replay(axes_b, replayed)
    axes_b[0].set_title("b  Held-out events regenerated from their own first contact",
                        loc="left", fontsize=9.5, fontweight="bold", pad=8)

    middle = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.34,
                                              width_ratios=[1.35, 1.0])
    ax_c = fig.add_subplot(middle[0])
    if a["sufficiency"]:
        panel_sufficiency(ax_c, a["sufficiency"])
    ax_c.set_title("c  Can it regenerate this patient's propagation?", loc="left",
                   fontsize=9.5, fontweight="bold", pad=8)

    d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=middle[1], wspace=0.60)
    ax_d1 = fig.add_subplot(d[0])
    panel_pareto(ax_d1, fig.add_subplot(d[1]), a["pareto"])
    ax_d1.set_title("d  What the wiring budget costs", loc="left", fontsize=9.5,
                    fontweight="bold", pad=20)

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.30,
                                              width_ratios=[1.55, 1.0])
    ax_e = fig.add_subplot(bottom[0])
    panel_ladder(ax_e, a["topology"])
    ax_e.set_title("e  Is the topology the task's doing?", loc="left", fontsize=9.5,
                   fontweight="bold", pad=26)
    ax_f = fig.add_subplot(bottom[1])
    panel_lesion(ax_f, a["lesion"])
    ax_f.set_title("f  Is a module the necessary unit?", loc="left", fontsize=9.5,
                   fontweight="bold", pad=6)

    figures = out_root / "figures"
    figures.mkdir(exist_ok=True)
    stem = figures / f"topic5_wiring_economy_rnn_main_{args.cell}"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    primary = a["pareto"]["contrasts"]["SPATIAL_SET__vs__RANDOM_SET"]
    (figures / f"{stem.name}_metadata.json").write_text(json.dumps({
        "cell": args.cell, "representative_fit": representative,
        "n_patients": primary["n"],
        "primary_contrast_median_delta": primary["median_delta"],
        "primary_contrast_p": primary["p"],
        "n_cross_mode_patients": 11,
    }, indent=2))
    print(f"wrote {stem}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
