"""Figures for the Topic 5 spatial latent propagation RNN v0.1.

One panel per independent question (CLAUDE.md §7):

A  what the model is           -- contacts, latent field and the local readout
B  does it predict             -- patient-level arm comparison
C  is structure recoverable    -- the three layers of the synthetic gate
D  does it reach unseen space  -- leave-contact-out per patient
E  did the wiring economy act  -- edge length and hop reachability

Colour follows the repository lock: viridis for anything ordered, a diverging
red-blue for signed differences, and no internal jargon in reader-facing text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1"
FIGURES = OUT / "figures"

ARM_LABEL = {
    "STATIC_CONTACT": "Static contact rate",
    "ORDINARY_GRU": "Unconstrained recurrent",
    "CONTACT_GRAPH_RNN": "Contact-node graph",
    "LATENT_FIXED_LOCAL_RNN": "Tissue field, fixed local graph",
    "LATENT_LEARNED_SPATIAL_RNN": "Tissue field, learned graph",
}
ARM_ORDER = list(ARM_LABEL)
ARM_COLOUR = dict(zip(ARM_ORDER, plt.get_cmap("viridis")(np.linspace(0.08, 0.88, 5))))


def panel_a(ax, subject: str) -> None:
    plane = np.load(OUT / "cache" / subject / "plane_coordinates.npz", allow_pickle=True)
    nodes = np.load(OUT / "cache" / subject / "latent_nodes.npz")["nodes_xy"]
    H = np.load(OUT / "cache" / subject / "seeg_operator.npz")["H"]
    xy = plane["xy_mm"]

    ax.scatter(nodes[:, 0], nodes[:, 1], s=12, c="0.82", marker="o",
               linewidths=0, label="Tissue units", zorder=1)
    example = int(np.argmin(np.linalg.norm(xy - xy.mean(0), axis=1)))
    weight = H[example]
    seen = weight > 0
    # Size alone carries the weight.  A colour bar here would collide with the
    # next panel's axis label and the exact share is not the point of the panel.
    ax.scatter(nodes[seen, 0], nodes[seen, 1],
               s=18 + 200 * weight[seen] / weight.max(),
               color="#3d5a80", alpha=0.75, linewidths=0, zorder=2,
               label="Units it reads (size = share)")
    ax.scatter(xy[:, 0], xy[:, 1], s=40, marker="s", facecolor="white",
               edgecolor="0.15", linewidth=1.1, zorder=3, label="Recording contacts")
    ax.scatter(xy[example, 0], xy[example, 1], s=80, marker="s", facecolor="#d1495b",
               edgecolor="0.15", linewidth=1.1, zorder=4, label="Contact shown reading")
    ax.set_xlabel("Along propagation axis (mm)")
    ax.set_ylabel("Across axis (mm)")
    ax.set_title("A  A contact reads a patch of tissue, it is not a node", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              frameon=False, fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    ax.set_aspect("equal", adjustable="datalim")


def panel_b(ax) -> None:
    path = OUT / "cohort_statistics.json"
    metrics = OUT / "patient_prediction_metrics.csv"
    if not metrics.exists():
        ax.text(0.5, 0.5, "cohort not yet aggregated", ha="center", va="center")
        ax.set_axis_off()
        return
    import csv
    rows = list(csv.DictReader(metrics.open()))
    by_arm: dict = {}
    for row in rows:
        if row["test_next_bce"]:
            by_arm.setdefault(row["arm"], {}).setdefault(row["subject"], []).append(
                float(row["test_next_bce"]))
    if "STATIC_CONTACT" not in by_arm:
        ax.text(0.5, 0.5, "baseline arm missing", ha="center", va="center")
        ax.set_axis_off()
        return
    # Paired differences from the static baseline, not raw error.  The raw level
    # falls as the montage grows -- with more contacts most are absent at any
    # step, so predicting absence is enough to look good -- which makes levels
    # incomparable between patients.  The within-patient difference is not.
    reference = {s: np.median(v) for s, v in by_arm["STATIC_CONTACT"].items()}
    arms = [a for a in ARM_ORDER if a in by_arm and a != "STATIC_CONTACT"]
    for i, arm in enumerate(arms):
        subjects = [s for s in by_arm[arm] if s in reference]
        values = np.array([reference[s] - np.median(by_arm[arm][s]) for s in subjects])
        jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.22
        ax.scatter(np.full(len(values), i) + jitter, values, s=18,
                   color=ARM_COLOUR[arm], alpha=0.8, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(values)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("Improvement on the static\ncontact rate, same patient")
    ax.set_title("B  One point per patient, seeds pooled within patient", loc="left")


def panel_c(ax) -> None:
    gate_path = OUT / "synthetic" / "RECOVERY_GATE.json"
    if not gate_path.exists():
        ax.text(0.5, 0.5, "recovery gate not run", ha="center", va="center")
        ax.set_axis_off()
        return
    gate = json.loads(gate_path.read_text())
    cells = gate["cells"]
    # The three layers are measured on different scales with different chance
    # levels -- a ranking score against 0.5, a proportion against 0.5, and a
    # correlation against 0.  Putting the raw numbers on one axis would invite
    # comparisons that mean nothing, so each is rescaled so that its own chance
    # sits at 0 and a perfect answer at 1.
    n_cells = len(cells)
    layers = [
        ("Which\nconnections\nexist", [c["edge_auc"] for c in cells], 0.5,
         gate["edge_identity"]["floor"], "continuous"),
        ("Which way\nactivity\ntravels", [float(c["flow_sign_agrees"]) for c in cells], 0.5,
         gate["axis_direction"]["floor"], "proportion"),
        ("Relative\norder of\npush", [c["flow_node_spearman"] for c in cells], 0.0,
         None, "continuous"),
    ]
    lows, highs = [], []
    for i, (name, raw, chance, floor, kind) in enumerate(layers):
        raw = np.array(raw, float)
        if kind == "proportion":
            # Individual outcomes here are 0 or 1; drawing them as points would
            # put half the cells off any sensible axis. The proportion with its
            # binomial interval is the quantity that actually carries meaning.
            from scipy.stats import binomtest
            k = int(raw.sum())
            interval = binomtest(k, n_cells, 0.5).proportion_ci()
            centre = (k / n_cells - chance) / (1 - chance)
            lo = (interval.low - chance) / (1 - chance)
            hi = (interval.high - chance) / (1 - chance)
            colour = "#2a9d8f" if (k / n_cells) >= floor else "#d1495b"
            ax.errorbar([i], [centre], yerr=[[centre - lo], [hi - centre]],
                        fmt="o", ms=6, color=colour, capsize=3, lw=1.2)
            lows.append(lo); highs.append(hi)
        else:
            values = (raw - chance) / (1.0 - chance)
            jitter = (np.random.default_rng(i).random(len(values)) - 0.5) * 0.24
            passed = floor is None or np.median(raw) >= floor
            colour = "#2a9d8f" if passed else "#d1495b"
            ax.scatter(np.full(len(values), i) + jitter, values, s=26, color=colour,
                       alpha=0.85, linewidths=0)
            ax.plot([i - 0.3, i + 0.3], [np.median(values)] * 2, color="0.15", lw=2)
            lows.append(values.min()); highs.append(values.max())
        if floor is not None:
            ax.plot([i - 0.34, i + 0.34], [(floor - chance) / (1 - chance)] * 2,
                    color="0.25", lw=1.1, ls="--")
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([n for n, *_ in layers], fontsize=6.5)
    ax.set_ylabel("Recovered, rescaled so that\nchance is 0 and perfect is 1")
    span = max(highs) - min(lows)
    ax.set_ylim(min(lows) - 0.12 * span, max(highs) + 0.22 * span)
    ax.set_title("C  Only the weakest structural layer survives", loc="left")
    ax.text(0.02, 0.97, "dotted 0 = chance\ndashed = required to report",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="0.35")


def panel_d(ax) -> None:
    path = OUT / "leave_contact_out_summary.json"
    if not path.exists():
        ax.text(0.5, 0.5, "leave-contact-out not run", ha="center", va="center")
        ax.set_axis_off()
        return
    summary = json.loads(path.read_text())
    modes = [m for m in ("weak", "strong")
             if summary["comparisons"].get(m, {}).get("status") == "COMPLETE"]
    for i, mode in enumerate(modes):
        entry = summary["comparisons"][mode]
        deltas = np.array(list(entry["per_patient_delta"].values()))
        jitter = (np.random.default_rng(i).random(len(deltas)) - 0.5) * 0.24
        colour = np.where(deltas > 0, "#2a9d8f", "#d1495b")
        ax.scatter(np.full(len(deltas), i) + jitter, deltas, s=24, c=colour,
                   alpha=0.85, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(deltas)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(["Contact still visible\nin the sequence",
                        "Contact fully removed"][:len(modes)], fontsize=7)
    ax.set_ylabel("Tissue field advantage at\ncontacts never trained on")
    ax.set_title("D  Predicting where no electrode taught the model", loc="left")


def panel_e(ax) -> None:
    metrics = OUT / "patient_prediction_metrics.csv"
    if not metrics.exists():
        ax.text(0.5, 0.5, "cohort not yet aggregated", ha="center", va="center")
        ax.set_axis_off()
        return
    import csv
    rows = [r for r in csv.DictReader(metrics.open())
            if r["arm"] == "LATENT_LEARNED_SPATIAL_RNN" and r.get("mean_edge_length")]
    if not rows:
        ax.text(0.5, 0.5, "learned-graph arm not yet run", ha="center", va="center")
        ax.set_axis_off()
        return
    length = np.array([float(r["mean_edge_length"]) for r in rows])
    reach = np.array([float(r["hop_reachability"]) for r in rows])
    ax.scatter(length, reach, s=26, color="#3d5a80", alpha=0.8, linewidths=0)
    ax.axvline(1.0, color="0.55", lw=1, ls=":")
    ax.set_xlabel("Mean connection length\n(1 = typical spacing between tissue units)")
    ax.set_ylabel("Observed steps the graph\ncan actually carry")
    ax.set_ylim(0, 1.02)
    ax.set_title("E  Short connections, and they reach far enough", loc="left")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1146")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(11.5, 6.4))
    grid = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)
    panel_a(fig.add_subplot(grid[0, 0]), args.subject)
    panel_b(fig.add_subplot(grid[0, 1]))
    panel_c(fig.add_subplot(grid[0, 2]))
    panel_d(fig.add_subplot(grid[1, 0]))
    panel_e(fig.add_subplot(grid[1, 1]))
    note = fig.add_subplot(grid[1, 2])
    note.set_axis_off()

    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"topic5_slp_rnn_v0_1_overview.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIGURES / 'topic5_slp_rnn_v0_1_overview.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
