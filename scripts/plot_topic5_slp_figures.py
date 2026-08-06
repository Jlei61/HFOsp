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
    ax.set_title("A  A contact samples tissue, not a node", loc="left", pad=16)
    # The plane was estimated from the whole recording, so it is not a geometry
    # that could have been known before the test events were seen.
    ax.text(0.0, 1.005, "retrospective, test-informed propagation plane",
            transform=ax.transAxes, fontsize=6.0, color="#8a5a00", style="italic")
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
    # Not a clean factorial decomposition: the arms differ in state
    # representation, recurrent parameterisation AND output mapping at once.
    # They share the prediction task and the loss, not the output head.
    ax.set_title("B  Only unconstrained recurrence helps", loc="left", pad=16)
    ax.text(0.0, 1.005, "same task and loss; output heads differ",
            transform=ax.transAxes, fontsize=6.0, color="0.35", style="italic")


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
    # "Relative order of push" is coarse reachability, not a recovered graph.
    ax.set_title("C  Only coarse reachability recovers", loc="left", pad=16)
    ax.text(0.02, 0.97, "dotted 0 = chance\ndashed = required to report",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="0.35")


def _loco_rows() -> list[dict] | None:
    import csv
    path = OUT / "leave_contact_out_patient_first.csv"
    if not path.exists():
        return None
    return list(csv.DictReader(path.open()))


def _loco_swarm(ax, values_by_mode: dict, ylabel: str, title: str,
                subtitle: str | None = None) -> None:
    modes = [m for m in ("weak", "strong") if m in values_by_mode]
    for i, mode in enumerate(modes):
        v = np.asarray(values_by_mode[mode], float)
        jitter = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.24
        colour = np.where(v > 0, "#2a9d8f", "#d1495b")
        ax.scatter(np.full(len(v), i) + jitter, v, s=24, c=colour,
                   alpha=0.85, linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(["Contact still visible\nin the sequence",
                        "Contact fully removed"][:len(modes)], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=16)
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=6.0, color="0.35", style="italic")


def panel_d1(ax) -> None:
    """Absolute score at the withheld contacts.  This is the question asked."""
    rows = _loco_rows()
    if not rows:
        ax.text(0.5, 0.5, "leave-contact-out not run", ha="center", va="center")
        ax.set_axis_off()
        return
    by_mode: dict[str, list[float]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(
            float(r["raw_heldout_tissue_minus_contact"])
        )
    _loco_swarm(
        ax, by_mode,
        "Tissue field better than the contact\ngraph AT the unseen contacts",
        "D1  Absolute score at unseen contacts",
        "positive = tissue field wins",
    )


def panel_d2(ax) -> None:
    """The same patients, scored on loss relative to each model's own baseline."""
    rows = _loco_rows()
    if not rows:
        ax.text(0.5, 0.5, "leave-contact-out not run", ha="center", va="center")
        ax.set_axis_off()
        return
    by_mode: dict[str, list[float]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(
            float(r["degradation_tissue_minus_contact"])
        )
    _loco_swarm(
        ax, by_mode,
        "Tissue field loses less relative\nto its OWN retained-contact score",
        "D2  Smoother degradation, worse score",
        "a smaller drop from a much lower baseline",
    )


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
    # Computational reach of the fitted sparse graph.  The recovery gate found
    # edge identity unrecoverable, so these are not biological pathways.
    ax.set_title("E  Sparse-graph reachability diagnostic", loc="left", pad=16)
    ax.text(0.0, 1.005, "computational reach, not recovered pathways",
            transform=ax.transAxes, fontsize=6.0, color="0.35", style="italic")


def panel_f(ax) -> None:
    """Does the wiring economy explain the shortfall?  Take it away and see.

    The flow-ordering comparison that used to sit here did not survive its
    geometry control -- an untrained graph on the same node positions produces
    the same within-versus-between gap -- so it moved to the supplement and this
    slot now carries the diagnostic that does separate cleanly.
    """
    stats_path = OUT / "cohort_statistics.json"
    if not stats_path.exists():
        ax.text(0.5, 0.5, "cohort not aggregated", ha="center", va="center")
        ax.set_axis_off()
        return
    entry = (json.loads(stats_path.read_text())["comparisons"]["primary"]
             .get("ceiling_dense_over_learned", {}).get("all", {}))
    if entry.get("status") != "COMPLETE":
        ax.text(0.5, 0.5, "dense-adjacency control not run", ha="center", va="center")
        ax.set_axis_off()
        return
    deltas = np.array(list(entry["per_patient_delta"].values()))
    jitter = (np.random.default_rng(0).random(len(deltas)) - 0.5) * 0.30
    colour = np.where(deltas > 0, "#2a9d8f", "#d1495b")
    ax.scatter(np.zeros(len(deltas)) + jitter, deltas, s=26, c=colour,
               alpha=0.85, linewidths=0)
    ax.plot([-0.35, 0.35], [np.median(deltas)] * 2, color="0.15", lw=2)
    ax.axhline(0.0, color="0.55", lw=1, ls=":")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([0])
    ax.set_xticklabels([f"Fully connected minus\nsparse tissue field\n(n={len(deltas)})"],
                       fontsize=6.5)
    ax.set_ylabel("Improvement from removing\nthe sparsity constraint")
    ax.set_title("F  Removing sparsity does not help", loc="left", pad=16)
    ax.text(0.0, 1.005, "so the wiring budget is not the limitation",
            transform=ax.transAxes, fontsize=6.0, color="0.35", style="italic")


def panel_scope(ax) -> None:
    """What this figure is not evidence for.

    Every panel above is a negative or a diagnostic, and a reader who takes one
    of them for a positive claim has read the figure wrong.  Saying so on the
    figure costs one slot and removes the most likely misreading.
    """
    ax.set_axis_off()
    ax.set_title("Scope of this figure", loc="left")
    lines = [
        ("Supported", "#2a6b4f", [
            "within-event history predicts the next rank",
            "  (unconstrained recurrence, 21/21 patients)",
        ]),
        ("Not supported", "#8c2f39", [
            "tissue field beating an unconstrained model",
            "prediction at contacts held out of training",
            "which connections exist, or their direction",
            "patient-specific ordering beyond node geometry",
        ]),
        ("Conditions", "#5a5a5a", [
            "plane fitted on the whole recording",
            "teacher-forced next-rank; no free rollout",
            "one seed carries coverage, two carry stability",
        ]),
    ]
    y = 0.93
    for heading, colour, items in lines:
        ax.text(0.0, y, heading, transform=ax.transAxes, fontsize=7.5,
                color=colour, fontweight="bold")
        y -= 0.085
        for item in items:
            ax.text(0.03, y, item, transform=ax.transAxes, fontsize=6.6,
                    color="0.25")
            y -= 0.075
        y -= 0.035


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1146")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})

    # Seven panels, not six: the held-out-contact question splits into an
    # absolute comparison and a relative one, and those are different questions
    # with opposite answers.  Collapsing them into the relative one alone was
    # what made the original figure read as a positive result.
    fig = plt.figure(figsize=(15.0, 7.6))
    grid = fig.add_gridspec(2, 4, hspace=0.78, wspace=0.46)
    panel_a(fig.add_subplot(grid[0, 0]), args.subject)
    panel_b(fig.add_subplot(grid[0, 1]))
    panel_c(fig.add_subplot(grid[0, 2]))
    panel_d1(fig.add_subplot(grid[0, 3]))
    panel_d2(fig.add_subplot(grid[1, 0]))
    panel_e(fig.add_subplot(grid[1, 1]))
    panel_f(fig.add_subplot(grid[1, 2]))
    panel_scope(fig.add_subplot(grid[1, 3]))

    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"topic5_slp_rnn_v0_1_overview.{extension}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {(FIGURES / 'topic5_slp_rnn_v0_1_overview.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
