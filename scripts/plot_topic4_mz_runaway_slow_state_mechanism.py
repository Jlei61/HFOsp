"""Single-panel MZ slow-state mechanism candidate from completed natural trajectories.

The shaded band is the observed D range at natural operational-runaway first crossings,
not a fitted separatrix or a proven bifurcation boundary.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slow_fast_transition")
PER_STATE = os.path.join(RESULT, "per_state")
FIG_DIR = os.path.join(RESULT, "figures")

CONDITIONS = ["z_only", "mz_runaway", "mz_edge", "mz_plateau"]
COLORS = {
    "z_only": "#7A7A7A",
    "mz_runaway": "#D97925",
    "mz_edge": "#7753A6",
    "mz_plateau": "#2878B5",
}
LABELS = {
    "z_only": "z only",
    "mz_runaway": r"z+m runaway  ($\tau_m$=0.5 s)",
    "mz_edge": r"z+m edge  ($\tau_m$=1 s)",
    "mz_plateau": r"z+m plateau  ($\tau_m$=2 s)",
}
SEEDS = (1, 3, 4)


def _load_natural(condition: str, seed: int) -> dict:
    path = os.path.join(PER_STATE, f"{condition}_seed{seed}_natural.npz")
    data = np.load(path)
    crossing = float(data["crossing_ms"])
    crossing = crossing if np.isfinite(crossing) else None
    t = np.asarray(data["t_ms"], float)
    keep = np.ones(t.size, dtype=bool) if crossing is None else t <= crossing
    return {
        "t": t[keep],
        "D": np.asarray(data["D"], float)[keep],
        "a": np.asarray(data["a"], float)[keep],
        "crossing": crossing,
    }


def _crossing_record(condition: str, seed: int) -> dict | None:
    path = os.path.join(PER_STATE, f"{condition}_seed{seed}_first_crossing.json")
    return json.load(open(path)) if os.path.exists(path) else None


def _mean_path(paths: list[dict], n: int = 700) -> tuple[np.ndarray, np.ndarray]:
    progress = np.linspace(0.0, 1.0, n)
    d_paths, a_paths = [], []
    for path in paths:
        source = np.linspace(0.0, 1.0, len(path["D"]))
        d_paths.append(np.interp(progress, source, path["D"]))
        a_paths.append(np.interp(progress, source, path["a"]))
    return np.mean(d_paths, axis=0), np.mean(a_paths, axis=0)


def _path_arrows(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    for frac in (0.35, 0.62, 0.82):
        i = min(len(x) - 2, max(0, int(round(frac * (len(x) - 1)))))
        j = min(len(x) - 1, i + max(2, len(x) // 45))
        ax.annotate(
            "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25, mutation_scale=9),
            zorder=7,
        )


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = {condition: [_load_natural(condition, seed) for seed in SEEDS] for condition in CONDITIONS}
    crossings = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            record = _crossing_record(condition, seed)
            if record is not None:
                crossings.append(record)
    crossing_d = np.asarray([record["D"] for record in crossings], float)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    lo, hi = float(crossing_d.min()), float(crossing_d.max())
    ax.axvspan(lo, hi, color="#C94C4C", alpha=0.10, lw=0, zorder=0)
    ax.axvline(float(crossing_d.mean()), color="#B53A3A", ls="--", lw=1.0, alpha=0.75, zorder=1)
    ax.text(
        0.5 * (lo + hi), 0.000795,
        "observed first-crossing D range\n(not a fitted separatrix)",
        ha="center", va="top", fontsize=8.0, color="#8E3333",
    )

    for condition in CONDITIONS:
        color = COLORS[condition]
        for path in paths[condition]:
            ax.plot(path["D"], path["a"], color=color, lw=0.75, alpha=0.28, zorder=2)
        mean_d, mean_a = _mean_path(paths[condition])
        line_style = "--" if condition == "mz_edge" else "-"
        line_alpha = 0.78 if condition == "mz_edge" else 0.98
        ax.plot(mean_d, mean_a, color=color, lw=2.25, ls=line_style, alpha=line_alpha, zorder=5)
        _path_arrows(ax, mean_d, mean_a, color)

    for record in crossings:
        condition = record["condition"]
        ax.scatter(
            record["D"], record["a"], marker="*", s=115,
            facecolor=COLORS[condition], edgecolor="white", linewidth=0.8, zorder=9,
        )

    for condition in ("mz_edge", "mz_plateau"):
        for path in paths[condition]:
            if path["crossing"] is None:
                ax.scatter(
                    path["D"][-1], path["a"][-1], s=28,
                    facecolor="white", edgecolor=COLORS[condition], linewidth=1.2, zorder=8,
                )

    ax.scatter(0.0, 0.0, s=42, facecolor="white", edgecolor="#555", linewidth=1.0, zorder=8)
    ax.annotate(
        "inhibitory efficacy $\\downarrow$  →  $D$ increases  (push)",
        xy=(0.060, 0.000028), xytext=(0.028, 0.000028),
        arrowprops=dict(arrowstyle="-|>", color="#9A5D3A", lw=1.35),
        ha="center", va="bottom", fontsize=8.3, color="#75452F",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82),
    )
    ax.annotate(
        "spiking  →  adaptation $a$ increases  (brake)",
        xy=(0.008, 0.00043), xytext=(0.008, 0.00017),
        arrowprops=dict(arrowstyle="-|>", color="#315F8A", lw=1.35),
        ha="left", va="center", fontsize=8.3, color="#315F8A",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82),
    )
    ax.annotate(
        "adaptation diverts / arrests the slow path",
        xy=(0.050, 0.000255), xytext=(0.058, 0.00047),
        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9),
        ha="center", va="bottom", fontsize=8.3, color="#444",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82),
    )

    edge_crossing = [record for record in crossings if record["condition"] == "mz_edge"]
    if edge_crossing:
        record = edge_crossing[0]
        ax.annotate(
            "edge: 1/3 seeds crosses",
            xy=(record["D"], record["a"]), xytext=(0.072, 0.00067),
            arrowprops=dict(arrowstyle="->", color=COLORS["mz_edge"], lw=0.9),
            fontsize=8.0, color=COLORS["mz_edge"], ha="center",
        )

    handles = [Line2D([0], [0], color=COLORS[c], lw=2.4, label=LABELS[c]) for c in CONDITIONS]
    handles += [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#B53A3A", markeredgecolor="white",
               markersize=11, label="natural operational-runaway crossing"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#666",
               markersize=6, label="bounded observation endpoint"),
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.98), ncol=2,
        frameon=True, framealpha=0.94, facecolor="white", edgecolor="none",
        fontsize=7.8, handlelength=2.2, columnspacing=1.4,
    )
    legend.set_zorder(30)

    ax.set_xlim(-0.002, 0.103)
    ax.set_ylim(-0.000025, 0.00082)
    ax.set_xlabel(r"disinhibition  $D=1-\bar z$", fontsize=10.5)
    ax.set_ylabel(r"adaptation  $a=\eta_m\bar m/I_{EE}$", fontsize=10.5)
    ax.set_title("Slow-state paths toward operational runaway", loc="left", fontsize=11.5, weight="bold")
    ax.text(1.0, -0.13, "MZ model · E1146 · natural trajectories · seeds 1/3/4",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="#555")
    ax.grid(color="#D9D9D9", lw=0.55, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.7)
    fig.tight_layout()

    base = os.path.join(FIG_DIR, "mz_runaway_slow_state_mechanism_candidate")
    fig.savefig(base + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(base + ".png")


if __name__ == "__main__":
    main()
