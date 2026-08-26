"""Paper-ready figure style for the Epi-PRSSM v0.1 asset packages.

Every constant here comes from the figure contract
``docs/superpowers/specs/2026-08-18-topic5-epi-prssm-figure-contract.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

MM = 1.0 / 25.4
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 180.0

#: semantic colours; objects keep the same colour across every asset
COLOR = {
    "scaffold": "#3F3F3F",       # fixed patient scaffold / baseline / reference
    "observer": "#9A9A9A",       # observer correction, thin dashed
    "G0": "#C58F3D",             # leaky baseline (real elapsed time)
    "G1": "#4C78A8",             # stable linear graph recurrent
    "G2": "#238A8D",             # nonlinear graph recurrent
    "G3": "#6A51A3",             # resource-anchored generator
    "exposure": "#A35E48",       # IED exposure R2/R3
    "onset": "#B33A3A",          # clinical onset
    "null": "#BFBFBF",           # null / control distributions
    "epilepsiae": "#2F5D8A",
    "yuquan": "#8A5A2F",
}
ARM_COLOR = {
    "static": COLOR["scaffold"], "frozen_state": "#B4B4B4", "frozen_state_node": "#5C5C5C", "event_index_ewma": "#B0B0B0",
    "ct_ewma_g0": "#C58F3D", "unconstrained_gru": "#8C6BA8",
    "g1_graph_clds": COLOR["G1"], "g2_graph_gru_ode": COLOR["G2"],
    "g3_resource": COLOR["G3"], "g3_flexible_resource_control": "#9A86C4",
    "g2_compressed_state": "#7FBFC0",
    # arms added after the first ladder pass: sensitivity, order-weighted, timing baseline
    "ct_ewma_g0_long_window": "#DDB57A", "g2_graph_gru_ode_long_window": "#7CBFC1",
    "g1_graph_clds_order_weighted": "#2F5D8A",
    "g3_resource_on_g1": "#8C6BA8",
    "nuisance_timing_baseline": "#9A9A9A",
    "nuisance_timing_baseline_order_weighted": "#C4C4C4",
}
ARM_LABEL = {
    "static": "fixed repertoire", "frozen_state": "frozen state (global)",
    "frozen_state_node": "frozen state (node-resolved)", "event_index_ewma": "event-index EWMA",
    "ct_ewma_g0": "G0 leaky (real time)", "unconstrained_gru": "unconstrained GRU",
    "g1_graph_clds": "G1 graph-CLDS", "g2_graph_gru_ode": "G2 graph-GRU-ODE",
    "g3_resource": "G3 + resource", "g3_flexible_resource_control": "G3 + flexible r-correction",
    "g2_compressed_state": "G2 compressed state",
    "ct_ewma_g0_long_window": "G0 leaky (long window)",
    "g2_graph_gru_ode_long_window": "G2 graph-GRU-ODE (long window)",
    "g1_graph_clds_order_weighted": "G1 graph-CLDS (order-weighted)",
    "g3_resource_on_g1": "G1 + resource",
    "nuisance_timing_baseline": "observable-timing baseline",
    "nuisance_timing_baseline_order_weighted": "observable-timing baseline (order-weighted)",
}

#: reserve colours handed to arms that reach a figure before they reach ARM_COLOR;
#: a missing arm must not abort an unattended chain, but it must stay visible.
_RESERVE = ("#B07AA1", "#59A14F", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC")
_UNMAPPED: dict[str, str] = {}


def arm_color(arm: str) -> str:
    """Colour for an arm; unmapped arms get a stable reserve colour and are recorded."""
    if arm in ARM_COLOR:
        return ARM_COLOR[arm]
    if arm not in _UNMAPPED:
        _UNMAPPED[arm] = _RESERVE[len(_UNMAPPED) % len(_RESERVE)]
        print(f"[figure_style] WARNING: arm {arm!r} has no assigned colour; "
              f"using reserve {_UNMAPPED[arm]}. Add it to ARM_COLOR/ARM_LABEL.",
              file=sys.stderr)
    return _UNMAPPED[arm]


def arm_label(arm: str) -> str:
    """Legend label for an arm; unmapped arms fall back to the raw arm name."""
    return ARM_LABEL.get(arm, arm)


def unmapped_arms() -> dict[str, str]:
    """Arms that reached a figure without an assigned colour, for figure metadata."""
    return dict(_UNMAPPED)

LW_MAIN, LW_INDIVIDUAL, LW_REFERENCE = 1.35, 0.65, 0.7
FS_PANEL, FS_TITLE, FS_AXIS, FS_TICK = 10.5, 8.5, 7.5, 6.8


def apply_style() -> None:
    rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FS_AXIS, "axes.labelsize": FS_AXIS, "axes.titlesize": FS_TITLE,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.solid_capstyle": "round",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.grid": False, "figure.autolayout": False,
    })


def figure(width_mm: float = DOUBLE_COLUMN_MM, height_mm: float = 120.0, **kwargs):
    apply_style()
    return plt.subplots(figsize=(width_mm * MM, height_mm * MM), **kwargs)


def panel_letter(ax, letter: str, dx: float = -0.085, dy: float = 1.06) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="top", ha="left")


def zero_line(ax, orientation: str = "h") -> None:
    if orientation == "h":
        ax.axhline(0.0, color="#4D4D4D", lw=LW_REFERENCE, ls=(0, (3, 2)), zorder=0)
    else:
        ax.axvline(0.0, color="#4D4D4D", lw=LW_REFERENCE, ls=(0, (3, 2)), zorder=0)


def save_asset(fig, asset_id: str, root: Path, *, metadata: dict[str, Any],
               readme_entries: list[dict[str, str]], suffix: str = "") -> dict[str, str]:
    """Write PNG, vector PDF, metadata JSON and README in one call.

    The README is written after the figure files exist, never as a placeholder.
    """
    directory = root / asset_id / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{asset_id}{suffix}"
    png, pdf = directory / f"{stem}.png", directory / f"{stem}.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)
    meta_path = root / asset_id / f"{asset_id}_metadata.json"
    payload = dict(metadata)
    payload.setdefault("asset_id", asset_id)
    payload.setdefault("paper_slot", "TBD")
    payload.setdefault("status", "EXPLORATORY")
    payload["files"] = {"png": str(png), "pdf": str(pdf)}
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    readme = directory / "README.md"
    lines = [f"# {asset_id}", ""]
    for entry in readme_entries:
        lines += [f"### {entry['filename']}", "", entry["body"].strip(), "",
                  f"**关注点**：{entry['focus']}", ""]
    readme.write_text("\n".join(lines))
    return {"png": str(png), "pdf": str(pdf), "metadata": str(meta_path), "readme": str(readme)}
