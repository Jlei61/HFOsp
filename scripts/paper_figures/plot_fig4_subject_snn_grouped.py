"""Render the formal grouped Figure 4 draft for the E1146 subject-SNN example.

The PPT layout is treated as three reader-facing panel groups, not a collection
of individually exported micro-panels:

  A | integrated connection/MZ mechanism + patient-specific placement
  B | opposite-direction event maps + one continuous electrode readout
  C | clustered readout + aligned rank summaries + model-data similarity

Plotting only.  All data-bearing panels consume the accepted E1146
``figdata/readout`` artifacts; this script never reruns the SNN.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig_subject_snn import (  # noqa: E402
    TA_COLOR,
    TB_COLOR,
    _registered_axis_display,
    compose as compose_readout,
)
from scripts.paper_figures.plot_fig_subject_snn_kmeans2 import compose as compose_kmeans  # noqa: E402
from scripts.paper_figures.plot_fig_subject_snn_mechanism import (  # noqa: E402
    AXIS_COL,
    E_COL,
    FWD_SHADE,
    I_COL,
    _load_figdata,
    _plot_mechanism,
    _reconstruct_posI,
)


DEFAULT_TAG = "epilepsiae_1146_gradient_shared_corefrozen_cr1p5_s5_20260722"
DEFAULT_FIG_NAME = "fig4_subject_snn_e1146"
RUN_ROOT = ROOT / "results/topic4_sef_hfo/field_swap_subject_snn"
def _paired_seed(path: Path) -> int:
    return int(path.stem.rsplit("_s", 1)[1].split("_", 1)[0])


DEFAULT_VALIDATION_TAGS = tuple(
    path.stem.removeprefix("readout_")
    for path in sorted(
        RUN_ROOT.glob("readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json"),
        key=_paired_seed,
    )
)
PARTICIPATION_ELIGIBILITY = 0.15
MIN_EVENTS_PER_DIRECTION = 100


def _clean_direction_counts(tags: tuple[str, ...]) -> tuple[int, int]:
    n_forward = 0
    n_reverse = 0
    for input_tag in tags:
        readout = json.loads((RUN_ROOT / f"readout_{input_tag}.json").read_text(encoding="utf-8"))
        k_dir = int(readout.get("k_dir", 2))
        selected = [
            event for event in readout["events"]
            if event.get("sign") is not None and int(event.get("n_part", 0)) >= 2 * k_dir
        ]
        n_forward += sum(float(event["sign"]) > 0.0 for event in selected)
        n_reverse += sum(float(event["sign"]) < 0.0 for event in selected)
    return int(n_forward), int(n_reverse)


def _working_point_row(label: str, tags: tuple[str, ...], panel_meta: dict) -> dict:
    events = []
    per_seed = []
    core_radii = []
    durations = []
    for input_tag in tags:
        readout = json.loads((RUN_ROOT / f"readout_{input_tag}.json").read_text(encoding="utf-8"))
        figdata = np.load(RUN_ROOT / f"figdata_{input_tag}.npz", allow_pickle=True)
        k_dir = int(readout.get("k_dir", 2))
        selected = [
            event for event in readout["events"]
            if event.get("sign") is not None and event.get("n_part", 0) >= 2 * k_dir
        ]
        n_forward = int(sum(float(event["sign"]) > 0 for event in selected))
        n_reverse = int(sum(float(event["sign"]) < 0 for event in selected))
        per_seed.append({
            "seed": int(readout["seed"]),
            "n_events": len(selected),
            "n_forward": n_forward,
            "n_reverse": n_reverse,
            "bidirectional": bool(n_forward > 0 and n_reverse > 0),
        })
        events.extend(selected)
        core_radii.append(float(figdata["core_r"]))
        times = np.asarray(figdata["times"], dtype=float)
        trace_duration = float(times[-1] + np.median(np.diff(times)))
        durations.append(float(readout.get("paired_simulation_duration_ms", trace_duration)))

    names = sorted({
        name for event in events
        for name, value in (event.get("ranks") or {}).items() if value is not None
    })
    fractions = {
        name: float(sum((event.get("ranks") or {}).get(name) is not None for event in events) / len(events))
        for name in names
    }
    eligible = [name for name in names if fractions[name] >= PARTICIPATION_ELIGIBILITY]
    low = [
        {"channel": name, "participation_frac": fractions[name]}
        for name in names if fractions[name] < PARTICIPATION_ELIGIBILITY
    ]
    return {
        "candidate": label,
        "input_tags": list(tags),
        "core_r": float(core_radii[0]),
        "n_seeds": len(tags),
        "duration_ms_per_seed": sorted(set(round(value, 6) for value in durations)),
        "total_simulation_duration_ms": float(sum(durations)),
        "n_events": len(events),
        "n_forward": int(sum(float(event["sign"]) > 0 for event in events)),
        "n_reverse": int(sum(float(event["sign"]) < 0 for event in events)),
        "bidirectional_seeds": int(sum(row["bidirectional"] for row in per_seed)),
        "n_channels_observed": len(names),
        "n_channels_eligible_at_15pct": len(eligible),
        "eligible_channels": eligible,
        "low_participation_channels": low,
        "direction_purity": panel_meta.get("kmeans", {}).get("direction_purity"),
        "matrix_valid": panel_meta.get("similarity_matrix_panel", {}).get("valid"),
        "per_seed": per_seed,
    }


def _write_working_point_audit(
    outdir: Path,
    validation_tags: tuple[str, ...],
    validation_meta: dict,
) -> tuple[Path, Path]:
    older = {
        "core_r=2.5 legacy sweep": (
            ("epilepsiae_1146_sweep_cr2.5",),
            ROOT / "results/paper-ready-figure/fig_subject_snn_epilepsiae_1146_CR2p5_SWEEP_CANDIDATE/figures/fig_subject_snn_epilepsiae_1146_CR2p5_SWEEP_CANDIDATE_kmeans2_metadata.json",
        ),
        "core_r=2.87 cohort-rule rerun": (
            ("epilepsiae_1146_cohort_cr2.87_s3_rerun20260706",),
            ROOT / "results/paper-ready-figure/fig_subject_snn_epilepsiae_1146_COHORT_CR2p87_RERUN20260706/figures/fig_subject_snn_epilepsiae_1146_COHORT_CR2p87_RERUN20260706_kmeans2_metadata.json",
        ),
        "core_r=6.0 coverage variant": (
            ("epilepsiae_1146_sweep_cr6.0",),
            ROOT / "results/paper-ready-figure/fig_subject_snn_epilepsiae_1146_COVERAGE_VARIANT/figures/fig_subject_snn_epilepsiae_1146_COVERAGE_VARIANT_kmeans2_metadata.json",
        ),
    }
    spontaneous_cr1p5 = (
        tuple(f"epilepsiae_1146_tsrc_cr1p5_highn_s{seed}_20260721" for seed in range(0, 3))
        + tuple(f"epilepsiae_1146_tsrc_cr1p5_s{seed}_rerun20260706" for seed in range(3, 10))
        + tuple(f"epilepsiae_1146_tsrc_cr1p5_highn_s{seed}_20260721" for seed in range(10, 26))
    )
    core2p5_t4 = (
        "epilepsiae_1146_sweep_cr2.5",
        *tuple(f"epilepsiae_1146_tsrc_cr2p5_t4s_s{seed}_20260721" for seed in range(4, 7)),
    )
    core2p5_t8 = tuple(
        f"epilepsiae_1146_tsrc_cr2p5_pilot_s{seed}_20260721" for seed in range(3, 7)
    )
    rows = [
        _working_point_row("core_r=1.5 paired-arm high-n", validation_tags, validation_meta),
        _working_point_row("core_r=1.5 spontaneous 8-s audit", spontaneous_cr1p5, {}),
        _working_point_row("core_r=2.5 spontaneous 4-s audit", core2p5_t4, {}),
        _working_point_row("core_r=2.5 spontaneous 8-s audit", core2p5_t8, {}),
    ]
    for label, (tags, meta_path) in older.items():
        rows.append(_working_point_row(label, tags, json.loads(meta_path.read_text(encoding="utf-8"))))
    rows[0]["selection"] = "retained formal working point"
    for row in rows[1:]:
        row["selection"] = "coverage tradeoff / not selected"
    audit = {
        "subject": "epilepsiae_1146",
        "participation_eligibility": PARTICIPATION_ELIGIBILITY,
        "candidates": rows,
        "verdict": (
            "Retain core_r=1.5 for the formal paired-arm validation. The 26-seed spontaneous "
            "core_r=1.5 audit is strongly reverse-biased (29/99); core_r=2.5 changes from 11/14 "
            "at 4 s to 4/28 at 8 s, so enlarging the core does not define a duration-robust balanced "
            "spontaneous working point. Exclude contacts participating in <15% of pooled clean events."
        ),
        "comparison_boundary": (
            "Panel B retains a two-core spontaneous example, whereas formal high-n Panel C uses "
            "paired source-only/sink-only arms on the same network seed. These are different tests."
        ),
    }
    json_path = outdir.parent / "fig4_working_point_audit.json"
    csv_path = outdir.parent / "fig4_working_point_audit.csv"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fields = [
        "candidate", "core_r", "n_seeds", "n_events", "n_forward", "n_reverse",
        "bidirectional_seeds", "n_channels_observed", "n_channels_eligible_at_15pct",
        "direction_purity", "matrix_valid", "selection",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    return json_path, csv_path
M_COL = "#7A5195"


def _draw_dotted_arrow(
    ax,
    start: tuple[float, float],
    tip: tuple[float, float],
    *,
    color: str = E_COL,
    head_length: float = 0.17,
    head_width: float = 0.14,
) -> None:
    """Draw a dotted shaft plus an explicit filled triangular arrowhead."""
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(tip, dtype=float)
    vec = p1 - p0
    norm = float(np.linalg.norm(vec))
    if norm <= head_length:
        return
    unit = vec / norm
    normal = np.asarray([-unit[1], unit[0]])
    base = p1 - head_length * unit
    ax.plot(
        [p0[0], base[0]], [p0[1], base[1]],
        color=color, lw=1.40, ls=(0, (1.2, 2.0)),
        dash_capstyle="round", zorder=7,
    )
    triangle = np.vstack(
        [p1, base + 0.5 * head_width * normal, base - 0.5 * head_width * normal]
    )
    ax.add_patch(
        Polygon(triangle, closed=True, fc=color, ec=color, lw=0.8, zorder=10)
    )


def _draw_integrated_mechanism(ax) -> None:
    """Combine the spatial connection rule and MZ feedback in one schematic."""
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 8.0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("white")

    e_y = 4.48
    i_y = 2.20

    # A genuinely two-dimensional footprint replaces the literature-like 1-D
    # bell curve.  Nested 2:1 contours state the anisotropic E->E rule visually;
    # the compact circular footprint below states local isotropic I->E.
    for width, height, alpha, lw in (
        (6.00, 3.00, 0.08, 1.25),
        (4.45, 2.22, 0.05, 1.45),
        (2.90, 1.45, 0.04, 1.65),
    ):
        ax.add_patch(
            Ellipse(
                (5.0, e_y), width=width, height=height,
                fc=E_COL, ec=E_COL, lw=lw, alpha=alpha, zorder=0,
            )
        )
    for radius, alpha, lw in ((0.82, 0.08, 1.25), (0.55, 0.05, 1.45)):
        ax.add_patch(
            Circle((5.0, i_y), radius, fc=I_COL, ec=I_COL,
                   lw=lw, alpha=alpha, zorder=0)
        )
    ax.text(0.88, 5.94, "E→E", color=E_COL, fontsize=12.0,
            fontweight="bold", ha="left")
    ax.text(0.88, 1.08, "I→E", color=I_COL, fontsize=12.0,
            fontweight="bold", ha="left")

    # Population rows.  Every displayed position contains an E cell and an I
    # cell; only the central E cell is filled to identify the reference unit.
    # The central E and I remain vertically aligned.  Reciprocal paths are
    # separated only by a small lateral line offset, as in the reference.
    cell_x = np.linspace(1.08, 8.92, 11)
    center_idx = len(cell_x) // 2
    i_cell_x = cell_x.copy()
    for idx, cx in enumerate(cell_x):
        tri = np.array([[cx - 0.23, e_y - 0.25], [cx + 0.23, e_y - 0.25], [cx, e_y + 0.25]])
        is_center = idx == center_idx
        ax.add_patch(
            Polygon(
                tri, closed=True,
                fc=E_COL if is_center else "white",
                ec=E_COL,
                lw=2.0 if is_center else 1.65,
                zorder=7 if is_center else 5,
            )
        )
        ax.add_patch(
            Circle(
                (i_cell_x[idx], i_y), 0.22,
                fc="white", ec=I_COL, lw=1.65,
                zorder=5,
            )
        )

    # Recurrent E arrows extend preferentially along the horizontal major axis.
    for dx, rad in ((-1.57, 0.24), (1.57, -0.24)):
        ax.annotate(
            "",
            xy=(5.0 + dx, e_y + 0.04),
            xytext=(5.0, e_y + 0.22),
            arrowprops=dict(
                arrowstyle="-|>", color=E_COL, lw=1.7,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=8,
        )
    # A single, subdued return arc at the central E-cell apex.  The endpoints
    # sit immediately to either side of the apex so the connection reads as a
    # short arch rather than a closed oval.
    apex = np.asarray([5.0, e_y + 0.25])
    loop_start = apex + np.asarray([-0.045, 0.0])
    loop_control = apex + np.asarray([0.0, 0.21])
    loop_end = apex + np.asarray([0.045, 0.0])
    loop_t = np.linspace(0.0, 1.0, 121)[:, None]
    loop_xy = (
        (1.0 - loop_t) ** 2 * loop_start
        + 2.0 * (1.0 - loop_t) * loop_t * loop_control
        + loop_t ** 2 * loop_end
    )
    loop_x, loop_y = loop_xy[:, 0], loop_xy[:, 1]
    ax.plot(loop_x, loop_y, color=E_COL, lw=1.55, zorder=9)
    loop_tip = np.asarray([loop_x[-1], loop_y[-1]])
    loop_tangent = loop_end - loop_control
    loop_tangent /= float(np.linalg.norm(loop_tangent))
    loop_normal = np.asarray([-loop_tangent[1], loop_tangent[0]])
    loop_base = loop_tip - 0.10 * loop_tangent
    ax.add_patch(
        Polygon(
            np.vstack([
                loop_tip,
                loop_base + 0.043 * loop_normal,
                loop_base - 0.043 * loop_normal,
            ]),
            closed=True, fc=E_COL, ec=E_COL, lw=0.8, zorder=10,
        )
    )

    # The central E recruits multiple local I cells.  The middle E->I edge is
    # straight and slightly left of the straight I->E return; the two lateral
    # E->I branches fan outward to make the one-to-many recruitment explicit.
    central_i_x = float(i_cell_x[center_idx])
    side_i_x = (float(i_cell_x[center_idx - 2]), float(i_cell_x[center_idx + 2]))
    recruit_starts = (
        np.asarray([4.80, e_y - 0.22]),
        np.asarray([central_i_x - 0.04, e_y - 0.25]),
        np.asarray([5.20, e_y - 0.22]),
    )
    recruit_centers = (
        np.asarray([side_i_x[0], i_y]),
        np.asarray([central_i_x, i_y]),
        np.asarray([side_i_x[1], i_y]),
    )
    for arrow_idx, (start, target_center) in enumerate(zip(recruit_starts, recruit_centers)):
        if arrow_idx == 1:
            # Force the central E->I shaft to be exactly vertical.
            tip = np.asarray([start[0], i_y + 0.235])
        else:
            direction = target_center - start
            direction /= float(np.linalg.norm(direction))
            tip = target_center - 0.235 * direction
        _draw_dotted_arrow(ax, tuple(start), tuple(tip))

    # Nearby I projections communicate a local inhibitory population.  Each
    # side interneuron inhibits the nearest E cell directly above it.
    # All anatomical I symbols and I->E edges are solid blue; T-bars, not
    # arrowheads, encode inhibitory synaptic terminals.
    for sx in side_i_x:
        ax.plot([sx, sx], [i_y + 0.23, e_y - 0.30], color=I_COL, lw=1.45,
                zorder=4)
        ax.plot([sx - 0.17, sx + 0.17], [e_y - 0.30, e_y - 0.30],
                color=I_COL, lw=1.75, zorder=8)

    central_i_edge_x = central_i_x + 0.04
    ax.plot(
        [central_i_edge_x, central_i_edge_x],
        [i_y + 0.20, e_y - 0.30],
        color=I_COL, lw=1.75, zorder=6,
    )
    ax.plot(
        [central_i_edge_x - 0.11, central_i_edge_x + 0.11],
        [e_y - 0.30, e_y - 0.30],
        color=I_COL, lw=2.40, solid_capstyle="butt", zorder=8,
    )

    # z reads the inhibitory drive received by this E cell and scales that same
    # I->E current.  A single short dashed branch points from the solid central
    # inhibitory edge into z; the solid edge itself remains uninterrupted.
    z_xy = (5.52, 2.82)
    ax.add_patch(
        Circle(
            z_xy, 0.21, fc="white", ec=I_COL, lw=1.60,
            ls=(0, (2.0, 1.7)), zorder=8,
        )
    )
    ax.text(*z_xy, "z↓", color=I_COL, fontsize=9.6,
            fontweight="bold", ha="center", va="center", zorder=9)
    _draw_dotted_arrow(
        ax,
        (central_i_edge_x, z_xy[1]),
        (z_xy[0] - 0.22, z_xy[1]),
        color=I_COL,
        head_length=0.085,
        head_width=0.075,
    )
    _draw_dotted_arrow(
        ax,
        (5.11, e_y + 0.02),
        (z_xy[0], z_xy[1] + 0.22),
        color=E_COL,
        head_length=0.12,
        head_width=0.10,
    )

    # Make the adaptation sequence explicit and spatially separated:
    # central E -> E spike train -> accumulating/decaying m trace -> brake.
    ax.annotate(
        "",
        xy=(6.05, 6.04), xytext=(5.18, e_y + 0.25),
        arrowprops=dict(arrowstyle="-|>", color=E_COL, lw=1.55,
                        connectionstyle="arc3,rad=-0.18"),
        zorder=8,
    )
    spike_x = np.array([6.05, 6.28, 6.51, 6.74])
    spike_h = np.array([0.42, 0.72, 0.54, 0.82])
    for sx, sh in zip(spike_x, spike_h):
        ax.plot([sx, sx], [5.96, 5.96 + sh], color=E_COL, lw=1.65, zorder=8)
    ax.text(6.40, 6.92, "E spikes", color=E_COL, fontsize=9.8,
            fontweight="bold", ha="center")
    ax.annotate(
        "",
        xy=(7.28, 6.24), xytext=(6.86, 6.24),
        arrowprops=dict(arrowstyle="-|>", color=E_COL, lw=1.55),
        zorder=8,
    )

    t = np.linspace(0.0, 1.0, 320)
    m_trace = np.zeros_like(t)
    for spike_t in (0.08, 0.27, 0.46, 0.65):
        mask = t >= spike_t
        m_trace[mask] += np.exp(-(t[mask] - spike_t) / 0.30)
    m_trace /= float(np.max(m_trace))
    mx = 7.32 + 1.48 * t
    my = 5.86 + 0.82 * m_trace
    ax.plot([7.28, 8.88], [5.86, 5.86], color="0.72", lw=0.85, zorder=6)
    ax.plot(mx, my, color=M_COL, lw=2.15, zorder=9)
    ax.text(8.95, 6.11, "m", color=M_COL, fontsize=13.0,
            fontweight="bold", ha="left", va="center")

    ax.annotate(
        "",
        xy=(5.28, e_y + 0.18), xytext=(8.35, 5.88),
        arrowprops=dict(arrowstyle="-[", color=M_COL, lw=1.8,
                        mutation_scale=11.0, connectionstyle="arc3,rad=0.26"),
        zorder=8,
    )


def _render_panel_a(tag: str, fig_name: str, output_stem: str, dpi: int) -> tuple[Path, dict]:
    fd, source_path = _load_figdata(tag)
    pos_i, pos_i_meta = _reconstruct_posI(fd, tag)
    plot_seed = int((pos_i_meta.get("seed") or 0) + 101)
    display = _registered_axis_display(fd)

    fig = plt.figure(figsize=(13.2, 4.80), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, width_ratios=[0.95, 1.16], left=0.042, right=0.985,
        bottom=0.10, top=0.93, wspace=0.08,
    )
    ax_h = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])
    _draw_integrated_mechanism(ax_h)
    setup_meta = _plot_mechanism(
        fd,
        ax_s,
        clean=True,
        posI=pos_i,
        plot_seed=plot_seed,
        display=display,
        homogeneous_cores=True,
        semantic_core_colors=True,
        show_basic_labels=True,
        show_title=False,
    )
    setup_meta["template_core_style"] = {
        "template_A": TA_COLOR,
        "template_B": TB_COLOR,
        "display_labels": {"template_A_core": "Core 1", "template_B_core": "Core 2"},
        "contact_edge": "Core 1 red and Core 2 blue replace the default black edge; no double ring",
        "basic_labels": [
            "E/I neuron legend", "Core 1", "Core 2", "anisotropic E-to-E",
        ],
    }
    # Match the restrained opacity and stroke language of the integrated
    # mechanism while retaining the exact registered montage and core marks.
    for collection in ax_s.collections:
        if collection.get_zorder() in (1, 2, 5, 6):
            collection.set_alpha(0.25)
            collection.set_sizes(collection.get_sizes() * 0.78)
    fig.text(0.012, 0.94, "A", fontsize=22, fontweight="bold")

    outdir = ROOT / "results" / "paper-ready-figure" / fig_name / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{output_stem}.png"
    pdf = outdir / f"{output_stem}.pdf"
    svg = outdir / f"{output_stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    meta = {
        "figure": output_stem,
        "panel": "A",
        "source_figdata": str(source_path.relative_to(ROOT)),
        "plotting_only": True,
        "hypothesis_contract": {
            "network": "current-based E/I LIF SNN",
            "anisotropic_connection": "E->E only; elliptical-exponential; AR=2",
            "connection_widths_mm": {
                "l_EE_geometric_mean": 0.380,
                "l_EE_long_axis_AR2": float(0.380 * np.sqrt(2.0)),
                "l_EI_l_IE_l_II": 0.250,
            },
            "other_connection_kernels": "local/isotropic; not drawn as broader than E->E",
            "microcircuit_schematic": (
                "aligned E and I population rows with a filled central E reference cell; "
                "three independently rooted dotted red shafts with explicit filled triangular "
                "heads show one E recruiting multiple I cells without a Y junction; the central "
                "E-to-I shaft is exactly vertical; both ends of the compact curved E-to-E "
                "return meet the central E-cell apex region; the arc is a population-recurrence "
                "shorthand rather than a literal autapse"
            ),
            "cores": "twoend_equal low-threshold E cores at the two template-source regions",
            "event_trigger": "background OU/Poisson noise; no external kick",
            "artifact_slow_variables": "off (slow=None)",
            "slow_variable_schematic": {
                "status": "conceptual MZ extension; not active in the source figdata",
                "scope": "per-neuron E-cell variables only",
                "z_i": "conceptual local slow state receiving separate local excitatory and inhibitory drives and modulating inhibitory efficacy",
                "z_visual_encoding": "separate red and blue dashed inputs terminate at the z boundary; neither line crosses the z circle",
                "implementation_alignment": "dual E/I drive is an intended conceptual extension; the source figdata has slow=None and does not instantiate this update",
                "m_i": "spike-triggered adaptation; each E spike increments m_i; subtractive current is eta_m * m_i",
                "quiet_recovery": "z_i -> 1 with tau_z; m_i -> 0 with tau_adp",
                "membrane_current": "I_net^E = I_EE - z_i I_EI - eta_m m_i",
            },
        },
        "registered_axis_display": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in display.items()
        },
        "patient_setup": setup_meta,
        "posI": pos_i_meta,
        "outputs": [str(p.relative_to(ROOT)) for p in (png, pdf, svg)],
        "claim_boundary": (
            "connection and MZ slow-state hypotheses plus subject-specific placement; "
            "the MZ schematic is not active in the source figdata and is not a patient mechanism proof"
        ),
    }
    (outdir / f"{output_stem}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return outdir, meta


def _write_readme(outdir: Path) -> None:
    (outdir / "README.md").write_text(
        """# Figure 4 — E1146 subject-specific SNN

### fig4_panel_a_model_setup.png / .pdf / .svg

左侧把连接核、E/I 微回路和慢变量合成一个机制示意：中央红色 E→I 虚线被强制为完全竖直，三条 E→I 从不同位置独立发出。中央 E 上方的单弧回旋线两端紧贴红色三角顶点，是局部 E→E population recurrence 的示意，不表示代码中被排除的单神经元 autapse。`z↓` 分别接收红色兴奋性和蓝色抑制性虚线输入，两条线均终止在 z 边界而不穿过圆；该双输入属于拟议的 MZ 概念扩展，当前 source artifact 仍为 `slow=None`。右侧使用 `Core 1/Core 2` 中性名称，Core 1 为红色、Core 2 为蓝色单层边框；右上角以不透明白底小图例显示红色 E 三角和蓝色 I 圆，并保留 `anisotropic E→E` 标签。Panel A/B 使用同一冻结 gradient shared-plane 几何；Panel C 仍是旧 per-template 几何的高 n 验证产物。

**关注点**：看中央红蓝竖线是否平行、E→E 回旋是否独立可读、z 是否清楚接收红蓝两类局部输入；再看右侧 Core 1/2 与 E/I 图例是否无歧义。

### fig4_panel_b_bidirectional_readout.png / .pdf

这一整行消费同一个 spontaneous twoend seed5 artifact：冻结 gradient shared-plane 上的 model forward/reverse 代表事件，以及同一患者 montage 上的连续虚拟 SEEG readout；不再重复 Panel A 已展示的机制/底物图。右侧 1200 ms signed 30–80 Hz 窗口同时包含一个完整 model forward 橙色事件和一个完整 model reverse 蓝色事件；不拼接轨迹，不加入 runaway/发作期标记。两个空间图采用紧凑组内间距，`relative firing onset` 色条与空间坐标框等高，并与右侧 readout 保留最小必要间隔；readout legend 位于波形轴上方。

**关注点**：看同一底物和同一 montage 是否能反复读出相反的传播次序，同时保留 seed 与 k_dir=2 的边界。

### fig4_panel_c_model_validation.png / .pdf

这一行改为与 Panel B 对齐的三块布局：model forward/reverse clustered heatmap、mean-rank profile、model–data correlation matrix；删除 Rank dist. panel。统计池使用 21 个 paired network seeds；每个 seed 在同一 network realization 上分别运行 4 s source-only 与 4 s sink-only arm，总仿真时长 168 s。固定 `core_r=1.5`、`core_mean=17.5` 和 `k_dir=2`，共得到 222 个 clean directional events（model forward/reverse=103/119）。相关矩阵的行写 model forward/reverse，列写 data forward/reverse。只保留参与率至少 15% 的触点，SCL9（7.7%）被排除，最终显示 ICL1–ICL11。该高 n 池仍来自旧 per-template 几何；在 shared-plane 下完成对应重跑前，不能写成与 Panel A/B 完全同几何的验证。

**关注点**：同时看事件级聚类、paired-seed LOSO 和模型—真实模板矩阵；不能把 222 个事件误写成 222 次独立仿真，也不能把控制 arm 的方向平衡写成双核同网自发平衡。
""",
        encoding="utf-8",
    )


def compose(
    tag: str,
    fig_name: str,
    dpi: int = 240,
    validation_tags: tuple[str, ...] = DEFAULT_VALIDATION_TAGS,
) -> Path:
    os.chdir(ROOT)
    n_forward, n_reverse = _clean_direction_counts(validation_tags)
    if min(n_forward, n_reverse) < MIN_EVENTS_PER_DIRECTION:
        raise RuntimeError(
            "formal Panel C requires at least "
            f"{MIN_EVENTS_PER_DIRECTION} clean events per direction; "
            f"current forward/reverse={n_forward}/{n_reverse}"
        )
    outdir, setup_meta = _render_panel_a(tag, fig_name, "fig4_panel_a_model_setup", dpi)
    _, readout_meta = compose_readout(
        tag,
        None,
        None,
        fig_name,
        "E1146",
        5000.0,
        output_stem="fig4_panel_b_bidirectional_readout",
        panel_letter="B",
        formal_layout=True,
    )
    compose_kmeans(
        tag,
        fig_name,
        3,
        "narrow",
        preview_style=False,
        display_min_channel_frac=0.15,
        output_stem="fig4_panel_c_model_validation",
        panel_letter="C",
        formal_layout=True,
        tags=list(validation_tags),
    )
    validation_meta = json.loads(
        (outdir / "fig4_panel_c_model_validation_metadata.json").read_text(encoding="utf-8")
    )
    with np.load(RUN_ROOT / f"figdata_{tag}.npz", allow_pickle=True) as panel_b_figdata:
        panel_b_reg = panel_b_figdata["reg"].item()
    audit_json, audit_csv = _write_working_point_audit(outdir, validation_tags, validation_meta)

    bundle = {
        "figure": "Figure 4 grouped draft",
        "subject": "epilepsiae_1146",
        "input_tag": tag,
        "plotting_only": True,
        "panels": {
            "A": setup_meta["figure"],
            "B": readout_meta["figure"],
            "C": validation_meta["figure"],
        },
        "shared_contract": {
            "same_subject": True,
            "same_subject_and_fixed_model_parameters_for_B_C": True,
            "same_coordinate_geometry_for_B_C": False,
            "panel_b_readout": "one continuous seed5 window containing both TA and TB events",
            "panel_c_readout": "fixed-parameter independent 8-s runs; >=100 clean events per direction",
            "same_snn_readout_artifact_for_B_C": False,
            "simulation_rerun": True,
            "simulation_check": {
                "artifact": setup_meta["source_figdata"],
                "stored_axis_unit": panel_b_reg["axis_unit"],
                "stored_theta_deg": setup_meta["patient_setup"]["theta_deg_native_sheet"],
                "verdict": "Panel A/B use the frozen gradient shared-plane; Panel C remains the prior per-template-plane high-n pool",
            },
            "coordinate_display": "frozen template-gradient shared-plane; sheet-centering translation only, no display rotation",
            "readout_display": "one continuous interictal signed 30-80 Hz window containing TA and TB events",
            "k_dir": 2,
            "validation_n_seeds": validation_meta["replication_statistics"]["n_seeds"],
            "validation_seeds": validation_meta["replication_statistics"]["seeds"],
            "validation_independent_unit": validation_meta["replication_statistics"]["independent_unit"],
            "validation_total_simulation_duration_ms": validation_meta["replication_statistics"]["total_simulation_duration_ms"],
            "clean_events": validation_meta["n_events"],
            "forward_events": validation_meta["n_forward"],
            "reverse_events": validation_meta["n_reverse"],
            "minimum_events_per_direction": MIN_EVENTS_PER_DIRECTION,
            "channel_participation_eligibility": PARTICIPATION_ELIGIBILITY,
        },
        "working_point_audit": [
            str(audit_json.relative_to(ROOT)), str(audit_csv.relative_to(ROOT)),
        ],
        "claim_boundary": (
            "single-subject model/readout feasibility and template-consistency example; "
            "not cohort evidence or proof of the biological mechanism"
        ),
    }
    (outdir / "fig4_grouped_metadata.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    _write_readme(outdir)
    print(f"wrote grouped Figure 4 draft to {outdir}")
    return outdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--fig-name", default=DEFAULT_FIG_NAME)
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument(
        "--validation-tags",
        default=",".join(DEFAULT_VALIDATION_TAGS),
        help="Comma-separated fixed-parameter seed tags used by Panel C.",
    )
    args = ap.parse_args()
    validation_tags = tuple(x.strip() for x in args.validation_tags.split(",") if x.strip())
    compose(args.tag, args.fig_name, args.dpi, validation_tags)


if __name__ == "__main__":
    main()
