"""Frozen native local-circuit producer; only three display parameters exposed."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon
E_COL = '#d62728'
I_COL = '#1f77b4'
M_COL = '#7A5195'

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


def _draw_integrated_mechanism(ax, *, central_offset=0.04, z_linewidth=1.60, z_dashes=(2.0, 1.7)) -> None:
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
        np.asarray([central_i_x - central_offset, e_y - 0.25]),
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

    central_i_edge_x = central_i_x + central_offset
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
            z_xy, 0.21, fc="white", ec=I_COL, lw=z_linewidth,
            ls=(0, z_dashes), zorder=8,
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
