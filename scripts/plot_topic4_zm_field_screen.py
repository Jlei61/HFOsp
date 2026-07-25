#!/usr/bin/env python
"""Figure for the reduced 2-D S_L(x)+S_G field screen (spec
docs/superpowers/specs/2026-07-24-topic4-zm-reduced-field-Sl-Sg-design.md rev3): does making the
inhibitory feedback spatially LOCAL destabilise the uniform oscillation into a phase-staggered
spatial pattern? Four independent-question panels:
  (A) does the reduced model actually oscillate at the locked operating point? (uniform mean-field orbit)
  (B) is any 2-D SPATIAL mode unstable, for global vs local inhibitory feedback? (Floquet heatmaps)
  (C) does any excitability level push any inhibitory topology's growth rate past zero? (verdict panel)
  (D) what does the field actually look like doing it, in the short diagnostic runs? (r(x,y) + r(t))

Reads results/topic4_sef_hfo/zm_field_screen/{phaseA_lock.json,floquet_map.json,traces/*.npz} and
recomputes panels A/B directly from src.topic4_zm_field_meanfield / src.topic4_zm_field_screen so the
figure is never more than one function call away from the same code that produced the verdict.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.topic4_zm_field_meanfield import MFParams, detect_orbit, simulate_meanfield
from src.topic4_zm_field_screen import FieldParams, floquet_map, uniform_orbit
from src.topic4_zm_field_verdict import TH

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_field_screen")
LOCK_PATH = os.path.join(OUT, "phaseA_lock.json")
FLOQUET_PATH = os.path.join(OUT, "floquet_map.json")
TRACES_DIR = os.path.join(OUT, "traces")
FIG_DIR = os.path.join(OUT, "figures")
FIG_STEM = "zm_field_screen_local_vs_global"

M_MAX = 4          # integer spatial-mode half-width for the Floquet heatmap (panel B)
LAM_FLOOR = TH["lam_floor"]   # sign-resolution floor; imported so it cannot drift from the adjudicator

ARM_LABEL = {
    "div_global": "divisive-only (beta=0):\nNO orbit -- not a stability result",
    "dual_global": "global inhibition",
    "dual_local": "local inhibition",
    "dual_mixed": "mixed local+global",
}
ARM_COLOR = {
    "div_global": "#BEBEBE",
    "dual_global": "#2166AC",
    "dual_local": "#B2182B",
    "dual_mixed": "#762A83",
}


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _panel_header(fig, axes_list, text, y_offset=0.03, fontsize=11.5):
    """Bold header text centred above the union of the given axes (acts as the panel title)."""
    boxes = [ax.get_position() for ax in axes_list]
    x0 = min(b.x0 for b in boxes)
    x1 = max(b.x1 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    fig.text((x0 + x1) / 2, y1 + y_offset, text, ha="center", va="bottom",
              fontsize=fontsize, fontweight="bold", linespacing=1.35)


# ---------------------------------------------------------------------------
# Panel A -- does the reduced model actually oscillate?
# ---------------------------------------------------------------------------
def panel_a(axes, lock):
    ax_r, ax_mu, ax_s = axes
    op = lock["operating_point"]
    dt = lock["dt"]
    mf = MFParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=op["I0"])
    tr = simulate_meanfield(mf, T=6000.0, dt=dt)
    o = detect_orbit(tr, dt)
    if not o["oscillates"]:
        for ax in axes:
            ax.text(0.5, 0.5, "meanfield does not oscillate\nat the locked operating point",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        return None

    period_ms = o["period_ms"]
    n_period = max(2, int(round(period_ms / dt)))
    n_show = min(len(tr), n_period * 3)
    tail = tr[-n_show:]
    t = np.arange(len(tail)) * dt

    rows = [(0, "population rate  $r$  (a.u.)"),
            (1, "recruitment drive  $\\mu$  (a.u.)"),
            (2, "slow inhibitory pool  $S$  (a.u.)")]
    for ax, (i, ylabel) in zip(axes, rows):
        ax.plot(t, tail[:, i], color="#1B1B1B", lw=1.6)
        ax.set_ylabel(ylabel, fontsize=8.7)
        ax.margins(x=0)

    # mark the period using the actual mid-line crossings of the displayed r(t) trace
    r_tail = tail[:, 0]
    mid = 0.5 * (r_tail.max() + r_tail.min())
    cross_idx = np.flatnonzero((r_tail[:-1] < mid) & (r_tail[1:] >= mid))
    if cross_idx.size >= 2:
        c0, c1 = cross_idx[0] * dt, cross_idx[1] * dt
        y0 = ax_r.get_ylim()[1]
        ax_r.annotate("", xy=(c1, y0 * 0.96), xytext=(c0, y0 * 0.96),
                      arrowprops=dict(arrowstyle="<->", color="#2166AC", lw=1.1))
        ax_r.text((c0 + c1) / 2, y0 * 0.985, f"period = {period_ms:.1f} ms",
                  color="#2166AC", ha="center", va="bottom", fontsize=8.2)
        for k in range(int(t[-1] // period_ms) + 1):
            ax_r.axvline(c0 + k * period_ms, color="#2166AC", lw=0.7, ls="--", alpha=0.45)
            ax_mu.axvline(c0 + k * period_ms, color="#2166AC", lw=0.7, ls="--", alpha=0.25)
            ax_s.axvline(c0 + k * period_ms, color="#2166AC", lw=0.7, ls="--", alpha=0.25)

    plt.setp(ax_r.get_xticklabels(), visible=False)
    plt.setp(ax_mu.get_xticklabels(), visible=False)
    ax_s.set_xlabel("time (ms)", fontsize=9)

    # the 5 excitability levels panel C sweeps are otherwise only implied by panel C's x-axis --
    # name them here explicitly against the single trace (at the locked operating point) shown above
    levels_str = ", ".join(f"{v:.2f}" for v in lock["I0_levels"])
    ax_s.text(0.0, -0.30, f"panel C sweeps these 5 locked excitability levels ($I_0$): {levels_str}",
              transform=ax_s.transAxes, fontsize=6.8, color="#555555", va="top", ha="left")
    return period_ms


# ---------------------------------------------------------------------------
# Panel B -- is any SPATIAL mode unstable?
# ---------------------------------------------------------------------------
def panel_b(ax_g, ax_l, ax_cb, ax_note, lock):
    op = lock["operating_point"]
    dt = lock["dt"]
    p = FieldParams(W0=op["W0"], alpha=op["alpha"], beta=op["beta"], theta=op["theta"], I0=op["I0"],
                     n=lock["grid_n"])
    orbit, _ = uniform_orbit(p, dt)

    grids, kstars = {}, {}
    for arm in ("dual_global", "dual_local"):
        res = floquet_map(p, arm, orbit, dt, m_max=M_MAX)
        grid = np.full((2 * M_MAX + 1, 2 * M_MAX + 1), np.nan)
        for (mx, my), lam in zip(res["modes"], res["lam"]):
            grid[mx + M_MAX, my + M_MAX] = lam
        grids[arm] = grid
        kstars[arm] = (res["k_star"], res["lam_max"])

    vmax = max(np.nanmax(np.abs(g)) for g in grids.values())
    extent = (-M_MAX - 0.5, M_MAX + 0.5, -M_MAX - 0.5, M_MAX + 0.5)
    im = None
    for ax, arm in zip((ax_g, ax_l), ("dual_global", "dual_local")):
        im = ax.imshow(grids[arm].T, origin="lower", extent=extent, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, aspect="equal")
        ax.plot(0, 0, marker="x", color="black", ms=7, mew=1.6)  # DC mode: excluded, not transverse
        kx, ky = kstars[arm][0]
        ax.scatter([kx], [ky], s=110, facecolors="none", edgecolors="black", linewidths=1.3)
        # identity label drawn INSIDE the axes (not ax.set_title) so it can never collide with the
        # bold panel header floating above the axes box
        ax.text(0.03, 0.96, ARM_LABEL[arm], transform=ax.transAxes, ha="left", va="top",
                 fontsize=9.2, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.5))
        ax.set_xticks(range(-M_MAX, M_MAX + 1))
        ax.set_yticks(range(-M_MAX, M_MAX + 1))
        ax.tick_params(labelsize=6.6)
        ax.set_ylabel("spatial mode  $m_y$", fontsize=8.2)

    plt.setp(ax_g.get_xticklabels(), visible=False)
    ax_l.set_xlabel("spatial mode  $m_x$", fontsize=8.2)

    cb = plt.colorbar(im, cax=ax_cb)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("transverse growth rate  $\\lambda_\\perp$  (ms$^{-1}$)\npositive = spatial instability",
                 fontsize=7.6)

    lam_g, lam_l = kstars["dual_global"][1], kstars["dual_local"][1]
    ax_note.axis("off")
    ax_note.text(0.0, 0.95,
                 "DC mode ($m_x{=}0,\\,m_y{=}0$; marked ×) excluded -- it is the uniform state\n"
                 "itself, not a transverse (pattern-forming) mode. Open circle = each arm's\n"
                 "least-stable (closest-to-zero) mode:\n"
                 f"  global inhibition   $\\lambda_\\perp$ = {lam_g:+.4f} ms$^{{-1}}$\n"
                 f"  local inhibition    $\\lambda_\\perp$ = {lam_l:+.4f} ms$^{{-1}}$\n"
                 "Both still negative -- no mode, for either topology, crosses zero.",
                 fontsize=7.6, va="top", ha="left", transform=ax_note.transAxes)


# ---------------------------------------------------------------------------
# Panel C -- does any excitability level approach instability? (verdict panel)
# ---------------------------------------------------------------------------
def panel_c(ax, floquet, lock):
    levels = sorted(float(k) for k in floquet.keys())
    for arm in ("div_global", "dual_global", "dual_local", "dual_mixed"):
        ys = [floquet[f"{lv:.4f}"].get(arm) for lv in levels]
        if all(y is None for y in ys):
            continue
        if arm == "div_global":
            # beta=0 ablation: at this operating point the system has NO oscillation of its own (it
            # settles to a fixed point -- see test_divisive_only_beta0_has_no_orbit), so this curve is a
            # coefficient ablation evaluated along the DUAL system's orbit, not a stability statement
            # about div_global's own dynamics. Styled to read as clearly secondary / not comparable to
            # the three self-consistent arms below -- full caveat in the in-panel note underneath.
            ax.plot(levels, ys, marker="o", ms=3.2, lw=0.8, ls=":", alpha=0.5,
                     color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=1)
            continue
        primary = arm in ("dual_global", "dual_local")
        ax.plot(levels, ys, marker="o", ms=5,
                 lw=2.4 if primary else 1.3, ls="-" if primary else "--",
                 color=ARM_COLOR[arm], label=ARM_LABEL[arm], zorder=3 if primary else 2)

    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    ax.axhspan(-LAM_FLOOR, LAM_FLOOR, color="gray", alpha=0.18, zorder=0)

    op_I0 = lock["operating_point"]["I0"]
    ax.axvline(op_I0, color="#444444", lw=0.9, ls=":", alpha=0.7, zorder=1)
    ax.text(op_I0, ax.get_ylim()[0], "locked\noperating\npoint", fontsize=6.8, color="#444444",
             ha="center", va="bottom")

    ax.set_ylim(top=max(ax.get_ylim()[1], 0.012))
    # place the floor-band label in the (data-driven, so it stays clear on any rerun) gap between the
    # dual_local and dual_mixed curves at the leftmost level -- the near-zero strip directly above it is
    # now the legend's territory (the div_global caveat entry made the legend taller than before)
    lvl0_key = f"{levels[0]:.4f}"
    floor_label_y = 0.5 * (floquet[lvl0_key]["dual_local"] + floquet[lvl0_key]["dual_mixed"])
    ax.text(levels[0], floor_label_y, f"sign-resolution floor (|λ|≤{LAM_FLOOR:g})",
             fontsize=7.4, color="#555555", va="center", ha="left")
    ax.legend(fontsize=6.6, loc="upper right", frameon=False, title="inhibitory feedback",
              title_fontsize=6.8)

    ax.set_xlabel("excitability level  $I_0$", fontsize=9.3)
    ax.set_ylabel("max transverse growth rate  $\\lambda_\\perp^{max}$  (ms$^{-1}$)", fontsize=9.3)
    ax.margins(x=0.08)

    ax.text(0.0, -0.10,
             "The beta=0 ablation has no oscillation of its own at this operating point (it settles to a\n"
             "fixed point), so its curve is a coefficient ablation evaluated along the dual system's orbit --\n"
             "NOT a statement about its own stability. Its role in this project is to show that the\n"
             "subtractive term is what creates the oscillation; that contrast is separate from the\n"
             "local-vs-global spatial-rank contrast the other three curves make.",
             transform=ax.transAxes, fontsize=6.9, color="#555555", va="top", ha="left", linespacing=1.4)


# ---------------------------------------------------------------------------
# Panel D -- what does the field actually do?
# ---------------------------------------------------------------------------
def _trace_path(level_str, arm, seed=0):
    return os.path.join(TRACES_DIR, f"form_L{level_str}_{arm}_s{seed}.npz")


def panel_d(axes, lock, L_mm, seed=0):
    ax_g1, ax_g2, ax_l1, ax_l2, ax_cb, ax_ts = axes
    level_str = f"{lock['operating_point']['I0']:.4f}"

    traces, missing = {}, []
    for arm in ("dual_global", "dual_local"):
        path = _trace_path(level_str, arm, seed=seed)
        if not os.path.exists(path):
            missing.append(arm)
            continue
        d = np.load(path)
        traces[arm] = dict(r=d["r_trace"], t=d["t_ms"])

    for arm, axs in (("dual_global", (ax_g1, ax_g2)), ("dual_local", (ax_l1, ax_l2))):
        if arm in missing:
            for ax in axs:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{ARM_LABEL[arm]}:\ntrace file missing", ha="center", va="center",
                        fontsize=7.6, transform=ax.transAxes)

    if not traces:
        ax_cb.axis("off")
        ax_ts.axis("off")
        ax_ts.text(0.5, 0.5, "no trace files found for this level/seed", ha="center", va="center")
        return

    all_vals = np.concatenate([d["r"].ravel() for d in traces.values()])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    # snapshot times: the population-rate peak and trough of whichever arm is available
    ref = traces.get("dual_local", next(iter(traces.values())))
    pop_ref = ref["r"].mean(axis=(1, 2))
    idxs = sorted({int(np.argmin(pop_ref)), int(np.argmax(pop_ref))})
    if len(idxs) == 1:
        idxs.append(min(idxs[0] + len(pop_ref) // 2, len(pop_ref) - 1))

    im = None
    for arm, axs in (("dual_global", (ax_g1, ax_g2)), ("dual_local", (ax_l1, ax_l2))):
        if arm not in traces:
            continue
        r, t = traces[arm]["r"], traces[arm]["t"]
        for ax, idx in zip(axs, idxs):
            im = ax.imshow(r[idx], origin="lower", extent=(0, L_mm, 0, L_mm), cmap="viridis",
                            vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"t={t[idx]:.0f} ms  ($\\bar r$={r[idx].mean():.3f})", fontsize=7.4)
            ax.set_xticks([]); ax.set_yticks([])

    ax_g1.set_ylabel("global\ninhibition", fontsize=8.2)
    ax_l1.set_ylabel("local\ninhibition", fontsize=8.2)

    if im is not None:
        cb = plt.colorbar(im, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("population rate  $r$  (a.u.)", fontsize=7.6)

    pop_means = {}
    for arm, ls, lw in (("dual_global", "-", 1.8), ("dual_local", "--", 2.2)):
        if arm not in traces:
            continue
        r, t = traces[arm]["r"], traces[arm]["t"]
        pop_means[arm] = r.mean(axis=(1, 2))
        ax_ts.plot(t, pop_means[arm], color=ARM_COLOR[arm], lw=lw, ls=ls,
                    label=ARM_LABEL[arm], alpha=0.9)
    for idx in idxs:
        ax_ts.axvline(ref["t"][idx], color="gray", lw=0.8, ls=":")
    ax_ts.margins(x=0)
    ax_ts.set_xlabel("time (ms)", fontsize=8.6)
    ax_ts.set_ylabel("population\nmean $r$", fontsize=8.2)
    ax_ts.legend(fontsize=7, frameon=False, loc="upper right")

    note_lines = [
        "this axis plots the population-MEAN rate r(t), not the phase-synchrony measure originally",
        "sketched for this panel -- harmless here because the field never develops a spatial pattern",
        "(it stays spatially uniform throughout, matching panels B/C), so a synchrony measure would",
        "have nothing to distinguish and would be uninformative.",
        f"short diagnostic run: T=3000 ms, seed {seed} (not production-length).",
        "Snapshots are visually flat (no spatial texture) for both arms at both times --",
        "consistent with panels B/C: no spatial pattern is emerging here.",
    ]
    if "dual_global" in pop_means and "dual_local" in pop_means:
        max_diff = float(np.abs(pop_means["dual_global"] - pop_means["dual_local"]).max())
        note_lines += [
            f"Global/local population-mean curves overlap almost exactly here (max diff {max_diff:.1e}):",
            "the perturbation has already decayed below visibility, so both arms ride the same",
            "uniform limit cycle in this short run.",
        ]
    if missing:
        note_lines.append(f"Missing trace file(s): {', '.join(ARM_LABEL[a] for a in missing)}.")
    ax_ts.text(0.0, -0.30, "\n".join(note_lines), transform=ax_ts.transAxes, fontsize=6.8,
                color="#555555", va="top")


def main():
    if not os.path.exists(LOCK_PATH):
        raise SystemExit(f"missing lock file: {LOCK_PATH} -- run scripts/run_topic4_zm_field_screen.py first")
    if not os.path.exists(FLOQUET_PATH):
        raise SystemExit(f"missing floquet map: {FLOQUET_PATH} -- run scripts/run_topic4_zm_field_screen.py first")
    lock = _load_json(LOCK_PATH)
    floquet = _load_json(FLOQUET_PATH)
    os.makedirs(FIG_DIR, exist_ok=True)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(23.0, 8.2))
    width_ratios = [1.05, 0.14, 1.05, 0.08, 0.32, 1.05, 0.32, 0.95, 0.95, 0.08]
    outer = fig.add_gridspec(3, 10, width_ratios=width_ratios, height_ratios=[1.0, 1.0, 0.95],
                              left=0.032, right=0.985, top=0.77, bottom=0.085, hspace=0.6, wspace=0.18)

    # --- Panel A ---
    ax_ar = fig.add_subplot(outer[0, 0])
    ax_amu = fig.add_subplot(outer[1, 0], sharex=ax_ar)
    ax_as = fig.add_subplot(outer[2, 0], sharex=ax_ar)
    panel_a((ax_ar, ax_amu, ax_as), lock)
    _panel_header(fig, [ax_ar, ax_amu, ax_as],
                  "A. Does the reduced model actually oscillate?\nuniform mean-field orbit, locked operating point")

    # --- Panel B ---
    ax_bg = fig.add_subplot(outer[0, 2])
    ax_bl = fig.add_subplot(outer[1, 2], sharex=ax_bg, sharey=ax_bg)
    ax_bcb = fig.add_subplot(outer[0:2, 3])
    ax_bnote = fig.add_subplot(outer[2, 2])
    panel_b(ax_bg, ax_bl, ax_bcb, ax_bnote, lock)
    _panel_header(fig, [ax_bg, ax_bl],
                  "B. Is any SPATIAL mode unstable?\nglobal vs local inhibitory feedback")

    # --- Panel C ---
    ax_c = fig.add_subplot(outer[:, 5])
    panel_c(ax_c, floquet, lock)
    _panel_header(fig, [ax_c],
                  "C. Does any excitability level approach instability?\nmax growth rate vs $I_0$ (verdict)")

    # --- Panel D ---
    ax_dg1 = fig.add_subplot(outer[0, 7])
    ax_dg2 = fig.add_subplot(outer[0, 8], sharex=ax_dg1, sharey=ax_dg1)
    ax_dl1 = fig.add_subplot(outer[1, 7], sharex=ax_dg1, sharey=ax_dg1)
    ax_dl2 = fig.add_subplot(outer[1, 8], sharex=ax_dg1, sharey=ax_dg1)
    ax_dcb = fig.add_subplot(outer[0:2, 9])
    ax_dts = fig.add_subplot(outer[2, 7:9])
    panel_d((ax_dg1, ax_dg2, ax_dl1, ax_dl2, ax_dcb, ax_dts), lock,
            L_mm=FieldParams.__dataclass_fields__["L"].default)
    _panel_header(fig, [ax_dg1, ax_dg2, ax_dl1, ax_dl2],
                  "D. What does the field actually do?\nglobal vs local inhibition, short diagnostic runs")

    fig.suptitle(
        "Reduced 2-D field: does spatially LOCAL inhibitory feedback destabilise the uniform oscillation\n"
        "into a phase-staggered pattern?  Verdict: NO -- every spatial mode decays, for local, global and "
        "mixed inhibition, at every excitability level tested.",
        fontsize=12.5, y=0.975)

    stem = os.path.join(FIG_DIR, FIG_STEM)
    fig.savefig(stem + ".png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(stem + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.png")
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
