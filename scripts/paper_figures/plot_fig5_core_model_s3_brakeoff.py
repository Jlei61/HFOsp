"""Paper-ready Stage-3 brake-off core model panel.

This replaces the old A/B/C stacked `core_model_s3_brakeoff.png` composition
with one compact AB panel:
  mechanism schematic | source-at-neg event | source-at-pos event | fused readout.

The script is plotting-only. It consumes the existing SNN readout artifacts and
does not re-run the simulation.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = OUT / "figures"
PAPER_FIG = ROOT / "results/paper-ready-figure/fig5_core_model_s3_brakeoff/figures"

FWD_SHADE = "#f4b266"
REV_SHADE = "#78a6d8"
SHAFT_A = "#e8743b"
SHAFT_B = "#1f9e9e"
AXIS_COL = "#a65f00"
OFF = 1.35
SHADE_PAD_MS = 22.0
SPATIAL_DOT_SIZE = 8.0
SPATIAL_ALPHA = 0.90


def _load_npz(tag: str):
    return np.load(OUT / "per_event" / f"rep_{tag}.npz", allow_pickle=True)


def _load_readout(tag: str) -> dict:
    with open(OUT / f"readout_{tag}.json", "r") as f:
        return json.load(f)


def _axis(theta_deg: float):
    th = np.deg2rad(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])
    p = np.array([-u[1], u[0]])
    return u, p


def _clean_events(events):
    return [
        e for e in events
        if e.get("returned")
        and e.get("sign") in (1.0, -1.0)
        and e.get("axis_err") is not None
        and e.get("axis_err") < 25
        and e.get("n_part", 0) >= 7
    ]


def _active_contacts(names, contacts, axis_unit, perp_unit, events):
    active = set()
    for e in events:
        active.update(n for n, v in (e.get("ranks") or {}).items() if v is not None)
    keep = [i for i, n in enumerate(names) if n in active]
    if not keep:
        keep = list(range(len(names)))
    pp = np.array([contacts[i] @ axis_unit for i in keep])
    qq = np.array([contacts[i] @ perp_unit for i in keep])
    order = np.lexsort((qq, pp))
    return [keep[i] for i in order]


def _focus_source_index(z):
    pos = np.asarray(z["posE"], float)
    onset = np.asarray(z["onset_core"], float)
    foci = np.asarray(z["foci"], float)
    radius = float(z["patch_r"])
    med = []
    for focus in foci:
        m = (np.linalg.norm(pos - focus, axis=1) <= radius) & np.isfinite(onset)
        med.append(np.nanmedian(onset[m]) if m.any() else np.inf)
    return int(np.nanargmin(med))


def _axis_range_patch(center, foci, axis_unit, perp_unit, l_ee, ar):
    l_par = float(l_ee) * np.sqrt(float(ar))
    l_perp = float(l_ee) / np.sqrt(float(ar))
    # Draw a visible 3-sigma E->E footprint corridor across the two pathology cores.
    half_w = max(0.80, 3.0 * l_perp)
    ext = max(1.20, 3.0 * l_par)
    proj = (np.asarray(foci) - center) @ axis_unit
    a = center + axis_unit * (float(proj.min()) - ext)
    b = center + axis_unit * (float(proj.max()) + ext)
    return np.vstack([a + half_w * perp_unit, b + half_w * perp_unit,
                      b - half_w * perp_unit, a - half_w * perp_unit])


def _draw_contacts(ax, contacts, names):
    contacts = np.asarray(contacts, float)
    for prefix, color, marker in (("A", SHAFT_A, "o"), ("B", SHAFT_B, "s")):
        idx = [i for i, n in enumerate(names) if str(n).startswith(prefix)]
        if not idx:
            continue
        c = contacts[idx]
        ax.plot(c[:, 0], c[:, 1], color=color, lw=1.0, alpha=0.60, zorder=5)
        ax.scatter(c[:, 0], c[:, 1], s=42, marker=marker, fc="white", ec=color, lw=1.0, zorder=6)
        for j in (idx[0], idx[-1]):
            ax.text(contacts[j, 0], contacts[j, 1], str(names[j]), fontsize=7,
                    color=color, fontweight="bold", ha="center", va="center", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def _style_spatial(ax, L):
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=7.6)
    ax.set_ylabel("y (mm)", fontsize=7.6)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("0.25")


def _plot_mechanism(ax, z, readout):
    pos = np.asarray(z["posE"], float)
    v = np.asarray(z["vth"], float)
    foci = np.asarray(z["foci"], float)
    contacts = np.asarray(z["contacts"], float)
    names = [str(x) for x in z["names"]]
    L = float(z["L"])
    theta = float(z["theta"])
    axis_unit, perp_unit = _axis(theta)
    center = np.array([L / 2.0, L / 2.0])
    cfg = readout.get("config", {})
    poly = _axis_range_patch(
        center, foci, axis_unit, perp_unit,
        cfg.get("l_EE", 0.38), cfg.get("AR", 2.0),
    )
    ax.scatter(
        pos[:, 0], pos[:, 1],
        c=np.clip(18.0 - v, 0.0, None),
        s=SPATIAL_DOT_SIZE, cmap="plasma", vmin=0.0, vmax=1.2,
        alpha=SPATIAL_ALPHA, linewidths=0, rasterized=True, zorder=2,
    )
    ax.add_patch(Polygon(poly, closed=True, fc=FWD_SHADE, ec=AXIS_COL,
                         lw=1.35, alpha=0.30, zorder=4, label="E->E long-axis range"))
    for i, f in enumerate(foci):
        ax.add_patch(plt.Circle(f, float(z["patch_r"]), fill=False, ec="crimson",
                                lw=1.25, ls="--", zorder=7))
        ax.text(f[0], f[1] + 1.0, "-" if i == 0 else "+", fontsize=9,
                color="crimson", fontweight="bold", ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    p0 = center - axis_unit * 8.3
    p1 = center + axis_unit * 8.3
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=AXIS_COL, lw=1.7),
                zorder=8)
    _draw_contacts(ax, contacts, names)
    ax.set_title("mechanism", fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, L)


def _plot_event(ax, z, title):
    pos = np.asarray(z["posE"], float)
    onset = np.asarray(z["onset_core"], float)
    foci = np.asarray(z["foci"], float)
    contacts = np.asarray(z["contacts"], float)
    names = [str(x) for x in z["names"]]
    L = float(z["L"])
    theta = float(z["theta"])
    axis_unit, _ = _axis(theta)
    fin = np.isfinite(onset)
    bg = np.zeros(len(pos), bool)
    bg[::4] = True
    ax.scatter(pos[bg & ~fin, 0], pos[bg & ~fin, 1], s=1.2, c="0.86",
               alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    rel = onset.copy()
    if fin.any():
        rel[fin] -= np.nanmin(rel[fin])
        vmax = max(1.0, float(np.nanpercentile(rel[fin], 98)))
        ax.scatter(pos[fin, 0], pos[fin, 1], c=rel[fin], s=SPATIAL_DOT_SIZE,
                   cmap="viridis", vmin=0.0, vmax=vmax, alpha=SPATIAL_ALPHA,
                   linewidths=0, rasterized=True, zorder=2)
    src_idx = _focus_source_index(z)
    for i, f in enumerate(foci):
        ax.add_patch(plt.Circle(f, float(z["patch_r"]), fill=False, ec="crimson",
                                lw=1.2, ls="--", zorder=5))
        if i == src_idx:
            ax.scatter([f[0]], [f[1]], marker="*", s=150, c="black",
                       ec="white", lw=0.8, zorder=7)
    p0 = np.array([L / 2.0, L / 2.0]) - axis_unit * 8.3
    p1 = np.array([L / 2.0, L / 2.0]) + axis_unit * 8.3
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.20", lw=1.2, alpha=0.75, zorder=4)
    _draw_contacts(ax, contacts, names)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    _style_spatial(ax, L)


def _plot_readout(ax, z, events, window_ms):
    names = [str(x) for x in z["names"]]
    contacts = np.asarray(z["contacts"], float)
    theta = float(z["theta"])
    axis_unit, perp_unit = _axis(theta)
    clean = _clean_events(events)
    combined = _active_contacts(names, contacts, axis_unit, perp_unit, clean)

    lfp = np.asarray(z["lfp"], float).T
    t = np.asarray(z["times"], float)
    win_hi = min(float(window_ms), float(t[-1]))
    sel = (t >= 0.0) & (t <= win_hi)
    ts = t[sel]
    sub = lfp[combined][:, sel]
    base = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(sub.max(axis=1, keepdims=True) - base, 1e-9)
    zt = (sub - base) / scale
    y = np.arange(len(combined)) * OFF

    for i, ci in enumerate(combined):
        color = SHAFT_B if names[ci].startswith("B") else SHAFT_A
        ax.plot(ts, zt[i] + y[i], color=color, lw=0.78, alpha=0.92, zorder=3)

    n_fwd = n_rev = 0
    for e in clean:
        if e["t_on"] < 0.0 or e["t_on"] > win_hi:
            continue
        sign = float(e["sign"])
        color = FWD_SHADE if sign > 0 else REV_SHADE
        n_fwd += int(sign > 0)
        n_rev += int(sign < 0)
        span0 = max(0.0, float(e["t_on"]) - SHADE_PAD_MS)
        span1 = min(win_hi, float(e["t_off"]) + SHADE_PAD_MS)
        ax.axvspan(span0, span1, color=color, alpha=0.28, lw=0, zorder=0)
        pts = []
        ranks = e.get("ranks") or {}
        for i, ci in enumerate(combined):
            if ranks.get(names[ci]) is None:
                continue
            m = (ts >= e["t_on"]) & (ts <= e["t_off"])
            if m.sum() < 2:
                continue
            pi = np.flatnonzero(m)[int(np.argmax(zt[i][m]))]
            pts.append((ts[pi], zt[i][pi] + y[i]))
            ax.plot(ts[pi], zt[i][pi] + y[i], "o", ms=2.3, mfc="black",
                    mec="white", mew=0.35, zorder=6)
        if len(pts) >= 2:
            px, py = zip(*sorted(pts))
            ax.plot(px, py, "-", color="black", lw=0.75, alpha=0.70, zorder=5)

    ax.set_xlim(0.0, win_hi)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in combined], fontsize=7.0)
    for tick, ci in zip(ax.get_yticklabels(), combined):
        tick.set_color(SHAFT_B if names[ci].startswith("B") else SHAFT_A)
    ax.tick_params(axis="y", length=2.5, labelsize=7.0, color="0.35")
    ax.tick_params(axis="x", length=3.0, labelsize=7.5, color="0.35")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("0.35")
        ax.spines[side].set_linewidth(0.8)
    ax.set_xlabel("time (ms)", fontsize=8.2)
    ax.legend(
        handles=[
            Patch(facecolor=FWD_SHADE, alpha=0.40, edgecolor="none", label="forward event"),
            Patch(facecolor=REV_SHADE, alpha=0.40, edgecolor="none", label="reverse event"),
            Line2D([0], [0], color="black", marker="o", lw=0.8, ms=3,
                   mfc="black", mec="white", label="peak order"),
        ],
        frameon=False, fontsize=7.4, loc="upper right", bbox_to_anchor=(1.0, 1.045),
        borderaxespad=0.0, ncol=3, handlelength=1.5, columnspacing=0.9,
    )
    return {"forward_events": n_fwd, "reverse_events": n_rev, "contacts": [names[i] for i in combined]}


def _write_readme():
    text = """# Fig5 core model s3 brake-off

### core_model_s3_brakeoff.png

这张图把 SNN 仿真图固定成一行：左侧是机制示意，中间两格分别是 tempA source 与 tempB source 的代表传播事件，右侧是同一虚拟 SEEG montage 的整段 electrode readout。readout 里暖色阴影表示正向间期传播事件，蓝色阴影表示反向间期传播事件，黑点/线只标 clean readable 事件的触点峰值顺序。

**关注点**：先看左侧机制是否定义清楚，再看 tempA/tempB 两种特异性组合的传播梯度是否相反，最后看右侧同一 montage 是否反复读出正向/反向事件。
"""
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    (PAPER_FIG / "README.md").write_text(text)


def _write_metadata(paths, stats):
    meta = {
        "figure": "Fig5 core model s3 brake-off",
        "source_artifacts": {
            "readout": str((OUT / "readout_s3_brakeoff.json").relative_to(ROOT)),
            "forward_event_npz": str((OUT / "per_event/rep_s3_brakeoff_neg.npz").relative_to(ROOT)),
            "reverse_event_npz": str((OUT / "per_event/rep_s3_brakeoff_pos.npz").relative_to(ROOT)),
        },
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "readout_stats": stats,
        "notes": [
            "Plotting-only regeneration; no SNN simulation was rerun.",
            "Forward/reverse shadings are clean readable propagation events from readout_s3_brakeoff.json.",
            "E->E long-axis range uses a visible 3-sigma footprint from l_EE and AR around the two pathology cores.",
        ],
    }
    for p in (PAPER_FIG / "core_model_s3_brakeoff_metadata.json",
              FIG / "core_model_s3_brakeoff_metadata.json"):
        p.write_text(json.dumps(meta, indent=2))


def compose(window_ms: float = 8000.0):
    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    readout = _load_readout("s3_brakeoff")
    z_base = _load_npz("s3_brakeoff")
    z_neg = _load_npz("s3_brakeoff_neg")
    z_pos = _load_npz("s3_brakeoff_pos")

    fig = plt.figure(figsize=(18.0, 4.45), facecolor="white")
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[1.0, 1.0, 1.0, 2.75],
        left=0.045,
        right=0.992,
        bottom=0.16,
        top=0.86,
        wspace=0.075,
    )
    _plot_mechanism(fig.add_subplot(gs[0, 0]), z_base, readout)
    _plot_event(fig.add_subplot(gs[0, 1]), z_neg, "tempA source")
    _plot_event(fig.add_subplot(gs[0, 2]), z_pos, "tempB source")
    stats = _plot_readout(fig.add_subplot(gs[0, 3]), z_base, readout["events"], window_ms)
    fig.text(0.012, 0.925, "A", fontsize=19, fontweight="bold")

    old_png = FIG / "core_model_s3_brakeoff.png"
    paper_png = PAPER_FIG / "core_model_s3_brakeoff.png"
    paper_pdf = PAPER_FIG / "core_model_s3_brakeoff.pdf"
    fig.savefig(old_png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(paper_png, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(paper_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paths = {"legacy_png": old_png, "paper_png": paper_png, "paper_pdf": paper_pdf}
    _write_readme()
    _write_metadata(paths, stats)
    return paths


def main():
    os.chdir(ROOT)
    paths = compose()
    for p in paths.values():
        print(f"wrote {p}")


if __name__ == "__main__":
    sys.exit(main())
