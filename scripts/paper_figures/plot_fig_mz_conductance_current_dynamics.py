"""Paper-ready visual diagnostic for the current L=20 MZ-conductance stage.

The producer is plotting-only.  It consumes one compact capture made by
``run_topic4_mz_conductance.py capture-figure`` and renders the honest current
trajectory:

    mechanism | returning event | early runaway | continuous electrode readout

This is not a recovery or bistability figure.  It deliberately labels the terminal
state as runaway and reuses the accepted Topic-4 spatial/contact painters.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-figure")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "paper_figures"))

# Reuse the locked Topic-4 painters instead of constructing look-alike contacts/ranges.
import plot_fig5_core_model_s3_brakeoff as F5  # noqa: E402
import plot_fig_m3a_v2_1_qigk_runaway_transition_gif as QIGK  # noqa: E402


FIG_NAME = "fig_mz_conductance_current_dynamics"
OUT = ROOT / "results" / "paper-ready-figure" / FIG_NAME / "figures"
LATEST = ROOT / "results" / "topic4_sef_hfo" / "mz_conductance" / "latest_figure_capture.json"

RETURN_SHADE = "#f4b266"
RUNAWAY = "#b2182b"
Z_COLOR = "#6a3d9a"
RATE_COLOR = "#222222"
AXIS_COLOR = "#a65f00"


def _shaft(name: str) -> str:
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _load(artifact: str | None):
    latest = json.loads(LATEST.read_text())
    npz_path = Path(artifact).resolve() if artifact else Path(latest["artifact"])
    meta_path = Path(latest["metadata"])
    if artifact:
        candidate = npz_path.with_suffix(".json")
        if candidate.exists():
            meta_path = candidate
    return np.load(npz_path, allow_pickle=True), json.loads(meta_path.read_text()), npz_path, meta_path


def _draw_cores(ax, z, *, stars=None):
    foci = [np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float)]
    r = float(np.asarray(z["core_radius"]).ravel()[0])
    for i, xy in enumerate(foci):
        ax.add_patch(plt.Circle(xy, r, fill=False, ec="crimson", lw=1.15, ls="--", zorder=7))
        if stars is not None and i in stars:
            ax.scatter([xy[0]], [xy[1]], marker="*", s=145, c="black", ec="white", lw=0.8, zorder=9)


def _spatial_style(ax, z):
    L = float(np.asarray(z["L"]).ravel()[0])
    QIGK._style_spatial(ax, L)
    QIGK._draw_contacts(ax, np.asarray(z["contacts"], float), [str(x) for x in z["names"]])


def _plot_mechanism(ax, z):
    pos = np.asarray(z["posE"], float)
    vth = np.asarray(z["vth"], float)
    center = np.asarray(z["center"], float)
    axis = np.asarray(z["axis_unit"], float)
    perp = np.array([-axis[1], axis[0]])
    foci = np.vstack([z["src_xy"], z["snk_xy"]]).astype(float)
    poly = F5._axis_range_patch(center, foci, axis, perp, l_ee=0.38, ar=2.0)
    ax.scatter(
        pos[:, 0], pos[:, 1], c=np.clip(18.0 - vth, 0.0, None),
        s=2.1, cmap="plasma", vmin=0.0, vmax=1.2,
        alpha=0.82, linewidths=0, rasterized=True, zorder=2,
    )
    ax.add_patch(Polygon(poly, closed=True, fc=RETURN_SHADE, ec=AXIS_COLOR,
                         lw=1.15, alpha=0.23, zorder=4))
    p0 = center - axis * 8.2
    p1 = center + axis * 8.2
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=AXIS_COLOR, lw=1.5, zorder=5)
    _draw_cores(ax, z)
    _spatial_style(ax, z)
    ax.set_title("mechanism", fontsize=9.5, fontweight="bold", pad=5)
    ax.text(
        0.04, 0.04,
        r"$g_i^{I}=z_i g_{i,local}^{I}+\beta\langle g_{local}^{I}\rangle$",
        transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.90),
        zorder=12,
    )


def _earliest_core(z, onset):
    pos = np.asarray(z["posE"], float)
    foci = [np.asarray(z["src_xy"], float), np.asarray(z["snk_xy"], float)]
    r = float(np.asarray(z["core_radius"]).ravel()[0])
    med = []
    for xy in foci:
        m = (np.linalg.norm(pos - xy, axis=1) <= r) & np.isfinite(onset)
        med.append(float(np.nanmedian(onset[m])) if m.any() else np.inf)
    return {int(np.argmin(med))} if np.isfinite(min(med)) else set()


def _plot_event(ax, z, rank_key, onset_key, title):
    pos = np.asarray(z["posE"], float)
    rank = np.asarray(z[rank_key], float)
    onset = np.asarray(z[onset_key], float)
    fin = np.isfinite(rank)
    bg = (~fin) & ((np.arange(len(pos)) % 8) == 0)
    ax.scatter(pos[bg, 0], pos[bg, 1], s=1.0, c="0.84", alpha=0.28,
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(pos[fin, 0], pos[fin, 1], c=rank[fin], s=3.0,
               cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.88,
               linewidths=0, rasterized=True, zorder=3)
    center = np.asarray(z["center"], float)
    axis = np.asarray(z["axis_unit"], float)
    p0, p1 = center - axis * 8.2, center + axis * 8.2
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.15", lw=1.0, alpha=0.70, zorder=5)
    _draw_cores(ax, z, stars=_earliest_core(z, onset))
    _spatial_style(ax, z)
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)


def _smooth(x, dt, win_ms=20.0):
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(np.asarray(x, float), np.ones(n) / n, mode="same")


def _lfp_scale(z, runaway_ms):
    t = np.asarray(z["times"], float)
    lfp = np.abs(np.asarray(z["lfp_trace"], float).T)
    pre = t < float(runaway_ms)
    base = np.median(lfp[:, pre], axis=1, keepdims=True)
    pre_scale = np.maximum(np.percentile(lfp[:, pre], 99.0, axis=1, keepdims=True) - base, 1e-9)
    full_scale = np.maximum(np.percentile(lfp, 99.0, axis=1, keepdims=True) - base, 1e-9)
    scale = np.maximum(pre_scale, 0.35 * full_scale)
    return (lfp - base) / scale


def _event_peak_line(ax, ts, traces, y, window):
    m = (ts >= float(window[0])) & (ts <= float(window[1]))
    if int(m.sum()) < 2:
        return
    idx = np.flatnonzero(m)
    pts = []
    for i in range(traces.shape[0]):
        local = idx[int(np.argmax(traces[i, m]))]
        pts.append((float(ts[local]), float(traces[i, local] + y[i])))
        ax.plot(pts[-1][0], pts[-1][1], "o", ms=2.1, mfc="black", mec="white", mew=0.3, zorder=7)
    px, py = zip(*sorted(pts, key=lambda q: q[1], reverse=True))
    ax.plot(px, py, color="black", lw=0.65, alpha=0.55, zorder=6)


def _plot_dynamics(fig, slot, z, meta):
    sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=slot, height_ratios=[0.92, 2.45], hspace=0.08,
    )
    ax_rate = fig.add_subplot(sub[0, 0])
    ax_lfp = fig.add_subplot(sub[1, 0], sharex=ax_rate)
    t_ms = np.asarray(z["times"], float)
    t_s = t_ms / 1000.0
    dt = float(np.median(np.diff(t_ms)))
    runaway_ms = float(np.asarray(z["runaway_ms"]).ravel()[0])
    selected = np.asarray(z["returning_window_ms"], float)
    returning = np.asarray(z["returning_event_intervals_ms"], float)

    rate = _smooth(z["rate_E"], dt)
    ax_rate.plot(t_s, rate, color=RATE_COLOR, lw=0.85, zorder=3)
    ax_rate.set_ylabel("E rate\n(Hz)", fontsize=7.5)
    ax_rate.set_ylim(bottom=0.0)
    ax_rate.tick_params(axis="y", labelsize=6.8, length=2.0)
    ax_rate.tick_params(axis="x", labelbottom=False, length=0)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.spines["right"].set_visible(False)
    ax_d = ax_rate.twinx()
    d_core = 1.0 - np.asarray(z["z_core_mean"], float)
    ax_d.plot(t_s, d_core, color=Z_COLOR, lw=1.1, zorder=4)
    ax_d.set_ylabel(r"core $D=1-z$", color=Z_COLOR, fontsize=7.5)
    ax_d.tick_params(axis="y", labelsize=6.8, colors=Z_COLOR, length=2.0)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_color(Z_COLOR)
    ax_d.set_ylim(0.0, max(0.36, float(np.nanmax(d_core)) * 1.08))
    for lo, hi in returning:
        ax_rate.axvspan(lo / 1000.0, hi / 1000.0, color=RETURN_SHADE, alpha=0.10, lw=0, zorder=0)
    ax_rate.axvspan(selected[0] / 1000.0, selected[1] / 1000.0,
                    color=RETURN_SHADE, alpha=0.42, lw=0, zorder=1)
    ax_rate.axvline(runaway_ms / 1000.0, color=RUNAWAY, lw=1.0, ls="--", zorder=5)

    traces = _lfp_scale(z, runaway_ms)
    names = [str(x) for x in z["names"]]
    shafts = sorted({_shaft(n) for n in names})
    y = np.arange(len(names)) * 1.25
    stride = max(1, int(np.ceil(len(t_s) / 9000)))
    for lo, hi in returning:
        ax_lfp.axvspan(lo / 1000.0, hi / 1000.0, color=RETURN_SHADE, alpha=0.06, lw=0, zorder=0)
    ax_lfp.axvspan(selected[0] / 1000.0, selected[1] / 1000.0,
                   color=RETURN_SHADE, alpha=0.30, lw=0, zorder=1)
    ax_lfp.axvspan(runaway_ms / 1000.0, t_s[-1],
                   color=RUNAWAY, alpha=0.10, lw=0, zorder=0)
    ax_lfp.axvline(runaway_ms / 1000.0, color=RUNAWAY, lw=1.0, ls="--", zorder=5)
    for i, name in enumerate(names):
        color = QIGK._shaft_color(name, shafts)
        ax_lfp.plot(t_s[::stride], traces[i, ::stride] + y[i], color=color,
                    lw=0.68, alpha=0.90, rasterized=True, zorder=3)
    _event_peak_line(ax_lfp, t_ms / 1000.0, traces, y, selected / 1000.0)
    ax_lfp.set_xlim(float(t_s[0]), float(t_s[-1]))
    ax_lfp.set_yticks(y)
    ax_lfp.set_yticklabels(names, fontsize=6.5)
    for tick, name in zip(ax_lfp.get_yticklabels(), names):
        tick.set_color(QIGK._shaft_color(name, shafts))
    ax_lfp.set_xlabel("time (s)", fontsize=8.0)
    ax_lfp.set_ylabel("contact", fontsize=7.8)
    ax_lfp.tick_params(axis="x", labelsize=7.0, length=2.5)
    ax_lfp.tick_params(axis="y", labelsize=6.5, length=2.0)
    ax_lfp.spines["top"].set_visible(False)
    ax_lfp.spines["right"].set_visible(False)
    ax_lfp.legend(
        handles=[
            Patch(facecolor=RETURN_SHADE, alpha=0.40, edgecolor="none", label="returning events"),
            Line2D([0], [0], color=Z_COLOR, lw=1.2, label="core Z depletion"),
            Line2D([0], [0], color=RUNAWAY, lw=1.0, ls="--", label="runaway onset"),
        ],
        frameon=False, fontsize=7.0, loc="upper left", bbox_to_anchor=(0.0, 1.02),
        ncol=3, handlelength=1.5, columnspacing=0.8, borderaxespad=0.0,
    )
    return ax_rate, ax_lfp


def _spatial_stats(z, rank_key):
    rank = np.asarray(z[rank_key], float)
    pos = np.asarray(z["posE"], float)
    center = np.asarray(z["center"], float)
    axis = np.asarray(z["axis_unit"], float)
    perp = np.array([-axis[1], axis[0]])
    valid = np.isfinite(rank)
    pa = (pos - center) @ axis
    pp = (pos - center) @ perp
    return dict(
        n_recruited=int(valid.sum()),
        recruited_fraction=float(valid.mean()),
        onset_axis_spearman=float(spearmanr(pa[valid], rank[valid]).statistic),
        onset_perp_spearman=float(spearmanr(pp[valid], rank[valid]).statistic),
        axis_p95_span_mm=float(np.ptp(np.percentile(pa[valid], [2.5, 97.5]))),
        perp_p95_span_mm=float(np.ptp(np.percentile(pp[valid], [2.5, 97.5]))),
    )


def _write_readme(meta, spatial):
    text = f"""# Current MZ-conductance dynamics

### mz_conductance_current_dynamics.png

这张图使用同一条 L=20、seed 1 自发轨迹：左侧给出当前 conductance + local-Z + protected additive-global GABA 结构；中间分别显示一个按固定规则选出的 returning event 和 terminal early-runaway 的空间招募顺序；右侧把群体放电率、core Z 消耗以及同一真实 E1146 montage 的 15 触点 readout 放在连续时间轴上。returning event 的 onset-axis Spearman 为 {spatial['returning']['onset_axis_spearman']:.2f}，early runaway 降为 {spatial['early_runaway']['onset_axis_spearman']:.2f}；该轨迹在 {meta['runaway_ms']:.1f} ms 进入 runaway，图中没有把它标成可恢复发作态。

**关注点**：先看 returning event 是否仍有局部时序结构，再看 runaway 是否表现为空间招募扩大；最后核对 Z 阶梯只把系统推向 runaway，而没有生成高活动后的回落段。
"""
    (OUT / "README.md").write_text(text)


def compose(artifact: str | None = None):
    z, source_meta, npz_path, meta_path = _load(artifact)
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.0, 4.75), facecolor="white")
    gs = gridspec.GridSpec(
        1, 4, width_ratios=[1.0, 1.0, 1.0, 2.85],
        left=0.045, right=0.992, bottom=0.15, top=0.86, wspace=0.09,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    _plot_mechanism(ax0, z)
    _plot_event(ax1, z, "onset_rank_returning", "onset_returning_ms", "returning event")
    _plot_event(ax2, z, "onset_rank_runaway", "onset_runaway_ms", "early runaway")
    rate_ax, lfp_ax = _plot_dynamics(fig, gs[0, 3], z, source_meta)

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.025, pad=0.02, aspect=28)
    cb.set_label("early → late", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, length=2.0)
    for label, ax in zip(("a", "b", "c", "d"), (ax0, ax1, ax2, rate_ax)):
        ax.text(-0.16 if label != "d" else -0.07, 1.10, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")

    png = OUT / "mz_conductance_current_dynamics.png"
    pdf = OUT / "mz_conductance_current_dynamics.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    spatial = {
        "returning": _spatial_stats(z, "onset_rank_returning"),
        "early_runaway": _spatial_stats(z, "onset_rank_runaway"),
    }

    meta = dict(
        figure=FIG_NAME,
        status="visual diagnostic; current positive route ends in runaway, not recovery",
        source_artifact=str(npz_path.relative_to(ROOT)),
        source_metadata=str(meta_path.relative_to(ROOT)),
        outputs={"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))},
        layout=["mechanism", "returning event", "early runaway", "continuous electrode readout"],
        renderer_reuse=[
            "plot_fig5_core_model_s3_brakeoff._axis_range_patch",
            "plot_fig_m3a_v2_1_qigk_runaway_transition_gif._draw_contacts/_style_spatial/_shaft_color",
        ],
        source_run=source_meta,
        spatial_readout=spatial,
        visual_contract=dict(
            onset_colormap="viridis, 0=early and 1=late",
            montage="registered E1146 narrow 15-contact plane",
            returning_selection=source_meta["selection_rule"],
            terminal_label="runaway",
        ),
        claim_boundary=[
            "supports a spontaneous event-locked Z staircase followed by delayed runaway",
            "does not support a bounded ictal attractor, limit cycle, bistability, or recovery",
            "single-seed spatial visual diagnostic; quantitative verdict remains the multiseed pilot report",
        ],
    )
    (OUT / "mz_conductance_current_dynamics_metadata.json").write_text(json.dumps(meta, indent=2))
    _write_readme(source_meta, spatial)
    print(png)
    print(pdf)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    compose(args.artifact)


if __name__ == "__main__":
    main()
