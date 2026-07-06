"""Stim vs no-stim q_I build-up -> runaway COMPARISON GIF (E1146 geometry).

Companion to `plot_fig_m3a_v2_2_hG_runaway_transition_gif.py` (--top qI). Same shared
substrate / seed / multi-pulse drive / q_I carrier (k_q, kick_boost, g_K visualized but
eta_K=0, h_G OFF), but here TWO arms are run side by side:

    row 0  no stim      -> the approved q_I build-up -> runaway trajectory
    row 1  stim         -> the FOUR central ICL contacts (ICL4-7) are V_th-clamped during a
                           finite window; fewer/smaller interictal events deplete q_I -> the
                           runaway is delayed or held off inside the window; after stim OFF the
                           drive continues so any rebound is visible too.

The two arms are byte-identical until stim_on (the clamp only changes the spike-threshold
comparison, never an rng draw -- see `_simulate_continuous`'s parity contract in the companion
script). That shared baseline is what makes the comparison meaningful.

SCIENTIFIC STATUS (held): visual diagnostic, ONE trajectory per arm, EXTERNAL preventive
suppression. This is NOT a treatment/recovery/closed-loop claim and NOT a statistical sweep;
runaway/tonic is never an ictal-like event. It is a different question from the (separately
NEGATIVE) internal-recovery h_G/g_K brakes, which try to ABORT a saturated avalanche -- here we
only show that suppressing the q_I-depleting driver DURING build-up postpones the transition.

Run:
    python scripts/paper_figures/plot_fig_m3a_v2_2_qI_stim_runaway_gif.py
    python scripts/paper_figures/plot_fig_m3a_v2_2_qI_stim_runaway_gif.py --stim-on 500 --stim-off 1400 --T 2300
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Ellipse, Patch

ROOT = Path(__file__).resolve().parents[2]
ENG = ROOT / "src" / "snn_engine"
for _p in (str(ROOT), str(ROOT / "scripts"), str(Path(__file__).resolve().parent), str(ENG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H  # noqa: E402
import run_m3a_v2_step2_qI as S2  # noqa: E402
from slow_field import firing_rate_field  # noqa: E402

FIG_NAME = "fig_m3a_v2_2_qI_stim_runaway"
STIM_COL = "#2f80ed"


# ===========================================================================
# Stim target: the four central ICL contacts -> nearby E cells
# ===========================================================================
def _select_middle_contacts(names, contacts, center, n: int = 4) -> np.ndarray:
    """Indices of the `n` ICL-shaft contacts closest to `center` (sorted ascending). Falls back
    to all contacts if the ICL shaft is absent. Mirrors the static E1146 stim figure's site rule."""
    contacts = np.asarray(contacts, float)
    center = np.asarray(center, float)
    icl = [i for i, nm in enumerate(names) if H._shaft(nm) == "ICL"]
    pool = icl or list(range(len(names)))
    ranked = sorted(pool, key=lambda i: float(np.linalg.norm(contacts[i] - center)))
    return np.array(sorted(ranked[:n]), dtype=int)


def _electrode_e_mask(pos, is_E, stim_contacts, radius: float) -> np.ndarray:
    """Full-network bool mask of E cells within `radius` mm of ANY stim contact."""
    pos = np.asarray(pos, float)
    sc = np.asarray(stim_contacts, float)
    d = np.linalg.norm(pos[:, None, :] - sc[None, :, :], axis=2)
    return np.asarray(is_E, bool) & (d.min(axis=1) <= radius)


def _stim_site_center(S, site: str) -> np.ndarray:
    """Anchor point the stim contacts are chosen around, per site policy:
      'middle'            -> the sheet center (between the two foci)
      'earliest-endpoint' -> the focus kicked FIRST (tempA = foci[0]; pulse k=0), i.e. the
                             earliest-onset axial endpoint."""
    if site == "middle":
        return np.asarray(S["center"], float)
    if site in ("earliest-endpoint", "endpoint"):
        return np.asarray(H._source_xy(S, "tempA"), float)
    raise ValueError(f"unknown stim site {site!r} (use 'middle' or 'earliest-endpoint')")


def _select_both_foci_contacts(names, contacts, focus_a, focus_b, n: int = 4) -> np.ndarray:
    """Union of the `n` ICL contacts nearest each focus (stimulate BOTH event generators),
    de-duplicated and sorted ascending."""
    ia = _select_middle_contacts(names, contacts, focus_a, n=n)
    ib = _select_middle_contacts(names, contacts, focus_b, n=n)
    return np.array(sorted(set(ia.tolist()) | set(ib.tolist())), dtype=int)


def _build_target(S, site: str, radius: float, n: int) -> dict:
    """Pick the ICL contacts for a stim `site` and the E-cell clamp mask around them.
      'middle' / 'earliest-endpoint' -> n contacts around one anchor point
      'both-foci'                    -> n contacts around EACH focus (both generators)."""
    contacts, names = H._contacts(S)
    if site in ("both-foci", "both"):
        idx = _select_both_foci_contacts(names, contacts, H._source_xy(S, "tempA"),
                                         H._source_xy(S, "tempB"), n=n)
    else:
        idx = _select_middle_contacts(names, contacts, _stim_site_center(S, site), n=n)
    mask = _electrode_e_mask(S["net"]["pos"], np.asarray(S["labels"]) == 0, contacts[idx], radius)
    return dict(idx=idx, contacts=contacts[idx], mask=mask,
                names=[names[i] for i in idx], n_E=int(mask.sum()))


# ===========================================================================
# Per-arm rendering (mirrors the companion qI layout, drawn into one grid row)
# ===========================================================================
def _zlfp(res):
    lfp = np.abs(res["lfp_trace"].T)
    base = np.median(lfp, axis=1, keepdims=True)
    scale = np.maximum(np.percentile(lfp, 98, axis=1, keepdims=True) - base, 1e-9)
    return (lfp - base) / scale


def _activity_fields(res, S, frame_steps, dt, window_ms):
    out, vals = [], []
    for step in frame_steps:
        lo = max(0, step - int(round(window_ms / dt)))
        fired = res["E_spk_bool"][lo: step + 1].any(axis=0)
        A = firing_rate_field(fired, S["posE"], S["L"], S2.N_GRID, sigma=0.5)
        out.append(A)
        if np.any(A > 0):
            vals.append(A[A > 0])
    return out, vals


def _range_label(names):
    """Compact same-shaft consecutive runs, e.g. [ICL8..ICL11] -> 'ICL8–11' and the two-group
    both-foci set [ICL1..ICL4, ICL8..ICL11] -> 'ICL1–4,ICL8–11'. Falls back to a plain join when
    the names are not a single-shaft integer sequence."""
    m = [re.match(r"([A-Za-z]+)(\d+)$", str(n)) for n in names]
    if not (names and all(m) and len({x.group(1) for x in m}) == 1):
        return ",".join(str(n) for n in names)
    pref = m[0].group(1)
    nums = sorted(int(x.group(2)) for x in m)
    runs, lo, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((lo, prev)); lo = prev = n
    runs.append((lo, prev))
    return ",".join(f"{pref}{a}–{b}" if b > a else f"{pref}{a}" for a, b in runs)


def _draw_arm(fig, row_spec, S, res, metrics, cfg, qi, frame_steps, q_frames,
              activity_field, zlfp, activity_vmax, *, row_title, tm_cursor,
              stim_contacts=None, stim_on=None, stim_off=None, baseline_qmean=None, x_end_ms=None):
    L = S["L"]
    contacts = res["contacts"]
    names = res["names"]
    shafts = sorted({H._shaft(n) for n in names})
    times = res["times"]
    runaway = metrics["runaway_start_ms"]
    T = float(S["p"].T)
    x_end = float(x_end_ms) if x_end_ms is not None else T
    win = None if stim_on is None else (stim_on, min(stim_off, T))
    rg = row_spec.subgridspec(1, 3, width_ratios=[1.0, 1.0, 2.15], wspace=0.18)

    def _stim_marks(ax):
        if stim_contacts is not None:
            ax.scatter(stim_contacts[:, 0], stim_contacts[:, 1], s=66, marker="s",
                       fc=STIM_COL, ec="white", lw=0.8, zorder=9)

    # --- col 0: permissivity (1 - q_I) ---
    ax0 = fig.add_subplot(rg[0, 0])
    im0 = ax0.imshow(1.0 - q_frames[qi], origin="lower", extent=[0, L, 0, L],
                     cmap="plasma", vmin=0.0, vmax=1.0)
    ec, ew, eh, ea = H._axis_ellipse(S)
    ax0.add_patch(Ellipse(ec, ew, eh, angle=ea, fc=H.PULSE_A, ec=H.AXIS_COL, lw=1.1, alpha=0.22, zorder=4))
    for source, label in (("tempA", "A"), ("tempB", "B")):
        xy = H._source_xy(S, source)
        ax0.add_patch(plt.Circle(xy, cfg.core_radius, fill=False, ec="crimson", lw=1.0, ls="--", zorder=7))
        ax0.text(xy[0], xy[1] + 0.44, label, fontsize=7.5, color="crimson", fontweight="bold",
                 ha="center", va="bottom", path_effects=[pe.withStroke(linewidth=1.8, foreground="white")])
    H._draw_contacts(ax0, contacts, names)
    _stim_marks(ax0)
    H._style_spatial(ax0, L)
    ax0.set_title(f"{row_title} · permissivity 1$-q_I$", fontsize=7.6, fontweight="bold", pad=3)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # --- col 1: real-time 2D SNN E activity ---
    ax1 = fig.add_subplot(rg[0, 1])
    im1 = ax1.imshow(activity_field, origin="lower", extent=[0, L, 0, L],
                     cmap="viridis", vmin=0.0, vmax=activity_vmax)
    tm = float(times[frame_steps[qi]])
    for pulse in res["pulses"]:
        if pulse["t0"] <= tm <= pulse["t1"] + 8.0:
            xy = H._source_xy(S, pulse["source"])
            ax1.scatter([xy[0]], [xy[1]], marker="*", s=120, c="white", ec="black", lw=0.8, zorder=8)
    c = np.asarray(S["center"]); u = np.asarray(S["axis_unit"])
    ax1.plot([c[0] - u[0] * 4.6, c[0] + u[0] * 4.6], [c[1] - u[1] * 4.6, c[1] + u[1] * 4.6],
             color="white", lw=1.2, alpha=0.9, zorder=5)
    H._draw_contacts(ax1, contacts, names)
    _stim_marks(ax1)
    H._style_spatial(ax1, L)
    ax1.set_title("2D SNN activity", fontsize=8.2, fontweight="bold", pad=3)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    # --- col 2 split: top q_I/g_K trace | bottom continuous readout ---
    sub = rg[0, 2].subgridspec(2, 1, height_ratios=[1.0, 2.4], hspace=0.08)
    axg = fig.add_subplot(sub[0, 0])
    if baseline_qmean is not None:
        axg.plot(times, baseline_qmean, color="0.62", lw=1.0, ls=(0, (4, 2)), zorder=2, label="mean $q_I$ (no stim)")
    axg.plot(times, res["trace_qI_mean"], color=H.QI_MEAN_COL, lw=1.6, zorder=4, label="mean $q_I$")
    axg.plot(times, res["trace_qI_min"], color=H.QI_MIN_COL, lw=1.1, ls="--", zorder=3, label="min $q_I$")
    axg.axhline(cfg.q_min, color="0.6", lw=0.8, ls=":", zorder=1, label="$q_{\\min}$ floor")
    if cfg.use_gK:
        axg.plot(times, res["trace_gK_axial"], color=H.GK_COL, lw=1.4, zorder=5, label="axial $g_K$")
    if win is not None:
        axg.axvspan(win[0], win[1], color=STIM_COL, alpha=0.12, lw=0, zorder=0)
    axg.axvline(tm_cursor, color="black", lw=1.1, alpha=0.9, zorder=7)
    if runaway is not None:
        axg.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
    axg.set_xlim(0.0, x_end); axg.set_ylim(-0.03, 1.05)
    axg.tick_params(axis="x", labelbottom=False, length=2.0)
    axg.tick_params(axis="y", labelsize=6.2, length=2.0)
    axg.set_ylabel("$q_I,\\ g_K$", fontsize=7.4)
    axg.spines["top"].set_visible(False); axg.spines["right"].set_visible(False)
    verdict = "no runaway in window" if runaway is None else f"runaway {runaway:.0f} ms"
    axg.set_title(verdict, fontsize=8.0, fontweight="bold", pad=2,
                  color=("crimson" if runaway is not None else "#2e7d32"))
    axg.legend(frameon=False, fontsize=5.7, loc="lower left", ncol=2, handlelength=1.2, columnspacing=0.7)

    ax2 = fig.add_subplot(sub[1, 0], sharex=axg)
    trace_y = np.arange(len(names)) * H.TRACE_OFF
    for pulse in res["pulses"]:
        if runaway is not None and pulse["t0"] >= runaway:
            continue
        ax2.axvspan(pulse["t0"], pulse["t1"], color=(H.PULSE_A if pulse["source"] == "tempA" else H.PULSE_B),
                    alpha=0.20, lw=0, zorder=0)
    if win is not None:
        ax2.axvspan(win[0], win[1], color=STIM_COL, alpha=0.12, lw=0, zorder=0)
    for i, nm in enumerate(names):
        ax2.plot(times, zlfp[i] + trace_y[i], color=H._shaft_color(nm, shafts), lw=0.6, alpha=0.88, zorder=3)
    ax2.axvline(tm_cursor, color="black", lw=1.1, alpha=0.9, zorder=7)
    if runaway is not None:
        ax2.axvline(runaway, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
    ax2.set_xlim(0.0, x_end)
    ax2.set_yticks(trace_y); ax2.set_yticklabels(names, fontsize=6.2)
    for tick, nm in zip(ax2.get_yticklabels(), names):
        tick.set_color(H._shaft_color(nm, shafts))
    ax2.tick_params(axis="x", labelsize=6.8, length=2.5); ax2.tick_params(axis="y", labelsize=6.2, length=2.0)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_ylabel("contact", fontsize=7.4); ax2.set_xlabel("time (ms)", fontsize=7.4)
    handles = [Patch(facecolor=H.PULSE_A, alpha=0.4, edgecolor="none", label="tempA drive"),
               Patch(facecolor=H.PULSE_B, alpha=0.4, edgecolor="none", label="tempB drive")]
    if win is not None:
        handles.append(Patch(facecolor=STIM_COL, alpha=0.22, edgecolor="none", label="stim ON"))
    ax2.legend(handles=handles, frameon=False, fontsize=6.2, loc="upper right",
               bbox_to_anchor=(1.0, 1.12), ncol=len(handles), handlelength=1.3, columnspacing=0.7)


# ===========================================================================
# Multi-arm run + generic multi-row render
# ===========================================================================
def _run(cfg, sites, *, radius, n, stim_on, stim_off):
    """Build ONE substrate, run the no-stim baseline plus one stim arm per site in `sites`. Every arm
    re-seeds identically, so each stim arm is byte-identical to the baseline until stim_on."""
    S = H._build(cfg)
    base_res = H._simulate_continuous(S, cfg, record_gif=True)
    base_m = H._activity_metrics(base_res, S, cfg)
    arms = {}
    for site in sites:
        tgt = _build_target(S, site, radius, n)
        res = H._simulate_continuous(S, cfg, record_gif=True,
                                     stim_target=tgt["mask"], stim_on=stim_on, stim_off=stim_off)
        arms[site] = dict(res=res, m=H._activity_metrics(res, S, cfg), tgt=tgt)
    return dict(S=S, base_res=base_res, base_m=base_m, arms=arms)


def _render_frames(bundle, cfg, rows, *, stim_on, stim_off, out_dir: Path, stem, main_title, footer_text,
                   gif_end_ms=None, x_end_ms=None):
    """Render N rows (one `_draw_arm` each) as an animated GIF + final PNG/PDF. Each row dict:
    {res, m, row_title, stim_contacts|None, baseline_qmean|None}. A shared activity vmax is used
    across all rows so the 2D-activity colormaps are comparable. `gif_end_ms` (optional) truncates
    the ANIMATION to frames with t <= gif_end_ms so the final frame freezes on a chosen contrast
    moment; the sim/metrics are unaffected (the full trajectory still drives runaway detection), and
    the final-frame time cursor reads gif_end_ms rather than T."""
    S = bundle["S"]; T = float(S["p"].T); dt = S["p"].dt
    frame_steps = bundle["base_res"]["q_frame_steps"]
    if gif_end_ms is not None:
        frame_steps = [s for s in frame_steps if float(bundle["base_res"]["times"][s]) <= float(gif_end_ms)]
    end_time = float(gif_end_ms) if gif_end_ms is not None else T
    af, z, vals = [], [], []
    for r in rows:
        a, v = _activity_fields(r["res"], S, frame_steps, dt, cfg.activity_window_ms)
        af.append(a); z.append(_zlfp(r["res"])); vals += v
    activity_vmax = max(1.0, float(np.percentile(np.concatenate(vals), 98))) if vals else 1.0
    out_dir.mkdir(parents=True, exist_ok=True)
    gif = out_dir / f"{stem}.gif"; png = out_dir / f"{stem}_final.png"; pdf = out_dir / f"{stem}_final.pdf"
    frames = []
    for qi, step in enumerate(frame_steps):
        last = qi == len(frame_steps) - 1
        tm_cursor = end_time if last else float(bundle["base_res"]["times"][step])
        fig = plt.figure(figsize=(14.0, 4.9 * len(rows)), facecolor="white")
        outer = fig.add_gridspec(len(rows), 1, left=0.055, right=0.985, bottom=0.065, top=0.9, hspace=0.30)
        for ri, r in enumerate(rows):
            is_stim = r.get("stim_contacts") is not None
            _draw_arm(fig, outer[ri], S, r["res"], r["m"], cfg, qi, frame_steps, r["res"]["q_frames"],
                      af[ri][qi], z[ri], activity_vmax, row_title=r["row_title"], tm_cursor=tm_cursor,
                      stim_contacts=r.get("stim_contacts"),
                      stim_on=(stim_on if is_stim else None), stim_off=(stim_off if is_stim else None),
                      baseline_qmean=r.get("baseline_qmean"), x_end_ms=x_end_ms)
        fig.text(0.016, 0.955, "A", fontsize=18, fontweight="bold")
        fig.suptitle(main_title.format(t=tm_cursor), fontsize=12.0, fontweight="bold", y=0.975)
        fig.text(0.50, 0.022, footer_text, fontsize=8.0, ha="center", color="0.2")
        if last:
            fig.savefig(png, dpi=170, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
    frames.extend([frames[-1]] * 8)
    imageio.mimsave(gif, frames, duration=0.11, loop=0)
    return gif, png, pdf, activity_vmax


def _delay_str(rt, base_rt):
    if rt is None:
        return "no runaway in window"
    if base_rt is None:
        return f"{rt:.0f} ms"
    return f"+{rt - base_rt:.0f} ms"


def _write_readme_site_compare(base_rt, ep, mid, stim_on, stim_off, out_dir: Path):
    ep_rt, mid_rt = ep["m"]["runaway_start_ms"], mid["m"]["runaway_start_ms"]
    ep_names = ", ".join(ep["tgt"]["names"]); mid_names = ", ".join(mid["tgt"]["names"])
    if ep_rt is not None and mid_rt is not None:
        better = "中段" if mid_rt > ep_rt else ("最早端点" if ep_rt > mid_rt else "两者相当")
        cmp_line = (f"最早端点把 runaway 推到 {ep_rt} ms（{_delay_str(ep_rt, base_rt)}），"
                    f"中段推到 {mid_rt} ms（{_delay_str(mid_rt, base_rt)}）；**{better}压得更后**。")
    else:
        cmp_line = (f"最早端点 runaway={ep_rt} ms，中段 runaway={mid_rt} ms（None=整段仿真窗口内没 runaway）；"
                    f"None 的那个把 runaway 完全压在窗口外，更狠。")
    text = f"""# M3A-v2.2 q_I runaway：刺激位点对照 — 最早端点 vs 中段（E1146 几何）

### qI_stim_site_compare.gif

连续单轨迹 **visual diagnostic**，不是统计 sweep。和 `fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146`
**同一套衬底/种子/多脉冲驱动/q_I 载体**；三条轨迹（不刺激基线 + 两种刺激位点）在刺激开之前**逐比特一致**
（刺激只改阈值比较、不动随机数）。图里只画两条刺激臂做直接对比。

**布局**：两行——上 = **刺激最早起始的端点** `{ep_names}`（最先被点火的灶 tempA 那一端）、
下 = **刺激中段** `{mid_names}`；两行都在 `[{stim_on:.0f}, {stim_off:.0f}] ms` 施刺激。每行三栏：
`permissivity(1-q_I)` | 实时 2D SNN 活动 |（上）`q_I`(mean+min)+轴向 `g_K` 疲劳（**叠一条 no-stim 的
mean q_I 灰虚线做基线对照**）／（下）连续 SEEG readout。刺激窗蓝色阴影、刺激触点蓝方块、runaway 红虚线。

**这条轨迹里看到的**：不刺激基线 runaway={base_rt} ms。{cmp_line}
机制直觉——打端点=直接掐掉那个灶的点火（但另一个灶还在放电磨刹车）；打中段=两灶都点得着但传不过去、
每次事件更小。谁更省抑制资源 `q_I`、把 runaway 推得更后，就是这张图要比的。

**红线（务必照读）**：
- **visual diagnostic**，外部预防式压制**示意**，**不**主张电刺激治发作 / recovery / 闭环 / 破轴。
- runaway / tonic 饱和**不是** ictal-like 事件。

### qI_stim_site_compare_final.png / .pdf

GIF 末帧静态快照，核对两条刺激臂末态、runaway 时刻、readout 是否非空。

no-stim runaway: `{base_rt}` ms；最早端点刺激 runaway: `{ep_rt}` ms（触点 `{ep_names}`）；
中段刺激 runaway: `{mid_rt}` ms（触点 `{mid_names}`）。
"""
    (out_dir / "README.md").write_text(text)


def _write_readme_no_stim(base_rt, arm, site_label, stem, stim_on, stim_off, out_dir: Path):
    stim_rt = arm["m"]["runaway_start_ms"]
    stim_names = ", ".join(arm["tgt"]["names"])
    if stim_rt is None:
        outcome = f"刺激臂在整段仿真窗口内**没有** runaway（no-stim 在 {base_rt} ms 已 runaway）。"
    elif base_rt is not None:
        outcome = f"刺激把 runaway 从 {base_rt} ms 推后到 {stim_rt} ms（晚了 {stim_rt - base_rt:.0f} ms）。"
    else:
        outcome = f"刺激臂 runaway={stim_rt} ms。"
    text = f"""# M3A-v2.2 q_I build-up -> runaway：刺激 @ {site_label} vs 不刺激（E1146 几何）

### {stem}.gif

连续单轨迹 **visual diagnostic**，不是统计 sweep。和 `fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146`
**同一套衬底/种子/多脉冲驱动/q_I 载体**；两臂唯一区别 = 刺激臂在 `[{stim_on:.0f}, {stim_off:.0f}] ms`
把 **{site_label}** 的真实触点 `{stim_names}` 附近的兴奋性细胞 V_th clamp（暂时点不着）。刺激只改阈值比较、
不动任何随机数，所以两臂在刺激开之前**逐比特一致**（对照才成立）。

**布局**：两行（上 = 不刺激，下 = 刺激）× 该图三栏——`permissivity(1-q_I)` | 实时 2D SNN 活动 |
（上）`q_I`(mean+min)+轴向 `g_K` 疲劳轨迹（刺激行额外叠一条 no-stim 的 mean q_I 灰虚线做直接对照）／
（下）连续 SEEG readout。刺激窗蓝色阴影、刺激触点蓝方块；各臂 runaway 红虚线。

**这条轨迹里看到的**：{outcome}机制=刺激按住 {site_label}→"要把它点着"的间期事件被压成局部/压掉→放电少→
抑制资源 `q_I` 少磨（甚至缓慢回血）→越晚到地板→越晚/不 runaway。

**红线**：**visual diagnostic**、外部预防式压制**示意**，**不**主张电刺激治发作 / recovery / 破轴；
和内部恢复变量（`h_G`/`g_K` 减法刹车拉不回饱和雪崩）是不同问题；runaway / tonic 不是 ictal-like 事件。

### {stem}_final.png / .pdf

GIF 末帧静态快照。no-stim runaway: `{base_rt}` ms；stim runaway: `{stim_rt}` ms；刺激触点：`{stim_names}`。
"""
    (out_dir / "README.md").write_text(text)


def _arm_metrics(m):
    return {k: m[k] for k in ("runaway_start_ms", "max_rate_hz", "q_mean_final", "q_min_final")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["no_stim_vs_middle", "no_stim_vs_both_foci", "endpoint_vs_middle"],
                    default="no_stim_vs_middle",
                    help="no_stim_vs_middle: baseline vs middle-ICL stim; "
                         "no_stim_vs_both_foci: baseline vs stim on BOTH foci; "
                         "endpoint_vs_middle: earliest-onset endpoint stim vs middle stim")
    ap.add_argument("--stim-on", type=float, default=500.0)
    ap.add_argument("--stim-off", type=float, default=1400.0)
    ap.add_argument("--stim-radius", type=float, default=2.0)
    ap.add_argument("--n-stim-contacts", type=int, default=4)
    ap.add_argument("--T", type=float, default=2300.0)
    ap.add_argument("--gif-end-ms", type=float, default=None,
                    help="truncate the ANIMATION to t<=this (sim/metrics unaffected); freezes the "
                         "final frame on a chosen contrast moment, e.g. --gif-end-ms 1400 ends at "
                         "stim-window close where endpoint has run away but middle still holds")
    ap.add_argument("--x-end-ms", type=float, default=None,
                    help="cap the trace/readout time axis at this (default: full T); keep T long "
                         "enough to still DETECT late runaways, e.g. --T 1750 --x-end-ms 1600")
    ap.add_argument("--n-pulses", type=int, default=17)
    ap.add_argument("--seed", type=int, default=H.ProtocolConfig.seed)
    ap.add_argument("--layout", choices=["stage5", "subject1146"], default="subject1146")
    ap.add_argument("--fig-name", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)
    on, off = float(args.stim_on), float(args.stim_off)

    # per-mode figure stem / dir + (for the no-stim modes) the single stim site + labels
    STEM = {"endpoint_vs_middle": "qI_stim_site_compare",
            "no_stim_vs_middle": "qI_stim_runaway_comparison",
            "no_stim_vs_both_foci": "qI_stim_both_foci_comparison"}
    DEFAULT_FIG = {"endpoint_vs_middle": "fig_m3a_v2_2_qI_stim_site_compare",
                   "no_stim_vs_middle": FIG_NAME,
                   "no_stim_vs_both_foci": "fig_m3a_v2_2_qI_stim_both_foci"}
    NO_STIM_SITE = {"no_stim_vs_middle": ("middle", "中段", "middle"),
                    "no_stim_vs_both_foci": ("both-foci", "两个灶", "both foci")}
    stem = STEM[args.mode]
    default_fig = DEFAULT_FIG[args.mode]
    fig_name = args.fig_name or (f"{default_fig}_epilepsiae_1146" if args.layout == "subject1146" else default_fig)
    cfg = H.ProtocolConfig(
        layout=str(args.layout), top="qI", use_gK=True, eta_K=0.0, use_hG=False,
        T=float(args.T), n_pulses=int(args.n_pulses), seed=int(args.seed), fig_name=fig_name,
    )
    t0 = time.time()
    sites = (["earliest-endpoint", "middle"] if args.mode == "endpoint_vs_middle"
             else [NO_STIM_SITE[args.mode][0]])
    bundle = _run(cfg, sites, radius=args.stim_radius, n=args.n_stim_contacts, stim_on=on, stim_off=off)
    S = bundle["S"]; geo = S.get("layout", {}).get("label", "Stage5 geometry")
    base_m = bundle["base_m"]; base_rt = base_m["runaway_start_ms"]
    base_qmean = bundle["base_res"]["trace_qI_mean"]
    out_dir = H._out_dir(fig_name)

    if args.mode == "endpoint_vs_middle":
        ep, mid = bundle["arms"]["earliest-endpoint"], bundle["arms"]["middle"]
        ep_rt, mid_rt = ep["m"]["runaway_start_ms"], mid["m"]["runaway_start_ms"]
        rows = [
            dict(res=ep["res"], m=ep["m"], stim_contacts=ep["tgt"]["contacts"], baseline_qmean=base_qmean,
                 row_title=f"stim @ earliest endpoint ({_range_label(ep['tgt']['names'])})"),
            dict(res=mid["res"], m=mid["m"], stim_contacts=mid["tgt"]["contacts"], baseline_qmean=base_qmean,
                 row_title=f"stim @ middle ({_range_label(mid['tgt']['names'])})"),
        ]
        main_title = f"q_I runaway — endpoint vs middle stim · {geo} · t={{t:.0f}} ms"
        footer_text = (f"no-stim runaway={base_rt} ms | earliest-endpoint={ep_rt} ms ({_delay_str(ep_rt, base_rt)}) | "
                       f"middle={mid_rt} ms ({_delay_str(mid_rt, base_rt)}) | "
                       f"stim E cells: endpoint={ep['tgt']['n_E']} middle={mid['tgt']['n_E']}")
        gif, png, pdf, activity_vmax = _render_frames(bundle, cfg, rows, stim_on=on, stim_off=off,
                                                      out_dir=out_dir, stem=stem, main_title=main_title,
                                                      footer_text=footer_text, gif_end_ms=args.gif_end_ms,
                                                      x_end_ms=args.x_end_ms)
        stim_meta = {"earliest_endpoint": {"contacts": ep["tgt"]["names"], "target_E_cells": ep["tgt"]["n_E"],
                                           **_arm_metrics(ep["m"])},
                     "middle": {"contacts": mid["tgt"]["names"], "target_E_cells": mid["tgt"]["n_E"],
                                **_arm_metrics(mid["m"])}}
        _write_readme_site_compare(base_rt, ep, mid, on, off, out_dir)
    else:
        site_key, zh_label, en_label = NO_STIM_SITE[args.mode]
        arm = bundle["arms"][site_key]; arm_rt = arm["m"]["runaway_start_ms"]
        rows = [
            dict(res=bundle["base_res"], m=base_m, stim_contacts=None, baseline_qmean=None, row_title="no stim"),
            dict(res=arm["res"], m=arm["m"], stim_contacts=arm["tgt"]["contacts"], baseline_qmean=base_qmean,
                 row_title=f"stim @ {en_label} ({_range_label(arm['tgt']['names'])})"),
        ]
        main_title = f"q_I runaway — {en_label} stim vs no stim · {geo} · t={{t:.0f}} ms"
        footer_text = (f"no-stim runaway={base_rt} ms | stim @ {en_label}={arm_rt} ms ({_delay_str(arm_rt, base_rt)}) | "
                       f"min q_I end: no-stim={base_m['q_min_final']} stim={arm['m']['q_min_final']} | "
                       f"stim E cells={arm['tgt']['n_E']}")
        gif, png, pdf, activity_vmax = _render_frames(bundle, cfg, rows, stim_on=on, stim_off=off,
                                                      out_dir=out_dir, stem=stem, main_title=main_title,
                                                      footer_text=footer_text, gif_end_ms=args.gif_end_ms,
                                                      x_end_ms=args.x_end_ms)
        stim_meta = {site_key: {"contacts": arm["tgt"]["names"], "target_E_cells": arm["tgt"]["n_E"],
                                **_arm_metrics(arm["m"])}}
        _write_readme_no_stim(base_rt, arm, zh_label, stem, on, off, out_dir)

    meta = {
        "figure": fig_name,
        "mode": args.mode,
        "status": ("visual diagnostic; EXTERNAL preventive V_th-clamp of ICL contacts during a finite "
                   "window; NOT a treatment/recovery/closed-loop claim, NOT a statistical sweep; "
                   "runaway/tonic is never ictal-like"),
        "companion_baseline": "fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146 (same q_I trajectory, no stim)",
        "geometry": geo,
        "config": {**{k: getattr(cfg, k) for k in (
            "seed", "T", "pulse_start", "pulse_interval", "n_pulses", "pulse_duration",
            "kick_boost", "r_kick", "q_min", "k_q", "sigma_q", "tau_q", "use_gK", "eta_K",
            "use_hG", "core_mean", "core_radius", "layout")},
            "stim_on_ms": on, "stim_off_ms": off,
            "stim_radius_mm": args.stim_radius, "n_stim_contacts": args.n_stim_contacts,
            "gif_end_ms": args.gif_end_ms, "x_end_ms": args.x_end_ms},
        "metrics": {"no_stim": _arm_metrics(base_m), **stim_meta,
                    "runaway_delay_ms": {site: (None if a["m"]["runaway_start_ms"] is None or base_rt is None
                                                else round(a["m"]["runaway_start_ms"] - base_rt, 1))
                                         for site, a in bundle["arms"].items()}},
        "outputs": {"gif": str(gif.relative_to(ROOT)), "final_png": str(png.relative_to(ROOT)),
                    "final_pdf": str(pdf.relative_to(ROOT))},
        "colorbars": {"permissivity_vmin": 0.0, "permissivity_vmax": 1.0,
                      "activity_vmin": 0.0, "activity_vmax": activity_vmax},
        "parity": "arms byte-identical until stim_on (clamp changes only the V_th comparison, no rng draw)",
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {gif}")
    print(f"wrote {png}")
    print(json.dumps(meta["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
