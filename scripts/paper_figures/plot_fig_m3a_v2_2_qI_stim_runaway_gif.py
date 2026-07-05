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


def _draw_arm(fig, row_spec, S, res, metrics, cfg, qi, frame_steps, q_frames,
              activity_field, zlfp, activity_vmax, *, row_title, tm_cursor,
              stim_contacts=None, stim_on=None, stim_off=None, baseline_qmean=None):
    L = S["L"]
    contacts = res["contacts"]
    names = res["names"]
    shafts = sorted({H._shaft(n) for n in names})
    times = res["times"]
    runaway = metrics["runaway_start_ms"]
    T = float(S["p"].T)
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
    ax0.set_title(f"{row_title} — permissivity (1-$q_I$)", fontsize=8.2, fontweight="bold", pad=3)
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
    axg.set_xlim(0.0, T); axg.set_ylim(-0.03, 1.05)
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
    ax2.set_xlim(0.0, T)
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
# Two-arm run + render
# ===========================================================================
def run_compare(cfg, *, stim_on, stim_off, stim_radius, n_stim):
    S = H._build(cfg)
    contacts, names = H._contacts(S)
    center = np.asarray(S["center"], float)
    stim_idx = _select_middle_contacts(names, contacts, center, n=n_stim)
    stim_contacts = contacts[stim_idx]
    is_E = (np.asarray(S["labels"]) == 0)
    target = _electrode_e_mask(S["net"]["pos"], is_E, stim_contacts, stim_radius)

    base_res = H._simulate_continuous(S, cfg, record_gif=True)
    stim_res = H._simulate_continuous(S, cfg, record_gif=True,
                                      stim_target=target, stim_on=stim_on, stim_off=stim_off)
    base_m = H._activity_metrics(base_res, S, cfg)
    stim_m = H._activity_metrics(stim_res, S, cfg)
    return dict(S=S, base_res=base_res, stim_res=stim_res, base_m=base_m, stim_m=stim_m,
                stim_idx=stim_idx, stim_contacts=stim_contacts,
                target_E=int(target.sum()), names=names)


def render(bundle, cfg, *, stim_on, stim_off, out_dir: Path):
    S = bundle["S"]; base_res = bundle["base_res"]; stim_res = bundle["stim_res"]
    base_m = bundle["base_m"]; stim_m = bundle["stim_m"]
    dt = S["p"].dt; T = float(S["p"].T)
    frame_steps = base_res["q_frame_steps"]
    af_base, vals_b = _activity_fields(base_res, S, frame_steps, dt, cfg.activity_window_ms)
    af_stim, vals_s = _activity_fields(stim_res, S, frame_steps, dt, cfg.activity_window_ms)
    allvals = vals_b + vals_s
    activity_vmax = max(1.0, float(np.percentile(np.concatenate(allvals), 98))) if allvals else 1.0
    z_base = _zlfp(base_res); z_stim = _zlfp(stim_res)
    base_qmean = base_res["trace_qI_mean"]
    geo = S.get("layout", {}).get("label", "Stage5 geometry")
    stim_names = [bundle["names"][i] for i in bundle["stim_idx"]]
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    gif = out_dir / "qI_stim_runaway_comparison.gif"
    png = out_dir / "qI_stim_runaway_comparison_final.png"
    pdf = out_dir / "qI_stim_runaway_comparison_final.pdf"
    for qi, step in enumerate(frame_steps):
        last = qi == len(frame_steps) - 1
        tm_cursor = T if last else float(base_res["times"][step])
        fig = plt.figure(figsize=(14.0, 9.7), facecolor="white")
        outer = fig.add_gridspec(2, 1, left=0.055, right=0.985, bottom=0.065, top=0.9, hspace=0.30)
        _draw_arm(fig, outer[0], S, base_res, base_m, cfg, qi, frame_steps, base_res["q_frames"],
                  af_base[qi], z_base, activity_vmax, row_title="no stim", tm_cursor=tm_cursor)
        _draw_arm(fig, outer[1], S, stim_res, stim_m, cfg, qi, frame_steps, stim_res["q_frames"],
                  af_stim[qi], z_stim, activity_vmax,
                  row_title=f"stim ({','.join(stim_names)}, {stim_on:.0f}–{stim_off:.0f} ms)",
                  tm_cursor=tm_cursor, stim_contacts=bundle["stim_contacts"],
                  stim_on=stim_on, stim_off=stim_off, baseline_qmean=base_qmean)
        fig.text(0.016, 0.955, "A", fontsize=18, fontweight="bold")
        fig.suptitle(f"q_I build-up $\\rightarrow$ runaway: stimulation vs no stimulation "
                     f"({geo}) | t={tm_cursor:.0f} ms", fontsize=12.0, fontweight="bold", y=0.975)
        base_rt = base_m["runaway_start_ms"]; stim_rt = stim_m["runaway_start_ms"]
        delay = ("prevented within T" if stim_rt is None else
                 (f"+{stim_rt - base_rt:.0f} ms later" if base_rt is not None else f"{stim_rt:.0f} ms"))
        fig.text(0.50, 0.022,
                 f"no-stim runaway={base_rt} ms | stim runaway={stim_rt} ms ({delay}) | "
                 f"min q_I end: no-stim={base_m['q_min_final']} stim={stim_m['q_min_final']} | "
                 f"stim E cells={bundle['target_E']}",
                 fontsize=8.0, ha="center", color="0.2")
        if last:
            fig.savefig(png, dpi=170, bbox_inches="tight", facecolor="white")
            fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    frames.extend([frames[-1]] * 8)
    imageio.mimsave(gif, frames, duration=0.11, loop=0)
    return gif, png, pdf, activity_vmax


def _write_readme(bundle, cfg, stim_on, stim_off, out_dir: Path):
    base_rt = bundle["base_m"]["runaway_start_ms"]
    stim_rt = bundle["stim_m"]["runaway_start_ms"]
    stim_names = ", ".join(bundle["names"][i] for i in bundle["stim_idx"])
    if stim_rt is None:
        outcome = f"刺激臂在整段仿真窗口内**没有** runaway（no-stim 在 {base_rt} ms 已 runaway）。"
    elif base_rt is not None:
        outcome = f"刺激把 runaway 从 {base_rt} ms 推后到 {stim_rt} ms（晚了 {stim_rt - base_rt:.0f} ms），关刺激后才反弹。"
    else:
        outcome = f"刺激臂 runaway={stim_rt} ms。"
    text = f"""# M3A-v2.2 q_I build-up -> runaway：刺激 vs 不刺激 对照 GIF（E1146 几何）

### qI_stim_runaway_comparison.gif

连续单轨迹 **visual diagnostic**，不是统计 sweep。和 `fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146`
**同一套衬底/种子/多脉冲驱动/q_I 载体**；两臂唯一区别 = 刺激臂在 `[{stim_on:.0f}, {stim_off:.0f}] ms`
把中段 4 个真实触点 `{stim_names}` 附近的兴奋性细胞 V_th clamp（暂时点不着）。刺激只改阈值比较、不动任何
随机数，所以两臂在刺激开之前**逐比特一致**（对照才成立）。

**布局**：两行（上 = 不刺激，下 = 刺激）× 该图三栏——`permissivity(1-q_I)` | 实时 2D SNN 活动 |
（上）`q_I`(mean+min)+轴向 `g_K` 疲劳轨迹（刺激行额外叠一条 no-stim 的 mean q_I 灰虚线做直接对照）／
（下）连续 SEEG readout。刺激窗蓝色阴影、刺激触点蓝方块；各臂 runaway 红虚线。

**这条轨迹里看到的**：{outcome}机制=刺激按住中段→该次"要把它点着"的间期事件被压成局部/压掉→放电少→
抑制资源 `q_I` 少磨（甚至缓慢回血）→越晚到地板→越晚/不 runaway。

**红线（务必照读）**：
- 这是 **visual diagnostic**，外部预防式压制的**示意**，**不**主张电刺激治发作 / recovery / 闭环成立 / 破轴。
- 和内部恢复变量（`h_G`/`g_K` 减法刹车，已另证拉不回饱和雪崩）是**不同问题**：这里压的是 runaway **形成前**的
  driver，不是去 abort 已经烧起来的 runaway。
- runaway / tonic 饱和**不是** ictal-like 事件。

### qI_stim_runaway_comparison_final.png / .pdf

GIF 末帧静态快照，核对两臂末态、runaway 时刻、readout 是否非空。

no-stim runaway: `{base_rt}` ms；stim runaway: `{stim_rt}` ms；刺激触点：`{stim_names}`；
刺激覆盖 E 细胞数：`{bundle['target_E']}`。
"""
    (out_dir / "README.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-on", type=float, default=500.0)
    ap.add_argument("--stim-off", type=float, default=1400.0)
    ap.add_argument("--stim-radius", type=float, default=2.0)
    ap.add_argument("--n-stim-contacts", type=int, default=4)
    ap.add_argument("--T", type=float, default=2300.0)
    ap.add_argument("--n-pulses", type=int, default=17)
    ap.add_argument("--seed", type=int, default=H.ProtocolConfig.seed)
    ap.add_argument("--layout", choices=["stage5", "subject1146"], default="subject1146")
    ap.add_argument("--fig-name", default=None)
    args = ap.parse_args()
    os.chdir(ROOT)

    fig_name = args.fig_name or (f"{FIG_NAME}_epilepsiae_1146" if args.layout == "subject1146" else FIG_NAME)
    cfg = H.ProtocolConfig(
        layout=str(args.layout), top="qI", use_gK=True, eta_K=0.0, use_hG=False,
        T=float(args.T), n_pulses=int(args.n_pulses), seed=int(args.seed), fig_name=fig_name,
    )
    t0 = time.time()
    bundle = run_compare(cfg, stim_on=args.stim_on, stim_off=args.stim_off,
                         stim_radius=args.stim_radius, n_stim=args.n_stim_contacts)
    out_dir = H._out_dir(fig_name)
    gif, png, pdf, activity_vmax = render(bundle, cfg, stim_on=args.stim_on, stim_off=args.stim_off, out_dir=out_dir)

    meta = {
        "figure": fig_name,
        "status": ("visual diagnostic, two arms (stim vs no-stim) on ONE shared trajectory; EXTERNAL "
                   "preventive V_th-clamp of the central ICL contacts during a finite window; NOT a "
                   "treatment/recovery/closed-loop claim, NOT a statistical sweep; runaway/tonic is never ictal-like"),
        "companion_baseline": "fig_m3a_v2_2_qI_runaway_transition_epilepsiae_1146 (same q_I trajectory, no stim)",
        "geometry": bundle["S"].get("layout", {}).get("label", "Stage5 geometry"),
        "config": {**{k: getattr(cfg, k) for k in (
            "seed", "T", "pulse_start", "pulse_interval", "n_pulses", "pulse_duration",
            "kick_boost", "r_kick", "q_min", "k_q", "sigma_q", "tau_q", "use_gK", "eta_K",
            "use_hG", "core_mean", "core_radius", "layout")},
            "stim_on_ms": args.stim_on, "stim_off_ms": args.stim_off,
            "stim_radius_mm": args.stim_radius, "n_stim_contacts": args.n_stim_contacts,
            "stim_contacts": [bundle["names"][i] for i in bundle["stim_idx"]],
            "stim_target_E_cells": bundle["target_E"]},
        "metrics": {
            "no_stim": {k: bundle["base_m"][k] for k in ("runaway_start_ms", "max_rate_hz", "q_mean_final", "q_min_final")},
            "stim": {k: bundle["stim_m"][k] for k in ("runaway_start_ms", "max_rate_hz", "q_mean_final", "q_min_final")},
            "runaway_delay_ms": (None if bundle["stim_m"]["runaway_start_ms"] is None or bundle["base_m"]["runaway_start_ms"] is None
                                 else round(bundle["stim_m"]["runaway_start_ms"] - bundle["base_m"]["runaway_start_ms"], 1)),
        },
        "outputs": {"gif": str(gif.relative_to(ROOT)), "final_png": str(png.relative_to(ROOT)),
                    "final_pdf": str(pdf.relative_to(ROOT))},
        "colorbars": {"permissivity_vmin": 0.0, "permissivity_vmax": 1.0,
                      "activity_vmin": 0.0, "activity_vmax": activity_vmax},
        "parity": "arms byte-identical until stim_on (clamp changes only the V_th comparison, no rng draw)",
        "wall_s": round(time.time() - t0, 1),
    }
    (out_dir / "qI_stim_runaway_comparison_metadata.json").write_text(json.dumps(meta, indent=2))
    _write_readme(bundle, cfg, args.stim_on, args.stim_off, out_dir)
    print(f"wrote {gif}")
    print(f"wrote {png}")
    print(json.dumps(meta["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
