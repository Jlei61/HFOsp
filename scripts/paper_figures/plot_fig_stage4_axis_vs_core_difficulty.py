"""Topic 4 axis-vs-core figures: (A) the 3-row difficulty figure — why a spontaneous single focus
cannot self-generate a discrete-event train (big核 synchronous blast / small核 front fills the sheet /
kick 两灶 externally-supplied train) — and (B) the axis-vs-core stimulation comparison — at a fixed
electrode footprint, blocking the propagation axis delays runaway at least as much as stimulating the
core, across substrate situations.

SCIENTIFIC STATUS (held, into every title/README/metadata): visual diagnostic, single trajectory +
small screen. runaway/tonic is NEVER an ictal event. "axis >= core" is a within-model fixed-footprint
efficiency statement — established in the multi-source/chokepoint geometry (E1146 kick, cited), tested
honestly (PASS/TIE/FAIL all reported) in the single central core. NOT "proves seizure mechanism", NOT
"treats seizures", NOT "closed-loop/recovery".

Figure A reuses the parity-tested Stage-4 machinery (_build_stage4_patch / _build_subject1146 /
_simulate_continuous); Figure B's kick row cites the committed E1146 stim delays (KICK_REF) and its
small row consumes results/topic4_sef_hfo/axis_vs_core/small_core_stim.json (Task 2 runner).

    python scripts/paper_figures/plot_fig_stage4_axis_vs_core_difficulty.py --figure both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "paper_figures"),
           str(ROOT / "src" / "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import topic4_axis_vs_core as AV  # noqa: E402

# Cited E1146 kick delays (committed fig_m3a_v2_2_qI_stim_site_compare metadata): baseline 757.5 ms;
# core/endpoint stim -> 1171.3 ms (+413.8); axis/middle stim -> 1591.9 ms (+834.4). Footprint 4 each.
KICK_REF = {"baseline_ms": 757.5, "core_ms": 1171.3, "axis_ms": 1591.9,
            "core_delay": 413.8, "axis_delay": 834.4, "footprint": 4}
FIG_DIR = "results/paper-ready-figure/fig_stage4_axis_vs_core_difficulty/figures"
QI_MEAN_COL, QI_MIN_COL, GK_COL, RATE_COL = "#1f7a5a", "#7fc4a6", "#b8860b", "#333333"
CORE_STIM_COL, AXIS_STIM_COL, NOSTIM_COL = "#c0392b", "#2e86c1", "#888888"
ROW_TITLE = {"big": "big核 r=6（自发）", "small": "small核 r=3（自发）", "kick": "kick 两灶（外部戳）"}


def _fmt(x):
    return "—" if x is None else f"{x:.0f}"


def simulate_row(kind):
    import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H
    import run_sef_hfo_snn_cm_spontaneous_readout as C
    if kind == "big":
        cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False, eta_K=0.8,
                               k_K=1.5, tau_K=150.0, sigma_K=0.5, k_q=0.25, tau_q=5000.0, sigma_q=1.5,
                               q_min=0.05, core_mean=16.5, core_std=1.5, core_radius=6.0, drive=0.6,
                               L=20.0, T=200.0, n_pulses=0, seed=1)
    elif kind == "small":
        cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, use_hG=False, eta_K=0.8,
                               k_K=1.5, tau_K=150.0, sigma_K=0.5, k_q=0.25, tau_q=5000.0, sigma_q=1.5,
                               q_min=0.05, core_mean=16.5, core_std=1.5, core_radius=3.0, drive=0.6,
                               L=20.0, T=200.0, n_pulses=0, seed=1)
    else:  # kick — E1146 two-foci; T=1000 spans the cited baseline runaway (~757 ms, config matches
           # the committed qI_stim_site_compare: k_q=0.18, eta_K=0, pulses @130/265/400/535/670…)
        cfg = H.ProtocolConfig(layout="subject1146", top="qI", use_gK=True, use_hG=False, eta_K=0.0,
                               k_q=0.18, tau_q=5000.0, sigma_q=1.5, q_min=0.05, T=1000.0, seed=1)
    S = H._build(cfg)
    DT = float(S["p"].dt); assert abs(DT - C.DT) < 1e-12
    vth = S["patch_vth"] if kind in ("big", "small") else None
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=vth)
    rate_hz = np.asarray(res["rate_E"], float)
    rate_s = H._smooth_rate(rate_hz, DT, 20.0)
    af, bin_w = C.active_fraction(res["E_spk_bool"], DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    n_events = len(C.detect_events(af, bin_w, event_on_frac=bar))
    # col1 shows ONE representative event's spatial ignition (is the front contained, or does it fill
    # the sheet?). For the self-igniting cores the single burst IS the whole trajectory; for the
    # kicked two-foci case, restrict col1 to the first evoked-event window (before the train floods
    # the sheet at the ~757 ms runaway) so it shows a CONTAINED corridor event, not the runaway flood.
    if kind == "kick":
        w_end = int(round((cfg.pulse_start + cfg.pulse_interval) / DT))   # up to the 2nd pulse (~265 ms)
        spk_col1 = res["E_spk_bool"][:w_end]
    else:
        spk_col1 = res["E_spk_bool"]
    onset = AV.onset_time_field(spk_col1, DT)
    # "front fills the sheet" = cumulative fraction of E cells that ever fired in this event; the
    # per-bin max_active_frac saturates at ~0.5 (tau_ref_E=2 ms -> <=half the cells fire per 1 ms bin).
    frac_ever_fired = float(np.isfinite(onset).mean())
    return dict(kind=kind, posE=np.asarray(S["posE"], float), onset=onset,
                frac_ever_fired=frac_ever_fired,
                times=np.asarray(res["times"], float), rate_s=rate_s,
                qI_mean=np.asarray(res["trace_qI_mean"], float), qI_min=np.asarray(res["trace_qI_min"], float),
                gK=np.asarray(res["trace_gK_axial"], float),
                runaway_ms=H._first_sustained(rate_s, DT, 120.0, 100.0),
                n_events=n_events, L=float(S["L"]), max_active_frac=float(af.max()),
                center=np.asarray(S["center"], float), core_r=float(S["layout"]["core_r"]),
                foci=np.asarray(S["layout"]["foci"], float))


def render_figure_a(rows_data, out_dir):
    out_dir = Path(out_dir)
    order = {"big": 0, "small": 1, "kick": 2}
    rows_data = sorted(rows_data, key=lambda d: order.get(d["kind"], 9))
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.8), gridspec_kw={"width_ratios": [1.0, 1.55]})
    for ri, d in enumerate(rows_data):
        ax_geom, ax_tr = axes[ri]
        # --- col1: source-space per-E-cell onset time (viridis early->late) ---
        onset = d["onset"]; finite = np.isfinite(onset)
        if (~finite).any():
            ax_geom.scatter(d["posE"][~finite, 0], d["posE"][~finite, 1], c="0.86", s=3,
                            linewidths=0, zorder=1)
        sc = ax_geom.scatter(d["posE"][finite, 0], d["posE"][finite, 1], c=onset[finite],
                             cmap="viridis", s=6, linewidths=0, zorder=2)
        for f in np.atleast_2d(d["foci"]):
            ax_geom.add_patch(plt.Circle(f, d["core_r"], fill=False, ec="crimson", lw=1.3,
                                         ls="--", zorder=5))
        ax_geom.set_xlim(0, d["L"]); ax_geom.set_ylim(0, d["L"]); ax_geom.set_aspect("equal")
        ax_geom.set_xlabel("x (mm)", fontsize=8); ax_geom.set_ylabel("y (mm)", fontsize=8)
        ax_geom.tick_params(labelsize=7)
        ax_geom.set_title(f"{ROW_TITLE[d['kind']]}｜源空间点火时刻", fontsize=10, fontweight="bold")
        cb = fig.colorbar(sc, ax=ax_geom, fraction=0.046, pad=0.02)
        cb.set_label("点火时刻 onset (ms) 早→晚", fontsize=7.5); cb.ax.tick_params(labelsize=6.5)
        # --- col2: E-rate (left) + q_I / g_K (right twin) ---
        ax_tr.plot(d["times"], d["rate_s"], color=RATE_COL, lw=1.3, label="E-rate (Hz)", zorder=3)
        ax_tr.set_ylabel("E-rate (Hz)", fontsize=8); ax_tr.set_xlabel("time (ms)", fontsize=8)
        ax_tr.tick_params(labelsize=7); ax_tr.set_xlim(0, float(d["times"][-1]))
        ax2 = ax_tr.twinx()
        ax2.plot(d["times"], d["qI_mean"], color=QI_MEAN_COL, lw=1.4, label="mean $q_I$", zorder=2)
        ax2.plot(d["times"], d["qI_min"], color=QI_MIN_COL, lw=1.1, ls="--", label="min $q_I$", zorder=2)
        ax2.plot(d["times"], d["gK"], color=GK_COL, lw=1.3, label="轴向 $g_K$（疲劳）", zorder=2)
        ax2.set_ylim(-0.03, 1.05); ax2.set_ylabel("$q_I,\\ g_K$", fontsize=8); ax2.tick_params(labelsize=7)
        if d["runaway_ms"] is not None:
            ax_tr.axvline(d["runaway_ms"], color="crimson", lw=1.1, ls="--", zorder=6)
        ax_tr.set_title(f"n_events={d['n_events']}｜runaway={_fmt(d['runaway_ms'])} ms｜"
                        f"铺满 frac_ever={d.get('frac_ever_fired', float('nan')):.2f}", fontsize=9)
        if ri == 0:
            h1, l1 = ax_tr.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax_tr.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=6.8, frameon=False, ncol=2)
    fig.suptitle("为什么自发单灶出不了「一串分开的小事件」：要能自燃就得过自持阈值，一过就铺满停不下来\n"
                 "big 整片同步爆（无梯度）｜small 前锋铺满到边界（只是变慢）｜kick 从外部供给事件串",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.006, "visual diagnostic；within-model 单轨迹；runaway/tonic 非 ictal 事件；"
             "col1 每行独立归一化（各行时标差 ~15×）", ha="center", fontsize=8, style="italic", color="0.3")
    fig.tight_layout(rect=[0, 0.02, 1, 0.945])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "difficulty_3row.png", dpi=150); fig.savefig(out_dir / "difficulty_3row.pdf")
    plt.close(fig)


def render_figure_b(small, kick_ref, out_dir):
    out_dir = Path(out_dir)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # --- kick row (cited E1146; established axis >= core at 2-source chokepoint geometry) ---
    axk_geom, axk_bar = axes[0]
    axk_geom.set_title("kick 两灶：core=端点, axis=中段走廊"); axk_geom.axis("off")
    axk_geom.text(0.5, 0.5, "E1146 两灶几何\ncore=端点电极\naxis=中段走廊\n（引用已提交结果）",
                  ha="center", va="center", transform=axk_geom.transAxes, fontsize=11)
    axk_bar.bar(["core-stim", "axis-stim"], [kick_ref["core_delay"], kick_ref["axis_delay"]],
                color=[CORE_STIM_COL, AXIS_STIM_COL])
    axk_bar.set_ylabel("runaway 推迟 (ms)")
    axk_bar.set_title(f"固定 footprint={kick_ref['footprint']}：axis ≥ core")
    for i, v in enumerate([kick_ref["core_delay"], kick_ref["axis_delay"]]):
        axk_bar.text(i, v, f"+{v:.0f}", ha="center", va="bottom")
    # --- small row (this run; honest test — single central core has no shared chokepoint) ---
    axs_geom, axs_bar = axes[1]
    contacts = np.asarray(small["contacts"], float)
    core_r = small.get("config", {}).get("core_r")
    if core_r is not None and len(contacts):
        ctr = contacts[len(contacts) // 2]
        axs_geom.add_patch(plt.Circle(ctr, float(core_r), fill=False, ec="crimson", lw=1.2,
                                      ls="--", zorder=1, label="核 core outline"))
    axs_geom.scatter(contacts[:, 0], contacts[:, 1], c="lightgray", s=30, zorder=2)
    axs_geom.scatter(contacts[small["core_contact_idx"], 0], contacts[small["core_contact_idx"], 1],
                     c=CORE_STIM_COL, s=45, zorder=3, label="core-stim")
    axs_geom.scatter(contacts[small["axis_contact_idx"], 0], contacts[small["axis_contact_idx"], 1],
                     c=AXIS_STIM_COL, s=45, zorder=3, label="axis-stim")
    axs_geom.set_title(f"small核 r=3：source {small['n_source_contacts']} 触点, footprint N={small['config']['N']}")
    axs_geom.legend(loc="upper right"); axs_geom.set_aspect("equal")
    cd = small["arms"]["core_stim"]["runaway_delay_ms"]; ad = small["arms"]["axis_stim"]["runaway_delay_ms"]
    axs_bar.bar(["core-stim", "axis-stim"], [cd, ad], color=[CORE_STIM_COL, AXIS_STIM_COL])
    axs_bar.set_ylabel("runaway 推迟 (ms)")
    verdict = "axis ≥ core" if ad >= cd - 10 else "core > axis (单核无咽喉)"
    axs_bar.set_title(f"固定 footprint=N：{verdict}")
    for i, v in enumerate([cd, ad]):
        axs_bar.text(i, v, f"+{v:.0f}", ha="center", va="bottom")
    fig.suptitle("固定电极预算下：挡轴的刺激效果 vs 打灶（跨情况）", fontweight="bold")
    fig.text(0.5, 0.005, "visual diagnostic；within-model 效率示意，非临床证明；runaway 非 ictal 事件",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "axis_vs_core.png", dpi=150); fig.savefig(out_dir / "axis_vs_core.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()
    os.chdir(ROOT)
    out_dir = ROOT / FIG_DIR
    if args.figure in ("A", "both"):
        import run_sef_hfo_snn_cm_spontaneous_readout as C
        C._engine_guard()
        rows = [simulate_row(k) for k in ("big", "small", "kick")]
        for d in rows:
            print(f"ROW {d['kind']} n_events={d['n_events']} runaway_ms={d['runaway_ms']} "
                  f"frac_ever={round(d['frac_ever_fired'], 4)} "
                  f"max_active_frac={round(d['max_active_frac'], 4)}", flush=True)
        render_figure_a(rows, out_dir)
        print(f"wrote {out_dir / 'difficulty_3row.png'}", flush=True)
    if args.figure in ("B", "both"):
        small_path = ROOT / "results" / "topic4_sef_hfo" / "axis_vs_core" / "small_core_stim.json"
        small = json.loads(small_path.read_text())
        render_figure_b(small, KICK_REF, out_dir)
        print(f"wrote {out_dir / 'axis_vs_core.png'}", flush=True)
    print("DONE_AXIS_VS_CORE_FIGURE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
