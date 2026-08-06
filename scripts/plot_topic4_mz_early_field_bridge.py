"""Plot the MZ early-field bridge (design §13). Plotting only — consumes per-seed artifacts,
runs no simulation.

Outputs (results/topic4_sef_hfo/mz_early_field_bridge/figures/):
  mz_early_field_bridge_seed1.png    — Fig5-grammar diagnostic: two labelled trace strips
      (slow-off interictal-template source + z-only operational runaway) + timing/energy fields.
  mz_early_field_bridge_multiseed.png — diagnostic grid: held-out reproducibility, early association
      with within-shaft null band, contact-vs-source agreement, support/dynamic-range, eligibility table.

Honest provenance (design §13): the timing template comes from matched slow-off, so slow-off and native
are drawn as TWO explicitly labelled strips; a slow-off event window is never shaded on the native trace.
Labelled "operational runaway" / "MZ diagnostic"; never overwrites results/paper-ready-figure/fig5_snn_state_readout/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from scipy.signal import butter, hilbert, sosfiltfilt        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge")
FIG = os.path.join(OUT, "figures")
BAND = (30.0, 80.0)
SHAFT_COLS = {"SCL": "#e8743b", "ICL": "#1f9e9e"}
PRIMARY_WK = "early_0_50_ms"


def _shaft(name):
    import re
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _load(seed):
    d = os.path.join(OUT, "per_seed", f"seed{seed}")
    out = {}
    for base in ("bridge_metrics", "native", "templates", "slowoff"):
        jp = os.path.join(d, base + ".json")
        if os.path.exists(jp):
            out[base] = json.load(open(jp))
    for base in ("slowoff", "native", "templates"):
        np_ = os.path.join(d, base + ".npz")
        if os.path.exists(np_):
            out[base + "_npz"] = np.load(np_, allow_pickle=True)
    return out


def _burst_trace(lfp, times, onset_ms):
    dt_ms = float(np.median(np.diff(times)))
    sos = butter(4, BAND, btype="bandpass", fs=1000.0 / dt_ms, output="sos")
    burst = sosfiltfilt(sos, lfp, axis=0)
    pre = times < float(onset_ms) if onset_ms is not None else np.ones(times.size, bool)
    if pre.sum() < 10:
        pre = np.ones(times.size, bool)
    scale = np.percentile(np.abs(burst[pre]), 95.0, axis=0)
    fp = scale[np.isfinite(scale) & (scale > 1e-12)]
    scale = np.maximum(scale, 0.15 * float(np.median(fp)) if fp.size else 1e-9)
    return 0.68 * burst / scale[None, :]


def _plot_trace(ax, lfp, times, names, title, *, onset_ms=None, shade=None, shade_color="#6F9FD8",
                shade_label=None, onset_label=None):
    tr = _burst_trace(lfp, times, onset_ms)
    off = 1.48
    y = np.arange(len(names)) * off
    for ci, nm in enumerate(names):
        ax.plot(times, tr[:, ci] + y[ci], color=SHAFT_COLS.get(_shaft(nm), "0.4"), lw=0.7, alpha=0.9)
    if shade is not None:
        ax.axvspan(shade[0], shade[1], color=shade_color, alpha=0.16, lw=0, zorder=0, label=shade_label)
    if onset_ms is not None:
        ax.axvline(onset_ms, color="crimson", lw=1.5, ls="--", zorder=8, label=onset_label)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-0.6, y[-1] + 1.7)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    for tick, nm in zip(ax.get_yticklabels(), names):
        tick.set_color(SHAFT_COLS.get(_shaft(nm), "0.4"))
    ax.set_xlabel("time (ms)"); ax.set_ylabel("contacts")
    ax.set_title(title, fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _project(contacts, axis_unit, center):
    au = np.asarray(axis_unit, float); au = au / np.linalg.norm(au)
    tv = np.array([-au[1], au[0]])
    c = np.asarray(contacts, float) - np.asarray(center, float)
    return np.column_stack([c @ au, c @ tv])


def _plot_field(ax, pts, values, names, title, cmap, cbar_label, *, src_pt=None, snk_pt=None):
    finite = np.isfinite(values)
    ax.scatter(pts[~finite, 0], pts[~finite, 1], s=70, facecolors="white", edgecolors="0.4",
               linewidths=0.9, zorder=2)
    sc = ax.scatter(pts[finite, 0], pts[finite, 1], c=values[finite], cmap=cmap, s=90,
                    edgecolors="black", linewidths=0.9, zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (pts[i, 0], pts[i, 1]), fontsize=5.5, ha="center", va="center", zorder=4)
    if src_pt is not None and snk_pt is not None:
        ax.annotate("", xy=snk_pt, xytext=src_pt,
                    arrowprops=dict(arrowstyle="->", color="0.3", lw=1.4, alpha=0.7))
        ax.annotate("source", src_pt, fontsize=7, color="0.3", ha="center", va="top")
        ax.annotate("sink", snk_pt, fontsize=7, color="0.3", ha="center", va="bottom")
    ax.set_aspect("equal"); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("along E1146 axis (mm)"); ax.set_ylabel("transverse (mm)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_seed1(seed=1):
    D = _load(seed)
    if "slowoff_npz" not in D or "native_npz" not in D:
        print(f"[plot] seed{seed} artifacts missing; skip seed figure"); return None
    so, na, tm = D["slowoff_npz"], D["native_npz"], D["templates_npz"]
    names = [str(x) for x in so["names"]]
    axis_map = D["templates"]["direction_axis_mapping"]
    center, axis_unit = axis_map["center"], axis_map["axis_unit"]
    pts = _project(so["contacts"], axis_unit, center)
    src_pt = _project(np.array([axis_map["src_xy"]]), axis_unit, center)[0]
    snk_pt = _project(np.array([axis_map["snk_xy"]]), axis_unit, center)[0]

    onset = D["native"]["onset"]; t120 = D["native"].get("t120_ms")
    t_recruit = onset.get("t_recruit_ms")
    # display the maxAB-WINNING eligible direction: its template concords with the energy field
    # (showing the losing/anti-correlated direction would make the two panels look discordant).
    elig = D["bridge_metrics"].get("template_eligibility", {})
    mx0 = ((((D["bridge_metrics"].get("by_window", {}).get(PRIMARY_WK, {}) or {}).get("contact", {}) or {})
            .get("all_support", {}).get("maxab")) or {})
    ra_v, rb_v = mx0.get("rho_a"), mx0.get("rho_b")
    elig_dirs = [d for d in ("A_to_B", "B_to_A") if elig.get(d, {}).get("contact")]
    if ra_v is not None and rb_v is not None:
        disp_dir = "A_to_B" if ra_v >= rb_v else "B_to_A"
    else:
        disp_dir = elig_dirs[0] if elig_dirs else "A_to_B"
    tmpl = tm["contact_A"] if disp_dir == "A_to_B" else tm["contact_B"]
    # pre-specified display example = the FIRST held-out event (odd chronological index) of the display
    # direction. The axis is BIDIRECTIONAL; the display direction is not a fixed phenotype.
    ev_dir = [str(x) for x in so["event_dir"]]
    disp_events = [i for i, dd in enumerate(ev_dir) if dd == disp_dir]
    disp_idx = disp_events[1] if len(disp_events) >= 2 else (disp_events[0] if disp_events else None)
    ev_shade = None
    if disp_idx is not None:
        ev_shade = (float(so["event_t_on"][disp_idx]), float(so["event_t_off"][disp_idx]) + 40.0)

    energy = na.get(f"contact_energy__{PRIMARY_WK}")
    energy = energy if energy is not None else np.full(len(names), np.nan)

    fig = plt.figure(figsize=(12.5, 9.5), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 0.9, 1.25], hspace=0.42, wspace=0.22,
                          left=0.07, right=0.965, top=0.9, bottom=0.07)
    ax_so = fig.add_subplot(gs[0, :])
    _plot_trace(ax_so, so["lfp_trace"], so["times"], names,
                "slow-off — interictal template source (30-80 Hz)",
                shade=ev_shade, shade_label="held-out event #1 (display-only)")
    if ev_shade:
        ax_so.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_na = fig.add_subplot(gs[1, :])
    en_win = (t_recruit, t_recruit + 50.0) if t_recruit is not None else None
    _plot_trace(ax_na, na["lfp_trace"], na["times"], names,
                "z-only operational runaway (30-80 Hz)", onset_ms=t120,
                shade=en_win, shade_color="crimson", shade_label="0-50 ms energy window",
                onset_label="t120 runaway onset")
    ax_na.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_t = fig.add_subplot(gs[2, 0])
    lab = "TA" if disp_dir == "A_to_B" else "TB"
    _plot_field(ax_t, pts, tmpl, names, f"interictal order (display dir {lab}, bidirectional axis)", "viridis",
                "contact rank (early->late)", src_pt=src_pt, snk_pt=snk_pt)
    ax_e = fig.add_subplot(gs[2, 1])
    _plot_field(ax_e, pts, np.asarray(energy, float), names, "pre-t120 early recruitment energy (0-50 ms)",
                "Blues", "virtual-LFP energy", src_pt=src_pt, snk_pt=snk_pt)

    mx = (((D["bridge_metrics"].get("by_window", {}).get(PRIMARY_WK, {}) or {}).get("contact", {})
           or {}).get("all_support", {}).get("maxab", {}) or {})
    sub = (f"onset {onset.get('status')}  t120={t120}  t_recruit={t_recruit}  "
           f"contact maxAB(mirror-inv)={mx.get('rho_maxab')}  "
           f"(pre-t120 early recruitment; operational runaway, model proxy, not seizure)")
    fig.suptitle(f"MZ diagnostic — E1146 seed{seed} early-field bridge\n{sub}",
                 fontsize=12, fontweight="bold", y=0.975)
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, f"mz_early_field_bridge_seed{seed}.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {p}")
    return p


def plot_multiseed(seeds=(1, 3, 4)):
    data = {s: _load(s) for s in seeds}
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32, left=0.06, right=0.97, top=0.9, bottom=0.09)
    colors = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}

    # Q1 held-out reproducibility (H1)
    ax = fig.add_subplot(gs[0, 0])
    for s in seeds:
        tj = data[s].get("templates", {}).get("templates", {})
        for j, d in enumerate(("A_to_B", "B_to_A")):
            sc = (tj.get(d, {}).get("contact", {}) or {}).get("heldout_scores") or []
            xs = np.full(len(sc), s + (j - 0.5) * 0.18)
            ax.scatter(xs, sc, color=colors[s], marker="o" if j == 0 else "^", s=28,
                       alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="0.6", lw=0.8); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(list(seeds)); ax.set_xlabel("seed"); ax.set_ylabel("held-out Spearman")
    ax.set_title("Q1 template reproducibility\n(o=A→B, ^=B→A)", fontsize=10)

    # Q2 per-seed rho_A, rho_B (grey rings, BOTH directions) + mirror-invariant maxAB (star) + null p95
    ax = fig.add_subplot(gs[0, 1])
    for s in seeds:
        c = (((data[s].get("bridge_metrics", {}).get("by_window", {}).get(PRIMARY_WK, {}) or {})
              .get("contact", {}) or {}).get("all_support", {}))
        mx = (c.get("maxab") or {}); nl = (c.get("within_shaft_null") or {})
        for v in (mx.get("rho_a"), mx.get("rho_b")):        # both direction associations, grey (non-winner visible)
            if v is not None:
                ax.scatter(s, v, s=48, facecolors="none", edgecolors="0.6", linewidths=1.3, zorder=2)
        if mx.get("rho_maxab") is not None:                 # mirror-invariant maxAB in seed colour
            ax.scatter(s, mx["rho_maxab"], color=colors[s], marker="*", s=200, edgecolors="k",
                       linewidths=0.5, zorder=4, label=f"seed{s} maxAB")
            if nl.get("null_p95") is not None:              # within-shaft null p95 (seed3 star sits at/below it)
                ax.plot([s - 0.24, s + 0.24], [nl["null_p95"]] * 2, color="0.35", lw=2, zorder=3)
    ax.axhline(0, color="0.6", lw=0.8); ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(list(seeds)); ax.set_xlabel("seed"); ax.set_ylabel("contact association (primary window)")
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("Q2 rho_A, rho_B (grey rings) + mirror-inv maxAB (star)\n(black line = within-shaft null p95)", fontsize=9)

    # Q3 contact vs source AXIS ENGAGEMENT magnitude (direction-free maxAB); source is supplementary,
    # NOT merged with contact into a direction-agreement claim.
    ax = fig.add_subplot(gs[0, 2])
    for s in seeds:
        bw = data[s].get("bridge_metrics", {}).get("by_window", {}).get(PRIMARY_WK, {})
        c = ((bw.get("contact", {}) or {}).get("all_support", {}).get("maxab") or {}).get("rho_maxab")
        sr = ((bw.get("source", {}) or {}).get("all_support", {}).get("maxab") or {}).get("rho_maxab")
        if c is not None and sr is not None:
            ax.scatter(c, sr, color=colors[s], s=70, label=f"seed{s}", edgecolors="k")
    ax.plot([0, 1], [0, 1], "0.8", lw=0.8, ls=":")
    ax.set_xlim(-0.1, 1); ax.set_ylim(-0.1, 1)
    ax.set_xlabel("contact axis engagement (mirror-inv maxAB)")
    ax.set_ylabel("source axis engagement (maxAB)"); ax.legend(fontsize=7)
    ax.set_title("Q3 contact vs source axis engagement\n(direction-free magnitude; source supplementary)", fontsize=9)

    # Q5 support & dynamic range (primary window contact field)
    ax = fig.add_subplot(gs[1, 0])
    for s in seeds:
        diag = (data[s].get("bridge_metrics", {}).get("by_window", {}).get(PRIMARY_WK, {})
                or {}).get("contact_field_diag", {})
        if diag.get("support"):
            ax.bar(s - 0.18, diag.get("support", 0), width=0.32, color=colors[s], alpha=0.7)
            dr = diag.get("dynamic_range")
            if dr is not None:
                ax.bar(s + 0.18, dr, width=0.32, color=colors[s], alpha=0.35, hatch="//")
    ax.set_xticks(list(seeds)); ax.set_xlabel("seed")
    ax.set_ylabel("support (solid) / dyn-range (hatched)")
    ax.set_title("Q5 field support & dynamic range\n(degenerate field = no bar)", fontsize=10)

    # Q6 eligibility / onset status table
    ax = fig.add_subplot(gs[1, 1:]); ax.axis("off")
    rows = [["seed", "onset", "t120", "t_recruit", "A→B elig", "B→A elig", "maxAB elig",
             "rho_maxAB", "within-p", "pre-runaway"]]
    for s in seeds:
        bm = data[s].get("bridge_metrics", {})
        if not bm:
            rows.append([str(s), "MISSING", "-", "-", "-", "-", "-", "-", "-", "-"]); continue
        on = bm.get("onset", {}); el = bm.get("template_eligibility", {})
        c = ((bm.get("by_window", {}).get(PRIMARY_WK, {}) or {}).get("contact", {}) or {}).get("all_support", {})
        mx = (c.get("maxab") or {}); nl = (c.get("within_shaft_null") or {})
        wt = bm.get("within_trajectory_audit", {})
        def f(x):
            return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
        def fp(x):                                            # p-value: keep small p honest (not "0.00")
            return "-" if x is None else (f"{x:.1e}" if x < 0.01 else f"{x:.3f}")
        rows.append([str(s), on.get("status", "-"), f(bm.get("t120_ms")), f(on.get("t_recruit_ms")),
                     str(el.get("A_to_B", {}).get("contact")), str(el.get("B_to_A", {}).get("contact")),
                     str(bm.get("maxab_eligible")), f(mx.get("rho_maxab")), fp(nl.get("p_one_sided")),
                     f"{wt.get('n_pre_runaway_returning', '-')}"])
    tbl = ax.table(cellText=rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.7)
    for j in range(len(rows[0])):
        tbl[(0, j)].set_facecolor("#dddddd"); tbl[(0, j)].set_text_props(fontweight="bold")
    ax.set_title("Q6 eligibility / onset (unresolved & ineligible shown)", fontsize=10)

    cs = json.load(open(os.path.join(OUT, "cohort_summary.json"))) if os.path.exists(
        os.path.join(OUT, "cohort_summary.json")) else {}
    sub = (f"n complete={cs.get('n_seeds_complete')}  mirror-inv maxAB median={cs.get('rho_maxab_median')}  "
           f"range={cs.get('rho_maxab_range')}  n_positive={cs.get('n_positive_maxab')}  "
           f"(3 seeds of ONE E1146 scaffold, not 3 patients; no cohort p; pre-t120 early recruitment; model proxy)")
    fig.suptitle(f"MZ early-field bridge — multiseed diagnostic (E1146 seeds {list(seeds)})\n{sub}",
                 fontsize=11, fontweight="bold", y=0.975)
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, "mz_early_field_bridge_multiseed.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {p}")
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,3,4")
    ap.add_argument("--seed1-only", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    plot_seed1(seeds[0])
    if not args.seed1_only:
        plot_multiseed(seeds)
