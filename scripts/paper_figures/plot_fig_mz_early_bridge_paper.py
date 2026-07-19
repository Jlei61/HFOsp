"""Paper-ready DRAFT figure for the MZ early-ictal bridge (E1146 model).

Plotting only — consumes results/topic4_sef_hfo/mz_early_field_bridge/per_seed/*. No simulation.

Layout (2026-07-19 review):
  Row 1 (2 wide):  slow-off held-out interictal event window | z-only pre-t120 transition window
                   (same scaffold, same seed, TWO INDEPENDENT state replays — not one continuous trace)
  Row 2 (4 cells): TA (A->B) timing field | TB (B->A) timing field | pre-t120 early energy field |
                   cross-seed observed maxAB vs within-shaft null

Both direction templates are shown neutrally; the maxAB winner appears only in the stats panel. No
neuron granular layer (local-participation audit incomplete). Clean canvas: model-proxy disclaimer,
candidate name, exact decimals and eligibility table live in the README/metadata, not on the figure.
Does NOT overwrite the diagnostic figures or the legacy fig5_snn_state_readout.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
from scipy.signal import butter, sosfiltfilt                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_early_field_bridge")
FIGDIR = os.path.join(ROOT, "results", "paper-ready-figure", "fig_mz_early_bridge", "figures")
BAND = (30.0, 80.0)
SHAFT_COLS = {"SCL": "#e8743b", "ICL": "#1f9e9e"}
TA_COL, TB_COL = "#B2182B", "#2166AC"
SEED_COLS = {1: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
CASE_SEED = 1
SEEDS = (1, 3, 4)
PRIMARY_WK = "early_0_50_ms"


def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _load(seed):
    d = os.path.join(OUT, "per_seed", f"seed{seed}")
    out = {}
    for base in ("bridge_metrics", "native", "templates"):
        p = os.path.join(d, base + ".json")
        if os.path.exists(p):
            out[base] = json.load(open(p))
    for base in ("slowoff", "native", "templates"):
        p = os.path.join(d, base + ".npz")
        if os.path.exists(p):
            out[base + "_npz"] = np.load(p, allow_pickle=True)
    return out


def _signed_burst(lfp, times, pre_mask):
    """Signed 30-80 Hz band-pass, per-contact scaled by the 95th-pct |burst| over pre_mask."""
    dt_ms = float(np.median(np.diff(times)))
    sos = butter(4, BAND, btype="bandpass", fs=1000.0 / dt_ms, output="sos")
    burst = sosfiltfilt(sos, lfp, axis=0)
    scale = np.percentile(np.abs(burst[pre_mask]), 95.0, axis=0)
    fp = scale[np.isfinite(scale) & (scale > 1e-12)]
    scale = np.maximum(scale, 0.15 * float(np.median(fp)) if fp.size else 1e-9)
    return burst / scale[None, :]


def _plot_trace(ax, burst, times, names, t0, t1, title, *, shade=None, shade_col="#6F9FD8",
                shade_label=None, vlines=()):
    sel = (times >= t0) & (times <= t1)
    t = times[sel]; b = burst[sel]
    off = 1.5
    y = np.arange(len(names)) * off
    for ci, nm in enumerate(names):
        ax.plot(t, 0.66 * b[:, ci] + y[ci], color=SHAFT_COLS.get(_shaft(nm), "0.4"), lw=0.8, alpha=0.92)
    if shade is not None:
        ax.axvspan(max(shade[0], t0), min(shade[1], t1), color=shade_col, alpha=0.16, lw=0, zorder=0,
                   label=shade_label)
    for xv, col, lab in vlines:
        if t0 <= xv <= t1:
            ax.axvline(xv, color=col, lw=1.5, ls="--", zorder=8, label=lab)
    ax.set_xlim(t0, t1); ax.set_ylim(-0.7, y[-1] + 1.8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    for tick, nm in zip(ax.get_yticklabels(), names):
        tick.set_color(SHAFT_COLS.get(_shaft(nm), "0.4"))
    ax.set_xlabel("time (ms)"); ax.set_title(title, fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if shade_label or vlines:
        ax.legend(loc="upper left", fontsize=6.8, framealpha=0.9)


def _project(contacts, axis_unit, center):
    au = np.asarray(axis_unit, float); au = au / np.linalg.norm(au)
    tv = np.array([-au[1], au[0]])
    c = np.asarray(contacts, float) - np.asarray(center, float)
    return np.column_stack([c @ au, c @ tv])


def _field(ax, pts, vals, names, title, title_col, cmap, cbar_label, src_pt, snk_pt):
    vals = np.asarray(vals, float)
    fin = np.isfinite(vals)
    ax.annotate("", xy=snk_pt, xytext=src_pt,
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.5, alpha=0.7), zorder=1)
    ax.scatter(pts[~fin, 0], pts[~fin, 1], s=64, facecolors="white", edgecolors="0.5",
               linewidths=0.9, zorder=2)
    sc = ax.scatter(pts[fin, 0], pts[fin, 1], c=vals[fin], cmap=cmap, s=88, edgecolors="black",
                    linewidths=0.8, zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (pts[i, 0], pts[i, 1]), fontsize=5.0, ha="center", va="center", zorder=4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=title_col)
    ax.set_xlabel("along axis (mm)", fontsize=8.5); ax.set_ylabel("transverse (mm)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=6.5)


def _stats_panel(ax):
    ax.axhline(0, color="0.7", lw=0.8)
    for s in SEEDS:
        c = (((_load(s).get("bridge_metrics", {}).get("by_window", {}).get(PRIMARY_WK, {}) or {})
              .get("contact", {}) or {}).get("all_support", {}))
        mx = (c.get("maxab") or {}); nl = (c.get("within_shaft_null") or {})
        for v in (mx.get("rho_a"), mx.get("rho_b")):
            if v is not None:
                ax.scatter(s, v, s=42, facecolors="none", edgecolors="0.6", linewidths=1.2, zorder=2)
        if mx.get("rho_maxab") is not None:
            ax.scatter(s, mx["rho_maxab"], color=SEED_COLS[s], marker="*", s=210, edgecolors="k",
                       linewidths=0.5, zorder=4)
        if nl.get("null_p95") is not None:
            ax.plot([s - 0.26, s + 0.26], [nl["null_p95"]] * 2, color="0.35", lw=2.2, zorder=3)
    ax.set_ylim(-1.02, 1.02); ax.set_xlim(0.5, 4.5); ax.set_xticks(list(SEEDS))
    ax.set_xlabel("seed", fontsize=9); ax.set_ylabel("contact association", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("cross-seed maxAB (star) vs null p95 (bar)", fontsize=10, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    D = _load(CASE_SEED)
    so, na, tm = D["slowoff_npz"], D["native_npz"], D["templates_npz"]
    names = [str(x) for x in so["names"]]
    amap = D["templates"]["direction_axis_mapping"]
    center, axis_unit = amap["center"], amap["axis_unit"]
    pts = _project(so["contacts"], axis_unit, center)
    src_pt = _project(np.array([amap["src_xy"]]), axis_unit, center)[0]
    snk_pt = _project(np.array([amap["snk_xy"]]), axis_unit, center)[0]
    onset = D["native"]["onset"]; t120 = float(D["native"]["t120_ms"]); t_rec = float(onset["t_recruit_ms"])

    # pre-specified display example = FIRST held-out event (odd chrono index) of the majority direction
    ev_dir = [str(x) for x in so["event_dir"]]
    counts = {d: ev_dir.count(d) for d in ("A_to_B", "B_to_A")}
    maj = max(counts, key=counts.get)
    maj_idx = [i for i, d in enumerate(ev_dir) if d == maj]
    ev_i = maj_idx[1] if len(maj_idx) >= 2 else maj_idx[0]
    ev_on, ev_off = float(so["event_t_on"][ev_i]), float(so["event_t_off"][ev_i])

    so_burst = _signed_burst(np.asarray(so["lfp_trace"], float), np.asarray(so["times"], float),
                             np.ones(np.asarray(so["times"]).size, bool))
    na_times = np.asarray(na["times"], float)
    na_burst = _signed_burst(np.asarray(na["lfp_trace"], float), na_times, na_times < t_rec)

    fig = plt.figure(figsize=(15.5, 9.6), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[0.95, 1.12], hspace=0.42, wspace=0.5,
                          left=0.055, right=0.975, top=0.88, bottom=0.075)

    ax_so = fig.add_subplot(gs[0, 0:2])
    _plot_trace(ax_so, so_burst, np.asarray(so["times"], float), names, ev_on - 25, ev_off + 60,
                "state A · slow-off — held-out interictal event", shade=(ev_on, ev_off),
                shade_label="interictal event")
    ax_na = fig.add_subplot(gs[0, 2:4])
    _plot_trace(ax_na, na_burst, na_times, names, t_rec - 30, t120 + 110,
                "state B · z-only — pre-t120 transition", shade=(t_rec, t_rec + 50),
                shade_col="crimson", shade_label="0-50 ms early window",
                vlines=[(t120, "crimson", "t120 runaway onset")])

    ax_ta = fig.add_subplot(gs[1, 0])
    _field(ax_ta, pts, tm["contact_A"], names, "TA interictal order (A->B)", TA_COL, "viridis",
           "rank early->late", src_pt, snk_pt)
    ax_tb = fig.add_subplot(gs[1, 1])
    _field(ax_tb, pts, tm["contact_B"], names, "TB interictal order (B->A)", TB_COL, "viridis",
           "rank early->late", src_pt, snk_pt)
    ax_en = fig.add_subplot(gs[1, 2])
    _field(ax_en, pts, na[f"contact_energy__{PRIMARY_WK}"], names, "pre-t120 early energy", "0.15",
           "Blues", "energy (a.u.)", src_pt, snk_pt)
    ax_st = fig.add_subplot(gs[1, 3])
    _stats_panel(ax_st)

    fig.suptitle("MZ early-ictal bridge — E1146 (model)", fontsize=15, fontweight="bold", y=0.965)
    fig.text(0.5, 0.915, "same scaffold · same seed · two independent state replays",
             ha="center", fontsize=10.5, style="italic", color="0.3")
    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, "fig_mz_early_bridge.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIGDIR, "fig_mz_early_bridge.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[paper-fig] wrote {png} (+ .pdf)")
    # metadata sidecar (exact numbers kept OFF the canvas)
    meta = {"case_seed": CASE_SEED, "display_event_index": int(ev_i), "display_event_direction": maj,
            "display_event_t_on_ms": ev_on, "display_event_t_off_ms": ev_off,
            "t_recruit_ms": t_rec, "t120_ms": t120, "primary_window": PRIMARY_WK,
            "framing": "operational runaway = model proxy, NOT clinical seizure; virtual-LFP energy proxy",
            "per_seed": {}}
    for s in SEEDS:
        c = (((_load(s).get("bridge_metrics", {}).get("by_window", {}).get(PRIMARY_WK, {}) or {})
              .get("contact", {}) or {}).get("all_support", {}))
        mx = (c.get("maxab") or {}); nl = (c.get("within_shaft_null") or {})
        meta["per_seed"][str(s)] = {"rho_a": mx.get("rho_a"), "rho_b": mx.get("rho_b"),
                                    "rho_maxab": mx.get("rho_maxab"),
                                    "within_shaft_p": nl.get("p_one_sided"),
                                    "within_shaft_null_p95": nl.get("null_p95")}
    json.dump(meta, open(os.path.join(FIGDIR, "fig_mz_early_bridge_metadata.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
