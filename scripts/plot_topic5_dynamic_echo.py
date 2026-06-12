"""Topic 5 Stage-2b dynamic-pattern-echo sentinel figures (MANUAL GATE).

Reads results/topic5_dynamic_echo/sentinel/<ds_sid>_<idx>.json and renders paper-grade
self-contained figures for the visual gate:
  - per seizure: (A) template-alignment over time for the two amplitude features, with the
    pre-registered confirmatory window, the alignment peak, and the channel-shuffle
    significance threshold (max-over-time null); (B) per-feature alignment direction bars
    (early-window mean intensity / rise-time latency / early ramp) to read cross-feature agreement.
  - cross-seizure summary: confirmatory alignment per seizure (sign-consistency across
    epilepsiae + yuquan) and the max-null p-values.

Plain-language axes only (no internal field names). Positive alignment = template-early
contacts (closer to the interictal source) are earlier / stronger / faster in the seizure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SENT = Path("results/topic5_dynamic_echo/sentinel")
MEAN_WIN = (0.0, 5.0)
GATE_FEATURES = ("broadband", "line_length")
FEATURE_LABEL = {"broadband": "broadband power", "line_length": "line-length",
                 "hfa": "high-freq (80–150 Hz)", "spectral_edge": "spectral edge"}
PRETTY_DS = {"epilepsiae": "Epilepsiae", "yuquan": "Yuquan"}


def _curve(rec, feat):
    f = rec["features"].get(feat)
    if not f:
        return None
    c = np.array([np.nan if x is None else x for x in f["echo_act"]["curve"]], float)
    return c


def plot_seizure(rec, out_path):
    t = np.asarray(rec["t_axis"], float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    # ---- Panel A: alignment over time (gate features) + null threshold ----
    ax = axes[0]
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.axvspan(MEAN_WIN[0], MEAN_WIN[1], color="0.92", zorder=0,
               label=f"confirmatory window {MEAN_WIN[0]:.0f}–{MEAN_WIN[1]:.0f}s")
    colors = {"broadband": "#1f77b4", "line_length": "#d62728"}
    for feat in GATE_FEATURES:
        c = _curve(rec, feat)
        if c is None:
            continue
        ax.plot(t, c, color=colors[feat], lw=1.8, label=FEATURE_LABEL[feat], zorder=3)
        g = rec["gate_maxnull"].get(feat, {})
        peak = g.get("echo_peak")
        tp = rec["features"][feat]["echo_act"]["t_peak"]
        if peak is not None and tp is not None:
            ax.plot([tp], [peak], "o", color=colors[feat], ms=7, zorder=4)
        thr = (g.get("channel") or {}).get("null_q95")
        if thr is not None:
            ax.axhline(thr, color=colors[feat], ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax.set_xlabel("time after clinical seizure onset (s)")
    ax.set_ylabel("template alignment\n(+: template-early contacts stronger)")
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("A  alignment of activation with the interictal template, over time\n"
                 "(dashed = channel-shuffle 95% threshold, max-over-time)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    # ---- Panel B: per-feature direction bars ----
    ax = axes[1]
    feats = [f for f in ("broadband", "line_length", "hfa", "spectral_edge")
             if f in rec["features"]]
    metrics = [("early-window\nmean intensity", lambda f: rec["features"][f]["echo_act"]["mean"]),
               ("rise-time\nlatency", lambda f: rec["features"][f]["latency_align"]["t50"]),
               ("early ramp\n(0–2s)", lambda f: rec["features"][f]["ramp_align"]["auc_0_2"])]
    x = np.arange(len(feats))
    w = 0.26
    mcolors = ["#4c72b0", "#dd8452", "#55a868"]
    for j, (mlabel, getter) in enumerate(metrics):
        vals = [(_nan(getter(f))) for f in feats]
        ax.bar(x + (j - 1) * w, vals, w, color=mcolors[j], label=mlabel)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABEL[f].split(" (")[0] for f in feats], fontsize=8)
    ax.set_ylabel("template alignment (+: template-early earlier/stronger/faster)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("B  alignment direction by feature and dynamic quantity\n"
                 "(agreement across bars = construct validity)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    g = rec["gate_maxnull"].get("broadband", {})
    ptxt = (f"broadband peak vs max-null:  p(channel)={_p(g.get('channel'))}  "
            f"p(within-shaft)={_p(g.get('within_shaft'))}  p(anchor)={_p(g.get('anchor_matched'))}")
    fig.suptitle(f"{PRETTY_DS.get(rec['dataset'], rec['dataset'])} {rec['ds_sid'].split('_',1)[1]} "
                 f"— seizure {rec['idx']}   (template channels n={rec['template_common']}, "
                 f"swap={rec['swap_class']}, B={rec['B']})\n{ptxt}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_summary(recs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    labels = [f"{PRETTY_DS.get(r['dataset'], r['dataset'])}\n{r['ds_sid'].split('_',1)[1]} sz{r['idx']}"
              for r in recs]
    x = np.arange(len(recs))
    # ---- left: confirmatory broadband mean (sign consistency) ----
    ax = axes[0]
    main = [_nan(r["confirmatory_broadband_echo_mean"]) for r in recs]
    c1 = [_nan(r["cluster1_broadband_echo_mean"]) for r in recs]
    er = [_nan(r["er_echo_mean"]) for r in recs]
    ax.bar(x - 0.27, main, 0.27, color="#1f77b4", label="main template")
    ax.bar(x, c1, 0.27, color="#9ecae1", label="swap template")
    ax.bar(x + 0.27, er, 0.27, color="0.6", label="ER proxy (held-out)")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("confirmatory alignment\n(broadband mean, 0–5s)")
    ax.set_ylim(-0.6, 0.6)
    ax.set_title("Confirmatory early-window alignment per seizure\n"
                 "(no time/feature selection — the honest direction)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    # ---- right: gate p-values heatmap ----
    ax = axes[1]
    modes = ["channel", "within_shaft", "anchor_matched"]
    mode_lab = ["channel\nshuffle", "within-shaft", "anchor-\nmatched"]
    P = np.full((len(recs), len(modes)), np.nan)
    for i, r in enumerate(recs):
        g = r["gate_maxnull"].get("broadband", {})
        for j, m in enumerate(modes):
            v = (g.get(m) or {}).get("p")
            if v is not None:
                P[i, j] = v
    im = ax.imshow(P, cmap="RdYlGn_r", vmin=0, vmax=0.2, aspect="auto")
    ax.set_xticks(np.arange(len(modes))); ax.set_xticklabels(mode_lab, fontsize=8)
    ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(recs)):
        for j in range(len(modes)):
            if np.isfinite(P[i, j]):
                ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="black")
    ax.set_title("Broadband alignment-peak p vs max-over-time null\n"
                 "(green ≤0.05; the peak must beat the channel shuffle)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    fig.suptitle("Stage 2b sentinel — early-ictal dynamic template echo (manual gate)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _nan(x):
    return np.nan if x is None else float(x)


def _p(g):
    v = (g or {}).get("p")
    return "nan" if v is None else f"{v:.3f}"


def main():
    figs = SENT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(p.read_text()) for p in sorted(SENT.glob("*.json"))]
    if not recs:
        print("no sentinel JSON found — run the sentinel first")
        return
    for r in recs:
        out = figs / f"{r['ds_sid']}_{r['idx']}.png"
        plot_seizure(r, out)
        print("wrote", out)
    plot_summary(recs, figs / "sentinel_summary.png")
    print("wrote", figs / "sentinel_summary.png")


if __name__ == "__main__":
    main()
