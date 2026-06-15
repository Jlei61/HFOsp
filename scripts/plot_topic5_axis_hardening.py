"""Topic5 A-line hardening — three paper-grade figures (self-contained, no codebase jargon).

Fig 1  per-patient  : each patient's interictal-axis vs ictal-activation alignment vs its own
                      coarse shuffle null -> the coarse skeleton is present per patient.
Fig 2  null hierarchy: alignment effect size as the control gets stricter (all contacts -> within
                      shaft -> within activity -> shaft & activity) -> broadband coarse-only,
                      HFA deeper but finest is borderline.
Fig 3  window sweep  : alignment effect size across post-onset windows + the distal pre-onset
                      negative control [-120,-90]s -> distal is NOT weaker => persistent scaffold,
                      NOT ictal-onset-specific.

Labels are reader-facing English; no 'joint/channel/within_shaft' jargon, no §/PR codes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AX = Path("results/topic5_ictal_recruitment/axis_alignment")
FIG = AX / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_BB, C_HF = "#1f6f8b", "#d1495b"     # broadband (teal), HFA (rose) — consistent across all 3 figs
C_OK, C_NO = "#2a9d8f", "#bdbdbd"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})

# control strictness order + reader-facing names (no codebase jargon)
NULL_ORDER = ["channel", "within_shaft", "anchor_matched", "joint"]
NULL_LABEL = {"channel": "all contacts\n(coarse)", "within_shaft": "within\nelectrode shaft",
              "anchor_matched": "within\nactivity level", "joint": "shaft &\nactivity (strictest)"}
WIN_ORDER = ["post_0_5", "post_5_10", "post_0_10", "post_0_20", "pre_prox_m10_0", "pre_distal_m120_m90"]
WIN_LABEL = {"post_0_5": "0–5 s", "post_5_10": "5–10 s", "post_0_10": "0–10 s", "post_0_20": "0–20 s",
             "pre_prox_m10_0": "−10–0 s\n(pre)", "pre_distal_m120_m90": "−120–−90 s\n(distal pre,\nneg. control)"}


def _epi(summ):
    return [r for r in summ["per_subject"] if r.get("status") == "ok" and r["dataset"] == "epilepsiae"]


def fig1_patient_level():
    bb = _epi(json.load(open(AX / "axis_alignment_broadband_B1000.json")))
    hf = _epi(json.load(open(AX / "axis_alignment_hfa_B1000.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, rows, name, col in [(axes[0], bb, "Broadband power", C_BB), (axes[1], hf, "Fast activity (60–100 Hz)", C_HF)]:
        rows = sorted(rows, key=lambda r: r["real_median_abs_corr"])
        x = np.arange(len(rows))
        real = [r["real_median_abs_corr"] for r in rows]
        p95 = [r["channel_null_p95"] for r in rows]
        passed = [r["real_median_abs_corr"] > r["channel_null_p95"] for r in rows]
        for xi, p in zip(x, p95):
            ax.plot([xi, xi], [0, p], color="#e6e6e6", lw=4, zorder=1, solid_capstyle="round")
        ax.scatter(x, p95, marker="_", s=130, color="#666666", lw=1.6, zorder=2,
                   label="chance (95th pct of shuffle)")
        xa, ra, pa = np.array(x), np.array(real), np.array(passed)
        ax.scatter(xa[pa], ra[pa], s=52, color=col, zorder=3, edgecolor="white", linewidth=0.7,
                   label="observed — beats chance")
        ax.scatter(xa[~pa], ra[~pa], s=46, facecolor="white", edgecolor="#9a9a9a", linewidth=1.3,
                   zorder=3, label="observed — at chance")
        ax.set_title(f"{name}\n{sum(passed)}/{len(rows)} patients beat chance", fontsize=11)
        ax.set_xlabel("patients (sorted)")
        ax.set_xticks([])
        ax.set_xlim(-0.7, len(rows) - 0.3)
    axes[0].set_ylabel("axis–activation alignment  |r|")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Per-patient: interictal axis reads out a coarse ictal-activation skeleton", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "axis_hardening_fig1_patient_level.png", bbox_inches="tight")
    plt.close(fig)


def fig2_null_hierarchy():
    tbl = json.load(open(AX / "axis_alignment_FINAL.json"))["table"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(NULL_ORDER))
    for metric, col, dx in [("broadband", C_BB, -0.06), ("hfa", C_HF, 0.06)]:
        eff, lo, hi = [], [], []
        for nl in NULL_ORDER:
            r = next((t for t in tbl if t["metric"] == metric and t["null"] == nl and t["B"] == 1000), None)
            eff.append(r["effect_size"] if r and r["effect_size"] is not None else np.nan)
            lo.append((r["effect_size"] - r["effect_ci_lo"]) if r and r["effect_ci_lo"] is not None else 0)
            hi.append((r["effect_ci_hi"] - r["effect_size"]) if r and r["effect_ci_hi"] is not None else 0)
        ax.errorbar(x + dx, eff, yerr=[lo, hi], color=col, marker="o", ms=7, lw=2, capsize=4,
                    label={"broadband": "Broadband power", "hfa": "Fast activity (60–100 Hz)"}[metric])
    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([NULL_LABEL[n] for n in NULL_ORDER], fontsize=9)
    ax.set_xlabel("control (left = coarse, right = strictest)")
    ax.set_ylabel("alignment effect size\n(observed − shuffle, patient median)")
    ax.set_title("How fine is the alignment? Effect shrinks as the control tightens", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9)
    ax.annotate("CI touches 0\nat strictest", xy=(3.06, 0.05), xytext=(2.3, 0.13), fontsize=8.5,
                color=C_HF, arrowprops=dict(arrowstyle="->", color=C_HF, lw=1))
    fig.tight_layout()
    fig.savefig(FIG / "axis_hardening_fig2_null_hierarchy.png", bbox_inches="tight")
    plt.close(fig)


def fig3_window():
    rows = json.load(open(AX / "window/window_summary.json"))["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    x = np.arange(len(WIN_ORDER))
    series = [("broadband", "channel", C_BB, "Broadband (coarse)"),
              ("hfa", "channel", C_HF, "Fast activity (coarse)")]
    for metric, layer, col, lab in series:
        eff, lo, hi = [], [], []
        for w in WIN_ORDER:
            r = next((t for t in rows if t["metric"] == metric and t["layer"] == layer and t["window"] == w), None)
            eff.append(r["effect_size"] if r else np.nan)
            lo.append((r["effect_size"] - r["effect_ci"][0]) if r else 0)
            hi.append((r["effect_ci"][1] - r["effect_size"]) if r else 0)
        ax.errorbar(x, eff, yerr=[lo, hi], color=col, marker="o", ms=6.5, lw=2, capsize=3.5, label=lab)
    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.axvspan(3.5, 5.5, color="#f3e0e0", alpha=0.5, zorder=0)   # pre-onset windows
    ax.set_xticks(x)
    ax.set_xticklabels([WIN_LABEL[w] for w in WIN_ORDER], fontsize=8.5)
    ax.set_xlabel("activation window (relative to seizure onset)")
    ax.set_ylabel("alignment effect size\n(observed − shuffle, patient median)")
    ax.set_title("Alignment is NOT seizure-onset-specific:\ndistal pre-onset is as strong as ictal → persistent scaffold", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.annotate("negative control\nnot weaker", xy=(5, eff[-1] if np.isfinite(eff[-1]) else 0.1),
                xytext=(3.7, 0.20), fontsize=8.5, color="#7a3b3b",
                arrowprops=dict(arrowstyle="->", color="#7a3b3b", lw=1))
    fig.tight_layout()
    fig.savefig(FIG / "axis_hardening_fig3_window_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_patient_level()
    fig2_null_hierarchy()
    fig3_window()
    print("wrote 3 figures to", FIG)
