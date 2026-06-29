"""M3A-v2.2 carrier-exploration RESULT-SUMMARY figure (reads the sweep, not a re-run).

Reads the autonomous exploration `per_run.jsonl` (3184 runs: main L=10 sweep + backup follow-up)
and renders the NEGATIVE gate result in three independent panels:

  1. slow-off failure mode vs r_hold  -- does sustained ramp+HOLD break the all-or-none mode?
  2. q_I+g_K carrier outcome           -- does the carrier make a partial-fill candidate?
  3. clean events vs the partial-fill target box -- are the rare clean slow-off events candidates?

This is a descriptive result summary (a real statistical figure over the sweep), NOT a mechanism
claim and NOT an SNN re-run. tonic/multiburst is fail-closed (never ictal-like).

Run: python scripts/paper_figures/plot_fig_m3a_v2_2_explore_summary.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPL = ROOT / "results" / "topic4_m3a_v2_2_explore"
OUT = ROOT / "results" / "paper-ready-figure" / "fig_m3a_v2_2_explore_summary" / "figures"

R_HOLDS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85]
TONIC, NOEV, CLEAN = "#c44e52", "#bbbbbb", "#4c72b0"   # red / grey / blue


def _load(target_L=10.0):
    """All per_run rows from the L=10 sweep + follow-up. Skip any run dir whose run_config records a
    different L (e.g. the L=16 sensitivity run, whose dir name carries no marker)."""
    rows = []
    for f in sorted(glob.glob(str(EXPL / "2026*/per_run.jsonl"))):
        rc = Path(f).parent / "run_config.json"
        if rc.exists():
            try:
                cfg = json.load(open(rc))
                if "L" in cfg and abs(float(cfg["L"]) - target_L) > 1e-9:
                    continue                      # different scale -> not this summary
            except Exception:
                pass
        for line in open(f):
            line = line.strip()
            if line and '"error"' not in line:
                rows.append(json.loads(line))
    return rows


def _bucket(r):
    if r.get("clean_single_event"):
        return "clean"
    if r.get("segmentation_status") == "no_event":
        return "noev"
    return "tonic"   # TONIC_OR_MULTIBURST / INSUFFICIENT — fail-closed


def main():
    rows = _load()
    s1 = [r for r in rows if r["stage"] == "stage1_slowoff"]
    s2 = [r for r in rows if r["stage"] == "stage2_qigk"]
    L = next((r.get("L") for r in rows if r.get("L")), 10.0)   # L=10 sweep (early rows predate L-logging)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor="white")
    fig.suptitle(f"M3A-v2.2 carrier under sustained ramp+HOLD — NEGATIVE / fail-closed "
                 f"(n={len(rows)} sims, L={L} mm)", fontsize=11.5, fontweight="bold", y=1.02)

    # ---- Panel 1: slow-off failure mode vs r_hold (stacked fraction over substrates+seeds) ----
    ax = axes[0]
    frac = {k: [] for k in ("tonic", "noev", "clean")}
    for rh in R_HOLDS:
        cell = [r for r in s1 if abs(r["r_hold"] - rh) < 1e-9]
        n = max(1, len(cell))
        b = [_bucket(r) for r in cell]
        for k in frac:
            frac[k].append(b.count(k) / n)
    x = np.arange(len(R_HOLDS))
    ax.bar(x, frac["tonic"], color=TONIC, label="tonic / fail-closed")
    ax.bar(x, frac["noev"], bottom=frac["tonic"], color=NOEV, label="no event")
    ax.bar(x, frac["clean"], bottom=np.array(frac["tonic"]) + np.array(frac["noev"]),
           color=CLEAN, label="clean single-event")
    ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in R_HOLDS], fontsize=8)
    ax.set_xlabel("drive hold level $r_{hold}$", fontsize=9.5)
    ax.set_ylabel("fraction of slow-off runs", fontsize=9.5)
    ax.set_title("slow-off: protocol preserves all-or-none\n(C1-A everywhere; clean events negligible)",
                 fontsize=9.5)
    ax.set_ylim(0, 1.0); ax.legend(fontsize=7.6, loc="lower center", frameon=False, ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- Panel 2: q_I+g_K carrier outcome (phenotype composition + candidate count) ----
    ax = axes[1]
    comp = {"tonic": sum(1 for r in s2 if _bucket(r) == "tonic"),
            "noev": sum(1 for r in s2 if _bucket(r) == "noev"),
            "clean": sum(1 for r in s2 if r.get("clean_single_event"))}
    cand = sum(1 for r in s2 if r.get("partial_fill_candidate"))
    bars = ax.bar([0, 1, 2], [comp["tonic"], comp["noev"], comp["clean"]],
                  color=[TONIC, NOEV, CLEAN], width=0.62)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["tonic /\nfail-closed", "no\nevent", "clean\nsingle-event"], fontsize=8.5)
    ax.set_ylabel("q_I+g_K runs", fontsize=9.5)
    ax.set_title(f"q_I+g_K carrier (n={len(s2)}): no partial-fill\n"
                 f"PARTIAL-FILL CANDIDATES = {cand}", fontsize=9.5)
    for b, v in zip(bars, [comp["tonic"], comp["noev"], comp["clean"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(comp.values()) * 0.01, str(v),
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(comp.values()) * 1.14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- Panel 3: the clean events vs the partial-fill TARGET box ----
    ax = axes[2]
    clean = [r for r in rows if r.get("clean_single_event")]
    ax.axvspan(0.10, 0.80, ymin=0.0, ymax=(0.70 / 1.05), color="#bfe3bf", alpha=0.45, zorder=0)
    ax.text(0.45, 0.34, "partial-fill\ntarget", ha="center", va="center", fontsize=8.5,
            color="#2e7d32", zorder=1)
    if clean:
        R = [r["R_area"] for r in clean]; S = [r["S_axis"] for r in clean]
        ax.scatter(R, S, s=70, c=CLEAN, ec="black", lw=0.8, zorder=3)
        ax.annotate(f"{len(clean)} clean events\n(backup r=0.85; all R<0.1, S>0.9)",
                    xy=(float(np.median(R)), float(np.min(S))), xytext=(0.30, 0.88),
                    fontsize=8.2, color=CLEAN, va="center",
                    arrowprops=dict(arrowstyle="->", color=CLEAN, lw=0.9))
    ax.axhline(0.70, color="0.4", ls="--", lw=0.9, zorder=2)
    ax.axvline(0.10, color="0.4", ls="--", lw=0.9, zorder=2)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.05)
    ax.set_xlabel("recruitment area $R$", fontsize=9.5)
    ax.set_ylabel("axis score $S_{axis}$ (1=fully axial)", fontsize=9.5)
    ax.set_title("the only clean events are tiny + high-axial\n(outside the partial-fill box → not candidates)",
                 fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"explore_summary.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sub_c1 = {}
    for sub in sorted({r["substrate"] for r in s1}):
        ss = [r for r in s1 if r["substrate"] == sub]
        sub_c1[sub] = sum(1 for r in ss if r.get("c1_branch") == "A_failure_mode_preserved")
    (OUT / "README.md").write_text(
        "# M3A-v2.2 carrier-exploration result summary\n\n"
        f"读取自主探索 `per_run.jsonl`（{len(rows)} 次真仿真，L={L} mm，sustained ramp+HOLD 协议）"
        "画的**结果汇总图**（非 SNN 重跑、非统计 sweep 主张）。三联：\n\n"
        "### explore_summary.png\n"
        "- **左**：slow-off 失败模式 vs 驱动强度——每个 `r_hold` 上 tonic（红）/ no-event（灰）/ 干净单事件（蓝）的比例。"
        "持续协议**没**把全或无变温和：tonic 全程压倒，干净事件可忽略。\n"
        f"- **中**：`q_I+g_K` 载体表型组成——**partial-fill 候选 = {sum(1 for r in s2 if r.get('partial_fill_candidate'))}**。\n"
        "- **右**：唯一那几个干净事件落在 partial-fill 目标框（R∈[0.1,0.8] 且 S<0.7）**之外**（范围太小、轴向性太高）→ 不是候选。\n\n"
        f"**关注点**：三个面板都指向同一结论——载体在 sustained 协议下 **fail-closed**，0 partial-fill 候选；"
        f"C1 失败模式保留 per substrate = {sub_c1}。tonic/multiburst 一律不当 ictal-like。\n")
    print(f"wrote {OUT/'explore_summary.png'} | s1={len(s1)} s2={len(s2)} clean={len(clean)} candidates={cand}")


if __name__ == "__main__":
    main()
