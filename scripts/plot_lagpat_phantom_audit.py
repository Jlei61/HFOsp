#!/usr/bin/env python3
"""Diagnostic figures for the lagPatRank phantom-rank audit.

Per plan ``docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md``
§2.3, produces three figures + README into
``results/lagpatrank_audit/figures/``.

Figures:
  1. ami_vs_noise_floor.{png,pdf}        — scatter audit AMI vs seed floor
  2. phantom_fraction_vs_delta.{png,pdf} — phantom_fraction vs Δ
  3. stable_k_confusion.{png,pdf}        — original vs masked stable_k table
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.plot_style import (  # noqa: E402
    FS_LABEL, FS_TICK, FS_TITLE,
    style_panel,
)

CSV_PATH = REPO_ROOT / "results" / "lagpatrank_audit" / "cohort_summary.csv"
OUT_DIR = REPO_ROOT / "results" / "lagpatrank_audit" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> List[Dict[str, str]]:
    with open(CSV_PATH) as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok"]


def _int(s):
    return int(s) if s not in ("", None, "None") else None


def fig_ami_vs_floor(rows):
    sk2 = [r for r in rows if _int(r["stable_k"]) == 2]
    skhi = [r for r in rows if _int(r["stable_k"]) is not None and _int(r["stable_k"]) > 2]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for group, marker, face, edge, label in [
        (sk2, "o", "#3F7A88", "#1F4F5C", "stable_k=2 (n=35)"),
        (skhi, "s", "#C77E48", "#7E4A22", "stable_k>2 (n=5)"),
    ]:
        if not group:
            continue
        x = [float(r["ami_seed_floor_original"]) for r in group]
        y = [float(r["ami_audit"]) for r in group]
        ax.scatter(x, y, marker=marker, s=70, facecolors=face, edgecolors=edge,
                   linewidths=0.8, label=label, zorder=4)

    # Diagonal — phantom audit equal to seed-jitter floor
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1.0,
            label="audit = seed floor", zorder=2)
    # -0.05 / -0.15 reference bands
    xs = np.linspace(0, 1, 50)
    ax.fill_between(xs, xs - 0.05, xs, color="0.85", alpha=0.4,
                    zorder=1, label="cosmetic band (Δ ≥ -0.05)")

    ax.set_xlabel("seed-jitter noise floor (AMI across 5 seeds on original features)",
                  fontsize=FS_LABEL)
    ax.set_ylabel("audit signal  AMI(original, masked) at seed=0", fontsize=FS_LABEL)
    ax.set_title("lagPatRank phantom-rank audit — cohort distribution",
                 fontsize=FS_TITLE, loc="left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=FS_TICK - 1, frameon=False)
    style_panel(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ami_vs_noise_floor.png", dpi=150)
    fig.savefig(OUT_DIR / "ami_vs_noise_floor.pdf")
    plt.close(fig)


def fig_phantom_vs_delta(rows):
    sk2 = [r for r in rows if _int(r["stable_k"]) == 2]
    skhi = [r for r in rows if _int(r["stable_k"]) is not None and _int(r["stable_k"]) > 2]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for group, marker, face, edge, label in [
        (sk2, "o", "#3F7A88", "#1F4F5C", "stable_k=2"),
        (skhi, "s", "#C77E48", "#7E4A22", "stable_k>2"),
    ]:
        x = [float(r["phantom_fraction"]) for r in group]
        y = [float(r["ami_audit_minus_floor"]) for r in group]
        ax.scatter(x, y, marker=marker, s=70, facecolors=face, edgecolors=edge,
                   linewidths=0.8, label=label, zorder=4)

    # gate threshold lines
    ax.axhline(-0.05, color="0.6", linestyle=":", linewidth=1.0,
               label="cosmetic threshold (-0.05)")
    ax.axhline(-0.15, color="0.3", linestyle="--", linewidth=1.0,
               label="broad-rederivation threshold (-0.15)")

    all_rows = sk2 + skhi
    x_all = np.array([float(r["phantom_fraction"]) for r in all_rows])
    y_all = np.array([float(r["ami_audit_minus_floor"]) for r in all_rows])
    rho, p = spearmanr(x_all, y_all)
    ax.text(0.02, 0.04, f"Spearman ρ = {rho:.2f}, p = {p:.1e}",
            transform=ax.transAxes, fontsize=FS_TICK, va="bottom", ha="left",
            color="0.3")

    ax.set_xlabel("phantom fraction  (~bools).sum() / bools.size", fontsize=FS_LABEL)
    ax.set_ylabel("ami_audit_minus_floor  Δ", fontsize=FS_LABEL)
    ax.set_title("Phantom fraction vs cluster-identity damage", fontsize=FS_TITLE, loc="left")
    ax.set_ylim(-1.05, 0.05)
    ax.legend(loc="lower right", fontsize=FS_TICK - 1, frameon=False)
    style_panel(ax)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "phantom_fraction_vs_delta.png", dpi=150)
    fig.savefig(OUT_DIR / "phantom_fraction_vs_delta.pdf")
    plt.close(fig)


def fig_stable_k_confusion(rows):
    pairs = []
    for r in rows:
        ok_ = _int(r["stable_k"])
        mk = _int(r["stable_k_masked"])
        if ok_ is None or mk is None:
            continue
        pairs.append((ok_, mk))
    ks = sorted({k for pair in pairs for k in pair})
    mat = np.zeros((len(ks), len(ks)), dtype=int)
    for orig, mask in pairs:
        i = ks.index(orig)
        j = ks.index(mask)
        mat[i, j] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(mat, cmap="Blues", aspect="equal")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v:
                color = "white" if v > mat.max() * 0.55 else "black"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=FS_LABEL, color=color)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks, fontsize=FS_TICK)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=FS_TICK)
    ax.set_xlabel("masked stable_k", fontsize=FS_LABEL)
    ax.set_ylabel("original stable_k", fontsize=FS_LABEL)
    ax.set_title("stable_k confusion — original vs masked features",
                 fontsize=FS_TITLE, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Annotate diagonal vs off-diagonal counts
    n_diag = int(sum(mat[i, i] for i in range(len(ks))))
    n_off = int(mat.sum() - n_diag)
    ax.text(1.05, -0.18, f"on-diagonal (no change): {n_diag}/{n_diag + n_off}\n"
                          f"off-diagonal (flip): {n_off}/{n_diag + n_off}",
            transform=ax.transAxes, fontsize=FS_TICK, va="top", ha="left",
            color="0.3")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "stable_k_confusion.png", dpi=150)
    fig.savefig(OUT_DIR / "stable_k_confusion.pdf")
    plt.close(fig)


def write_readme(rows):
    sk2 = [r for r in rows if _int(r["stable_k"]) == 2]
    delta_sk2 = np.array([float(r["ami_audit_minus_floor"]) for r in sk2])
    text = f"""# lagPatRank Phantom-Rank Audit — Diagnostic Figures

Source data: `../cohort_summary.csv` (n=40 subjects; all status=ok).
Code: `scripts/audit_kmeans_phantom_rank.py` +
`scripts/augment_lagpat_audit_masked_stable_k.py` +
`scripts/plot_lagpat_phantom_audit.py`.
Plan: `docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md`.

---

### ami_vs_noise_floor.png

主图：x = original feature 上跨 5 seed 的 pairwise median AMI（seed-jitter noise floor），
y = AMI(original@seed=0, masked@seed=0) = audit signal。对角线 = "audit AMI 等于 seed floor"
（cosmetic 通过区域）。灰色带 = cosmetic 阈值 (Δ ≥ -0.05)。

**关注点**：所有 40 个 subject 都远低于对角线。stable_k=2 cohort (n=35, 圆点)
median 音频 = 0.39，floor = 1.0；stable_k>2 cohort (n=5, 方块) 同样落在底部。
无 subject 进 cosmetic band。Cosmetic 出口被实证排除。

### phantom_fraction_vs_delta.png

x = phantom_fraction = `(~bools).sum() / bools.size`，y = `ami_audit_minus_floor`。
两条参考线 = pre-registered cosmetic (-0.05) / broad-rederivation (-0.15) 阈值。
标注 Spearman ρ。

**关注点**：所有 40 subject 都在 broad-rederivation 阈值 (-0.15) 之下；35/40 在
-0.50 之下。phantom fraction 越高，Δ 越负 (ρ = -0.42, p = 0.007)，符合
"more phantom → more cluster-identity damage" 的预期。phantom fraction 范围
[0.140, 0.458]，median 0.328 —— 即便最低 phantom 占比的 subject 仍越线。

### stable_k_confusion.png

行 = original stable_k，列 = masked stable_k 重新跑后的选择 (k_range=[2..6])。
对角线 = 重选 k 与原一致；off-diagonal = stable_k 翻转。

**关注点**：36/40 对角线 (k 选择不变)。4/40 翻转：
3 个高 k 翻转 (huangwanling 4→3, zhaojinrui 5→6, zhangjinhan 6→5) 都在 n_ch ≤ 5
的 stable_k>2 cohort 内 (这些 subject 已被 Topic 4 H3 主分析以独立结构理由排除)。
**唯一 stable_k=2 cohort 内的翻转：epilepsiae_916 (2→4)**，Δ=-0.81，
masked floor=1.0；material flip。

---

## 一句话验收

Cohort gate 走 **Broad re-derivation**。
- median Δ = -0.609 (stable_k=2 cohort median -0.609; stable_k>2 median -0.252)
- 35/35 stable_k=2 subject 越 broad 阈值
- 4/40 stable_k 翻转，1 个在主线 cohort 内 (epilepsiae_916: 2→4)
- 已写归档 `docs/archive/topic1/propagation/lagpatrank_phantom_audit_diagnostic_2026-05-20.md`
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> int:
    rows = _load()
    fig_ami_vs_floor(rows)
    fig_phantom_vs_delta(rows)
    fig_stable_k_confusion(rows)
    write_readme(rows)
    print(f"figures + README -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
