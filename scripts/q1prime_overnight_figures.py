"""Stage E figures for Q1' overnight exploration.

Generates:
  q1prime_overnight_feature_corr_heatmap.png — subject × feature Spearman ρ matrix
  q1prime_overnight_feature_subtype_discrim_bar.png — per-feature cohort hit count
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

RESULTS = REPO / "results"
BRIDGE_DIR = RESULTS / "topic1_topic5_bridge"
CORR_CSV = BRIDGE_DIR / "q1prime_feature_delta_rho_correlation.csv"
DISCRIM_CSV = BRIDGE_DIR / "q1prime_feature_subtype_discrimination.csv"
OUT_DIR = BRIDGE_DIR / "figures"

FEATURE_COLS = [
    "n_active",
    "active_fraction",
    "onset_spread_sec",
    "median_onset_latency_sec",
]
DELTA_COL = "delta_rho_swap"

# Morandi-ish palette
PAL = ["#8AA6A3", "#C4A484", "#A9908A", "#7C8B96", "#D4B996"]


def make_corr_heatmap(corr_df: pd.DataFrame, out_path: Path) -> None:
    """Subject × feature heatmap of Spearman ρ (delta_rho_swap)."""
    sub = corr_df[
        (corr_df["delta_rho_col"] == DELTA_COL) &
        (corr_df["skip_reason"].isna())
    ].copy()

    # Pivot: rows=subject, cols=feature
    sub["subj_key"] = sub["dataset"] + "_" + sub["subject"]
    pivot = sub.pivot(index="subj_key", columns="feature", values="rho")
    # Keep only feature columns we care about, in order
    cols_present = [c for c in FEATURE_COLS if c in pivot.columns]
    pivot = pivot[cols_present]

    # Sort rows by n_active rho (descending)
    sort_col = "n_active" if "n_active" in pivot.columns else cols_present[0]
    pivot = pivot.sort_values(sort_col, ascending=False)

    n_subj, n_feat = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, n_feat * 1.5), max(4, n_subj * 0.35)), dpi=150)

    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r",
        vmin=-1.0, vmax=1.0
    )
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols_present], fontsize=9)
    ax.set_yticks(range(n_subj))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title(f"Per-subject Spearman ρ(feature, {DELTA_COL})\nn_eligible per cell shown; NaN=white", fontsize=10)

    # Annotate cells
    for i in range(n_subj):
        for j in range(n_feat):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_discrim_bar(discrim_df: pd.DataFrame, out_path: Path) -> None:
    """Per-feature bar: n_sig (p<0.05) and n_strong (|eff|>0.3) across eligible subjects."""
    eligible = discrim_df[discrim_df["skip_reason"].isna()].copy()

    n_features = len(FEATURE_COLS)
    x = np.arange(n_features)
    width = 0.35

    n_sig = []
    n_strong = []
    n_total = []
    dir_pos = []
    dir_neg = []

    for feat in FEATURE_COLS:
        fe = eligible[eligible["feature"] == feat]
        n_sig.append((fe["pval"] < 0.05).sum())
        n_strong.append((fe["effect"].abs() > 0.3).sum())
        n_total.append(len(fe))
        dir_pos.append((fe["direction"] == "pos").sum())
        dir_neg.append((fe["direction"] == "neg").sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Panel 1: n_sig and n_strong
    ax = axes[0]
    bars_sig = ax.bar(x - width/2, n_sig, width, label="p < 0.05", color=PAL[0])
    bars_str = ax.bar(x + width/2, n_strong, width, label="|eff| > 0.3", color=PAL[1])
    ax.axhline(int(n_total[0]) * 0.05, ls="--", color="grey", alpha=0.7,
               label=f"5% chance ({int(n_total[0]*0.05):.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in FEATURE_COLS], fontsize=9)
    ax.set_ylabel("# subjects")
    ax.set_title(f"Feature × subtype discrimination\n(n_eligible={n_total[0] if n_total else 0} subjects per feature)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(max(n_sig + n_strong), 3) + 1)
    for bar in bars_sig:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)), ha="center", va="bottom", fontsize=8)
    for bar in bars_str:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)), ha="center", va="bottom", fontsize=8)

    # Panel 2: direction distribution
    ax2 = axes[1]
    ax2.bar(x - width/2, dir_pos, width, label="subtype 0 > others", color=PAL[2])
    ax2.bar(x + width/2, dir_neg, width, label="subtype 0 < others", color=PAL[3])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f.replace("_", "\n") for f in FEATURE_COLS], fontsize=9)
    ax2.set_ylabel("# subjects")
    ax2.set_title("Direction of subtype 0 vs others")
    ax2.legend(fontsize=9)

    fig.suptitle("Stage D: Feature × subtype discrimination cohort summary", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    corr_df = pd.read_csv(CORR_CSV)
    discrim_df = pd.read_csv(DISCRIM_CSV)

    print("Generating Stage E figures...")
    make_corr_heatmap(corr_df, OUT_DIR / "q1prime_overnight_feature_corr_heatmap.png")
    make_discrim_bar(discrim_df, OUT_DIR / "q1prime_overnight_feature_subtype_discrim_bar.png")
    print("Done.")


if __name__ == "__main__":
    main()
