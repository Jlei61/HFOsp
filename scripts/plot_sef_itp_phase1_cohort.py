"""SEF-ITP Phase 1 cohort verdict bar charts.

Inputs:
  - results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json

Outputs (per AGENTS.md figures/ convention):
  - figures/cohort_h1_verdict_distribution.png
  - figures/cohort_h2_verdict_distribution.png
  - figures/cohort_h6_verdict_distribution.png
  - figures/cohort_coord_coverage.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Verdict orderings — keep "interpretable" verdicts before "untestable" ones
H1_ORDER = [
    "PASS",
    "partial_PASS",
    "PASS_one_side_untestable",
    "NULL",
    "NULL_one_side_untestable",
    "FAIL",
    "FAIL_DIFFUSE",
    "UNTESTABLE_BOTH_SIDES",
    "INCOMPLETE_GATED_ON_COORDS",
    "INCONCLUSIVE_ENVELOPE_INDETERMINATE",
]
H2_ORDER = ["PASS", "partial_PASS", "NULL", "FAIL", "EMPTY_SET",
            "GATED_NO_COORDS", "INSUFFICIENT_NULL", "DEGENERATE"]
H6_ORDER = ["PASS", "PARTIAL", "NULL", "INSUFFICIENT_SPLIT",
            "EXCLUDED_SINGLE_SHAFT", "INSUFFICIENT_NULL", "GATED_NO_COORDS"]

# Colors by category
COLOR = {
    "pass":      "#3a7d44",   # green
    "partial":   "#94c97a",   # light green
    "null":      "#7d7d7d",   # grey
    "fail":      "#c0382b",   # red
    "untestable":"#d3d3d3",   # light grey
}


def _color_for(verdict: str) -> str:
    if verdict == "PASS":
        return COLOR["pass"]
    if verdict in ("partial_PASS", "PARTIAL"):  # PARTIAL = H6's 1-of-3-significant tier
        return COLOR["partial"]
    if verdict == "PASS_one_side_untestable":
        return COLOR["pass"]
    if verdict in ("NULL", "NULL_one_side_untestable"):
        return COLOR["null"]
    if verdict in ("FAIL", "FAIL_DIFFUSE"):
        return COLOR["fail"]
    return COLOR["untestable"]


def _bar(ax, counts: dict, order, title: str, total: int):
    keys = [k for k in order if k in counts]
    extra = [k for k in counts if k not in order]
    keys = keys + extra
    vals = [counts[k] for k in keys]
    colors = [_color_for(k) for k in keys]
    ypos = range(len(keys))
    ax.barh(list(ypos), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(keys, fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=9)
    ax.set_xlabel(f"n (total = {total})")
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("results/topic4_sef_itp/phase1_spatial_geometry/cohort_summary.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/topic4_sef_itp/phase1_spatial_geometry/figures"))
    args = parser.parse_args(argv)

    summary = json.load(open(args.summary))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_subj = summary["n_subjects"]

    # === H1 ===
    fig, ax = plt.subplots(figsize=(8, 4.5))
    h1d = summary["h1"]["verdict_distribution_per_cluster"]
    _bar(ax, h1d, H1_ORDER,
         f"H1 endpoint compactness — verdict distribution per cluster\n"
         f"(n={summary['h1']['n_clusters_total']} clusters from {n_subj} subjects, 23/cohort)",
         total=summary['h1']['n_clusters_total'])
    plt.tight_layout()
    p = args.output_dir / "cohort_h1_verdict_distribution.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")

    # === H2 ===
    fig, ax = plt.subplots(figsize=(8, 3.5))
    h2d = summary["h2"]["verdict_distribution_per_pair_integrated"]
    _bar(ax, h2d, H2_ORDER,
         f"H2 source/sink reversal — integrated verdict per fwd/rev pair\n"
         f"(n={summary['h2']['n_pairs_total']} pairs from {n_subj} subjects)",
         total=summary['h2']['n_pairs_total'])
    plt.tight_layout()
    p = args.output_dir / "cohort_h2_verdict_distribution.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")

    # === H6 ===
    fig, ax = plt.subplots(figsize=(8, 3.5))
    h6d = summary["h6"]["verdict_distribution"]
    _bar(ax, h6d, H6_ORDER,
         f"H6 participation field segregation — verdict per subject\n"
         f"(n={summary['h6']['n_subjects']} subjects)",
         total=summary['h6']['n_subjects'])
    plt.tight_layout()
    p = args.output_dir / "cohort_h6_verdict_distribution.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")

    # === Coord coverage ===
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    cc = summary["coord_coverage"]
    cs = cc["coord_space_distribution"]
    nc = cc["normalization_certainty_distribution"]
    keys, vals = list(cs.keys()), list(cs.values())
    axes[0].barh(keys, vals, color="#3a7d44", edgecolor="black")
    for i, v in enumerate(vals):
        axes[0].text(v + max(vals) * 0.02, i, str(v), va="center", fontsize=9)
    axes[0].set_xlabel(f"n subjects (cohort = {n_subj})")
    axes[0].set_title("Coord space distribution", fontsize=11)
    keys2, vals2 = list(nc.keys()), list(nc.values())
    axes[1].barh(keys2, vals2, color="#94c97a", edgecolor="black")
    for i, v in enumerate(vals2):
        axes[1].text(v + max(vals2) * 0.02, i, str(v), va="center", fontsize=9)
    axes[1].set_xlabel(f"n subjects (cohort = {n_subj})")
    axes[1].set_title("Normalization certainty distribution", fontsize=11)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p = args.output_dir / "cohort_coord_coverage.png"
    plt.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
