"""Plot rev9 alpha-star selection versus unseen-network diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


DEFAULT_SUMMARY = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "node_edge_calibration/alpha_star_out_of_selection_summary.json")
DEFAULT_OUT = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "node_edge_calibration/figures")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _scatter(axis, rows, feature, title):
    colors = {"field_component": "#D55E00", "matched_off_field": "#0072B2"}
    names = {"field_component": "field component",
             "matched_off_field": "off-field control"}
    all_x, all_y = [], []
    for role in colors:
        subset = [row for row in rows if row["role"] == role
                  and row["paired_eligible"]]
        x = np.asarray([row["node_scalars"][feature] for row in subset], float)
        y = np.asarray([row["edge_scalars"][feature] for row in subset], float)
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[valid], y[valid], s=20, color=colors[role], alpha=0.8,
                     edgecolor="white", linewidth=0.35, label=names[role])
        all_x.extend(x[valid].tolist())
        all_y.extend(y[valid].tolist())
    all_x, all_y = np.asarray(all_x), np.asarray(all_y)
    lower = float(min(all_x.min(), all_y.min()))
    upper = float(max(all_x.max(), all_y.max()))
    pad = max(0.05 * (upper - lower), 1e-4)
    axis.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
              color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(lower - pad, upper + pad)
    axis.set_ylim(lower - pad, upper + pad)
    axis.axhline(0.0, color="#BBBBBB", linewidth=0.5)
    axis.axvline(0.0, color="#BBBBBB", linewidth=0.5)
    rho = spearmanr(all_x, all_y).statistic
    axis.text(0.04, 0.95, f"rho={rho:.2f} | n={len(all_x)}",
              transform=axis.transAxes, va="top", fontsize=7.5)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Node slope")
    axis.set_ylabel("Edge slope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    payload = json.loads(Path(args.summary).read_text())
    comparison = payload["comparison_to_selection"]
    selection = comparison["selection"]
    unseen = comparison["out_of_selection"]
    intervals = payload["seed_bootstrap_95_interval"]
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)

    axis = axes[0]
    bars = np.asarray([
        [selection["response_loss_median"],
         1.0 - selection["paired_coverage"],
         0.5 * selection["off_field_loss_median"],
         0.25 * selection["baseline_shift_median"]],
        [unseen["response_loss_median"], unseen["missing_pair_penalty"],
         0.5 * unseen["off_field_loss_median"],
         0.25 * unseen["baseline_shift_median"]],
    ])
    labels = ("paired response", "missing pairs", "off-field", "sham baseline")
    colors = ("#4C78A8", "#B8B8B8", "#F2CF5B", "#E45756")
    bottom = np.zeros(2)
    for values, label, color in zip(bars.T, labels, colors):
        axis.bar([0, 1], values, bottom=bottom, width=0.58, color=color,
                 edgecolor="white", linewidth=0.4, label=label)
        bottom += values
    low, high = intervals["J_eval"]
    axis.errorbar(1, unseen["J_eval"],
                  yerr=[[unseen["J_eval"] - low], [high - unseen["J_eval"]]],
                  color="#222222", capsize=3, linewidth=1.0, zorder=5)
    axis.set_xticks([0, 1], ["selection seeds", "unseen seeds"])
    axis.set_ylabel("frozen objective")
    axis.set_title("reference transfer", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(-0.12, 1.07, "a", transform=axis.transAxes,
              fontsize=10, fontweight="bold")

    paired_rows = [row for row in payload["pair_rows"]
                   if row["paired_eligible"]]
    _scatter(axes[1], paired_rows, 0, "source response")
    axes[1].text(-0.12, 1.07, "b", transform=axes[1].transAxes,
                 fontsize=10, fontweight="bold")
    _scatter(axes[2], paired_rows, 1, "downstream response")
    axes[2].legend(frameon=False, loc="lower right")
    axes[2].text(-0.12, 1.07, "c", transform=axes[2].transAxes,
                 fontsize=10, fontweight="bold")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "rev9_alpha_star_out_of_selection"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    metadata = dict(
        status="REV9_ALPHA_STAR_OUT_OF_SELECTION_DIAGNOSTIC",
        scientific_role=(
            "Unseen-network local-response diagnostic with frozen alpha; not "
            "patient validation or Node-Edge equivalence"),
        alpha_star=payload["alpha_star"], observed=payload["observed"],
        intervals=intervals,
        source=dict(path=args.summary, sha256=_sha256(args.summary)),
        git_commit=_git_commit())
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    readme = out_dir / "README.md"
    text = readme.read_text() if readme.exists() else ""
    heading = "### rev9_alpha_star_out_of_selection.png"
    if heading not in text:
        text += (
            "\n" + heading + "\n\n"
            "这张图检验冻结 alpha 在未参与选择的 12 张网络上是否保留 Node-Edge 局部响应关系。左侧分解 selection 与 unseen seeds 的同一目标函数，误差线为 unseen seeds 的 seed-level bootstrap 95% 区间；右侧分别比较 source 和 downstream slope。\n\n"
            "它只支持网络 seed 外推的描述性判断，不是患者盲检，也不证明 Node 与 Edge 是同一机制。\n\n"
            "**关注点**：unseen seeds 的 response loss 是否升高、区间是否过宽，以及两类位置是否仍沿 identity line 排列。\n")
        readme.write_text(text)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
