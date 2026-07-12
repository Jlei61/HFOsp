"""Topic5 contact-similarity ladder — cohort visualisation.

Reads  results/topic5_ictal_recruitment/contact_similarity/cohort_summary_{activation}.json
Writes results/topic5_ictal_recruitment/contact_similarity/figures/*.png

Four independent panels (CLAUDE.md §7 — one question per panel):
  A  per-subject grouped bars  R1/R2/R3 obs  + per-rung null p95 tick
     Q: do all three similarity rungs agree per subject?
  B  per-subject delta bars — smooth_delta (R1→R2, same-plane smoothing) + grid_delta (R2→R3, grid)
     Q: how much does in-plane Gaussian smoothing (R1→R2) vs grid
        interpolation (R2→R3) each contribute?
  C  σ-sweep  R2 obs vs bandwidth multiplier {0.5×, 1×, 2×}
     Q: is the same-plane smoothing claim robust to bandwidth choice?
  D  sequence-sanity (small)  Spearman + Kendall obs vs null p95
     Q: does plain rank-order correlation (no geometry) also see a signal?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
# --out-dir is the ladder's namespaced dir (matches axis_alignment convention: one
# dir, filenames carry the activation); figures land in <out-dir>/figures/.
_DEF_OUT_DIR = "results/topic5_ictal_recruitment/contact_similarity"


def _default_summary_path(out_dir, activation):
    """Default cohort_summary path for (out_dir, activation) — must match the
    runner's `cohort_summary_{activation}.json` naming
    (scripts/run_topic5_contact_similarity.py:350)."""
    return Path(out_dir) / f"cohort_summary_{activation}.json"


RUNG_LABELS = {"R1": "R1 (raw Pearson)", "R2": "R2 (in-plane smoothed)", "R3": "R3 (grid field)"}
RUNG_COLORS = {"R1": "#4575b4", "R2": "#fdae61", "R3": "#d73027"}
SIGMA_COLORS = {0.5: "#1b7837", 1.0: "#762a83", 2.0: "#c51b7d"}
SEQ_COLORS = {"spearman": "#3182bd", "kendall": "#de2d26"}


def _safe(d, *keys, default=np.nan):
    """Nested dict access with NaN fallback."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return d if (d is not None and not isinstance(d, dict)) else default


def load_subjects(summary_path: Path):
    """Return list of ok-status subject dicts."""
    data = json.load(open(summary_path))
    return [s for s in data.get("per_subject", []) if s.get("status") == "ok"]


# ---------------------------------------------------------------------------
# Panel A — per-subject grouped bars
# ---------------------------------------------------------------------------
def panel_A(ax, subjects):
    """Grouped bar: R1/R2/R3 obs per subject, null p95 horizontal tick overlay."""
    n = len(subjects)
    if n == 0:
        ax.text(0.5, 0.5, "No ok subjects", ha="center", va="center", transform=ax.transAxes)
        return

    rungs = ["R1", "R2", "R3"]
    bar_w = 0.24
    group_offsets = np.array([-1, 0, 1]) * bar_w
    x = np.arange(n)

    for i, rung in enumerate(rungs):
        obs = [_safe(s, rung, "within_shaft", "obs_subject") for s in subjects]
        p95 = [_safe(s, rung, "within_shaft", "null_q", "p95") for s in subjects]
        bars = ax.bar(x + group_offsets[i], obs, width=bar_w * 0.85,
                      color=RUNG_COLORS[rung], alpha=0.8, label=RUNG_LABELS[rung], zorder=3)
        # null p95 tick: horizontal line across each bar
        for xi, (xb, p) in enumerate(zip(x + group_offsets[i], p95)):
            if np.isfinite(p):
                ax.plot([xb - bar_w * 0.4, xb + bar_w * 0.4], [p, p],
                        color="black", lw=1.5, zorder=4)

    labels = [s["subject_id"].replace("epilepsiae_", "E").replace("yuquan_", "Y")
              for s in subjects]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Median|corr| (within-shaft null, obs)", fontsize=9)
    ax.set_title("A  Per-subject similarity: raw → smoothed → grid", fontsize=9, fontweight="bold")
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)
    # horizontal tick legend note
    ax.plot([], [], color="black", lw=1.5, label="Null p95 threshold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    # tight y
    all_obs = [_safe(s, r, "within_shaft", "obs_subject") for s in subjects for r in rungs]
    all_p95 = [_safe(s, r, "within_shaft", "null_q", "p95") for s in subjects for r in rungs]
    vals = [v for v in all_obs + all_p95 if np.isfinite(v)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.15 + 0.02
        ax.set_ylim(max(0, lo - pad), min(1.05, hi + pad))


# ---------------------------------------------------------------------------
# Panel B — per-step delta plot
# ---------------------------------------------------------------------------
def panel_B(ax, subjects):
    """Grouped bars: smooth_delta (R1→R2) and grid_delta (R2→R3) per subject.

    Answers: how much does each geometry step move the similarity?  Distinct from
    Panel A's absolute-height-vs-null view (CLAUDE.md §7 — one question per panel).
    """
    n = len(subjects)
    if n == 0:
        ax.text(0.5, 0.5, "No ok subjects", ha="center", va="center", transform=ax.transAxes)
        return

    labels = [s["subject_id"].replace("epilepsiae_", "E").replace("yuquan_", "Y")
              for s in subjects]
    x = np.arange(n)
    bar_w = 0.35

    smooth_vals, grid_vals = [], []
    for s in subjects:
        sd = _safe(s, "smooth_delta")
        if not np.isfinite(sd):  # robustness fallback: compute from R1/R2 obs
            sd = (_safe(s, "R2", "within_shaft", "obs_subject")
                  - _safe(s, "R1", "within_shaft", "obs_subject"))
        gd = _safe(s, "grid_delta")
        if not np.isfinite(gd):  # robustness fallback: compute from R2/R3 obs
            gd = (_safe(s, "R3", "within_shaft", "obs_subject")
                  - _safe(s, "R2", "within_shaft", "obs_subject"))
        smooth_vals.append(sd)
        grid_vals.append(gd)

    ax.bar(x - bar_w / 2, smooth_vals, width=bar_w, color="#fdae61", alpha=0.85,
           label="same-plane smoothing (R1→R2)", zorder=3)
    ax.bar(x + bar_w / 2, grid_vals, width=bar_w, color="#d73027", alpha=0.85,
           label="grid (R2→R3)", zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Δ similarity (contribution of each geometry step)", fontsize=9)
    ax.set_title("B  Per-subject geometry step contributions (R1→R2 vs R2→R3)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)

    all_vals = [v for v in smooth_vals + grid_vals if np.isfinite(v)]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.15, 0.05)
        ax.set_ylim(lo - pad, hi + pad)


# ---------------------------------------------------------------------------
# Panel C — σ-sweep
# ---------------------------------------------------------------------------
def panel_C(ax, subjects):
    """R2 obs at σ×{0.5, 1.0, 2.0}; one line per subject."""
    keys = ["0.5x", "1.0x", "2.0x"]
    x_vals = [0.5, 1.0, 2.0]
    labels_x = ["0.5×σ", "1×σ (canonical)", "2×σ"]

    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(subjects))]

    for si, s in enumerate(subjects):
        obs = [_safe(s, "R2_sigma_sweep", k, "obs_subject") for k in keys]
        subj_label = s["subject_id"].replace("epilepsiae_", "E").replace("yuquan_", "Y")
        finite = [(x, o) for x, o in zip(x_vals, obs) if np.isfinite(o)]
        if len(finite) < 2:
            continue
        xs, ys = zip(*finite)
        ax.plot(xs, ys, "o-", color=colors[si], alpha=0.8, linewidth=1.5,
                markersize=5, label=subj_label)

    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels_x, fontsize=8)
    ax.set_xlabel("Gaussian bandwidth multiplier (applied to canonical σ)", fontsize=9)
    ax.set_ylabel("R2 obs (within-shaft null)", fontsize=9)
    ax.set_title("C  Bandwidth robustness: in-plane smoothed similarity (R2) across σ-sweep",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7, ncol=2)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    all_obs = [_safe(s, "R2_sigma_sweep", k, "obs_subject") for s in subjects for k in keys]
    vals = [v for v in all_obs if np.isfinite(v)]
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.15 + 0.02
        ax.set_ylim(max(0, lo - pad), min(1.05, hi + pad))
    ax.set_xlim(0.3, 2.2)


# ---------------------------------------------------------------------------
# Panel D — sequence-sanity (small)
# ---------------------------------------------------------------------------
def panel_D(ax, subjects):
    """Scatter Spearman + Kendall obs vs null p95 — are rank-order stats above chance?"""
    methods = ["spearman", "kendall"]
    method_labels = {"spearman": "Spearman ρ", "kendall": "Kendall τ"}
    markers = {"spearman": "o", "kendall": "s"}

    all_vals = []
    for method in methods:
        obs_list = [_safe(s, "sequence", method, "obs_subject") for s in subjects]
        p95_list = [_safe(s, "sequence", method, "null_q", "p95") for s in subjects]
        for ob, p in zip(obs_list, p95_list):
            if np.isfinite(ob) and np.isfinite(p):
                all_vals.extend([ob, p])
                ax.scatter(p, ob, color=SEQ_COLORS[method],
                           marker=markers[method], s=40, alpha=0.8,
                           label=method_labels[method])

    # identity line
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="obs = null p95")

    # remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen, unique = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    if unique:
        ax.legend(*zip(*unique), fontsize=7, loc="lower right", framealpha=0.7)

    ax.set_xlabel("Null p95 threshold", fontsize=9)
    ax.set_ylabel("Observed value (within-shaft)", fontsize=9)
    ax.set_title("D  Sequence sanity: rank-order correlation", fontsize=9, fontweight="bold")
    ax.grid(linestyle=":", linewidth=0.6, alpha=0.6)

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = (hi - lo) * 0.12 + 0.02
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")


# ---------------------------------------------------------------------------
# main figure
# ---------------------------------------------------------------------------
def make_figure(summary_path: Path, out_dir: Path, activation: str = "broadband"):
    subjects = load_subjects(summary_path)
    if not subjects:
        print(f"[plot] No ok subjects in {summary_path}; nothing to plot.")
        return

    act_label = activation.replace("broadband", "broadband (80–500 Hz)")
    n_ok = len(subjects)
    print(f"[plot] {n_ok} ok subjects | activation={activation}")

    fig = plt.figure(figsize=(14, 10))
    # 2×2 grid, panel D slightly smaller
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.07)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    panel_A(ax_A, subjects)
    panel_B(ax_B, subjects)
    panel_C(ax_C, subjects)
    panel_D(ax_D, subjects)

    fig.suptitle(
        f"Contact-similarity ladder · {act_label} · n={n_ok} subjects\n"
        "Within-shaft null (B=permutation draws); Panel A horizontal tick = null p95",
        fontsize=10, y=0.97
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"contact_similarity_{activation}_cohort.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=["broadband", "hfa"], default="broadband")
    ap.add_argument("--summary", default=None,
                    help="override path to cohort_summary_{activation}.json "
                         "(default: <out-dir>/cohort_summary_{activation}.json)")
    ap.add_argument("--out-dir", default=_DEF_OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary) if args.summary else _default_summary_path(out_dir, args.activation)
    if not summary_path.exists():
        raise FileNotFoundError(f"cohort_summary_{args.activation}.json not found: {summary_path}\n"
                                "Run scripts/run_topic5_contact_similarity.py first.")

    make_figure(summary_path, out_dir / "figures", args.activation)


if __name__ == "__main__":
    main()
