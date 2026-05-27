"""Per-subject figures for SEF-ITP H2b direction-axis phase.

Plan: docs/archive/topic4/sef_itp_direction_axis/phase_h2b_direction_axis_plan_2026-05-25.md

Each subject gets a 3-panel figure:
  Panel A — cos-similarity bars: cos(v_A, ±v_B) + permutation null distribution.
  Panel B — axis projection scatter: rank vs projection on u_A, separate lines for
            cluster A (trivially monotone +) and cluster B (discriminative slope).
  Panel C — 3D spatial map: source_A / sink_A / source_B / sink_B centroids and
            individual channels + SOZ overlay if available. Title carries the caveat
            "SEEG endpoint-axis direction (rank-derived), not UEA traveling-wave velocity".

Cohort figure (one combined): stacked-bar verdict distribution by swap_class.

Usage:
    python scripts/plot_sef_itp_direction_axis.py --subject epilepsiae_1146
    python scripts/plot_sef_itp_direction_axis.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.seeg_coord_loader import load_subject_coords

PER_SUBJECT_DIR = Path("results/topic4_sef_itp/direction_axis/per_subject")
FIG_DIR = Path("results/topic4_sef_itp/direction_axis/figures")
COHORT_JSON = Path("results/topic4_sef_itp/direction_axis/cohort_summary.json")

VERDICT_COLOR = {
    "axis_reversal": "#d6191b",
    "dual_source": "#1f78b4",
    "same_direction": "#33a02c",
    "inconclusive": "#999999",
    "degenerate_geometry": "#000000",
    "exit_no_universe": "#cccccc",
}
DESCRIPTIVE_COLOR = {
    "axis_reversal_shaped": "#d6191b",
    "dual_source_shaped": "#1f78b4",
    "same_direction_shaped": "#33a02c",
    "unclear": "#999999",
}


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p) as f:
        return json.load(f)


def _split_stem(stem: str):
    if stem.startswith("yuquan_"):
        return "yuquan", stem[len("yuquan_"):]
    if stem.startswith("epilepsiae_"):
        return "epilepsiae", stem[len("epilepsiae_"):]
    raise ValueError(f"unknown stem prefix: {stem!r}")


def plot_subject(stem: str, out_dir: Path = FIG_DIR) -> Optional[Path]:
    rec_path = PER_SUBJECT_DIR / f"{stem}.json"
    if not rec_path.exists():
        return None
    rec = _read_json(rec_path)
    verdict = (rec.get("verdict") or {}).get("label", "?")
    descriptive = (rec.get("descriptive_geometry") or {}).get("label", "?")
    swap_class = rec.get("swap_class", "?")
    decision_k = rec.get("decision_k_from_rank_displacement", "?")
    n_universe = rec.get("n_universe", 0)

    if rec.get("exit_reason") == "insufficient_universe" or rec.get("exit_reason") == "coord_load_failed":
        # Make a minimal exit card so the directory has 1:1 fig:subject coverage
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.axis("off")
        ax.set_title(f"{stem} — {verdict}\nSEEG endpoint-axis direction (rank-derived), not UEA velocity",
                     fontsize=10)
        ax.text(0.05, 0.5,
                f"exit_reason: {rec.get('exit_reason')}\n"
                f"swap_class={swap_class}, decision_k={decision_k}, n_universe={n_universe}\n"
                f"reason: {(rec.get('verdict') or {}).get('reason', '')}",
                fontsize=9, family="monospace", va="center")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path

    align = rec.get("axis_pair_alignment", {})
    slope_AA = rec.get("slope_A_on_axisA", {})
    slope_BA = rec.get("slope_B_on_axisA", {})
    deg = rec.get("degeneracy", {})
    ta = rec.get("template_A", {})
    tb = rec.get("template_B", {})

    cos_neg = align.get("cos_A_neg_B", float("nan"))
    cos_pos = align.get("cos_A_pos_B", float("nan"))
    null_med = align.get("null_cos_A_neg_B_median", float("nan"))
    null_95 = align.get("null_cos_A_neg_B_95", float("nan"))
    p_align = align.get("p_one_sided_axis_reversal", float("nan"))

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.4], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2], projection="3d")

    # Panel A — cos bars + null reference line
    bars = ax1.bar(["cos(v_A,+v_B)", "cos(v_A,-v_B)"], [cos_pos, cos_neg],
                   color=["#666666", VERDICT_COLOR.get(verdict, "#444444")],
                   edgecolor="black", linewidth=0.8)
    ax1.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax1.axhline(-0.5, color="grey", linestyle=":", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.5)
    if np.isfinite(null_med):
        ax1.axhline(null_med, color="#1f78b4", linestyle="--", linewidth=1.0,
                    label=f"null median={null_med:.2f}")
    if np.isfinite(null_95):
        ax1.axhline(null_95, color="#1f78b4", linestyle=":", linewidth=1.0,
                    label=f"null 95th={null_95:.2f}")
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("cos similarity")
    ax1.set_title(f"Axis pair alignment\n(perm p={p_align:.3f}, n_perm={align.get('n_perm_completed', 0)})",
                  fontsize=10)
    ax1.legend(fontsize=7, loc="lower left")
    for b, v in zip(bars, [cos_pos, cos_neg]):
        ax1.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.08),
                 f"{v:.3f}", ha="center", fontsize=9)

    # Panel B — axis projection
    # Reconstruct projection from coords + v_axis
    try:
        dataset, subject_id = _split_stem(stem)
        coords_result = load_subject_coords(dataset, subject_id, rec["channel_names"])
        coords = coords_result.coords_array_in_requested_order
        mapped = coords_result.mapped_mask_in_requested_order
    except Exception:
        coords = None
        mapped = None

    if coords is not None:
        v_A = np.asarray(ta["v_axis"], dtype=float)
        norm = float(np.linalg.norm(v_A))
        if norm > 1e-9:
            u_A = v_A / norm
            # joint_valid not stored in record; approximate via mapped + non-NaN coords
            # Use channels where coords mapped AND that appeared as endpoint (universe proxy)
            # Strictly we should pull joint_valid from rank_displacement; cheap path here.
            valid_for_plot = mapped & np.isfinite(coords).all(axis=1)
            proj = coords[valid_for_plot] @ u_A
            # We don't store full rank vectors in the record; rebuild from rank_displacement
            rd_path = Path("results/interictal_propagation_masked/rank_displacement/per_subject") / f"{stem}.json"
            if rd_path.exists():
                rd = _read_json(rd_path)
                pp = (rd.get("pairs") or [{}])[0]
                jv = np.asarray(pp["joint_valid"], dtype=bool)
                rank_a = np.asarray(pp["rank_a_dense_full"], dtype=float)
                rank_b = np.asarray(pp["rank_b_dense_full"], dtype=float)
                universe = jv & mapped
                xs = coords[universe] @ u_A
                ya = rank_a[universe]
                yb = rank_b[universe]
                ax2.scatter(xs, ya, s=44, c="#33a02c", alpha=0.85,
                            edgecolor="black", linewidth=0.5, label="cluster A rank")
                ax2.scatter(xs, yb, s=44, c="#d6191b", alpha=0.85, marker="s",
                            edgecolor="black", linewidth=0.5, label="cluster B rank")
                # regression lines
                if slope_AA.get("slope") is not None and np.isfinite(slope_AA["slope"]):
                    xl = np.linspace(xs.min(), xs.max(), 50)
                    ax2.plot(xl, slope_AA["intercept"] + slope_AA["slope"] * xl,
                             color="#33a02c", linewidth=1.2, alpha=0.6)
                if slope_BA.get("slope") is not None and np.isfinite(slope_BA["slope"]):
                    xl = np.linspace(xs.min(), xs.max(), 50)
                    ax2.plot(xl, slope_BA["intercept"] + slope_BA["slope"] * xl,
                             color="#d6191b", linewidth=1.2, alpha=0.6)
    ax2.set_xlabel("projection on $\\hat{u}_A$ (mm)")
    ax2.set_ylabel("dense rank")
    title2 = (f"Axis projection slopes  (B on axis_A: slope={slope_BA.get('slope', float('nan')):.3f}, "
              f"r²={slope_BA.get('r2', float('nan')):.2f}, p_neg={slope_BA.get('p_one_sided_neg_slope', float('nan')):.3f})")
    ax2.set_title(title2, fontsize=9)
    ax2.legend(fontsize=7, loc="best")
    ax2.axhline(0, color="grey", linewidth=0.3)

    # Panel C — 3D spatial map
    if coords is not None:
        src_a = ta.get("source_indices", [])
        snk_a = ta.get("sink_indices", [])
        src_b = tb.get("source_indices", [])
        snk_b = tb.get("sink_indices", [])
        all_mapped = mapped if mapped is not None else np.ones(coords.shape[0], dtype=bool)
        # other channels
        endpoint_set = set(src_a) | set(snk_a) | set(src_b) | set(snk_b)
        others = np.array([i for i in range(coords.shape[0]) if all_mapped[i] and i not in endpoint_set], dtype=int)
        if others.size:
            ax3.scatter(coords[others, 0], coords[others, 1], coords[others, 2],
                        s=12, c="#cccccc", alpha=0.5)
        if src_a:
            P = coords[np.array(src_a, dtype=int)]
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, c="#33a02c", marker="o",
                        edgecolor="black", linewidth=0.7, label="source A")
        if snk_a:
            P = coords[np.array(snk_a, dtype=int)]
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, c="#33a02c", marker="s",
                        edgecolor="black", linewidth=0.7, label="sink A")
        if src_b:
            P = coords[np.array(src_b, dtype=int)]
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, c="#d6191b", marker="o",
                        edgecolor="black", linewidth=0.7, label="source B")
        if snk_b:
            P = coords[np.array(snk_b, dtype=int)]
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], s=60, c="#d6191b", marker="s",
                        edgecolor="black", linewidth=0.7, label="sink B")
        # template axis vectors as arrows from centroid
        cA_src = np.asarray(ta.get("source_centroid", [0, 0, 0]))
        cA_snk = np.asarray(ta.get("sink_centroid", [0, 0, 0]))
        cB_src = np.asarray(tb.get("source_centroid", [0, 0, 0]))
        cB_snk = np.asarray(tb.get("sink_centroid", [0, 0, 0]))
        ax3.plot([cA_src[0], cA_snk[0]], [cA_src[1], cA_snk[1]], [cA_src[2], cA_snk[2]],
                 color="#33a02c", linewidth=2.5, alpha=0.75)
        ax3.plot([cB_src[0], cB_snk[0]], [cB_src[1], cB_snk[1]], [cB_src[2], cB_snk[2]],
                 color="#d6191b", linewidth=2.5, alpha=0.75)
        # SOZ overlay (if exists)
        soz_rel = rec.get("soz_relation") or {}
        if soz_rel.get("soz_centroid") is not None:
            c_SOZ = np.asarray(soz_rel["soz_centroid"], dtype=float)
            ax3.scatter([c_SOZ[0]], [c_SOZ[1]], [c_SOZ[2]], s=150, c="#fcae1e",
                        marker="*", edgecolor="black", linewidth=0.8,
                        label=f"SOZ centroid (n={soz_rel.get('n_soz_in_universe', 0)})")
        ax3.set_xlabel("x (mm)", fontsize=8)
        ax3.set_ylabel("y (mm)", fontsize=8)
        ax3.set_zlabel("z (mm)", fontsize=8)
        ax3.legend(fontsize=7, loc="upper left")
    ax3.set_title("Source/sink centroids and template axes\n(SEEG rank-derived direction, NOT UEA velocity)",
                  fontsize=9)

    fig.suptitle(
        f"{stem}  —  swap_class={swap_class}, decision_k={decision_k}, n_universe={n_universe}\n"
        f"Strict verdict: {verdict}  |  Descriptive geometry: {descriptive}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_cohort(out_dir: Path = FIG_DIR) -> Optional[Path]:
    if not COHORT_JSON.exists():
        return None
    coh = _read_json(COHORT_JSON)
    by_class = coh.get("verdict_by_swap_class") or {}
    desc_by_class = coh.get("descriptive_geometry_by_swap_class") or {}
    classes = ["strict", "candidate", "none", "inconclusive", "unknown"]
    classes = [c for c in classes if c in by_class or c in desc_by_class]
    verdict_labels = ["axis_reversal", "dual_source", "same_direction",
                      "inconclusive", "degenerate_geometry", "exit_no_universe"]
    desc_labels = ["axis_reversal_shaped", "dual_source_shaped",
                   "same_direction_shaped", "unclear"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, labels, by_c, color_map, title in [
        (axes[0], verdict_labels, by_class, VERDICT_COLOR, "Strict verdict by swap_class"),
        (axes[1], desc_labels, desc_by_class, DESCRIPTIVE_COLOR, "Descriptive geometry by swap_class"),
    ]:
        bottom = np.zeros(len(classes))
        x = np.arange(len(classes))
        for label in labels:
            counts = np.array([by_c.get(c, {}).get(label, 0) for c in classes], dtype=int)
            if counts.sum() == 0:
                continue
            ax.bar(x, counts, bottom=bottom, color=color_map.get(label, "#777"),
                   edgecolor="black", linewidth=0.5, label=label)
            for xi, ci, bi in zip(x, counts, bottom):
                if ci > 0:
                    ax.text(xi, bi + ci / 2, str(int(ci)), ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
            bottom += counts
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylabel("n subjects")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"SEF-ITP H2b direction-axis cohort summary (n={coh.get('n_subjects', '?')})\n"
        "Per-subject mechanism disambiguation — NOT a cohort claim. SEEG rank-derived direction.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cohort_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = ap.parse_args()

    if args.all and not args.subject:
        stems = sorted([p.stem for p in PER_SUBJECT_DIR.glob("*.json")])
    elif args.subject:
        stems = list(args.subject)
    else:
        stems = []
    for s in stems:
        out = plot_subject(s, out_dir=args.out_dir)
        print(f"{s}: {out}")
    if args.cohort or args.all:
        out = plot_cohort(out_dir=args.out_dir)
        if out:
            print(f"cohort: {out}")


if __name__ == "__main__":
    main()
