"""Diagnostic A/B determination — render existing per-seizure PNGs in a
cluster-grouped grid so we can see whether algo's t_onset clusters align
with post-onset z-ER pattern (visible in atlas).

Per topic5 PR-1 plan v2 reviewer guidance:
  (A) algo cluster post-onset patterns are consistent within cluster
      → algo found a real subtype axis, just orthogonal to user's visual
  (B) algo cluster post-onset patterns are jumbled within cluster
      → t_onset clustering is noise; need feature upgrade to z-ER tensor

Usage::

  python scripts/diagnostic_cluster_grid.py --subject epilepsiae/442 --band gamma_ER
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.atlas_loading import LAYER_A_DIR  # noqa: E402

CLUSTER_OUT_DIR = LAYER_A_DIR / "seizure_clusters"
PER_SEIZURE_PNG_DIR = LAYER_A_DIR / "atlas_v2_3" / "figures" / "per_seizure"
DIAG_OUT_DIR = CLUSTER_OUT_DIR / "figures" / "diagnostic"


def render_diagnostic_grid(
    subject: str,
    band: str,
    cluster_json_path: Path,
    out_path: Path,
) -> Path:
    """Render a grid of per-seizure thumbnails grouped by algo cluster.

    Parameters
    ----------
    subject : "epilepsiae/442"
    band : "gamma_ER" or "broad_ER"
    cluster_json_path : path to topic5 PR-1 per-subject JSON
    """
    sid = subject.replace("/", "_")

    with open(cluster_json_path, "r") as fh:
        result = json.load(fh)
    bres = result["per_band"][band]
    if bres["status"] != "ok":
        raise ValueError(f"{subject}/{band} status={bres['status']}; cannot render")

    seizure_ids_kept = bres["seizure_ids_kept"]
    subtype_label = bres["subtype_label"]
    outlier_flag = bres["outlier_flag"]
    # Reconstruct seizure_idx for each kept (need per-subject JSON)
    from src.atlas_loading import load_per_subject_json
    src = "_sentinel" if not (LAYER_A_DIR / "per_subject" / f"{sid}.json").exists() else "per_subject"
    pj = load_per_subject_json(subject, source=src)
    recs = pj["per_er"][band]["seizure_records"]
    id_to_idx = {r["seizure_id"]: r["seizure_idx"] for r in recs}
    kept_idx = [id_to_idx[sid_] for sid_ in seizure_ids_kept]

    # Group by subtype label (-1 = outlier)
    groups: Dict[int, List[int]] = {}
    for i, lbl in enumerate(subtype_label):
        groups.setdefault(int(lbl), []).append(kept_idx[i])
    # Sort: subtypes by id, outliers (-1) last
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (kv[0] == -1, kv[0]),
    )

    # Build figure: each group has its own row-block (thumbnails wrap at
    # 5 per row inside the block). Each block separated by a band header.
    MAX_COLS_PER_BLOCK = 4
    THUMB_W, THUMB_H = 7.0, 4.5   # inches per thumbnail (so source PNG aspect preserved)

    blocks = []  # list of (row_label, row_color, idx_list, n_thumb_rows)
    total_thumb_rows = 0
    for lbl, idx_list in sorted_groups:
        n_rows_this = (len(idx_list) + MAX_COLS_PER_BLOCK - 1) // MAX_COLS_PER_BLOCK
        if lbl == -1:
            blocks.append((f"OUTLIERS (n={len(idx_list)})", "#7f8c8d",
                            idx_list, n_rows_this))
        else:
            blocks.append((f"subtype {lbl} (n={len(idx_list)})", "#2c3e50",
                            idx_list, n_rows_this))
        total_thumb_rows += n_rows_this

    fig_w = max(16.0, THUMB_W * MAX_COLS_PER_BLOCK + 1.0)
    fig_h = max(6.0, THUMB_H * total_thumb_rows + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(
        f"{subject} | {band} | algo subtypes  "
        f"(n_subtypes={bres['n_subtypes']}, n_outliers={bres['n_outliers']}, "
        f"silhouette={bres['silhouette_k']:.3f}, "
        f"gap_perm={bres['gap_perm_k']:.3f}, "
        f"chosen_k={bres['chosen_k']})",
        fontsize=15, y=0.998,
    )

    gs = fig.add_gridspec(total_thumb_rows, MAX_COLS_PER_BLOCK,
                           hspace=0.35, wspace=0.05,
                           left=0.04, right=0.99, top=0.965, bottom=0.01)

    cur_row = 0
    for row_label, row_color, idx_list, n_rows_this in blocks:
        for k_in_block, sz_idx in enumerate(idx_list):
            r = cur_row + (k_in_block // MAX_COLS_PER_BLOCK)
            c = k_in_block % MAX_COLS_PER_BLOCK
            ax = fig.add_subplot(gs[r, c])
            png = PER_SEIZURE_PNG_DIR / f"{sid}_seizure_{sz_idx:02d}.png"
            if not png.exists():
                ax.text(0.5, 0.5, f"sz_{sz_idx}\nno PNG",
                        transform=ax.transAxes, ha="center", va="center",
                        color="red", fontsize=9)
                ax.axis("off")
                continue
            try:
                img = mpimg.imread(str(png))
                ax.imshow(img)
                ax.axis("off")
                title_color = row_color
                ax.set_title(f"sz_idx={sz_idx}", fontsize=10,
                              color=title_color, pad=2)
            except Exception as exc:
                ax.text(0.5, 0.5, f"sz_{sz_idx}\nerr {exc}",
                        transform=ax.transAxes, ha="center", va="center",
                        color="red", fontsize=8)
                ax.axis("off")
        # Add a left-edge bracket label spanning the block's row range
        first_ax = fig.add_subplot(gs[cur_row, 0])
        first_ax.text(
            -0.06, 1.05, row_label,
            transform=first_ax.transAxes,
            fontsize=13, fontweight="bold",
            color=row_color, ha="left", va="bottom",
        )
        first_ax.axis("off")
        cur_row += n_rows_this

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--band", default="both",
                         choices=("gamma_ER", "broad_ER", "both"))
    parser.add_argument("--feature-mode", default="t_onset",
                         choices=("t_onset", "zer_binned"))
    args = parser.parse_args()

    sid = args.subject.replace("/", "_")
    suffix = "" if args.feature_mode == "t_onset" else "__zer_binned"
    cluster_json = CLUSTER_OUT_DIR / "per_subject" / f"{sid}{suffix}.json"
    if not cluster_json.exists():
        print(f"ERROR: cluster JSON missing — run cluster_ictal_seizures.py "
              f"per-subject --subject {args.subject} --feature-mode "
              f"{args.feature_mode} first", file=sys.stderr)
        return 1

    bands = ("gamma_ER", "broad_ER") if args.band == "both" else (args.band,)
    for band in bands:
        out = DIAG_OUT_DIR / f"{sid}_{band}{suffix}_cluster_grid.png"
        try:
            render_diagnostic_grid(args.subject, band, cluster_json, out)
            print(f"  wrote {out}")
        except Exception as exc:
            print(f"  [{band}] FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
