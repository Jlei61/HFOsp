"""Combined CORE model figure (user 2026-06-11): keep each propagation mode on ONE row.

Only the c/d electrode read-out is fused into the integrated long-time train; the a/b panels
(heterogeneity/lesion map + event-propagation map) are kept for both propagation modes.

Layout:
  A row = forward mode: a/b maps at left + fused electrode train at right.
  B row = reverse mode: a/b maps at left + fused electrode train at right.
  C row = real per-subject pipeline on the pooled model record.

Pure image composition (both upstream plotters untouched); re-run the upstream plotters first
(mechanism + train + model_propagation) if their PNGs changed. NO re-sim.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

D = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures"
W_IN = 18.0
COL_RATIOS = (1.0, 1.35)
# mechanism_4panel layout (src/sef_hfo_plot): a/b occupy the left ~half (axA x=0.030 .. caxB 0.483).
# Crop that strip (drop the suptitle band on top + the contact-note band at the bottom).
MAPS_X = (0.018, 0.500)
MAPS_Y = (0.095, 0.915)         # fraction from the TOP of the image


def _crop_maps(path):
    im = mpimg.imread(path)
    h, w = im.shape[:2]
    crop = im[int(MAPS_Y[0] * h):int(MAPS_Y[1] * h), int(MAPS_X[0] * w):int(MAPS_X[1] * w)].copy()
    # The source mechanism figure already has an internal panel "b" letter. In the
    # combined figure A/B are the row-level labels, so remove that redundant letter.
    ch, cw = crop.shape[:2]
    crop[:int(0.10 * ch), int(0.425 * cw):int(0.485 * cw), :3] = 1.0
    if crop.shape[2] == 4:
        crop[:int(0.10 * ch), int(0.425 * cw):int(0.485 * cw), 3] = 1.0
    return crop


def _row_height(left_im, right_im):
    usable_w = W_IN * 0.96
    left_w = usable_w * COL_RATIOS[0] / sum(COL_RATIOS)
    right_w = usable_w * COL_RATIOS[1] / sum(COL_RATIOS)
    return max(left_w * left_im.shape[0] / left_im.shape[1],
               right_w * right_im.shape[0] / right_im.shape[1])


def _full_width_height(im):
    return W_IN * 0.96 * im.shape[0] / im.shape[1]


def _show(ax, im, title=""):
    ax.imshow(im, aspect="auto")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=5, color="0.15")


def _row_letter(fig, ax, letter):
    bb = ax.get_position()
    fig.text(0.018, bb.y1 + 0.004, letter, fontsize=18, fontweight="bold",
             ha="left", va="bottom", color="0.05")


def compose(config):
    paths = {
        "maps_neg": f"{D}/mechanism_spont_{config}_neg.png",
        "train_neg": f"{D}/train_{config}_neg.png",
        "maps_pos": f"{D}/mechanism_spont_{config}_pos.png",
        "train_pos": f"{D}/train_{config}_pos.png",
        "pipeline": f"{D}/model_propagation/model_{config}_bidir_propagation.png",
    }
    for path in paths.values():
        if not os.path.exists(path):
            print(f"  (skip {config}: missing {os.path.basename(path)})"); return

    maps_neg = _crop_maps(paths["maps_neg"])
    train_neg = mpimg.imread(paths["train_neg"])
    maps_pos = _crop_maps(paths["maps_pos"])
    train_pos = mpimg.imread(paths["train_pos"])
    pipeline = mpimg.imread(paths["pipeline"])

    h_a = _row_height(maps_neg, train_neg)
    h_b = _row_height(maps_pos, train_pos)
    h_c = _full_width_height(pipeline)
    fig_h = h_a + h_b + h_c + 1.0
    fig = plt.figure(figsize=(W_IN, fig_h), facecolor="white")
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[h_a, h_b, h_c],
        width_ratios=COL_RATIOS,
        hspace=0.18,
        wspace=0.035,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _show(ax_a, maps_neg, "Forward mode: a/b maps (heterogeneity + event propagation)")
    ax = fig.add_subplot(gs[0, 1])
    _show(ax, train_neg, "Forward mode: fused electrode read-out (one locus across active contacts)")

    ax_b = fig.add_subplot(gs[1, 0])
    _show(ax_b, maps_pos, "Reverse mode: a/b maps (same substrate, reversed propagation)")
    ax = fig.add_subplot(gs[1, 1])
    _show(ax, train_pos, "Reverse mode: fused electrode read-out (same montage)")

    ax_c = fig.add_subplot(gs[2, :])
    _show(ax_c, pipeline)
    _row_letter(fig, ax_a, "A")
    _row_letter(fig, ax_b, "B")
    _row_letter(fig, ax_c, "C")

    out = f"{D}/core_model_{config}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for c in (sys.argv[1:] or ["stage2_main", "stage2_low_abnormality"]):
        compose(c)


if __name__ == "__main__":
    main()
