"""Combined CORE model figure (user 2026-06-11): the mechanism/simulation figure is the core
figure; only the c/d electrode read-out is FUSED into the integrated long-time train, while the
a/b panels (heterogeneity/lesion map + event-propagation map) are KEPT for BOTH propagation modes.
The real per-subject clustering (model_propagation) goes at the bottom.

Per mode (forward at −end, reverse at +end):
  maps row = a heterogeneity/lesion map + b event-propagation map (cropped from the mechanism
             4-panel's left half — a/b unchanged),
  train row = ∥+⊥ electrodes integrated into one panel (active electrodes only), long-time train.
Bottom = real per-subject pipeline on the pooled record (active electrodes) — KMeans k=2 → two
         opposite templates.

Pure image composition (both upstream plotters untouched); re-run the upstream plotters first
(mechanism + train + model_propagation) if their PNGs changed. NO re-sim.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

D = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures"
W_IN = 17.0
# mechanism_4panel layout (src/sef_hfo_plot): a/b occupy the left ~half (axA x=0.030 .. caxB 0.483).
# Crop that strip (drop the suptitle band on top + the contact-note band at the bottom).
MAPS_X = (0.018, 0.500)
MAPS_Y = (0.095, 0.915)         # fraction from the TOP of the image


def _crop_maps(path):
    im = mpimg.imread(path)
    h, w = im.shape[:2]
    return im[int(MAPS_Y[0] * h):int(MAPS_Y[1] * h), int(MAPS_X[0] * w):int(MAPS_X[1] * w)]


def compose(config):
    items = [
        (_crop_maps, f"{D}/mechanism_spont_{config}_neg.png",
         "A1 · MODEL forward (lesion −end) — heterogeneity/lesion map (left) + event-propagation map (right)"),
        (mpimg.imread, f"{D}/train_{config}_neg.png",
         "A2 · MODEL forward — electrode event TRAIN (∥+⊥ integrated, active electrodes only; one peak locus / event)"),
        (_crop_maps, f"{D}/mechanism_spont_{config}_pos.png",
         "B1 · MODEL reverse (lesion +end) — heterogeneity/lesion map + event-propagation map (direction reversed)"),
        (mpimg.imread, f"{D}/train_{config}_pos.png",
         "B2 · MODEL reverse — electrode event TRAIN"),
        (mpimg.imread, f"{D}/model_propagation/model_{config}_bidir_propagation.png",
         "C · REAL PER-SUBJECT PIPELINE on pooled events (active electrodes) — lagPat + KMeans k=2 → two opposite templates (inter-corr −0.95)"),
    ]
    imgs, titles = [], []
    for loader, path, ttl in items:
        if not os.path.exists(path):
            print(f"  (skip {config}: missing {os.path.basename(path)})"); return
        imgs.append(loader(path)); titles.append(ttl)

    pad = 0.05
    ratios = [im.shape[0] / im.shape[1] + pad for im in imgs]
    fig = plt.figure(figsize=(W_IN, W_IN * sum(ratios)), facecolor="white")
    gs = gridspec.GridSpec(len(imgs), 1, height_ratios=ratios, hspace=0.06)
    for i, (im, ttl) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im, aspect="auto"); ax.axis("off")
        ax.set_title(ttl, fontsize=11, fontweight="bold", loc="left", pad=4, color="0.15")
    out = f"{D}/core_model_{config}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for c in (sys.argv[1:] or ["stage2_main", "stage2_low_abnormality"]):
        compose(c)


if __name__ == "__main__":
    main()
