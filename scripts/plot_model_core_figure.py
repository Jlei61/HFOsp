"""Combined CORE model figure (user 2026-06-11): the mechanism/simulation figure is the core
figure; the real-pipeline per-subject clustering (model_propagation) goes BELOW it as one figure.

Top rows  = model simulation (mechanism 4-panel): lesion/heterogeneity map · propagation map ·
            ∥/⊥ electrode read-out — forward (lesion at −end) then reverse (lesion at +end).
Bottom row = real per-subject pipeline on the POOLED record: lagPat heatmap + KMeans k=2 → two
            opposite templates (the same figure real patients get; instrument parity).

Pure image composition — stacks the already-rendered PNGs (mechanism_spont_<config>_{neg,pos}.png +
model_propagation/model_<config>_bidir_propagation.png) at a common width, no distortion. Re-run the
upstream plotters first if their PNGs changed. NO re-sim.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

D = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/figures"
W_IN = 17.0   # common width (inches)


def compose(config):
    panels = [
        (f"{D}/train_{config}_neg.png",
         "A · MODEL — forward event TRAIN (lesion at −end): ∥+⊥ electrodes (active only, one panel), "
         "∥ peaks slant = direction, repeats event-to-event"),
        (f"{D}/train_{config}_pos.png",
         "B · MODEL — reverse event TRAIN (lesion at +end): same electrodes, peaks slant the other way"),
        (f"{D}/model_propagation/model_{config}_bidir_propagation.png",
         "C · REAL PER-SUBJECT PIPELINE on pooled events (active electrodes) — lagPat + KMeans k=2 → "
         "two opposite templates (inter-corr −0.95, forward/reverse pair; model: labelled)"),
    ]
    imgs, aspects, titles = [], [], []
    for path, ttl in panels:
        if not os.path.exists(path):
            print(f"  (skip {config}: missing {os.path.basename(path)})"); return
        im = mpimg.imread(path)
        imgs.append(im); aspects.append(im.shape[0] / im.shape[1]); titles.append(ttl)

    pad = 0.045                                    # header strip per panel (fraction of width)
    ratios = [a + pad for a in aspects]
    fig_h = W_IN * sum(ratios)
    fig = plt.figure(figsize=(W_IN, fig_h), facecolor="white")
    gs = gridspec.GridSpec(len(imgs), 1, height_ratios=ratios, hspace=0.04)
    for i, (im, ttl) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im, aspect="auto"); ax.axis("off")
        ax.set_title(ttl, fontsize=11, fontweight="bold", loc="left", pad=4, color="0.15")
    out = f"{D}/core_model_{config}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    configs = sys.argv[1:] or ["stage2_main", "stage2_low_abnormality"]
    for c in configs:
        compose(c)


if __name__ == "__main__":
    main()
