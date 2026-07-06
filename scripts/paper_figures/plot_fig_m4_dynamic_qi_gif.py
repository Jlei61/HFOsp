"""M4 dynamic-q_I runaway GIF (Topic 4). The one-shot runaway spreads spatially from the source core to
the whole sheet; the shared divisive pool S_G brakes the intensity. Animates the spatial activity field
(viridis) for no_pool vs pool (k_q=0.35) over the first ~1500 ms, foci overlaid, per-frame time + rate.

Reads results/topic4_m4_dynamic/dynamic_qi_traces.npz (the `movie` = per-25ms-frame E-active fraction on a
24x24 grid). Output: results/paper-ready-figure/fig_m4_dynamic_qi/figures/m4_dynamic_qi_runaway.gif (+ _final.png).
Plotting-only. Follows the imageio.mimsave pattern of plot_fig_m3a_v2_2_qI_stim_runaway_gif.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/topic4_m4_dynamic/dynamic_qi_traces.npz"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"

NE, DT = 32000, 0.1
MOVIE_BIN_MS = 25.0
CORE_R = 1.5
N_FRAMES = 32                                        # 0..800 ms (quiescent -> ignition ~350 -> whole-field spread)
ARMS = [("kq0.35_no_pool", "no pool  ($\\alpha_G$=0)"), ("kq0.35_pool_aG6", "pool  ($\\alpha_G$=6)")]


def _rate_hz_in_frame(rate, fi):
    bs = int(round(MOVIE_BIN_MS / DT))
    seg = rate[fi * bs:(fi + 1) * bs]
    return float(seg.mean()) / NE / DT * 1e3 if seg.size else 0.0


def main():
    d = np.load(NPZ, allow_pickle=True)
    L = float(d["L"]); src = d["src_xy"]; snk = d["snk_xy"]
    G = d[f"{ARMS[0][0]}__movie"].shape[1]
    data = {a: dict(movie=d[f"{a}__movie"], rate=d[f"{a}__rate"]) for a, _ in ARMS}
    OUT.mkdir(parents=True, exist_ok=True)
    ext = [0, L, 0, L]

    def draw(fi):
        fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.9))
        for ax, (a, title) in zip(axs, ARMS):
            ax.imshow(data[a]["movie"][fi], origin="lower", extent=ext, cmap="viridis", vmin=0.0, vmax=0.6)
            for c, col in [(src, "#e8743b"), (snk, "#ffffff")]:
                ax.add_patch(Circle((c[0], c[1]), CORE_R, fill=False, ec=col, lw=1.6))
            hz = _rate_hz_in_frame(data[a]["rate"], fi)
            ax.set_title(f"{title}\n{hz:.0f} Hz/neuron", fontsize=10)
            ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"M4 dynamic $q_I$ — spatial activity   t = {fi * MOVIE_BIN_MS:.0f} ms   (E1146, L=20)",
                     fontsize=11.5, y=1.00)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    frames = [draw(fi) for fi in range(min(N_FRAMES, data[ARMS[0][0]]["movie"].shape[0]))]
    frames.extend([frames[-1]] * 8)                  # hold the final frame
    gif = OUT / "m4_dynamic_qi_runaway.gif"
    imageio.mimsave(gif, frames, duration=0.10, loop=0)
    imageio.imwrite(OUT / "m4_dynamic_qi_runaway_final.png", frames[-9])
    print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
