"""M4 aG16 BOUNDED-state GIF (Topic 4). The confirmed single-seed bounded sustained non-runaway state
(E1146 twoend_equal, k_q=0.10, alpha_G=16, seed 1, T=15000). Shows LEFT the spatial E-activity settling
into a sustained BROAD (~64% sheet) state that never fills/saturates, RIGHT the q_I(t) sheet-mean holding
above the 0.05 floor + the population rate holding below the 120 Hz runaway line -- for the full 15 s.

Framing discipline: this is a bounded BROAD high-activity state with sheet-MEAN q_I preserved above floor
(q_min still touches 0.05); it is NOT a localized ictal core. Supports "M4 can bound the q_I-depletion
runaway", not "spatially-localized seizure-like core".

Reads results/topic4_m4_dynamic_longconfirm/dynamic_qi_traces.npz. Output:
results/paper-ready-figure/fig_m4_dynamic_qi/figures/m4_dynamic_qi_bounded_aG16.gif (+ _final.png). res
rate_E is already Hz. Plotting-only.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results/topic4_m4_dynamic_longconfirm/dynamic_qi_traces.npz"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"
LBL = "kq0.10_aG16.0"
DT, MOVIE_BIN_MS, CORE_R = 0.1, 25.0, 1.5
FRAME_STRIDE = 8                                      # movie frames per GIF frame (8*25ms = 200ms)
RUNAWAY_HZ, Q_FLOOR = 120.0, 0.05


def main():
    d = np.load(NPZ, allow_pickle=True)
    L = float(d["L"]); src = d["src_xy"]; snk = d["snk_xy"]
    movie = d[LBL + "__movie"]; qI = d[LBL + "__trace_qI_mean"]; rate = d[LBL + "__rate"]
    nfr = movie.shape[0]
    t_q = np.arange(len(qI)) * DT
    OUT.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, nfr, FRAME_STRIDE))
    ext = [0, L, 0, L]

    def draw(k):
        tm = k * MOVIE_BIN_MS
        fig = plt.figure(figsize=(10.6, 4.7))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1, 1], wspace=0.28, hspace=0.32)
        # --- spatial activity ---
        axs = fig.add_subplot(gs[:, 0])
        axs.imshow(movie[k], origin="lower", extent=ext, cmap="viridis", vmin=0.0, vmax=0.7)
        for c, col in [(src, "#e8743b"), (snk, "#ffffff")]:
            axs.add_patch(Circle((c[0], c[1]), CORE_R, fill=False, ec=col, lw=1.6))
        axs.set_xticks([]); axs.set_yticks([])
        axs.set_title(f"E activity   t={tm:.0f} ms", fontsize=10)
        # --- q_I(t) ---
        aq = fig.add_subplot(gs[0, 1])
        aq.plot(t_q, qI, color="#1f6fb2", lw=1.3)
        aq.axhline(Q_FLOOR, color="0.5", lw=0.8, ls=":")
        aq.text(t_q[-1], Q_FLOOR + 0.02, "$q_{min}$ floor 0.05", ha="right", va="bottom", fontsize=7, color="0.5")
        aq.axvline(tm, color="k", lw=1.0)
        aq.set_xlim(0, t_q[-1]); aq.set_ylim(0, 1.02)
        aq.set_ylabel("sheet-mean $q_I$", fontsize=8.5); aq.tick_params(labelsize=7, labelbottom=False)
        aq.set_title("inhibitory resource holds above floor $\\Rightarrow$ no separatrix crossing",
                     fontsize=8.5)
        # --- rate(t) ---
        ar = fig.add_subplot(gs[1, 1], sharex=aq)
        ar.plot(t_q, rate, color="#c1272d", lw=0.5, alpha=0.8)
        ar.axhline(RUNAWAY_HZ, color="0.35", lw=0.9, ls="--")
        ar.text(t_q[-1], RUNAWAY_HZ + 8, "runaway 120 Hz", ha="right", va="bottom", fontsize=7, color="0.35")
        ar.axvline(tm, color="k", lw=1.0)
        ar.set_xlim(0, t_q[-1]); ar.set_ylim(0, max(200, float(rate.max()) * 1.05))
        ar.set_ylabel("rate (Hz/neuron)", fontsize=8.5); ar.set_xlabel("time (ms)", fontsize=8.5)
        ar.tick_params(labelsize=7)
        fig.suptitle("M4 CONFIRMED bounded state — $\\alpha_G$=16 (E1146, $k_q$=0.10, seed 1, 15 s): "
                     "bounded BROAD ~64% state, NOT a localized core", fontsize=10.5, y=0.99)
        fig.subplots_adjust(left=0.02, right=0.9, top=0.9, bottom=0.12)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    frames = [draw(k) for k in idx]
    frames.extend([frames[-1]] * 10)
    gif = OUT / "m4_dynamic_qi_bounded_aG16.gif"
    imageio.mimsave(gif, frames, duration=0.09, loop=0)
    imageio.imwrite(OUT / "m4_dynamic_qi_bounded_aG16_final.png", frames[-11])
    print(f"wrote {gif}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
