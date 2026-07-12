"""M4-2 P1 mechanism figure (supplementary): slow-variable trajectories for representative
persist / fragment / suppress cells, so the mechanism is shown, not asserted. Top row = activity
envelope; bottom row = sheet-mean q_I + E->E STD availability x_dep (mean/min). q_min=0.05 reference.

What the trajectories show: fragment = x_dep recovers fast (tau=1000) between bursts -> recurrent drive
returns -> re-ignition (q_I is still draining, not pinned); suppress = x_dep stays depleted -> activity
killed -> q_I never drains (stays high). Neither yields one sustained event + clean offset.
Seed=1, k_q=0.10, alpha_G=16. S_G (pool output) is not saved in the sweep npz; this figure shows
activity + q_I + x_dep (3 of the 4 slow signals), enough for the recovery-timescale reading."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "topic4_m4_dynamic_p1_sweep", "figures")
DT_MS = 0.1
Q_MIN = 0.05

z = np.load(os.path.join(RES, "topic4_m4_dynamic_p1_sweep", "p1_sweep_traces.npz"))
cells = [("p1_arm0", "persist  (u=0, no STD)", "#4C6EDB"),
         ("p1_u0.15_tau1000", "fragment  (u=0.15, tau=1000)", "#E8833A"),
         ("p1_u0.5_tau5000", "suppress  (u=0.5, tau=5000)", "#8A8F98")]


def smooth(a, k):
    n = (len(a) // k) * k
    return a[:n].reshape(-1, k).mean(1)


fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True)
for j, (lbl, name, col) in enumerate(cells):
    af = np.asarray(z[f"{lbl}__af"], float)          # active fraction, 1ms bins
    qi = np.asarray(z[f"{lbl}__qI"], float)          # sheet-mean q_I, per-step (dt=0.1ms)
    xm = np.asarray(z[f"{lbl}__xdep_mean"], float)   # x_dep mean, per-step
    xn = np.asarray(z[f"{lbl}__xdep_min"], float)    # x_dep min, per-step
    t_af = np.arange(len(af)) / 1000.0 * (1.0)       # af bin = 1ms -> s ; len 15000 -> 15s
    ka = 100                                          # 100ms envelope for activity
    t_ae = smooth(t_af, ka); ae = smooth(af, ka)
    ks = 2000                                         # 200ms for slow vars (per-step dt=0.1ms -> 2000 steps)
    t_s = np.arange(len(qi)) * DT_MS / 1000.0
    t_se = smooth(t_s, ks); qie = smooth(qi, ks); xme = smooth(xm, ks); xne = smooth(xn, ks)

    ax0 = axes[0, j]
    ax0.plot(t_ae, ae, color=col, lw=1.4)
    ax0.set_title(name, fontsize=10)
    ax0.set_ylim(0, max(0.02, ae.max() * 1.15))
    if j == 0:
        ax0.set_ylabel("activity\n(active fraction, 100ms)")
    ax0.spines[["top", "right"]].set_visible(False)

    ax1 = axes[1, j]
    ax1.plot(t_se, qie, color="#2E7D32", lw=1.6, label="q_I (sheet mean)")
    ax1.axhline(Q_MIN, color="#2E7D32", ls=":", lw=1, alpha=0.7)
    ax1.plot(t_se, xme, color="#B23A48", lw=1.4, label="x_dep mean")
    ax1.plot(t_se, xne, color="#B23A48", lw=1.0, ls="--", label="x_dep min")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("time (s)")
    if j == 0:
        ax1.set_ylabel("slow vars [0,1]")
        ax1.legend(fontsize=7.5, frameon=False, loc="center right")
    ax1.spines[["top", "right"]].set_visible(False)

axes[1, 0].text(0.02, Q_MIN + 0.03, "q_min", color="#2E7D32", fontsize=7, transform=axes[1, 0].get_yaxis_transform())
fig.suptitle("M4-2 P1 mechanism (seed=1): fast STD (x_dep) recovery re-drives recurrent bursts (fragment); "
             "sustained depletion kills activity (suppress); q_I drains but no single sustained event + clean offset",
             fontsize=9.5, y=1.0)
fig.tight_layout()
out = os.path.join(OUT, "m4_2_p1_mechanism.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
