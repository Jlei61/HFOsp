"""FCXR-LC1 diagnostic figure (bounded-negative, per sprint §16 — NOT a paper-ready Figure 5).

Three panels tell the E4 mechanistic story on subject epilepsiae_1146 (seed 1):
  A  dynamic Z alone (q75): the system self-drives from interictal into a bounded, metastable dense-event
     oscillation (population rate stays ~10 Hz; inhibition depletion 1-z climbs as an event-locked staircase).
  B  Z + X at q75: X is inert -- the dense episodes self-terminate (Z recovers) too fast for the persistence
     sensor, so relay depletion 1-x stays ~0 and nothing changes.
  C  Z + X at q50: X ENGAGES and terminates the sustained bout -- relay depletion 1-x rises to ~0.5, the rate
     stays bounded (q50 alone runs away to ~450 Hz), but the onset is too fast (~1 s interictal) and Z does
     not recover, so there is no clean statistical recovery.

Loads the recorded traces (latest run of each kind). Pure post-processing; runs no simulation.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay", "lifecycle_closure")
FIGDIR = os.path.join(OUT, "figures")
RATE_C, DZ_C, DX_C = "#3b4b8a", "#c0504d", "#4e9a51"   # rate / inhibition-depletion / relay-depletion


def _latest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not hits:
        raise SystemExit(f"no trace matches {pattern}")
    return hits[-1]


def _rate_t(d):
    r = np.asarray(d["rate_E"], float)
    return np.arange(r.size) * float(d["rate_dt_ms"][0]) / 1000.0, r


def _panel(ax, t_r, rate, t_s, dz, dx, title):
    ax.fill_between(t_r, 0, rate, color=RATE_C, alpha=0.18, lw=0)
    ax.plot(t_r, rate, color=RATE_C, lw=0.7, label="population rate (Hz)")
    ax.set_ylabel("population rate (Hz)", color=RATE_C, fontsize=9)
    ax.tick_params(axis="y", labelcolor=RATE_C, labelsize=8)
    ax.set_ylim(0, max(15.0, float(np.percentile(rate, 99)) * 1.2))
    ax.set_xlim(0, t_r[-1])
    ax.set_title(title, fontsize=9.5, loc="left")
    ax2 = ax.twinx()
    ax2.plot(t_s, dz, color=DZ_C, lw=1.6, label=r"inhibition depletion  $1-\bar z$")
    if dx is not None:
        ax2.plot(t_s, dx, color=DX_C, lw=1.6, label=r"relay depletion  $1-\bar x$")
    ax2.set_ylabel("slow-variable depletion", fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis="y", labelsize=8)
    return ax2


def main():
    zonly = np.load(_latest(os.path.join(OUT, "runs", "*zonly_seed1_q75*", "zonly_traces.npz")))
    q75 = np.load(_latest(os.path.join(OUT, "runs", "*lifecycle_seed1_q75_xm0.1_td500*", "lifecycle_traces.npz")))
    q50 = np.load(_latest(os.path.join(OUT, "runs", "*lifecycle_seed1_q50_xm0.1*", "lifecycle_traces.npz")))

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2))
    fig.suptitle("Dynamic slow-feedback lifecycle — mechanism (epilepsiae 1146, seed 1)\n"
                 "the two allowed inhibition-failure rates bracket a clean single-bout lifecycle",
                 fontsize=11, y=0.985)

    tr, r = _rate_t(zonly)
    ax2a = _panel(axes[0], tr, r, np.asarray(zonly["DZ_t_ms"]) / 1000.0, np.asarray(zonly["DZ"]), None,
                  "A   Dynamic inhibition-failure alone (mild):  interictal → bounded metastable dense oscillation")

    tr, r = _rate_t(q75)
    _panel(axes[1], tr, r, np.asarray(q75["t_ms"]) / 1000.0, np.asarray(q75["D_Z"]), np.asarray(q75["D_X"]),
           "B   + relay resource (mild failure):  relay stays available (episodes self-terminate) → no bout to grab")

    tr, r = _rate_t(q50)
    ax2c = _panel(axes[2], tr, r, np.asarray(q50["t_ms"]) / 1000.0, np.asarray(q50["D_Z"]), np.asarray(q50["D_X"]),
                  "C   + relay resource (strong failure):  relay depletes and TERMINATES the sustained bout "
                  "(bounded) — but onset too fast, no recovery")
    axes[2].set_xlabel("time (s)", fontsize=9.5)

    # one shared legend (rate + both depletion lines), drawn outside the panels so nothing is occluded
    handles = [plt.Line2D([], [], color=RATE_C, lw=1.4), plt.Line2D([], [], color=DZ_C, lw=1.8),
               plt.Line2D([], [], color=DX_C, lw=1.8)]
    labels = ["population rate (Hz)", r"inhibition depletion  $1-\bar z$", r"relay depletion  $1-\bar x$"]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"lifecycle_diagnostic.{ext}"), dpi=170, bbox_inches="tight")
    print(f"wrote {FIGDIR}/lifecycle_diagnostic.png")


if __name__ == "__main__":
    main()
