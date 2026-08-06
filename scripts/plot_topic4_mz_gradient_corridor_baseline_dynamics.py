"""Baseline-dynamics diagnostic for MZ gradient-corridor stimulation.

Answers ONE QC question: does the frozen Z+M candidate, on each patient's gradient-mapped substrate,
produce the assumed "recoverable interictal events -> distinct late runaway" trajectory, or something
else (slow monotonic ramp / no runaway / immediate high firing)? This is the eligibility diagnostic that
gates the site comparison; for a NO-GO it is the cohort-level finding.

One row per subject: population E-rate (Hz) + active-fraction with the frozen event bar + mean z, over
time, with the runaway time, the [45%,75%] stim window, and eligibility annotated. Reads the saved
baseline_no_stim.{json,npz} per subject-seed.
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_gradient_corridor_stimulation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--seed", default="1")
    args = ap.parse_args()
    per = os.path.join(args.res, "per_run")
    figdir = os.path.join(args.res, "figures")
    os.makedirs(figdir, exist_ok=True)
    subs = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                   for p in glob.glob(os.path.join(per, "*", args.seed, "baseline_no_stim.json"))})
    if not subs:
        print("no baseline artifacts yet")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(subs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.5 * n), squeeze=False)
    for i, s in enumerate(subs):
        ax = axes[i][0]
        rec = json.load(open(os.path.join(per, s, args.seed, "baseline_no_stim.json")))
        d = np.load(os.path.join(per, s, args.seed, "baseline_no_stim.npz"))
        v = rec.get("baseline_verdict", {})
        T = rec["n_steps"] * rec["dt"]
        rate = d["rate"]
        af = d["active_frac"]
        z = d["z_mean"]
        tr = np.linspace(0, T, len(rate))
        ta = np.linspace(0, T, len(af))
        tz = np.linspace(0, T, len(z))
        ax.plot(tr, rate, color="#333", lw=0.7, label="E-rate (Hz)")
        ax.axhline(120, color="#e63946", lw=0.8, ls=":", label="120 Hz runaway")
        ax2 = ax.twinx()
        ax2.plot(ta, af, color="#4c78a8", lw=0.7, alpha=0.8, label="active frac")
        bar = v.get("frozen_event_bar")
        if bar:
            ax2.axhline(bar, color="#4c78a8", lw=0.7, ls="--", alpha=0.7)
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.06))
        ax3.plot(tz, z, color="#2a9d8f", lw=0.9, alpha=0.7)
        ax3.set_ylabel("mean z", color="#2a9d8f", fontsize=8)
        ax3.tick_params(axis="y", labelcolor="#2a9d8f", labelsize=7)
        ra = v.get("baseline_runaway_ms")
        if ra:
            ax.axvline(ra, color="#e63946", lw=1.2)
        on, off = v.get("stim_on_ms"), v.get("stim_off_ms")
        if on and off:
            ax.axvspan(on, off, color="#f4a261", alpha=0.25)
        ax.set_ylabel("E-rate (Hz)", fontsize=8)
        ax2.set_ylabel("active frac", color="#4c78a8", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#4c78a8", labelsize=7)
        elig = v.get("eligible")
        ax.set_title(f"{s}  runaway={ra:.0f}ms  eligible={elig}  ({v.get('reason') or 'ok'})  "
                     f"n_pre_recover={v.get('n_pre_stim_recoverable')}", loc="left", fontsize=9)
        ax.set_xlim(0, T)
        if i == n - 1:
            ax.set_xlabel("time (ms)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, f"baseline_dynamics_seed{args.seed}.{ext}"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {figdir}/baseline_dynamics_seed{args.seed}.png")


if __name__ == "__main__":
    main()
