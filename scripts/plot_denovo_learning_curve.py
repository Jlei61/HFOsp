#!/usr/bin/env python3
"""Figure: de novo event-count learning curve.

Panel A: RANDOM-DRAW learning curves -- recovery of full-recording structure vs number of
         events drawn (from anywhere). Three recoveries (direction / axis line / endpoint set).
         Answers: how many events does each need, where does it saturate.
Panel B: REAL quiet windows vs the random-draw baseline at the SAME event count.
         If real falls BELOW random at matched M -> quiet periods are a different state,
         not just fewer events. Shows direction recovery (primary) + rate reference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
FIG = DIR / "figures"

C_DIR = "#4c72b0"    # direction (signed)
C_AXIS = "#dd8452"   # axis line (|rho|)
C_EP = "#2ca02c"     # endpoint set
C_RATE = "#c44e52"   # rate reference


def _series(d, key):
    Ms = sorted(int(m) for m in d)
    return np.array(Ms, float), np.array([d[str(m)][key] for m in Ms], float)


def main() -> int:
    o = json.loads((DIR / "cohort_learning_curve.json").read_text())
    rc = o["cohort_all"]["random_curve"]
    rb = o["cohort_all"]["real_binned"]

    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- Panel A: random-draw learning curves, three recoveries ---
    for key, c, lbl, mk in [("axis_signed", C_DIR, "direction (which end is source)", "o"),
                            ("axis_abs", C_AXIS, "axis line (which channels are extremes)", "s"),
                            ("endpoint", C_EP, "endpoint set (first+last to fire)", "^")]:
        M, y = _series(rc, key)
        axA.plot(M, y, mk + "-", color=c, label=lbl)
    axA.set_xscale("log")
    axA.set_xlabel("events drawn from the whole recording")
    axA.set_ylabel("recovery of full-recording structure\n(median across subjects)")
    axA.set_title("How many events to re-discover the structure?\n(random draw = ideal sampling, no state change)")
    axA.legend(frameon=False, fontsize=8.5, loc="lower right")
    axA.set_ylim(0.0, 1.02)
    axA.grid(alpha=0.25)

    # --- Panel B: real quiet windows vs random baseline (direction recovery) + rate ref ---
    M_r, y_rand = _series(rc, "axis_signed")
    M_b, y_real = _series(rb, "axis_signed")
    _, rate_rand = _series(rc, "rate_axis")
    _, rate_real = _series(rb, "rate_axis")

    axB.plot(M_r, y_rand, "o-", color=C_DIR, label="propagation direction — random draw")
    axB.plot(M_b, y_real, "o--", color=C_DIR, alpha=0.55,
             label="propagation direction — real quiet window")
    axB.plot(M_r, rate_rand, "^-", color=C_RATE, alpha=0.7, label="firing count — random draw")
    axB.plot(M_b, rate_real, "^--", color=C_RATE, alpha=0.4, label="firing count — real quiet window")
    # shade the state-change gap for propagation
    common_M = sorted(set(M_r.astype(int)) & set(M_b.astype(int)))
    axB.fill_between(common_M,
                     [rc[str(m)]["axis_signed"] for m in common_M],
                     [rb[str(m)]["axis_signed"] for m in common_M],
                     color=C_DIR, alpha=0.12)
    axB.set_xscale("log")
    axB.set_xlabel("events in window")
    axB.set_ylabel("direction recovery (signed Spearman vs full axis)")
    axB.set_title("Real quiet windows fall BELOW the random baseline\n(same #events → worse: quiet periods are a different state)")
    axB.legend(frameon=False, fontsize=8, loc="lower right")
    axB.set_ylim(0.0, 1.02)
    axB.grid(alpha=0.25)

    fig.tight_layout()
    out = FIG / "denovo_learning_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
