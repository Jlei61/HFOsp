"""Kick-amplitude robustness figure (2026-06-09). One 2-panel figure, each a
distinct question (CLAUDE.md §7):
  a  matched (variance-only, mean=18) evoked co-activation vs kick amplitude —
     is the matched NULL robust to a stronger stimulus? (flat ~0 = robust)
  b  sanity: whole-net evoked peak grows with amplitude (kick does something) +
     baseline never self-ignites; pre-kick-invariant flag annotated.

Read-only over kick_amp_metrics.json — no re-run.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"


def main():
    d = json.loads((OUT / "kick_amp_metrics.json").read_text())
    amps = [f"{a:.0f}x" for a in d["amps"]]
    x = np.arange(len(amps))
    ag = d["aggregate"]
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # a — matched evoked synchrony vs amplitude
    ys = [ag[a]["matched_d_core_paf"]["mean"] if ag[a]["matched_d_core_paf"] else np.nan for a in amps]
    es = [ag[a]["matched_d_core_paf"]["sem"] if ag[a]["matched_d_core_paf"] else 0.0 for a in amps]
    ns = [ag[a]["matched_d_core_paf"]["n"] if ag[a]["matched_d_core_paf"] else 0 for a in amps]
    axA.axhline(0, color="k", lw=0.9)
    axA.errorbar(x, ys, yerr=es, fmt="o-", color="steelblue", lw=1.8, ms=8, capsize=5)
    for xi, yi, ni in zip(x, ys, ns):
        if np.isfinite(yi):
            axA.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 10), fontsize=7, ha="center")
    axA.set_xticks(x); axA.set_xticklabels([f"{a}·ν_θ" for a in amps])
    axA.set_xlabel("kick amplitude (× locked operating kick)")
    axA.set_ylabel("matched Δ core co-activation\n(variance-only − healthy)")
    rng = max(0.12, max(abs(np.nanmin(ys)), abs(np.nanmax(ys))) * 1.6) if any(np.isfinite(ys)) else 0.12
    axA.set_ylim(-rng, rng)
    axA.set_title("a · matched (variance-only) evoked effect vs stimulus strength\n"
                  "null at locked (2×) + stronger (3×); weak (1×) = wide tail helps recruitment",
                  fontsize=10)
    axA.grid(axis="y", alpha=0.3)

    # b — sanity: evoked peak grows; baseline never ignites
    peak = [ag[a]["base_whole_paf"]["mean"] for a in amps]
    pe = [ag[a]["base_whole_paf"]["sem"] for a in amps]
    bign = [ag[a]["base_ignition_rate"] for a in amps]
    axB.errorbar(x, peak, yerr=pe, fmt="s-", color="seagreen", lw=1.8, ms=8, capsize=5,
                 label="whole-net evoked peak (substantial at all amps)")
    axB.set_xticks(x); axB.set_xticklabels([f"{a}·ν_θ" for a in amps])
    axB.set_xlabel("kick amplitude (× locked operating kick)")
    axB.set_ylabel("whole-net peak active fraction\n(evoked window)")
    pinv = d.get("prekick_invariant")
    axB.set_title(f"b · sanity: kick does something + baseline stays quiet\n"
                  f"baseline ignition rate = {bign} · pre-kick-invariant: {pinv}", fontsize=10)
    axB.legend(fontsize=8); axB.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Kick-amplitude robustness: matched-null vs stimulus strength "
                 f"({len(d['conn_seeds'])} networks × {len(d['fn_seeds'])} field-noise, mid core + end kick)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "kick_amp.png", dpi=140); plt.close(fig)
    print("wrote kick_amp.png")


if __name__ == "__main__":
    main()
