"""Core-size axis figure (autonomous exploration C, 2026-06-09). 2 panels, each a
distinct question (CLAUDE.md §7):
  a  ignition probability vs core mean for 3 core radii (wide solid / narrow dashed)
     — does a BIGGER core ignite at a HIGHER mean (boundary lowers)? mean=18 wide
     points circled = the GUARD (must be ~0; else the clean reference is gone).
  b  ignition boundary (mean @ 0.5) vs core size (n_core) — the summary trend, with
     the guard verdict in the title.

Read-only over core_size_metrics.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/topic4_sef_hfo/snn_heterogeneity")
FIG = OUT / "figures"
RCOLOR = {0.3: "#4c72b0", 0.45: "#dd8452", 0.6: "#c44e52"}


def main():
    d = json.loads((OUT / "core_size_metrics.json").read_text())
    radii = d["radii"]; means = sorted(d["means"])
    ag = d["aggregate"]; bnd = d["ignition_boundary"]
    guard = d["guard_baseline_wide18_ignition"]; gclean = d["guard_clean"]
    ncore = {r: int(np.median([x["n_core"] for x in d["raw"] if x["radius"] == r])) for r in radii}
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # a — ignition prob vs mean, per radius, wide solid / narrow dashed
    for r in radii:
        cw = RCOLOR.get(r, "0.4")
        wide = [ag["wide"][f"r{r}_m{m}"]["ignition_rate"] for m in means]
        narrow = [ag["narrow"][f"r{r}_m{m}"]["ignition_rate"] for m in means]
        axA.plot(means, wide, "-o", color=cw, lw=2, ms=6, label=f"r={r}mm wide (~{ncore[r]} cells)")
        axA.plot(means, narrow, "--s", color=cw, lw=1.5, ms=5, alpha=0.7,
                 label=f"r={r}mm narrow")
    axA.axhline(0.5, color="0.6", lw=0.8, ls=":")
    # guard markers: mean=18 wide must be ~0
    for r in radii:
        axA.scatter([18.0], [ag["wide"][f"r{r}_m18.0"]["ignition_rate"]], s=130,
                    facecolors="none", edgecolors="k", zorder=6, lw=1.3)
    axA.annotate("mean=18 wide = GUARD\n(must be ~0 = clean reference)", xy=(18.0, 0.03),
                 xytext=(17.4, 0.45), fontsize=7.5, ha="left",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    axA.set_xlabel("core mean threshold (mV)   — lower = more excitable →")
    axA.set_ylabel("P(core self-ignites before stimulus)")
    axA.set_title("a · ignition vs mean for 3 core sizes\nbigger core → ignites at higher mean?",
                  fontsize=10)
    axA.invert_xaxis(); axA.set_ylim(-0.05, 1.10)
    axA.legend(fontsize=6.5, loc="center left", ncol=1, framealpha=0.9)
    axA.grid(alpha=0.3)

    # b — boundary (mean@0.5) vs core size
    xs = [ncore[r] for r in radii]
    bw = [bnd["wide"][f"r{r}"] for r in radii]
    bn = [bnd["narrow"][f"r{r}"] for r in radii]
    axB.plot(xs, bw, "-o", color="firebrick", lw=2, ms=8, label="wide")
    axB.plot(xs, bn, "--s", color="steelblue", lw=2, ms=8, label="narrow")
    for x, r in zip(xs, radii):
        axB.annotate(f"r={r}", (x, axB.get_ylim()[0]), fontsize=7, ha="center", va="bottom", color="0.4")
    axB.set_xlabel("core size (number of E neurons in core)")
    axB.set_ylabel("ignition boundary: core mean @ P=0.5 (mV)\nhigher = easier to ignite")
    gtxt = "CLEAN (baseline never self-ignites)" if gclean else f"FLAGGED: baseline ignites {guard}"
    axB.set_title(f"b · ignition boundary vs core size\nguard: {gtxt}", fontsize=10)
    axB.legend(fontsize=8); axB.grid(alpha=0.3)

    fig.suptitle(f"Core-size axis: does a bigger pathology core (more low-threshold tail cells) "
                 f"lower the ignition boundary? ({len(d['conn_seeds'])} networks; mid core)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "core_size.png", dpi=140); plt.close(fig)
    print(f"wrote core_size.png (guard_clean={gclean})")


if __name__ == "__main__":
    main()
