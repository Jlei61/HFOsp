"""M2 brake OFF-vs-ON spatial-extent comparison (Topic 4 cm-SNN, stage-3 multi-focus substrate).

Two independent questions, one panel each (CLAUDE.md §7):
  A. Does the event bound itself, or fill the sheet? -> along-axis reach vs sheet size L.
     A self-limited event would be L-INVARIANT (flat); a boundary-limited one TRACKS the sheet.
  B. What does turning the brake on actually do? -> clean forward/reverse event yield.

Pure plotting from the already-computed gate-pilot artifacts (NO re-sim):
  fullfield_<tag>.json  -> per-event reach_axis_mm (readable n_part>=7)
  readout_<tag>.json    -> n_clean_forward / n_clean_reverse
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
GP = os.path.join(OUT, "m2_gate_pilot")
FIG = os.path.join(OUT, "figures")

# (label, color, tag_at_L20, tag_at_L32) — front-inhibition brake = the wide I->E veto gate.
CONDS = [
    ("brake OFF",            "#1f77b4", "g_off_L20",       "g_off_L32"),
    ("brake ON (moderate)",  "#ff7f0e", "g0.5_lg1.5_L20",  "g0.5_lg1.5_L32"),
    ("brake ON (strong)",    "#d62728", "g0.7_lg1.5_L20",  "g0.7_lg1.5_L32"),
]
PART_MIN = 7


def _reach(tag):
    p = os.path.join(GP, f"fullfield_{tag}.json")
    if not os.path.exists(p):
        return np.array([])
    evs = json.load(open(p))["events"]
    return np.array([e["reach_axis_mm"] for e in evs
                     if e.get("n_part", 0) >= PART_MIN and e.get("reach_axis_mm") is not None])


def _clean_counts(tag):
    p = os.path.join(GP, f"readout_{tag}.json")
    d = json.load(open(p))
    return d["n_clean_forward"], d["n_clean_reverse"]


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="white")

    # ---- Panel A: along-axis reach vs sheet size (self-limit discriminator) ----
    Ls = [20.0, 32.0]
    for label, color, t20, t32 in CONDS:
        cells = [(20.0, _reach(t20)), (32.0, _reach(t32))]
        meds = []
        for j, (L, r) in enumerate(cells):
            if r.size:
                x = L + np.linspace(-0.7, 0.7, r.size) if r.size > 1 else np.array([L])
                axA.scatter(x, r, s=26, color=color, alpha=0.55, edgecolor="none", zorder=2)
                meds.append((L, float(np.median(r))))
        if len(meds) == 2:
            xs, ys = zip(*meds)
            axA.plot(xs, ys, "-o", color=color, lw=2.4, ms=8, zorder=3,
                     label=f"{label}  (median)")
    # full-sheet (boundary) reach along the 45° axis = L*sqrt(2)
    diag = [L * np.sqrt(2) for L in Ls]
    axA.plot(Ls, diag, "--", color="0.45", lw=1.8, zorder=1)
    axA.annotate("fills the sheet (boundary-limited ceiling = L·√2)",
                 xy=(32, diag[1]), xytext=(21.5, diag[1] + 1.5), fontsize=9.5, color="0.35")
    axA.annotate("a self-limited event would be FLAT (L-invariant)",
                 xy=(26, 24), fontsize=10, color="0.2", style="italic",
                 ha="center")
    axA.set_xticks(Ls); axA.set_xticklabels(["20×20", "32×32"])
    axA.set_xlabel("sheet size  L  (mm)", fontsize=12)
    axA.set_ylabel("event along-axis reach (mm)", fontsize=12)
    axA.set_xlim(18.5, 33.5); axA.set_ylim(0, 48)
    axA.set_title("A. Reach tracks the sheet, not a fixed scale\n(brake ON or OFF)",
                  fontsize=12.5, fontweight="bold")
    axA.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    axA.grid(alpha=0.25)

    # ---- Panel B: brake effect on clean event yield + direction (at L=20) ----
    labels = [c[0] for c in CONDS]
    fwd = []; rev = []
    for _, _, t20, _ in CONDS:
        f, r = _clean_counts(t20)
        fwd.append(f); rev.append(r)
    x = np.arange(len(labels)); w = 0.38
    axB.bar(x - w / 2, fwd, w, color="#2c7fb8", label="forward (clean)")
    axB.bar(x + w / 2, rev, w, color="#c51b8a", label="reverse (clean)")
    for i in range(len(labels)):
        axB.text(x[i] - w / 2, fwd[i] + 0.15, str(fwd[i]), ha="center", fontsize=10)
        axB.text(x[i] + w / 2, rev[i] + 0.15, str(rev[i]), ha="center", fontsize=10)
    axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=10)
    axB.set_ylabel("clean event count  (L=20, T=8 s)", fontsize=12)
    axB.set_ylim(0, max(fwd) + 1.5)
    axB.set_title("B. The brake suppresses events (reverse first),\nit does not bound them",
                  fontsize=12.5, fontweight="bold")
    axB.legend(fontsize=10, loc="upper right")
    axB.grid(axis="y", alpha=0.25)

    fig.suptitle("Front-inhibition brake does NOT spatially self-limit the cm-SNN event "
                 "(stage-3 multi-focus substrate)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIG, "m2_brake_extent_comparison.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
