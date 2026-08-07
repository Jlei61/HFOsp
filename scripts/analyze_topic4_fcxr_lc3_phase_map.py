#!/usr/bin/env python
"""Turn the frozen-state probes into a map, and say what the map licenses.

Three panels, three separate questions, per the project's figure rule:

1. **from a low start** -- at which frozen states does the tissue ignite on its own?
2. **from a high start** -- at which frozen states does a discharge already under way survive?
3. **where the two disagree** -- the empirically bistable region.

Overlaying the real trajectory on panel 3 is the natural next step and is deliberately not done
here: the trajectory lives in absolute D and relay units while the grid is in scale factors, and a
path drawn through the wrong conversion would be read as a route the lifecycle took.

Panels 1 and 2 are not the same construct seen twice: the first is an ignition boundary and the
second an extinction boundary, and the gap between them is the whole point.

The colour is the regime, not "seizure": at this working point every high state on this substrate
is a train that re-ignites from silence every 86 ms, which the project's own criterion excludes
from counting as an ictal carrier.  A map that painted those cells "ictal" would be asserting the
thing the classifier exists to deny.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc3_bistability import (  # noqa: E402
    bistable_points,
    evidence_summary,
)

RUN = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/phase_map"
FIGS = RUN / "figures"

ORDER = ["R0_interictal_only", "R2_bounded_high", "R3_carrier", "R4_burst_train", "R1_runaway"]
COLOURS = ["#f2f2f2", "#9ecae1", "#e8743b", "#f4b266", "#7a1f1f"]
LABELS = {"R0_interictal_only": "interictal", "R2_bounded_high": "bounded continuous",
          "R3_carrier": "carrier (troughs recruited)", "R4_burst_train": "burst train (re-ignites)",
          "R1_runaway": "runaway / saturated"}


def _grid(rows, ic, alphas_d, alphas_x, key):
    g = np.full((len(alphas_x), len(alphas_d)), np.nan)
    for r in rows:
        if r["ic"] != ic:
            continue
        i, j = alphas_x.index(r["alpha_x"]), alphas_d.index(r["alpha_d"])
        g[i, j] = key(r)
    return g


def _panel(ax, grid, alphas_d, alphas_x, title):
    cmap = ListedColormap(COLOURS)
    ax.imshow(grid, origin="lower", cmap=cmap, vmin=-0.5, vmax=len(ORDER) - 0.5,
              aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(alphas_d)), [f"{a:g}" for a in alphas_d], fontsize=7.5)
    ax.set_yticks(range(len(alphas_x)), [f"{a:g}" for a in alphas_x], fontsize=7.5)
    ax.set_xlabel(r"disinhibition scale  $\alpha_D$", fontsize=8.5)
    ax.set_ylabel(r"relay-load scale  $\alpha_X$", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rows = [json.load(open(p)) for p in sorted((RUN / "points").glob("*.json"))]
    rows = [r for r in rows if r.get("status") == "COMPLETE"]
    if not rows:
        raise SystemExit(f"no completed points under {RUN / 'points'}")
    alphas_d = sorted({r["alpha_d"] for r in rows})
    alphas_x = sorted({r["alpha_x"] for r in rows})
    print(f"  {len(rows)} probes over {len(alphas_d)}x{len(alphas_x)} points")

    ev = evidence_summary(rows)
    pts = bistable_points(rows)
    for line in (f"  bistability: {ev['bistability']}  "
                 f"({ev['n_bistable']}/{ev['n_points']} points)",
                 f"  hysteresis:  {ev['hysteresis']}  {ev['hysteresis_detail'] or ''}",
                 f"  jump:        {ev['jump']}",
                 f"  allowed:     {ev['claim_allowed']}",
                 f"  forbidden:   {ev['claim_forbidden']}"):
        print(line)
    n_carrier = sum(1 for r in rows if r.get("carrier"))
    print(f"  carriers among all probes: {n_carrier}/{len(rows)} "
          f"(a high state that never falls back to interictal between bursts)")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), facecolor="white")
    for ax, ic, title in ((axes[0], "interictal", "from a low start: where it ignites"),
                          (axes[1], "ictal", "from a high start: where it survives")):
        g = _grid(rows, ic, alphas_d, alphas_x,
                  lambda r: ORDER.index(r["regime"]) if r["regime"] in ORDER else np.nan)
        _panel(ax, g, alphas_d, alphas_x, title)

    bi = np.full((len(alphas_x), len(alphas_d)), 0.0)
    for p in pts:
        bi[alphas_x.index(p["alpha_x"]), alphas_d.index(p["alpha_d"])] = (
            2.0 if p["bistable"] else (1.0 if p["from_ictal"] == "high" else 0.0))
    axes[2].imshow(bi, origin="lower", cmap=ListedColormap(["#f2f2f2", "#9ecae1", "#b2182b"]),
                   vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")
    axes[2].set_xticks(range(len(alphas_d)), [f"{a:g}" for a in alphas_d], fontsize=7.5)
    axes[2].set_yticks(range(len(alphas_x)), [f"{a:g}" for a in alphas_x], fontsize=7.5)
    axes[2].set_xlabel(r"disinhibition scale  $\alpha_D$", fontsize=8.5)
    axes[2].set_title("where the two starts disagree", fontsize=9.5, fontweight="bold")
    axes[2].legend(handles=[Patch(fc="#b2182b", label="bistable"),
                            Patch(fc="#9ecae1", label="high from both"),
                            Patch(fc="#f2f2f2", label="low from both")],
                   frameon=False, fontsize=7, loc="upper left")

    handles = [Patch(fc=c, label=LABELS[k]) for k, c in zip(ORDER, COLOURS)]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / "phase_map_D_X.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "phase_map_D_X.pdf", bbox_inches="tight")
    plt.close(fig)

    out = RUN / "adjudication.json"
    out.write_text(json.dumps(dict(
        schema="fcxr-lc3-phase-map-1.0", n_probes=len(rows),
        alpha_d=alphas_d, alpha_x=alphas_x, evidence=ev, points=pts,
        n_carrier_probes=n_carrier,
        boundary=("the map's colours are regimes, not seizures: every high state here is a train "
                  "that re-ignites from silence, which the project's carrier criterion excludes "
                  "from counting as an ictal carrier"),
        rows=rows), indent=2, default=float))
    print(f"\n  wrote {out} and {FIGS / 'phase_map_D_X.png'}")


if __name__ == "__main__":
    main()
