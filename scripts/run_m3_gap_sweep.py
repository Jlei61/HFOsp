#!/usr/bin/env python3
"""M3 controlled gap sweep (fast, NO eigs): hold the corridor geometry FIXED and vary ONLY the
spatial gap between corridor and global, to cleanly answer "does a structural barrier make a real
hub gate?" (the adjacent-vs-separated comparison in the sigma maps was uncontrolled — corridor
size also changed). Uses only the cheap crossing path-gain (sparse sums), so it runs in minutes.

The gate quantity is hub_broadcast = mean gap-weighted hub->global drive:
  at hub_gain=0  -> the LOCAL leak (hub's local E->E edges into global). Should DROP as the gap grows
                   (the hub moves away from global, out of local range).
  at hub_gain>0  -> restored by the long-range broadcast (l_hub_long >> local), which bridges the gap.
A clean gate = leak small AND broadcast restores crossing -> the broadcast/leak ratio RISES with gap.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from params import Params                       # noqa: E402
from connectivity import place_neurons          # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from src.topic4_corridor_substrate import corridor_regions, hub_mask_E  # noqa: E402
from src.topic4_hub_criticality import recruitment_operator, crossing_path_gain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=14.0)
    ap.add_argument("--density", type=float, default=70.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--sep-frac", type=float, default=0.4)
    ap.add_argument("--corridor-half-frac", type=float, default=0.55)  # FIXED across the sweep
    ap.add_argument("--hub-frac", type=float, default=0.03)
    ap.add_argument("--core-mean", type=float, default=17.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--hub-long-range-c", type=int, default=12)
    ap.add_argument("--l-hub-long", type=float, default=6.0)
    ap.add_argument("--drive-rest", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    ap.add_argument("--gain", type=float, default=1.0, help="broadcast gain for the 'on' condition")
    ap.add_argument("--out", default="results/topic4_sef_hfo/m3_hub_scaffold/gap_sweep")
    a = ap.parse_args()

    out_dir = os.path.join(ROOT, a.out)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    p = Params(L=a.L, density=a.density, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=a.seed)
    rng0 = np.random.default_rng(a.seed)
    pos, labels, NE, NI = place_neurons(p, rng0)
    posE = pos[:NE]
    center = np.array([a.L / 2, a.L / 2]); half = a.L / 2
    axis_unit = np.array([1.0, 0.0])
    # FIXED base lesion V_th (two cores) — identical across all gaps (the controlled part)
    vth0 = np.full(NE + NI, 18.0)
    for s in (-1.0, 1.0):
        focus = center + s * a.sep_frac * half * axis_unit
        vth0[:NE][np.linalg.norm(posE - focus, axis=1) <= a.core_r] = a.core_mean

    rows = []
    for gap in a.gaps:
        reg = corridor_regions(posE, center, axis_unit, half, corridor_half_frac=a.corridor_half_frac,
                               hub_frac=a.hub_frac, global_gap_frac=float(gap))
        hub = hub_mask_E(NE, reg["hub_idx"])
        n_glob = reg["global_idx"].size
        out = {"gap": float(gap), "n_corridor": int(reg["corridor_idx"].size),
               "n_hub": int(reg["hub_idx"].size), "n_global": int(n_glob)}
        if n_glob == 0:
            out.update(leak=float("nan"), broadcast_on=float("nan"), ratio=float("nan"))
            rows.append(out); print(f"gap={gap}: global EMPTY -> skip", flush=True); continue
        for cond, gain in [("off", 0.0), ("on", a.gain)]:
            rng_b = np.random.default_rng(a.seed)
            net = build_connectivity_rot(p, pos, labels, NE, NI, rng_b, theta_EE=0.0, AR=a.AR,
                                         hub_mask_E=hub, hub_long_range_C=a.hub_long_range_c,
                                         l_hub_long=a.l_hub_long, hub_gain=float(gain))
            M = recruitment_operator(net, vth0, NE, a.drive_rest)
            pg = crossing_path_gain(M, reg["corridor_idx"], reg["hub_idx"], reg["global_idx"])
            out["hub_recruit"] = round(pg["hub_recruit"], 3)  # ~constant across gap (corridor fixed)
            out["leak" if cond == "off" else "broadcast_on"] = round(pg["hub_broadcast"], 4)
        out["ratio"] = round(out["broadcast_on"] / out["leak"], 3) if out["leak"] > 1e-9 else float("inf")
        rows.append(out)
        print(f"gap={gap}: n_global={n_glob} leak(hub->global,gain0)={out['leak']} "
              f"broadcast(gain{a.gain})={out['broadcast_on']} ratio={out['ratio']}", flush=True)

    with open(os.path.join(out_dir, "gap_sweep.json"), "w") as f:
        json.dump(dict(config=vars(a), rows=rows), f, indent=2)

    # figure: leak vs gap (should fall) + broadcast vs gap (held by l_hub_long) + ratio (should rise)
    valid = [r for r in rows if np.isfinite(r.get("leak", np.nan))]
    if valid:
        gaps = [r["gap"] for r in valid]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(gaps, [r["leak"] for r in valid], "o-", label="leak (gain=0, hub→global local)")
        ax[0].plot(gaps, [r["broadcast_on"] for r in valid], "s-", label=f"with broadcast (gain={a.gain})")
        ax[0].set_xlabel("global_gap_frac (corridor↔global separation)")
        ax[0].set_ylabel("hub→global drive (gap-weighted)")
        ax[0].set_title("does a bigger gap remove the local leak?"); ax[0].legend(fontsize=8)
        ax[1].plot(gaps, [r["ratio"] for r in valid], "d-", color="C3")
        ax[1].axhline(1.0, ls="--", color="gray", lw=0.8)
        ax[1].set_xlabel("global_gap_frac"); ax[1].set_ylabel("broadcast / leak ratio")
        ax[1].set_title("gate quality (>1 = broadcast beats local leak)")
        fig.suptitle("M3 controlled gap sweep (corridor FIXED; structural diagnostic)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(out_dir, "figures", "gap_sweep.png"), dpi=130)
    print(f"[gap-sweep] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
