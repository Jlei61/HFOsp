#!/usr/bin/env python3
"""M3 plan Task 8: pre-registration + structural sigma phase map (NO dynamics).

Two outputs, both BEFORE any SNN dynamics result:
  1. preregistration.json — the degree-normalization 3-scheme comparison plan + main-readout
     rule, the Layer-2 subject-level tolerance band (computed from Task-0 per_subject medians
     axial_obs/lateral_obs and PERSISTED here so it is self-contained), and the hub/global
     numeric thresholds. Frozen; the dynamics runs (Tasks 9-12) must not change it.
  2. sigma phase maps — for each degnorm scheme, the recruitment-operator branching ratio
     sigma_corridor (can events propagate along the corridor?) and sigma_crossing (can they
     cross the hub into the global region?) over a (degnorm_alpha x hub_gain) grid. This is a
     STRUCTURAL diagnostic (linear algebra on the built connectivity, drive_rest is a proxy) to
     pick candidate workpoints and answer the M3.0 gate "does the topology support a gate?".
     It is NOT a pass/fail gate — the SNN two-layer test is (spec §5.3 lock).

Honest scope: absolute sigma depends on the proxy drive_rest + weight units; the INFORMATIVE
signal is the SHAPE — whether a regime exists where sigma_corridor stays >~1 (propagates) while
sigma_crossing drops <1 (contained at the hub), and whether lowering the hub threshold (smaller
alpha) lifts sigma_crossing back above 1 (broadcast). Pilots verify in the real dynamics.
"""
from __future__ import annotations
import argparse
import csv
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
from src.topic4_degnorm import degnorm_vth_delta                         # noqa: E402
from src.topic4_hub_criticality import (recruitment_operator, branching_ratio,  # noqa: E402
                                        crossing_path_gain)
from src.topic4_m3_acceptance import subject_tolerance_band              # noqa: E402

SCHEMES = ["out_strength", "in_strength", "hybrid"]
# Task-0 per_subject medians live in the MAIN checkout (results/ is gitignored, not in worktree).
TASK0_CSV = "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/event_extent_audit/per_subject.csv"


def write_preregistration(out_dir, args):
    """Persist the frozen pre-registration (Layer-2 band + degnorm plan + hub/global thresholds)."""
    af, lr = [], []
    with open(TASK0_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("excluded_reason", "none") not in ("none", ""):
                continue
            try:
                a, l = float(row["axial_obs"]), float(row["lateral_obs"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(a) and np.isfinite(l):
                af.append(a); lr.append(l)
    band = subject_tolerance_band(np.array(af), np.array(lr), q=(10, 90))
    prereg = {
        "frozen": True,
        "note": ("Pre-registered BEFORE any M3 SNN dynamics result (plan Task 8). The dynamics "
                 "runs must not change these. Layer-2 PASS = model per-subject median AF/LR inside "
                 "the band AND AF>=min_af (NOT a non-significant p-value)."),
        "layer2_tolerance_band": band,
        "layer2_min_af": 0.75,
        "layer2_band_q": [10, 90],
        "layer2_band_source": "Task-0 per_subject.csv axial_obs/lateral_obs (real per-subject medians)",
        "layer2_n_real_subjects": len(af),
        "real_subject_af_median": float(np.median(af)),
        "real_subject_lr_median": float(np.median(lr)),
        "degnorm_schemes": SCHEMES,
        "degnorm_main_readout_rule": ("Compare all 3 schemes; the main scheme is chosen by which "
                                      "produces a stable gate regime (sigma_corridor>~1 & sigma_crossing<1 "
                                      "with a lift to >1 as the hub threshold drops) AND the cleanest "
                                      "two-layer pilot. No single primary pre-set; no result-peeking on "
                                      "the cohort claim — the choice rule is fixed here."),
        "hub_global_interictal_thresholds": {
            "hub_recruited_fraction_max": 0.5,
            "global_E_spike_fraction_max": 0.10,
            "note": "interictal gate (Layer 1): event dies at the hub; global region ~ background.",
        },
        "ictal_bridge_requirements": {
            "global_first_spike_after_hub_ms_min": 0.0,
            "global_E_spike_fraction_lift": "global fraction markedly above interictal threshold",
            "note": "ictal bridge: onset->hub->global timing; synthetic feasibility only (C4).",
        },
    }
    with open(os.path.join(out_dir, "preregistration.json"), "w") as f:
        json.dump(prereg, f, indent=2)
    return prereg


def build_substrate(args):
    p = Params(L=args.L, density=args.density, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    posE = pos[:NE]
    center = np.array([args.L / 2, args.L / 2]); half = args.L / 2
    th = np.radians(args.theta); axis_unit = np.array([np.cos(th), np.sin(th)])
    regions = corridor_regions(posE, center, axis_unit, half,
                               corridor_half_frac=args.corridor_half_frac, hub_frac=args.hub_frac,
                               global_gap_frac=args.global_gap_frac)
    hub = hub_mask_E(NE, regions["hub_idx"])
    # base lesion V_th: two excitable cores at +/- sep_frac*half (twoend_equal, inline, no noise)
    vth0 = np.full(NE + NI, 18.0)
    for s in (-1.0, 1.0):
        focus = center + s * args.sep_frac * half * axis_unit
        d = np.linalg.norm(posE - focus, axis=1)
        vth0[:NE][d <= args.core_r] = args.core_mean
    return p, pos, labels, NE, NI, posE, regions, hub, vth0, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--density", type=float, default=80.0)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--sep-frac", type=float, default=0.5)
    ap.add_argument("--core-mean", type=float, default=17.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--corridor-half-frac", type=float, default=0.7)
    ap.add_argument("--hub-frac", type=float, default=0.03)
    ap.add_argument("--global-gap-frac", type=float, default=0.0,
                    help="spatial gap between corridor and global (0=adjacent)")
    ap.add_argument("--hub-long-range-c", type=int, default=12)
    ap.add_argument("--l-hub-long", type=float, default=6.0)
    ap.add_argument("--drive-rest", type=float, default=15.0, help="proxy resting drive for gap_factor")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-alpha", type=int, default=5)
    ap.add_argument("--alpha-max", type=float, default=2.5)
    ap.add_argument("--n-gain", type=int, default=5)
    ap.add_argument("--gain-max", type=float, default=1.0)
    ap.add_argument("--out", default="results/topic4_sef_hfo/m3_hub_scaffold")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(os.path.join(out_dir, "sigma_phase_map"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    print("[M3.8] writing pre-registration ...", flush=True)
    prereg = write_preregistration(out_dir, args)
    print(f"  Layer-2 band: {prereg['layer2_tolerance_band']} (n={prereg['layer2_n_real_subjects']} real subjects)",
          flush=True)

    print("[M3.8] building substrate ...", flush=True)
    p, pos, labels, NE, NI, posE, regions, hub, vth0, th = build_substrate(args)
    print(f"  N={NE+NI} NE={NE} corridor={regions['corridor_idx'].size} "
          f"hub={regions['hub_idx'].size} global={regions['global_idx'].size}", flush=True)

    alpha_grid = np.linspace(0.0, args.alpha_max, args.n_alpha)
    gain_grid = np.linspace(0.0, args.gain_max, args.n_gain)

    # build one net per gain (connectivity depends on gain, not alpha/scheme) — controlled: fresh
    # rng per build so only hub_gain differs between gains.
    print("[M3.8] building nets per gain ...", flush=True)
    nets = []
    for gain in gain_grid:
        rng_b = np.random.default_rng(args.seed)
        net = build_connectivity_rot(p, pos, labels, NE, NI, rng_b, theta_EE=th, AR=args.AR,
                                     hub_mask_E=hub, hub_long_range_C=args.hub_long_range_c,
                                     l_hub_long=args.l_hub_long, hub_gain=float(gain))
        nets.append(net)
        print(f"    gain={gain:.3f} built", flush=True)

    results = {}
    for scheme in SCHEMES:
        sc = np.full((args.n_alpha, args.n_gain), np.nan)          # corridor self-sustain
        recruit = np.full((args.n_alpha, args.n_gain), np.nan)     # corridor -> hub drive
        cross = np.full((args.n_alpha, args.n_gain), np.nan)       # corridor -> hub -> global gain
        for gi, net in enumerate(nets):
            for ai, alpha in enumerate(alpha_grid):
                vth = vth0 + degnorm_vth_delta(net, NE, NI, float(alpha), scheme)
                M = recruitment_operator(net, vth, NE, args.drive_rest)
                sc[ai, gi] = branching_ratio(M, regions["corridor_idx"])
                pg = crossing_path_gain(M, regions["corridor_idx"],
                                        regions["hub_idx"], regions["global_idx"])
                recruit[ai, gi] = pg["hub_recruit"]
                cross[ai, gi] = pg["gain"]
            print(f"  [{scheme}] gain col {gi+1}/{args.n_gain} done", flush=True)
        results[scheme] = dict(sigma_corridor=sc, hub_recruit=recruit, crossing_gain=cross)
        np.savez(os.path.join(out_dir, "sigma_phase_map", f"{scheme}.npz"),
                 alpha_grid=alpha_grid, gain_grid=gain_grid,
                 sigma_corridor=sc, hub_recruit=recruit, crossing_gain=cross)

    # figure: rows = schemes, cols = 3 distinct questions (CLAUDE.md §7).
    # absolute scale of these structural quantities is NOT calibrated to the real "sigma~1"
    # (the gap-factor linearization ignores refractory saturation), so each panel is normalized
    # to its own max -> read the SHAPE (how it varies with alpha / gain), not absolute level.
    fig, axes = plt.subplots(len(SCHEMES), 3, figsize=(13, 4 * len(SCHEMES)))
    ext = [gain_grid[0], gain_grid[-1], alpha_grid[0], alpha_grid[-1]]
    titles = ["σ_corridor — corridor sustains an event?",
              "hub_recruit — corridor lights the hub? (alpha closes it)",
              "crossing_gain — lit hub reaches global? (the gate)"]
    for r, scheme in enumerate(SCHEMES):
        for c, key in enumerate(["sigma_corridor", "hub_recruit", "crossing_gain"]):
            data = results[scheme][key]
            norm = data / (np.nanmax(np.abs(data)) or 1.0)
            ax = axes[r, c] if len(SCHEMES) > 1 else axes[c]
            im = ax.imshow(norm, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=0, vmax=1)
            ax.set_xlabel("hub_gain"); ax.set_ylabel("degnorm_alpha")
            ax.set_title(f"{scheme}\n{titles[c]}", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("M3 structural σ probe (per-panel-normalized; SHAPE diagnostic, NOT an absolute gate)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(out_dir, "figures", "sigma_phase_map.png"), dpi=130)
    print(f"[M3.8] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
