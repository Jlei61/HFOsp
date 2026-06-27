#!/usr/bin/env python3
"""M3 static-μ SPONTANEOUS no-kick pilot (§3 primary; spec m3_static_mu_pilot_2026-06-24 v2).

The MAIN claim gate: with NO kick, does the network ITSELF go R2 (finite returned) -> R3 (large
returned) -> R4a (W-aligned sustained) as the slow permissivity μ deepens the susceptibility
field? (kick is NOT the seizure mechanism — see run_m3_kick_calibration for the basin support.)

Per seed: build the core field, apply_mu (μ=0 = current field, bit-parity), run ONE long no-kick
sim (KICK_BOOST=0), detect spontaneous events in the population active-fraction trace, classify
each (R0-R4 via sef_hfo_mu_basin), aggregate event rate + size/duration distributions + R-class
fractions. High μ self-ignition is the PHENOTYPE, not contamination (no kick-sham differencing).

PILOT-FIRST: tiny by default; --run to execute. Outputs results/topic4_sef_hfo/m3_static_mu/.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.sef_hfo_event_figure import active_fraction_trace          # noqa: E402
from src.sef_hfo_mu_basin import (apply_mu, detect_events, event_props,    # noqa: E402
                                  classify_event, aggregate_spontaneous, DEFAULT_CAPS)
from run_m3_kick_calibration import (_bin_spike_counts_in_window, _spatial_extent,  # noqa: E402
                                     TRACE_BIN_MS)

R95_CAP = DEFAULT_CAPS["R95_CAP"]
FAR_CAP = DEFAULT_CAPS["FAR_CAP"]


def _event_spatial(E_spk, bin_of_cell, n_bins, bin_centers, src_bin, far_radius, t_lo, t_hi, dt):
    """Per-event spatial metrics (r95/far/n_active from the core) + sustained_front_score (how
    spatially CONCENTRATED the late-window activity is: 1 - active_bins/n_bins; high = front/R4a,
    low = uniform tonic/R4b)."""
    res = {"E_spk_bool": E_spk}
    bins = _bin_spike_counts_in_window(res, bin_of_cell, n_bins, t_lo, t_hi, dt)
    n_act, r95, far = _spatial_extent(bins, bin_centers, src_bin, far_radius)
    # sustained front: spatial concentration of the LAST ~50ms of the event window
    tail_lo = max(t_lo, t_hi - 50.0)
    tail = _bin_spike_counts_in_window(res, bin_of_cell, n_bins, tail_lo, t_hi, dt)
    active_bins = int(np.sum(tail > 0))
    front_score = 1.0 - active_bins / n_bins
    return float(r95), float(far), int(n_act), float(front_score)


def run(args):
    from params import Params
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from kick_probe import simulate_kick
    from src.topic4_propagation_operator import spatial_bins
    from src.sef_hfo_heterogeneity import sample_core_field

    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    p = Params(L=args.L, density=args.density, T=args.T, dt=args.dt,
               nu_ext_ratio=args.nu_ext_ratio, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.radians(args.theta_ee), AR=args.AR)
    posE = pos[:NE]
    bins_info = spatial_bins(posE, args.n_bins_per_axis)
    bin_centers = bins_info["bin_centers"]; bin_of_cell = bins_info["bin_of_cell"]
    n_bins = bin_centers.shape[0]
    is_E = (labels == 0)
    vth_uniform = np.full(NE + NI, args.vth0)

    core_mode = args.core_mean is not None
    if core_mode:
        core_center = (np.asarray(args.core_center_xy, float) if args.core_center_xy is not None
                       else bin_centers.mean(axis=0))
        vth_core = sample_core_field(pos, is_E, core_center, args.core_r,
                                     np.random.default_rng(args.seed + 7),
                                     core_mean=args.core_mean, core_std=args.core_std,
                                     base_mean=args.vth0)["vth"]
    else:
        core_center = bin_centers.mean(axis=0); vth_core = vth_uniform
    core_mean_for_mu = args.core_mean if args.core_mean is not None else args.vth0
    vth_eff = apply_mu(vth_core, args.vth0, core_mean_for_mu, args.mu, args.dvth_at_mu1,
                       args.h_mode, np.random.default_rng(args.seed + 13))
    src_bin = int(np.argmin(np.linalg.norm(bin_centers - core_center[None, :], axis=1)))
    far_radius = args.far_radius_mm if args.far_radius_mm is not None else 0.35 * args.L
    record_ms = float(args.T)

    per_event_rows, all_classes, per_seed = [], [], []
    for s in range(args.seeds):
        net_c = dict(net); net_c["rng"] = np.random.default_rng(s + 200)
        res = simulate_kick(p, net_c, KICK_BOOST=0.0, kick_center=core_center,
                            V_th_per_neuron=vth_eff, r_kick=args.r_kick, t_kick=0.0)
        E_spk = res["E_spk_bool"]
        trace = active_fraction_trace(E_spk, p.dt, TRACE_BIN_MS)
        med = float(np.median(trace))
        mad = float(np.median(np.abs(trace - med))) * 1.4826
        thresh = max(args.thresh_floor, med + args.thresh_k * mad)
        events = detect_events(trace, thresh, min_gap_bins=args.min_gap_bins)
        n_rec_bins = len(trace)
        seed_classes = []
        for (b0, b1) in events:
            t_lo = b0 * TRACE_BIN_MS; t_hi = (b1 + 1) * TRACE_BIN_MS
            ep = event_props(trace, (b0, b1), TRACE_BIN_MS, n_rec_bins)
            r95, far, n_act, front = _event_spatial(E_spk, bin_of_cell, n_bins, bin_centers,
                                                    src_bin, far_radius, t_lo, t_hi, p.dt)
            metrics = {"event_detected": True, "returned": ep["returned"],
                       "runaway": ep["sustained"], "r95_ea": r95, "far_ea": far,
                       "active_peak": ep["peak_active"], "sustained_front_score": front}
            cls = classify_event(metrics)
            seed_classes.append(cls); all_classes.append(cls)
            per_event_rows.append({"seed": s, "t_start_ms": round(t_lo, 1),
                                   "duration_ms": ep["duration_ms"], "peak_active": round(ep["peak_active"], 4),
                                   "r95_mm": round(r95, 2), "far_frac": round(far, 3),
                                   "n_active_bins": n_act, "front_score": round(front, 3),
                                   "returned": ep["returned"], "class": cls})
        per_seed.append({"seed": s, "n_events": len(events), "thresh": round(thresh, 5),
                         "classes": seed_classes})

    agg = aggregate_spontaneous(len(all_classes), record_ms * args.seeds, all_classes)
    durations = [r["duration_ms"] for r in per_event_rows]
    sizes = [r["n_active_bins"] for r in per_event_rows]
    summary = {"mu": args.mu, "dvth_mV": round(args.dvth_at_mu1 * args.mu, 3), "h_mode": args.h_mode,
               "substrate": ("bare" if not core_mode else f"core{args.core_mean:g}"),
               "L": args.L, "seeds": args.seeds, "T_ms": args.T,
               "n_events_total": len(all_classes),
               "event_rate_hz_per_seed": round(agg["event_rate_hz"], 4),
               "R_fractions": {k: round(v, 3) for k, v in agg["frac"].items()},
               "duration_ms": {"median": float(np.median(durations)) if durations else 0.0,
                               "max": float(np.max(durations)) if durations else 0.0},
               "size_active_bins": {"median": float(np.median(sizes)) if sizes else 0.0,
                                    "max": float(np.max(sizes)) if sizes else 0.0}}
    with open(os.path.join(out_dir, "spontaneous_per_event.csv"), "w", newline="") as f:
        if per_event_rows:
            w = csv.DictWriter(f, fieldnames=list(per_event_rows[0].keys()))
            w.writeheader(); w.writerows(per_event_rows)
        else:
            f.write("(no spontaneous events detected)\n")
    json.dump({"summary": summary, "per_seed": per_seed}, open(os.path.join(out_dir, "spontaneous_summary.json"), "w"), indent=1)
    print(f"[spontaneous] μ={args.mu} ΔVth={summary['dvth_mV']}mV h={args.h_mode} "
          f"{summary['substrate']} L={args.L}: {len(all_classes)} events, "
          f"rate={summary['event_rate_hz_per_seed']}Hz, R={summary['R_fractions']}")
    print(f"[spontaneous] wrote -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="M3 static-μ spontaneous no-kick pilot (§3 primary)")
    p.add_argument("--L", type=float, default=20.0); p.add_argument("--density", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--T", type=float, default=8000.0, help="record length (ms); spontaneous needs long T")
    p.add_argument("--dt", type=float, default=0.1); p.add_argument("--nu-ext-ratio", type=float, default=0.6)
    p.add_argument("--theta-ee", type=float, default=45.0); p.add_argument("--AR", type=float, default=2.0)
    p.add_argument("--vth0", type=float, default=18.0); p.add_argument("--n-bins-per-axis", type=int, default=5)
    p.add_argument("--core-mean", type=float, default=None); p.add_argument("--core-std", type=float, default=0.5)
    p.add_argument("--core-r", type=float, default=1.5); p.add_argument("--core-center-xy", type=float, nargs=2, default=None)
    p.add_argument("--mu", type=float, default=0.0); p.add_argument("--dvth-at-mu1", type=float, default=1.333)
    p.add_argument("--h-mode", choices=["core_susceptibility", "uniform", "shuffled"], default="core_susceptibility")
    p.add_argument("--r-kick", type=float, default=0.5); p.add_argument("--far-radius-mm", type=float, default=None)
    p.add_argument("--thresh-k", type=float, default=6.0, help="event threshold = median + k*MAD of the trace")
    p.add_argument("--thresh-floor", type=float, default=0.01, help="min event threshold (active fraction)")
    p.add_argument("--min-gap-bins", type=int, default=3, help="merge events separated by < this many quiet bins")
    p.add_argument("--out-dir", type=str, default="results/topic4_sef_hfo/m3_static_mu/spontaneous/tmp")
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    if not args.run:
        print("[spontaneous] PILOT-FIRST: pass --run. Nothing was run."); return 0
    run(args); return 0


if __name__ == "__main__":
    sys.exit(main())
