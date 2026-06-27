#!/usr/bin/env python3
"""M3A-A1 quasi-static slow-state no-kick pilot (spec 2026-06-24-sef-hfo-m3a-quasistatic-slowstate).

ONE frozen slow variable per invocation. Question: if the tissue were ALREADY clamped at this slow
state, does the network's OWN (no-kick) spontaneous event phenotype shift from "small returned"
(R2/R3) toward "sustained recruitment" (R4a) — beyond a mere event-rate increase?

This is NOT the static-μ runner. It writes to results/topic4_sef_hfo/m3a_slowvars/quasistatic/<tag>/.
It excludes W / h(W); external kick is OFF (KICK_BOOST=0, the spontaneous record IS the evidence).

Engine paths (verified, src/snn_engine/kick_probe.py):
  z / phi / gK  -> slow=FrozenSlowVars (slow= path; uses uniform p.V_th, bypasses shunt_gaba AND
                   V_th_per_neuron). Substrate = uniform vth0.
  egaba         -> slow=None + shunt_gaba=True + e_gaba (membrane shunt path; V_th_per_neuron honored).
  off           -> slow=None current-based baseline (the interictal anchor).
The two paths NEVER combine (the "do not mix z and e_GABA" trap) — one --slow-var per run, and
_build_slow_membrane() raises if asked to combine.

PILOT-FIRST: tiny by default; --run to execute.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.sef_hfo_event_figure import active_fraction_trace               # noqa: E402
from src.sef_hfo_mu_basin import (detect_events, event_props, classify_event,   # noqa: E402
                                  aggregate_spontaneous, DEFAULT_CAPS)
from src.sef_hfo_slowvars_quasistatic import build_frozen_slowvars       # noqa: E402

try:  # Optional legacy M3 dependency. M3A only needs the bin width constant.
    from run_m3_kick_calibration import TRACE_BIN_MS                      # noqa: E402
except ModuleNotFoundError:  # keep this runner importable when old M3 scripts are not merged
    TRACE_BIN_MS = 2.0

try:
    from run_m3_static_mu_spontaneous import _event_spatial                # noqa: E402
except ModuleNotFoundError:
    from src.sef_hfo_a2 import _bin_spike_counts, _spatial_extent          # noqa: E402

    def _event_spatial(E_spk, bin_of_cell, n_bins, bin_centers, src_bin,
                       far_radius, t_lo, t_hi, dt):
        """Local fallback for static-mu spatial metrics used by the A1 runner."""
        lo = int(np.floor(float(t_lo) / float(dt)))
        hi = int(np.ceil(float(t_hi) / float(dt)))
        bins = _bin_spike_counts(E_spk, bin_of_cell, n_bins, lo, hi)
        n_act, r95, far = _spatial_extent(bins, bin_centers, src_bin, far_radius)
        tail_lo = max(float(t_lo), float(t_hi) - 50.0)
        tlo = int(np.floor(tail_lo / float(dt)))
        thi = int(np.ceil(float(t_hi) / float(dt)))
        tail_bins = _bin_spike_counts(E_spk, bin_of_cell, n_bins, tlo, thi)
        front_score = 1.0 - float(np.sum(tail_bins > 0)) / max(int(n_bins), 1)
        return float(r95), float(far), int(n_act), float(front_score)

SLOW_VARS = ("off", "z", "phi", "gK", "egaba")
_SLOW_KEYS = ("z", "phi", "gK", "e_gaba")          # per-event slow-state columns (plan §4)
_LANDMARKS = ("pre", "onset", "peak", "end")


def _frozen_slow_scalar(var, level, vth0):
    """The single frozen scalar entering the engine for this run, keyed to the plan's slow column.
    Returns (active_key in {z,phi,gK,e_gaba} or None for 'off', scalar value or None)."""
    if var == "off":
        return None, None
    if var == "z":
        return "z", float(level)
    if var == "phi":
        return "phi", float(vth0) + float(level)     # absolute adaptive threshold = vth0 + offset
    if var == "gK":
        return "gK", float(level)
    if var == "egaba":
        return "e_gaba", float(level)
    raise ValueError(f"unknown slow var {var!r}")


def _slow_state_fields(var, level, vth0):
    """Per-event slow-state dict (z_pre..e_gaba_end). Quasi-static = frozen: the active variable is
    CONSTANT across pre/onset/peak/end; every inactive variable is 'NA' (never 0 — plan §4)."""
    out = {f"{k}_{lm}": "NA" for k in _SLOW_KEYS for lm in _LANDMARKS}
    key, val = _frozen_slow_scalar(var, level, vth0)
    if key is not None:
        for lm in _LANDMARKS:
            out[f"{key}_{lm}"] = round(val, 4)
    return out


def _build_slow_membrane(var, level, N, vth0, g_gaba_scale):
    """Map one slow variable to the (slow, V_th_per_neuron, shunt_gaba, e_gaba, g_gaba_scale) engine
    args. The z/φ/g_K path and the e_GABA shunt path are MUTUALLY EXCLUSIVE (raise on any attempt to
    combine — slow!=None silently bypasses shunt_gaba, so combining z and e_GABA would be a silent
    science bug)."""
    vth_uniform = np.full(N, float(vth0))
    if var == "off":
        return None, vth_uniform, False, None, 0.0
    if var in ("z", "phi", "gK"):
        kw = {"z": {"z": level}, "phi": {"phi_offset": level}, "gK": {"gK": level}}[var]
        slow = build_frozen_slowvars(N, vth0, **kw)
        return slow, None, False, None, 0.0           # slow path: shunt OFF, V_th_per_neuron bypassed
    if var == "egaba":
        return None, vth_uniform, True, float(level), float(g_gaba_scale)
    raise ValueError(f"unknown slow var {var!r}")


def run(args):
    from params import Params
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot
    from kick_probe import simulate_kick
    from src.topic4_propagation_operator import spatial_bins

    tag = args.tag or (args.slow_var if args.slow_var == "off" else f"{args.slow_var}{args.level:g}")
    out_dir = os.path.join(ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    out_dir = os.path.join(out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Bare UNIFORM substrate (A1 scope): vth0 everywhere. p.V_th pinned to vth0 so the slow= path
    # (uniform p.V_th) and the off/egaba path (V_th_per_neuron=vth0) share one threshold base.
    p = Params(L=args.L, density=args.density, T=args.T, dt=args.dt,
               nu_ext_ratio=args.nu_ext_ratio, seed=args.seed, V_th=args.vth0)
    rng = np.random.default_rng(args.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.radians(args.theta_ee), AR=args.AR)
    N = NE + NI
    posE = pos[:NE]
    bins_info = spatial_bins(posE, args.n_bins_per_axis)
    bin_centers = bins_info["bin_centers"]; bin_of_cell = bins_info["bin_of_cell"]
    n_bins = bin_centers.shape[0]
    core_center = bin_centers.mean(axis=0)
    src_bin = int(np.argmin(np.linalg.norm(bin_centers - core_center[None, :], axis=1)))
    far_radius = args.far_radius_mm if args.far_radius_mm is not None else 0.35 * args.L
    record_ms = float(args.T)

    state_label = tag
    active_key, active_val = _frozen_slow_scalar(args.slow_var, args.level, args.vth0)
    slow_fields = _slow_state_fields(args.slow_var, args.level, args.vth0)

    per_event_rows, slow_sample_rows, all_classes, per_seed = [], [], [], []
    eid = 0
    for s in range(args.seeds):
        slow, vth_pn, shunt, e_gaba, g_scale = _build_slow_membrane(
            args.slow_var, args.level, N, args.vth0, args.g_gaba_scale)
        net_c = dict(net); net_c["rng"] = np.random.default_rng(s + 200)
        res = simulate_kick(p, net_c, KICK_BOOST=0.0, kick_center=core_center,
                            slow=slow, V_th_per_neuron=vth_pn, shunt_gaba=shunt,
                            e_gaba=e_gaba, g_gaba_scale=g_scale, r_kick=args.r_kick, t_kick=0.0)
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
            active_mass = float(np.sum(trace[b0:b1 + 1]))
            seed_classes.append(cls); all_classes.append(cls)
            row = {"event_id": eid, "seed": s, "state_label": state_label,
                   "onset_ms": round(t_lo, 1), "end_ms": round(t_hi, 1),
                   "duration_ms": ep["duration_ms"], "size_bins": n_act,
                   "active_mass": round(active_mass, 4), "peak_active": round(ep["peak_active"], 4),
                   "r95_mm": round(r95, 2), "far_frac": round(far, 3),
                   "return_to_baseline": ep["returned"], "R_class": cls,
                   "sustained_front_score": round(front, 3)}
            row.update(slow_fields)
            per_event_rows.append(row)
            # slow_state_samples: landmark samples of the single active slow scalar (frozen => flat).
            s_val = (round(active_val, 4) if active_val is not None else "NA")
            slow_sample_rows.append({"event_id": eid, "seed": s, "state_label": state_label,
                                     "active_var": (active_key or "NA"), "frozen": True,
                                     "s_pre": s_val, "s_onset": s_val, "s_peak": s_val, "s_end": s_val,
                                     "s_post_50ms": s_val, "s_post_200ms": s_val, "s_post_1s": s_val})
            eid += 1
        per_seed.append({"seed": s, "n_events": len(events), "thresh": round(thresh, 5),
                         "classes": seed_classes})

    agg = aggregate_spontaneous(len(all_classes), record_ms * args.seeds, all_classes)
    durations = [r["duration_ms"] for r in per_event_rows]
    sizes = [r["size_bins"] for r in per_event_rows]
    masses = [r["active_mass"] for r in per_event_rows]
    returns = [r["return_to_baseline"] for r in per_event_rows]
    summary = {
        "slow_var": args.slow_var, "level": args.level, "state_label": state_label,
        "active_key": active_key, "active_value": active_val,
        "substrate": f"uniform vth0={args.vth0:g}", "L": args.L, "seeds": args.seeds, "T_ms": args.T,
        "nu_ext_ratio": args.nu_ext_ratio, "g_gaba_scale": (args.g_gaba_scale if args.slow_var == "egaba" else None),
        "n_events_total": len(all_classes),
        "event_rate_hz_per_seed": round(agg["event_rate_hz"], 4),
        "return_probability": (round(float(np.mean(returns)), 3) if returns else None),
        "R_fractions": {k: round(v, 3) for k, v in agg["frac"].items()},
        "duration_ms": {"median": float(np.median(durations)) if durations else 0.0,
                        "max": float(np.max(durations)) if durations else 0.0},
        "size_bins": {"median": float(np.median(sizes)) if sizes else 0.0,
                      "max": float(np.max(sizes)) if sizes else 0.0},
        "active_mass": {"median": float(np.median(masses)) if masses else 0.0,
                        "max": float(np.max(masses)) if masses else 0.0},
    }

    # ---- write per-event csv (full slow-state schema; inactive vars = NA) ----
    pe_cols = (["event_id", "seed", "state_label", "onset_ms", "end_ms", "duration_ms",
                "size_bins", "active_mass", "peak_active", "r95_mm", "far_frac",
                "return_to_baseline", "R_class", "sustained_front_score"]
               + [f"{k}_{lm}" for k in _SLOW_KEYS for lm in _LANDMARKS])
    with open(os.path.join(out_dir, "spontaneous_per_event.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pe_cols)
        w.writeheader()
        w.writerows(per_event_rows)
        if not per_event_rows:
            f.write("# (no spontaneous events detected)\n")
    with open(os.path.join(out_dir, "slow_state_samples.csv"), "w", newline="") as f:
        ss_cols = ["event_id", "seed", "state_label", "active_var", "frozen",
                   "s_pre", "s_onset", "s_peak", "s_end", "s_post_50ms", "s_post_200ms", "s_post_1s"]
        w = csv.DictWriter(f, fieldnames=ss_cols)
        w.writeheader(); w.writerows(slow_sample_rows)
    json.dump({"summary": summary, "per_seed": per_seed},
              open(os.path.join(out_dir, "spontaneous_summary.json"), "w"), indent=1)

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_sha = "unknown"
    config = {
        "spec": "git show a1213ee:docs/superpowers/plans/2026-06-24-sef-hfo-m3a-quasistatic-slowstate-plan.md",
        "engine_git_sha": git_sha,
        "slow_var": args.slow_var, "level": args.level, "tag": tag,
        "substrate": f"uniform vth0={args.vth0:g}",
        "params": {"L": args.L, "density": args.density, "T": args.T, "dt": args.dt,
                   "nu_ext_ratio": args.nu_ext_ratio, "V_th": args.vth0, "seed": args.seed,
                   "seeds": args.seeds, "theta_ee": args.theta_ee, "AR": args.AR,
                   "n_bins_per_axis": args.n_bins_per_axis,
                   "g_gaba_scale": args.g_gaba_scale if args.slow_var == "egaba" else None},
        "detector": {"trace_bin_ms": TRACE_BIN_MS, "thresh_k": args.thresh_k,
                     "thresh_floor": args.thresh_floor, "min_gap_bins": args.min_gap_bins,
                     "r_kick": args.r_kick, "far_radius_mm": far_radius},
        "default_caps": DEFAULT_CAPS,
    }
    json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    print(f"[A1 quasi-static] {state_label} (substrate {summary['substrate']}, L={args.L}): "
          f"{len(all_classes)} events, rate={summary['event_rate_hz_per_seed']}Hz/seed, "
          f"return_p={summary['return_probability']}, R={summary['R_fractions']}, "
          f"size_med={summary['size_bins']['median']}, dur_med={summary['duration_ms']['median']}ms")
    print(f"[A1 quasi-static] wrote -> {out_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description="M3A-A1 quasi-static slow-state spontaneous pilot")
    p.add_argument("--slow-var", choices=SLOW_VARS, default="off",
                   help="ONE frozen slow variable per run (off=interictal baseline anchor)")
    p.add_argument("--level", type=float, default=1.0,
                   help="frozen value: z in [0,1] | phi OFFSET mV | gK mV-equiv | egaba reversal mV")
    p.add_argument("--g-gaba-scale", type=float, default=1.0, help="shunt conductance scale (egaba only)")
    p.add_argument("--tag", type=str, default=None, help="output subdir name (default from var+level)")
    p.add_argument("--L", type=float, default=20.0); p.add_argument("--density", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--T", type=float, default=8000.0, help="record length (ms); spontaneous needs long T")
    p.add_argument("--dt", type=float, default=0.1); p.add_argument("--nu-ext-ratio", type=float, default=0.6)
    p.add_argument("--theta-ee", type=float, default=45.0); p.add_argument("--AR", type=float, default=2.0)
    p.add_argument("--vth0", type=float, default=18.0); p.add_argument("--n-bins-per-axis", type=int, default=5)
    p.add_argument("--r-kick", type=float, default=0.5); p.add_argument("--far-radius-mm", type=float, default=None)
    p.add_argument("--thresh-k", type=float, default=6.0, help="event threshold = median + k*MAD")
    p.add_argument("--thresh-floor", type=float, default=0.01, help="min event threshold (active fraction)")
    p.add_argument("--min-gap-bins", type=int, default=3, help="merge events < this many quiet bins apart")
    p.add_argument("--out-dir", type=str, default="results/topic4_sef_hfo/m3a_slowvars/quasistatic")
    p.add_argument("--run", action="store_true")
    args = p.parse_args(argv)
    if not args.run:
        print("[A1 quasi-static] PILOT-FIRST: pass --run. Nothing was run."); return 0
    run(args); return 0


if __name__ == "__main__":
    sys.exit(main())
