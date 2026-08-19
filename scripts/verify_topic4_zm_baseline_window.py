#!/usr/bin/env python
"""Decide whether the low-activity state may be called a baseline.

The rule is not a clock reading. A window earns the name `baseline` only if the
population rate AND the h-weighted z AND the h-weighted m all sit inside the
Z/M-off support. The 2 s checkpoint already fails the rate clause, which is why
it is labelled `early_transition`.

With Z/M off there is no z or m at all, so the two slow clauses would be
untestable. The reference therefore comes from a SHADOW run: the same seed with
z and m integrating but never reaching the membrane. That run's trajectory is
bit-identical to the Z/M-off run (tests/test_zm_passive_mode.py), so its slow
variables describe exactly the network the baseline claim is about.

Two references are reported, and both must be read:

  pooled     the user's literal definition -- q95 over every 500 ms window of
             the 20 s shadow run. z has tau = 5 s, so over 20 s it drifts a
             long way, making this a LOOSE bound that an early window passes
             almost automatically.
  matched    the shadow run's own value in the SAME [500, 1000] ms window. This
             is the tight, informative comparison and is the one to quote.

`found=False` is a legitimate outcome: the figure is then labelled
`early transition vs pre-ictal` and the word baseline is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_baseline_discovery import (  # noqa: E402
    ema_rate_hz, find_baseline_window, window_medians)

WORKERS = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/workers"
OUT = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/baseline_window_verdict.json"


def _load(stem):
    npz = WORKERS / f"{stem}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)
    with np.load(npz, allow_pickle=False) as z:
        return {"active_fraction": np.asarray(z["active_fraction"], float),
                "bin_ms": float(z["active_fraction_bin_ms"]),
                "t": np.asarray(z["mz_h_weighted_time_ms"], float),
                "z": np.asarray(z["mz_h_weighted_z_weighted_mean"], float),
                "m": np.asarray(z["mz_h_weighted_m_weighted_mean"], float)}


def _window_median(t, values, lo, hi):
    inside = (t >= lo) & (t < hi)
    return float(np.median(values[inside])) if inside.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1801, 1802, 1803])
    ap.add_argument("--candidate-id", default="joint_04_control")
    ap.add_argument("--window-ms", type=float, default=500.0)
    ap.add_argument("--burn-in-ms", type=float, default=500.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    on, shadow = {}, {}
    for seed in args.seeds:
        on[seed] = _load(f"{args.candidate_id}_seed_{seed}")
        shadow[seed] = _load(f"{args.candidate_id}_seed_{seed}_zmshadow")
        if shadow[seed]["z"].size == 0:
            raise RuntimeError(f"shadow run for seed {seed} recorded no z/m; it was "
                               "not run with --zm-mode shadow")

    # ---- pooled Z/M-off support, one distribution per quantity ----
    rate_pool = np.concatenate([window_medians(ema_rate_hz(s["active_fraction"], s["bin_ms"]),
                                               s["bin_ms"], args.window_ms)
                                for s in shadow.values()])
    def _slow_pool(key):
        out = []
        for s in shadow.values():
            edges = np.arange(0.0, s["t"].max() + args.window_ms, args.window_ms)
            for lo, hi in zip(edges[:-1], edges[1:]):
                v = _window_median(s["t"], s[key], lo, hi)
                if np.isfinite(v):
                    out.append(v)
        return np.asarray(out, float)
    z_pool, m_pool = _slow_pool("z"), _slow_pool("m")
    # The bound is on DISINHIBITION, not on z. z falls from 1, so an upper bound
    # on z would demand that a candidate window be at least as disinhibited as
    # the reference's 95th percentile -- it rejects the quietest windows, which
    # is what it did on all three seeds before this was corrected.
    disinhibition_pool = 1.0 - z_pool

    support = {"rate_q95": float(np.percentile(rate_pool, 95)),
               "disinhibition_q95": float(np.percentile(disinhibition_pool, 95)),
               "z_floor_equivalent": float(1.0 - np.percentile(disinhibition_pool, 95)),
               "m_q95": float(np.percentile(m_pool, 95)),
               "n_windows": {"rate": int(rate_pool.size), "z": int(z_pool.size),
                             "m": int(m_pool.size)}}

    per_seed, all_found, windows = {}, True, []
    for seed in args.seeds:
        s = on[seed]
        verdict = find_baseline_window(
            s["active_fraction"], s["bin_ms"], rate_q95=support["rate_q95"],
            z_trace=s["z"], m_trace=s["m"], zm_time_ms=s["t"],
            disinhibition_q95=support["disinhibition_q95"],
            m_q95=support["m_q95"],
            burn_in_ms=args.burn_in_ms, window_ms=args.window_ms,
            search_end_ms=None)
        entry = {"pooled_reference": verdict}
        if verdict["found"]:
            windows.append(tuple(verdict["window_ms"]))
            lo, hi = verdict["window_ms"]
            ref = shadow[seed]
            ema_on = ema_rate_hz(s["active_fraction"], s["bin_ms"])
            ema_ref = ema_rate_hz(ref["active_fraction"], ref["bin_ms"])
            i0, i1 = int(lo / s["bin_ms"]), int(hi / s["bin_ms"])
            # The tight comparison: the SAME window on the shadow trajectory.
            entry["time_matched_reference"] = {
                "window_ms": [lo, hi],
                "rate_hz": {"zm_on": float(np.median(ema_on[i0:i1])),
                            "shadow": float(np.median(ema_ref[i0:i1]))},
                "disinhibition": {
                    "zm_on": 1.0 - _window_median(s["t"], s["z"], lo, hi),
                    "shadow": 1.0 - _window_median(ref["t"], ref["z"], lo, hi)},
                "m": {"zm_on": _window_median(s["t"], s["m"], lo, hi),
                      "shadow": _window_median(ref["t"], ref["m"], lo, hi)}}
        else:
            all_found = False
        per_seed[seed] = entry

    same_window = len(set(windows)) == 1 if windows else False
    report = {
        "status": "ZM_BASELINE_WINDOW_VERDICT",
        "definition": ("first post-burn-in window whose population rate, h-weighted z "
                       "and h-weighted m all sit inside the Z/M-off support"),
        "reference_construction": ("shadow run: z and m integrate but never reach the "
                                   "membrane, so the trajectory is bit-identical to "
                                   "Z/M-off while the slow variables are observable"),
        "pooled_support": support,
        "per_seed": per_seed,
        "all_seeds_found": bool(all_found),
        "same_window_on_every_seed": bool(same_window),
        "window_ms": list(windows[0]) if same_window else None,
        "label_to_use": ("baseline" if all_found and same_window
                         else "early transition vs pre-ictal"),
        "caveat": ("the pooled q95 spans 20 s of z drift at tau_z = 5 s and is a loose "
                   "bound; quote the time_matched_reference numbers, which compare the "
                   "same window on a trajectory that never transitions"),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("all_seeds_found", "same_window_on_every_seed", "window_ms",
                       "label_to_use", "pooled_support")}, indent=2))


if __name__ == "__main__":
    main()
