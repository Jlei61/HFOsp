"""Calibrate the M3A-A2 phase mapping ONCE for a substrate (SIGN calibration only).

Runs the frozen-q engine sign test for q_core / q_global and writes a calibrated mapping
+ ranges + a calibration report. The sweep references this dir with --calibration-dir;
without it the sweep stays fail-closed (pre-calibration scaffold, cond1=false).

This is SIGN calibration: it locks the q -> excitability DIRECTION, NOT a fitted rate
response curve (the transform a/b/input_min/input_max stay the normalized placeholders).
If M3B needs to interpret coordinate-distance magnitudes, add response-curve calibration.

Usage:  python scripts/calibrate_a2_mapping.py [sim args] --out DIR
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "src/snn_engine")
import scripts.plot_a2p_synchronous_burst_figure as F  # noqa: E402
from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges  # noqa: E402
from src.sef_hfo_m3a_calibration import calibrate_axisbreak_mapping  # noqa: E402

_ENGINE_FILES = ("src/snn_engine/slow_vars.py", "src/snn_engine/kick_probe.py",
                 "src/snn_engine/params.py")


def _engine_sha():
    h = hashlib.sha256()
    for f in _ENGINE_FILES:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=2000.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.7)
    ap.add_argument("--dephase", type=float, default=0.3)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    F._add_a2_args(ap)
    ap.add_argument("--mapping-id", default="m3a_a2_sign_cal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sha = _engine_sha()
    mapping, ranges = default_precalib_mapping_and_ranges(a.mapping_id)
    cal, sts = calibrate_axisbreak_mapping(a, mapping, engine_sha=sha)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(out / "calibrated_mapping.json", "w"), indent=2)
    json.dump(ranges, open(out / "phase_coord_ranges.json", "w"), indent=2)
    report = {
        "calibration_kind": "sign_only",
        "caveat": ("sign-calibrated normalized phase mapping: DIRECTION locked, quantitative "
                   "response curve NOT fitted"),
        "engine_sha": sha,
        "q_values": [0.4, 0.7, 1.0],
        "mapping_id": a.mapping_id,
        "substrate": {"L": a.L, "density": a.density, "theta": a.theta, "AR": a.AR,
                      "core_mean": a.core_mean, "a2_mode": a.a2_mode},
        "sign_tests": sts,
    }
    json.dump(report, open(out / "calibration_report.json", "w"), indent=2)
    print("[calibrate] engine_sha=%s  %s  -> %s" % (
        sha,
        " ".join("%s:slope=%d,passed=%s" % (k, v["observed_slope_sign"], v["passed"])
                 for k, v in sts.items()),
        out))


if __name__ == "__main__":
    main()
