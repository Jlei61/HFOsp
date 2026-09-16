#!/usr/bin/env python3
"""Rule out a locally deep but spatially incoherent oscillation.

Criterion 10 measures how much the *whole sheet's* firing rate moves in one
cycle.  A traveling wave with random phase across the sheet would flatten that
global average even if every small patch were deeply modulated, so the global
number alone cannot separate "tonic everywhere" from "deep but incoherent".

This script recomputes the same modulation-depth estimator on the local spike
counts recorded around each virtual contact, at several radii.  If the local
depth matches the global depth, the shallowness is local and the incoherence
explanation is dead; if local is much larger, it is alive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_tonic_fixed_point import (  # noqa: E402
    population_rate_modulation,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def analyse(npz_path: Path, span_ms=1000.0, minimum_neurons=5):
    with np.load(npz_path) as artifact:
        dt = float(artifact["lfp_dt_ms"])
        rate = np.asarray(artifact["rate_E_hz"], float)
        names = np.asarray(artifact["contact_names"]).astype(str)
        keys = [key for key in artifact.files
                if key.startswith("local_spike_count_r")
                and not key.endswith("n_neurons")]
        local = {key: (np.asarray(artifact[key], float),
                       np.asarray(artifact[f"{key}_n_neurons"], int))
                 for key in sorted(keys)}
    n = int(round(span_ms / dt))
    if len(rate) < n:
        raise RuntimeError("trace shorter than the analysis span")
    glob = population_rate_modulation(rate[-n:], dt_ms=dt)
    radii = {}
    for key, (counts, sizes) in local.items():
        counts = counts[-n:]
        rows = []
        for index, name in enumerate(names):
            if int(sizes[index]) < int(minimum_neurons):
                continue
            local_rate = counts[:, index] / float(sizes[index]) / (dt * 1e-3)
            try:
                got = population_rate_modulation(local_rate, dt_ms=dt)
            except ValueError:
                continue
            rows.append({"contact": str(name),
                         "n_neurons": int(sizes[index]),
                         "dominant_hz": got["dominant_hz"],
                         "mean_rate_hz": got["mean_rate_hz"],
                         "modulation_depth": got["modulation_depth"]})
        if not rows:
            continue
        depths = np.array([row["modulation_depth"] for row in rows])
        freqs = np.array([row["dominant_hz"] for row in rows])
        radii[key] = {
            "n_contacts_scored": len(rows),
            "n_neurons_min": int(min(row["n_neurons"] for row in rows)),
            "n_neurons_max": int(max(row["n_neurons"] for row in rows)),
            "median_local_modulation_depth": float(np.median(depths)),
            "min_local_modulation_depth": float(depths.min()),
            "max_local_modulation_depth": float(depths.max()),
            "median_local_dominant_hz": float(np.median(freqs)),
            "local_over_global_depth_ratio": float(
                np.median(depths) / max(glob["modulation_depth"], 1e-12)),
            "per_contact": rows,
        }
    return {"global": {key: value for key, value in glob.items()
                       if key != "cycle_profile_hz"},
            "by_radius": radii,
            "analysis_span_ms": span_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {"status": "SPATIAL_ZM_LOCAL_DEPTH_ANALYSIS_COMPLETE",
              "question": ("is the shallow global modulation an averaging "
                           "artefact of a spatially incoherent deep rhythm?"),
              "runs": {}}
    for item in args.npz:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        report["runs"][path.stem] = analyse(path)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_json_safe(report), str(out.with_suffix(".json")))
    for name, run in report["runs"].items():
        glob = run["global"]
        print(f"{name}: global f0={glob['dominant_hz']:.0f} Hz "
              f"mean={glob['mean_rate_hz']:.1f} Hz "
              f"depth={glob['modulation_depth']:.4f}")
        for key, value in run["by_radius"].items():
            print(f"   {key}: n={value['n_neurons_min']}-{value['n_neurons_max']} "
                  f"local depth median={value['median_local_modulation_depth']:.4f} "
                  f"(range {value['min_local_modulation_depth']:.4f}-"
                  f"{value['max_local_modulation_depth']:.4f}) "
                  f"local/global={value['local_over_global_depth_ratio']:.2f} "
                  f"local f0={value['median_local_dominant_hz']:.0f} Hz")


if __name__ == "__main__":
    main()
