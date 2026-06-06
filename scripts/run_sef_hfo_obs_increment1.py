# scripts/run_sef_hfo_obs_increment1.py
"""Increment-1 known-direction contract gate: run toy wave (30/60/0/90/135 deg) +
C1 (radial) + C2 (synchronous-amplitude) through virtual contacts; emit verdict
JSON + figures. Thresholds are LOCKED (spec §10): τ_pass=0.9, τ_fail=0.3.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sef_hfo_observation import (
    build_shaft, merge_montages, read_direction_from_source, axis_angle_error_deg,
    write_legacy_npz, write_packed_times, write_montage_manifest,
)
from src.sef_hfo_toywave import (
    traveling_wave, radial_source, synchronous_amplitude_source,
)

TAU_PASS = 0.9
TAU_FAIL = 0.3
OUT = Path("results/topic4_sef_hfo/observation_layer/increment1_toywave")


def _montage():
    # even contact count + both shafts through origin -> centroid at origin, no
    # coincident point (matches the C1 centered-symmetric protocol, spec §3.5)
    a = build_shaft(np.deg2rad(10.0), 4.0, 8, (0.0, 0.0), "A")
    b = build_shaft(np.deg2rad(100.0), 4.0, 8, (0.0, 0.0), "B")
    return merge_montages([a, b])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT.parent / "figures").mkdir(parents=True, exist_ok=True)
    montage = _montage()
    verdict = {"tau_pass": TAU_PASS, "tau_fail": TAU_FAIL, "waves": {}, "controls": {}}

    for deg in (0.0, 30.0, 60.0, 90.0, 135.0):
        src = traveling_wave(64, 64.0, np.deg2rad(deg), c=0.4, dt=0.25,
                             t_max=200.0, width=8.0)
        out = read_direction_from_source(src, montage, kernel_width=3.0)
        err = (axis_angle_error_deg(out["axis"], np.deg2rad(deg))
               if out["axis"] is not None else None)
        verdict["waves"][f"{deg:g}deg"] = {
            "spearman": out["spearman"], "axis_err_deg": err,
            "pass": bool(out["spearman"] >= TAU_PASS)}

    c1 = read_direction_from_source(radial_source(64, 64.0, 0.35, 0.25, 160.0, 6.0),
                                    montage, 3.0)
    verdict["controls"]["C1_radial"] = {
        "readability": c1["readability"],
        "must_fail_ok": bool(c1["readability"] < TAU_FAIL)}

    c2 = read_direction_from_source(
        synchronous_amplitude_source(64, 64.0, 0.25, 120.0, 10.0, np.deg2rad(10.0)),
        montage, 3.0)
    r2 = c2["readability"]
    verdict["controls"]["C2_synchronous"] = {
        "readability": r2,
        "must_fail_ok": bool(np.isnan(r2) or r2 < TAU_FAIL)}

    verdict["GATE_PASS"] = bool(
        all(w["pass"] for w in verdict["waves"].values() if w["spearman"] == w["spearman"])
        and verdict["controls"]["C1_radial"]["must_fail_ok"]
        and verdict["controls"]["C2_synchronous"]["must_fail_ok"])

    (OUT / "gate_verdict.json").write_text(json.dumps(verdict, indent=2,
                                                      default=lambda o: None))

    # Figure: read-out axis vs imposed angle for the waves + control markers
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    degs = [0, 30, 60, 90, 135]
    errs = [verdict["waves"][f"{d:g}deg"]["axis_err_deg"] or np.nan for d in degs]
    rhos = [verdict["waves"][f"{d:g}deg"]["spearman"] for d in degs]
    ax[0].bar([str(d) for d in degs], rhos)
    ax[0].axhline(TAU_PASS, ls="--", color="k")
    ax[0].set_title("Toy-wave read-out Spearman vs imposed angle")
    ax[0].set_ylabel("Spearman rho"); ax[0].set_xlabel("imposed direction (deg)")
    ax[1].bar([str(d) for d in degs], errs)
    ax[1].axhline(25.0, ls="--", color="k")
    ax[1].set_title("Endpoint-axis angle error"); ax[1].set_ylabel("deg")
    fig.tight_layout()
    fig.savefig(OUT.parent / "figures" / "increment1_gate.png", dpi=130)
    plt.close(fig)

    # Persist one example artifact (30deg) to prove the legacy write path runs
    ex = read_direction_from_source(
        traveling_wave(64, 64.0, np.deg2rad(30.0), 0.4, 0.25, 200.0, 8.0),
        montage, 3.0)["artifact"]
    write_legacy_npz(ex, OUT / "example30_lagPat_withFreqCent.npz")
    write_packed_times(ex, OUT / "example30_packedTimes_withFreqCent.npy")
    write_montage_manifest(ex, OUT / "example30_montage.json")
    print("GATE_PASS =", verdict["GATE_PASS"])


if __name__ == "__main__":
    main()
