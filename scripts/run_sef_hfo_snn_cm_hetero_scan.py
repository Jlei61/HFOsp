"""Step-3 cm heterogeneity parameter scan — find the operating range where a pathology
core on the VALIDATED cm read-out sheet (L=20/d=100, four-control closed) ignites and
propagates a directional event readable through virtual SEEG.

Reuses scripts/run_sef_hfo_snn_cm_wave_read.py --core (byte-identical read-out) per cell.
The single empirical axis (advisor 2026-06-09): core_mean x core_std. Operating point
(g=3.6/drive=0.6), sheet (L=20/d=100), core radius (~= measured E->E reach 1.5mm) are
FIXED, not scanned. Core at the kick origin so the kick + core co-seed the wave.

Two regimes the scan separates (= the plan's evoked_clean vs core_prekick_ignited split):
  - EVOKED-READABLE window: core_mean where the kicked event reads directionally
    (n_part>=7 AND axis_err<25 on BOTH reads) AND the no-kick OFF run stays quiet
    (evoked_clean=True). This is the Step-3-ready kick-evoked regime (read-out validated).
  - SELF-IGNITION boundary: the core_mean where the OFF run first self-ignites
    (core_self_ignite=True). Below it = the spontaneous/pathology leg (NOT-YET: needs
    multi-seed/finite-size, per the snn_heterogeneity_result memory).

Writes per-cell JSON via the runner + an aggregate cm_hetero_scan_summary.json that encodes
the two ranges (the conclusion, not just the cells).
"""
import os
import sys
import json
import subprocess

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_wave"
RUNNER = "scripts/run_sef_hfo_snn_cm_wave_read.py"

# FIXED (validated read-out + measured reach); the scan varies only core_mean x core_std.
FIXED = ["--L", "20", "--density", "100", "--nc", "6", "--r-kick", "0.6",
         "--theta", "45", "--kick-mode", "negend", "--core", "--core-r", "1.5",
         "--core-at", "kick", "--seed", "1"]
CORE_MEANS = [17.0, 16.0, 15.0, 14.0, 13.0]   # base V_th=18 -> shift 1..5 mV
CORE_STDS = [("wide", 1.5), ("narrow", 0.5)]


def run_cell(mean, sname, std):
    tag = f"L20d100_CORE_m{mean:g}_{sname}"
    cmd = ["python3", RUNNER, *FIXED, "--core-mean", str(mean), "--core-std", str(std),
           "--tag", tag]
    log = os.path.join(OUT, f"{tag}.log")
    with open(log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    p = os.path.join(OUT, f"wave_read_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def readable(r):
    return (r["axis_err"] is not None and r["n_part"] >= 7 and r["axis_err"] < 25.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = []
    for sname, std in CORE_STDS:
        for mean in CORE_MEANS:
            j = run_cell(mean, sname, std)
            if j is None:
                cells.append(dict(core_mean=mean, spread=sname, ok=False))
                continue
            ig = j["ignition"]
            both_read = readable(j["firing"]) and readable(j["current_lfp"])
            cells.append(dict(
                core_mean=mean, spread=sname, n_core=j["core"]["n_core"],
                ign_on=ig["peak_inst"], ign_off=ig["off_peak_inst"],
                self_ignite=ig["core_self_ignite"], evoked_clean=ig["evoked_clean"],
                returned=ig["returned"], oracle=(j["oracle"] or {}).get("axis_deg"),
                fir_n=j["firing"]["n_part"], fir_err=j["firing"]["axis_err"],
                lfp_n=j["current_lfp"]["n_part"], lfp_err=j["current_lfp"]["axis_err"],
                both_read=both_read,
                step3_evoked_ready=bool(both_read and ig["evoked_clean"]
                                        and ig["returned"] and ig["peak_inst"] > 5e-3)))
            print(f"  m={mean:g} {sname}: ign_on={ig['peak_inst']:.3f} ign_off={ig['off_peak_inst']:.3f} "
                  f"self={ig['core_self_ignite']} clean={ig['evoked_clean']} read={both_read} "
                  f"-> step3_evoked_ready={cells[-1]['step3_evoked_ready']}", flush=True)

    # derive the two ranges (the conclusion)
    evoked_ready = sorted({c["core_mean"] for c in cells if c.get("step3_evoked_ready")})
    self_ig = sorted({c["core_mean"] for c in cells if c.get("self_ignite")})
    summary = dict(
        config=dict(L=20, density=100, nc=6, core_r=1.5, core_at="kick", theta=45,
                    r_kick=0.6, op_point="g=3.6/drive=0.6", reach_mm=1.46),
        cells=cells,
        step3_evoked_ready_core_means=evoked_ready,
        self_ignition_core_means=self_ig,
        verdict=dict(
            evoked_readable_window=(f"core_mean in {evoked_ready}" if evoked_ready else "NONE"),
            self_ignition_onset=(min(self_ig) if self_ig else None)))
    with open(os.path.join(OUT, "cm_hetero_scan_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== SUMMARY ===")
    print("Step-3 evoked-readable core_means:", evoked_ready or "NONE")
    print("self-ignition (spontaneous leg) core_means:", self_ig or "NONE")
    with open(os.path.join(OUT, "cm_hetero_scan.done"), "w") as f:
        f.write("DONE")


if __name__ == "__main__":
    main()
