"""A2 axis-break experiment — ONE cell.

Question (user, 2026-06-26): the committed best point only drained the LOCAL core inhibition
(core_only -> q_global stayed 1.0), so events only grew along the inter-core axis (grad_align=1.0).
The seizure state we want is OFF-AXIS / global: activity stops respecting the wiring's axial
texture, the whole sheet engages, and the axial read-out loses meaning. This sweep engages the
GLOBAL inhibitory tank (mode=two_tank, q_global drops with whole-sheet firing) and pushes it down,
and asks whether that flips events from axis-aligned (align~1, isotropy<1) to off-axis/global
(align down, isotropy up). Controls: core_only at the same drive isolate "global tank vs just more
drive".

Re-uses the bit-faithful in-process sim (scripts.plot_a2p_synchronous_burst_figure). Dumps a small
per-cell summary JSON (medians + the biggest event + did the global tank actually deplete + a
runaway guard) — NO spike matrix, NO figure.

Usage:  python scripts/run_a2_axisbreak_sweep.py [sim args] --a2-mode two_tank ... --tag X --out DIR
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.getcwd()); sys.path.insert(0, "src/snn_engine")
import scripts.plot_a2p_synchronous_burst_figure as F  # noqa: E402  (sim + read_events live here)
import scripts.run_sef_hfo_snn_cm_spontaneous_readout as C  # noqa: E402


def _med(xs, k):
    v = [e[k] for e in xs]
    return float(np.median(v)) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--drive", type=float, default=0.6)
    ap.add_argument("--T", type=float, default=8000.0)
    ap.add_argument("--core-mean", type=float, default=17.5)
    ap.add_argument("--core-std", type=float, default=1.0)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.7)
    ap.add_argument("--dephase", type=float, default=0.3)
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    F._add_a2_args(ap)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration-dir", default=None,
                    help="dir with calibrated_mapping.json + phase_coord_ranges.json from "
                         "calibrate_a2_mapping.py; absent -> fail-closed pre-calibration handoff")
    a = ap.parse_args()
    C._engine_guard()
    sim = F.simulate_a2(a)
    events = F.read_events(sim)
    big = [e for e in events if e["n_fired"] >= 3]
    Tsec = sim["spk"].shape[0] * F.DT / 1000.0
    global_rate_hz = float(sim["spk"].sum() / sim["NE"] / Tsec)
    biggest = max(big, key=lambda e: e["r95_src"]) if big else None
    # off-axis GLOBAL = round (isotropy >= 0.5, above the axial max ~0.39) AND global (n_fired >= 18000,
    # well above the biggest AXIAL recruitment wave ~13.8k; the field is NE=32000). Both gates needed:
    # n_fired alone catches a big elongated wave; isotropy alone catches a small round blob. grad_r2
    # is NOT a gate (small local blobs also have low grad_r2); it's reported as "synchronous" sub-flag.
    OFF_ISO, OFF_NFIRED = 0.50, 18000
    def is_off_axis(e):
        return e["isotropy"] >= OFF_ISO and e["n_fired"] >= OFF_NFIRED
    off = [e for e in big if is_off_axis(e)]
    off_axis_frac = len(off) / max(len(big), 1)
    off_sync = [e for e in off if e["grad_r2"] < 0.30]   # off-axis AND no directed front = synchronous
    # SELF-LIMITING / discreteness: duty cycle of the network being "in an event" (low = discrete
    # punctuated bursts that terminate; high = tonic-on). bar = same record_peak rule as read_events.
    af, bin_w = C.active_fraction(sim["spk"], F.DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (af.max() - floor)
    tonic_fraction = float((af > bar).mean())

    # regime — separate "broke the axis" (off list non-empty) from "self-limiting" (terminates):
    tonic = global_rate_hz > 60 or tonic_fraction > 0.5
    if not big or (biggest and biggest["n_fired"] < 1500):
        regime = "quiet_or_tiny"
    elif off and tonic:
        regime = "off_axis_TONIC"                     # broke the axis but stays ON (not self-limiting)
    elif len(off) >= 2:
        regime = "off_axis_SELF_LIMITING"             # TARGET: discrete off-axis events that TERMINATE
    elif len(off) == 1:
        regime = "off_axis_oneshot"                   # one off-axis event (promising, weak)
    elif tonic:
        regime = "tonic_axial"                        # stays on but never broke the corridor
    else:
        regime = "axial_only"

    summary = dict(
        tag=a.tag, mode=a.a2_mode, k_use=a.a2_k_use, drive=a.drive, gk_max=a.a2_gk_max,
        q_min=a.a2_q_min, tau_rec=a.a2_tau_rec, T=a.T, seed=a.seed,
        n_events=len(events), n_big=len(big), n_off_axis=len(off), n_off_sync=len(off_sync),
        off_axis_frac=round(off_axis_frac, 3), tonic_fraction=round(tonic_fraction, 4),
        events_brief=[dict(n_fired=e["n_fired"], isotropy=round(e["isotropy"], 3),
                           reach_perp=round(e["reach_perp"], 2), reach_along=round(e["reach_along"], 2),
                           grad_r2=round(e["grad_r2"], 3), r95=round(e["r95_src"], 2)) for e in big],
        q_core_min=round(sim["q_core_min"], 4), q_global_min=round(sim["q_global_min"], 4),
        global_rate_hz=round(global_rate_hz, 3), regime=regime,
        median_big=dict(grad_align=_med(big, "grad_align"), grad_r2=_med(big, "grad_r2"),
                        isotropy=_med(big, "isotropy"), reach_along=_med(big, "reach_along"),
                        reach_perp=_med(big, "reach_perp"), r95_src=_med(big, "r95_src"),
                        n_fired=_med(big, "n_fired")),
        biggest_event=(dict(t_on=biggest["t_on"], rho_pre=round(biggest["rho_pre"], 3),
                            r95_src=round(biggest["r95_src"], 2), n_fired=biggest["n_fired"],
                            grad_align=round(biggest["grad_align"], 3), grad_r2=round(biggest["grad_r2"], 3),
                            isotropy=round(biggest["isotropy"], 3),
                            reach_along=round(biggest["reach_along"], 2),
                            reach_perp=round(biggest["reach_perp"], 2)) if biggest else None),
    )
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(Path(a.out) / f"summary_{a.tag}.json", "w"), indent=2, default=float)

    # M3A->M3B-R2 handoff artifacts (canonical contract src/sef_hfo_m3_interface.py; fail-closed --
    # the self-audit REFUSES the overlay until A1 calibrates the mapping). event_phase_samples.csv is
    # deferred (needs R_class + the absolute tail_to_baseline definition, contract §9).
    from src.sef_hfo_m3a_export import build_handoff_from_sim, write_handoff_artifacts
    from src.sef_hfo_a2 import event_peak_ms
    for e in events:                            # decision C: real activity-fraction peak (af from above)
        e["t_peak"] = event_peak_ms(af, bin_w, e["t_on"], e["t_off"])
    cal_map = cal_ranges = None                 # P1-1: use the once-calibrated mapping if provided
    if a.calibration_dir:
        cd = Path(a.calibration_dir)
        cal_map = json.load(open(cd / "calibrated_mapping.json"))
        cal_ranges = json.load(open(cd / "phase_coord_ranges.json"))
    h = build_handoff_from_sim(sim, events, F.DT, mapping_id=f"m3a_a2_{a.tag}",
                               gk_enabled=a.a2_gk_max > 0, af=af, bin_w=bin_w, L=a.L,
                               mapping=cal_map, ranges=cal_ranges)
    audit = write_handoff_artifacts(str(Path(a.out) / f"handoff_{a.tag}"), **h)
    _why = "needs gate_A PASS (science)" if audit["cond1_sign_tests_passed"] else "uncalibrated mapping"
    print(f"[{a.tag}] handoff overlay_verdict={audit['overlay_verdict']} ({_why}) "
          f"n_landmarks={len(h['landmark_rows'])}")
    print(f"[{a.tag}] regime={regime} qGmin={summary['q_global_min']} rate={global_rate_hz:.1f}Hz "
          f"tonic_frac={summary['tonic_fraction']} n_off_axis={len(off)}/{len(big)} "
          f"biggest: align={summary['biggest_event']['grad_align'] if biggest else None} "
          f"iso={summary['biggest_event']['isotropy'] if biggest else None} "
          f"grad_r2={summary['biggest_event']['grad_r2'] if biggest else None}")


if __name__ == "__main__":
    main()
