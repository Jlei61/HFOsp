"""THROWAWAY smoke for Increment-2 (NOT the formal runner). Per the 2026-06-07 review:
run ONLY theta_EE=45/AR=2 center kick + AR=1 center kick, confirm "can read the axis /
can't read a fake axis", and find observation knobs to FREEZE. NO full four-control run,
NO threshold tuning. If the estimator can't pass the pre-lock here -> STOP, report.

Verdict bars (LOCKED, never tuned here): AR=2 -> axis within 25deg of 45 AND ratio>1.3;
AR=1 -> ratio<1.3 (or degenerate None). Observation knobs (pitch/front_ms/kernel_width/
bin_ms/smooth_ms) ARE swept here to find a working read, then frozen.
"""
import sys
import os
import json
import numpy as np

ENG = os.path.join("src", "snn_engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta            # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402 (patched: kick_center)

from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,
                                     attach_geometry, onset_front_axis, angle_error_deg)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run

# substrate (smoke scale: big enough that an event is sub-global so an onset front is readable)
L, DENSITY, T, DT, DRIVE = 3.0, 1000.0, 200.0, 0.1, 0.6
CENTER = (L / 2, L / 2)
AXIS_ERR_MAX, RATIO_MIN = 25.0, 1.3      # LOCKED verdict bars (NOT tuned)


def run_condition(theta_deg, AR, seed=1):
    """Build net, CENTER kick + ref, return (E_spk_bool, posE, dt) for kick + ref."""
    p = Params(g=3.6, L=L, density=DENSITY, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(theta_deg), AR=AR, verbose=False)
    posE = net["pos"][:NE]
    nu_theta = compute_nu_theta(p)[0]
    net["rng"] = np.random.default_rng(seed)
    rk = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=list(CENTER))
    net["rng"] = np.random.default_rng(seed)
    rr = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(CENTER))
    return rk["E_spk_bool"], rr["E_spk_bool"], posE


def montage(pitch, nc=8, shafts=(10.0, 100.0)):
    return merge_montages([build_shaft(np.deg2rad(shafts[0]), pitch, nc, CENTER, "A"),
                           build_shaft(np.deg2rad(shafts[1]), pitch, nc, CENTER, "B")])


def read(spk_k, spk_r, posE, pitch, front_ms, kernel_width, bin_ms, smooth_ms):
    m = montage(pitch)
    env, fdt, agg = snn_event_envelope(spk_k, posE, m, DT, bin_ms=bin_ms,
                                       smooth_ms=smooth_ms, kernel_width=kernel_width)
    _, _, aggr = snn_event_envelope(spk_r, posE, m, DT, bin_ms=bin_ms,
                                    smooth_ms=smooth_ms, kernel_width=kernel_width)
    win = event_window_for_run(agg, aggr, fdt)
    if win is None:
        return {"win": None}
    art = extract_lagpat(env, fdt, event_windows=[win], participation_floor=float(env.min()),
                         participation_margin=0.5 * (float(env.max()) - float(env.min())),
                         timing_frac=0.5, tie_tol=fdt)
    art = attach_geometry(art, m)
    n_part = int(art.bools[:, 0].sum())
    angle, ratio, nf = onset_front_axis(art.lag_raw[:, 0], art.bools[:, 0],
                                        art.contact_coords, front_ms)
    return {"win": win, "n_part": n_part, "n_front": nf, "angle": angle, "ratio": ratio}


def main():
    print(f"[smoke] building AR=2 theta=45 (L={L}, density={DENSITY}, drive={DRIVE}) ...", flush=True)
    k2, r2, posE2 = run_condition(45.0, 2.0)
    print("[smoke] building AR=1 (iso) theta=0 ...", flush=True)
    k1, r1, posE1 = run_condition(0.0, 1.0)

    # event footprint (radius of active E region during the event) vs montage extent
    def footprint_radius(spk):
        active = spk[int(120 / DT):int(180 / DT)].any(axis=0)
        if not active.any():
            return 0.0
        c = posE2[active].mean(0)
        return float(np.percentile(np.linalg.norm(posE2[active] - c, axis=1), 90))
    fr = footprint_radius(k2)
    print(f"[smoke] AR=2 event footprint p90 radius ~ {fr:.2f} mm", flush=True)

    out = {"locked": {"axis_err_max": AXIS_ERR_MAX, "ratio_min": RATIO_MIN,
                      "footprint_p90_mm": fr}, "sweep": []}
    best = None
    for pitch in (0.25, 0.35, 0.45):
        for front_ms in (6.0, 8.0, 12.0):
            for kw in (0.2, 0.3):
                a2 = read(k2, r2, posE2, pitch, front_ms, kw, 2.0, 5.0)
                a1 = read(k1, r1, posE1, pitch, front_ms, kw, 2.0, 5.0)
                extent = pitch * (8 - 1) / 2.0      # half-extent of an 8-contact shaft
                rec = {"pitch": pitch, "front_ms": front_ms, "kernel_width": kw,
                       "montage_half_extent_mm": extent, "AR2": a2, "AR1": a1}
                # pass: extent > footprint; AR2 reads ~45 + ratio>1.3; AR1 ratio<1.3 (or None)
                ar2_ok = (a2.get("angle") is not None and a2.get("ratio") is not None
                          and np.isfinite(a2["ratio"]) and a2["ratio"] > RATIO_MIN
                          and angle_error_deg(a2["angle"], 45.0) < AXIS_ERR_MAX
                          and a2.get("n_part", 0) >= 7)
                ar1_ok = (a1.get("angle") is None or a1.get("ratio") is None
                          or (np.isfinite(a1["ratio"]) and a1["ratio"] < RATIO_MIN))
                rec["pass"] = bool(extent > fr and ar2_ok and ar1_ok)
                out["sweep"].append(rec)
                if rec["pass"] and best is None:
                    best = rec
    out["FROZEN"] = best
    out["SMOKE_PASS"] = best is not None
    os.makedirs("results/topic4_sef_hfo/observation_layer/increment2_snn_slice", exist_ok=True)
    p = "results/topic4_sef_hfo/observation_layer/increment2_snn_slice/smoke_verdict.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: None)
    print("[smoke] SMOKE_PASS =", out["SMOKE_PASS"], "| frozen knobs:", best, flush=True)
    print("[smoke] verdict ->", p, flush=True)


if __name__ == "__main__":
    main()
