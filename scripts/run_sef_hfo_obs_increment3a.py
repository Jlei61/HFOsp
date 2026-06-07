"""Increment-3a rate-model parity runner (TRACKED): the SAME virtual-SEEG observation
chain run on the LIF RATE FIELD instead of the SNN, to confirm the modality is
substrate-independent (the bi-model gate that Step 3 consumes).

Mirrors scripts/run_sef_hfo_obs_inc2_smoke.py EXACTLY — same locked verdict bars
(AXIS_ERR_MAX / KDIR / PART_MIN / TAU_FAIL), same estimator (endpoint_centroid_axis),
same event window (event_window_for_run), same four controls. ONLY the substrate
changes: SNN build/simulate_kick  ->  mean_field + integrate_lif_field(return_frames)
with a pulse_stim_fn; snn_event_envelope -> rate_event_envelope.

Substrate parity notes:
  * The rate _grid is CENTERED AT ORIGIN (x in [-L/2, +L/2]); SNN neurons are in
    [0, L]. So center = (0, 0) here, NOT (L/2, L/2).
  * frame_dt = integration dt (the rate field is already smooth; no spike binning).
  * iso control = ell_par == ell_perp (AR=1); aniso = ELL_PAR/ELL_PERP (AR=2).
  * Deterministic pulse (the homogeneous rate field's noise-spontaneous events were
    honest-NULL, 2026-06-04) — parity is about the read-out chain, not noise.

Geometry-smoke ONLY: sweeps montage/kick/sheet GEOMETRY; NEVER the verdict bars and
never the operating point beyond the locked excitable window. See plan
docs/superpowers/plans/2026-06-07-sef-hfo-virtual-seeg-observation-increment3.md.
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import (mean_field, integrate_lif_field, _grid, DETECT,   # noqa: E402
                             ELL_PAR, ELL_PERP, L_INH)
from src.sef_hfo_rate_adapter import pulse_stim_fn, rate_event_envelope        # noqa: E402
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,  # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import event_window_for_run                        # noqa: E402

# LOCKED verdict bars — identical to Increment-2; NEVER tuned by the geometry smoke
AXIS_ERR_MAX = 25.0
KDIR = 3
PART_MIN = 2 * KDIR + 1          # 7
TAU_FAIL = 0.3

# Operating point + pulse: the locked Step-0b excitable window + validated disk pulse
RATIO = 0.6                       # quiet drive (self_limited_propagation across 0.5-1.3)
DT, T_MAX = 0.25, 250.0
PULSE = dict(radius=2.0, amp=8.0, t_on=0.0, t_off=30.0)   # _STIM_R / A / _STIM_T


def montage(center, pitch, n_contacts, shafts):
    """N non-parallel shafts at `pitch` mm, all centered on `center`."""
    return merge_montages([build_shaft(np.deg2rad(a), pitch, n_contacts, center, chr(65 + i))
                           for i, a in enumerate(shafts)])


def _integrate(op, theta_rad, AR, kick_xy, n, L):
    """One on-run + one off-run (no kick) of the rate field; returns frame stacks."""
    ell_par = ELL_PAR if AR != 1 else float(np.sqrt(ELL_PAR * ELL_PERP))
    ell_perp = ELL_PERP if AR != 1 else float(np.sqrt(ELL_PAR * ELL_PERP))
    sf = pulse_stim_fn(tuple(kick_xy), n=n, L=L, **PULSE)
    on = integrate_lif_field(op, sf, dt=DT, t_max=T_MAX, theta_EE=theta_rad, n=n, L=L,
                             ell_par=ell_par, ell_perp=ell_perp, l_inh=L_INH,
                             return_frames=True)
    off = integrate_lif_field(op, lambda t: 0.0, dt=DT, t_max=T_MAX, theta_EE=theta_rad,
                              n=n, L=L, ell_par=ell_par, ell_perp=ell_perp, l_inh=L_INH,
                              return_frames=True)
    return on[-1], off[-1]               # rE_frames (nsteps, n, n)


def read(op, theta_rad, AR, kick_xy, m, theta_ref_rad, n, L, kernel_width):
    """One single-end-kick read: n_part, axis_err (undirected vs theta_ref), readability.
    SAME chain as the SNN: rate_event_envelope -> event_window_for_run -> extract_lagpat
    -> endpoint_centroid_axis."""
    on_f, off_f = _integrate(op, theta_rad, AR, kick_xy, n, L)
    env = rate_event_envelope(on_f, n, L, m, kernel_width)
    env_ref = rate_event_envelope(off_f, n, L, m, kernel_width)
    agg, aggr = env.mean(axis=0), env_ref.mean(axis=0)
    win = event_window_for_run(agg, aggr, DT)
    if win is None:
        return {"n_part": 0, "axis_err": None, "readability": None, "win": None}
    art = extract_lagpat(env, DT, [win], float(env.min()),
                         0.5 * (float(env.max()) - float(env.min())), 0.5, DT)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    eps_deg = 0.5 * float(np.linalg.norm(m.contacts[1] - m.contacts[0]))   # 0.5 * pitch
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=eps_deg)
    rd = direction_readability(r0, b0, art.contact_coords)
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_ref_rad)), 1)
    return {"n_part": n_part, "axis_err": err,
            "readability": (None if rd is None or rd != rd else round(float(rd), 3)),
            "win": [round(win[0], 1), round(win[1], 1)]}


def footprint(op, theta_rad, AR, kick_xy, n, L):
    """Lobe footprint of the on-run event (advisor gate: size the lobe before montage)."""
    on_f, _ = _integrate(op, theta_rad, AR, kick_xy, n, L)
    thr = op["nuE"] + DETECT
    active_any = (on_f > thr).any(axis=0)
    X, Y = _grid(n, L)
    if not active_any.any():
        return {"active": False}
    xs, ys = X[active_any], Y[active_any]
    return {"active": True, "x_span_mm": round(float(xs.max() - xs.min()), 2),
            "y_span_mm": round(float(ys.max() - ys.min()), 2),
            "cx": round(float(xs.mean()), 2), "cy": round(float(ys.mean()), 2),
            "area_frac": round(float(active_any.mean()), 3)}


def run_smoke(L=24.0, n=96, pitch=4.0, n_contacts=4, shafts=(10.0, 70.0, 130.0),
              kick_end_frac=0.6, kicktrack_off=0.7):
    center = np.array([0.0, 0.0])                 # rate grid is origin-centered
    m = montage(tuple(center), pitch, n_contacts, shafts)
    kernel_width = 0.5 * pitch
    half_extent = pitch * (n_contacts - 1) / 2.0
    op = mean_field(RATIO)
    cfg = dict(substrate="lif_rate_field", L=L, n=n, pitch=pitch, n_contacts=n_contacts,
               shafts=shafts, kick_end_frac=kick_end_frac, kicktrack_off=kicktrack_off,
               ratio=RATIO, montage_half_extent_mm=round(half_extent, 3),
               kernel_width_mm=kernel_width)
    res = {"config": cfg, "C_track": {}, "kick_track": {}, "iso": {}}

    # --- lobe footprint at theta_EE=45 (advisor sizing gate, before trusting montage) ---
    end45 = center + kick_end_frac * (L / 2) * np.array([np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))])
    res["footprint_45"] = footprint(op, np.deg2rad(45), 2.0, end45, n, L)

    # C-track: kick at the theta_EE end (AR=2)
    for th in (0.0, 45.0, 90.0):
        end = center + kick_end_frac * (L / 2) * np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
        res["C_track"][f"{th:g}deg"] = read(op, np.deg2rad(th), 2.0, end, m, np.deg2rad(th),
                                            n, L, kernel_width)

    # kick-track: theta_EE FIXED 45, kick offset PERPENDICULAR (wave sweeps montage) -> axis stays 45
    perp = np.array([-np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))])
    for d in (-kicktrack_off, kicktrack_off):
        kxy = center + d * (L / 2) * perp
        res["kick_track"][f"perp{d:+.2f}"] = read(op, np.deg2rad(45), 2.0, kxy, m,
                                                  np.deg2rad(45), n, L, kernel_width)

    # iso: AR=1 single-end at the 45 end (expected honest fizzle / no axis)
    res["iso"] = read(op, np.deg2rad(45), 1.0, end45, m, np.deg2rad(45), n, L, kernel_width)

    def ok_track(r):
        return r["axis_err"] is not None and r["n_part"] >= PART_MIN and r["axis_err"] < AXIS_ERR_MAX
    ctrack_ok = all(ok_track(res["C_track"][k]) for k in ("45deg", "90deg"))
    kt_ok = all(ok_track(r) for r in res["kick_track"].values())
    iso = res["iso"]
    iso_honest = (iso["n_part"] < PART_MIN) or (iso["readability"] is not None and iso["readability"] < TAU_FAIL) or (iso["axis_err"] is None)
    res["SMOKE_PASS"] = bool(ctrack_ok and kt_ok and iso_honest)
    res["checks"] = {"ctrack_45_90_ok": ctrack_ok, "kick_track_stays_45": kt_ok,
                     "iso_honest_fizzle_or_noaxis": iso_honest}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=24.0)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--n-contacts", type=int, default=4)
    ap.add_argument("--kick-end-frac", type=float, default=0.6)
    ap.add_argument("--kicktrack-off", type=float, default=0.7)
    a = ap.parse_args()
    res = run_smoke(L=a.L, n=a.n, pitch=a.pitch, n_contacts=a.n_contacts,
                    kick_end_frac=a.kick_end_frac, kicktrack_off=a.kicktrack_off)
    out = "results/topic4_sef_hfo/observation_layer/increment3a_rate_parity"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "smoke_rate_parity_verdict.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None)
    print("footprint@45:", res["footprint_45"])
    print("SMOKE_PASS =", res["SMOKE_PASS"], "| checks:", res["checks"])
    print("C_track:", {k: (v["axis_err"], v["n_part"]) for k, v in res["C_track"].items()})
    print("kick_track:", {k: (v["axis_err"], v["n_part"]) for k, v in res["kick_track"].items()})
    print("iso:", {"n_part": res["iso"]["n_part"], "readability": res["iso"]["readability"],
                   "axis_err": res["iso"]["axis_err"]})


if __name__ == "__main__":
    main()
