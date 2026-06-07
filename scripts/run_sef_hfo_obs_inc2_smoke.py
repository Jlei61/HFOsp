"""Increment-2 Option-B smoke runner (TRACKED): single-end kick + endpoint_centroid_axis,
per-condition n_part / axis_err / readability / event_window. Geometry-smoke ONLY — sweeps
montage/kick/sheet GEOMETRY; NEVER the verdict bars (AXIS_ERR_MAX / KDIR / TAU_FAIL) and never
seed-picking. See docs/superpowers/plans/2026-06-07-sef-hfo-obs-increment2-optionB.md.

Conditions:
  C-track     : AR=2, kick at the theta_EE END, theta_EE in {0,45,90}.
  kick-track  : AR=2, theta_EE FIXED 45, kick OFFSET PERPENDICULAR to theta_EE (so the
                unidirectional wave SWEEPS the montage) -> axis must STAY ~45 (not follow kick).
  iso         : AR=1, single-end -> expected fizzle (honest INSUFFICIENT; supports anisotropy-necessary).
"""
import sys
import os
import json
import argparse
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta            # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot     # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402 (patched: kick_center)

from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,        # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run             # noqa: E402

# LOCKED verdict bars — NEVER tuned by the geometry smoke
AXIS_ERR_MAX = 25.0
KDIR = 3
PART_MIN = 2 * KDIR + 1          # 7
TAU_FAIL = 0.3
DT, DRIVE, T = 0.1, 0.6, 220.0


def build(theta_deg, AR, L, density, seed=1):
    p = Params(g=3.6, L=L, density=density, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(theta_deg), AR=AR, verbose=False)
    return p, net, net["pos"][:NE], compute_nu_theta(p)[0]


def montage(center, pitch, n_contacts, shafts):
    return merge_montages([build_shaft(np.deg2rad(shafts[0]), pitch, n_contacts, center, "A"),
                           build_shaft(np.deg2rad(shafts[1]), pitch, n_contacts, center, "B")])


def read(p, net, posE, nut, kick_xy, m, theta_ref_rad, kernel_width=0.3):
    """One single-end-kick read: returns n_part, axis_err (undirected vs theta_ref), readability."""
    net["rng"] = np.random.default_rng(p.seed)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(kick_xy))
    net["rng"] = np.random.default_rng(p.seed)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(kick_xy))
    env, fdt, agg = snn_event_envelope(on["E_spk_bool"], posE, m, DT, kernel_width=kernel_width)
    _, _, aggr = snn_event_envelope(off["E_spk_bool"], posE, m, DT, kernel_width=kernel_width)
    win = event_window_for_run(agg, aggr, fdt)
    if win is None:
        return {"n_part": 0, "axis_err": None, "readability": None, "win": None}
    art = extract_lagpat(env, fdt, [win], float(env.min()),
                         0.5 * (float(env.max()) - float(env.min())), 0.5, fdt)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    n_part = int(b0.sum())
    eps_deg = 0.5 * float(np.linalg.norm(m.contacts[1] - m.contacts[0]))   # 0.5 * contact pitch
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=eps_deg)
    rd = direction_readability(r0, b0, art.contact_coords)
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_ref_rad)), 1)
    return {"n_part": n_part, "axis_err": err,
            "readability": (None if rd is None or rd != rd else round(float(rd), 3)),
            "win": None if win is None else [round(win[0], 1), round(win[1], 1)]}


def run_smoke(L=3.0, density=1000.0, pitch=0.45, n_contacts=8, shafts=(10.0, 100.0),
              kick_end_frac=0.6, kicktrack_off=0.7, seed=1):
    center = np.array([L / 2, L / 2])
    m = montage(tuple(center), pitch, n_contacts, shafts)
    half_extent = pitch * (n_contacts - 1) / 2.0
    cfg = dict(L=L, density=density, pitch=pitch, n_contacts=n_contacts, shafts=shafts,
               kick_end_frac=kick_end_frac, kicktrack_off=kicktrack_off,
               montage_half_extent_mm=round(half_extent, 3))
    res = {"config": cfg, "C_track": {}, "kick_track": {}, "iso": {}}

    # C-track: kick at the theta_EE end
    for th in (0.0, 45.0, 90.0):
        p, net, posE, nut = build(th, 2.0, L, density, seed)
        end = center + kick_end_frac * (L / 2) * np.array([np.cos(np.deg2rad(th)), np.sin(np.deg2rad(th))])
        res["C_track"][f"{th:g}deg"] = read(p, net, posE, nut, end, m, np.deg2rad(th))

    # kick-track: theta_EE FIXED 45, kick offset PERPENDICULAR (wave sweeps the montage)
    p, net, posE, nut = build(45.0, 2.0, L, density, seed)
    perp = np.array([-np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))])
    for d in (-kicktrack_off, kicktrack_off):
        kxy = center + d * (L / 2) * perp
        res["kick_track"][f"perp{d:+.2f}"] = read(p, net, posE, nut, kxy, m, np.deg2rad(45.0))

    # iso: AR=1, single-end at the 45 end (expected fizzle)
    p, net, posE, nut = build(45.0, 1.0, L, density, seed)
    end = center + kick_end_frac * (L / 2) * np.array([np.cos(np.deg2rad(45.0)), np.sin(np.deg2rad(45.0))])
    res["iso"] = read(p, net, posE, nut, end, m, np.deg2rad(45.0))

    # verdict (geometry smoke pass = the 3 things; thresholds NOT tuned)
    def ok_track(r):
        return r["axis_err"] is not None and r["n_part"] >= PART_MIN and r["axis_err"] < AXIS_ERR_MAX
    ctrack_ok = all(ok_track(res["C_track"][k]) for k in ("45deg", "90deg"))   # 45 & 90 the binding ones
    kt_ok = all(ok_track(r) for r in res["kick_track"].values())               # axis stays ~45
    iso = res["iso"]
    iso_honest = (iso["n_part"] < PART_MIN) or (iso["readability"] is not None and iso["readability"] < TAU_FAIL) or (iso["axis_err"] is None)
    res["SMOKE_PASS"] = bool(ctrack_ok and kt_ok and iso_honest)
    res["checks"] = {"ctrack_45_90_ok": ctrack_ok, "kick_track_stays_45": kt_ok,
                     "iso_honest_fizzle_or_noaxis": iso_honest}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=3.0)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--pitch", type=float, default=0.45)
    ap.add_argument("--n-contacts", type=int, default=8)
    ap.add_argument("--kick-end-frac", type=float, default=0.6)
    ap.add_argument("--kicktrack-off", type=float, default=0.7)
    a = ap.parse_args()
    res = run_smoke(L=a.L, density=a.density, pitch=a.pitch, n_contacts=a.n_contacts,
                    kick_end_frac=a.kick_end_frac, kicktrack_off=a.kicktrack_off)
    out = "results/topic4_sef_hfo/observation_layer/increment2_snn_slice"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "smoke_optionB_verdict.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None)
    print("SMOKE_PASS =", res["SMOKE_PASS"], "| checks:", res["checks"])
    print("C_track:", {k: (v["axis_err"], v["n_part"]) for k, v in res["C_track"].items()})
    print("kick_track:", {k: (v["axis_err"], v["n_part"]) for k, v in res["kick_track"].items()})
    print("iso:", {"n_part": res["iso"]["n_part"], "readability": res["iso"]["readability"], "axis_err": res["iso"]["axis_err"]})


if __name__ == "__main__":
    main()
