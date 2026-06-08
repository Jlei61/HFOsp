"""SNN cm-scale TRAVELING-WAVE read (user 2026-06-08, option A + scale-law tuning).

Reframe: at cm scale the SNN event is a sustained traveling FRONT, not a self-limited blob.
So read it as a wave: kick EARLY (t_kick~20ms, so the front has the whole sim to cross), use a
FIXED first-pass window (NOT event_window_for_run, which needs a self-terminating event ->
n=0 for a sustained wave), and read per-contact FIRST-crossing onset times -> endpoint_centroid_axis
(early-onset centroid -> late-onset centroid = propagation direction; first-crossing captures the
FIRST front pass even if periodic-BC re-entry follows). Estimator unchanged (the WF-A perpendicular
trap is onset_front_axis only). Reads firing-envelope AND current-LFP.

Scale law used: mean-field operating point is N-invariant (fixed in-degree) so g/drive fixed across
scales; only finite-size noise ~1/sqrt(N) changes; event speed v is intensive so the crossing time
(and read window) scale with L, not N. Run ONE config per process (fresh memory); CLI-parametrized.
"""
import sys
import os
import json
import argparse
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta                # noqa: E402
from connectivity import place_neurons                      # noqa: E402
from connectivity_rot import build_connectivity_rot         # noqa: E402
from kick_probe import simulate_kick, DUR_KICK              # noqa: E402 (patched: r_kick, t_kick)
from lfp import LFPRecorder                                 # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope      # noqa: E402

PITCH, NC, SHAFTS = 4.0, 6, (15.0, 75.0, 135.0)
MARGIN_FRAC, KDIR, AXIS_ERR_MAX, PART_MIN = 0.10, 3, 25.0, 7
DT, DRIVE = 0.1, 0.6
OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_wave"


def montage(center, rot_deg=0.0):
    rot = np.deg2rad(rot_deg)
    return merge_montages([build_shaft(np.deg2rad(a) + rot, PITCH, NC, tuple(center), chr(65 + i))
                           for i, a in enumerate(SHAFTS)])


def read_fixed_window(env, fdt, m, win, theta_ref_rad, label):
    """First-pass read: extract_lagpat on a FIXED window (first-crossing onset) -> endpoint axis."""
    floor = float(env.min())
    margin = MARGIN_FRAC * (float(env.max()) - floor)
    art = extract_lagpat(env, fdt, [win], floor, margin, 0.5, fdt)
    art = attach_geometry(art, m)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5 * PITCH)
    rd = direction_readability(r0, b0, art.contact_coords)
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_ref_rad)), 1)
    return {"signal": label, "n_part": int(b0.sum()), "axis_err": err,
            "readability": (None if rd is None or rd != rd else round(float(rd), 3))}


def front_speed(on_spk, posE, theta_rad, t_kick, dt):
    """Front position along theta vs time -> speed (mm/ms). Confirms a single-pass crossing + v."""
    th = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    proj = posE @ th
    bs = max(1, int(round(2.0 / dt)))
    nb = on_spk.shape[0] // bs
    binned = on_spk[:nb * bs].reshape(nb, bs, -1).any(axis=1)
    ikb = int(t_kick / dt) // bs
    fronts, times = [], []
    for b in range(ikb, nb):
        active = binned[b]
        if active.sum() >= 5:
            fronts.append(float(np.percentile(proj[active], 95)))   # leading edge along theta
            times.append(b * bs * dt)
    if len(fronts) < 4:
        return None, None
    fr = np.array(fronts); tm = np.array(times)
    # speed = slope of leading-edge advance over the rising phase
    v = float(np.polyfit(tm[:max(4, len(tm) // 2)], fr[:max(4, len(fr) // 2)], 1)[0])
    return round(v, 4), round(float(fr.max() - fr.min()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=8.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--density", type=float, default=1000.0)
    ap.add_argument("--r-kick", type=float, default=0.6)
    ap.add_argument("--t-kick", type=float, default=20.0)
    ap.add_argument("--t-window", type=float, default=180.0)   # post-kick window (>= crossing)
    ap.add_argument("--kick-mode", choices=["negend", "center", "perp"], default="negend")
    ap.add_argument("--perp-j", type=float, default=0.0)
    ap.add_argument("--montage-rot", type=float, default=0.0)
    ap.add_argument("--theta-ref", type=float, default=None)
    ap.add_argument("--tag", default="L8")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "wave.pid"), "w") as f:
        f.write(str(os.getpid()))
    L, theta_rad = a.L, np.deg2rad(a.theta)
    theta_ref = np.deg2rad(a.theta if a.theta_ref is None else a.theta_ref)
    T = a.t_kick + DUR_KICK + a.t_window
    p = Params(g=3.6, L=L, density=a.density, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    print(f"[{a.tag}] placing+building N~{int(a.density*L*L)} (T={T:.0f}ms t_kick={a.t_kick}) ...", flush=True)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    posE = net["pos"][:NE]
    nut = compute_nu_theta(p)[0]
    center = np.array([L / 2, L / 2])
    half = L / 2
    if a.kick_mode == "center":
        kxy = center
    elif a.kick_mode == "perp":
        base = center - 0.6 * half * np.array([np.cos(theta_rad), np.sin(theta_rad)])
        perp = np.array([-np.sin(theta_rad), np.cos(theta_rad)])
        kxy = base + a.perp_j * half * perp
    else:                                   # negend (single-end, source at -theta end)
        kxy = center - 0.6 * half * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    m = montage(center, a.montage_rot)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    print(f"[{a.tag}] simulating on/off (kick {a.kick_mode} r={a.r_kick} at {np.round(kxy,2)}) ...", flush=True)
    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(kxy), lfp_recorder=rec,
                       r_kick=a.r_kick, t_kick=a.t_kick)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(kxy), lfp_recorder=rec,
                        r_kick=a.r_kick, t_kick=a.t_kick)

    # ignition (instantaneous peak + returned) + dense oracle + front speed
    spk = on["E_spk_bool"]
    bs = max(1, int(round(2.0 / DT)))
    nb = spk.shape[0] // bs
    inst = spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    ikb = int(a.t_kick / DT) // bs
    peak = float(inst[ikb:].max()) if nb > ikb else 0.0
    returned = bool(peak <= 1e-9 or float(inst[-10:].mean()) < 0.3 * peak)
    ik = int(a.t_kick / DT)
    idx = np.where(spk[ik:].any(axis=0))[0]
    oracle = None
    if len(idx) >= 5:
        c = posE[idx] - center
        w, V = np.linalg.eigh((c.T @ c) / len(c))
        mj = V[:, np.argmax(w)]
        oa = float(np.degrees(np.arctan2(mj[1], mj[0])) % 180.0)
        oracle = dict(axis_deg=round(oa, 1),
                      err_vs_ref=round(float(axis_angle_error_deg(
                          np.array([np.cos(np.deg2rad(oa)), np.sin(np.deg2rad(oa))]), theta_ref)), 1),
                      ratio=round(float(np.sqrt(w.max() / max(w.min(), 1e-12))), 2))
    v, reach = front_speed(spk, posE, theta_rad, a.t_kick, DT)

    # FIXED first-pass read window (kick-end -> end) — no event_window_for_run
    win = (a.t_kick + DUR_KICK + 2.0, T - 5.0)
    env_f, fdt, _ = snn_event_envelope(on["E_spk_bool"], posE, m, DT)
    env_l = on["lfp_trace"].T
    fire = read_fixed_window(env_f, fdt, m, win, theta_ref, "firing")
    lfp = read_fixed_window(env_l, DT, m, win, theta_ref, "current_lfp")

    res = dict(tag=a.tag, L=L, theta=a.theta, AR=a.AR, density=a.density, r_kick=a.r_kick,
               t_kick=a.t_kick, T=T, kick_mode=a.kick_mode, montage_rot=a.montage_rot,
               theta_ref=(a.theta if a.theta_ref is None else a.theta_ref),
               NE=int(NE), peak_inst=round(peak, 4), returned=returned,
               front_v_mm_per_ms=v, front_reach_mm=reach, win=[round(win[0], 1), round(win[1], 1)],
               oracle=oracle, firing=fire, current_lfp=lfp)
    with open(os.path.join(OUT, f"wave_read_{a.tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda o: None)
    print(f"[{a.tag}] peak_inst={peak:.3f} returned={returned} v={v}mm/ms reach={reach}mm | "
          f"oracle={oracle['axis_deg'] if oracle else None}deg(err {oracle['err_vs_ref'] if oracle else None}) | "
          f"firing n={fire['n_part']} err={fire['axis_err']} | LFP n={lfp['n_part']} err={lfp['axis_err']}",
          flush=True)
    ok = (fire["axis_err"] is not None and fire["n_part"] >= PART_MIN and fire["axis_err"] < AXIS_ERR_MAX
          and lfp["axis_err"] is not None and lfp["n_part"] >= PART_MIN and lfp["axis_err"] < AXIS_ERR_MAX)
    print(f"[{a.tag}] WAVE-READ {'PASS' if ok else 'not-yet'} (both reads >=7 contacts & <25deg)", flush=True)


if __name__ == "__main__":
    main()
