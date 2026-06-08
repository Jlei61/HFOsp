"""OVERNIGHT SNN cm-scale exploration (user 2026-06-07): ignite the SNN at cm scale AND
reproduce the small-scale / rate-field four-control direction read THROUGH the 4mm electrodes.

Mirrors the PROVEN rate runner-v2 geometry EXACTLY (scripts/run_sef_hfo_obs_increment3a.py):
shafts 15/75/135 deg, n_contacts=6, pitch=4mm, NEGATIVE-end kick (avoids periodic-BC wrap that
ran the positive-end read away to 187GB), participation margin 0.10, kicktrack_off 0.35, same
locked bars (AXIS_ERR_MAX=25, KDIR=3, PART_MIN=7, TAU_FAIL=0.3). ONLY the substrate changes:
rate field -> SNN (Params/build_connectivity_rot/simulate_kick, r_kick=0.6 supra-nucleus), and
each event is read TWO ways: firing-density envelope AND current-based LFP (|I_E|+|I_I|).

Ignition classified by INSTANTANEOUS peak active-fraction + returned-to-rest (self_limited vs
runaway), NOT cumulative count. Resumable (--start-idx); writes incremental JSON + a running
markdown report after EACH condition, and a per-condition try/except so one failure/runaway does
not lose the rest. Priority order: the core (theta=45) + connectivity controls come first.
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
from kick_probe import simulate_kick, T_KICK                # noqa: E402 (patched: kick_center, lfp_recorder, r_kick)
from lfp import LFPRecorder                                  # noqa: E402 (patched: sites=)

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,   # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability)
from src.sef_hfo_snn_adapter import snn_event_envelope, event_window_for_run         # noqa: E402

# ---- geometry + bars mirroring the rate runner v2 ----
PITCH, NC, SHAFTS = 4.0, 6, (15.0, 75.0, 135.0)
MARGIN_FRAC = 0.10
KDIR, AXIS_ERR_MAX, PART_MIN, TAU_FAIL = 3, 25.0, 7, 0.3
DT, DRIVE, T = 0.1, 0.6, 180.0
R_KICK_DEF, KICK_END_FRAC, KICKTRACK_OFF = 0.6, 0.6, 0.35

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_explore"
JSON = os.path.join(OUT, "snn_cm_explore_results.json")
REPORT = os.path.join(OUT, "RUNNING_REPORT.md")

_NET_CACHE = {}


def get_net(theta_deg, AR, L, density, seed=1):
    key = (theta_deg, AR, L, density, seed)
    if key not in _NET_CACHE:
        p = Params(g=3.6, L=L, density=density, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=seed)
        rng = np.random.default_rng(seed)
        pos, labels, NE, NI = place_neurons(p, rng)
        net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                     theta_EE=np.deg2rad(theta_deg), AR=AR, verbose=False)
        _NET_CACHE.clear()                       # keep only ONE net in memory at a time
        _NET_CACHE[key] = (p, net, net["pos"][:NE], compute_nu_theta(p)[0], int(NE))
    return _NET_CACHE[key]


def montage(center, rotation_deg=0.0):
    rot = np.deg2rad(rotation_deg)
    return merge_montages([build_shaft(np.deg2rad(a) + rot, PITCH, NC, tuple(center), chr(65 + i))
                           for i, a in enumerate(SHAFTS)])


def kick_xy(mode, theta_deg, L, perp_j=0.0):
    center = np.array([L / 2, L / 2])
    half = L / 2
    if mode == "center":
        return center
    th = np.deg2rad(theta_deg)
    base = center - KICK_END_FRAC * half * np.array([np.cos(th), np.sin(th)])   # NEGATIVE-end
    if mode == "negend":
        return base
    if mode == "perp":                            # C2 kick-track: offset perpendicular to theta
        perp = np.array([-np.sin(th), np.cos(th)])
        return base + perp_j * half * perp
    raise ValueError(mode)


def _read_chain(env, env_ref, fdt, m, theta_ref_rad, label):
    win = event_window_for_run(env.mean(0), env_ref.mean(0), fdt)
    if win is None:
        return {"signal": label, "n_part": 0, "axis_err": None, "readability": None}
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


def ignition_class(on_spk, off_spk, posE, center, theta_ref_rad):
    bs = max(1, int(round(2.0 / DT)))
    nb = on_spk.shape[0] // bs
    inst = on_spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    inst_off = off_spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    ikb = int(T_KICK / DT) // bs
    peak = float(inst[ikb:].max()) if nb > ikb else 0.0
    peak_off = float(inst_off[ikb:].max()) if nb > ikb else 0.0
    returned = bool(peak <= 1e-9 or float(inst[-15:].mean()) < 0.3 * peak)
    # dense-neuron oracle: principal axis of recruited E (excess over background)
    ik = int(T_KICK / DT)
    idx = np.where(on_spk[ik:].any(axis=0))[0]
    ang = ratio = None
    if len(idx) >= 5:
        c = posE[idx] - center
        cov = (c.T @ c) / len(c)
        w, V = np.linalg.eigh(cov)
        mj = V[:, np.argmax(w)]
        ang = round(float(np.degrees(np.arctan2(mj[1], mj[0])) % 180.0), 1)
        ratio = round(float(np.sqrt(w.max() / max(w.min(), 1e-12))), 2)
    oracle_err = None if ang is None else round(float(axis_angle_error_deg(
        np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))]), theta_ref_rad)), 1)
    supra = peak > 3 * max(peak_off, 1e-4)
    if not supra:
        klass = "extinction"
    elif not returned:
        klass = "runaway"
    elif ratio is None or ratio < 1.5:
        klass = "isotropic_event"
    else:
        klass = "self_limited_directional"
    return dict(event_class=klass, peak_inst=round(peak, 4), peak_inst_off=round(peak_off, 4),
                returned=returned, oracle_axis_deg=ang, oracle_ratio=ratio,
                oracle_err_vs_ref=oracle_err)


def run_condition(cond):
    theta, AR, L, density = cond["theta"], cond["AR"], cond["L"], cond["density"]
    rk = cond.get("r_kick", R_KICK_DEF)
    p, net, posE, nut, NE = get_net(theta, AR, L, density)
    center = np.array([L / 2, L / 2])
    kxy = kick_xy(cond["kick"], theta, L, cond.get("perp_j", 0.0))
    m = montage(center, cond.get("rotation_deg", 0.0))
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(p.seed)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(kxy), lfp_recorder=rec, r_kick=rk)
    net["rng"] = np.random.default_rng(p.seed)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(kxy), lfp_recorder=rec, r_kick=rk)
    theta_ref = np.deg2rad(cond.get("theta_ref", theta))
    ig = ignition_class(on["E_spk_bool"], off["E_spk_bool"], posE, center, theta_ref)
    env_f, fdt, _ = snn_event_envelope(on["E_spk_bool"], posE, m, DT)
    env_f_ref, _, _ = snn_event_envelope(off["E_spk_bool"], posE, m, DT)
    fire = _read_chain(env_f, env_f_ref, fdt, m, theta_ref, "firing")
    lfp = _read_chain(on["lfp_trace"].T, off["lfp_trace"].T, DT, m, theta_ref, "current_lfp")
    return dict(name=cond["name"], theta=theta, AR=AR, L=L, density=density, r_kick=rk,
                kick=cond["kick"], rotation_deg=cond.get("rotation_deg", 0.0),
                theta_ref=cond.get("theta_ref", theta), NE=NE, ignition=ig,
                firing=fire, current_lfp=lfp)


# ---- condition list (priority order: core theta=45 + connectivity first) ----
def build_conditions():
    """Grouped so same (theta,AR,L,density) net is consecutive (one build per group; the cache
    holds ONE net at a time). theta=45 L=12 group FIRST = core + most conditions reuse one build."""
    C = []
    # --- group 1: theta=45 AR=2 L=12 d=1000 (ONE build, 8 conditions) — core + robustness ---
    C.append(dict(name="C1_theta45", theta=45, AR=2, L=12, density=1000, kick="negend"))
    C.append(dict(name="rk0.4_theta45", theta=45, AR=2, L=12, density=1000, kick="negend", r_kick=0.4))
    C.append(dict(name="rk0.8_theta45", theta=45, AR=2, L=12, density=1000, kick="negend", r_kick=0.8))
    for j, tag in [(-KICKTRACK_OFF, "perp-"), (KICKTRACK_OFF, "perp+")]:
        C.append(dict(name=f"C2_{tag}", theta=45, AR=2, L=12, density=1000, kick="perp",
                      perp_j=j, theta_ref=45))
    for rot in (30.0, 60.0, 90.0):
        C.append(dict(name=f"C3_rot{rot:g}", theta=45, AR=2, L=12, density=1000, kick="negend",
                      rotation_deg=rot, theta_ref=45))
    # --- connectivity-axis controls (one build each) ---
    C.append(dict(name="C1_theta0", theta=0, AR=2, L=12, density=1000, kick="negend"))
    C.append(dict(name="C1_theta90", theta=90, AR=2, L=12, density=1000, kick="negend"))
    C.append(dict(name="C4_iso", theta=45, AR=1, L=12, density=1000, kick="center"))
    # --- scale / density robustness ---
    C.append(dict(name="L16_theta45", theta=45, AR=2, L=16, density=1000, kick="negend"))
    C.append(dict(name="L16_theta90", theta=90, AR=2, L=16, density=1000, kick="negend"))
    C.append(dict(name="dens2000_theta45", theta=45, AR=2, L=12, density=2000, kick="negend"))
    return C


def write_report(results):
    lines = ["# SNN cm-scale exploration — running report", "",
             "Goal: cm SNN ignites + reproduces the four-control direction read (firing + current-LFP)",
             "through 4mm electrodes, mirroring the rate runner-v2 geometry. Bars: axis_err<25, n_part>=7.",
             "", "| condition | ignition | oracle axis(err) | firing n_part/err | LFP n_part/err |",
             "|---|---|---|---|---|"]
    for r in results:
        ig = r["ignition"]; f = r["firing"]; l = r["current_lfp"]
        lines.append(f"| {r['name']} (th={r['theta']},AR={r['AR']},L={r['L']},rk={r['r_kick']}) "
                     f"| {ig['event_class']} (pk {ig['peak_inst']}) "
                     f"| {ig['oracle_axis_deg']}({ig['oracle_err_vs_ref']}) "
                     f"| {f['n_part']}/{f['axis_err']} | {l['n_part']}/{l['axis_err']} |")
    with open(REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def avail_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
    except Exception:
        pass
    return 999.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-idx", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "explorer.pid"), "w") as f:   # for a PID-based watchdog
        f.write(str(os.getpid()))
    conds = build_conditions()
    results = []
    if os.path.exists(JSON):
        try:
            results = json.load(open(JSON)).get("results", [])
        except Exception:
            results = []
    done = {r["name"] for r in results}
    for i, cond in enumerate(conds):
        if i < a.start_idx or cond["name"] in done:
            continue
        if avail_gb() < 40:                       # in-process memory guard (between conditions)
            print(f"ABORTING before {cond['name']}: low memory ({avail_gb():.0f}GB avail)", flush=True)
            break
        print(f"\n===== [{i}] {cond['name']} (avail {avail_gb():.0f}GB) =====", flush=True)
        try:
            r = run_condition(cond)
            ig = r["ignition"]
            print(f"  {ig['event_class']} peak={ig['peak_inst']} oracle={ig['oracle_axis_deg']}deg"
                  f"(err {ig['oracle_err_vs_ref']}) | firing n={r['firing']['n_part']} "
                  f"err={r['firing']['axis_err']} | LFP n={r['current_lfp']['n_part']} "
                  f"err={r['current_lfp']['axis_err']}", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  CONDITION FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
            results.append(dict(name=cond["name"], error=f"{type(e).__name__}: {str(e)[:200]}"))
        json.dump({"results": results}, open(JSON, "w"), indent=1, default=lambda o: None)
        write_report(results)
    print("\nDONE all conditions.", flush=True)


if __name__ == "__main__":
    main()
