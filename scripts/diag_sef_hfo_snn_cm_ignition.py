"""SNN-cm ignition DIAGNOSIS (user 2026-06-07 Step 1+2): WHY does cm not ignite, and what
is the minimum density that does? CHEAP — NO LFP, NO montage, NO figure, ignition summary only.

Mechanism hypothesis (arithmetic): each E needs C_EE=800 presynaptic E; # E within one
l_EE(0.38mm) disk = pi*l_EE^2*0.8*density. Below density~2204 the sampler must reach >l_EE to
gather 800 -> non-local E->E -> a local kick cannot sustain a front. Above it, 800 pack locally
-> propagation. This sweeps DENSITY ONLY (one knob) at FIXED small L (L=6, no periodic-BC wrap
unlike L=3, and N stays affordable: density{1000,2000,4000} -> N={36k,72k,144k}). Kick fixed.

Per density measures: kick-disk E count, realized E->E connection radius (empirical, confirms
the arithmetic), recruited E at 50/100ms post-kick, spatial spread (flash vs front) + principal
axis (does it elongate along theta_EE=45). Ignited iff recruitment >> disk AND spread > disk.
"""
import sys
import os
import json
import numpy as np

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
sys.path.insert(0, ENG)
from params import Params, compute_nu_theta            # noqa: E402
from connectivity import place_neurons                  # noqa: E402
from connectivity_rot import build_connectivity_rot, _sample_partners_rot  # noqa: E402
from kick_probe import simulate_kick, T_KICK, R_KICK     # noqa: E402

import argparse

T, DT, DRIVE, THETA = 220.0, 0.1, 0.6, 45.0
l_EE_default = 0.380


def principal_axis_deg_ratio(coords, center):
    c = coords - center
    if len(c) < 5:
        return None, None
    cov = (c.T @ c) / len(c)
    w, V = np.linalg.eigh(cov)
    major = V[:, np.argmax(w)]
    ang = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    ratio = float(np.sqrt(w.max() / max(w.min(), 1e-12)))
    return round(ang, 1), round(ratio, 2)


def realized_ee_radius(p, pos, NE, rng, theta_rad, kick_center, n_sample=20):
    """Median distance of the C_EE sampled presyn E for E neurons near the kick."""
    posE = pos[:NE]
    l_par = p.l_EE * np.sqrt(2.0)
    l_perp = p.l_EE / np.sqrt(2.0)
    near = np.argsort(np.linalg.norm(posE - kick_center, axis=1))[:n_sample]
    radii = []
    for i in near:
        cols = _sample_partners_rot(posE[i], posE, p.C_EE, l_par, l_perp, theta_rad,
                                    rng, self_local=int(i))
        if cols.size:
            radii.append(float(np.median(np.linalg.norm(posE[cols] - posE[i], axis=1))))
    return round(float(np.median(radii)), 3) if radii else None


def run_one(density, L, kick_mult, theta_rad, r_kick=None):
    p = Params(g=3.6, L=L, density=density, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=theta_rad, AR=2.0, verbose=False)
    posE = net["pos"][:NE]
    nut = compute_nu_theta(p)[0]
    center = np.array([L / 2, L / 2])
    rk = R_KICK if r_kick is None else float(r_kick)
    n_disk = int(np.sum(np.linalg.norm(posE - center, axis=1) <= rk))
    ee_r = realized_ee_radius(p, pos, NE, np.random.default_rng(7), theta_rad, center)

    net["rng"] = np.random.default_rng(1)
    on = simulate_kick(p, net, KICK_BOOST=kick_mult * nut, kick_center=list(center), r_kick=rk)
    net["rng"] = np.random.default_rng(1)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(center), r_kick=rk)

    dt = DT
    ik = int(T_KICK / dt)
    spk = on["E_spk_bool"]
    spk_off = off["E_spk_bool"]

    # INSTANTANEOUS active fraction (per 2ms bin) + returned-to-rest: the correct
    # self-limited-vs-global discriminator (classify_response semantics), NOT cumulative
    # recruited (a directional front sweeping a small sheet recruits high cumulative but is
    # self-limited). global synchronous = high peak AND not returned; self-limited = returned.
    bs = max(1, int(round(2.0 / dt)))
    nb = spk.shape[0] // bs
    inst = spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)   # frac E active / 2ms bin
    inst_off = spk_off[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    ikb = ik // bs
    peak_inst = float(inst[ikb:].max()) if nb > ikb else 0.0
    peak_inst_off = float(inst_off[ikb:].max()) if nb > ikb else 0.0
    tail = float(inst[-15:].mean())                                      # last ~30ms
    returned = bool(peak_inst <= 1e-9 or tail < 0.3 * peak_inst)

    def recruited(win_ms):
        w = slice(ik, ik + int(win_ms / dt))
        idx = np.where(spk[w].any(axis=0))[0]
        idx_off = np.where(spk_off[w].any(axis=0))[0]
        return idx, idx_off

    theta_hat = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    def par_span(idx):
        """Span + reach of recruited E projected onto the theta_EE axis (propagation
        distance along the connectivity axis, NOT isotropic max radius)."""
        if len(idx) < 2:
            return 0.0, 0.0
        proj = (posE[idx] - center) @ theta_hat
        return float(np.ptp(proj)), float(np.abs(proj).max())

    out = {"density": density, "L": L, "kick_mult": kick_mult, "r_kick_mm": rk,
           "N": int(density * L * L), "NE": int(NE), "n_disk_E": n_disk,
           "realized_ee_radius_mm": ee_r, "l_EE_mm": p.l_EE,
           "ee_radius_over_lEE": (None if ee_r is None else round(ee_r / p.l_EE, 2)),
           "peak_inst_frac": round(peak_inst, 4), "peak_inst_off": round(peak_inst_off, 4),
           "returned": returned}
    for win in (50, 100):
        idx, idx_off = recruited(win)
        n_on, n_off = len(idx), len(idx_off)
        max_dist = (round(float(np.linalg.norm(posE[idx] - center, axis=1).max()), 2)
                    if n_on else 0.0)
        ang, ratio = principal_axis_deg_ratio(posE[idx], center) if n_on >= 5 else (None, None)
        span_on, reach_on = par_span(idx)
        span_off, _ = par_span(idx_off)
        out[f"recruited_{win}ms"] = dict(
            n_on=n_on, n_off=n_off, max_dist_mm=max_dist,
            span_par_mm=round(span_on, 2), reach_par_mm=round(reach_on, 2),
            span_par_off_mm=round(span_off, 2), axis_deg=ang, axis_ratio=ratio)
    r = out["recruited_100ms"]
    # Classify the event with the CORRECT self-limited-vs-global discriminator:
    supra_bg = peak_inst > 3 * max(peak_inst_off, 1e-4)        # ignited above background
    directional = (r["axis_ratio"] is not None and r["axis_ratio"] > 1.5)
    propagated = r["reach_par_mm"] > 3 * rk
    if not (supra_bg and propagated):
        klass = "extinction"                                   # fizzle / sub-nucleus
    elif not returned:
        klass = "global_runaway"                               # ignited but never self-terminates
    elif not directional:
        klass = "isotropic_event"                              # ignited+self-limited but no axis
    else:
        klass = "self_limited_directional"                     # the target state
    out["event_class"] = klass
    # IGNITED (the target) = a self-limited DIRECTIONAL propagating front:
    out["IGNITED"] = bool(klass == "self_limited_directional")
    return out


# one-knob-at-a-time grids: (density, L, kick_mult, r_kick_mm) tuples
SWEEPS = {
    "density": [(1000.0, 6.0, 2.0, None), (2000.0, 6.0, 2.0, None), (4000.0, 6.0, 2.0, None)],
    "L":       [(1000.0, 6.0, 2.0, None), (1000.0, 8.0, 2.0, None), (1000.0, 12.0, 2.0, None)],
    "kick":    [(1000.0, 12.0, 2.0, None), (1000.0, 12.0, 4.0, None), (1000.0, 12.0, 8.0, None)],
    # kick_radius: fixed L=8 (failed at R_KICK=0.15) — does the rate-field-predicted
    # critical nucleus (~0.6-0.8mm) ignite the SNN deterministically?
    "kick_radius": [(1000.0, 8.0, 2.0, 0.15), (1000.0, 8.0, 2.0, 0.8), (1000.0, 8.0, 2.0, 1.2)],
    # kick_radius_safe: map the fizzle->self_limited transition WITHOUT r>=1.2 (which runs
    # away + blows up memory). Uses the instantaneous-peak+returned classifier.
    "kick_radius_safe": [(1000.0, 8.0, 2.0, 0.4), (1000.0, 8.0, 2.0, 0.6), (1000.0, 8.0, 2.0, 0.8)],
    # L12_oracle: take the confirmed window (r_kick=0.6, supra-nucleus) to cm scale L=12
    # (fits 4mm >=7 contacts). axis_deg here IS the dense-neuron oracle (principal axis of
    # recruited E). Must stay self_limited_directional ~45 deg before any electrode read-out.
    "L12_oracle": [(1000.0, 12.0, 2.0, 0.6)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=list(SWEEPS), default="density")
    a = ap.parse_args()
    theta_rad = np.deg2rad(THETA)
    rows = []
    for density, L, km, rk in SWEEPS[a.sweep]:
        print(f"--- density={density} L={L} kick={km}x r_kick={rk} (N~{int(density*L*L)}) building+simulating ---", flush=True)
        r = run_one(density, L, km, theta_rad, r_kick=rk)
        rows.append(r)
        rr = r["recruited_100ms"]
        print(f"  disk_E={r['n_disk_E']} | peak_inst={r['peak_inst_frac']} (off {r['peak_inst_off']}) "
              f"returned={r['returned']} | reach_par={rr['reach_par_mm']}mm (3*rk={3*r['r_kick_mm']:.2f}) "
              f"axis={rr['axis_deg']}/{rr['axis_ratio']} -> {r['event_class'].upper()}", flush=True)
    out = "results/topic4_sef_hfo/observation_layer/lfp_forward_validation"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"cm_ignition_{a.sweep}_diagnosis.json"), "w") as f:
        json.dump({"sweep": a.sweep, "theta": THETA, "rows": rows}, f, indent=2, default=lambda o: None)
    print(f"\nSUMMARY (sweep={a.sweep}; one knob varied, others fixed):", flush=True)
    for r in rows:
        rr = r['recruited_100ms']
        print(f"  density={r['density']:>6} L={r['L']:>4} r_kick={r['r_kick_mm']}mm: "
              f"peak_inst={r['peak_inst_frac']} returned={r['returned']} axisratio={rr['axis_ratio']} "
              f"-> {r['event_class']}", flush=True)


if __name__ == "__main__":
    main()
