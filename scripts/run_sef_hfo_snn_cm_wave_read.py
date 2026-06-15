"""SNN cm-scale TRAVELING-WAVE read (user 2026-06-08, option A) — AUDITED v2.

Reframe: at cm scale the SNN event is a sustained traveling FRONT (not a self-limited blob);
read the FIRST front pass (first-crossing onset -> endpoint_centroid_axis). v2 adds the rigor
the user 2026-06-08 demanded:
  1. VALID-CONTACT MASK: a contact counts ONLY if it is inside the sheet [0,L]^2 AND has >=1 real
     E neuron within Rr. The engine LFPRecorder falls back to the NEAREST single neuron for empty
     (off-sheet) sites, and sample_envelopes' normalized Gaussian likewise reads the nearest
     neurons — so off-sheet contacts are boundary-neuron extrapolations, NOT real 4mm reads. We
     exclude them. NC=4 (the largest montage that FITS the 12mm sheet: shaft length 12mm) is the
     default so there are no off-sheet contacts to begin with; the mask is the audit on top.
  2. POSITIVE/NEGATIVE VERDICT SCHEMA: control_type drives the pass logic. positive (C1/C2/C3):
     pass = BOTH reads n_valid_part>=7 AND axis_err<25 (the pre-registered gate; NOT loosened to
     'LFP-only'). negative (C4 iso): pass = NO readable axis (readability<TAU_FAIL or axis None).
  3. PROVENANCE: git_sha + engine file hashes + full CLI/params (kxy, perp_j, nc, r_kick, t_kick,
     seed) + contact coords + valid mask + which contacts participated + their rank/lag — so the
     axis is auditable ("which contacts read it") months later.

C2 CONTRACT NOTE (plan-vs-code reconciliation): the plan said "perpendicular offset FROM montage
center". This code offsets perpendicular FROM the single-end kick BASE (center - end_frac*half*dir),
i.e. kick = base +/- perp_j*half*perp. This is a documented CONTRACT CHANGE (the kick-track control
still moves ONLY the kick perpendicular while montage + theta_EE stay fixed -> it still isolates
"seed position must not drag direction"; it just starts from the single-end base, consistent with
the single-end-kick reframe). Not "plan tests A, code tests B" silently.

Parity comparator = the FIRING envelope (rate runner-v2 reads firing-density; the rate field has no
current-LFP). current-LFP is an SNN-only bonus and does NOT carry the parity claim.
"""
import sys
import os
import json
import argparse
import subprocess
import hashlib
import numpy as np

ENG = os.path.join("src", "snn_engine")
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
from src.sef_hfo_heterogeneity import sample_core_field     # noqa: E402 (Step-3 pathology core)
from src.sef_hfo_snn_engine_guard import assert_versions    # noqa: E402 (loud-fail on engine drift)

PITCH, SHAFTS = 4.0, (15.0, 75.0, 135.0)
# Participation margin: cm-scale SNN single-end kick is a sustained traveling FRONT with a 3-10x
# amplitude gradient along the wave (near-kick contacts >> wavefront contacts), like the rate field
# (spec §4.2) and UNLIKE the L=3 self-limited blob that spec §4.1 locked at 0.5. 0.10 = the
# max-coverage end (a tighter margin only drops wavefront contacts -> n_part<7 -> INSUFFICIENT;
# it is not a cherry-picked sweet spot). Run --margin-scan to audit: axis_err must stay invariant
# across margins (margin is a detection front-end / coverage knob, NOT the parity claim, which is
# estimator+threshold = endpoint_centroid_axis / 25 deg / k_dir=3). Locked in spec §4.3.
MARGIN_FRAC, KDIR, AXIS_ERR_MAX, PART_MIN, TAU_FAIL = 0.10, 3, 25.0, 7, 0.3
MARGIN_SCAN_FRACS = (0.10, 0.20, 0.30, 0.50)
DT, DRIVE = 0.1, 0.6
OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_wave"
_ENG_FILES = ("kick_probe.py", "lfp.py", "connectivity.py", "connectivity_rot.py", "params.py")
# Shared engine-version baseline (same engine as the hetero grid runner); assert_versions() at
# main() startup loud-fails if any engine file drifts from this recorded snapshot.
ENGINE_VERSIONS = os.path.join("results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")


def _engine_guard():
    """Loud-fail if the engine drifted from the recorded baseline (spec §7 hard
    contract; same guard the hetero grid runner uses). The engine is patched (r_kick/t_kick/
    lfp sites; patch-note scripts/engine_patches/kick_center.patch); the guard asserts its sha256 so an
    unreviewed in-place edit fails loudly before a run."""
    if not os.path.exists(ENGINE_VERSIONS):
        raise RuntimeError(
            f"engine baseline missing: {ENGINE_VERSIONS}. Re-snapshot it with "
            "src.sef_hfo_snn_engine_guard.record_versions after reviewing the engine.")
    assert_versions(json.loads(open(ENGINE_VERSIONS).read()))


def _provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        sha = None
    eng = {}
    for fn in _ENG_FILES:
        try:
            eng[fn] = hashlib.sha1(open(os.path.join(ENG, fn), "rb").read()).hexdigest()[:12]
        except Exception:
            eng[fn] = None
    return dict(git_sha=sha, engine_sha=eng, argv=sys.argv)


def montage(center, rot_deg, nc):
    rot = np.deg2rad(rot_deg)
    return merge_montages([build_shaft(np.deg2rad(a) + rot, PITCH, nc, tuple(center), chr(65 + i))
                           for i, a in enumerate(SHAFTS)])


def valid_mask(m, posE, L, Rr):
    """valid contact = inside sheet [0,L]^2 AND >=1 real E neuron within Rr (no fallback)."""
    C = np.asarray(m.contacts, float)
    inside = (C[:, 0] >= 0) & (C[:, 0] <= L) & (C[:, 1] >= 0) & (C[:, 1] <= L)
    has_n = np.array([int((np.linalg.norm(posE - c, axis=1) <= Rr).sum()) >= 1 for c in C])
    return inside & has_n, inside, has_n


def read_valid(env, fdt, m, valid, win, theta_ref_rad, label, margin_frac=MARGIN_FRAC):
    """Read direction using ONLY valid contacts (subset env+montage), with contact-level detail."""
    vi = np.where(valid)[0]
    if len(vi) < PART_MIN:
        return dict(signal=label, n_valid=int(valid.sum()), n_part=0, axis_err=None,
                    readability=None, axis_vec=None, participating=[], ranks={}, lags_s={})
    env_v = env[vi]
    names_v = [m.names[i] for i in vi]
    from src.sef_hfo_observation import VirtualMontage
    m_v = VirtualMontage(np.asarray(m.contacts)[vi], names_v, "valid_subset")
    floor = float(env_v.min())
    margin = margin_frac * (float(env_v.max()) - floor)
    art = extract_lagpat(env_v, fdt, [win], floor, margin, 0.5, fdt)
    art = attach_geometry(art, m_v)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5 * PITCH)
    rd = direction_readability(r0, b0, art.contact_coords)
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_ref_rad)), 1)
    part = [names_v[j] for j in range(len(b0)) if b0[j]]
    ranks = {names_v[j]: round(float(r0[j]), 2) for j in range(len(b0)) if b0[j]}
    lags = {names_v[j]: round(float(art.lag_raw[j, 0]), 4) for j in range(len(b0)) if b0[j]}
    return dict(signal=label, n_valid=int(valid.sum()), n_part=int(b0.sum()), axis_err=err,
                readability=(None if rd is None or rd != rd else round(float(rd), 3)),
                axis_vec=(None if ax is None else [round(float(ax[0]), 4), round(float(ax[1]), 4)]),
                participating=part, ranks=ranks, lags_s=lags)


def margin_scan_read(env, fdt, m, valid, win, theta_ref, label, margins=MARGIN_SCAN_FRACS):
    """Read the SAME recorded envelope at several participation margins (cheap; the sim is unchanged).
    Decisive evidence that margin is a detection front-end (coverage knob), not the parity claim:
    axis_err must stay invariant where n_part>=7; a tighter margin only drops wavefront contacts."""
    out = []
    for mf in margins:
        r = read_valid(env, fdt, m, valid, win, theta_ref, label, margin_frac=mf)
        out.append(dict(margin_frac=mf, n_part=r["n_part"], n_valid=r["n_valid"],
                        axis_err=r["axis_err"], readability=r["readability"]))
    return out


def front_speed(on_spk, posE, theta_rad, t_kick, dt):
    th = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    proj = posE @ th
    bs = max(1, int(round(2.0 / dt)))
    nb = on_spk.shape[0] // bs
    binned = on_spk[:nb * bs].reshape(nb, bs, -1).any(axis=1)
    ikb = int(t_kick / dt) // bs
    fronts, times = [], []
    for b in range(ikb, nb):
        if binned[b].sum() >= 5:
            fronts.append(float(np.percentile(proj[binned[b]], 95)))
            times.append(b * bs * dt)
    if len(fronts) < 4:
        return None
    fr = np.array(fronts); tm = np.array(times)
    h = max(4, len(tm) // 2)
    return round(float(np.polyfit(tm[:h], fr[:h], 1)[0]), 4)


def positive_verdict(fire, lfp):
    def ok(r):
        return r["axis_err"] is not None and r["n_part"] >= PART_MIN and r["axis_err"] < AXIS_ERR_MAX
    return dict(control_type="positive",
                firing_pass=ok(fire), lfp_pass=ok(lfp),
                PASS=bool(ok(fire) and ok(lfp)),                       # pre-registered gate: BOTH
                parity_comparator="firing", parity_pass=ok(fire))


def negative_verdict(fire, lfp):
    def no_axis(r):
        return (r["axis_err"] is None) or (r["readability"] is not None and r["readability"] < TAU_FAIL)
    return dict(control_type="negative",
                firing_no_axis=no_axis(fire), lfp_no_axis=no_axis(lfp),
                PASS=bool(no_axis(fire) and no_axis(lfp)),             # correct = NO readable axis
                parity_comparator="firing", parity_pass=no_axis(fire))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=12.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--density", type=float, default=500.0)
    ap.add_argument("--r-kick", type=float, default=0.6)
    ap.add_argument("--t-kick", type=float, default=20.0)
    ap.add_argument("--t-window", type=float, default=180.0)
    ap.add_argument("--kick-mode", choices=["negend", "center", "perp"], default="negend")
    ap.add_argument("--perp-j", type=float, default=0.0)
    ap.add_argument("--montage-rot", type=float, default=0.0)
    ap.add_argument("--nc", type=int, default=4)             # NC=4 FITS L=12 (no off-sheet contacts)
    ap.add_argument("--theta-ref", type=float, default=None)
    ap.add_argument("--control-type", choices=["positive", "negative"], default="positive")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="L12")
    # Step-3 pathology core (heterogeneity): lower + spread the firing threshold in a
    # localized core at the kick origin, then read the core-seeded event through the SAME
    # validated virtual-SEEG read-out. --core-mean<18 = more excitable; std 1.5 wide / 0.5 narrow.
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--core-mean", type=float, default=18.0)
    ap.add_argument("--core-std", type=float, default=1.5)
    ap.add_argument("--core-r", type=float, default=1.5)        # ~= measured E->E reach at d=100
    ap.add_argument("--core-at", choices=["kick", "center"], default="kick")
    ap.add_argument("--margin-frac", type=float, default=MARGIN_FRAC,
                    help="participation margin fraction (cm-SNN locked 0.10, spec §4.3)")
    ap.add_argument("--margin-scan", action="store_true",
                    help="also read the envelope at margins 0.10/0.20/0.30/0.50 (sensitivity)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    _engine_guard()
    with open(os.path.join(OUT, "wave.pid"), "w") as f:
        f.write(str(os.getpid()))
    L, theta_rad = a.L, np.deg2rad(a.theta)
    theta_ref = np.deg2rad(a.theta if a.theta_ref is None else a.theta_ref)
    T = a.t_kick + DUR_KICK + a.t_window
    p = Params(g=3.6, L=L, density=a.density, T=T, dt=DT, nu_ext_ratio=DRIVE, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"[{a.tag}] N~{int(a.density*L*L)} seed={a.seed} nc={a.nc} ...", flush=True)
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
    else:
        kxy = center - 0.6 * half * np.array([np.cos(theta_rad), np.sin(theta_rad)])
    # Step-3 pathology core: localized threshold heterogeneity at the kick origin (or center).
    vth, core_info = None, None
    if a.core:
        is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True
        core_xy = (kxy if a.core_at == "kick" else center)
        cf = sample_core_field(net["pos"], is_E, core_xy, a.core_r,
                               np.random.default_rng(a.seed + 7),
                               core_mean=a.core_mean, core_std=a.core_std, base_mean=18.0)
        vth = cf["vth"]
        core_info = dict(core_mean=a.core_mean, core_std=a.core_std, core_r=a.core_r,
                         core_at=a.core_at, core_xy=[round(float(core_xy[0]), 3), round(float(core_xy[1]), 3)],
                         n_core=int(cf["core_mask"].sum()))
    m = montage(center, a.montage_rot, a.nc)
    valid, inside, has_n = valid_mask(m, posE, L, p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(a.seed)
    on = simulate_kick(p, net, KICK_BOOST=2 * nut, kick_center=list(kxy), lfp_recorder=rec,
                       r_kick=a.r_kick, t_kick=a.t_kick, V_th_per_neuron=vth)
    net["rng"] = np.random.default_rng(a.seed)
    off = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(kxy), lfp_recorder=rec,
                        r_kick=a.r_kick, t_kick=a.t_kick, V_th_per_neuron=vth)

    spk = on["E_spk_bool"]
    bs = max(1, int(round(2.0 / DT)))
    nb = spk.shape[0] // bs
    inst = spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    ikb = int(a.t_kick / DT) // bs
    peak = float(inst[ikb:].max()) if nb > ikb else 0.0
    returned = bool(peak <= 1e-9 or float(inst[-10:].mean()) < 0.3 * peak)
    # self-ignition probe: the OFF run has the SAME core vth but NO kick. If the core fires
    # spontaneously the off run bursts -> the on event is NOT a clean evoked read (the
    # spontaneous/pathology leg, NOT-YET multi-seed-grade). evoked_clean gates the read-out.
    off_spk = off["E_spk_bool"]
    off_inst = off_spk[:nb * bs].reshape(nb, bs, -1).any(axis=1).mean(axis=1)
    off_peak = float(off_inst.max()) if nb else 0.0
    core_self_ignite = bool(core_info is not None and off_peak > max(5e-3, 0.2 * peak))
    evoked_clean = bool(core_info is None or not core_self_ignite)
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
    v = front_speed(spk, posE, theta_rad, a.t_kick, DT)

    win = (a.t_kick + DUR_KICK + 2.0, T - 5.0)
    env_f, fdt, _ = snn_event_envelope(on["E_spk_bool"], posE, m, DT)
    fire = read_valid(env_f, fdt, m, valid, win, theta_ref, "firing", margin_frac=a.margin_frac)
    lfp = read_valid(on["lfp_trace"].T, DT, m, valid, win, theta_ref, "current_lfp", margin_frac=a.margin_frac)
    verdict = (negative_verdict if a.control_type == "negative" else positive_verdict)(fire, lfp)
    margin_sensitivity = None
    if a.margin_scan:
        margin_sensitivity = dict(
            margins=list(MARGIN_SCAN_FRACS),
            firing=margin_scan_read(env_f, fdt, m, valid, win, theta_ref, "firing"),
            current_lfp=margin_scan_read(on["lfp_trace"].T, DT, m, valid, win, theta_ref, "current_lfp"),
            note="margin = detection front-end (coverage); parity claim = estimator+threshold "
                 "(endpoint_centroid_axis / 25deg / k_dir=3, unchanged). axis_err invariance across "
                 "margins => margin moves coverage not the read; 0.10 = max-coverage end (spec §4.3).")

    res = dict(tag=a.tag, control_type=a.control_type, provenance=_provenance(),
               config=dict(L=L, theta=a.theta, theta_ref=(a.theta if a.theta_ref is None else a.theta_ref),
                           AR=a.AR, density=a.density, NE=int(NE), r_kick=a.r_kick, t_kick=a.t_kick,
                           T=T, kick_mode=a.kick_mode, perp_j=a.perp_j, montage_rot=a.montage_rot,
                           nc=a.nc, seed=a.seed, Rr=p.Rr, margin_frac=a.margin_frac,
                           kxy=[round(float(kxy[0]), 3), round(float(kxy[1]), 3)]),
               geometry_audit=dict(n_contacts=len(m.contacts), n_inside=int(inside.sum()),
                                   n_valid=int(valid.sum()), n_offsheet=int((~inside).sum()),
                                   contact_coords=np.asarray(m.contacts).round(2).tolist(),
                                   valid=valid.astype(int).tolist()),
               ignition=dict(peak_inst=round(peak, 4), returned=returned, front_v_mm_per_ms=v,
                             off_peak_inst=round(off_peak, 4), core_self_ignite=core_self_ignite,
                             evoked_clean=evoked_clean),
               core=core_info, margin_sensitivity=margin_sensitivity,
               oracle=oracle, firing=fire, current_lfp=lfp, verdict=verdict)
    with open(os.path.join(OUT, f"wave_read_{a.tag}.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=lambda o: None)
    cstr = ("" if core_info is None else
            f"| CORE m={a.core_mean} s={a.core_std} r={a.core_r} ign_on={peak:.3f} ign_off={off_peak:.3f} "
            f"self_ignite={core_self_ignite} evoked_clean={evoked_clean} ")
    print(f"[{a.tag}] {a.control_type} {cstr}| valid {valid.sum()}/{len(m.contacts)} (offsheet {(~inside).sum()}) "
          f"| oracle={oracle['axis_deg'] if oracle else None}({oracle['err_vs_ref'] if oracle else None}) "
          f"| fir n={fire['n_part']}/{fire['n_valid']} err={fire['axis_err']} rd={fire['readability']} "
          f"| lfp n={lfp['n_part']}/{lfp['n_valid']} err={lfp['axis_err']} rd={lfp['readability']} "
          f"-> VERDICT {'PASS' if verdict['PASS'] else 'FAIL'} (parity[firing] {'pass' if verdict['parity_pass'] else 'FAIL'})",
          flush=True)


if __name__ == "__main__":
    main()
