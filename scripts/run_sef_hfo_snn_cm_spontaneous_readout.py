"""Spontaneous (noise-driven, NO kick) cm-SNN multi-event READ-OUT + 4-panel figure data
(user 2026-06-10: option B, real interictal events nucleate from the lesion, no external kick).

The lesion (low-threshold heterogeneity core) self-ignites a TRAIN of events under the background
noise drive; each event is read through the SAME validated virtual-SEEG montage (3 non-parallel
4mm shafts) -> per-event rank order + endpoint-centroid direction -> a multi-event rank matrix
(the building block for the masked PR-2/PR-2.5/rank-displacement record). For a REPRESENTATIVE
clean event we save a per_cell-style npz so the kick-version `mechanism_4panel` figure reproduces
for the spontaneous version (the lesion IS the nucleation source — no kick marker).

Lesion configs (--lesion):
  oneend_neg   one focus near the -axis end  -> forward train  ((a)-pool forward building block)
  oneend_pos   one focus near the +axis end  -> reverse train  ((a)-pool reverse building block)
  twoend_deph  two foci with DIFFERENT means (dephased) so they fire at separated times (config b)

Surround stays sub-critical (bare sheet is quiet at this drive) so events nucleate ONLY from the
lesion — NOT a near-critical whole-sheet artifact. Margin/estimator/threshold identical to the
validated cm read-out (endpoint_centroid_axis / k_dir=3 / margin 0.10 spec §4.3).
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
from kick_probe import simulate_kick                        # noqa: E402
from lfp import LFPRecorder                                 # noqa: E402

sys.path.insert(0, os.getcwd())
from src.sef_hfo_observation import (build_shaft, merge_montages, extract_lagpat,       # noqa: E402
                                     attach_geometry, endpoint_centroid_axis,
                                     axis_angle_error_deg, direction_readability, VirtualMontage,
                                     write_legacy_npz, write_packed_times, write_montage_manifest)
from src.sef_hfo_snn_adapter import snn_event_envelope      # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field     # noqa: E402
from src.sef_hfo_events import detect_events                 # noqa: E402
from src.sef_hfo_snn_engine_guard import assert_versions     # noqa: E402
from src.sef_hfo_stage3 import build_sidecar               # noqa: E402
from src.topic4_corridor_substrate import corridor_regions, hub_mask_E  # noqa: E402  (M3)
from src.topic4_degnorm import degnorm_vth_delta, ee_degree              # noqa: E402  (M3)
from src.topic4_propagation_operator import (spatial_bins, build_w_resp,  # noqa: E402  (M3-final)
                                             make_step_operator, spectral_radius, h_field)
from src.topic4_permissivity import permissivity_vth_delta              # noqa: E402  (M3-final)
from src.sef_hfo_stage4 import nucleation_centroid, compute_t0_gate   # noqa: E402
from src.sef_hfo_a1b import a1b_weight_lesion, local_global_ratio     # noqa: E402  (A1b)

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
ENGINE_VERSIONS = os.path.join("results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
PITCH, SHAFT_OFFSETS = 4.0, (0.0, 90.0)   # ∥ and ⊥ to the EE axis (theta_EE), like the small-scale grid
MARGIN_FRAC, KDIR, PART_MIN = 0.10, 3, 7
DT, DRIVE = 0.1, 0.6
BIN_MS = 1.0
BASELINE_MS = (5.0, 50.0)
CAL_FRAC = 0.5
_ENG_FILES = ("kick_probe.py", "lfp.py", "connectivity.py", "connectivity_rot.py", "params.py")
# M3-final W_resp measurement defaults (--h-source resp). Calibrated values are frozen by Task
# 1.5/Task 4 pre-registration; these are the pilot defaults (win starts after DUR_KICK=18ms =
# one propagation generation, t_kick clear of the boundary, kick on the bin center).
W_RESP_KICK_BOOST, W_RESP_R_KICK, W_RESP_T_KICK, W_RESP_WIN_MS = 1.0, 1.5, 100.0, (18.0, 28.0)
W_RESP_SEEDS = 3


def _engine_guard():
    if not os.path.exists(ENGINE_VERSIONS):
        raise RuntimeError(f"engine baseline missing: {ENGINE_VERSIONS}")
    assert_versions(json.loads(open(ENGINE_VERSIONS).read()))


def _provenance():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        sha = None
    eng = {fn: (hashlib.sha1(open(os.path.join(ENG, fn), "rb").read()).hexdigest()[:12]
                if os.path.exists(os.path.join(ENG, fn)) else None) for fn in _ENG_FILES}
    return dict(git_sha=sha, engine_sha=eng, argv=sys.argv)


def montage(center, theta_deg, rot_deg, nc, pitch=PITCH):
    """2 shafts: A = ∥ EE axis (theta_EE), B = ⊥ (theta_EE+90), like the small-scale grid. Aligning
    the ∥ shaft to the propagation axis means BOTH forward and reverse waves traverse the SAME ∥
    contacts -> high shared participation -> cleanly rank-reversed (opposite) templates. The earlier
    misaligned 3-shaft montage (15/75/135) gave only 6/18 shared contacts -> masked imputation
    inflated the inter-cluster corr to +0.74 and the forward/reverse opposition went undetected
    (even though each template was monotonic with axis-projection). cm four-control C3 already proved
    shaft-rotation-invariance, so aligning here is illustration + coverage, not tuning to the answer."""
    rot = np.deg2rad(rot_deg)
    return merge_montages([build_shaft(np.deg2rad(theta_deg + off) + rot, pitch, nc, tuple(center), chr(65 + i))
                           for i, off in enumerate(SHAFT_OFFSETS)])


def valid_mask(m, posE, L, Rr):
    C = np.asarray(m.contacts, float)
    inside = (C[:, 0] >= 0) & (C[:, 0] <= L) & (C[:, 1] >= 0) & (C[:, 1] <= L)
    has_n = np.array([int((np.linalg.norm(posE - c, axis=1) <= Rr).sum()) >= 1 for c in C])
    return inside & has_n


def _mirror_core_field(cf_src, cf_dst, pos, src_xy, dst_xy):
    """ideal-symmetry probe: transplant cf_src's core threshold-vs-radius profile onto cf_dst's
    core neurons (matched by within-core distance-to-focus rank) so the two cores carry an
    IDENTICAL realized field. If the read-out still biases one direction under this -> the
    asymmetry is geometry/connectivity/read-out, NOT a threshold-draw difference."""
    src_idx = np.flatnonzero(cf_src["core_mask"]); dst_idx = np.flatnonzero(cf_dst["core_mask"])
    rs = src_idx[np.argsort(np.linalg.norm(pos[src_idx] - src_xy, axis=1))]
    rd = dst_idx[np.argsort(np.linalg.norm(pos[dst_idx] - dst_xy, axis=1))]
    n = min(len(rs), len(rd))
    new_vth = cf_dst["vth"].copy()
    new_vth[rd[:n]] = cf_src["vth"][rs[:n]]   # same vth-vs-radius profile on both cores
    return dict(vth=new_vth, core_mask=cf_dst["core_mask"])


def build_lesion_vth(net, NE, axis_unit, center, half, lesion, core_mean, core_std, core_r,
                     dephase, seed, sep_frac=0.6, swap_vth=False, mirror_vth=False, elongation=1.0):
    """Per-neuron threshold field. Returns (vth, core_mask, foci[xy list], core_masks[per-focus
    FULL-network bool masks]). The per-focus masks let the caller compute a core-LEVEL onset per end.
    `sep_frac`: each focus sits at center ± sep_frac*half along the axis; larger = farther apart =
    weaker coupling between the two ends (regime-screen geometry knob; default 0.6 = the pilot value).
    twoend_equal: both foci at the SAME core_mean (Stage 3; collisions handled downstream by censoring,
    NOT by dephasing). twoend_deph: pos-end mean RAISED by `dephase` so the two run at different rates
    (identical means collide/merge — diagnostic 2026-06-10)."""
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True
    neg_xy = center - sep_frac * half * axis_unit
    pos_xy = center + sep_frac * half * axis_unit
    if lesion == "oneend_neg":
        cf = sample_core_field(net["pos"], is_E, neg_xy, core_r, np.random.default_rng(seed + 7),
                               core_mean=core_mean, core_std=core_std, base_mean=18.0)
        return cf["vth"], cf["core_mask"], [neg_xy], [cf["core_mask"]]
    if lesion == "oneend_pos":
        cf = sample_core_field(net["pos"], is_E, pos_xy, core_r, np.random.default_rng(seed + 7),
                               core_mean=core_mean, core_std=core_std, base_mean=18.0)
        return cf["vth"], cf["core_mask"], [pos_xy], [cf["core_mask"]]
    if lesion == "twoend_equal":
        # both foci SAME mean/std; distinct rng seeds only de-correlate the threshold draws.
        # paired-swap probe (swap_vth): swap which RNG draw each core gets (connectivity + OU noise
        # from rng(seed) held fixed) -> if the per-network 'winner' flips end, the asymmetry is
        # threshold-draw-driven (per-run luck), not a fixed neg/pos structural bias.
        s_neg, s_pos = (seed + 8, seed + 7) if swap_vth else (seed + 7, seed + 8)
        cf1 = sample_core_field(net["pos"], is_E, neg_xy, core_r, np.random.default_rng(s_neg),
                                core_mean=core_mean, core_std=core_std, base_mean=18.0)
        cf2 = sample_core_field(net["pos"], is_E, pos_xy, core_r, np.random.default_rng(s_pos),
                                core_mean=core_mean, core_std=core_std, base_mean=18.0)
        if mirror_vth:
            cf2 = _mirror_core_field(cf1, cf2, net["pos"], neg_xy, pos_xy)
        return (np.minimum(cf1["vth"], cf2["vth"]), (cf1["core_mask"] | cf2["core_mask"]),
                [neg_xy, pos_xy], [cf1["core_mask"], cf2["core_mask"]])
    if lesion == "extended_patch":
        # Stage 4: ONE large excitable disk at the sheet centre; interior low-Vth, exterior base
        # (18.0). core_r = patch radius (~6-8 mm to span 4-5 contacts). Single core -> foci /
        # core_masks are length-1, so the dual-core build_sidecar (len==2 gate) is NOT triggered.
        cf = sample_core_field(net["pos"], is_E, center, core_r, np.random.default_rng(seed + 7),
                               core_mean=core_mean, core_std=core_std, base_mean=18.0,
                               elongation=elongation, axis_unit=axis_unit)
        return cf["vth"], cf["core_mask"], [center], [cf["core_mask"]]
    # twoend_deph: neg focus at core_mean, pos focus at core_mean + dephase (slower) -> drift apart
    cf1 = sample_core_field(net["pos"], is_E, neg_xy, core_r, np.random.default_rng(seed + 7),
                            core_mean=core_mean, core_std=core_std, base_mean=18.0)
    cf2 = sample_core_field(net["pos"], is_E, pos_xy, core_r, np.random.default_rng(seed + 8),
                            core_mean=core_mean + dephase, core_std=core_std, base_mean=18.0)
    return (np.minimum(cf1["vth"], cf2["vth"]), (cf1["core_mask"] | cf2["core_mask"]),
            [neg_xy, pos_xy], [cf1["core_mask"], cf2["core_mask"]])


def active_fraction(E_spk_bool, dt, bin_ms):
    bs = max(1, int(round(bin_ms / dt)))
    nb = E_spk_bool.shape[0] // bs
    binned = E_spk_bool[:nb * bs].reshape(nb, bs, -1).any(axis=1)
    return binned.mean(axis=1), bs * dt


def read_event(env_f, fdt, m, valid, win, axis_unit, k_dir=KDIR, part_min=PART_MIN, pitch=PITCH):
    """Direction read of ONE event window via valid contacts (endpoint-centroid axis on firing
    envelope). Returns dict(n_part, corr_sign vs axis, axis_err_deg, readability, ranks per name).
    `k_dir`/`part_min` default to the module KDIR/PART_MIN (=3/7); the Phase-2 `--k-dir` knob lowers
    the readable participant floor `part_min = 2*k_dir+1` here AND at the endpoint_centroid_axis call
    so the floor moves consistently (reviewer P2 2026-06-17)."""
    vi = np.where(valid)[0]
    if len(vi) < part_min:
        return dict(n_part=0, axis_err=None, sign=None, readability=None, ranks=None, names=None)
    env_v = env_f[vi]
    names_v = [m.names[i] for i in vi]
    m_v = VirtualMontage(np.asarray(m.contacts)[vi], names_v, "valid_subset")
    floor = float(env_v.min()); margin = MARGIN_FRAC * (float(env_v.max()) - floor)
    art = extract_lagpat(env_v, fdt, [win], floor, margin, 0.5, fdt)
    art = attach_geometry(art, m_v)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=k_dir, eps_deg=0.5 * pitch)
    rd = direction_readability(r0, b0, art.contact_coords)
    # signed direction: project the early->late axis onto axis_unit (forward=+)
    sign = None if ax is None else float(np.sign(np.dot(ax, axis_unit)))
    theta_ref = np.arctan2(axis_unit[1], axis_unit[0])
    err = None if ax is None else round(float(axis_angle_error_deg(ax, theta_ref)), 1)
    ranks = {names_v[j]: (None if not b0[j] else round(float(r0[j]), 2)) for j in range(len(b0))}
    return dict(n_part=int(b0.sum()), axis_err=err, sign=sign,
                readability=(None if rd is None or rd != rd else round(float(rd), 3)),
                ranks=ranks, names=names_v)


def per_neuron_onset(E_spk_bool, t_on, t_off, dt):
    s, e = int(round(t_on / dt)), int(round(t_off / dt))
    seg = E_spk_bool[s:e]
    fired = seg.any(axis=0)
    onset = np.full(seg.shape[1], np.nan)
    idx = np.flatnonzero(fired)
    onset[idx] = np.argmax(seg[:, idx], axis=0).astype(float) * dt
    return onset


def event_field_geometry(P, axis_unit, L):
    """Spatial geometry of one event's fired E-neuron cloud P (n,2 positions in mm).
    Returns (ext, r95, reach_axis, cx, cy, edge_margin), all floats.

    edge_margin is a SPATIAL OVERLAP SCREEN: nearest-wall distance of the cloud centroid minus the
    r95 bulk radius. >0 => bulk fired-field stayed clear of the nearest sheet wall; <=0 => it
    reached/overlapped the wall margin. This is NOT a temporal T_stop<T_edge (front-hits-wall-then-
    stops) criterion -- that needs a per-timestep front trace this dump does not save. Degenerate
    (<2 fired) -> all zeros."""
    if len(P) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cen = P.mean(0)
    rad = np.linalg.norm(P - cen, axis=1)
    ext = float(np.std(rad)); r95 = float(np.percentile(rad, 95))
    u = (P - cen) @ axis_unit                      # projection onto the EE long axis
    reach_axis = float(np.percentile(u, 97.5) - np.percentile(u, 2.5))
    cx, cy = float(cen[0]), float(cen[1])
    edge_margin = float(min(cx, L - cx, cy, L - cy) - r95)
    return ext, r95, reach_axis, cx, cy, edge_margin


def _struct_bin_h(net, NE, bins, scheme):
    """h-like field from the STRUCTURAL prior (reuses the --degnorm-alpha path's ee_degree):
    per-E-cell degree (out/in/hybrid) averaged into each spatial bin. h-scheme post=target
    recruitability -> in_strength (E->E row sum); out=source broadcast -> out_strength (col sum);
    hybrid -> hybrid. Same question as h_field(W_resp, scheme), measured structurally not by kick."""
    scheme_map = {"post": "in_strength", "out": "out_strength", "hybrid": "hybrid"}
    g = ee_degree(net, NE, scheme_map[scheme])            # length-NE, median-normalized
    boc = np.asarray(bins["bin_of_cell"], int)
    n_bins = bins["bin_centers"].shape[0]
    tot = np.bincount(boc, weights=g, minlength=n_bins)
    cnt = np.bincount(boc, minlength=n_bins)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0)


def apply_permissivity_vth_delta(vth, net, NE, NI, posE, a, *, bins, p, V_th0, rng):
    """M3-final W-coupled permissivity (h-coupling) V_th pre-transform. Returns (vth, provenance).

    mu=0 SHORT-CIRCUIT: V_th_per_neuron is returned UNTOUCHED (no h built, no delta) so the spike
    output is bit-identical to the engine baseline (M3_BASE_SHA). Rides the SAME V_th_per_neuron
    path as --degnorm-alpha -- zero engine change.

    mu>0: build h (struct=ee_degree prior; resp=measured W_resp at the mu=0 baseline, optionally
    cached) -> delta = permissivity_vth_delta(...) -> vth += delta (BEFORE simulate_kick).
    """
    prov = dict(mu=a.mu, delta_theta=a.delta_theta, h_source=a.h_source, h_scheme=a.h_scheme,
                h_control=a.h_control, mu_impl=a.mu_impl,
                n_bins=int(bins["bin_centers"].shape[0]) if bins is not None else None,
                lambda0=None)
    if a.mu == 0.0:
        return vth, prov                       # bit-parity: V_th untouched
    if a.mu_impl == "inhibition":
        raise NotImplementedError("--mu-impl inhibition is Phase-2 (inhibition-restraint mu); "
                                  "Phase-1 implements only --mu-impl threshold.")
    if a.h_source == "struct":
        h = _struct_bin_h(net, NE, bins, a.h_scheme)
    else:   # resp: measure (or load) W_resp at the mu=0 baseline, then h_field
        wr = None
        if a.w_resp_cache and os.path.exists(a.w_resp_cache):
            wr = dict(np.load(a.w_resp_cache))
            prov["w_resp_source"] = "cache"
            prov["w_resp_cache_path"] = a.w_resp_cache
        if wr is None:
            # Resolve the W-measurement kick/window. Priority: frozen calibration JSON >
            # pilot-default (ONLY if explicitly allowed). Fail closed otherwise — never silently
            # mix pilot-default h into a canonical --mu>0 --h-source resp run (review P1, 2026-06-22):
            # the prereg freezes calibrated_kick_boost/win, so resp-mode must consume them.
            if a.w_resp_calib_json:
                with open(a.w_resp_calib_json, encoding="utf-8") as f:
                    _cal = json.load(f)
                kb = float(_cal["calibrated_kick_boost"])
                win = tuple(float(x) for x in _cal["calibrated_win_ms"])
                src = "calibrated"
            elif a.allow_pilot_default_wresp:
                kb, win, src = W_RESP_KICK_BOOST, W_RESP_WIN_MS, "pilot_default"
            else:
                raise RuntimeError(
                    "--mu>0 --h-source resp needs a measured-W source: pass --w-resp-cache "
                    "(existing npz) or --w-resp-calib-json (frozen calibration). Refusing to use "
                    "pilot-default W kick/window in a canonical run. Use "
                    "--allow-pilot-default-wresp ONLY for explicit smoke/testing.")
            wr = build_w_resp(p, net, NE, NI, bins, V_th0,
                              kick_boost=kb, r_kick=W_RESP_R_KICK,
                              t_kick=W_RESP_T_KICK, win_ms=win, seeds=W_RESP_SEEDS)
            prov["w_resp_source"] = src
            prov["w_resp_kick_boost"] = float(kb)
            prov["w_resp_win_ms"] = list(win)
            if a.w_resp_cache:
                np.savez_compressed(a.w_resp_cache, **wr)
        h = h_field(wr["W_resp"], a.h_scheme)
        # Lambda0 = rho(W_step) is only meaningful for the measured-response operator
        w_step = make_step_operator(wr["W_resp"], wr["src_mass"], injected_mass=wr.get("injected_mass"))
        prov["lambda0"] = round(float(spectral_radius(w_step)), 6)
    delta = permissivity_vth_delta(h, np.asarray(bins["bin_of_cell"], int), NE, NI,
                                   mu=a.mu, delta_theta=a.delta_theta, control=a.h_control, rng=rng)
    return vth + delta, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
    ap.add_argument("--l-ee", type=float, default=None,
                    help="override Params.l_EE (mm) E->E kernel spatial spread; None=paper 0.380. "
                         "Lower -> tighter front reach -> self-limitation / boundary-audit screen.")
    ap.add_argument("--c-ee", type=int, default=None,
                    help="override Params.C_EE E->E in-degree; None=paper 800. Lower -> sparser coupling "
                         "(ALSO lowers mean recurrent drive -> operating point shifts).")
    # ---- M2 TRUE gate: SEPARATE ADDED wide I->E veto path (rate-toy two-path structure).
    #      Built in build_connectivity_rot; default gate_scale=0 -> bit-parity. Ported from main 2026-06-19. ----
    ap.add_argument("--gate-scale", type=float, default=0.0,
                    help="M2 gate: extra wide I->E inhibitory gain as a fraction of w_EI (0=off, default). "
                         "ADDED on top of the UNCHANGED local I->E.")
    ap.add_argument("--l-gate", type=float, default=None,
                    help="M2 gate: extra wide I->E kernel reach (mm); required if --gate-scale>0 (try ~2-3x l_EE).")
    ap.add_argument("--c-gate", type=int, default=None,
                    help="M2 gate: extra wide I->E in-degree; required if --gate-scale>0 (try ~100-200).")
    # ---- M3 hub-gated scaffold (all default OFF -> bit-parity): long-range hub broadcast edges +
    #      degree-normalized threshold + hub permissivity. Region overlay from topic4_corridor_substrate. ----
    ap.add_argument("--corridor-half-frac", type=float, default=0.75,
                    help="M3: corridor spans along-axis <= this*half (must exceed --sep-frac so cores are inside).")
    ap.add_argument("--hub-frac", type=float, default=0.03, help="M3: hub = this fraction of corridor cells at the +edge.")
    ap.add_argument("--global-gap-frac", type=float, default=0.0,
                    help="M3: spatial gap between corridor and global region (0=adjacent).")
    ap.add_argument("--hub-gain", type=float, default=0.0,
                    help="M3: long-range hub E->E broadcast weight as a fraction of w_EE (0=off, default).")
    ap.add_argument("--hub-long-range-c", type=int, default=12, help="M3: hub broadcast out-degree (if --hub-gain>0).")
    ap.add_argument("--l-hub-long", type=float, default=6.0, help="M3: hub broadcast kernel reach (mm).")
    ap.add_argument("--degnorm-alpha", type=float, default=0.0,
                    help="M3: degree-normalized threshold strength theta+=alpha*g_deg (0=off, default).")
    ap.add_argument("--degnorm-scheme", choices=["out_strength", "in_strength", "hybrid"],
                    default="out_strength", help="M3: degree measure (pre-registered 3-scheme comparison).")
    ap.add_argument("--hub-theta-delta", type=float, default=0.0,
                    help="M3: extra threshold on hub cells (interictal=0 high baseline; ictal<0 opens the gate).")
    # ---- M3-final W-coupled permissivity (h-coupling); default mu=0 -> short-circuit, V_th UNTOUCHED
    #      = bit-parity. delta[E] = -delta_theta * mu * h_eff[bin]. Rides the SAME V_th_per_neuron
    #      pre-transform path as --degnorm-alpha (zero engine change, no re-bless). ----
    ap.add_argument("--mu", type=float, default=0.0,
                    help="M3-final: W-coupled permissivity scalar (0=off -> V_th_per_neuron untouched = bit-parity).")
    ap.add_argument("--delta-theta", type=float, default=3.0,
                    help="M3-final: per-bin threshold-shift magnitude (mV) the h-coupling can lower (if --mu>0).")
    ap.add_argument("--h-source", choices=["struct", "resp"], default="struct",
                    help="M3-final h source: struct=ee_degree structural prior (reuses --degnorm path); "
                         "resp=measured W_resp at the mu=0 baseline (build_w_resp).")
    ap.add_argument("--h-scheme", choices=["post", "out", "hybrid"], default="post",
                    help="M3-final h scheme: post=target recruitability (primary), out=source broadcast, hybrid=mean.")
    ap.add_argument("--h-control", choices=["none", "uniform", "shuffle"], default="none",
                    help="M3-final C5 control: none=use h; uniform=flat mu; shuffle=permute h across bins.")
    ap.add_argument("--mu-impl", choices=["threshold", "inhibition"], default="threshold",
                    help="M3-final mu implementation: threshold (Phase-1); inhibition is Phase-2 (raises NotImplementedError).")
    ap.add_argument("--n-bins-per-axis", type=int, default=5,
                    help="M3-final: spatial bins per axis for the h field (n_bins = this**2).")
    ap.add_argument("--w-resp-cache", default=None,
                    help="M3-final: optional npz path for a measured W_resp (--h-source resp); load if present else measure+save.")
    ap.add_argument("--w-resp-calib-json", default=None,
                    help="M3-final: frozen calibration JSON (run_m3_kick_calibration output) "
                         "supplying calibrated_kick_boost + calibrated_win_ms for --h-source resp. "
                         "Required (or --w-resp-cache) for canonical --mu>0 --h-source resp runs.")
    ap.add_argument("--allow-pilot-default-wresp", action="store_true",
                    help="M3-final: allow pilot-default W_resp kick/window for --h-source resp "
                         "WITHOUT a calibration/cache. For smoke/testing ONLY — forbidden in "
                         "canonical pilots (which must consume frozen calibration).")
    # ---- M2 E->I recruit gate: wide E->I onto I targets (front recruits I AHEAD), default OFF ----
    ap.add_argument("--ei-gate-scale", type=float, default=0.0,
                    help="M2 recruit gate: extra wide E->I gain as a fraction of w_IE (0=off, default).")
    ap.add_argument("--l-ei-gate", type=float, default=None,
                    help="M2 recruit gate: extra wide E->I kernel reach (mm); required if --ei-gate-scale>0.")
    ap.add_argument("--c-ei-gate", type=int, default=None,
                    help="M2 recruit gate: extra wide E->I in-degree; required if --ei-gate-scale>0.")
    ap.add_argument("--prune-radius", type=float, default=None,
                    help="tail-bounded E->E candidate radius (mm); None=full pop (L<=24). "
                         "L>=28: set ~8*l_EE*sqrt(AR) ~= 4.3 mm (Stage 4 Phase 0).")
    ap.add_argument("--drive", type=float, default=DRIVE,
                    help="nu_ext_ratio (background drive); lower -> sparser / more local events")
    ap.add_argument("--T", type=float, default=1500.0)
    ap.add_argument("--core-mean", type=float, default=17.0)
    ap.add_argument("--core-std", type=float, default=1.5)
    ap.add_argument("--core-r", type=float, default=1.5)
    ap.add_argument("--sep-frac", type=float, default=0.6,
                    help="focus offset from center as a fraction of half-L (twoend separation; higher = farther apart / less coupled)")
    ap.add_argument("--dephase", type=float, default=0.3, help="twoend_deph: +mV on pos focus to separate firing times")
    ap.add_argument("--nc", type=int, default=6)
    ap.add_argument("--pitch", type=float, default=PITCH,
                    help="contact pitch (mm) per shaft; default 4.0. Phase-2 readout-escape E2 uses a "
                         "smaller pitch (e.g. 3.0) -> denser in-patch sampling so small edge events "
                         "clear the floor at the stable default k_dir. Sampling-coverage intervention.")
    ap.add_argument("--patch-elongation", type=float, default=1.0,
                    help="extended_patch axis elongation (Phase-2 readout-escape E3): 1.0=isotropic "
                         "disk (default), >1 stretches the core along the EE axis (semi-major "
                         "core_r*elongation) so end-events propagate a longer stretch -> more "
                         "participants -> readable. T0 gate catches two-hotspot degeneracy.")
    ap.add_argument("--tau-nuc-ms", type=float, default=2.0,
                    help="extended_patch nucleation-window half-width (ms) for the per-event "
                         "ground-truth seed location (analysis window, not a dynamics param).")
    ap.add_argument("--k-dir", type=int, default=KDIR,
                    help="endpoint-centroid k_dir; the readable participant floor is 2*k_dir+1 "
                         "(default 3 -> 7). Phase-2 readout-escape E1 lowers it (k_dir=2 -> floor 5) "
                         "to admit smaller edge events; moves the floor consistently everywhere.")
    ap.add_argument("--lesion", choices=["oneend_neg", "oneend_pos", "twoend_deph", "twoend_equal",
                                         "extended_patch",
                                         "oneend_inhib", "oneend_recur", "oneend_combined"],
                    default="oneend_neg")
    ap.add_argument("--ei-scale", type=float, default=0.5,
                    help="oneend_inhib/combined: w_EI multiplier for in-core E targets (<1 = perisomatic inhibition collapse)")
    ap.add_argument("--ee-gain", type=float, default=1.5,
                    help="oneend_recur/combined: w_EE multiplier for both-in-core E->E edges (>1 = recurrent cluster)")
    # ---- A1b state-topography knobs on the twoend_equal core (default 1.0 -> bit-parity). local loop =
    #      core-ei-scale (local I->E onto core E) + core-ee-gain (core recurrent E->E); global restraint =
    #      global-ei-scale (scales GABA input to EVERY E target; core gets the extra *core-ei-scale). ----
    ap.add_argument("--core-ei-scale", type=float, default=1.0,
                    help="A1b local loop: w_EI multiplier for in-core E targets (<1 = weaker local inhibition).")
    ap.add_argument("--core-ee-gain", type=float, default=1.0,
                    help="A1b local loop: w_EE multiplier for both-in-core E->E (>1 = stronger local recurrent E).")
    ap.add_argument("--global-ei-scale", type=float, default=1.0,
                    help="A1b global restraint: w_EI multiplier for ALL E targets (>1 = stronger global feedback inhibition).")
    ap.add_argument("--ei-vth-seed", type=float, default=None,
                    help="E/I lesions: if set, lower the in-core E cells' V_th to this (mild threshold seed) "
                         "instead of flat 18.0 -- nucleation-vs-propagation-relay screen (does E/I need a seed to relay out)")
    ap.add_argument("--delta-onset", type=float, default=30.0,
                    help="ms; two cores igniting within this -> collision (Stage 3 twoend_equal)")
    ap.add_argument("--n-min", type=int, default=5, help="min core E cells to count an onset")
    ap.add_argument("--out", default=None, help="output root (default: canonical OUT; set for tests/worktree)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--swap-vth", action="store_true",
                    help="source-asymmetry probe: swap the two cores' threshold RNG draws (twoend_equal)")
    ap.add_argument("--mirror-vth", action="store_true",
                    help="source-asymmetry probe: identical threshold-vs-radius profile on both cores (twoend_equal)")
    ap.add_argument("--dump-fullfield", action="store_true",
                    help="write per-event FULL-neuron-field spatial extent + n_fired_E (local-vs-global spread, not just n_part)")
    ap.add_argument("--dump-i-spikes", action="store_true",
                    help="M2 diag: record I-cell spike bool (front_lead_by_axis ahead-of-front); readout-only, bit-parity.")
    ap.add_argument("--dump-drive", action="store_true",
                    help="M2 diag: snapshot I_E/I_I at peak-active frame (clamp_check); readout-only, bit-parity.")
    # ---- A1c dynamic global feedback RESTRAINT (default off=bit-parity). I_global = gain*EMA(global E rate Hz)
    #      on E cells. NOT inhibitory exhaustion (that=A2). --dump-fb writes I_global trace + magnitude sanity. ----
    ap.add_argument("--feedback-gain", type=float, default=0.0,
                    help="A1c dynamic global feedback gain (0=off=bit-parity); I_global=gain*EMA(global E rate Hz).")
    ap.add_argument("--feedback-tau-ms", type=float, default=0.0,
                    help="A1c EMA low-pass time constant (ms); required >0 if --feedback-gain>0.")
    ap.add_argument("--dump-fb", action="store_true",
                    help="A1c: dump per-1ms-binned I_global + global E rate + I_global/median(I_I on E) magnitude sanity.")
    ap.add_argument("--fb-control", action="store_true",
                    help="A1c P1-3: after the dynamic run, RE-RUN the same network with a PRESCRIBED brake = "
                         "the dynamic I_global time-averaged-constant + time-shuffled (matched magnitude, no "
                         "causal timing). If those also terminate, termination is NOT dynamic-led. Needs --dump-fb.")
    ap.add_argument("--event-bar-mode", choices=["record_peak", "prefix_peak", "fixed_bar"],
                    default="record_peak",
                    help="event-detection bar calibration. record_peak=legacy global-max (length-dependent!); "
                         "prefix_peak=bar from first --cal-prefix-ms (length-stable); fixed_bar=--event-bar")
    ap.add_argument("--cal-prefix-ms", type=float, default=3000.0, help="prefix_peak: window for peak calibration")
    ap.add_argument("--event-bar", type=float, default=None, help="fixed_bar: explicit event-on active fraction")
    ap.add_argument("--dump-af", action="store_true",
                    help="save af trace + bin_w + bar + detected windows + detector config (length-stable warming diagnostic)")
    ap.add_argument("--ee-std-u", type=float, default=0.0,
                    help="M1: presynaptic E->E depletion fraction per spike (0=off, M0 bit-parity)")
    ap.add_argument("--ee-std-tau-ms", type=float, default=0.0,
                    help="M1: E->E availability recovery time constant (ms); required if --ee-std-u>0")
    # ---- M2 conductance shunting inhibition (default OFF = current-LIF, bit-parity) ----
    ap.add_argument("--shunt-gaba", action="store_true",
                    help="M2 faithful: conductance-based shunting inhibition (default OFF=current-LIF).")
    ap.add_argument("--e-gaba", type=float, default=None,
                    help="GABA reversal (mV); None=Params.E_gaba (=V_reset).")
    ap.add_argument("--g-gaba-scale", type=float, default=0.0,
                    help="GABA conductance scale (I_I -> g_I); required >0 with --shunt-gaba.")
    # ---- M3A quasi-static frozen slow variable layered on the Stage-3 core (z/phi/gK; default
    #      none = bit-parity). Rides the core: z/gK via the simulate_kick V_th_per_neuron hook, phi
    #      via vth_field=vth. e_GABA uses the existing --shunt-gaba/--e-gaba path (slow!=None bypasses
    #      the shunt path, so z/phi/gK and e_GABA are MUTUALLY EXCLUSIVE — one mechanism per run). ----
    ap.add_argument("--slow-var", choices=["none", "z", "phi", "gK"], default="none",
                    help="M3A frozen slow variable on the core (none=off, bit-parity).")
    ap.add_argument("--slow-level", type=float, default=None,
                    help="frozen value: z in [0,1] | phi offset mV | gK mV-equiv (required if --slow-var!=none).")
    ap.add_argument("--dump-trajectory", action="store_true",
                    help="M1: save per-event front-centroid + min-edge-distance over time (for T_stop<T_edge)")
    ap.add_argument("--dump-fwd-rev-reps", action="store_true",
                    help="extended_patch figure data: dump rep_{tag}_fwd.npz + rep_{tag}_rev.npz (cleanest "
                         "readable interior forward/reverse events from the SAME run), each with locus = that "
                         "event's nucleation centroid -> 'same patch, ~same origin, opposite read-out' figure")
    a = ap.parse_args()
    # M3A slow-var arg validation (fail fast, before the expensive network build).
    if a.slow_var != "none":
        if a.slow_level is None:
            raise SystemExit("--slow-var requires --slow-level")
        if a.shunt_gaba:
            raise SystemExit("--slow-var z/phi/gK cannot combine with --shunt-gaba (slow!=None "
                             "bypasses the shunt path); run e_GABA via --shunt-gaba separately.")
    # A1c dynamic global feedback validation (rides the default current-based membrane path).
    if a.feedback_gain > 0.0:
        if a.feedback_tau_ms <= 0.0:
            raise SystemExit("--feedback-gain>0 requires --feedback-tau-ms>0")
        if a.slow_var != "none" or a.shunt_gaba:
            raise SystemExit("A1c (--feedback-gain) is incompatible with --slow-var / --shunt-gaba "
                             "(it rides the default current-based membrane path)")
    tag = a.tag or f"{a.lesion}_s{a.seed}"
    out_dir = a.out or OUT
    k_dir = a.k_dir
    part_min = 2 * k_dir + 1          # readable participant floor; moves with --k-dir everywhere (P2)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "per_event"), exist_ok=True)
    _engine_guard()

    L, theta_rad = a.L, np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    _ee_over = {}
    if a.l_ee is not None: _ee_over["l_EE"] = a.l_ee
    if a.c_ee is not None: _ee_over["C_EE"] = a.c_ee
    p = Params(g=3.6, L=L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed, **_ee_over)
    _gate_kw = dict(gate_scale=a.gate_scale, l_gate=a.l_gate, C_gate=a.c_gate,  # M2 wide I->E veto gate
                    ei_gate_scale=a.ei_gate_scale, l_ei_gate=a.l_ei_gate, C_ei_gate=a.c_ei_gate)  # + E->I recruit gate
    rng = np.random.default_rng(a.seed)
    print(f"[{tag}] N~{int(a.density*L*L)} seed={a.seed} T={a.T} lesion={a.lesion} m={a.core_mean} ...", flush=True)
    pos, labels, NE, NI = place_neurons(p, rng)
    center = np.array([L / 2, L / 2]); half = L / 2
    # M3 region overlay + hub mask (computed always for diagnostics; mechanisms gated by their scales).
    _m3_regions = corridor_regions(pos[:NE], center, axis_unit, half,
                                   corridor_half_frac=a.corridor_half_frac, hub_frac=a.hub_frac,
                                   global_gap_frac=a.global_gap_frac)
    _hub_mask = hub_mask_E(NE, _m3_regions["hub_idx"])
    _hub_kw = dict(hub_mask_E=_hub_mask, hub_long_range_C=a.hub_long_range_c,
                   l_hub_long=a.l_hub_long, hub_gain=a.hub_gain)
    EI_LESIONS = ("oneend_inhib", "oneend_recur", "oneend_combined")
    if a.lesion in EI_LESIONS:
        # Axis-A local E/I lesion: scale synaptic weights inside ONE core (neg end), FLAT V_th --
        # excitability comes from the weight lesion, not a lowered threshold. The weight field is
        # threaded into build_connectivity_rot (NOT simulate_kick: weights are baked at build time).
        focus = center - a.sep_frac * half * axis_unit
        core_mask_E = np.linalg.norm(pos[:NE] - focus, axis=1) <= a.core_r
        local_scale_EI = None
        w_EE_gain_core = 1.0
        if a.lesion in ("oneend_inhib", "oneend_combined"):
            local_scale_EI = np.ones(NE + NI); local_scale_EI[:NE][core_mask_E] = a.ei_scale
        if a.lesion in ("oneend_recur", "oneend_combined"):
            w_EE_gain_core = a.ee_gain
        net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR,
                                     local_scale_EI=local_scale_EI, w_EE_gain_core=w_EE_gain_core,
                                     core_mask_E=core_mask_E, verbose=False, prune_radius=a.prune_radius,
                                     **_gate_kw, **_hub_kw)
        vth = np.full(NE + NI, 18.0)   # flat bare-sheet threshold (same base as the V_th lesions)
        if a.ei_vth_seed is not None:
            vth[:NE][core_mask_E] = a.ei_vth_seed   # mild in-core V_th seed (nucleation-vs-relay screen)
        core_mask = np.zeros(NE + NI, bool); core_mask[:NE] = core_mask_E
        foci = [focus]; core_masks = [core_mask]
    else:
        # A1b state-topography knobs (local loop + global restraint) on the twoend_equal core. Active
        # iff any knob != 1.0; default -> no weight lesion -> bit-parity with the V_th-only core.
        _a1b_on = (a.core_ei_scale != 1.0 or a.core_ee_gain != 1.0 or a.global_ei_scale != 1.0)
        _a1b_kw = {}
        if _a1b_on:
            neg_xy = center - a.sep_frac * half * axis_unit
            pos_xy = center + a.sep_frac * half * axis_unit
            core_mask_E = ((np.linalg.norm(pos[:NE] - neg_xy, axis=1) <= a.core_r)
                           | (np.linalg.norm(pos[:NE] - pos_xy, axis=1) <= a.core_r))  # union of both foci
            _ls, _gain = a1b_weight_lesion(NE, NI, core_mask_E, a.core_ei_scale, a.core_ee_gain, a.global_ei_scale)
            _a1b_kw = dict(local_scale_EI=_ls, w_EE_gain_core=_gain, core_mask_E=core_mask_E)
        net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR,
                                     verbose=False, prune_radius=a.prune_radius, **_a1b_kw, **_gate_kw, **_hub_kw)
        vth, core_mask, foci, core_masks = build_lesion_vth(net, NE, axis_unit, center, half, a.lesion,
                                                            a.core_mean, a.core_std, a.core_r, a.dephase, a.seed,
                                                            sep_frac=a.sep_frac, swap_vth=a.swap_vth,
                                                            mirror_vth=a.mirror_vth, elongation=a.patch_elongation)
    # M3: degree-normalized threshold (all E cells) + hub-cell permissivity (interictal high baseline /
    # ictal negative delta opens the gate). Both default OFF -> vth unchanged -> bit-parity.
    if a.degnorm_alpha != 0.0:
        vth = vth + degnorm_vth_delta(net, NE, NI, a.degnorm_alpha, a.degnorm_scheme)
    if a.hub_theta_delta != 0.0:
        vth[:NE][_hub_mask] += a.hub_theta_delta
    posE = net["pos"][:NE]
    # M3-final W-coupled permissivity (h-coupling): SAME V_th pre-transform path as --degnorm-alpha.
    # mu=0 -> short-circuit, vth UNTOUCHED -> bit-parity (M3_BASE_SHA). mu>0 -> build h (struct prior
    # or measured W_resp) -> lower E thresholds by -delta_theta*mu*h_eff[bin] BEFORE simulate_kick.
    _m3_bins = spatial_bins(posE, a.n_bins_per_axis)
    vth, _m3_perm_prov = apply_permissivity_vth_delta(
        vth, net, NE, NI, posE, a, bins=_m3_bins, p=p, V_th0=np.full(NE + NI, 18.0),
        rng=np.random.default_rng(a.seed))
    # Stage 4: per-event ground-truth nucleation centroid (single extended patch only)
    patch_E_idx = np.where(core_mask[:NE])[0] if a.lesion == "extended_patch" else None
    tau_nuc_steps = int(round(a.tau_nuc_ms / DT))        # nucleation window (--tau-nuc-ms; default 2 ms)

    m = montage(center, a.theta, 0.0, a.nc, pitch=a.pitch)
    valid = valid_mask(m, posE, L, p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(a.seed)
    # M3A: build the frozen slow variable on the core (off-by-default; one mechanism per run).
    slow = None
    if a.slow_var != "none":            # args already validated right after parse_args
        from src.sef_hfo_slowvars_quasistatic import build_frozen_slowvars
        _slow_kw = {"z": {"z": a.slow_level}, "phi": {"phi_offset": a.slow_level, "vth_field": vth},
                    "gK": {"gK": a.slow_level}}[a.slow_var]
        slow = build_frozen_slowvars(NE + NI, p.V_th, **_slow_kw)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(center), r_kick=a.core_r,
                        t_kick=1e9, V_th_per_neuron=vth, slow=slow, lfp_recorder=rec,
                        ee_std_u=a.ee_std_u, ee_std_tau_ms=a.ee_std_tau_ms,
                        shunt_gaba=a.shunt_gaba, e_gaba=a.e_gaba, g_gaba_scale=a.g_gaba_scale,
                        dump_i_spikes=a.dump_i_spikes, dump_drive=(a.dump_drive or a.dump_fb),
                        feedback_gain=a.feedback_gain, feedback_tau_ms=a.feedback_tau_ms, dump_fb=a.dump_fb)
    spk = res["E_spk_bool"]; lfp_trace = res["lfp_trace"]; times = res["times"]

    af, bin_w = active_fraction(spk, DT, BIN_MS)
    nb0, nb1 = int(BASELINE_MS[0] / bin_w), int(BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    # P1 (review 2026-06-15): the event bar must not be silently record-length-dependent. Default
    # 'record_peak' = legacy global-max (a big LATE event raises the bar for ALL earlier events ->
    # same deterministic early events read n_part=8 at T=3000 but n_part=2 at T=30000). 'prefix_peak'
    # calibrates from the first cal_prefix_ms only (short/long records share the bar; M1 workpoint);
    # 'fixed_bar' uses an explicit bar. bar_source -> provenance.
    if a.event_bar_mode == "record_peak":
        peak = float(af.max()); bar = floor + CAL_FRAC * (peak - floor); bar_src = "record_peak"
    elif a.event_bar_mode == "prefix_peak":
        npf = max(1, int(round(a.cal_prefix_ms / bin_w)))
        peak = float(af[:npf].max()); bar = floor + CAL_FRAC * (peak - floor)
        bar_src = f"prefix_peak({a.cal_prefix_ms}ms)"
    elif a.event_bar_mode == "fixed_bar":
        peak = float(af.max()); bar = float(a.event_bar); bar_src = f"fixed_bar({a.event_bar})"
    else:
        raise ValueError(f"unknown --event-bar-mode {a.event_bar_mode}")
    events = detect_events(af, bin_w, event_on_frac=bar)

    # A1b activity readouts (READOUT-only, from spk -> no bit-parity impact). Distinguish silent /
    # interictal axial-self-limited / seizure-like large-synchronized / runaway by GLOBAL rate +
    # core-vs-surround + tonic duty cycle, not just event-rate.
    _Tsec = spk.shape[0] * DT / 1000.0
    _coreE = core_mask[:NE]
    _n_core = max(int(_coreE.sum()), 1); _n_surr = max(int((~_coreE).sum()), 1)
    global_E_rate_mean_hz = round(float(spk.sum() / NE / _Tsec), 4)
    core_E_rate_mean_hz = round(float(spk[:, _coreE].sum() / _n_core / _Tsec), 4)
    surround_E_rate_mean_hz = round(float(spk[:, ~_coreE].sum() / _n_surr / _Tsec), 4)
    _bs = max(1, int(round(BIN_MS / DT))); _nb = spk.shape[0] // _bs
    _rate_bins = (spk[:_nb * _bs].reshape(_nb, _bs, NE).sum(axis=(1, 2)) / NE / (_bs * DT / 1000.0)
                  if _nb else np.zeros(1))
    global_E_rate_p95_hz = round(float(np.percentile(_rate_bins, 95)), 4)
    active_E_fraction_peak = round(float(af.max()) if len(af) else 0.0, 5)
    tonic_fraction = round(float((af > bar).mean()) if len(af) else 0.0, 4)   # duty cycle above event bar
    # A1c absolute termination readouts (raw-rate primary, §5.2): a clamped-but-elevated PLATEAU must
    # NOT read as "terminated" -> use absolute tail-to-baseline ratio, NOT the per-event-relative return.
    _rate_hz = np.asarray(res["rate_E"], float)            # per-step global E rate (Hz)
    _blo, _bhi = int(BASELINE_MS[0] / DT), int(BASELINE_MS[1] / DT)
    baseline_abs_hz = float(_rate_hz[_blo:_bhi].mean()) if _bhi > _blo else float(_rate_hz.mean())
    _tlo = max(0, len(_rate_hz) - int(500.0 / DT))
    tail_to_baseline_ratio = round(float(_rate_hz[_tlo:].mean()) / max(baseline_abs_hz, 1e-9), 3)
    _activity = dict(global_E_rate_mean_hz=global_E_rate_mean_hz, global_E_rate_p95_hz=global_E_rate_p95_hz,
                     tonic_fraction=tonic_fraction, active_E_fraction_peak=active_E_fraction_peak,
                     core_E_rate_mean_hz=core_E_rate_mean_hz, surround_E_rate_mean_hz=surround_E_rate_mean_hz,
                     tail_to_baseline_ratio=tail_to_baseline_ratio, baseline_abs_hz=round(baseline_abs_hz, 4),
                     peak_E_rate_hz=round(float(_rate_hz.max()), 3), completed=True)
    # A1c magnitude sanity (P1-3): is I_global the same order as the inhibitory currents on E cells, or a
    # 10x-1000x unit error? Report I_global_peak / median(|I_I| on E) BEFORE trusting any silent/null result.
    a1c_block = None
    if a.feedback_gain > 0.0 and a.dump_fb:
        _IIe = np.abs(res["I_I_peak"][:NE]); _IEe = np.abs(res["I_E_peak"][:NE])
        _ig = np.asarray(res["I_global_trace"], float); _medII = float(np.median(_IIe))
        a1c_block = dict(
            feedback_gain=a.feedback_gain, feedback_tau_ms=a.feedback_tau_ms,
            alpha_fb=round(float(1.0 - np.exp(-DT / a.feedback_tau_ms)), 6),
            I_global_peak=round(float(_ig.max()), 4),
            I_I_on_E_median=round(_medII, 4), I_I_on_E_p95=round(float(np.percentile(_IIe, 95)), 4),
            I_E_on_E_median=round(float(np.median(_IEe)), 4),
            I_I_core_median=round(float(np.median(_IIe[_coreE])), 4),
            I_I_surround_median=round(float(np.median(_IIe[~_coreE])), 4),
            I_global_to_I_I_ratio=round(float(_ig.max() / max(_medII, 1e-9)), 4))
        _igb = (_ig[:_nb * _bs].reshape(_nb, _bs).mean(axis=1) if _nb else _ig)   # 1ms-binned
        np.savez_compressed(os.path.join(out_dir, f"fb_{tag}.npz"),
                            I_global_bin=_igb.astype(np.float32), global_E_rate_bin=_rate_bins.astype(np.float32),
                            rate_E_hz=_rate_hz.astype(np.float32), bin_ms=BIN_MS,
                            **{k: v for k, v in a1c_block.items()})

    # A1c P1-3 matched-static / time-shuffled control (adversarial-review-hardened): re-run the SAME network
    # (rng reset to a.seed => identical draws) with a PRESCRIBED brake instead of the closed-loop EMA, to test
    # whether the dynamic termination NEEDS the brake causally locked to the rate or just enough timing-agnostic
    # restraint. Three hardenings vs the naive from-t0 design:
    #  S6 ONSET-GATE: the dynamic brake is ~0 during the early self-ignition window (r_ema starts at 0). So the
    #     control brake is held at 0 until the dynamic trace first engages (>5% of its peak), then applied over
    #     the POST-onset window only -> ignition is UNBRAKED in all conditions; a low control tail then means
    #     "terminated an IGNITED runaway", not "prevented ignition" (which a from-t0 DC could fake).
    #  S7 IGNITION GATE: record peak rate + ignited; 'terminated' = ignited AND tail<=gate; a non-ignited low
    #     tail is labeled 'prevented_no_ignition' (NOT a termination).
    #  S8 COMMON BASELINE: normalize all three tails by the DYNAMIC run's (unbraked-early) 5-50ms baseline, so
    #     dynamic/const/shuffle tails are apples-to-apples (not each by its own brake-depressed baseline).
    #  const = matched DC over the post-onset window (timing destroyed); shuffle = permutation of the post-onset
    #  dynamic trace (magnitude DISTRIBUTION preserved, causal lock destroyed).
    if a.fb_control and a.feedback_gain > 0.0 and a.dump_fb:
        _b0, _b1 = int(BASELINE_MS[0] / DT), int(BASELINE_MS[1] / DT)
        _rh_dyn = np.asarray(res["rate_E"], float)
        _bl = float(_rh_dyn[_b0:_b1].mean()) if _b1 > _b0 else float(_rh_dyn.mean())   # S8 common unbraked baseline
        def _classify(_r):
            rh = np.asarray(_r["rate_E"], float)
            tl = max(0, len(rh) - int(500.0 / DT)); pk = float(rh.max())
            ign = bool(pk >= 3.0); tail = round(float(rh[tl:].mean()) / max(_bl, 1e-9), 3)
            st = "runaway" if tail > 1.5 else ("terminated" if ign else "prevented_no_ignition")
            return dict(tail=tail, peak_rate_hz=round(pk, 2), ignited=ign, state=st)
        _igf = np.asarray(res["I_global_trace"], float); _ns = len(_igf); _pk = float(_igf.max())
        _onset = int(np.argmax(_igf > 0.05 * _pk)) if _pk > 0 else 0    # S6 first step the dynamic brake engages
        _post = _igf[_onset:]; _dc = float(_post.mean()) if len(_post) else 0.0
        _crng = np.random.default_rng(a.seed + 9173)
        _const = np.zeros(_ns); _const[_onset:] = _dc                  # 0 through ignition, then matched DC
        _shuf = np.zeros(_ns); _shuf[_onset:] = _crng.permutation(_post)  # 0 through ignition, then time-scrambled
        _ctl = {}
        for _nm, _ov in (("const", _const), ("shuffle", _shuf)):
            net["rng"] = np.random.default_rng(a.seed)           # SAME network/poisson draws as the dynamic run
            _rc = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(center), r_kick=a.core_r,
                                t_kick=1e9, V_th_per_neuron=vth, slow=None, lfp_recorder=None,
                                ee_std_u=a.ee_std_u, ee_std_tau_ms=a.ee_std_tau_ms,
                                feedback_gain=0.0, fb_override_trace=_ov)
            _ctl[_nm] = _classify(_rc)
        _dyn = _classify(res)
        fbctrl = dict(tag=tag, seed=a.seed, feedback_gain=a.feedback_gain, feedback_tau_ms=a.feedback_tau_ms,
                      tail_gate=1.5, baseline_common_hz=round(_bl, 5), onset_ms=round(_onset * DT, 1),
                      const_dc=round(_dc, 4), ig_peak=round(_pk, 4), ig_mean_full=round(float(_igf.mean()), 4),
                      control_informative=bool(_dyn["tail"] <= 1.5 and _dyn["ignited"]),   # N1
                      dynamic=_dyn, const=_ctl["const"], shuffle=_ctl["shuffle"],
                      dynamic_tail=_dyn["tail"], const_tail=_ctl["const"]["tail"], shuffle_tail=_ctl["shuffle"]["tail"],
                      note="onset-gated controls (brake=0 through the ignition window, then matched-DC / time-shuffled). "
                           "state 'terminated'=ignited AND tail<=gate; 'prevented_no_ignition'=never ignited (NOT a "
                           "termination). Read const/shuffle vs dynamic ONLY if control_informative=true.")
        json.dump(fbctrl, open(os.path.join(out_dir, f"fbctrl_{tag}.json"), "w"), indent=2)
        print(f"[{tag}] FB-CONTROL informative={fbctrl['control_informative']} onset={fbctrl['onset_ms']}ms "
              f"dyn={_dyn['tail']}/{_dyn['state']} const={_ctl['const']['tail']}/{_ctl['const']['state']} "
              f"shuffle={_ctl['shuffle']['tail']}/{_ctl['shuffle']['state']}",
              flush=True)

    env_f, fdt, _ = snn_event_envelope(spk, posE, m, DT)
    ev_recs = []
    for ev in events:
        win = (ev["t_on"], ev["t_off"])
        rd = read_event(env_f, fdt, m, valid, win, axis_unit, k_dir=k_dir, part_min=part_min, pitch=a.pitch)
        # event peak time = bin of max active fraction inside [t_on,t_off]
        s, e = int(ev["t_on"] / bin_w), int(ev["t_off"] / bin_w)
        ep = (s + int(np.argmax(af[s:e]))) * bin_w if e > s else ev["t_on"]
        # first-active contact = valid-contact name with the smallest rank (None if unreadable)
        part = {k: v for k, v in (rd["ranks"] or {}).items() if v is not None}
        first_contact = min(part, key=part.get) if part else None
        # Stage 4 ground-truth nucleation centroid (extended_patch only)
        nuc = None
        if patch_E_idx is not None:
            nc = nucleation_centroid(spk, patch_E_idx, posE, t_on_idx=int(round(ev["t_on"] / DT)),
                                     tau_nuc_steps=tau_nuc_steps, axis_unit=axis_unit,
                                     patch_center=center, k_min=5)
            nuc = None if nc is None else {
                "xy": [round(float(nc["centroid_xy"][0]), 3), round(float(nc["centroid_xy"][1]), 3)],
                "s_nuc": round(nc["s_nuc"], 3), "r_off": round(nc["r_off"], 3),
                "n_early_cells": nc["n_early_cells"]}
        ev_recs.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                            event_peak_t=round(ep, 1), returned=bool(ev["returned"]),
                            n_part=rd["n_part"], axis_err=rd["axis_err"], sign=rd["sign"],
                            readability=rd["readability"], ranks=rd["ranks"],
                            first_contact=first_contact, nucleation=nuc))

    # --- source-asymmetry probe ④: per-event FULL-neuron-field spread/duration (not just the 12
    # virtual contacts) so local vs global can be compared on the real neural field. Reuses
    # per_neuron_onset; spatial extent = std of fired E-neuron radial distances about their centroid.
    if a.dump_fullfield:
        ff = []
        for e in ev_recs:
            on = per_neuron_onset(spk, e["t_on"], e["t_off"], DT)
            fired = np.isfinite(on)
            # edge_margin is a SPATIAL OVERLAP SCREEN, not a temporal T_stop<T_edge criterion (see fn docstring)
            ext, r95, reach_axis, cx, cy, edge_margin = event_field_geometry(posE[fired], axis_unit, L)
            ff.append(dict(t_on=e["t_on"], t_off=e["t_off"], duration=round(e["t_off"] - e["t_on"], 1),
                           n_fired_E=int(fired.sum()), fullfield_extent_mm=round(ext, 3),
                           r95_mm=round(r95, 3), reach_axis_mm=round(reach_axis, 3),
                           centroid_x=round(cx, 3), centroid_y=round(cy, 3),
                           edge_margin_mm=round(edge_margin, 3),
                           n_part=e["n_part"], sign=e["sign"], returned=e["returned"]))
        json.dump(dict(tag=tag, events=ff),
                  open(os.path.join(out_dir, f"fullfield_{tag}.json"), "w"), indent=2)

    # M1 boundary-audit instrument: per-event ACTIVE-FRONT trajectory. Sort fired E neurons by onset,
    # then sample the cumulative-front at 12 growth quantiles; for each, record the front centroid
    # (cx,cy), the front's min distance to any sheet wall (edge_dist), and the onset time of the
    # last-recruited neuron at that quantile (t). edge_dist reaching ~0 LATER than t_off (or never)
    # is the per-timestep evidence for T_stop < T_edge (self-limit before boundary), distinct from
    # the static event-level edge_margin spatial screen.
    if a.dump_trajectory:
        traj = []
        for e in ev_recs:
            on = per_neuron_onset(spk, e["t_on"], e["t_off"], DT)
            fired = np.where(np.isfinite(on))[0]
            if fired.size < 2:
                continue
            tt = on[fired]
            order = np.argsort(tt); P = posE[fired][order]; tt = tt[order]
            qs = np.linspace(0, 1, 12)
            cx = []; cy = []; edge = []; margin_r95 = []; tg = []
            for q in qs:
                k = max(1, int(q * len(P)))
                Q = P[:k]; c = Q.mean(0)
                cx.append(float(c[0])); cy.append(float(c[1]))
                # edge_dist = cumulative-front BOUNDING-BOX nearest-wall distance (legacy field).
                edge.append(float(np.min([Q[:, 0].min(), L - Q[:, 0].max(),
                                          Q[:, 1].min(), L - Q[:, 1].max()])))
                # margin_r95 = PLAN edge_margin_r95(t): centroid nearest-wall MINUS r95 bulk radius of
                # the cumulative front. <=0 => the r95 bulk overflows the wall margin at that growth
                # quantile (the contract used by self-limitation: T_edge = first t with margin_r95<=0).
                r95_q = float(np.percentile(np.linalg.norm(Q - c, axis=1), 95))
                margin_r95.append(float(min(c[0], L - c[0], c[1], L - c[1]) - r95_q))
                tg.append(float(tt[k - 1]))
            traj.append(dict(t_on=e["t_on"], t_off=e["t_off"], t=tg, cx=cx, cy=cy,
                             edge_dist=edge, margin_r95=margin_r95,
                             min_edge_margin_r95=float(np.min(margin_r95))))
        json.dump(dict(tag=tag, events=traj),
                  open(os.path.join(out_dir, f"trajectory_{tag}.json"), "w"), indent=2)

    # representative event for the figure = clean, self-terminating, enough contacts, readable axis.
    # Skip the FIRST event (index 0): the network is still settling from t=0 so its pre-event
    # baseline is a transient -> faint/distorted electrode traces (the rest of the train is clean).
    clean = [(i, r) for i, r in enumerate(ev_recs)
             if r["returned"] and r["n_part"] >= part_min and r["axis_err"] is not None]
    clean_interior = [ir for ir in clean if ir[0] > 0]
    pool = clean_interior or clean
    rep_i = (max(pool, key=lambda ir: (ir[1]["readability"] or 0))[0] if pool
             else (max(range(len(ev_recs)), key=lambda i: ev_recs[i]["n_part"]) if ev_recs else None))
    rep = ev_recs[rep_i] if rep_i is not None else None

    if rep is not None:
        onset = per_neuron_onset(spk, rep["t_on"], rep["t_off"], DT)
        # the rep event's SOURCE focus: forward (sign>=0) nucleates at the -end focus (foci[0]),
        # reverse at the +end focus (foci[-1]). marked with the star; all foci are drawn as cores.
        src_focus = foci[0] if (rep["sign"] is None or rep["sign"] >= 0) else foci[-1]
        np.savez_compressed(
            os.path.join(out_dir, "per_event", f"rep_{tag}.npz"),
            posE=posE, onset_core=onset, vth=vth[:NE], is_E=np.ones(NE, bool),
            lfp=lfp_trace, times=times, contacts=np.asarray(m.contacts), names=np.array(m.names),
            nc=a.nc, kick=np.asarray(src_focus), patch=np.asarray(src_focus), patch_r=a.core_r,
            foci=np.asarray(foci), valid=valid.astype(int), L=L, theta=a.theta,
            event_peak_t=rep["event_peak_t"], event_t_on=rep["t_on"], event_t_off=rep["t_off"],
            lesion=a.lesion, sign=(rep["sign"] if rep["sign"] is not None else 0.0))

    # Stage 4 figure data: dump a forward AND a reverse readable event from the SAME run/substrate,
    # each with its OWN nucleation centroid as the marked locus (NOT foci[0]) -> the core figure can
    # show "same patch, ~same origin, opposite read-out direction" (co-primary-A null illustration).
    if a.dump_fwd_rev_reps:
        def _readable_interior(sign_want):
            cands = [(i, r) for i, r in enumerate(ev_recs)
                     if i > 0 and r["returned"] and r["sign"] == sign_want
                     and r["axis_err"] is not None and r["axis_err"] < 25 and r["n_part"] >= part_min]
            return max(cands, key=lambda ir: (ir[1]["readability"] or 0)) if cands else None
        for sign_want, suffix in ((1.0, "fwd"), (-1.0, "rev")):
            sel = _readable_interior(sign_want)
            if sel is None:
                print(f"[{tag}] dump-fwd-rev: no readable interior {suffix} event", flush=True); continue
            si, sr = sel
            onset_s = per_neuron_onset(spk, sr["t_on"], sr["t_off"], DT)
            locus = (np.asarray(sr["nucleation"]["xy"], float) if sr.get("nucleation")
                     else np.asarray(foci[0], float))
            np.savez_compressed(
                os.path.join(out_dir, "per_event", f"rep_{tag}_{suffix}.npz"),
                posE=posE, onset_core=onset_s, vth=vth[:NE], is_E=np.ones(NE, bool),
                lfp=lfp_trace, times=times, contacts=np.asarray(m.contacts), names=np.array(m.names),
                nc=a.nc, kick=locus, patch=np.asarray(foci[0], float), patch_r=a.core_r,
                foci=np.asarray(foci), valid=valid.astype(int), L=L, theta=a.theta,
                event_peak_t=sr["event_peak_t"], event_t_on=sr["t_on"], event_t_off=sr["t_off"],
                lesion=a.lesion, sign=sr["sign"])
            print(f"[{tag}] dumped rep_{tag}_{suffix}.npz (event {si}, sign={sr['sign']}, "
                  f"s_nuc={sr['nucleation']['s_nuc'] if sr.get('nucleation') else None})", flush=True)

    # "clean" = self-terminated (returned) AND readable (axis_err<25, n_part>=part_min). A boundary
    # event truncated by the sim end has returned=False -> it is NOT counted clean (it is reported
    # separately as a truncated boundary event), keeping this consistent with the rep-event gate.
    def _clean(r, s):
        return r["returned"] and r["sign"] == s and (r["axis_err"] is not None and r["axis_err"] < 25) and r["n_part"] >= part_min
    # --- legacy lagPat record (full montage, SELF-TERMINATED events) for the masked pipeline ---
    # extract_lagpat over all returned events on the full 18-contact montage -> (n_ch, n_ev) with
    # phantom-mask (non-participating contacts -> NaN). Written in the real loader's legacy keys so
    # the pooled forward+reverse record traverses masked PR-2/PR-2.5/rank-displacement unchanged.
    ret_wins = [(e["t_on"], e["t_off"]) for e in ev_recs if e["returned"]]
    env_ref_src = None
    if ret_wins:
        assert valid.all(), "off-sheet contacts present — record would be boundary-extrapolated; refuse"
        # P1/P2 (review 2026-06-15): the record-lagPat participation margin must NOT be record-length
        # -dependent either (env_f.max() is whole-record -> a big late event raises the margin -> earlier
        # events' participation/ranks shift, same length-bug as the event bar -> would bias the template
        # matrix). prefix_peak -> calibrate env ref from the first cal_prefix_ms; else legacy max.
        if a.event_bar_mode == "prefix_peak":
            npf_e = max(1, int(round(a.cal_prefix_ms / fdt)))
            env_ref_max = float(env_f[:, :npf_e].max()); env_ref_src = f"prefix({a.cal_prefix_ms}ms)"
        else:
            env_ref_max = float(env_f.max()); env_ref_src = "record_max"
        floor_g = float(env_f.min()); margin_g = MARGIN_FRAC * (env_ref_max - floor_g)
        rec_art = attach_geometry(extract_lagpat(env_f, fdt, ret_wins, floor_g, margin_g, 0.5, fdt), m)
        rec_dir = os.path.join(out_dir, "record", tag); os.makedirs(rec_dir, exist_ok=True)
        base = os.path.join(rec_dir, f"model_{tag}")
        write_legacy_npz(rec_art, base + "_lagPat_withFreqCent.npz")
        write_packed_times(rec_art, base + "_packedTimes_withFreqCent.npy")
        write_montage_manifest(rec_art, base + "_montage.json")

    # Stage 3 sidecar + the downstream synthetic-label timing controls rearrange labels in ARRAY
    # order within each collision-free block (src.sef_hfo_stage3) -> they assume events are in TIME
    # order. detect_events already emits time-order; pin it here at the data-production boundary so
    # the sidecar (and anything built from it) is guaranteed time-sorted (user 2026-06-13).
    _ton = [e["t_on"] for e in ev_recs]
    assert _ton == sorted(_ton), "events not time-sorted — sidecar/synthetic-controls assume time order"
    # --- Stage 3 sidecar (two-focus runs): hidden core-level source label per RETURNED event,
    # aligned 1:1 to the record columns (plan P1-3). build_sidecar is pure (unit-tested w/o a sim).
    stage3_source_counts = None   # sidecar-derived hidden-SOURCE counts (distinct from direction)
    if len(core_masks) == 2:
        payload = build_sidecar(ev_recs, spk, core_masks, NE, dt=DT, bin_ms=BIN_MS,
                                part_min=part_min, delta_onset=a.delta_onset, n_min=a.n_min)
        json.dump(dict(tag=tag, **payload),
                  open(os.path.join(out_dir, f"sidecar_{tag}.json"), "w"), indent=2)
        # Stage 3 GATE counts = hidden SOURCE end (neg/pos), NOT the read-out direction. A run can be
        # all-forward by direction yet have 0 neg-source events (pilot 2026-06-13) — do not conflate.
        _se = payload["events"]
        stage3_source_counts = dict(
            neg_clean=sum(1 for e in _se if e["hidden_source_label"] == "neg" and e["clean_for_timing"]),
            pos_clean=sum(1 for e in _se if e["hidden_source_label"] == "pos" and e["clean_for_timing"]),
            collision=sum(1 for e in _se if e["hidden_source_label"] == "collision"),
            ambiguous=sum(1 for e in _se if e["hidden_source_label"] == "ambiguous"),
            collision_rate=payload["collision_rate"])

    n_fwd = sum(1 for r in ev_recs if _clean(r, 1.0))
    n_rev = sum(1 for r in ev_recs if _clean(r, -1.0))
    n_trunc_dir = sum(1 for r in ev_recs if (not r["returned"]) and r["sign"] in (1.0, -1.0)
                      and (r["axis_err"] is not None and r["axis_err"] < 25) and r["n_part"] >= part_min)
    # TRUE inter-event baseline = p95 active fraction OUTSIDE detected event windows (the detector's
    # floor is only the 5-50ms calibration window). Gates "core quasi-continuous" in the stage-2 gate.
    _imask = np.ones(len(af), bool)
    for e in ev_recs:
        _imask[max(0, int((e["t_on"] - 10) / bin_w)):int((e["t_off"] + 10) / bin_w)] = False
    true_floor = round(float(np.percentile(af[_imask], 95)), 5) if _imask.any() else None
    summary = dict(tag=tag, provenance=_provenance(),
                   config=dict(L=L, density=a.density, theta=a.theta, AR=a.AR, drive=a.drive, T=a.T, NE=int(NE),
                               l_EE=p.l_EE, C_EE=p.C_EE,
                               l_IE=p.l_IE, C_IE=p.C_IE, l_EI=p.l_EI, C_EI=p.C_EI,
                               gate_scale=a.gate_scale, l_gate=a.l_gate, c_gate=a.c_gate,
                               ei_gate_scale=a.ei_gate_scale, l_ei_gate=a.l_ei_gate, c_ei_gate=a.c_ei_gate,
                               shunt_gaba=a.shunt_gaba, e_gaba=a.e_gaba, g_gaba_scale=a.g_gaba_scale,
                               slow_var=a.slow_var, slow_level=a.slow_level,
                               core_ei_scale=a.core_ei_scale, core_ee_gain=a.core_ee_gain,
                               global_ei_scale=a.global_ei_scale,
                               feedback_gain=a.feedback_gain, feedback_tau_ms=a.feedback_tau_ms,
                               local_global_ratio=round(local_global_ratio(a.core_ee_gain, a.core_ei_scale,
                                                                           a.global_ei_scale), 4),
                               ee_std_u=a.ee_std_u, ee_std_tau_ms=a.ee_std_tau_ms,
                               hub_gain=a.hub_gain, hub_long_range_c=a.hub_long_range_c, l_hub_long=a.l_hub_long,
                               degnorm_alpha=a.degnorm_alpha, degnorm_scheme=a.degnorm_scheme,
                               **_m3_perm_prov,   # M3-final: mu/delta_theta/h_source/h_scheme/h_control/mu_impl/n_bins/lambda0
                               hub_theta_delta=a.hub_theta_delta, corridor_half_frac=a.corridor_half_frac,
                               hub_frac=a.hub_frac, global_gap_frac=a.global_gap_frac,
                               n_corridor=int(_m3_regions["corridor_idx"].size),
                               n_hub=int(_m3_regions["hub_idx"].size),
                               n_global=int(_m3_regions["global_idx"].size),
                               lesion=a.lesion, core_mean=a.core_mean, core_std=a.core_std,
                               core_r=a.core_r, sep_frac=a.sep_frac, dephase=a.dephase, nc=a.nc, seed=a.seed,
                               ei_scale=a.ei_scale, ee_gain=a.ee_gain, ei_vth_seed=a.ei_vth_seed,
                               foci=[[round(float(f[0]), 2), round(float(f[1]), 2)] for f in foci],
                               margin_frac=MARGIN_FRAC, n_core=int(core_mask.sum()),
                               n_valid_contacts=int(valid.sum()), tau_nuc_ms=a.tau_nuc_ms,
                               k_dir=k_dir, part_min=part_min, pitch=a.pitch,
                               patch_elongation=a.patch_elongation),
                   detector=dict(floor=round(floor, 4), peak=round(peak, 4), bar=round(bar, 4),
                                 event_bar_mode=a.event_bar_mode, bar_source=bar_src,
                                 record_env_ref=env_ref_src,  # prefix_peak length-stable record-margin source
                                 true_inter_event_floor=true_floor),
                   activity=_activity,   # A1b: global / core / surround E rate + tonic + peak active frac
                   a1c=a1c_block,        # A1c: feedback gain/tau + I_global magnitude sanity (P1-3)
                   n_events=len(ev_recs),
                   # n_clean_* are DIRECTION (read-out sign) counts, NOT hidden-source counts;
                   # Stage 3 gates read stage3_source_counts (sidecar hidden neg/pos), see above.
                   n_clean_forward=n_fwd, n_clean_reverse=n_rev,
                   n_truncated_directional=n_trunc_dir,
                   stage3_source_counts=stage3_source_counts,
                   rep_event_index=rep_i, events=ev_recs)
    json.dump(summary, open(os.path.join(out_dir, f"readout_{tag}.json"), "w"), indent=2)
    if a.dump_af:
        # length-stable warming diagnostic: full af trace + the windows the CURRENT bar produced.
        # Lets an analyzer recompute rolling p50/p95 af (bar-independent) and re-detect under any bar.
        np.savez_compressed(os.path.join(out_dir, f"af_{tag}.npz"),
                            af=af.astype(np.float32), bin_w=bin_w, floor=floor, peak=peak, bar=bar,
                            event_bar_mode=a.event_bar_mode, bar_source=bar_src,
                            win_on=np.array([e["t_on"] for e in ev_recs], float),
                            win_off=np.array([e["t_off"] for e in ev_recs], float),
                            n_part=np.array([e["n_part"] for e in ev_recs], int))
    # T0 continuous-patch gate artifact (extended_patch): a covert-two-focus / low-n run must NOT be
    # silently pooled as an extended-patch pass (Phase 2 plan §T0). Written alongside the readout.
    if a.lesion == "extended_patch":
        _sn = np.array([e["nucleation"]["s_nuc"] if e.get("nucleation") else np.nan for e in ev_recs], float)
        _ro = np.array([e["nucleation"]["r_off"] if e.get("nucleation") else np.nan for e in ev_recs], float)
        t0 = compute_t0_gate(_sn, _ro, patch_r=a.core_r, elongation=a.patch_elongation)
        json.dump(dict(tag=tag, patch_r=a.core_r, k_dir=k_dir, part_min=part_min, **t0),
                  open(os.path.join(out_dir, f"t0_gate_{tag}.json"), "w"), indent=2)
        print(f"[{tag}] T0 gate: {t0['hotspot_degeneracy']['verdict']} "
              f"(n_valid_nucleation={t0['n_valid_nucleation']})", flush=True)
    if stage3_source_counts is not None:
        print(f"[{tag}] stage3 SOURCE counts (gate): {stage3_source_counts}", flush=True)
    print(f"[{tag}] events={len(ev_recs)} clean DIRECTION fwd/rev={n_fwd}/{n_rev} (+{n_trunc_dir} truncated boundary) "
          f"| rep_event={rep_i} (rd={rep['readability'] if rep else None} err={rep['axis_err'] if rep else None}) "
          f"| bar={bar:.4f} peak={peak:.4f}", flush=True)


if __name__ == "__main__":
    main()
