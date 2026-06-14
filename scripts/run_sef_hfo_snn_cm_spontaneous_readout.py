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

ENG = os.path.join("results", "topic4_sef_hfo", "lif_snn", "engine")
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

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
ENGINE_VERSIONS = os.path.join("results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
PITCH, SHAFT_OFFSETS = 4.0, (0.0, 90.0)   # ∥ and ⊥ to the EE axis (theta_EE), like the small-scale grid
MARGIN_FRAC, KDIR, PART_MIN = 0.10, 3, 7
DT, DRIVE = 0.1, 0.6
BIN_MS = 1.0
BASELINE_MS = (5.0, 50.0)
CAL_FRAC = 0.5
_ENG_FILES = ("kick_probe.py", "lfp.py", "connectivity.py", "connectivity_rot.py", "params.py")


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


def montage(center, theta_deg, rot_deg, nc):
    """2 shafts: A = ∥ EE axis (theta_EE), B = ⊥ (theta_EE+90), like the small-scale grid. Aligning
    the ∥ shaft to the propagation axis means BOTH forward and reverse waves traverse the SAME ∥
    contacts -> high shared participation -> cleanly rank-reversed (opposite) templates. The earlier
    misaligned 3-shaft montage (15/75/135) gave only 6/18 shared contacts -> masked imputation
    inflated the inter-cluster corr to +0.74 and the forward/reverse opposition went undetected
    (even though each template was monotonic with axis-projection). cm four-control C3 already proved
    shaft-rotation-invariance, so aligning here is illustration + coverage, not tuning to the answer."""
    rot = np.deg2rad(rot_deg)
    return merge_montages([build_shaft(np.deg2rad(theta_deg + off) + rot, PITCH, nc, tuple(center), chr(65 + i))
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
                     dephase, seed, sep_frac=0.6, swap_vth=False, mirror_vth=False):
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


def read_event(env_f, fdt, m, valid, win, axis_unit):
    """Direction read of ONE event window via valid contacts (endpoint-centroid axis on firing
    envelope). Returns dict(n_part, corr_sign vs axis, axis_err_deg, readability, ranks per name)."""
    vi = np.where(valid)[0]
    if len(vi) < PART_MIN:
        return dict(n_part=0, axis_err=None, readability=None, ranks=None, names=None)
    env_v = env_f[vi]
    names_v = [m.names[i] for i in vi]
    m_v = VirtualMontage(np.asarray(m.contacts)[vi], names_v, "valid_subset")
    floor = float(env_v.min()); margin = MARGIN_FRAC * (float(env_v.max()) - floor)
    art = extract_lagpat(env_v, fdt, [win], floor, margin, 0.5, fdt)
    art = attach_geometry(art, m_v)
    r0, b0 = art.ranks[:, 0], art.bools[:, 0]
    ax = endpoint_centroid_axis(r0, b0, art.contact_coords, k_dir=KDIR, eps_deg=0.5 * PITCH)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--density", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--AR", type=float, default=2.0)
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
    ap.add_argument("--lesion", choices=["oneend_neg", "oneend_pos", "twoend_deph", "twoend_equal"],
                    default="oneend_neg")
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
    a = ap.parse_args()
    tag = a.tag or f"{a.lesion}_s{a.seed}"
    out_dir = a.out or OUT
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "per_event"), exist_ok=True)
    _engine_guard()

    L, theta_rad = a.L, np.deg2rad(a.theta)
    axis_unit = np.array([np.cos(theta_rad), np.sin(theta_rad)])
    p = Params(g=3.6, L=L, density=a.density, T=a.T, dt=DT, nu_ext_ratio=a.drive, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    print(f"[{tag}] N~{int(a.density*L*L)} seed={a.seed} T={a.T} lesion={a.lesion} m={a.core_mean} ...", flush=True)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta_rad, AR=a.AR, verbose=False)
    posE = net["pos"][:NE]
    center = np.array([L / 2, L / 2]); half = L / 2
    vth, core_mask, foci, core_masks = build_lesion_vth(net, NE, axis_unit, center, half, a.lesion,
                                                        a.core_mean, a.core_std, a.core_r, a.dephase, a.seed,
                                                        sep_frac=a.sep_frac, swap_vth=a.swap_vth,
                                                        mirror_vth=a.mirror_vth)

    m = montage(center, a.theta, 0.0, a.nc)
    valid = valid_mask(m, posE, L, p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=m.contacts)
    net["rng"] = np.random.default_rng(a.seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(center), r_kick=a.core_r,
                        t_kick=1e9, V_th_per_neuron=vth, lfp_recorder=rec)
    spk = res["E_spk_bool"]; lfp_trace = res["lfp_trace"]; times = res["times"]

    af, bin_w = active_fraction(spk, DT, BIN_MS)
    nb0, nb1 = int(BASELINE_MS[0] / bin_w), int(BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    peak = float(af.max()); bar = floor + CAL_FRAC * (peak - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)

    env_f, fdt, _ = snn_event_envelope(spk, posE, m, DT)
    ev_recs = []
    for ev in events:
        win = (ev["t_on"], ev["t_off"])
        rd = read_event(env_f, fdt, m, valid, win, axis_unit)
        # event peak time = bin of max active fraction inside [t_on,t_off]
        s, e = int(ev["t_on"] / bin_w), int(ev["t_off"] / bin_w)
        ep = (s + int(np.argmax(af[s:e]))) * bin_w if e > s else ev["t_on"]
        ev_recs.append(dict(t_on=round(ev["t_on"], 1), t_off=round(ev["t_off"], 1),
                            event_peak_t=round(ep, 1), returned=bool(ev["returned"]),
                            n_part=rd["n_part"], axis_err=rd["axis_err"], sign=rd["sign"],
                            readability=rd["readability"], ranks=rd["ranks"]))

    # --- source-asymmetry probe ④: per-event FULL-neuron-field spread/duration (not just the 12
    # virtual contacts) so local vs global can be compared on the real neural field. Reuses
    # per_neuron_onset; spatial extent = std of fired E-neuron radial distances about their centroid.
    if a.dump_fullfield:
        ff = []
        for e in ev_recs:
            on = per_neuron_onset(spk, e["t_on"], e["t_off"], DT)
            fired = np.isfinite(on)
            if fired.sum() > 1:
                P = posE[fired]; ext = float(np.std(np.linalg.norm(P - P.mean(0), axis=1)))
            else:
                ext = 0.0
            ff.append(dict(t_on=e["t_on"], t_off=e["t_off"], duration=round(e["t_off"] - e["t_on"], 1),
                           n_fired_E=int(fired.sum()), fullfield_extent_mm=round(ext, 3),
                           n_part=e["n_part"], sign=e["sign"], returned=e["returned"]))
        json.dump(dict(tag=tag, events=ff),
                  open(os.path.join(out_dir, f"fullfield_{tag}.json"), "w"), indent=2)

    # representative event for the figure = clean, self-terminating, enough contacts, readable axis.
    # Skip the FIRST event (index 0): the network is still settling from t=0 so its pre-event
    # baseline is a transient -> faint/distorted electrode traces (the rest of the train is clean).
    clean = [(i, r) for i, r in enumerate(ev_recs)
             if r["returned"] and r["n_part"] >= PART_MIN and r["axis_err"] is not None]
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

    # "clean" = self-terminated (returned) AND readable (axis_err<25, n_part>=PART_MIN). A boundary
    # event truncated by the sim end has returned=False -> it is NOT counted clean (it is reported
    # separately as a truncated boundary event), keeping this consistent with the rep-event gate.
    def _clean(r, s):
        return r["returned"] and r["sign"] == s and (r["axis_err"] is not None and r["axis_err"] < 25) and r["n_part"] >= PART_MIN
    # --- legacy lagPat record (full montage, SELF-TERMINATED events) for the masked pipeline ---
    # extract_lagpat over all returned events on the full 18-contact montage -> (n_ch, n_ev) with
    # phantom-mask (non-participating contacts -> NaN). Written in the real loader's legacy keys so
    # the pooled forward+reverse record traverses masked PR-2/PR-2.5/rank-displacement unchanged.
    ret_wins = [(e["t_on"], e["t_off"]) for e in ev_recs if e["returned"]]
    if ret_wins:
        assert valid.all(), "off-sheet contacts present — record would be boundary-extrapolated; refuse"
        floor_g = float(env_f.min()); margin_g = MARGIN_FRAC * (float(env_f.max()) - floor_g)
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
                                part_min=PART_MIN, delta_onset=a.delta_onset, n_min=a.n_min)
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
                      and (r["axis_err"] is not None and r["axis_err"] < 25) and r["n_part"] >= PART_MIN)
    # TRUE inter-event baseline = p95 active fraction OUTSIDE detected event windows (the detector's
    # floor is only the 5-50ms calibration window). Gates "core quasi-continuous" in the stage-2 gate.
    _imask = np.ones(len(af), bool)
    for e in ev_recs:
        _imask[max(0, int((e["t_on"] - 10) / bin_w)):int((e["t_off"] + 10) / bin_w)] = False
    true_floor = round(float(np.percentile(af[_imask], 95)), 5) if _imask.any() else None
    summary = dict(tag=tag, provenance=_provenance(),
                   config=dict(L=L, density=a.density, theta=a.theta, AR=a.AR, drive=a.drive, T=a.T, NE=int(NE),
                               lesion=a.lesion, core_mean=a.core_mean, core_std=a.core_std,
                               core_r=a.core_r, sep_frac=a.sep_frac, dephase=a.dephase, nc=a.nc, seed=a.seed,
                               foci=[[round(float(f[0]), 2), round(float(f[1]), 2)] for f in foci],
                               margin_frac=MARGIN_FRAC, n_core=int(core_mask.sum())),
                   detector=dict(floor=round(floor, 4), peak=round(peak, 4), bar=round(bar, 4),
                                 true_inter_event_floor=true_floor),
                   n_events=len(ev_recs),
                   # n_clean_* are DIRECTION (read-out sign) counts, NOT hidden-source counts;
                   # Stage 3 gates read stage3_source_counts (sidecar hidden neg/pos), see above.
                   n_clean_forward=n_fwd, n_clean_reverse=n_rev,
                   n_truncated_directional=n_trunc_dir,
                   stage3_source_counts=stage3_source_counts,
                   rep_event_index=rep_i, events=ev_recs)
    json.dump(summary, open(os.path.join(out_dir, f"readout_{tag}.json"), "w"), indent=2)
    if stage3_source_counts is not None:
        print(f"[{tag}] stage3 SOURCE counts (gate): {stage3_source_counts}", flush=True)
    print(f"[{tag}] events={len(ev_recs)} clean DIRECTION fwd/rev={n_fwd}/{n_rev} (+{n_trunc_dir} truncated boundary) "
          f"| rep_event={rep_i} (rd={rep['readability'] if rep else None} err={rep['axis_err'] if rep else None}) "
          f"| bar={bar:.4f} peak={peak:.4f}", flush=True)


if __name__ == "__main__":
    main()
