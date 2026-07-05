"""M3A-v2.2 pilot: sustained ramp+HOLD protocol -> (1) slow-off baseline C1 branch,
(2) Exp-0 sensor calibration eligibility over a small r_hold LADDER (C6 fail-closed),
(3) q_I+g_K sustained pilot. Builds the CARRIER only -- NO closed-loop h_G grid, NO
surrogate battery, NO ablation (Deferred).

C1-A 'failure_mode_preserved': slow-off under ramp+HOLD still cannot self-recover (runaway /
  not returned) -> the stress baseline suitable for testing h_G.
C1-B 'protocol_changed_substrate': ramp+HOLD itself already returns / grades the event ->
  do NOT attribute recovery to h_G; re-calibrate or change protocol.

This is a necessary-condition SCREEN, not a seizure-mechanism validation (meta.screen_type).
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_m3a_v2_step2_qI as S2                          # SUBSTRATES, build (dict!), N_GRID, CORRIDOR_HW
from params import compute_nu_theta                       # snn_engine local; returns a TUPLE
from kick_probe import simulate_kick                       # NO T_override; sim length = p.T
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field
from src.topic4_m3a_v2_2_protocol import ramp_hold_drive
from src.topic4_m3a_v2_phenotype import (recruitment_area, axis_score, offaxis_fraction,
                                         participation_ratio, make_field_grid_xy, classify_event)

N_GRID, CORRIDOR_HW = S2.N_GRID, S2.CORRIDOR_HW


def _participation(E_spk, i_on, i_off, dt):
    """(time, neuron) SEMANTICS LOCK: E_spk_bool is [time, neuron]. Slice the TIME window
    [i_on, i_off), collapse TIME (axis=0) -> per-NEURON participation; argmax(axis=0) -> per-neuron
    onset step. Returns (ever[NE] bool, onset[NE] float, NaN for non-participants)."""
    seg = np.asarray(E_spk)[i_on:i_off]                   # (window, NE)
    ever = seg.any(axis=0)                                # per-NEURON (NOT per-time)
    onset = np.where(ever, seg.argmax(axis=0) * dt + i_on * dt, np.nan)
    return ever, onset


def _event_window(rate, dt, settle_ms=40.0, gap_ms=10.0, tonic_thr=0.5):
    """Baseline-crossing segmentation of the SUSTAINED-protocol rate. Returns
    (i_on, i_off, seg_status, n_components, tonic_fraction) or None if no event.
    seg_status='single_event' ONLY when ONE contiguous burst spans < tonic_thr of the recording;
    multi-burst (>1 component, gap>gap_ms) or long-tonic (span >= tonic_thr) -> 'TONIC_OR_MULTIBURST'
    (fail-closed downstream -- a tonic/multi-burst run is NOT one event)."""
    rate = np.asarray(rate, float)
    nb = max(1, int(settle_ms / dt))
    base_mu, base_sd = float(rate[:nb].mean()), float(rate[:nb].std())
    above = np.where(rate > base_mu + 3.0 * (base_sd + 1e-9))[0]
    if above.size == 0:
        return None
    i_on, i_off = int(above[0]), int(above[-1]) + 1
    bursts = np.split(above, np.where(np.diff(above) > max(1, int(gap_ms / dt)))[0] + 1)
    n_components = len(bursts)
    tonic_fraction = float(i_off - i_on) / max(1, len(rate))
    seg_status = "single_event" if (n_components == 1 and tonic_fraction < tonic_thr) else "TONIC_OR_MULTIBURST"
    return i_on, i_off, seg_status, n_components, tonic_fraction


def _segment_and_classify(res, S, settle_ms=40.0, tail_ms=80.0):
    """Segment the sustained-protocol event, then phenotype it. FAIL-CLOSED:
      * no event window           -> class_label = 'INSUFFICIENT'
      * multi-burst / long tonic  -> class_label = 'INSUFFICIENT_FOR_EVENT_PHENOTYPE'
    (never hard-classify a tonic/multi-burst run as a single-event phenotype). Always reports
    segmentation_status / n_components / tonic_fraction."""
    dt = S["p"].dt
    rate = np.asarray(res["rate_E"], float)
    posE, L, u, c = S["posE"], S["L"], S["axis_unit"], S["center"]
    nb = max(1, int(settle_ms / dt))
    base_mu, base_sd = float(rate[:nb].mean()), float(rate[:nb].std())
    seg = _event_window(rate, dt, settle_ms=settle_ms)
    if seg is None:
        return dict(n_onsets=0, R_area=0.0, S_axis=float("nan"), F_offaxis=0.0, G_PR=0.0,
                    recovery=True, peak_rate=float(rate.max()), t_on=None, t_off=None,
                    segmentation_status="no_event", n_components=0, tonic_fraction=0.0,
                    class_label="INSUFFICIENT")
    i_on, i_off, seg_status, n_comp, tonic_frac = seg
    ever, onset = _participation(res["E_spk_bool"], i_on, i_off, dt)
    A = firing_rate_field(ever, posE, L, N_GRID, 0.5)
    gxy = make_field_grid_xy(L, N_GRID)
    A_thr = 0.20 * A.max() if A.max() > 0 else 0.0
    tail = rate[i_off:min(len(rate), i_off + int(tail_ms / dt))]
    returned = bool(tail.size > 0 and tail.mean() <= base_mu + 1.5 * base_sd + 1e-9)
    m = dict(n_onsets=int(ever.sum()),
             R_area=float(recruitment_area(A, A_thr)),
             S_axis=float(axis_score(posE, onset, u)),
             F_offaxis=float(offaxis_fraction(A, gxy, c, u, CORRIDOR_HW)),
             G_PR=float(participation_ratio(A)),
             recovery=returned, peak_rate=float(rate.max()),
             t_on=float(i_on * dt), t_off=float(i_off * dt),
             segmentation_status=seg_status, n_components=int(n_comp), tonic_fraction=float(tonic_frac))
    m["class_label"] = ("INSUFFICIENT_FOR_EVENT_PHENOTYPE" if seg_status != "single_event"
                        else classify_event(m))            # fail-closed: tonic/multi-burst != event
    return m


def _drive(S, r_hold, t0=50.0, t_ramp=200.0):
    nu_theta, _, _ = compute_nu_theta(S["p"])             # TUPLE return
    return ramp_hold_drive(nu_theta, r0=S["p"].nu_ext_ratio, r_hold=r_hold, t0=t0, t_ramp=t_ramp)


def _run(S, slow, nu_fn, seed):
    # PAIRED comparison + ORDER-INVARIANCE: reset the noise stream before EVERY arm so slow-off /
    # ladder / q_I+g_K see the IDENTICAL OU realization -> attribution is to the variable, not to run
    # order (mirrors tests/test_m3a_v2_spatial_slowvars._run, which resets net["rng"] per run).
    S["net"]["rng"] = np.random.default_rng(seed)
    return simulate_kick(S["p"], S["net"], KICK_BOOST=0.0, slow=slow, nu_signal_fn=nu_fn,
                         kick_center=S["core_xy"], r_kick=0.3, t_kick=50.0,
                         V_th_per_neuron=S["vth"])         # T comes from p.T (set in build)


def _c1_branch(m0):
    """C1 contract (encoded, not inferred): B 'protocol_changed_substrate' iff the slow-off run
    ALREADY produced a returned single-event phenotype (interictal_axial / expanded_axial /
    ictal_like_candidate -- ANY clean returned event, axial OR broken) -> do NOT attribute recovery
    to slow vars. A 'failure_mode_preserved' otherwise (runaway / tonic-or-multiburst / no clean
    event = the v2.1 all-or-none failure mode persists -> the stress baseline for testing h_G)."""
    graded_returned = m0["recovery"] and m0["class_label"] in (
        "interictal_axial", "expanded_axial", "ictal_like_candidate")
    return "B_protocol_changed_substrate" if graded_returned else "A_failure_mode_preserved"


def run_pilot(substrate="primary", seed=1, T=500.0, r_hold=0.6, fast=False):
    sub = S2.SUBSTRATES[substrate]
    S = S2.build(sub, seed, T=T)                          # build returns a DICT; T flows into p.T

    # (1) slow-off baseline -> C1 branch  (seed reset inside _run -> paired)
    m0 = _segment_and_classify(_run(S, None, _drive(S, r_hold), seed), S)
    c1 = _c1_branch(m0)

    # (2) Exp-0 sensor calibration over a small r_hold LADDER (slow-off) -> C6 fail-closed.
    #     A single run cannot be BOTH returned-axial AND runaway -> a ladder is required.
    ladder = [0.50] if fast else [0.50, 0.60, 0.75]
    anchors = []
    for rh in ladder:
        a = _segment_and_classify(_run(S, None, _drive(S, rh), seed), S)
        a["r_hold"] = rh
        anchors.append(a)
    n_ret_axial = sum(a["recovery"] and a["class_label"] in ("interictal_axial", "expanded_axial")
                      for a in anchors)
    n_runaway = sum((not a["recovery"]) for a in anchors)
    eligible = n_ret_axial >= 1 and n_runaway >= 1       # need BOTH anchor kinds across the ladder
    exp0 = dict(eligibility="eligible" if eligible else "UNCALIBRATED",
                n_returned_axial=int(n_ret_axial), n_runaway=int(n_runaway),
                ladder=[a["r_hold"] for a in anchors],
                note="anchors from an r_hold ladder; M50/B50/Pi50 frozen ONLY when eligible (C6)")

    # (3) q_I + g_K sustained pilot (carrier check; NO h_G) -- does the brake make a partial fill?
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=True, use_hG=False,
                                 k_q=0.3, sigma_q=1.5, q_min=0.25,
                                 k_K=1.0, sigma_K=0.5, eta_K=1.0, tau_a=20.0)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    m2 = _segment_and_classify(_run(S, slow, _drive(S, r_hold), seed), S)

    return dict(meta=dict(substrate=substrate, seed=seed, T=S["p"].T, r_hold=r_hold,
                          protocol="ramp_hold", carrier_only=True,
                          screen_type="pilot-gate / necessary-condition screen -- NOT seizure mechanism validation"),
                slow_off=dict(c1_branch=c1, **m0), exp0=exp0, qI_gK_pilot=m2)


def _json_safe(obj):
    """Recursively map non-finite floats (NaN/Inf -- e.g. S_axis=NaN on INSUFFICIENT events) to None
    so the artifact is STRICT JSON. A research artifact must not rely on a lax parser; callers pair
    this with json.dumps(..., allow_nan=False)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=["primary"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--T", type=float, default=500.0)
    ap.add_argument("--out", default="results/topic4_m3a_v2_2_pilot")
    a = ap.parse_args()
    rows = [run_pilot(s, sd, T=a.T) for s in a.substrates for sd in a.seeds]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_results.json").write_text(json.dumps(
        _json_safe(dict(meta=dict(substrates=a.substrates, seeds=a.seeds, T=a.T, carrier_only=True), rows=rows)),
        indent=2, allow_nan=False))                       # strict JSON (NaN S_axis -> null)
    print(f"wrote {out / 'pilot_results.json'}")


if __name__ == "__main__":
    main()
