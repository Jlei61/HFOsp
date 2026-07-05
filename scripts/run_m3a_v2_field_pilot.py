"""M3A-v2 spatial slow-variable field — reproducible pilot runner (DESCRIPTIVE SCREEN ONLY).

Calibration/pilot is a plan Deferred item; this regenerates the numbers + interpretation in
docs/archive/topic4/m3a_v2_field_pilot_2026-06-28.md. Emits JSON to
results/topic4_m3a_v2_field_pilot/ so the pilot is a reproducible artifact (not a hand table):
others can re-run, change params, and audit the grid convention + proxy computation.

Scientific status (do NOT overstate): field-only probe = mechanism SANITY (a wide-disinhibition /
narrow-fatigue carrier CAN manufacture an off-axis excitability advantage); the closed-loop SNN
ictal-like broken-axis transition is NOT established (the substrate enters full-field recruitment
before any localized axial propagation). Axis-breaking is gated behind ablation and is unlocked by
neither result. Source-space onset gradient is the locked axis instrument.

Usage:
  python scripts/run_m3a_v2_field_pilot.py --experiment all          # ~10-15 min (SNN runs)
  python scripts/run_m3a_v2_field_pilot.py --experiment fieldmap     # fast (no SNN)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                                          # noqa: E402
from connectivity import place_neurons                             # noqa: E402
from connectivity_rot import build_connectivity_rot                # noqa: E402
from kick_probe import simulate_kick                               # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field            # noqa: E402
from src.sef_hfo_snn_metrics import self_limit, pre_kick_ignition  # noqa: E402
from src.topic4_m3a_v2_phenotype import (                          # noqa: E402
    recruitment_area, axis_score, offaxis_fraction, participation_ratio, classify_event,
    make_field_grid_xy, region_masks, proxy_phase_point)

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_field_pilot")
N_GRID = 32
CORRIDOR_HW = 1.5


# ===================== substrate / run / source-space readout =====================
def build_two_core(L=10.0, density=100.0, theta_deg=45.0, AR=2.0, drive=0.6,
                   core_mean=17.5, core_std=1.5, core_r=1.5, sep_frac=0.6, seed=1, T=400.0, g=3.6):
    """Anisotropic E-I SNN + two excitable cores along the theta_EE axis (build_lesion_vth
    twoend_equal recipe): vth = min of two single-core fields (both lowered to core_mean)."""
    theta = np.radians(theta_deg)
    axis_unit = np.array([np.cos(theta), np.sin(theta)])
    p = Params(g=g, L=L, density=density, T=T, dt=0.1, nu_ext_ratio=drive, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta, AR=AR, verbose=False)
    pos = net["pos"]; N = NE + NI
    is_E = np.zeros(N, bool); is_E[:NE] = True
    center = np.array([L / 2, L / 2]); half = L / 2
    neg_xy = center - sep_frac * half * axis_unit
    pos_xy = center + sep_frac * half * axis_unit
    f_neg = sample_core_field(pos, is_E, neg_xy, core_r, np.random.default_rng(seed + 7),
                              core_mean=core_mean, core_std=core_std, base_mean=18.0)
    f_pos = sample_core_field(pos, is_E, pos_xy, core_r, np.random.default_rng(seed + 8),
                              core_mean=core_mean, core_std=core_std, base_mean=18.0)
    vth = np.minimum(f_neg["vth"], f_pos["vth"])
    return dict(p=p, net=net, NE=NE, NI=NI, posE=pos[:NE], posI=pos[NE:], N=N, vth=vth,
                foci=[neg_xy, pos_xy], axis_unit=axis_unit, center=center, L=L)


def run_kicked(sub, slow_cfg=None, KICK_BOOST=2.0, t_kick=150.0, r_kick=0.5, kick_focus=0, seed=1):
    sub["net"]["rng"] = np.random.default_rng(seed)
    slow = (SpatialSlowField(sub["N"], 18.0, sub["posE"], sub["posI"], sub["L"], cfg=slow_cfg)
            if slow_cfg is not None else None)
    res = simulate_kick(sub["p"], sub["net"], KICK_BOOST=KICK_BOOST, slow=slow,
                        kick_center=sub["foci"][kick_focus], r_kick=r_kick, t_kick=t_kick,
                        V_th_per_neuron=sub["vth"])
    res["_slow"] = slow
    return res


def _round(x, nd=4):
    return None if (x is None or (isinstance(x, float) and x != x)) else round(float(x), nd)


def readout(res, sub, t_kick=150.0, event_dur=120.0, theta_A_frac=0.20):
    """Source-space readout + four-state classify. grid_xy via make_field_grid_xy (tested
    field[iy,ix] convention); axis/off q_I diagnostics via region_masks (same convention)."""
    dt = sub["p"].dt
    posE, L, u_axis, center = sub["posE"], sub["L"], sub["axis_unit"], sub["center"]
    E_spk, rate_E = res["E_spk_bool"], res["rate_E"]

    i_lo, i_hi = int(round(t_kick / dt)), int(round((t_kick + event_dur) / dt))
    post = E_spk[i_lo:i_hi]
    ever = post.any(axis=0)
    onset_win = np.where(ever, post.argmax(axis=0) * dt + t_kick, np.nan)
    A = firing_rate_field(ever, posE, L, N_GRID, sigma=0.5)
    grid_xy = make_field_grid_xy(L, N_GRID)
    A_thr = theta_A_frac * A.max() if A.max() > 0 else 0.0
    sl = self_limit(rate_E, dt, t_kick)
    ignited, ig_lat = pre_kick_ignition(rate_E, dt, t_kick)

    metrics = dict(n_onsets=int(ever.sum()), R_area=recruitment_area(A, A_thr),
                   S_axis=axis_score(posE, onset_win, u_axis),
                   F_offaxis=offaxis_fraction(A, grid_xy, center, u_axis, CORRIDOR_HW),
                   G_PR=participation_ratio(A), recovery=bool(sl["returned"]))
    out = dict(label=classify_event(metrics), pre_kick_ignited=bool(ignited),
               pre_kick_latency=_round(ig_lat, 1), peak_rate=_round(sl["peak"], 2),
               rest_rate=_round(sl["rest_rate"], 3),
               **{k: (_round(v) if isinstance(v, float) else v) for k, v in metrics.items()})
    slow = res.get("_slow")
    if slow is not None:
        masks = region_masks(L, slow.cfg.n_grid, center, u_axis, CORRIDOR_HW)
        out.update(qI_axis=_round(float(slow.q_I[masks["axis"]].mean())),
                   qI_offaxis=_round(float(slow.q_I[masks["offaxis"]].mean())),
                   qI_min=_round(float(slow.q_I.min())),
                   gK_axis=_round(float(slow.g_K[masks["axis"]].mean())))
    return out


def run_field_probe(cfg, L=10.0, theta_deg=45.0, sep_frac=0.6, n_E=4000, n_I=1000,
                    n_steps=4000, dt=0.1, seed=1, lgr=1.0):
    """Field-only probe: PRESCRIBE sustained axis-corridor firing, advance the field, return the
    end-state mask-mean q_I/g_K (axis vs off), net excitability, and proxy X at beta=0.3 and
    beta=eta_K. net = (1-q_I) - eta_K*g_K (disinhibition minus fatigue)."""
    rng = np.random.default_rng(seed)
    posE = rng.uniform(0, L, (n_E, 2)); posI = rng.uniform(0, L, (n_I, 2))
    N = n_E + n_I; labels = np.r_[np.zeros(n_E, int), np.ones(n_I, int)]
    center = np.array([L / 2, L / 2]); half = L / 2
    theta = np.radians(theta_deg); u_axis = np.array([np.cos(theta), np.sin(theta)])
    aperp = np.array([-u_axis[1], u_axis[0]])
    perpE = np.abs((posE - center) @ aperp); alongE = (posE - center) @ u_axis
    corridor = (perpE <= CORRIDOR_HW) & (np.abs(alongE) <= sep_frac * half)
    fire = np.zeros(N, bool); fire[:n_E] = corridor

    fld = SpatialSlowField(N, 18.0, posE, posI, L, cfg=cfg)
    masks = region_masks(L, cfg.n_grid, center, u_axis, CORRIDOR_HW)
    am, om = masks["axis"], masks["offaxis"]
    for _ in range(n_steps):
        fld.step(fire, labels, dt)
    qa, qo = float(fld.q_I[am].mean()), float(fld.q_I[om].mean())
    ga, go = float(fld.g_K[am].mean()), float(fld.g_K[om].mean())
    net_axis, net_off = (1.0 - qa) - cfg.eta_K * ga, (1.0 - qo) - cfg.eta_K * go
    X03, _ = proxy_phase_point(fld, masks, lgr, 0.3)
    Xeta, _ = proxy_phase_point(fld, masks, lgr, cfg.eta_K)
    return dict(n_corridor=int(corridor.sum()), qI_axis=_round(qa), qI_off=_round(qo),
                gK_axis=_round(ga), gK_off=_round(go), net_axis=_round(net_axis),
                net_off=_round(net_off), mech=_round(net_off - net_axis), X_b03=_round(X03),
                X_beta_etaK=_round(Xeta))


# ===================== experiments (deterministic, seed=1) =====================
def exp_baseline_regime():
    rows = []
    for core_mean in (17.5, 17.8, 18.0):
        for drive in (0.40, 0.55, 0.70):
            sub = build_two_core(L=10.0, T=400.0, seed=1, core_mean=core_mean, drive=drive)
            o = readout(run_kicked(sub, slow_cfg=None), sub)
            rows.append(dict(core_mean=core_mean, drive=drive, **o))
            print(f"  baseline core_mean={core_mean} drive={drive}: preig={o['pre_kick_ignited']} "
                  f"S_axis={o['S_axis']} R_area={o['R_area']} rec={o['recovery']} -> {o['label']}")
    return rows


def exp_field_map():
    base = dict(tau_q=500.0, tau_K=500.0, tau_a=20.0, k_q=0.05, k_K=0.05, q_min=0.0,
                sigma_K=0.5, eta_K=1.0)
    rows = []
    for sigma_q in (0.5001, 1.0, 1.5, 2.0, 2.5, 3.0):
        cfg = SpatialSlowFieldConfig(sigma_q=sigma_q, **base)
        r = run_field_probe(cfg, n_steps=4000)
        rows.append(dict(sigma_q=round(sigma_q, 2), **r))
        print(f"  fieldmap sigma_q={sigma_q:.2f}: net[ax/off]={r['net_axis']}/{r['net_off']} "
              f"mech={r['mech']} X(b.3)={r['X_b03']} X(b=etaK)={r['X_beta_etaK']}")
    return rows


def exp_closed_loop():
    sub = build_two_core(L=10.0, T=400.0, seed=1, core_mean=17.5, drive=0.5)
    off = readout(run_kicked(sub, slow_cfg=None), sub)
    on = []
    for sq in (1.5, 2.5):
        cfg = SpatialSlowFieldConfig(sigma_q=sq, sigma_K=0.5, k_q=0.5, k_K=0.5, eta_K=1.0,
                                     tau_q=500.0, tau_K=500.0, tau_a=20.0, q_min=0.0)
        o = readout(run_kicked(sub, slow_cfg=cfg), sub)
        on.append(dict(sigma_q=sq, **o))
        print(f"  closed-loop slow_on sigma_q={sq}: qI[ax/off]={o.get('qI_axis')}/{o.get('qI_offaxis')} "
              f"R_area={o['R_area']} -> {o['label']}")
    return dict(slow_off=off, slow_on=on)


def exp_localize():
    rows = []
    for g in (3.6, 6.0):
        for kick in (0.5, 1.0, 2.0):
            sub = build_two_core(L=10.0, T=400.0, seed=1, core_mean=17.5, drive=0.45, g=g)
            o = readout(run_kicked(sub, slow_cfg=None, KICK_BOOST=kick), sub)
            rows.append(dict(g=g, KICK_BOOST=kick, **o))
            print(f"  localize g={g} kick={kick}: n_on={o['n_onsets']} S_axis={o['S_axis']} "
                  f"R_area={o['R_area']} -> {o['label']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("all", "baseline", "fieldmap", "closedloop", "localize"),
                    default="all")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runners = dict(baseline=("baseline_regime", exp_baseline_regime),
                   fieldmap=("field_map", exp_field_map),
                   closedloop=("closed_loop", exp_closed_loop),
                   localize=("localize", exp_localize))
    todo = list(runners) if a.experiment == "all" else [a.experiment]
    payload = dict(meta=dict(date="2026-06-28", seed=1, n_grid=N_GRID, corridor_hw=CORRIDOR_HW,
                             status="descriptive screen; field-only=mechanism sanity, closed-loop "
                                    "ictal-like transition NOT established (substrate full-field "
                                    "recruitment before localized axial propagation); ablation-gated"))
    for key in todo:
        name, fn = runners[key]
        print(f">>> {name}")
        payload[name] = fn()
    out_path = os.path.join(a.out, "pilot_results.json")
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
