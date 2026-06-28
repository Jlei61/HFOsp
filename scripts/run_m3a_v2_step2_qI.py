"""M3A-v2 Step 2 — q_I ONLY (g_K=0, D_EE=1) on the qualified robust substrate(s).

Goal (narrow, per user 2026-06-28): show q_I(x,t) turns a local interictal axial event into an
EXPANDED AXIAL recruitment -- R_area UP, T_event UP, but S_axis STILL HIGH, F_off LOW-to-MODERATE,
returned=True. NOT seizure / NOT axis-breaking (that needs g_K, Step 3). If q_I-only already breaks
the axis / globalizes, that is q_I too wide/strong, not ictal-like.

Design choices (faithful to the spec):
- k_q is parameterized as a TARGET AXIAL DEPLETION dq_axis_target = 1 - q_axis_min, calibrated per
  (substrate, seed, sigma_q) by replaying the BASELINE event's spikes through the field with a small
  test k_q and scaling (K_q is mass-normalized, so sigma_q is a pure spatial-spread knob). dq=0 -> the
  q_I-off baseline. Achieved q_axis_min is measured (closed-loop disinhibition usually exceeds target).
- sigma_q swept narrow->wide; q_min swept; tau_q/tau_a FIXED (no tau sweep this round).
- per-seed raw rows keep the full schema incl. q_axis/off/global_min + q_depl_gap, and PAIRED deltas
  vs the same-seed baseline. A-E class per run.

Output -> results/topic4_m3a_v2_step2_qI/. DESCRIPTIVE screen. Multi-seed.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                          # noqa: E402
import json                                              # noqa: E402
import multiprocessing as mp                             # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402
from collections import defaultdict                      # noqa: E402

import numpy as np                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                                # noqa: E402
from connectivity import place_neurons                   # noqa: E402
from connectivity_rot import build_connectivity_rot      # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig, firing_rate_field  # noqa: E402
from src.sef_hfo_heterogeneity import sample_core_field  # noqa: E402
from src.sef_hfo_snn_metrics import self_limit, pre_kick_ignition  # noqa: E402
from src.topic4_m3a_v2_phenotype import (                # noqa: E402
    recruitment_area, axis_score, offaxis_fraction, make_field_grid_xy, region_masks)

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_step2_qI")
N_GRID, CORRIDOR_HW, THETA_A_FRAC = 32, 1.5, 0.20
T_KICK, EVENT_DUR, T_SIM = 120.0, 130.0, 500.0
TAU_Q, TAU_A = 5000.0, 20.0                              # fixed this round (no tau sweep)
KQ_TEST = 0.02                                           # calibration probe

# the 3 qualified substrates (Step-1 sweep): primary=default I->E, sensitivity, headroom backup
SUBSTRATES = {
    "primary":     dict(AR=4.0, g=8.0, l_EI=0.25, C_EI=200, nu=0.46),
    "sensitivity": dict(AR=4.0, g=8.0, l_EI=0.50, C_EI=200, nu=0.46),
    "backup":      dict(AR=4.0, g=5.0, l_EI=1.00, C_EI=400, nu=0.46),
}
SIGMA_Q = {"narrow": 0.75, "matched": 1.0, "moderate": 1.5, "wide": 2.0}
Q_MIN = [0.8, 0.7, 0.6]
DQ_TARGET = [0.0, 0.05, 0.10, 0.20, 0.30]
KICK = 3.0


def _round(x, nd=4):
    return None if (x is None or (isinstance(x, float) and x != x)) else round(float(x), nd)


class RecordingSlowField(SpatialSlowField):
    """SpatialSlowField that records <q_I> over axis / offaxis / whole-sheet masks each step, so we
    can read q_axis_min / q_off_min / q_global_min after the run."""
    def set_masks(self, am, om):
        self._am, self._om = am, om
        self.q_axis_tr, self.q_off_tr, self.q_glob_tr = [], [], []

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)
        self.q_axis_tr.append(float(self.q_I[self._am].mean()))
        self.q_off_tr.append(float(self.q_I[self._om].mean()))
        self.q_glob_tr.append(float(self.q_I.mean()))


def build(sub, seed, T=T_SIM):
    """Single-core anisotropic substrate (Step-1 recipe)."""
    L, theta = 10.0, np.radians(45.0)
    axis_unit = np.array([np.cos(theta), np.sin(theta)])
    p = Params(g=sub["g"], L=L, density=100.0, T=T, dt=0.1, nu_ext_ratio=sub["nu"], seed=seed,
               w_EE=0.1575, l_EE=0.380, C_EE=800, l_EI=sub["l_EI"], C_EI=sub["C_EI"])
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=theta, AR=sub["AR"], verbose=False)
    pos = net["pos"]; N = NE + NI
    is_E = np.zeros(N, bool); is_E[:NE] = True
    center = np.array([L / 2, L / 2]); half = L / 2
    core_xy = center - 0.6 * half * axis_unit
    vth = sample_core_field(pos, is_E, core_xy, 1.0, np.random.default_rng(seed + 7),
                            core_mean=16.5, core_std=1.0, base_mean=18.0)["vth"]
    masks = region_masks(L, N_GRID, center, axis_unit, CORRIDOR_HW)
    return dict(p=p, net=net, NE=NE, NI=NI, posE=pos[:NE], posI=pos[NE:], N=N, labels=labels,
                vth=vth, core_xy=core_xy, axis_unit=axis_unit, center=center, L=L, masks=masks)


def _slow_cfg(sigma_q, q_min, k_q):
    return SpatialSlowFieldConfig(use_qI=True, use_gK=False, k_q=k_q, k_K=0.0, sigma_q=sigma_q,
                                  sigma_K=0.5, q_min=q_min, tau_q=TAU_Q, tau_a=TAU_A, q_init=1.0)


def _readout(res, S, slow=None):
    """source-space metrics + (if slow) q_I diagnostics. Returns the per-run row fields."""
    dt = S["p"].dt; posE, L, u, c = S["posE"], S["L"], S["axis_unit"], S["center"]
    E_spk, rate = res["E_spk_bool"], res["rate_E"]
    i_lo, i_hi = int(round(T_KICK / dt)), int(round((T_KICK + EVENT_DUR) / dt))
    post = E_spk[i_lo:i_hi]; ever = post.any(axis=0)
    onset = np.where(ever, post.argmax(axis=0) * dt + T_KICK, np.nan)
    A = firing_rate_field(ever, posE, L, N_GRID, sigma=0.5)
    gxy = make_field_grid_xy(L, N_GRID)
    A_thr = THETA_A_FRAC * A.max() if A.max() > 0 else 0.0
    sl = self_limit(rate, dt, T_KICK)
    ig, _ = pre_kick_ignition(rate, dt, T_KICK)
    # tail AUC over [peak_t, T_SIM): area of rate above rest (returned-energy proxy)
    t = np.arange(len(rate)) * dt
    pk_t = sl["peak_t"]; tail = rate[t >= pk_t]
    tail_auc = float(np.maximum(tail - sl["rest_rate"], 0).sum() * dt)
    row = dict(n_onsets=int(ever.sum()), R_area=_round(recruitment_area(A, A_thr)),
               S_axis=_round(axis_score(posE, onset, u)),
               F_off=_round(offaxis_fraction(A, gxy, c, u, CORRIDOR_HW)),
               T_event=_round(sl["burst_duration_ms"], 1), peak_rate=_round(sl["peak"], 1),
               returned=bool(sl["returned"] and sl["tail_complete"]), T_return=_round(pk_t, 1),
               tail_AUC=_round(tail_auc, 1), pre_ignited=bool(ig))
    if slow is not None:
        qa, qo, qg = min(slow.q_axis_tr), min(slow.q_off_tr), min(slow.q_glob_tr)
        row.update(q_axis_min=_round(qa), q_off_min=_round(qo), q_global_min=_round(qg),
                   q_axis_depl=_round(1 - qa), q_off_depl=_round(1 - qo),
                   q_depl_gap=_round((1 - qa) - (1 - qo)))
    return row


def classify(row, base):
    """A no-effect | B expanded-axial (TARGET) | C over-expanded | D global-disinhibition | E runaway."""
    R, S, F, R0 = row["R_area"], row["S_axis"], row["F_off"], base["R_area"]
    if not row["returned"]:
        return "E_runaway"
    if S is None or S != S:
        return "E_runaway"                                # unreadable axis after a strong event
    if S < 0.85 and F is not None and F > 0.30:
        return "D_global_disinhibition"                   # axis dropped + off-axis rose (q_I too wide/strong)
    grew = (R > R0 + 0.05) or (R0 > 0 and R / R0 > 1.2)
    if R > 0.70 and S >= 0.85:
        return "C_over_expanded"                          # big but still axial
    if grew and S > 0.90 and (F is None or F < 0.30):
        return "B_expanded_axial"                         # TARGET
    if not grew:
        return "A_no_effect"
    return "B_expanded_axial" if S > 0.85 else "D_global_disinhibition"


def worker(task):
    """One (substrate, seed, sigma_q): build, run baseline (q_I off), calibrate k_q to each dq_target,
    run q_I closed-loop for every (q_min, dq>0). Returns the baseline row + the q_I rows (paired)."""
    sub_id, seed, sq_name = task
    sub = SUBSTRATES[sub_id]; sigma_q = SIGMA_Q[sq_name]
    S = build(sub, seed)
    p, net, NE = S["p"], S["net"], S["NE"]
    base_meta = dict(substrate=sub_id, seed=seed, sigma_q=sigma_q, sigma_q_name=sq_name,
                     **{k: sub[k] for k in ("AR", "g", "l_EI", "C_EI", "nu")}, kick=KICK)

    # --- baseline (slow off, dump I spikes for the calibration replay) ---
    net["rng"] = np.random.default_rng(seed)
    res0 = simulate_kick(p, net, KICK_BOOST=KICK, slow=None, kick_center=S["core_xy"], r_kick=0.3,
                         t_kick=T_KICK, V_th_per_neuron=S["vth"], dump_i_spikes=True)
    base_row = dict(**base_meta, q_min=None, dq_target=0.0, k_q=0.0,
                    **_readout(res0, S), class_label="baseline")
    base_row["dR"] = base_row["dS"] = base_row["dF"] = 0.0
    spk_full = np.concatenate([res0["E_spk_bool"], res0["I_spk_bool"]], axis=1)   # (nsteps, N)

    # --- calibrate: replay baseline spikes with a small test k_q -> dq per unit k_q ---
    cal = RecordingSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"],
                             cfg=_slow_cfg(sigma_q, q_min=0.0, k_q=KQ_TEST))
    cal.set_masks(S["masks"]["axis"], S["masks"]["offaxis"])
    for tstep in range(spk_full.shape[0]):
        cal.step(spk_full[tstep], S["labels"], p.dt)
    dq_test = 1.0 - min(cal.q_axis_tr)                    # axial depletion from the test k_q
    kq_per_dq = (KQ_TEST / dq_test) if dq_test > 1e-9 else None

    rows = [base_row]
    for q_min in Q_MIN:
        for dq in [d for d in DQ_TARGET if d > 0]:
            k_q = (dq * kq_per_dq) if kq_per_dq is not None else 0.0
            slow = RecordingSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"],
                                      cfg=_slow_cfg(sigma_q, q_min, k_q))
            slow.set_masks(S["masks"]["axis"], S["masks"]["offaxis"])
            net["rng"] = np.random.default_rng(seed)
            res = simulate_kick(p, net, KICK_BOOST=KICK, slow=slow, kick_center=S["core_xy"],
                                r_kick=0.3, t_kick=T_KICK, V_th_per_neuron=S["vth"])
            r = _readout(res, S, slow=slow)
            row = dict(**base_meta, q_min=q_min, dq_target=dq, k_q=_round(k_q, 5), **r)
            row["class_label"] = classify(row, base_row)
            row["dR"] = _round(row["R_area"] - base_row["R_area"])
            row["dS"] = _round((row["S_axis"] or 0) - (base_row["S_axis"] or 0))
            row["dF"] = _round((row["F_off"] or 0) - (base_row["F_off"] or 0))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--sigmas", nargs="+", default=list(SIGMA_Q))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(s, sd, sq) for s in a.substrates for sd in a.seeds for sq in a.sigmas]
    print(f"Step 2 q_I: {len(tasks)} (substrate,seed,sigma_q) tasks x {len(Q_MIN) * (len(DQ_TARGET)-1)} "
          f"closed-loop each = {len(tasks) * (1 + len(Q_MIN) * (len(DQ_TARGET)-1))} SNN runs", flush=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        raw = [r for rows in pool.map(worker, tasks) for r in rows]
    wall = time.time() - t0

    cls = defaultdict(int)
    for r in raw:
        if r["class_label"] != "baseline":
            cls[r["class_label"]] += 1
    # config-level aggregate (substrate,sigma_q,q_min,dq): class distribution + mean dR/dS/dF over seeds
    by = defaultdict(list)
    for r in raw:
        if r["class_label"] != "baseline":
            by[(r["substrate"], r["sigma_q_name"], r["q_min"], r["dq_target"])].append(r)
    agg = []
    for key, rs in by.items():
        n = len(rs)
        labs = defaultdict(int)
        for r in rs:
            labs[r["class_label"]] += 1
        agg.append(dict(substrate=key[0], sigma_q=key[1], q_min=key[2], dq_target=key[3], n_seeds=n,
                        n_B_expanded=labs["B_expanded_axial"], n_returned=sum(r["returned"] for r in rs),
                        classes=dict(labs), dR_mean=_round(np.mean([r["dR"] for r in rs])),
                        dS_mean=_round(np.mean([r["dS"] for r in rs])),
                        dF_mean=_round(np.mean([r["dF"] for r in rs])),
                        R_mean=_round(np.mean([r["R_area"] for r in rs])),
                        S_mean=_round(np.mean([(r["S_axis"] or 0) for r in rs])),
                        qgap_mean=_round(np.mean([(r["q_depl_gap"] or 0) for r in rs])),
                        q_axis_min_mean=_round(np.mean([(r["q_axis_min"] or 1) for r in rs]))))
    agg.sort(key=lambda r: (-r["n_B_expanded"], -r["dR_mean"]))

    payload = dict(meta=dict(date="2026-06-28", step="Step 2 q_I only (g_K=0, D_EE=1)",
                             substrates=SUBSTRATES, sigma_q=SIGMA_Q, q_min=Q_MIN, dq_target=DQ_TARGET,
                             tau_q=TAU_Q, tau_a=TAU_A, n_runs=len(raw), wall_s=round(wall, 1),
                             class_counts=dict(cls)),
                   aggregates=agg, raw_rows=raw)
    json.dump(payload, open(os.path.join(a.out, "step2_results.json"), "w"), indent=2)

    print(f"\n{len(raw)} rows in {wall:.0f}s. class counts: {dict(cls)}", flush=True)
    print("\nTOP config cells by #B_expanded (substrate sigma_q q_min dq | nB/n returned dR dS qgap):")
    for r in agg[:15]:
        print(f"  {r['substrate']:11} sq={r['sigma_q']:8} qmin={r['q_min']} dq={r['dq_target']} | "
              f"B={r['n_B_expanded']}/{r['n_seeds']} ret={r['n_returned']} dR={r['dR_mean']:+.3f} "
              f"dS={r['dS_mean']:+.3f} qgap={r['qgap_mean']:+.3f}")
    print(f"\nwrote {os.path.join(a.out, 'step2_results.json')}", flush=True)


if __name__ == "__main__":
    main()
