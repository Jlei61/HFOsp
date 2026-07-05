"""M3A-v2 Step 3 — q_I + g_K (D_EE=1) boundary-oriented RESCUE scout.

Step 2 showed q_I-only either does nothing (returned, no growth) or, when strong, runs away with axis
readout collapse (NOT demonstrated off-axis recruitment). The decisive Step-3 question (user
2026-06-28): can a use-dependent recovery brake g_K convert that q_I-driven destabilization into a
CONTROLLED, REVERSIBLE off-axis recruitment (= ictal-like candidate)?

Boundary-oriented: don't re-scan all q_I. For each (substrate, q_I boundary setting (dq,q_min), sigma_q,
seed), the Gamma_K=0 cell IS the q_I-only reference; we then add g_K at calibrated strengths and ask
whether the runaway/near-boundary references become returned + off-axis.

Calibrations (relative, not raw):
- k_q -> TARGET axial depletion dq (baseline-event replay, Step-2 method; K_q mass-normalized).
- k_K -> TARGET Gamma_K = eta_K*g_K_axis^peak / (dI_disinh_axis^peak + eps), dI_disinh_axis =
  (1-q_axis_min)*<I_I>_axis (I_I from simulate_kick dump_drive at the peak frame). g_K_axis^peak per
  unit k_K from replaying the q_I-only event's E-activity through the g_K field.

KEY judgment lock (review P1-2): 'off-axis recruitment' REQUIRES F_off (or global/low-k) RISING, not
just S_axis dropping (a collapse of the axis readout is not recruitment).

Output -> results/topic4_m3a_v2_step3_qIgK/. DESCRIPTIVE screen. Multi-seed, per-seed auditable.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                          # noqa: E402
import itertools                                         # noqa: E402
import json                                              # noqa: E402
import multiprocessing as mp                             # noqa: E402
import sys                                               # noqa: E402
import time                                              # noqa: E402
from collections import defaultdict                      # noqa: E402

import numpy as np                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m3a_v2_step2_qI as S2                          # noqa: E402  (build, SUBSTRATES, _readout, etc.)
from kick_probe import simulate_kick                     # noqa: E402
from slow_field import SpatialSlowFieldConfig            # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "topic4_m3a_v2_step3_qIgK")
KK_TEST = 0.02
SAT_FACTOR = 3.0      # k_K = SAT_FACTOR/gK_per -> over-drives g_K to ~gK_max (saturated brake footprint)
GK_MAX = 1.0
TAU_K = S2.TAU_Q
GAMMA_K = [0.0, 0.5, 1.0, 1.5, 2.0]          # 0 == q_I-only reference
DQ_QMIN = [(0.20, 0.7), (0.30, 0.6)]          # near-boundary, deep (Step-2 runaway-prone region)
SIGMA_Q = [1.5, 2.0]
RATIOS = [1.0, 1.5, 2.0]                       # sigma_q/sigma_K (1.0 -> no width gap control; >1 -> gap)
_round = S2._round


class S3Field(S2.RecordingSlowField):
    """Records q_I AND g_K over axis/offaxis masks each step."""
    def set_masks(self, am, om):
        super().set_masks(am, om)
        self.gK_axis_tr, self.gK_off_tr = [], []

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)            # advances field + records q_axis/off/global
        self.gK_axis_tr.append(float(self.g_K[self._am].mean()))
        self.gK_off_tr.append(float(self.g_K[self._om].mean()))


def _cfg(sigma_q, sigma_K, q_min, k_q, k_K, eta_K=1.0):
    return SpatialSlowFieldConfig(use_qI=True, use_gK=(k_K > 0), k_q=k_q, k_K=k_K, sigma_q=sigma_q,
                                  sigma_K=sigma_K, q_min=q_min, eta_K=eta_K, gK_max=GK_MAX,
                                  tau_q=S2.TAU_Q, tau_K=TAU_K, tau_a=S2.TAU_A, q_init=1.0)


def _field(S, cfg):
    f = S3Field(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    f.set_masks(S["masks"]["axis"], S["masks"]["offaxis"])
    return f


def _replay_peak_gK(S, spk_E, sigma_q, sigma_K):
    """Replay the q_I-only event's E-activity through a g_K-only field (test k_K); return peak <g_K>_axis
    per unit k_K (linear calibration handle)."""
    cfg = SpatialSlowFieldConfig(use_qI=False, use_gK=True, k_q=0.0, k_K=KK_TEST, sigma_q=sigma_q,
                                 sigma_K=sigma_K, q_min=0.0, eta_K=1.0, tau_q=S2.TAU_Q, tau_K=TAU_K,
                                 tau_a=S2.TAU_A)
    f = _field(S, cfg)
    N = S["N"]; nE = S["NE"]
    spk = np.zeros(N, bool)
    for tstep in range(spk_E.shape[0]):
        spk[:nE] = spk_E[tstep]                  # g_K only uses E activity; I spikes irrelevant
        f.step(spk, S["labels"], S["p"].dt)
    peak = max(f.gK_axis_tr) if f.gK_axis_tr else 0.0
    return peak / KK_TEST if peak > 0 else 0.0


def classify_s3(row, ref):
    """A_ictal_like (rescued if ref ran away) | B_gK_oversuppress | C1_still_runaway | C2_still_axial.
    off-axis recruitment REQUIRES F_off rising (not just S_axis dropping)."""
    if not row["returned"]:
        return "C1_still_runaway"
    R, S, F = row["R_area"], row["S_axis"], row["F_off"]
    Rref, Fref = ref["R_area"], (ref["F_off"] or 0)
    off_axis = (F is not None) and (F > 0.30 or (F - Fref) > 0.05)     # F_off RISE (locked)
    axis_dropped = (S is not None) and (S < 0.85)
    gk_axial = (row.get("gK_axis_peak") or 0) >= (row.get("gK_off_peak") or 0)
    if off_axis and axis_dropped and gk_axial:
        return "A_rescued_ictal_like" if not ref["returned"] else "A_ictal_like_candidate"
    if R is not None and Rref is not None and R < Rref - 0.05 and not ref["returned"]:
        return "B_gK_oversuppress"                                     # g_K killed a would-be-runaway event
    if R is not None and Rref is not None and R < 0.10:
        return "B_gK_oversuppress"                                     # event suppressed away
    return "C2_still_axial"                                            # returned but no off-axis recruitment


def worker(task):
    sub_id, dq, q_min, sigma_q, seed = task
    sub = S2.SUBSTRATES[sub_id]
    S = S2.build(sub, seed)
    p, nE = S["p"], S["NE"]
    meta = dict(substrate=sub_id, seed=seed, dq_target=dq, q_min=q_min, sigma_q=sigma_q,
                **{k: sub[k] for k in ("AR", "g", "l_EI", "C_EI", "nu")})
    # axis-corridor E-cell mask (neuron space) for <I_I>_axis
    u = S["axis_unit"]; uperp = np.array([-u[1], u[0]])
    corr_E = np.abs((S["posE"] - S["center"]) @ uperp) <= S2.CORRIDOR_HW

    # --- baseline (slow off) -> k_q calibration (Step-2 method) ---
    S["net"]["rng"] = np.random.default_rng(seed)
    res0 = simulate_kick(p, S["net"], KICK_BOOST=S2.KICK, slow=None, kick_center=S["core_xy"],
                         r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"], dump_i_spikes=True)
    spk_base = np.concatenate([res0["E_spk_bool"], res0["I_spk_bool"]], axis=1)
    cal = _field(S, _cfg(sigma_q, sigma_q / 1.001, 0.0, S2.KQ_TEST, 0.0))
    for t in range(spk_base.shape[0]):
        cal.step(spk_base[t], S["labels"], p.dt)
    dq_test = 1.0 - min(cal.q_axis_tr)
    k_q = (dq * S2.KQ_TEST / dq_test) if dq_test > 1e-9 else 0.0

    # --- q_I-only reference (Gamma_K=0): k_q on, g_K off, dump_drive for I_I ---
    ref_f = _field(S, _cfg(sigma_q, sigma_q / 1.001, q_min, k_q, 0.0))
    S["net"]["rng"] = np.random.default_rng(seed)
    res_ref = simulate_kick(p, S["net"], KICK_BOOST=S2.KICK, slow=ref_f, kick_center=S["core_xy"],
                            r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"], dump_drive=True)
    ref_row = dict(**meta, sigma_K=None, ratio=None, Gamma_K=0.0, k_q=_round(k_q, 5), k_K=0.0,
                   **S2._readout(res_ref, S, slow=ref_f), gK_axis_peak=0.0, gK_off_peak=0.0,
                   Gamma_K_achieved=0.0)
    ref_row["class_label"] = "qonly_runaway" if not ref_row["returned"] else "qonly_returned"
    q_axis_min = min(ref_f.q_axis_tr)
    II_axis = float(np.mean(res_ref["I_I_peak"][:nE][corr_E])) if res_ref.get("I_I_peak") is not None else 0.0
    dI_disinh = max((1.0 - q_axis_min) * II_axis, 1e-6)
    spk_ref_E = res_ref["E_spk_bool"]

    rows = [ref_row]
    for ratio in RATIOS:
        sigma_K = sigma_q / ratio if ratio > 1.0 else sigma_q / 1.001
        gK_per_kK = _replay_peak_gK(S, spk_ref_E, sigma_q, sigma_K)
        k_K = (SAT_FACTOR / gK_per_kK) if gK_per_kK > 0 else 0.0   # over-drive -> g_K_axis saturates ~gK_max
        for G in [g for g in GAMMA_K if g > 0]:
            eta_K = G * dI_disinh / GK_MAX                         # Gamma_K tuned via the COUPLING eta_K
            f = _field(S, _cfg(sigma_q, sigma_K, q_min, k_q, k_K, eta_K))
            S["net"]["rng"] = np.random.default_rng(seed)
            res = simulate_kick(p, S["net"], KICK_BOOST=S2.KICK, slow=f, kick_center=S["core_xy"],
                                r_kick=0.3, t_kick=S2.T_KICK, V_th_per_neuron=S["vth"])
            r = S2._readout(res, S, slow=f)
            gKa, gKo = max(f.gK_axis_tr), max(f.gK_off_tr)
            Gach = eta_K * gKa / dI_disinh                         # achieved Gamma_K (g_K may not fully saturate)
            row = dict(**meta, sigma_K=_round(sigma_K), ratio=ratio, Gamma_K=G, k_q=_round(k_q, 5),
                       k_K=_round(k_K, 5), eta_K=_round(eta_K, 3), **r, gK_axis_peak=_round(gKa),
                       gK_off_peak=_round(gKo), gK_gap=_round(gKa - gKo), Gamma_K_achieved=_round(Gach))
            row["class_label"] = classify_s3(row, ref_row)
            row["dR"] = _round(row["R_area"] - ref_row["R_area"])
            row["dF"] = _round((row["F_off"] or 0) - (ref_row["F_off"] or 0))
            row["dS"] = _round((row["S_axis"] or 0) - (ref_row["S_axis"] or 0))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrates", nargs="+", default=["primary", "sensitivity"])
    ap.add_argument("--backup-control", action="store_true", default=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = [(s, dq, qm, sq, sd) for s in a.substrates for (dq, qm) in DQ_QMIN
             for sq in SIGMA_Q for sd in a.seeds]
    if a.backup_control:                          # backup = low-excitability negative control (deep point)
        tasks += [("backup", 0.30, 0.6, 1.5, sd) for sd in a.seeds]
    print(f"Step 3: {len(tasks)} (sub,dq,qmin,sigma_q,seed) tasks x {len(RATIOS)*(len(GAMMA_K)-1)} "
          f"g_K cells each (+ref) = {len(tasks)*(1 + len(RATIOS)*(len(GAMMA_K)-1))} SNN runs", flush=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, len(tasks))) as pool:
        raw = [r for rows in pool.map(worker, tasks) for r in rows]
    wall = time.time() - t0

    gk = [r for r in raw if r["Gamma_K"] > 0]
    cls = defaultdict(int)
    for r in gk:
        cls[r["class_label"]] += 1
    # rescue table: q-only-runaway references, did any g_K cell rescue them?
    refs = [r for r in raw if r["Gamma_K"] == 0]
    n_runaway_ref = sum(1 for r in refs if not r["returned"])
    rescued = [r for r in gk if r["class_label"] == "A_rescued_ictal_like"]
    payload = dict(meta=dict(date="2026-06-28", step="Step 3 q_I+g_K rescue scout",
                             Gamma_K=GAMMA_K, dq_qmin=DQ_QMIN, sigma_q=SIGMA_Q, ratios=RATIOS,
                             gK_max=GK_MAX, tau_K=TAU_K, n_runs=len(raw), wall_s=round(wall, 1),
                             n_qonly_runaway_refs=n_runaway_ref, gk_class_counts=dict(cls),
                             n_rescued=len(rescued)),
                   raw_rows=raw)
    json.dump(payload, open(os.path.join(a.out, "step3_results.json"), "w"), indent=2)

    print(f"\n{len(raw)} rows in {wall:.0f}s. q-only-runaway refs={n_runaway_ref}. "
          f"g_K class counts: {dict(cls)}  RESCUED={len(rescued)}", flush=True)
    print("\nReturned+off-axis (A) candidates (substrate dq qmin sigma_q ratio Gamma | R S F dF gKgap ref):")
    for r in sorted([r for r in gk if r["class_label"].startswith("A_")],
                    key=lambda r: -(r["dF"] or 0))[:12]:
        rf = "RUNAWAY" if r["class_label"] == "A_rescued_ictal_like" else "ret"
        print(f"  {r['substrate']:11} dq={r['dq_target']} qmin={r['q_min']} sq={r['sigma_q']} "
              f"r={r['ratio']} G={r['Gamma_K']} | R={r['R_area']} S={r['S_axis']} F={r['F_off']} "
              f"dF={r['dF']:+.3f} gKgap={r['gK_gap']:+.3f} (ref {rf})")
    if not rescued and not [r for r in gk if r["class_label"].startswith("A_")]:
        print("  (none) -- no controlled off-axis recruitment found; see gk_class_counts")
    print(f"\nwrote {os.path.join(a.out, 'step3_results.json')}", flush=True)


if __name__ == "__main__":
    main()
