"""FCXR-HYB1 — pure adjudication logic for the Z / activity-excess-K / X lifecycle sprint.

Plan of record: docs/superpowers/plans/2026-07-29-topic4-fcxr-hyb1.md

Nothing here runs a simulation.  Every threshold in this module is fixed by plan sections 2, 3 and
5 and must not be edited to make a result pass -- the q75 / q50 / q50-without-X bad-data
regressions in tests/test_topic4_fcxr_hyb1.py exist precisely to catch that.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ plan 2: activity-excess K
Q_BG = 0.99                      # per-voxel background upper-envelope quantile (plan 2.1)
EPS_FRAC = 0.10                  # deadband softness = EPS_FRAC * median_v(b_v)  (plan 2.2)
TAU_K_S = 0.6546                 # B2.1 measured tau_Ko at the working point     (plan 2.3)
D_K = 2.5e-4                     # B2.1 diffusion constant                        (plan 2.3)
Q_K = 0.013615797289152352       # B2.1 q_ion at the T7.1 f'=1.0 anchor           (plan 2.3)
G_DELTA_K = 1.0                  # B2.1 anchor; NOT swept this sprint             (plan 2.4)
ETA_PUMP = 0.0                   # locked: no Na in HYB1

# plan 2.5 baseline-preservation gate
BASE_DUTY_MAX = 3.0 * (1.0 - Q_BG)   # 0.03
BASE_DK_Q99_MAX_MM = 0.05

# ------------------------------------------------------------------ plan 3: the Z hazard axis
TAU_Z_DOWN_MS = 5000.0
TAU_Z_UP_MS = 20000.0
# --- anchors, CORRECTED 2026-07-31 -------------------------------------------------------------
# The first pre-registered attempt registered h_Z_obs as D_Z_end / T.  That is the 24 s AVERAGE of
# a curve that saturates: measured on the existing q75 seed-1 trace the slope is ~2.0e-2 /s over
# [0,4) s, 8.2e-3 by [6,8) s and ~0 by [8,10) s.  The axis quantity a_p/tau_z is the slope at t=0,
# so the two are different observables and the identifiability test compared them.  It failed, and
# the failed verdict is preserved at superseded/z_axis_calibration_avg_slope_anchors.json.
#
# The anchor OBSERVABLE is corrected to what the plan actually names -- the D_Z slope -- measured
# on the same stored traces over [0, 5] s.  Nothing else moves: the +-50% tolerance, the geometric
# spacing and the strictly-between rule are unchanged.  The correction is falsifiable and it is
# checked: with the right statistic the seed-1 probe re-predicts BOTH seed-1 anchors to 14% / 10%.
#
# The seed-3 anchor is NOT predictable from a seed-1 probe (different substrate -> different GABA
# survival curve) and is therefore recorded as provenance only, not as an identifiability check.
Z_ANCHORS = {
    "q75_seed1": dict(I_th_EI=95.19851312666987, tau_z_ms=5000.0, h_Z=2.1307e-2,
                      source="D_Z slope over [0,5]s, zonly_seed1_q75_T24000/zonly_traces.npz"),
    "q50_seed1": dict(I_th_EI=1.6652801609959704, tau_z_ms=10000.0, h_Z=3.4896e-2,
                      source="D_Z slope over [0,5]s, zonly_seed1_q50_T24000/zonly_traces.npz"),
}
Z_ANCHORS_PROVENANCE_ONLY = {
    "q75_seed3": dict(I_th_EI=95.19851312666987, tau_z_ms=5000.0, h_Z=1.3669e-2,
                      note="different substrate; a seed-1 survival curve cannot predict it"),
}
Z_ANCHORS_SUPERSEDED_AVG = {"q75_seed1": 6.9221e-3, "q75_seed3": 7.7734e-3, "q50_seed1": 3.3694e-2}
H_LO_HI = (Z_ANCHORS["q75_seed1"]["h_Z"], Z_ANCHORS["q50_seed1"]["h_Z"])
_R = (H_LO_HI[1] / H_LO_HI[0]) ** 0.25
H_TARGETS = {"H_LO": H_LO_HI[0] * _R, "H_MID": H_LO_HI[0] * _R ** 2, "H_HI": H_LO_HI[0] * _R ** 3}
Z_AXIS_ANCHOR_TOL = 0.50         # plan 3.5 identifiability: +-50% on both anchors

# ------------------------------------------------------------------ plan 5: the seven gates
PRE_MIN_MS = 8000.0
BOUT_MIN_MS, BOUT_MAX_MS = 1000.0, 5000.0
RECRUIT_MIN = 12                 # of 15 virtual contacts (HEO3 threshold, at-risk -- plan 5.1)
X_DELAY_MIN_MS = 100.0
X_ACTIVE_D_X = 0.02
POST_MIN_MS = 8000.0
IEI_CV_MIN = 0.5
RUNAWAY_RATE_HZ = 300.0          # scientific early-stop / Gate 7 ceiling
WALL_KILL_S = 3600.0


def deadband_positive(u, eps):
    """R_eps(u): exactly 0 at or below background, C1 at the origin, ~u-eps far above (plan 2.2).

    softplus is NOT usable here -- the plan requires STRICTLY zero below the registered background,
    and softplus leaks a positive source into every interictal voxel at every step.
    """
    u = np.asarray(u, float)
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return np.where(u > 0.0, u * u / (u + eps), 0.0)


def background_envelope(load_tv, q=Q_BG):
    """Per-voxel registered background upper envelope b_v from a sensor-only interictal run."""
    a = np.asarray(load_tv, float)
    if a.ndim != 2:
        raise ValueError("load_tv must be (n_time, n_voxel)")
    return np.quantile(a, q, axis=0)


def deadband_eps(b_v, frac=EPS_FRAC):
    return float(frac * np.median(np.asarray(b_v, float)))


def hazard_from_survival(sensor_I, p_weights, theta, tau_z_ms):
    """h_Z = a_p(theta)/tau_z, the EXACT t=0 slope of D_Z (plan 3.1), in s^-1."""
    I = np.asarray(sensor_I, float)
    p = np.asarray(p_weights, float)
    if I.shape != p.shape:
        raise ValueError(f"sensor and weights must match: {I.shape} vs {p.shape}")
    s = float(p.sum())
    if not (s > 0):
        raise ValueError("p_weights must have positive sum")
    a_p = float(np.sum(p * (I >= float(theta))) / s)
    return a_p / (float(tau_z_ms) / 1000.0), a_p


def invert_survival_for_theta(sensor_I, p_weights, a_target):
    """Smallest threshold whose p-weighted survival is <= a_target (the axis is monotone in theta)."""
    I = np.asarray(sensor_I, float)
    p = np.asarray(p_weights, float)
    o = np.argsort(I)
    Is, ps = I[o], p[o]
    surv = (ps[::-1].cumsum())[::-1] / float(p.sum())     # surv[k] = a_p(Is[k])
    k = int(np.searchsorted(-surv, -float(a_target), side="left"))
    return float(Is[min(k, Is.size - 1)])


def adjudicate_z_axis(sensor_I, p_weights):
    """plan 3.5: the ONE probe must re-predict both observed anchors and be monotone between them."""
    checks, pred = {}, {}
    for name, a in Z_ANCHORS.items():
        h, a_p = hazard_from_survival(sensor_I, p_weights, a["I_th_EI"], a["tau_z_ms"])
        pred[name] = dict(h_Z_pred=h, a_p=a_p, h_Z_obs=a["h_Z"],
                          rel_err=(h - a["h_Z"]) / a["h_Z"])
        checks[f"anchor_{name}"] = dict(ok=bool(abs(pred[name]["rel_err"]) <= Z_AXIS_ANCHOR_TOL),
                                        value=pred[name]["rel_err"], tol=Z_AXIS_ANCHOR_TOL)
    lo, hi = Z_ANCHORS["q50_seed1"]["I_th_EI"], Z_ANCHORS["q75_seed1"]["I_th_EI"]
    grid = np.linspace(lo, hi, 64)
    a_grid = [hazard_from_survival(sensor_I, p_weights, t, 1000.0)[1] for t in grid]
    checks["monotone"] = dict(ok=bool(np.all(np.diff(a_grid) <= 1e-12)),
                              rule="a_p must be non-increasing in the threshold")
    levels = {}
    for name, h in H_TARGETS.items():
        a_t = h * (TAU_Z_DOWN_MS / 1000.0)
        th = invert_survival_for_theta(sensor_I, p_weights, a_t)
        h_got, a_got = hazard_from_survival(sensor_I, p_weights, th, TAU_Z_DOWN_MS)
        levels[name] = dict(h_Z_target=h, a_p_target=a_t, I_th_EI=th, h_Z_realised=h_got,
                            a_p_realised=a_got)
    checks["levels_between_anchors"] = dict(
        ok=bool(all(H_LO_HI[0] < v["h_Z_realised"] < H_LO_HI[1] for v in levels.values())),
        rule="every realised hazard must lie strictly between the q75 and q50 observations")
    ok = all(c["ok"] for c in checks.values())
    return dict(status="Z_AXIS_LOCKED" if ok else "DESIGN_BLOCKED_Z_AXIS", checks=checks,
                anchor_prediction=pred, levels=levels,
                tau_z_down_ms=TAU_Z_DOWN_MS, tau_z_up_ms=TAU_Z_UP_MS)


def adjudicate_baseline_preservation(m):
    """plan 2.5.  Every clause is measured; 'preserved by construction' is not accepted as evidence."""
    c = dict(
        dk_duty=dict(ok=bool(m["dk_duty"] <= BASE_DUTY_MAX), value=m["dk_duty"],
                     threshold=BASE_DUTY_MAX),
        dk_amplitude=dict(
            # plan 2.5 writes q99_{t,v}(dK) <= 0.05 mM.  That is EXACTLY P_{t,v}(dK > 0.05) <= 0.01,
            # which is streamable; the time-q99 of the per-block spatial MAX is a different and
            # much harsher statistic and must not be substituted for it.
            ok=bool(m["dk_frac_over"] <= 1.0 - Q_BG), value=m["dk_frac_over"],
            threshold=1.0 - Q_BG, amplitude_mM=BASE_DK_Q99_MAX_MM,
            spatial_max_q99_mM=m.get("dk_spatial_max_q99_mM")),
        event_rate_in_band=dict(ok=bool(m["event_rate_in_band"])),
        iei_cv_in_band=dict(ok=bool(m["iei_cv_in_band"] and m["iei_cv"] >= IEI_CV_MIN),
                            value=m["iei_cv"], floor=IEI_CV_MIN),
        duration_in_band=dict(ok=bool(m["duration_in_band"])),
        participation_in_band=dict(ok=bool(m["participation_in_band"])),
        numerically_safe=dict(ok=bool(m["clip_frac_max"] == 0.0 and not m["numerical_unsafe"]),
                              clip=m["clip_frac_max"]),
    )
    ok = all(v["ok"] for v in c.values())
    return dict(status="BASELINE_PRESERVED" if ok else "STOP_BASELINE_DISTURBED", checks=c,
                rescue_forbidden="do not touch drive, connectivity, g_deltaK or Q_BG")


def _in_band(v, band):
    return bool(band is not None and band[0] <= v <= band[1])


def adjudicate_lifecycle(m, *, spatial_leg="UNRESOLVED"):
    """The seven gates of plan 5.  `m` is one run's reduced summary.

    `spatial_leg` is passed in rather than derived here: plan 5.1 pre-registers that if neither
    recruitment nor the onset gradient separates a structured event from the synchronous negative
    control, Gate 4 is UNRESOLVED -- and that separation is a property of the CONTROLS, not of the
    run being judged, so it cannot be decided from `m`.
    """
    if spatial_leg not in ("PASS", "FAIL", "UNRESOLVED"):
        raise ValueError("spatial_leg must be PASS, FAIL or UNRESOLVED")
    g = {}
    g["1_spontaneous"] = dict(ok=bool(m["kick_boost"] == 0.0 and m["t_kick_ms"] >= 1e8
                                      and m["onset_detected"]),
                              kick_boost=m["kick_boost"], onset=m["onset_detected"])
    g["2_pre_interictal"] = dict(ok=bool(m["pre_interictal_ms"] >= PRE_MIN_MS),
                                 value=m["pre_interictal_ms"], threshold=PRE_MIN_MS)
    bout = m.get("bout_ms")
    g["3_bounded_high_state"] = dict(
        ok=bool(bout is not None and BOUT_MIN_MS <= bout <= BOUT_MAX_MS and m["bounded"]
                and m["clip_frac_max"] == 0.0),
        bout_ms=bout, window=[BOUT_MIN_MS, BOUT_MAX_MS], bounded=m["bounded"])
    g["4_spatial"] = dict(ok=(spatial_leg == "PASS"), leg=spatial_leg,
                          recruit=m.get("recruit_contacts"), recruit_min=RECRUIT_MIN,
                          onset_gradient_r2=m.get("onset_gradient_r2"),
                          note="UNRESOLVED is not a pass and not a direction claim")
    dly = m.get("x_activation_delay_ms")
    g["5_x_after_onset"] = dict(ok=bool(dly is not None and dly >= X_DELAY_MIN_MS),
                                value=dly, threshold=X_DELAY_MIN_MS)
    g["6_statistical_recovery"] = dict(
        ok=bool(m.get("post_return_ms", 0.0) >= POST_MIN_MS
                and m.get("label") == "RECOVERED_INTERICTAL"
                and m.get("post_iei_cv", 0.0) >= IEI_CV_MIN
                and _in_band(m.get("post_event_rate_hz", -1.0), m.get("band_event_rate"))
                and _in_band(m.get("post_duration_ms", -1.0), m.get("band_duration"))
                and _in_band(m.get("post_participation", -1.0), m.get("band_participation"))
                and not m.get("post_silent", False)),
        post_return_ms=m.get("post_return_ms"), label=m.get("label"),
        post_iei_cv=m.get("post_iei_cv"))
    g["7_numerical"] = dict(
        ok=bool(m["clip_frac_max"] == 0.0 and m["finite"] and not m["numerical_unsafe"]
                and m["end_rate_hz"] < RUNAWAY_RATE_HZ),
        end_rate_hz=m["end_rate_hz"], ceiling=RUNAWAY_RATE_HZ)
    failed = [k for k, v in g.items() if not v["ok"]]
    hard = [k for k in failed if k != "4_spatial"]
    if not failed:
        status = "LIFECYCLE_CANDIDATE"
    elif not hard and spatial_leg == "UNRESOLVED":
        status = "LIFECYCLE_CANDIDATE_SPATIAL_UNRESOLVED"
    else:
        status = "NOT_A_CANDIDATE"
    return dict(status=status, gates=g, failed=failed,
                failure_layer=_failure_layer(failed),
                forbidden=["limit cycle", "bistability", "real patient ion mechanism",
                           "promotes propagation", "source/sink success wording"])


_LAYER = {"1_spontaneous": "onset", "2_pre_interictal": "onset",
          "3_bounded_high_state": "persistence", "4_spatial": "recruitment",
          "5_x_after_onset": "termination", "6_statistical_recovery": "recovery",
          "7_numerical": "persistence"}


def _failure_layer(failed):
    """Which of onset / persistence / recruitment / termination / recovery / waveform broke."""
    order = ["onset", "persistence", "recruitment", "termination", "recovery"]
    hit = {_LAYER[f] for f in failed}
    return [x for x in order if x in hit]


# ------------------------------------------------------------------ plan 5.1 spatial separation
def adjudicate_spatial_separation(structured, synchronous):
    """Can EITHER leg tell a structured event from the synchronous negative control?

    Pre-registered before any HYB1 run.  HEO2.1 already measured recruitment at >=13/15 in 48/48
    working points including the purely synchronous tonic state, so the recruitment leg is expected
    to be non-discriminative; the onset gradient is the remaining hope.  If neither separates, the
    lifecycle's spatial leg is UNRESOLVED for the whole sprint.
    """
    rec_sep = bool(min(structured["recruit"]) > max(synchronous["recruit"]))
    gr_sep = bool(min(structured["onset_gradient_r2"]) > max(synchronous["onset_gradient_r2"]))
    return dict(leg="PASS" if (rec_sep or gr_sep) else "UNRESOLVED",
                recruit_separates=rec_sep, onset_gradient_separates=gr_sep,
                structured=structured, synchronous=synchronous,
                note=("recruitment alone is known to be near-saturated across working points "
                      "(HEO2.1: 48/48 >=13/15), so a PASS driven by recruitment only would be "
                      "suspect; the onset gradient is the discriminative candidate"))
