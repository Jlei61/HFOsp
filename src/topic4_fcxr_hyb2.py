"""FCXR-HYB2 — pure adjudication logic.  Nothing here runs a simulation.

Plan of record: docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md

Every threshold is fixed by the plan and must not be edited to make a result pass.  The HYB1
seven gates and their q75 / q50 / q50-without-X bad-data regressions are reused verbatim from
`src.topic4_fcxr_hyb1`; this module adds only what HYB2 introduces.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ plan 2: locked ELR constants
DT_R_MS = 0.5
Q_BG = 0.99                      # per-voxel background upper-envelope quantile
EPS_S_FRAC = 0.10                # eps_s = 0.10 * median_v(b_v)
EPS_Q_FRAC = 0.10                # eps_q = 0.10 * Q_on
Q_ON_MARGIN = 1.10               # Q_on = 1.10 * max event peak on the calibration half
I_R_MAX = 4.134151260609386      # B2.1 force anchor (dE_K at 0.6715 mM, g=1), engine-drive units
CAL_SPLIT_FRAC = 0.5             # first half = calibration, second half = validation

# ------------------------------------------------------------------ plan 3: event timescale
T_EVENT_GUARD_MS = 22.0          # two-seed MAXIMUM event duration, LC1 24 s frozen bar
RESIDUAL_TARGET = 0.01           # exp(-GAP_05/tau_R) must not exceed this

# ------------------------------------------------------------------ plan 7.1: Gate B0
B0_ACTIVE_OCCUPANCY_MAX = 0.01
B0_RESIDUAL_FRAC = 0.01          # q_v in the 2 ms pre-onset window, as a fraction of Q_on
B0_PRE_ONSET_WINDOW_MS = 2.0
B0_DRIFT_MAX = 0.01              # (floor_last - floor_first) / Q_on

# ------------------------------------------------------------------ plan 7.2 / 5.3: Gate A0
A0_WINDOW_MS = 1000.0
A0_MIN_IMPROVEMENT = 0.10        # >= +10% on at least two of three extent measures
A0_MIN_MEASURES = 2
A0_CEILING_FRAC = 0.90           # off arm at >= 90% of cells / occupied voxels = ceiling-confounded
A0_RUNAWAY_HZ = 300.0

# ------------------------------------------------------------------ plan 6: S_Z response axis
TAU_Z_DOWN_MS = 5000.0
TAU_Z_UP_MS = 20000.0
T_CAL_MS_MIN = 3.0 * TAU_Z_DOWN_MS       # 15 s; below this S_Z degenerates to h_Z
T_CAL_MS = 24000.0
I_TH_Q75 = 95.19851312666987
I_TH_Q50 = 1.6652801609959704
S_Z_FRACTIONS = (0.25, 0.50, 0.75)


# ================================================================== plan 2: calibration
def background_envelope(load_tv, occupied, q=Q_BG):
    """b_v = Q99_t[s_v] over OCCUPIED voxels only; unoccupied get +inf so they never source."""
    a = np.asarray(load_tv, float)
    if a.ndim != 2:
        raise ValueError("load_tv must be (n_time, n_voxel)")
    occ = np.asarray(occupied, bool)
    b = np.full(a.shape[1], np.inf)
    b[occ] = np.quantile(a[:, occ], float(q), axis=0)
    return b


def eps_s_from_background(b_v):
    finite = np.asarray(b_v, float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        raise ValueError("no occupied voxel has a positive background")
    return float(EPS_S_FRAC * np.median(finite))


def event_gaps(onsets_ms, offsets_ms):
    """GAP_k = t_on[k+1] - t_off[k].  Not onset-to-onset: q_v is still driven until the event ends,
    so only the true inter-event silence is available to decay."""
    on = np.asarray(onsets_ms, float)
    off = np.asarray(offsets_ms, float)
    if on.shape != off.shape:
        raise ValueError(f"onsets and offsets must match: {on.shape} vs {off.shape}")
    if on.size < 2:
        return np.zeros(0)
    o = np.argsort(on)
    on, off = on[o], off[o]
    if np.any(off < on):
        raise ValueError("every offset must be at or after its own onset")
    g = on[1:] - off[:-1]
    if np.any(g < 0):
        raise ValueError("overlapping events: an onset precedes the previous offset")
    return g


def tau_R_from_timescale(T_event_guard_ms, gap_05_ms):
    """Geometric midpoint of [T_event_guard, GAP_05/ln(100)].  Pre-registered, never swept."""
    hi = float(gap_05_ms) / np.log(1.0 / RESIDUAL_TARGET)
    lo = float(T_event_guard_ms)
    return dict(tau_R_ms=float(np.sqrt(lo * hi)) if lo < hi else None,
                interval=[lo, float(hi)], feasible=bool(lo < hi),
                headroom_ms=float(hi - lo))


def residual(gap_ms, tau_R_ms):
    return float(np.exp(-float(gap_ms) / float(tau_R_ms)))


def event_peak_values(q_tv, onsets_ms, occupied, tau_R_ms, dt_ms):
    """q_v peak inside each event window, over ALL events and occupied voxels (plan 4.2).

    ONE function with TWO call sites (calibration pass 2 and finalize) because the first
    implementation duplicated this loop and both copies carried the same defect: they restricted
    the peak search to `onsets < CAL_SPLIT_FRAC * T`.  CAL_SPLIT_FRAC governs `b_v` (plan 4.1
    estimates the background envelope on the calibration half); plan 4.2 says pass 2 replays
    "the same record" and takes `max_{event, occupied voxel}`, with no half-restriction.  The
    deviation made Q_on 1.51x too small on seed1, which in turn broke the by-construction argument
    that plan 7.1 uses to declare clauses 1 and 2 non-independent.
    """
    q = np.asarray(q_tv, float)
    occ = np.asarray(occupied, bool)
    span = int(round(3.0 * float(tau_R_ms) / float(dt_ms))) + 1
    out = []
    for t_on in np.asarray(onsets_ms, float):
        a0 = int(t_on / float(dt_ms))
        a1 = min(q.shape[0], a0 + span)
        if a1 > a0:
            out.append(float(q[a0:a1][:, occ].max()))
    return out


def b0_envelope_statistics(pre_windows, Q_on):
    """plan 7.1 clauses 2 and 3, on the JOINT (event x block x occupied voxel) pool.

    `pre_windows` is one 2-D array (n_block, n_occupied_voxel) per event, covering the 2 ms
    immediately before that event's onset.

    The contract says "q99 across events x occupied voxels".  The first implementation reduced each
    block to `max_v q_v` first and took the q99 of THAT -- a maximum over ~1000 voxels is not a 99th
    percentile over them, and it ran 2x high on seed1.  Same error class as the B2.1 amplitude
    clause (time-q99 of a spatial max instead of the joint tail); the fix is the same.

    Clause 3 aggregates the SAME per-event statistic so the two clauses cannot disagree about what
    "the floor" means.
    """
    Q = float(Q_on)
    if not (Q > 0):
        raise ValueError("Q_on must be > 0")
    per_event = [np.asarray(w, float).ravel() for w in pre_windows if np.asarray(w).size]
    if len(per_event) < 4:
        return dict(pre_onset_residual_frac=float("nan"), q_floor_drift=float("nan"),
                    n_pre_onset=len(per_event), insufficient=True)
    joint = np.concatenate(per_event)
    floors = np.asarray([float(np.quantile(e, 0.99)) for e in per_event])
    k = max(2, floors.size // 4)
    return dict(pre_onset_residual_frac=float(np.quantile(joint, 0.99) / Q),
                q_floor_drift=float((floors[-k:].mean() - floors[:k].mean()) / Q),
                floor_first=float(floors[:k].mean()), floor_last=float(floors[-k:].mean()),
                n_pre_onset=int(floors.size), segment_k=int(k),
                n_joint_samples=int(joint.size), insufficient=False)


def q_on_from_event_peaks(peaks):
    p = np.asarray(peaks, float)
    if p.size == 0:
        raise ValueError("no calibration event peaks")
    return float(Q_ON_MARGIN * p.max())


def adjudicate_calibration(m):
    """plan 8 stage 2.  Two distinct stops: an empty timescale interval, or degenerate thresholds."""
    tr = tau_R_from_timescale(m["T_event_guard_ms"], m["gap_05_ms"])
    if not tr["feasible"]:
        return dict(status="DESIGN_BLOCKED_EVENT_TIMESCALE", tau=tr,
                    reason=("T_event_guard >= GAP_05/ln(100): no memory can hold a whole event "
                            "AND clear between events"))
    q_on, q_scale = m["Q_on"], m["Q_scale"]
    if not (q_on > 0 and q_scale > 0):
        return dict(status="CALIBRATION_INVALID", tau=tr, Q_on=q_on, Q_scale=q_scale)
    t = tr["tau_R_ms"]
    return dict(status="CALIBRATION_LOCKED", tau=tr, tau_R_ms=t, Q_on=q_on, Q_scale=q_scale,
                eps_q=EPS_Q_FRAC * q_on, I_R_max=I_R_MAX,
                residual_tail={k: residual(m[k], t) for k in
                               ("gap_05_ms", "gap_01_ms", "gap_min_ms") if k in m},
                note=("the 1% rule covers 95% of gaps; the GAP_01 and shortest-gap residuals are "
                      "reported so the short tail stays visible"))


# ================================================================== plan 7.1: Gate B0
def adjudicate_gate_B0(m):
    """Baseline invisibility.  Clauses 1 and 2 pass BY CONSTRUCTION -- see `construction_note`."""
    c = dict(
        active_occupancy=dict(ok=bool(m["active_occupancy"] <= B0_ACTIVE_OCCUPANCY_MAX),
                              value=m["active_occupancy"], threshold=B0_ACTIVE_OCCUPANCY_MAX,
                              independent=False),
        pre_onset_residual=dict(ok=bool(m["pre_onset_residual_frac"] <= B0_RESIDUAL_FRAC),
                                value=m["pre_onset_residual_frac"], threshold=B0_RESIDUAL_FRAC,
                                window_ms=B0_PRE_ONSET_WINDOW_MS, independent=False),
        q_floor_drift=dict(ok=bool(m["q_floor_drift"] <= B0_DRIFT_MAX), value=m["q_floor_drift"],
                           threshold=B0_DRIFT_MAX, independent=True,
                           rule="(floor_last - floor_first) / Q_on, a DIFFERENCE not a ratio"),
        event_stats_in_band=dict(ok=bool(m["event_stats_in_band"]), independent=True,
                                 detail=m.get("event_stats_detail")),
        numerically_safe=dict(ok=bool(m["clip_frac_max"] == 0.0 and not m["numerical_unsafe"]),
                              clip=m["clip_frac_max"], independent=True),
    )
    ok = all(v["ok"] for v in c.values())
    return dict(
        status="BASELINE_INVISIBLE" if ok else "STOP_ELR_BASELINE_VISIBLE", checks=c,
        construction_note=(
            "Q_on = 1.10 * (max interictal event peak), so clause 1 (R_evt occupancy) and "
            "clause 2 (pre-onset residual) BOTH hold by construction: clause 2 written out is "
            "exp(-GAP/tau_R) <= 0.011, which IS the tau_R selection rule. They can only fail on "
            "gaps shorter than GAP_05. The independent criteria are the q_v ratchet (clause 3, "
            "exactly where HYB1 failed) and the event statistics (clause 4)."),
        allowed_wording=("no baseline disturbance was OBSERVED on the validation half and the "
                         "second seed"),
        forbidden_wording=("the event-scale actuator has been shown not to disturb the baseline"),
        rescue_forbidden="do not touch drive, connectivity, Q_on, tau_R or I_R_max")


# ================================================================== plan 7.2 / 5.3: Gate A0
def adjudicate_gate_A0(m):
    """Three-way, so 'the actuator is ineffective' cannot be confused with 'the input was'."""
    if not m["crossed_Q_on"] or m["ms_after_t_gate"] < A0_WINDOW_MS:
        return dict(status="A0_INPUT_INSUFFICIENT", crossed=m["crossed_Q_on"],
                    ms_after_t_gate=m["ms_after_t_gate"], window_ms=A0_WINDOW_MS,
                    note="the input never reached the actuator's threshold, or left too little "
                         "window; this says nothing about the actuator")
    off = m["off"]
    ceiling = dict(
        participants=bool(off["window_participants"] >= A0_CEILING_FRAC * m["n_E"]),
        voxels=bool(off["participant_voxels"] >= A0_CEILING_FRAC * m["n_occupied_voxels"]),
        runaway=bool(off.get("end_rate_hz", 0.0) >= A0_RUNAWAY_HZ or off.get("early_stopped", False)))
    if any(ceiling.values()):
        return dict(status="A0_CEILING_CONFOUNDED", ceiling=ceiling, frac=A0_CEILING_FRAC,
                    note="the off arm is already near global recruitment, so there is no headroom "
                         "for a >=10% improvement; this says nothing about the actuator")
    on = m["on"]
    keys = ("window_participants", "recruitment_radius_mm", "participant_voxels")
    rel = {k: (on[k] - off[k]) / off[k] for k in keys}
    n_up = sum(1 for k in keys if rel[k] >= A0_MIN_IMPROVEMENT)
    bounded = bool(m["max_R_evt"] <= I_R_MAX and m["clip_frac_max"] == 0.0 and m["finite"])
    ok = bool(n_up >= A0_MIN_MEASURES and bounded)
    return dict(status="A0_RECRUITMENT_EFFECTIVE" if ok else "NO_GO_EVENT_LIMITED_ACTUATOR",
                eligible=True, relative=rel, n_measures_up=n_up, required=A0_MIN_MEASURES,
                threshold=A0_MIN_IMPROVEMENT, bounded=bounded,
                allowed_wording=("the event-scale bounded actuator RETAINED the B2.1 recruitment "
                                 "EXTENT effect" if ok else
                                 "THIS short-memory, diffusion-free, thresholded ELR did not "
                                 "retain the B2.1 recruitment extent effect"),
                forbidden_wording=("B2.1's spatial recruitment depends on cross-event "
                                   "concentration memory -- HYB2 changed memory, diffusion AND "
                                   "the deadband/threshold/saturation at once, so one failing arm "
                                   "cannot attribute the loss to any single one"))


# ================================================================== plan 6: S_Z response axis
def c_analytic(T_cal_ms, tau_ms):
    """S_Z = a_p * C(T_cal, tau) when a cell's above/below-threshold status is constant.

    C = 1 - (tau/T)(1 - e^{-T/tau}).  Near-linear for T << tau, which is why T_cal must be at
    least 3*tau_Z_down: below that S_Z is proportional to h_Z and the new axis adds nothing.
    """
    T, tau = float(T_cal_ms), float(tau_ms)
    return 1.0 - (tau / T) * (1.0 - np.exp(-T / tau))


def s_z_response(sensor_tv, p_weights, theta, *, dt_ms, tau_down_ms=TAU_Z_DOWN_MS,
                 tau_up_ms=TAU_Z_UP_MS):
    """Open-loop cumulative-depletion coordinate: replay z on a FROZEN slow-off load trace.

    A frozen replay cannot express self-limitation (that is the closed loop z -> I_I -> z).  What
    S_Z adds over the t=0 hazard is the cells that CROSS threshold during the window, i.e. it
    measures time-occupancy above threshold rather than an instantaneous fraction.  It is a
    parameter coordinate for spacing three Z points; it predicts nothing about closed-loop
    branching.
    """
    S = np.asarray(sensor_tv, float)
    if S.ndim != 2:
        raise ValueError("sensor_tv must be (n_time, n_cell)")
    p = np.asarray(p_weights, float)
    if p.shape != (S.shape[1],):
        raise ValueError(f"p_weights must be ({S.shape[1]},), got {p.shape}")
    if float(p.sum()) <= 0:
        raise ValueError("p_weights must have positive sum")
    T_cal = S.shape[0] * float(dt_ms)
    if T_cal < T_CAL_MS_MIN:
        raise ValueError(f"T_cal {T_cal} ms < {T_CAL_MS_MIN} ms: S_Z would degenerate to h_Z "
                         f"(C = {c_analytic(T_cal, tau_down_ms):.3f}, still near-linear)")
    z = np.ones(S.shape[1])
    acc = 0.0
    for t in range(S.shape[0]):
        z_inf = (S[t] < float(theta)).astype(float)
        tau = np.where(z_inf < z, tau_down_ms, tau_up_ms)
        z = np.clip(z + (float(dt_ms) / tau) * (z_inf - z), 0.0, 1.0)
        acc += float(np.sum(p * (1.0 - z)) / p.sum())
    return acc / S.shape[0]


def adjudicate_z_response_axis(sensor_tv, p_weights, *, dt_ms, n_grid=48):
    """plan 6.3: monotone in the threshold, then invert S_Z at 25 / 50 / 75%."""
    def S(th):
        return s_z_response(sensor_tv, p_weights, th, dt_ms=dt_ms)
    grid = np.linspace(I_TH_Q50, I_TH_Q75, int(n_grid))
    vals = np.array([S(t) for t in grid])
    monotone = bool(np.all(np.diff(vals) <= 1e-12))
    s_hi, s_lo = float(vals[0]), float(vals[-1])          # q50 strong -> q75 weak
    if not monotone or not (s_lo < s_hi):
        return dict(status="DESIGN_BLOCKED_Z_RESPONSE_AXIS", monotone=monotone,
                    S_Z_q50=s_hi, S_Z_q75=s_lo,
                    reason="S_Z must be strictly decreasing in I_th between the two anchors")
    levels = {}
    for f in S_Z_FRACTIONS:
        target = s_lo + f * (s_hi - s_lo)
        k = int(np.argmin(np.abs(vals - target)))
        levels[f"S{int(f * 100)}"] = dict(fraction=f, S_Z_target=float(target),
                                          I_th_EI=float(grid[k]), S_Z_realised=float(vals[k]))
    return dict(status="Z_RESPONSE_AXIS_LOCKED", monotone=True, S_Z_q75=s_lo, S_Z_q50=s_hi,
                levels=levels, T_cal_ms=sensor_tv.shape[0] * float(dt_ms),
                tau_z_down_ms=TAU_Z_DOWN_MS, tau_z_up_ms=TAU_Z_UP_MS,
                scope=("open-loop cumulative depletion under a frozen load; a parameter "
                       "coordinate only. It does NOT measure self-limitation and does not "
                       "predict whether the three points branch differently in closed loop."))
