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
B0_PASS_STATUS = "BASELINE_PRACTICALLY_INVISIBLE"   # one name, so a downstream gate cannot
                                                    # silently keep checking a retired one
B0_N_SEGMENTS = 4                 # equal time segments for the membrane-level rise check
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


def event_peak_values(q_tv, onsets_ms, occupied, tau_R_ms, dt_ms, calibration_end_ms):
    """q_v peak inside each CALIBRATION-half event window (plan 4.2 step 1, spec 207).

    `calibration_end_ms` is REQUIRED and has no default.  Q_on must be locked on the calibration
    half alone so that the validation half stays out-of-sample for the baseline false-activation
    check (spec 207 step 3).  An earlier revision of this function used every event in the record:
    that let seed1's Q_on be set 1.51x higher by a peak the validation half was supposed to TEST
    against (112.505 -> 169.846, above the validation maximum of 154.405), which turns "the
    actuator never fires interictally" into circular validation.  A default would let a caller
    restore that path silently, so there is none.

    ONE function with TWO call sites (calibration pass 2 and finalize); the loop used to be
    duplicated and both copies had to be corrected in lockstep.
    """
    q = np.asarray(q_tv, float)
    occ = np.asarray(occupied, bool)
    span = int(round(3.0 * float(tau_R_ms) / float(dt_ms))) + 1
    out = []
    for t_on in np.asarray(onsets_ms, float):
        if t_on >= float(calibration_end_ms):
            continue
        a0 = int(t_on / float(dt_ms))
        a1 = min(q.shape[0], a0 + span)
        if a1 > a0:
            out.append(float(q[a0:a1][:, occ].max()))
    return out


def revt_activation_profile(n_active_per_block, n_occupied, n_segments=4):
    """R_evt occupancy per equal time segment -- the MEMBRANE-level reading of "is the layer
    creeping up", which is what Gate B0 is entitled to gate on.

    q_v is a hidden sensor: nothing about q reaches a membrane until it crosses Q_on and becomes
    R_evt.  HYB1's ratchet was a delta-K floor that entered the membrane directly, so transplanting
    that gate onto a hidden q floor is not a level-preserving translation.  A rising R_evt floor
    WOULD raise the later segments' occupancy, so the same 1% bound applies per segment and no new
    threshold is introduced.
    """
    a = np.asarray(n_active_per_block, float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("n_active_per_block must be a non-empty 1-D array")
    if not (n_occupied > 0):
        raise ValueError("n_occupied must be > 0")
    seg = np.array_split(a, int(n_segments))
    return [float(x.sum() / (x.size * float(n_occupied))) for x in seg]


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
    """Baseline invisibility, gated at the MEMBRANE level.

    Gating clauses: R_evt occupancy, R_evt occupancy per segment (no sustained rise), on/off
    interictal event statistics, numerical safety.

    DIAGNOSTIC, not gating: the two q_v measures.  Both were demoted after the first run, on two
    independent grounds.  (a) Level: q_v never reaches a membrane; only R_evt does (see
    revt_activation_profile).  (b) The pre-onset residual is a broken instrument -- its derivation
    assumes e_v == 0 through the gap, but b_v := Q99_t[s_v] puts 1% of blocks above background in
    every voxel by construction, and the 2 ms window it samples sits inside the NEXT event's local
    build-up.  Measured gap-resolved (figures/b0_gap_resolved_envelope_seed1.png): the envelope
    does clear, to 0.0029-0.0051 of Q_on 30-75 ms before onset, while the contract window reads
    0.161.  A demotion decided after seeing a failure needs its reason to be structural, and both
    of these are; they are still reported, on every run, so the decision stays auditable.
    """
    seg = list(m.get("revt_occupancy_by_segment") or [])
    c = dict(
        active_occupancy=dict(ok=bool(m["active_occupancy"] <= B0_ACTIVE_OCCUPANCY_MAX),
                              value=m["active_occupancy"], threshold=B0_ACTIVE_OCCUPANCY_MAX,
                              gating=True),
        revt_no_sustained_rise=dict(
            ok=bool(seg and max(seg) <= B0_ACTIVE_OCCUPANCY_MAX),
            by_segment=seg, threshold=B0_ACTIVE_OCCUPANCY_MAX, gating=True,
            rule="R_evt occupancy in EVERY equal segment, same 1% bound; a creeping floor fails it"),
        event_stats_in_band=dict(ok=bool(m["event_stats_in_band"]), gating=True,
                                 detail=m.get("event_stats_detail")),
        numerically_safe=dict(ok=bool(m["clip_frac_max"] == 0.0 and not m["numerical_unsafe"]),
                              clip=m["clip_frac_max"], gating=True),
    )
    diag = dict(
        q_pre_onset_residual=dict(value=m["pre_onset_residual_frac"], reference=B0_RESIDUAL_FRAC,
                                  gating=False, window_ms=B0_PRE_ONSET_WINDOW_MS,
                                  note="broken instrument: samples the next event's build-up"),
        q_floor_drift=dict(value=m["q_floor_drift"], reference=B0_DRIFT_MAX, gating=False,
                           note="hidden-sensor level; the membrane-level reading is "
                                "revt_no_sustained_rise"),
    )
    ok = all(v["ok"] for v in c.values())
    return dict(
        status=B0_PASS_STATUS if ok else "STOP_ELR_BASELINE_VISIBLE",
        checks=c, diagnostics=diag,
        allowed_wording=("under the PRE-REGISTERED calibration-half Q_on, seed1 showed a very rare "
                         "validation-half activation and seed3 showed none; on neither seed was any "
                         "disturbance of the interictal event statistics observed, over 24 s and "
                         "two connectivity seeds"),
        forbidden_wording=("the actuator never fired at all / baseline invisibility has been proven "
                           "bit-exactly / the event-scale actuator has been shown not to disturb "
                           "the baseline"),
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
