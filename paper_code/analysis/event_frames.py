"""Unchanged frame selection from plot_topic5_interictal_event_envelope_field.py."""
import numpy as np
from scipy.stats import spearmanr
from analysis import envelope_normalization as ief
FRAME_AVG_MS,N_FRAMES=3.,4
FRAME_PRE_MS,FRAME_MARGIN_MS,FRAME_MIN_POST_MS,FRAME_MAX_POST_MS=8.,4.,35.,50.
STATIC_FRAME_GRID_STEP_MS=2.
STATIC_FRAME_MIN_GAP_MS=8.
STATIC_FRAME_MIN_JOINT_VISIBILITY=.24
STATIC_FRAME_MIN_AXIS_SPAN_MM=12.
STATIC_FRAME_MIN_STATE_DISTANCE=.09
STATIC_FRAME_MIN_CENTROID_RHO=.70
STATIC_FRAME_MIN_ENDPOINT_HANDOFF=.10
STATIC_FRAME_MIN_FULL_VISIBILITY=.30
STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM=2.
STATIC_FRAME_MIN_HOTSPOT_STEP_MM=4.

def _frame_window(ea, eb):
    """Tight frame window around the displayed centroids, without an empty late tail."""
    return ief.display_window_ms(
        [e["centroid_ms"][e["usable"]] for e in (ea, eb)],
        pre_ms=FRAME_PRE_MS,
        margin_ms=FRAME_MARGIN_MS,
        min_post_ms=FRAME_MIN_POST_MS,
        max_post_ms=FRAME_MAX_POST_MS,
    )

def _static_frame_times(t_lo, t_hi, n_frames=N_FRAMES):
    """Use a sparse uniform grid; the readout cursor, not a redundant frame, marks t=0."""
    t_lo, t_hi = float(t_lo), float(t_hi)
    if int(n_frames) < 3:
        raise ValueError("static frame grid requires at least three frames")
    times = ief.frame_times_ms(t_lo, t_hi, int(n_frames))
    if len(times) != int(n_frames) or len(np.unique(np.round(times, 1))) != len(times):
        raise ValueError(f"invalid static frame times: {np.round(times, 1)}")
    return np.asarray(times, float)

def _display_vmax_events(events, t_lo, t_hi):
    """One scale for any candidate set, computed from complete intervals, not sampled frames."""
    win = [
        np.clip(
            e["env_z"][e["part"]][
                :, (e["t"] >= e["t0"] + t_lo / ief.MS)
                & (e["t"] <= e["t0"] + t_hi / ief.MS)
            ],
            0.0,
            None,
        )
        for e in events
    ]
    return ief.pooled_vmax(win, q=99.0)

def _event_normalization_scales(ea, eb, t_lo, t_hi, override=None):
    """Return one robust-z denominator per event; the displayed field is clipped to 0..1."""
    if override is None:
        scales = {
            "TA": _display_vmax_events((ea,), t_lo, t_hi),
            "TB": _display_vmax_events((eb,), t_lo, t_hi),
        }
    else:
        scales = {lab: float(override[lab]) for lab in ("TA", "TB")}
    if any(not np.isfinite(value) or value <= 0 for value in scales.values()):
        raise ValueError("normalization scales must be positive and finite")
    return scales

def _normalized_axial_frame_states(event, fz, frame_times, scale):
    """Return visibility, centroid and axial contact state for each candidate frame."""
    part = np.asarray(event["part"], bool)
    axial = part & np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    if int(axial.sum()) < 3:
        raise ValueError("joint-visible frame selection requires >=3 axial participants")
    x = np.asarray(fz["ax_mm"], float)[axial]
    visibility, centroids, states = [], [], []
    for frame_ms in np.asarray(frame_times, float):
        raw = np.clip(
            ief.values_at(
                event["env_z"], event["t"], event["t0"], frame_ms,
                avg_ms=FRAME_AVG_MS,
            ),
            0.0, None,
        )
        values = np.clip(raw / float(scale), 0.0, 1.0)[axial]
        states.append(values)
        top_k = min(3, len(values))
        visibility.append(float(np.mean(np.sort(values)[-top_k:])))
        weight = float(np.sum(values))
        centroids.append(float(np.sum(x * values) / weight) if weight > 0 else float("nan"))
    return (
        np.asarray(visibility, float),
        np.asarray(centroids, float),
        np.asarray(states, float),
    )

def _normalized_full_frame_states(event, fz, frame_times, scale):
    """Return full-participant visibility, centroid and top-3 hotspot along the shared axis.

    These quantities are computed from the exact contact values and frozen coordinates consumed
    by the final 2-D field.  They deliberately include every participating shaft, closing the old
    mismatch where an ICL-only selector could approve frames whose rendered all-shaft hotspot did
    not move.
    """
    part = np.asarray(event["part"], bool)
    if int(part.sum()) < 3:
        raise ValueError("full-field frame selection requires >=3 participants")
    x = np.asarray(fz["ax_mm"], float)[part]
    visibility, centroids, hotspots, states = [], [], [], []
    for frame_ms in np.asarray(frame_times, float):
        raw = np.clip(
            ief.values_at(
                event["env_z"], event["t"], event["t0"], frame_ms,
                avg_ms=FRAME_AVG_MS,
            ),
            0.0, None,
        )
        values = np.clip(raw / float(scale), 0.0, 1.0)[part]
        states.append(values)
        top_k = min(3, len(values))
        top = np.argsort(values)[-top_k:]
        top_values = values[top]
        visibility.append(float(np.mean(top_values)))
        weight = float(np.sum(values))
        top_weight = float(np.sum(top_values))
        centroids.append(float(np.sum(x * values) / weight) if weight > 0 else float("nan"))
        hotspots.append(
            float(np.sum(x[top] * top_values) / top_weight)
            if top_weight > 0 else float("nan")
        )
    return (
        np.asarray(visibility, float),
        np.asarray(centroids, float),
        np.asarray(hotspots, float),
        np.asarray(states, float),
    )

def _axial_endpoint_contrast(event, fz, states):
    """Return right-minus-left energy on the fixed outer thirds of the axial shaft.

    This contact-level readability metric does not inspect rendered field pixels.  Unlike a
    centroid, it only changes when energy is actually handed from one shaft end to the other, so
    a diffuse energy plateau cannot pass as a distinct propagation frame.
    """
    part = np.asarray(event["part"], bool)
    shaft = np.asarray(fz["shafts"] == fz["axial_shaft"], bool)
    axial = part & shaft
    x_all = np.asarray(fz["ax_mm"], float)[shaft]
    x = np.asarray(fz["ax_mm"], float)[axial]
    states = np.asarray(states, float)
    if states.ndim != 2 or states.shape[1] != len(x):
        raise ValueError("axial endpoint contrast received a misaligned contact-state matrix")
    q1, q2 = np.quantile(x_all, [1.0 / 3.0, 2.0 / 3.0])
    left, right = x <= q1, x >= q2
    if not left.any() or not right.any():
        raise ValueError("axial endpoint contrast requires participants at both shaft ends")
    return np.mean(states[:, right], axis=1) - np.mean(states[:, left], axis=1)

def _select_joint_visible_frame_times(ea, eb, fz, t_lo, t_hi, scales, n_frames=N_FRAMES):
    """Pick equally spaced frames with opposite motion in the final all-shaft field inputs.

    Selection uses normalized contact envelopes only.  The strict path searches arithmetic time
    sequences on the biological 2-ms grid.  It retains the axial direction gates, then additionally
    requires the full-participant centroid and top-3 hotspot to move at every step in the direction
    that the final 2-D renderer is expected to show.  A generic uniform fallback keeps other
    subjects renderable.
    """
    n_frames = int(n_frames)
    grid = np.arange(
        float(t_lo), float(t_hi) + 0.25 * STATIC_FRAME_GRID_STEP_MS,
        STATIC_FRAME_GRID_STEP_MS,
    )
    if grid.size < n_frames:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {"selection_mode": "uniform_fallback_grid_too_short"}
    try:
        vis_a, cen_a, state_a = _normalized_axial_frame_states(
            ea, fz, grid, scales["TA"],
        )
        vis_b, cen_b, state_b = _normalized_axial_frame_states(
            eb, fz, grid, scales["TB"],
        )
        endpoint_a = _axial_endpoint_contrast(ea, fz, state_a)
        endpoint_b = _axial_endpoint_contrast(eb, fz, state_b)
        full_vis_a, full_cen_a, hot_a, _ = _normalized_full_frame_states(
            ea, fz, grid, scales["TA"],
        )
        full_vis_b, full_cen_b, hot_b, _ = _normalized_full_frame_states(
            eb, fz, grid, scales["TB"],
        )
    except ValueError:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {"selection_mode": "uniform_fallback_axial_geometry"}
    joint = np.minimum(vis_a, vis_b)
    full_joint = np.minimum(full_vis_a, full_vis_b)
    feasible = []
    late_floor = float(t_hi) - STATIC_FRAME_MIN_GAP_MS
    min_step_idx = int(np.ceil(STATIC_FRAME_MIN_GAP_MS / STATIC_FRAME_GRID_STEP_MS))
    for step_idx in range(min_step_idx, len(grid)):
        for start_idx in range(len(grid)):
            idx = start_idx + np.arange(n_frames, dtype=int) * step_idx
            if idx[-1] >= len(grid):
                break
            times = grid[idx]
            if times[0] > STATIC_FRAME_GRID_STEP_MS or times[-1] < late_floor:
                continue
            if (
                not np.all(np.isfinite(cen_a[idx]))
                or not np.all(np.isfinite(cen_b[idx]))
                or not np.all(np.isfinite(full_cen_a[idx]))
                or not np.all(np.isfinite(full_cen_b[idx]))
                or not np.all(np.isfinite(hot_a[idx]))
                or not np.all(np.isfinite(hot_b[idx]))
            ):
                continue
            if float(np.min(joint[idx])) < STATIC_FRAME_MIN_JOINT_VISIBILITY:
                continue
            rho_a = float(spearmanr(times, cen_a[idx]).correlation)
            rho_b = float(spearmanr(times, cen_b[idx]).correlation)
            if (
                not np.isfinite(rho_a)
                or not np.isfinite(rho_b)
                or rho_a < STATIC_FRAME_MIN_CENTROID_RHO
                or rho_b > -STATIC_FRAME_MIN_CENTROID_RHO
            ):
                continue
            span_a = float(cen_a[idx][-1] - cen_a[idx][0])
            span_b = float(cen_b[idx][0] - cen_b[idx][-1])
            if span_a < STATIC_FRAME_MIN_AXIS_SPAN_MM or span_b < STATIC_FRAME_MIN_AXIS_SPAN_MM:
                continue
            state_step_a = np.sqrt(np.mean(np.diff(state_a[idx], axis=0) ** 2, axis=1))
            state_step_b = np.sqrt(np.mean(np.diff(state_b[idx], axis=0) ** 2, axis=1))
            joint_state_step = np.minimum(state_step_a, state_step_b)
            if float(np.min(joint_state_step)) < STATIC_FRAME_MIN_STATE_DISTANCE:
                continue
            handoff_a = np.diff(endpoint_a[idx])
            handoff_b = -np.diff(endpoint_b[idx])
            joint_handoff = np.minimum(handoff_a, handoff_b)
            if float(np.min(joint_handoff)) < STATIC_FRAME_MIN_ENDPOINT_HANDOFF:
                continue
            full_centroid_step_a = np.diff(full_cen_a[idx])
            full_centroid_step_b = -np.diff(full_cen_b[idx])
            joint_full_centroid_step = np.minimum(
                full_centroid_step_a, full_centroid_step_b,
            )
            hotspot_step_a = np.diff(hot_a[idx])
            hotspot_step_b = -np.diff(hot_b[idx])
            joint_hotspot_step = np.minimum(hotspot_step_a, hotspot_step_b)
            if float(np.min(full_joint[idx])) < STATIC_FRAME_MIN_FULL_VISIBILITY:
                continue
            if (
                float(np.min(joint_full_centroid_step))
                < STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM
            ):
                continue
            if float(np.min(joint_hotspot_step)) < STATIC_FRAME_MIN_HOTSPOT_STEP_MM:
                continue
            score = float(
                2.0 * np.min(joint_hotspot_step)
                + np.mean(joint_hotspot_step)
                + np.min(joint_full_centroid_step)
                + 0.5 * np.mean(joint_full_centroid_step)
                + 2.0 * np.min(joint_handoff)
                + np.mean(joint_handoff)
                + 2.0 * np.min(joint[idx])
                + np.mean(joint[idx])
                + 2.0 * np.min(joint_state_step)
                + np.mean(joint_state_step)
                + 0.1 * (rho_a - rho_b)
            )
            feasible.append(
                (
                    score, tuple(float(x) for x in times), idx, span_a, span_b,
                    rho_a, rho_b, state_step_a, state_step_b, joint_state_step,
                    handoff_a, handoff_b, joint_handoff,
                    full_centroid_step_a, full_centroid_step_b,
                    joint_full_centroid_step, hotspot_step_a, hotspot_step_b,
                    joint_hotspot_step,
                    float(times[1] - times[0]),
                )
            )
    if not feasible:
        fallback = _static_frame_times(t_lo, t_hi, n_frames)
        return fallback, {
            "selection_mode": "uniform_fallback_no_strict_equal_interval_solution",
            "grid_step_ms": STATIC_FRAME_GRID_STEP_MS,
            "times_are_equally_spaced": True,
        }
    (
        score, selected, idx, span_a, span_b, rho_a, rho_b,
        state_step_a, state_step_b, joint_state_step,
        handoff_a, handoff_b, joint_handoff,
        full_centroid_step_a, full_centroid_step_b, joint_full_centroid_step,
        hotspot_step_a, hotspot_step_b, joint_hotspot_step, equal_interval_ms,
    ) = max(
        feasible, key=lambda row: (row[0], tuple(-x for x in row[1]))
    )
    return np.asarray(selected, float), {
        "selection_mode": "equal_interval_full_field_hotspot_v6",
        "rendered_pixels_used": False,
        "times_are_equally_spaced": True,
        "equal_interval_ms": equal_interval_ms,
        "grid_step_ms": STATIC_FRAME_GRID_STEP_MS,
        "minimum_gap_ms": STATIC_FRAME_MIN_GAP_MS,
        "minimum_joint_visibility": STATIC_FRAME_MIN_JOINT_VISIBILITY,
        "minimum_axis_span_mm": STATIC_FRAME_MIN_AXIS_SPAN_MM,
        "minimum_contact_state_distance": STATIC_FRAME_MIN_STATE_DISTANCE,
        "minimum_abs_centroid_time_rho": STATIC_FRAME_MIN_CENTROID_RHO,
        "minimum_endpoint_handoff_per_step": STATIC_FRAME_MIN_ENDPOINT_HANDOFF,
        "minimum_full_participant_visibility": STATIC_FRAME_MIN_FULL_VISIBILITY,
        "minimum_full_centroid_step_mm": STATIC_FRAME_MIN_FULL_CENTROID_STEP_MM,
        "minimum_top3_hotspot_step_mm": STATIC_FRAME_MIN_HOTSPOT_STEP_MM,
        "objective_score": float(score),
        "joint_visibility": [float(x) for x in joint[idx]],
        "ta_axis_centroids_mm": [float(x) for x in cen_a[idx]],
        "tb_axis_centroids_mm": [float(x) for x in cen_b[idx]],
        "ta_centroid_vs_time_rho": rho_a,
        "tb_centroid_vs_time_rho": rho_b,
        "ta_contact_state_step_rms": [float(x) for x in state_step_a],
        "tb_contact_state_step_rms": [float(x) for x in state_step_b],
        "joint_contact_state_step_rms": [float(x) for x in joint_state_step],
        "ta_right_minus_left_endpoint_energy": [float(x) for x in endpoint_a[idx]],
        "tb_right_minus_left_endpoint_energy": [float(x) for x in endpoint_b[idx]],
        "ta_endpoint_handoff_step": [float(x) for x in handoff_a],
        "tb_endpoint_handoff_step": [float(x) for x in handoff_b],
        "joint_endpoint_handoff_step": [float(x) for x in joint_handoff],
        "joint_full_participant_visibility": [float(x) for x in full_joint[idx]],
        "ta_full_centroids_mm": [float(x) for x in full_cen_a[idx]],
        "tb_full_centroids_mm": [float(x) for x in full_cen_b[idx]],
        "ta_full_centroid_step_mm": [float(x) for x in full_centroid_step_a],
        "tb_full_centroid_step_mm": [float(x) for x in full_centroid_step_b],
        "joint_full_centroid_step_mm": [float(x) for x in joint_full_centroid_step],
        "ta_top3_hotspots_mm": [float(x) for x in hot_a[idx]],
        "tb_top3_hotspots_mm": [float(x) for x in hot_b[idx]],
        "ta_top3_hotspot_step_mm": [float(x) for x in hotspot_step_a],
        "tb_top3_hotspot_step_mm": [float(x) for x in hotspot_step_b],
        "joint_top3_hotspot_step_mm": [float(x) for x in joint_hotspot_step],
        "ta_axis_span_mm": span_a,
        "tb_axis_span_mm": span_b,
    }
