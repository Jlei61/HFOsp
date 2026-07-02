"""Pure geometry / event helpers for the Topic 4 off-axis surround stimulation experiment.

Given a patient-like planar montage and a pathological propagation axis (source->sink), pick
NON-axis surround contacts (flanking the axis, outside the source/sink core and outside the axis
corridor) for stimulation, plus a matched on-axis-corridor comparator. No SNN engine imports --
fully unit-testable. See docs/superpowers/plans/2026-07-02-topic4-offaxis-surround-stimulation.md."""
from __future__ import annotations

import numpy as np


def axis_frame(source_xy, sink_xy):
    """Pathological-axis coordinate frame from the source and sink centroids (registered 2-D mm).
    Returns source/sink/center, the axis unit (source->sink), the perpendicular unit, and the
    inter-core distance."""
    s = np.asarray(source_xy, float); k = np.asarray(sink_xy, float)
    d = k - s
    L = float(np.linalg.norm(d))
    u = d / max(L, 1e-9)
    perp = np.array([-u[1], u[0]])                      # +90 deg rotation
    return {"source": s, "sink": k, "center": (s + k) / 2.0,
            "axis_unit": u, "perp_unit": perp, "inter_core_mm": L}


def project_contacts(contacts, frame):
    """Along-axis and off-axis (perpendicular) coordinates of each contact, relative to the axis
    center (mm). Returns {"along": (n,), "off": (n,)}."""
    C = np.asarray(contacts, float)
    rel = C - np.asarray(frame["center"], float)[None, :]
    return {"along": rel @ frame["axis_unit"], "off": rel @ frame["perp_unit"]}


def classify_axis_corridor(contacts, frame, corridor_halfwidth_mm, along_pad_mm=0.0):
    """Bool mask of contacts inside the axis corridor: within ``corridor_halfwidth_mm`` of the axis
    perpendicular-wise AND with along-axis position between/near the source-sink interval."""
    pr = project_contacts(contacts, frame)
    half_L = frame["inter_core_mm"] / 2.0
    in_off = np.abs(pr["off"]) <= float(corridor_halfwidth_mm)
    in_along = (pr["along"] >= -half_L - float(along_pad_mm)) & (pr["along"] <= half_L + float(along_pad_mm))
    return in_off & in_along


def _near_interval(pr, frame, along_pad_mm):
    half_L = frame["inter_core_mm"] / 2.0
    return (pr["along"] >= -half_L - float(along_pad_mm)) & (pr["along"] <= half_L + float(along_pad_mm))


def select_offaxis_surround_contacts(contacts, frame, core_contact_mask, N, corridor_halfwidth_mm,
                                     offaxis_min_mm, along_pad_mm=0.0):
    """N non-axis surround contacts flanking the axis, BALANCED N/2 on each side: perpendicular
    distance ``>= offaxis_min_mm``, NOT a source/sink core contact, NOT in the axis corridor, and
    along-axis near the source-sink interval. Deterministic tie-break: nearest the axis center
    along-wise, then lower index. Raises ValueError if either side cannot be filled."""
    if N % 2 != 0:
        raise ValueError("N must be even for a balanced off-axis split")
    pr = project_contacts(contacts, frame)
    core = np.asarray(core_contact_mask, bool)
    corridor = classify_axis_corridor(contacts, frame, corridor_halfwidth_mm, along_pad_mm)
    near = _near_interval(pr, frame, along_pad_mm)
    elig = (np.abs(pr["off"]) >= float(offaxis_min_mm)) & (~core) & (~corridor) & near

    def pick(side_mask, k):
        idx = np.flatnonzero(elig & side_mask).tolist()
        idx.sort(key=lambda i: (abs(float(pr["along"][i])), i))
        return idx[:k]

    pos = pick(pr["off"] > 0, N // 2)
    neg = pick(pr["off"] < 0, N // 2)
    if len(pos) < N // 2 or len(neg) < N // 2:
        raise ValueError(f"insufficient off-axis contacts: {len(pos)}+{len(neg)} < {N} "
                         f"(need N/2={N // 2} on each side of the axis; "
                         f"offaxis_min_mm={offaxis_min_mm}, corridor_halfwidth_mm={corridor_halfwidth_mm})")
    return np.array(sorted(pos + neg))


def select_onaxis_corridor_contacts(contacts, frame, core_contact_mask, N, corridor_halfwidth_mm,
                                    along_pad_mm=0.0):
    """N on-axis-corridor comparator contacts: in the axis corridor, NOT a core contact. Nearest the
    axis center along-wise, lower index first. Raises ValueError if fewer than N are available."""
    pr = project_contacts(contacts, frame)
    core = np.asarray(core_contact_mask, bool)
    corridor = classify_axis_corridor(contacts, frame, corridor_halfwidth_mm, along_pad_mm)
    idx = np.flatnonzero(corridor & (~core)).tolist()
    idx.sort(key=lambda i: (abs(float(pr["along"][i])), i))
    if len(idx) < N:
        raise ValueError(f"insufficient on-axis corridor contacts: {len(idx)} < {N} "
                         f"(corridor_halfwidth_mm={corridor_halfwidth_mm})")
    return np.array(sorted(idx[:N]))


def electrode_e_mask(posE, contacts, indices, radius_mm):
    """Bool mask over ``posE`` (E-cell positions): True where within ``radius_mm`` of ANY selected
    contact (``contacts[indices]``). Empty selection -> all-False."""
    P = np.asarray(posE, float)
    sel = np.asarray(contacts, float)[np.asarray(indices, int)] if len(np.asarray(indices)) else np.empty((0, 2))
    if len(sel) == 0:
        return np.zeros(len(P), bool)
    d = np.linalg.norm(P[:, None, :] - sel[None, :, :], axis=2)
    return d.min(axis=1) <= float(radius_mm)
