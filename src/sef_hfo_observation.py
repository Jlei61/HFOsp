# src/sef_hfo_observation.py
"""Virtual-SEEG observation layer (Increment 1, spec 2026-06-06).

Small pure functions: montage -> envelope sampler -> lag extractor -> direction
estimators -> legacy-key artifact writer/validator. Model adapters live next to
the models, NOT here. No model dynamics in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VirtualMontage:
    contacts: np.ndarray   # (n_contact, 2) coords in model frame, mm
    names: list            # length n_contact
    provenance: str

    def spans_2d(self, tol: float = 1e-6) -> bool:
        c = self.contacts - self.contacts.mean(axis=0, keepdims=True)
        return int(np.linalg.matrix_rank(c, tol=tol)) >= 2


def build_shaft(angle_rad, pitch, n_contacts, origin=(0.0, 0.0),
                name_prefix="A") -> VirtualMontage:
    """One linear shaft: n_contacts evenly spaced (pitch mm) along angle_rad,
    centered on origin. Contacts named <prefix>0..<prefix>(n-1)."""
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    offs = (np.arange(n_contacts) - (n_contacts - 1) / 2.0) * pitch
    contacts = np.asarray(origin, float)[None, :] + offs[:, None] * d[None, :]
    names = [f"{name_prefix}{i}" for i in range(n_contacts)]
    return VirtualMontage(contacts, names, provenance="parametric_shaft")


def merge_montages(montages) -> VirtualMontage:
    """Combine shafts into one montage (the ≥2-non-parallel-shaft 2D read-out)."""
    contacts = np.vstack([m.contacts for m in montages])
    names = [nm for m in montages for nm in m.names]
    return VirtualMontage(contacts, names, provenance="parametric_multi_shaft")


def from_real_geometry(*args, **kwargs):
    """Layer-2 stub (spec §7): 3D real SEEG coords -> 2D model frame. Loud-fail
    until the per-patient heterogeneity round builds it."""
    raise NotImplementedError(
        "real-geometry montage (3D->2D registration) is layer 2; see spec §7")


def grid_coords(n: int, L: float) -> np.ndarray:
    """Flattened (n*n, 2) mm coords matching src.sef_hfo_field._grid (ij-indexed,
    centered, spacing L/n). Row-major .ravel() order aligns with a field reshaped
    via field.reshape(n_time, -1)."""
    x = (np.arange(n) - n // 2) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])


def sample_envelopes(source_frames, grid_xy, montage, kernel_width,
                     Rr=None) -> np.ndarray:
    """Per-contact activity envelope = distance-weighted average of source over
    nearby grid pixels (generalizes engine/lfp.py to arbitrary contact coords).

    source_frames : (n_time, n_pix) — a field time series flattened per frame.
    grid_xy       : (n_pix, 2) mm coords (from grid_coords).
    Gaussian footprint exp(-d^2 / 2 kernel_width^2), normalized per contact.
    Optional Rr (mm) hard cutoff; None = all pixels.
    Returns (n_contact, n_time).
    """
    frames = np.asarray(source_frames, float)
    out = np.empty((len(montage.contacts), frames.shape[0]))
    for ci, c in enumerate(montage.contacts):
        d = np.linalg.norm(grid_xy - c[None, :], axis=1)
        mask = np.ones(d.shape, bool) if Rr is None else (d <= Rr)
        dd = d[mask]
        w = np.exp(-(dd ** 2) / (2.0 * kernel_width ** 2))
        w = w / max(w.sum(), 1e-12)
        out[ci] = frames[:, mask] @ w
    return out


@dataclass
class LagPatArtifact:
    bools: np.ndarray          # (n_contact, n_event) bool
    ranks: np.ndarray          # (n_contact, n_event) float; non-participant = NaN
    lag_raw: np.ndarray        # (n_contact, n_event) float ms; non-participant = NaN
    contact_coords: np.ndarray # (n_contact, 2)
    names: list
    event_rel_times: np.ndarray      # (n_event,) ms, event onset
    event_rel_end_times: np.ndarray  # (n_event,) ms, event end


def _dense_ranks_with_tie_tol(lags, tie_tol):
    """Dense ranks (0,1,2,...) over a 1-D lag array; lags within tie_tol of a
    group's first value share a rank."""
    order = np.argsort(lags, kind="mergesort")
    sorted_lags = lags[order]
    grp = np.zeros(len(lags), dtype=float)
    start = sorted_lags[0]
    g = 0.0
    for i in range(1, len(lags)):
        if sorted_lags[i] - start > tie_tol:
            g += 1.0
            start = sorted_lags[i]
        grp[i] = g
    ranks = np.empty(len(lags))
    ranks[order] = grp
    return ranks


def extract_lagpat(envelopes, dt, event_windows, participation_floor,
                   participation_margin, timing_frac=0.5, tie_tol=None) -> LagPatArtifact:
    """Per event window: participation (max env > floor+margin) sets bools; among
    participants, activation time = first crossing of timing_frac * own in-window
    peak (spec §4.1 timing 阈, per-contact relative); ranks = tie-tolerant dense
    rank of those times. Non-participants -> NaN rank/lag (no phantom finite rank).

    contact_coords/names are filled by the caller via attach_geometry()."""
    env = np.asarray(envelopes, float)
    n_c = env.shape[0]
    n_ev = len(event_windows)
    if tie_tol is None:
        tie_tol = dt
    bools = np.zeros((n_c, n_ev), bool)
    ranks = np.full((n_c, n_ev), np.nan)
    lag_raw = np.full((n_c, n_ev), np.nan)
    bar = participation_floor + participation_margin
    ev_on = np.array([w[0] for w in event_windows], float)
    ev_off = np.array([w[1] for w in event_windows], float)
    for ev, (t_on, t_off) in enumerate(event_windows):
        s = int(round(t_on / dt))
        e = int(round(t_off / dt))
        seg = env[:, s:e]                      # (n_c, win)
        if seg.shape[1] == 0:
            continue
        peak = seg.max(axis=1)
        part = peak > bar
        bools[part, ev] = True
        for ci in np.flatnonzero(part):
            thr = timing_frac * peak[ci]
            crossings = np.flatnonzero(seg[ci] >= thr)
            lag_raw[ci, ev] = t_on + crossings[0] * dt   # first-crossing time (ms)
        idx = np.flatnonzero(part)
        if idx.size:
            ranks[idx, ev] = _dense_ranks_with_tie_tol(lag_raw[idx, ev], tie_tol)
    return LagPatArtifact(bools=bools, ranks=ranks, lag_raw=lag_raw,
                          contact_coords=np.zeros((n_c, 2)), names=[""] * n_c,
                          event_rel_times=ev_on, event_rel_end_times=ev_off)


def attach_geometry(artifact: LagPatArtifact, montage: VirtualMontage) -> LagPatArtifact:
    """Fill coords/names from the montage (order = contact order). Asserts length."""
    assert len(montage.names) == artifact.bools.shape[0]
    artifact.contact_coords = np.asarray(montage.contacts, float)
    artifact.names = list(montage.names)
    return artifact


def _rankdata_avg(a):
    """Average ranks (ties share mean rank), pure numpy."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def rank_vs_projection_spearman(ranks_ev, bools_ev, coords, n_hat) -> float:
    """Increment-1 main gate: Spearman between per-event participant ranks and
    their projection onto the known direction n_hat. NaN if <2 participants or
    no variance (e.g. all tied)."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 2:
        return float("nan")
    r = np.asarray(ranks_ev, float)[idx]
    proj = (np.asarray(coords, float)[idx] @ np.asarray(n_hat, float))
    if np.ptp(r) == 0 or np.ptp(proj) == 0:
        return float("nan")            # no order to compare (e.g. synchronous source)
    rr = _rankdata_avg(r)
    rp = _rankdata_avg(proj)
    return float(np.corrcoef(rr, rp)[0, 1])


def endpoint_centroid_axis(ranks_ev, bools_ev, coords, k_dir=3, eps_deg=None):
    """Increment-2 main estimator (also Increment-1 C1 no-axis check). Axis =
    centroid(k_dir earliest-rank contacts) -> centroid(k_dir latest). Returns a
    unit 2-vector, or None when: <2*k_dir+1 participants, endpoint sets overlap,
    or ||axis|| < eps_deg (degenerate / no-axis)."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 2 * k_dir + 1:
        return None
    r = np.asarray(ranks_ev, float)[idx]
    xy = np.asarray(coords, float)[idx]
    if np.ptp(r) == 0:        # all participants tied -> no early/late distinction
        return None
    order = np.argsort(r, kind="mergesort")
    early = order[:k_dir]
    late = order[-k_dir:]
    if set(early.tolist()) & set(late.tolist()):
        return None
    vec = xy[late].mean(0) - xy[early].mean(0)
    norm = float(np.linalg.norm(vec))
    if eps_deg is not None and norm < eps_deg:
        return None
    if norm < 1e-9:
        return None
    return vec / norm


def axis_angle_error_deg(axis, theta_ref) -> float:
    """Undirected (mod 180°) angle between a 2-vector axis and a reference angle."""
    a = np.arctan2(axis[1], axis[0])
    diff = np.rad2deg(a - theta_ref) % 180.0
    return float(min(diff, 180.0 - diff))
