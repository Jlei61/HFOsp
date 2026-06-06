# src/sef_hfo_observation.py
"""Virtual-SEEG observation layer (Increment 1, spec 2026-06-06).

Small pure functions: montage -> envelope sampler -> lag extractor -> direction
estimators -> legacy-key artifact writer/validator. Model adapters live next to
the models, NOT here. No model dynamics in this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

# Time-unit contract: the model/sim and the toy-wave run in MILLISECONDS (sim
# convention, matches src.sef_hfo_events DT/MIN_DUR_MS etc.). The real legacy
# artifact the loader consumes is in SECONDS (verified: src.event_periodicity uses
# block_dur = 3600.0 s, detections + start_t in seconds). So LagPatArtifact carries
# ms internally and the legacy WRITERS convert ms -> s. Read-out gates (increment
# 1/2) only use rank ORDER, so units do not affect them; rate/IEI/window analyses
# (later) require the correct seconds, hence the conversion at the write boundary.
MS_TO_S: float = 1e-3


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


def validate_artifact(artifact: LagPatArtifact) -> None:
    """Assert the spec invariants before writing: 2D montage, name/coord order
    match, and non-participant entries carry NO finite rank (phantom-mask discipline)."""
    b = artifact.bools
    assert artifact.ranks.shape == b.shape == artifact.lag_raw.shape
    assert artifact.contact_coords.shape == (b.shape[0], 2)
    assert len(artifact.names) == b.shape[0]
    assert VirtualMontage(artifact.contact_coords, artifact.names, "x").spans_2d(), \
        "read-out montage must span 2D (>=2 non-parallel shafts) — spec D6"
    nonpart = ~b
    bad = np.isfinite(artifact.ranks[nonpart])
    assert not bad.any(), "non-participating contacts must have NaN rank (no phantom)"
    bad_lag = np.isfinite(artifact.lag_raw[nonpart])
    assert not bad_lag.any(), "non-participating contacts must have NaN lag"


def write_legacy_npz(artifact: LagPatArtifact, path) -> None:
    """Write the *_lagPat_withFreqCent.npz with the EXACT keys the real loader
    reads (src/interictal_propagation.py L344-353): lagPatRank / eventsBool /
    lagPatRaw / chnNames / start_t.

    UNIT: lagPatRaw is written in SECONDS (legacy on-disk unit). The artifact carries
    ms (sim unit); convert here via MS_TO_S. ranks are unit-free."""
    path = str(path)
    assert path.endswith("_lagPat_withFreqCent.npz"), \
        "filename must end with _lagPat_withFreqCent.npz for the loader's glob"
    validate_artifact(artifact)
    np.savez(
        path,
        lagPatRank=artifact.ranks.astype(float),
        eventsBool=artifact.bools.astype(np.int8),
        lagPatRaw=(artifact.lag_raw * MS_TO_S).astype(float),   # ms -> s
        chnNames=np.array(artifact.names, dtype=object),
        start_t=np.array(0.0),
    )


def write_packed_times(artifact: LagPatArtifact, path) -> None:
    """Companion *_packedTimes_withFreqCent.npy: (n_event, 2) rel start/end times.
    The model gives these directly (D1: no packer, but the loader needs them).

    UNIT: written in SECONDS (legacy on-disk unit); artifact carries ms -> convert."""
    path = str(path)
    assert path.endswith("_packedTimes_withFreqCent.npy")
    packed = np.column_stack([artifact.event_rel_times * MS_TO_S,
                              artifact.event_rel_end_times * MS_TO_S])   # ms -> s
    np.save(path, packed.astype(float))


def write_montage_manifest(artifact: LagPatArtifact, path) -> None:
    """Sidecar JSON carrying contact coords (legacy lagPat has none; D6 needs them).
    Order is asserted == chnNames."""
    payload = {"contact_coords": artifact.contact_coords.tolist(),
               "chn_names": list(artifact.names)}
    from pathlib import Path as _P
    _P(str(path)).write_text(json.dumps(payload), encoding="utf-8")


def direction_readability(ranks_ev, bools_ev, coords, n_axes=180) -> float:
    """C1/C2 must-fail scalar: the BEST rank-vs-projection Spearman over a sweep of
    candidate axes (0..180 deg) — gives a directionless source every chance to show an
    axis. High for a real wave (~main-gate Spearman), low for radial (rank=radius is
    monotone along no line), NaN for a fully-tied (synchronous) read-out. More robust
    than 'is the endpoint axis None' (tied ranks bias argsort into a spurious axis)."""
    idx = np.flatnonzero(np.asarray(bools_ev, bool))
    if idx.size < 2:
        return float("nan")
    r = np.asarray(ranks_ev, float)[idx]
    if np.ptp(r) == 0:
        return float("nan")
    xy = np.asarray(coords, float)[idx]
    rr = _rankdata_avg(r)
    best = 0.0
    for k in range(n_axes):
        th = np.pi * k / n_axes
        proj = xy @ np.array([np.cos(th), np.sin(th)])
        if np.ptp(proj) == 0:
            continue
        rho = abs(float(np.corrcoef(rr, _rankdata_avg(proj))[0, 1]))
        if rho > best:
            best = rho
    return best


def read_direction_from_source(source, montage, kernel_width,
                               participation_frac=0.5, timing_frac=0.5,
                               k_dir=3) -> dict:
    """End-to-end Increment-1 read-out: sample the analytic source through the montage,
    extract one event's lagPat, report the direction metrics.

    Returns dict with:
      spearman    : rank-vs-(true n_hat)-projection Spearman (nan when source has no n_hat)
      readability : best rank-vs-projection Spearman over all axes (C1/C2 must-fail)
      axis        : endpoint-centroid unit axis or None (for the wave angle-error report)
      artifact    : the LagPatArtifact (for persistence)
    """
    frames = source["frames"]
    coords = source["grid_xy"]
    dt = source["dt"]
    env = sample_envelopes(frames, coords, montage, kernel_width)
    # participation bar = TEMPORAL noise floor + margin (spec §4.1) — NOT a cross-contact
    # peak percentile. floor = global quiescent baseline (min over time, ~0 for the
    # noiseless toy); margin scales to the global event peak so every bell-crossed
    # contact robustly participates. Anti-fake-order: the TIMING threshold (inside
    # extract_lagpat) is per-contact relative to own peak.
    floor = float(env.min())
    margin = participation_frac * (float(env.max()) - floor)
    art = extract_lagpat(env, dt, event_windows=[source["window"]],
                         participation_floor=floor, participation_margin=margin,
                         timing_frac=timing_frac, tie_tol=dt)
    art = attach_geometry(art, montage)
    ranks0, bools0 = art.ranks[:, 0], art.bools[:, 0]
    pitch = source.get("pitch_hint", 1.0)
    axis = endpoint_centroid_axis(ranks0, bools0, art.contact_coords,
                                  k_dir=k_dir, eps_deg=0.5 * pitch)
    n_hat = source.get("n_hat")
    spearman = (rank_vs_projection_spearman(ranks0, bools0, art.contact_coords, n_hat)
                if n_hat is not None else float("nan"))
    readability = direction_readability(ranks0, bools0, art.contact_coords)
    return dict(spearman=spearman, readability=readability, axis=axis, artifact=art)
