"""Pure helpers for SEF-HFO SNN Stage 4 (extended single-patch stochastic readout).

Spec:  docs/superpowers/specs/2026-06-15-sef-hfo-snn-stage4-extended-patch-stochastic-readout-design.md
Plan:  docs/superpowers/plans/2026-06-15-sef-hfo-snn-stage4-extended-patch.md
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def _trim_mean(x, frac=0.2):
    x = np.sort(np.asarray(x, float))
    k = int(len(x) * frac)
    return x[k:len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def nucleation_centroid(spk, patch_E_idx, posE, t_on_idx, tau_nuc_steps,
                        axis_unit, patch_center, k_min=5):
    """Robust ground-truth seed location of an event inside the patch (spec §4.1).

    The onset is anchored on the **k_min-th** earliest patch spike (NOT the single
    first spike), so a temporally isolated stray cell cannot move the window; the
    centroid is a coordinate-wise trimmed mean. Returns None if fewer than k_min
    patch cells fire (unstable centroid).

    Returns dict(centroid_xy, s_nuc, r_off, n_early_cells):
      s_nuc  = centroid projection on axis_unit (relative to patch_center)
      r_off  = transverse (perpendicular) component
    """
    sub = spk[t_on_idx:, patch_E_idx]                       # (T', npatch)
    fired = sub.any(axis=0)
    if int(fired.sum()) < k_min:
        return None
    first = sub.argmax(axis=0).astype(float)                # first-spike step per cell
    first[~fired] = np.inf
    onset = np.sort(first[fired])[k_min - 1]                # k_min-th earliest = robust onset
    early = fired & (first >= onset - tau_nuc_steps) & (first <= onset + tau_nuc_steps)
    n_early = int(early.sum())
    if n_early < k_min:
        return None
    pts = posE[patch_E_idx][early]
    centroid = np.array([_trim_mean(pts[:, 0]), _trim_mean(pts[:, 1])])
    rel = centroid - np.asarray(patch_center, float)
    au = np.asarray(axis_unit, float); au = au / np.linalg.norm(au)
    perp = np.array([-au[1], au[0]])
    return dict(centroid_xy=centroid, s_nuc=float(rel @ au),
                r_off=float(rel @ perp), n_early_cells=n_early)


def _binary_entropy_bits(pfwd):
    """Binary Shannon entropy in bits: 0 when unidirectional, 1 at a 50/50 split."""
    if pfwd in (0.0, 1.0):
        return 0.0
    return float(-(pfwd * np.log2(pfwd) + (1 - pfwd) * np.log2(1 - pfwd)))


def readout_direction_distribution(signs, angles_deg, axis_angle_deg,
                                   near_axis_tol_deg=30.0):
    """Co-primary B summary of per-event readouts (spec §4.3).

    sign_entropy = binary Shannon entropy (bits) of the forward/reverse split ->
    THE bidirectionality measure (1 = 50/50, 0 = unidirectional).
    axis_concentration = |mean exp(i*2*theta)| over readable -> how tightly the
    readable AXES hug ONE line (sign-folded); high = on-axis, NOT a bidirectionality
    claim (doubled-angle collapses forward 0 deg and reverse 180 deg onto the same axis).
    Unreadable events (sign None / NaN angle) are counted, never coerced to a sign.
    """
    signs = list(signs)
    readable = [i for i, s in enumerate(signs) if s in (1, -1)]
    n_read = len(readable)
    n_unread = len(signs) - n_read
    fwd = sum(1 for i in readable if signs[i] == 1)
    forward_frac = (fwd / n_read) if n_read else float("nan")
    sign_entropy = _binary_entropy_bits(forward_frac) if n_read else float("nan")
    th2 = np.radians([angles_deg[i] for i in readable]) * 2.0
    axis_concentration = float(np.abs(np.mean(np.exp(1j * th2)))) if n_read else float("nan")

    def _axdist(a):
        d = abs((a - axis_angle_deg) % 180.0)
        return min(d, 180.0 - d)
    near = sum(1 for i in readable if _axdist(angles_deg[i]) <= near_axis_tol_deg)
    near_axis_frac = (near / n_read) if n_read else float("nan")
    return dict(n_readable=n_read, n_unreadable=n_unread, forward_frac=forward_frac,
                sign_entropy=sign_entropy, axis_concentration=axis_concentration,
                near_axis_frac=near_axis_frac)


def first_contact_entropy(first_contacts, n_contacts):
    """Normalized Shannon entropy of the first-active-contact distribution over the
    categorical contact labels. 0 = always the same contact; 1 = uniform over n_contacts."""
    labels = [c for c in first_contacts if c is not None]
    if not labels or n_contacts <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    H = -(p * np.log(p)).sum()
    return float(H / np.log(n_contacts))


def _auc(scores, labels):
    """ROC AUC via the Mann-Whitney U identity with average ranks (ties -> 0.5).
    Returns nan if labels are one-class."""
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(int)
    n1 = int(labels.sum()); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def correspondence_two_stage(s_nuc, r_off, readable, sign, rng, n_shuffle=500):
    """Co-primary A (spec §4.2). Stage 1: does nucleation position predict readability
    (AUC of |s_nuc| and -|r_off| -> readable). Stage 2: among readable events, does
    s_nuc predict sign, beating a within-event shuffle null. `sign` may contain None
    for unreadable events -> only readable indices are converted (None never reaches
    astype(int))."""
    s_nuc = np.asarray(s_nuc, float); r_off = np.asarray(r_off, float)
    readable = np.asarray(readable, bool); sign = list(sign)
    stage1_s = _auc(np.abs(s_nuc), readable.astype(int))       # end-like -> readable
    stage1_r = _auc(-np.abs(r_off), readable.astype(int))      # small offset -> readable
    ridx = np.where(readable)[0]
    if ridx.size < 4:
        return dict(stage1_auc_s_nuc=stage1_s, stage1_auc_r_off=stage1_r,
                    stage2_auc_sign=float("nan"), stage2_p_shuffle=float("nan"),
                    n_readable=int(ridx.size))
    sr = s_nuc[ridx]
    bad = [int(i) for i in ridx if sign[i] not in (1, -1)]
    if bad:   # readable ⟹ sign ∈ {+1,-1}; a None here would be silently mislabelled as reverse
        raise ValueError(f"readable event {bad[0]} has sign {sign[bad[0]]!r}, expected +1/-1")
    lab = np.array([1 if sign[i] == 1 else 0 for i in ridx])   # readable -> sign is +1/-1
    obs = _auc(sr, lab); obs_c = max(obs, 1.0 - obs)
    null = np.array([(lambda a: max(a, 1.0 - a))(_auc(sr, rng.permutation(lab)))
                     for _ in range(n_shuffle)])
    return dict(stage1_auc_s_nuc=stage1_s, stage1_auc_r_off=stage1_r,
                stage2_auc_sign=obs, stage2_p_shuffle=float((null >= obs_c).mean()),
                n_readable=int(ridx.size))


def nucleation_dispersion(s_nuc, r_off, patch_r, grid_n=6, elongation=1.0):
    """Dispersion summary of per-event nucleation centroids in axis coords (s_nuc = along-axis,
    r_off = transverse). Returns std along/transverse, a normalized spatial entropy (0-1) over a
    patch-area grid, and the top-2 grid-cell occupancy fraction.

    `elongation` (P1-3, 2026-06-17): the grid is ellipse-aware — the long (s) axis spans
    ±patch_r*elongation, the transverse (r) axis ±patch_r — so an elongated patch's far-along-axis
    nucleations are not mis-binned as edge/spread. Default 1.0 = isotropic disk (unchanged).
    `top2_occupancy` is a WARNING metric only — NOT a degeneracy verdict (clause C2; the verdict is
    `hotspot_degeneracy`). n<2 -> all nan (cannot summarize a single point) (clause C3)."""
    s = np.asarray(s_nuc, float); r = np.asarray(r_off, float)
    n = int(s.size)
    if n < 2:
        return dict(n=n, std_s_nuc=float("nan"), std_r_off=float("nan"),
                    spatial_entropy=float("nan"), top2_occupancy=float("nan"))
    s_ext = patch_r * elongation
    s_edges = np.linspace(-s_ext, s_ext, grid_n + 1)
    r_edges = np.linspace(-patch_r, patch_r, grid_n + 1)
    H, _, _ = np.histogram2d(np.clip(s, -s_ext, s_ext), np.clip(r, -patch_r, patch_r),
                             bins=[s_edges, r_edges])
    counts = H.ravel(); tot = counts.sum()
    p = counts[counts > 0] / tot
    # C1: Shannon entropy of grid-cell occupancy, normalized by log(n_cells) -> 0..1
    spatial_entropy = float(-(p * np.log(p)).sum() / np.log(grid_n * grid_n))
    top2_occupancy = float(np.sort(counts)[::-1][:2].sum() / tot)        # C2: WARNING only
    return dict(n=n, std_s_nuc=float(s.std()), std_r_off=float(r.std()),
                spatial_entropy=spatial_entropy, top2_occupancy=top2_occupancy)


def hotspot_degeneracy(s_nuc, r_off, patch_r, n_min=6, tight_frac=0.2, elongation=1.0):
    """Three-state continuous-patch gate (anti-two-hotspot; Phase 2 plan 2026-06-17). The single
    extended patch must stay a CONTINUOUS source, not collapse to <=2 spatially TIGHT hot-spots
    (a covert two-focus = Stage 2 in disguise). Returns `verdict` in:
      - `indeterminate_low_n`  (clause C4): < `n_min` nucleation-valid events -> cannot judge;
        NOT an extended-patch pass. Checked FIRST, before any clustering (k=2 on 3-4 pts is noise).
      - `two_hotspot_degenerate` (clause C5): a k=2 split gives BOTH clusters TIGHT (within-cluster
        RMS radius < `tight_frac`*patch_r). Fires on <=2 fixed tight points; control / diagnostic only.
      - `healthy` (clause C6): at least one k=2 cluster is spread (continuous), even if denser at the
        two ends.
    Clause C7 (CRITICAL): bimodality is NOT degeneracy. spec §3.1 EXPECTS the seed to favour the two
    ends (forward/reverse = which end seeded); an end-favouring but *spread* distribution has a LARGE
    within-cluster radius and stays `healthy`. Only TIGHTNESS triggers the fail — never the s_nuc
    histogram shape, and never `top2_occupancy` (clause C8: that is a warning, computed separately).
    NOTE: a single synchronous hot-spot (no nucleation variability at all) is normally already
    excluded upstream by the Phase-1 s_nuc dispersion gate; this gate targets the two-focus case.
    `elongation` (P1-3, 2026-06-17): clustering is done in NORMALIZED (possibly elliptical) patch
    coordinates — s/(patch_r*elongation), r/patch_r — so the unit patch is the (elliptical) core and
    `cluster_radii` / `tight_thresh` are in normalized units (fraction of the patch). For
    elongation=1.0 this equals the isotropic disk (radius/patch_r vs tight_frac) — verdicts unchanged."""
    from sklearn.cluster import KMeans
    s = np.asarray(s_nuc, float); r = np.asarray(r_off, float)
    n = int(s.size)
    if n < n_min:                                          # C4: low-n first, never hard-pass
        return dict(verdict="indeterminate_low_n", n=n, n_min=int(n_min))
    P = np.column_stack([s / (patch_r * elongation), r / patch_r])   # P1-3: normalized ellipse coords
    lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(P)
    radii = []
    for k in (0, 1):
        pts = P[lab == k]
        c = pts.mean(axis=0)
        radii.append(float(np.sqrt(((pts - c) ** 2).sum(axis=1).mean())))   # RMS dist to centroid (normalized)
    degenerate = max(radii) < tight_frac                  # C5: BOTH clusters tight (normalized < tight_frac)
    return dict(verdict="two_hotspot_degenerate" if degenerate else "healthy",
                n=n, cluster_radii=[round(x, 3) for x in radii], tight_thresh=round(float(tight_frac), 3))


def compute_t0_gate(s_nuc, r_off, patch_r, n_min=6, elongation=1.0):
    """Run-level T0 continuous-patch gate artifact (Phase 2 plan 2026-06-17): from a run's per-event
    nucleation centroids, drop events without a valid centroid (NaN), then package
    `nucleation_dispersion` + the 3-state `hotspot_degeneracy` verdict + `n_valid_nucleation`. This is
    what a smoke / ensemble run writes alongside its readout so a `two_hotspot_degenerate` or
    `indeterminate_low_n` run is never silently pooled as an extended-patch pass."""
    s = np.asarray(s_nuc, float); r = np.asarray(r_off, float)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    return dict(n_valid_nucleation=int(s.size), elongation=float(elongation),
                nucleation_dispersion=nucleation_dispersion(s, r, patch_r, elongation=elongation),
                hotspot_degeneracy=hotspot_degeneracy(s, r, patch_r, n_min=n_min, elongation=elongation))
