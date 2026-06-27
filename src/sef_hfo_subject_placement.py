# src/sef_hfo_subject_placement.py
"""Subject-specific SNN placement helpers for the field-swap plan (2026-06-26).

Pure (no sim, no engine). Loads montage-consistent swap endpoints + the precomputed
2D contact plane, then isotropically registers the patient plane into the blessed
L=20 SNN sheet so the two swap cores and the virtual electrodes share ONE frame.

Contract (CLAUDE.md §6 ritual, field-swap plan §2/§3A/§3B):
- montage-consistency: swap + geometry come from the SAME montage tree (`MONTAGE_TREES`).
- channel-overlap: swap source/sink names intersected with geometry names; LOUD-fail
  (raise) on any swap node missing from geometry -- never silently drop.
- coordinate-frame: one isotropic transform applied to electrodes AND core centroids.
- cores-inside-sheet: assert all transformed coords in [margin, L-margin].
Reuses src.sef_hfo_observation.from_real_geometry / VirtualMontage (no re-invent).
"""
from __future__ import annotations

import json
import os

import numpy as np

from src.sef_hfo_observation import from_real_geometry, VirtualMontage

MONTAGE_TREES = {
    "narrow": dict(
        rd="results/interictal_propagation_masked/rank_displacement/per_subject",
        geo="results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"),
    "broad": dict(
        rd="results/interictal_propagation_masked_broad/rank_displacement/per_subject",
        geo="results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"),
}


def load_swap_endpoints(subject, montage, root="."):
    """Source/sink swap-k nodes from the montage-assigned masked rank_displacement.

    Source-side = lowest decision_k ranks in rank_a_dense_full; sink-side = highest
    (field-swap plan §2). Returns names + swap metadata. Montage-consistency clause:
    `montage` selects the tree -- caller must pass the same montage used for geometry.
    """
    if montage not in MONTAGE_TREES:
        raise ValueError(f"unknown montage {montage!r}; use {list(MONTAGE_TREES)}")
    path = os.path.join(root, MONTAGE_TREES[montage]["rd"], f"{subject}.json")
    d = json.load(open(path))
    p = d["pairs"][0]
    ss = p["swap_sweep"]
    names = p["channel_names"]
    ranks = p["rank_a_dense_full"]
    dk = int(ss["decision_k"])
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    source = [names[i] for i in order[:dk]]
    sink = [names[i] for i in order[-dk:]]
    return dict(source=source, sink=sink, swap_class=ss["swap_class"],
                decision_k=dk, T_obs=ss["T_obs"], p_fw=ss["p_fw"],
                montage=montage, subject=subject)


def template_source_foci(subject, montage, k_early=3, root="."):
    """Two cores = the EARLIEST few electrodes of EACH interictal template (the two template
    SOURCES), using the field's typical_rank (low = early). For a swap pair, t_a's source is one
    end of the axis and t_b's source is the other end -- so the two cores sit at the two true ends,
    tight foci (not the middle-pulled swap-k centroids). Positions come from t_a's geometry (same
    contacts in both templates); electrodes co-registered from the same plane.

    Returns (montage[t_a positions], core_a_names[t_a earliest-k], core_b_names[t_b earliest-k]).
    """
    if montage not in MONTAGE_TREES:
        raise ValueError(f"unknown montage {montage!r}")
    geo = MONTAGE_TREES[montage]["geo"]
    ga = json.load(open(os.path.join(root, geo, f"{subject}_t_a.json")))
    gb = json.load(open(os.path.join(root, geo, f"{subject}_t_b.json")))
    m = from_real_geometry(ga)

    def earliest(g, k):
        chs = [c for c in g["channels"] if c.get("typical_rank") is not None]
        return [c["name"] for c in sorted(chs, key=lambda c: c["typical_rank"])[:k]]
    core_a = earliest(ga, k_early)          # t_a source (one end)
    core_b = earliest(gb, k_early)          # t_b source (other end)
    # both cores must exist in the t_a position frame
    missing = [n for n in core_a + core_b if n not in m.names]
    if missing:
        raise ValueError(f"template_source_foci: {missing} not in t_a geometry")
    return m, core_a, core_b


def load_subject_montage(subject, montage, template="t_a", root="."):
    """Patient contact plane (mm) as a VirtualMontage via from_real_geometry.

    Montage-consistency clause: geometry tree keyed by the same `montage`.
    """
    if montage not in MONTAGE_TREES:
        raise ValueError(f"unknown montage {montage!r}; use {list(MONTAGE_TREES)}")
    path = os.path.join(root, MONTAGE_TREES[montage]["geo"], f"{subject}_{template}.json")
    geom = json.load(open(path))
    return from_real_geometry(geom)


def _centroid(montage, names):
    idx = [montage.names.index(n) for n in names]
    return np.asarray(montage.contacts)[idx].mean(axis=0)


def register_to_sheet(montage, source_names, sink_names, L=20.0, margin=2.0,
                      target_inter_core_mm=None):
    """Isotropically register the patient plane into the L=20 sheet.

    Single transform (scale s, offset b) applied to ALL contacts -> the two core
    centroids are read off the SAME transformed frame (coordinate-frame clause).
    Channel-overlap clause: any swap node not in `montage.names` raises (loud).

    Two modes (one transform either way; preserves relative geometry + orientation):
    - `target_inter_core_mm=None` (PLANE-FIT): scale so the whole contact plane fits
      [margin, L-margin]. All contacts inside the sheet; cores can land closer than the
      blessed `sep_frac` separation (E958: 6.8 mm). Use for geometry display.
    - `target_inter_core_mm=d` (CORE-ANCHORED): scale so the two core centroids sit `d`
      mm apart, centered on [L/2, L/2]. This makes the cores land exactly where
      `build_lesion_vth(twoend_equal, sep_frac)` places them (d = sep_frac*L), so the
      SNN substrate is the blessed one; electrodes sit at their true RELATIVE positions
      and distant contacts may fall outside the sheet (auto-excluded by valid_mask).
      Use for the subject SNN run so the blessed spontaneous dynamics transfer.

    Returns the sheet-frame montage, the two core centroids, EE-axis angle
    (source -> sink = forward), center (= core midpoint), inter-core mm, transform.
    """
    missing = [n for n in list(source_names) + list(sink_names) if n not in montage.names]
    if missing:
        raise ValueError(
            f"register_to_sheet: swap nodes missing from geometry montage: {missing} "
            "(channel-overlap clause -- caller must reconcile the montage)")

    C = np.asarray(montage.contacts, float)
    src_cen_p = _centroid(montage, source_names)
    snk_cen_p = _centroid(montage, sink_names)

    if target_inter_core_mm is None:
        lo = C.min(axis=0)
        span = float(max((C.max(axis=0) - lo).max(), 1e-9))
        usable = L - 2.0 * margin
        s = usable / span                   # ISOTROPIC scale (preserve geometry)
        extent = (C.max(axis=0) - lo) * s
        b = margin + (usable - extent) / 2.0 - lo * s
        anchor = "plane_fit"
    else:
        inter_p = float(np.linalg.norm(snk_cen_p - src_cen_p))
        if inter_p < 1e-9:
            raise ValueError("register_to_sheet: source/sink centroids coincide; cannot core-anchor")
        s = float(target_inter_core_mm) / inter_p
        mid_p = 0.5 * (src_cen_p + snk_cen_p)
        b = np.array([L / 2.0, L / 2.0]) - s * mid_p   # core midpoint -> sheet center
        anchor = "core_anchored"

    Csheet = C * s + b
    msheet = VirtualMontage(Csheet, list(montage.names), provenance=f"real_geometry_2d_{anchor}")
    src_cen = _centroid(msheet, source_names)
    snk_cen = _centroid(msheet, sink_names)

    # cores-inside-sheet clause (cores must always be inside; electrodes only in plane-fit mode)
    for cen, tag in ((src_cen, "source"), (snk_cen, "sink")):
        assert (margin - 1e-6 <= cen).all() and (cen <= L - margin + 1e-6).all(), \
            f"{tag} centroid outside usable sheet"
    if anchor == "plane_fit":
        assert (Csheet >= -1e-6).all() and (Csheet <= L + 1e-6).all(), "transformed contacts outside [0,L]"

    v = snk_cen - src_cen               # forward axis = source(-end,early) -> sink(+end,late)
    theta_deg = float(np.degrees(np.arctan2(v[1], v[0])))
    center = 0.5 * (src_cen + snk_cen)
    inter = float(np.linalg.norm(v))
    n_offsheet = int(((Csheet < 0) | (Csheet > L)).any(axis=1).sum())
    return dict(montage_sheet=msheet, source_centroid=src_cen, sink_centroid=snk_cen,
                center=center, theta_deg=theta_deg, inter_core_mm_sheet=inter,
                scale=s, offset=b, L=L, margin=margin, anchor=anchor, n_contacts_offsheet=n_offsheet,
                source_names=list(source_names), sink_names=list(sink_names))
