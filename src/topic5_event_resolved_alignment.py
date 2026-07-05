"""Topic 5 A-line — event-resolved interictal axis_bias (SECONDARY, exploratory).

Spec: docs/superpowers/specs/2026-06-25-topic5-event-resolved-axis-bias-design.md (v2)
Plan: docs/superpowers/plans/2026-06-25-topic5-event-resolved-axis-bias.md

WHAT (白话, §0): the A-line primary collapses a subject's thousands of interictal HFO
propagations into ONE averaged spatial-gradient field and asks if it is collinear with the
seizure-onset gradient. This module does NOT average: it takes each interictal event already
labelled A/B (stable_k=2 KMeans cluster), builds that single event's small gradient field on
its OWN class plane, and asks — with the SAME sign-free mirror-invariant field correlation the
primary uses — how aligned that one event is to the ictal field. The spread of these per-event
alignments within a class is the within-class dispersion (the "std effect"); the A-vs-B
location difference is the class bias.

This is a SECONDARY descriptive analysis. It does NOT touch or extend the A-line primary
cohort claim. Reuses (does not reinvent): src.interictal_propagation.{load_subject_propagation_events,
_valid_event_indices,_legacy_hist_mean_rank}, src.lagpat_rank_audit.mask_phantom_ranks,
src.propagation_contact_plane_readout.{R_smooth_rank,corr_pair_mirror_invariant,make_plane_grid,
S_THRESH,OVERLAP_MIN}.

CONTRACTS (spec §5; each honoured by name in the code):
  C1  positional label↔event alignment — three hard raises (channel_names / per-class counts /
      exact producer template via _legacy_hist_mean_rank on RAW ranks).
  C2  cluster_id↔t_a/t_b — signed corr, 2x2 bijection + margin; else ambiguous (exclude).
  C3  each class uses its OWN plane (A→t_a, B→t_b); never a single shared plane.
  C4  per-event support = THIS event's participation (1.0 / dropped), not aggregate support;
      sigma_xy pinned to the class full-channel template value (passed in, not re-derived).
  C5  phantom mask via mask_phantom_ranks (NaN-drop), NEVER build_masked_kmeans_features (0.5 impute).
  C7  block-awareness: separation null permutes BLOCK→label (not per-event); report n_blocks.
  C8  real & null pass through the identical mirror-invariant + sign-free reduction.
  C10 Stage B/C entry points raise NotImplementedError (no plausible stub).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr, rankdata as _rankdata

from src.interictal_propagation import (
    load_subject_propagation_events,
    _valid_event_indices,
    _legacy_hist_mean_rank,
)
from src.lagpat_rank_audit import mask_phantom_ranks
from src.topic5_axis_alignment import channel_shuffle
from src.propagation_contact_plane_readout import (
    R_smooth_rank,
    corr_pair_mirror_invariant,
    make_plane_grid,
    S_THRESH,
    OVERLAP_MIN,
)

MIN_PARTICIPATING = 3          # valid-event gate (matches upstream labels)
MIN_PART_EVENT = 5             # M1d per-event lower bound
CHANNEL_HEADROOM = 3           # M1d: event must leave >=3 channels OUT (n_part <= n_ch - 3)
MIN_PLANE_CONTACTS = 3         # M field: need >=3 participating contacts on the plane to smooth


# ----------------------------------------------------------------------------- C1: loader
def _broad_lagpat_dir(dataset: str, subject: str) -> Path:
    """Broad-pool lagPat dir that PRODUCED the broad labels.
    epi: lagpat_broad_epilepsiae (verified 2026-06-25). yuquan: lagpat_broad_dyn — the canonical
    pool the masked_broad labels were recomputed from (verified 2026-06-26: xuxinyi 5684, zhangjinhan
    1802 reproduce exactly; older results/lagpat_broad gave a different xuxinyi pool). If a yuquan
    subject's labels came from a different pool, §C1 raises (not a silent mismatch)."""
    if dataset == "yuquan":
        return Path(f"results/lagpat_broad_dyn/{subject}")
    return Path(f"results/lagpat_broad_epilepsiae/{subject}")


def load_event_labels_ranks(
    dataset: str,
    subject: str,
    *,
    broad: bool = True,
    labels_dir: str = "results/interictal_propagation_masked_broad/per_subject",
    lagpat_dir: Optional[str] = None,
    template_tol: float = 0.99,
) -> dict:
    """Load broad A/B labels + per-event ranks, with the §C1 positional-alignment proof.

    Returns dict with: masked (n_ch,n_valid, phantom-masked, NaN non-participating),
    bools (n_ch,n_valid), ranks_raw (n_ch,n_valid), labels (n_valid,), valid_ev,
    event_abs_times (n_valid,), block_ids (n_valid,), channel_names, n_blocks,
    cluster_template_ranks {0:list,1:list} (the producer templates), dataset, subject.

    Raises ValueError on ANY §C1 mismatch (length identity is NOT sufficient — broad/glob
    drift can make n_valid coincide while events differ; we prove positional identity by
    reproducing the producer's per-cluster template exactly).
    """
    if not broad:
        raise NotImplementedError("narrow substrate path for the field metric is not built; "
                                  "narrow is companion-only (M1d). See spec §4.")
    js = json.load(open(Path(labels_dir) / f"{dataset}_{subject}.json"))
    ac = js["adaptive_cluster"]
    if not (ac.get("stable_k") == 2 and ac.get("chosen_k") == 2):
        raise ValueError(f"{dataset}_{subject}: not stable_k==chosen_k==2 "
                         f"(stable_k={ac.get('stable_k')}, chosen_k={ac.get('chosen_k')})")
    json_names = list(js["channel_names"])
    labels = np.asarray(ac["labels"], dtype=int)
    clusters = ac["clusters"]

    lp = Path(lagpat_dir) if lagpat_dir else _broad_lagpat_dir(dataset, subject)
    ev = load_subject_propagation_events(lp)
    ranks, bools, ch = ev["ranks"], ev["bools"], list(ev["channel_names"])
    valid_ev = _valid_event_indices(bools, min_participating=MIN_PARTICIPATING)

    # C1.0 length
    if valid_ev.size != labels.size:
        raise ValueError(f"{dataset}_{subject}: n_valid {valid_ev.size} != labels {labels.size}")
    # C1.1 channel_names elementwise
    if ch != json_names:
        raise ValueError(f"{dataset}_{subject}: channel_names mismatch "
                         f"(loaded {ch[:4]}... vs json {json_names[:4]}...)")
    # C1.2 per-cluster counts + C1.3 exact producer template (RAW ranks via _legacy_hist_mean_rank)
    cluster_template_ranks = {}
    for k in (0, 1):
        sel = valid_ev[labels == k]
        if int(sel.size) != int(clusters[k]["n_events"]):
            raise ValueError(f"{dataset}_{subject}: cluster {k} count {sel.size} "
                             f"!= json n_events {clusters[k]['n_events']}")
        templ = _legacy_hist_mean_rank(ranks[:, sel], bools[:, sel])
        tr = np.argsort(np.argsort(templ)).tolist()
        cluster_template_ranks[k] = tr
        jr = list(clusters[k]["template_rank"])
        if tr != jr:
            rho = spearmanr(tr, jr).correlation if len(tr) > 2 else 0.0
            if not (np.isfinite(rho) and rho >= template_tol):
                raise ValueError(f"{dataset}_{subject}: cluster {k} producer-template mismatch "
                                 f"(exact={tr==jr}, rank-corr={rho:.3f} < {template_tol}); "
                                 f"positional label↔event alignment NOT proven")

    masked = mask_phantom_ranks(ranks, bools, normalize=True)[:, valid_ev]   # C5: NaN-drop
    block_ids = np.asarray(ev["block_ids"])[valid_ev]
    abs_t = np.asarray(ev["event_abs_times"])[valid_ev]
    return {
        "dataset": dataset, "subject": subject,
        "masked": masked, "bools": bools[:, valid_ev].astype(bool),
        "ranks_raw": ranks[:, valid_ev], "labels": labels, "valid_ev": valid_ev,
        "event_abs_times": abs_t, "block_ids": block_ids,
        "channel_names": ch, "n_blocks": int(np.unique(block_ids).size),
        "cluster_template_ranks": cluster_template_ranks,
    }


# ----------------------------------------------------------------------------- C2: map
def map_clusters_to_templates(
    cluster_rank_0: Sequence[float],
    cluster_rank_1: Sequence[float],
    t_a_rank: Sequence[float],
    t_b_rank: Sequence[float],
    *,
    margin: float = 0.30,
) -> dict:
    """Map cluster_id {0,1} -> {"t_a","t_b"} by SIGNED Spearman, requiring a clean 2x2.

    All four rank vectors MUST be aligned to the same channel ordering (caller's job).
    Returns {"map": {0:..,1:..}, "diag_minus_offdiag": float, "ambiguous": bool}.
    Ambiguous (weak diagonal OR non-bijection) -> caller EXCLUDES the subject (C2: never
    silently pick). Forward/reverse near-mirror templates are the common case here, so the
    correlation must be SIGNED, not abs.
    """
    def _rho(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        return spearmanr(x[m], y[m]).correlation

    C = np.array([[_rho(cluster_rank_0, t_a_rank), _rho(cluster_rank_0, t_b_rank)],
                  [_rho(cluster_rank_1, t_a_rank), _rho(cluster_rank_1, t_b_rank)]], float)
    if not np.isfinite(C).all():
        return {"map": None, "diag_minus_offdiag": float("nan"), "ambiguous": True,
                "corr_matrix": C.tolist()}
    a0 = int(np.argmax(C[0])); a1 = int(np.argmax(C[1]))
    bijection = (a0 != a1)
    if bijection and a0 == 0:           # c0->t_a, c1->t_b
        diag = C[0, 0] + C[1, 1]; off = C[0, 1] + C[1, 0]; mp = {0: "t_a", 1: "t_b"}
    elif bijection:                     # c0->t_b, c1->t_a
        diag = C[0, 1] + C[1, 0]; off = C[0, 0] + C[1, 1]; mp = {0: "t_b", 1: "t_a"}
    else:
        diag = off = np.nan; mp = None
    dmo = float((diag - off) / 2.0) if bijection else float("nan")
    ambiguous = (not bijection) or (not np.isfinite(dmo)) or (dmo < margin)
    return {"map": (None if ambiguous else mp), "diag_minus_offdiag": dmo,
            "ambiguous": bool(ambiguous), "corr_matrix": C.tolist()}


# ----------------------------------------------------------------------------- field helpers
def build_plane_xy(plane_record: dict) -> Dict[str, tuple]:
    """name -> (x_norm, y_norm) for channels with finite coords (a class's contact plane)."""
    out = {}
    for c in plane_record["channels"]:
        x, y = c.get("x_norm"), c.get("y_norm")
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            out[c["name"]] = (float(x), float(y))
    return out


def _event_field(part_names, part_vals, plane_xy, X, Y, sigma, s_thresh):
    """Smooth ONE event's masked ranks on the class plane. C4: support=1.0 per participating
    contact (NOT aggregate support); sigma pinned (passed in). Returns smoothed field or None
    if too few plane contacts."""
    chans = []
    for n, v in zip(part_names, part_vals):
        if n in plane_xy and np.isfinite(v):
            x, y = plane_xy[n]
            chans.append({"name": n, "x_norm": x, "y_norm": y,
                          "typical_rank": float(v), "support": 1.0})   # C4 support=event participation
    if len(chans) < MIN_PLANE_CONTACTS:
        return None
    return R_smooth_rank({"channels": chans}, X, Y, sigma, s_thresh)


def make_subject_ictal_field(plane_record: dict, ictal_by_channel: Dict[str, float],
                             *, sigma: float, X, Y, s_thresh: float = S_THRESH) -> Optional[dict]:
    """Subject-mean ictal field on a CLASS plane (C3): channels that are on the plane AND have
    an ictal activation value, with the plane's aggregate support, pinned sigma.

    NOTE (C6): this is the subject-MEAN bb_auc field — a MORE-AVERAGED estimator than the
    primary's per-seizure-median; it is NOT identical to the primary's construction.
    """
    chans = []
    for c in plane_record["channels"]:
        nm = c["name"]
        if nm in ictal_by_channel and np.isfinite(ictal_by_channel[nm]) \
           and np.isfinite(c.get("x_norm", np.nan)) and np.isfinite(c.get("y_norm", np.nan)) \
           and c.get("support", 0) > 0:
            chans.append({"name": nm, "x_norm": float(c["x_norm"]), "y_norm": float(c["y_norm"]),
                          "typical_rank": float(ictal_by_channel[nm]),
                          "support": float(c["support"])})
    if len(chans) < MIN_PLANE_CONTACTS:
        return None
    return R_smooth_rank({"channels": chans}, X, Y, sigma, s_thresh)


def class_template_sigma(plane_record: dict, *, X, Y, s_thresh: float = S_THRESH) -> float:
    """The class's full-channel template smoothing sigma (sigma=None -> NN-spacing derived),
    pinned and reused for every per-event field of that class (C4)."""
    chans = [{"name": c["name"], "x_norm": float(c["x_norm"]), "y_norm": float(c["y_norm"]),
              "typical_rank": float(c.get("typical_rank", 0.0)), "support": float(c.get("support", 1.0))}
             for c in plane_record["channels"]
             if np.isfinite(c.get("x_norm", np.nan)) and np.isfinite(c.get("y_norm", np.nan))]
    f = R_smooth_rank({"channels": chans}, X, Y, None, s_thresh)
    return float(f["sigma_xy"])


# ----------------------------------------------------------------------------- M: per-event field metric
def per_event_field_alignment(
    bundle: dict,
    *,
    plane_by_label: Dict[int, dict],
    ictal_field_by_label: Dict[int, dict],
    sigma_by_label: Dict[int, float],
    overlap_min: int = OVERLAP_MIN,
    s_thresh: float = S_THRESH,
) -> dict:
    """Primary metric M (spec §3.1): per-event mirror-invariant field alignment, by A/B class.

    For each valid event e (class g = labels[e]): build F_e on plane_by_label[g] (C3) with
    per-event support (C4) and sigma_by_label[g] (C4), then
    align_e = | corr_pair_mirror_invariant(F_e, ictal_field_by_label[g]) | (C8 sign-free).
    Events failing overlap (or too few plane contacts) -> status, counted in usable_fraction.
    """
    masked = bundle["masked"]; bools = bundle["bools"]; labels = bundle["labels"]
    names = bundle["channel_names"]; block_ids = bundle["block_ids"]
    abs_t = bundle["event_abs_times"]
    X, Y = make_plane_grid()
    plane_xy = {k: build_plane_xy(plane_by_label[k]) for k in (0, 1)}
    per_event = []
    n_ev = masked.shape[1]
    for i in range(n_ev):
        g = int(labels[i])
        part = np.where(bools[:, i])[0]
        pnames = [names[j] for j in part]
        pvals = masked[part, i]
        rec = {"event_idx": int(bundle["valid_ev"][i]), "abs_time": float(abs_t[i]),
               "block_id": int(block_ids[i]), "label": g, "n_part": int(part.size)}
        # P0 fix: evaluate EACH event under BOTH class planes/ictal fields, so the R2 label
        # permutation can re-pick the value under the shuffled label's plane (pays the
        # class-specific plane-selection cost). align0 = on plane_by_label[0] vs ictal[0], etc.
        aligns = {}
        for k in (0, 1):
            ict = ictal_field_by_label.get(k)
            F_e = (_event_field(pnames, pvals, plane_xy[k], X, Y, sigma_by_label[k], s_thresh)
                   if ict is not None else None)
            if ict is None or F_e is None:
                aligns[k] = None; continue
            # C8: same mirror-invariant + abs reduction for observed and (later) null
            r = corr_pair_mirror_invariant(F_e["T"], F_e["S"], ict["T"], ict["S"], s_thresh, overlap_min)
            aligns[k] = (float(abs(r["corr"])) if (not r["insufficient_overlap"]
                         and r["corr"] is not None and np.isfinite(r["corr"])) else None)
        rec["align0"] = aligns[0]; rec["align1"] = aligns[1]
        rec["align"] = aligns[g]                          # assigned = own class's plane
        rec["status"] = "ok" if aligns[g] is not None else "unresolved"
        per_event.append(rec)
    usable = [r for r in per_event if r["status"] == "ok"]
    n_blocks_usable = int(np.unique([r["block_id"] for r in usable]).size) if usable else 0
    return {"per_event": per_event, "n_events": n_ev, "n_usable": len(usable),
            "usable_fraction": (len(usable) / n_ev if n_ev else 0.0),
            "n_blocks_usable": n_blocks_usable}


def class_aggregate_contact_values(bundle: dict, label: int) -> dict:
    """Per-contact weight-normalized aggregate of masked ranks over ONE class's events
    (the per-class FIELD inputs). value[c] = nanmean over class events of masked[c,·]
    (channel never participating -> NaN); support[c] = fraction of class events the contact
    participates in (the natural participation weight). Returns name -> {value, support}.
    """
    masked = bundle["masked"]; bools = bundle["bools"]; labels = bundle["labels"]
    names = bundle["channel_names"]
    cols = np.where(np.asarray(labels) == label)[0]
    out = {}
    if cols.size == 0:
        return {n: {"value": np.nan, "support": 0.0} for n in names}
    sub = masked[:, cols]; subb = np.asarray(bools)[:, cols].astype(bool)
    with np.errstate(invalid="ignore"):
        val = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmean(sub, axis=1))
    sup = subb.sum(axis=1) / float(cols.size)
    for c, n in enumerate(names):
        out[n] = {"value": float(val[c]) if np.isfinite(val[c]) else np.nan,
                  "support": float(sup[c])}
    return out


# ----------------------------------------------------------------------------- M1d: 1D companion
def per_event_1d_alignment(
    bundle: dict,
    ictal_by_channel: Dict[str, float],
    *,
    min_part: int = MIN_PART_EVENT,
    headroom: int = CHANNEL_HEADROOM,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Companion metric M1d (spec §3.2): per-event 1D collinear |Spearman(order, activation)|.

    REPLAY-ADJACENT — a strictly more replay-flavoured construct than the primary 2D field
    metric. NO sign is stored (§6). Only ever summarised as a class-level distribution, never
    per-event-named. Eligible only when the montage has channel headroom: n_ch >= min_part +
    headroom, and per event min_part <= n_part <= n_ch - headroom (the event must leave >=
    headroom channels OUT, else |Spearman| just re-measures the subject axis).
    Per-event null: permute activation among the event's OWN participating contacts.
    """
    rng = rng or np.random.default_rng()
    masked = bundle["masked"]; bools = bundle["bools"]; labels = bundle["labels"]
    names = bundle["channel_names"]
    n_ch = len(names)
    if n_ch < min_part + headroom:
        return {"eligible": False, "n_channels": n_ch, "per_event": [],
                "reason": f"n_ch {n_ch} < min_part+headroom {min_part + headroom}"}
    a_by = ictal_by_channel
    per_event = []
    n_ev = masked.shape[1]
    for i in range(n_ev):
        part = np.where(bools[:, i])[0]
        if not (min_part <= part.size <= n_ch - headroom):
            continue
        pnames = [names[j] for j in part]
        if not all(n in a_by and np.isfinite(a_by[n]) for n in pnames):
            continue
        r = masked[part, i]; a = np.array([a_by[n] for n in pnames], float)
        if np.std(r) < 1e-12 or np.std(a) < 1e-12:
            continue
        # Spearman = Pearson on ranks; vectorize the null as rank-Pearson over a permutation
        # matrix (equivalent to per-perm spearmanr but ~100x faster). C8 sign-free; NO sign.
        rr = _rankdata(r); ra = _rankdata(a)
        rrc = rr - rr.mean(); rac = ra - ra.mean()
        denom = (np.linalg.norm(rrc) * np.linalg.norm(rac))
        if denom < 1e-12:
            continue
        obs = abs(float(rrc @ rac) / denom)
        P = np.array([rng.permutation(rac) for _ in range(n_perm)])   # (n_perm, k)
        null = np.abs(P @ rrc) / denom                                # (n_perm,)
        null_p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        per_event.append({"event_idx": int(bundle["valid_ev"][i]), "label": int(labels[i]),
                          "n_part": int(part.size), "align1d": float(obs), "null_p": null_p})
    return {"eligible": True, "n_channels": n_ch, "per_event": per_event,
            "n_usable": len(per_event),
            "usable_fraction": (len(per_event) / n_ev if n_ev else 0.0)}


# ----------------------------------------------------------------------------- R2: block label-shuffle
def class_separation_block_null(
    align0: Sequence[float],
    align1: Sequence[float],
    labels: Sequence[int],
    block_ids: Sequence[int],
    *,
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """R2 (spec §3.3, C7; P0 fix 2026-06-25): A-vs-B Δmedian + dispersion ratio vs a
    WITHIN-BLOCK constrained label-permutation null that PAYS the plane-selection cost.

    align0[i] / align1[i] = event i's alignment computed on class-0 / class-1's OWN plane
    (per_event_field_alignment now returns both). The assigned value uses the event's true
    label; the null permutes labels WITHIN each block (preserving each block's class counts —
    so event-level class sizes AND within-block A/B mixing are both preserved), and for each
    permuted label it RE-PICKS align0/align1 accordingly (so a shuffled event is scored on the
    plane of its shuffled class — the class-specific plane selection cost is present in the null,
    not just in the observed value). The earlier "collapse block to dominant label then shuffle
    whole blocks" null is REMOVED: it destroyed within-block mixing (blocks are mostly mixed)
    and let class sizes drift. block_ids REQUIRED (C7).
    """
    rng = rng or np.random.default_rng()
    a0 = np.asarray(align0, float); a1 = np.asarray(align1, float)
    lab = np.asarray(labels, int); blk = np.asarray(block_ids)
    fin = np.isfinite(a0) & np.isfinite(a1)              # need BOTH planes to re-pick under perm
    a0, a1, lab, blk = a0[fin], a1[fin], lab[fin], blk[fin]
    if a0.size == 0 or np.unique(lab).size < 2:
        return {"status": "insufficient", "n": int(a0.size)}

    def _stats(labvec):
        assigned = np.where(labvec == 0, a0, a1)
        A = assigned[labvec == 0]; B = assigned[labvec == 1]
        if A.size == 0 or B.size == 0:
            return np.nan, np.nan
        dmed = float(np.median(A) - np.median(B))
        iqrA = float(np.subtract(*np.percentile(A, [75, 25])))
        iqrB = float(np.subtract(*np.percentile(B, [75, 25])))
        return dmed, (float(iqrA / iqrB) if iqrB > 1e-12 else np.nan)

    obs_dmed, obs_dratio = _stats(lab)

    # within-block permutation indices (precompute block -> positions)
    ublk = np.unique(blk)
    block_pos = [np.where(blk == b)[0] for b in ublk]
    nd, nr = [], []
    for _ in range(n_perm):
        perm_lab = lab.copy()
        for pos in block_pos:
            if pos.size > 1:
                perm_lab[pos] = lab[pos][rng.permutation(pos.size)]   # preserves block class counts
        d, r = _stats(perm_lab)
        nd.append(d); nr.append(r)
    nd = np.array(nd, float); nr = np.array(nr, float)
    p_dmed = (float((np.sum(np.abs(nd) >= abs(obs_dmed)) + 1) / (n_perm + 1))
              if np.isfinite(obs_dmed) else None)
    finr = np.isfinite(nr) & (nr > 0)
    p_dratio = (float((np.sum(np.abs(np.log(nr[finr])) >= abs(np.log(obs_dratio))) + 1)
                      / (int(finr.sum()) + 1))
                if (np.isfinite(obs_dratio) and obs_dratio > 0) else None)

    # size-matched IQR ratio (down-sample larger class to n_min) on assigned values
    assigned = np.where(lab == 0, a0, a1)
    A = assigned[lab == 0]; B = assigned[lab == 1]; n_min = int(min(A.size, B.size))
    iqrA = np.subtract(*np.percentile(rng.choice(A, n_min, replace=False), [75, 25]))
    iqrB = np.subtract(*np.percentile(rng.choice(B, n_min, replace=False), [75, 25]))
    size_matched = float(iqrA / iqrB) if iqrB > 1e-12 else None

    return {"status": "ok", "n": int(a0.size), "n_blocks": int(ublk.size),
            "n_a": int((lab == 0).sum()), "n_b": int((lab == 1).sum()),
            "delta_median_obs": obs_dmed, "delta_median_null_p": p_dmed,
            "disp_ratio_obs": obs_dratio, "disp_ratio_null_p": p_dratio,
            "size_matched_iqr_ratio": size_matched}


# ----------------------------------------------------------------------------- diagnostics + stubs
def participation_diagnostics(bools: np.ndarray, labels: Sequence[int],
                              block_ids: Sequence[int]) -> dict:
    """Per-class participation distribution + n_blocks (C7 effective-N reporting)."""
    bools = np.asarray(bools).astype(bool)
    lab = np.asarray(labels, int)
    blk = np.asarray(block_ids)
    out = {}
    for g, name in ((0, "class_0"), (1, "class_1")):
        cols = np.where(lab == g)[0]
        if cols.size == 0:
            out[name] = {"n_events": 0}
            continue
        npart = bools[:, cols].sum(axis=0)
        out[name] = {
            "n_events": int(cols.size),
            "n_blocks": int(np.unique(blk[cols]).size),
            "median_n_part": float(np.median(npart)),
            "frac_ge5": float(np.mean(npart >= 5)),
            "frac_ge6": float(np.mean(npart >= 6)),
            "frac_ge10": float(np.mean(npart >= 10)),
        }
    return out


def stage_b_window_bias(*_a, **_k):
    """STUB (C10) — Stage B (A/B class bias across pre/post/background windows).
    Not built; gated behind the Stage-A pilot + advisor sign-off (spec §2 S2, §8.3)."""
    raise NotImplementedError("Stage B (window bias) not built — see spec §2 S2 / plan Phase deferred")


def stage_c_sequential_effects(*_a, **_k):
    """STUB (C10) — Stage C (sequential same/opposite-class reach/rank/rate).
    Not built; gated behind the Stage-A pilot + advisor sign-off (spec §2 S3, §8.3)."""
    raise NotImplementedError("Stage C (sequential effects) not built — see spec §2 S3 / plan Phase deferred")


# --------------------------------------------------------------- class-vs-template max-AB statistic
def field_from_contact_values(plane_record: dict, values_by_name: Dict[str, float],
                              *, support_by_name: Optional[Dict[str, float]] = None,
                              sigma: float, X, Y, s_thresh: float = S_THRESH) -> Optional[dict]:
    """Generic smoothed field from a name->value map on a class plane (template typical_rank,
    class-aggregate rank, or seizure activation). support_by_name overrides the plane's aggregate
    support (use it for the class field's participation weights); None -> plane aggregate support.
    """
    chans = []
    for c in plane_record["channels"]:
        nm = c["name"]
        v = values_by_name.get(nm)
        if v is None or not np.isfinite(v):
            continue
        if not (np.isfinite(c.get("x_norm", np.nan)) and np.isfinite(c.get("y_norm", np.nan))):
            continue
        sup = (support_by_name.get(nm, 0.0) if support_by_name is not None else c.get("support", 0.0))
        if sup is None or sup <= 0:
            continue
        chans.append({"name": nm, "x_norm": float(c["x_norm"]), "y_norm": float(c["y_norm"]),
                      "typical_rank": float(v), "support": float(sup)})
    if len(chans) < MIN_PLANE_CONTACTS:
        return None
    return R_smooth_rank({"channels": chans}, X, Y, sigma, s_thresh)


def _absmir(F1, F2, overlap_min, s_thresh):
    if F1 is None or F2 is None:
        return np.nan
    r = corr_pair_mirror_invariant(F1["T"], F1["S"], F2["T"], F2["S"], s_thresh, overlap_min)
    return abs(r["corr"]) if (not r["insufficient_overlap"] and r["corr"] is not None
                              and np.isfinite(r["corr"])) else np.nan


def maxab_alignment_vs_target(
    F_inter_a: dict, F_inter_b: dict, plane_a: dict, plane_b: dict,
    sigma_a: float, sigma_b: float, target_seizures: Sequence[Dict[str, float]],
    *, n_null: int, rng: np.random.Generator, X, Y,
    overlap_min: int = OVERLAP_MIN, s_thresh: float = S_THRESH,
) -> dict:
    """A-line "max_ab" statistic (per-seizure, selection-cost null) for ONE interictal
    representation (template OR class) vs a target window.

    Candidate = MAX over {A-field on plane_a vs target-on-plane_a, B-field on plane_b vs
    target-on-plane_b}. Per eligible seizure: max(|corr_A|, |corr_B|); real = median over
    seizures. Null: per seizure, channel-shuffle the target activation and recompute the SAME
    max-over-AB (so the selection cost is paid in the null too — handoff rule + spec §C8);
    n_null draws; null_p95 = 95th pct of median-over-seizures-per-draw.
    """
    real, null_per_sz = [], []
    for act in target_seizures:
        names = list(act.keys()); vals = np.array([act[n] for n in names], float)
        Fta = field_from_contact_values(plane_a, act, sigma=sigma_a, X=X, Y=Y, s_thresh=s_thresh)
        Ftb = field_from_contact_values(plane_b, act, sigma=sigma_b, X=X, Y=Y, s_thresh=s_thresh)
        aA = _absmir(F_inter_a, Fta, overlap_min, s_thresh)
        aB = _absmir(F_inter_b, Ftb, overlap_min, s_thresh)
        if not (np.isfinite(aA) or np.isfinite(aB)):
            continue
        real.append(np.nanmax([aA, aB]))
        draws = []
        for _ in range(n_null):
            sh = channel_shuffle(vals, rng)
            actsh = {n: float(v) for n, v in zip(names, sh)}
            Fa_s = field_from_contact_values(plane_a, actsh, sigma=sigma_a, X=X, Y=Y, s_thresh=s_thresh)
            Fb_s = field_from_contact_values(plane_b, actsh, sigma=sigma_b, X=X, Y=Y, s_thresh=s_thresh)
            draws.append(np.nanmax([_absmir(F_inter_a, Fa_s, overlap_min, s_thresh),
                                    _absmir(F_inter_b, Fb_s, overlap_min, s_thresh)]))
        null_per_sz.append(draws)
    if not real:
        return {"status": "no_resolvable_seizure", "n_seizures": 0}
    real_med = float(np.nanmedian(real))
    if null_per_sz:
        dist = np.nanmedian(np.asarray(null_per_sz, float), axis=0)   # median-over-seizures per draw
        null_p95 = float(np.nanpercentile(dist, 95)); null_med = float(np.nanmedian(dist))
    else:
        null_p95 = null_med = None
    return {"status": "ok", "n_seizures": len(real), "real_median_maxab": real_med,
            "channel_null_median": null_med, "channel_null_p95": null_p95,
            "pass_channel_null": (bool(real_med > null_p95) if null_p95 is not None else None)}


def maxab_two_reps_vs_target(
    reps: Dict[str, tuple], plane_a: dict, plane_b: dict, sigma_a: float, sigma_b: float,
    target_seizures: Sequence[Dict[str, float]], *, n_null: int, rng: np.random.Generator, X, Y,
    overlap_min: int = OVERLAP_MIN, s_thresh: float = S_THRESH,
) -> dict:
    """Same statistic as maxab_alignment_vs_target but for SEVERAL interictal representations at
    once (e.g. {"template":(F_tplA,F_tplB), "class":(F_clsA,F_clsB)}), building each target field
    (real + every null draw) ONCE and correlating against all reps. Halves the cost vs calling the
    single-rep function per representation, AND makes the per-rep nulls use the SAME shuffles (so a
    template-vs-class comparison is paired at the null level). Returns {rep_name: result_dict}.
    """
    real = {nm: [] for nm in reps}; null = {nm: [] for nm in reps}
    for act in target_seizures:
        names = list(act.keys()); vals = np.array([act[n] for n in names], float)
        Fta = field_from_contact_values(plane_a, act, sigma=sigma_a, X=X, Y=Y, s_thresh=s_thresh)
        Ftb = field_from_contact_values(plane_b, act, sigma=sigma_b, X=X, Y=Y, s_thresh=s_thresh)
        null_fields = []
        for _ in range(n_null):
            sh = channel_shuffle(vals, rng); actsh = {n: float(v) for n, v in zip(names, sh)}
            null_fields.append((field_from_contact_values(plane_a, actsh, sigma=sigma_a, X=X, Y=Y, s_thresh=s_thresh),
                                field_from_contact_values(plane_b, actsh, sigma=sigma_b, X=X, Y=Y, s_thresh=s_thresh)))
        for nm, (FA, FB) in reps.items():
            m = np.nanmax([_absmir(FA, Fta, overlap_min, s_thresh), _absmir(FB, Ftb, overlap_min, s_thresh)])
            if not np.isfinite(m):
                continue
            real[nm].append(m)
            null[nm].append([np.nanmax([_absmir(FA, fa, overlap_min, s_thresh),
                                        _absmir(FB, fb, overlap_min, s_thresh)]) for fa, fb in null_fields])
    out = {}
    for nm in reps:
        if not real[nm]:
            out[nm] = {"status": "no_resolvable_seizure", "n_seizures": 0}; continue
        rm = float(np.nanmedian(real[nm]))
        if null[nm]:
            dist = np.nanmedian(np.asarray(null[nm], float), axis=0)
            p95 = float(np.nanpercentile(dist, 95)); md = float(np.nanmedian(dist))
        else:
            p95 = md = None
        out[nm] = {"status": "ok", "n_seizures": len(real[nm]), "real_median_maxab": rm,
                   "channel_null_median": md, "channel_null_p95": p95,
                   "pass_channel_null": (bool(rm > p95) if p95 is not None else None)}
    return out
