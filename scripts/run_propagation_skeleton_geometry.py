#!/usr/bin/env python3
"""Per-subject interictal propagation skeleton geometry (descriptive model-input).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md
Outputs: results/spatial_modulation/propagation_geometry/components/path_axis/{per_subject/{ds}_{subj}.json, cohort_summary.json}
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src import propagation_skeleton_geometry as G

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RANKDISP = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
SOZ_JSON = {ds: _ROOT / f"results/{ds}_soz_core_channels.json" for ds in ("yuquan", "epilepsiae")}
ALL_SUBJECTS = None  # default cohort discovered from RANKDISP + coord availability

# Bad-data exclusions (spec 2026-06-08 §2 + memory project_topic4_soz_localization_plan:
# pengzihang total_hours=2.0, gpu=1 — not a valid recording). Recorded, not silently dropped.
EXCLUDE_BAD_DATA = {("yuquan", "pengzihang")}


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


# Coord-loader failure fingerprints (yuquan chnXyzDict, epilepsiae SQL/MRI).
# A known coord miss is benign (subject lacks coords); anything else is a real
# bug (loader misuse, contract violation) and must NOT be swept up as benign.
_COORD_MISS_MARKERS = (
    "coord file not found", "chnXyzDict",
    "SQL not found", "MRI not found", "coords missing",
)


# Fix B: only these error categories may be recorded-and-continued. Everything
# else re-raises and fails the whole run loudly (the 9 yuquan chnXyzDict misses
# are the sole expected benign category).
WHITELIST_ERROR_CATEGORIES = {"no_coord_file"}


def _error_category(status):
    s = str(status).lower()
    if any(m.lower() in s for m in _COORD_MISS_MARKERS):
        return "no_coord_file"
    return "unexpected"


def _swap_class(ds, subj):
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        return "none"
    d = json.loads(f.read_text())
    pairs = d.get("pairs") or [{}]
    return (((pairs[0].get("swap_sweep") or {}).get("swap_class")) or "none")


def _load_accepted_templates(ds, subj, names):
    """Load the two ACCEPTED cluster templates (rank-displacement pair[0]) and
    align them to the propagation-events `names` ordering by NAME.

    Returns (template_a, template_b, swap_class). Each template is a length-len(names)
    float array, NaN where that channel is absent from the rank-displacement pair
    (NaN-at-non-participating, so it never enters a phantom core). A missing
    rank-displacement JSON is an UNEXPECTED error (Fix A): the runner must NOT
    silently fall back to fresh KMeans — raise loudly.
    """
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        raise FileNotFoundError(
            f"rank-displacement JSON missing for {ds}:{subj} ({f}); accepted "
            "cluster templates are required (no KMeans fallback)")
    d = json.loads(f.read_text())
    pair = (d.get("pairs") or [{}])[0]
    rd_names = list(pair.get("channel_names") or [])
    ra = np.asarray(pair.get("rank_a_dense_full"), float)
    rb = np.asarray(pair.get("rank_b_dense_full"), float)
    if rd_names == [] or ra.size != len(rd_names) or rb.size != len(rd_names):
        raise ValueError(
            f"rank-displacement templates malformed for {ds}:{subj}: "
            f"n_names={len(rd_names)} ra={ra.size} rb={rb.size}")
    # Map rank-displacement (name -> template value) onto the propagation ordering;
    # NaN where a propagation channel is absent from the rank-displacement pair.
    idx_a = {nm: ra[i] for i, nm in enumerate(rd_names)}
    idx_b = {nm: rb[i] for i, nm in enumerate(rd_names)}
    template_a = np.array([idx_a.get(nm, np.nan) for nm in names], float)
    template_b = np.array([idx_b.get(nm, np.nan) for nm in names], float)
    swap_class = (((pair.get("swap_sweep") or {}).get("swap_class")) or "none")
    return template_a, template_b, swap_class


def _soz_set(ds, subj):
    """Descriptive-only SOZ core channel set for one subject.

    Real schema (verified 2026-06-08): both SOZ JSONs are
    {subject_id_str: [chan_names]} (epilepsiae keys numeric str e.g. '253',
    yuquan keys subject name e.g. 'chengshuai'). Older dict-valued schema
    {subject: {core_channels: [...]}} is also tolerated. NEVER crashes the
    subject — any miss returns set() (soz_relation is purely descriptive).
    """
    try:
        d = json.loads(SOZ_JSON[ds].read_text())
        entry = d.get(subj, d.get(str(subj)))
        if isinstance(entry, dict):
            entry = entry.get("core_channels", [])
        return set(entry or [])
    except Exception:
        return set()


def process_subject(ds, subj):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    if not ev["channel_names"] or np.asarray(ev["ranks"]).size == 0:
        return {"dataset": ds, "subject": subj, "status": "no_events"}
    names = list(ev["channel_names"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)

    # Fix A: reuse the two ACCEPTED cluster templates (rank-displacement pair[0]),
    # NOT a fresh KMeans. Assign every event to its nearest accepted template by
    # Spearman correlation; no re-clustering, no global re-alignment "糊平均".
    template_a, template_b, swap_class = _load_accepted_templates(ds, subj, names)
    labels = G.assign_events_to_templates(masked, template_a, template_b)
    clustering_provenance = "accepted_rankdisp_templates"

    # Swap-positive: the dominant accepted cluster's OWN frame feeds axis +
    # stereotypy (the two templates are anti-parallel; averaging them washes the
    # excess). Non-swap: clusters are not anti-parallel, so the full recording is
    # used directly (no flipping needed).
    minority_masked = minority_bools = None
    if swap_class in ("strict", "candidate"):
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        dom = 0 if n0 >= n1 else 1
        sel = labels == dom
        ster_masked = masked[:, sel]
        ster_bools = bools[:, sel]
        template_source = "dominant_cluster"
        minority = 1 - dom
        msel = labels == minority
        if msel.any():
            minority_masked = masked[:, msel]
            minority_bools = bools[:, msel]
    else:
        ster_masked = masked
        ster_bools = bools
        template_source = "full_recording"
    template_axis = np.array(
        [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in ster_masked])
    full_count = ster_bools.sum(axis=1).astype(float)
    comps = G.channel_stereotypy_components(ster_masked, ster_bools,
                                            rng=np.random.default_rng(0))

    cr = load_subject_coords(ds, subj, names)
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    eligible = (~np.isnan(template_axis)) & mapped

    cores = G.build_endpoint_cores(template_axis, eligible, k_primary=3)
    rec = {
        "dataset": ds, "subject": subj, "status": "ok",
        "swap_class": swap_class, "template_source": template_source,
        "clustering_provenance": clustering_provenance,
        "coord_space": cr.coord_space, "n_eff": cores["n_eff"],
        "k_used": cores["k_used"], "eligibility_tier": cores["tier"],
        "channel_names": names,
        "missing_coords": [m.channel for m in cr.missing],
    }
    # phantom-safe core assertion (acceptance §9.1)
    rec["phantom_core_violation"] = bool(
        any(not eligible[i] for i in cores["source_idx"] + cores["sink_idx"]))

    if cores["tier"] == "descriptive_only":
        return rec

    fr = G.compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    rec["degenerate_axis"] = fr["degenerate_axis"]
    if fr["degenerate_axis"]:
        rec["eligibility_tier"] = "descriptive_only"
        return rec

    excess = comps["excess"]
    samp = G.classify_sampling_geometry(
        names, eligible, fr["off_axis"],
        spacing_mm=3.5 if ds == "yuquan" else 4.6)
    along = np.asarray(fr["along_axis"], dtype=float)
    # Span the FULL along range so the profile shows a<0 (before source) and a>L
    # (beyond sink) — spec §5 wants endpoints tested as hard boundaries, not assumed.
    # Right-exclusive bins (a < hi) would drop the channel at exactly along==hi;
    # bump the top edge just past it so the final bin includes it.
    amin = float(np.nanmin(along)) if np.any(~np.isnan(along)) else 0.0
    amax = float(np.nanmax(along)) if np.any(~np.isnan(along)) else max(fr["axis_length"], 1e-6)
    lo = min(0.0, amin)
    hi = max(fr["axis_length"], amax)
    edges = list(np.linspace(lo, hi, 6))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    along_profile = G.axis_stereotypy_profile(along, excess, edges=edges)
    # Per-bin mean event count over the SAME channels whose excess was averaged
    # (axis_stereotypy_profile masks on ~isnan(along) & ~isnan(excess)). Lets a
    # reader confirm firing rate is roughly flat across along-axis bins (control).
    bin_ok = ~np.isnan(along) & ~np.isnan(np.asarray(excess, dtype=float))
    for b in along_profile:
        sel = bin_ok & (along >= b["a_lo"]) & (along < b["a_hi"])
        cnts = full_count[sel]
        b["mean_event_count"] = float(cnts.mean()) if cnts.size else float("nan")
    rec["along_axis_profile_metric"] = "raw_excess_obs_minus_nullmean"
    # Per-channel record so the plotter is self-contained (no recompute). One
    # entry per ELIGIBLE channel (participating AND coord-mapped); channels with
    # NaN along/off (out of frame, e.g. NaN coords) are skipped. NO new geometry
    # — these are values already computed above (fr along/off, comps excess).
    source_set = set(cores["source_idx"])
    sink_set = set(cores["sink_idx"])
    excess_arr = np.asarray(excess, dtype=float)
    channels = []
    for i in np.where(eligible)[0]:
        a_i = float(along[i])
        o_i = float(fr["off_axis"][i])
        if np.isnan(a_i) or np.isnan(o_i):
            continue
        if i in source_set:
            role = "source_core"
        elif i in sink_set:
            role = "sink_core"
        else:
            role = "interior"
        channels.append({
            "name": str(names[i]),
            "along_axis_mm": a_i,
            "off_axis_mm": o_i,
            "stereotypy_excess": float(excess_arr[i]),
            "role": role,
            "shaft": str(G.parse_shaft(names[i])[0]),
        })
    rec["channels"] = channels
    # Modality tag (inferred from channel naming): a channel is "grid" if its
    # parsed shaft starts with "G" — but ONLY for Epilepsiae (grid naming
    # GA1/GB3...). Yuquan is all depth by recording convention; its bare-"G"
    # shafts are single-letter DEPTH labels, not grids, so the heuristic is
    # dataset-gated to avoid a false-positive grid tag. Infer over the
    # PARTICIPATING set (broadest-correct: a grid channel that participates but
    # is unmapped/out-of-frame would be missed if we used only the in-frame
    # `channels` list, under-counting depth+grid). This 3-layer breakdown makes
    # the SEEG-vs-mixed confound visible at cohort level.
    participating = bools.any(axis=1)
    has_grid = ds == "epilepsiae" and any(
        (G.parse_shaft(names[i])[0] or "").startswith("G")
        for i in np.where(participating)[0])
    rec["modality"] = "depth+grid" if has_grid else "depth_only"
    rec["modality_inferred_from"] = "channel_naming"
    # Weak-axis flag: when a sink-core channel projects to a SMALLER along-axis
    # position than some source-core channel, the cores interleave along the axis
    # -> source/sink centroids collapse toward each other -> axis_length cancels
    # and is unreliable. This is the real split-core detector; degenerate_axis
    # (L<1e-9) does NOT catch it. Downstream model code should gate on this.
    src_along = [c["along_axis_mm"] for c in channels if c["role"] == "source_core"]
    snk_along = [c["along_axis_mm"] for c in channels if c["role"] == "sink_core"]
    weak_axis = bool(src_along and snk_along and (min(snk_along) < max(src_along)))
    rec["weak_axis"] = weak_axis
    rec.update({
        "source_radius": G.core_radii(coords[cores["source_idx"]],
                                      np.array(fr["source_centroid"])),
        "sink_radius": G.core_radii(coords[cores["sink_idx"]],
                                    np.array(fr["sink_centroid"])),
        "axis_length_mm": fr["axis_length"],
        "perp_spread": G.perp_spread(fr["off_axis"], eligible),
        "perp_spread_participation_sweep":
            G.perp_spread_participation_sweep(fr["off_axis"], full_count),
        "sampling_geometry": samp,
        "perp_width_measurable": samp["measurable"],
        "along_axis_profile": along_profile,
        "soz_relation": {
            "source_core": [names[i] for i in cores["source_idx"]],
            "sink_core": [names[i] for i in cores["sink_idx"]],
            "soz_channels_in_montage":
                sorted(_soz_set(ds, subj) & set(names)),
        },
    })

    # Fix C: per-cluster (multi-template) geometry for swap subjects. Build the
    # MINORITY accepted cluster's own source/sink cores + axis, then report the
    # cosine between the dominant and minority axis unit vectors. Descriptive only
    # (no significance): cos ~ -1 -> same spatial axis, opposite directions;
    # cos ~ 0/positive -> divergent paths. NEVER raises (Fix B): a minority cluster
    # too small/interleaved -> minority_axis=None, axes_cos_angle=None.
    rec["minority_axis"] = None
    rec["axes_cos_angle"] = None
    if minority_masked is not None and minority_masked.shape[1] >= 1:
        min_axis_vec = _cluster_axis(coords, minority_masked, eligible)
        if min_axis_vec is not None:
            m_src_c, m_snk_c, m_len = min_axis_vec
            rec["minority_axis"] = {
                "axis_length_mm": m_len,
                "source_centroid": m_src_c.tolist(),
                "sink_centroid": m_snk_c.tolist(),
            }
            dom_axis = np.array(fr["sink_centroid"]) - np.array(fr["source_centroid"])
            min_axis = m_snk_c - m_src_c
            dn = np.linalg.norm(dom_axis)
            mn = np.linalg.norm(min_axis)
            if dn > 1e-9 and mn > 1e-9:
                rec["axes_cos_angle"] = float(
                    np.dot(dom_axis, min_axis) / (dn * mn))

    # Fix D: split-half path validation (anti-tautology) + per-core radius null.
    # Fixed seed (np.random.default_rng(0)) for reproducibility. eligible_idx is
    # the eligible-channel index set; k matches the tier's core size.
    eligible_idx = np.where(eligible)[0]
    rng = np.random.default_rng(0)
    rec["split_half_validation"] = G.split_half_axis_validation(
        ster_masked, coords, eligible_idx, k=cores["k_used"],
        rng=rng, n_boot=200)
    rec["source_radius_null"] = G.core_radius_null(
        coords, eligible_idx, k=cores["k_used"],
        observed_radius_rms=rec["source_radius"]["rms_mm"],
        n_null=2000, rng=rng)
    rec["sink_radius_null"] = G.core_radius_null(
        coords, eligible_idx, k=cores["k_used"],
        observed_radius_rms=rec["sink_radius"]["rms_mm"],
        n_null=2000, rng=rng)
    return rec


def _cluster_axis(coords, cluster_masked, eligible):
    """Source/sink cores + axis for one cluster's per-channel mean rank. Returns
    (source_centroid, sink_centroid, axis_length_mm) or None if the cluster can't
    form a valid (non-degenerate, non-descriptive) frame. NEVER raises."""
    n_ch = coords.shape[0]
    with np.errstate(invalid="ignore"):
        axis = np.array([
            np.nanmean(cluster_masked[c]) if np.any(~np.isnan(cluster_masked[c]))
            else np.nan for c in range(n_ch)])
    elig = np.asarray(eligible, bool) & ~np.isnan(axis)
    cores = G.build_endpoint_cores(axis, elig, k_primary=3)
    if cores["tier"] == "descriptive_only" or not cores["source_idx"] or not cores["sink_idx"]:
        return None
    try:
        fr = G.compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    except ValueError:
        return None
    if fr["degenerate_axis"]:
        return None
    return (np.array(fr["source_centroid"]), np.array(fr["sink_centroid"]),
            fr["axis_length"])


def discover_cohort():
    subs = []
    for f in sorted(RANKDISP.glob("*.json")):
        ds, subj = f.stem.split("_", 1)
        subs.append((ds, subj))
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="ds:subj tokens; default = full cohort")
    ap.add_argument("--out", default=str(
        _ROOT / "results/spatial_modulation/propagation_geometry/components/path_axis"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "per_subject").mkdir(parents=True, exist_ok=True)

    if args.subjects:
        cohort = [tuple(s.split(":", 1)) for s in args.subjects]
    else:
        cohort = discover_cohort()

    # Bad-data exclusions apply to BOTH branches (spec §2). Excluded subjects are
    # recorded by name with provenance; they are NOT processed and write NO record.
    excluded = [(ds, subj) for ds, subj in cohort if (ds, subj) in EXCLUDE_BAD_DATA]
    cohort = [(ds, subj) for ds, subj in cohort if (ds, subj) not in EXCLUDE_BAD_DATA]
    for ds, subj in excluded:
        print(f"  [exclude bad-data] {ds}:{subj}", flush=True)

    recs = []
    for ds, subj in cohort:
        try:
            rec = process_subject(ds, subj)
        except Exception as e:  # noqa: BLE001
            status = f"error: {e}"
            category = _error_category(status)
            # Fix B: only KNOWN-benign coord misses are recorded-and-continued.
            # Anything else (loader misuse, missing rank-disp JSON, contract
            # violation) is a systemic bug — re-raise loudly rather than hide it
            # under a cohort summary built on a silent failure.
            if category not in WHITELIST_ERROR_CATEGORIES:
                print(f"  [FATAL {ds}:{subj}] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                raise
            rec = {"dataset": ds, "subject": subj, "status": status,
                   "error_category": category}
            print(f"  [error {ds}:{subj}] {type(e).__name__}: {e}", flush=True)
        (out / "per_subject" / f"{ds}_{subj}.json").write_text(
            json.dumps(rec, indent=2, default=float))
        recs.append(rec)

    ok = [r for r in recs if r.get("status") == "ok"]
    errored = [r for r in recs if str(r.get("status", "")).startswith("error")]
    tiers = {}
    for r in ok:
        tiers[r["eligibility_tier"]] = tiers.get(r["eligibility_tier"], 0) + 1
    by_dataset = {}
    for r in ok:
        d = by_dataset.setdefault(r["dataset"], {"n": 0, "axis_length_mm": []})
        d["n"] += 1
        if r.get("eligibility_tier") in ("primary", "fallback") and "axis_length_mm" in r:
            d["axis_length_mm"].append(r["axis_length_mm"])
    weak_recs = [r for r in ok if r.get("weak_axis")]
    # Error-category tally + vacuous-green guard (refuse a summary built on zero
    # successes; never silently swallow a loader-misuse "unexpected" error).
    error_categories = {}
    for r in errored:
        cat = r.get("error_category") or _error_category(r.get("status", ""))
        error_categories[cat] = error_categories.get(cat, 0) + 1
    n_ok = len(ok)
    if n_ok == 0:
        raise SystemExit(
            "all subjects failed — systemic error, refusing to write a vacuous "
            "cohort summary")
    n_unexpected = error_categories.get("unexpected", 0)
    if n_unexpected:
        print(f"WARNING: {n_unexpected} unexpected errors", file=sys.stderr,
              flush=True)
    # Modality 3-layer breakdown over ok primary/fallback subjects (modality is
    # only set on the full-record path). Makes the SEEG-vs-mixed confound visible.
    modality = {"yuquan_depth_only": 0, "epilepsiae_depth_only": 0,
                "epilepsiae_depth+grid": 0}
    n_tagged = 0
    for r in ok:
        m = r.get("modality")
        if m is None:
            continue
        n_tagged += 1
        key = f"{r['dataset']}_{m}"
        if key in modality:
            modality[key] += 1
    # Exhaustiveness: every modality-tagged ok rec must land in a bucket. A new
    # (dataset, modality) combo with no bucket would otherwise be silently
    # dropped (the bug that surfaced yuquan bare-"G" depth shafts as grid).
    if sum(modality.values()) != n_tagged:
        raise SystemExit(
            f"modality breakdown drops subjects: sum(buckets)="
            f"{sum(modality.values())} != n_tagged={n_tagged}")
    cohort_summary = {
        "n_processed": len(recs), "n_ok": n_ok,
        "n_error": len(errored),
        "errors": [{"dataset": r["dataset"], "subject": r["subject"],
                    "status": r["status"],
                    "error_category": r.get("error_category")} for r in errored],
        "error_categories": error_categories,
        "excluded_bad_data": [f"{ds}:{subj}" for ds, subj in excluded],
        "phantom_core_violations":
            int(sum(bool(r.get("phantom_core_violation")) for r in ok)),
        "weak_axis": len(weak_recs),
        "weak_axis_subjects": [f"{r['dataset']}:{r['subject']}" for r in weak_recs],
        "tiers": tiers,
        "modality": modality,
        "sampling_geometry": {
            g: int(sum(r.get("sampling_geometry", {}).get("geometry") == g for r in ok))
            for g in ("1D", "distributed", "shaft_parse_uncertain")},
        "swap_tiers": {
            sc: int(sum(r.get("swap_class") == sc for r in ok))
            for sc in ("none", "candidate", "strict")},
        "by_dataset": by_dataset,
    }
    (out / "cohort_summary.json").write_text(json.dumps(cohort_summary, indent=2, default=float))
    print(json.dumps(cohort_summary, indent=2, default=float))


if __name__ == "__main__":
    main()
