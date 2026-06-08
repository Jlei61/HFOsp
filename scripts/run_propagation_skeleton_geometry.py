#!/usr/bin/env python3
"""Per-subject interictal propagation skeleton geometry (descriptive model-input).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md
Outputs: results/topic4_sef_hfo/skeleton_geometry/{per_subject/{ds}_{subj}.json, cohort_summary.json}
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

from sklearn.cluster import KMeans

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks, build_masked_kmeans_features
from src.low_rate_template_stability import align_template_events
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
    labels = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(
        build_masked_kmeans_features(ranks, bools))

    # Single event-set/frame feeds axis + stereotypy + count (spec: no anti-parallel
    # "糊平均"). For swap subjects the dominant cluster's OWN frame is used — do NOT
    # global-align (that re-merges the anti-parallel clusters and washes the excess).
    swap_class = _swap_class(ds, subj)
    if swap_class in ("strict", "candidate"):
        dom = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
        sel = labels == dom
        ster_masked = masked[:, sel]
        ster_bools = bools[:, sel]
        template_source = f"dominant_cluster_{dom}"
    else:
        aligned, _ = align_template_events(masked, labels)
        ster_masked = aligned
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
    return rec


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
        _ROOT / "results/topic4_sef_hfo/skeleton_geometry"))
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
        except Exception as e:  # noqa: BLE001 — record, don't crash the cohort
            status = f"error: {e}"
            rec = {"dataset": ds, "subject": subj, "status": status,
                   "error_category": _error_category(status)}
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
