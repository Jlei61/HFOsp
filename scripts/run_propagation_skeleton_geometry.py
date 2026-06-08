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


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _swap_class(ds, subj):
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        return "none"
    d = json.loads(f.read_text())
    pairs = d.get("pairs") or [{}]
    return (((pairs[0].get("swap_sweep") or {}).get("swap_class")) or "none")


def _cluster_axis(masked, labels, cluster):
    sel = labels == cluster
    sub = masked[:, sel]
    return np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in sub])


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
    aligned, _ = align_template_events(masked, labels)

    swap_class = _swap_class(ds, subj)
    if swap_class in ("strict", "candidate"):
        dom = 0 if (labels == 0).sum() >= (labels == 1).sum() else 1
        template_axis = _cluster_axis(masked, labels, dom)
        template_source = f"dominant_cluster_{dom}"
    else:
        template_axis = np.array(
            [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])
        template_source = "full_recording"

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

    full_count = bools.sum(axis=1).astype(float)
    comps = G.channel_stereotypy_components(masked, bools, rng=np.random.default_rng(0))
    excess = comps["excess"]
    samp = G.classify_sampling_geometry(
        names, eligible, fr["off_axis"],
        spacing_mm=3.5 if ds == "yuquan" else 4.6)
    # Right-exclusive profile bins (a < hi) would drop a sink-core channel at
    # exactly along==L; bump the top edge just past L so the final bin includes it.
    edges = list(np.linspace(0.0, max(fr["axis_length"], 1e-6), 5))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    along = np.asarray(fr["along_axis"], dtype=float)
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

    recs = []
    for ds, subj in cohort:
        try:
            rec = process_subject(ds, subj)
        except Exception as e:  # noqa: BLE001 — record, don't crash the cohort
            rec = {"dataset": ds, "subject": subj, "status": f"error: {e}"}
        (out / "per_subject" / f"{ds}_{subj}.json").write_text(
            json.dumps(rec, indent=2, default=float))
        recs.append(rec)

    ok = [r for r in recs if r.get("status") == "ok"]
    tiers = {}
    for r in ok:
        tiers[r["eligibility_tier"]] = tiers.get(r["eligibility_tier"], 0) + 1
    by_dataset = {}
    for r in ok:
        d = by_dataset.setdefault(r["dataset"], {"n": 0, "axis_length_mm": []})
        d["n"] += 1
        if r.get("eligibility_tier") in ("primary", "fallback") and "axis_length_mm" in r:
            d["axis_length_mm"].append(r["axis_length_mm"])
    cohort_summary = {
        "n_processed": len(recs), "n_ok": len(ok),
        "phantom_core_violations":
            int(sum(bool(r.get("phantom_core_violation")) for r in ok)),
        "tiers": tiers,
        "sampling_geometry": {
            g: int(sum(r.get("sampling_geometry", {}).get("geometry") == g for r in ok))
            for g in ("1D", "distributed")},
        "swap_tiers": {
            sc: int(sum(r.get("swap_class") == sc for r in ok))
            for sc in ("none", "candidate", "strict")},
        "by_dataset": by_dataset,
    }
    (out / "cohort_summary.json").write_text(json.dumps(cohort_summary, indent=2, default=float))
    print(json.dumps(cohort_summary, indent=2, default=float))


if __name__ == "__main__":
    main()
