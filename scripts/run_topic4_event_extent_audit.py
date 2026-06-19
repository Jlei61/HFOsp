#!/usr/bin/env python3
"""Task 0 (M2 data-side audit) runner — do real interictal HFO group events axially
self-limit (cover only a SEGMENT of the propagable axis) or merely run laterally narrow
along it? Gates whether the M2 shunting/ahead-gate model work is even the right target.

PINNED DATA CONTRACT (clauses verified before writing this body; see plan Task 0):
  [c1] broad source = ONLY results/lagpat_broad_epilepsiae/<subj> (epi) +
       results/lagpat_broad/<subj> (yuquan); _k00/_k05/_km10/_topn40/_dyn EXCLUDED.
  [c2] events via src.interictal_propagation.load_subject_propagation_events (union
       channel_names + bools[n_ch,n_events]); NO hand-parsing of the NPZ.
  [c3] coords via src.seeg_coord_loader.load_subject_coords(ds, subj, names) ->
       coords_array_in_requested_order / mapped_mask_in_requested_order / coord_space.
  [c4] axis = the ACCEPTED reproducible axis: source_core/sink_core names from
       skeleton_geometry/per_subject/<ds>_<subj>.json; build the frame by reusing
       compute_axis_frame(broad_coords, source_idx, sink_idx) (project broad channels onto
       the accepted source/sink centroids — NOT a re-derived axis on the broad pool).
  [c5] eligibility spine = skeleton status=='ok' AND not degenerate_axis AND finite
       axis_length_mm AND source_core|sink_core subset of broad channel_names.
  [c6] per-event exclusions tallied: empty block (loader), n_part<5 (low_n; 3-4 sensitivity),
       NaN-coord channels (~mapped_mask) dropped from projection.
  [c7] 3-mode matched null (uniform always; rate=participation fraction; shaft_matched via
       event_shaft_counts), n_draw>=200; primary = shaft_matched (most conservative).
  [c8] per-subject diagnostics columns: n_events_total/used/excluded, excluded_reason
       breakdown, n_channels_in_pool/mapped, coord_space, template_source.
  [c9] fail-loud asserts: coords.shape==(len(names),3); source_core|sink_core subset of names
       (else core_unmapped — a real contract break, recorded, NOT a benign skip).
  Verdict via cohort_verdict (pre-registered Step-9 thresholds); INCONCLUSIVE is the default.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.interictal_propagation import load_subject_propagation_events  # [c2]
from src.propagation_skeleton_geometry import compute_axis_frame, parse_shaft  # [c4]
from src.seeg_coord_loader import load_subject_coords  # [c3]
from src.topic4_event_extent_audit import (  # pure metrics (TDD'd)
    cohort_verdict,
    event_extent,
    event_shaft_counts,
    matched_null_extent,
)

SKELETON = _ROOT / "results/topic4_sef_hfo/skeleton_geometry/per_subject"
BROAD = {  # [c1] canonical broad pool ONLY
    "epilepsiae": _ROOT / "results/lagpat_broad_epilepsiae",
    "yuquan": _ROOT / "results/lagpat_broad",
}
OUT = _ROOT / "results/topic4_sef_hfo/event_extent_audit"
MIN_N_PART = 5            # [c6] p5/p95 extent needs >=5 points
N_DRAW = 200             # [c7]
N_EVENT_CAP = 400        # bound runtime; deterministic subsample (recorded, no silent cap)


def _cores(skeleton_json):
    """[c4] accepted source_core / sink_core channel names from the skeleton card."""
    chans = skeleton_json.get("channels", [])
    src = [c["name"] for c in chans if c.get("role") == "source_core"]
    snk = [c["name"] for c in chans if c.get("role") == "sink_core"]
    return src, snk


def _eligible_subjects():
    """[c5] eligibility spine: status ok, real axis, both cores present in the broad pool."""
    subjects = []
    for f in sorted(SKELETON.glob("*.json")):
        d = json.loads(f.read_text())
        ds = d.get("dataset")
        subj = str(d.get("subject"))
        if d.get("status") != "ok":
            continue
        if d.get("degenerate_axis"):
            continue
        axl = d.get("axis_length_mm")
        if axl is None or not np.isfinite(axl):
            continue
        subjects.append((ds, subj, d))
    return subjects


def _audit_subject(ds, subj, card, rng):
    rec = dict(dataset=ds, subject=subj, template_source=card.get("template_source"),
               coord_space=None, n_events_total=0, n_events_used=0, n_events_excluded=0,
               n_low_n=0, n_channels_in_pool=0, n_channels_mapped=0,
               excluded_reason=None, axial_obs=np.nan, axial_null=np.nan,
               lateral_obs=np.nan, lateral_null=np.nan, n_events_subsampled=0)

    sdir = BROAD[ds] / subj
    if not sdir.exists():
        rec["excluded_reason"] = "no_broad_dir"
        return rec, []
    try:
        ev = load_subject_propagation_events(sdir)  # [c2]
    except FileNotFoundError:
        rec["excluded_reason"] = "no_events"
        return rec, []
    names = list(ev["channel_names"])
    bools = np.asarray(ev["bools"], bool)  # (n_ch, n_events)
    rec["n_channels_in_pool"] = len(names)
    if bools.size == 0 or bools.shape[1] == 0:
        rec["excluded_reason"] = "empty_block"
        return rec, []
    rec["n_events_total"] = int(bools.shape[1])

    cr = load_subject_coords(ds, subj, names)  # [c3]
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    assert coords.shape == (len(names), 3), f"[c9] coords shape {coords.shape} != ({len(names)},3)"  # [c9]
    rec["coord_space"] = cr.coord_space
    rec["n_channels_mapped"] = int(mapped.sum())

    src_names, snk_names = _cores(card)  # [c4]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    missing_core = [c for c in (src_names + snk_names) if c not in name_to_idx]
    if missing_core:  # [c9] a core not in the broad pool is a real break, NOT a benign skip
        rec["excluded_reason"] = "core_unmapped"
        return rec, []
    source_idx = [name_to_idx[c] for c in src_names]
    sink_idx = [name_to_idx[c] for c in snk_names]

    fr = compute_axis_frame(coords, source_idx, sink_idx)  # [c4] reuse, don't re-derive
    if fr.get("degenerate_axis"):
        rec["excluded_reason"] = "degenerate_axis"
        return rec, []
    along = np.asarray(fr["along_axis"], float)
    off = np.asarray(fr["off_axis"], float)
    axis_length = float(fr["axis_length"])

    # [c6] eligible = coord-mapped + in-frame + recruited (participates in >=1 event). The
    # recruited territory defines BOTH the null pool and the axial denominator, so axial
    # coverage is normalized by what the propagation can actually reach (NOT the core-to-core
    # distance — broad events recruit far past the two endpoint cores, which sends AF past 1).
    participates_ever = bools.any(axis=1)
    eligible = mapped & ~np.isnan(along) & ~np.isnan(off) & participates_ever
    elig_idx = np.where(eligible)[0]
    if len(elig_idx) < MIN_N_PART + 1:
        rec["excluded_reason"] = "too_few_eligible_channels"
        return rec, []
    along_e = along[elig_idx]
    off_e = off[elig_idx]
    rate_e = bools[elig_idx].mean(axis=1)  # [c7] per-channel participation fraction
    shaft_e = np.array([parse_shaft(names[i])[0] for i in elig_idx], object)
    # axial extent of the recruited territory = the denominator for axial_fraction (in [0,1])
    axial_available = float(np.percentile(along_e, 95) - np.percentile(along_e, 5))
    axial_available = max(axial_available, 1e-9)
    rec["axis_core_mm"] = axis_length
    rec["axial_available_mm"] = axial_available

    # [c6] subsample events for runtime (recorded); per-subject median is robust at 400
    n_ev = bools.shape[1]
    ev_order = np.arange(n_ev)
    if n_ev > N_EVENT_CAP:
        ev_order = np.sort(rng.choice(n_ev, size=N_EVENT_CAP, replace=False))
        rec["n_events_subsampled"] = int(n_ev - N_EVENT_CAP)

    reasons = Counter()
    obs_af, obs_lr, null_af, null_lr = [], [], [], []
    per_event = []
    for e in ev_order:
        part_idx = np.where(bools[:, e] & eligible)[0]
        n_part = len(part_idx)
        if n_part < MIN_N_PART:
            reasons["low_n_part"] += 1
            if 3 <= n_part <= 4:
                rec["n_low_n"] += 1
            continue
        ee = event_extent(along[part_idx], off[part_idx], axial_available)
        scnt = event_shaft_counts([names[i] for i in part_idx])  # [c7]
        null = matched_null_extent(along_e, off_e, n_part, axial_available, N_DRAW, rng,
                                   shaft=shaft_e, shaft_counts=scnt, rate=rate_e)  # [c7]
        obs_af.append(ee["axial_fraction"])
        obs_lr.append(ee["lateral_ratio"])
        sm = null.get("shaft_matched", null["uniform"])  # primary = most conservative
        null_af.append(sm["axial_fraction_med"])
        null_lr.append(sm["lateral_ratio_med"])
        per_event.append(dict(
            dataset=ds, subject=subj, event=int(e), n_part=int(n_part),
            axial_fraction=ee["axial_fraction"], lateral_ratio=ee["lateral_ratio"],
            null_shaft_axial=sm["axial_fraction_med"], null_shaft_lateral=sm["lateral_ratio_med"],
            null_uniform_axial=null["uniform"]["axial_fraction_med"],
            null_rate_axial=null.get("rate", {}).get("axial_fraction_med", np.nan)))

    rec["n_events_used"] = len(obs_af)
    rec["n_events_excluded"] = int(rec["n_events_total"] - rec["n_events_used"])
    rec["excluded_reason"] = "|".join(f"{k}:{v}" for k, v in reasons.items()) or "none"
    if obs_af:  # [c8] per-subject obs vs shaft_matched-null medians (cohort_verdict inputs)
        rec["axial_obs"] = float(np.median(obs_af))
        rec["axial_null"] = float(np.median(null_af))
        rec["lateral_obs"] = float(np.median(obs_lr))
        rec["lateral_null"] = float(np.median(null_lr))
    return rec, per_event


def _write_outputs(records, per_event_all, verdict):
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "subject", "n_events_total", "n_events_used", "n_events_excluded",
            "excluded_reason", "n_low_n", "n_events_subsampled", "n_channels_in_pool",
            "n_channels_mapped", "coord_space", "template_source",
            "axis_core_mm", "axial_available_mm",
            "axial_obs", "axial_null", "lateral_obs", "lateral_null"]
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (OUT / "per_subject.csv").write_text("\n".join(lines) + "\n")

    used = [r for r in records if r["n_events_used"] > 0]
    summary = dict(
        verdict=verdict["verdict"], n_eligible_subjects=len(records),
        n_subjects_with_events=len(used),
        AF_cohort_median=verdict["AF"], LR_cohort_median=verdict["LR"],
        axial_delta_mean=verdict["axial_delta_mean"], axial_ci=verdict["axial_ci"],
        lateral_delta_mean=verdict["lateral_delta_mean"], lateral_ci=verdict["lateral_ci"],
        axial_wilcoxon_p=verdict["axial_wilcoxon_p"],
        lateral_wilcoxon_p=verdict["lateral_wilcoxon_p"],
        excluded=[{k: r[k] for k in ("dataset", "subject", "excluded_reason")}
                  for r in records if r["n_events_used"] == 0],
        pre_registered_thresholds=dict(
            AXIAL_EXTENDED_LATERAL_NARROW="AF>=0.75 & LR<=0.5 & lateral below shaft-null",
            AXIAL_SEGMENT="AF<=0.5 & axial below shaft-null",
            SAMPLING_ARTIFACT="AF<=0.5 & axial CI includes 0",
            INCONCLUSIVE="n<10, or 0.5<AF<0.75, or CI straddles 0 (default)"),
        n_draw=N_DRAW, min_n_part=MIN_N_PART, n_event_cap=N_EVENT_CAP)
    (OUT / "cohort_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _figure(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    used = [r for r in records if r["n_events_used"] > 0]
    if not used:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(used))
    labels = [f"{r['dataset'][:3]}_{r['subject']}" for r in used]
    # Panel A: axial_fraction observed vs shaft-matched null (do events fill the axis?)
    ax[0].plot(x, [r["axial_obs"] for r in used], "o-", color="#1f77b4", label="observed")
    ax[0].plot(x, [r["axial_null"] for r in used], "s--", color="#999999", label="shaft-matched null")
    ax[0].axhline(0.75, color="#2ca02c", ls=":", lw=1)
    ax[0].axhline(0.5, color="#d62728", ls=":", lw=1)
    ax[0].set_ylabel("axial_fraction (axis filled)")
    ax[0].set_title("A  Does each event fill the propagation axis?")
    ax[0].legend(frameon=False, fontsize=8)
    # Panel B: lateral_ratio observed vs null (how narrow sideways?)
    ax[1].plot(x, [r["lateral_obs"] for r in used], "o-", color="#9467bd", label="observed")
    ax[1].plot(x, [r["lateral_null"] for r in used], "s--", color="#999999", label="shaft-matched null")
    ax[1].axhline(0.5, color="#d62728", ls=":", lw=1)
    ax[1].set_ylabel("lateral_ratio (sideways / axial)")
    ax[1].set_title("B  How narrow are events transverse to the axis?")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "event_extent.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap subjects")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    subjects = _eligible_subjects()
    if a.limit:
        subjects = subjects[: a.limit]
    print(f"[event-extent] {len(subjects)} eligibility-spine subjects")

    records, per_event_all = [], []
    for ds, subj, card in subjects:
        rec, pe = _audit_subject(ds, subj, card, rng)
        records.append(rec)
        per_event_all.extend(pe)
        tag = rec["excluded_reason"] if rec["n_events_used"] == 0 else f"used={rec['n_events_used']}"
        print(f"  {ds:11s} {subj:14s} {tag}")

    used = [r for r in records if r["n_events_used"] > 0]
    verdict = cohort_verdict(used, rng) if used else dict(
        verdict="INCONCLUSIVE", n_subjects=0, AF=float("nan"), LR=float("nan"),
        axial_delta_mean=float("nan"), lateral_delta_mean=float("nan"),
        axial_ci=(float("nan"), float("nan")), lateral_ci=(float("nan"), float("nan")),
        axial_wilcoxon_p=float("nan"), lateral_wilcoxon_p=float("nan"))
    summary = _write_outputs(records, per_event_all, verdict)
    _figure(records)
    print(f"\n[event-extent] VERDICT = {summary['verdict']}  "
          f"(AF={summary['AF_cohort_median']:.3f}, LR={summary['LR_cohort_median']:.3f}, "
          f"n_used={summary['n_subjects_with_events']})")
    print(f"  axial Δ CI = {summary['axial_ci']}   lateral Δ CI = {summary['lateral_ci']}")
    print(f"  wrote {OUT}/per_subject.csv + cohort_summary.json + figures/event_extent.png")


if __name__ == "__main__":
    main()
