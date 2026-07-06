#!/usr/bin/env python3
"""Topic 5 V2 Phase 1 -- Task 9: order-null DEPENDENCY check (per axis_set).

The HFO-rate-preserving order null (``order_null_rank_pair``) rebuilds each template's typical
timing rank from its own event table, then destroys the within-event timing order while keeping
each contact's participation (HFO-rate) topography. For that null to be trustworthy per subject,
``rebuild_typical_rank`` on the REAL (un-permuted) events must reproduce the PRODUCER geometry
G_HFO (the ``typical_rank`` field in the propagation_geometry ``*_t_{a,b}.json`` planes). This
script computes that sanity per template and grades each subject.

What is fed as ``event_lag``: the per-event phantom-MASKED normalized rank
(``mask_phantom_ranks(ranks, bools, normalize=True)``) -- i.e. the per-event timing ORDER whose
aggregation IS the producer ``typical_rank`` (nanmedian over events). This is the quantity the
order null permutes; feeding raw lag TIMES would test a different, inconsistent null (the producer
never uses lag times for ``typical_rank``).

Event tables + labels + the label<->event positional-identity proof (C1) are reused from
``src.topic5_event_resolved_alignment.load_event_labels_ranks`` -- its null contract (prove the
loaded events line up with the producer's cluster labels) is exactly what this sanity needs, so we
must NOT re-derive events by hand. Its ``broad`` flag only guards the (unrelated) per-event field
metric; the loader mechanics + C1 proof are substrate-agnostic, so narrow reuses it with the narrow
labels/lagpat dirs.

Out: ``{outdir}/{axis_set}/phase1_order_null_depcheck.csv``
Cols: subject, axis_set, has_event_data_a, has_event_data_b,
      corr_rebuilt_vs_geo_a, corr_rebuilt_vs_geo_b, order_null_strength(strong|weak_downgrade|missing)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.topic5_event_resolved_alignment as erm
from src.topic5_v2_band_scan import load_phase1_config, rebuild_typical_rank

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

# Per axis_set: producer geometry planes (typical_rank = G_HFO), broad/narrow masked labels,
# and how to reach the SAME lagPat pool the labels were computed from. broad lagpat = None ->
# erm._broad_lagpat_dir (yuquan lagpat_broad_dyn / epi lagpat_broad_epilepsiae). narrow lagpat =
# the on-mount narrow pool that produced the narrow masked labels.
_SUBSTRATE = {
    "broad": {
        "geo": _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects",
        "labels": "results/interictal_propagation_masked_broad/per_subject",
        "narrow_lagpat": False,
    },
    "narrow": {
        "geo": _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects",
        "labels": "results/interictal_propagation_masked/per_subject",
        "narrow_lagpat": True,
    },
}
CSV_COLS = ["subject", "axis_set", "has_event_data_a", "has_event_data_b",
            "corr_rebuilt_vs_geo_a", "corr_rebuilt_vs_geo_b", "order_null_strength"]


def _narrow_lagpat_dir(ds: str, subj: str) -> Path:
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _load_bundle(ds: str, subj: str, cfg: dict) -> dict:
    """Reuse the C1-proven loader; narrow overrides labels/lagpat dirs (see module docstring)."""
    if cfg["narrow_lagpat"]:
        return erm.load_event_labels_ranks(ds, subj, labels_dir=cfg["labels"],
                                           lagpat_dir=str(_narrow_lagpat_dir(ds, subj)))
    return erm.load_event_labels_ranks(ds, subj, labels_dir=cfg["labels"])


def _producer_typical_rank(plane_json: dict, order):
    """G_HFO in `order` channel order, NaN where the channel is absent / has no typical_rank."""
    if "channels" not in plane_json:
        return None                                     # descriptive_only / no_events plane
    d = {c["name"]: c.get("typical_rank") for c in plane_json["channels"]}
    return np.array([(d[n] if (n in d and d[n] is not None) else np.nan) for n in order], float)


def _spearman_shared(rebuilt, producer):
    """Spearman over channels finite in BOTH; None if <3 shared or a degenerate (0-variance) side."""
    m = np.isfinite(rebuilt) & np.isfinite(producer)
    if int(m.sum()) < 3 or np.std(rebuilt[m]) == 0 or np.std(producer[m]) == 0:
        return None
    return float(spearmanr(rebuilt[m], producer[m]).correlation)


def eval_subject(ds_sid: str, cfg: dict, axis_set: str, min_corr: float) -> dict:
    ds, subj = ds_sid.split("_", 1)
    row = {"subject": ds_sid, "axis_set": axis_set, "has_event_data_a": False,
           "has_event_data_b": False, "corr_rebuilt_vs_geo_a": "",
           "corr_rebuilt_vs_geo_b": "", "order_null_strength": "missing"}
    taf, tbf = cfg["geo"] / f"{ds_sid}_t_a.json", cfg["geo"] / f"{ds_sid}_t_b.json"
    if not (taf.exists() and tbf.exists()):
        return row                                       # no producer geometry -> missing
    try:
        bundle = _load_bundle(ds, subj, cfg)             # FileNotFoundError / ValueError(C1) -> missing
    except (FileNotFoundError, ValueError, KeyError):
        return row
    order = bundle["channel_names"]
    ta = _producer_typical_rank(json.load(open(taf)), order)
    tb = _producer_typical_rank(json.load(open(tbf)), order)
    if ta is None or tb is None:
        return row                                       # degenerate plane -> can't map -> missing
    cmap = erm.map_clusters_to_templates(bundle["cluster_template_ranks"][0],
                                         bundle["cluster_template_ranks"][1], ta, tb)
    if cmap["ambiguous"]:
        return row                                       # can't bind events to a template -> missing
    inv = {v: k for k, v in cmap["map"].items()}         # "t_a"/"t_b" -> cluster label 0/1
    corrs = {}
    for tid, producer, key in (("t_a", ta, "a"), ("t_b", tb, "b")):
        sel = bundle["labels"] == inv[tid]
        if int(sel.sum()) == 0:
            continue
        # (n_ch, n_sel) -> (n_sel, n_ch) event x channel; masked normalized rank = event_lag.
        # agg="median" explicit: must match the producer's own typical_rank aggregator
        # (nanmedian over events, see module docstring above) or this compares rebuilt
        # geometry against a different geometry than what Gate A actually observed.
        rebuilt = rebuild_typical_rank(bundle["bools"][:, sel].T, bundle["masked"][:, sel].T, agg="median")
        rho = _spearman_shared(rebuilt, producer)
        if rho is None:
            continue
        row[f"has_event_data_{key}"] = True
        row[f"corr_rebuilt_vs_geo_{key}"] = round(rho, 6)
        corrs[key] = rho
    if not corrs:
        row["order_null_strength"] = "missing"
    else:
        # conservative: the A/B-max downstream may invoke either template's geometry, so the null
        # is `strong` only if EVERY available template reproduces the producer (>= min_corr).
        row["order_null_strength"] = "strong" if all(c >= min_corr for c in corrs.values()) else "weak_downgrade"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--substrate", choices=list(_SUBSTRATE), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit ds_sid list (e.g. epilepsiae_139); default = every subject with a "
                         "producer geometry plane in this axis_set's geo dir")
    ap.add_argument("--outdir", default=None,
                    help="override output ROOT (default results/.../v2_band_scan); "
                         "writes {outdir}/{axis_set}/phase1_order_null_depcheck.csv")
    args = ap.parse_args()

    cfg = _SUBSTRATE[args.substrate]
    min_corr = float(load_phase1_config()["nulls"]["order_null_min_corr_to_geo"])
    subjects = args.subjects or sorted(p.name[:-len("_t_a.json")] for p in cfg["geo"].glob("*_t_a.json"))

    rows = []
    for ds_sid in subjects:
        row = eval_subject(ds_sid, cfg, args.substrate, min_corr)
        rows.append(row)
        print(f"[{args.substrate}] {ds_sid:>26} a={row['corr_rebuilt_vs_geo_a'] or 'NA':>8} "
              f"b={row['corr_rebuilt_vs_geo_b'] or 'NA':>8} -> {row['order_null_strength']}", flush=True)

    out_root = Path(args.outdir) if args.outdir else OUT_ROOT
    outdir = out_root / args.substrate
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "phase1_order_null_depcheck.csv"
    with open(outpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(rows)

    dist = {k: sum(r["order_null_strength"] == k for r in rows)
            for k in ("strong", "weak_downgrade", "missing")}
    print(f"[done] {args.substrate}: {len(rows)} subjects -> {outpath} | "
          f"strong={dist['strong']} weak_downgrade={dist['weak_downgrade']} missing={dist['missing']} "
          f"| min_corr_to_geo={min_corr}", flush=True)
    if dist["strong"] == 0:
        # QC print only (issue #12): the order null still runs downstream on weak_downgrade subjects
        # with the flag propagated; we do NOT hard-fail the dependency check.
        print(f"[QC-WARN] {args.substrate}: 0 subjects reach order_null_strength=strong "
              f"(corr_rebuilt_vs_geo >= {min_corr}); downstream A/B-max order null carries the "
              f"weak_downgrade flag.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
