#!/usr/bin/env python3
"""Full-cohort A/B gradient axes, collinearity, and own/shared ictal-field readout.

Contract: docs/superpowers/specs/2026-07-14-topic5-template-gradient-shared-field-design.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.propagation_skeleton_geometry import assign_events_to_templates, parse_shaft  # noqa: E402
from src.seeg_coord_loader import load_subject_coords  # noqa: E402
from src.topic5_axis_alignment import (channel_shuffle, effective_shuffle_n,  # noqa: E402
                                       within_shaft_shuffle)
from src.topic5_template_axis_field import (compute_template_axis_pair,  # noqa: E402
                                              make_field_scorer,
                                              make_normalized_plane,
                                              score_scorer_bundle_batch,
                                              score_scorer_bundle,
                                              z_earliness)

RANKDISP = ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
DEFAULT_CACHE = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
DEFAULT_OUT = ROOT / "results/topic5_ictal_recruitment/template_axis_field"
TEMPLATE_RECORDS = (ROOT / "results/spatial_modulation/propagation_geometry/"
                    "observation_readout/real_subjects")
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")


def _subject_dir(dataset: str, subject: str) -> Path:
    return YUQUAN_ROOT / subject if dataset == "yuquan" else EPILEPSIAE_ROOT / subject / "all_recs"


def _seed(token: str, base: int = 0) -> int:
    return int((zlib.crc32(token.encode("utf-8")) + int(base)) % (2**32 - 1))


def _jsonable(x):
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _load_axis_input(sid: str) -> Dict[str, object]:
    path = RANKDISP / f"{sid}.json"
    d = json.loads(path.read_text())
    dataset, subject = sid.split("_", 1)
    base = {"subject_id": sid, "dataset": dataset, "subject": subject,
            "stable_k": d.get("stable_k")}
    if d.get("stable_k") != 2:
        return dict(base, status="stable_k_not_2")
    pairs = d.get("pairs") or []
    if not pairs:
        return dict(base, status="missing_pair")
    pair = pairs[0]
    names = list(pair.get("channel_names") or [])
    joint = np.asarray(pair.get("joint_valid"), bool)
    rank_a = np.asarray(pair.get("rank_a_dense_full"), float)
    rank_b = np.asarray(pair.get("rank_b_dense_full"), float)
    if not (len(names) == len(joint) == len(rank_a) == len(rank_b)):
        return dict(base, status="rankdisp_shape_mismatch")
    names_joint = [names[i] for i in np.where(joint)[0]]
    try:
        cr = load_subject_coords(dataset, subject, names_joint)
    except Exception as exc:
        return dict(base, status="coordinate_load_failed", error=str(exc)[:200])
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    ra, rb = rank_a[joint], rank_b[joint]
    valid = mapped & np.isfinite(coords).all(1) & np.isfinite(ra) & np.isfinite(rb)
    if int(valid.sum()) < 6:
        return dict(base, status="insufficient_joint_mapped", n_joint_mapped=int(valid.sum()))
    names_used = [names_joint[i] for i in np.where(valid)[0]]
    coords_used, ra_used, rb_used = coords[valid], ra[valid], rb[valid]
    shafts = [parse_shaft(n)[0] for n in names_used]
    pair_out = compute_template_axis_pair(
        coords_used, ra_used, rb_used, shafts,
        n_axis_boot=200, n_pair_boot=500, seed=_seed(sid, 17), line_threshold=0.50,
    )
    return dict(base, status=pair_out.get("status"), names=names_used, coords=coords_used,
                rank_a=ra_used, rank_b=rb_used, shafts=shafts, axis_pair=pair_out,
                swap_class=((pair.get("swap_sweep") or {}).get("swap_class") or "none"),
                decision_k=(pair.get("swap_sweep") or {}).get("decision_k"))


def _load_interictal_support(dataset: str, subject: str, rank_a_by_name: Mapping[str, float],
                             rank_b_by_name: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    # Canonical readout records already contain the masked, template-assigned participation
    # fraction.  Reuse that producer output; support is independent of its legacy endpoint axis.
    pa = TEMPLATE_RECORDS / f"{dataset}_{subject}_t_a.json"
    pb = TEMPLATE_RECORDS / f"{dataset}_{subject}_t_b.json"
    if pa.exists() and pb.exists():
        da, db = json.loads(pa.read_text()), json.loads(pb.read_text())
        sa = {str(c["name"]): float(c["support"]) for c in da.get("channels", [])}
        sb = {str(c["name"]): float(c["support"]) for c in db.get("channels", [])}
        # Common support remains a direct all-event participation fraction.
        ev = load_subject_propagation_events(_subject_dir(dataset, subject))
        bools = np.asarray(ev["bools"], bool)
        common = {str(n): float(v) for n, v in zip(ev["channel_names"], bools.mean(axis=1))}
        return {"a": sa, "b": sb, "common": common,
                "n_events": {"a": None, "b": None, "unassigned": None},
                "support_source": "canonical_template_readout_records"}
    ev = load_subject_propagation_events(_subject_dir(dataset, subject))
    bools = np.asarray(ev["bools"], bool)
    if bools.ndim != 2 or bools.shape[1] == 0:
        return {}
    names = [str(x) for x in ev["channel_names"]]
    ranks = np.asarray(ev["ranks"], float)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    ta = np.asarray([rank_a_by_name.get(n, np.nan) for n in names], float)
    tb = np.asarray([rank_b_by_name.get(n, np.nan) for n in names], float)
    labels = assign_events_to_templates(masked, ta, tb)

    def support_for(label):
        sel = labels == label
        if not np.any(sel):
            return np.zeros(len(names), float)
        return bools[:, sel].mean(axis=1)

    sa, sb, common = support_for(0), support_for(1), bools.mean(axis=1)
    return {"a": {n: float(v) for n, v in zip(names, sa)},
            "b": {n: float(v) for n, v in zip(names, sb)},
            "common": {n: float(v) for n, v in zip(names, common)},
            "n_events": {"a": int(np.sum(labels == 0)), "b": int(np.sum(labels == 1)),
                         "unassigned": int(np.sum(labels < 0))}}


def _load_ictal(cache_dir: Path, sid: str, feature_key: str):
    npz_path, json_path = cache_dir / f"{sid}.npz", cache_dir / f"{sid}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(json_path.read_text())
    channels = [str(x) for x in data["channels"]]
    vectors = {}
    for idx in meta.get("eligible_idxs", []):
        key = f"{feature_key}__{idx}"
        if key in data.files:
            vectors[str(idx)] = np.asarray(data[key], float)
    return {"channels": channels, "vectors": vectors, "meta": meta}


def _make_field_bundle(axis_record: Mapping[str, object], cache_dir: Path,
                       feature_key: str, support_mode: str) -> Dict[str, object]:
    sid = str(axis_record["subject_id"])
    ictal = _load_ictal(cache_dir, sid, feature_key)
    if ictal is None:
        return {"status": "missing_ictal_cache"}
    try:
        rank_a_by_name = dict(zip(axis_record["names"], np.asarray(axis_record["rank_a"], float)))
        rank_b_by_name = dict(zip(axis_record["names"], np.asarray(axis_record["rank_b"], float)))
        supports = _load_interictal_support(str(axis_record["dataset"]),
                                            str(axis_record["subject"]),
                                            rank_a_by_name, rank_b_by_name)
    except Exception as exc:
        return {"status": "interictal_support_failed", "error": str(exc)[:200]}
    if not supports:
        return {"status": "interictal_support_failed", "error": "empty support"}
    support_a_by_name = supports["common"] if support_mode == "common" else supports["a"]
    support_b_by_name = supports["common"] if support_mode == "common" else supports["b"]
    cache_index = {n: i for i, n in enumerate(ictal["channels"])}
    names_all = list(axis_record["names"])
    keep = np.array([n in cache_index and support_a_by_name.get(n, 0.0) > 0
                     and support_b_by_name.get(n, 0.0) > 0 for n in names_all])
    if int(keep.sum()) < 6:
        return {"status": "insufficient_field_contacts", "n_contacts": int(keep.sum())}
    names = [names_all[i] for i in np.where(keep)[0]]
    coords = np.asarray(axis_record["coords"], float)[keep]
    rank_a = np.asarray(axis_record["rank_a"], float)[keep]
    rank_b = np.asarray(axis_record["rank_b"], float)[keep]
    support_a = np.asarray([support_a_by_name[n] for n in names], float)
    support_b = np.asarray([support_b_by_name[n] for n in names], float)
    seizure_values = {}
    cache_ix = np.asarray([cache_index[n] for n in names], int)
    for sz, vec in ictal["vectors"].items():
        vals = np.asarray(vec, float)[cache_ix]
        if int(np.isfinite(vals).sum()) >= 6 and np.nanstd(vals) > 1e-12:
            seizure_values[sz] = vals
    if not seizure_values:
        return {"status": "no_resolvable_seizure", "n_contacts": len(names)}

    pair = axis_record["axis_pair"]
    ax_a, ax_b = pair["axis_a"], pair["axis_b"]
    own_a = make_normalized_plane(coords, ax_a["u"], origin=ax_a["xbar"])
    own_b = make_normalized_plane(coords, ax_b["u"], origin=ax_b["xbar"])
    if own_a.get("status") != "ok" or own_b.get("status") != "ok":
        return {"status": "own_plane_failed", "plane_a": own_a.get("status"),
                "plane_b": own_b.get("status")}
    ea, eb = z_earliness(rank_a), z_earliness(rank_b)
    scorers = {
        "own_a": make_field_scorer(ea, own_a["points"], support_a, own_a["sigma"]),
        "own_b": make_field_scorer(eb, own_b["points"], support_b, own_b["sigma"]),
    }
    plane_meta = {"own_a": own_a, "own_b": own_b}
    if pair["relation"]["collinear"] and pair["shared_axis"].get("status") == "ok":
        shared = make_normalized_plane(coords, pair["shared_axis"]["u"], origin=ax_a["xbar"])
        if shared.get("status") == "ok":
            scorers["shared_a"] = make_field_scorer(ea, shared["points"], support_a, shared["sigma"])
            scorers["shared_b"] = make_field_scorer(eb, shared["points"], support_b, shared["sigma"])
            plane_meta["shared"] = shared
    seizure_mean = np.nanmean(np.vstack(list(seizure_values.values())), axis=0)
    return {"status": "ok", "names": names, "coords": coords, "rank_a": rank_a,
            "rank_b": rank_b, "earliness_a": ea, "earliness_b": eb,
            "support_a": support_a, "support_b": support_b, "support_mode": support_mode,
            "support_source": supports.get("support_source", "recomputed_from_masked_events"),
            "template_event_counts": supports["n_events"], "seizure_mean": seizure_mean,
            "seizure_values": seizure_values,
            "scorers": scorers, "planes": plane_meta, "n_contacts": len(names),
            "n_seizures": len(seizure_values)}


def _fold_metric(obs: Sequence[float], null: np.ndarray, *, effective_n: int) -> Dict[str, object]:
    obs_arr = np.asarray(obs, float)
    null_arr = np.asarray(null, float)
    obs_subject = float(np.nanmedian(obs_arr))
    dist = np.nanmedian(null_arr, axis=0)
    finite = dist[np.isfinite(dist)]
    if not np.isfinite(obs_subject) or finite.size == 0:
        return {"status": "not_computable"}
    q = {f"p{p}": float(np.percentile(finite, p)) for p in (5, 50, 95, 99)}
    status = "ok" if effective_n >= 4 else "INSUFFICIENT_NULL"
    return {"status": status, "obs_subject": obs_subject,
            "obs_per_seizure": obs_arr.tolist(), "null_q": q,
            "null_dist": finite.tolist(),
            "margin_vs_null_median": float(obs_subject - q["p50"]),
            "p_upper": float((1 + np.sum(finite >= obs_subject)) / (1 + len(finite))),
            "passed": bool(status == "ok" and obs_subject > q["p95"]),
            "effective_shuffle_n": int(effective_n), "n_seizures": int(len(obs_arr))}


def _run_field_nulls(bundle: Mapping[str, object], *, B: int, seed: int) -> Dict[str, object]:
    scorers = bundle["scorers"]
    values_by_sz = bundle["seizure_values"]
    names = bundle["names"]
    observed = {sz: score_scorer_bundle(scorers, vals) for sz, vals in values_by_sz.items()}
    metric_keys = sorted({k for r in observed.values() for k in r
                          if k.endswith("_abs") or k.endswith("_maxab")})
    signed_keys = sorted({k for r in observed.values() for k in r if k.endswith("_signed")})
    signed_subject = {k: float(np.nanmedian([r.get(k, np.nan) for r in observed.values()]))
                      for k in signed_keys}
    out = {"observed_by_seizure": observed, "observed_signed_subject": signed_subject,
           "nulls": {}}
    for mode_i, mode in enumerate(("channel", "within_shaft")):
        rng = np.random.default_rng(seed + 100003 * mode_i)
        effective_n = effective_shuffle_n(names, None, mode)
        per_metric_obs = {k: [] for k in metric_keys}
        per_metric_null = {k: [] for k in metric_keys}
        for sz, values in values_by_sz.items():
            obs = observed[sz]
            for k in metric_keys:
                per_metric_obs[k].append(obs.get(k, np.nan))
            n = len(values)
            indices = np.tile(np.arange(n), (int(B), 1))
            if mode == "channel":
                for b in range(int(B)):
                    indices[b] = rng.permutation(n)
            else:
                groups = {}
                for j, name in enumerate(names):
                    groups.setdefault(parse_shaft(name)[0], []).append(j)
                for idx in groups.values():
                    ix = np.asarray(idx, int)
                    for b in range(int(B)):
                        indices[b, ix] = ix[rng.permutation(len(ix))]
            shuffled = np.asarray(values, float)[indices]
            draws = score_scorer_bundle_batch(scorers, shuffled)
            for k in metric_keys:
                per_metric_null[k].append(np.asarray(draws.get(k, np.full(B, np.nan)), float))
        metrics = {k: _fold_metric(per_metric_obs[k], np.asarray(per_metric_null[k], float),
                                   effective_n=effective_n)
                   for k in metric_keys}
        out["nulls"][mode] = {"effective_shuffle_n": int(effective_n), "metrics": metrics}
    return out


def _axis_csv_row(record: Mapping[str, object], field: Mapping[str, object]) -> Dict[str, object]:
    row = {k: record.get(k) for k in ("subject_id", "dataset", "subject", "stable_k", "status",
                                      "swap_class", "decision_k")}
    row.update({"field_status": field.get("status"), "n_field_contacts": field.get("n_contacts"),
                "n_seizures": field.get("n_seizures")})
    pair = record.get("axis_pair")
    if not isinstance(pair, Mapping) or pair.get("status") != "ok":
        return row
    a, b, rel, boot = pair["axis_a"], pair["axis_b"], pair["relation"], pair["pair_bootstrap"]
    row.update({
        "n_joint": pair.get("n_joint"),
        "axis_pair_estimable": pair.get("axis_pair_estimable", True),
        "geometry_2d_supported": pair.get("geometry_2d_supported"),
        "strict_stability_pass": pair.get("strict_stability_pass", pair.get("axis_pair_qc_pass")),
        "axis_pair_qc_pass": pair.get("axis_pair_qc_pass"),
        "n_shafts": min(int(a.get("n_shafts") or 0), int(b.get("n_shafts") or 0)),
        "effective_rank": min(int(a.get("effective_rank") or 0), int(b.get("effective_rank") or 0)),
        "axis_a_R2": a.get("R2"), "axis_b_R2": b.get("R2"),
        "axis_a_bootstrap_cosine": a.get("bootstrap_cosine"),
        "axis_b_bootstrap_cosine": b.get("bootstrap_cosine"),
        "axis_a_loso_cosine": a.get("loso_cosine"), "axis_b_loso_cosine": b.get("loso_cosine"),
        "cos_uA_uB": rel.get("cosine"), "abs_cos_uA_uB": rel.get("abs_cosine"),
        "line_angle_deg": rel.get("line_angle_deg"), "collinear_60deg": rel.get("collinear"),
        "relation": rel.get("relation"), "pair_boot_p_collinear": boot.get("p_collinear"),
        "pair_boot_p_sign_stable": boot.get("p_sign_stable"),
        "robust_collinear": boot.get("robust_collinear"),
    })
    return row


def _cohort_metric(records: Sequence[Mapping[str, object]], mode: str,
                   metric: str) -> Dict[str, object]:
    usable = []
    for r in records:
        m = (((r.get("field") or {}).get("statistics") or {}).get("nulls", {})
             .get(mode, {}).get("metrics", {}).get(metric, {}))
        if m.get("status") == "ok" and np.isfinite(m.get("obs_subject", np.nan)):
            usable.append((r["subject_id"], m))
    if not usable:
        return {"status": "no_eligible_subject", "n": 0}
    B = min(len(m["null_dist"]) for _, m in usable)
    obs = np.asarray([m["obs_subject"] for _, m in usable], float)
    null = np.vstack([np.asarray(m["null_dist"][:B], float) for _, m in usable])
    cohort_null = np.nanmedian(null, axis=0)
    obs_cohort = float(np.nanmedian(obs))
    return {"status": "ok", "n": len(usable), "subjects": [s for s, _ in usable],
            "obs_median": obs_cohort, "obs_iqr": [float(x) for x in np.percentile(obs, [25, 75])],
            "null_median": float(np.nanmedian(cohort_null)),
            "null_p95": float(np.nanpercentile(cohort_null, 95)),
            "p_upper": float((1 + np.sum(cohort_null >= obs_cohort)) / (1 + len(cohort_null))),
            "n_subject_pass": int(sum(bool(m.get("passed")) for _, m in usable))}


def _paired_comparison(records: Sequence[Mapping[str, object]], mode: str,
                       seed: int = 0) -> Dict[str, object]:
    rows = []
    for r in records:
        metrics = (((r.get("field") or {}).get("statistics") or {}).get("nulls", {})
                   .get(mode, {}).get("metrics", {}))
        own, shared = metrics.get("own_maxab", {}), metrics.get("shared_maxab", {})
        if own.get("status") == shared.get("status") == "ok":
            rows.append((r["subject_id"], float(own["obs_subject"]), float(shared["obs_subject"]),
                         float(own["margin_vs_null_median"]),
                         float(shared["margin_vs_null_median"])))
    if not rows:
        return {"status": "no_paired_subject", "n": 0}
    d_obs = np.asarray([r[2] - r[1] for r in rows])
    d_margin = np.asarray([r[4] - r[3] for r in rows])
    rng = np.random.default_rng(seed)
    boot = np.asarray([np.median(d_obs[rng.integers(0, len(d_obs), len(d_obs))])
                       for _ in range(10000)])
    try:
        p_two = float(wilcoxon(d_obs, alternative="two-sided").pvalue)
    except ValueError:
        p_two = np.nan
    return {"status": "ok", "n": len(rows), "subjects": [r[0] for r in rows],
            "shared_minus_own": [float(x) for x in d_obs],
            "median_shared_minus_own": float(np.median(d_obs)),
            "bootstrap_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "wilcoxon_p_two_sided": p_two,
            "median_margin_difference": float(np.median(d_margin))}


def _summarize(records: Sequence[Mapping[str, object]], B: int, feature_key: str,
               support_mode: str) -> Dict[str, object]:
    axis_ok = [r for r in records if r.get("status") == "ok" and isinstance(r.get("axis_pair"), Mapping)]
    field_ok = [r for r in axis_ok if (r.get("field") or {}).get("status") == "ok"]
    geometry_2d = [r for r in axis_ok if r["axis_pair"].get("geometry_2d_supported")]
    strict_stability = [r for r in axis_ok if r["axis_pair"].get("strict_stability_pass")]
    own_geometry = [r for r in field_ok if r["axis_pair"].get("geometry_2d_supported")]
    own_strict = [r for r in field_ok if r["axis_pair"].get("strict_stability_pass")]
    shared_all = [r for r in field_ok if r["axis_pair"]["relation"].get("collinear")]
    shared_geometry = [r for r in own_geometry if r["axis_pair"]["relation"].get("collinear")]
    shared_strict = [r for r in own_strict if r["axis_pair"]["relation"].get("collinear")]
    subsets = {
        "own_axis_defined": field_ok,
        "own_2d_geometry": own_geometry,
        "own_strict_stability": own_strict,
        "shared_all_axis_defined_60deg": shared_all,
        "shared_2d_geometry_60deg": shared_geometry,
        "shared_strict_stability_60deg": shared_strict,
        "shared_45deg": [r for r in shared_geometry if r["axis_pair"]["relation"]["abs_cosine"] >= np.sqrt(0.5)],
        "shared_30deg": [r for r in shared_geometry if r["axis_pair"]["relation"]["abs_cosine"] >= np.sqrt(3) / 2],
        "shared_robust_pair_bootstrap": [r for r in shared_geometry
                                         if r["axis_pair"]["pair_bootstrap"].get("robust_collinear")],
    }
    metrics_by_subset = {}
    for subset_name, subset in subsets.items():
        metrics_by_subset[subset_name] = {}
        metrics = ("own_a_abs", "own_b_abs", "own_maxab")
        if subset_name.startswith("shared"):
            metrics += ("shared_a_abs", "shared_b_abs", "shared_maxab")
        for mode in ("channel", "within_shaft"):
            metrics_by_subset[subset_name][mode] = {
                metric: _cohort_metric(subset, mode, metric) for metric in metrics
            }
    paired = {name: {mode: _paired_comparison(subset, mode, _seed(name + mode))
                     for mode in ("channel", "within_shaft")}
              for name, subset in subsets.items() if name.startswith("shared")}
    relation_counts = {k: sum(r["axis_pair"]["relation"]["relation"] == k for r in axis_ok)
                       for k in ("same", "reversed", "different")}
    geometry_relation_counts = {k: sum(r["axis_pair"]["relation"]["relation"] == k
                                       for r in geometry_2d)
                                for k in ("same", "reversed", "different")}
    strict_relation_counts = {k: sum(r["axis_pair"]["relation"]["relation"] == k
                                     for r in strict_stability)
                              for k in ("same", "reversed", "different")}
    return {"contract": "template_gradient_shared_field_v2_early_to_late", "feature_key": feature_key,
            "axis_definition": "template_propagation_axis_v2",
            "axis_direction_convention": "positive_early_to_late",
            "support_mode": support_mode,
            "B": int(B), "denominators": {"rankdisp_total": len(records), "axis_pair_ok": len(axis_ok),
                                            "field_cache_ok": len(field_ok),
                                            **{k: len(v) for k, v in subsets.items()}},
            "axis_relation_counts_all_axis_ok": relation_counts,
            "axis_relation_counts_2d_geometry": geometry_relation_counts,
            "axis_relation_counts_strict_stability": strict_relation_counts,
            # Backward-compatible alias for the old strict-stability label.
            "axis_relation_counts_axis_qc": strict_relation_counts,
            "metrics": metrics_by_subset, "paired_shared_vs_own": paired,
            "claim_boundary": "exploratory; axis and collinearity frozen without ictal outcomes; early-ictal readout, not early-ictal-specific"}


def run(subjects: Sequence[str], cache_dir: Path, out_dir: Path, *,
        feature_key: str, support_mode: str, B: int) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_subject").mkdir(exist_ok=True)
    records = []
    rows = []
    for i, sid in enumerate(subjects, 1):
        rec = _load_axis_input(sid)
        field: Dict[str, object] = {"status": "axis_not_available"}
        if rec.get("status") == "ok":
            bundle = _make_field_bundle(rec, cache_dir, feature_key, support_mode)
            field = {k: v for k, v in bundle.items()
                     if k not in {"scorers", "seizure_values"}}
            if bundle.get("status") == "ok":
                field["statistics"] = _run_field_nulls(bundle, B=B, seed=_seed(sid, 991))
        rec["field"] = field
        rows.append(_axis_csv_row(rec, field))
        serial = _jsonable(rec)
        (out_dir / "per_subject" / f"{sid}.json").write_text(
            json.dumps(serial, ensure_ascii=False, indent=2))
        records.append(rec)
        pair = rec.get("axis_pair") or {}
        rel = (pair.get("relation") or {}).get("relation", "-")
        print(f"[{i:02d}/{len(subjects)}] {sid}: axis={rec.get('status')} "
              f"qc={pair.get('axis_pair_qc_pass')} relation={rel} field={field.get('status')}",
              flush=True)
    cols = sorted({k for row in rows for k in row})
    with (out_dir / "axis_cohort.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(_jsonable(row))
    summary = _summarize(records, B, feature_key, support_mode)
    (out_dir / "cohort_summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None, help="dataset_subject tokens")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--feature-key", default="bb_auc")
    ap.add_argument("--support-mode", choices=["template", "common"], default="template")
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    subjects = args.subjects or sorted(p.stem for p in RANKDISP.glob("*.json"))
    summary = run(subjects, args.cache_dir, args.out_dir, feature_key=args.feature_key,
                  support_mode=args.support_mode, B=args.B)
    print(json.dumps(_jsonable(summary["denominators"]), ensure_ascii=False, indent=2))
    print(f"wrote {args.out_dir / 'axis_cohort.csv'}")
    print(f"wrote {args.out_dir / 'cohort_summary.json'}")


if __name__ == "__main__":
    main()
