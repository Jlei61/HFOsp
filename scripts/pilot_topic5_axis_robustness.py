#!/usr/bin/env python3
"""PILOT (exploratory, not a cohort run) — Topic 5 TA/TB field-reversal §6a axis-level
robustness supplement.

Spec: docs/superpowers/specs/2026-07-06-topic5-tatb-field-reversal-design.md §6a

Real scientific question (§6a, not the already-built §6b per-contact LOO sanity check):
does coordinate-aware field smoothing give a MORE ROBUST / GEOMETRICALLY-CORRECT propagation
AXIS than the raw contact reading, and does it fix the specific 1146-style failure mode where
a coordinate-blind (electrode-identity-only) reading mistakes "early contacts on shaft A and
shaft B, both near the shared entrance" for "propagation runs shaft A -> shaft B"?

This script is throwaway pilot code (not the shipped src/ module) run on 5 subjects, broad
substrate only, to check the mechanism BEFORE any cohort-scale implementation. It reuses the
existing loaders (load_event_labels_ranks / map_clusters_to_templates / pick_reference /
class_aggregate_contact_values / class_template_sigma / field_from_contact_values) exactly as
scripts/run_topic5_field_reversal.py does, on the SAME shared t_a reference plane (P0).

Three axes, same shared 2D plane, same weighted-least-squares gradient estimator
(value ~ 1 + x + y, weighted), differing ONLY in what per-contact "value" is fit:
  - raw_contact_axis: the contact's own class-aggregate value (no smoothing).
  - field_axis:       the smoothed field's value at supported grid pixels (smoothing only).
  - sequence_axis:    each contact's own SHAFT's support-weighted mean value, broadcast to
                       every contact on that shaft (electrode-identity-only "smoothing" that
                       throws away within-shaft position -- coordinate-blind by construction,
                       though the resulting vector still needs real coordinates to be expressed
                       as a direction, same as the other two axes).

Main metric = held-out axis score: split a class's events into train/held-out halves; fit
raw_contact_axis and field_axis from the train half; project each contact's REAL position onto
each train axis; score = Spearman(projection, held-out per-contact mean rank). This is the
"stable AND correct" test, immune to amplitude inflation (spec §6a).

Secondary = 1146-style geometric diagnostic (full-class, no split): cos(sequence_axis,
field_axis) and cos(raw_contact_axis, field_axis), per class, per subject.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.topic5_event_resolved_alignment import (
    load_event_labels_ranks, map_clusters_to_templates, class_aggregate_contact_values,
    class_template_sigma, field_from_contact_values, build_plane_xy)
from src.propagation_contact_plane_readout import make_plane_grid, S_THRESH
from src.propagation_skeleton_geometry import parse_shaft

GEOM_DIR = _ROOT / "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
SUBJECTS = ["epilepsiae_1146", "epilepsiae_1077", "epilepsiae_1084", "epilepsiae_1125", "epilepsiae_1150"]
RNG_SEED = 20260706
N_SPLIT = 30          # per class (TA, TB) -- "a few bootstrap splits", pilot-scale
MIN_POINTS = 3         # same floor as MIN_PLANE_CONTACTS elsewhere in this pipeline


# --------------------------------------------------------------------- P0 shared-frame loading
def pick_reference(cmap, plane_a, plane_b):
    """P0 reference frame = the plane mapped to t_a (same convention as run_topic5_field_reversal.py)."""
    inv = {v: k for k, v in cmap["map"].items()}
    plane_of = {"t_a": plane_a, "t_b": plane_b}
    return plane_of["t_a"], inv["t_a"], inv["t_b"]


def _vec_in_order(name_to_val, order):
    return np.array([name_to_val.get(n, np.nan) for n in order], float)


def _aggregate_over_events(masked, names, cols):
    """Per-contact masked-rank mean over the given event columns -> {name:{value,support}}
    (same construction as topic5_field_reversal._aggregate_over_events; small enough to keep
    local to this pilot script rather than importing a leading-underscore cross-module name)."""
    sub = masked[:, cols]
    with np.errstate(invalid="ignore"):
        val = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmean(sub, axis=1))
    sup = np.isfinite(sub).mean(axis=1)
    return {n: {"value": float(val[i]), "support": float(sup[i])} for i, n in enumerate(names)}


# --------------------------------------------------------------------------- the shared estimator
def weighted_ls_gradient_2d(value, x, y, weight):
    """value_i ~ 1 + x_i + y_i, weighted LS (weight_i). Returns unit vector pointing toward
    increasing value (early->late by construction: moving along +unit increases predicted value).
    ok=False if <MIN_POINTS usable rows or the fitted gradient is ~0 (no resolvable direction)."""
    value = np.asarray(value, float); x = np.asarray(x, float)
    y = np.asarray(y, float); weight = np.asarray(weight, float)
    finite = np.isfinite(value) & np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    n = int(finite.sum())
    if n < MIN_POINTS:
        return {"beta": (float("nan"), float("nan")), "unit": (float("nan"), float("nan")),
                "n": n, "ok": False}
    v = value[finite]; xx = x[finite]; yy = y[finite]; w = weight[finite]
    A = np.column_stack([np.ones_like(xx), xx, yy])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    _b0, bx, by = coef
    mag = float(np.hypot(bx, by))
    if mag < 1e-12:
        return {"beta": (float(bx), float(by)), "unit": (float("nan"), float("nan")),
                "n": n, "ok": False}
    return {"beta": (float(bx), float(by)), "unit": (float(bx / mag), float(by / mag)),
            "n": n, "ok": True}


def raw_contact_axis(cav: dict, plane_xy: dict) -> dict:
    """Coordinate LS, NO smoothing: contact's own class-aggregate value ~ 1+x+y, weight=support."""
    names = [n for n in cav if n in plane_xy and np.isfinite(cav[n]["value"])]
    val = np.array([cav[n]["value"] for n in names], float)
    w = np.array([cav[n]["support"] for n in names], float)
    x = np.array([plane_xy[n][0] for n in names], float)
    y = np.array([plane_xy[n][1] for n in names], float)
    return weighted_ls_gradient_2d(val, x, y, w)


def field_axis_from_field(field: dict, X: np.ndarray, Y: np.ndarray, s_thresh: float = S_THRESH) -> dict:
    """Smooth-first LS: field value ~ 1+x+y over supported grid pixels, weight=field support S.
    Same estimator as raw_contact_axis -- differs ONLY by the prior spatial-smoothing step."""
    if field is None:
        return {"beta": (float("nan"), float("nan")), "unit": (float("nan"), float("nan")),
                "n": 0, "ok": False}
    mask = field["S"] >= s_thresh
    return weighted_ls_gradient_2d(field["T"][mask], X[mask], Y[mask], field["S"][mask])


def sequence_axis(cav: dict, plane_xy: dict) -> dict:
    """COORDINATE-BLIND direction: group contacts by shaft (parse_shaft, electrode identity
    only), collapse every contact's value to its OWN SHAFT's support-weighted mean value (so
    within-shaft spatial position carries zero information -- two contacts on the same shaft
    get the identical value regardless of where on the shaft they sit), then feed that
    shaft-collapsed value into the SAME weighted_ls_gradient_2d estimator raw_contact_axis uses.
    If early contacts happen to sit on one shaft and late contacts on another (the 1146 failure
    mode), this reads as the inter-shaft direction rather than the true within-shaft gradient."""
    names = [n for n in cav if n in plane_xy and np.isfinite(cav[n]["value"])]
    shaft_of = {n: parse_shaft(n)[0] for n in names}
    by_shaft: dict = {}
    for n in names:
        by_shaft.setdefault(shaft_of[n], []).append(n)
    shaft_mean = {}
    for s, ns in by_shaft.items():
        vals = np.array([cav[n]["value"] for n in ns], float)
        ws = np.array([cav[n]["support"] for n in ns], float)
        wsum = float(ws.sum())
        shaft_mean[s] = float(np.sum(vals * ws) / wsum) if wsum > 0 else float(np.mean(vals))
    seq_val = np.array([shaft_mean[shaft_of[n]] for n in names], float)
    w = np.array([cav[n]["support"] for n in names], float)
    x = np.array([plane_xy[n][0] for n in names], float)
    y = np.array([plane_xy[n][1] for n in names], float)
    out = weighted_ls_gradient_2d(seq_val, x, y, w)
    out["shaft_mean"] = shaft_mean
    out["n_shafts"] = len(by_shaft)
    return out


def cos_unit(u1, u2) -> float:
    a = np.asarray(u1, float); b = np.asarray(u2, float)
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    return float(np.dot(a, b))


# ------------------------------------------------------------------- held-out axis score (main)
def held_out_axis_scores(bundle: dict, plane_ref: dict, label: int, *, X, Y, sigma,
                         n_split: int, rng: np.random.Generator, s_thresh: float = S_THRESH) -> list:
    """For ONE class label (TA or TB): n_split random halves of THAT class's events.
    train-half -> raw_contact_axis + field_axis (same sigma/grid/s_thresh as the primary stat
    path, §3.1 P0). held-out-half -> per-contact mean rank. score = Spearman(each contact's
    REAL-position projection onto the train axis, that contact's held-out mean rank).
    Returns a list of {"raw_rho","field_rho","n_common"} dicts, one per usable split (splits
    that can't form BOTH a raw and a field axis, or that leave <MIN_POINTS common contacts,
    are silently dropped from this descriptive pilot list -- not a formal accountability tier)."""
    masked = bundle["masked"]; names = list(bundle["channel_names"])
    labels = np.asarray(bundle["labels"])
    cols = np.where(labels == label)[0]
    plane_xy = build_plane_xy(plane_ref)
    out = []
    for _ in range(n_split):
        perm = rng.permutation(cols)
        half = perm.size // 2
        if half < 1:
            continue
        train_cav = _aggregate_over_events(masked, names, perm[:half])
        held_cav = _aggregate_over_events(masked, names, perm[half:])

        rc = raw_contact_axis(train_cav, plane_xy)
        if not rc["ok"]:
            continue
        v = {n: d["value"] for n, d in train_cav.items()}
        s = {n: d["support"] for n, d in train_cav.items()}
        field = field_from_contact_values(plane_ref, v, support_by_name=s, sigma=sigma,
                                          X=X, Y=Y, s_thresh=s_thresh)
        fa = field_axis_from_field(field, X, Y, s_thresh)
        if not fa["ok"]:
            continue

        common = [n for n in names
                  if n in plane_xy
                  and train_cav[n]["support"] > 0
                  and np.isfinite(held_cav[n]["value"])]
        if len(common) < MIN_POINTS:
            continue
        xs = np.array([plane_xy[n][0] for n in common], float)
        ys = np.array([plane_xy[n][1] for n in common], float)
        held = np.array([held_cav[n]["value"] for n in common], float)
        proj_raw = xs * rc["unit"][0] + ys * rc["unit"][1]
        proj_field = xs * fa["unit"][0] + ys * fa["unit"][1]
        raw_rho = spearmanr(proj_raw, held).correlation
        field_rho = spearmanr(proj_field, held).correlation
        if np.isfinite(raw_rho) and np.isfinite(field_rho):
            out.append({"raw_rho": float(raw_rho), "field_rho": float(field_rho),
                        "n_common": len(common)})
    return out


# --------------------------------------------------------------------------------- per-subject
def run_subject(ds_sid: str, *, X, Y, rng: np.random.Generator, n_split: int) -> dict:
    dataset, subject = ds_sid.split("_", 1)
    out = {"ds_sid": ds_sid}

    ta_f = GEOM_DIR / f"{ds_sid}_t_a.json"
    tb_f = GEOM_DIR / f"{ds_sid}_t_b.json"
    if not (ta_f.exists() and tb_f.exists()):
        out["reason"] = "no_planes"
        return out
    bundle = load_event_labels_ranks(dataset, subject, broad=True)
    plane_a = json.load(open(ta_f)); plane_b = json.load(open(tb_f))
    if "channels" not in plane_a or "channels" not in plane_b:
        out["reason"] = "plane_not_built"
        return out

    order = bundle["channel_names"]
    ta_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_a["channels"]}, order)
    tb_rank = _vec_in_order({c["name"]: c["typical_rank"] for c in plane_b["channels"]}, order)
    c0 = np.asarray(bundle["cluster_template_ranks"][0], float)
    c1 = np.asarray(bundle["cluster_template_ranks"][1], float)
    cmap = map_clusters_to_templates(c0, c1, ta_rank, tb_rank)
    if cmap["ambiguous"]:
        out["reason"] = "cluster_map_ambiguous"
        return out
    plane_ref, ta_label, tb_label = pick_reference(cmap, plane_a, plane_b)
    plane_xy = build_plane_xy(plane_ref)
    sigma = class_template_sigma(plane_ref, X=X, Y=Y)     # P0: ONE sigma, from geometry only

    out.update(reason="ok", n_channels=plane_a["n_channels"],
               poor_planarity=bool(plane_a["flags"].get("poor_planarity")),
               n_shafts_on_plane=len(set(parse_shaft(n)[0] for n in plane_xy)),
               sigma=float(sigma))

    per_class = {}
    pooled_raw, pooled_field = [], []
    for label, cls_name in [(ta_label, "TA"), (tb_label, "TB")]:
        cav = class_aggregate_contact_values(bundle, label)
        rc = raw_contact_axis(cav, plane_xy)
        seq = sequence_axis(cav, plane_xy)
        v = {n: d["value"] for n, d in cav.items()}
        s = {n: d["support"] for n, d in cav.items()}
        field = field_from_contact_values(plane_ref, v, support_by_name=s, sigma=sigma, X=X, Y=Y)
        fa = field_axis_from_field(field, X, Y)

        splits = held_out_axis_scores(bundle, plane_ref, label, X=X, Y=Y, sigma=sigma,
                                      n_split=n_split, rng=rng)
        raw_rhos = [d["raw_rho"] for d in splits]
        field_rhos = [d["field_rho"] for d in splits]
        pooled_raw.extend(raw_rhos); pooled_field.extend(field_rhos)

        per_class[cls_name] = {
            "raw_contact_unit": rc["unit"], "field_unit": fa["unit"], "sequence_unit": seq["unit"],
            "n_shafts": seq.get("n_shafts"), "shaft_mean_value": seq.get("shaft_mean"),
            "cos_sequence_field": cos_unit(seq["unit"], fa["unit"]),
            "cos_raw_field": cos_unit(rc["unit"], fa["unit"]),
            "held_out_n_splits": len(splits),
            "held_out_raw_median": float(np.median(raw_rhos)) if raw_rhos else float("nan"),
            "held_out_field_median": float(np.median(field_rhos)) if field_rhos else float("nan"),
            "held_out_field_beats_raw_frac": (
                float(np.mean([f > r for f, r in zip(field_rhos, raw_rhos)])) if raw_rhos else float("nan")),
        }
    out["per_class"] = per_class
    out["held_out_pooled_n_splits"] = len(pooled_raw)
    out["held_out_pooled_raw_median"] = float(np.median(pooled_raw)) if pooled_raw else float("nan")
    out["held_out_pooled_field_median"] = float(np.median(pooled_field)) if pooled_field else float("nan")
    out["held_out_pooled_field_beats_raw_frac"] = (
        float(np.mean([f > r for f, r in zip(pooled_field, pooled_raw)])) if pooled_raw else float("nan"))
    return out


def main():
    X, Y = make_plane_grid()
    rng = np.random.default_rng(RNG_SEED)
    results = []
    for ds_sid in SUBJECTS:
        print(f"[run] {ds_sid} ...", flush=True)
        res = run_subject(ds_sid, X=X, Y=Y, rng=rng, n_split=N_SPLIT)
        print(f"    reason={res.get('reason')}", flush=True)
        results.append(res)

    out_path = _ROOT / ".superpowers/sdd/pilot_axis_robustness_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"[done] wrote {out_path}")

    for r in results:
        if r.get("reason") != "ok":
            continue
        print(f"\n=== {r['ds_sid']} (n_ch={r['n_channels']}, poor_planarity={r['poor_planarity']}, "
              f"n_shafts={r['n_shafts_on_plane']}) ===")
        print(f"  pooled held-out (TA+TB, n_splits={r['held_out_pooled_n_splits']}): "
              f"raw_median={r['held_out_pooled_raw_median']:.3f}  "
              f"field_median={r['held_out_pooled_field_median']:.3f}  "
              f"field_beats_raw_frac={r['held_out_pooled_field_beats_raw_frac']:.2f}")
        for cls_name, d in r["per_class"].items():
            print(f"  [{cls_name}] cos(seq,field)={d['cos_sequence_field']:.3f}  "
                  f"cos(raw,field)={d['cos_raw_field']:.3f}  n_shafts={d['n_shafts']}  "
                  f"held_out raw={d['held_out_raw_median']:.3f} field={d['held_out_field_median']:.3f} "
                  f"(n={d['held_out_n_splits']})")


if __name__ == "__main__":
    main()
