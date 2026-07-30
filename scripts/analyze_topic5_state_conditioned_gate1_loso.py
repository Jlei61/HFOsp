#!/usr/bin/env python
"""Canonical Gate 1: nested-LOSO static interictal scaffold readout.

The shared mapping is learned from other patients only. A held-out patient's
prediction uses its prefix-only TA/TB/support features and target-independent
electrode coordinates; every seizure label remains hidden until scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft


def weighted_center_norm(y, q):
    y = np.asarray(y, float)
    q = np.asarray(q, float)
    good = np.isfinite(y) & np.isfinite(q) & (q > 0)
    out = np.full_like(y, np.nan)
    if np.sum(good) < 4:
        return out
    w = q[good] / q[good].sum()
    centered = y[good] - np.sum(w * y[good])
    scale = np.sqrt(np.sum(w * centered**2))
    if scale <= 1e-12:
        return out
    out[good] = centered / scale
    return out


def weighted_corr(a, b, q):
    aa = weighted_center_norm(a, q)
    bb = weighted_center_norm(b, q)
    good = np.isfinite(aa) & np.isfinite(bb) & (np.asarray(q) > 0)
    if np.sum(good) < 4:
        return np.nan
    w = np.asarray(q)[good]
    w = w / w.sum()
    return float(np.sum(w * aa[good] * bb[good]))


def frozen_coordinates(subject, names):
    """Target-independent shaft geometry; never read a full-record template plane."""
    parsed = [parse_shaft(name) for name in names]
    groups = {}
    for i, (shaft, number) in enumerate(parsed):
        groups.setdefault(shaft or "__none__", []).append((i, number))
    out = np.zeros((len(names), 3), dtype=float)
    n_shafts = max(len(groups), 1)
    for shaft_order, (_shaft, members) in enumerate(sorted(groups.items())):
        numbers = np.asarray(
            [float(number) if number is not None else float(j) for j, (_i, number) in enumerate(members)]
        )
        center = float(np.mean(numbers))
        scale = float(np.std(numbers)) or 1.0
        for (idx, _number), value in zip(members, numbers):
            out[idx, 0] = (value - center) / scale
            out[idx, 1] = len(members) / max(len(names), 1)
            out[idx, 2] = shaft_order / max(n_shafts - 1, 1)
    return out


def load_subject(dataset, targets, subject):
    with np.load(dataset / "per_subject" / f"{subject}.npz", allow_pickle=True) as z:
        arrays = {key: z[key] for key in z.files}
    axis = json.loads((dataset / "per_subject" / f"{subject}.json").read_text())
    names = [str(x) for x in arrays["channel_names"]]
    q = np.asarray(arrays["support_q"], float)
    ta = np.asarray(arrays["template_a"], float)
    tb = np.asarray(arrays["template_b"], float)
    sa = np.asarray(axis["support_a"], float)
    sb = np.asarray(axis["support_b"], float)
    early_a = 1.0 - ta
    early_b = 1.0 - tb
    geometry = frozen_coordinates(subject, names)
    support = np.c_[q, 0.5 * (sa + sb), np.abs(sa - sb)]
    scaffold = np.c_[
        geometry,
        support,
        0.5 * (early_a + early_b),
        np.abs(early_a - early_b),
        early_a * early_b,
    ]
    geometry_support = np.c_[geometry, support]
    subject_targets = targets[
        (targets.subject == subject) & np.isfinite(targets.target_low_1_8)
    ]
    fields = []
    for row in subject_targets.itertuples():
        key = f"field_low_1_8__{int(row.seizure_idx)}"
        if key in arrays:
            fields.append(weighted_center_norm(np.asarray(arrays[key], float), q))
    if not fields:
        raise ValueError(f"{subject}: no primary field")
    mean_field = weighted_center_norm(np.nanmean(np.stack(fields), axis=0), q)
    valid = (
        np.isfinite(mean_field)
        & np.isfinite(scaffold).all(axis=1)
        & np.isfinite(geometry_support).all(axis=1)
        & (q > 0)
    )
    return {
        "subject": subject,
        "names": names,
        "q": q,
        "scaffold": scaffold,
        "geometry_support": geometry_support,
        "target": mean_field,
        "valid": valid,
        "n_seizures": len(fields),
        "rank_feature_start": geometry_support.shape[1],
    }


def fit_model(records, feature_key, alpha):
    X, y, weight = [], [], []
    for record in records:
        valid = record["valid"]
        xx = record[feature_key][valid]
        yy = record["target"][valid]
        X.append(xx)
        y.append(yy)
        weight.append(np.full(len(yy), 1.0 / len(yy)))
    X = np.vstack(X)
    y = np.concatenate(y)
    weight = np.concatenate(weight)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(X), y, sample_weight=weight
    )
    return scaler, model


def predict(record, feature_key, scaler, model, override=None):
    X = record[feature_key] if override is None else override
    pred = np.full(len(record["names"]), np.nan)
    valid = record["valid"]
    pred[valid] = model.predict(scaler.transform(X[valid]))
    return pred


def choose_alpha(records, feature_key, alphas):
    scores = []
    for alpha in alphas:
        fold = []
        for heldout in records:
            train = [record for record in records if record["subject"] != heldout["subject"]]
            scaler, model = fit_model(train, feature_key, alpha)
            pred = predict(heldout, feature_key, scaler, model)
            fold.append(weighted_corr(pred, heldout["target"], heldout["q"]))
        scores.append(float(np.nanmean(fold)))
    return float(alphas[int(np.nanargmax(scores))]), scores


def permutation_index(names, rng, within_shaft):
    idx = np.arange(len(names))
    if not within_shaft:
        return rng.permutation(idx)
    out = idx.copy()
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(parse_shaft(name)[0], []).append(i)
    for group in groups.values():
        if len(group) >= 2:
            out[group] = rng.permutation(group)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = (
        args.dataset_dir
        if args.dataset_dir is not None and args.dataset_dir.is_absolute()
        else ROOT / args.dataset_dir
        if args.dataset_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    out = dataset / "gate1_static_scaffold_loso"
    out.mkdir(parents=True, exist_ok=True)
    attr = pd.read_csv(dataset / "gate0_attrition.csv")
    passed = attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    subjects = attr.loc[passed, "subject"].astype(str).tolist()
    targets = pd.read_csv(dataset / "seizure_targets.csv")
    records = [load_subject(dataset, targets, subject) for subject in subjects]
    alpha_grid = cfg["validation"]["probe_alpha_grid"]
    rows, full_draws, shaft_draws = [], [], []
    rng = np.random.default_rng(args.seed)
    for heldout in records:
        train = [r for r in records if r["subject"] != heldout["subject"]]
        alpha_static, inner_static = choose_alpha(train, "scaffold", alpha_grid)
        alpha_geo, inner_geo = choose_alpha(train, "geometry_support", alpha_grid)
        static_scaler, static_model = fit_model(train, "scaffold", alpha_static)
        geo_scaler, geo_model = fit_model(train, "geometry_support", alpha_geo)
        static_pred = predict(heldout, "scaffold", static_scaler, static_model)
        geo_pred = predict(heldout, "geometry_support", geo_scaler, geo_model)
        static_r = weighted_corr(static_pred, heldout["target"], heldout["q"])
        geo_r = weighted_corr(geo_pred, heldout["target"], heldout["q"])
        null_full = np.full(args.n_perm, np.nan)
        null_shaft = np.full(args.n_perm, np.nan)
        rank_start = int(heldout["rank_feature_start"])
        for draw in range(args.n_perm):
            for within, output in ((False, null_full), (True, null_shaft)):
                order = permutation_index(heldout["names"], rng, within)
                shuffled = heldout["scaffold"].copy()
                shuffled[:, rank_start:] = shuffled[order, rank_start:]
                pred = predict(
                    heldout, "scaffold", static_scaler, static_model, override=shuffled
                )
                output[draw] = weighted_corr(pred, heldout["target"], heldout["q"])
        rows.append(
            {
                "subject": heldout["subject"],
                "n_contacts": int(np.sum(heldout["valid"])),
                "n_seizures": heldout["n_seizures"],
                "static_loso_r": static_r,
                "geometry_support_loso_r": geo_r,
                "increment_over_geometry": static_r - geo_r,
                "alpha_static": alpha_static,
                "alpha_geometry": alpha_geo,
                "full_contact_null_median": float(np.nanmedian(null_full)),
                "within_shaft_null_median": float(np.nanmedian(null_shaft)),
            }
        )
        full_draws.append(null_full)
        shaft_draws.append(null_shaft)
    table = pd.DataFrame(rows)
    table.to_csv(out / "subject_level.csv", index=False)
    observed = float(table.static_loso_r.median())
    full_cohort = np.nanmedian(np.vstack(full_draws), axis=0)
    shaft_cohort = np.nanmedian(np.vstack(shaft_draws), axis=0)
    p_full = float((1 + np.sum(full_cohort >= observed)) / (args.n_perm + 1))
    p_shaft = float((1 + np.sum(shaft_cohort >= observed)) / (args.n_perm + 1))
    delta = table.increment_over_geometry.to_numpy(float)
    nonzero = delta[np.abs(delta) > 1e-12]
    p_geometry = (
        float(wilcoxon(nonzero, alternative="greater").pvalue)
        if len(nonzero)
        else np.nan
    )
    verdict = {
        "contract": cfg["contract"]["name"],
        "gate": 1,
        "method": "nested-LOSO shared static mapping; held-out patient contributes no seizure labels",
        "n_subjects": len(table),
        "n_seizures": int(table.n_seizures.sum()),
        "static_scaffold_cohort_median_r": observed,
        "geometry_support_cohort_median_r": float(table.geometry_support_loso_r.median()),
        "median_increment_over_geometry": float(table.increment_over_geometry.median()),
        "paired_wilcoxon_static_greater_geometry_p": p_geometry,
        "full_contact_rank_shuffle_null_median": float(np.nanmedian(full_cohort)),
        "full_contact_rank_shuffle_p_greater": p_full,
        "within_shaft_rank_shuffle_null_median": float(np.nanmedian(shaft_cohort)),
        "within_shaft_rank_shuffle_p_greater": p_shaft,
        "gate1_pass": bool(
            observed > np.nanmedian(full_cohort)
            and p_full < 0.05
            and np.nanmedian(delta) > 0
        ),
        "gate1_strong_pass": bool(
            observed > np.nanmedian(shaft_cohort)
            and p_shaft < 0.05
            and np.nanmedian(delta) > 0
            and np.isfinite(p_geometry)
            and p_geometry < 0.05
        ),
        "claim_boundary": "static interictal scaffold readability only",
    }
    (out / "gate1_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
