#!/usr/bin/env python
"""Gate 1: test the prefix-only static scaffold against spatial nulls."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft


def weighted_corr(a, b, weights):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(weights, float)
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(w) & (w > 0)
    if np.sum(good) < 4:
        return np.nan
    a, b, w = a[good], b[good], w[good]
    w = w / w.sum()
    ac = a - np.sum(w * a)
    bc = b - np.sum(w * b)
    den = np.sqrt(np.sum(w * ac * ac) * np.sum(w * bc * bc))
    return float(np.sum(w * ac * bc) / den) if den > 1e-12 else np.nan


def permutation_indices(names, rng, within_shaft: bool):
    idx = np.arange(len(names))
    if not within_shaft:
        return rng.permutation(idx)
    out = idx.copy()
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(parse_shaft(str(name))[0], []).append(i)
    for group in groups.values():
        if len(group) >= 2:
            out[group] = rng.permutation(group)
    return out


def subject_scores(subject, rows, arrays, n_perm, seed):
    names = [str(x) for x in arrays["channel_names"]]
    q = np.asarray(arrays["support_q"], float)
    ta = np.asarray(arrays["template_a"], float)
    tb = np.asarray(arrays["template_b"], float)
    # Earlier prefix rank predicts stronger early recruitment.
    static = -0.5 * (ta + tb)
    fields = []
    observed = []
    used_rows = []
    for row in rows.itertuples():
        key = f"field_low_1_8__{int(row.seizure_idx)}"
        if key not in arrays:
            continue
        field = np.asarray(arrays[key], float)
        score = weighted_corr(static, field, q)
        if np.isfinite(score):
            fields.append(field)
            observed.append(score)
            used_rows.append(row)
    if not observed:
        return None, []
    rng = np.random.default_rng(seed)
    null_full = np.full((n_perm, len(fields)), np.nan)
    null_shaft = np.full((n_perm, len(fields)), np.nan)
    for draw in range(n_perm):
        full_idx = permutation_indices(names, rng, False)
        shaft_idx = permutation_indices(names, rng, True)
        for j, field in enumerate(fields):
            null_full[draw, j] = weighted_corr(static[full_idx], field, q)
            null_shaft[draw, j] = weighted_corr(static[shaft_idx], field, q)
    obs_subject = float(np.median(observed))
    full_subject = np.nanmedian(null_full, axis=1)
    shaft_subject = np.nanmedian(null_shaft, axis=1)
    summary = {
        "subject": subject,
        "n_seizures": len(observed),
        "observed_median_r": obs_subject,
        "full_contact_null_median": float(np.nanmedian(full_subject)),
        "full_contact_p_greater": float(
            (1 + np.sum(full_subject >= obs_subject)) / (n_perm + 1)
        ),
        "within_shaft_null_median": float(np.nanmedian(shaft_subject)),
        "within_shaft_p_greater": float(
            (1 + np.sum(shaft_subject >= obs_subject)) / (n_perm + 1)
        ),
        "_full_draws": full_subject,
        "_shaft_draws": shaft_subject,
    }
    event_rows = []
    for row, score in zip(used_rows, observed):
        event_rows.append(
            {
                "dataset": row.dataset,
                "subject": subject,
                "seizure_idx": int(row.seizure_idx),
                "static_scaffold_r": score,
            }
        )
    return summary, event_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dataset = ROOT / cfg["outputs"]["dataset"]
    out = dataset / "gate1_static_scaffold"
    out.mkdir(parents=True, exist_ok=True)
    attr = pd.read_csv(dataset / "gate0_attrition.csv")
    subjects = attr[
        attr.gate0_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    ].subject.astype(str)
    targets = pd.read_csv(dataset / "seizure_targets.csv")
    subject_rows, event_rows, full_draws, shaft_draws = [], [], [], []
    for offset, subject in enumerate(subjects):
        with np.load(dataset / "per_subject" / f"{subject}.npz", allow_pickle=True) as z:
            arrays = {key: z[key] for key in z.files}
        rows = targets[
            (targets.subject == subject) & np.isfinite(targets.target_low_1_8)
        ]
        summary, events = subject_scores(
            subject, rows, arrays, args.n_perm, args.seed + offset * 1009
        )
        if summary is None:
            continue
        full_draws.append(summary.pop("_full_draws"))
        shaft_draws.append(summary.pop("_shaft_draws"))
        subject_rows.append(summary)
        event_rows.extend(events)
    subject_df = pd.DataFrame(subject_rows)
    subject_df.to_csv(out / "subject_level.csv", index=False)
    pd.DataFrame(event_rows).to_csv(out / "seizure_level.csv", index=False)
    observed = float(subject_df.observed_median_r.median()) if len(subject_df) else np.nan
    full_cohort = np.nanmedian(np.vstack(full_draws), axis=0) if full_draws else np.array([])
    shaft_cohort = np.nanmedian(np.vstack(shaft_draws), axis=0) if shaft_draws else np.array([])
    p_full = (
        float((1 + np.sum(full_cohort >= observed)) / (len(full_cohort) + 1))
        if full_cohort.size
        else np.nan
    )
    p_shaft = (
        float((1 + np.sum(shaft_cohort >= observed)) / (len(shaft_cohort) + 1))
        if shaft_cohort.size
        else np.nan
    )
    verdict = {
        "contract": cfg["contract"]["name"],
        "gate": 1,
        "n_subjects": len(subject_df),
        "n_seizures": len(event_rows),
        "observed_cohort_median_r": observed,
        "full_contact_null_median": float(np.nanmedian(full_cohort)) if full_cohort.size else np.nan,
        "full_contact_cohort_p_greater": p_full,
        "within_shaft_null_median": float(np.nanmedian(shaft_cohort)) if shaft_cohort.size else np.nan,
        "within_shaft_cohort_p_greater": p_shaft,
        "gate1_pass": bool(np.isfinite(p_full) and p_full < 0.05 and observed > np.nanmedian(full_cohort)),
        "primary_null": "coherent subject-level full-contact template-rank shuffle",
        "within_shaft": "anatomy-controlled sensitivity",
        "claim_boundary": "static interictal scaffold readability; not seizure-specific dynamic prediction",
    }
    (out / "gate1_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
