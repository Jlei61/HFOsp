"""V3c-1: interictal-axis coverage of clinical SOZ (primary). Subject-first,
same-shaft null, cohort-median null + LOSO. broad primary / narrow sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts  # noqa: E402
from scripts._topic5_v3c_io import V3C_SUBJECTS, axis_soz_join, load_soz  # noqa: E402
from src.topic5_v3_mode_transition import load_v3_config  # noqa: E402
from src.topic5_v3c_coverage import coverage_null_distribution  # noqa: E402

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_axis", "n_soz", "n_covered", "n_surplus", "n_missed",
        "coverage", "surplus_fraction", "jaccard", "coverage_null_p",
        "null_q05", "null_q50", "null_q95", "eligible"]   # q05/q50/q95 for the forest figure


def coverage_subject_row(ds_sid: str, cohort: str, cfg: dict) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    row = {c: float("nan") for c in COLS}
    row.update({"subject": ds_sid, "cohort": cohort, "eligible": False})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        vc = cfg["v3c"]
        eligible = j["n_soz"] >= vc["coverage"]["min_soz"] and j["n_axis"] >= vc["coverage"]["min_axis"]
        null = coverage_null_distribution(
            cls["is_axis"], cls["all_clean"], j["soz_in_pool"], cls["shaft_by_name"],
            n_perm=vc["nulls"]["n_perm"], rng=vc["nulls"]["seed"],
        ) if eligible else np.array([])
        p = float((np.sum(null >= j["coverage"]) + 1) / (null.size + 1)) if null.size else float("nan")
        q05, q50, q95 = (np.percentile(null, [5, 50, 95]) if null.size else (np.nan, np.nan, np.nan))
        row.update({k: j[k] for k in ("n_axis", "n_soz", "n_covered", "n_surplus", "n_missed",
                                      "coverage", "surplus_fraction", "jaccard")})
        row.update({"coverage_null_p": p, "null_q05": float(q05), "null_q50": float(q50),
                    "null_q95": float(q95), "eligible": bool(eligible),
                    "_null": null, "_obs": j["coverage"]})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def cohort_median_null(subject_obs: list, subject_nulls: list) -> dict:
    """Nested subject-level (cohort-median) null (spec §7): per perm take the median
    across subjects of that perm's null coverage; compare to the observed cohort median.
    """
    obs_med = float(np.median(subject_obs))
    n_perm = min(len(n) for n in subject_nulls)
    stacked = np.vstack([n[:n_perm] for n in subject_nulls])   # (n_subj, n_perm)
    perm_medians = np.median(stacked, axis=0)                  # (n_perm,)
    p = float((np.sum(perm_medians >= obs_med) + 1) / (n_perm + 1))
    return {"obs_cohort_median": obs_med, "p_value": p, "n_perm": n_perm,
            "n_subjects": len(subject_obs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [coverage_subject_row(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]

    elig = [r for r in rows if r.get("eligible")]
    cohort = {}
    if elig:
        cohort = cohort_median_null([r["_obs"] for r in elig], [r["_null"] for r in elig])
        # LOSO only defined with >=2 subjects (review P2: leave-one-out of 1 -> empty vstack)
        if len(elig) >= 2:
            cohort["loso"] = [
                {"dropped": elig[k]["subject"],
                 **cohort_median_null([r["_obs"] for i, r in enumerate(elig) if i != k],
                                      [r["_null"] for i, r in enumerate(elig) if i != k])}
                for k in range(len(elig))]
            cohort["loso_status"] = "ok"
        else:
            cohort["loso"] = []
            cohort["loso_status"] = "not_enough_subjects"
        cohort["n_pass_own_null"] = int(sum(r["coverage_null_p"] < cfg["v3c"]["nulls"].get("alpha", 0.05)
                                            for r in elig if np.isfinite(r["coverage_null_p"])))

    with open(outdir / "coverage_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLS})
    (outdir / "coverage_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort}: {len(elig)}/{len(rows)} eligible; cohort={cohort.get('p_value')}")


if __name__ == "__main__":
    main()
