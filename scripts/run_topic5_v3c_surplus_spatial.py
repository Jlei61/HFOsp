"""V3c-3 (secondary descriptive): spatial organization of axis-surplus A∖S.
Shaft spread + contiguous runs + distance-to-SOZ (coords-gated) vs same-shaft null.
Emits surplus_spatial_cohort.json (cohort-median distance null) for the coverage
DOUBLE-condition in Task 14 (spec §4.4).
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._topic5_v3_io import classify_subject_contacts
from scripts._topic5_v3c_io import V3C_SUBJECTS, axis_soz_join, load_axis_coords, load_soz
from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_coverage import distance_null_distribution, surplus_spatial_metrics

OUT = _ROOT / "results/topic5_ictal_recruitment/v3c_soz_axis_coverage"
COLS = ["subject", "cohort", "n_surplus", "n_shafts_with_surplus", "shaft_gini",
        "max_contiguous_run", "mean_min_dist_to_soz", "dist_null_p"]


def surplus_row(ds_sid, cohort, cfg) -> dict:
    dataset, subj = ds_sid.split("_", 1)
    row = {c: float("nan") for c in COLS}; row.update({"subject": ds_sid, "cohort": cohort})
    try:
        cls = classify_subject_contacts(ds_sid, cohort, cfg)
        j = axis_soz_join(cls, load_soz(dataset, subj))
        coords = load_axis_coords(dataset, subj, cls["all_clean"])
        m = surplus_spatial_metrics(j["surplus"], j["soz_in_pool"], coords, cls["shaft_by_name"])
        null = distance_null_distribution(j["surplus"], cls["all_clean"], j["soz_in_pool"], coords,
                                          cls["shaft_by_name"], n_perm=cfg["v3c"]["nulls"]["n_perm"],
                                          rng=cfg["v3c"]["nulls"]["seed"])
        # lower observed distance than null -> surplus closer to SOZ than random -> structured
        p = (float((np.sum(null <= m["mean_min_dist_to_soz"]) + 1) / (null.size + 1))
             if null.size and np.isfinite(m["mean_min_dist_to_soz"]) else float("nan"))
        row.update({"n_surplus": j["n_surplus"], **{k: m[k] for k in
                    ("n_shafts_with_surplus", "shaft_gini", "max_contiguous_run", "mean_min_dist_to_soz")},
                    "dist_null_p": p, "_null": null, "_obs": m["mean_min_dist_to_soz"]})
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {ds_sid} ({cohort}): {type(exc).__name__}: {exc}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=["broad", "narrow"], required=True)
    args = ap.parse_args()
    cfg = load_v3_config()
    outdir = OUT / args.cohort / "surplus_spatial"; outdir.mkdir(parents=True, exist_ok=True)
    rows = [surplus_row(s, args.cohort, cfg) for s in V3C_SUBJECTS[args.cohort]]
    with open(outdir / "surplus_subject.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows: w.writerow({c: r[c] for c in COLS})

    # P1-1: cohort-median distance null feeds the coverage DOUBLE-condition (spec §4.4).
    # A subject is spatial-eligible only if it has coords -> a finite distance null.
    elig = [r for r in rows if isinstance(r.get("_null"), np.ndarray) and r["_null"].size
            and np.isfinite(r.get("_obs", float("nan")))]
    cohort = {"n_spatial_eligible": len(elig), "n_subjects": len(rows)}
    if elig:
        n_perm = min(r["_null"].size for r in elig)
        perm_med = np.median(np.vstack([r["_null"][:n_perm] for r in elig]), axis=0)
        obs = float(np.median([r["_obs"] for r in elig]))
        cohort.update({
            "obs_cohort_median_dist": obs, "n_perm": int(n_perm),
            "p_value": float((np.sum(perm_med <= obs) + 1) / (n_perm + 1)),   # lower-tail (closer = structured)
            "dist_null_q05": float(np.percentile(perm_med, 5)),
            "dist_null_q50": float(np.percentile(perm_med, 50)),
            "dist_null_q95": float(np.percentile(perm_med, 95)),
        })
        # LOSO only defined with >=2 subjects (review P2: leave-one-out of 1 -> empty vstack)
        if len(elig) >= 2:
            cohort["loso"] = [
                {"dropped": elig[k]["subject"],
                 "p_value": float((np.sum(
                     np.median(np.vstack([r["_null"][:n_perm] for i, r in enumerate(elig) if i != k]), axis=0)
                     <= np.median([r["_obs"] for i, r in enumerate(elig) if i != k])) + 1) / (n_perm + 1))}
                for k in range(len(elig))]
            cohort["loso_status"] = "ok"
        else:
            cohort["loso"] = []
            cohort["loso_status"] = "not_enough_subjects"
    (outdir / "surplus_spatial_cohort.json").write_text(json.dumps(cohort, indent=2))
    print(f"[done] {args.cohort} surplus-spatial ({len(rows)} subjects, {len(elig)} spatial-eligible); "
          f"cohort_dist_p={cohort.get('p_value')}")


if __name__ == "__main__":
    main()
