#!/usr/bin/env python
"""Cohort tables for the Raw-SEEG evolvable prediction-state model (R0.1).

Patients are the unit. Every cohort number is a statistic over per-patient
values; a patient with ten times more usable minutes does not get ten times the
weight, and each row keeps its own denominator so no "cohort n" of minutes is
ever formed (execution plan section 9).

Writes into ``results/epi_prssm/raw_seeg_state/r0_1/``:
    cohort_horizon_metrics.csv   one row per (subject, horizon, arm)
    cohort_state_swap.csv        one row per (subject, horizon)
    cohort_consistency.csv       one row per subject
    COHORT_SUMMARY.json          cohort statistics + explicit claim boundaries

Example
-------
    LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:$LD_LIBRARY_PATH \
    /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
      scripts/topic5_raw_seeg_state/aggregate_cohort.py --arm full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_raw_seeg_state import analysis, contract  # noqa: E402

#: Which per-subject file supplies which arm's MSE.
_MODEL_ARMS = {"model": "full", "identity_dynamics": "identity"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-subject-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--arm", default="full", help="which arm is the reported model")
    p.add_argument("--identity-suffix", default="__identity",
                   help="directory suffix holding the identity-dynamics arm")
    return p


def _read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def collect_subject(sub_dir: Path, identity_suffix: str):
    """Assemble one subject's per-horizon MSEs for all five arms."""
    metrics = _read_json(sub_dir / "validation_horizon_metrics.json")
    if metrics is None:
        return None
    baseline = _read_json(sub_dir / "baseline_metrics.json") or {}
    ident = _read_json(sub_dir.with_name(sub_dir.name + identity_suffix)
                       / "validation_horizon_metrics.json") or {}
    per_h = {}
    for key, entry in metrics["per_horizon"].items():
        h = int(key)
        row = {"model": float(entry["model_mse"]),
               "persistence": float(entry["persistence_mse"]),
               "n_windows": int(entry["n_windows"]),
               "n_elements": int(entry["n_elements"])}
        bh = (baseline.get("per_horizon") or {}).get(str(h))
        if bh:
            row["patient_mean"] = float(bh["patient_mean"])
            row["feature_ar"] = float(bh["feature_ar"])
            if abs(float(bh["persistence"]) - row["persistence"]) > 1e-6:
                raise ValueError(
                    f"{sub_dir.name} h={h}: persistence differs between the model "
                    f"evaluation ({row['persistence']:.6g}) and the baseline script "
                    f"({float(bh['persistence']):.6g}) -- the two are not on the "
                    "same validation windows"
                )
        ih = (ident.get("per_horizon") or {}).get(str(h))
        if ih:
            row["identity_dynamics"] = float(ih["model_mse"])
        per_h[h] = row

    # contract.EVAL_SET_PRIMARY -- the windows scoreable at every horizon. This
    # is the load-bearing set for the cohort horizon curve; the per-horizon set
    # above is secondary and exists so subjects with no h=100 window still
    # report at h=1/5/10.
    prim_src = (metrics.get(contract.EVAL_SET_PRIMARY) or {}).get("per_horizon") or {}
    prim_base = ((baseline.get(contract.EVAL_SET_PRIMARY) or {}).get("per_horizon") or {})
    prim_ident = ((ident.get(contract.EVAL_SET_PRIMARY) or {}).get("per_horizon") or {})
    per_h_common = {}
    for key, entry in prim_src.items():
        h = int(key)
        if not int(entry.get("n_windows", 0)):
            continue
        row = {"model": float(entry["model_mse"]),
               "persistence": float(entry["persistence_mse"]),
               "n_windows": int(entry["n_windows"]),
               "n_elements": int(entry["n_elements"])}
        bh = prim_base.get(str(h))
        if bh:
            row["patient_mean"] = float(bh["patient_mean"])
            row["feature_ar"] = float(bh["feature_ar"])
        ih2 = prim_ident.get(str(h))
        if ih2:
            row["identity_dynamics"] = float(ih2["model_mse"])
        per_h_common[h] = row

    return {"subject": sub_dir.name, "selected_epoch": metrics.get("selected_epoch"),
            "per_horizon": per_h, "per_horizon_common": per_h_common,
            "primary_set_empty": bool(metrics.get("primary_set_empty",
                                                  not bool(per_h_common))),
            "e_cons": metrics.get("e_cons", {})}


def main(argv=None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    per_subject_dir = Path(args.per_subject_dir) if args.per_subject_dir else contract.PER_SUBJECT_DIR
    out_dir = Path(args.out_dir) if args.out_dir else contract.RESULT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = {}
    for sub_dir in sorted(p for p in per_subject_dir.iterdir() if p.is_dir()):
        # arm and seed variants live in sibling directories named
        # "<subject>__<arm>[__s<seed>]"; only the bare subject directory is the
        # canonical (full, seed 0) run.
        if "__" in sub_dir.name:
            continue
        info = collect_subject(sub_dir, args.identity_suffix)
        if info is not None:
            subjects[info["subject"]] = info
    if not subjects:
        print(f"no per-subject metrics under {per_subject_dir}", file=sys.stderr)
        return 1

    # PRIMARY table -- the one Figure R2 plots. Subjects whose common set is
    # empty are named, not dropped silently.
    with_primary = {s: v["per_horizon_common"] for s, v in subjects.items()
                    if v["per_horizon_common"]}
    excluded_from_primary = sorted(s for s, v in subjects.items()
                                   if not v["per_horizon_common"])
    if with_primary:
        analysis.horizon_curve(with_primary).to_csv(
            out_dir / "cohort_horizon_metrics.csv", index=False)
    # SECONDARY table -- each horizon on its own windows and its own denominator.
    horizon_frame = analysis.horizon_curve(
        {s: v["per_horizon"] for s, v in subjects.items()})
    horizon_frame.to_csv(out_dir / "cohort_horizon_metrics_per_horizon.csv", index=False)
    if excluded_from_primary:
        print(f"[primary set empty, reported in the secondary table only] "
              f"{', '.join(excluded_from_primary)}", file=sys.stderr)

    swap_rows = []
    for subject, sub_dir in ((s, per_subject_dir / s) for s in sorted(subjects)):
        summary = _read_json(sub_dir / "state_swap_summary.json")
        if not summary:
            continue
        for key, entry in summary["per_horizon"].items():
            swap_rows.append({
                "subject": subject, "horizon_min": int(key),
                "median_dmse": entry["median_dmse"],
                "frac_positive_windows": entry["frac_positive"],
                "sign_test_p_windows": entry["sign_test_p_windows"],
                "n_windows": entry["n_windows"],
                "median_match_distance": summary["match_quality"]["median_distance"],
                "median_match_ratio_to_median": summary["match_quality"]["median_distance_ratio_to_median"],
                "median_separation_minutes": summary["match_quality"]["median_separation_minutes"],
            })
    swap_frame = pd.DataFrame(swap_rows)
    swap_frame.to_csv(out_dir / "cohort_state_swap.csv", index=False)

    cons_rows = []
    for subject in sorted(subjects):
        summary = _read_json(per_subject_dir / subject / "state_consistency_summary.json")
        if not summary:
            summary = dict(subjects[subject]["e_cons"] or {})
            summary["subject"] = subject
        cons_rows.append({
            "subject": subject,
            "n_windows": summary.get("n_windows", summary.get("n", 0)),
            "e_cons_median": summary.get("median", float("nan")),
            "e_cons_q25": summary.get("q25", float("nan")),
            "e_cons_q75": summary.get("q75", float("nan")),
            "frac_below_one": summary.get("frac_below_one", float("nan")),
        })
    cons_frame = pd.DataFrame(cons_rows)
    cons_frame.to_csv(out_dir / "cohort_consistency.csv", index=False)

    # Cohort skill is computed on the PRIMARY set only. Mixing window sets
    # across horizons would let "the far horizon kept only the easy windows"
    # masquerade as skill.
    skill_rows = []
    skill_rows_secondary = []
    for subject, info in subjects.items():
        for key, dest in (("per_horizon_common", skill_rows),
                          ("per_horizon", skill_rows_secondary)):
            for h, row in (info.get(key) or {}).items():
                for arm in analysis.BASELINE_ARMS:
                    if arm not in row:
                        continue
                    dest.append({"subject": subject, "horizon_min": h,
                                 "baseline": arm, "n_windows": row.get("n_windows"),
                                 "skill": analysis.skill_score(row["model"], row[arm])})
    summary = {
        "revision": contract.REVISION,
        "code_revision": contract.code_revision(),
        "package_hash": contract.package_hash(contract.r0_1_source_files()),
        "arm": args.arm,
        "n_subjects": len(subjects),
        "subjects": sorted(subjects),
        "unit_of_analysis": "patient",
        "eval_set_used_for_cohort_claim": contract.EVAL_SET_PRIMARY,
        "n_subjects_in_primary_set": len(with_primary),
        "excluded_from_primary_set": excluded_from_primary,
        "excluded_from_primary_reason": (
            "no validation window is scoreable at every horizon -- the "
            "validation span is shorter than context + longest horizon. These "
            "subjects appear in cohort_horizon_metrics_per_horizon.csv only and "
            "are NOT imputed into the cohort curve."),
        "skill_vs_baseline": {
            baseline: analysis.cohort_summary_from_rows(
                [r for r in skill_rows if r["baseline"] == baseline],
                value_key="skill", group_keys=("horizon_min",), null_value=0.0)
            for baseline in sorted({r["baseline"] for r in skill_rows})
        },
        "skill_vs_baseline_per_horizon_set": {
            baseline: analysis.cohort_summary_from_rows(
                [r for r in skill_rows_secondary if r["baseline"] == baseline],
                value_key="skill", group_keys=("horizon_min",), null_value=0.0)
            for baseline in sorted({r["baseline"] for r in skill_rows_secondary})
        },
        "state_swap": analysis.cohort_summary_from_rows(
            swap_rows, value_key="median_dmse", group_keys=("horizon_min",),
            null_value=0.0) if swap_rows else {},
        "state_consistency": analysis.cohort_statistic(
            [r["e_cons_median"] for r in cons_rows], null_value=1.0),
        "claim_boundary": (
            "Forecast layer and consistency layer are separate statements. A "
            "positive horizon curve with E_cons at or above 1 supports a "
            "forecastable latent code only, not a single evolvable state. "
            "Nothing here speaks to seizure risk, interictal spike generation, "
            "or activity above 100 Hz."
        ),
    }
    contract.atomic_write_json(out_dir / "COHORT_SUMMARY.json", summary)
    print(json.dumps({"n_subjects": summary["n_subjects"],
                      "outputs": sorted(p.name for p in out_dir.glob("cohort_*.csv"))},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
