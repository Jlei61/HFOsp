#!/usr/bin/env python3
"""Cohort statistics for the autonomous-generation round.

Each nested step is scored as "how much better did the richer motif generate the future
than the simpler one it inherited", on held-out events, in nats per decision.  Patients
are the independent unit.

Three things here exist because the previous round got them wrong:

* **A lower bound of exactly zero touches zero.**  A warm-started child reproduces its
  parent bit for bit whenever its new parameter stays at zero, so the differences pile
  up on 0.0 and the bootstrap's lower bound lands exactly there.  Testing ``low < 0``
  then reads that as excluding zero and turns a null into a finding.  Exclusion must be
  strict, and a bound sitting on zero is printed as touching it.
* **Ties are counted, not dropped.**  The same spike makes exact ties common; a sign
  test that silently discards them inflates the apparent consistency, so positive,
  negative and tied counts are all reported and the test is run on the full cohort.
* **The spread between optimisation starts is not a noise floor.**  It is a validation
  quantity about basins, not the random variable the held-out effect is drawn from.  It
  is reported alongside every comparison as an annotation and gates nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results/topic5_motif_autonomous_v0_4"
COMPARISONS = (
    ("M1_over_M0", "M0_ISOTROPIC_DIFFUSION", "M1_AXIAL_CORRIDOR",
     "沿这位患者的一条轴拉长，比往四面八方一样扩好多少"),
    ("M2_over_M1", "M1_AXIAL_CORRIDOR", "M2_DIRECTED_TRANSPORT",
     "由事件早期的位移选一个方向一直推，比只有走廊好多少"),
    ("M3_over_M2", "M2_DIRECTED_TRANSPORT", "M3_AXIAL_FEEDFORWARD_TRANSIENT",
     "沿轴的有限时程放大，比方向输运好多少"),
)
N_BOOTSTRAP = 10000


def bootstrap_median(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    medians = np.median(values[draws], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def interval_verdict(low: float, high: float) -> str:
    """Strictly excluding zero, touching it, or spanning it."""
    if low > 0.0 or high < 0.0:
        return "excludes_zero_strictly"
    if low == 0.0 or high == 0.0:
        return "touches_zero"
    return "spans_zero"


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, position in enumerate(order):
        value = (len(p_values) - rank) * p_values[position]
        running = max(running, value)
        adjusted[position] = min(1.0, running)
    return [float(v) for v in adjusted]


def load(tag: str) -> list[dict]:
    directory = RESULT_ROOT / tag / "per_patient"
    if not directory.is_dir():
        raise SystemExit(f"no per-patient directory at {directory}")
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def arm_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("state") != "ok":
            continue
        for arm in record["arms"]:
            starts = [s["validation_primary_nll"] for s in arm["starts"]]
            chosen = arm["starts"][arm["chosen_start"]]
            rows.append({
                "patient": record["patient"], "arm": arm["arm"],
                "n_contacts": record["n_contacts"], "n_shafts": record["n_shafts"],
                "n_events_kept": record["n_events_kept"],
                "n_events_too_short": record["n_events_too_short"],
                "test_primary_nll": arm["test_primary_nll"],
                "test_sensitivity_nll": arm["test_sensitivity_nll"],
                "validation_primary_nll": chosen["validation_primary_nll"],
                "n_starts": arm["n_starts"],
                "start_spread": float(max(starts) - min(starts)),
                "chosen_head_mode": chosen["head_mode"],
                "chosen_theta_init": chosen["theta_init"],
                "starts_are_bit_identical": arm["starts_are_bit_identical"],
                "any_start_hit_epoch_cap": any(s["hit_epoch_cap"] for s in arm["starts"]),
                "max_warm_start_gap": max(
                    (s["warm_start_gap"] for s in arm["starts"]
                     if s["warm_start_gap"] is not None), default=float("nan")),
                **{f"fitted_{k}": v for k, v in arm["fitted_parameters"].items()},
            })
    return pd.DataFrame(rows)


def increments(table: pd.DataFrame) -> pd.DataFrame:
    wide = table.pivot(index="patient", columns="arm", values="test_primary_nll")
    spread = table.pivot(index="patient", columns="arm", values="start_spread")
    rows = []
    for key, parent, child, plain in COMPARISONS:
        if parent not in wide or child not in wide:
            continue
        for patient in wide.index:
            rows.append({
                "patient": patient, "comparison": key, "plain_language": plain,
                # lower negative log likelihood is better, so the parent minus the child
                # is how much the extra mechanism bought
                "improvement_nats": float(wide.loc[patient, parent]
                                          - wide.loc[patient, child]),
                "start_spread_child": float(spread.loc[patient, child]),
            })
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> dict:
    blocks, raw_p = [], []
    for key, _, _, plain in COMPARISONS:
        part = table[table.comparison == key]
        if part.empty:
            continue
        values = part.improvement_nats.to_numpy()
        positive = int((values > 0).sum())
        negative = int((values < 0).sum())
        tied = int((values == 0).sum())
        # ties are kept in the denominator: they are the warm start reproducing the
        # parent, which is evidence of no gain, not a missing observation
        p_value = float(stats.binomtest(positive, len(values), 0.5).pvalue)
        low, high = bootstrap_median(values)
        median = float(np.median(values))
        verdict = interval_verdict(low, high)
        blocks.append({
            "comparison": key, "plain_language": plain, "n_patients": len(values),
            "median_improvement_nats": median,
            "ci95_low": low, "ci95_high": high,
            "interval_verdict": verdict,
            "positive": positive, "negative": negative, "tied": tied,
            "sign_test_p": p_value,
            # the two-sided p carries no direction: with ties kept in the denominator a
            # cohort where the child is mostly *worse* also produces a small p, and
            # reading that as support would invert the finding
            "direction": "child_better" if median > 0
                         else ("child_worse" if median < 0 else "no_difference"),
            "supports_improvement": bool(median > 0 and p_value < 0.05
                                         and verdict == "excludes_zero_strictly"),
            "median_start_spread_child": float(np.median(part.start_spread_child)),
        })
        raw_p.append(p_value)
    for block, adjusted in zip(blocks, holm(raw_p)):
        block["sign_test_p_holm"] = adjusted
    return {"comparisons": blocks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run")
    arguments = parser.parse_args()

    records = load(arguments.tag)
    out = RESULT_ROOT / arguments.tag
    states = {}
    for record in records:
        states[record["state"]] = states.get(record["state"], 0) + 1

    table = arm_rows(records)
    if table.empty:
        raise SystemExit(f"no completed patients in {arguments.tag}: {states}")
    table.to_csv(out / "PER_ARM_SCORES.csv", index=False)

    steps = increments(table)
    steps.to_csv(out / "NESTED_INCREMENTS.csv", index=False)

    summary = summarise(steps)
    summary["states"] = states
    summary["engineering"] = {
        "patients_with_a_start_at_the_epoch_cap": sorted(
            table[table.any_start_hit_epoch_cap].patient.unique().tolist()),
        "arms_whose_starts_were_bit_identical": (
            table[table.starts_are_bit_identical & (table.n_starts > 1)]
            [["patient", "arm", "n_starts"]].to_dict("records")),
        "max_warm_start_gap": float(np.nanmax(table.max_warm_start_gap))
        if table.max_warm_start_gap.notna().any() else None,
        "median_events_dropped_as_too_short": float(
            table.groupby("patient").n_events_too_short.first().median()),
    }
    (out / "COHORT_SUMMARY.json").write_text(json.dumps(summary, indent=2))

    for block in summary["comparisons"]:
        print(f"{block['comparison']:12s} n={block['n_patients']:3d} "
              f"median={block['median_improvement_nats']:+.5f} "
              f"CI[{block['ci95_low']:+.5f},{block['ci95_high']:+.5f}] "
              f"{block['interval_verdict']:24s} "
              f"+/-/tie={block['positive']}/{block['negative']}/{block['tied']} "
              f"p={block['sign_test_p']:.4f} holm={block['sign_test_p_holm']:.4f} "
              f"{block['direction']:14s} support={block['supports_improvement']} "
              f"start_spread={block['median_start_spread_child']:.5f}")
    print(json.dumps(summary["engineering"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
