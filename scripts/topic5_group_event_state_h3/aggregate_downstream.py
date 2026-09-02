#!/usr/bin/env python3
"""Cohort aggregation for the perturbation replay and the impulse response.

Two things are reported here that are easy to conflate and mean different things:

*The ablation* (``real`` minus ``no_event_feedback``) asks what happens when the
edge is switched off inside a model that was **trained with it**.  Nothing
compensates, so it measures how much that model leans on the edge.

*The model comparison* (M1 minus M0, in ``aggregate_h3.py``) asks what happens
when a model is **trained without** the edge, and is free to lean harder on the
background and the clock instead.

A large ablation with a small model-comparison gain is not a contradiction: it is
the statement that the edge is load-bearing inside the fitted model but largely
recoverable from what else the model can see.  The plan's acceptance rules call
that "event observation is informative", not feedback -- so the two numbers are
kept apart here and never summed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.analysis import (  # noqa: E402
    bootstrap_median_ci,
    sign_test,
    wilcoxon,
)
from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def _cohort(values: dict[str, float], label: str) -> dict:
    delta = np.asarray([values[s] for s in sorted(values)], dtype=np.float64)
    lo, hi = bootstrap_median_ci(delta)
    return {
        "label": label,
        "n_patients": int(delta.size),
        "median": float(np.median(delta)) if delta.size else float("nan"),
        "median_ci95": [lo, hi],
        "p_wilcoxon": wilcoxon(delta),
        **sign_test(delta),
        "per_subject": {s: float(values[s]) for s in sorted(values)},
    }


def aggregate_perturbation(tag: str, horizons: list[int]) -> dict:
    directory = OUT_ROOT / "machine" / f"perturbation_{tag}"
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    records = [r for r in records if r.get("status") == "ok"]
    if not records:
        return {"status": "not_available", "directory": str(directory)}

    by_subject: dict[str, list[dict]] = {}
    for rec in records:
        by_subject.setdefault(rec["subject"], []).append(rec)

    out: dict = {
        "status": "ok",
        "n_runs": len(records),
        "n_patients": len(by_subject),
        "seeds": sorted({r["seed"] for r in records}),
        "primary_arms": records[0]["primary_arms"],
        "secondary_arms": records[0].get("secondary_arms", []),
        "estimands": {
            "burden": "real_sequence minus no_event_feedback; exposure count is NOT matched out",
            "content": "real_sequence minus state_matched_mark_replacement; count and instants preserved",
        },
        "horizons": {},
    }
    rows: list[dict] = []
    for horizon in horizons:
        arms = out["primary_arms"] + out["secondary_arms"]
        per_arm: dict[str, dict[str, float]] = {a: {} for a in arms}
        per_arm_mark: dict[str, dict[str, float]] = {a: {} for a in arms}
        pairs: dict[str, float] = {}
        for subject, runs in by_subject.items():
            entries = [r["horizons"].get(str(horizon), {}) for r in runs]
            entries = [e for e in entries if e.get("status") == "ok"]
            if not entries:
                continue
            pairs[subject] = float(np.median([e["n_disjoint_pairs"] for e in entries]))
            for arm in arms:
                if arm not in entries[0]:
                    continue
                # Seeds are repeated fits: collapse within patient first.
                per_arm[arm][subject] = float(
                    np.median([e[arm]["median_count_delta_vs_real"] for e in entries])
                )
                per_arm_mark[arm][subject] = float(
                    np.median([e[arm]["median_mark_delta_vs_real"] for e in entries])
                )
        if not pairs:
            out["horizons"][str(horizon)] = {"status": "no_pairs"}
            continue
        entry = {
            "status": "ok",
            "n_patients": len(pairs),
            "median_disjoint_pairs_per_patient": float(np.median(list(pairs.values()))),
            "n_disjoint_pairs_per_patient": pairs,
            "count_endpoint": {
                arm: _cohort(per_arm[arm], f"{arm}: real minus perturbed, count")
                for arm in arms
                if per_arm[arm]
            },
            "mark_endpoint": {
                arm: _cohort(per_arm_mark[arm], f"{arm}: real minus perturbed, conditional mark")
                for arm in arms
                if per_arm_mark[arm]
            },
        }
        out["horizons"][str(horizon)] = entry
        for arm in arms:
            if not per_arm[arm]:
                continue
            for subject, value in per_arm[arm].items():
                rows.append(
                    {
                        "subject": subject,
                        "horizon_minutes": horizon,
                        "perturbation_arm": arm,
                        "tier": "primary" if arm in out["primary_arms"] else "secondary",
                        "count_delta_real_minus_perturbed": value,
                        "mark_delta_real_minus_perturbed": per_arm_mark[arm].get(subject, ""),
                        "n_disjoint_pairs": pairs.get(subject, ""),
                    }
                )
    out["_rows"] = rows
    return out


def aggregate_impulse(tag: str, horizons: list[int]) -> dict:
    directory = OUT_ROOT / "machine" / f"impulse_{tag}"
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    records = [r for r in records if r.get("status") == "ok"]
    if not records:
        return {"status": "not_available", "directory": str(directory)}

    by_subject: dict[str, list[dict]] = {}
    for rec in records:
        by_subject.setdefault(rec["subject"], []).append(rec)

    out: dict = {
        "status": "ok",
        "n_runs": len(records),
        "n_patients": len(by_subject),
        "readout": "fractional change in the expected number of events in the next block",
        "sign_note": "nothing in the model forces this to be positive; both signs are reported",
        "horizons": {},
    }
    rows: list[dict] = []
    for horizon in horizons:
        medians: dict[str, float] = {}
        positive: dict[str, float] = {}
        channels: dict[str, dict[str, float]] = {"count": {}, "mark": {}}
        for subject, runs in by_subject.items():
            stats = [r["primary"]["horizons"].get(str(horizon)) for r in runs]
            stats = [s for s in stats if s and np.isfinite(s["median_count_fraction"])]
            if not stats:
                continue
            medians[subject] = float(np.median([s["median_count_fraction"] for s in stats]))
            positive[subject] = float(np.median([s["fraction_events_positive"] for s in stats]))
            for channel in ("count", "mark"):
                values = [
                    s["median_count_fraction_by_channel"].get(channel)
                    for s in stats
                    if s["median_count_fraction_by_channel"].get(channel) is not None
                ]
                if values:
                    channels[channel][subject] = float(np.median(values))
            rows.append(
                {
                    "subject": subject,
                    "horizon_minutes": horizon,
                    "median_count_fraction": medians[subject],
                    "fraction_events_positive": positive[subject],
                    "count_channel": channels["count"].get(subject, ""),
                    "mark_channel": channels["mark"].get(subject, ""),
                }
            )
        if not medians:
            out["horizons"][str(horizon)] = {"status": "no_records"}
            continue
        out["horizons"][str(horizon)] = {
            "status": "ok",
            "per_patient_median": _cohort(medians, "per-event median fractional change"),
            "fraction_of_events_that_raise_the_next_block": _cohort(
                positive, "fraction of events with a positive response"
            ),
            "by_channel": {
                channel: _cohort(values, f"{channel} channel only")
                for channel, values in channels.items()
                if values
            },
        }

    # Which continuous event coordinates the signed response follows.
    axes: dict[str, dict[str, list[float]]] = {}
    for subject, runs in by_subject.items():
        for rec in runs:
            for horizon, per_axis in rec.get("continuous_axis_spearman", {}).items():
                for name, rho in per_axis.items():
                    if rho is not None and np.isfinite(rho):
                        axes.setdefault(horizon, {}).setdefault(name, []).append(float(rho))
    out["continuous_axis_spearman"] = {
        horizon: {
            name: {
                "median_rho": float(np.median(values)),
                "n_records": len(values),
                "n_negative": int(sum(v < 0 for v in values)),
            }
            for name, values in sorted(per_axis.items())
        }
        for horizon, per_axis in axes.items()
    }
    out["_rows"] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--horizons", nargs="*", type=int, default=[5, 30, 120])
    args = parser.parse_args()

    perturbation = aggregate_perturbation(args.tag, args.horizons)
    impulse = aggregate_impulse(args.tag, args.horizons)
    for payload, name in ((perturbation, "perturbation"), (impulse, "impulse")):
        rows = payload.pop("_rows", [])
        write_json_atomic(payload, OUT_ROOT / "machine" / f"cohort_{name}_{args.tag}.json")
        if rows:
            path = OUT_ROOT / "machine" / f"per_patient_{name}_{args.tag}.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    if perturbation.get("status") == "ok":
        print("=== perturbation replay (frozen model; ablation, not a retrained control) ===")
        for horizon in args.horizons:
            entry = perturbation["horizons"].get(str(horizon), {})
            if entry.get("status") != "ok":
                print(f"{horizon:>4}m: {entry.get('status')}")
                continue
            print(f"{horizon:>4}m n_patients={entry['n_patients']} "
                  f"pairs/patient median={entry['median_disjoint_pairs_per_patient']:.0f}")
            for arm, stats in entry["count_endpoint"].items():
                if arm == "real_sequence":
                    continue
                print(f"      count  {arm:34s} median={stats['median']:+.4f} "
                      f"CI[{stats['median_ci95'][0]:+.4f},{stats['median_ci95'][1]:+.4f}] "
                      f"{stats['n_positive']}/{stats['n_nonzero']} p_sign={stats['p_sign']:.4f}")
    if impulse.get("status") == "ok":
        print("\n=== signed impulse response (readout of the fitted edge) ===")
        for horizon in args.horizons:
            entry = impulse["horizons"].get(str(horizon), {})
            if entry.get("status") != "ok":
                print(f"{horizon:>4}m: {entry.get('status')}")
                continue
            main_stats = entry["per_patient_median"]
            pos = entry["fraction_of_events_that_raise_the_next_block"]
            print(f"{horizon:>4}m n_patients={main_stats['n_patients']} "
                  f"median={main_stats['median']:+.5f} "
                  f"CI[{main_stats['median_ci95'][0]:+.5f},{main_stats['median_ci95'][1]:+.5f}] "
                  f"{main_stats['n_positive']}/{main_stats['n_nonzero']} raise; "
                  f"median share of events raising = {pos['median']:.3f}")
    print("\nwrote cohort_perturbation / cohort_impulse JSON + per-patient CSV")


if __name__ == "__main__":
    main()
