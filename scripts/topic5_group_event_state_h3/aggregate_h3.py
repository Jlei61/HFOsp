#!/usr/bin/env python3
"""Blocks -> patients -> cohort for the M0/M1/M2 comparison.

The two contrasts the acceptance rules name, and nothing merged:

``M1 - M0``  does an event's *occurrence and burden* entering the state transition
             improve an unseen future block?
``M2 - M1``  at the same event count and the same instants, does the event's
             *content* add anything further?

Reported per endpoint (count, conditional mark) and per named mark part, because
an arm that sharpens the rate while blurring the spatial pattern is a different
scientific statement from one that does both.
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
    ENDPOINTS,
    collapse_seeds,
    contrast,
    load_runs,
    patient_means,
    seed_swap_null,
    seeds_are_distinct_fits,
)
from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.models import ARM_NAMES  # noqa: E402
from src.topic5_group_event_state_h3.runtime import AGENT_C_ROOT  # noqa: E402
from src.topic5_group_event_state_h3.support import MAIN_HORIZONS_MINUTES  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"

CONTRASTS = (
    ("M1_count_rate_feedback", "M0_no_feedback", "M1_minus_M0_burden_channel"),
    ("M2_mark_specific_feedback", "M1_count_rate_feedback", "M2_minus_M1_content_channel"),
    ("M2_mark_specific_feedback", "M0_no_feedback", "M2_minus_M0_any_event_channel"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--horizons", nargs="*", type=int, default=list(MAIN_HORIZONS_MINUTES))
    parser.add_argument("--split", default="development_test")
    parser.add_argument(
        "--min-blocks", type=int, default=6,
        help="pre-registered minimum number of non-overlapping held-out blocks a "
             "patient must contribute at a horizon to enter that horizon's contrast",
    )
    parser.add_argument("--primary-seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--null-seeds", nargs="*", type=int, default=[3, 4, 5])
    args = parser.parse_args()

    # The no-state floor every arm has to beat before any contrast between arms
    # is worth reading.
    climatology_path = OUT_ROOT / "machine" / "climatology_reference.json"
    climatology: dict[str, dict[str, float]] = {}
    if climatology_path.exists():
        for row in json.loads(climatology_path.read_text())["subjects"]:
            for horizon, entry in row["horizons"].items():
                if entry.get("status") == "ok":
                    climatology.setdefault(horizon, {})[row["subject"]] = entry[
                        "mean_count_logscore"
                    ]

    machine_dir = OUT_ROOT / "machine" / args.tag
    records = load_runs(machine_dir, AGENT_C_ROOT / "checkpoints" / args.tag)
    if not records:
        raise SystemExit(f"no completed runs under {machine_dir}")

    seed_check = seeds_are_distinct_fits(records)
    if seed_check["n_duplicate_seed_fits"]:
        raise SystemExit(
            "byte-identical seed fits detected; identical seeds are one fit, not two: "
            f"{seed_check['duplicates'][:5]}"
        )

    summary: dict = {
        "tag": args.tag,
        "split": args.split,
        "arms": list(ARM_NAMES),
        "n_runs": len(records),
        "n_subjects": len({r.subject for r in records}),
        "seeds": sorted({r.seed for r in records}),
        "seed_distinctness": seed_check,
        "horizons": {},
    }
    per_block_rows: list[dict] = []
    per_patient_rows: list[dict] = []

    for horizon in args.horizons:
        # The main contrast uses seeds 0-2 exactly as the plan specifies; seeds 3-5,
        # when present, exist only to build a refit floor at the same aggregation.
        collapsed = collapse_seeds(records, horizon, args.split, seeds=args.primary_seeds)
        # The eligibility rule set in C0 is enforced HERE too, not only in the
        # support inventory.  A patient contributing three 120-minute blocks would
        # otherwise carry the same weight in the cohort median as one contributing
        # twenty-four, and the figure would show points the plan had already ruled
        # out.
        dropped = sorted(
            {sub for (sub, _arm), sc in collapsed.items() if sc["n_blocks"] < args.min_blocks}
        )
        collapsed = {
            key: sc for key, sc in collapsed.items()
            if sc["n_blocks"] >= args.min_blocks
        }
        if not collapsed:
            summary["horizons"][str(horizon)] = {
                "status": "no_patient_meets_the_minimum_block_count",
                "min_blocks": args.min_blocks,
                "subjects_below_minimum": dropped,
            }
            continue
        arms_present = sorted({arm for _s, arm in collapsed})
        block_counts = {
            subject: int(scores["n_blocks"])
            for (subject, _arm), scores in collapsed.items()
        }
        entry: dict = {
            "status": "ok",
            "min_blocks_required": args.min_blocks,
            "subjects_below_minimum_block_count": dropped,
            "n_subjects_dropped_below_minimum": len(dropped),
            "arms_present": arms_present,
            "n_subjects_with_blocks": len(block_counts),
            "n_disjoint_blocks_per_subject": block_counts,
            "total_disjoint_blocks": int(sum(block_counts.values())),
            "median_disjoint_blocks_per_subject": float(np.median(list(block_counts.values()))),
            "endpoints": {},
        }

        for endpoint in ENDPOINTS:
            means = patient_means(collapsed, endpoint)
            per_endpoint: dict = {
                "arm_medians": {
                    arm: float(np.median(list(values.values())))
                    for arm, values in means.items()
                },
                "contrasts": {},
            }
            for arm_a, arm_b, label in CONTRASTS:
                if arm_a in means and arm_b in means:
                    per_endpoint["contrasts"][label] = contrast(means[arm_a], means[arm_b], label)
            # The scale a contrast has to clear before it means more than a refit.
            null = seed_swap_null(
                records, horizon, endpoint,
                primary_seeds=args.primary_seeds, null_seeds=args.null_seeds,
            )
            per_endpoint["seed_swap_null"] = null
            if null.get("status") == "ok":
                floor = null["median_absolute_refit_delta"]
                for label, stats in per_endpoint["contrasts"].items():
                    delta = np.abs(
                        np.asarray(list(stats["per_subject_delta"].values()), dtype=float)
                    )
                    # Reported next to every contrast, because the p-value is a test
                    # against zero and zero is not the comparison that matters here.
                    stats["refit_floor"] = {
                        "median_absolute_refit_delta": floor,
                        "effect_over_floor": (
                            abs(stats["median_delta"]) / floor if floor > 0 else float("inf")
                        ),
                        "n_subjects_exceeding_floor": int((delta > floor).sum()),
                        "n_subjects": int(delta.size),
                        "clears_floor": bool(abs(stats["median_delta"]) > floor),
                    }
            if endpoint == "count" and str(horizon) in climatology:
                reference = climatology[str(horizon)]
                per_endpoint["gain_over_no_state_reference"] = {
                    arm: contrast(
                        values,
                        {s: reference[s] for s in values if s in reference},
                        f"{arm}_minus_train_rate_only_reference",
                    )
                    for arm, values in means.items()
                    if any(s in reference for s in values)
                }
            entry["endpoints"][endpoint] = per_endpoint

        # Named parts of the conditional-mark endpoint.
        group_scores: dict[str, dict[str, dict[str, float]]] = {}
        n_groups = next(iter(collapsed.values()))["mark_groups"].shape[1]
        for gi in range(n_groups):
            per_arm: dict[str, dict[str, float]] = {}
            for (subject, arm), scores in collapsed.items():
                has = np.asarray(scores["has_events"], dtype=bool)
                if not has.any():
                    continue
                per_arm.setdefault(arm, {})[subject] = float(
                    np.mean(scores["mark_groups"][has, gi])
                )
            group_scores[f"mark_group_{gi}"] = {
                label: contrast(per_arm[a], per_arm[b], label)
                for a, b, label in CONTRASTS
                if a in per_arm and b in per_arm
            }
        entry["mark_group_contrasts"] = group_scores
        summary["horizons"][str(horizon)] = entry

        for (subject, arm), scores in sorted(collapsed.items()):
            has = np.asarray(scores["has_events"], dtype=bool)
            per_patient_rows.append(
                {
                    "subject": subject,
                    "arm": arm,
                    "horizon_minutes": horizon,
                    "n_disjoint_blocks": int(scores["n_blocks"]),
                    "n_seeds": int(scores["n_seeds"]),
                    "mean_count_logscore": float(np.mean(scores["count"])),
                    "mean_mark_logscore": float(np.mean(scores["mark"][has])) if has.any() else "",
                    "median_count_true": float(np.median(scores["count_true"])),
                }
            )
            for i in range(int(scores["n_blocks"])):
                per_block_rows.append(
                    {
                        "subject": subject,
                        "arm": arm,
                        "horizon_minutes": horizon,
                        "segment": int(scores["block_key"][i, 0]),
                        "anchor_index": int(scores["block_key"][i, 1]),
                        "anchor_time": float(scores["anchor_time"][i]),
                        "count_logscore": float(scores["count"][i]),
                        "mark_logscore": float(scores["mark"][i]),
                        "has_events": bool(has[i]),
                        "count_true": int(scores["count_true"][i]),
                    }
                )

    write_json_atomic(summary, OUT_ROOT / "machine" / f"cohort_summary_{args.tag}.json")
    for rows, name in ((per_block_rows, "per_block"), (per_patient_rows, "per_patient")):
        if not rows:
            continue
        path = OUT_ROOT / "machine" / f"{name}_scores_{args.tag}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for horizon in args.horizons:
        entry = summary["horizons"].get(str(horizon), {})
        if entry.get("status") != "ok":
            print(f"{horizon:>4}m: {entry.get('status', 'missing')}")
            continue
        print(
            f"{horizon:>4}m  n_subj={entry['n_subjects_with_blocks']:2d} "
            f"(dropped {entry['n_subjects_dropped_below_minimum']} below "
            f"{entry['min_blocks_required']} blocks)  "
            f"blocks/subj median={entry['median_disjoint_blocks_per_subject']:.0f}"
        )
        reference_gain = entry["endpoints"]["count"].get("gain_over_no_state_reference", {})
        if reference_gain:
            summary_line = "  ".join(
                f"{arm.split('_')[0]}={stats['median_delta']:+.3f}"
                for arm, stats in sorted(reference_gain.items())
            )
            print(f"       count  [gain over a train-rate-only reference] {summary_line}")
        for endpoint in ENDPOINTS:
            null = entry["endpoints"][endpoint].get("seed_swap_null", {})
            if null.get("status") == "ok":
                print(
                    f"       {endpoint:>5s} {'[refit floor, same arm, same width]':34s} "
                    f"median|delta|={null['median_absolute_refit_delta']:.4f} "
                    f"p90={null['p90_absolute_refit_delta']:.4f} "
                    f"(signed median={null['median_delta']:+.4f}, "
                    f"{null['n_positive']}/{null['n_nonzero']})"
                )
            for label, stats in entry["endpoints"][endpoint]["contrasts"].items():
                floor = stats.get("refit_floor")
                tail = (
                    f" | vs refit floor: {floor['effect_over_floor']:.2f}x, "
                    f"{floor['n_subjects_exceeding_floor']}/{floor['n_subjects']} patients over"
                    if floor else ""
                )
                print(
                    f"       {endpoint:>5s} {label:34s} median={stats['median_delta']:+.4f} "
                    f"CI[{stats['median_ci95'][0]:+.4f},{stats['median_ci95'][1]:+.4f}] "
                    f"{stats['n_positive']}/{stats['n_nonzero']} p_sign={stats['p_sign']:.4f} "
                    f"p_wilcoxon={stats['p_wilcoxon']:.4f}{tail}"
                )
    print(f"\nwrote {OUT_ROOT / 'machine' / f'cohort_summary_{args.tag}.json'}")


if __name__ == "__main__":
    main()
