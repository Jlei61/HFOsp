#!/usr/bin/env python3
"""Aggregate three-seed Zhangjiaqi long total-effect results without p-values."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_long_total import LONG_TOTAL_REVISION


WINDOWS = ("event_count_10000", "physical_6h")
SEEDS = (0, 1, 2)


def _delta(result: dict, arm: str, reference: str, metric: str) -> float:
    values = result["validation_decoder_space"]
    return float(values[arm][metric] - values[reference][metric])


def _next_delta(result: dict, arm: str, reference: str, metric: str) -> float:
    values = result["next_event_exact_likelihood_secondary"]
    return float(values[arm][metric] - values[reference][metric])


def _optional_contrast(result: dict, name: str, metric: str) -> float | None:
    value = result.get("contrasts", {}).get(name)
    return None if value is None else float(value[metric])


def _payload_fingerprint(result: dict) -> str:
    """Hash a result with everything seed-specific removed.

    Three R1.3 seeds that share one initialisation and run a deterministic
    chronological trajectory produce byte-identical arms.  Counting them as
    "3/3 seeds" would turn one computation into three replicates.
    """
    value = copy.deepcopy(result)
    for key in ("seed", "t1", "parameters_and_support",
                "parameters_and_support_sha256"):
        value.pop(key, None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "t2_long_total_effect",
    )
    args = parser.parse_args()
    rows = []
    payloads: dict[str, list[dict]] = {window: [] for window in WINDOWS}
    for window in WINDOWS:
        for seed in SEEDS:
            path = (
                args.root / "human/yuquan_zhangjiaqi" / window
                / f"seed_{seed}/result.json"
            )
            result = json.loads(path.read_text())
            if (result.get("status") != "COMPLETE"
                    or result.get("sealed_opened") is not False
                    or result.get("formal_test_partition_opened") is not False):
                raise ValueError(f"invalid long total-effect result: {path}")
            admissible = result.get("instrument_admissibility", {}).get(
                "human_biological_contrasts_admissible",
                bool(
                    result["decoder_readout"]["rank"] > 0
                    and result["t1"].get("selected_total_epoch", 0) > 0
                ),
            )
            payloads[window].append(result)
            row = {
                "window": window,
                "seed": seed,
                "payload_fingerprint": _payload_fingerprint(result),
                "instrument_revision": result.get("revision", ""),
                "train_windows": result["denominators"]["train_windows"],
                "validation_windows": result["denominators"]["validation_windows"],
                "validation_next_event_pairs": result["denominators"][
                    "validation_next_event_pairs"
                ],
                "median_window_hours_validation": result["denominators"][
                    "median_window_hours_validation"
                ],
                "median_events_per_window_validation": result["denominators"][
                    "median_events_per_window_validation"
                ],
                "decoder_rank": result["decoder_readout"]["rank"],
                "decoder_blocks_at_scale_floor": len(
                    result["decoder_readout"].get("blocks_at_scale_floor", [])
                ),
                "biological_contrasts_admissible": bool(admissible),
                "real_minus_intercept_matched_decoder": _optional_contrast(
                    result, "real_minus_intercept_matched",
                    "decoder_total_equal_block_mse",
                ),
                "delayed_minus_intercept_matched_decoder": _optional_contrast(
                    result, "delayed_minus_intercept_matched",
                    "decoder_total_equal_block_mse",
                ),
                "real_minus_no_edge_decoder": result["contrasts"][
                    "real_minus_no_edge"
                ]["decoder_total_equal_block_mse"],
                "real_minus_delayed_decoder": result["contrasts"][
                    "real_minus_causal_delayed"
                ]["decoder_total_equal_block_mse"],
                "delayed_minus_no_edge_decoder": _delta(
                    result, "causal_delayed_load_1000",
                    "no_edge_natural_flow", "decoder_total_equal_block_mse",
                ),
                "real_minus_no_edge_next_joint": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "joint_nll_per_event",
                ),
                "real_minus_delayed_next_joint": _next_delta(
                    result, "real_occurrence_plus_load",
                    "causal_delayed_load_1000", "joint_nll_per_event",
                ),
                "real_minus_no_edge_next_timing": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "timing_nll_per_event",
                ),
                "real_minus_no_edge_next_mark": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "mark_nll_per_event",
                ),
                "real_minus_no_edge_next_stop": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "stop_nll_per_event",
                ),
                "real_minus_no_edge_next_first_subset": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "first_group_subset_nll_per_event",
                ),
                "real_minus_no_edge_next_continuation": _next_delta(
                    result, "real_occurrence_plus_load",
                    "no_edge_natural_flow", "continuation_subset_nll_per_event",
                ),
            }
            rows.append(row)
    summary = {}
    for window in WINDOWS:
        selected = [row for row in rows if row["window"] == window]
        numeric = {}
        for key in selected[0]:
            if key in {"window", "seed", "payload_fingerprint",
                       "instrument_revision"}:
                continue
            if any(row[key] is None for row in selected):
                numeric[key] = None
                continue
            value = np.asarray([row[key] for row in selected], dtype=np.float64)
            numeric[key] = {
                "median": float(np.median(value)),
                "min": float(np.min(value)),
                "max": float(np.max(value)),
            }
        real_no = np.asarray(
            [row["real_minus_no_edge_decoder"] for row in selected]
        )
        real_delayed = np.asarray(
            [row["real_minus_delayed_decoder"] for row in selected]
        )
        delayed_no = np.asarray(
            [row["delayed_minus_no_edge_decoder"] for row in selected]
        )
        admissible = np.asarray(
            [row["biological_contrasts_admissible"] for row in selected], dtype=bool
        )
        real_intercept = [
            row["real_minus_intercept_matched_decoder"] for row in selected
        ]
        distinct = {row["payload_fingerprint"] for row in selected}
        stale = [
            row["instrument_revision"] for row in selected
            if row["instrument_revision"] != LONG_TOTAL_REVISION
        ]
        summary[window] = {
            "three_seed_range": numeric,
            "seed_independence": {
                "seeds_run": int(len(selected)),
                "distinct_seed_payloads": int(len(distinct)),
                "seeds_are_independent_computations": bool(
                    len(distinct) == len(selected)
                ),
                "warning": (
                    "identical payloads mean one deterministic computation was "
                    "repeated; seed counts are not replicates"
                ),
            },
            "decoder_space_evidence_vector": {
                "admissible_seeds": int(admissible.sum()),
                "real_beats_intercept_matched_seeds": (
                    None if any(value is None for value in real_intercept)
                    else int(sum(value < 0 for value in real_intercept))
                ),
                "real_beats_delayed_seeds": int((real_delayed < 0).sum()),
                "median_real_minus_intercept_matched": (
                    None if any(value is None for value in real_intercept)
                    else float(np.median(np.asarray(real_intercept)))
                ),
                "median_real_minus_delayed": float(np.median(real_delayed)),
                "instrument_revision_stale": sorted(set(stale)),
                "scientific_status": (
                    "exploratory_contrast_available" if bool(admissible.all())
                    else "UNTESTABLE_T1_INSTRUMENT_DEGENERATE"
                ),
            },
            "free_intercept_artefact": {
                "median_real_minus_no_edge": float(np.median(real_no)),
                "median_delayed_minus_no_edge": float(np.median(delayed_no)),
                "not_exposure_evidence": True,
                "reason": (
                    "the exposure arms carry a saturated occurrence block, i.e. a "
                    "free state-space intercept the no-edge arm lacks; any offset "
                    "between the frozen natural flow and the observed target wins "
                    "this comparison with no exposure information"
                ),
            },
            "interpretation_key": (
                "score against the intercept-matched reference, never against "
                "raw no-edge: real and delayed both below intercept-matched with "
                "no real-delayed advantage supports only occurrence-like "
                "cumulative signal; real below both adds correct load-timing "
                "evidence"
            ),
        }
    t1 = []
    for seed in SEEDS:
        path = (
            args.root / "t1_r1_3/fits/yuquan_zhangjiaqi"
            / f"explicit_seed_{seed}/result.json"
        )
        value = json.loads(path.read_text())
        t1.append({
            "seed": seed,
            "selected_total_epoch": value["fit_trace"]["selected_total_epoch"],
            "persistent_minus_memoryless_joint": value["validation"][
                "persistent_minus_memoryless"
            ]["joint_nll_per_event"],
            "correct_minus_wrong_joint": value["validation"][
                "strict_matched_wrong_time"
            ]["correct_minus_wrong_median"]["joint_nll_per_event"],
            "persistent_minus_memoryless_subset": value["validation"][
                "persistent_minus_memoryless"
            ]["subset_nll_per_event"],
            "correct_minus_wrong_subset": value["validation"][
                "strict_matched_wrong_time"
            ]["correct_minus_wrong_median"]["subset_nll_per_event"],
            "sealed_opened": value["sealed_opened"],
        })
    synthetic_path = args.root / "synthetic/recovery.json"
    synthetic = json.loads(synthetic_path.read_text())
    result = {
        "status": "COMPLETE",
        "revision": LONG_TOTAL_REVISION,
        "subject": "yuquan_zhangjiaqi",
        "t1_three_seed": t1,
        "windows": summary,
        "per_seed": rows,
        "synthetic": {
            "path": str(synthetic_path),
            "sha256": contract.sha256_file(synthetic_path),
            "status": synthetic["status"],
            "acceptance": synthetic["acceptance"],
        },
        "primary_unit": "one development patient; three seeds are stability, not n=3 biology",
        "primary_contrasts": [
            "real_minus_intercept_matched", "real_minus_causal_delayed",
        ],
        "demoted_contrasts": {
            "real_minus_no_edge": "free-intercept artefact, not exposure evidence",
            "delayed_minus_no_edge": "free-intercept artefact, not exposure evidence",
        },
        "overlap_warning": (
            "long windows overlap; seed counts and window counts are not independent "
            "biological replicates and no p-value is reported"
        ),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    report_root = args.root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "per_seed_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result["per_seed_csv"] = str(csv_path)
    result["per_seed_csv_sha256"] = contract.sha256_file(csv_path)
    contract.atomic_json(report_root / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
