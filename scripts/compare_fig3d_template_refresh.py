#!/usr/bin/env python3
"""Compare legacy, matched Timing-only and Timing+Space Fig. 3D results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _wilcoxon_two_sided(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative="two-sided").pvalue)


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    seed: int,
    n_boot: int = 10000,
) -> list[float | None]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, len(values), size=(n_boot, len(values)))],
        axis=1,
    )
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _load_variant(label: str, subject_path: Path, cohort_path: Path):
    subject = pd.read_csv(subject_path)
    cohort = pd.read_csv(cohort_path)
    subject["variant"] = label
    cohort["variant"] = label
    return subject, cohort


def run(args: argparse.Namespace) -> dict:
    variants = {
        "legacy": _load_variant("legacy", args.legacy_subject, args.legacy_cohort),
        "matched_timing_only": _load_variant(
            "matched_timing_only", args.timing_subject, args.timing_cohort
        ),
        "timing_plus_space": _load_variant(
            "timing_plus_space", args.space_subject, args.space_cohort
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort_table = pd.concat(
        [pair[1] for pair in variants.values()], ignore_index=True
    )
    cohort_table.to_csv(args.out_dir / "fig3d_variant_summary.csv", index=False)

    delta_rows = []
    comparisons = []
    for comparator in ("legacy", "matched_timing_only"):
        old = variants[comparator][0]
        new = variants["timing_plus_space"][0]
        for group_id in sorted(set(old["group_id"]) | set(new["group_id"])):
            left = old.loc[old["group_id"] == group_id]
            right = new.loc[new["group_id"] == group_id]
            matched = left.merge(
                right,
                on="subject",
                suffixes=("_reference", "_space"),
            )
            for row in matched.itertuples(index=False):
                delta_rows.append({
                    "comparison": f"timing_plus_space_minus_{comparator}",
                    "group_id": group_id,
                    "subject": row.subject,
                    "data_delta": row.data_space - row.data_reference,
                    "null_delta": (
                        row.channel_null_median_space
                        - row.channel_null_median_reference
                    ),
                    "margin_delta": row.margin_space - row.margin_reference,
                })
            data_delta = matched["data_space"] - matched["data_reference"]
            null_delta = (
                matched["channel_null_median_space"]
                - matched["channel_null_median_reference"]
            )
            margin_delta = matched["margin_space"] - matched["margin_reference"]
            comparisons.append({
                "comparison": f"timing_plus_space_minus_{comparator}",
                "group_id": group_id,
                "n_reference": int(len(left)),
                "n_space": int(len(right)),
                "n_matched": int(len(matched)),
                "lost_subjects": sorted(set(left["subject"]) - set(right["subject"])),
                "gained_subjects": sorted(set(right["subject"]) - set(left["subject"])),
                "data_delta_median": float(np.median(data_delta)),
                "data_delta_bootstrap_95ci": _bootstrap_median_ci(
                    data_delta.to_numpy(float), seed=args.seed + len(comparisons) * 11
                ),
                "data_delta_wilcoxon_two_sided_p": _wilcoxon_two_sided(
                    data_delta.to_numpy(float)
                ),
                "null_delta_median": float(np.median(null_delta)),
                "margin_delta_median": float(np.median(margin_delta)),
                "margin_delta_bootstrap_95ci": _bootstrap_median_ci(
                    margin_delta.to_numpy(float), seed=args.seed + len(comparisons) * 17
                ),
                "margin_delta_wilcoxon_two_sided_p": _wilcoxon_two_sided(
                    margin_delta.to_numpy(float)
                ),
            })

    pd.DataFrame(delta_rows).to_csv(
        args.out_dir / "fig3d_matched_subject_deltas.csv", index=False
    )
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(
        args.out_dir / "fig3d_matched_comparisons.csv", index=False
    )
    field_summaries = {}
    for label, path in (
        ("legacy", args.legacy_field_summary),
        ("matched_timing_only", args.timing_field_summary),
        ("timing_plus_space", args.space_field_summary),
    ):
        field_summaries[label] = json.loads(path.read_text())
    payload = {
        "question": (
            "How does adding event spatial direction during interictal clustering "
            "change frozen fields and Fig. 3D?"
        ),
        "matched_control": (
            "Timing-only and Timing+Space use identical QC-clean events, joint-support "
            "contact rule, dense rank representation and field producer."
        ),
        "field_summaries": field_summaries,
        "fig3d_cohort_statistics": cohort_table.to_dict("records"),
        "fig3d_paired_comparisons": comparisons,
    }
    out = args.out_dir / "fig3d_template_refresh_comparison.json"
    out.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("legacy", "timing", "space"):
        parser.add_argument(f"--{prefix}-subject", type=Path, required=True)
        parser.add_argument(f"--{prefix}-cohort", type=Path, required=True)
        parser.add_argument(f"--{prefix}-field-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["fig3d_paired_comparisons"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
