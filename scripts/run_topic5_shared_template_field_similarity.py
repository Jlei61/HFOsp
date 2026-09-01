#!/usr/bin/env python3
"""TA/TB similarity on the frozen shared axis, without axis-relation grouping.

The estimand is the signed Pearson correlation between the contact-evaluated
TA and TB fields on one frozen shared 2D plane.  Eligibility is fixed upstream:
the subject must have a saved shared field and supported 2D geometry.  Axis
stability and the sign of the TA/TB axis cosine are reported but are not used
as inclusion gates or strata.

The primary null jointly permutes TB earliness and participation support across
all contacts, then rebuilds the TB field on the frozen shared plane.  A
within-shaft shuffle is retained as the stricter anatomy-controlled sensitivity.
Neither null refits the axes, the shared plane, the contact set, or the bandwidth.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import wilcoxon

from paper_figures.plot_fig3_field_concordance_cohort_stat import (
    _fmt_p,
    plot_paired_data_null_groups,
)
from plot_topic5_interictal_template_ab_fields import (
    DEFAULT_YUQUAN_CROSSWALK,
    _display_name,
    _load_yuquan_crosswalk,
)
from run_topic5_template_field_negative_null import (
    NULL_MODES,
    _add_fdr,
    _cohort_summary,
    _null_correlations,
    _pearson_r,
    _permutation_indices,
    _stable_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUTPUT = DEFAULT_INPUT
OBSERVED_KEY = "observed_shared_field_r"
PRIMARY_NULL = "channel"


def _subject_record(
    artifact_path: Path,
    *,
    n_perm: int,
    base_seed: int,
    yuquan_labels: dict[str, str],
) -> tuple[dict, dict[str, np.ndarray]] | None:
    record = json.loads(artifact_path.read_text())
    pair = record.get("axis_pair") or {}
    field = record.get("interictal_field") or {}
    models = field.get("field_models") or {}
    if (
        record.get("status") != "ok"
        or not pair.get("geometry_2d_supported")
        or "shared_a" not in models
        or "shared_b" not in models
    ):
        return None

    shared_a = models["shared_a"]
    shared_b = models["shared_b"]
    points_a = np.asarray(shared_a["points"], dtype=float)
    points_b = np.asarray(shared_b["points"], dtype=float)
    sigma_a = float(shared_a["sigma"])
    sigma_b = float(shared_b["sigma"])
    if not np.allclose(points_a, points_b, atol=1e-12, rtol=1e-12):
        raise ValueError(f"shared-plane point mismatch for {record['subject_id']}")
    if not np.isclose(sigma_a, sigma_b, atol=1e-12, rtol=1e-12):
        raise ValueError(f"shared-plane sigma mismatch for {record['subject_id']}")

    field_a = np.asarray(shared_a["template_field"], dtype=float)
    field_b = np.asarray(shared_b["template_field"], dtype=float)
    earliness_b = np.asarray(field["earliness_b"], dtype=float)
    support_b = np.asarray(field["support_b"], dtype=float)
    observed = _pearson_r(field_a, field_b)
    rebuilt = _null_correlations(
        field_a,
        earliness_b,
        support_b,
        points_b,
        sigma_b,
        np.arange(len(field_b), dtype=int)[None, :],
    )
    if len(rebuilt) != 1 or not np.isclose(rebuilt[0], observed, atol=1e-10, rtol=1e-10):
        raise ValueError(f"frozen shared-field rebuild mismatch for {record['subject_id']}")

    row = {
        "subject_id": record["subject_id"],
        "display_id": _display_name(record["subject_id"], yuquan_labels),
        "dataset": record["dataset"],
        "subject": record["subject"],
        "axis_cosine_sign": "negative" if float(pair["relation"]["cosine"]) < 0 else "positive",
        "cos_ta_tb": float(pair["relation"]["cosine"]),
        "line_angle_deg": float(pair["relation"]["line_angle_deg"]),
        "strict_stability_pass": bool(pair.get("strict_stability_pass")),
        "n_contacts": int(len(field_a)),
        OBSERVED_KEY: float(observed),
    }
    null_arrays: dict[str, np.ndarray] = {}
    shafts = [str(value) for value in field["shafts"]]
    for mode in NULL_MODES:
        rng = np.random.default_rng(_stable_seed(f"shared:{record['subject_id']}:{mode}", base_seed))
        permutations, n_unique, exact = _permutation_indices(shafts, mode, n_perm, rng)
        null = _null_correlations(
            field_a,
            earliness_b,
            support_b,
            points_b,
            sigma_b,
            permutations,
        )
        if exact:
            p_negative = float(np.mean(null <= observed + 1e-12))
        else:
            p_negative = float((1 + np.sum(null <= observed)) / (len(null) + 1))
        null_arrays[mode] = null
        row.update(
            {
                f"{mode}_null_median": float(np.median(null)),
                f"{mode}_null_q05": float(np.percentile(null, 5)),
                f"{mode}_null_q95": float(np.percentile(null, 95)),
                f"{mode}_p_negative": p_negative,
                f"{mode}_n_draws": int(len(null)),
                f"{mode}_n_unique": int(n_unique),
                f"{mode}_exact": bool(exact),
                f"{mode}_resolution_adequate": bool(n_unique >= 20),
            }
        )
    return row, null_arrays


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], summary: dict, out_png: Path, out_pdf: Path, *, seed: int) -> None:
    plot_rows = [
        {
            "subject_id": row["subject_id"],
            "data": row[OBSERVED_KEY],
            "null": row[f"{PRIMARY_NULL}_null_median"],
        }
        for row in rows
    ]
    # Match the established Data-vs-Null field-reversal panel: the displayed
    # cohort test is paired Wilcoxon(Data, subject-null median), alternative less.
    p_value = float(summary["paired_wilcoxon_observed_less_than_null_median_p"])
    plot_paired_data_null_groups(
        [
            {
                "label": "Shared-axis patients",
                "rows": plot_rows,
                "summary": {"n": len(plot_rows)},
                "display_p": p_value,
                "caption": "",
            }
        ],
        out_png,
        out_pdf,
        ylabel="TA–TB field reversal (signed r)",
        seed=seed,
        figsize=(3.75, 4.25),
        pair_gap=0.72,
        group_gap=2.05,
        ylim=(-1.05, 1.05),
        connect_pairs=True,
        pair_tick_labels=("Data", "Null"),
        zero_reference=True,
        bottom=0.13,
        ylabel_fontsize=10.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yuquan-crosswalk", type=Path, default=DEFAULT_YUQUAN_CROSSWALK)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--n-cohort-perm", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=12,
        help=(
            "Expected eligible shared-plane cohort size. Use 0 to accept the "
            "data-derived denominator while still requiring at least two subjects."
        ),
    )
    args = parser.parse_args()

    yuquan_labels = _load_yuquan_crosswalk(args.yuquan_crosswalk)
    rows: list[dict] = []
    null_arrays: dict[str, dict[str, np.ndarray]] = {}
    for artifact_path in sorted((args.input_root / "per_subject").glob("*.json")):
        result = _subject_record(
            artifact_path,
            n_perm=args.n_perm,
            base_seed=args.seed,
            yuquan_labels=yuquan_labels,
        )
        if result is None:
            continue
        row, subject_nulls = result
        rows.append(row)
        null_arrays[row["subject_id"]] = subject_nulls
    if args.expected_subjects > 0 and len(rows) != args.expected_subjects:
        raise ValueError(
            "expected "
            f"{args.expected_subjects} shared-axis subjects with supported 2D geometry, "
            f"found {len(rows)}"
        )
    if len(rows) < 2:
        raise ValueError(
            "shared-axis cohort inference requires at least two eligible subjects"
        )

    _add_fdr(rows, observed_key=OBSERVED_KEY)
    summaries = {
        mode: _cohort_summary(
            rows,
            null_arrays,
            mode,
            n_cohort_perm=args.n_cohort_perm,
            base_seed=args.seed,
            observed_key=OBSERVED_KEY,
        )
        for mode in NULL_MODES
    }
    observed = np.asarray([row[OBSERVED_KEY] for row in rows], dtype=float)
    observed_wilcoxon_vs_zero = float(
        wilcoxon(observed, alternative="less", method="auto").pvalue
    )
    for mode in NULL_MODES:
        null_median = np.asarray([row[f"{mode}_null_median"] for row in rows], dtype=float)
        summaries[mode].update(
            {
                "paired_wilcoxon_observed_less_than_null_median_p": float(
                    wilcoxon(observed, null_median, alternative="less", method="auto").pvalue
                ),
                "n_observed_below_subject_null_median": int(np.sum(observed < null_median)),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "shared_field_similarity_subjects.csv"
    json_path = args.output_root / "shared_field_similarity_statistics.json"
    npz_path = args.output_root / "shared_field_similarity_null_draws.npz"
    png_path = figure_dir / "template_shared_field_similarity_data_vs_null.png"
    pdf_path = figure_dir / "template_shared_field_similarity_data_vs_null.pdf"
    _write_csv(rows, csv_path)
    np.savez_compressed(
        npz_path,
        **{
            f"{row['subject_id']}__{mode}": null_arrays[row["subject_id"]][mode]
            for row in rows
            for mode in NULL_MODES
        },
    )
    json_path.write_text(
        json.dumps(
            {
                "contract": "topic5_interictal_template_fields_v1",
                "cohort": "shared_field_available_and_geometry_2d_supported",
                "n_subjects": len(rows),
                "expected_subjects": (
                    int(args.expected_subjects) if args.expected_subjects > 0 else None
                ),
                "inclusion": (
                    "saved shared_a/shared_b fields and supported 2D geometry; no strict-stability "
                    "gate and no grouping by signed axis cosine"
                ),
                "field_metric": (
                    "signed Pearson r between contact-evaluated TA and TB fields on the same "
                    "frozen shared 2D plane"
                ),
                "primary_null": PRIMARY_NULL,
                "primary_cohort_test": (
                    "paired Wilcoxon observed shared-field r vs subject channel-null median, "
                    "alternative='less'; the hierarchical cohort-median randomization is "
                    "reported as a secondary statistic"
                ),
                "permutation_payload": "TB earliness and support jointly permuted; TB field rebuilt",
                "frozen": ["contact set", "shared axis", "shared plane", "sigma", "TA field"],
                "nulls": {
                    "channel": "primary; permute TB payload across all contacts",
                    "within_shaft": (
                        "anatomy-controlled sensitivity; permute TB payload within electrode shaft"
                    ),
                },
                "n_permutations_requested": args.n_perm,
                "n_cohort_permutations": args.n_cohort_perm,
                "seed": args.seed,
                "descriptive_observed_distribution": {
                    "median_r": float(np.median(observed)),
                    "n_negative": int(np.sum(observed < 0)),
                    "n_positive": int(np.sum(observed > 0)),
                    "wilcoxon_r_less_than_zero_p": observed_wilcoxon_vs_zero,
                },
                "cohort_summary": summaries,
                "interpretation_boundary": (
                    "This is a direct shared-field reversal analysis conditional on the upstream "
                    "shared-axis definition. Negative r denotes opposite TA/TB field organization. "
                    "It does not compare cosine-sign classes, test different-axis patients, refit "
                    "KMeans, or independently validate the axis-selection rule."
                ),
                "per_subject_csv": str(csv_path.relative_to(ROOT)),
                "null_draws_npz": str(npz_path.relative_to(ROOT)),
                "figure": str(png_path.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    _plot(rows, summaries[PRIMARY_NULL], png_path, pdf_path, seed=args.seed)
    print(json.dumps(summaries, indent=2))
    print(
        f"[done] shared 2D n={len(rows)}; primary {PRIMARY_NULL} "
        "paired-Wilcoxon P="
        f"{_fmt_p(summaries[PRIMARY_NULL]['paired_wilcoxon_observed_less_than_null_median_p'])}"
    )
    print(f"[done] wrote {png_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")


if __name__ == "__main__":
    main()
