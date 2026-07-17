#!/usr/bin/env python3
"""Plot TA/TB interictal-field similarity by template-axis relation.

The primary figure follows the frozen field contract:

* reversed/same collinear pairs use the two fields on their shared plane;
* different-axis pairs use the two template-specific own fields.

All groups are restricted to ``geometry_2d_supported`` subjects.  An own-field
only analysis is saved as an estimator-matched sensitivity because a shared
plane is intentionally undefined for different-axis pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, kruskal, mannwhitneyu, rankdata, wilcoxon

from paper_figures.plot_fig3_field_concordance_cohort_stat import (
    _add_sig_bracket,
    _add_violin_box_points,
    _fmt_p,
    _p_stars,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUTPUT = DEFAULT_INPUT
GROUP_ORDER = ("reversed", "same", "different")
GROUP_STYLE = {
    "reversed": {
        "label": "Reversed\ncollinear",
        "face": "#65B7A6",
        "edge": "#2F7F72",
        "point": "#2F7F72",
    },
    "same": {
        "label": "Same-direction\ncollinear",
        "face": "#D8B56A",
        "edge": "#9B7430",
        "point": "#9B7430",
    },
    "different": {
        "label": "Different-axis",
        "face": "#A8B7C3",
        "edge": "#667B8A",
        "point": "#667B8A",
    },
}


def _pearson_r(a: list[float], b: list[float]) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3:
        raise ValueError("field correlation requires at least three finite paired contacts")
    x = x[keep]
    y = y[keep]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("field correlation is undefined for a constant contact field")
    return float(np.corrcoef(x, y)[0, 1])


def _load_rows(input_root: Path) -> list[dict]:
    cohort_path = input_root / "axis_cohort.csv"
    per_subject = input_root / "per_subject"
    cohort_rows = list(csv.DictReader(cohort_path.open()))
    rows: list[dict] = []
    for cohort in cohort_rows:
        if cohort.get("status") != "ok" or cohort.get("geometry_2d_supported") != "True":
            continue
        relation = cohort.get("relation")
        if relation not in GROUP_ORDER:
            continue
        artifact_path = per_subject / f"{cohort['subject_id']}.json"
        record = json.loads(artifact_path.read_text())
        pair = record["axis_pair"]
        artifact_relation = pair["relation"]["relation"]
        if artifact_relation != relation:
            raise ValueError(
                f"axis relation drift for {cohort['subject_id']}: "
                f"cohort={relation}, artifact={artifact_relation}"
            )
        if not pair.get("geometry_2d_supported"):
            raise ValueError(f"2D eligibility drift for {cohort['subject_id']}")

        models = record["interictal_field"]["field_models"]
        own_r = _pearson_r(
            models["own_a"]["template_field"], models["own_b"]["template_field"]
        )
        shared_r = None
        if relation in ("reversed", "same"):
            shared_r = _pearson_r(
                models["shared_a"]["template_field"],
                models["shared_b"]["template_field"],
            )
            field_mode = "shared"
            similarity_r = shared_r
        else:
            field_mode = "own"
            similarity_r = own_r

        rows.append(
            {
                "subject_id": cohort["subject_id"],
                "dataset": cohort["dataset"],
                "subject": cohort["subject"],
                "relation": relation,
                "cos_ta_tb": float(cohort["cos_ta_tb"]),
                "line_angle_deg": float(cohort["line_angle_deg"]),
                "strict_stability_pass": cohort.get("strict_stability_pass") == "True",
                "field_mode": field_mode,
                "field_similarity_r": float(similarity_r),
                "own_field_similarity_r": float(own_r),
                "shared_field_similarity_r": (
                    "" if shared_r is None else float(shared_r)
                ),
                "n_field_contacts": int(cohort["n_field_contacts"]),
            }
        )
    counts = {relation: sum(r["relation"] == relation for r in rows) for relation in GROUP_ORDER}
    if counts != {"reversed": 7, "same": 5, "different": 14}:
        raise ValueError(f"unexpected frozen 2D cohort counts: {counts}")
    return rows


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (len(p_values) - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _kruskal_label_permutation(
    groups: list[np.ndarray], *, n_permutations: int, seed: int
) -> dict:
    """Monte-Carlo label permutation of the Kruskal-Wallis H statistic."""
    pooled = np.concatenate(groups)
    sizes = [int(len(group)) for group in groups]
    ranks = rankdata(pooled)
    n_total = int(len(pooled))
    _, tie_counts = np.unique(pooled, return_counts=True)
    tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts)) / (
        n_total**3 - n_total
    )
    observed = float(kruskal(*groups).statistic)
    rng = np.random.default_rng(seed)
    n_exceed = 0
    batch_size = 10_000
    for start in range(0, n_permutations, batch_size):
        n_batch = min(batch_size, n_permutations - start)
        # Ranking independent continuous random numbers gives uniform label permutations.
        permutations = np.argsort(rng.random((n_batch, n_total)), axis=1)
        rank_sums = []
        offset = 0
        for size in sizes:
            rank_sums.append(ranks[permutations[:, offset : offset + size]].sum(axis=1))
            offset += size
        h_perm = (
            12.0
            / (n_total * (n_total + 1.0))
            * sum(rank_sum**2 / size for rank_sum, size in zip(rank_sums, sizes))
            - 3.0 * (n_total + 1.0)
        ) / tie_correction
        n_exceed += int(np.count_nonzero(h_perm >= observed - 1e-12))
    return {
        "statistic_h": observed,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "n_permutations_ge_observed": int(n_exceed),
        "p_value": float((n_exceed + 1) / (n_permutations + 1)),
        "asymptotic_p_value": float(kruskal(*groups).pvalue),
    }


def _statistics(
    rows: list[dict],
    *,
    value_key: str,
    n_permutations: int,
    seed: int,
) -> dict:
    groups = {
        relation: np.asarray(
            [row[value_key] for row in rows if row["relation"] == relation], dtype=float
        )
        for relation in GROUP_ORDER
    }
    omnibus = _kruskal_label_permutation(
        [groups[relation] for relation in GROUP_ORDER],
        n_permutations=n_permutations,
        seed=seed,
    )

    pairs = (("reversed", "same"), ("reversed", "different"), ("same", "different"))
    pairwise = []
    raw_p = []
    for group_a, group_b in pairs:
        a = groups[group_a]
        b = groups[group_b]
        test = mannwhitneyu(a, b, alternative="two-sided", method="exact")
        p_value = float(test.pvalue)
        raw_p.append(p_value)
        pairwise.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "mann_whitney_u": float(test.statistic),
                "p_raw_two_sided_exact": p_value,
                "cliffs_delta_a_minus_b": float(
                    2.0 * float(test.statistic) / (len(a) * len(b)) - 1.0
                ),
            }
        )
    for row, adjusted in zip(pairwise, _holm_adjust(raw_p)):
        row["p_holm"] = float(adjusted)

    group_summary = {}
    for relation in GROUP_ORDER:
        values = groups[relation]
        wilcoxon_result = wilcoxon(
            values, zero_method="wilcox", alternative="two-sided", method="auto"
        )
        n_positive = int(np.sum(values > 0))
        n_negative = int(np.sum(values < 0))
        group_summary[relation] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "iqr": [
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)),
            ],
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "wilcoxon_vs_zero_p_two_sided": float(wilcoxon_result.pvalue),
            "sign_test_vs_zero_p_two_sided": float(
                binomtest(n_positive, n_positive + n_negative, 0.5).pvalue
            ),
        }
    return {
        "value_key": value_key,
        "group_order": list(GROUP_ORDER),
        "group_summary": group_summary,
        "omnibus": omnibus,
        "pairwise": pairwise,
    }


def _plot(rows: list[dict], stats: dict, out_png: Path, out_pdf: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(5.15, 4.35))
    positions = {relation: float(index + 1) for index, relation in enumerate(GROUP_ORDER)}

    for relation in GROUP_ORDER:
        values = np.asarray(
            [row["field_similarity_r"] for row in rows if row["relation"] == relation],
            dtype=float,
        )
        style = GROUP_STYLE[relation]
        _add_violin_box_points(
            ax,
            values,
            positions[relation],
            facecolor=style["face"],
            edgecolor=style["edge"],
            rng=rng,
            point_face=style["point"],
            point_edge="white",
        )

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(3.42, 0.015, "r = 0", ha="left", va="bottom", fontsize=8.5, color="0.45")

    significant_pairs = [row for row in stats["pairwise"] if row["p_holm"] < 0.05]
    bracket_y = 0.76
    for index, comparison in enumerate(significant_pairs):
        _add_sig_bracket(
            ax,
            positions[comparison["group_a"]],
            positions[comparison["group_b"]],
            bracket_y + 0.15 * index,
            _p_stars(comparison["p_holm"]),
        )

    overall_p = stats["omnibus"]["p_value"]
    ax.text(
        0.02,
        0.985,
        f"Overall permutation P = {_fmt_p(overall_p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
    )
    ax.set_ylabel("TA–TB field similarity (signed r)", fontsize=11)
    ax.set_xticks([positions[relation] for relation in GROUP_ORDER])
    ax.set_xticklabels(
        [
            f"{GROUP_STYLE[relation]['label']}\nn={stats['group_summary'][relation]['n']}"
            for relation in GROUP_ORDER
        ],
        fontsize=9.2,
    )
    ax.set_xlim(0.55, 3.72)
    ax.set_ylim(-1.05, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=1.0)
    ax.yaxis.grid(False)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.97, bottom=0.19)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def _write_rows(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "subject_id",
        "dataset",
        "subject",
        "relation",
        "cos_ta_tb",
        "line_angle_deg",
        "strict_stability_pass",
        "field_mode",
        "field_similarity_r",
        "own_field_similarity_r",
        "shared_field_similarity_r",
        "n_field_contacts",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    rows = _load_rows(args.input_root)
    primary_stats = _statistics(
        rows,
        value_key="field_similarity_r",
        n_permutations=args.permutations,
        seed=args.seed,
    )
    own_only_stats = _statistics(
        rows,
        value_key="own_field_similarity_r",
        n_permutations=args.permutations,
        seed=args.seed,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_png = figure_dir / "template_field_similarity_by_axis_relation.png"
    out_pdf = figure_dir / "template_field_similarity_by_axis_relation.pdf"
    out_csv = args.output_root / "field_similarity_by_axis_relation.csv"
    out_json = args.output_root / "field_similarity_by_axis_relation_statistics.json"
    _plot(rows, primary_stats, out_png, out_pdf, seed=args.seed)
    _write_rows(rows, out_csv)
    out_json.write_text(
        json.dumps(
            {
                "contract": "topic5_interictal_template_fields_v1",
                "cohort": "geometry_2d_supported",
                "primary_metric": (
                    "signed Pearson r across contact-evaluated TA/TB template fields; "
                    "shared plane for reversed/same collinear pairs and own planes for "
                    "different-axis pairs"
                ),
                "primary_relation_appropriate": primary_stats,
                "sensitivity_estimator_matched_own_fields_for_all_groups": own_only_stats,
                "interpretation_boundary": (
                    "The primary statistic describes each relation class using its contract-"
                    "appropriate field representation. Because plane construction differs for "
                    "the different-axis group, the own-only analysis is the estimator-matched "
                    "between-group sensitivity. Axis and field are derived from the same TA/TB "
                    "rank templates, so this is internal structural consistency, not independent "
                    "biological validation."
                ),
                "per_subject_csv": str(out_csv.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[done] wrote {out_png}")
    print(f"[done] wrote {out_pdf}")
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
