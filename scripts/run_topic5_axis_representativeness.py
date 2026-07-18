#!/usr/bin/env python3
"""Test whether fitted TA/TB gradient axes represent single-event directions.

Primary statistic: subject-folded mean signed cosine above a montage-controlled
template-rank-shuffle null. Gradient events must pass independent geometry +
LOCO QC, and the frozen template gradient axis must pass strict stability.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    DEFAULT_MAX_EVENTS,
    DEFAULT_SEED,
    FROZEN_ROOT,
    _jsonable,
    _load_masked_events_and_labels,
    _pretty_subject,
)
from scripts.plot_topic5_interictal_template_direction_rose_endpoint import (  # noqa: E402
    _load_endpoint_input_record,
)
from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (  # noqa: E402
    plot_paired_data_null_groups,
)
from src.topic5_axis_representativeness import (  # noqa: E402
    rank_shuffle_axis_null,
    summarize_direction_representativeness,
)
from src.topic5_interictal_direction_rose import (  # noqa: E402
    assess_event_direction_qc,
    fit_event_directions_3d,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    bootstrap_median_ci,
)

DEFAULT_OUT = ROOT / "results/interictal_propagation_masked/axis_representativeness"
PAPER_OUT = ROOT / "results/paper-ready-figure/fig_axis_representativeness"
TEMPLATE_LABELS = {0: "TA", 1: "TB"}
DEFAULT_N_PERM = 1000
DEFAULT_MIN_EVENTS = 20


def _seed_for(subject_id: str, method: str, template: str, seed: int) -> int:
    token = f"{subject_id}|{method}|{template}".encode("utf-8")
    return int((zlib.crc32(token) + int(seed)) % (2**32 - 1))


def _template_row(
    *,
    subject_id: str,
    dataset: str,
    subject: str,
    template: str,
    template_rank: np.ndarray,
    coords: np.ndarray,
    event_directions: np.ndarray,
    axis: np.ndarray,
    axis_strict: bool,
    n_perm: int,
    min_events: int,
    seed: int,
) -> Dict[str, object]:
    metrics = summarize_direction_representativeness(event_directions, axis)
    reasons = []
    if int(metrics["n_events"]) < min_events:
        reasons.append(f"fewer_than_{min_events}_event_directions")
    eligible = not reasons
    row: Dict[str, object] = {
        "subject_id": subject_id,
        "dataset": dataset,
        "subject": subject,
        "pretty_subject": _pretty_subject(subject_id),
        "method": "gradient",
        "template": template,
        "analysis_eligible": bool(eligible),
        "ineligibility_reasons": reasons,
        "axis_strict_stability": bool(axis_strict),
        **metrics,
        "null_n": 0,
        "null_mean_cosine_median": float("nan"),
        "null_mean_cosine_q05": float("nan"),
        "null_mean_cosine_q95": float("nan"),
        "alignment_margin": float("nan"),
        "rank_shuffle_p_greater": float("nan"),
        "null_main_gap_median_deg": float("nan"),
        "main_gap_improvement_deg": float("nan"),
    }
    if not eligible:
        return row
    null = rank_shuffle_axis_null(
        template_rank,
        coords,
        event_directions,
        method="gradient",
        n_perm=n_perm,
        seed=seed,
    )
    null_cos = np.asarray(null["mean_signed_cosine"], float)
    null_gap = np.asarray(null["axis_to_main_direction_deg"], float)
    observed = float(metrics["mean_signed_cosine"])
    null_median = float(np.nanmedian(null_cos))
    gap_median = float(np.nanmedian(null_gap))
    row.update({
        "null_n": int(len(null_cos)),
        "null_mean_cosine_median": null_median,
        "null_mean_cosine_q05": float(np.nanpercentile(null_cos, 5)),
        "null_mean_cosine_q95": float(np.nanpercentile(null_cos, 95)),
        "alignment_margin": observed - null_median,
        "rank_shuffle_p_greater": float(
            (1 + np.sum(null_cos >= observed - 1e-15)) / (len(null_cos) + 1)
        ),
        "null_main_gap_median_deg": gap_median,
        "main_gap_improvement_deg": gap_median - float(metrics["axis_to_main_direction_deg"]),
        "null_attempts": int(null["attempts"]),
        "_null_cosine": null_cos,
        "_null_main_gap": null_gap,
    })
    return row


def process_subject(
    subject_id: str,
    *,
    n_perm: int,
    min_events: int,
    max_events: int,
    seed: int,
) -> Sequence[Dict[str, object]]:
    try:
        record = _load_endpoint_input_record(subject_id)
    except Exception as exc:
        message = str(exc).replace(
            "endpoint_inputs_unavailable", "axis_inputs_unavailable"
        )
        if message != str(exc):
            raise ValueError(message) from exc
        raise
    events = _load_masked_events_and_labels(record, max_events=max_events, seed=seed)
    coords = np.asarray(record["coords"], float)
    labels = np.asarray(events["labels"], int)
    ranks = {"TA": np.asarray(record["rank_a"], float),
             "TB": np.asarray(record["rank_b"], float)}
    dataset, subject = subject_id.split("_", 1)
    rows = []

    gradient_fit = fit_event_directions_3d(events["event_ranks"], coords, min_contacts=3)
    gradient_qc = assess_event_direction_qc(
        events["event_ranks"],
        coords,
        record["shafts"],
        directions=gradient_fit["directions"],
        n_valid_contacts=gradient_fit["n_valid_contacts"],
        effective_rank=gradient_fit["effective_rank"],
    )
    pair = record["axis_pair"]
    validity = record.get("direction_validity") or {}
    for label, template, axis_key, validity_key in (
        (0, "TA", "axis_a", "ta"), (1, "TB", "axis_b", "tb")
    ):
        mask = (labels == label) & np.asarray(gradient_qc["passes"], bool)
        direction_values = np.asarray(gradient_fit["directions"], float)[mask]
        axis = np.asarray(pair[axis_key]["u"], float)
        strict = bool((validity.get(validity_key) or {}).get("strict_stability_pass"))
        rows.append(_template_row(
            subject_id=subject_id,
            dataset=dataset,
            subject=subject,
            template=template,
            template_rank=ranks[template],
            coords=coords,
            event_directions=direction_values,
            axis=axis,
            axis_strict=strict,
            n_perm=n_perm,
            min_events=min_events,
            seed=_seed_for(subject_id, "gradient", template, seed),
        ))

    return rows


def fold_subject_rows(template_rows: Sequence[Mapping[str, object]]) -> Sequence[Dict[str, object]]:
    grouped = defaultdict(list)
    for row in template_rows:
        if row.get("analysis_eligible"):
            grouped[(row["subject_id"], row["method"])].append(row)
    out = []
    for (subject_id, method), rows in sorted(grouped.items()):
        if {row["template"] for row in rows} != {"TA", "TB"}:
            continue
        rows = sorted(rows, key=lambda row: str(row["template"]))
        null_cos = np.mean(np.vstack([row["_null_cosine"] for row in rows]), axis=0)
        null_gap = np.mean(np.vstack([row["_null_main_gap"] for row in rows]), axis=0)
        observed = float(np.mean([row["mean_signed_cosine"] for row in rows]))
        observed_gap = float(np.mean([row["axis_to_main_direction_deg"] for row in rows]))
        null_cos_median = float(np.nanmedian(null_cos))
        null_gap_median = float(np.nanmedian(null_gap))
        out.append({
            "subject_id": subject_id,
            "pretty_subject": rows[0]["pretty_subject"],
            "dataset": rows[0]["dataset"],
            "subject": rows[0]["subject"],
            "method": method,
            "strict_stability_pass": bool(
                all(bool(row["axis_strict_stability"]) for row in rows)
            ),
            "n_events_ta": int(rows[0]["n_events"]),
            "n_events_tb": int(rows[1]["n_events"]),
            "mean_signed_cosine": observed,
            "null_mean_cosine_median": null_cos_median,
            "alignment_margin": observed - null_cos_median,
            "rank_shuffle_p_greater": float(
                (1 + np.sum(null_cos >= observed - 1e-15)) / (len(null_cos) + 1)
            ),
            "resultant_length_3d": float(
                np.mean([row["resultant_length_3d"] for row in rows])
            ),
            "axis_to_main_direction_deg": observed_gap,
            "null_main_gap_median_deg": null_gap_median,
            "main_gap_improvement_deg": null_gap_median - observed_gap,
            "fraction_within_30deg": float(
                np.mean([row["fraction_within_30deg"] for row in rows])
            ),
            "fraction_within_45deg": float(
                np.mean([row["fraction_within_45deg"] for row in rows])
            ),
        })
    return out


def _cohort_stat(values: Sequence[float], *, seed: int) -> Dict[str, object]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    lo, hi = bootstrap_median_ci(x, n_boot=5000, seed=seed)
    try:
        p = float(wilcoxon(x, alternative="greater").pvalue) if np.any(x != 0) else 1.0
    except ValueError:
        p = float("nan")
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "bootstrap_median_ci95": [lo, hi],
        "wilcoxon_greater_than_zero_p": p,
        "n_positive": int(np.sum(x > 0)),
    }


def summarize_cohort(subject_rows: Sequence[Mapping[str, object]], *, seed: int):
    gradient_rows = [row for row in subject_rows if row["method"] == "gradient"]
    strict_rows = [row for row in gradient_rows if row["strict_stability_pass"]]

    def summarize_rows(rows: Sequence[Mapping[str, object]], *, seed_offset: int):
        observed = np.asarray([row["mean_signed_cosine"] for row in rows], float)
        null = np.asarray([row["null_mean_cosine_median"] for row in rows], float)
        paired_p = (
            float(wilcoxon(observed, null, alternative="greater").pvalue)
            if observed.size and np.any(observed != null)
            else 1.0
        )
        return {
            "observed_vs_rank_shuffle_null": {
            "n": int(len(observed)),
            "observed_median": float(np.median(observed)),
            "observed_iqr": [
                float(np.percentile(observed, 25)),
                float(np.percentile(observed, 75)),
            ],
            "null_median": float(np.median(null)),
            "null_iqr": [
                float(np.percentile(null, 25)),
                float(np.percentile(null, 75)),
            ],
            "n_observed_gt_null": int(np.sum(observed > null)),
            "wilcoxon_greater_p": paired_p,
            },
            "alignment_margin": _cohort_stat(
                [row["alignment_margin"] for row in rows], seed=seed + seed_offset
            ),
            "main_gap_improvement_deg": _cohort_stat(
                [row["main_gap_improvement_deg"] for row in rows],
                seed=seed + seed_offset + 10,
            ),
        }

    primary = summarize_rows(gradient_rows, seed_offset=0)
    strict = summarize_rows(strict_rows, seed_offset=100)
    return {
        "observed_vs_rank_shuffle_null": primary["observed_vs_rank_shuffle_null"],
        "alignment_margin": {"gradient": primary["alignment_margin"]},
        "main_gap_improvement_deg": {"gradient": primary["main_gap_improvement_deg"]},
        "strict_stability_sensitivity": strict,
    }


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    clean = []
    for row in rows:
        clean.append({
            key: value for key, value in row.items()
            if not str(key).startswith("_") and key != "mean_direction"
        })
    columns = sorted({key for row in clean for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_jsonable(clean))


def plot_cohort(
    subject_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    out_dir: Path,
) -> Dict[str, Path]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    gradient_rows = [row for row in subject_rows if row["method"] == "gradient"]
    if not gradient_rows:
        raise ValueError("no subject has both eligible gradient templates")
    paired = summary["observed_vs_rank_shuffle_null"]
    rows = [
        {
            "subject_id": row["subject_id"],
            "data": float(row["mean_signed_cosine"]),
            "null": float(row["null_mean_cosine_median"]),
        }
        for row in gradient_rows
    ]
    png = figures / "axis_representativeness_cohort.png"
    pdf = figures / "axis_representativeness_cohort.pdf"
    plot_paired_data_null_groups(
        [{
            "label": "Estimable gradient axes",
            "rows": rows,
            "summary": {
                "n": paired["n"],
                "wilcoxon_p_data_gt_null_median": paired["wilcoxon_greater_p"],
            },
            "caption": "",
            "bracket_text": "***",
            "bracket_fontsize": 13,
            "bracket_fontweight": "bold",
        }],
        png,
        pdf,
        ylabel=(
            "Axis–event propagation alignment\n"
            "(signed cosine: 0 = none, 1 = same)"
        ),
        seed=20260718,
        figsize=(4.8, 4.15),
        pair_gap=0.85,
        ylim=(-0.08, 1.02),
        connect_pairs=False,
        point_jitter_sd=0.0,
        pair_tick_labels=(
            "Fitted template-gradient\naxis",
            "Montage-matched\nrank-shuffled axis",
        ),
        zero_reference=True,
        bottom=0.25,
        ylabel_fontsize=9.5,
    )
    paper_figures = PAPER_OUT / "figures"
    paper_figures.mkdir(parents=True, exist_ok=True)
    paper_png = paper_figures / png.name
    paper_pdf = paper_figures / pdf.name
    shutil.copy2(png, paper_png)
    shutil.copy2(pdf, paper_pdf)
    return {
        "png": png,
        "pdf": pdf,
        "paper_png": paper_png,
        "paper_pdf": paper_pdf,
    }


def _write_readme(out_dir: Path, summary: Mapping[str, object]) -> None:
    g = summary["alignment_margin"]["gradient"]
    lines = [
        "# 模板轴对单事件主方向的代表性统计",
        "",
        "### axis_representativeness_cohort.png",
        "",
        (
            "主图检验所有可拟合且具有足量二维 QC-clean 单事件的 gradient axis。以 subject 为统计单位，先在每个患者内等权折叠 TA/TB，再比较真实轴的 mean signed cosine "
            "与同一 montage 上 template-rank shuffle 重建假轴的中位数；正值表示真实轴比几何匹配的假轴更能代表事件方向。"
        ),
        (
            f"图形复用 paper-ready `fig3_field_concordance_cohort_stat` 的 Data-vs-Null 语法：violin + IQR box + subject points + 显著性括号。"
            "由于真实值与 null 已完全分离，正式版不画患者内连线；所有 subject 点固定在对应类别中心，不使用随机 jitter。"
            f"Strict stability 不作为主分析纳入门；主分析 n={g['n']}，strict 子集仅作为 sensitivity。"
        ),
        "箱体为四分位距、黑线为中位数；事件均向角度改善作为次要表达仅保留在 summary JSON 和 CSV，不重复占用正式画布。",
        "主效应量是患者级 alignment margin：真实 template-gradient axis 与单事件传播方向的 mean signed cosine，减去同患者 montage-matched rank-shuffle null 的中位数。",
        "",
        "**关注点**：看真实 gradient axis 是否在患者内系统高于 rank-shuffle null；这是同数据 descriptive representativeness，不是 held-out generalization。",
        "",
    ]
    (out_dir / "figures" / "README.md").write_text("\n".join(lines))


def _write_paper_ready_sidecars(result: Mapping[str, object]) -> None:
    figures = PAPER_OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    flow = result["cohort_flow"]
    gradient_n = flow["n_subjects_with_both_templates_by_method"]["gradient"]
    strict_n = flow["n_subjects_strict_stability_sensitivity"]
    observed_null = result["cohort"]["observed_vs_rank_shuffle_null"]
    gradient_stat = result["cohort"]["alignment_margin"]["gradient"]
    readme = [
        "# Axis representativeness",
        "",
        "### axis_representativeness_cohort.png",
        "",
        "The panel tests every estimable interictal gradient axis with sufficient two-dimensional QC-clean single-event directions. Observed mean signed cosine is paired within patient to the median from axes rebuilt after shuffling template ranks over the same contact geometry; Template A and Template B are folded with equal weight.",
        (
            "The visual grammar is adapted from the accepted paper-ready `fig3_field_concordance_cohort_stat`: blue observed and gray null violins, IQR boxes, centered patient points, and a significance bracket. Pairing lines and random horizontal jitter are omitted because the distributions are completely separated. "
            f"The source pool contained {flow['n_subjects_requested']} candidate patients, of whom {flow['n_subjects_with_traceable_inputs']} had traceable axis inputs; both templates had sufficient event directions in {gradient_n} patients. Strict stability is not an admission gate and is retained as an n={strict_n} sensitivity analysis."
        ),
        "The analysis is descriptive and in-sample because the template axis and event pool originate from the same recording corpus; it does not establish held-out generalization.",
        "",
        "**关注点**：真实 gradient axis 是否在患者内系统高于 rank-shuffle null；不要把事件数当成统计样本量。",
        "",
    ]
    (figures / "README.md").write_text("\n".join(readme))
    paper_result = {
        "contract": {
            key: value for key, value in result["contract"].items()
            if not key.startswith("endpoint_")
        },
        "cohort": {
            "gradient_observed_vs_rank_shuffle_null": observed_null,
            "gradient_alignment_margin": gradient_stat,
            "main_gap_improvement_deg": result["cohort"]["main_gap_improvement_deg"]["gradient"],
            "strict_stability_sensitivity": result["cohort"]["strict_stability_sensitivity"],
        },
        "cohort_flow": {
            "n_subjects_requested": flow["n_subjects_requested"],
            "n_subjects_with_traceable_inputs": flow["n_subjects_with_traceable_inputs"],
            "n_subjects_skipped": flow["n_subjects_skipped"],
            "n_gradient_template_rows": flow["n_gradient_template_rows"],
            "n_eligible_gradient_template_rows": flow["n_eligible_gradient_template_rows"],
            "n_subjects_with_both_gradient_templates": gradient_n,
            "n_subjects_strict_stability_sensitivity": strict_n,
        },
        "visual_reference": (
            "results/paper-ready-figure/fig3_field_concordance_cohort_stat/"
            "figures/field_concordance_cohort_stat.png"
        ),
        "visual_contract": {
            "pairing_lines": False,
            "horizontal_point_jitter": False,
            "x_tick_observed": "Fitted template-gradient axis",
            "x_tick_null": "Montage-matched rank-shuffled axis",
            "zero_reference": "signed cosine = 0",
            "p_value_display": "stars only",
            "right_side_numeric_annotation": False,
        },
        "outputs": result["outputs"],
    }
    (figures / "axis_representativeness_cohort_metadata.json").write_text(
        json.dumps(_jsonable(paper_result), ensure_ascii=False, indent=2)
    )


def run(
    subjects: Sequence[str],
    *,
    out_dir: Path,
    n_perm: int,
    min_events: int,
    max_events: int,
    seed: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    template_rows = []
    run_rows = []
    for index, subject_id in enumerate(subjects, 1):
        try:
            rows = process_subject(
                subject_id,
                n_perm=n_perm,
                min_events=min_events,
                max_events=max_events,
                seed=seed,
            )
            template_rows.extend(rows)
            run_rows.append({"subject_id": subject_id, "status": "ok", "reason": ""})
            n_eligible = sum(bool(row["analysis_eligible"]) for row in rows)
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"eligible gradient template rows={n_eligible}/2",
                flush=True,
            )
        except Exception as exc:
            run_rows.append({"subject_id": subject_id, "status": "skip", "reason": str(exc)[:240]})
            print(f"[{index:02d}/{len(subjects)}] {subject_id}: skip ({exc})", flush=True)
    subject_rows = fold_subject_rows(template_rows)
    summary = summarize_cohort(subject_rows, seed=seed)
    outputs = plot_cohort(subject_rows, summary, out_dir)
    _write_csv(template_rows, out_dir / "per_template_axis_representativeness.csv")
    _write_csv(subject_rows, out_dir / "subject_folded_axis_representativeness.csv")
    _write_csv(run_rows, out_dir / "run_status.csv")
    result = {
        "contract": {
            "statistical_unit": "subject",
            "within_subject_fold": "equal mean of TA and TB",
            "primary_metric": "mean signed cosine observed minus template-rank-shuffle null median",
            "primary_effect_interpretation": "how much better the fitted template-gradient axis represents single-event interictal propagation directions than montage-matched rank-shuffled axes",
            "null": "shuffle template ranks over the same mapped contacts and rebuild the same axis estimator",
            "gradient_event_gate": "mapped contacts>=6; shafts>=2; effective rank>=2; LOCO valid>=0.8; median signed cosine>=0.8",
            "gradient_template_gate": "axis estimable; strict frozen-axis stability is sensitivity only, not an admission gate",
            "minimum_events_per_template": int(min_events),
            "n_rank_shuffles": int(n_perm),
            "max_events_per_subject": int(max_events),
            "seed": int(seed),
            "claim_boundary": "in-sample descriptive representativeness; not held-out generalization",
            "ictal_input": "none",
            "paper_ready_producer": "scripts/paper_figures/plot_axis_representativeness.py",
        },
        "cohort": summary,
        "cohort_flow": {
            "n_subjects_requested": int(len(subjects)),
            "n_subjects_with_traceable_inputs": int(
                sum(row["status"] == "ok" for row in run_rows)
            ),
            "n_subjects_skipped": int(sum(row["status"] == "skip" for row in run_rows)),
            "n_gradient_template_rows": int(len(template_rows)),
            "n_eligible_gradient_template_rows": int(sum(
                bool(row["analysis_eligible"]) for row in template_rows
            )),
            "n_subjects_with_both_templates_by_method": {
                "gradient": int(len(subject_rows))
            },
            "n_subjects_strict_stability_sensitivity": int(sum(
                bool(row["strict_stability_pass"]) for row in subject_rows
            )),
        },
        "n_template_rows": len(template_rows),
        "n_subject_rows": len(subject_rows),
        "outputs": outputs,
    }
    (out_dir / "axis_representativeness_summary.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2)
    )
    _write_readme(out_dir, summary)
    _write_paper_ready_sidecars(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    parser.add_argument("--min-events", type=int, default=DEFAULT_MIN_EVENTS)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    subjects = args.subjects or sorted(
        path.stem for path in (FROZEN_ROOT / "per_subject").glob("*.json")
    )
    run(
        subjects,
        out_dir=args.out_dir,
        n_perm=args.n_perm,
        min_events=args.min_events,
        max_events=args.max_events,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
