#!/usr/bin/env python3
"""Clinical-SOZ contact compactness against subject-specific spatial nulls.

Primary null: equal-size subsets from every mapped invasive contact.
Sensitivity null: exact SOZ-contact count preserved within each electrode
lead/array, using the channel-name prefix as the grouping unit.

Outputs live under results/spatial_modulation/soz_contact_compactness/ and are
subject-first: one row/JSON per patient, one cohort summary, and one two-panel
candidate supplementary figure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft
from src.seeg_coord_loader import (
    _canonicalize_epilepsiae_subject_id,
    _read_epilepsiae_electrode_sql,
    _read_yuquan_chnXyzDict,
    load_subject_coords,
)
from src.soz_spatial_compactness import analyze_subject_compactness


SOZ_FILES = {
    "yuquan": ROOT / "results/yuquan_soz_core_channels.json",
    "epilepsiae": ROOT / "results/epilepsiae_soz_core_channels.json",
}
DEFAULT_OUT = ROOT / "results/spatial_modulation/soz_contact_compactness"
DATASET_COLORS = {"yuquan": "#86A9C2", "epilepsiae": "#8A4C9C"}
DATASET_LABELS = {"yuquan": "Yuquan", "epilepsiae": "Epilepsiae"}


def _p_to_stars(p_value: float) -> str:
    """Conventional significance label used only for the cohort-vs-null test."""
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _seed_for_subject(seed: int, dataset: str, subject: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _all_source_contact_names(dataset: str, subject: str) -> tuple[list[str], dict[str, int]]:
    """Enumerate all invasive contacts with finite source coordinates."""
    if dataset == "yuquan":
        shaft_dict = _read_yuquan_chnXyzDict(subject)
        names: list[str] = []
        n_source = 0
        for shaft, raw in sorted(shaft_dict.items()):
            arr = np.asarray(raw, dtype=float)
            n_source += int(arr.shape[0])
            for i, xyz in enumerate(arr, start=1):
                if np.isfinite(xyz).all():
                    names.append(f"{shaft}{i}")
        return names, {"n_source_invasive": n_source, "n_source_finite": len(names)}

    canonical = _canonicalize_epilepsiae_subject_id(subject)
    rows, _ = _read_epilepsiae_electrode_sql(canonical)
    invasive = [row for row in rows if row.invasive is True]
    names = []
    seen = set()
    for row in invasive:
        if row.name in seen:
            raise ValueError(f"duplicate invasive contact {dataset}:{subject}:{row.name}")
        seen.add(row.name)
        if None not in (row.coord_x, row.coord_y, row.coord_z):
            names.append(row.name)
    return names, {
        "n_source_invasive": len(invasive),
        "n_source_finite": len(names),
    }


def _load_all_mapped_contacts(dataset: str, subject: str) -> tuple[list[str], np.ndarray, str, dict]:
    names, inventory = _all_source_contact_names(dataset, subject)
    coord = load_subject_coords(dataset, subject, names, allow_voxel_fallback=False)
    mask = np.asarray(coord.mapped_mask_in_requested_order, dtype=bool)
    pts = np.asarray(coord.coords_array_in_requested_order, dtype=float)
    keep = mask & np.isfinite(pts).all(axis=1)
    mapped_names = [name for name, ok in zip(names, keep) if ok]
    mapped_coords = pts[keep]
    inventory = dict(inventory)
    inventory["n_loader_mapped"] = int(keep.sum())
    return mapped_names, mapped_coords, coord.coord_space, inventory


def _eligibility_reason(
    *, n_all: int, n_soz_labeled: int, n_soz_mapped: int, coverage: float,
    min_soz_coverage: float,
) -> str:
    if n_soz_labeled == 0:
        return "no_clinical_soz_labels"
    if n_soz_mapped < 2:
        return "fewer_than_2_mapped_soz_contacts"
    if coverage < min_soz_coverage:
        return f"soz_coordinate_coverage_below_{min_soz_coverage:.2f}"
    if n_all <= n_soz_mapped:
        return "no_mapped_nonsoz_control_contacts"
    return ""


def _analyze_one(
    dataset: str,
    subject: str,
    soz_names: list[str],
    *,
    n_null: int,
    seed: int,
    min_soz_coverage: float,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset": dataset,
        "subject": subject,
        "status": "excluded",
        "exclusion_reason": "",
        "n_soz_labeled": int(len(set(soz_names))),
        "min_soz_coordinate_coverage": float(min_soz_coverage),
    }
    try:
        names, coords, coord_space, inventory = _load_all_mapped_contacts(dataset, subject)
    except FileNotFoundError as exc:
        base["exclusion_reason"] = "coordinate_source_unavailable"
        base["error"] = str(exc)
        return base

    name_set = set(names)
    soz_unique = sorted(set(soz_names))
    mapped_soz = [name for name in soz_unique if name in name_set]
    coverage = len(mapped_soz) / len(soz_unique) if soz_unique else 0.0
    base.update(
        coord_space=coord_space,
        contact_inventory=inventory,
        n_mapped_contacts=int(len(names)),
        n_soz_mapped=int(len(mapped_soz)),
        soz_coordinate_coverage=float(coverage),
        mapped_soz_contacts=mapped_soz,
        missing_soz_contacts=sorted(set(soz_unique) - name_set),
    )
    reason = _eligibility_reason(
        n_all=len(names),
        n_soz_labeled=len(soz_unique),
        n_soz_mapped=len(mapped_soz),
        coverage=coverage,
        min_soz_coverage=min_soz_coverage,
    )
    if reason:
        base["exclusion_reason"] = reason
        return base

    shafts: list[str | None] = [parse_shaft(name)[0] for name in names]
    rng = np.random.default_rng(_seed_for_subject(seed, dataset, subject))
    result = analyze_subject_compactness(
        names,
        coords,
        mapped_soz,
        shafts,
        n_null=n_null,
        rng=rng,
    )
    base.update(
        status="included",
        exclusion_reason="",
        n_shafts_all=int(len({s for s in shafts if s is not None})),
        n_shafts_soz=int(len({parse_shaft(name)[0] for name in mapped_soz} - {None})),
        n_unparseable_shafts=int(sum(s is None for s in shafts)),
        analysis=result,
    )
    return base


def _bootstrap_median_ci(values: np.ndarray, *, seed: int, n_boot: int = 20000) -> list[float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, vals.size), replace=True)
    meds = np.median(draws, axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _summarize_records(
    records: list[dict],
    null_key: str,
    *,
    metric_key: str = "rms_radius",
    seed: int,
) -> dict[str, Any]:
    usable = []
    for rec in records:
        if rec.get("status") != "included":
            continue
        null = rec["analysis"].get(null_key, {})
        if not null.get("available"):
            continue
        ratio = null[metric_key]["observed_to_null_median_ratio"]
        p_left = null[metric_key]["p_left"]
        if np.isfinite(ratio) and ratio > 0 and np.isfinite(p_left):
            usable.append((rec["dataset"], ratio, p_left))

    def one(rows: list[tuple[str, float, float]], offset: int) -> dict[str, Any]:
        ratios = np.asarray([row[1] for row in rows], dtype=float)
        pvals = np.asarray([row[2] for row in rows], dtype=float)
        n = int(ratios.size)
        if n == 0:
            return {"n": 0}
        log_ratio = np.log2(ratios)
        try:
            w = wilcoxon(log_ratio, alternative="less", zero_method="wilcox", method="auto")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        n_subject_sig = int(np.count_nonzero(pvals < 0.05))
        return {
            "n": n,
            "median_observed_to_null_ratio": float(np.median(ratios)),
            "median_ratio_bootstrap_95ci": _bootstrap_median_ci(
                ratios, seed=seed + offset
            ),
            "n_ratio_below_1": int(np.count_nonzero(ratios < 1.0)),
            "n_subject_p_lt_0_05": n_subject_sig,
            "wilcoxon_log2_ratio_less_than_0": {
                "statistic": w_stat,
                "p_value": w_p,
            },
            "binomial_subject_pass_rate_vs_0.05": {
                "successes": n_subject_sig,
                "trials": n,
                "p_value": float(binomtest(n_subject_sig, n, 0.05, alternative="greater").pvalue),
            },
        }

    return {
        "overall": one(usable, 0),
        "yuquan": one([row for row in usable if row[0] == "yuquan"], 101),
        "epilepsiae": one([row for row in usable if row[0] == "epilepsiae"], 202),
    }


def _flatten_record(rec: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": rec["dataset"],
        "subject": rec["subject"],
        "status": rec["status"],
        "exclusion_reason": rec.get("exclusion_reason", ""),
        "coord_space": rec.get("coord_space", ""),
        "n_mapped_contacts": rec.get("n_mapped_contacts"),
        "n_soz_labeled": rec.get("n_soz_labeled"),
        "n_soz_mapped": rec.get("n_soz_mapped"),
        "soz_coordinate_coverage": rec.get("soz_coordinate_coverage"),
        "n_shafts_all": rec.get("n_shafts_all"),
        "n_shafts_soz": rec.get("n_shafts_soz"),
    }
    if rec.get("status") != "included":
        return row
    analysis = rec["analysis"]
    row.update(
        observed_soz_rms_mm=analysis["observed_soz"]["rms_radius_mm"],
        observed_soz_median_pairwise_mm=analysis["observed_soz"]["median_pairwise_mm"],
        all_contacts_rms_mm=analysis["all_contacts"]["rms_radius_mm"],
        soz_to_all_rms_ratio=analysis["soz_to_all_rms_ratio"],
    )
    for prefix, key in (("all_null", "all_contact_null"), ("shaft_null", "shaft_stratified_null")):
        null = analysis[key]
        row[f"{prefix}_available"] = bool(null.get("available"))
        row[f"{prefix}_reason"] = null.get("reason", "")
        if null.get("available"):
            for metric_name, metric in (("rms", null["rms_radius"]),
                                        ("pairwise", null["median_pairwise"])):
                row[f"{prefix}_{metric_name}_median_mm"] = metric["null_median"]
                row[f"{prefix}_{metric_name}_ratio"] = metric["observed_to_null_median_ratio"]
                row[f"{prefix}_{metric_name}_p_left"] = metric["p_left"]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw_compactness_panel(
    ax,
    records: list[dict],
    null_key: str,
    summary: dict,
    *,
    dataset_colors: dict[str, str] | None = None,
    point_size: float = 42.0,
    annotation_fontsize: float = 7.2,
    tick_fontsize: float | None = None,
    line_scale: float = 1.0,
) -> None:
    """Draw the accepted subject-level SOZ/null compactness panel.

    Optional style parameters let paper-ready composite builders reuse the
    accepted painter without copying its statistical or visual encoding.
    """
    colors = DATASET_COLORS if dataset_colors is None else dataset_colors
    rng = np.random.default_rng(20260819)
    counts: dict[str, int] = {}
    dataset_medians: dict[str, float] = {}
    for xpos, dataset in enumerate(("yuquan", "epilepsiae")):
        vals = []
        pvals = []
        for rec in records:
            if rec.get("status") != "included" or rec["dataset"] != dataset:
                continue
            null = rec["analysis"].get(null_key, {})
            if not null.get("available"):
                continue
            vals.append(null["rms_radius"]["observed_to_null_median_ratio"])
            pvals.append(null["rms_radius"]["p_left"])
        vals_arr = np.asarray(vals, dtype=float)
        pvals_arr = np.asarray(pvals, dtype=float)
        if vals_arr.size == 0:
            continue
        counts[dataset] = int(vals_arr.size)
        jitter = rng.uniform(-0.09, 0.09, size=vals_arr.size)
        significant = pvals_arr < 0.05
        ax.scatter(
            xpos + jitter[~significant], vals_arr[~significant], s=point_size * 0.90,
            facecolors="white", edgecolors=colors[dataset],
            linewidths=1.2 * line_scale,
            zorder=3,
        )
        ax.scatter(
            xpos + jitter[significant], vals_arr[significant], s=point_size,
            facecolors=colors[dataset], edgecolors="white",
            linewidths=0.7 * line_scale,
            zorder=4,
        )
        med = float(np.median(vals_arr))
        dataset_medians[dataset] = med
        q1, q3 = np.percentile(vals_arr, [25, 75])
        ax.plot(
            [xpos - 0.18, xpos + 0.18], [med, med],
            color="black", lw=2.2 * line_scale, zorder=5,
        )
        ax.plot(
            [xpos, xpos], [q1, q3],
            color="black", lw=1.6 * line_scale, zorder=5,
        )

    ax.axhline(1.0, color="#777777", lw=1.2 * line_scale, ls="--", zorder=1)
    ax.text(
        -0.22, 1.025, "Null", color="#666666",
        fontsize=annotation_fontsize, ha="left", va="bottom",
    )

    # Each bracket is the within-dataset subject-level Wilcoxon against null=1.
    for xpos, dataset in enumerate(("yuquan", "epilepsiae")):
        group_median = dataset_medians.get(dataset, float("nan"))
        group_p = (
            summary.get(dataset, {})
            .get("wilcoxon_log2_ratio_less_than_0", {})
            .get("p_value", float("nan"))
        )
        if not (np.isfinite(group_median) and group_median > 0):
            continue
        bracket_x = xpos + 0.22
        cap = 0.055
        ax.plot(
            [bracket_x - cap, bracket_x, bracket_x, bracket_x - cap],
            [1.0, 1.0, group_median, group_median],
            color="black", lw=1.0 * line_scale, clip_on=False, zorder=6,
        )
        group_p_text = f"P={group_p:.2g}" if np.isfinite(group_p) else "P=n/a"
        ax.text(
            bracket_x + 0.035,
            float(np.sqrt(group_median)),
            f"{_p_to_stars(group_p)}\n{group_p_text}",
            ha="left", va="center", fontsize=annotation_fontsize,
            fontweight="bold",
        )

    ax.set_yscale("log", base=2)
    ax.set_ylim(0.0625, 2.0)
    ax.set_xlim(-0.25, 1.75)
    ax.set_yticks([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
    ax.set_yticklabels(["0.0625", "0.125", "0.25", "0.5", "1", "2"])
    ax.set_xticks(
        [0, 1],
        [
            f"{DATASET_LABELS['yuquan']}\nn={counts.get('yuquan', 0)}",
            f"{DATASET_LABELS['epilepsiae']}\nn={counts.get('epilepsiae', 0)}",
        ],
    )
    for tick, dataset in zip(ax.get_xticklabels(), ("yuquan", "epilepsiae")):
        tick.set_color(colors[dataset])
        tick.set_fontweight("bold")
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.spines[["top", "right"]].set_visible(False)


def _plot(records: list[dict], summary: dict, figure_dir: Path) -> tuple[Path, Path]:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.4))
    draw_compactness_panel(
        ax, records, "all_contact_null",
        summary["all_contact_primary"],
    )
    ax.set_ylabel("SOZ RMS radius / null median")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.97)
    png = figure_dir / "soz_contact_spatial_compactness.png"
    pdf = figure_dir / "soz_contact_spatial_compactness.pdf"
    fig.savefig(png, dpi=400, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def _write_figure_readme(figure_dir: Path, summary: dict) -> None:
    p = summary["all_contact_primary"]["overall"]
    s = summary["within_shaft_sensitivity"]["overall"]
    pp = summary["all_contact_pairwise_sensitivity"]["overall"]
    sp = summary["within_shaft_pairwise_sensitivity"]["overall"]
    text = f"""### soz_contact_spatial_compactness.png

**Clinical SOZ contact compactness relative to the subject-specific all-contact null.** 图只保留最关键的主分析：比较每位患者 clinical SOZ 触点的三维 RMS 半径与该患者全部可映射颅内触点中等数量随机子集的 null。点是患者，实心点表示患者内经验 P<0.05，黑线为中位数和 IQR，虚线 1 表示与 null 中位数相同。Yuquan 与 Epilepsiae 各自右侧的括号分别连接该数据集的中位比值与 null=1，星号和精确 P 值来自数据集内患者级 log2 比值的单侧 Wilcoxon 检验（* P<0.05，** P<0.01，*** P<0.001）。

主分析共 n={p['n']}，SOZ/null 半径比中位数为 {p['median_observed_to_null_ratio']:.3f}。未画出的 within-lead/array 敏感性共 n={s['n']}，中位数为 {s['median_observed_to_null_ratio']:.3f}；稳健的 median pairwise distance 也给出同向结果：all-contact 比值中位数 {pp['median_observed_to_null_ratio']:.3f}，within-lead/array 比值中位数 {sp['median_observed_to_null_ratio']:.3f}。敏感性结果继续保留在 `cohort_summary.json`，不占用当前主图空间。

**关注点**：图中只展示 all-contact 主 null；结论成立仍要求同时核对未画出的 within-lead/array 敏感性。本图不检验传播 endpoint 是否位于 SOZ，也不证明群体事件局限于 clinical SOZ。
"""
    (figure_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-null", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--min-soz-coverage", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not (0 < args.min_soz_coverage <= 1):
        raise ValueError("--min-soz-coverage must be in (0, 1]")

    out = args.output_dir.resolve()
    per_subject_dir = out / "per_subject"
    figure_dir = out / "figures"
    per_subject_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for dataset in ("yuquan", "epilepsiae"):
        labels = json.loads(SOZ_FILES[dataset].read_text())
        for subject, soz_names in sorted(labels.items()):
            rec = _analyze_one(
                dataset,
                str(subject),
                list(soz_names),
                n_null=args.n_null,
                seed=args.seed,
                min_soz_coverage=args.min_soz_coverage,
            )
            records.append(rec)
            path = per_subject_dir / f"{dataset}_{subject}.json"
            path.write_text(json.dumps(_json_safe(rec), indent=2, ensure_ascii=False), encoding="utf-8")

    included = [rec for rec in records if rec["status"] == "included"]
    summary = {
        "schema_version": "soz_contact_spatial_compactness_v1",
        "topic": "topic3_spatial_soz_modulation",
        "status": "analysis_complete_figure_candidate",
        "question": "Are clinical SOZ contacts spatially compact relative to each patient's mapped invasive-contact geometry?",
        "primary_metric": "RMS distance of mapped clinical SOZ contacts to their own centroid",
        "primary_null": "equal-size subsets from all mapped invasive contacts within subject",
        "sensitivity_null": "exact SOZ-contact count preserved within every electrode lead/array (channel-name prefix)",
        "independent_unit": "subject",
        "n_null_per_subject": int(args.n_null),
        "seed": int(args.seed),
        "min_soz_coordinate_coverage": float(args.min_soz_coverage),
        "n_labelled_subjects": int(len(records)),
        "n_primary_included": int(len(included)),
        "n_excluded": int(len(records) - len(included)),
        "exclusion_counts": {},
        "all_contact_primary": _summarize_records(
            records, "all_contact_null", seed=args.seed
        ),
        "within_shaft_sensitivity": _summarize_records(
            records, "shaft_stratified_null", seed=args.seed + 1000
        ),
        "all_contact_pairwise_sensitivity": _summarize_records(
            records, "all_contact_null", metric_key="median_pairwise",
            seed=args.seed + 2000,
        ),
        "within_shaft_pairwise_sensitivity": _summarize_records(
            records, "shaft_stratified_null", metric_key="median_pairwise",
            seed=args.seed + 3000,
        ),
        "claim_boundary": [
            "Supports or rejects spatial compactness of clinical SOZ contacts in 3D SEEG contact space.",
            "Does not test whether propagation endpoints are SOZ-enriched.",
            "Does not establish that population-event contacts are confined to clinical SOZ.",
            "Yuquan native-RAS and Epilepsiae MNI coordinates are analyzed only within subject; point clouds are never pooled.",
        ],
    }
    for rec in records:
        if rec["status"] == "included":
            continue
        reason = rec.get("exclusion_reason", "unknown")
        summary["exclusion_counts"][reason] = summary["exclusion_counts"].get(reason, 0) + 1

    flat = [_flatten_record(rec) for rec in records]
    _write_csv(out / "subject_compactness.csv", flat)
    _write_csv(out / "exclusion_inventory.csv", [row for row in flat if row["status"] != "included"])
    (out / "cohort_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    png, pdf = _plot(records, summary, figure_dir)
    _write_figure_readme(figure_dir, summary)
    metadata = {
        "figure": "Clinical SOZ contact spatial compactness (candidate supplementary figure)",
        "source_summary": str(out / "cohort_summary.json"),
        "source_table": str(out / "subject_compactness.csv"),
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "displayed_analysis": summary["all_contact_primary"],
        "supporting_analysis_not_displayed": summary["within_shaft_sensitivity"],
        "significance_encoding": {
            "brackets": "within-dataset subject-level observed/null ratio versus null=1",
            "test": "one-sided Wilcoxon signed-rank on log2 ratio",
            "thresholds": {"*": "P<0.05", "**": "P<0.01", "***": "P<0.001"},
        },
        "claim_boundary": summary["claim_boundary"],
    }
    (out / "figure_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
