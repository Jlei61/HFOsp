#!/usr/bin/env python3
"""Render geometry-stable interictal TA/TB single-event direction roses.

This is the QC-clean companion to ``plot_topic5_interictal_template_direction_rose``.
The original n>=3 plot remains the all-event sensitivity view.  Here an event
must independently support a stable signed gradient; proximity to either frozen
template direction is never part of the inclusion rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    DEFAULT_MAX_EVENTS,
    DEFAULT_SEED,
    FROZEN_ROOT,
    TA_COLOR,
    TB_COLOR,
    _jsonable,
    _load_frozen_record,
    _load_masked_events_and_labels,
    _pretty_subject,
)
from src.topic5_interictal_direction_rose import (  # noqa: E402
    assess_event_direction_qc,
    axis_pair_display_basis,
    fit_event_directions_3d,
    project_directions_to_angles,
    resultant_length,
)

DEFAULT_OUT = ROOT / "results/interictal_propagation_masked/template_direction_rose_qc_clean"
DEFAULT_MIN_CONTACTS = 6
DEFAULT_MIN_SHAFTS = 2
DEFAULT_MIN_EFFECTIVE_RANK = 2
DEFAULT_MIN_LOCO_VALID_FRACTION = 0.8
DEFAULT_MIN_LOCO_MEDIAN_SIGNED_COSINE = 0.8


def _qc_contract(
    *,
    min_contacts: int,
    min_shafts: int,
    min_effective_rank: int,
    min_loco_valid_fraction: float,
    min_loco_median_signed_cosine: float,
) -> Dict[str, object]:
    return {
        "minimum_mapped_participating_contacts": int(min_contacts),
        "minimum_participating_shafts": int(min_shafts),
        "minimum_effective_coordinate_rank": int(min_effective_rank),
        "minimum_loco_valid_fraction": float(min_loco_valid_fraction),
        "minimum_loco_median_signed_cosine": float(min_loco_median_signed_cosine),
        "loco_valid_refit_requires_effective_rank": int(min_effective_rank),
        "template_angle_used_for_inclusion": False,
        "r_squared_used_for_inclusion": False,
    }


def build_subject_payload(
    subject_id: str,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    seed: int = DEFAULT_SEED,
    min_contacts: int = DEFAULT_MIN_CONTACTS,
    min_shafts: int = DEFAULT_MIN_SHAFTS,
    min_effective_rank: int = DEFAULT_MIN_EFFECTIVE_RANK,
    min_loco_valid_fraction: float = DEFAULT_MIN_LOCO_VALID_FRACTION,
    min_loco_median_signed_cosine: float = DEFAULT_MIN_LOCO_MEDIAN_SIGNED_COSINE,
) -> Dict[str, object]:
    record = _load_frozen_record(subject_id)
    events = _load_masked_events_and_labels(record, max_events=max_events, seed=seed)
    pair = record["axis_pair"]
    axis_a, axis_b = pair["axis_a"], pair["axis_b"]
    basis = axis_pair_display_basis(
        axis_a["u"], axis_b["u"], fallback_transverse=axis_a.get("w")
    )
    coords = np.asarray(record["coords"], float)
    fitted = fit_event_directions_3d(events["event_ranks"], coords, min_contacts=3)
    projected = project_directions_to_angles(
        fitted["directions"], basis["axis_a"], basis["transverse"]
    )
    qc = assess_event_direction_qc(
        events["event_ranks"],
        coords,
        record["shafts"],
        directions=fitted["directions"],
        n_valid_contacts=fitted["n_valid_contacts"],
        effective_rank=fitted["effective_rank"],
        min_contacts=min_contacts,
        min_shafts=min_shafts,
        min_effective_rank=min_effective_rank,
        min_loco_valid_fraction=min_loco_valid_fraction,
        min_loco_median_signed_cosine=min_loco_median_signed_cosine,
    )

    labels = np.asarray(events["labels"], int)
    angles = np.asarray(projected["angles"], float)
    angle_valid = np.isfinite(angles)
    qc_pass = np.asarray(qc["passes"], bool) & angle_valid
    groups_all = {
        label: angles[(labels == label) & angle_valid] for label in (0, 1)
    }
    groups_qc = {
        label: angles[(labels == label) & qc_pass] for label in (0, 1)
    }
    relation = pair.get("relation") or {}
    return {
        "subject_id": subject_id,
        "pretty_subject": _pretty_subject(subject_id),
        "groups_all": groups_all,
        "groups_qc": groups_qc,
        "basis": basis,
        "relation": relation,
        "geometry_2d_supported": bool(pair.get("geometry_2d_supported")),
        "strict_stability_pass": bool(pair.get("strict_stability_pass")),
        "n_events_total": events["n_events_total"],
        "n_events_sampled": int(len(events["selection"])),
        "n_assigned_ta": int(np.sum(labels == 0)),
        "n_assigned_tb": int(np.sum(labels == 1)),
        "n_unassigned": int(np.sum(labels < 0)),
        "n_direction_all_ta": int(groups_all[0].size),
        "n_direction_all_tb": int(groups_all[1].size),
        "n_direction_qc_ta": int(groups_qc[0].size),
        "n_direction_qc_tb": int(groups_qc[1].size),
        "resultant_qc_ta": resultant_length(groups_qc[0]),
        "resultant_qc_tb": resultant_length(groups_qc[1]),
        "event_selection_indices": events["selection"],
        "event_projection_norm": projected["projection_norm"],
        "event_qc": qc,
        "qc_contract": _qc_contract(
            min_contacts=min_contacts,
            min_shafts=min_shafts,
            min_effective_rank=min_effective_rank,
            min_loco_valid_fraction=min_loco_valid_fraction,
            min_loco_median_signed_cosine=min_loco_median_signed_cosine,
        ),
        "axis_a_u": np.asarray(axis_a["u"], float),
        "axis_b_u": np.asarray(axis_b["u"], float),
        "axis_definition": record["axis_definition"],
        "axis_direction_convention": record["axis_direction_convention"],
        "frozen_artifact": FROZEN_ROOT / "per_subject" / f"{subject_id}.json",
        "rank_source": events["rank_source"],
        "max_events": int(max_events),
        "seed": int(seed),
    }


def _format_resultant(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2f}"


def plot_subject(
    payload: Mapping[str, object], out_dir: Path, *, bins: int = 18
) -> Dict[str, Path]:
    subject_id = str(payload["subject_id"])
    figures = out_dir / "figures"
    metadata_dir = out_dir / "per_subject"
    figures.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    centers = edges[:-1] + 0.5 * np.diff(edges)
    width = float(np.diff(edges)[0] * 0.95)
    fig = plt.figure(figsize=(9.1, 9.2))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.86, bottom=0.23)
    ax = fig.add_subplot(111, projection="polar")
    rmax = 1
    for label, color, name in ((0, TA_COLOR, "TA"), (1, TB_COLOR, "TB")):
        values = np.asarray(payload["groups_qc"][label], float)
        all_count = int(payload[f"n_direction_all_{name.lower()}"])
        counts, _ = np.histogram(values, bins=edges)
        if counts.size:
            rmax = max(rmax, int(counts.max()))
        ax.bar(
            centers,
            counts,
            width=width,
            facecolor=to_rgba(color, 0.20),
            edgecolor=color,
            linewidth=1.9,
            zorder=2,
            label=(
                f"{name} QC-clean  n={values.size}/{all_count}, "
                f"R={_format_resultant(float(payload[f'resultant_qc_{name.lower()}']))}"
            ),
        )

    theta_b = float(payload["basis"]["theta_b_rad"])
    line_top = rmax * 1.12
    ax.plot(
        [0.0, 0.0], [0.0, line_top], color=TA_COLOR, lw=4.4, ls="-", zorder=6,
        label="TA frozen template direction",
    )
    ax.plot(
        [theta_b, theta_b], [0.0, line_top], color=TB_COLOR, lw=4.4, ls="-", zorder=6,
        label="TB frozen template direction",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(100)
    ax.set_ylim(0.0, rmax * 1.16)
    ax.grid(color="#A9A9A9", alpha=0.75, linewidth=0.9)
    relation = payload["relation"].get("relation", "unknown")
    ax.set_title(
        f"{payload['pretty_subject']}: QC-clean interictal TA/TB event directions\n"
        f"TA frozen early→late axis → 0° · TB={payload['basis']['theta_b_deg']:.1f}° "
        f"({relation})",
        fontsize=15,
        pad=16,
    )
    if not payload["n_direction_qc_ta"] and not payload["n_direction_qc_tb"]:
        ax.text(
            0.5, 0.5, "No QC-clean event direction", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color="#555555", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
        )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1,
        frameon=False, fontsize=11.3,
    )
    contract = payload["qc_contract"]
    fig.text(
        0.5,
        0.035,
        (
            f"Independent event QC: mapped contacts ≥{contract['minimum_mapped_participating_contacts']} · "
            f"shafts ≥{contract['minimum_participating_shafts']} · "
            f"effective rank ≥{contract['minimum_effective_coordinate_rank']} · "
            f"LOCO valid ≥{contract['minimum_loco_valid_fraction']:.1f} · "
            f"median signed cosine ≥{contract['minimum_loco_median_signed_cosine']:.1f}"
        ),
        ha="center", va="bottom", fontsize=9.6, color="#444444",
    )

    stem = f"{subject_id}_interictal_template_direction_rose_qc_clean"
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        key: value for key, value in payload.items()
        if key not in {"groups_all", "groups_qc"}
    }
    metadata["single_event_direction_contract"] = {
        "input": "masked per-event rank; non-participating contacts are NaN",
        "fit": "3D least-squares rank gradient with rcond=0.05; positive is early_to_late",
        "all_event_sensitivity_minimum_contacts": 3,
        "main_histogram": "QC-clean events only",
        "display_plane": payload["basis"]["basis_source"],
        "reference": "frozen TA early-to-late axis at 0 degrees",
        "ictal_input": "none",
    }
    metadata["outputs"] = {"png": png, "pdf": pdf}
    metadata_path = metadata_dir / f"{subject_id}.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2))
    return {"png": png, "pdf": pdf, "metadata": metadata_path}


def _retention(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _index_row(
    payload: Mapping[str, object], outputs: Mapping[str, Path]
) -> Dict[str, object]:
    return {
        "subject_id": payload["subject_id"],
        "status": "ok",
        "drop_reason": "",
        "n_events_total": payload["n_events_total"],
        "n_events_sampled": payload["n_events_sampled"],
        "n_assigned_ta": payload["n_assigned_ta"],
        "n_assigned_tb": payload["n_assigned_tb"],
        "n_unassigned": payload["n_unassigned"],
        "n_direction_all_ta": payload["n_direction_all_ta"],
        "n_direction_all_tb": payload["n_direction_all_tb"],
        "n_direction_qc_ta": payload["n_direction_qc_ta"],
        "n_direction_qc_tb": payload["n_direction_qc_tb"],
        "retention_ta": _retention(
            int(payload["n_direction_qc_ta"]), int(payload["n_direction_all_ta"])
        ),
        "retention_tb": _retention(
            int(payload["n_direction_qc_tb"]), int(payload["n_direction_all_tb"])
        ),
        "resultant_qc_ta": payload["resultant_qc_ta"],
        "resultant_qc_tb": payload["resultant_qc_tb"],
        "tb_angle_deg_in_ta_frame": payload["basis"]["theta_b_deg"],
        "axis_cosine": payload["relation"].get("cosine"),
        "axis_relation": payload["relation"].get("relation"),
        "geometry_2d_supported": payload["geometry_2d_supported"],
        "strict_stability_pass": payload["strict_stability_pass"],
        "png": str(outputs["png"].relative_to(ROOT)),
        "pdf": str(outputs["pdf"].relative_to(ROOT)),
        "metadata": str(outputs["metadata"].relative_to(ROOT)),
    }


INDEX_COLUMNS = [
    "subject_id", "status", "drop_reason", "n_events_total", "n_events_sampled",
    "n_assigned_ta", "n_assigned_tb", "n_unassigned", "n_direction_all_ta",
    "n_direction_all_tb", "n_direction_qc_ta", "n_direction_qc_tb",
    "retention_ta", "retention_tb", "resultant_qc_ta", "resultant_qc_tb",
    "tb_angle_deg_in_ta_frame", "axis_cosine", "axis_relation",
    "geometry_2d_supported", "strict_stability_pass", "png", "pdf", "metadata",
]


def _write_index(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    with (out_dir / "subject_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    (out_dir / "subject_index.json").write_text(
        json.dumps(_jsonable(list(rows)), ensure_ascii=False, indent=2)
    )


def _write_figure_readme(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    lines = [
        "# 间期 TA/TB 单事件方向图（QC-clean）",
        "",
        "本目录是只保留几何稳定事件的主分析版；原 `template_direction_rose/` 保留为 n≥3 全事件 sensitivity。",
        "筛选只判断事件自身能否稳定估计有符号梯度，不使用事件与模板的夹角，也不读取发作、onset 或 ictal energy 数据。",
        "",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        png = Path(str(row["png"])).name
        lines.extend([
            f"### {png}",
            "",
            (
                f"红/蓝直方图只纳入通过独立几何与 LOCO 稳定性 QC 的 TA/TB 间期事件；"
                f"TA 保留 {row['n_direction_qc_ta']}/{row['n_direction_all_ta']}，"
                f"TB 保留 {row['n_direction_qc_tb']}/{row['n_direction_all_tb']}。"
            ),
            (
                f"粗红/蓝实线仍直接读取冻结模板轴，TA 固定为 0°，TB 为 "
                f"{float(row['tb_angle_deg_in_ta_frame']):.1f}°；空直方图表示该类没有可可靠估计方向的事件。"
            ),
            "同名 PDF 是矢量版本，完整逐事件 QC 数值见上一级 `per_subject/` 元数据。",
            "",
            "**关注点**：QC 后方向峰是否仍偏离同色冻结模板轴；若仍偏离，说明仅靠参与 contact 数量和本版基本几何稳定性 gate 不能解释该差异。",
            "",
        ])
    (out_dir / "figures" / "README.md").write_text("\n".join(lines))


def run(
    subjects: Sequence[str],
    *,
    out_dir: Path = DEFAULT_OUT,
    bins: int = 18,
    max_events: int = DEFAULT_MAX_EVENTS,
    seed: int = DEFAULT_SEED,
    min_contacts: int = DEFAULT_MIN_CONTACTS,
    min_shafts: int = DEFAULT_MIN_SHAFTS,
    min_effective_rank: int = DEFAULT_MIN_EFFECTIVE_RANK,
    min_loco_valid_fraction: float = DEFAULT_MIN_LOCO_VALID_FRACTION,
    min_loco_median_signed_cosine: float = DEFAULT_MIN_LOCO_MEDIAN_SIGNED_COSINE,
) -> Sequence[Mapping[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, subject_id in enumerate(subjects, 1):
        try:
            payload = build_subject_payload(
                subject_id,
                max_events=max_events,
                seed=seed,
                min_contacts=min_contacts,
                min_shafts=min_shafts,
                min_effective_rank=min_effective_rank,
                min_loco_valid_fraction=min_loco_valid_fraction,
                min_loco_median_signed_cosine=min_loco_median_signed_cosine,
            )
            outputs = plot_subject(payload, out_dir, bins=bins)
            row = _index_row(payload, outputs)
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"TA={payload['n_direction_qc_ta']}/{payload['n_direction_all_ta']} "
                f"TB={payload['n_direction_qc_tb']}/{payload['n_direction_all_tb']} "
                f"TB_angle={payload['basis']['theta_b_deg']:.1f}°",
                flush=True,
            )
        except Exception as exc:
            row = {column: "" for column in INDEX_COLUMNS}
            row.update(subject_id=subject_id, status="skip", drop_reason=str(exc)[:240])
            print(f"[{index:02d}/{len(subjects)}] {subject_id}: skip ({exc})", flush=True)
        rows.append(row)
    _write_index(rows, out_dir)
    _write_figure_readme(rows, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-contacts", type=int, default=DEFAULT_MIN_CONTACTS)
    parser.add_argument("--min-shafts", type=int, default=DEFAULT_MIN_SHAFTS)
    parser.add_argument("--min-effective-rank", type=int, default=DEFAULT_MIN_EFFECTIVE_RANK)
    parser.add_argument(
        "--min-loco-valid-fraction", type=float,
        default=DEFAULT_MIN_LOCO_VALID_FRACTION,
    )
    parser.add_argument(
        "--min-loco-median-signed-cosine", type=float,
        default=DEFAULT_MIN_LOCO_MEDIAN_SIGNED_COSINE,
    )
    args = parser.parse_args()
    subjects = args.subjects or sorted(
        path.stem for path in (FROZEN_ROOT / "per_subject").glob("*.json")
    )
    run(
        subjects,
        out_dir=args.out_dir,
        bins=args.bins,
        max_events=args.max_events,
        seed=args.seed,
        min_contacts=args.min_contacts,
        min_shafts=args.min_shafts,
        min_effective_rank=args.min_effective_rank,
        min_loco_valid_fraction=args.min_loco_valid_fraction,
        min_loco_median_signed_cosine=args.min_loco_median_signed_cosine,
    )


if __name__ == "__main__":
    main()
