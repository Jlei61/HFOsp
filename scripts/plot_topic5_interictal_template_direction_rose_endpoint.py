#!/usr/bin/env python3
"""Render interictal TA/TB roses with the legacy endpoint-centroid method.

Both the thick template axes and every single-event direction are constructed
from early/source and late/sink endpoint centroids.  No least-squares gradient
direction and no ictal input enter the estimator.
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
    _load_masked_events_and_labels,
    _pretty_subject,
)
from src.topic5_interictal_direction_rose import (  # noqa: E402
    axis_pair_display_basis,
    fit_endpoint_direction_3d,
    fit_event_endpoint_directions_3d,
    project_directions_to_angles,
    resultant_length,
)

DEFAULT_OUT = ROOT / "results/interictal_propagation_masked/template_direction_rose_endpoint"
DEFAULT_K_PRIMARY = 3


def _load_endpoint_input_record(subject_id: str) -> Dict[str, object]:
    """Load frozen masked ranks/coordinates without consuming gradient axes."""
    path = FROZEN_ROOT / "per_subject" / f"{subject_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen interictal template input: {path}")
    record = json.loads(path.read_text())
    if int(record.get("stable_k", -1)) != 2:
        raise ValueError(f"stable_k_not_2:{record.get('stable_k')}")
    required = ("names", "coords", "shafts", "rank_a", "rank_b", "source")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"endpoint_inputs_unavailable:{record.get('status')}:{','.join(missing)}")
    n = len(record["names"])
    if np.asarray(record["coords"], float).shape != (n, 3):
        raise ValueError(f"{subject_id}: endpoint coordinate shape mismatch")
    for key in ("shafts", "rank_a", "rank_b"):
        if len(record[key]) != n:
            raise ValueError(f"{subject_id}: endpoint {key} shape mismatch")
    return record


def _geometry_transverse(coords: np.ndarray, direction: np.ndarray):
    """Contact-geometry fallback only for displaying collinear endpoint axes."""
    xyz = np.asarray(coords, float)
    u = np.asarray(direction, float)
    good = np.isfinite(xyz).all(axis=1)
    if int(good.sum()) < 2:
        return None
    rel = xyz[good] - xyz[good].mean(axis=0)
    residual = rel - np.outer(rel @ u, u)
    _, singular, vh = np.linalg.svd(residual, full_matrices=False)
    if not singular.size or float(singular[0]) < 1e-9:
        return None
    return vh[0]


def build_subject_payload(
    subject_id: str,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    seed: int = DEFAULT_SEED,
    k_primary: int = DEFAULT_K_PRIMARY,
) -> Dict[str, object]:
    record = _load_endpoint_input_record(subject_id)
    events = _load_masked_events_and_labels(record, max_events=max_events, seed=seed)
    coords = np.asarray(record["coords"], float)
    template_a = fit_endpoint_direction_3d(record["rank_a"], coords, k_primary=k_primary)
    template_b = fit_endpoint_direction_3d(record["rank_b"], coords, k_primary=k_primary)
    for name, fitted in (("TA", template_a), ("TB", template_b)):
        if not np.isfinite(np.asarray(fitted["direction"], float)).all():
            raise ValueError(f"{subject_id}: {name} endpoint axis not estimable ({fitted['tier']})")

    fallback = _geometry_transverse(coords, np.asarray(template_a["direction"], float))
    basis = axis_pair_display_basis(
        template_a["direction"], template_b["direction"], fallback_transverse=fallback
    )
    fitted = fit_event_endpoint_directions_3d(
        events["event_ranks"], coords, k_primary=k_primary
    )
    projected = project_directions_to_angles(
        fitted["directions"], basis["axis_a"], basis["transverse"]
    )
    labels = np.asarray(events["labels"], int)
    angles = np.asarray(projected["angles"], float)
    angle_valid = np.isfinite(angles)
    groups = {label: angles[(labels == label) & angle_valid] for label in (0, 1)}
    k_used = np.asarray(fitted["k_used"], int)

    def _count(label: int, k: int) -> int:
        return int(np.sum((labels == label) & angle_valid & (k_used == k)))

    return {
        "subject_id": subject_id,
        "pretty_subject": _pretty_subject(subject_id),
        "groups": groups,
        "basis": basis,
        "template_a": template_a,
        "template_b": template_b,
        "n_events_total": events["n_events_total"],
        "n_events_sampled": int(len(events["selection"])),
        "n_assigned_ta": int(np.sum(labels == 0)),
        "n_assigned_tb": int(np.sum(labels == 1)),
        "n_unassigned": int(np.sum(labels < 0)),
        "n_direction_ta": int(groups[0].size),
        "n_direction_tb": int(groups[1].size),
        "n_primary_k3_ta": _count(0, 3),
        "n_primary_k3_tb": _count(1, 3),
        "n_fallback_k2_ta": _count(0, 2),
        "n_fallback_k2_tb": _count(1, 2),
        "resultant_ta": resultant_length(groups[0]),
        "resultant_tb": resultant_length(groups[1]),
        "event_selection_indices": events["selection"],
        "event_n_valid_contacts": fitted["n_valid_contacts"],
        "event_k_used": fitted["k_used"],
        "event_tier": fitted["tier"],
        "event_axis_length": fitted["axis_length"],
        "event_projection_norm": projected["projection_norm"],
        "template_input_artifact": FROZEN_ROOT / "per_subject" / f"{subject_id}.json",
        "rank_source": events["rank_source"],
        "max_events": int(max_events),
        "seed": int(seed),
        "k_primary": int(k_primary),
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
        values = np.asarray(payload["groups"][label], float)
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
                f"{name} endpoint events  n={values.size} "
                f"(k3={payload[f'n_primary_k3_{name.lower()}']}, "
                f"k2={payload[f'n_fallback_k2_{name.lower()}']}), "
                f"R={_format_resultant(float(payload[f'resultant_{name.lower()}']))}"
            ),
        )

    theta_b = float(payload["basis"]["theta_b_rad"])
    line_top = rmax * 1.12
    k_a = int(payload["template_a"]["k_used"])
    k_b = int(payload["template_b"]["k_used"])
    ax.plot(
        [0.0, 0.0], [0.0, line_top], color=TA_COLOR, lw=4.4, ls="-", zorder=6,
        label=f"TA endpoint template direction (k={k_a})",
    )
    ax.plot(
        [theta_b, theta_b], [0.0, line_top], color=TB_COLOR, lw=4.4, ls="-", zorder=6,
        label=f"TB endpoint template direction (k={k_b})",
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(100)
    ax.set_ylim(0.0, rmax * 1.16)
    ax.grid(color="#A9A9A9", alpha=0.75, linewidth=0.9)
    ax.set_title(
        f"{payload['pretty_subject']}: endpoint-centroid interictal TA/TB directions\n"
        f"TA endpoint early→late axis → 0° · TB={payload['basis']['theta_b_deg']:.1f}°",
        fontsize=15,
        pad=16,
    )
    if not payload["n_direction_ta"] and not payload["n_direction_tb"]:
        ax.text(
            0.5, 0.5, "No endpoint-estimable event direction", transform=ax.transAxes,
            ha="center", va="center", fontsize=13, color="#555555", zorder=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
        )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1,
        frameon=False, fontsize=10.8,
    )
    fig.text(
        0.5,
        0.035,
        "Legacy endpoint contract: n≥7 → top/bottom-3 · n=5–6 → top/bottom-2 fallback · source centroid → sink centroid",
        ha="center", va="bottom", fontsize=9.6, color="#444444",
    )

    stem = f"{subject_id}_interictal_template_direction_rose_endpoint"
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {key: value for key, value in payload.items() if key != "groups"}
    metadata["endpoint_direction_contract"] = {
        "rank_input": "masked per-event/template rank; non-participating contacts are NaN",
        "template_estimator": "build_endpoint_cores + source-to-sink centroid vector",
        "event_estimator": "same endpoint-centroid estimator as template",
        "primary": "n_eff>=7 uses k=3 earliest and k=3 latest contacts",
        "fallback": "n_eff in {5,6} uses k=2 earliest and k=2 latest contacts",
        "below_floor": "n_eff<5 has no direction",
        "gradient_axis_input": "none",
        "ictal_input": "none",
        "reference": "TA endpoint direction at 0 degrees",
    }
    metadata["outputs"] = {"png": png, "pdf": pdf}
    metadata_path = metadata_dir / f"{subject_id}.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2))
    return {"png": png, "pdf": pdf, "metadata": metadata_path}


INDEX_COLUMNS = [
    "subject_id", "status", "drop_reason", "n_events_total", "n_events_sampled",
    "n_assigned_ta", "n_assigned_tb", "n_unassigned", "n_direction_ta",
    "n_direction_tb", "n_primary_k3_ta", "n_primary_k3_tb", "n_fallback_k2_ta",
    "n_fallback_k2_tb", "resultant_ta", "resultant_tb",
    "template_k_ta", "template_k_tb", "tb_angle_deg_in_ta_frame",
    "axis_cosine", "png", "pdf", "metadata",
]


def _index_row(payload: Mapping[str, object], outputs: Mapping[str, Path]):
    return {
        "subject_id": payload["subject_id"],
        "status": "ok",
        "drop_reason": "",
        "n_events_total": payload["n_events_total"],
        "n_events_sampled": payload["n_events_sampled"],
        "n_assigned_ta": payload["n_assigned_ta"],
        "n_assigned_tb": payload["n_assigned_tb"],
        "n_unassigned": payload["n_unassigned"],
        "n_direction_ta": payload["n_direction_ta"],
        "n_direction_tb": payload["n_direction_tb"],
        "n_primary_k3_ta": payload["n_primary_k3_ta"],
        "n_primary_k3_tb": payload["n_primary_k3_tb"],
        "n_fallback_k2_ta": payload["n_fallback_k2_ta"],
        "n_fallback_k2_tb": payload["n_fallback_k2_tb"],
        "resultant_ta": payload["resultant_ta"],
        "resultant_tb": payload["resultant_tb"],
        "template_k_ta": payload["template_a"]["k_used"],
        "template_k_tb": payload["template_b"]["k_used"],
        "tb_angle_deg_in_ta_frame": payload["basis"]["theta_b_deg"],
        "axis_cosine": payload["basis"]["cosine"],
        "png": str(outputs["png"].relative_to(ROOT)),
        "pdf": str(outputs["pdf"].relative_to(ROOT)),
        "metadata": str(outputs["metadata"].relative_to(ROOT)),
    }


def _write_index(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    with (out_dir / "subject_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    (out_dir / "subject_index.json").write_text(
        json.dumps(_jsonable(list(rows)), ensure_ascii=False, indent=2)
    )


def _write_readme(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    lines = [
        "# 间期 TA/TB endpoint-centroid 单事件方向图",
        "",
        "本目录把模板粗线和单事件方向都改回旧 endpoint-centroid 定义，不读取 gradient 方向或任何发作数据。",
        "n≥7 使用最早/最晚各 3 个 contacts；n=5–6 是旧合同的 k=2 fallback，图例分别报告两档数量。",
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
                f"红/蓝直方图是 TA/TB 单事件的 endpoint-centroid early-to-late 方向；"
                f"TA 使用 {row['n_direction_ta']} 个事件，TB 使用 {row['n_direction_tb']} 个事件。"
            ),
            (
                f"粗红/蓝实线是同一 endpoint 方法从聚合模板 rank 得到的方向，TA 固定为 0°，"
                f"TB 为 {float(row['tb_angle_deg_in_ta_frame']):.1f}°。"
            ),
            "同名 PDF 是矢量版；逐事件 n、k、endpoint axis length 见上一级 `per_subject/` 元数据。",
            "",
            "**关注点**：单事件 endpoint 方向是否比 gradient 版本更集中于各自模板 endpoint 粗线，同时留意 k=2 fallback 是否占主导。",
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
    k_primary: int = DEFAULT_K_PRIMARY,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, subject_id in enumerate(subjects, 1):
        try:
            payload = build_subject_payload(
                subject_id, max_events=max_events, seed=seed, k_primary=k_primary
            )
            outputs = plot_subject(payload, out_dir, bins=bins)
            row = _index_row(payload, outputs)
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"TA={payload['n_direction_ta']} "
                f"(k3={payload['n_primary_k3_ta']},k2={payload['n_fallback_k2_ta']}) "
                f"TB={payload['n_direction_tb']} "
                f"(k3={payload['n_primary_k3_tb']},k2={payload['n_fallback_k2_tb']}) "
                f"TB_angle={payload['basis']['theta_b_deg']:.1f}°",
                flush=True,
            )
        except Exception as exc:
            row = {column: "" for column in INDEX_COLUMNS}
            row.update(subject_id=subject_id, status="skip", drop_reason=str(exc)[:240])
            print(f"[{index:02d}/{len(subjects)}] {subject_id}: skip ({exc})", flush=True)
        rows.append(row)
    _write_index(rows, out_dir)
    _write_readme(rows, out_dir)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=18)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--k-primary", type=int, default=DEFAULT_K_PRIMARY)
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
        k_primary=args.k_primary,
    )


if __name__ == "__main__":
    main()
