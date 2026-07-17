#!/usr/bin/env python3
"""Render interictal-only TA/TB single-event direction roses.

TA's frozen early-to-late axis is the 0-degree reference.  TA/TB single-event
directions are estimated from masked ranks on real 3D contact coordinates, and
the two thick solid lines are read directly from the frozen per-subject template
axis artifact.  This script never reads seizure, onset, or ictal-energy inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
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

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.lagpat_rank_audit import mask_phantom_ranks  # noqa: E402
from src.propagation_skeleton_geometry import assign_events_to_templates  # noqa: E402
from src.topic5_interictal_direction_rose import (  # noqa: E402
    axis_pair_display_basis,
    fit_event_directions_3d,
    project_directions_to_angles,
    resultant_length,
)
from src.topic5_template_axis_field import (  # noqa: E402
    TEMPLATE_AXIS_DEFINITION,
    TEMPLATE_AXIS_DIRECTION,
)

FROZEN_ROOT = ROOT / "results/interictal_propagation_masked/template_gradient_fields"
DEFAULT_OUT = ROOT / "results/interictal_propagation_masked/template_direction_rose"
YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")

TA_COLOR = "#B2182B"
TB_COLOR = "#2166AC"
DEFAULT_MAX_EVENTS = 6000
DEFAULT_SEED = 20260717


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    if dataset == "epilepsiae":
        return EPILEPSIAE_ROOT / subject / "all_recs"
    raise ValueError(f"unsupported dataset: {dataset}")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_absolute() and value.is_relative_to(ROOT) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def _pretty_subject(subject_id: str) -> str:
    return subject_id.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _subject_seed(subject_id: str, seed: int) -> int:
    return int((zlib.crc32(subject_id.encode("utf-8")) + int(seed)) % (2**32 - 1))


def _load_frozen_record(subject_id: str) -> Dict[str, object]:
    path = FROZEN_ROOT / "per_subject" / f"{subject_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen template-axis artifact: {path}")
    record = json.loads(path.read_text())
    if record.get("axis_definition") != TEMPLATE_AXIS_DEFINITION:
        raise ValueError(f"{subject_id}: unexpected axis definition")
    if record.get("axis_direction_convention") != TEMPLATE_AXIS_DIRECTION:
        raise ValueError(f"{subject_id}: unexpected direction convention")
    pair = record.get("axis_pair") or {}
    if pair.get("status") != "ok":
        raise ValueError(f"axis_pair_not_estimable:{record.get('status')}")
    for template in ("axis_a", "axis_b"):
        axis = pair.get(template) or {}
        if axis.get("status") != "ok" or axis.get("axis_definition") != TEMPLATE_AXIS_DEFINITION:
            raise ValueError(f"{subject_id}: {template} is not a frozen canonical axis")
        if axis.get("direction_convention") != TEMPLATE_AXIS_DIRECTION:
            raise ValueError(f"{subject_id}: {template} direction is not early-to-late")
    return record


def _load_masked_events_and_labels(
    record: Mapping[str, object],
    *,
    max_events: int,
    seed: int,
) -> Dict[str, object]:
    subject_id = str(record["subject_id"])
    dataset, subject = subject_id.split("_", 1)
    rank_source = ROOT / str(record["source"]["rank_displacement"])
    rank_record = json.loads(rank_source.read_text())
    pairs = rank_record.get("pairs") or []
    pair_index = int(record["source"].get("pair_index", 0))
    if pair_index >= len(pairs):
        raise ValueError(f"{subject_id}: frozen pair index missing from rank-displacement source")
    pair = pairs[pair_index]

    event_record = load_subject_propagation_events(_subject_dir(dataset, subject))
    event_names = [str(name) for name in event_record["channel_names"]]
    pair_names = [str(name) for name in pair.get("channel_names") or []]
    if event_names != pair_names:
        raise ValueError(f"{subject_id}: event/rank-displacement channel order mismatch")

    ranks = np.asarray(event_record["ranks"], float)
    bools = np.asarray(event_record["bools"], bool)
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    n_events_total = int(masked.shape[1])
    if max_events > 0 and n_events_total > max_events:
        selection = np.sort(
            np.random.default_rng(_subject_seed(subject_id, seed)).choice(
                n_events_total, max_events, replace=False
            )
        )
        masked = masked[:, selection]
    else:
        selection = np.arange(n_events_total)

    joint = np.asarray(pair.get("joint_valid"), bool)
    template_a = np.asarray(pair.get("rank_a_dense_full"), float)
    template_b = np.asarray(pair.get("rank_b_dense_full"), float)
    if not (joint.shape == template_a.shape == template_b.shape == (len(event_names),)):
        raise ValueError(f"{subject_id}: rank-displacement template shape mismatch")
    template_a = np.where(joint, template_a, np.nan)
    template_b = np.where(joint, template_b, np.nan)
    labels = assign_events_to_templates(masked, template_a, template_b)

    frozen_names = [str(name) for name in record.get("names") or []]
    field_names = [str(name) for name in (record.get("interictal_field") or {}).get("contact_order") or []]
    if frozen_names != field_names:
        raise ValueError(f"{subject_id}: frozen axis/field contact order mismatch")
    name_to_index = {name: index for index, name in enumerate(event_names)}
    if any(name not in name_to_index for name in frozen_names):
        raise ValueError(f"{subject_id}: frozen axis contact absent from event universe")
    aligned = masked[[name_to_index[name] for name in frozen_names], :]
    return {
        "event_ranks": aligned,
        "labels": labels,
        "selection": selection,
        "n_events_total": n_events_total,
        "rank_source": rank_source,
    }


def build_subject_payload(
    subject_id: str,
    *,
    max_events: int = DEFAULT_MAX_EVENTS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, object]:
    record = _load_frozen_record(subject_id)
    events = _load_masked_events_and_labels(record, max_events=max_events, seed=seed)
    pair = record["axis_pair"]
    axis_a, axis_b = pair["axis_a"], pair["axis_b"]
    basis = axis_pair_display_basis(axis_a["u"], axis_b["u"],
                                    fallback_transverse=axis_a.get("w"))
    fitted = fit_event_directions_3d(
        events["event_ranks"], np.asarray(record["coords"], float), min_contacts=3
    )
    projected = project_directions_to_angles(
        fitted["directions"], basis["axis_a"], basis["transverse"]
    )
    labels = np.asarray(events["labels"], int)
    angles = np.asarray(projected["angles"], float)
    groups = {
        label: angles[(labels == label) & np.isfinite(angles)] for label in (0, 1)
    }
    relation = pair.get("relation") or {}
    return {
        "subject_id": subject_id,
        "pretty_subject": _pretty_subject(subject_id),
        "groups": groups,
        "basis": basis,
        "relation": relation,
        "geometry_2d_supported": bool(pair.get("geometry_2d_supported")),
        "strict_stability_pass": bool(pair.get("strict_stability_pass")),
        "n_events_total": events["n_events_total"],
        "n_events_sampled": int(len(events["selection"])),
        "n_assigned_ta": int(np.sum(labels == 0)),
        "n_assigned_tb": int(np.sum(labels == 1)),
        "n_unassigned": int(np.sum(labels < 0)),
        "n_direction_ta": int(groups[0].size),
        "n_direction_tb": int(groups[1].size),
        "resultant_ta": resultant_length(groups[0]),
        "resultant_tb": resultant_length(groups[1]),
        "event_n_valid_contacts": fitted["n_valid_contacts"],
        "event_effective_rank": fitted["effective_rank"],
        "event_projection_norm": projected["projection_norm"],
        "axis_a_u": np.asarray(axis_a["u"], float),
        "axis_b_u": np.asarray(axis_b["u"], float),
        "axis_definition": record["axis_definition"],
        "axis_direction_convention": record["axis_direction_convention"],
        "frozen_artifact": FROZEN_ROOT / "per_subject" / f"{subject_id}.json",
        "rank_source": events["rank_source"],
        "max_events": int(max_events),
        "seed": int(seed),
    }


def plot_subject(payload: Mapping[str, object], out_dir: Path, *, bins: int = 18) -> Dict[str, Path]:
    subject_id = str(payload["subject_id"])
    figures = out_dir / "figures"
    metadata_dir = out_dir / "per_subject"
    figures.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    edges = np.linspace(0.0, 2.0 * np.pi, bins + 1)
    centers = edges[:-1] + 0.5 * np.diff(edges)
    width = float(np.diff(edges)[0] * 0.95)
    fig = plt.figure(figsize=(9.1, 8.9), constrained_layout=True)
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
            label=(f"{name} single events  n={values.size}, "
                   f"R={float(payload[f'resultant_{name.lower()}']):.2f}"),
        )

    theta_b = float(payload["basis"]["theta_b_rad"])
    line_top = rmax * 1.12
    ax.plot([0.0, 0.0], [0.0, line_top], color=TA_COLOR, lw=4.4, ls="-", zorder=6,
            label="TA frozen template direction")
    ax.plot([theta_b, theta_b], [0.0, line_top], color=TB_COLOR, lw=4.4, ls="-", zorder=6,
            label="TB frozen template direction")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlabel_position(100)
    ax.set_ylim(0.0, rmax * 1.16)
    ax.grid(color="#A9A9A9", alpha=0.75, linewidth=0.9)
    relation = payload["relation"].get("relation", "unknown")
    ax.set_title(
        f"{payload['pretty_subject']}: interictal TA/TB single-event directions\n"
        f"TA frozen early→late axis → 0° · TB={payload['basis']['theta_b_deg']:.1f}° "
        f"({relation})",
        fontsize=15,
        pad=16,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=1,
              frameon=False, fontsize=11.5)

    stem = f"{subject_id}_interictal_template_direction_rose"
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        key: value for key, value in payload.items()
        if key not in {"groups", "event_n_valid_contacts", "event_effective_rank", "event_projection_norm"}
    }
    metadata["single_event_direction_contract"] = {
        "input": "masked per-event rank; non-participating contacts are NaN",
        "fit": "3D least-squares rank gradient with rcond=0.05; positive is early_to_late",
        "minimum_mapped_participating_contacts": 3,
        "display_plane": payload["basis"]["basis_source"],
        "reference": "frozen TA early-to-late axis at 0 degrees",
        "ictal_input": "none",
    }
    metadata["event_direction_qc"] = {
        "n_valid_contacts": payload["event_n_valid_contacts"],
        "effective_rank": payload["event_effective_rank"],
        "projection_norm": payload["event_projection_norm"],
    }
    metadata["outputs"] = {"png": png, "pdf": pdf}
    metadata_path = metadata_dir / f"{subject_id}.json"
    metadata_path.write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2))
    return {"png": png, "pdf": pdf, "metadata": metadata_path}


def _index_row(payload: Mapping[str, object], outputs: Mapping[str, Path]) -> Dict[str, object]:
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
        "resultant_ta": payload["resultant_ta"],
        "resultant_tb": payload["resultant_tb"],
        "tb_angle_deg_in_ta_frame": payload["basis"]["theta_b_deg"],
        "axis_cosine": payload["relation"].get("cosine"),
        "axis_relation": payload["relation"].get("relation"),
        "geometry_2d_supported": payload["geometry_2d_supported"],
        "strict_stability_pass": payload["strict_stability_pass"],
        "png": str(outputs["png"].relative_to(ROOT)),
        "pdf": str(outputs["pdf"].relative_to(ROOT)),
        "metadata": str(outputs["metadata"].relative_to(ROOT)),
    }


def _write_index(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    columns = [
        "subject_id", "status", "drop_reason", "n_events_total", "n_events_sampled",
        "n_assigned_ta", "n_assigned_tb", "n_unassigned", "n_direction_ta",
        "n_direction_tb", "resultant_ta", "resultant_tb", "tb_angle_deg_in_ta_frame",
        "axis_cosine", "axis_relation", "geometry_2d_supported", "strict_stability_pass",
        "png", "pdf", "metadata",
    ]
    with (out_dir / "subject_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_jsonable(rows))
    (out_dir / "subject_index.json").write_text(
        json.dumps(_jsonable(list(rows)), ensure_ascii=False, indent=2)
    )


def _write_figure_readme(rows: Sequence[Mapping[str, object]], out_dir: Path) -> None:
    lines = [
        "# 间期 TA/TB 单事件方向图",
        "",
        "本目录只使用冻结的间期模板轴与 masked 间期群体事件，不读取任何发作、onset 或 ictal energy 数据。",
        "每张 PNG 都有同名 PDF 矢量版；完整输入、抽样和方向 QC 见上一级 `per_subject/` 元数据。",
        "",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        png = Path(str(row["png"])).name
        lines.extend([
            f"### {png}",
            "",
            (f"红/蓝直方图分别是 TA/TB 类别内单个间期群体事件的 early-to-late 方向分布；"
             f"本图使用 {row['n_direction_ta']} 个 TA 事件和 {row['n_direction_tb']} 个 TB 事件。"),
            (f"粗红/蓝实线直接读取该患者冻结的 TA/TB 模板轴，TA 固定为 0°，TB 为 "
             f"{float(row['tb_angle_deg_in_ta_frame']):.1f}°；轴关系为 {row['axis_relation']}。"),
            "同名 PDF 是矢量版本，图中的方向不是发作方向，也不是传播速度。",
            "",
            "**关注点**：单事件方向峰是否围绕各自同色的冻结模板方向聚集，以及 A/B 两类是否形成可分的方向结构。",
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
) -> Sequence[Mapping[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, subject_id in enumerate(subjects, 1):
        try:
            payload = build_subject_payload(subject_id, max_events=max_events, seed=seed)
            outputs = plot_subject(payload, out_dir, bins=bins)
            row = _index_row(payload, outputs)
            print(
                f"[{index:02d}/{len(subjects)}] {subject_id}: "
                f"TA={payload['n_direction_ta']} TB={payload['n_direction_tb']} "
                f"TB_angle={payload['basis']['theta_b_deg']:.1f}°",
                flush=True,
            )
        except Exception as exc:
            row = {column: "" for column in (
                "n_events_total", "n_events_sampled", "n_assigned_ta", "n_assigned_tb",
                "n_unassigned", "n_direction_ta", "n_direction_tb", "resultant_ta",
                "resultant_tb", "tb_angle_deg_in_ta_frame", "axis_cosine", "axis_relation",
                "geometry_2d_supported", "strict_stability_pass", "png", "pdf", "metadata",
            )}
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
    args = parser.parse_args()
    subjects = args.subjects or sorted(path.stem for path in (FROZEN_ROOT / "per_subject").glob("*.json"))
    run(subjects, out_dir=args.out_dir, bins=args.bins,
        max_events=args.max_events, seed=args.seed)


if __name__ == "__main__":
    main()
