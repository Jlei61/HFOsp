#!/usr/bin/env python3
"""Add canonical and real-geometry observation layouts to the frozen targets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_cohort_formal_layout import build_subject_layout  # noqa: E402

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
TRACKED_MODULES = (
    "scripts/build_topic4_data_driven_snn_cohort_formal_layouts.py",
    "src/topic4_cohort_formal_layout.py",
    "src/topic4_canonical_shaft_layout.py",
    "src/topic4_data_driven_cohort_formal.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        with open(temporary, "w") as stream:
            json.dump(_json_ready(payload), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        generated = temporary if temporary.endswith(".npz") else temporary + ".npz"
        os.replace(generated, path)
    finally:
        for candidate in (temporary, temporary + ".npz"):
            if os.path.exists(candidate):
                os.unlink(candidate)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _provenance(config_path: Path) -> dict:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *TRACKED_MODULES,
         str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_modules_dirty": bool(dirty),
        "runtime_file_sha256": {
            name: _sha256(ROOT / name) for name in TRACKED_MODULES
        },
        "config_sha256": _sha256(config_path),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _draw(records: list[dict], output: Path) -> None:
    order = sorted(records, key=lambda row: row["within_shaft_null"]["effective_null_size"])
    labels = [row["subject_id"].replace("epilepsiae_", "E").replace("yuquan_", "Y")
              for row in order]
    sizes = np.asarray([row["within_shaft_null"]["effective_null_size"] for row in order])
    exhaustive = np.asarray([row["within_shaft_null"]["exhaustive"] for row in order])

    fig = plt.figure(figsize=(15.0, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))

    ax = fig.add_subplot(grid[0, :])
    colors = np.where(exhaustive, "#b2182b", "#2166ac")
    ax.bar(np.arange(len(sizes)), sizes, color=colors)
    ax.set_ylim(0, 92)
    requested = ax.axhline(64, color="#222222", linewidth=1.0, linestyle="--")
    floor = ax.axhline(19, color="#777777", linewidth=0.9, linestyle=":")
    ax.legend(
        [requested, floor],
        ["64 requested", "19 distinct alternatives = floor for p<=0.05"],
        frameon=False, fontsize=9, loc="upper left", ncol=2,
    )
    for index, (value, is_exhaustive) in enumerate(zip(sizes, exhaustive)):
        if is_exhaustive:
            ax.text(index, value + 1.5, str(value), ha="center", va="bottom", fontsize=8,
                    color="#b2182b")
    ax.set_xticks(np.arange(len(sizes)), labels, rotation=90, fontsize=7)
    ax.set_ylabel("distinct within-shaft permutations")
    ax.set_title(
        "how many matched contact-identity alternatives each montage actually has "
        "(red = whole group enumerated, blue = 64 distinct draws)",
        loc="left", fontweight="bold", fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    paired = [row for row in records if row["real_geometry_layout"] is not None]
    canonical_minor = np.asarray([
        min(row["canonical_layout"]["x_span_mm"], row["canonical_layout"]["y_span_mm"])
        for row in paired
    ])
    real_minor = np.asarray([
        min(row["real_geometry_layout"]["x_span_mm"],
            row["real_geometry_layout"]["y_span_mm"])
        for row in paired
    ])
    order_minor = np.argsort(real_minor)
    positions = np.arange(len(paired))
    ax.vlines(positions, real_minor[order_minor], canonical_minor[order_minor],
              color="#cccccc", linewidth=1.0)
    ax.scatter(positions, real_minor[order_minor], s=26, color="#9e9e9e",
               label="real geometry")
    ax.scatter(positions, canonical_minor[order_minor], s=26, color="#2166ac",
               label="canonical")
    ax.set_ylim(-0.6, 17.5)
    ax.set_xticks([])
    ax.set_xlabel(f"subject, sorted by real minor span (n={len(paired)})")
    ax.set_ylabel("minor-axis span (mm)")
    ax.set_title("both layouts fill 16 mm on their principal axis;\n"
                 "they differ only on the second axis",
                 loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    examples = sorted(records, key=lambda row: row["n_contacts"])
    for column, row in enumerate((examples[0], examples[-1]), start=1):
        ax = fig.add_subplot(grid[1, column])
        arrays = row["_arrays"]
        canonical = arrays["canonical_coords_sheet"]
        shafts = arrays["canonical_shaft_ids"]
        if "real_coords_sheet" in arrays:
            real = arrays["real_coords_sheet"]
            ax.scatter(real[:, 0], real[:, 1], s=13, color="#c8c8c8", marker="x",
                       linewidth=0.8, zorder=1)
            ax.text(0.5, 19.3, "grey x = real geometry", fontsize=7, color="#8a8a8a",
                    va="top")
        for shaft in sorted(set(shafts.tolist())):
            selected = shafts == shaft
            ax.plot(canonical[selected, 0], canonical[selected, 1], "-o",
                    markersize=3.5, linewidth=1.0, color="#2166ac", zorder=2)
            ax.text(canonical[selected, 0].max() + 0.35,
                    canonical[selected, 1][0], shaft, fontsize=6.5,
                    color="#2166ac", va="center")
        ax.set_xlim(0, 21.5)
        ax.set_ylim(0, 20)
        ax.set_aspect("equal")
        ax.set_title(
            f"{row['subject_id']} ({row['n_contacts']} contacts, "
            f"{len(set(shafts.tolist()))} shafts)",
            loc="left", fontweight="bold", fontsize=10,
        )
        ax.set_xlabel("sheet x (mm)")
        if column == 1:
            ax.set_ylabel("sheet y (mm)")

    fig.suptitle(
        "Topic 4 formal cohort observation layouts | target-blind readout geometry "
        "and matched nulls; no SNN result",
        fontsize=13, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def build(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "topic4_data_driven_snn_cohort_formal_v1":
        raise RuntimeError("formal layout builder received another config schema")
    for name, record in config["inputs"].items():
        if "sha256" not in record:
            continue
        actual = _sha256(ROOT / record["path"])
        if actual != record["sha256"]:
            raise RuntimeError(f"formal input hash changed for {name}: {record['path']}")

    source_root = ROOT / config["inputs"]["source_target_root"]["path"]
    output_root = ROOT / config["output_root"]
    per_subject = output_root / "per_subject"
    audit = json.loads(
        (ROOT / config["inputs"]["cohort_target_audit"]["path"]).read_text()
    )
    subject_ids = [row["subject_id"] for row in audit["subjects"]]
    if len(subject_ids) != int(config["observation"]["primary_denominator"]):
        raise RuntimeError("frozen target audit does not hold the primary denominator")

    observation = config["observation"]
    null_config = config["null"]
    records, rows = [], []
    for subject_id in subject_ids:
        source_json = source_root / f"{subject_id}.json"
        source_npz = source_root / f"{subject_id}_target.npz"
        source = json.loads(source_json.read_text())
        if not source.get("patient_target_eligible"):
            raise RuntimeError(f"formal cohort subject lost target eligibility: {subject_id}")
        with np.load(source_npz, allow_pickle=False) as loaded:
            contact_order = [str(value) for value in loaded["contact_order"]]
            real = (
                np.asarray(loaded["geometry_coords_sheet"], float)
                if "geometry_coords_sheet" in loaded else None
            )
        built = build_subject_layout(
            subject_id, contact_order, real_coords_sheet=real,
            n_permutations=int(null_config["n_permutations"]),
            base_seed=int(null_config["base_seed"]),
            sheet_size_mm=float(observation["sheet_size_mm"]),
            margin_mm=float(observation["sheet_margin_mm"]),
        )
        layout_npz = per_subject / f"{subject_id}_layout.npz"
        _atomic_npz(layout_npz, built["arrays"])
        record = {
            "schema_version": "topic4_data_driven_snn_cohort_formal_layout_v1",
            **built["record"],
            "in_primary_canonical_cohort": True,
            "in_real_geometry_sensitivity_cohort": real is not None,
            "sources": {
                "target_json": str(source_json.relative_to(ROOT)),
                "target_json_sha256": _sha256(source_json),
                "target_npz": str(source_npz.relative_to(ROOT)),
                "target_npz_sha256": _sha256(source_npz),
            },
            "layout_npz": str(layout_npz.relative_to(ROOT)),
            "layout_npz_sha256": _sha256(layout_npz),
        }
        _atomic_json(per_subject / f"{subject_id}.json", record)
        records.append({**record, "_arrays": built["arrays"]})
        null = record["within_shaft_null"]
        rows.append({
            "subject_id": subject_id,
            "n_contacts": record["n_contacts"],
            "n_shafts": record["canonical_layout"]["n_shafts"],
            "canonical_x_span_mm": f"{record['canonical_layout']['x_span_mm']:.4f}",
            "canonical_y_span_mm": f"{record['canonical_layout']['y_span_mm']:.4f}",
            "real_geometry": str(real is not None),
            "real_x_span_mm": (
                "" if real is None
                else f"{record['real_geometry_layout']['x_span_mm']:.4f}"
            ),
            "real_y_span_mm": (
                "" if real is None
                else f"{record['real_geometry_layout']['y_span_mm']:.4f}"
            ),
            "null_effective_size": null["effective_null_size"],
            "null_exhaustive": str(null["exhaustive"]),
            "null_minimum_reachable_p": f"{null['minimum_reachable_p']:.6f}",
            "null_seed": null["seed"],
        })

    _atomic_csv(output_root / "cohort_layout.csv", rows)
    figures = output_root / "figures"
    _draw(records, figures / "formal_cohort_observation_layouts")

    coarse = [
        row["subject_id"] for row in records
        if row["within_shaft_null"]["minimum_reachable_p"] > 0.05
    ]
    n_real = sum(row["in_real_geometry_sensitivity_cohort"] for row in records)
    summary = {
        "schema_version": "topic4_data_driven_snn_cohort_formal_layout_audit_v1",
        "status": "FORMAL_LAYOUTS_FROZEN_SNN_NOT_RUN",
        "denominators": {
            "primary_canonical_layout": len(records),
            "real_geometry_sensitivity": n_real,
        },
        "within_shaft_null": {
            "requested": int(null_config["n_permutations"]),
            "subjects_with_exhaustive_null": sum(
                row["within_shaft_null"]["exhaustive"] for row in records
            ),
            "subjects_that_cannot_reach_p_0_05_alone": coarse,
        },
        "scientific_boundary": (
            "This artifact freezes target-blind readout geometry and matched nulls "
            "only. It contains no SNN result. The canonical layout is a contact-order "
            "readout, not patient anatomy."
        ),
        "subjects": [
            {key: value for key, value in row.items() if key != "_arrays"}
            for row in records
        ],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": _provenance(config_path),
    }
    _atomic_json(output_root / "cohort_layout_audit.json", summary)

    readme = f"""# 图说明

### formal_cohort_observation_layouts.png

这张图只审计正式 cohort 的"读出几何"与"配对零假设"，不含任何 SNN 仿真结果。上排是每位患者真正拥有多少个**互不相同的**同杆触点重排方案：红色的 {summary['within_shaft_null']['subjects_with_exhaustive_null']} 位患者的方案总数少于请求的 64 个，于是整组被穷举（这是精确零假设，但很粗）；其中 {len(coarse)} 位患者无论模型多好，单看这个零假设也够不到 p<=0.05。下排左是 {n_real} 位有真实三维坐标的患者，比较"名字排布"与"真实几何"两种读出覆盖的片上面积；下排右两格画出实际触点摆位（线是同一根杆，灰色叉是真实几何）。

**关注点**：canonical 摆位只保留"杆身份 + 杆上序号"，不读任何 rank / 模式标签 / 事件计数，因此它支撑的是"触点顺序刻板结构"，**不是解剖定位**；分母必须写成 {len(records)} 人 canonical + {n_real} 人真实几何敏感性，红色患者的零假设分辨率必须随结果一起报出。
"""
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "README.md").write_text(readme)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build(args.config.resolve())
    print(json.dumps({
        "status": summary["status"],
        "denominators": summary["denominators"],
        "within_shaft_null": summary["within_shaft_null"],
    }, indent=2))


if __name__ == "__main__":
    main()
