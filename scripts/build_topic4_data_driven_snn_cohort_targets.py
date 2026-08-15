#!/usr/bin/env python3
"""Build the 34-subject target/geometry audit for the Topic 4 cohort SNN."""
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

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_data_driven_cohort import (  # noqa: E402
    TargetConfig,
    build_crossfit_patient_target,
    canonical_pair_contract,
    geometry_only_sheet_projection,
    subject_raw_root,
    subset_pair_contract,
)


DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_v1.json"


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
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


def _raw_lagpat_files(raw_root: Path) -> list[Path]:
    files = sorted(raw_root.glob("*_lagPat_withFreqCent.npz"))
    return files or sorted(raw_root.glob("*_lagPat.npz"))


def _raw_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target_config(config: dict) -> TargetConfig:
    source = config["patient_target"]
    return TargetConfig(
        minimum_participating_contacts=int(source["minimum_participating_contacts"]),
        heldout_block_fraction=float(source["heldout_block_fraction"]),
        split_seed=int(source["split_seed"]),
        kmeans_fit_max_events=int(source["kmeans_fit_max_events"]),
        kmeans_n_init=int(source["kmeans_n_init"]),
        kmeans_seed=int(source["kmeans_seed"]),
        stability_seeds=tuple(int(value) for value in source["stability_seeds"]),
        stored_events_per_mode_per_split=int(
            source["stored_events_per_mode_per_split"]
        ),
    )


def _descriptor_arrays(prefix: str, descriptor: dict, arrays: dict) -> None:
    arrays[f"{prefix}_pair_indices"] = np.asarray(
        descriptor["pair_indices"], dtype=np.int16,
    )
    for mode in ("TA", "TB"):
        row = descriptor[mode]
        arrays[f"{prefix}_{mode.lower()}_profile"] = np.asarray(row["profile"], float)
        arrays[f"{prefix}_{mode.lower()}_recruitment"] = np.asarray(
            row["recruitment"], float,
        )
        arrays[f"{prefix}_{mode.lower()}_precedence"] = np.asarray(
            row["precedence"], float,
        )


def _target_arrays(target: dict, pair: dict, geometry: dict | None) -> dict:
    arrays = {
        "contact_order": np.asarray(target["contact_order"], dtype="U64"),
        "canonical_rank_a": np.asarray(pair["rank_a"], float),
        "canonical_rank_b": np.asarray(pair["rank_b"], float),
        "kmeans_centers": np.asarray(target["kmeans_centers"], float),
        "train_profiles": np.asarray(target["train_profiles"], float),
        "heldout_profiles": np.asarray(target["heldout_profiles"], float),
        "train_to_heldout_matrix": np.asarray(target["train_to_heldout_matrix"], float),
        "train_mode_counts": np.asarray(target["train_mode_counts"], int),
        "heldout_mode_counts": np.asarray(target["heldout_mode_counts"], int),
        "train_block_ids": np.asarray(target["train_block_ids"], int),
        "heldout_block_ids": np.asarray(target["heldout_block_ids"], int),
    }
    for split in ("train", "heldout"):
        for mode in ("TA", "TB"):
            arrays[f"{split}_{mode.lower()}_rank_samples"] = np.asarray(
                target[f"{split}_samples"][mode], float,
            )
        _descriptor_arrays(
            split, target[f"{split}_descriptors"], arrays,
        )
    if geometry is not None:
        for key in ("coords_sheet", "coords_projected", "basis", "center_3d",
                    "singular_values", "offset"):
            arrays[f"geometry_{key}"] = np.asarray(geometry[key])
        arrays["geometry_scale"] = np.asarray(float(geometry["scale"]))
    return arrays


def _compact_target(target: dict) -> dict:
    return {
        "contact_order": target["contact_order"],
        "n_contacts": len(target["contact_order"]),
        "n_train_events": int(len(target["train_event_indices"])),
        "n_heldout_events": int(len(target["heldout_event_indices"])),
        "n_train_blocks": int(len(target["train_block_ids"])),
        "n_heldout_blocks": int(len(target["heldout_block_ids"])),
        "train_mode_counts": target["train_mode_counts"],
        "heldout_mode_counts": target["heldout_mode_counts"],
        "cluster_to_semantic_mode": target["cluster_to_semantic_mode"],
        "cluster_to_frozen_template_matrix": target[
            "cluster_to_frozen_template_matrix"
        ],
        "train_to_heldout_matrix": target["train_to_heldout_matrix"],
        "train_to_heldout_diagonal": target["train_to_heldout_diagonal"],
        "train_to_heldout_crossed": target["train_to_heldout_crossed"],
        "train_to_heldout_margin": target["train_to_heldout_margin"],
        "train_distance_q95": target["train_distance_q95"],
        "heldout_ood_fraction": target["heldout_ood_fraction"],
        "kmeans_stability": target["kmeans_stability"],
    }


def _eligibility(target: dict, geometry: dict | None, config: dict) -> tuple[bool, bool, list[str]]:
    contract = config["patient_target"]
    reasons = []
    if len(target["contact_order"]) < int(contract["minimum_contacts"]):
        reasons.append("fewer_than_minimum_contacts")
    if len(target["train_block_ids"]) < int(contract["minimum_train_blocks"]):
        reasons.append("insufficient_train_blocks")
    if len(target["heldout_block_ids"]) < int(contract["minimum_heldout_blocks"]):
        reasons.append("insufficient_heldout_blocks")
    minimum_events = int(contract["minimum_events_per_mode_per_split"])
    if int(np.min(target["train_mode_counts"])) < minimum_events:
        reasons.append("insufficient_train_mode_events")
    if int(np.min(target["heldout_mode_counts"])) < minimum_events:
        reasons.append("insufficient_heldout_mode_events")
    if target["train_to_heldout_margin"] <= float(
        contract["heldout_patient_margin_min"]
    ):
        reasons.append("nonpositive_patient_heldout_margin")
    target_eligible = not reasons
    geometry_eligible = geometry is not None
    return target_eligible, target_eligible and geometry_eligible, reasons


def _draw_diagnostic(rows: list[dict], output: Path) -> None:
    ordered = sorted(rows, key=lambda row: float(row["patient_heldout_margin"]))
    x = np.arange(len(ordered))
    geometry = np.asarray([row["geometry_eligible"] == "True" for row in ordered])
    diag = np.asarray([float(row["patient_heldout_diagonal"]) for row in ordered])
    crossed = np.asarray([float(row["patient_heldout_crossed"]) for row in ordered])
    margins = diag - crossed
    colors = np.where(geometry, "#2166ac", "#9e9e9e")

    fig = plt.figure(figsize=(14.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(0.85, 1.55, 1.15))
    ax = fig.add_subplot(grid[:, 0])
    counts = [len(rows), sum(row["patient_target_eligible"] == "True" for row in rows),
              int(geometry.sum()), sum(row["snn_eligible"] == "True" for row in rows)]
    labels = ["masked\nK=2", "target\nstable", "2-D real\ngeometry", "SNN\neligible"]
    ax.bar(np.arange(4), counts, color=["#4d4d4d", "#4d4d4d", "#2166ac", "#2166ac"])
    for index, value in enumerate(counts):
        ax.text(index, value + 0.5, str(value), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(0, max(counts) + 5)
    ax.set_ylabel("subjects")
    ax.set_title("eligibility funnel", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 1:])
    ax.scatter(x, diag, c=colors, s=32, label="same mode")
    ax.scatter(x, crossed, facecolors="none", edgecolors=colors, s=32, label="crossed mode")
    ax.plot(x, diag, color="#2166ac", alpha=0.25, linewidth=0.8)
    ax.plot(x, crossed, color="#b2182b", alpha=0.25, linewidth=0.8)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_ylabel("train-heldout Spearman")
    ax.set_xticks([])
    ax.set_title("patient target cross-block stability", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1])
    ax.bar(x, margins, color=colors, width=0.8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("same - crossed margin")
    ax.set_xticks([])
    ax.set_title("held-out patient mode margin", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2])
    contacts = np.asarray([int(row["n_contacts"]) for row in rows])
    events = np.asarray([int(row["n_heldout_events"]) for row in rows])
    ax.scatter(contacts, events, c=["#2166ac" if row["geometry_eligible"] == "True" else "#9e9e9e" for row in rows], s=38)
    ax.set_yscale("log")
    ax.set_xlabel("joint-valid contacts")
    ax.set_ylabel("held-out events")
    ax.set_title("target support", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Topic 4 data-driven SNN cohort preflight | patient targets only; no SNN result",
        fontsize=13, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _provenance(config_path: Path, input_paths: list[Path]) -> dict:
    tracked = [
        Path(__file__).resolve(),
        ROOT / "src/topic4_data_driven_cohort.py",
        config_path.resolve(),
    ]
    relative = [str(path.relative_to(ROOT)) for path in tracked]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_modules_dirty": bool(dirty),
        "runtime_file_sha256": {name: _sha256(ROOT / name) for name in relative},
        "input_file_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in input_paths
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def build(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"] / "target_audit"
    per_subject = output_root / "per_subject"
    inputs = config["inputs"]
    cohort_path = ROOT / inputs["stable_k2_cohort_csv"]
    cohort_rows = list(csv.DictReader(cohort_path.open()))
    if len(cohort_rows) != 34:
        raise RuntimeError(f"expected 34 masked stable-K=2 subjects, found {len(cohort_rows)}")
    target_config = _target_config(config)
    rows = []
    subject_records = []

    for cohort_row in cohort_rows:
        subject_id = cohort_row["sid"]
        rank_path = ROOT / inputs["rank_displacement_root"] / f"{subject_id}.json"
        geometry_path = ROOT / inputs["gradient_geometry_root"] / f"{subject_id}.json"
        rank_record = json.loads(rank_path.read_text())
        geometry_record = json.loads(geometry_path.read_text())
        pair = canonical_pair_contract(rank_record)
        geometry = None
        geometry_reason = geometry_record.get("status", "missing")
        target_pair = pair
        coords_3d = None
        if geometry_record.get("status") == "ok":
            field = geometry_record.get("interictal_field") or {}
            contact_order = [str(value) for value in field.get("contact_order", [])]
            coords_3d = np.asarray(field.get("coords", []), dtype=float)
            try:
                target_pair = subset_pair_contract(pair, contact_order)
                geometry = geometry_only_sheet_projection(
                    coords_3d,
                    sheet_size_mm=float(config["geometry"]["sheet_size_mm"]),
                    margin_mm=float(config["geometry"]["sheet_margin_mm"]),
                )
                geometry_reason = "ok"
            except (ValueError, RuntimeError) as exc:
                geometry_reason = f"geometry_projection_failed:{exc}"
                geometry = None
                target_pair = pair

        raw_root = subject_raw_root(
            subject_id,
            epilepsiae_root=inputs["epilepsiae_raw_root"],
            yuquan_root=inputs["yuquan_raw_root"],
        )
        raw_files = _raw_lagpat_files(raw_root)
        if not raw_files:
            raise FileNotFoundError(f"no lagPat inputs for {subject_id}: {raw_root}")
        data = load_subject_propagation_events(raw_root)
        target = build_crossfit_patient_target(data, target_pair, config=target_config)
        target_eligible, snn_eligible, reasons = _eligibility(target, geometry, config)
        dataset, subject = subject_id.split("_", 1)
        npz_path = per_subject / f"{subject_id}_target.npz"
        _atomic_npz(npz_path, _target_arrays(target, target_pair, geometry))
        record = {
            "schema_version": "topic4_data_driven_snn_cohort_subject_target_v1",
            "subject_id": subject_id,
            "dataset": dataset,
            "subject": subject,
            "patient_target_eligible": target_eligible,
            "geometry_eligible": geometry is not None,
            "snn_eligible": snn_eligible,
            "exclusion_reasons": reasons,
            "geometry_status": geometry_reason,
            "geometry_contract": None if geometry is None else {
                "projection": config["geometry"]["projection"],
                "matrix_rank": geometry["matrix_rank"],
                "scale": geometry["scale"],
                "forbidden_projection_inputs": config["geometry"][
                    "forbidden_projection_inputs"
                ],
            },
            "target": _compact_target(target),
            "sources": {
                "raw_root": str(raw_root),
                "raw_lagpat_file_count": len(raw_files),
                "raw_lagpat_sha256": _raw_fingerprint(raw_files),
                "rank_displacement": str(rank_path.relative_to(ROOT)),
                "rank_displacement_sha256": _sha256(rank_path),
                "geometry": str(geometry_path.relative_to(ROOT)),
                "geometry_sha256": _sha256(geometry_path),
            },
            "target_npz": str(npz_path.relative_to(ROOT)),
        }
        json_path = per_subject / f"{subject_id}.json"
        _atomic_json(json_path, record)
        subject_records.append(record)
        row = {
            "subject_id": subject_id,
            "dataset": dataset,
            "patient_target_eligible": str(target_eligible),
            "geometry_eligible": str(geometry is not None),
            "snn_eligible": str(snn_eligible),
            "geometry_status": geometry_reason,
            "exclusion_reasons": ";".join(reasons),
            "n_contacts": len(target["contact_order"]),
            "n_train_blocks": len(target["train_block_ids"]),
            "n_heldout_blocks": len(target["heldout_block_ids"]),
            "n_train_events": len(target["train_event_indices"]),
            "n_heldout_events": len(target["heldout_event_indices"]),
            "train_ta_events": int(target["train_mode_counts"][0]),
            "train_tb_events": int(target["train_mode_counts"][1]),
            "heldout_ta_events": int(target["heldout_mode_counts"][0]),
            "heldout_tb_events": int(target["heldout_mode_counts"][1]),
            "patient_heldout_diagonal": f"{target['train_to_heldout_diagonal']:.9f}",
            "patient_heldout_crossed": f"{target['train_to_heldout_crossed']:.9f}",
            "patient_heldout_margin": f"{target['train_to_heldout_margin']:.9f}",
            "heldout_ood_fraction": f"{target['heldout_ood_fraction']:.9f}",
            "kmeans_stability_ami": f"{target['kmeans_stability']['pairwise_ami_median']:.9f}",
        }
        rows.append(row)
        print(json.dumps({
            "subject": subject_id,
            "target": target_eligible,
            "geometry": geometry is not None,
            "snn": snn_eligible,
            "margin": target["train_to_heldout_margin"],
        }), flush=True)

    _atomic_csv(output_root / "cohort_eligibility.csv", rows)
    figures = output_root / "figures"
    figure_base = figures / "data_driven_snn_cohort_target_preflight"
    _draw_diagnostic(rows, figure_base)
    n_target = sum(record["patient_target_eligible"] for record in subject_records)
    n_geometry = sum(record["geometry_eligible"] for record in subject_records)
    n_snn = sum(record["snn_eligible"] for record in subject_records)
    input_paths = [
        cohort_path,
        ROOT / inputs["masked_propagation_root"] / "epilepsiae_1146.json",
    ]
    summary = {
        "schema_version": "topic4_data_driven_snn_cohort_target_audit_v1",
        "status": "TARGET_AUDIT_COMPLETE_SNN_NOT_RUN",
        "denominators": {
            "masked_stable_k2": len(subject_records),
            "patient_target_eligible": n_target,
            "geometry_eligible": n_geometry,
            "snn_eligible": n_snn,
        },
        "scientific_boundary": (
            "This artifact validates patient targets and geometry only. It contains "
            "no SNN cohort result and cannot support a model-recovery claim."
        ),
        "subjects": [
            {
                "subject_id": record["subject_id"],
                "patient_target_eligible": record["patient_target_eligible"],
                "geometry_eligible": record["geometry_eligible"],
                "snn_eligible": record["snn_eligible"],
                "geometry_status": record["geometry_status"],
            }
            for record in subject_records
        ],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": _provenance(config_path, input_paths),
    }
    _atomic_json(output_root / "cohort_target_audit.json", summary)
    readme = f"""# 图说明

### data_driven_snn_cohort_target_preflight.png

这张图只审计患者 target 与二维几何资格，不包含 SNN 仿真结果。上游 masked stable-K=2 共 {len(rows)} 人，患者 training/held-out target 合格 {n_target} 人，真实二维几何可实例化 {n_geometry} 人，最终 SNN canary 资格 {n_snn} 人；灰色点是坐标技术排除，不是模型失败。

**关注点**：所有患者的 KMeans 都只在 training recording blocks 上拟合，held-out margin 必须为正；正式 cohort 结论的分母必须同时报告 {n_snn}/{len(rows)}，不得写成 34 人都完成了 SNN。
"""
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "README.md").write_text(readme)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = build(args.config.resolve())
    print(json.dumps(summary["denominators"], sort_keys=True))


if __name__ == "__main__":
    main()
