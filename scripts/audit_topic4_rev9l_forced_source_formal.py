"""Post-run audit of the rev9-L forced-source formal fit assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_forced_source_capacity import (  # noqa: E402
    source_mode_correlation_summary,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_forced_source_formal.json"
DEFAULT_SUMMARY = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/"
    "forced_source_capacity/formal_fit/forced_source_capacity_summary.json")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "q05": None, "q95": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "n": int(len(values)),
    }


def _matrix_spread(matrices):
    values = np.asarray(matrices, float)
    output = np.full(values.shape[1:], np.nan)
    finite = np.isfinite(values).any(axis=0)
    output[finite] = (
        np.nanmax(values[:, finite], axis=0)
        - np.nanmin(values[:, finite], axis=0)
    )
    return output


def _json_matrix(values):
    return [[None if not np.isfinite(value) else float(value) for value in row]
            for row in np.asarray(values, float)]


def _load_arm(summary, arm):
    rows, primary, inclusive = [], [], []
    seeds = []
    for record in summary["worker_inputs"]:
        if record["arm"] != arm:
            continue
        if _sha256(record["json"]["path"]) != record["json"]["sha256"]:
            raise RuntimeError(f"worker JSON changed: {record['json']['path']}")
        if _sha256(record["npz"]["path"]) != record["npz"]["sha256"]:
            raise RuntimeError(f"worker NPZ changed: {record['npz']['path']}")
        payload = json.loads(Path(record["json"]["path"]).read_text())
        if payload["arrays"]["sha256"] != record["npz"]["sha256"]:
            raise RuntimeError("worker JSON and summary disagree on array hash")
        with np.load(record["npz"]["path"], allow_pickle=False) as loaded:
            source_ids = np.asarray(loaded["source_ids"]).astype(str)
            primary_curves = np.asarray(loaded["rank_curves"], float)
            inclusive_curves = np.asarray(
                loaded["inclusive_packet_frame_rank_curves"], float)
        if len(payload["runs"]) != len(source_ids):
            raise RuntimeError("worker rows and source ids do not align")
        for index, row in enumerate(payload["runs"]):
            if row["source_id"] != source_ids[index]:
                raise RuntimeError("worker source order changed")
            rows.append({"seed": int(record["seed"]), **row})
            primary.append(primary_curves[index])
            inclusive.append(inclusive_curves[index])
        seeds.append(int(record["seed"]))
    return rows, np.asarray(primary), np.asarray(inclusive), seeds


def _mode_pair(curves, rows, prototypes, source, intended, *, clean_only=False):
    selected = [index for index, row in enumerate(rows)
                if row["source_id"] == source and (
                    not clean_only or row["sham_triggered_event"] is None)]
    if not selected:
        return None
    result = source_mode_correlation_summary(
        curves[selected], [source] * len(selected), prototypes,
        source_order=[source])
    pair = np.asarray(result["sources"][source][
        "per_network_correlation_to_A_B"], float)
    return {
        "intended_mode": "A" if intended == 0 else "B",
        "intended_correlation": _summary(pair[:, intended]),
        "cross_correlation": _summary(pair[:, 1 - intended]),
        "intended_minus_cross": _summary(
            pair[:, intended] - pair[:, 1 - intended]),
        "seeds": [int(rows[index]["seed"]) for index in selected],
    }


def _plot(audit, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = audit["arm_order"]
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.5),
                                constrained_layout=True)
    x = np.arange(len(arms))
    for source, label, color, marker in (
            ("component_2", "component 2 to A", "#d1495b", "o"),
            ("component_1", "component 1 to B", "#277da1", "s")):
        values = [audit["arms"][arm]["primary"][source][
            "intended_correlation"]["median"] for arm in arms]
        axes[0].plot(x, values, marker=marker, color=color, label=label)
    axes[0].set_xticks(x, arms, rotation=20)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("patient prototype Spearman")
    axes[0].set_title("a  route shape is arm-invariant", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=7)

    effects = audit["edge_minus_null_forced_mass"]
    for index, source in enumerate(("component_2", "component_1")):
        record = effects[source]
        estimate = record["estimate"]
        interval = record["interval_95"]
        axes[1].errorbar(index, estimate,
                         yerr=[[estimate - interval[0]], [interval[1] - estimate]],
                         fmt="o", color="#d55e00", capsize=4)
    axes[1].axhline(0.0, color="0.45", lw=1)
    axes[1].set_xticks([0, 1], ["component 2\nto A", "component 1\nto B"])
    axes[1].set_ylabel("Edge - Null downstream spike mass")
    axes[1].set_title("b  Edge amplifies forced relay", loc="left", weight="bold")

    offset = {"component_2": -0.08, "component_1": 0.08}
    for source, color, marker in (
            ("component_2", "#d1495b", "o"),
            ("component_1", "#277da1", "s")):
        all_values = [audit["arms"][arm]["primary"][source][
            "intended_minus_cross"]["median"] for arm in arms]
        clean_values = [audit["arms"][arm]["sham_clear"][source][
            "intended_minus_cross"]["median"] for arm in arms]
        axes[2].scatter(x + offset[source], all_values, color=color,
                        marker=marker, s=32)
        axes[2].scatter(x + offset[source], clean_values, facecolor="white",
                        edgecolor=color, marker=marker, s=40)
    axes[2].axhline(0.0, color="0.45", ls=":", lw=1)
    axes[2].set_xticks(x, arms, rotation=20)
    axes[2].set_ylabel("intended minus cross Spearman")
    axes[2].set_title("c  sham-clear sensitivity", loc="left", weight="bold")
    axes[2].text(0.02, 0.04, "filled: all; open: sham-clear",
                 transform=axes[2].transAxes, fontsize=7)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"rev9l_l1_review_audit.{suffix}", dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()
    config_path, summary_path = Path(args.config), Path(args.summary)
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary["status"] != "REV9L_L1_FORCED_FIT_COMPLETE":
        raise RuntimeError("formal forced-source summary is incomplete")
    if summary["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("formal config changed after aggregation")
    with np.load(config["inputs"]["patient_training_target"]["path"],
                 allow_pickle=False) as loaded:
        prototypes = np.asarray(loaded["patient_train_mode_prototypes"], float)

    mapping = {
        config["primary_mapping"]["mode_A_source"]: 0,
        config["primary_mapping"]["mode_B_source"]: 1,
    }
    arms = {}
    primary_matrices = []
    inclusive_matrices = []
    for arm in config["arms"]:
        rows, primary, inclusive, seeds = _load_arm(summary, arm)
        primary_result = source_mode_correlation_summary(
            primary, [row["source_id"] for row in rows], prototypes,
            source_order=config["packet"]["formal_sources"])
        inclusive_result = source_mode_correlation_summary(
            inclusive, [row["source_id"] for row in rows], prototypes,
            source_order=config["packet"]["formal_sources"])
        primary_matrix = np.asarray(
            primary_result["median_correlation_matrix"], float)
        inclusive_matrix = np.asarray(
            inclusive_result["median_correlation_matrix"], float)
        primary_matrices.append(primary_matrix)
        inclusive_matrices.append(inclusive_matrix)
        collision_seeds = sorted({int(row["seed"]) for row in rows
                                  if row["sham_triggered_event"] is not None})
        arms[arm] = {
            "network_seeds": seeds,
            "sham_overlap_seeds": collision_seeds,
            "primary": {
                source: _mode_pair(primary, rows, prototypes, source, intended)
                for source, intended in mapping.items()
            },
            "inclusive_packet_frame": {
                source: _mode_pair(inclusive, rows, prototypes, source, intended)
                for source, intended in mapping.items()
            },
            "sham_clear": {
                source: _mode_pair(
                    primary, rows, prototypes, source, intended, clean_only=True)
                for source, intended in mapping.items()
            },
            "max_abs_correlation_change_when_packet_frame_is_included": float(
                np.nanmax(np.abs(primary_matrix - inclusive_matrix))),
        }

    effects = {
        source: summary["paired_factorial_by_source"][source][
            "downstream_positive_spike_mass"]["delta_edge"]
        for source in mapping
    }
    payload = {
        "status": "REV9L_L1_REVIEW_AUDIT_COMPLETE",
        "scientific_role": (
            "zero-simulation development audit; patient held-out is not read"),
        "claim_boundary": (
            "forced source capacity and relay modulation only; not spontaneous "
            "patient interictal reproduction and not core causality"),
        "arm_order": config["arms"],
        "arms": arms,
        "between_arm_primary_matrix_spread": _json_matrix(
            _matrix_spread(primary_matrices)),
        "edge_minus_null_forced_mass": effects,
        "mode_descriptor_snapshot": {
            arm: summary["arms"][arm]["primary_mode_descriptors"]["modes"]
            for arm in config["arms"]
        },
        "interpretation": {
            "ignition": (
                "forcing makes component 1 and component 2 readable in every arm, "
                "so Edge-only spontaneous failure is primarily an ignition/access issue"),
            "mode_A": (
                "component 2 remains only weakly patient-A-like after forcing in every "
                "arm; forced initiation does not repair mode-A geometry"),
            "edge": (
                "Edge increases downstream mass relative to Null while leaving source-"
                "conditioned rank profiles nearly unchanged; it acts as a conditional relay"),
            "sham_collision": (
                "Node and Node+Edge mass/interaction estimates include seeds with a sham "
                "event in the trigger window and are descriptive, not clean inhibition evidence"),
            "beta": (
                "do not open beta: the unresolved defect is mode-A route geometry, not an "
                "isolated radial response-width mismatch"),
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
        },
        "provenance": {
            "git_commit_at_start": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "producer_sha256_at_start": _sha256(__file__),
            "numeric_modules_dirty_at_start": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--",
                 "scripts/audit_topic4_rev9l_forced_source_formal.py",
                 "src/topic4_forced_source_capacity.py"],
                cwd=ROOT, text=True).strip()),
            "systemd_unit": os.environ.get("REV9L_SYSTEMD_UNIT"),
        },
    }
    output_dir = summary_path.parent / "review_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, output_dir / "rev9l_l1_review_audit.json")
    _plot(payload, output_dir / "figures")
    (output_dir / "figures" / "README.md").write_text(
        "### rev9l_l1_review_audit.png\n"
        "左图比较四臂强制 source 响应与患者训练集 A/B prototype 的绝对相关；中图给出 Edge 相对 Null 的逐网络 downstream mass 增量；右图比较全样本和 sham-clear 子集。全部是 development audit，不读取 patient held-out。\n\n"
        "**关注点**：Edge 是否只增加传播质量而不改变传播形状，以及 mode A 的弱绝对匹配是否在排除 sham-window collision 后仍存在。\n"
    )
    print(json.dumps({
        "status": payload["status"],
        "edge_minus_null_forced_mass": effects,
        "sham_overlap_seeds": {
            arm: record["sham_overlap_seeds"] for arm, record in arms.items()},
        "out": str(output_dir / "rev9l_l1_review_audit.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
