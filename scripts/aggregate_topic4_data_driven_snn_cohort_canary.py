#!/usr/bin/env python3
"""Aggregate fit-pool multisubject cohort canary workers."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_data_driven_cohort import score_model_ranks_against_target  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_canary_v1.json"


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


def _patient_target(config: dict, subject_id: str) -> tuple[dict, dict]:
    root = ROOT / config["target_root"]
    metadata = json.loads((root / f"{subject_id}.json").read_text())
    with np.load(root / f"{subject_id}_target.npz", allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    recruitment = np.asarray([
        arrays["train_ta_recruitment"], arrays["train_tb_recruitment"]
    ])
    precedence = np.asarray([
        arrays["train_ta_precedence"], arrays["train_tb_precedence"]
    ])
    target = {
        "centers": arrays["kmeans_centers"],
        "profiles": arrays["train_profiles"],
        "recruitment": recruitment,
        "precedence": precedence,
        "ood_threshold": float(metadata["target"]["train_distance_q95"]),
    }
    return metadata, target


def _rows(config: dict, manifest: dict, *, expected_commit: str) -> list[dict]:
    output_root = ROOT / config["output_root"]
    candidates = manifest["candidate_set"]["candidates"]
    targets = {
        subject: _patient_target(config, subject) for subject in config["subjects"]
    }
    rows = []
    for candidate in candidates:
        for seed in config["search"]["fit_network_seeds"]:
            stem = f"{candidate['candidate_id']}_seed_{int(seed)}"
            json_path = output_root / "workers" / f"{stem}.json"
            npz_path = output_root / "workers" / f"{stem}.npz"
            if not json_path.exists() or not npz_path.exists():
                raise RuntimeError(f"missing cohort worker: {stem}")
            worker = json.loads(json_path.read_text())
            if worker["output_npz_sha256"] != _sha256(npz_path):
                raise RuntimeError(f"cohort worker NPZ hash changed: {stem}")
            provenance = worker.get("provenance", {})
            if not (
                provenance.get("expected_git_commit") == expected_commit
                and provenance.get("runtime_modules_match_expected_commit") is True
                and not provenance.get("runtime_modules_dirty")
            ):
                raise RuntimeError(f"cohort worker provenance is invalid: {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                returned = np.asarray(loaded["event_returned"], bool)
                for subject_index, subject_id in enumerate(config["subjects"]):
                    ranks = np.asarray(
                        loaded[f"subject_{subject_index:02d}_ranks"], float,
                    )[returned]
                    _, target = targets[subject_id]
                    score = score_model_ranks_against_target(
                        ranks,
                        patient_centers=target["centers"],
                        patient_profiles=target["profiles"],
                        patient_recruitment=target["recruitment"],
                        patient_precedence=target["precedence"],
                        patient_ood_threshold=target["ood_threshold"],
                    )
                    row = {
                        "subject_id": subject_id,
                        "candidate_id": candidate["candidate_id"],
                        "arm": candidate["arm"],
                        "rotation_deg": int(candidate["node_field"]["transform"]["rotation_deg"]),
                        "seed": int(seed),
                        "worker_status": worker["status"],
                        **score,
                    }
                    rows.append(row)
    return rows


def _select(rows: list[dict], subjects: list[str]) -> dict:
    selection = {}
    for subject in subjects:
        subject_rows = [row for row in rows if row["subject_id"] == subject]
        candidate_ids = sorted({row["candidate_id"] for row in subject_rows})
        summaries = []
        for candidate_id in candidate_ids:
            current = [
                row for row in subject_rows
                if row["candidate_id"] == candidate_id and row["status"] == "EVALUABLE"
                and row["worker_status"] == "COMPLETE"
            ]
            if not current:
                continue
            summaries.append({
                "candidate_id": candidate_id,
                "arm": current[0]["arm"],
                "rotation_deg": current[0]["rotation_deg"],
                "n_evaluable_networks": len(current),
                "selection_score_median": float(np.median([
                    row["selection_score"] for row in current
                ])),
                "supervised_margin_median": float(np.median([
                    row["supervised_margin"] for row in current
                ])),
                "natural_margin_median": float(np.median([
                    row["natural_margin"] for row in current
                ])),
                "natural_seed_ami_median": float(np.median([
                    row["natural_seed_ami_median"] for row in current
                ])),
                "ood_fraction_median": float(np.median([
                    row["ood_fraction"] for row in current
                ])),
            })
        summaries.sort(key=lambda row: (row["selection_score_median"], row["candidate_id"]))
        selection[subject] = {
            "status": "SELECTED" if summaries else "NOT_EVALUABLE",
            "selected_candidate_id": summaries[0]["candidate_id"] if summaries else None,
            "candidate_summaries": summaries,
        }
    return selection


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append({
            key: value for key, value in _json_ready(row).items()
            if not isinstance(value, (dict, list))
        })
    fields = sorted({key for row in scalar_rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _draw(config: dict, manifest: dict, rows: list[dict], selection: dict,
          output: Path) -> None:
    subjects = config["subjects"]
    candidates = [row["candidate_id"] for row in manifest["candidate_set"]["candidates"]]
    supervised = np.full((len(subjects), len(candidates)), np.nan)
    natural = np.full_like(supervised, np.nan)
    for i, subject in enumerate(subjects):
        for j, candidate in enumerate(candidates):
            current = [
                row for row in rows if row["subject_id"] == subject
                and row["candidate_id"] == candidate
                and row["status"] == "EVALUABLE"
                and row["worker_status"] == "COMPLETE"
            ]
            if current:
                supervised[i, j] = np.median([row["supervised_margin"] for row in current])
                natural[i, j] = np.median([row["natural_margin"] for row in current])
    selected_supervised = []
    selected_natural = []
    labels = []
    for i, subject in enumerate(subjects):
        candidate = selection[subject]["selected_candidate_id"]
        if candidate is None:
            continue
        j = candidates.index(candidate)
        selected_supervised.append(supervised[i, j])
        selected_natural.append(natural[i, j])
        labels.append(subject.replace("epilepsiae_", "E").replace("yuquan_", "Y:"))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=True)
    image = axes[0, 0].imshow(supervised, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=2)
    axes[0, 0].set_xticks(range(len(candidates)), candidates, rotation=45, ha="right")
    axes[0, 0].set_yticks(range(len(subjects)), [s.replace("epilepsiae_", "E").replace("yuquan_", "Y:") for s in subjects])
    axes[0, 0].set_title("supervised patient margin", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 0], shrink=0.8)

    image = axes[0, 1].imshow(natural, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=2)
    axes[0, 1].set_xticks(range(len(candidates)), candidates, rotation=45, ha="right")
    axes[0, 1].set_yticks(range(len(subjects)), [s.replace("epilepsiae_", "E").replace("yuquan_", "Y:") for s in subjects])
    axes[0, 1].set_title("natural KMeans patient margin", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[0, 1], shrink=0.8)

    x = np.arange(len(labels))
    axes[1, 0].scatter(x, selected_supervised, color="#b2182b", label="supervised")
    axes[1, 0].scatter(x, selected_natural, color="#2166ac", label="natural KMeans")
    for index in x:
        axes[1, 0].plot(
            [index, index], [selected_supervised[index], selected_natural[index]],
            color="#bbbbbb", linewidth=1,
        )
    axes[1, 0].axhline(0, color="#222222", linewidth=0.8)
    axes[1, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 0].set_ylabel("same - crossed margin")
    axes[1, 0].set_title("per-subject selected candidate", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    selected_rows = []
    for subject in subjects:
        chosen = selection[subject]["selected_candidate_id"]
        selected_rows.extend([
            row for row in rows if row["subject_id"] == subject
            and row["candidate_id"] == chosen
            and row["status"] == "EVALUABLE"
            and row["worker_status"] == "COMPLETE"
        ])
    axes[1, 1].scatter(
        [row["n_in_distribution_events"] for row in selected_rows],
        [row["natural_seed_ami_median"] for row in selected_rows],
        c=["#b2182b" if row["subject_id"] == "epilepsiae_1146" else "#2166ac" for row in selected_rows],
        s=40,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("in-distribution events per network")
    axes[1, 1].set_ylabel("KMeans seed AMI")
    axes[1, 1].set_title("same-network support", loc="left", fontweight="bold")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Data-driven SNN cohort canary | pretrained E1146 morphology transfer; exploratory",
        fontsize=13, fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def aggregate(config_path: Path, *, expected_commit: str) -> dict:
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_commit = manifest.get("provenance", {}).get("expected_git_commit")
    if manifest_commit != expected_commit:
        raise RuntimeError("cohort manifest and aggregator commit differ")
    provenance = _runtime_provenance(expected_commit)
    if (provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("cohort aggregator runtime is not frozen")
    rows = _rows(config, manifest, expected_commit=expected_commit)
    selection = _select(rows, config["subjects"])
    _write_csv(output_root / "fit_subject_candidate_network_scores.csv", rows)
    evaluable = sum(value["status"] == "SELECTED" for value in selection.values())
    source_subject = config["pretrained_source"]["subject"]
    transfer_subjects = [subject for subject in config["subjects"] if subject != source_subject]
    transfer_evaluable = sum(selection[subject]["status"] == "SELECTED" for subject in transfer_subjects)
    status = (
        "CANARY_FIT_EVALUABLE"
        if evaluable >= int(json.loads((ROOT / config["inputs"]["cohort_config"]["path"]).read_text())["canary"]["minimum_subjects_with_evaluable_same_network_k2"])
        else "CANARY_FIT_INSUFFICIENT"
    )
    payload = {
        "status": status,
        "scientific_role": config["scientific_role"],
        "denominators": {
            "canary_subjects": len(config["subjects"]),
            "evaluable_subjects": evaluable,
            "transfer_subjects_excluding_E1146": len(transfer_subjects),
            "evaluable_transfer_subjects": transfer_evaluable,
        },
        "selection": selection,
        "claim_boundary": config["canary_boundary"],
        "manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": _sha256(manifest_path),
        },
        "provenance": provenance,
    }
    atomic_write_json(_json_ready(payload), output_root / "fit_selection.json")
    figure_base = output_root / "figures" / "data_driven_snn_cohort_canary_fit"
    _draw(config, manifest, rows, selection, figure_base)
    readme = f"""# 图说明

### data_driven_snn_cohort_canary_fit.png

这张图比较 E1146 预训练连续场的四个旋转，以及 Node-only 和 Node+EE+E→I 两种机制臂，在 {len(config['subjects'])} 位几何多样患者上的 fit-pool 结果。它同时分开显示 supervised patient margin 与 natural KMeans margin；E1146 是开发来源病例，跨患者 transfer 统计必须排除它。

**关注点**：状态为 `{status}`，但这仍是探索性 canary，不是 34 人 cohort 结论；只有进入独立 selection/final network 后才能判断可迁移的内部刻板结构。
"""
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    (figure_base.parent / "README.md").write_text(readme)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    print(json.dumps(_json_ready(aggregate(
        args.config.resolve(), expected_commit=commit,
    )), indent=2))


if __name__ == "__main__":
    main()
