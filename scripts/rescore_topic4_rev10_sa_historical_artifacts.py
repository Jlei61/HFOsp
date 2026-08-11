"""SA4 zero-simulation shaft-aware audit of retained historical model events."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from scripts.build_topic4_rev10_sa_shaft_aware_target import (  # noqa: E402
    _atomic_json,
    _calibrated_mode_score,
    _distance_row,
    _flatten_distances,
)
from scripts.run_topic4_rev9l_objective_replay import _load_reference  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.sef_hfo_observation import extract_lagpat  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fixed_count_indices,
    normalized_rank_curve,
)
from src.topic4_core_field_rev9 import assign_frozen_modes  # noqa: E402
from src.topic4_core_field_runner import _placement  # noqa: E402
from src.topic4_shaft_aware import (  # noqa: E402
    PAIR_CLASS_ORDER,
    SHAFT_ORDER,
    centered_smooth_max,
    contract_groups,
    contract_pairs,
    floor_excess,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_shaft_aware.json"
FIXED_EVENTS_PER_MODE = 6


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runtime_provenance(config_path, target_path):
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/sef_hfo_observation.py",
        config_path.resolve(),
        target_path.resolve(),
    ]
    tracked = [path for path in paths if path.is_relative_to(ROOT)]
    relative = [str(path.relative_to(ROOT)) for path in tracked]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_files_dirty": bool(dirty),
        "runtime_file_sha256": {
            str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): _sha256(path)
            for path in paths
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _target_descriptor(loaded, prefix, mode):
    stem = f"{prefix}mode_{mode}_"
    return {
        "recruitment": {
            shaft: np.asarray(loaded[f"{stem}recruitment_{shaft.lower()}"])
            for shaft in SHAFT_ORDER
        },
        "precedence": {
            pair_class: np.asarray(loaded[
                f"{stem}precedence_{pair_class.lower().replace('-', '_')}"
            ]) for pair_class in PAIR_CLASS_ORDER
        },
        "profile": {
            "ICL": np.asarray(loaded[f"{stem}profile_icl"]),
            "SCL": np.asarray(loaded[f"{stem}profile_scl"]),
            "cross": np.asarray(loaded[f"{stem}profile_cross"]),
        },
    }


def load_scoring_contract(target_path, floors_path, semantics,
                          fixed_events_per_mode=FIXED_EVENTS_PER_MODE):
    if semantics not in {"FULL_TIMING", "ORDINAL_COMPATIBLE"}:
        raise ValueError("unknown scoring semantics")
    prefix = "" if semantics == "FULL_TIMING" else "ordinal_"
    with np.load(target_path, allow_pickle=False) as loaded:
        embedding_prefix = "" if semantics == "FULL_TIMING" else "ordinal_"
        embedding = {
            "center": np.asarray(loaded[f"{embedding_prefix}feature_center"]),
            "scale": np.asarray(loaded[f"{embedding_prefix}feature_scale"]),
            "components": np.asarray(loaded[f"{embedding_prefix}pca_components"]),
            "directions": np.asarray(loaded[f"{embedding_prefix}sw_directions"]),
        }
        targets = {
            mode: {
                "descriptor": _target_descriptor(loaded, prefix, mode),
                "reference_z": np.asarray(loaded[f"{prefix}mode_{mode}_reference_z"]),
            } for mode in (0, 1)
        }
        contact_names = np.asarray(loaded["contact_names"]).astype(str)
    floor_payload = json.loads(Path(floors_path).read_text())
    floor_key = (
        "full_timing_floors" if semantics == "FULL_TIMING"
        else "ordinal_compatible_floors"
    )
    floors = floor_payload[floor_key][str(int(fixed_events_per_mode))]
    return contact_names, embedding, targets, floors


def _component_excess(flat, floor):
    output = {}
    for key, value in flat.items():
        if key == "multishaft_fraction":
            continue
        output[key] = floor_excess(
            value, floor[key]["median"], floor[key]["q95"],
        )
    return output


def score_mode_conditioned_events(
    values,
    labels,
    *,
    groups,
    pairs,
    embedding,
    targets,
    floors,
    config,
    fixed_events_per_mode=FIXED_EVENTS_PER_MODE,
):
    """Score a fixed count per old direction mode without deleting missing shafts."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("event values and labels do not align")
    modes = {}
    for mode in (0, 1):
        available = np.flatnonzero(labels == mode)
        chosen_local = fixed_count_indices(len(available), fixed_events_per_mode)
        if chosen_local is None:
            modes[str(mode)] = {
                "status": "INSUFFICIENT_MODE_SUPPORT",
                "n_available": int(len(available)),
                "n_required": int(fixed_events_per_mode),
            }
            continue
        selected = available[chosen_local]
        row = _flatten_distances(_distance_row(
            values[selected], targets[mode], groups, pairs, embedding,
        ))
        calibrated = _calibrated_mode_score(
            row, floors[str(mode)], config,
        )
        modes[str(mode)] = {
            "status": "OK",
            "n_available": int(len(available)),
            "n_scored": int(len(selected)),
            "selected_event_indices": selected,
            "raw": row,
            "floor_excess": _component_excess(row, floors[str(mode)]),
            "objective": calibrated,
        }
    if any(row["status"] != "OK" for row in modes.values()):
        return {
            "status": "INSUFFICIENT_MODE_SUPPORT",
            "n_events": int(len(values)),
            "mode_counts": np.bincount(labels[labels >= 0], minlength=2),
            "modes": modes,
        }
    scores = [modes[str(mode)]["objective"]["mode_score"] for mode in (0, 1)]
    return {
        "status": "OK",
        "n_events": int(len(values)),
        "mode_counts": np.bincount(labels, minlength=2),
        "fixed_events_per_mode": int(fixed_events_per_mode),
        "modes": modes,
        "mean_mode_score": float(np.mean(scores)),
        "weak_mode_score": centered_smooth_max(scores, 0.25),
        "weak_mode": "A" if scores[0] >= scores[1] else "B",
        "pooled_multishaft_fraction": float(np.mean([
            np.isfinite(row[groups["ICL"]]).any()
            and np.isfinite(row[groups["SCL"]]).any()
            for row in values
        ])) if len(values) else float("nan"),
        "pooled_scl_recruitment_fraction": float(np.isfinite(
            values[:, groups["SCL"]]
        ).mean()) if len(values) else float("nan"),
    }


def _reorder_rows(values, source_names, target_names):
    source_names = [str(value) for value in source_names]
    lookup = {name: index for index, name in enumerate(source_names)}
    if set(source_names) != set(target_names):
        raise RuntimeError("artifact contact set differs from SA0 contract")
    return np.asarray(values, dtype=float)[:, [lookup[name] for name in target_names]]


def _score_ordinal_npz(path, rank_key, label_key, name_key, contract, scoring, config,
                       ood=None):
    with np.load(path, allow_pickle=False) as loaded:
        values = _reorder_rows(
            loaded[rank_key], loaded[name_key], scoring["contact_names"],
        )
        labels = np.asarray(loaded[label_key], dtype=int)
    result = score_mode_conditioned_events(
        values, labels, groups=contract["groups"], pairs=contract["pairs"],
        embedding=scoring["embedding"], targets=scoring["targets"],
        floors=scoring["floors"], config=config,
    )
    if ood is not None:
        result["frozen_ood_fraction"] = float(np.mean(ood))
    return result


def _classifier(config):
    frozen_path = ROOT / "results/topic4_sef_hfo/data_driven_core_field_rev9/frozen_readouts.npz"
    reference_path = ROOT / config["inputs"]["old_rank_curve_reference"]
    with np.load(frozen_path, allow_pickle=False) as loaded:
        classifier = {
            "embedding_centroids": np.asarray(loaded["classifier_embedding_centroids"]),
            "ood_distance_thresholds": np.asarray(loaded["classifier_ood_thresholds"]),
        }
    return classifier, _load_reference(reference_path)


def _json_rank_family(paths, contact_names, classifier, reference):
    axial = axial_map()
    grid = np.asarray(reference["grid"])
    rows, curves, source_files = [], [], []
    for path in sorted(paths):
        payload = json.loads(Path(path).read_text())
        for event in payload.get("events", []):
            ranks = event.get("ranks") or {}
            curve = normalized_rank_curve(ranks, axial, grid=grid)
            if curve is None:
                continue
            rows.append([
                np.nan if ranks.get(name) is None else float(ranks[name])
                for name in contact_names
            ])
            curves.append(curve)
            source_files.append(str(path))
    if not rows:
        return np.empty((0, len(contact_names))), np.empty(0, int), np.empty(0, bool), []
    assigned = assign_frozen_modes(np.asarray(curves), classifier, reference)
    return (
        np.asarray(rows, dtype=float), assigned["labels"], assigned["ood"], source_files,
    )


def reconstruct_worker_onsets(npz_path, json_path, *, montage_names,
                              montage_xy, engine_l, readout_rr, target_names):
    """Re-extract timing from retained envelopes and prove rank parity."""
    metadata = json.loads(Path(json_path).read_text())
    with np.load(npz_path, allow_pickle=False) as loaded:
        envelopes = np.asarray(loaded["excess_contact_envelope"], dtype=float)
        envelope_dt = float(loaded["envelope_dt_ms"])
        positions = np.asarray(loaded["positions_E"], dtype=float)
        stored_names = np.asarray(loaded["contact_names"]).astype(str)
        stored_ranks = np.asarray(loaded["contact_ranks"], dtype=float)
        source_ids = np.asarray(loaded["source_ids"]).astype(str)
        assigned_ood = np.asarray(loaded["assigned_ood"], dtype=bool)
    inside = (
        (montage_xy[:, 0] >= 0.0) & (montage_xy[:, 0] <= float(engine_l))
        & (montage_xy[:, 1] >= 0.0) & (montage_xy[:, 1] <= float(engine_l))
    )
    has_neuron = np.asarray([
        np.any(np.linalg.norm(positions - contact, axis=1) <= float(readout_rr))
        for contact in montage_xy
    ])
    valid = inside & has_neuron
    valid_names = np.asarray(montage_names)[valid]
    output, parity = [], []
    for event_index, envelope in enumerate(envelopes):
        selected = envelope[valid]
        floor = float(selected.min())
        margin = 0.1 * (float(selected.max()) - floor)
        window = tuple(metadata["runs"][event_index]["paired_excess_readout"]["window_ms"])
        artifact = extract_lagpat(
            selected, envelope_dt, [window], floor, margin,
            timing_frac=0.5, tie_tol=envelope_dt,
        )
        lag_by_name = {
            name: artifact.lag_raw[index, 0]
            for index, name in enumerate(valid_names)
        }
        rank_by_name = {
            name: artifact.ranks[index, 0]
            for index, name in enumerate(valid_names)
        }
        lag_row = np.asarray([lag_by_name.get(name, np.nan) for name in target_names])
        rank_row = np.asarray([rank_by_name.get(name, np.nan) for name in stored_names])
        equal = bool(np.allclose(rank_row, stored_ranks[event_index], equal_nan=True))
        if not equal:
            raise RuntimeError(f"envelope/rank parity failed: {npz_path} row {event_index}")
        output.append(lag_row)
        parity.append(equal)
    return {
        "candidate_id": metadata.get("candidate_id", "scalar_baseline"),
        "network_seed": int(metadata.get("network_seed", metadata["seed"])),
        "dynamics_seed": int(metadata.get("dynamics_seed", metadata["seed"])),
        "source_ids": source_ids,
        "onsets": np.asarray(output, dtype=float),
        "assigned_ood": assigned_ood,
        "rank_parity": parity,
    }


def _forced_candidate_scores(worker_dir, contract, scoring, config, *, repeated):
    stage = json.loads((ROOT / config["inputs"]["stage_config"]).read_text())
    placement = _placement(stage)
    montage_names = np.asarray(placement["montage_sheet"].names).astype(str)
    montage_xy = np.asarray(placement["montage_sheet"].contacts, dtype=float)
    records = []
    for npz_path in sorted(Path(worker_dir).glob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        records.append(reconstruct_worker_onsets(
            npz_path, json_path, montage_names=montage_names, montage_xy=montage_xy,
            engine_l=stage["engine"]["L"],
            readout_rr=config["readout_contract"]["lfp_Rr_mm"],
            target_names=scoring["contact_names"],
        ))
    grouped = defaultdict(lambda: defaultdict(lambda: {0: [], 1: [], "ood": []}))
    source_mode = {"component_2": 0, "component_1": 1}
    for record in records:
        replicate = record["dynamics_seed"] if repeated else 0
        bucket = grouped[record["candidate_id"]][replicate]
        for index, source in enumerate(record["source_ids"]):
            if source not in source_mode:
                continue
            bucket[source_mode[source]].append(record["onsets"][index])
            bucket["ood"].append(bool(record["assigned_ood"][index]))
    candidate_rows = []
    for candidate_id, replicates in sorted(grouped.items()):
        scores = []
        for replicate_id, events in sorted(replicates.items()):
            values = np.asarray(events[0] + events[1], dtype=float)
            labels = np.r_[np.zeros(len(events[0]), int), np.ones(len(events[1]), int)]
            score = score_mode_conditioned_events(
                values, labels, groups=contract["groups"], pairs=contract["pairs"],
                embedding=scoring["embedding"], targets=scoring["targets"],
                floors=scoring["floors"], config=config,
            )
            score["replicate_id"] = int(replicate_id)
            score["assigned_ood_fraction"] = float(np.mean(events["ood"]))
            scores.append(score)
        eligible = [score for score in scores if score["status"] == "OK"]
        row = {
            "candidate_id": candidate_id,
            "n_replicates": len(scores),
            "n_eligible_replicates": len(eligible),
            "replicate_scores": scores,
        }
        if eligible:
            for key in ("weak_mode_score", "mean_mode_score",
                        "pooled_multishaft_fraction", "pooled_scl_recruitment_fraction"):
                row[f"median_{key}"] = float(np.median([score[key] for score in eligible]))
            for mode in (0, 1):
                for metric in ("recruitment.ICL", "recruitment.SCL",
                               "precedence.ICL-ICL", "precedence.SCL-SCL",
                               "precedence.ICL-SCL"):
                    row[f"median_mode_{mode}_{metric}_excess"] = float(np.median([
                        score["modes"][str(mode)]["floor_excess"][metric]
                        for score in eligible
                    ]))
        candidate_rows.append(row)
    return candidate_rows, len(records), int(sum(sum(record["rank_parity"]) for record in records))


def _old_objective_maps():
    l2_path = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/"
        "relaxed_edge_oracle/sobol_fit/sobol_fit_summary.json"
    )
    l3_path = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/"
        "relaxed_edge_oracle/network_oracle_repeated/fit/l3b_repeated_fit_oracle.json"
    )
    l2 = json.loads(l2_path.read_text())
    l3 = json.loads(l3_path.read_text())
    return (
        {row["candidate_id"]: row.get("score", {}).get("objective")
         for row in l2["candidates"]},
        {row["candidate_id"]: row.get("median_objective")
         for row in l3["candidate_summary"]},
    )


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row if key != "replicate_scores"})
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in keys})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _association(rows):
    valid = [
        row for row in rows
        if row.get("old_objective") is not None
        and row.get("median_weak_mode_score") is not None
        and np.isfinite(row["old_objective"])
        and np.isfinite(row["median_weak_mode_score"])
    ]
    if len(valid) < 3:
        return {"rho": None, "pvalue": None, "n": len(valid)}
    result = spearmanr(
        [row["old_objective"] for row in valid],
        [row["median_weak_mode_score"] for row in valid],
    )
    return {"rho": float(result.statistic), "pvalue": float(result.pvalue), "n": len(valid)}


def _plot(summary, l2_rows, l3_rows, output_root):
    fig, axes = plt.subplots(1, 4, figsize=(15.4, 3.9))
    family_rows = [row for row in summary["datasets"].values()
                   if row.get("score", {}).get("status") == "OK"]
    labels = [row["label"] for row in family_rows]
    coverage = [row["score"]["pooled_scl_recruitment_fraction"] for row in family_rows]
    not_evaluable = [
        row["score"].get("patient_mode_status") == "NOT_EVALUABLE"
        for row in family_rows
    ]
    bars = axes[0].bar(
        np.arange(len(labels)), coverage,
        color=["#B8BDC2" if flag else "#4C78A8" for flag in not_evaluable],
    )
    for bar, flag in zip(bars, not_evaluable):
        if flag:
            bar.set_hatch("///")
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                "OOD", ha="center", va="bottom", fontsize=7, rotation=90,
            )
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
    axes[0].set_ylabel("SCL recruitment fraction")
    axes[0].set_title("A  SCL recruitment; OOD marked", loc="left", weight="bold")

    for rows, color, label in ((l2_rows, "#F28E2B", "L2"), (l3_rows, "#2A9D8F", "L3")):
        x = [row.get("median_mode_0_precedence.ICL-ICL_excess", np.nan) for row in rows]
        y = [row.get("median_mode_0_recruitment.SCL_excess", np.nan) for row in rows]
        axes[1].scatter(x, y, s=18, alpha=0.6, color=color, label=label)
    axes[1].set_xlabel("mode A ICL precedence excess")
    axes[1].set_ylabel("mode A SCL recruitment excess")
    axes[1].set_title("B  Missing shaft vs route", loc="left", weight="bold")
    axes[1].legend(frameon=False)

    for axis, rows, title in ((axes[2], l2_rows, "C  L2 ranking"),
                              (axes[3], l3_rows, "D  L3 ranking")):
        x = np.asarray([row.get("old_objective", np.nan) for row in rows], dtype=float)
        y = np.asarray([row.get("median_weak_mode_score", np.nan) for row in rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[valid], y[valid], s=20, color="#6C757D", alpha=0.7)
        association = _association(rows)
        axis.text(0.04, 0.96, f"rho={association['rho']:.2f}\nn={association['n']}",
                  transform=axis.transAxes, va="top")
        axis.set_xlabel("old shaft-blind objective")
        axis.set_ylabel("new SA weak-mode score")
        axis.set_title(title, loc="left", weight="bold")
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    figure_dir = output_root / "figures"
    stem = figure_dir / "rev10_sa_historical_artifact_rescore"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    readme = figure_dir / "README.md"
    if "### rev10_sa_historical_artifact_rescore" not in readme.read_text():
        with readme.open("a") as stream:
            stream.write("""

### rev10_sa_historical_artifact_rescore

这张图对保留逐事件 contact 数据的历史模型做零仿真重评分。A 显示各历史家族的 SCL recruitment，斜线灰柱表示 OOD 超过 50%、患者模式不可评价；B 将 mode A 的 ICL 内 precedence 误差与 SCL recruitment 误差分开；C/D 比较旧 shaft-blind objective 与新 shaft-aware score 在 L2/L3 候选中的排序。

**关注点**：没有逐事件 identity 的 48 个 field-fit candidates 不进图；FULL_TIMING 与 ORDINAL_COMPATIBLE 使用各自患者 floors，不跨语义比较绝对分数。
""")
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    target_path = output_root / "shaft_aware_patient_training_target.npz"
    floors_path = output_root / "shaft_aware_patient_floors.json"
    factor_path = output_root / "direction_extent_factorization_audit.json"
    factor = json.loads(factor_path.read_text())
    if factor["status"] != "DIRECTION_AND_EXTENT_FACTORS_BOTH_SUPPORTED_EXPLORATORY":
        raise RuntimeError("SA4 requires the patient mode-factorization resolution")
    contract_payload = json.loads((output_root / "contact_shaft_contract.json").read_text())
    contract = {
        "groups": contract_groups(contract_payload),
        "pairs": contract_pairs(contract_payload),
    }
    ordinal_names, ordinal_embedding, ordinal_targets, ordinal_floors = (
        load_scoring_contract(target_path, floors_path, "ORDINAL_COMPATIBLE"))
    timing_names, timing_embedding, timing_targets, timing_floors = (
        load_scoring_contract(target_path, floors_path, "FULL_TIMING"))
    if not np.array_equal(ordinal_names, timing_names):
        raise RuntimeError("timing/ordinal contact contracts differ")
    ordinal = {"contact_names": ordinal_names, "embedding": ordinal_embedding,
               "targets": ordinal_targets, "floors": ordinal_floors}
    timing = {"contact_names": timing_names, "embedding": timing_embedding,
              "targets": timing_targets, "floors": timing_floors}

    datasets = {}
    final_path = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_stage3/"
        "joint_confirmation_rev8_1/final_event_profiles.npz"
    )
    datasets["rev8_1_final"] = {
        "label": "rev8.1 final", "semantics": "ORDINAL_COMPATIBLE",
        "path": str(final_path.relative_to(ROOT)),
    }
    # final rank matrix is contacts x events, unlike every later artifact.
    with np.load(final_path, allow_pickle=False) as loaded:
        final_values = _reorder_rows(
            np.asarray(loaded["model_rank_matrix"]).T,
            loaded["contact_names"], ordinal_names,
        )
        final_labels = np.asarray(loaded["model_labels"], dtype=int)
    datasets["rev8_1_final"]["score"] = score_mode_conditioned_events(
        final_values, final_labels, groups=contract["groups"], pairs=contract["pairs"],
        embedding=ordinal_embedding, targets=ordinal_targets,
        floors=ordinal_floors, config=config,
    )
    datasets["rev8_1_final"]["score"]["patient_mode_status"] = "DEVELOPMENT_ONLY"

    factorial_path = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9/"
        "node_edge_factorial/factorial_summary.npz"
    )
    with np.load(factorial_path, allow_pickle=False) as loaded:
        for slug, label in (("null", "Null"), ("node", "Node"),
                            ("edge", "Edge"), ("node_edge", "Node+Edge")):
            values = _reorder_rows(
                loaded[f"{slug}_ranks"], loaded[f"{slug}_contact_names"], ordinal_names,
            )
            labels = np.asarray(loaded[f"{slug}_frozen_labels"], dtype=int)
            score = score_mode_conditioned_events(
                values, labels, groups=contract["groups"], pairs=contract["pairs"],
                embedding=ordinal_embedding, targets=ordinal_targets,
                floors=ordinal_floors, config=config,
            )
            score["frozen_ood_fraction"] = float(np.mean(loaded[f"{slug}_frozen_ood"]))
            score["patient_mode_status"] = (
                "NOT_EVALUABLE" if score["frozen_ood_fraction"] > 0.5
                else "DEVELOPMENT_ONLY"
            )
            datasets[f"rev9_{slug}"] = {
                "label": label, "semantics": "ORDINAL_COMPATIBLE",
                "path": str(factorial_path.relative_to(ROOT)), "score": score,
            }

    classifier, old_reference = _classifier(config)
    hand_paths = glob.glob(str(ROOT / (
        "results/topic4_sef_hfo/field_swap_subject_snn/"
        "readout_epilepsiae_1146_paired_tsrc_highn_s*_20260721.json"
    )))
    filament_paths = glob.glob(str(ROOT / (
        "results/topic4_sef_hfo/field_swap_subject_snn/"
        "readout_epilepsiae_1146_learned_core_field_pool_s*.json"
    )))
    for key, label, paths in (("hand_dual_core", "hand dual-core", hand_paths),
                              ("stage2_filament", "Stage 2 filament", filament_paths)):
        values, labels, ood, source_files = _json_rank_family(
            paths, ordinal_names, classifier, old_reference,
        )
        score = score_mode_conditioned_events(
            values, labels, groups=contract["groups"], pairs=contract["pairs"],
            embedding=ordinal_embedding, targets=ordinal_targets,
            floors=ordinal_floors, config=config,
        )
        score["frozen_ood_fraction"] = float(np.mean(ood)) if len(ood) else None
        score["patient_mode_status"] = (
            "NOT_EVALUABLE"
            if score["frozen_ood_fraction"] is not None
            and score["frozen_ood_fraction"] > 0.5
            else "DEVELOPMENT_ONLY"
        )
        datasets[key] = {
            "label": label, "semantics": "ORDINAL_COMPATIBLE",
            "path_pattern": str(Path(paths[0]).parent.relative_to(ROOT)) if paths else None,
            "n_files": len(paths), "n_events_with_rank_curve": len(values), "score": score,
        }

    l2_dir = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/"
        "relaxed_edge_oracle/sobol_fit/workers"
    )
    l3_dir = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/"
        "relaxed_edge_oracle/network_oracle_repeated/fit/workers"
    )
    l2_rows, l2_workers, l2_parity = _forced_candidate_scores(
        l2_dir, contract, timing, config, repeated=False,
    )
    l3_rows, l3_workers, l3_parity = _forced_candidate_scores(
        l3_dir, contract, timing, config, repeated=True,
    )
    l2_old, l3_old = _old_objective_maps()
    for row in l2_rows:
        row["old_objective"] = l2_old.get(row["candidate_id"])
    for row in l3_rows:
        row["old_objective"] = l3_old.get(row["candidate_id"])
    _atomic_csv(output_root / "l2_shaft_aware_candidate_scores.csv", l2_rows)
    _atomic_csv(output_root / "l3_shaft_aware_candidate_scores.csv", l3_rows)

    fit_checkpoint = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_stage3/"
        "joint_fit_kmeans_rev8_1/checkpoint_K3_r0.json"
    )
    fit = json.loads(fit_checkpoint.read_text())
    n_fit_candidates = len(fit.get("history", []))
    inventory = [
        {
            "family": "rev8_1_field_fit_history",
            "semantics": "NOT_RESCORABLE",
            "n_candidates": n_fit_candidates,
            "reason": "per-event fixed-contact values were not retained",
            "path": str(fit_checkpoint.relative_to(ROOT)),
        },
        *[
            {
                "family": key,
                "semantics": row["semantics"],
                "n_events": row["score"].get("n_events"),
                "score_status": row["score"].get("status"),
                "path": row.get("path", row.get("path_pattern")),
            } for key, row in datasets.items()
        ],
        {
            "family": "L2_sobol_component_pair_edge",
            "semantics": "FULL_TIMING",
            "n_candidates": len(l2_rows), "n_worker_artifacts": l2_workers,
            "rank_parity_rows": l2_parity,
            "path": str(l2_dir.relative_to(ROOT)),
        },
        {
            "family": "L3_repeated_component_pair_edge",
            "semantics": "FULL_TIMING",
            "n_candidates": len(l3_rows), "n_worker_artifacts": l3_workers,
            "rank_parity_rows": l3_parity,
            "path": str(l3_dir.relative_to(ROOT)),
        },
    ]
    _atomic_csv(output_root / "historical_artifact_inventory.csv", inventory)

    all_scored = [row["score"] for row in datasets.values()
                  if row["score"].get("status") == "OK"]
    ordinal_support = {
        key: {
            "pooled_multishaft_fraction": row["score"].get(
                "pooled_multishaft_fraction"
            ),
            "pooled_scl_recruitment_fraction": row["score"].get(
                "pooled_scl_recruitment_fraction"
            ),
            "frozen_ood_fraction": row["score"].get("frozen_ood_fraction"),
            "patient_mode_status": row["score"].get("patient_mode_status"),
        }
        for key, row in datasets.items()
        if row["score"].get("status") == "OK"
    }
    l2_best = min((row for row in l2_rows if row.get("median_weak_mode_score") is not None),
                  key=lambda row: row["median_weak_mode_score"])
    l3_best = min((row for row in l3_rows if row.get("median_weak_mode_score") is not None),
                  key=lambda row: row["median_weak_mode_score"])
    summary = {
        "status": "SA4_HISTORICAL_RESCORING_COMPLETE",
        "scientific_verdict": (
            "FROZEN_LEARNED_NODE_FIELDS_HAVE_ZERO_SCL_SUPPORT; "
            "RIGID_CONTROLS_HAVE_SPARSE_SCL_SUPPORT; "
            "EDGE_NULL_MULTISHAFT_EVENTS_ARE_PATIENT_MODE_NOT_EVALUABLE; "
            "OLD_OBJECTIVE_FIELD_SELECTION_MISS_NOT_TESTABLE"
        ),
        "development_boundary": "patient training target only; no patient held-out score read",
        "mode_factorization": (
            "old A/B direction labels primary; shaft extent remains continuous within mode"
        ),
        "scoring_semantics": {
            "FULL_TIMING": "retained excess envelopes, original threshold re-extraction, rank parity required",
            "ORDINAL_COMPATIBLE": "retained ranks, separate ordinal patient embedding/floors",
            "NOT_RESCORABLE": "no per-event fixed-contact identity; no reconstruction from aggregate curves",
            "cross_semantics_absolute_score_comparison": "FORBIDDEN",
        },
        "fixed_events_per_mode": FIXED_EVENTS_PER_MODE,
        "inventory": inventory,
        "datasets": datasets,
        "l2": {
            "n_candidates": len(l2_rows), "n_workers": l2_workers,
            "rank_parity_rows": l2_parity,
            "old_vs_new_rank_association": _association(l2_rows),
            "best_new_score_candidate": l2_best,
        },
        "l3": {
            "n_candidates": len(l3_rows), "n_workers": l3_workers,
            "rank_parity_rows": l3_parity,
            "old_vs_new_rank_association": _association(l3_rows),
            "best_new_score_candidate": l3_best,
        },
        "field_selection_miss": {
            "status": "NOT_TESTABLE",
            "reason": (
                f"all {n_fit_candidates} rev8.1 fit-history candidates lack per-event "
                "fixed-contact values; retained controls do not establish a shaft-aware "
                "field-candidate ranking"
            ),
        },
        "historical_multishaft_support": {
            "ordinal_datasets": ordinal_support,
            "any_ordinal_dataset_positive": bool(any(
                score.get("pooled_multishaft_fraction", 0.0) > 0.0 for score in all_scored
            )),
            "any_l2_candidate_positive": bool(any(
                row.get("median_pooled_multishaft_fraction", 0.0) > 0.0 for row in l2_rows
            )),
            "any_l3_candidate_positive": bool(any(
                row.get("median_pooled_multishaft_fraction", 0.0) > 0.0 for row in l3_rows
            )),
            "interpretation": (
                "rev8.1 final, Node, and Node+Edge have zero SCL support; hand "
                "dual-core and Stage 2 filament have sparse support; Null and Edge "
                "have high SCL recruitment but OOD>0.5 and cannot define patient modes"
            ),
        },
        "inputs": {
            "target": {"path": str(target_path.relative_to(ROOT)), "sha256": _sha256(target_path)},
            "floors": {"path": str(floors_path.relative_to(ROOT)), "sha256": _sha256(floors_path)},
            "factorization": {"path": str(factor_path.relative_to(ROOT)), "sha256": _sha256(factor_path)},
        },
        "provenance": _runtime_provenance(config_path, target_path),
    }
    summary_path = output_root / "historical_artifact_rescore_summary.json"
    _atomic_json(summary_path, summary)
    stem = _plot(summary, l2_rows, l3_rows, output_root)
    print(summary["status"])
    print(summary["scientific_verdict"])
    print("L2 best", l2_best["candidate_id"], l2_best["median_weak_mode_score"])
    print("L3 best", l3_best["candidate_id"], l3_best["median_weak_mode_score"])
    print("multishaft", summary["historical_multishaft_support"])
    print(f"wrote {summary_path} and {stem}.png/.pdf")


if __name__ == "__main__":
    main()
