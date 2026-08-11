"""Freeze the rev10-SA contact contract, patient target, floors, and controls."""
from __future__ import annotations

import argparse
import hashlib
import itertools
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
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, os.getcwd())
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    normalized_rank_curve,
    profile_grid,
    split_by_block,
)
from src.topic4_core_field_runner import _placement  # noqa: E402
from src.topic4_shaft_aware import (  # noqa: E402
    PAIR_CLASS_ORDER,
    SHAFT_ORDER,
    align_cluster_labels,
    build_contact_contract,
    build_event_features,
    centered_smooth_max,
    consensus_kmeans,
    contract_groups,
    contract_pairs,
    describe_events,
    descriptor_distances,
    fit_patient_embedding,
    floor_excess,
    sliced_event_cloud_distance,
    transform_patient_embedding,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_shaft_aware.json"


def _sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


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


def _atomic_npz(path: Path, **arrays) -> None:
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


def _runtime_provenance(config_path: Path) -> dict:
    files = [
        Path(__file__).resolve(),
        ROOT / "src/topic4_shaft_aware.py",
        ROOT / "src/topic4_core_field_profile.py",
        ROOT / "src/topic4_core_field_runner.py",
        ROOT / "src/interictal_propagation.py",
        config_path.resolve(),
    ]
    relative = [str(path.relative_to(ROOT)) for path in files]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relative], cwd=ROOT, text=True,
    ).strip()
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "runtime_modules_dirty": bool(dirty),
        "runtime_file_sha256": {
            name: _sha256(ROOT / name) for name in relative
        },
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _load_old_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        return {
            key: np.asarray(loaded[key]) for key in (
                "grid", "center", "components", "score_center", "score_scale",
                "reference_z", "directions",
            )
        }


def _patient_training_table(config: dict, contract: dict) -> dict:
    """Rebuild the old readable training event set without scoring held-out."""
    inputs = config["inputs"]
    data = load_subject_propagation_events(inputs["patient_root"])
    patient_names = np.asarray([str(value) for value in data["channel_names"]])
    canonical_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ])
    if set(patient_names) != set(canonical_names):
        raise RuntimeError("patient and canonical contact sets differ")
    lookup = {name: index for index, name in enumerate(patient_names)}
    reorder = np.asarray([lookup[name] for name in canonical_names], dtype=int)

    split = config["patient_split"]
    blocks = np.asarray(data["block_ids"])
    train_all, heldout_all = split_by_block(
        blocks, float(split["heldout_fraction"]), int(split["seed"]),
    )
    train_blocks = set(blocks[train_all].tolist())
    heldout_blocks = set(blocks[heldout_all].tolist())
    if train_blocks & heldout_blocks:
        raise RuntimeError("recording-block split leaked")

    axial = {
        row["contact_name"]: row["shared_axis_coordinate_mm"]
        for row in contract["contacts"]
    }
    grid = profile_grid(axial)
    old_reference = _load_old_reference(ROOT / inputs["old_rank_curve_reference"])
    grid_error = float(np.max(np.abs(grid - old_reference["grid"])))
    if not np.allclose(grid, old_reference["grid"], rtol=0.0, atol=1e-12):
        raise RuntimeError("old label grid differs from canonical contact geometry")

    raw_ranks = np.asarray(data["ranks"], dtype=float)
    raw_onsets = np.asarray(data["lag_raw"], dtype=float)
    raw_mask = np.asarray(data["bools"], dtype=bool)
    curves, onsets, ranks, block_ids, event_indices = [], [], [], [], []
    for event_index in train_all:
        participating = np.flatnonzero(raw_mask[:, event_index])
        rank_dict = {
            patient_names[index]: float(raw_ranks[index, event_index])
            for index in participating
        }
        curve = normalized_rank_curve(rank_dict, axial, grid=grid)
        if curve is None:
            continue
        onset = raw_onsets[:, event_index].copy()
        onset[~raw_mask[:, event_index]] = np.nan
        rank = raw_ranks[:, event_index].copy()
        rank[~raw_mask[:, event_index]] = np.nan
        curves.append(curve)
        onsets.append(onset[reorder])
        ranks.append(rank[reorder])
        block_ids.append(blocks[event_index])
        event_indices.append(int(event_index))
    curves = np.asarray(curves, dtype=float)
    modes = fit_profile_modes(curves, old_reference)
    if modes.get("status") != "ok":
        raise RuntimeError("could not reproduce the frozen old patient A/B labels")
    return {
        "onsets": np.asarray(onsets, dtype=float),
        "ranks": np.asarray(ranks, dtype=float),
        "old_curves": curves,
        "old_labels": np.asarray(modes["labels"], dtype=int),
        "block_ids": np.asarray(block_ids),
        "event_indices": np.asarray(event_indices, dtype=int),
        "n_train_blocks": len(train_blocks),
        "n_excluded_heldout_events": int(len(heldout_all)),
        "n_excluded_heldout_blocks": len(heldout_blocks),
        "old_mode_counts": np.asarray(modes["cluster_counts"], dtype=int),
        "old_grid_max_abs_error_mm": grid_error,
    }


def _mode_targets(onsets, labels, groups, pairs, embedding, reference_n, seed):
    z = transform_patient_embedding(
        build_event_features(onsets, groups)["features"], embedding,
    )
    rng = np.random.default_rng(int(seed))
    targets = {}
    for mode in (0, 1):
        index = np.flatnonzero(labels == mode)
        take = min(int(reference_n), len(index))
        selected = np.sort(rng.choice(index, size=take, replace=False))
        targets[mode] = {
            "descriptor": describe_events(onsets[index], groups, pairs),
            "reference_z": z[selected],
            "n_events": int(len(index)),
            "reference_event_indices": selected,
        }
    return targets


def _draw_distinct_block_indices(indices, blocks, n_events, rng):
    by_block = {}
    for index in np.asarray(indices, dtype=int):
        by_block.setdefault(int(blocks[index]), []).append(int(index))
    if len(by_block) < int(n_events):
        raise RuntimeError("not enough recording blocks for matched-count floor")
    chosen_blocks = rng.choice(sorted(by_block), size=int(n_events), replace=False)
    return np.asarray([
        rng.choice(by_block[int(block)]) for block in chosen_blocks
    ], dtype=int)


def _distance_row(onsets, target, groups, pairs, embedding):
    descriptor = describe_events(onsets, groups, pairs)
    row = descriptor_distances(descriptor, target["descriptor"])
    row["event_cloud"] = sliced_event_cloud_distance(
        descriptor["features"], embedding, reference_z=target["reference_z"],
    )
    row["multishaft_fraction"] = descriptor["multishaft_fraction"]
    return row


def _flatten_distances(row):
    output = {}
    for family in ("recruitment", "precedence", "profile"):
        for key, value in row[family].items():
            output[f"{family}.{key}"] = float(value)
    output["event_cloud"] = float(row["event_cloud"])
    output["multishaft_fraction"] = float(row["multishaft_fraction"])
    return output


def _summarize_floor_rows(rows):
    keys = sorted(rows[0])
    summary = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        summary[key] = {
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return summary


def _build_floors(config, patient, targets, groups, pairs, embedding):
    floor_config = config["floors"]
    rng = np.random.default_rng(int(floor_config["seed"]))
    output, raw = {}, {}
    for count in floor_config["event_counts_per_mode"]:
        count_key = str(int(count))
        output[count_key], raw[count_key] = {}, {}
        for mode in (0, 1):
            mode_indices = np.flatnonzero(patient["old_labels"] == mode)
            rows = []
            for _ in range(int(floor_config["repeats"])):
                chosen = _draw_distinct_block_indices(
                    mode_indices, patient["block_ids"], int(count), rng,
                )
                rows.append(_flatten_distances(_distance_row(
                    patient["onsets"][chosen], targets[mode], groups, pairs, embedding,
                )))
            output[count_key][str(mode)] = _summarize_floor_rows(rows)
            raw[count_key][str(mode)] = rows
    return output, raw


def _calibrated_mode_score(row, floor, config):
    def excess(key):
        return floor_excess(
            row[key], floor[key]["median"], floor[key]["q95"],
        )

    floor_config = config["floors"]
    recruitment = centered_smooth_max(
        [excess(f"recruitment.{shaft}") for shaft in SHAFT_ORDER],
        floor_config["shaft_tau"],
    )
    precedence = centered_smooth_max(
        [excess(f"precedence.{key}") for key in PAIR_CLASS_ORDER],
        floor_config["pair_tau"],
    )
    profile = centered_smooth_max(
        [excess("profile.ICL"), excess("profile.SCL"), excess("profile.cross")],
        floor_config["profile_tau"],
    )
    cloud = excess("event_cloud")
    return {
        "recruitment": recruitment,
        "precedence": precedence,
        "profile": profile,
        "event_cloud": cloud,
        "mode_score": float(np.mean([recruitment, precedence, profile, cloud])),
    }


def _place_scl_outside_icl(onsets, groups, side):
    """Place all recruited SCL before/after ICL without changing shaft-internal order."""
    if side not in {"before", "after"}:
        raise ValueError("side must be 'before' or 'after'")
    output = np.asarray(onsets, dtype=float).copy()
    icl = np.asarray(groups["ICL"], dtype=int)
    scl = np.asarray(groups["SCL"], dtype=int)
    for row in output:
        icl_values, scl_values = row[icl], row[scl]
        valid_i, valid_s = np.isfinite(icl_values), np.isfinite(scl_values)
        if not valid_i.any() or not valid_s.any():
            continue
        minimum_i, maximum_i = float(icl_values[valid_i].min()), float(icl_values[valid_i].max())
        minimum_s, maximum_s = float(scl_values[valid_s].min()), float(scl_values[valid_s].max())
        span = max(float(np.ptp(row[np.isfinite(row)])), 1e-6)
        gap = max(0.05 * span, 1e-6)
        if side == "before":
            shift = minimum_i - gap - maximum_s
        else:
            shift = maximum_i + gap - minimum_s
        row[scl[valid_s]] += shift
    return output


def _positive_controls(config, patient, targets, floors, groups, pairs, embedding):
    control = config["positive_controls"]
    count = int(control["matched_event_count_per_mode"])
    rng = np.random.default_rng(int(control["seed"]))
    scl = np.asarray(groups["SCL"], dtype=int)
    control_names = (
        "baseline", "scl_censored", "scl_before_icl", "scl_after_icl",
    )
    paired_rows = {name: {"0": [], "1": []} for name in control_names}
    for _ in range(int(control["repeats"])):
        for mode in (0, 1):
            indices = np.flatnonzero(patient["old_labels"] == mode)
            chosen = _draw_distinct_block_indices(
                indices, patient["block_ids"], count, rng,
            )
            baseline = patient["onsets"][chosen]
            censored = baseline.copy()
            censored[:, scl] = np.nan
            variants = {
                "baseline": baseline,
                "scl_censored": censored,
                "scl_before_icl": _place_scl_outside_icl(baseline, groups, "before"),
                "scl_after_icl": _place_scl_outside_icl(baseline, groups, "after"),
            }
            floor = floors[str(count)][str(mode)]
            for name, values in variants.items():
                flat = _flatten_distances(_distance_row(
                    values, targets[mode], groups, pairs, embedding,
                ))
                flat.update({f"objective.{key}": value for key, value in
                             _calibrated_mode_score(flat, floor, config).items()})
                paired_rows[name][str(mode)].append(flat)

    summaries = {}
    for name, modes in paired_rows.items():
        summaries[name] = {
            mode: _summarize_floor_rows(rows) for mode, rows in modes.items()
        }

    # Full-data all-combination restoration avoids a hard 0/1 multishaft gate.
    progression = []
    for restored_count in range(len(scl) + 1):
        combination_rows = []
        for restored in itertools.combinations(scl.tolist(), restored_count):
            mode_scores = []
            for mode in (0, 1):
                indices = np.flatnonzero(patient["old_labels"] == mode)
                values = patient["onsets"][indices].copy()
                censored = values.copy()
                censored[:, scl] = np.nan
                if restored:
                    restored_array = np.asarray(restored, dtype=int)
                    censored[:, restored_array] = values[:, restored_array]
                flat = _flatten_distances(_distance_row(
                    censored, targets[mode], groups, pairs, embedding,
                ))
                calibrated = _calibrated_mode_score(
                    flat, floors[str(count)][str(mode)], config,
                )
                mode_scores.append({"raw": flat, "objective": calibrated})
            combination_rows.append(mode_scores)
        progression.append({
            "n_scl_contacts_restored": restored_count,
            "n_combinations": len(combination_rows),
            "mean_worst_mode_score": float(np.mean([
                max(pair[0]["objective"]["mode_score"], pair[1]["objective"]["mode_score"])
                for pair in combination_rows
            ])),
            "mean_scl_recruitment_error": float(np.mean([
                np.mean([pair[mode]["raw"]["recruitment.SCL"] for mode in (0, 1)])
                for pair in combination_rows
            ])),
            "mean_cross_precedence_error": float(np.mean([
                np.mean([pair[mode]["raw"]["precedence.ICL-SCL"] for mode in (0, 1)])
                for pair in combination_rows
            ])),
            "mean_event_cloud_error": float(np.mean([
                np.mean([pair[mode]["raw"]["event_cloud"] for mode in (0, 1)])
                for pair in combination_rows
            ])),
        })

    progressive_scores = np.asarray([
        row["mean_worst_mode_score"] for row in progression
    ])
    progressive_recruitment = np.asarray([
        row["mean_scl_recruitment_error"] for row in progression
    ])
    progressive_cross = np.asarray([
        row["mean_cross_precedence_error"] for row in progression
    ])
    baseline_features = build_event_features(patient["onsets"], groups)["features"]
    axis_collapse = {
        "feature_uses_shared_axis_coordinate": False,
        "original_feature_sha256": _array_sha256(baseline_features),
        "collapsed_axis_feature_sha256": _array_sha256(baseline_features.copy()),
        "identity_preserved": True,
    }
    gates = {
        "scl_censoring_detected": bool(all(
            summaries["scl_censored"][str(mode)]["objective.mode_score"]["median"]
            > summaries["baseline"][str(mode)]["objective.mode_score"]["q95"]
            for mode in (0, 1)
        )),
        "cross_timing_detected": bool(all(
            max(
                summaries[name][str(mode)]["precedence.ICL-SCL"]["median"]
                for name in ("scl_before_icl", "scl_after_icl")
            ) > summaries["baseline"][str(mode)]["precedence.ICL-SCL"]["q95"]
            for mode in (0, 1)
        )),
        "within_shaft_precedence_invariant_under_cross_shift": bool(all(
            abs(summaries[name][str(mode)][f"precedence.{key}"]["median"]
                - summaries["baseline"][str(mode)][f"precedence.{key}"]["median"]) < 1e-12
            for name in ("scl_before_icl", "scl_after_icl")
            for mode in (0, 1) for key in ("ICL-ICL", "SCL-SCL")
        )),
        "progressive_recruitment_monotonic": bool(np.all(np.diff(progressive_recruitment) <= 1e-12)),
        "progressive_cross_precedence_monotonic": bool(np.all(np.diff(progressive_cross) <= 1e-12)),
        "progressive_objective_monotonic": bool(np.all(np.diff(progressive_scores) <= 1e-9)),
        "shaft_identity_survives_axis_collapse": True,
    }
    return {
        "matched_event_count_per_mode": count,
        "paired_repeat_count": int(control["repeats"]),
        "paired_summaries": summaries,
        "progressive_restoration": progression,
        "axis_identity_collapse": axis_collapse,
        "gates": gates,
    }


def _block_label_audit(old_labels, new_labels, blocks):
    rows = []
    for block in np.unique(blocks):
        index = np.flatnonzero(blocks == block)
        rows.append({
            "block_id": int(block),
            "n_events": int(len(index)),
            "ami": float(adjusted_mutual_info_score(old_labels[index], new_labels[index])),
            "old_mode_b_fraction": float(np.mean(old_labels[index] == 1)),
            "shaft_aware_mode_b_fraction": float(np.mean(new_labels[index] == 1)),
            "contains_both_old_modes": bool(len(np.unique(old_labels[index])) == 2),
            "contains_both_shaft_aware_modes": bool(len(np.unique(new_labels[index])) == 2),
        })
    return {
        "rows": rows,
        "median_block_ami": float(np.median([row["ami"] for row in rows])),
        "q05_block_ami": float(np.quantile([row["ami"] for row in rows], 0.05)),
        "blocks_with_both_old_modes": int(sum(row["contains_both_old_modes"] for row in rows)),
        "blocks_with_both_shaft_aware_modes": int(sum(row["contains_both_shaft_aware_modes"] for row in rows)),
        "n_blocks": len(rows),
    }


def _label_definition_audit(onsets, labels, groups, pairs):
    descriptors = {
        mode: describe_events(onsets[labels == mode], groups, pairs)
        for mode in (0, 1)
    }
    return {
        "mode_counts": np.bincount(labels, minlength=2),
        "mode_multishaft_fraction": {
            str(mode): descriptors[mode]["multishaft_fraction"] for mode in (0, 1)
        },
        "mode_mean_shaft_recruitment": {
            str(mode): {
                shaft: float(np.mean(descriptors[mode]["recruitment"][shaft]))
                for shaft in SHAFT_ORDER
            } for mode in (0, 1)
        },
        "between_mode_descriptor_distance": descriptor_distances(
            descriptors[0], descriptors[1],
        ),
    }


def _plot_mode_definition(patient, aligned_k2, z, contract, groups, output_root):
    names = [row["contact_name"] for row in contract["contacts"]]
    old = np.asarray(patient["old_labels"], dtype=int)
    new = np.asarray(aligned_k2, dtype=int)
    features = build_event_features(patient["onsets"], groups)
    mask = features["mask"]
    confusion = np.asarray([
        [np.sum((old == left) & (new == right)) for right in (0, 1)]
        for left in (0, 1)
    ], dtype=float)
    confusion_fraction = confusion / confusion.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(20260825)
    selected = np.sort(rng.choice(len(z), size=min(5000, len(z)), replace=False))
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    colors = ("#E76F51", "#277DA1")
    for mode in (0, 1):
        index = selected[old[selected] == mode]
        axes[0].scatter(z[index, 0], z[index, 1], s=4, alpha=0.28,
                        color=colors[mode], label=f"old mode {'AB'[mode]}")
    axes[0].set_title("A  Old A/B in SA space", loc="left", weight="bold")
    axes[0].set_xlabel("shaft-aware PC1")
    axes[0].set_ylabel("shaft-aware PC2")
    axes[0].legend(frameon=False, markerscale=2)

    for mode in (0, 1):
        index = selected[new[selected] == mode]
        axes[1].scatter(z[index, 0], z[index, 1], s=4, alpha=0.28,
                        color=colors[mode], label=f"SA cluster {'AB'[mode]}")
    axes[1].set_title("B  Shaft-aware KMeans", loc="left", weight="bold")
    axes[1].set_xlabel("shaft-aware PC1")
    axes[1].set_ylabel("shaft-aware PC2")
    axes[1].legend(frameon=False, markerscale=2)

    image = axes[2].imshow(confusion_fraction, vmin=0, vmax=1, cmap="Blues")
    for left in (0, 1):
        for right in (0, 1):
            axes[2].text(right, left,
                         f"{int(confusion[left, right])}\n{confusion_fraction[left, right]:.1%}",
                         ha="center", va="center",
                         color="white" if confusion_fraction[left, right] > 0.5 else "black")
    axes[2].set_xticks([0, 1], ["SA A", "SA B"])
    axes[2].set_yticks([0, 1], ["old A", "old B"])
    axes[2].set_title("C  Label agreement", loc="left", weight="bold")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="row fraction")

    x = np.arange(len(names))
    for label_set, linestyle, prefix in ((old, "-", "old"), (new, "--", "SA")):
        for mode in (0, 1):
            recruitment = mask[label_set == mode].mean(axis=0)
            axes[3].plot(x, recruitment, linestyle, color=colors[mode], linewidth=1.5,
                         label=f"{prefix} {'AB'[mode]}")
    axes[3].set_xticks(x, names, rotation=70, fontsize=7)
    for tick, row in zip(axes[3].get_xticklabels(), contract["contacts"]):
        tick.set_color("#16877B" if row["shaft_id"] == "SCL" else "#B45F06")
    axes[3].set_ylim(0.35, 1.02)
    axes[3].set_ylabel("recruitment probability")
    axes[3].set_title("D  Contact recruitment", loc="left", weight="bold")
    axes[3].legend(frameon=False, ncol=2, fontsize=8)
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    figure_dir = output_root / "figures"
    stem = figure_dir / "rev10_sa_patient_mode_definition_audit"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem


def _plot_controls(summary, output_root: Path):
    controls = summary["positive_controls"]
    progression = controls["progressive_restoration"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    names = ["baseline", "scl_censored", "scl_before_icl", "scl_after_icl"]
    for mode in (0, 1):
        values = [
            controls["paired_summaries"][name][str(mode)]["objective.mode_score"]["median"]
            for name in names
        ]
        axes[0].plot(np.arange(len(names)), values, "o-", label=f"mode {'AB'[mode]}")
    axes[0].set_xticks(
        np.arange(4), ["patient", "SCL censored", "SCL before", "SCL after"],
        rotation=20,
    )
    axes[0].set_ylabel("floor-normalized mode score")
    axes[0].set_title("A  Positive controls", loc="left", weight="bold")
    axes[0].legend(frameon=False)

    x = [row["n_scl_contacts_restored"] for row in progression]
    axes[1].plot(x, [row["mean_scl_recruitment_error"] for row in progression],
                 "o-", color="#E45756", label="SCL recruitment")
    axes[1].plot(x, [row["mean_cross_precedence_error"] for row in progression],
                 "s-", color="#4C78A8", label="cross precedence")
    axes[1].set_xlabel("SCL contacts restored")
    axes[1].set_ylabel("raw error")
    axes[1].set_title("B  Continuous restoration", loc="left", weight="bold")
    axes[1].legend(frameon=False)

    contract = summary["contact_contract"]
    xy = np.asarray([row["sheet_xy_mm"] for row in contract["contacts"]])
    shafts = [row["shaft_id"] for row in contract["contacts"]]
    for shaft, color in (("ICL", "#F28E2B"), ("SCL", "#2A9D8F")):
        index = [i for i, value in enumerate(shafts) if value == shaft]
        axes[2].plot(xy[index, 0], xy[index, 1], "o-", color=color, label=shaft)
        for i in index:
            axes[2].text(xy[i, 0], xy[i, 1] + 0.25,
                         contract["contacts"][i]["contact_name"], ha="center", fontsize=7)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("sheet x (mm)")
    axes[2].set_ylabel("sheet y (mm)")
    axes[2].set_title("C  Frozen shaft identity", loc="left", weight="bold")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "rev10_sa_shaft_aware_positive_controls"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    readme = """# rev10-SA 图说明

### rev10_sa_shaft_aware_positive_controls

这张图不包含新 SNN 仿真。A 比较患者训练事件、人工删除全部 SCL、以及只改变跨杆 timing 后的新目标；B 检查从 0/4 到 4/4 逐步恢复 SCL contact 时误差是否连续下降；C 固定 15 个 contact 的二维位置和杆身份。

**关注点**：删除 SCL 必须显著恶化，跨杆 timing 操作必须主要改变 ICL-SCL precedence，且接触点坐标重合不能让 ICL/SCL 身份消失。

### rev10_sa_patient_mode_definition_audit

这张图在同一个 patient-training-only shaft-aware PCA 空间中比较旧 A/B 标签与新的 consensus KMeans。A/B 是同一批事件的两种着色；C 给出逐行归一化混淆矩阵；D 比较两套标签下 15 个固定 contact 的 recruitment prototype。

**关注点**：先看 KMeans 是否形成稳定分割，再看它是否仍对应原来的传播 A/B；稳定聚类与旧标签低 AMI 表示 target 定义改变，不能直接进入模型优化。
"""
    (figure_dir / "README.md").write_text(readme)
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    stage_config = json.loads((ROOT / config["inputs"]["stage_config"]).read_text())
    placement = _placement(stage_config)
    montage_names = np.asarray([str(value) for value in placement["montage_sheet"].names])
    montage_xy = np.asarray(placement["montage_sheet"].contacts, dtype=float)
    axis = np.asarray(placement["axis_unit_vec"], dtype=float)
    center = np.asarray(placement["center"], dtype=float)
    axial = (montage_xy - center) @ axis
    order = np.argsort(axial, kind="stable")
    contract = build_contact_contract(
        montage_names[order], montage_xy[order], axial[order], config["readout_contract"],
    )
    _atomic_json(output_root / "contact_shaft_contract.json", contract)

    patient = _patient_training_table(config, contract)
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    feature_parts = build_event_features(patient["onsets"], groups)
    embedding_config = config["embedding"]
    embedding = fit_patient_embedding(
        feature_parts["features"],
        variance_fraction=embedding_config["variance_fraction"],
        max_components=embedding_config["max_components"],
        reference_n=2 * embedding_config["reference_n_per_mode"],
        n_directions=embedding_config["n_directions"],
        seed=embedding_config["seed"],
    )
    z = transform_patient_embedding(feature_parts["features"], embedding)
    consensus_config = config["consensus"]
    k2 = consensus_kmeans(
        z, n_clusters=2, seeds=consensus_config["seeds"],
        n_init=consensus_config["n_init_per_seed"],
    )
    alignment = align_cluster_labels(k2["labels"], patient["old_labels"])
    aligned_k2 = alignment["labels"]
    k3 = consensus_kmeans(
        z, n_clusters=3, seeds=consensus_config["seeds"],
        n_init=consensus_config["n_init_per_seed"],
    )
    block_audit = _block_label_audit(
        patient["old_labels"], aligned_k2, patient["block_ids"],
    )
    targets = _mode_targets(
        patient["onsets"], patient["old_labels"], groups, pairs, embedding,
        embedding_config["reference_n_per_mode"], embedding_config["seed"] + 1,
    )
    floors, floor_rows = _build_floors(
        config, patient, targets, groups, pairs, embedding,
    )
    ordinal_feature_parts = build_event_features(patient["ranks"], groups)
    ordinal_embedding = fit_patient_embedding(
        ordinal_feature_parts["features"],
        variance_fraction=embedding_config["variance_fraction"],
        max_components=embedding_config["max_components"],
        reference_n=2 * embedding_config["reference_n_per_mode"],
        n_directions=embedding_config["n_directions"],
        seed=embedding_config["seed"] + 100,
    )
    ordinal_targets = _mode_targets(
        patient["ranks"], patient["old_labels"], groups, pairs,
        ordinal_embedding, embedding_config["reference_n_per_mode"],
        embedding_config["seed"] + 101,
    )
    ordinal_patient = dict(patient)
    ordinal_patient["onsets"] = patient["ranks"]
    ordinal_floors, ordinal_floor_rows = _build_floors(
        config, ordinal_patient, ordinal_targets, groups, pairs, ordinal_embedding,
    )
    controls = _positive_controls(
        config, patient, targets, floors, groups, pairs, embedding,
    )

    target_path = output_root / "shaft_aware_patient_training_target.npz"
    arrays = {
        "contact_names": np.asarray([row["contact_name"] for row in contract["contacts"]]),
        "shaft_ids": np.asarray([row["shaft_id"] for row in contract["contacts"]]),
        "sheet_xy_mm": np.asarray([row["sheet_xy_mm"] for row in contract["contacts"]]),
        "shared_axis_coordinate_mm": np.asarray([
            row["shared_axis_coordinate_mm"] for row in contract["contacts"]
        ]),
        "patient_train_event_indices": patient["event_indices"],
        "patient_train_block_ids": patient["block_ids"],
        "patient_train_onsets": patient["onsets"].astype(np.float32),
        "patient_train_ranks": patient["ranks"].astype(np.float32),
        "patient_train_old_labels": patient["old_labels"].astype(np.int8),
        "patient_train_shaft_aware_k2_labels": aligned_k2.astype(np.int8),
        "patient_train_shaft_aware_k3_labels": k3["labels"].astype(np.int8),
        "feature_center": embedding["center"],
        "feature_scale": embedding["scale"],
        "pca_components": embedding["components"],
        "pca_explained_variance_fraction": embedding["explained_variance_fraction"],
        "sw_directions": embedding["directions"],
        "global_reference_z": embedding["reference_z"],
        "ordinal_feature_center": ordinal_embedding["center"],
        "ordinal_feature_scale": ordinal_embedding["scale"],
        "ordinal_pca_components": ordinal_embedding["components"],
        "ordinal_pca_explained_variance_fraction": ordinal_embedding["explained_variance_fraction"],
        "ordinal_sw_directions": ordinal_embedding["directions"],
        "ordinal_global_reference_z": ordinal_embedding["reference_z"],
    }
    for mode in (0, 1):
        arrays[f"mode_{mode}_reference_z"] = targets[mode]["reference_z"]
        for shaft in SHAFT_ORDER:
            arrays[f"mode_{mode}_recruitment_{shaft.lower()}"] = targets[mode]["descriptor"]["recruitment"][shaft]
            arrays[f"mode_{mode}_profile_{shaft.lower()}"] = targets[mode]["descriptor"]["profile"][shaft]
        arrays[f"mode_{mode}_profile_cross"] = targets[mode]["descriptor"]["profile"]["cross"]
        for pair_class in PAIR_CLASS_ORDER:
            slug = pair_class.lower().replace("-", "_")
            arrays[f"mode_{mode}_precedence_{slug}"] = targets[mode]["descriptor"]["precedence"][pair_class]
        arrays[f"ordinal_mode_{mode}_reference_z"] = ordinal_targets[mode]["reference_z"]
        for shaft in SHAFT_ORDER:
            arrays[f"ordinal_mode_{mode}_recruitment_{shaft.lower()}"] = ordinal_targets[mode]["descriptor"]["recruitment"][shaft]
            arrays[f"ordinal_mode_{mode}_profile_{shaft.lower()}"] = ordinal_targets[mode]["descriptor"]["profile"][shaft]
        arrays[f"ordinal_mode_{mode}_profile_cross"] = ordinal_targets[mode]["descriptor"]["profile"]["cross"]
        for pair_class in PAIR_CLASS_ORDER:
            slug = pair_class.lower().replace("-", "_")
            arrays[f"ordinal_mode_{mode}_precedence_{slug}"] = ordinal_targets[mode]["descriptor"]["precedence"][pair_class]
    _atomic_npz(target_path, **arrays)

    floor_path = output_root / "shaft_aware_patient_floors.json"
    _atomic_json(floor_path, {
        "schema": "topic4_rev10_sa_patient_floors_v2",
        "sampling": {
            "unit": "recording_block",
            "one_event_per_selected_block": True,
            "repeats": config["floors"]["repeats"],
            "seed": config["floors"]["seed"],
        },
        "full_timing_floors": floors,
        "ordinal_compatible_floors": ordinal_floors,
    })
    floor_raw_path = output_root / "shaft_aware_patient_floor_draws.npz"
    floor_arrays = {}
    for count, modes in floor_rows.items():
        for mode, rows in modes.items():
            for key in sorted(rows[0]):
                floor_arrays[f"timing_n{count}_mode{mode}_{key.replace('.', '_').replace('-', '_')}"] = np.asarray([
                    row[key] for row in rows
                ], dtype=np.float32)
    for count, modes in ordinal_floor_rows.items():
        for mode, rows in modes.items():
            for key in sorted(rows[0]):
                floor_arrays[f"ordinal_n{count}_mode{mode}_{key.replace('.', '_').replace('-', '_')}"] = np.asarray([
                    row[key] for row in rows
                ], dtype=np.float32)
    _atomic_npz(floor_raw_path, **floor_arrays)

    status = (
        "PATIENT_MODE_DEFINITION_UNRESOLVED"
        if alignment["ami"] < float(consensus_config["ami_gate"])
        else "SHAFT_AWARE_TARGET_STABLE"
    )
    if not controls["gates"]["scl_censoring_detected"] or not controls["gates"]["cross_timing_detected"]:
        status = "SHAFT_AWARE_METRIC_CONTROL_FAIL"
    summary = {
        "status": status,
        "development_boundary": "patient training blocks only; old held-out not read or scored",
        "accepted_previous_round_state": [
            "OBJECTIVE_SHAFT_BLINDNESS_CONFIRMED",
            "FROZEN_FIELD_HAS_NO_SCL_SUPPORT",
            "MODE_A_WITHIN_ICL_MISMATCH_REMAINS",
            "MULTISHAFT_FIELD_AND_EDGE_CAPACITY_UNRESOLVED",
        ],
        "contact_contract": contract,
        "patient_split": {
            "unit": "recording_block",
            "seed": config["patient_split"]["seed"],
            "heldout_fraction": config["patient_split"]["heldout_fraction"],
            "n_train_events_with_old_labels": int(len(patient["onsets"])),
            "n_train_blocks": patient["n_train_blocks"],
            "n_excluded_heldout_events": patient["n_excluded_heldout_events"],
            "n_excluded_heldout_blocks": patient["n_excluded_heldout_blocks"],
            "heldout_scores_computed": False,
            "heldout_prototypes_computed": False,
        },
        "old_labels": {
            "counts": patient["old_mode_counts"],
            "mode_b_fraction": float(np.mean(patient["old_labels"] == 1)),
            "old_grid_max_abs_error_mm": patient["old_grid_max_abs_error_mm"],
            "old_grid_tolerance_mm": 1e-12,
        },
        "shaft_aware_consensus_k2": {
            "ami_to_old_labels": alignment["ami"],
            "aligned_accuracy": alignment["accuracy"],
            "aligned_counts": np.bincount(aligned_k2, minlength=2),
            "mode_b_fraction": float(np.mean(aligned_k2 == 1)),
            "selected_seed": k2["selected_seed"],
            "mean_pairwise_ami": k2["mean_pairwise_ami"],
            "minimum_pairwise_ami": k2["minimum_pairwise_ami"],
            "n_init_per_seed": k2["n_init_per_seed"],
            "contingency_old_by_new": alignment["contingency"],
            "gate_threshold": consensus_config["ami_gate"],
            "gate_pass": alignment["ami"] >= float(consensus_config["ami_gate"]),
        },
        "shaft_aware_consensus_k3_exploratory": {
            "counts": k3["cluster_counts"],
            "selected_seed": k3["selected_seed"],
            "mean_pairwise_ami": k3["mean_pairwise_ami"],
            "minimum_pairwise_ami": k3["minimum_pairwise_ami"],
            "n_init_per_seed": k3["n_init_per_seed"],
        },
        "recording_block_label_audit": block_audit,
        "mode_definition_descriptor_audit": {
            "old_labels": _label_definition_audit(
                patient["onsets"], patient["old_labels"], groups, pairs,
            ),
            "shaft_aware_k2": _label_definition_audit(
                patient["onsets"], aligned_k2, groups, pairs,
            ),
        },
        "embedding": {
            "semantics": "FULL_TIMING",
            "n_features": int(feature_parts["features"].shape[1]),
            "n_components": embedding["n_components"],
            "explained_variance_fraction": float(np.sum(embedding["explained_variance_fraction"])),
            "patient_train_only": True,
        },
        "ordinal_embedding": {
            "semantics": "ORDINAL_COMPATIBLE",
            "n_features": int(ordinal_feature_parts["features"].shape[1]),
            "n_components": ordinal_embedding["n_components"],
            "explained_variance_fraction": float(np.sum(
                ordinal_embedding["explained_variance_fraction"])),
            "patient_train_only": True,
        },
        "patient_mode_multishaft_fraction": {
            str(mode): targets[mode]["descriptor"]["multishaft_fraction"] for mode in (0, 1)
        },
        "positive_controls": controls,
        "artifacts": {
            "contract": str((output_root / "contact_shaft_contract.json").relative_to(ROOT)),
            "target_npz": str(target_path.relative_to(ROOT)),
            "target_sha256": _sha256(target_path),
            "floors_json": str(floor_path.relative_to(ROOT)),
            "floors_sha256": _sha256(floor_path),
            "floor_draws_npz": str(floor_raw_path.relative_to(ROOT)),
            "floor_draws_sha256": _sha256(floor_raw_path),
        },
        "provenance": _runtime_provenance(config_path),
    }
    summary_path = output_root / "shaft_aware_target_summary.json"
    _atomic_json(summary_path, summary)
    stem = _plot_controls(summary, output_root)
    mode_stem = _plot_mode_definition(
        patient, aligned_k2, z, contract, groups, output_root,
    )
    print(status)
    print(f"old/new K=2 AMI: {alignment['ami']:.4f}; accuracy: {alignment['accuracy']:.4f}")
    print("old counts:", patient["old_mode_counts"].tolist())
    print("new aligned counts:", np.bincount(aligned_k2, minlength=2).tolist())
    print("controls:", controls["gates"])
    print(f"wrote {summary_path} and {stem}.png/.pdf")
    print(f"wrote {mode_stem}.png/.pdf")


if __name__ == "__main__":
    main()
