"""Aggregate V4 with factorized direction and all-event shaft objectives."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, os.getcwd())
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
    score_mode_conditioned_events,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import continuous_surface  # noqa: E402
from src.topic4_shaft_aware import (  # noqa: E402
    build_event_features,
    contract_groups,
    contract_pairs,
    transform_patient_embedding,
)
from src.topic4_shaft_aware_direction import (  # noqa: E402
    all_event_shaft_participation,
    assign_direction_modes,
    mode_conditioned_joint_support,
)
from src.topic4_spectral_field import uniform_sheet_grid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v4.json"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True,
        ))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    os.close(handle)
    try:
        with open(temporary, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _worker_complete(payload, npz_path, config_sha, commit):
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == "REV10SA_SPECTRAL_FIELD_WORKER_COMPLETE"
        and payload.get("config", {}).get("sha256") == config_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
    )


def _classifier_from_manifest(manifest):
    classifier = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], dtype=float)
    return classifier


def _kmeans_audit(z, direction_labels):
    z = np.asarray(z, dtype=float)
    if len(z) < 4 or np.unique(z, axis=0).shape[0] < 2:
        return {"status": "INSUFFICIENT", "n_events": int(len(z))}
    label_sets = [
        KMeans(n_clusters=2, n_init=20, random_state=20260812 + seed).fit_predict(z)
        for seed in range(12)
    ]
    pairwise = [
        adjusted_mutual_info_score(label_sets[left], label_sets[right])
        for left in range(len(label_sets)) for right in range(left + 1, len(label_sets))
    ]
    return {
        "status": "OK", "n_events": int(len(z)), "labels": label_sets[0],
        "cluster_counts": np.bincount(label_sets[0], minlength=2),
        "pairwise_seed_ami_median": float(np.median(pairwise)),
        "pairwise_seed_ami_min": float(np.min(pairwise)),
        "ami_with_direction_labels": float(
            adjusted_mutual_info_score(label_sets[0], direction_labels)
        ),
    }


def _objective(score6, score3, counts, participation, ood_fraction, config,
               mode_support=None):
    objective = config["search"]["objective"]
    required = int(objective["fixed_events_per_mode"])
    support_penalty = float(sum(
        max(0, required - int(counts[mode])) / required for mode in (0, 1)
    ))
    if score6["status"] == "OK":
        route_score = float(score6["weak_mode_score"])
        route_source = "n6"
    elif score3["status"] == "OK":
        route_score = float(score3["weak_mode_score"])
        route_source = "n3_fallback"
    else:
        route_score = 8.0
        route_source = "unsupported_penalty"
    target = float(objective["minimum_target_joint_fraction"])
    floor_q05 = float(objective["joint_floor_q05"])
    joint_excess = max(
        0.0, (target - participation["joint_fraction"])
        / max(target - floor_q05, 1e-12),
    )
    mode_required = int(objective.get(
        "minimum_joint_in_distribution_events_per_mode_for_objective", 0,
    ))
    mode_support_penalty = 0.0
    if mode_required > 0:
        if mode_support is None:
            raise ValueError("mode-conditioned joint support is required")
        mode_support_penalty = float(sum(
            max(0, mode_required - int(
                mode_support[name]["n_joint_in_distribution"]
            )) / mode_required
            for name in ("A", "B")
        ))
    score = (
        route_score
        + float(objective["joint_weight"]) * joint_excess
        + float(objective["direction_support_weight"]) * support_penalty
        + float(objective["ood_weight"]) * float(ood_fraction)
        + float(objective.get(
            "mode_conditioned_joint_support_weight", 0.0,
        )) * mode_support_penalty
    )
    return {
        "selection_score": float(score),
        "route_score": route_score,
        "route_score_source": route_source,
        "joint_excess": float(joint_excess),
        "direction_support_penalty": support_penalty,
        "mode_conditioned_joint_support_penalty": mode_support_penalty,
    }


def _pareto_flags(rows):
    values = np.asarray([
        [row["route_score"], row["joint_excess"],
         row["direction_support_penalty"], row["ood_fraction"]]
        for row in rows
    ], dtype=float)
    flags = []
    for index, row in enumerate(values):
        dominated = any(
            np.all(other <= row) and np.any(other < row)
            for other_index, other in enumerate(values) if other_index != index
        )
        flags.append(not dominated)
    return flags


def _selection_verdict(rows, config):
    """Keep a descriptive scalar minimum separate from a valid selection."""
    objective = config["search"]["objective"]
    minimum_joint = int(objective.get("minimum_joint_events_for_selection", 1))
    minimum_joint_seeds = int(
        objective.get("minimum_seeds_with_joint_for_selection", 1)
    )
    minimum_mode_joint = int(objective.get(
        "minimum_joint_in_distribution_events_per_mode_for_selection", 0,
    ))
    minimum_mode_joint_seeds = int(objective.get(
        "minimum_seeds_with_joint_in_distribution_per_mode_for_selection", 0,
    ))
    diagnostic = min(
        rows,
        key=lambda row: (
            row["n_runaway_networks"] > 0,
            row["selection_score"],
            row["candidate_id"],
        ),
    )
    eligible = [
        row for row in rows
        if (
            row["n_runaway_networks"] == 0
            and row["n_joint"] >= minimum_joint
            and int(row.get(
                "n_seeds_with_joint", 1 if row["n_joint"] > 0 else 0,
            )) >= minimum_joint_seeds
            and int(row.get(
                "weak_mode_joint_in_distribution_count", 0,
            )) >= minimum_mode_joint
            and int(row.get(
                "weak_mode_joint_in_distribution_seed_count", 0,
            )) >= minimum_mode_joint_seeds
        )
    ]
    labels = config.get("aggregation", {})
    if not eligible:
        return {
            "status": labels.get(
                "no_joint_status", "REV10SA_V4_NO_JOINT_SHAFT_CANDIDATE",
            ),
            "selected": None,
            "diagnostic": diagnostic,
            "minimum_joint_events_for_selection": minimum_joint,
            "minimum_seeds_with_joint_for_selection": minimum_joint_seeds,
            "minimum_joint_in_distribution_events_per_mode_for_selection": minimum_mode_joint,
            "minimum_seeds_with_joint_in_distribution_per_mode_for_selection": minimum_mode_joint_seeds,
        }
    selected = min(
        eligible,
        key=lambda row: (row["selection_score"], row["candidate_id"]),
    )
    return {
        "status": labels.get(
            "success_status", "REV10SA_V4_JOINT_SHAFT_CANDIDATE_FOUND",
        ),
        "selected": selected,
        "diagnostic": diagnostic,
        "minimum_joint_events_for_selection": minimum_joint,
        "minimum_seeds_with_joint_for_selection": minimum_joint_seeds,
        "minimum_joint_in_distribution_events_per_mode_for_selection": minimum_mode_joint,
        "minimum_seeds_with_joint_in_distribution_per_mode_for_selection": minimum_mode_joint_seeds,
    }


def _relative_field(candidate, positions):
    surface = continuous_surface(
        candidate["coefficients"], positions,
        n_basis=candidate["n_basis"], degree=candidate["degree"], L=20.0,
    )
    return np.exp(np.clip(surface - surface.max(), -30.0, 0.0))


def _plot_search(summary, manifest, config, output_root):
    rows = summary["candidate_rows"]
    candidates = {row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]}
    selected = summary["selected_candidate_id"]
    display = summary["display_candidate_id"]
    joint_best = max(
        rows,
        key=lambda row: (
            row["joint_fraction"], row["scl_participation_fraction"],
            -row["selection_score"],
        ),
    )
    reference = config.get("aggregation", {}).get(
        "plot_reference_candidate_id",
        manifest["candidate_set"]["candidates"][0]["candidate_id"],
    )
    map_ids = [reference, display, joint_best["candidate_id"]]
    joint_title = (
        "highest joint fraction" if joint_best["n_joint"] > 0
        else "highest SCL participation (no joint)"
    )
    titles = ["reference field", "scalar diagnostic", joint_title]
    grid = uniform_sheet_grid(120, L=20.0)
    axis = (np.arange(120) + 0.5) * 20.0 / 120
    fig = plt.figure(figsize=(15.8, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    for column, (candidate_id, title) in enumerate(zip(map_ids, titles)):
        ax = fig.add_subplot(gs[0, column])
        field = _relative_field(candidates[candidate_id], grid).reshape(120, 120)
        image = ax.contourf(axis, axis, field, levels=14, cmap="magma")
        ax.set_title(f"{chr(65 + column)}  {title}", loc="left", weight="bold")
        ax.set_xlabel("sheet x (mm)")
        if column == 0:
            ax.set_ylabel("sheet y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 0])
    role_styles = {
        "stage3_uniform_spline_warm": ("*", "#111111", "Stage 3 warm"),
        "uniform_sheet_allocation_refinement": ("o", "#4E79A7", "uniform probe"),
        "observation_free_smooth_random_residual": ("^", "#E15759", "random field"),
        "uniform_negative_control": ("s", "#7F7F7F", "uniform control"),
        "v3_spectral_to_spline_bridge": ("D", "#4E79A7", "V3 spline bridge"),
        "adaptive_training_anchor": ("D", "#111111", "training anchor"),
        "adaptive_latent_linear_interpolation": ("o", "#4E79A7", "latent interpolation"),
        "adaptive_density_mixture_interpolation": ("^", "#E15759", "density interpolation"),
        "selection_confirmation_adaptive_training_anchor": ("D", "#111111", "anchor confirmation"),
        "selection_confirmation_adaptive_latent_linear_interpolation": ("o", "#4E79A7", "latent confirmation"),
        "selection_confirmation_adaptive_density_mixture_interpolation": ("^", "#E15759", "density confirmation"),
        "final_confirmation_score_winner": ("^", "#E15759", "score winner"),
        "final_confirmation_joint_support": ("D", "#59A14F", "joint-support anchor"),
        "final_confirmation_stage3_reference": ("s", "#111111", "Stage 3 reference"),
        "mode_conditioned_density_boundary": ("o", "#B07AA1", "mode boundary"),
    }
    for role in sorted({row["role"] for row in rows}):
        marker, color, label = role_styles.get(role, ("o", "#7F7F7F", role))
        selected_rows = [row for row in rows if row["role"] == role]
        ax.scatter(
            [row["route_score"] for row in selected_rows],
            [row["joint_fraction"] for row in selected_rows],
            marker=marker, c=color, s=48, alpha=0.75, label=label,
        )
    shown_candidate = next(row for row in rows if row["candidate_id"] == display)
    ax.scatter(
        shown_candidate["route_score"], shown_candidate["joint_fraction"],
        marker="*" if selected is not None else "X", s=170,
        c="#59A14F" if selected is not None else "#D62728",
        edgecolor="black", zorder=5,
        label="eligible selection" if selected is not None else "diagnostic only",
    )
    ax.axhline(summary["joint_fraction_target"], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("direction weak-mode score (lower better)")
    ax.set_ylabel("all-event joint-shaft fraction")
    ax.set_title("D  Factorized objective plane", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    shown = sorted(rows, key=lambda row: row["selection_score"])[:10]
    x = np.arange(len(shown))
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x, [row["n_icl_only"] for row in shown], color="#4E79A7", label="ICL only")
    ax.bar(x, [row["n_joint"] for row in shown],
           bottom=[row["n_icl_only"] for row in shown], color="#59A14F", label="joint")
    bottom = [row["n_icl_only"] + row["n_joint"] for row in shown]
    ax.bar(x, [row["n_scl_only"] for row in shown], bottom=bottom,
           color="#E15759", label="SCL only")
    ax.set_xticks(x, [str(i + 1) for i in range(len(shown))])
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("detected events")
    ax.set_title("E  All-event shaft participation", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x - 0.2, [row["mode_A_count"] for row in shown], width=0.4,
           color="#F28E2B", label="direction A")
    ax.bar(x + 0.2, [row["mode_B_count"] for row in shown], width=0.4,
           color="#76B7B2", label="direction B")
    ax.axhline(6, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [str(i + 1) for i in range(len(shown))])
    ax.set_xlabel("candidate rank")
    ax.set_ylabel("assigned events")
    ax.set_title("F  Direction support", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Stable continuous-field screen | " + summary["status"],
        fontsize=14, weight="bold",
    )
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("aggregation", {}).get(
        "figure_prefix", "rev10_sa_v4_spline_field",
    )
    stem = figure_dir / f"{prefix}_search"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem


def _plot_fig4(summary, config, event_records, output_root, contact_names,
               patient_onsets, patient_labels, patient_z):
    selected_id = summary["display_candidate_id"]
    selected = event_records[selected_id]
    labels, ood, z = selected["labels"], selected["ood"], selected["embedding"]
    kmeans = selected["kmeans"]
    kmeans_labels = np.asarray(kmeans.get("labels", labels), dtype=int)
    colors = {0: "#F28E2B", 1: "#4E79A7"}
    fig = plt.figure(figsize=(15.8, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0])

    ax = fig.add_subplot(gs[:, 0])
    shown = 0
    names = np.asarray(contact_names).astype(str)
    for mode in (0, 1):
        eligible = np.flatnonzero((labels == mode) & ~ood)
        if not len(eligible):
            eligible = np.flatnonzero(labels == mode)
        if not len(eligible):
            continue
        record = selected["records"][int(eligible[0])]
        with np.load(record["npz"], allow_pickle=False) as loaded:
            envelope = np.asarray(loaded["contact_envelope"], float)
            dt = float(loaded["contact_envelope_dt_ms"])
            t_on = float(loaded["event_t_on_ms"][record["local_index"]])
            t_off = float(loaded["event_t_off_ms"][record["local_index"]])
        start = max(0, int((t_on - 80.0) / dt))
        stop = min(envelope.shape[1], int((t_off + 150.0) / dt))
        time_axis = np.arange(stop - start) * dt + start * dt - t_on + shown * 420.0
        segment = envelope[:, start:stop]
        scale = max(float(np.percentile(np.abs(segment), 99)), 1e-9)
        for contact, trace in enumerate(segment):
            ax.plot(time_axis, trace / scale * 0.34 + contact,
                    color=colors[mode], linewidth=0.75, alpha=0.85)
        ax.axvspan(shown * 420.0, shown * 420.0 + t_off - t_on,
                   color=colors[mode], alpha=0.10)
        ax.text(shown * 420.0 + 4, len(names) - 0.3, f"model direction {'AB'[mode]}",
                color=colors[mode], fontsize=9, weight="bold")
        shown += 1
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("time aligned to event onset (ms; examples separated)")
    ax.set_ylabel("virtual contacts")
    ax.set_title("A  Direct model-current readout", loc="left", weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(20260812)
    take = rng.choice(len(patient_z), size=min(1200, len(patient_z)), replace=False)
    ax.scatter(patient_z[take, 0], patient_z[take, 1], c="#BBBBBB", s=7,
               alpha=0.18, label="patient train")
    for cluster, marker in ((0, "o"), (1, "^")):
        use = kmeans_labels == cluster
        ax.scatter(z[use, 0], z[use, 1], s=30, marker=marker,
                   c=[colors[int(mode)] for mode in labels[use]],
                   edgecolor="white", linewidth=0.4, alpha=0.85,
                   label=f"model KMeans {cluster + 1}")
    ax.set_xlabel("shaft-aware patient PCA 1")
    ax.set_ylabel("shaft-aware patient PCA 2")
    ax.set_title("B  Patient/model event cloud", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    matrix = np.zeros((2, 2), int)
    for cluster, mode in zip(kmeans_labels, labels):
        matrix[int(cluster), int(mode)] += 1
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], ["direction A", "direction B"])
    ax.set_yticks([0, 1], ["KMeans 1", "KMeans 2"])
    ami = float(kmeans.get("ami_with_direction_labels", np.nan))
    ax.set_title(f"C  KMeans vs direction | AMI={ami:.2f}",
                 loc="left", weight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 1:])
    x = np.arange(len(names))
    for mode in (0, 1):
        model = np.isfinite(selected["onsets"][labels == mode]).mean(axis=0)
        patient = np.isfinite(patient_onsets[patient_labels == mode]).mean(axis=0)
        ax.plot(x, model, color=colors[mode], linewidth=2.0,
                label=f"model {'AB'[mode]}")
        ax.plot(x, patient, color=colors[mode], linewidth=1.5, linestyle="--",
                label=f"patient {'AB'[mode]}")
    ax.axvspan(-0.5, 3.5, color="#76B7B2", alpha=0.08, label="SCL contacts")
    ax.set_xticks(x, names, rotation=45, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("recruitment probability")
    ax.set_title("D  Fixed-contact recruitment prototypes", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    qualifier = "eligible" if summary["selected_candidate_id"] else "diagnostic only"
    fig.suptitle(f"Fig.4-style shaft-aware readout | {selected_id} | {qualifier}",
                 fontsize=14, weight="bold")
    prefix = config.get("aggregation", {}).get(
        "figure_prefix", "rev10_sa_v4_spline_field",
    )
    stem = Path(output_root) / "figures" / f"{prefix}_fig4_modes"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--worker-commit")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    worker_commit = subprocess.check_output(
        ["git", "rev-parse", args.worker_commit or args.expected_commit],
        cwd=ROOT, text=True,
    ).strip()
    output_root = ROOT / config["output_root"]
    manifest = json.loads((output_root / "candidate_manifest.json").read_text())
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("V4 manifest uses another config")
    contract = _load_json_input(config["inputs"]["contact_contract"])
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    scoring_config = _load_json_input(config["inputs"]["shaft_aware_scoring_config"])
    fixed_n = int(config["search"]["objective"]["fixed_events_per_mode"])
    fallback_n = int(config["search"]["objective"]["fallback_events_per_mode"])
    target_path = config["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = config["inputs"]["shaft_aware_floors"]["path"]
    contact_names, embedding, targets, floors6 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=fixed_n,
    )
    _, _, _, floors3 = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=fallback_n,
    )
    contract_names = np.asarray([row["contact_name"] for row in contract["contacts"]])
    if not np.array_equal(contact_names, contract_names):
        raise RuntimeError("scoring and contact contract orders differ")
    with np.load(target_path, allow_pickle=False) as loaded:
        patient_onsets = np.asarray(loaded["patient_train_onsets"], float)
        patient_labels = np.asarray(loaded["patient_train_old_labels"], int)
    patient_z = transform_patient_embedding(
        build_event_features(patient_onsets, groups)["features"], embedding,
    )
    classifier = _classifier_from_manifest(manifest)
    seeds = list(args.seeds or config["search"]["network_seeds"])
    worker_dir = output_root / "workers"
    rows, details, event_records, worker_inputs = [], {}, {}, []
    for candidate in manifest["candidate_set"]["candidates"]:
        onset_blocks, records, metadata = [], [], []
        for seed in seeds:
            stem = worker_dir / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(payload, npz_path, config_sha, worker_commit):
                raise RuntimeError(f"incomplete or stale V4 worker: {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                names = np.asarray(loaded["contact_names"]).astype(str)
                onsets = np.asarray(loaded["onsets"], float)
            if not np.array_equal(names, contact_names):
                raise RuntimeError(f"contact order changed: {stem}")
            offset = sum(len(block) for block in onset_blocks)
            onset_blocks.append(onsets)
            records.extend({
                "seed": int(seed), "local_index": int(local),
                "global_index": int(offset + local), "npz": str(npz_path),
            } for local in range(len(onsets)))
            metadata.append(payload)
            worker_inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": int(seed),
                "json_sha256": _sha256(json_path), "npz_sha256": _sha256(npz_path),
            })
        onsets = np.concatenate(onset_blocks, axis=0) if onset_blocks else np.empty((0, len(contact_names)))
        if len(onsets):
            assigned = assign_direction_modes(
                onsets, groups=groups, embedding=embedding, classifier=classifier,
            )
            labels, ood, z = assigned["labels"], assigned["ood"], assigned["embedding"]
        else:
            labels, ood = np.empty(0, int), np.empty(0, bool)
            z = np.empty((0, len(classifier["coef"])), float)
        counts = np.bincount(labels, minlength=2)
        participation = all_event_shaft_participation(onsets, groups)
        mode_support = mode_conditioned_joint_support(
            onsets, labels, ood, groups,
        )
        seed_readouts = {}
        cursor = 0
        for seed, block in zip(seeds, onset_blocks):
            stop = cursor + len(block)
            block_labels = labels[cursor:stop]
            block_ood = ood[cursor:stop]
            block_participation = all_event_shaft_participation(block, groups)
            block_mode_support = mode_conditioned_joint_support(
                block, block_labels, block_ood, groups,
            )
            block_counts = np.bincount(block_labels, minlength=2)
            seed_readouts[str(seed)] = {
                **block_participation,
                "mode_A_count": int(block_counts[0]),
                "mode_B_count": int(block_counts[1]),
                "ood_fraction": (
                    float(np.mean(block_ood)) if len(block_ood) else 1.0
                ),
                "mode_conditioned_joint_support": block_mode_support,
            }
            cursor = stop
        score6 = score_mode_conditioned_events(
            onsets, labels, groups=groups, pairs=pairs, embedding=embedding,
            targets=targets, floors=floors6, config=scoring_config,
            fixed_events_per_mode=fixed_n,
        )
        score3 = score_mode_conditioned_events(
            onsets, labels, groups=groups, pairs=pairs, embedding=embedding,
            targets=targets, floors=floors3, config=scoring_config,
            fixed_events_per_mode=fallback_n,
        )
        ood_fraction = float(np.mean(ood)) if len(ood) else 1.0
        objective = _objective(
            score6, score3, counts, participation, ood_fraction, config,
            mode_support=mode_support,
        )
        runaway = int(sum(row["run"]["runaway_early_stop_ms"] is not None for row in metadata))
        if runaway:
            objective["selection_score"] = 1000.0 + runaway
        row = {
            "candidate_id": candidate["candidate_id"], "version": candidate["version"],
            "role": candidate["role"], "roughness": candidate["roughness"],
            **objective, **participation,
            "n_detected_events": int(len(onsets)),
            "mode_A_count": int(counts[0]), "mode_B_count": int(counts[1]),
            "ood_fraction": ood_fraction,
            "n_runaway_networks": runaway,
            "n_seeds_with_joint": int(sum(
                value["n_joint"] > 0 for value in seed_readouts.values()
            )),
            "mode_A_joint_count": mode_support["A"]["n_joint"],
            "mode_B_joint_count": mode_support["B"]["n_joint"],
            "mode_A_joint_in_distribution_count": mode_support["A"][
                "n_joint_in_distribution"
            ],
            "mode_B_joint_in_distribution_count": mode_support["B"][
                "n_joint_in_distribution"
            ],
            "mode_A_joint_in_distribution_fraction": mode_support["A"][
                "joint_in_distribution_fraction"
            ],
            "mode_B_joint_in_distribution_fraction": mode_support["B"][
                "joint_in_distribution_fraction"
            ],
            "weak_mode_joint_in_distribution_count": min(
                mode_support[name]["n_joint_in_distribution"]
                for name in ("A", "B")
            ),
            "weak_mode_joint_in_distribution_seed_count": min(
                sum(
                    value["mode_conditioned_joint_support"][name][
                        "n_joint_in_distribution"
                    ] > 0
                    for value in seed_readouts.values()
                )
                for name in ("A", "B")
            ),
            "score6_status": score6["status"], "score3_status": score3["status"],
        }
        rows.append(row)
        kmeans = _kmeans_audit(z, labels)
        details[candidate["candidate_id"]] = {
            "score_n6": score6, "score_n3": score3, "kmeans": kmeans,
            "mode_conditioned_joint_support": mode_support,
            "event_count_by_seed": {
                str(seed): int(metadata[index]["run"]["n_common_detector_events"])
                for index, seed in enumerate(seeds)
            },
            "shaft_and_direction_by_seed": seed_readouts,
        }
        event_records[candidate["candidate_id"]] = {
            "onsets": onsets, "labels": labels, "ood": ood,
            "embedding": z, "kmeans": kmeans, "records": records,
        }
    for row, flag in zip(rows, _pareto_flags(rows)):
        row["pareto_nondominated"] = bool(flag)
    rows.sort(key=lambda row: (
        row["n_runaway_networks"] > 0, row["selection_score"], row["candidate_id"],
    ))
    verdict = _selection_verdict(rows, config)
    selected = verdict["selected"]
    diagnostic = verdict["diagnostic"]
    summary = {
        "status": verdict["status"],
        "scientific_role": config["scientific_role"],
        "safe_claim": (
            "candidate fields were frozen without observation geometry; old A/B direction "
            "and all-event joint-shaft participation were scored as separate factors; "
            "a scalar minimum is not selected when no joint-shaft event exists"
        ),
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else None
        ),
        "selected_selection_score": (
            selected["selection_score"] if selected is not None else None
        ),
        "display_candidate_id": (
            selected["candidate_id"] if selected is not None
            else diagnostic["candidate_id"]
        ),
        "diagnostic_candidate_id": diagnostic["candidate_id"],
        "minimum_joint_events_for_selection": verdict[
            "minimum_joint_events_for_selection"
        ],
        "minimum_seeds_with_joint_for_selection": verdict[
            "minimum_seeds_with_joint_for_selection"
        ],
        "minimum_joint_in_distribution_events_per_mode_for_selection": verdict[
            "minimum_joint_in_distribution_events_per_mode_for_selection"
        ],
        "minimum_seeds_with_joint_in_distribution_per_mode_for_selection": verdict[
            "minimum_seeds_with_joint_in_distribution_per_mode_for_selection"
        ],
        "joint_fraction_target": config["search"]["objective"][
            "minimum_target_joint_fraction"
        ],
        "candidate_rows": rows, "candidate_details": details,
        "network_seeds": seeds, "worker_inputs": worker_inputs,
        "direction_classifier": {
            key: value for key, value in manifest["direction_classifier"].items()
            if key not in {"coef", "class_centers", "class_precisions"}
        },
        "representation_preflight": manifest["representation_preflight"],
        "observation_boundary": config["observation_boundary"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": _runtime_provenance(args.expected_commit),
    }
    prefix = config.get("aggregation", {}).get(
        "artifact_prefix", "v4_spline_field",
    )
    _atomic_csv(output_root / f"{prefix}_candidate_summary.csv", rows)
    _atomic_json(output_root / f"{prefix}_search_summary.json", summary)
    search_stem = _plot_search(summary, manifest, config, output_root)
    fig4_stem = _plot_fig4(
        summary, config, event_records, output_root, contact_names,
        patient_onsets, patient_labels, patient_z,
    )
    (output_root / "figures" / "README.md").write_text(
        f"""### {search_stem.name}

这张图在数值稳定的均匀连续 spline 场中比较 Stage 3 warm、全 sheet 等距 allocation directions 和 observation-free 平滑随机残差。方向 A/B 与同一事件双杆参与被拆成两个目标；触点和杆信息不参与候选场生成。

**关注点**：状态为 {summary['status']}；候选必须实际产生 joint-shaft event 才能成为 selection，SCL-only 事件不能冒充患者多杆恢复。

### {fig4_stem.name}

这张图展示 {summary['display_candidate_id']} 的直接 model-current、patient/model shaft-aware event cloud、de novo KMeans 与监督式 A/B 方向标签的一致性，以及 15 个固定 contact 的招募 prototype。若没有 eligible selection，它只作为 scalar diagnostic 展示。

**关注点**：KMeans 稳定、A/B 可分和双杆招募是三个不同证据，必须同时查看。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_candidate_id": summary["selected_candidate_id"],
        "selected_score": summary["selected_selection_score"],
        "figures": [str(search_stem), str(fig4_stem)],
    }, indent=2))


if __name__ == "__main__":
    main()
