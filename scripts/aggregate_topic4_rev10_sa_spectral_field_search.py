"""Aggregate observation-invariant spectral-field spontaneous searches."""
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
from scripts.build_topic4_rev10_sa_shaft_aware_target import _atomic_json  # noqa: E402
from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
    score_mode_conditioned_events,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _load_reference,
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_profile import (  # noqa: E402
    normalized_rank_curve,
    transform_rank_curves,
)
from src.topic4_core_field_rev9 import assign_frozen_modes  # noqa: E402
from src.topic4_core_field_stage3 import params_to_q  # noqa: E402
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402
from src.topic4_spectral_field import spectral_surface  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field.json"


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


def _classifier(config):
    classifier_path = ROOT / config["inputs"]["frozen_mode_classifier"]["path"]
    reference_path = ROOT / config["inputs"]["rank_curve_reference"]["path"]
    with np.load(classifier_path, allow_pickle=False) as loaded:
        classifier = {
            "embedding_centroids": np.asarray(
                loaded["classifier_embedding_centroids"], float
            ),
            "ood_distance_thresholds": np.asarray(
                loaded["classifier_ood_thresholds"], float
            ),
        }
    return classifier, _load_reference(reference_path)


def _curves_from_ranks(ranks, contact_names, reference):
    axial = axial_map()
    grid = np.asarray(reference["grid"], float)
    curves, keep = [], []
    for index, row in enumerate(np.asarray(ranks, float)):
        rank_dict = {
            str(name): float(value)
            for name, value in zip(contact_names, row) if np.isfinite(value)
        }
        curve = normalized_rank_curve(rank_dict, axial, grid=grid)
        if curve is not None:
            curves.append(curve)
            keep.append(index)
    return np.asarray(curves, float).reshape((-1, len(grid))), np.asarray(keep, int)


def _kmeans_audit(curves, frozen_labels, reference):
    if len(curves) < 4 or np.unique(curves, axis=0).shape[0] < 2:
        return {"status": "INSUFFICIENT", "n_events": int(len(curves))}
    embedded = transform_rank_curves(curves, reference)
    label_sets = [
        KMeans(n_clusters=2, n_init=20, random_state=20260811 + seed).fit_predict(
            embedded
        )
        for seed in range(12)
    ]
    pairwise = [
        adjusted_mutual_info_score(label_sets[left], label_sets[right])
        for left in range(len(label_sets)) for right in range(left + 1, len(label_sets))
    ]
    primary = label_sets[0]
    return {
        "status": "OK",
        "n_events": int(len(curves)),
        "labels": primary,
        "cluster_counts": np.bincount(primary, minlength=2),
        "pairwise_seed_ami_median": float(np.median(pairwise)),
        "pairwise_seed_ami_min": float(np.min(pairwise)),
        "ami_with_frozen_patient_classifier": float(
            adjusted_mutual_info_score(primary, frozen_labels)
        ),
    }


def _relative_surface(candidate, positions, config):
    if candidate["field_type"] == "gaussian_k3_benchmark":
        q = params_to_q(candidate["theta"], positions, K=3, L=20.0)
        surface = np.log(np.maximum(q, 1e-12))
    else:
        surface = spectral_surface(
            candidate["coefficients"], positions,
            max_harmonic=config["field"]["max_harmonic"], L=20.0,
        )
    return np.exp(np.clip(surface - np.max(surface), -30.0, 0.0))


def _patient_profile_prototypes(config, reference, *, n_per_mode=2000):
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    with np.load(target_path, allow_pickle=False) as loaded:
        ranks = np.asarray(loaded["patient_train_ranks"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
        names = np.asarray(loaded["contact_names"]).astype(str)
    prototypes = []
    for mode in (0, 1):
        available = np.flatnonzero(labels == mode)
        chosen = available[np.linspace(
            0, len(available) - 1, min(int(n_per_mode), len(available)), dtype=int,
        )]
        curves, _ = _curves_from_ranks(ranks[chosen], names, reference)
        prototypes.append(np.mean(curves, axis=0))
    return np.asarray(prototypes)


def _plot_landscape(summary, manifest, config, output_root):
    rows = summary["candidate_rows"]
    candidates = {
        row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]
    }
    is_v3 = config["scientific_role"] == (
        "development_only_observation_invariant_uniform_allocation_refinement"
    )
    if is_v3:
        map_ids = [
            "v3_warm_scale_1p00", "v3_initial_selected",
            summary["selected_candidate_id"],
        ]
        titles = ["spectral warm field", "initial-search reference", "best V3 field"]
    else:
        map_ids = [
            "v0_exact_stage3_k3", "v0_stage3_spectral_projection",
            summary["selected_candidate_id"],
        ]
        titles = ["old Stage 3 field", "uniform spectral projection", "best search field"]
    axis = np.linspace(0.0, 20.0, 140)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fig = plt.figure(figsize=(15.8, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    for column, (candidate_id, title) in enumerate(zip(map_ids, titles)):
        ax = fig.add_subplot(gs[0, column])
        relative = _relative_surface(
            candidates[candidate_id], grid, config,
        ).reshape(xx.shape)
        image = ax.contourf(xx, yy, relative, levels=np.linspace(0, 1, 18), cmap="magma")
        ax.set_aspect("equal")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_title(f"{chr(65 + column)}  {title}", loc="left", weight="bold")
        ax.set_xlabel("sheet x (mm)")
        if column == 0:
            ax.set_ylabel("sheet y (mm)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="relative field")

    palette = {"V0": "#7F7F7F", "V1": "#4E79A7", "V2": "#E15759",
               "V3": "#59A14F"}
    ax = fig.add_subplot(gs[1, 0])
    for version in sorted({row["version"] for row in rows}):
        selected = [row for row in rows if row["version"] == version]
        ax.scatter(
            [row["weak_mode_score"] for row in selected],
            [row["frozen_ood_fraction"] for row in selected],
            s=[35 + 4 * row["n_usable_events"] for row in selected],
            color=palette[version], alpha=0.8, label=version,
            edgecolor="white", linewidth=0.5,
        )
    best = next(row for row in rows if row["candidate_id"] == summary["selected_candidate_id"])
    ax.scatter(best["weak_mode_score"], best["frozen_ood_fraction"], s=180,
               marker="*", color="black", zorder=5)
    ax.set_xlabel("shaft-aware weak-mode score (lower better)")
    ax.set_ylabel("frozen-classifier OOD fraction")
    ax.set_title("D  Patient-objective plane", loc="left", weight="bold")
    ax.legend(frameon=False)

    shown = sorted(rows, key=lambda row: row["selection_score"])[:8]
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(len(shown))
    ax.barh(y, [row["mode_A_ICL_precedence_excess"] for row in shown],
            color="#4E79A7", label="mode A ICL precedence")
    ax.barh(y, [row["worst_mode_SCL_recruitment_excess"] for row in shown],
            left=[row["mode_A_ICL_precedence_excess"] for row in shown],
            color="#F28E2B", label="worst SCL recruitment")
    ax.set_yticks(y, [row["candidate_id"] for row in shown], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("patient-floor excess (stacked diagnostic)")
    ax.set_title("E  Limiting objective terms", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    x = np.arange(len(shown))
    ax.bar(x - 0.2, [row["mode_A_count"] for row in shown], width=0.4,
           color="#59A14F", label="mode A")
    ax.bar(x + 0.2, [row["mode_B_count"] for row in shown], width=0.4,
           color="#B07AA1", label="mode B")
    ax.axhline(config["search"]["objective"]["minimum_usable_events_per_mode"],
               color="black", linestyle="--", linewidth=0.9)
    ax.set_xticks(x, [str(index + 1) for index in range(len(shown))])
    ax.set_xlabel("candidate rank in panel E")
    ax.set_ylabel("usable events")
    ax.set_title("F  Spontaneous repertoire support", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        ("Observation-invariant V3 uniform allocation | contacts enter after simulation"
         if is_v3 else
         "Observation-invariant continuous field search | contacts enter after simulation"),
        fontsize=14, weight="bold",
    )
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "rev10_sa_spectral_field_search"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return stem


def _plot_fig4(summary, config, reference, output_root, event_records):
    selected_id = summary["selected_candidate_id"]
    selected = event_records[selected_id]
    curves = selected["curves"]
    frozen_labels = selected["labels"]
    kmeans_labels = selected["kmeans_labels"]
    embedded = transform_rank_curves(curves, reference)
    patient_proto = _patient_profile_prototypes(config, reference)
    model_proto = np.asarray([
        curves[frozen_labels == mode].mean(axis=0) for mode in (0, 1)
    ])

    fig = plt.figure(figsize=(15.8, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    shown = 0
    colors = {0: "#E15759", 1: "#4E79A7"}
    for mode in (0, 1):
        eligible = np.flatnonzero((frozen_labels == mode) & ~selected["ood"])
        if not len(eligible):
            eligible = np.flatnonzero(frozen_labels == mode)
        if not len(eligible):
            continue
        record = selected["records"][int(eligible[0])]
        with np.load(record["npz"], allow_pickle=False) as loaded:
            envelope = np.asarray(loaded["contact_envelope"], float)
            dt = float(loaded["contact_envelope_dt_ms"])
            names = np.asarray(loaded["contact_names"]).astype(str)
            t_on = float(loaded["event_t_on_ms"][record["local_index"]])
            t_off = float(loaded["event_t_off_ms"][record["local_index"]])
        start = max(0, int((t_on - 80.0) / dt))
        stop = min(envelope.shape[1], int((t_off + 150.0) / dt))
        time_axis = np.arange(stop - start) * dt + start * dt - t_on + shown * 420.0
        segment = envelope[:, start:stop]
        scale = max(float(np.percentile(np.abs(segment), 99)), 1e-9)
        for contact, trace in enumerate(segment):
            normalized = trace / scale * 0.34
            ax.plot(time_axis, normalized + contact, color=colors[mode],
                    linewidth=0.75, alpha=0.85)
        ax.axvspan(shown * 420.0, shown * 420.0 + (t_off - t_on),
                   color=colors[mode], alpha=0.10)
        ax.text(shown * 420.0 + 4, len(names) - 0.3, f"model mode {'AB'[mode]}",
                color=colors[mode], fontsize=9, weight="bold")
        shown += 1
    ax.set_yticks(np.arange(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("time aligned to event onset (ms; examples separated)")
    ax.set_ylabel("virtual contacts")
    ax.set_title("A  Direct model-current readout", loc="left", weight="bold")

    ax = fig.add_subplot(gs[0, 1])
    markers = ["o", "^"]
    for cluster in (0, 1):
        use = kmeans_labels == cluster
        ax.scatter(embedded[use, 0], embedded[use, 1], s=28,
                   marker=markers[cluster], c=[colors[int(label)] for label in frozen_labels[use]],
                   edgecolor="white", linewidth=0.4, alpha=0.8,
                   label=f"KMeans cluster {cluster + 1}")
    ax.set_xlabel("frozen patient embedding 1")
    ax.set_ylabel("frozen patient embedding 2")
    ax.set_title("B  KMeans event cloud", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    matrix = np.zeros((2, 2), int)
    for kmeans, frozen in zip(kmeans_labels, frozen_labels):
        matrix[int(kmeans), int(frozen)] += 1
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_xticks([0, 1], ["frozen A", "frozen B"])
    ax.set_yticks([0, 1], ["KMeans 1", "KMeans 2"])
    ax.set_title(
        f"C  Label consistency | AMI={summary['selected_kmeans_ami']:.2f}",
        loc="left", weight="bold",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 1:])
    x = np.asarray(reference["grid"], float)
    for mode in (0, 1):
        ax.plot(x, model_proto[mode], color=colors[mode], linewidth=2.0,
                label=f"model {'AB'[mode]}")
        ax.plot(x, patient_proto[mode], color=colors[mode], linewidth=1.6,
                linestyle="--", label=f"patient {'AB'[mode]}")
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    ax.set_xlabel("shared-axis coordinate (mm)")
    ax.set_ylabel("normalized rank profile")
    ax.set_title("D  Frozen-mode prototypes", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=2)
    fig.suptitle(
        f"Fig.4-style repertoire readout | {selected_id}", fontsize=14, weight="bold",
    )
    figure_dir = Path(output_root) / "figures"
    stem = figure_dir / "rev10_sa_spectral_field_fig4_modes"
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
        raise RuntimeError("spectral manifest uses another config")

    contract = _load_json_input(config["inputs"]["contact_contract"])
    scoring_config = json.loads(
        (ROOT / "config/topic4_rev10_sa_shaft_aware.json").read_text()
    )
    fixed_n = int(config["search"]["objective"]["fixed_events_per_mode"])
    contact_names, embedding, targets, floors = load_scoring_contract(
        config["inputs"]["shaft_aware_target_npz"]["path"],
        config["inputs"]["shaft_aware_floors"]["path"], "FULL_TIMING",
        fixed_events_per_mode=fixed_n,
    )
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    contract_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ]).astype(str)
    if not np.array_equal(contact_names, contract_names):
        raise RuntimeError("scoring and contact contract orders differ")
    classifier, reference = _classifier(config)
    seeds = list(args.seeds or config["search"]["network_seeds"])
    worker_dir = output_root / "workers"
    candidate_rows, details, event_records, worker_inputs = [], {}, {}, []
    ood_weight = float(config["search"]["objective"]["ood_weight"])
    minimum = int(config["search"]["objective"]["minimum_usable_events_per_mode"])

    for candidate in manifest["candidate_set"]["candidates"]:
        onset_blocks, rank_blocks, records, metadata = [], [], [], []
        for seed in seeds:
            stem = worker_dir / f"{candidate['candidate_id']}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            payload = json.loads(json_path.read_text())
            if not _worker_complete(payload, npz_path, config_sha, worker_commit):
                raise RuntimeError(f"incomplete or stale spectral worker: {stem}")
            if payload["candidate"]["field_sha256"] != candidate["field_sha256"]:
                raise RuntimeError(f"candidate field hash changed: {stem}")
            with np.load(npz_path, allow_pickle=False) as loaded:
                names = np.asarray(loaded["contact_names"]).astype(str)
                if not np.array_equal(names, contact_names):
                    raise RuntimeError(f"contact order changed: {stem}")
                onsets = np.asarray(loaded["onsets"], float)
                ranks = np.asarray(loaded["ranks"], float)
            offset = sum(len(block) for block in rank_blocks)
            onset_blocks.append(onsets)
            rank_blocks.append(ranks)
            records.extend({
                "seed": int(seed), "local_index": int(local),
                "global_index": int(offset + local), "npz": str(npz_path),
            } for local in range(len(ranks)))
            metadata.append(payload)
            worker_inputs.append({
                "candidate_id": candidate["candidate_id"], "seed": int(seed),
                "json": str(json_path.relative_to(ROOT)),
                "json_sha256": _sha256(json_path), "npz_sha256": _sha256(npz_path),
            })
        all_onsets = np.concatenate(onset_blocks, axis=0)
        all_ranks = np.concatenate(rank_blocks, axis=0)
        curves, keep = _curves_from_ranks(all_ranks, contact_names, reference)
        usable_onsets = all_onsets[keep]
        usable_records = [records[index] for index in keep]
        if len(curves):
            assigned = assign_frozen_modes(curves, classifier, reference)
            labels, ood = assigned["labels"], assigned["ood"]
        else:
            labels, ood = np.empty(0, int), np.empty(0, bool)
        score = score_mode_conditioned_events(
            usable_onsets, labels, groups=groups, pairs=pairs,
            embedding=embedding, targets=targets, floors=floors,
            config=scoring_config, fixed_events_per_mode=fixed_n,
        )
        kmeans = _kmeans_audit(curves, labels, reference)
        counts = np.bincount(labels, minlength=2)
        support_deficit = int(sum(max(0, minimum - int(value)) for value in counts))
        runaway = int(sum(row["run"]["runaway_early_stop_ms"] is not None
                          for row in metadata))
        ood_fraction = float(np.mean(ood)) if len(ood) else 1.0
        if runaway:
            selection_score = 1000.0 + runaway
        elif score["status"] != "OK":
            selection_score = 100.0 + support_deficit + ood_weight * ood_fraction
        else:
            selection_score = float(score["weak_mode_score"] + ood_weight * ood_fraction)

        def excess(mode, key):
            if score["status"] != "OK":
                return float("nan")
            return float(score["modes"][str(mode)]["floor_excess"][key])

        scl_values = [
            excess(0, "recruitment.SCL"), excess(1, "recruitment.SCL")
        ]
        finite_scl = [value for value in scl_values if np.isfinite(value)]
        row = {
            "candidate_id": candidate["candidate_id"],
            "version": candidate["version"], "role": candidate["role"],
            "selection_score": selection_score,
            "score_status": score["status"],
            "weak_mode_score": (float(score["weak_mode_score"])
                                if score["status"] == "OK" else float("nan")),
            "mean_mode_score": (float(score["mean_mode_score"])
                                if score["status"] == "OK" else float("nan")),
            "n_detected_events": int(sum(row_["run"]["n_common_detector_events"]
                                         for row_ in metadata)),
            "n_usable_events": int(len(curves)),
            "mode_A_count": int(counts[0]), "mode_B_count": int(counts[1]),
            "support_deficit": support_deficit,
            "frozen_ood_fraction": ood_fraction,
            "returned_event_fraction": float(
                sum(row_["run"]["n_returned_events"] for row_ in metadata)
                / max(1, sum(row_["run"]["n_common_detector_events"] for row_ in metadata))
            ),
            "n_runaway_networks": runaway,
            "kmeans_seed_ami_median": float(kmeans.get(
                "pairwise_seed_ami_median", np.nan
            )),
            "kmeans_frozen_ami": float(kmeans.get(
                "ami_with_frozen_patient_classifier", np.nan
            )),
            "mode_A_ICL_precedence_excess": excess(0, "precedence.ICL-ICL"),
            "mode_B_ICL_precedence_excess": excess(1, "precedence.ICL-ICL"),
            "mode_A_SCL_recruitment_excess": excess(0, "recruitment.SCL"),
            "mode_B_SCL_recruitment_excess": excess(1, "recruitment.SCL"),
            "worst_mode_SCL_recruitment_excess": (
                float(max(finite_scl)) if finite_scl else float("nan")
            ),
        }
        candidate_rows.append(row)
        details[candidate["candidate_id"]] = {
            "score": score, "kmeans": kmeans,
            "event_count_by_seed": {
                str(seed): int(metadata[index]["run"]["n_common_detector_events"])
                for index, seed in enumerate(seeds)
            },
        }
        event_records[candidate["candidate_id"]] = {
            "curves": curves, "labels": labels, "ood": ood,
            "kmeans_labels": np.asarray(kmeans.get("labels", labels), int),
            "records": usable_records,
        }
    candidate_rows.sort(key=lambda row: (
        row["n_runaway_networks"] > 0, row["support_deficit"] > 0,
        row["selection_score"], row["candidate_id"],
    ))
    selected = candidate_rows[0]
    summary = {
        "status": (
            "REV10SA_SPECTRAL_V3_SEARCH_COMPLETE"
            if config["scientific_role"]
            == "development_only_observation_invariant_uniform_allocation_refinement"
            else "REV10SA_SPECTRAL_INITIAL_SEARCH_COMPLETE"
        ),
        "scientific_role": config["scientific_role"],
        "safe_claim": (
            "whole-sheet stationary spectral fields were compared using only "
            "post-simulation patient readout and shaft-aware loss"
        ),
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_version": selected["version"],
        "selected_selection_score": selected["selection_score"],
        "selected_kmeans_ami": selected["kmeans_frozen_ami"],
        "candidate_rows": candidate_rows,
        "candidate_details": details,
        "network_seeds": seeds,
        "worker_inputs": worker_inputs,
        "representation_preflight": manifest["representation_preflight"],
        "observation_boundary": config["observation_boundary"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": _runtime_provenance(args.expected_commit),
    }
    _atomic_csv(output_root / "spectral_field_candidate_summary.csv", candidate_rows)
    _atomic_json(output_root / "spectral_field_search_summary.json", summary)
    field_stem = _plot_landscape(summary, manifest, config, output_root)
    fig4_stem = _plot_fig4(summary, config, reference, output_root, event_records)
    readme = output_root / "figures" / "README.md"
    phase_text = (
        "V3 在整张 sheet 的 4x4 等距位置施加完全相同的平滑 allocation direction；"
        "位置集合不读取 contact 或 shaft geometry，由仿真后的 shaft-aware loss 选择。"
        if config["scientific_role"]
        == "development_only_observation_invariant_uniform_allocation_refinement"
        else
        "V0-V2 比较旧 Stage 3 场、其均匀谱投影和平稳随机残差。"
    )
    attention = (
        "比较 warm/reference 与 V3 是否改善最弱模式、SCL recruitment 和 OOD；"
        "均匀位置只是搜索方向，不是候选 core。"
        if config["scientific_role"]
        == "development_only_observation_invariant_uniform_allocation_refinement"
        else
        "先看 V0 投影是否复现旧场，再看 V1/V2 是否在不增加观测先验的情况下"
        "改善最弱模式、SCL recruitment 和 OOD。"
    )
    readme.write_text(
        f"""### rev10_sa_spectral_field_search

{phase_text} 场生成不读取接触点、杆轨迹、患者 onset 或患者 mode；患者信息只在自发 SNN 仿真后的虚拟电极 readout 和 shaft-aware loss 中使用。

**关注点**：{attention}

### rev10_sa_spectral_field_fig4_modes

这张图按 Fig.4 风格展示最低开发集目标候选的直接 model-current 波形、de novo KMeans event cloud、KMeans 与冻结患者分类器的一致性，以及模型/患者 mode prototype。KMeans 是独立结构诊断，shaft-aware patient objective 仍是候选选择标准。

**关注点**：双簇稳定不等于患者模式恢复；必须同时查看 AMI、每模式事件数、OOD 和患者 prototype 偏差。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": summary["status"],
        "selected_candidate_id": summary["selected_candidate_id"],
        "selected_score": summary["selected_selection_score"],
        "figures": [str(field_stem), str(fig4_stem)],
    }, indent=2))


if __name__ == "__main__":
    main()
