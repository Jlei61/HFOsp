#!/usr/bin/env python3
"""Render rev12 Node confirmation dynamics and natural-KMeans companion figures."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
)
from src.topic4_node_dualmode import normalize_event_ranks  # noqa: E402
from src.topic4_node_intervention import (  # noqa: E402
    representative_event_index,
    select_representative_seed,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
MODE_COLORS = ("#C43C39", "#277DA1")
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_bundle(npz_path: Path, worker: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as loaded:
        returned = np.asarray(loaded["event_returned"], bool)
        evaluable = np.asarray(loaded["source_onset_evaluable"], bool)[returned]
        maps = np.asarray(loaded["source_onset_maps_ms"], float)[returned][evaluable]
    return maps, np.asarray(worker["labels"], int)[evaluable], np.flatnonzero(evaluable)


def _load_contract(config: dict, artifact_root: Path):
    cohort = json.loads(
        (artifact_root / config["inputs"]["cohort_config"]["path"]).read_text()
    )
    classifier_config = json.loads(
        (artifact_root / config["inputs"]["classifier_config"]["path"]).read_text()
    )
    patient = _patient_data(cohort, artifact_root)
    classifier = _classifier_contract(classifier_config, artifact_root)
    semantics = _old_to_patient_label_map(patient, classifier)
    patient = _reorder_patient_contract(patient, classifier["names"])
    return patient, classifier, semantics


def _workers(config: dict, aggregate: dict, candidate_id: str,
             patient: dict, classifier: dict, semantics: dict,
             artifact_root: Path) -> dict[int, tuple[Path, dict]]:
    output_root = artifact_root / config["output_root"]
    output = {}
    for seed in aggregate["requested_seeds"]:
        path = output_root / "workers" / f"{candidate_id}_seed_{seed}.npz"
        output[int(seed)] = (
            path,
            _load_network_worker(
                path, patient["contact_names"], classifier,
                semantics["raw_to_patient"],
            ),
        )
    return output


def _aggregate_row(aggregate: dict, candidate_id: str) -> dict:
    rows = [row for row in aggregate["rows"] if row["candidate_id"] == candidate_id]
    if len(rows) != 1:
        raise RuntimeError("candidate is not unique in aggregate")
    return rows[0]


def _selected_seed(row: dict) -> int:
    source_counts = {
        int(record["seed"]): int(record["n_source_maps"])
        for record in row["per_seed"]
    }
    return select_representative_seed(row["score"]["network_scores"], source_counts)


def _event_records(npz_path: Path, worker: dict) -> tuple[list[dict], dict[int, int]]:
    payload = json.loads(npz_path.with_suffix(".json").read_text())
    returned_detected = [
        index for index, event in enumerate(payload["events"])
        if bool(event["returned"])
    ]
    return payload["events"], {
        detected: returned for returned, detected in enumerate(returned_detected)
    }


def _representative_detected_events(npz_path: Path, worker: dict) -> dict[int, int]:
    maps, labels, evaluable_positions = _source_bundle(npz_path, worker)
    with np.load(npz_path, allow_pickle=False) as loaded:
        returned_indices = np.flatnonzero(np.asarray(loaded["event_returned"], bool))
    output = {}
    for mode in (0, 1):
        local = representative_event_index(
            maps, np.asarray(worker["ranks"])[evaluable_positions], labels, mode,
        )
        returned_position = int(evaluable_positions[local])
        output[mode] = int(returned_indices[returned_position])
    return output


def _plot_contacts(ax, xy: np.ndarray, names: np.ndarray) -> None:
    shafts = np.asarray([
        "".join(character for character in str(name) if not character.isdigit())
        for name in names
    ])
    for shaft in ("ICL", "SCL"):
        selected = shafts == shaft
        ax.plot(xy[selected, 0], xy[selected, 1], color=SHAFT_COLORS[shaft], lw=1.0)
        ax.scatter(
            xy[selected, 0], xy[selected, 1], s=26, color=SHAFT_COLORS[shaft],
            edgecolor="white", linewidth=0.6, zorder=5,
        )


def _plot_mechanism(ax, loaded) -> None:
    positions = np.asarray(loaded["positions_E"], float)
    h = np.asarray(loaded["h"], float)
    names = np.asarray(loaded["contact_names"]).astype(str)
    contact_xy = np.asarray(loaded["contact_xy_mm"], float)
    edges = np.linspace(0.0, 20.0, 101)
    weighted, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges), weights=h,
    )
    counts, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges),
    )
    smooth_weighted = gaussian_filter(weighted, sigma=1.2, mode="nearest")
    smooth_counts = gaussian_filter(counts, sigma=1.2, mode="nearest")
    field = np.divide(
        smooth_weighted, smooth_counts,
        out=np.zeros_like(smooth_weighted), where=smooth_counts > 1e-9,
    )
    image = ax.imshow(
        field, origin="lower", extent=(0, 20, 0, 20), cmap="plasma",
        vmin=0.0, vmax=max(float(np.quantile(h, 0.995)), 1e-6),
        interpolation="bilinear",
    )
    _plot_contacts(ax, contact_xy, names)
    ax.set_title("continuous node field", weight="bold")
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("h", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)


def _plot_source(ax, loaded, detected_index: int, mode: int) -> None:
    onset = np.asarray(loaded["source_onset_maps_ms"], float)[detected_index]
    finite = np.isfinite(onset)
    shown = np.ma.masked_invalid(onset)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    if np.any(finite):
        low, high = float(np.min(onset[finite])), float(np.max(onset[finite]))
    else:
        low, high = -20.0, 80.0
    image = ax.imshow(
        shown, origin="lower", extent=(0, 20, 0, 20), cmap=cmap,
        vmin=low, vmax=max(low + 1.0, high), interpolation="nearest",
    )
    _plot_contacts(
        ax, np.asarray(loaded["contact_xy_mm"], float),
        np.asarray(loaded["contact_names"]).astype(str),
    )
    ax.set_title(f"model mode {mode + 1}", weight="bold")
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)")
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_ticks((low, high))
    colorbar.set_ticklabels(("early", "late"))
    colorbar.ax.tick_params(labelsize=7)


def _bandpass(values: np.ndarray, dt_ms: float) -> np.ndarray:
    fs = 1000.0 / float(dt_ms)
    sos = butter(3, [30.0, 80.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(values, float), axis=1)


def _settled_episode_pair(payload_events: list[dict], labels: np.ndarray,
                          detected_to_returned: dict[int, int],
                          representative_events: dict[int, int]) -> tuple[tuple[int, int], list[set[int]]]:
    pair = tuple(int(representative_events[mode]) for mode in (0, 1))
    if pair[0] == pair[1]:
        raise RuntimeError("representative modes share the same settled episode")
    for mode, detected in enumerate(pair):
        returned = detected_to_returned.get(detected)
        if returned is None or int(labels[returned]) != mode:
            raise RuntimeError("representative event does not match its patient mode")
    fragment_sets = [
        set(int(value) for value in payload_events[index].get("fragment_indices", [index]))
        for index in pair
    ]
    if fragment_sets[0] & fragment_sets[1]:
        raise RuntimeError("representative settled episodes share detector fragments")
    return pair, fragment_sets


def _plot_readout(ax, loaded, worker: dict, payload_events: list[dict],
                  detected_to_returned: dict[int, int],
                  representative_events: dict[int, int]) -> dict:
    pair, fragment_sets = _settled_episode_pair(
        payload_events, np.asarray(worker["labels"], int), detected_to_returned,
        representative_events,
    )

    envelope = np.asarray(loaded["contact_envelope"], float)
    dt_ms = float(loaded["contact_envelope_dt_ms"])
    traces_full = _bandpass(envelope, dt_ms)
    names = np.asarray(loaded["contact_names"]).astype(str)
    shafts = np.asarray(loaded["shaft_ids"]).astype(str)
    order = np.concatenate([
        np.flatnonzero(shafts == shaft) for shaft in ("ICL", "SCL")
    ])
    names, shafts, traces_full = names[order], shafts[order], traces_full[order]
    time_ms = np.arange(traces_full.shape[1]) * dt_ms
    durations = [
        float(payload_events[index]["t_off_ms"] - payload_events[index]["t_on_ms"])
        for index in pair
    ]
    pre_ms = 125.0
    width_ms = max(400.0, max(durations) + 250.0)
    windows = []
    trace_windows = []
    for detected in pair:
        onset = float(payload_events[detected]["t_on_ms"])
        start = max(0.0, onset - pre_ms)
        stop = min(float(time_ms[-1]), start + width_ms)
        selected = (time_ms >= start) & (time_ms <= stop)
        windows.append((start, stop, onset))
        trace_windows.append(traces_full[:, selected])
    amplitude = max(
        float(np.quantile(np.abs(np.concatenate(trace_windows, axis=1)), 0.995)),
        1e-12,
    )
    offsets = np.arange(len(names))[::-1] * 1.35
    ax.set_axis_off()
    displayed = []
    for mode, (detected, window, trace_window) in enumerate(zip(pair, windows, trace_windows)):
        start, stop, onset = window
        child = ax.inset_axes([0.02 + 0.50 * mode, 0.0, 0.46, 1.0])
        selected_time = time_ms[(time_ms >= start) & (time_ms <= stop)] - onset
        for row, offset in enumerate(offsets):
            child.plot(
                selected_time, 0.52 * trace_window[row] / amplitude + offset,
                color=SHAFT_COLORS[shafts[row]], lw=0.8,
            )
        event = payload_events[detected]
        child.axvspan(
            0.0, float(event["t_off_ms"] - onset),
            color=MODE_COLORS[mode], alpha=0.10, lw=0,
        )
        child.axvline(0.0, color=MODE_COLORS[mode], lw=0.9, ls="--")
        child.set(
            xlim=(-pre_ms, width_ms - pre_ms),
            ylim=(-0.9, offsets[0] + 0.9),
            xlabel="time from episode onset (ms)",
        )
        child.set_title(f"mode {mode + 1} episode", color=MODE_COLORS[mode], fontsize=9)
        if mode == 0:
            child.set_yticks(offsets, names, fontsize=6.2)
            child.set_ylabel("30-80 Hz virtual-contact activity", fontsize=8)
        else:
            child.set_yticks(offsets, [])
        child.tick_params(axis="y", length=0, pad=2)
        child.spines[["top", "right", "left"]].set_visible(False)
        displayed.append({
            "detected_index": detected,
            "mode": mode,
            "fragment_indices": sorted(fragment_sets[mode]),
            "absolute_window_ms": [start, stop],
            "absolute_onset_ms": onset,
        })
    return {
        "window_rule": "separate windows around mode-specific settled-episode medoids",
        "shared_amplitude_scale": amplitude,
        "events": displayed,
    }


def _block_profile_band(ranks: np.ndarray, labels: np.ndarray,
                        blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_event_ranks(ranks)
    profiles = {0: [], 1: []}
    for block in np.unique(blocks):
        for mode in (0, 1):
            selected = (blocks == block) & (labels == mode)
            if np.any(selected):
                values = normalized[selected]
                finite_count = np.sum(np.isfinite(values), axis=0)
                profile = np.divide(
                    np.nansum(values, axis=0), finite_count,
                    out=np.full(values.shape[1], np.nan), where=finite_count > 0,
                )
                profiles[mode].append(profile)
    low, high = [], []
    for mode in (0, 1):
        values = np.asarray(profiles[mode], float)
        low.append(np.nanquantile(values, 0.10, axis=0))
        high.append(np.nanquantile(values, 0.90, axis=0))
    return np.asarray(low), np.asarray(high)


def _similarity_matrix(model: np.ndarray, patient: np.ndarray) -> np.ndarray:
    output = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            finite = np.isfinite(model[row]) & np.isfinite(patient[column])
            if np.sum(finite) >= 3:
                output[row, column] = float(spearmanr(
                    model[row, finite], patient[column, finite],
                ).statistic)
    return output


def _matrix_permutation_p(model: np.ndarray, patient: np.ndarray,
                          names: np.ndarray, observed: np.ndarray, *,
                          draws: int = 4096, seed: int = 20260822) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    shafts = np.asarray([
        "".join(character for character in str(name) if not character.isdigit())
        for name in names
    ])
    null = np.empty((int(draws), 2, 2), float)
    for draw in range(int(draws)):
        shuffled = patient.copy()
        for shaft in np.unique(shafts):
            indices = np.flatnonzero(shafts == shaft)
            for mode in (0, 1):
                shuffled[mode, indices] = patient[mode, rng.permutation(indices)]
        null[draw] = _similarity_matrix(model, shuffled)
    p = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            if row == column:
                p[row, column] = (1 + np.sum(null[:, row, column] >= observed[row, column])) / (draws + 1)
            else:
                p[row, column] = (1 + np.sum(null[:, row, column] <= observed[row, column])) / (draws + 1)
    return p


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_dynamics(config: dict, aggregate: dict, candidate_id: str,
                    workers: dict, output: Path) -> dict:
    row = _aggregate_row(aggregate, candidate_id)
    seed = _selected_seed(row)
    npz_path, worker = workers[seed]
    detected = _representative_detected_events(npz_path, worker)
    payload_events, mapping = _event_records(npz_path, worker)
    with np.load(npz_path, allow_pickle=False) as loaded:
        fig = plt.figure(figsize=(16.2, 4.6), facecolor="white")
        grid = fig.add_gridspec(
            1, 4, width_ratios=(1.0, 0.95, 0.95, 1.75),
            left=0.055, right=0.99, bottom=0.15, top=0.88, wspace=0.24,
        )
        axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
        _plot_mechanism(axes[0], loaded)
        _plot_source(axes[1], loaded, detected[0], 0)
        _plot_source(axes[2], loaded, detected[1], 1)
        readout = _plot_readout(
            axes[3], loaded, worker, payload_events, mapping, detected,
        )
        output.mkdir(parents=True, exist_ok=True)
        stem = output / "node_dualmode_dynamics"
        fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
    return {
        "seed": seed,
        "seed_rule": "dual-mode source-evaluable seed nearest median objective",
        "representative_detected_events": {str(key): value for key, value in detected.items()},
        "event_rule": "joint source-topology/contact-rank medoid within each model mode",
        "readout": readout,
        "worker_npz": str(npz_path),
        "worker_npz_sha256": _sha256(npz_path),
    }


def render_kmeans(config: dict, aggregate: dict, candidate_id: str,
                  workers: dict, patient: dict, output: Path) -> dict:
    ranks, direction_labels, network_ids = [], [], []
    for seed, (_, worker) in workers.items():
        ranks.append(np.asarray(worker["ranks"], float))
        direction_labels.append(np.asarray(worker["labels"], int))
        network_ids.extend([int(seed)] * len(worker["ranks"]))
    ranks = np.concatenate(ranks)
    direction_labels = np.concatenate(direction_labels)
    natural = natural_kmeans(ranks, direction_labels, random_state=20260822)
    if natural["status"] != "OK":
        raise RuntimeError("pooled confirmation events do not support natural K=2")
    valid = np.asarray(natural["valid_event_mask"], bool)
    ranks = ranks[valid]
    direction_labels = direction_labels[valid]
    network_ids = np.asarray(network_ids, int)[valid]
    raw_labels = np.asarray(natural["cluster_labels"], int)
    alignment = best_binary_alignment(raw_labels, direction_labels)
    labels = np.asarray(alignment["mapped_labels"], int)
    normalized = normalize_event_ranks(ranks)
    order = np.lexsort((network_ids, labels))
    shown = np.ma.masked_invalid(normalized[order].T)
    model_profile = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    model_std = np.asarray([
        np.nanstd(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    patient_normalized = normalize_event_ranks(patient["heldout_ranks"])
    patient_profile = np.asarray([
        np.nanmean(patient_normalized[patient["heldout_labels"] == mode], axis=0)
        for mode in (0, 1)
    ])
    patient_low, patient_high = _block_profile_band(
        patient["heldout_ranks"], patient["heldout_labels"], patient["heldout_blocks"],
    )
    matrix = _similarity_matrix(model_profile, patient_profile)
    p_values = _matrix_permutation_p(
        model_profile, patient_profile, patient["contact_names"], matrix,
    )

    fig = plt.figure(figsize=(15.8, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(2.7, 0.08, 0.85, 1.25, 1.0),
        left=0.055, right=0.985, bottom=0.16, top=0.87, wspace=0.32,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])
    ax_dist = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[0, 3])
    ax_matrix = fig.add_subplot(grid[0, 4])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#D8D8D8")
    image = ax_heat.imshow(shown, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
    split = int(np.sum(labels[order] == 0))
    ax_heat.axvline(split - 0.5, color="#B22222", lw=1.2)
    ax_heat.set_yticks(np.arange(len(patient["contact_names"])), patient["contact_names"], fontsize=7)
    ax_heat.set(xlabel=f"{aggregate['seed_pool']} events", ylabel="virtual contact")
    ax_heat.set_title("clustered event heatmap", weight="bold", pad=24)
    ax_heat.text(
        max(0, split / 2), -0.75, f"mode 1  {split / len(labels):.0%}",
        color=MODE_COLORS[0], ha="center", fontsize=8,
    )
    ax_heat.text(
        split + max(0, (len(labels) - split) / 2), -0.75,
        f"mode 2  {(len(labels) - split) / len(labels):.0%}",
        color=MODE_COLORS[1], ha="center", fontsize=8,
    )
    colorbar = fig.colorbar(image, cax=ax_cbar)
    colorbar.set_ticks((0, 1), labels=("first", "last"))
    colorbar.ax.tick_params(labelsize=7)

    values, positions = [], []
    for contact in range(normalized.shape[1]):
        finite = normalized[:, contact][np.isfinite(normalized[:, contact])]
        if len(finite):
            values.append(finite)
            positions.append(contact)
    violin = ax_dist.violinplot(values, positions=positions, vert=False, widths=0.75,
                                showmeans=False, showmedians=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#777777")
        body.set_alpha(0.42)
        body.set_edgecolor("none")
    violin["cmedians"].set_color("#222222")
    ax_dist.set(xlim=(-0.05, 1.05), ylim=(len(patient["contact_names"]) - 0.5, -0.5),
                xlabel="rank")
    ax_dist.set_yticks(np.arange(len(patient["contact_names"])), [])
    ax_dist.set_title("rank distribution", weight="bold")
    ax_dist.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(patient["contact_names"]))
    for mode in (0, 1):
        ax_profile.fill_betweenx(
            y, model_profile[mode] - model_std[mode], model_profile[mode] + model_std[mode],
            color=MODE_COLORS[mode], alpha=0.12, lw=0,
        )
        ax_profile.fill_betweenx(
            y, patient_low[mode], patient_high[mode], color=MODE_COLORS[mode], alpha=0.08, lw=0,
        )
        ax_profile.plot(model_profile[mode], y, "-o", color=MODE_COLORS[mode], lw=1.7, ms=3.0,
                        label=f"model {mode + 1}")
        ax_profile.plot(patient_profile[mode], y, "--", color=MODE_COLORS[mode], lw=1.2,
                        label=f"patient {mode + 1}")
    ax_profile.set(xlim=(-0.08, 1.08), ylim=(len(y) - 0.5, -0.5), xlabel="mean rank")
    ax_profile.set_yticks(y, [])
    ax_profile.set_title("cluster rank profile", weight="bold")
    ax_profile.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=2)
    ax_profile.spines[["top", "right"]].set_visible(False)

    matrix_image = ax_matrix.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for row in (0, 1):
        for column in (0, 1):
            text = f"{matrix[row, column]:+.2f}{_stars(p_values[row, column])}"
            ax_matrix.text(column, row, text, ha="center", va="center", weight="bold",
                           color="white" if abs(matrix[row, column]) > 0.55 else "#222222")
    ax_matrix.set_xticks((0, 1), ("patient 1", "patient 2"), fontsize=8)
    ax_matrix.set_yticks((0, 1), ("model 1", "model 2"), fontsize=8)
    ax_matrix.set_title("model vs patient", weight="bold")
    matrix_bar = ax_matrix.inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(matrix_image, cax=matrix_bar)
    matrix_bar.tick_params(labelsize=7)

    output.mkdir(parents=True, exist_ok=True)
    stem = output / "node_dualmode_natural_kmeans"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return {
        "n_events": int(len(labels)),
        "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "direction_purity": alignment["purity"],
        "direction_balanced_alignment": alignment["balanced_alignment"],
        "kmeans_seed_ami_median": natural["kmeans_seed_ami_median"],
        "silhouette": natural["silhouette"],
        "heldout_gmm_k2_minus_k1_loglik_per_event": natural[
            "heldout_gmm_k2_minus_k1_loglik_per_event"
        ],
        "similarity_matrix": matrix.tolist(),
        "directional_permutation_p": p_values.tolist(),
        "permutation_contract": "within-shaft patient-contact shuffle, 4096 draws",
        "network_event_counts": {
            str(seed): int(np.sum(network_ids == seed))
            for seed in np.unique(network_ids)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.resolve().read_text())
    aggregate = json.loads(args.summary.resolve().read_text())
    patient, classifier, semantics = _load_contract(config, artifact_root)
    workers = _workers(
        config, aggregate, args.candidate_id, patient, classifier, semantics,
        artifact_root,
    )
    output = args.out or (
        artifact_root / config["output_root"] / "figures" / args.candidate_id
    )
    dynamics = render_dynamics(
        config, aggregate, args.candidate_id, workers, output,
    )
    kmeans = render_kmeans(
        config, aggregate, args.candidate_id, workers, patient, output,
    )
    metadata = {
        "status": "REV12ND_NODE_FIGURES_COMPLETE",
        "seed_pool": aggregate["seed_pool"],
        "candidate_id": args.candidate_id,
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config.resolve())},
        "summary": {"path": str(args.summary.resolve()), "sha256": _sha256(args.summary.resolve())},
        "dynamics": dynamics,
        "natural_kmeans": kmeans,
        "claim_boundary": (
            "Plots show model-current readout and development-only patient consistency; "
            "they do not establish patient causality or clinical generalization."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(f"""### node_dualmode_dynamics.png

连续 Node field、两个算法选出的模式事件和同一网络的 virtual-contact readout。代表网络取 {aggregate['seed_pool']} 池中最接近中位目标的 seed {dynamics['seed']}；两次事件分别是各模式在空间起始图与触点 rank 联合空间中的 medoid，并以互相独立的时间窗显示。

**关注点**：两种触点顺序背后是否具有可区分的局部起始拓扑，而不只是同一多点活动被分成两类。

### node_dualmode_natural_kmeans.png

汇总所有 {aggregate['seed_pool']} 网络的 returned episodes 后独立执行 K=2；患者标签只在聚类完成后用于语义对齐。左侧显示逐 episode rank 与 missing contact，中间比较模型和患者留出 prototype，右侧给出杆内 contact-shuffle 的方向性检验。

**关注点**：自然两簇是否稳定、是否在两种模式上同时对齐患者，而不是仅恢复占优模式。
""")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
