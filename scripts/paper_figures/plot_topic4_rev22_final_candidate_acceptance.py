#!/usr/bin/env python3
"""Render the rev22 final-candidate direct readout, activity GIF and KMeans audit.

The producer consumes immutable confirmation artifacts only.  It never replays an
SNN and never selects a parameter.  A candidate id must already have been frozen
and present in the validation aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_rev20_dual_core_endpoint import embedding_from_training_arrays  # noqa: E402
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


STAGE = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability"
)
DEFAULT_CONFIG = ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json"
FORMATS = ("png", "pdf", "svg")
MODE_COLORS = ("#B8323C", "#277DA1")
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON input is not an object: {path}")
    return value


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]).copy() for name in loaded.files}


def _resolve(record: Mapping, artifact_root: Path) -> Path:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        tracked = ROOT / path
        path = tracked if tracked.is_file() else artifact_root / path
    if not path.is_file() or _sha256(path) != str(record["sha256"]):
        raise RuntimeError(f"frozen input changed: {path}")
    return path


def _contact_order(names: Sequence[str], shafts: Sequence[str]) -> np.ndarray:
    def key(index: int) -> tuple[int, int, str]:
        name = str(names[index])
        match = re.search(r"(\d+)$", name)
        number = int(match.group(1)) if match else 10**6
        shaft = str(shafts[index])
        return (0 if shaft == "ICL" else 1, number, name)

    return np.asarray(sorted(range(len(names)), key=key), dtype=int)


def _worker_rows(worker_dir: Path, candidate_id: str) -> list[dict]:
    rows = []
    for json_path in sorted(worker_dir.glob(f"{candidate_id}_topo_*_dyn_*.json")):
        payload = _read_json(json_path)
        if not str(payload.get("status", "")).endswith("_WORKER_COMPLETE"):
            raise RuntimeError(f"incomplete confirmation worker: {json_path}")
        record = payload.get("arrays") or {}
        npz_path = Path(record.get("path") or json_path.with_suffix(".npz"))
        if not npz_path.is_absolute():
            npz_path = worker_dir / npz_path
        if not npz_path.is_file() or _sha256(npz_path) != record.get("sha256"):
            raise RuntimeError(f"confirmation NPZ hash mismatch: {npz_path}")
        arrays = _load_npz(npz_path)
        rows.append({
            "json_path": json_path,
            "npz_path": npz_path,
            "json_sha256": _sha256(json_path),
            "npz_sha256": _sha256(npz_path),
            "topology_seed": int(np.asarray(arrays["topology_seed"]).item()),
            "dynamics_seed": int(np.asarray(arrays["dynamics_seed"]).item()),
            "arrays": arrays,
        })
    if not rows:
        raise RuntimeError(f"no confirmation workers for candidate {candidate_id}")
    return rows


def _patient_band(training: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = normalize_event_ranks(training["patient_train_ranks"])
    labels = np.asarray(training["patient_train_old_labels"], int)
    blocks = np.asarray(training["patient_train_block_ids"])
    profiles = patient_profiles(training["patient_train_ranks"], labels)
    low = np.full_like(profiles, np.nan)
    high = np.full_like(profiles, np.nan)
    for mode in (0, 1):
        values = []
        for block in np.unique(blocks[labels == mode]):
            selected = (labels == mode) & (blocks == block)
            if np.any(selected):
                count = np.sum(np.isfinite(ranks[selected]), axis=0)
                values.append(np.divide(
                    np.nansum(ranks[selected], axis=0), count,
                    out=np.full(ranks.shape[1], np.nan), where=count > 0,
                ))
        if values:
            low[mode] = np.nanquantile(values, 0.05, axis=0)
            high[mode] = np.nanquantile(values, 0.95, axis=0)
    return profiles, low, high


def _prototype_matrix(model_ranks: np.ndarray, labels: np.ndarray,
                      patient_ranks: np.ndarray, patient_labels: np.ndarray) -> np.ndarray:
    model = np.asarray([
        np.nanmean(normalize_event_ranks(model_ranks[labels == mode]), axis=0)
        for mode in (0, 1)
    ])
    patient = patient_profiles(patient_ranks, patient_labels)
    matrix = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            valid = np.isfinite(model[row]) & np.isfinite(patient[column])
            if np.sum(valid) >= 3:
                matrix[row, column] = float(spearmanr(
                    model[row, valid], patient[column, valid]
                ).statistic)
    return matrix


def _matrix_permutation(model_ranks: np.ndarray, labels: np.ndarray,
                        patient_ranks: np.ndarray, patient_labels: np.ndarray,
                        shafts: Sequence[str], *, draws: int = 4096,
                        seed: int = 20260904) -> np.ndarray:
    observed = _prototype_matrix(model_ranks, labels, patient_ranks, patient_labels)
    model = np.asarray([
        np.nanmean(normalize_event_ranks(model_ranks[labels == mode]), axis=0)
        for mode in (0, 1)
    ])
    patient = patient_profiles(patient_ranks, patient_labels)
    rng = np.random.default_rng(seed)
    null = np.full((draws, 2, 2), np.nan)
    shafts = np.asarray(shafts).astype(str)
    for draw in range(draws):
        shuffled = model.copy()
        for shaft in np.unique(shafts):
            selected = np.flatnonzero(shafts == shaft)
            shuffled[:, selected] = shuffled[:, rng.permutation(selected)]
        for row in (0, 1):
            for column in (0, 1):
                valid = np.isfinite(shuffled[row]) & np.isfinite(patient[column])
                if np.sum(valid) >= 3:
                    null[draw, row, column] = spearmanr(
                        shuffled[row, valid], patient[column, valid]
                    ).statistic
    p = np.full((2, 2), np.nan)
    for row in (0, 1):
        for column in (0, 1):
            values = null[:, row, column]
            values = values[np.isfinite(values)]
            if len(values):
                if row == column:
                    p[row, column] = (1 + np.sum(values >= observed[row, column])) / (len(values) + 1)
                else:
                    p[row, column] = (1 + np.sum(values <= observed[row, column])) / (len(values) + 1)
    return p


def _collect_events(rows: Sequence[Mapping], groups: Mapping, embedding: Mapping,
                    classifier: Mapping, kmeans_seed: int) -> dict:
    ranks_list, onsets_list, event_map, ood_list = [], [], [], []
    names = None
    shafts = None
    for unit_index, row in enumerate(rows):
        arrays = row["arrays"]
        unit_names = np.asarray(arrays["contact_names"]).astype(str)
        unit_shafts = np.asarray(arrays["shaft_ids"]).astype(str)
        if names is None:
            names, shafts = unit_names, unit_shafts
        elif not np.array_equal(names, unit_names) or not np.array_equal(shafts, unit_shafts):
            raise RuntimeError("confirmation workers use inconsistent contact order")
        returned = np.asarray(arrays["event_returned"], bool)
        onsets = np.asarray(arrays["onsets"], float)[returned]
        ranks = np.asarray(arrays["ranks"], float)[returned]
        source_indices = np.flatnonzero(returned)
        direction = assign_direction_modes(
            onsets, groups=groups, embedding=embedding, classifier=classifier,
        )
        readable = np.sum(np.isfinite(onsets), axis=1) >= 3
        for local_index in np.flatnonzero(readable):
            event_map.append((unit_index, int(source_indices[local_index])))
        ranks_list.append(ranks[readable])
        onsets_list.append(onsets[readable])
        ood_list.append(np.asarray(direction["ood"], bool)[readable])
    ranks = np.concatenate(ranks_list)
    onsets = np.concatenate(onsets_list)
    ood = np.concatenate(ood_list)
    direction = assign_direction_modes(
        onsets, groups=groups, embedding=embedding, classifier=classifier,
    )
    natural = natural_kmeans(
        ranks, np.asarray(direction["labels"], int), random_state=int(kmeans_seed),
    )
    if natural["status"] != "OK":
        raise RuntimeError("final candidate does not support natural KMeans K=2")
    valid = np.asarray(natural["valid_event_mask"], bool)
    alignment = best_binary_alignment(
        np.asarray(natural["cluster_labels"], int),
        np.asarray(direction["labels"], int)[valid],
    )
    return {
        "names": names, "shafts": shafts, "ranks": ranks[valid], "onsets": onsets[valid],
        "ood": ood[valid], "event_map": [event_map[index] for index in np.flatnonzero(valid)],
        "labels": np.asarray(alignment["mapped_labels"], int),
        "alignment": alignment, "natural": natural,
    }


def _representative_pair(events: Mapping, rows: Sequence[Mapping],
                         *, maximum_span_ms: float = 2000.0) -> dict:
    normalized = normalize_event_ranks(events["ranks"])
    labels = np.asarray(events["labels"], int)
    profiles = np.asarray([np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)])
    distances = np.full(len(labels), np.inf)
    for index, mode in enumerate(labels):
        valid = np.isfinite(normalized[index]) & np.isfinite(profiles[mode])
        if np.sum(valid) >= 3:
            distances[index] = float(np.mean(np.abs(normalized[index, valid] - profiles[mode, valid])))
    best = None
    for unit_index in range(len(rows)):
        candidates = [np.flatnonzero((labels == mode) & np.asarray([
            mapping[0] == unit_index for mapping in events["event_map"]
        ])) for mode in (0, 1)]
        for first in candidates[0]:
            for second in candidates[1]:
                source_a = events["event_map"][int(first)][1]
                source_b = events["event_map"][int(second)][1]
                arrays = rows[unit_index]["arrays"]
                starts = np.asarray(arrays["event_t_on_ms"], float)[[source_a, source_b]]
                stops = np.asarray(arrays["event_t_off_ms"], float)[[source_a, source_b]]
                span = float(np.max(stops) - np.min(starts))
                if span > maximum_span_ms:
                    continue
                score = float(distances[first] + distances[second] + span / 10000.0)
                record = (score, unit_index, int(first), int(second), span)
                if best is None or record < best:
                    best = record
    if best is None:
        raise RuntimeError("no same-network two-mode representative pair within 2000 ms")
    _, unit_index, first, second, span = best
    return {
        "unit_index": unit_index,
        "pooled_event_indices": [first, second],
        "source_event_indices": [events["event_map"][first][1], events["event_map"][second][1]],
        "span_ms": span,
        "selection_rule": "same network; one event per aligned KMeans mode; minimum medoid distance plus span/10000; span <=2000 ms",
    }


def _bandpass(values: np.ndarray, dt_ms: float) -> np.ndarray:
    fs = 1000.0 / float(dt_ms)
    sos = butter(3, [30.0, 80.0], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(values, float), axis=1)


def _activity_and_readout(rows: Sequence[Mapping], pair: Mapping, events: Mapping,
                          output_dir: Path, candidate_id: str) -> tuple[list[Path], dict]:
    row = rows[int(pair["unit_index"])]
    arrays = row["arrays"]
    source = np.asarray(pair["source_event_indices"], int)
    starts = np.asarray(arrays["event_t_on_ms"], float)[source]
    stops = np.asarray(arrays["event_t_off_ms"], float)[source]
    envelope_dt = float(np.asarray(arrays["contact_envelope_dt_ms"]).item())
    activity_dt = float(np.asarray(arrays["sheet_activity_frame_ms"]).item())
    duration = np.asarray(arrays["contact_envelope"]).shape[1] * envelope_dt
    selected_start, selected_stop = float(np.min(starts)), float(np.max(stops))
    returned = np.asarray(arrays["event_returned"], bool)
    other = returned.copy()
    other[source] = False
    other_starts = np.asarray(arrays["event_t_on_ms"], float)[other]
    other_stops = np.asarray(arrays["event_t_off_ms"], float)[other]
    if np.any((other_starts < selected_stop) & (other_stops > selected_start)):
        raise RuntimeError("representative pair overlaps another returned causal family")
    previous = other_stops[other_stops <= selected_start]
    following = other_starts[other_starts >= selected_stop]
    window_start = max(0.0, selected_start - 80.0)
    window_stop = min(duration, selected_stop + 80.0)
    if len(previous):
        window_start = max(window_start, float(np.max(previous) + 5.0))
    if len(following):
        window_stop = min(window_stop, float(np.min(following) - 5.0))
    if window_start >= selected_start or window_stop <= selected_stop:
        raise RuntimeError("representative pair lacks an isolated display margin")
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    order = _contact_order(names, shafts)
    names, shafts = names[order], shafts[order]
    contact_xy = np.asarray(arrays["contact_xy_mm"], float)[order]
    envelope = _bandpass(np.asarray(arrays["contact_envelope"], float)[order], envelope_dt)
    lo = int(np.floor(window_start / envelope_dt))
    hi = int(np.ceil(window_stop / envelope_dt)) + 1
    traces = envelope[:, lo:hi]
    trace_time = np.arange(lo, hi) * envelope_dt
    scale = max(float(np.quantile(np.abs(traces), 0.995)), 1e-12)
    traces = traces / scale
    activity = np.asarray(arrays["sheet_activity_counts"], float)
    alo = int(np.floor(window_start / activity_dt))
    ahi = min(len(activity), int(np.ceil(window_stop / activity_dt)) + 1)
    activity = np.asarray([gaussian_filter(frame, 0.65) for frame in activity[alo:ahi]])
    frame_time = np.arange(alo, ahi) * activity_dt
    vmax = max(1.0, float(np.quantile(activity, 0.995)))
    selected_frame = int(np.argmax(np.sum(activity, axis=(1, 2))))

    def make_figure(frame_index: int):
        fig = plt.figure(figsize=(11.2, 4.4), facecolor="white")
        grid = fig.add_gridspec(1, 2, width_ratios=(0.82, 1.75), left=0.06, right=0.985,
                                bottom=0.14, top=0.91, wspace=0.25)
        ax_field = fig.add_subplot(grid[0, 0])
        ax_trace = fig.add_subplot(grid[0, 1])
        image = ax_field.imshow(activity[frame_index], origin="lower", extent=(0, 20, 0, 20),
                                cmap="viridis", vmin=0, vmax=vmax, interpolation="bilinear")
        for shaft in ("ICL", "SCL"):
            mask = shafts == shaft
            ax_field.scatter(contact_xy[mask, 0], contact_xy[mask, 1], s=30,
                             color=SHAFT_COLORS[shaft], edgecolor="white", linewidth=0.7)
        ax_field.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)")
        ax_field.set_aspect("equal")
        ax_field.set_title("SNN activity", loc="left", weight="bold")
        ax_field.spines[["top", "right"]].set_visible(False)
        bar = fig.colorbar(image, ax=ax_field, fraction=0.045, pad=0.04)
        bar.set_label(f"E spikes / {activity_dt:g} ms bin", fontsize=7)
        bar.ax.tick_params(labelsize=6)
        time_label = ax_field.text(0.02, 0.02, "", transform=ax_field.transAxes,
                                   color="white", fontsize=8, weight="bold")

        offsets = np.arange(len(names))[::-1] * 1.45
        for index, offset in enumerate(offsets):
            ax_trace.plot(trace_time - window_start, traces[index] * 0.55 + offset,
                          color=SHAFT_COLORS[shafts[index]], lw=0.9)
        for mode, event_index in enumerate(source):
            start = float(np.asarray(arrays["event_t_on_ms"])[event_index] - window_start)
            stop = float(np.asarray(arrays["event_t_off_ms"])[event_index] - window_start)
            ax_trace.axvspan(start, stop, color=MODE_COLORS[mode], alpha=0.12, lw=0)
        cursor = ax_trace.axvline(frame_time[frame_index] - window_start,
                                 color="#202020", lw=0.9)
        ax_trace.set_yticks(offsets, names, fontsize=6.5)
        ax_trace.tick_params(axis="y", length=0)
        ax_trace.set(xlim=(0, window_stop - window_start), ylim=(-1, offsets[0] + 1),
                     xlabel="time in displayed window (ms)")
        ax_trace.set_title("Virtual-contact readout (30-80 Hz)", loc="left", weight="bold")
        ax_trace.spines[["top", "right", "left"]].set_visible(False)
        ax_trace.legend(handles=[Patch(facecolor=MODE_COLORS[0], alpha=0.18, label="mode A"),
                                 Patch(facecolor=MODE_COLORS[1], alpha=0.18, label="mode B")],
                        loc="upper right", frameon=False, ncol=2, fontsize=7)
        xbar = max(10.0, (window_stop - window_start) * 0.025)
        ybar = -0.15
        ax_trace.plot([xbar, xbar], [ybar - 0.55, ybar], color="#222222", lw=1.2)
        ax_trace.text(xbar + 4, ybar - 0.28, "1 a.u.", va="center", fontsize=6.3)
        time_label.set_text(f"t = {frame_time[frame_index] - window_start:.0f} ms")
        return fig, image, cursor, time_label

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, _, _, _ = make_figure(selected_frame)
    stem = output_dir / "rev22_dci_final_candidate_direct_readout"
    paths = []
    for suffix in FORMATS:
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300 if suffix == "png" else None, facecolor="white",
                    bbox_inches="tight")
        paths.append(path)
    plt.close(fig)

    fig, image, cursor, time_label = make_figure(0)
    step = max(1, int(round(10.0 / activity_dt)))
    frame_indices = np.arange(0, len(activity), step, dtype=int)

    def update(animation_index: int):
        frame_index = int(frame_indices[animation_index])
        image.set_data(activity[frame_index])
        cursor.set_xdata([frame_time[frame_index] - window_start] * 2)
        time_label.set_text(f"t = {frame_time[frame_index] - window_start:.0f} ms")
        return image, cursor, time_label

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=False)
    gif_path = output_dir / "rev22_dci_final_candidate_direct_readout.gif"
    animation.save(gif_path, writer=PillowWriter(fps=12.5), dpi=100)
    plt.close(fig)
    encoded = Image.open(gif_path)
    durations = []
    for frame in range(encoded.n_frames):
        encoded.seek(frame)
        durations.append(int(encoded.info.get("duration", -1)))
    if encoded.n_frames != len(frame_indices) or set(durations) != {80}:
        raise RuntimeError("encoded GIF timing or frame count differs from the render contract")
    paths.append(gif_path)
    return paths, {
        "candidate_id": candidate_id, "topology_seed": row["topology_seed"],
        "dynamics_seed": row["dynamics_seed"], "window_ms": [window_start, window_stop],
        "source_event_indices": source.tolist(), "representative_pair": dict(pair),
        "activity_frame_ms": activity_dt, "envelope_dt_ms": envelope_dt,
        "activity_vmax_q995": vmax, "trace_normalization_q995": scale,
        "gif_frame_count": int(encoded.n_frames), "gif_frame_duration_ms": 80,
        "gif_loop": int(encoded.info.get("loop", -1)),
        "worker_npz_sha256": row["npz_sha256"],
    }


def _kmeans_figure(events: Mapping, training: Mapping, output_dir: Path) -> tuple[list[Path], dict]:
    labels = np.asarray(events["labels"], int)
    ranks = np.asarray(events["ranks"], float)
    normalized = normalize_event_ranks(ranks)
    names = np.asarray(events["names"]).astype(str)
    shafts = np.asarray(events["shafts"]).astype(str)
    contact_order = _contact_order(names, shafts)
    names, shafts = names[contact_order], shafts[contact_order]
    ranks = ranks[:, contact_order]
    normalized = normalized[:, contact_order]
    patient_ranks = np.asarray(training["patient_train_ranks"], float)[:, contact_order]
    order = np.lexsort((np.arange(len(labels)), labels))
    patient_profile, patient_low, patient_high = _patient_band(training)
    patient_profile = patient_profile[:, contact_order]
    patient_low = patient_low[:, contact_order]
    patient_high = patient_high[:, contact_order]
    model_profile = np.asarray([
        np.nanmean(normalized[labels == mode], axis=0) for mode in (0, 1)
    ])
    matrix = _prototype_matrix(
        ranks, labels, patient_ranks, training["patient_train_old_labels"],
    )
    p_values = _matrix_permutation(
        ranks, labels, patient_ranks, training["patient_train_old_labels"],
        shafts,
    )
    fig = plt.figure(figsize=(12.0, 4.3), facecolor="white")
    grid = fig.add_gridspec(1, 12, left=0.07, right=0.975, bottom=0.15, top=0.90,
                            wspace=0.75)
    ax_heat = fig.add_subplot(grid[0, 0:6])
    ax_dist = fig.add_subplot(grid[0, 6:8], sharey=ax_heat)
    ax_profile = fig.add_subplot(grid[0, 8:10], sharey=ax_heat)
    ax_matrix = fig.add_subplot(grid[0, 10:12])
    shown = np.ma.masked_invalid(normalized[order].T)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#D6D6D6")
    image = ax_heat.imshow(shown, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                           cmap=cmap)
    split = int(np.sum(labels == 0))
    ax_heat.axvline(split - 0.5, color="#B8323C", lw=1.2)
    ax_heat.set_yticks(np.arange(len(names)), names, fontsize=6.5)
    ax_heat.set(xlabel=f"returned causal families (n={len(labels)})", ylabel="virtual contact")
    ax_heat.set_title("Clustered event ranks", loc="left", weight="bold", pad=23)
    ax_heat.text(max(0, split / 2), 1.015, f"mode A  {split / len(labels):.0%}",
                 transform=ax_heat.get_xaxis_transform(), color=MODE_COLORS[0],
                 ha="center", va="bottom", fontsize=7)
    ax_heat.text(split + max(0, (len(labels) - split) / 2), 1.015,
                 f"mode B  {(len(labels) - split) / len(labels):.0%}",
                 transform=ax_heat.get_xaxis_transform(), color=MODE_COLORS[1],
                 ha="center", va="bottom", fontsize=7)
    cax = ax_heat.inset_axes([1.015, 0, 0.018, 1])
    bar = fig.colorbar(image, cax=cax)
    bar.set_ticks((0, 1), labels=("first", "last"))
    bar.ax.tick_params(labelsize=6)

    positions = np.arange(len(names))
    for mode, offset in ((0, -0.13), (1, 0.13)):
        data, pos = [], []
        for contact in range(len(names)):
            values = normalized[labels == mode, contact]
            values = values[np.isfinite(values)]
            if len(values):
                data.append(values)
                pos.append(contact + offset)
        if data:
            violin = ax_dist.violinplot(data, positions=pos, vert=False, widths=0.22,
                                        showmeans=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(MODE_COLORS[mode])
                body.set_edgecolor("none")
                body.set_alpha(0.55)
    ax_dist.set(xlim=(-0.05, 1.05), xlabel="normalized rank")
    ax_dist.set_title("Rank distribution", weight="bold")
    ax_dist.tick_params(axis="y", left=False, labelleft=False)
    ax_dist.spines[["top", "right", "left"]].set_visible(False)

    for mode in (0, 1):
        ax_profile.fill_betweenx(positions, patient_low[mode], patient_high[mode],
                                 color=MODE_COLORS[mode], alpha=0.12, lw=0)
        ax_profile.plot(model_profile[mode], positions, "-o", color=MODE_COLORS[mode],
                        lw=1.4, ms=2.5, label=f"model {chr(65 + mode)}")
        ax_profile.plot(patient_profile[mode], positions, "--", color=MODE_COLORS[mode],
                        lw=1.0, label=f"patient {chr(65 + mode)}")
    ax_profile.set(xlim=(-0.05, 1.05), xlabel="mean normalized rank")
    ax_profile.set_title("Cluster rank profile", weight="bold", pad=23)
    ax_profile.tick_params(axis="y", left=False, labelleft=False)
    ax_profile.spines[["top", "right", "left"]].set_visible(False)
    ax_profile.legend(frameon=False, fontsize=5.8, ncol=2, loc="lower center",
                      bbox_to_anchor=(0.5, 1.01), columnspacing=0.7, handlelength=1.6)

    matrix_image = ax_matrix.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for row in (0, 1):
        for column in (0, 1):
            p = p_values[row, column]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax_matrix.text(column, row, stars, ha="center", va="center", fontsize=10,
                           color="white" if abs(matrix[row, column]) > 0.55 else "#222222")
    ax_matrix.set_xticks((0, 1), ("patient A", "patient B"), fontsize=6.5)
    ax_matrix.set_yticks((0, 1), ("model A", "model B"), fontsize=6.5)
    ax_matrix.set_title("Model vs patient", weight="bold")
    matrix_cax = ax_matrix.inset_axes([1.08, 0, 0.06, 1])
    fig.colorbar(matrix_image, cax=matrix_cax, label="Spearman rho")
    matrix_cax.tick_params(labelsize=6)
    ax_heat.set_ylim(len(names) - 0.5, -0.5)
    fig.savefig(output_dir / "rev22_dci_final_candidate_kmeans.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / "rev22_dci_final_candidate_kmeans.pdf", facecolor="white",
                bbox_inches="tight")
    fig.savefig(output_dir / "rev22_dci_final_candidate_kmeans.svg", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    paths = [output_dir / f"rev22_dci_final_candidate_kmeans.{suffix}" for suffix in FORMATS]
    return paths, {
        "n_events": int(len(labels)), "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "direction_balanced_alignment": events["alignment"]["balanced_alignment"],
        "direction_purity": events["alignment"]["purity"],
        "kmeans_seed_ami_median": events["natural"]["kmeans_seed_ami_median"],
        "prototype_spearman_matrix": matrix.tolist(),
        "directional_channel_permutation_p": p_values.tolist(),
        "ood_fraction": float(np.mean(events["ood"])),
    }


def render(*, config_path: Path, validation_path: Path, frozen_path: Path,
           candidate_manifest_path: Path, worker_dir: Path, candidate_id: str,
           output_dir: Path, artifact_root: Path = Path("/home/honglab/leijiaxin/HFOsp")) -> dict:
    config = _read_json(config_path)
    validation = _read_json(validation_path)
    frozen = _read_json(frozen_path)
    manifest = _read_json(candidate_manifest_path)
    if candidate_id not in set(map(str, frozen.get("candidate_ids", []))):
        raise RuntimeError("candidate is not in the frozen candidate set")
    confirmation = {str(row["candidate_id"]): row
                    for row in (validation.get("phases") or {}).get("confirmation", [])}
    if candidate_id not in confirmation:
        raise RuntimeError("candidate is absent from confirmation validation")
    if confirmation[candidate_id].get("primary_status") not in {
            "OK", "PRIMARY_ENDPOINT_PARTIALLY_ESTIMABLE"}:
        raise RuntimeError("candidate has no estimable confirmation endpoints")
    candidates = {str(row["candidate_id"]) for row in manifest.get("candidates", [])}
    if candidate_id not in candidates:
        raise RuntimeError("candidate is absent from the execution manifest")
    training = _load_npz(_resolve(config["inputs"]["patient_training_target"], artifact_root))
    contract = _read_json(_resolve(config["inputs"]["contact_contract"], artifact_root))
    support = _read_json(_resolve(config["inputs"]["patient_support_config"], artifact_root))
    rev20 = _read_json(_resolve(config["inputs"]["rev20_config"], artifact_root))
    classifier_path = _resolve(support["inputs"]["old_ab_train_only_classifier"], artifact_root)
    classifier = _read_json(classifier_path)["direction_classifier"]
    classifier = {key: (np.asarray(value, float) if key in {
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds"} else value)
                  for key, value in classifier.items()}
    rows = _worker_rows(worker_dir, candidate_id)
    if len(rows) != 12:
        raise RuntimeError(f"final acceptance requires 12 confirmation workers, found {len(rows)}")
    groups = contract_groups(contract)
    embedding = embedding_from_training_arrays(training)
    events = _collect_events(rows, groups, embedding, classifier,
                             int(rev20["validation"]["natural_kmeans_seed"]))
    pair = _representative_pair(events, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_paths, direct = _activity_and_readout(rows, pair, events, output_dir, candidate_id)
    kmeans_paths, kmeans = _kmeans_figure(events, training, output_dir)
    all_paths = direct_paths + kmeans_paths
    metadata = {
        "schema_id": "topic4_rev22_dci_final_candidate_acceptance_v1",
        "status": "FINAL_CANDIDATE_VISUAL_ACCEPTANCE_RENDERED",
        "candidate_id": candidate_id,
        "selection_role": "post-freeze visual acceptance only",
        "snn_simulation_run": False,
        "event_unit": "returned edge-supported causal family",
        "event_filter": "all returned families with at least three readable contacts",
        "direct_readout": direct,
        "kmeans": kmeans,
        "worker_hashes": {str(row["topology_seed"]): {
            "json": row["json_sha256"], "npz": row["npz_sha256"]} for row in rows},
        "input_hashes": {
            "config": _sha256(config_path), "validation": _sha256(validation_path),
            "frozen_candidates": _sha256(frozen_path),
            "candidate_manifest": _sha256(candidate_manifest_path),
            "patient_training_target": _sha256(_resolve(
                config["inputs"]["patient_training_target"], artifact_root)),
        },
        "output_sha256": {path.name: _sha256(path) for path in all_paths},
        "claim_boundary": (
            "Development-only visual acceptance of one frozen dual-core interictal candidate. "
            "KMeans/template agreement is selection-blind, not independent, and does not prove mechanism."
        ),
    }
    metadata_path = output_dir / "rev22_dci_final_candidate_acceptance_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "### rev22_dci_final_candidate_direct_readout.png\n\n"
        "冻结候选在同一张确认网络中的两类代表事件。左侧为逐 2 ms 保存的兴奋性神经元放电场，右侧为同一连续时间窗的 15 触点模型电流读出；红蓝阴影只表示自然 KMeans 对齐后的两种传播模式。GIF 与静态图使用同一 NPZ、同一事件窗，未重跑仿真。\n\n"
        "**关注点**：观察两类事件是否各自形成连续传播，而不是一个长事件被截成两段；并检查 SCL 招募是否来自可见的空间传播。\n\n"
        "### rev22_dci_final_candidate_kmeans.png\n\n"
        "同一冻结候选的全部可读 returned causal families。四块依次显示聚类事件热图、逐触点 rank 分布、模型与患者 rank profile、以及模型和患者模板的 Spearman 矩阵；矩阵星号来自杆内标签置换。\n\n"
        "**关注点**：双簇是否由多数网络共同支持、两条对角是否同时为正且交叉为负，以及改善是否伴随高 OOD 或模式塌缩。\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation", type=Path, default=STAGE / "validation/validation_aggregate.json")
    parser.add_argument("--frozen-candidates", type=Path,
                        default=STAGE / "response_fit/frozen_candidates.json")
    parser.add_argument("--candidate-manifest", type=Path,
                        default=STAGE / "response_fit/final_execution_candidate_manifest.json")
    parser.add_argument("--confirmation-workers", type=Path,
                        default=STAGE / "confirmation/workers")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=STAGE / "figures/final_candidate")
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    args = parser.parse_args()
    result = render(
        config_path=args.config, validation_path=args.validation,
        frozen_path=args.frozen_candidates, candidate_manifest_path=args.candidate_manifest,
        worker_dir=args.confirmation_workers, candidate_id=args.candidate_id,
        output_dir=args.output_dir, artifact_root=args.artifact_root,
    )
    print(json.dumps({"status": result["status"], "candidate_id": result["candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
