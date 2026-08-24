#!/usr/bin/env python3
"""Render complete root-coactivity events with all sheet activity visible."""
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
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_topic4_rev12_node_event_gif import (  # noqa: E402
    MODE_COLORS,
    SHAFT_COLORS,
    _bandpass,
)
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.lagpat_rank_audit import build_masked_kmeans_features  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lineage_display_frames(*, event_on_ms: float, event_off_ms: float,
                           frame_ms: float, n_frames: int,
                           context_ms: float = 20.0) -> np.ndarray:
    """Return a bounded movie window around one complete event."""
    if not 0.0 <= event_on_ms < event_off_ms or frame_ms <= 0.0:
        raise ValueError("invalid directed-lineage display interval")
    start = max(0, int(np.floor((event_on_ms - context_ms) / frame_ms)))
    stop = min(n_frames, int(np.ceil((event_off_ms + context_ms) / frame_ms)) + 1)
    if stop <= start:
        raise RuntimeError("directed lineage has no stored movie frames")
    return np.arange(start, stop, dtype=int)


def lineage_bin_coordinates(labels: np.ndarray,
                            lineage_ids: int | list[int]) -> np.ndarray:
    """Return x/y bin centers for every root in one complete observation."""
    ids = np.atleast_1d(np.asarray(lineage_ids, int))
    coordinates = np.argwhere(np.isin(np.asarray(labels, int), ids))
    if not len(coordinates):
        return np.empty((0, 2), float)
    return np.column_stack((coordinates[:, 1] + 0.5, coordinates[:, 0] + 0.5))


def _representative_lineages(worker: dict, worker_json: dict,
                             source_evaluable: np.ndarray,
                             *, random_state: int) -> tuple[list[dict], dict]:
    """Select one medoid-like complete event from each natural KMeans cluster."""
    result = natural_kmeans(
        worker["ranks"], worker["labels"], random_state=int(random_state),
    )
    if result.get("status") != "OK":
        raise RuntimeError("natural KMeans is not evaluable for this network")
    valid = np.asarray(result["valid_event_mask"], bool)
    clusters = np.asarray(result["cluster_labels"], int)
    valid_positions = np.flatnonzero(valid)
    alignment = best_binary_alignment(clusters, worker["labels"][valid])
    mapped = np.asarray(alignment["mapped_labels"], int)

    ranks = np.asarray(worker["ranks"], float)[valid]
    features = build_masked_kmeans_features(
        ranks.T, np.isfinite(ranks.T), impute="event_median",
    )
    returned_detected = np.flatnonzero(np.asarray([
        bool(row["returned"]) for row in worker_json["events"]
    ]))
    if len(returned_detected) != len(worker["ranks"]):
        raise RuntimeError("returned-event order differs between JSON and NPZ")

    records = []
    for mode in (0, 1):
        local = np.flatnonzero(
            (mapped == mode) & source_evaluable[valid_positions]
        )
        if not len(local):
            raise RuntimeError(f"no source-evaluable natural cluster for pattern {mode + 1}")
        in_support = ~np.asarray(worker["ood"], bool)[valid_positions[local]]
        if np.any(in_support):
            local = local[in_support]
        centroid = np.mean(features[mapped == mode], axis=0)
        distance = np.sum((features[local] - centroid) ** 2, axis=1)

        candidates = []
        for local_index, feature_distance in zip(local, distance):
            returned_position = int(valid_positions[local_index])
            detected_index = int(returned_detected[returned_position])
            event = worker_json["events"][detected_index]
            candidates.append((
                float(feature_distance),
                float(event["maximum_fragment_collision_fraction"]),
                -int(event["n_recruited_contacts"]),
                detected_index,
                returned_position,
                int(clusters[local_index]),
            ))
        selected = min(candidates)
        records.append({
            "mode": mode,
            "returned_position": int(selected[4]),
            "detected_index": int(selected[3]),
            "natural_cluster_id": int(selected[5]),
            "feature_distance_to_cluster_centroid": float(selected[0]),
            "in_support_preferred": bool(np.any(in_support)),
        })
    audit = {
        "status": result["status"],
        "cluster_counts": result["cluster_counts"],
        "direction_purity": float(alignment["purity"]),
        "direction_balanced_alignment": float(alignment["balanced_alignment"]),
        "direction_contingency": np.asarray(alignment["contingency"]).tolist(),
        "kmeans_seed_ami_median": float(result["kmeans_seed_ami_median"]),
        "silhouette": float(result["silhouette"]),
    }
    return records, audit


def _render_lineage(*, npz_path: Path, worker_json: dict, patient: dict,
                    selection: dict, output: Path) -> dict:
    mode = int(selection["mode"])
    detected_index = int(selection["detected_index"])
    event = worker_json["events"][detected_index]
    lineage_ids = [
        int(value) for value in event.get("lineage_ids", [event["lineage_id"]])
    ]
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {
            "sheet_activity_counts", "sheet_activity_frame_ms",
            "directed_lineage_labels", "directed_lineage_collision_mask",
            "contact_envelope", "contact_envelope_dt_ms", "contact_names",
            "shaft_ids", "contact_xy_mm", "event_t_on_ms", "event_t_off_ms",
            "event_trigger_t_on_ms", "onsets",
        }
        if not required.issubset(loaded.files):
            raise RuntimeError("worker lacks directed-lineage movie arrays")
        movie = np.asarray(loaded["sheet_activity_counts"], float)
        frame_ms = float(loaded["sheet_activity_frame_ms"])
        lineage_labels = np.asarray(loaded["directed_lineage_labels"], int)
        collisions = np.asarray(loaded["directed_lineage_collision_mask"], bool)
        names = np.asarray(loaded["contact_names"]).astype(str)
        shafts = np.asarray(loaded["shaft_ids"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
        event_on = float(loaded["event_t_on_ms"][detected_index])
        event_off = float(loaded["event_t_off_ms"][detected_index])
        trigger_on = float(loaded["event_trigger_t_on_ms"][detected_index])
        contact_onsets = np.asarray(loaded["onsets"][detected_index], float)
    if worker_json["event_unit"].get("name") not in {
            "directed_spatiotemporal_lineage",
            "persistent_directed_spatiotemporal_lineage",
            "persistent_root_coactivity_episode"}:
        raise RuntimeError("GIF input is not the frozen directed event unit")
    if not bool(event["returned"]):
        raise RuntimeError("selected directed lineage is not returned")

    frames = lineage_display_frames(
        event_on_ms=event_on, event_off_ms=event_off,
        frame_ms=frame_ms, n_frames=len(movie),
    )
    activity = np.asarray([gaussian_filter(movie[index], 0.7) for index in frames])
    selected_labels = lineage_labels[frames]
    selected_collisions = collisions[frames]
    relative_frames = frames * frame_ms - event_on
    positive = activity[activity > 0]
    vmax = max(1.0, float(np.quantile(positive, 0.98)) if len(positive) else 1.0)

    order = np.asarray([
        int(np.flatnonzero(names == name)[0]) for name in patient["contact_names"]
    ])
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]
    contact_onsets = contact_onsets[order]
    envelope = _bandpass(envelope[order], envelope_dt)
    sample_time = np.arange(envelope.shape[1]) * envelope_dt
    trace_start = max(0.0, event_on - 20.0)
    trace_stop = min(envelope.shape[1] * envelope_dt, event_off + 20.0)
    sample = (sample_time >= trace_start) & (sample_time <= trace_stop)
    trace_time = sample_time[sample] - event_on
    traces = envelope[:, sample]
    scale = max(float(np.quantile(np.abs(traces), 0.995)), 1e-12)
    traces = traces / scale

    fig = plt.figure(figsize=(12.2, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.92, 1.75), left=0.055, right=0.985,
        bottom=0.13, top=0.90, wspace=0.18,
    )
    ax_field, ax_trace = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    image = ax_field.imshow(
        activity[0], origin="lower", extent=(0, 20, 0, 20), cmap="viridis",
        vmin=0.0, vmax=vmax, interpolation="bilinear",
    )
    root_outline = ax_field.scatter(
        [], [], s=30, facecolors="none", edgecolors="white", linewidths=0.8,
        label="complete event roots",
    )
    collision_marks = ax_field.scatter(
        [], [], s=18, marker="x", color="#D73027", linewidths=0.8,
        label="collision boundary",
    )
    for shaft in ("ICL", "SCL"):
        mask = shafts == shaft
        ax_field.scatter(
            contact_xy[mask, 0], contact_xy[mask, 1], s=32,
            color=SHAFT_COLORS[shaft], edgecolor="white", linewidth=0.7,
        )
    ax_field.set(
        xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)",
        title=f"Model pattern {mode + 1}",
    )
    ax_field.set_aspect("equal")
    ax_field.title.set_fontweight("bold")
    ax_field.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_field, fraction=0.046, pad=0.025)
    colorbar.set_label("active E neurons", fontsize=8)
    time_label = ax_field.text(
        0.02, 0.02, "", transform=ax_field.transAxes, color="white",
        fontsize=8.5, weight="bold", va="bottom",
    )

    offsets = np.arange(len(names))[::-1] * 1.45
    for row, offset in enumerate(offsets):
        ax_trace.plot(
            trace_time, 0.58 * traces[row] + offset,
            color=SHAFT_COLORS[shafts[row]], lw=1.0,
        )
    recruited = np.isfinite(contact_onsets)
    ax_trace.scatter(
        contact_onsets[recruited] - event_on, offsets[recruited],
        s=18, color=MODE_COLORS[mode], edgecolor="white", linewidth=0.45,
        zorder=5,
    )
    ax_trace.axvspan(
        0.0, event_off - event_on, color=MODE_COLORS[mode], alpha=0.10, lw=0,
    )
    ax_trace.axvline(
        trigger_on - event_on, color=MODE_COLORS[mode], ls="--", lw=1.0,
    )
    cursor = ax_trace.axvline(relative_frames[0], color="#202020", lw=1.0)
    ax_trace.set_yticks(offsets, names, fontsize=7.0)
    ax_trace.tick_params(axis="y", length=0, pad=3)
    ax_trace.set(
        xlim=(trace_time[0], trace_time[-1]), ylim=(-1.0, offsets[0] + 1.0),
        xlabel=(
            "time from causal-root onset (ms)"
            if worker_json["event_unit"]["name"] == "causal_root_observation"
            else "time from complete-event onset (ms)"
        ),
        ylabel="30-80 Hz virtual-contact activity",
    )
    ax_trace.spines[["top", "right", "left"]].set_visible(False)

    def update(frame: int):
        image.set_data(activity[frame])
        root_outline.set_offsets(lineage_bin_coordinates(
            selected_labels[frame], lineage_ids,
        ))
        collision = np.argwhere(selected_collisions[frame])
        collision_marks.set_offsets(
            np.column_stack((collision[:, 1] + 0.5, collision[:, 0] + 0.5))
            if len(collision) else np.empty((0, 2), float)
        )
        cursor.set_xdata([relative_frames[frame], relative_frames[frame]])
        time_label.set_text(f"t = {relative_frames[frame]:+.0f} ms")
        return image, root_outline, collision_marks, cursor, time_label

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)
    return {
        **selection,
        "lineage_ids": lineage_ids,
        "root_count": len(lineage_ids),
        "event_on_ms": event_on,
        "event_off_ms": event_off,
        "event_duration_ms": event_off - event_on,
        "detector_trigger_on_ms": trigger_on,
        "detector_fragment_indices": event["detector_fragment_indices"],
        "minimum_fragment_dominance": event["minimum_fragment_dominance"],
        "maximum_fragment_collision_fraction": event[
            "maximum_fragment_collision_fraction"
        ],
        "n_recruited_contacts": event["n_recruited_contacts"],
        "contact_onsets_ms": [
            None if not np.isfinite(value) else float(value)
            for value in contact_onsets
        ],
        "display_context_ms": 20.0,
        "all_sheet_activity_visible": True,
        "selected_lineage_encoding": "white open bin outline",
        "collision_encoding": "red x",
        "output": str(output),
        "output_sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(args.config.resolve().read_text())
    output_root = artifact_root / config["output_root"]
    stem = f"{args.candidate_id}_seed_{args.seed}"
    npz_path = output_root / "workers" / f"{stem}.npz"
    worker_json = json.loads(npz_path.with_suffix(".json").read_text())

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
    worker = _load_network_worker(
        npz_path, patient["contact_names"], classifier,
        semantics["raw_to_patient"],
    )
    with np.load(npz_path, allow_pickle=False) as loaded:
        returned = np.asarray(loaded["event_returned"], bool)
        source_evaluable = np.asarray(
            loaded["source_onset_evaluable"], bool,
        )[returned]
    selections, kmeans_audit = _representative_lineages(
        worker, worker_json, source_evaluable, random_state=args.seed,
    )
    figures = output_root / "figures" / stem
    causal_root = worker_json["event_unit"]["name"] == "causal_root_observation"
    suffix = "causal_root" if causal_root else "complete_event"
    records = [
        _render_lineage(
            npz_path=npz_path, worker_json=worker_json, patient=patient,
            selection=selection,
            output=figures / f"{stem}_pattern{selection['mode'] + 1}_{suffix}.gif",
        )
        for selection in selections
    ]
    metadata = {
        "status": (
            "REV12ND_CAUSAL_ROOT_GIFS_COMPLETE" if causal_root
            else "REV12ND_ROOT_COACTIVITY_GIFS_COMPLETE"
        ),
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "event_unit": worker_json["event_unit"],
        "selection": (
            "natural KMeans cluster representative nearest its cluster centroid; "
            "source-evaluable and in-support events preferred; no visual selection"
        ),
        "natural_kmeans": kmeans_audit,
        "records": records,
        "not_clinical_seeg": True,
    }
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if causal_root:
        readme = f"""### {stem}_pattern1_causal_root.gif

纯 Node field 在 seed {args.seed} 中由自然 KMeans 第一种传播模式自动选出的单根 causal event。左侧 `viridis` 底图保留同一时刻全场全部 E 神经元活动，白色空框只标用于读出的 causal root；其他热点不会被隐藏，但混合 detector 窗已作为 compound 留在 A/B 之外。右侧圆点是该 root 的 exact-neuron contact onset，阴影严格使用 latent root 起止，虚线只表示电极 detector 首次越阈。

**关注点**：检查白框活动是否形成一段连续传播，以及框外并发活动是否被诚实显示但未混入该模式。

### {stem}_pattern2_causal_root.gif

使用完全相同的因果根、接触点读出和自动选例合同，展示自然 KMeans 第二种传播模式；不按患者模板外观或动画效果挑选。

**关注点**：比较两种 root 的起点和传播方向，确认 A/B 不再来自同一长 detector 窗的不同片段。
"""
    else:
        readme = f"""### {stem}_pattern1_complete_event.gif

纯 Node field 在 seed {args.seed} 中由自然 KMeans 第一种传播模式自动选出的完整事件。左侧 `viridis` 底图显示同一时刻全场所有 E 神经元活动；白色空框标出该事件包含的全部根，红叉标记根相遇时无法唯一归属的边界。右侧保留全部 15 个 virtual-contact readout，彩色圆点标完整事件的 exact-neuron contact onset，阴影范围是完整事件窗口，虚线是原 detector 首次越阈时刻。

**关注点**：检查该模式是否真是一段完整传播，而不是从长窗口里截出的局部波包；同时检查多根是否属于同一次共激活事件。

### {stem}_pattern2_complete_event.gif

与上图使用完全相同的事件、颜色和时间合同，展示自然 KMeans 第二种传播模式。事件按簇中心距离自动选择，不按患者模板外观或动画效果挑选。

**关注点**：比较两种模式的根位置和传播方向；不能再把同一 detector 长窗口中互不相干的上下活动合并成一个双向事件。
"""
    (figures / "README.md").write_text(readme)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
