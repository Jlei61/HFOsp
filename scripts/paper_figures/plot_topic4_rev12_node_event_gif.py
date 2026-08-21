#!/usr/bin/env python3
"""Render algorithmically selected rev12 Node events as sheet/readout GIFs."""
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
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.topic4_node_dualmode import (  # noqa: E402
    event_features,
    normalize_event_ranks,
    shaft_balanced_feature_weights,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
MODE_COLORS = ("#C43C39", "#277DA1")
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bandpass(values: np.ndarray, dt_ms: float) -> np.ndarray:
    fs_hz = 1000.0 / float(dt_ms)
    sos = butter(3, [30.0, 80.0], btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(values, float), axis=1)


def _representative_returned_index(worker: dict, patient: dict,
                                   evaluable: np.ndarray, mode: int) -> int:
    selected = evaluable & (worker["labels"] == int(mode))
    supported = selected & ~worker["ood"]
    if np.any(supported):
        selected = supported
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise RuntimeError(f"no source-evaluable returned event for mode {mode + 1}")
    features = event_features(normalize_event_ranks(worker["ranks"][indices]))
    patient_features = event_features(
        normalize_event_ranks(patient["train_ranks"]),
    )
    prototype = patient_features[patient["train_labels"] == mode].mean(axis=0)
    weights = shaft_balanced_feature_weights(patient["contact_names"])
    distance = np.sum((features - prototype) ** 2 * weights, axis=1)
    return int(indices[int(np.argmin(distance))])


def _render_mode(*, npz_path: Path, worker_json: dict, patient: dict,
                 worker: dict, mode: int, output: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {"source_activity_counts", "source_activity_relative_ms"}
        if not required.issubset(loaded.files):
            raise RuntimeError("worker predates compact source-activity movie storage")
        names = np.asarray(loaded["contact_names"]).astype(str)
        shafts = np.asarray(loaded["shaft_ids"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        returned = np.asarray(loaded["event_returned"], bool)
        evaluable = np.asarray(loaded["source_onset_evaluable"], bool)[returned]
        returned_position = _representative_returned_index(
            worker, patient, evaluable, mode,
        )
        detected_index = int(np.flatnonzero(returned)[returned_position])
        activity = np.asarray(
            loaded["source_activity_counts"][detected_index], float,
        )
        relative = np.asarray(loaded["source_activity_relative_ms"], float)
        event_onset = float(loaded["event_t_on_ms"][detected_index])
        event_offset = float(loaded["event_t_off_ms"][detected_index])
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
    order = np.asarray([
        int(np.flatnonzero(names == name)[0]) for name in patient["contact_names"]
    ])
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]
    envelope = _bandpass(envelope[order], envelope_dt)
    trace_start = max(0.0, event_onset - 50.0)
    trace_stop = min(envelope.shape[1] * envelope_dt, event_onset + 100.0)
    sample_time = np.arange(envelope.shape[1]) * envelope_dt
    sample = (sample_time >= trace_start) & (sample_time <= trace_stop)
    trace_time = sample_time[sample] - event_onset
    traces = envelope[:, sample]
    scale = max(float(np.quantile(np.abs(traces), 0.995)), 1e-12)
    traces = traces / scale
    smoothed = np.asarray([gaussian_filter(frame, 0.7) for frame in activity])
    positive = smoothed[smoothed > 0]
    vmax = max(1.0, float(np.quantile(positive, 0.98)) if len(positive) else 1.0)

    fig = plt.figure(figsize=(12.2, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(0.92, 1.75), left=0.055, right=0.985,
        bottom=0.13, top=0.90, wspace=0.18,
    )
    ax_field, ax_trace = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    image = ax_field.imshow(
        smoothed[0], origin="lower", extent=(0, 20, 0, 20), cmap="viridis",
        vmin=0.0, vmax=vmax, interpolation="bilinear",
    )
    for shaft in ("ICL", "SCL"):
        mask = shafts == shaft
        ax_field.scatter(
            contact_xy[mask, 0], contact_xy[mask, 1], s=32,
            color=SHAFT_COLORS[shaft], edgecolor="white", linewidth=0.7,
        )
    ax_field.set(
        xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)",
        title=f"Model mode {mode + 1}",
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
    ax_trace.axvspan(
        0.0, event_offset - event_onset, color=MODE_COLORS[mode], alpha=0.10,
        lw=0,
    )
    cursor = ax_trace.axvline(relative[0], color="#202020", lw=1.0)
    ax_trace.set_yticks(offsets, names, fontsize=7.0)
    ax_trace.tick_params(axis="y", length=0, pad=3)
    ax_trace.set(
        xlim=(trace_time[0], trace_time[-1]), ylim=(-1.0, offsets[0] + 1.0),
        xlabel="time from event onset (ms)",
        ylabel="30-80 Hz virtual-contact activity",
    )
    ax_trace.spines[["top", "right", "left"]].set_visible(False)

    def update(frame: int):
        image.set_data(smoothed[frame])
        cursor.set_xdata([relative[frame], relative[frame]])
        time_label.set_text(f"t = {relative[frame]:+.0f} ms")
        return image, cursor, time_label

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(relative), interval=80, blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)
    return {
        "mode": mode + 1,
        "returned_event_position": returned_position,
        "detected_event_index": detected_index,
        "event_onset_ms": event_onset,
        "event_offset_ms": event_offset,
        "output": str(output),
        "output_sha256": _sha256(output),
        "worker_commit": worker_json["provenance"]["expected_git_commit"],
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
    json_path = npz_path.with_suffix(".json")
    worker_json = json.loads(json_path.read_text())
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
    figures = output_root / "figures" / stem
    records = [
        _render_mode(
            npz_path=npz_path, worker_json=worker_json, patient=patient,
            worker=worker, mode=mode,
            output=figures / f"{stem}_mode{mode + 1}.gif",
        )
        for mode in (0, 1)
    ]
    metadata = {
        "status": "REV12ND_NODE_EVENT_GIFS_COMPLETE",
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "event_selection": (
            "closest complete-event representation to the corresponding patient-training "
            "prototype among source-evaluable returned events; in-support preferred"
        ),
        "records": records,
        "not_clinical_seeg": True,
    }
    (figures / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (figures / "README.md").write_text(f"""### {stem}_mode1.gif

纯 Node field 在 seed {args.seed} 上算法选出的 model mode 1 returned event。左侧为每 2 ms 的 1 mm sheet activity counts，经固定空间平滑后用 `viridis` 显示；右侧为同一次事件的 15 个 virtual-contact 30-80 Hz readout。

**关注点**：看活动是从可重复局部区域传播，还是多处近同时点亮；本图不把 model-current 称为临床 SEEG。

### {stem}_mode2.gif

与上图相同，但展示 model mode 2。事件由患者训练 prototype 距离算法选择，不按视觉效果挑选。

**关注点**：比较两种模式的起始拓扑和扩展顺序，而不只比较右侧 contact rank。
""")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
