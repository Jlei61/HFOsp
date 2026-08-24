#!/usr/bin/env python3
"""Render complete population excursions without contact-defined cropping."""
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
    _representative_returned_index,
)
from scripts.rescore_topic4_rev12_node_historical import (  # noqa: E402
    _classifier_contract,
    _load_network_worker,
    _old_to_patient_label_map,
    _patient_data,
    _reorder_patient_contract,
)
from src.sef_hfo_events import detect_events  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_frame_indices(*, start_ms: float, stop_ms: float,
                          frame_ms: float, n_frames: int) -> np.ndarray:
    if not 0.0 <= start_ms < stop_ms or frame_ms <= 0.0 or n_frames <= 0:
        raise ValueError("invalid complete-excursion movie interval")
    first = max(0, int(np.floor(start_ms / frame_ms)))
    final = min(n_frames, int(np.ceil(stop_ms / frame_ms)) + 1)
    if final <= first:
        raise ValueError("complete excursion has no stored movie frames")
    return np.arange(first, final, dtype=int)


def detector_fragment_spans(active: np.ndarray, active_dt_ms: float, *,
                            event_threshold: float,
                            fragment_indices: list[int],
                            trigger_ms: float) -> list[tuple[float, float]]:
    fragments = detect_events(
        np.asarray(active, float), float(active_dt_ms),
        event_on_frac=float(event_threshold),
    )
    if any(index < 0 or index >= len(fragments) for index in fragment_indices):
        raise RuntimeError("population excursion references an absent detector fragment")
    return [
        (
            float(fragments[index]["t_on"]) - trigger_ms,
            float(fragments[index]["t_off"]) - trigger_ms,
        )
        for index in fragment_indices
    ]


def _render_mode(*, npz_path: Path, worker_json: dict, patient: dict,
                 worker: dict, mode: int, output: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as loaded:
        required = {
            "sheet_activity_counts", "sheet_activity_frame_ms",
            "active_fraction", "active_fraction_bin_ms", "contact_envelope",
            "contact_envelope_dt_ms", "event_returned", "event_t_on_ms",
            "event_trigger_t_on_ms", "event_t_off_ms",
        }
        if not required.issubset(loaded.files):
            raise RuntimeError("worker lacks the whole-run population-excursion movie")
        names = np.asarray(loaded["contact_names"]).astype(str)
        shafts = np.asarray(loaded["shaft_ids"]).astype(str)
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        returned = np.asarray(loaded["event_returned"], bool)
        evaluable = np.asarray(loaded["source_onset_evaluable"], bool)[returned]
        returned_position = _representative_returned_index(
            worker, patient, evaluable, mode,
        )
        detected_index = int(np.flatnonzero(returned)[returned_position])
        event_on = float(loaded["event_t_on_ms"][detected_index])
        trigger_on = float(loaded["event_trigger_t_on_ms"][detected_index])
        reset_start = float(loaded["event_t_off_ms"][detected_index])
        movie = np.asarray(loaded["sheet_activity_counts"], float)
        frame_ms = float(loaded["sheet_activity_frame_ms"])
        active = np.asarray(loaded["active_fraction"], float)
        active_dt = float(loaded["active_fraction_bin_ms"])
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
    event_row = worker_json["events"][detected_index]
    if (not event_row["returned"]
            or worker_json["event_unit"].get("name") not in {
                "population_excursion", "causal_population_excursion",
            }):
        raise RuntimeError("selected event is not a returned population excursion")
    reset_ms = float(worker_json["event_unit"]["reset_ms"])
    display_stop = min(movie.shape[0] * frame_ms, reset_start + reset_ms)
    frames = display_frame_indices(
        start_ms=event_on, stop_ms=display_stop,
        frame_ms=frame_ms, n_frames=len(movie),
    )
    activity = np.asarray([
        gaussian_filter(movie[index], 0.7) for index in frames
    ])
    relative_frames = frames * frame_ms - trigger_on
    positive = activity[activity > 0]
    vmax = max(1.0, float(np.quantile(positive, 0.98)) if len(positive) else 1.0)
    fragment_indices = [
        int(index) for index in event_row["detector_fragment_indices"]
    ]
    fragment_spans = detector_fragment_spans(
        active, active_dt,
        event_threshold=float(worker_json["event_unit"]["event_on_threshold"]),
        fragment_indices=fragment_indices, trigger_ms=trigger_on,
    )

    order = np.asarray([
        int(np.flatnonzero(names == name)[0]) for name in patient["contact_names"]
    ])
    names, shafts, contact_xy = names[order], shafts[order], contact_xy[order]
    envelope = _bandpass(envelope[order], envelope_dt)
    sample_time = np.arange(envelope.shape[1]) * envelope_dt
    sample = (sample_time >= event_on) & (sample_time <= display_stop)
    trace_time = sample_time[sample] - trigger_on
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
    for shaft in ("ICL", "SCL"):
        mask = shafts == shaft
        ax_field.scatter(
            contact_xy[mask, 0], contact_xy[mask, 1], s=32,
            color=SHAFT_COLORS[shaft], edgecolor="white", linewidth=0.7,
        )
    ax_field.set(
        xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)",
        title=f"Model pattern {mode + 1}: complete excursion",
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
    for left, right in fragment_spans:
        ax_trace.axvspan(left, right, color=MODE_COLORS[mode], alpha=0.10, lw=0)
    reset_relative = reset_start - trigger_on
    ax_trace.axvspan(
        reset_relative, display_stop - trigger_on,
        color="#B8B8B8", alpha=0.16, lw=0,
    )
    ax_trace.axvline(0.0, color=MODE_COLORS[mode], ls="--", lw=1.0)
    ax_trace.axvline(reset_relative, color="#606060", ls=":", lw=1.0)
    cursor = ax_trace.axvline(relative_frames[0], color="#202020", lw=1.0)
    ax_trace.set_yticks(offsets, names, fontsize=7.0)
    ax_trace.tick_params(axis="y", length=0, pad=3)
    ax_trace.set(
        xlim=(trace_time[0], trace_time[-1]), ylim=(-1.0, offsets[0] + 1.0),
        xlabel="time from first high-threshold crossing (ms)",
        ylabel="30-80 Hz virtual-contact activity",
    )
    ax_trace.spines[["top", "right", "left"]].set_visible(False)

    def update(frame: int):
        image.set_data(activity[frame])
        cursor.set_xdata([relative_frames[frame], relative_frames[frame]])
        time_label.set_text(f"t = {relative_frames[frame]:+.0f} ms")
        return image, cursor, time_label

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=80, blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=12.5), dpi=115)
    plt.close(fig)
    return {
        "mode": mode + 1,
        "returned_event_position": returned_position,
        "population_excursion_index": detected_index,
        "analysis_on_ms": event_on,
        "trigger_on_ms": trigger_on,
        "reset_start_ms": reset_start,
        "reset_end_ms": display_stop,
        "detector_fragment_indices": fragment_indices,
        "detector_fragment_spans_from_trigger_ms": fragment_spans,
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
    worker_json = json.loads(npz_path.with_suffix(".json").read_text())
    cohort = json.loads((ROOT / config["inputs"]["cohort_config"]["path"]).read_text())
    classifier_config = json.loads(
        (ROOT / config["inputs"]["classifier_config"]["path"]).read_text()
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
            output=figures / f"{stem}_pattern{mode + 1}_complete_excursion.gif",
        )
        for mode in (0, 1)
    ]
    metadata = {
        "status": "REV12ND_POPULATION_EXCURSION_GIFS_COMPLETE",
        "candidate_id": args.candidate_id,
        "seed": args.seed,
        "event_selection": (
            "patient-training prototype medoid among returned population excursions; "
            "in-support events preferred; contact geometry never sets boundaries"
        ),
        "records": records,
        "not_clinical_seeg": True,
    }
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "population_excursion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (figures / "README.md").write_text(f"""### {stem}_pattern1_complete_excursion.gif

纯 Node field 在 seed {args.seed} 上算法选出的 pattern 1 完整 population excursion。左侧从第一次高阈值 crossing 前的预卷积窗口一直播放到完整低状态 reset；右侧显示同一绝对时间窗的 15 个 virtual-contact readout。红色浅带是该 excursion 内的 detector fragments，灰色区是确认状态已恢复的低活动驻留。

**关注点**：检查多个局部波包究竟属于同一次未复位的递归过程，还是被完整 reset 分开的独立事件。

### {stem}_pattern2_complete_excursion.gif

与上图相同，但展示冻结 classifier 下的 pattern 2。事件由算法选取，不按局部动画是否像某个方向传播进行挑选。

**关注点**：只有在完整 excursion 上仍呈现不同传播顺序，才能继续讨论同一网络中的双模式。
""")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
