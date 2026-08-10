"""Render rev9 Fig4-style direct waveforms and KMeans mode diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt


ROOT = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/node_edge_factorial")
SUMMARY = ROOT / "factorial_summary.json"
ARRAYS = ROOT / "factorial_summary.npz"
OUT = ROOT / "figures"
ARMS = ("Null", "Node", "Edge", "Node+Edge")
MODE_COLORS = ("#C43C39", "#277DA1")
SHAFT_COLORS = ("#E67E22", "#159EAE", "#6A51A3", "#2A9D55")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _slug(arm):
    return arm.lower().replace("+", "_")


def _shaft(name):
    return "".join(character for character in str(name) if not character.isdigit())


def _event_order(curves, labels):
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    slopes = np.asarray([
        np.polyfit(np.arange(curves.shape[1]), curve, 1)[0]
        for curve in curves])
    return np.lexsort((slopes, labels))


def _window_bounds(t_on, t_off, start, stop, width=700.0):
    center = 0.5 * (float(t_on) + float(t_off))
    left = max(float(start), center - 0.5 * float(width))
    right = min(float(stop), left + float(width))
    left = max(float(start), right - float(width))
    return left, right


def _load_capture(summary, arm):
    representative = summary["arm_summaries"][arm]["representative"]
    capture_json = Path(representative["capture_json"])
    capture_npz = Path(representative["capture_npz"])
    payload = json.loads(capture_json.read_text())
    if payload["arrays"]["sha256"] != _sha256(capture_npz):
        raise RuntimeError(f"capture hash mismatch for {arm}")
    if payload["arm"] != arm or int(payload["seed"]) != int(representative["seed"]):
        raise RuntimeError(f"capture identity mismatch for {arm}")
    if not payload["capture_lfp"]:
        raise RuntimeError(f"representative capture lacks LFP for {arm}")
    canonical_json = Path(representative["worker_json"])
    canonical_payload = json.loads(canonical_json.read_text())
    canonical_npz = Path(canonical_payload["arrays"]["path"])
    if canonical_payload["arrays"]["sha256"] != _sha256(canonical_npz):
        raise RuntimeError(f"canonical worker hash mismatch for {arm}")
    with np.load(capture_npz, allow_pickle=False) as loaded:
        capture = {key: loaded[key] for key in loaded.files}
    with np.load(canonical_npz, allow_pickle=False) as loaded:
        canonical = {key: loaded[key] for key in loaded.files}
    for key in ("event_t_on_ms", "event_t_off_ms", "event_curves",
                "event_ranks", "active_fraction"):
        if not np.array_equal(capture[key], canonical[key], equal_nan=True):
            raise RuntimeError(f"LFP capture changed {key} for {arm}")
    return payload, capture


def _assignment_map(arrays, arm, seed):
    key = _slug(arm)
    selected = np.asarray(arrays[f"{key}_seed_ids"], int) == int(seed)
    return {
        int(local): dict(frozen=int(frozen), de_novo=int(de_novo), ood=bool(ood))
        for local, frozen, de_novo, ood in zip(
            np.asarray(arrays[f"{key}_local_event_indices"], int)[selected],
            np.asarray(arrays[f"{key}_frozen_labels"], int)[selected],
            np.asarray(arrays[f"{key}_de_novo_labels"], int)[selected],
            np.asarray(arrays[f"{key}_frozen_ood"], bool)[selected])
    }


def _prepare_waveforms(summary, arrays):
    prepared = {}
    amplitude_samples = []
    for arm in ARMS:
        payload, capture = _load_capture(summary, arm)
        times = np.asarray(capture["times"], float)
        raw = np.asarray(capture["lfp_trace"], float)
        dt = float(np.median(np.diff(times)))
        filtered = sosfiltfilt(
            butter(4, (30.0, 80.0), btype="bandpass",
                   fs=1000.0 / dt, output="sos"), raw, axis=0)
        assignments = _assignment_map(arrays, arm, payload["seed"])
        events = {int(row["local_event_index"]): row for row in payload["events"]}
        windows = {}
        representative = summary["arm_summaries"][arm]["representative"]
        for mode in (0, 1):
            local = representative["local_event_index_by_frozen_mode"][str(mode)]
            if local is None:
                windows[mode] = None
                continue
            event = events[int(local)]
            left, right = _window_bounds(
                event["t_on_ms"], event["t_off_ms"], times[0], times[-1])
            selected = (times >= left) & (times <= right)
            trace = filtered[selected]
            amplitude_samples.append(np.abs(trace).ravel())
            windows[mode] = dict(
                local_index=int(local), event=event,
                left=left, right=right, selected=selected,
                annotation=assignments[int(local)])
        prepared[arm] = dict(
            payload=payload, arrays=capture, times=times,
            filtered=filtered, assignments=assignments, events=events,
            windows=windows)
    combined = np.concatenate(amplitude_samples)
    scale = float(np.percentile(combined, 95)) if len(combined) else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return prepared, scale


def _plot_waveforms(summary, arrays, out_dir):
    prepared, scale = _prepare_waveforms(summary, arrays)
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 6.5,
        "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(
        4, 2, figsize=(14.5, 11.0), sharex=False, sharey=False,
        constrained_layout=True)
    display_per_unit = 0.62 / scale
    metadata_windows = {}
    for row_index, arm in enumerate(ARMS):
        row = prepared[arm]
        names = np.asarray(row["arrays"]["contact_names"]).astype(str)
        offsets = np.arange(len(names)) * 1.05
        shafts = sorted({_shaft(name) for name in names})
        metadata_windows[arm] = {}
        for mode in (0, 1):
            axis = axes[row_index, mode]
            window = row["windows"][mode]
            if window is None:
                axis.text(0.5, 0.5, "frozen mode absent",
                          transform=axis.transAxes, ha="center", va="center")
                axis.set_title(f"{arm} | frozen mode {chr(65 + mode)}")
                continue
            selected = window["selected"]
            t = row["times"][selected] - window["left"]
            trace = row["filtered"][selected] * display_per_unit
            for local, event in row["events"].items():
                annotation = row["assignments"].get(int(local))
                if annotation is None:
                    continue
                if event["t_off_ms"] < window["left"] or event["t_on_ms"] > window["right"]:
                    continue
                start = max(0.0, event["t_on_ms"] - window["left"])
                stop = min(window["right"] - window["left"],
                           event["t_off_ms"] - window["left"])
                axis.axvspan(
                    start, stop, facecolor=MODE_COLORS[annotation["frozen"]],
                    alpha=0.13, edgecolor="#222222" if annotation["ood"] else "none",
                    linewidth=0.45, hatch="//" if annotation["ood"] else None)
            for contact_index, name in enumerate(names):
                color = SHAFT_COLORS[shafts.index(_shaft(name)) % len(SHAFT_COLORS)]
                axis.plot(t, trace[:, contact_index] + offsets[contact_index],
                          color=color, lw=0.62, alpha=0.95)
            annotation = window["annotation"]
            state = "OOD" if annotation["ood"] else "in-distribution"
            axis.set_title(
                f"{arm} | frozen mode {chr(65 + mode)} | {state}", loc="left",
                color="#A33A2A" if annotation["ood"] else "#222222")
            axis.set_xlim(0.0, window["right"] - window["left"])
            axis.set_ylim(-0.65, offsets[-1] + 0.85)
            axis.set_yticks(offsets)
            axis.set_yticklabels(names if mode == 0 else [])
            axis.set_xlabel("time around event (ms)" if row_index == 3 else "")
            if mode == 0:
                axis.set_ylabel("contact")
            bar_x = 0.025 * (window["right"] - window["left"])
            bar_y = offsets[-1] + 0.08
            axis.plot([bar_x, bar_x], [bar_y - 0.62, bar_y],
                      color="#222222", lw=1.35, clip_on=False)
            if row_index == 0:
                axis.text(bar_x + 7.0, bar_y - 0.31,
                          f"{scale:.3g} model-current units",
                          fontsize=6.7, va="center")
            metadata_windows[arm][str(mode)] = dict(
                local_event_index=window["local_index"],
                start_ms=float(window["left"]), stop_ms=float(window["right"]),
                ood=bool(annotation["ood"]))
    axes[0, 1].legend(handles=[
        Patch(facecolor=MODE_COLORS[0], alpha=0.18, label="frozen A event"),
        Patch(facecolor=MODE_COLORS[1], alpha=0.18, label="frozen B event"),
        Patch(facecolor="white", edgecolor="#222222", hatch="//",
              label="frozen OOD"),
        *[Line2D([0], [0], color=SHAFT_COLORS[index], lw=1.6, label=shaft)
          for index, shaft in enumerate(sorted({_shaft(name) for name in
              np.asarray(prepared["Node"]["arrays"]["contact_names"]).astype(str)}))],
    ], frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.42))
    fig.suptitle(
        "rev9 node-edge factorization | direct 30-80 Hz electrode readout",
        fontsize=12.5, fontweight="bold")
    stem = out_dir / "rev9_factorial_direct_waveforms"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return dict(
        common_amplitude_scale=float(scale), windows=metadata_windows,
        amplitude_contract=(
            "one common 95th-percentile scale over all eight displayed windows; "
            "model-current proxy, not calibrated clinical voltage"))


def _plot_kmeans(summary, arrays, out_dir):
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(17.5, 10.2), constrained_layout=True)
    grid_spec = fig.add_gridspec(3, 4, height_ratios=(1.55, 1.0, 0.92))
    grid = np.asarray(arrays["grid"], float)
    patient = np.asarray(arrays["patient_train_mode_prototypes"], float)
    band_low = np.asarray(arrays["patient_block_band_low"], float)
    band_high = np.asarray(arrays["patient_block_band_high"], float)
    heatmap_image = None
    matrix_image = None
    heat_axes, matrix_axes = [], []
    for column, arm in enumerate(ARMS):
        key = _slug(arm)
        curves = np.asarray(arrays[f"{key}_curves"], float)
        ranks = np.asarray(arrays[f"{key}_normalized_ranks"], float)
        frozen = np.asarray(arrays[f"{key}_frozen_labels"], int)
        de_novo = np.asarray(arrays[f"{key}_de_novo_labels"], int)
        ood = np.asarray(arrays[f"{key}_frozen_ood"], bool)
        names = np.asarray(arrays[f"{key}_contact_names"]).astype(str)
        order = _event_order(curves, de_novo)

        axis = fig.add_subplot(grid_spec[0, column])
        heat_axes.append(axis)
        shown = np.ma.masked_invalid(ranks[order].T)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("#D9D9D9")
        heatmap_image = axis.imshow(
            shown, aspect="auto", interpolation="nearest", cmap=cmap,
            vmin=0.0, vmax=1.0)
        split = int(np.sum(de_novo[order] == 0))
        if 0 < split < len(order):
            axis.axvline(split - 0.5, color="white", lw=1.2)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names if column == 0 else [])
        axis.set_xlabel("events ordered by de novo KMeans")
        if column == 0:
            axis.set_ylabel("contact along axis")
        arm_summary = summary["arm_summaries"][arm]
        ami = arm_summary["de_novo"]["frozen_assignment_ami"]
        axis.set_title(
            f"{arm} | usable n={len(curves)}\n"
            f"frozen/de novo AMI={ami:.2f} | OOD={ood.mean():.0%}",
            fontweight="bold")
        strip = axis.inset_axes([0.0, 1.01, 1.0, 0.095])
        strip_values = np.vstack((de_novo[order], frozen[order], ood[order].astype(int)))
        strip_cmap = ListedColormap(
            [MODE_COLORS[0], MODE_COLORS[1], "#FFFFFF", "#222222"])
        display = strip_values.copy()
        display[2] = np.where(ood[order], 3, 2)
        strip.imshow(display, aspect="auto", interpolation="nearest",
                     cmap=strip_cmap, vmin=0, vmax=3)
        strip.set_yticks((0, 1, 2))
        strip.set_yticklabels(("K", "F", "O"), fontsize=5.8)
        strip.set_xticks([])
        for spine in strip.spines.values():
            spine.set_visible(False)

        axis = fig.add_subplot(grid_spec[1, column])
        prototypes = np.asarray(arrays[f"{key}_de_novo_prototypes"], float)
        for mode in (0, 1):
            axis.fill_between(
                grid, band_low[mode], band_high[mode],
                color=MODE_COLORS[mode], alpha=0.09, lw=0)
            axis.plot(grid, patient[mode], "--", color=MODE_COLORS[mode],
                      lw=1.25, label=f"patient {chr(65 + mode)}")
            axis.plot(grid, prototypes[mode], color=MODE_COLORS[mode],
                      lw=1.8, label=f"model {chr(65 + mode)}")
        axis.axhline(0.0, color="#BBBBBB", lw=0.55)
        axis.set_xlabel("shared axis (mm)")
        if column == 0:
            axis.set_ylabel("normalized rank curve")
        axis.set_title("de novo prototype vs patient", loc="left")
        if column == 3:
            axis.legend(frameon=False, ncol=2, fontsize=6.8,
                        loc="upper right")

        axis = fig.add_subplot(grid_spec[2, column])
        matrix_axes.append(axis)
        matrix = np.asarray(arrays[f"{key}_de_novo_similarity"], float)
        matrix_image = axis.imshow(
            matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
        for row in range(2):
            for col in range(2):
                value = float(matrix[row, col])
                axis.text(col, row, f"{value:+.2f}", ha="center", va="center",
                          fontsize=10, fontweight="bold",
                          color="white" if abs(value) > 0.58 else "#222222")
        axis.set_xticks((0, 1), ("patient A", "patient B"))
        axis.set_yticks((0, 1), ("model A", "model B") if column == 0 else ("", ""))
        matched = arm_summary["de_novo"].get("matched_mean")
        counts = arm_summary["de_novo"].get("cluster_counts")
        axis.set_title(
            f"KMeans vs patient | mean={matched:.2f} | n={counts[0]}/{counts[1]}",
            fontsize=8.2)
    heat_cbar = fig.colorbar(
        heatmap_image, ax=heat_axes, fraction=0.012, pad=0.008)
    heat_cbar.set_label("within-event rank")
    matrix_cbar = fig.colorbar(
        matrix_image, ax=matrix_axes, fraction=0.012, pad=0.008)
    matrix_cbar.set_label("Spearman rho")
    fig.suptitle(
        "rev9 node-edge factorization | de novo KMeans against frozen patient modes",
        fontsize=12.5, fontweight="bold")
    stem = out_dir / "rev9_factorial_kmeans_modes"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return dict(
        mode_strip="K=de novo KMeans; F=frozen classifier; O=black means frozen OOD",
        patient_band="recording-block patient-training variability from rev8 diagnostics")


def _write_readme(out_dir):
    text = """# Topic 4 rev9 node-edge factorial figures

### rev9_factorial_direct_waveforms.png

这张图直接比较 Null、Node、Edge 和 Node+Edge 四臂代表网络中的 frozen mode A/B 电极波形。所有面板使用同一 30–80 Hz filter 和同一幅度尺度；彩色背景是 frozen assignment，斜线表示该事件超出原 rev8 mode centroid 的 p99 距离。

**关注点**：Node/Node+Edge 是否出现可重复的两类传播窗口，以及 Null/Edge 的少量可用事件是否多数为 OOD，不能只看局部波形相似。

### rev9_factorial_kmeans_modes.png

这张图对四臂分别展示逐事件 rank heatmap、de novo KMeans prototype 与冻结 patient-training prototype，以及 2×2 Spearman 一致性矩阵。K/F/O 三行标注分别是 de novo 标签、frozen 标签和 frozen OOD；患者虚线周围的浅色带来自 recording-block variability。

**关注点**：两簇支持数、frozen/de novo AMI、OOD 比例和矩阵的正对角/负交叉应一起判断；稳定 KMeans 本身不等于匹配患者模式。
"""
    (out_dir / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(SUMMARY))
    parser.add_argument("--arrays", default=str(ARRAYS))
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text())
    if summary["arrays"]["sha256"] != _sha256(args.arrays):
        raise RuntimeError("factorial summary/array hash mismatch")
    with np.load(args.arrays, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    waveform = _plot_waveforms(summary, arrays, out_dir)
    kmeans = _plot_kmeans(summary, arrays, out_dir)
    _write_readme(out_dir)
    metadata = dict(
        status="REV9_FACTORIAL_FIG4_STYLE_COMPLETE",
        scientific_role=(
            "exploratory four-arm direct-waveform and KMeans diagnostics; "
            "not patient blind validation or Node-Edge equivalence"),
        source_summary=dict(path=args.summary, sha256=_sha256(args.summary)),
        source_arrays=dict(path=args.arrays, sha256=_sha256(args.arrays)),
        waveform=waveform, kmeans=kmeans,
        arms={arm: summary["arm_summaries"][arm] for arm in ARMS},
        git_commit=_git_commit(), producer_sha256=_sha256(__file__))
    (out_dir / "rev9_factorial_fig4_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(
        status=metadata["status"],
        direct_waveforms=str(out_dir / "rev9_factorial_direct_waveforms.png"),
        kmeans=str(out_dir / "rev9_factorial_kmeans_modes.png"),
        common_amplitude_scale=waveform["common_amplitude_scale"]), indent=2))


if __name__ == "__main__":
    main()
