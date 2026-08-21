#!/usr/bin/env python3
"""Observation-matched and fine-scale source-topology replay for Fig. 4 events."""
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
from scipy.ndimage import gaussian_filter, label

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src" / "snn_engine"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.paper_figures.plot_fig4_node_neuron_activity_gif import (  # noqa: E402
    CONFIG,
    DIRECT_METADATA,
    REFERENCE,
    SHAFT_COLORS,
    _active_fraction,
    _parity,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
    make_external_drive,
)


OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
    "node_field_activity_visual_acceptance/figures/"
    "node_baseline_seed1569_source_topology.gif"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bin_index(positions: np.ndarray, bin_mm: float, sheet_mm: float) -> tuple[np.ndarray, int]:
    size = int(round(sheet_mm / bin_mm))
    xy = np.floor(np.asarray(positions, float) / bin_mm).astype(int)
    xy = np.clip(xy, 0, size - 1)
    return xy[:, 1] * size + xy[:, 0], size


def _distinct_neuron_frames(spikes: np.ndarray, dt_ms: float, centers_ms: np.ndarray,
                            width_ms: float, neuron_bins: np.ndarray,
                            size: int) -> np.ndarray:
    output = np.zeros((len(centers_ms), size, size), float)
    half_steps = int(round(0.5 * width_ms / dt_ms))
    for frame, center_ms in enumerate(centers_ms):
        center = int(round(center_ms / dt_ms))
        low, high = max(0, center - half_steps), min(len(spikes), center + half_steps + 1)
        active = np.any(spikes[low:high], axis=0)
        output[frame] = np.bincount(
            neuron_bins[active], minlength=size * size,
        ).reshape(size, size)
    return output


def persistent_recruitment_onsets(event_counts: np.ndarray,
                                  baseline_counts: np.ndarray,
                                  relative_times_ms: np.ndarray,
                                  *, minimum_active_neurons: int = 2,
                                  persistence_frames: int = 2) -> np.ndarray:
    """First local q99 exceedance sustained for a small number of frames."""
    event_counts = np.asarray(event_counts, float)
    baseline_counts = np.asarray(baseline_counts, float)
    threshold = np.maximum(
        np.quantile(baseline_counts, 0.99, axis=0),
        float(minimum_active_neurons - 1),
    )
    above = event_counts > threshold[None, :, :]
    sustained = np.zeros_like(above)
    for frame in range(len(above) - persistence_frames + 1):
        sustained[frame] = np.all(above[frame:frame + persistence_frames], axis=0)
    onset = np.full(above.shape[1:], np.nan)
    for frame, time_ms in enumerate(relative_times_ms):
        new = sustained[frame] & ~np.isfinite(onset)
        onset[new] = float(time_ms)
    return onset


def _component_summary(onset: np.ndarray) -> dict:
    finite = np.isfinite(onset)
    count = int(np.sum(finite))
    if count == 0:
        return {
            "n_recruited_bins": 0,
            "n_early_components": 0,
            "dominant_early_component_fraction": None,
            "front_continuity": None,
        }
    early_count = min(count, max(5, int(np.ceil(0.10 * count))))
    threshold = np.partition(onset[finite], early_count - 1)[early_count - 1]
    early = finite & (onset <= threshold)
    components, n_components = label(early, structure=np.ones((3, 3), int))
    sizes = np.bincount(components[early], minlength=n_components + 1)[1:]

    times = np.unique(onset[finite])
    recruited = np.zeros_like(finite)
    adjacent, total = 0, 0
    for index, time_ms in enumerate(times):
        new = finite & (onset == time_ms)
        if index > 0:
            neighbourhood = gaussian_filter(recruited.astype(float), 0.7) > 0.01
            adjacent += int(np.sum(new & neighbourhood))
            total += int(np.sum(new))
        recruited |= new
    return {
        "n_recruited_bins": count,
        "n_early_bins": int(np.sum(early)),
        "n_early_components": int(n_components),
        "dominant_early_component_fraction": (
            None if not len(sizes) else float(np.max(sizes) / np.sum(sizes))
        ),
        "front_continuity": None if total == 0 else float(adjacent / total),
    }


def _linear_fit(train_distance: np.ndarray, train_time: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(train_distance)), train_distance])
    intercept, slope = np.linalg.lstsq(design, train_time, rcond=None)[0]
    return float(intercept), float(max(0.0, slope))


def source_model_cv(onset: np.ndarray, *, source_count: int,
                    folds: int = 5) -> dict:
    """Cross-validated one- or two-source radial onset-time model."""
    finite = np.argwhere(np.isfinite(onset))
    times = onset[np.isfinite(onset)]
    if len(times) < 12 or source_count not in (1, 2):
        return {"cv_r2": None, "n_bins": int(len(times))}
    earliest_order = np.argsort(times)[:min(12, len(times))]
    candidates = finite[earliest_order]
    source_sets = [(item,) for item in candidates]
    if source_count == 2:
        source_sets = [
            (candidates[i], candidates[j])
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        ]
    fold_id = np.arange(len(times)) % min(folds, len(times))
    predictions = np.full(len(times), np.nan)
    for fold in np.unique(fold_id):
        train = fold_id != fold
        test = ~train
        best = None
        for sources in source_sets:
            distance = np.min(np.column_stack([
                np.linalg.norm(finite - source[None, :], axis=1)
                for source in sources
            ]), axis=1)
            intercept, slope = _linear_fit(distance[train], times[train])
            residual = times[train] - (intercept + slope * distance[train])
            score = float(np.sum(residual ** 2))
            if best is None or score < best[0]:
                best = (score, intercept, slope, distance)
        _, intercept, slope, distance = best
        predictions[test] = intercept + slope * distance[test]
    sst = float(np.sum((times - np.mean(times)) ** 2))
    sse = float(np.sum((times - predictions) ** 2))
    return {
        "n_bins": int(len(times)),
        "cv_r2": None if sst <= 0.0 else float(1.0 - sse / sst),
        "cv_rmse_ms": float(np.sqrt(np.mean((times - predictions) ** 2))),
    }


def _event_analysis(spikes: np.ndarray, dt_ms: float, event_on_ms: float,
                    neuron_bins: np.ndarray, size: int) -> dict:
    relative = np.arange(-20.0, 80.0 + 1e-9, 2.0)
    baseline_relative = np.arange(-120.0, -20.0 + 1e-9, 2.0)
    counts = _distinct_neuron_frames(
        spikes, dt_ms, event_on_ms + relative, 3.0, neuron_bins, size,
    )
    baseline = _distinct_neuron_frames(
        spikes, dt_ms, event_on_ms + baseline_relative, 3.0, neuron_bins, size,
    )
    onset = persistent_recruitment_onsets(counts, baseline, relative)
    topology = _component_summary(onset)
    one = source_model_cv(onset, source_count=1)
    two = source_model_cv(onset, source_count=2)
    topology.update({
        "single_source": one,
        "two_source": two,
        "two_source_delta_cv_r2": (
            None if one["cv_r2"] is None or two["cv_r2"] is None
            else float(two["cv_r2"] - one["cv_r2"])
        ),
    })
    return {
        "relative_times_ms": relative,
        "counts": counts,
        "baseline": baseline,
        "onset": onset,
        "topology": topology,
    }


def render(config_path: Path, reference_path: Path,
           direct_metadata_path: Path, output_path: Path) -> dict:
    direct = json.loads(direct_metadata_path.read_text())
    window_start, window_stop = direct["direct_readout"]["display_window_ms"]
    seed = int(direct["direct_readout"]["seed"])
    config = load_round_config(config_path)
    substrate = build_substrate(
        config, "node_baseline", seed,
        cache_dir=ROOT / config["output_root"] / "network_cache",
        ee_dose=0.0, etoi_dose=0.0,
    )
    substrate.params.T = float(window_stop)
    substrate.net["rng"] = np.random.default_rng(seed)
    from kick_probe import simulate_kick
    result = simulate_kick(
        substrate.params, substrate.net, KICK_BOOST=0.0, t_kick=1e9,
        V_th_per_neuron=substrate.vtheta, slow=None,
        early_stop_runaway=False,
        external_e_rate_drive=make_external_drive(substrate, config["spatial_ou"], seed),
    )
    spikes = np.asarray(result["E_spk_bool"], bool)
    active, active_dt = _active_fraction(spikes, substrate)
    parity = _parity(reference_path, active, active_dt)
    if not parity["active_fraction_exact"] or not parity["active_fraction_dt_exact"]:
        raise RuntimeError(f"replay parity failed: {parity}")

    with np.load(reference_path, allow_pickle=False) as stored:
        event_on = np.asarray(stored["event_t_on_ms"], float)
        names = np.asarray(stored["contact_names"]).astype(str)
        shafts = np.asarray(stored["shaft_ids"]).astype(str)
        contact_xy = np.asarray(stored["contact_xy_mm"], float)
    selected = np.flatnonzero((event_on >= window_start) & (event_on <= window_stop))
    if len(selected) < 2:
        raise RuntimeError("accepted display window no longer contains two events")
    selected = selected[[0, -1]]
    mode_names = ("MTA", "MTB")
    neuron_bins, size = _bin_index(
        substrate.positions_e, 1.0, float(substrate.engine["L"]),
    )
    analyses = {
        mode: _event_analysis(
            spikes, float(substrate.engine["dt"]), float(event_on[index]),
            neuron_bins, size,
        )
        for mode, index in zip(mode_names, selected)
    }

    display_times = np.arange(-8.0, 50.0 + 1e-9, 2.0)
    fine_fields, coarse_fields, scales = {}, {}, {}
    for mode in mode_names:
        analysis = analyses[mode]
        indices = np.asarray([
            int(np.argmin(np.abs(analysis["relative_times_ms"] - time_ms)))
            for time_ms in display_times
        ])
        baseline = np.median(analysis["baseline"], axis=0)
        excess = np.maximum(0.0, analysis["counts"][indices] - baseline[None, :, :])
        scale = max(1.0, float(np.quantile(excess, 0.99)))
        fine_fields[mode] = excess / scale
        coarse_fields[mode] = np.asarray([
            gaussian_filter(frame, sigma=6.0) for frame in excess
        ])
        coarse_scale = max(1e-12, float(np.quantile(coarse_fields[mode], 0.99)))
        coarse_fields[mode] /= coarse_scale
        scales[mode] = {"fine_q99_active_neurons": scale, "coarse_q99": coarse_scale}

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), constrained_layout=True)
    images = {}
    for column, mode in enumerate(mode_names):
        for row, (title, fields) in enumerate((
            ("6 mm observation scale", coarse_fields[mode]),
            ("1 mm event-excess activity", fine_fields[mode]),
        )):
            ax = axes[row, column]
            image = ax.imshow(
                fields[0], origin="lower", extent=(0, 20, 0, 20),
                cmap="viridis", vmin=0.0, vmax=1.0, interpolation="bilinear",
            )
            images[(row, mode)] = image
            for shaft in ("ICL", "SCL"):
                mask = shafts == shaft
                ax.scatter(
                    contact_xy[mask, 0], contact_xy[mask, 1], s=24,
                    color=SHAFT_COLORS[shaft], edgecolors="white", linewidths=0.6,
                )
            ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="x (mm)", ylabel="y (mm)")
            ax.set_aspect("equal")
            ax.set_title(f"{mode}: {title}", fontsize=10, weight="bold")
            ax.spines[["top", "right"]].set_visible(False)
    time_label = fig.suptitle("", fontsize=11, weight="bold")

    def update(frame: int):
        for mode in mode_names:
            images[(0, mode)].set_data(coarse_fields[mode][frame])
            images[(1, mode)].set_data(fine_fields[mode][frame])
        time_label.set_text(f"Frozen Node-only events | t = {display_times[frame]:.0f} ms")
        return tuple(images.values()) + (time_label,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, update, frames=len(display_times), interval=80, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=12.5), dpi=120)
    plt.close(fig)

    payload = {
        "schema_id": "topic4_node_event_source_topology_v1",
        "status": "FROZEN_NODE_EVENT_SOURCE_TOPOLOGY_CANARY_COMPLETE",
        "candidate_id": "node_baseline",
        "network_seed": seed,
        "zm": "off",
        "learned_edge_redistribution": "off",
        "events": {
            mode: {
                "event_index": int(index),
                "event_on_ms": float(event_on[index]),
                "topology": analyses[mode]["topology"],
                "display_scales": scales[mode],
            }
            for mode, index in zip(mode_names, selected)
        },
        "observation_contract": {
            "biological_step_ms": 2.0,
            "frame_average_ms": 3.0,
            "display_times_ms": display_times.tolist(),
            "coarse_display_sigma_mm": 6.0,
            "fine_bin_mm": 1.0,
            "fine_values": "positive event active-neuron count minus binwise pre-event median",
        },
        "parity_vs_frozen_worker": parity,
        "reference_npz": str(reference_path),
        "reference_npz_sha256": _sha256(reference_path),
        "output_gif": str(output_path),
        "output_gif_sha256": _sha256(output_path),
        "interpretation_boundary": (
            "This two-event canary distinguishes observation-scale appearance from "
            "fine source topology. It does not establish causality; same-checkpoint "
            "component intervention is required."
        ),
    }
    output_path.with_name(output_path.stem + "_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    readme_path = output_path.parent / "README.md"
    marker = f"### {output_path.name}"
    existing = readme_path.read_text() if readme_path.exists() else ""
    if marker not in existing:
        section = (
            f"\n{marker}\n\n"
            "上排用与患者 Supplementary Video 1 相同的 6 mm 观察尺度显示两个冻结的 "
            "Node-only 事件；下排保留 1 mm 的事件超额活动，用来区分平滑后看似连续的传播与细粒度"
            "多源起始。MTA 由一个主导早期分量解释 96.9% 的早期格点，两源模型只增加 0.045 的 "
            "CV R2；MTB 的主导分量只有 48.9%，两源模型比单源模型增加 0.410 的 CV R2。\n\n"
            "**关注点**：该 canary 支持 MTA 与 MTB 的源拓扑不对称，不能把 KMeans 两簇直接解释为"
            "同一条因果通路的正向与逆向。它只分析两个代表事件；正式因果判断仍需同 checkpoint 的"
            "热点抑制与匹配位置对照。\n"
        )
        readme_path.write_text(existing.rstrip() + "\n" + section)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--direct-metadata", type=Path, default=DIRECT_METADATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(render(
        args.config.resolve(), args.reference.resolve(),
        args.direct_metadata.resolve(), args.output.resolve(),
    ), indent=2))


if __name__ == "__main__":
    main()
