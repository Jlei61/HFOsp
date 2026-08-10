"""Render Fig4A for the frozen rev8 data-driven core-field candidate.

Plotting only: field, representative mode events, and direct 30--80 Hz virtual
SEEG all come from the hash-verified representative capture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.getcwd())


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CAPTURE = f"{ROOT}/joint_confirmation_rev8/representative_readout.json"
FIGDATA = f"{ROOT}/joint_confirmation_rev8/representative_figdata.npz"
CONFIRM = f"{ROOT}/joint_confirmation_rev8/final_confirmation.json"
ONSET_JSON = f"{ROOT}/joint_confirmation_rev8/all_event_onset_diagnostics.json"
ONSET_NPZ = f"{ROOT}/joint_confirmation_rev8/all_event_onset_diagnostics.npz"
OUT = "results/paper-ready-figure/fig4_data_driven_core_field_rev8/figures"
MODE_COLORS = ("#c43c39", "#277da1")
SHAFT_COLORS = ("#e67e22", "#159eae", "#6a51a3", "#2a9d55")
VERDICT_LABELS = {
    "RIGID_TEMPLATE_MATCH_NOT_BEATEN": "fails rigid-mode benchmark",
}


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _shaft(name):
    return "".join(character for character in str(name) if not character.isdigit())


def _verdict_label(verdict):
    return VERDICT_LABELS.get(str(verdict), str(verdict).lower().replace("_", " "))


def _plot_contacts(ax, contacts, names):
    shafts = sorted({_shaft(name) for name in names})
    for index, shaft in enumerate(shafts):
        selected = [i for i, name in enumerate(names) if _shaft(name) == shaft]
        xy = contacts[selected]
        color = SHAFT_COLORS[index % len(SHAFT_COLORS)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.0, alpha=0.8, zorder=5)
        ax.scatter(xy[:, 0], xy[:, 1], s=33, c=color, ec="white", lw=0.65, zorder=6)


def _style_sheet(ax, L, title, show_ylabel=False):
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_xlabel("sheet x (mm)")
    ax.set_ylabel("sheet y (mm)" if show_ylabel else "")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)


def _field_landscape_grid(pos, h, L, resolution=72):
    axis = np.linspace(0.0, float(L), int(resolution))
    xx, yy = np.meshgrid(axis, axis)
    zz = griddata(pos, h, (xx, yy), method="linear")
    if np.isnan(zz).any():
        nearest = griddata(pos, h, (xx, yy), method="nearest")
        zz = np.where(np.isfinite(zz), zz, nearest)
    return xx, yy, zz


def _plot_field(ax, data, onset_data=None):
    pos = np.asarray(data["posE"], float)
    h = np.asarray(data["h"], float)
    contacts = np.asarray(data["contacts"], float)
    names = [str(value) for value in data["names"]]
    reg = data["reg"].item()
    L = float(reg["L"])
    xx, yy, zz = _field_landscape_grid(pos, h, L)
    vmax = max(float(np.quantile(h, 0.995)), 1e-6)
    surface = ax.plot_surface(
        xx, yy, np.minimum(zz, vmax), cmap="plasma", vmin=0.0, vmax=vmax,
        linewidth=0, antialiased=True, shade=False, alpha=0.96,
        rasterized=True,
    )
    ax.contour(
        xx, yy, zz, zdir="z", offset=0.0, levels=8,
        cmap="plasma", vmin=0.0, vmax=vmax, linewidths=0.55, alpha=0.72,
    )
    contact_h = griddata(pos, h, contacts, method="linear")
    if np.isnan(contact_h).any():
        nearest = griddata(pos, h, contacts, method="nearest")
        contact_h = np.where(np.isfinite(contact_h), contact_h, nearest)
    shafts = sorted({_shaft(name) for name in names})
    for index, shaft in enumerate(shafts):
        selected = [i for i, name in enumerate(names) if _shaft(name) == shaft]
        xyz = contacts[selected]
        z = np.minimum(contact_h[selected], vmax) + 0.025 * vmax
        color = SHAFT_COLORS[index % len(SHAFT_COLORS)]
        ax.plot(xyz[:, 0], xyz[:, 1], z, color=color, lw=1.25, zorder=7)
        ax.scatter(
            xyz[:, 0], xyz[:, 1], z, s=23, c=color,
            edgecolor="white", linewidth=0.5, depthshade=False, zorder=8,
        )
    center = np.asarray(reg["center"], float)
    axis = np.asarray(reg["axis_unit"], float)
    endpoints = np.vstack((center - 9.5 * axis, center + 9.5 * axis))
    ax.plot(
        endpoints[:, 0], endpoints[:, 1], np.full(2, 1.06 * vmax),
        color="#222222", lw=1.0, ls=(0, (3, 3)), alpha=0.9, zorder=9,
    )
    if onset_data is not None and "component_centers" in onset_data.files:
        centers = np.asarray(onset_data["component_centers"], float)
        ax.scatter(
            centers[:, 0], centers[:, 1], np.full(len(centers), 1.09 * vmax),
            marker="x", s=42, c="white", linewidth=1.4, depthshade=False,
            zorder=10,
        )
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, 1.12 * vmax)
    ax.set_xlabel("sheet x (mm)", labelpad=5)
    ax.set_ylabel("sheet y (mm)", labelpad=5)
    ax.set_zlabel("h", labelpad=3)
    ax.set_title("data-driven field landscape", fontsize=11.5,
                 fontweight="bold", pad=7)
    ax.view_init(elev=31, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.58))
    ax.tick_params(labelsize=7.5, pad=1)
    cbar = plt.colorbar(surface, ax=ax, fraction=0.04, pad=0.01, shrink=0.74)
    cbar.set_label("pathology field h", fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _plot_signed_delta_vth(ax, data, show_ylabel=False):
    pos = np.asarray(data["posE"], float)
    delta = np.asarray(data["vth"], float) - 18.0
    contacts = np.asarray(data["contacts"], float)
    names = [str(value) for value in data["names"]]
    reg = data["reg"].item()
    vmax = max(float(np.quantile(np.abs(delta), 0.995)), 1e-6)
    image = ax.scatter(
        pos[:, 0], pos[:, 1], c=delta, s=3.0, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, lw=0, alpha=0.80, rasterized=True)
    _plot_contacts(ax, contacts, names)
    _style_sheet(ax, float(reg["L"]), r"actual $\Delta V_\theta=-h d$ (mV)",
                 show_ylabel=show_ylabel)
    cbar = plt.colorbar(image, ax=ax, fraction=0.047, pad=0.025)
    cbar.set_label(r"$V_{\theta,i}-V_{\theta,0}$ (mV)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    return dict(vmin=-vmax, vmax=vmax)


def _density_grid(onset_data, mode):
    density = np.asarray(onset_data["density"], float)[int(mode)]
    edges = np.asarray(onset_data["density_edges"], float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, centers, density


def _plot_mode_event(ax, data, onset_data, mode, show_ylabel=False):
    representative = data[f"mode_{mode}"].item()
    reg = data["reg"].item()
    if representative is None:
        ax.text(0.5, 0.5, "mode absent in representative network",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        _style_sheet(ax, float(reg["L"]), f"model mode {chr(65 + mode)}",
                     show_ylabel=show_ylabel)
        return None
    onset = np.asarray(representative["onset"], float)
    pos = np.asarray(data["posE"], float)
    finite = np.isfinite(onset)
    relative = onset[finite] - np.nanmin(onset[finite])
    scale = max(float(np.quantile(relative, 0.98)), 1.0)
    phase = np.clip(relative / scale, 0.0, 1.0)
    x_density, y_density, density = _density_grid(onset_data, mode)
    positive = density[density > 0]
    if len(positive):
        levels = np.unique(np.quantile(positive, (0.55, 0.75, 0.90)))
        if len(levels):
            ax.contour(
                x_density, y_density, density, levels=levels,
                colors=MODE_COLORS[mode], linewidths=(0.7, 1.0, 1.35)[:len(levels)],
                alpha=0.92, zorder=2)
    pos_all = np.asarray(data["posE"], float)
    h = np.asarray(data["h"], float)
    reg = data["reg"].item()
    xx, yy, hh = _field_landscape_grid(pos_all, h, float(reg["L"]), resolution=72)
    h_levels = np.unique(np.quantile(h, (0.80, 0.92, 0.98)))
    ax.contour(xx, yy, hh, levels=h_levels, colors="#222222",
               linestyles="--", linewidths=0.55, alpha=0.58, zorder=2)
    image = ax.scatter(
        pos[finite, 0], pos[finite, 1], c=phase, s=7.0,
        cmap="viridis", vmin=0.0, vmax=1.0, lw=0,
        alpha=0.80, rasterized=True,
    )
    contacts = np.asarray(data["contacts"], float)
    names = [str(value) for value in data["names"]]
    _plot_contacts(ax, contacts, names)
    event_modes = np.asarray(onset_data["event_modes"], int)
    source_centroids = np.asarray(onset_data["source_centroids"], float)
    all_sources = source_centroids[event_modes == mode]
    ax.scatter(all_sources[:, 0], all_sources[:, 1], marker="o", s=18,
               facecolor="none", edgecolor=MODE_COLORS[mode], lw=0.65,
               alpha=0.72, zorder=7)
    _style_sheet(ax, float(reg["L"]), f"model mode {chr(65 + mode)}",
                 show_ylabel=show_ylabel)
    return image


def _closest_mode_pair(events):
    mode_events = [event for event in events if event.get("mode") in (0, 1)]
    pairs = [
        (left, right)
        for i, left in enumerate(mode_events)
        for right in mode_events[i + 1:]
        if left["mode"] != right["mode"]
    ]
    if not pairs:
        return None
    return min(pairs, key=lambda pair: abs(
        0.5 * (pair[0]["t_on"] + pair[0]["t_off"])
        - 0.5 * (pair[1]["t_on"] + pair[1]["t_off"])))


def _nice_amplitude_scale(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0 ** exponent)
    nice = next(level for level in (1.0, 2.0, 5.0, 10.0) if fraction <= level)
    return float(nice * 10.0 ** exponent)


def _plot_readout(ax, data, events):
    pair = _closest_mode_pair(events)
    raw = np.asarray(data["lfp_trace"], float)
    times = np.asarray(data["times"], float)
    dt = float(np.median(np.diff(times)))
    filtered = sosfiltfilt(
        butter(4, (30.0, 80.0), btype="bandpass", fs=1000.0 / dt,
               output="sos"), raw, axis=0)
    if pair is None:
        start, stop = float(times[0]), min(float(times[-1]), float(times[0]) + 1200.0)
        displayed = []
    else:
        pair_start = min(event["t_on"] for event in pair)
        pair_stop = max(event["t_off"] for event in pair)
        width = max(1200.0, pair_stop - pair_start + 240.0)
        start = max(float(times[0]), 0.5 * (pair_start + pair_stop - width))
        stop = min(float(times[-1]), start + width)
        start = max(float(times[0]), stop - width)
        displayed = [
            event for event in events
            if event.get("mode") in (0, 1)
            and event["t_on"] >= start and event["t_off"] <= stop
        ]
    selected = (times >= start) & (times <= stop)
    trace = filtered[selected]
    t = times[selected] - start
    scale = _nice_amplitude_scale(np.percentile(np.abs(trace), 95))
    display_per_unit = 0.68 / scale
    trace = trace * display_per_unit
    names = [str(value) for value in data["names"]]
    offsets = np.arange(len(names)) * 1.22
    shafts = sorted({_shaft(name) for name in names})
    for event in displayed:
        ax.axvspan(event["t_on"] - start, event["t_off"] - start,
                   color=MODE_COLORS[int(event["mode"])], alpha=0.14, lw=0)
    for index, name in enumerate(names):
        color = SHAFT_COLORS[shafts.index(_shaft(name)) % len(SHAFT_COLORS)]
        ax.plot(t, trace[:, index] + offsets[index], color=color,
                lw=0.78, alpha=0.94)
    ax.set_xlim(0, float(t[-1])); ax.set_ylim(-0.7, offsets[-1] + 1.0)
    ax.set_yticks(offsets); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("simulation time (ms)")
    ax.set_ylabel("virtual-SEEG (30–80 Hz)")
    ax.set_title("direct electrode readout", fontsize=11.5,
                 fontweight="bold", pad=7, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    mode_legend = ax.legend(handles=[
        Patch(facecolor=MODE_COLORS[0], alpha=0.22, label="model mode A"),
        Patch(facecolor=MODE_COLORS[1], alpha=0.22, label="model mode B"),
    ], frameon=False, ncol=2, fontsize=8.5, loc="lower right")
    ax.add_artist(mode_legend)
    ax.legend(handles=[
        Line2D([0], [0], color=SHAFT_COLORS[index], lw=1.8, label=shaft)
        for index, shaft in enumerate(shafts)
    ], title="contact family", frameon=False, ncol=min(2, len(shafts)),
       fontsize=8.2, title_fontsize=8.2, loc="upper right")
    bar_x = float(t[-1]) * 0.025
    bar_y = offsets[-1] + 0.05
    ax.plot([bar_x, bar_x], [bar_y - 0.68, bar_y], color="#222222", lw=1.5,
            clip_on=False)
    ax.text(bar_x + 0.012 * float(t[-1]), bar_y - 0.34,
            f"{scale:g} mV\nmodel-current proxy", ha="left", va="center",
            fontsize=7.7, color="#222222")
    return dict(
        start_ms=float(start), stop_ms=float(stop),
        displayed_mode_counts={
            str(mode): int(sum(event["mode"] == mode for event in displayed))
            for mode in (0, 1)
        },
        contains_both_modes=bool({event["mode"] for event in displayed} == {0, 1}),
        common_amplitude_scale_mV=float(scale),
        amplitude_contract=(
            "one common scale for every contact; LFP is the model current proxy, "
            "not calibrated clinical SEEG voltage"),
    )


def _write_readme(out_dir):
    path = os.path.join(out_dir, "README.md")
    existing = open(path).read() if os.path.exists(path) else "# Fig. 4 data-driven core-field rev8.1\n\n"
    entry = """### fig4a_data_driven_core_field_waveforms

这张图使用最终冻结候选的同一代表网络：左侧同时显示优化得到的 h envelope 和神经元实际承受的 signed ΔVθ；中间的代表传播事件叠加了全部 50 个 final events 的 event-equal earliest-activation density 与 h 等高线；右侧是未经模板平均、所有 contact 共用一个幅度标尺的 30–80 Hz model-current readout。阴影只表示无监督模式身份，不预先当作 forward/reverse 标签。

**关注点**：两个模式是否都在同一网络中出现、空间传播是否不同，以及直接电极波形是否支持而不是掩盖 KMeans 结论。

"""
    if "### fig4a_data_driven_core_field_waveforms" not in existing:
        with open(path, "w") as handle:
            handle.write(existing + entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", default=CAPTURE)
    parser.add_argument("--figdata", default=FIGDATA)
    parser.add_argument("--confirmation", default=CONFIRM)
    parser.add_argument("--onset-json", default=ONSET_JSON)
    parser.add_argument("--onset-diagnostics", default=ONSET_NPZ)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    capture = json.load(open(args.capture))
    confirmation = json.load(open(args.confirmation))
    if capture["figdata"]["sha256"] != _sha256(args.figdata):
        raise RuntimeError("capture/figdata hash mismatch")
    if capture["input_confirmation"]["sha256"] != _sha256(args.confirmation):
        raise RuntimeError("capture/confirmation hash mismatch")
    onset_summary = json.load(open(args.onset_json))
    if onset_summary["arrays"]["sha256"] != _sha256(args.onset_diagnostics):
        raise RuntimeError("onset-summary/arrays hash mismatch")
    if onset_summary["input_profiles"]["sha256"] != confirmation["event_profiles"]["sha256"]:
        raise RuntimeError("onset diagnostics use a different final event pool")
    data = np.load(args.figdata, allow_pickle=True)
    onset_data = np.load(args.onset_diagnostics)

    fig = plt.figure(figsize=(22.0, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1.12, 1.0, 1.0, 1.0, 2.35),
        left=0.038, right=0.992, bottom=0.15, top=0.88, wspace=0.23)
    axes = [fig.add_subplot(grid[0, 0], projection="3d")]
    axes.extend(fig.add_subplot(grid[0, index]) for index in range(1, 5))
    _plot_field(axes[0], data, onset_data)
    delta_stats = _plot_signed_delta_vth(axes[1], data)
    image_a = _plot_mode_event(axes[2], data, onset_data, 0)
    image_b = _plot_mode_event(axes[3], data, onset_data, 1)
    image = image_b if image_b is not None else image_a
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[2:4], fraction=0.025, pad=0.025)
        colorbar.set_ticks((0.0, 1.0)); colorbar.set_ticklabels(("early", "late"))
        colorbar.set_label("relative firing onset", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    readout_stats = _plot_readout(axes[4], data, capture["events"])
    verdict = confirmation["candidates"][0]["confirm"]["verdict"]
    fig.suptitle(
        f"Data-driven core field: direct model readout  |  {_verdict_label(verdict)}",
        fontsize=13.0, fontweight="bold", y=0.985)

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "fig4a_data_driven_core_field_waveforms")
    fig.savefig(stem + ".png", dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    metadata = dict(
        figure="Fig4A data-driven core-field direct waveforms",
        plotting_only=True,
        input_capture=dict(path=args.capture, sha256=_sha256(args.capture)),
        input_figdata=dict(path=args.figdata, sha256=_sha256(args.figdata)),
        input_confirmation=dict(path=args.confirmation, sha256=_sha256(args.confirmation)),
        input_onset_summary=dict(path=args.onset_json, sha256=_sha256(args.onset_json)),
        input_onset_diagnostics=dict(
            path=args.onset_diagnostics, sha256=_sha256(args.onset_diagnostics)),
        candidate_id=capture["candidate_id"], seed=capture["seed"],
        mode_counts=capture["mode_counts"], verdict=verdict,
        direct_readout=readout_stats, signed_delta_vth=delta_stats,
        all_event_onset=dict(
            n_events=int(onset_summary["n_events"]),
            mode_counts=onset_summary["mode_counts"],
            probability_component_given_mode=onset_summary[
                "probability_component_given_mode"],
            contract=onset_summary["earliest_activation_contract"]),
        claim_boundary=(
            "single-subject final-confirmation rendering; mode shading is an "
            "unsupervised readout identity, not a causal direction label"),
    )
    with open(stem + "_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    _write_readme(args.out)
    print(f"wrote {stem}.png / .pdf / _metadata.json")


if __name__ == "__main__":
    main()
