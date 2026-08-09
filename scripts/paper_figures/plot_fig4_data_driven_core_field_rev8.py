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
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.getcwd())


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
CAPTURE = f"{ROOT}/joint_confirmation_rev8/representative_readout.json"
FIGDATA = f"{ROOT}/joint_confirmation_rev8/representative_figdata.npz"
CONFIRM = f"{ROOT}/joint_confirmation_rev8/final_confirmation.json"
OUT = "results/paper-ready-figure/fig4_data_driven_core_field_rev8/figures"
MODE_COLORS = ("#c43c39", "#277da1")
SHAFT_COLORS = ("#e67e22", "#159eae", "#6a51a3", "#2a9d55")


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _shaft(name):
    return "".join(character for character in str(name) if not character.isdigit())


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


def _plot_field(ax, data):
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


def _plot_mode_event(ax, data, mode, show_ylabel=False):
    representative = data[f"mode_{mode}"].item()
    reg = data["reg"].item()
    if representative is None:
        ax.text(0.5, 0.5, "mode absent in representative network",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        _style_sheet(ax, float(reg["L"]), f"patient mode {chr(65 + mode)}",
                     show_ylabel=show_ylabel)
        return None
    onset = np.asarray(representative["onset"], float)
    pos = np.asarray(data["posE"], float)
    finite = np.isfinite(onset)
    relative = onset[finite] - np.nanmin(onset[finite])
    scale = max(float(np.quantile(relative, 0.98)), 1.0)
    phase = np.clip(relative / scale, 0.0, 1.0)
    image = ax.scatter(
        pos[finite, 0], pos[finite, 1], c=phase, s=7.0,
        cmap="viridis", vmin=0.0, vmax=1.0, lw=0,
        alpha=0.80, rasterized=True,
    )
    contacts = np.asarray(data["contacts"], float)
    names = [str(value) for value in data["names"]]
    _plot_contacts(ax, contacts, names)
    early = finite.copy()
    early[finite] = relative <= np.quantile(relative, 0.01)
    source = pos[early].mean(axis=0)
    ax.scatter(source[0], source[1], marker="*", s=170, c=MODE_COLORS[mode],
               ec="white", lw=0.9, zorder=8)
    _style_sheet(ax, float(reg["L"]), f"patient mode {chr(65 + mode)}",
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
    scale = np.percentile(np.abs(trace), 95, axis=0)
    positive = scale[scale > 1e-12]
    floor = 0.15 * float(np.median(positive)) if len(positive) else 1.0
    trace = 0.68 * trace / np.maximum(scale, floor)[None, :]
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
    ax.legend(handles=[
        Patch(facecolor=MODE_COLORS[0], alpha=0.22, label="patient mode A"),
        Patch(facecolor=MODE_COLORS[1], alpha=0.22, label="patient mode B"),
    ], frameon=False, ncol=2, fontsize=9, loc="upper right")
    return dict(
        start_ms=float(start), stop_ms=float(stop),
        displayed_mode_counts={
            str(mode): int(sum(event["mode"] == mode for event in displayed))
            for mode in (0, 1)
        },
        contains_both_modes=bool({event["mode"] for event in displayed} == {0, 1}),
    )


def _write_readme(out_dir):
    path = os.path.join(out_dir, "README.md")
    existing = open(path).read() if os.path.exists(path) else "# Fig. 4 data-driven core-field rev8\n\n"
    entry = """### fig4a_data_driven_core_field_waveforms

这张图使用最终冻结候选的同一代表网络：左侧显示学得的病理场，中间显示由全体最终事件 KMeans 后选出的两个模式代表传播，右侧显示未经模板平均的 30–80 Hz virtual-SEEG 直接波形。阴影只表示无监督模式身份，不预先当作 forward/reverse 标签。

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
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    capture = json.load(open(args.capture))
    confirmation = json.load(open(args.confirmation))
    if capture["figdata"]["sha256"] != _sha256(args.figdata):
        raise RuntimeError("capture/figdata hash mismatch")
    if capture["input_confirmation"]["sha256"] != _sha256(args.confirmation):
        raise RuntimeError("capture/confirmation hash mismatch")
    data = np.load(args.figdata, allow_pickle=True)

    fig = plt.figure(figsize=(18.8, 4.8), facecolor="white")
    grid = fig.add_gridspec(
        1, 4, width_ratios=(1.12, 1.0, 1.0, 2.35),
        left=0.045, right=0.99, bottom=0.15, top=0.88, wspace=0.22)
    axes = [fig.add_subplot(grid[0, 0], projection="3d")]
    axes.extend(fig.add_subplot(grid[0, index]) for index in range(1, 4))
    _plot_field(axes[0], data)
    image_a = _plot_mode_event(axes[1], data, 0)
    image_b = _plot_mode_event(axes[2], data, 1)
    image = image_b if image_b is not None else image_a
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[1:3], fraction=0.025, pad=0.025)
        colorbar.set_ticks((0.0, 1.0)); colorbar.set_ticklabels(("early", "late"))
        colorbar.set_label("relative firing onset", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    readout_stats = _plot_readout(axes[3], data, capture["events"])
    verdict = confirmation["candidates"][0]["confirm"]["verdict"]
    fig.suptitle(
        f"Data-driven core field: direct model readout  |  {verdict}",
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
        candidate_id=capture["candidate_id"], seed=capture["seed"],
        mode_counts=capture["mode_counts"], verdict=verdict,
        direct_readout=readout_stats,
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
