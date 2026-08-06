#!/usr/bin/env python3
"""Render endpoint versus middle stimulation near operational runaway in MZ.

The right-hand readout deliberately follows the accepted Figure-4 burst
grammar: signed 30--80 Hz virtual-SEEG bursts, event shading and contact peak
order.  The spatial panels use the registered E1146 montage in centered shared-
axis coordinates and share identical axes across both arms.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DT_MS = 0.1
TRACE_BAND_HZ = (30.0, 80.0)
TRACE_OFFSET = 1.24
TRACE_GAIN = 0.74
SHAFT_COLORS = {"ICL": "#F47A42", "SCL": "#1FA4A5"}
STIM_COLOR = "#2F80ED"
RUNAWAY_COLOR = "#D6274B"
BURST_SHADE = "#F6C98E"
STATE_CMAP = "viridis"
ACTIVITY_CMAP = "magma"


def _shaft(name: str) -> str:
    match = re.match(r"[A-Za-z]+", str(name))
    return match.group(0).upper() if match else str(name)


def _project(points: np.ndarray, axis: np.ndarray, center: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    transverse = np.array([-axis[1], axis[0]], float)
    centered = np.asarray(points, float) - np.asarray(center, float)[None, :]
    return np.column_stack((centered @ axis, centered @ transverse))


def _smooth_grid(points: np.ndarray, values: np.ndarray, *, bins: int = 92, sigma: float = 1.8):
    edges = np.linspace(-10.0, 10.0, bins + 1)
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    weighted, _, _ = np.histogram2d(
        points[finite, 0], points[finite, 1], bins=(edges, edges), weights=values[finite]
    )
    counts, _, _ = np.histogram2d(points[finite, 0], points[finite, 1], bins=(edges, edges))
    weighted = gaussian_filter(weighted, sigma=sigma, mode="nearest")
    counts = gaussian_filter(counts, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        field = weighted / counts
    field[counts < 1e-5] = np.nan
    return field.T, edges


def _signed_burst(lfp: np.ndarray, times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    times = np.asarray(times, float)
    dt_ms = float(np.median(np.diff(times)))
    sos = butter(4, TRACE_BAND_HZ, btype="bandpass", fs=1000.0 / dt_ms, output="sos")
    burst = sosfiltfilt(sos, np.asarray(lfp, float), axis=0)
    scale = np.percentile(np.abs(burst[np.asarray(mask, bool)]), 95.0, axis=0)
    positive = scale[np.isfinite(scale) & (scale > 1e-12)]
    if positive.size == 0:
        raise ValueError("virtual-SEEG burst trace has zero scale")
    scale = np.maximum(scale, 0.15 * float(np.median(positive)))
    return TRACE_GAIN * burst / scale[None, :]


def _components(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    result = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index))
            start = None
    if start is not None:
        result.append((start, len(mask)))
    return result


def _close_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, bool).copy()
    for lo, hi in _components(~out):
        if lo > 0 and hi < len(out) and (hi - lo) <= int(max_gap):
            out[lo:hi] = True
    return out


def _burst_windows(times: np.ndarray, trace: np.ndarray, t_run: float, xlim_abs: tuple[float, float]):
    global_env = gaussian_filter1d(np.median(np.abs(trace), axis=1), sigma=12.0)
    baseline_start = max(float(times[0]), min(float(xlim_abs[0]), float(t_run) - 2000.0))
    baseline = (times >= baseline_start) & (times < float(t_run) - 50.0)
    ref = global_env[baseline]
    if ref.size < 10:
        raise ValueError("insufficient pre-runaway samples for burst threshold")
    med = float(np.median(ref))
    mad = float(1.4826 * np.median(np.abs(ref - med)))
    threshold = med + 2.8 * max(mad, 1e-9)
    active = (
        (global_env > threshold)
        & (times >= xlim_abs[0])
        & (times <= xlim_abs[1])
        & (times < t_run - 10.0)
    )
    active = _close_gaps(active, int(round(8.0 / DT_MS)))
    windows = []
    for lo, hi in _components(active):
        t0, t1 = float(times[lo]), float(times[min(hi, len(times) - 1)])
        if t1 - t0 >= 8.0:
            windows.append((max(t0 - 8.0, xlim_abs[0]), min(t1 + 12.0, t_run - 4.0)))
    return windows, threshold


def _draw_contacts(
    ax,
    contacts: np.ndarray,
    names: list[str],
    stim_indices: np.ndarray,
    *,
    label_stim: bool = True,
) -> None:
    stim = set(int(index) for index in np.asarray(stim_indices, int))
    for index, (point, name) in enumerate(zip(contacts, names)):
        color = SHAFT_COLORS.get(_shaft(name), "0.35")
        if index in stim:
            ax.scatter(
                [point[0]], [point[1]], s=54, marker="s", facecolor=STIM_COLOR,
                edgecolor="white", linewidth=0.8, zorder=9,
            )
            if label_stim:
                ax.text(
                    point[0], point[1] + 0.55, name, color=STIM_COLOR, fontsize=6.4,
                    ha="center", va="bottom", fontweight="bold", zorder=10,
                )
        else:
            ax.scatter(
                [point[0]], [point[1]], s=31, marker="s", facecolor="white",
                edgecolor=color, linewidth=0.9, zorder=8,
            )


def _spatial_style(ax, *, show_y: bool) -> None:
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("TA shared axis (mm)", fontsize=8.7)
    if show_y:
        ax.set_ylabel("transverse axis (mm)", fontsize=8.7)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=7.3, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def _plot_state_panel(ax, payload: dict, d_limits: tuple[float, float], adap_levels: np.ndarray):
    data = payload["data"]
    pos = payload["pos"]
    contacts = payload["contacts"]
    names = payload["names"]
    d_field, edges = _smooth_grid(pos, 1.0 - np.asarray(data["z_snapshot_e"], float))
    adap = payload["eta_m"] * np.asarray(data["m_snapshot_e"], float)
    a_field, _ = _smooth_grid(pos, adap)
    image = ax.imshow(
        d_field,
        origin="lower",
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        cmap=STATE_CMAP,
        vmin=d_limits[0],
        vmax=d_limits[1],
        interpolation="bilinear",
        rasterized=True,
    )
    if np.unique(adap_levels).size >= 2:
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.contour(
            centers,
            centers,
            a_field,
            levels=adap_levels,
            colors="white",
            linewidths=0.65,
            alpha=0.76,
        )
    _draw_contacts(ax, contacts, names, data["stim_contact_indices"], label_stim=False)
    ax.plot([-10, 10], [0, 0], color="white", lw=0.75, alpha=0.55, zorder=6)
    for core, color in ((payload["source"], "#B2182B"), (payload["sink"], "#D95F0E")):
        ax.add_patch(plt.Circle(core, 1.5, fill=False, ec=color, lw=1.0, ls="--", zorder=7))
    _spatial_style(ax, show_y=True)
    return image


def _plot_activity_panel(ax, payload: dict, activity_vmax: float):
    data = payload["data"]
    counts = np.asarray(data["spike_count_window"], float)
    duration_s = float(payload["meta"]["activity_window_ms"]) / 1000.0
    activity_hz = counts / max(duration_s, 1e-9)
    field, edges = _smooth_grid(payload["pos"], activity_hz)
    image = ax.imshow(
        field,
        origin="lower",
        extent=[edges[0], edges[-1], edges[0], edges[-1]],
        cmap=ACTIVITY_CMAP,
        vmin=0.0,
        vmax=activity_vmax,
        interpolation="bilinear",
        rasterized=True,
    )
    active = counts > 0
    if np.any(active):
        latency = np.asarray(data["first_spike_rel_ms"], float)
        ax.scatter(
            payload["pos"][active, 0],
            payload["pos"][active, 1],
            c=latency[active],
            s=1.0,
            cmap="viridis",
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=4,
        )
    _draw_contacts(ax, payload["contacts"], payload["names"], data["stim_contact_indices"], label_stim=False)
    ax.plot([-10, 10], [0, 0], color="white", lw=0.75, alpha=0.55, zorder=6)
    _spatial_style(ax, show_y=False)
    return image


def _common_unbroken_window(
    payloads: list[dict],
    *,
    before_stim_ms: float = 2000.0,
) -> tuple[float, float]:
    stim_on = {float(payload["meta"]["stim_on_ms"]) for payload in payloads}
    stim_off = {float(payload["meta"]["stim_off_ms"]) for payload in payloads}
    if len(stim_on) != 1 or len(stim_off) != 1:
        raise ValueError("stimulation windows differ between arms")
    stim_on_ms = next(iter(stim_on))
    common = (
        stim_on_ms - float(before_stim_ms),
        min(float(np.asarray(payload["data"]["times"], float)[-1]) for payload in payloads),
    )
    for payload in payloads:
        arm = str(payload["meta"]["arm"])
        times = np.asarray(payload["data"]["times"], float)
        if common[0] < float(times[0]) - 1e-6:
            raise ValueError(f"{arm}: pre-stimulation display window is unavailable")
    endpoint = next(payload for payload in payloads if payload["meta"]["arm"] == "endpoint")
    middle = next(payload for payload in payloads if payload["meta"]["arm"] == "middle")
    if not (
        float(endpoint["meta"]["t_run_ms"]) < common[1]
        < float(middle["meta"]["t_run_ms"])
    ):
        raise ValueError("common window must contain endpoint runaway but precede middle runaway")
    return common


def _plot_burst_readout(
    ax,
    payload: dict,
    xlim_abs: tuple[float, float],
    *,
    x_origin_abs: float,
    show_xticklabels: bool,
    show_ylabels: bool,
):
    data, meta = payload["data"], payload["meta"]
    times_abs = np.asarray(data["times"], float)
    lfp = np.asarray(data["lfp_trace"], float)
    scale_mask = times_abs < min(float(meta["stim_on_ms"]), float(meta["t_run_ms"]) - 200.0)
    trace_all = _signed_burst(lfp, times_abs, scale_mask)
    keep = (times_abs >= xlim_abs[0]) & (times_abs <= min(xlim_abs[1], float(times_abs[-1])))
    times = times_abs[keep]
    trace = trace_all[keep]
    names = payload["names"]
    order = list(range(len(names) - 1, -1, -1))
    y = np.arange(len(order), dtype=float) * TRACE_OFFSET
    x = (times - float(x_origin_abs)) / 1000.0
    stim_set = set(int(index) for index in data["stim_contact_indices"])

    stim_lo = float(meta["stim_on_ms"] - x_origin_abs) / 1000.0
    stim_hi = float(meta["stim_off_ms"] - x_origin_abs) / 1000.0
    if xlim_abs[0] < float(meta["stim_off_ms"]) and xlim_abs[1] > float(meta["stim_on_ms"]):
        ax.axvspan(stim_lo, stim_hi, color=STIM_COLOR, alpha=0.18, lw=0, zorder=0)
    for boundary in (stim_lo, stim_hi):
        if -1e-9 <= boundary <= (xlim_abs[1] - x_origin_abs) / 1000.0 + 1e-9:
            ax.axvline(boundary, color=STIM_COLOR, lw=1.05, zorder=7)
    runaway_x = float(meta["t_run_ms"] - x_origin_abs) / 1000.0
    if xlim_abs[0] <= float(meta["t_run_ms"]) <= xlim_abs[1]:
        ax.axvline(runaway_x, color=RUNAWAY_COLOR, lw=1.35, ls="--", zorder=8)
        ax.axvspan(
            runaway_x,
            (xlim_abs[1] - x_origin_abs) / 1000.0,
            color=RUNAWAY_COLOR,
            alpha=0.035,
            lw=0,
            zorder=0,
        )
    windows, threshold = _burst_windows(times_abs, trace_all, float(meta["t_run_ms"]), xlim_abs)
    for lo, hi in windows:
        ax.axvspan(
            (lo - x_origin_abs) / 1000.0, (hi - x_origin_abs) / 1000.0,
            color=BURST_SHADE, alpha=0.28, lw=0, zorder=1,
        )

    for row, contact_index in enumerate(order):
        name = names[contact_index]
        color = SHAFT_COLORS.get(_shaft(name), "0.35")
        width = 1.0 if contact_index in stim_set else 0.82
        ax.plot(x, trace[:, contact_index] + y[row], color=color, lw=width, alpha=0.96, zorder=3)

    shown_windows = []
    for lo, hi in windows:
        event_mask = (times >= lo) & (times <= hi)
        if event_mask.sum() < 3:
            continue
        shown_windows.append({"start_ms": lo, "end_ms": hi})

    ax.set_xlim(0.0, (xlim_abs[1] - x_origin_abs) / 1000.0)
    ax.set_ylim(-0.6, y[-1] + 1.35)
    if show_ylabels:
        ax.set_yticks(y)
        ax.set_yticklabels([names[index] for index in order], fontsize=6.6)
        for tick, contact_index in zip(ax.get_yticklabels(), order):
            tick.set_color(
                STIM_COLOR if contact_index in stim_set
                else SHAFT_COLORS.get(_shaft(names[contact_index]), "0.35")
            )
            if contact_index in stim_set:
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=6.8, length=2.5)
    ax.tick_params(axis="x", labelbottom=show_xticklabels)
    ax.tick_params(axis="y", length=2.0)
    ax.spines[["top", "right"]].set_visible(False)
    return {
        "burst_windows": shown_windows,
        "burst_threshold": float(threshold),
        "xlim_absolute_ms": list(xlim_abs),
        "x_reference": "display_start",
        "x_unit": "s",
        "display_start_absolute_ms": float(x_origin_abs),
        "stim_on_relative_s": stim_lo,
        "stim_off_relative_s": stim_hi,
        "runaway_relative_s": runaway_x,
    }
def _load_arm(input_dir: Path, arm: str) -> dict:
    metadata = json.loads((input_dir / f"{arm}.json").read_text(encoding="utf-8"))
    arrays = np.load(input_dir / f"{arm}.npz", allow_pickle=True)
    names = [str(name) for name in arrays["contact_names"]]
    axis = np.asarray(arrays["axis_unit"], float)
    center = np.asarray(arrays["center"], float)
    return {
        "meta": metadata,
        "data": arrays,
        "names": names,
        "pos": _project(arrays["pos_e"], axis, center),
        "contacts": _project(arrays["contacts"], axis, center),
        "source": _project(np.asarray(arrays["source_xy"], float)[None, :], axis, center)[0],
        "sink": _project(np.asarray(arrays["sink_xy"], float)[None, :], axis, center)[0],
        "eta_m": float(metadata["candidate_cfg"]["eta_m"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "results/topic4_sef_hfo/mz_stim_site_compare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/paper-ready-figure/fig_mz_stim_site_sequence/figures",
    )
    parser.add_argument("--stem", default="fig_mz_stim_site_sequence")
    args = parser.parse_args()

    summary = json.loads((args.input_dir / "summary.json").read_text(encoding="utf-8"))
    arms = [_load_arm(args.input_dir, arm) for arm in ("endpoint", "middle")]
    if arms[0]["names"] != arms[1]["names"]:
        raise ValueError("contact order differs between stimulation arms")
    if not np.allclose(arms[0]["contacts"], arms[1]["contacts"], atol=1e-7):
        raise ValueError("registered montage differs between stimulation arms")
    common_xlim = _common_unbroken_window(arms)

    fig = plt.figure(figsize=(14.4, 6.4), facecolor="white")
    outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.0, 1.0],
        left=0.095,
        right=0.985,
        bottom=0.070,
        top=0.965,
        hspace=0.24,
    )
    readout_axes = []
    trace_metadata = {}
    for row, arm in enumerate(arms):
        readout_ax = fig.add_subplot(
            outer[row, 0],
            sharex=readout_axes[0] if readout_axes else None,
        )
        readout_axes.append(readout_ax)
        readout_meta = _plot_burst_readout(
            readout_ax,
            arm,
            common_xlim,
            x_origin_abs=common_xlim[0],
            show_xticklabels=(row == 1),
            show_ylabels=True,
        )
        arm_name = str(arm["meta"]["arm"])
        trace_metadata[arm_name] = readout_meta

        row_label = "endpoint" if row == 0 else "middle"
        fig.text(
            0.018,
            0.735 if row == 0 else 0.295,
            row_label,
            ha="left",
            va="center",
            fontsize=9.0,
            color="#1F2D3D",
            fontweight="bold",
            rotation=90,
        )

    fig.text(0.012, 0.985, "A", fontsize=15.0, fontweight="bold", ha="left", va="top")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.stem}.png"
    pdf = args.output_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    epoch_burst_rates = {}
    for arm in arms:
        arm_name = str(arm["meta"]["arm"])
        windows = trace_metadata[arm_name]["burst_windows"]
        stim_on = float(arm["meta"]["stim_on_ms"])
        stim_off = float(arm["meta"]["stim_off_ms"])
        midpoints = [0.5 * (item["start_ms"] + item["end_ms"]) for item in windows]
        n_pre = sum(common_xlim[0] <= midpoint < stim_on for midpoint in midpoints)
        n_stim = sum(stim_on <= midpoint <= stim_off for midpoint in midpoints)
        epoch_burst_rates[arm_name] = {
            "pre_stim_hz": n_pre / ((stim_on - common_xlim[0]) / 1000.0),
            "during_stim_hz": n_stim / ((stim_off - stim_on) / 1000.0),
        }

    metadata = {
        "schema_id": "topic4_mz_stim_site_sequence_figure_v1",
        "status": "model-side visual diagnostic",
        "canonical_producer": "scripts/paper_figures/plot_fig_mz_stim_site_near_runaway.py",
        "simulation_producer": "scripts/run_topic4_mz_stim_site_compare.py",
        "input_summary": str(args.input_dir / "summary.json"),
        "model_contract": {
            "candidate": summary["candidate"],
            "slow_state": "per-neuron postsynaptic inhibitory efficacy z_i plus adaptation m_i",
            "excluded_paths": summary["forbidden_model_paths"],
        },
        "stimulation_site_contract": {
            "contact_order": arms[0]["names"],
            "encoding": "stimulated contact labels are blue in each readout row",
            "omitted_panels": (
                "phase-matched near-runaway state and activity maps were removed because "
                "they do not identify the intervention timing effect"
            ),
        },
        "readout_contract": {
            "signal": "signed virtual-LFP bandpassed 30-80 Hz",
            "grammar": (
                "Figure-4 signed burst readout without peak markers or event-order lines; "
                "single continuous common time axis"
            ),
            "all_montage_contacts": True,
            "continuous_display_window_absolute_ms": list(common_xlim),
            "endpoint_runaway_visible": (
                common_xlim[0] <= float(arms[0]["meta"]["t_run_ms"]) <= common_xlim[1]
            ),
            "middle_runaway_visible": (
                common_xlim[0] <= float(arms[1]["meta"]["t_run_ms"]) <= common_xlim[1]
            ),
            "epoch_burst_rates_hz": epoch_burst_rates,
            "trace_metadata": trace_metadata,
        },
        "arms": {arm["meta"]["arm"]: arm["meta"] for arm in arms},
        "comparison": {
            "endpoint_runaway_after_stim_off_ms": float(
                arms[0]["meta"]["t_run_ms"] - arms[0]["meta"]["stim_off_ms"]
            ),
            "middle_runaway_after_stim_off_ms": float(
                arms[1]["meta"]["t_run_ms"] - arms[1]["meta"]["stim_off_ms"]
            ),
            "middle_minus_endpoint_runaway_ms": float(
                arms[1]["meta"]["t_run_ms"] - arms[0]["meta"]["t_run_ms"]
            ),
            "interpretation_boundary": (
                "runaway timing supports a site effect; the readout does not by itself "
                "identify propagation-order disruption as the mediator"
            ),
        },
        "pre_stim_parity": summary["pre_stim_parity"],
        "claim_boundary": (
            "single-seed model-only external threshold clamp; operational runaway proxy, "
            "not a clinical seizure, treatment result, or biophysical stimulation mechanism"
        ),
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    metadata_path = args.output_dir / f"{args.stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""### {png.name}

这张图只比较冻结 MZ `z_i+m_i` 候选的两种刺激位置。两行分别是端点刺激和中段刺激，蓝色 contact label 直接标出受刺激触点；原来的 near-runaway state/activity 四幅空间图已删除，因为 phase-matched 截帧不能解释刺激位置造成的时间差。

readout 采用当前 Figure 4 的 30–80 Hz signed virtual-SEEG burst 语法，不绘制峰值点或事件内连线。上下使用同一个连续时间窗：先显示 2 s 正常间歇事件，再进入明显的蓝色刺激区；刺激期间 burst 变稀疏，刺激结束后端点组进入 runaway，中段组在本观察窗内不进入 runaway。

**关注点**：上下行刺激结束后的 runaway 间隔分别约为 {arms[0]['meta']['t_run_ms'] - arms[0]['meta']['stim_off_ms']:.1f} ms 和 {arms[1]['meta']['t_run_ms'] - arms[1]['meta']['stim_off_ms']:.1f} ms；这是单 seed 的位置效应，不单凭本图认定其机制是打乱传播顺序。
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {png}\nwrote {pdf}\nwrote {metadata_path}")


if __name__ == "__main__":
    main()
