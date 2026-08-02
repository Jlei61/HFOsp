#!/usr/bin/env python3
"""Render development-only fast, M, and finite-control diagnostic figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import analyze_topic4_zm_lifecycle_sprint as ANALYSIS  # noqa: E402
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
FIG = OUT / "figures"
# Keys must be the labels `analyze_topic4_zm_lifecycle_sprint.analyze_one`
# actually emits; a stale key silently draws its cells grey and drops them from
# the legend instead of failing.
PHENO_COLORS = {
    "relaxation_burst_train": "#2166AC",
    "spreading_plateau": "#D6604D",
    "tonic_patch": "#B2182B",
    "structured_dynamic_candidate": "#1B7837",
    "weak_or_fragmented": "#BBBBBB",
    "no_onset": "#4D4D4D",
    "runaway": "#762A83",
}
ROLE_NAMES = {
    0: "highest continuous energy",
    1: "strongest spatial dynamics",
    2: "burst-patch transition",
    3: "stable local high activity",
}


def _load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def m_current_mV(dynamic_slow_flow, m_trace):
    """M current actually subtracted by the engine, in mV.

    `eta_m_applied` already carries the `g_M` scaling applied in `_make_slow`,
    and `SpatialSlowField.apply_currents` subtracts `eta_m * m`.  Re-applying
    `g_M` here would report a curve `g_M` times too large.
    """
    eta_applied = float((dynamic_slow_flow or {}).get("eta_m_applied", 0.001))
    return eta_applied * np.asarray(m_trace, float)


def plot_fast_phase_map(payload, path):
    rows = payload.get("rows", [])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    for row in rows:
        mechanism = row["mechanism"]
        dep = mechanism["i2e_depression"]
        arm = mechanism["arm"]
        color = PHENO_COLORS.get(row.get("phenotype"), "#777777")
        occupancy = row.get("intensity", {}).get("occupancy_above_6db") or 0.0
        axes[0].scatter(
            dep["tau_D_ms"], dep["d_star_nominal"],
            s=35 + 150 * occupancy, c=color,
            marker="^" if arm == "combined" else "o",
            edgecolor="black", linewidth=0.5, alpha=0.9,
        )
        spatial = row.get("within_episode_spatial", {})
        gain = row.get("intensity", {}).get("median_gain_db_across_contacts")
        pc1 = spatial.get("common_mode_pc1_fraction")
        if gain is not None and pc1 is not None:
            axes[1].scatter(
                gain, pc1, s=35 + 150 * occupancy, c=color,
                marker="^" if arm == "combined" else "o",
                edgecolor="black", linewidth=0.5, alpha=0.9,
            )
    axes[0].set(xlabel=r"I$\to$E depression recovery $\tau_D$ (ms)", ylabel=r"target resource $d^*$")
    axes[1].set(xlabel="median virtual-SEEG gain (dB)", ylabel="axial common-mode PC1 fraction")
    axes[1].axhline(0.9, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set_title("fast inhibitory phase map", loc="left", fontweight="bold")
    axes[1].set_title("energy versus spatial freedom", loc="left", fontweight="bold")
    handles = []
    for label, color in PHENO_COLORS.items():
        if any(row.get("phenotype") == label for row in rows):
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=color, label=label.replace("_", " ")))
    handles += [
        plt.Line2D([], [], marker="o", linestyle="", color="black", label="depression only"),
        plt.Line2D([], [], marker="^", linestyle="", color="black", label="+ I adaptation"),
    ]
    axes[1].legend(
        handles=handles, frameon=False, fontsize=7,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _series(rows, rank, tau_m):
    return sorted(
        [
            row for row in rows
            if row.get("status") == "complete"
            and int(row["selection_rank"]) == rank
            and float(row["tau_M_ms"]) == float(tau_m)
        ],
        key=lambda row: float(row["g_M"]),
    )


def plot_m_surface(payload, path):
    rows = payload.get("rows", [])
    ranks = sorted({int(row["selection_rank"]) for row in rows})
    fig, axes = plt.subplots(len(ranks), 2, figsize=(8.4, 2.25 * len(ranks)), sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    tau_colors = {500.0: "#B2182B", 2000.0: "#2166AC"}
    all_complete = [row for row in rows if row.get("status") == "complete"]
    rate_values = [
        row.get("paired_M_response", {}).get("ratio_core_mean_hz") for row in all_complete
    ]
    rate_values = [float(value) for value in rate_values if value is not None and np.isfinite(value)]
    rate_upper = max(1.10, max(rate_values, default=1.0) * 1.05)
    z_values = [
        row.get("paired_M_response", {}).get("delta_z_core_final") for row in all_complete
    ]
    z_values = [abs(float(value)) for value in z_values if value is not None and np.isfinite(value)]
    z_limit = max(0.05, max(z_values, default=0.0) * 1.15)
    for row_index, rank in enumerate(ranks):
        rank_rows = [row for row in rows if int(row["selection_rank"]) == rank]
        role = ROLE_NAMES.get(rank, f"phenotype {rank}")
        for tau in (500.0, 2000.0):
            series = _series(rows, rank, tau)
            if not series:
                continue
            x = np.asarray([row["g_M"] for row in series], float)
            rate_ratio = np.asarray([
                row.get("paired_M_response", {}).get("ratio_core_mean_hz", np.nan) for row in series
            ], float)
            z_delta = np.asarray([
                row.get("paired_M_response", {}).get("delta_z_core_final", np.nan) for row in series
            ], float)
            label = rf"$\tau_M$={tau / 1000:g} s"
            axes[row_index, 0].plot(x, rate_ratio, "o-", color=tau_colors[tau], label=label)
            axes[row_index, 1].plot(x, z_delta, "o-", color=tau_colors[tau], label=label)
            for axis, values in ((axes[row_index, 0], rate_ratio), (axes[row_index, 1], z_delta)):
                for source, xv, yv in zip(series, x, values):
                    if source.get("causal_exit_candidate") and np.isfinite(yv):
                        axis.scatter([xv], [yv], marker="*", s=90, color="#1B7837", zorder=5)
                    if source.get("returning_event_candidate") and np.isfinite(yv):
                        axis.scatter([xv], [yv], facecolors="none", edgecolors="#1B7837", s=90, zorder=6)
        axes[row_index, 0].axhline(1.0, color="#888888", linestyle="--", linewidth=0.8)
        axes[row_index, 1].axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
        axes[row_index, 0].set_ylabel(f"{rank}: {role}\ncore-rate ratio")
        axes[row_index, 1].set_ylabel(r"$\Delta z_{core,final}$")
        axes[row_index, 0].set_ylim(0.0, rate_upper)
        axes[row_index, 1].set_ylim(-z_limit, z_limit)
        for axis in axes[row_index]:
            axis.set_xscale("symlog", linthresh=1.0, linscale=1.0)
    for axis in axes[-1]:
        axis.set_xlabel(r"M coupling gain $g_M$")
    axes[0, 0].set_title("persistent-state response", loc="left", fontweight="bold")
    axes[0, 1].set_title("Z recovery response", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_control_dose(payload, path):
    rows = [row for row in payload.get("rows", []) if row.get("status") == "complete"]
    ranks = sorted({int(row["selection_rank"]) for row in rows})
    if not ranks:
        return False
    fig, axes = plt.subplots(
        len(ranks), 2, figsize=(8.2, 3.2 * len(ranks)),
        squeeze=False, constrained_layout=True,
    )
    duration_colors = {50.0: "#2166AC", 200.0: "#B2182B"}
    for row_index, rank in enumerate(ranks):
        subset = [row for row in rows if int(row["selection_rank"]) == rank]
        durations = sorted({float(row["control_duration_ms"]) for row in subset})
        for duration in durations:
            series = sorted(
                [row for row in subset if float(row["control_duration_ms"]) == duration],
                key=lambda row: float(row["dose_multiplier"]),
            )
            x = np.asarray([row["dose_multiplier"] for row in series], float)
            drop = np.asarray([
                row.get("control_response", {}).get("fractional_core_drop", np.nan)
                for row in series
            ], float)
            dwell = np.asarray([
                row.get("control_response", {}).get("longest_global_zero_rate_ms", np.nan)
                for row in series
            ], float)
            color = duration_colors.get(duration, "#555555")
            label = f"{duration:g} ms"
            axes[row_index, 0].plot(x, drop, "o-", color=color, label=label)
            axes[row_index, 1].plot(x, dwell, "o-", color=color, label=label)
            for source, xv, yv in zip(series, x, dwell):
                if source.get("causal_control_exit_candidate") and np.isfinite(yv):
                    axes[row_index, 1].scatter(
                        [xv], [yv], marker="*", s=100, color="#1B7837", zorder=5,
                    )
        n_exit = sum(bool(row.get("causal_control_exit_candidate")) for row in subset)
        n_return = sum(bool(row.get("returning_event_candidate")) for row in subset)
        axes[row_index, 0].axhline(0.5, color="#888888", linestyle="--", linewidth=0.8)
        axes[row_index, 0].set_ylim(0.0, 1.05)
        axes[row_index, 0].set(
            xlabel=r"dose / $u_{ref}$",
            ylabel="paired core spike-count reduction",
        )
        axes[row_index, 1].axhline(
            100.0, color="#888888", linestyle="--", linewidth=0.8,
            label="100 ms silencing guard",
        )
        axes[row_index, 1].set(
            xlabel=r"dose / $u_{ref}$",
            ylabel="longest all-E zero-rate dwell (ms)",
        )
        axes[row_index, 1].text(
            0.02, 0.96, f"durable exits {n_exit}/{len(subset)}\nreturning events {n_return}/{len(subset)}",
            transform=axes[row_index, 1].transAxes, va="top", ha="left", fontsize=8,
        )
    axes[0, 0].set_title("acute paired suppression", loc="left", fontweight="bold")
    axes[0, 1].set_title("persistence after the pulse", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def plot_trajectory(root, path):
    summary = json.loads((root / "summary.json").read_text())
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    fine_t = np.asarray(arrays["fine_time_ms"], float) / 1000.0
    coarse_t = np.arange(arrays["coarse_core_rate_hz"].size) * 0.025
    baseline, _, _ = ANALYSIS.resolve_episode_baseline(arrays["coarse_core_rate_hz"])
    rms, rms_status = ANALYSIS.contact_rms_from_baseline(
        arrays["lfp_raw_synaptic_proxy"], float(arrays["lfp_fs_hz"]), baseline,
    )
    db = None
    if rms is not None:
        power = rms ** 2
        base = np.mean(power[baseline[:power.shape[0]]], axis=0)
        valid = base > max(float(np.max(base)) * 1e-12, np.finfo(float).tiny)
        db = np.full_like(power, np.nan)
        db[:, valid] = 10.0 * np.log10(np.maximum(power[:, valid], np.finfo(float).tiny) / base[valid])

    fig, axes = plt.subplots(5, 1, figsize=(11.5, 10.0), sharex=True, constrained_layout=True)
    axes[0].plot(fine_t, arrays["fine_core_rate_hz"], color="#B2182B", lw=0.55, label="core E")
    axes[0].plot(fine_t, arrays["fine_surround_rate_hz"], color="#2166AC", lw=0.45, label="surround E")
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[0].set_title(root.name, loc="left", fontsize=9, fontweight="bold")
    if db is not None:
        vmax = max(6.0, float(np.nanpercentile(db, 99)))
        image = axes[1].imshow(
            db.T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
            vmin=-6.0, vmax=vmax, extent=(0, db.shape[0] * 0.025, 0.5, db.shape[1] + 0.5),
        )
        fig.colorbar(image, ax=axes[1], pad=0.01, label="dB vs event-free baseline")
    else:
        axes[1].text(0.5, 0.5, f"virtual-SEEG unavailable: {rms_status}", transform=axes[1].transAxes, ha="center")
    axes[1].set_ylabel("virtual contact")
    axes[1].set_title("virtual-SEEG energy", loc="left", fontsize=9, fontweight="bold")
    kymo = np.asarray(arrays["coarse_kymo_axial"], float)
    image = axes[2].imshow(
        kymo, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
        extent=(0, kymo.shape[1] * 0.025, 0, kymo.shape[0]),
    )
    fig.colorbar(image, ax=axes[2], pad=0.01, label="spikes / bin")
    axes[2].set_ylabel("pathological axis")
    axes[2].set_title("axial activity", loc="left", fontsize=9, fontweight="bold")

    n_trace = len(arrays["trace_z_core_mean"])
    trace_t = np.arange(n_trace) * float(summary["dt_ms"]) / 1000.0
    axes[3].plot(trace_t, arrays["trace_z_core_mean"], color="#1B9E77", lw=0.8, label="z core")
    axes[3].plot(trace_t, arrays["trace_S_G"], color="#E66101", lw=0.8, label=r"$S_G$")
    axes[3].set_ylabel("z / shared inhibition")
    axes[3].legend(frameon=False, ncol=2, loc="upper right")
    ax_m = axes[3].twinx()
    mechanism = summary.get("mechanism", {}).get("dynamic_slow_flow", {})
    m_current = m_current_mV(mechanism, arrays["trace_m_core_mean"])
    ax_m.plot(trace_t, m_current, color="#6A3D9A", lw=0.75, label=r"$\eta_M m$")
    ax_m.set_ylabel("M current (mV)", color="#6A3D9A")
    axes[3].set_title("Z–M slow flow", loc="left", fontsize=9, fontweight="bold")

    axes[4].plot(trace_t, arrays["trace_i2e_resource_mean"], color="#4D4D4D", lw=0.75, label="I→E resource")
    axes[4].plot(trace_t, arrays["trace_phi_core_mean"], color="#A6761D", lw=0.75, label=r"core $\phi$")
    if "trace_i_adaptation_mean" in arrays and arrays["trace_i_adaptation_mean"].size == trace_t.size:
        axes[4].plot(trace_t, arrays["trace_i_adaptation_mean"], color="#7570B3", lw=0.65, label="I adaptation")
    axes[4].set_ylabel("fast feedback")
    axes[4].set_xlabel("time after pre-entry checkpoint (s)")
    axes[4].legend(frameon=False, ncol=3, loc="upper right")
    axes[4].set_xlim(0, float(summary["observed_ms"]) / 1000.0)
    for axis in axes:
        axis.margins(x=0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    generated = []
    fast = _load(OUT / "batch1_phase_map.json")
    if fast is not None:
        path = FIG / "fast_inhibitory_phase_map.png"
        plot_fast_phase_map(fast, path); generated.append(path.name)
    m_surface = _load(OUT / "m_response_surface.json")
    if m_surface is not None:
        path = FIG / "m_response_surface.png"
        plot_m_surface(m_surface, path); generated.append(path.name)
    control = _load(OUT / "control_dose_analysis.json")
    if control is not None:
        path = FIG / "control_dose_response.png"
        if plot_control_dose(control, path):
            generated.append(path.name)
    for trajectory_root in args.trajectory_root:
        root = trajectory_root.resolve()
        path = FIG / f"trajectory_{root.name}.png"
        plot_trajectory(root, path)
        generated.append(path.name)
    descriptions = {
        "fast_inhibitory_phase_map.png": (
            "左图展示 I→E depression 参数与 burst/plateau 表型，右图检查宏观能量是否伴随真正的空间自由度。"
            "它是 seed-1 development phase map，不是 ictal carrier 验收。\n\n**关注点**：高能量点是否仍集中在 PC1 接近 1 的 common-mode 区。"
        ),
        "m_response_surface.png": (
            "四行对应四种 fast phenotype；左列是相对 g_M=0 的核心放电变化，右列是 Z 终值变化。"
            "绿色星号表示配对因果 offset，绿色空圈表示 returning-event candidate。\n\n**关注点**：M 是否形成连续剂量响应并把 Z 推向恢复，而不只是 prevention。"
        ),
        "control_dose_response.png": (
            "左图给出有限阈值脉冲相对同 checkpoint、同 future noise 无控制轨迹的即时 core spike-count 降幅；"
            "右图给出脉冲期间最长的兴奋细胞群（all-E）零放电时长，并标出 100 ms silencing guard——"
            "阈值抬升只作用在兴奋细胞上，抑制细胞仍在放电，因此这不是整片网络静默。"
            "每一行只对应**一个**被选中的持续状态，剂量×时长是同一条轨迹上的重复施加，不是独立样本。"
            "该图只描述 seed-1 development control。\n\n**关注点**：即时抑制能否转化为持久退出；"
            "若只随剂量增加而延长静默、脉冲后仍恢复原 burst train，则不是可控 lifecycle。"
        ),
    }
    # README is an inventory of the directory, not merely of figures touched
    # by this invocation.  A trajectory-only diagnostic call must not erase
    # the descriptions of trajectories generated earlier in the sprint.
    inventory = sorted({path.name for path in FIG.glob("*.png")} | set(generated))
    text = [f"### {name}\n\n{descriptions[name]}" for name in inventory if name in descriptions]
    text += [
        f"### {name}\n\n单条全动态轨迹的核区/外围放电、虚拟电极能量、病理轴活动与 Z–M/fast-feedback 轨迹。"
        "该图用于诊断状态类型，不单独构成 lifecycle 验收。\n\n**关注点**：高能活动是否持续且具有空间组织，"
        "以及退出前 Z 与 M 是否形成方向相反的 slow flow。"
        for name in inventory if name.startswith("trajectory_")
    ]
    (FIG / "README.md").write_text("\n\n".join(text) + "\n")
    print(json.dumps({"generated": generated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
