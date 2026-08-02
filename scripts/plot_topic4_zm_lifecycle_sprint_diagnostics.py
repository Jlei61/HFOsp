#!/usr/bin/env python3
"""Render development-only fast, M, and finite-control diagnostic figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"
FIG = OUT / "figures"
PHENO_COLORS = {
    "relaxation_burst_train": "#2166AC",
    "spreading_plateau": "#D6604D",
    "localized_tonic_patch": "#B2182B",
    "structured_candidate": "#1B7837",
}
ROLE_NAMES = {
    0: "highest continuous energy",
    1: "strongest spatial dynamics",
    2: "burst-patch transition",
    3: "stable local high activity",
}


def _load(path):
    return json.loads(path.read_text()) if path.is_file() else None


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
    fig, axes = plt.subplots(1, len(ranks), figsize=(3.2 * len(ranks), 3.2), squeeze=False, constrained_layout=True)
    for axis, rank in zip(axes[0], ranks):
        subset = [row for row in rows if int(row["selection_rank"]) == rank]
        durations = sorted({float(row["control_duration_ms"]) for row in subset})
        multipliers = sorted({float(row.get("dose_multiplier", np.nan)) for row in subset})
        grid = np.full((len(durations), len(multipliers)), np.nan)
        for row in subset:
            i = durations.index(float(row["control_duration_ms"]))
            j = multipliers.index(float(row["dose_multiplier"]))
            grid[i, j] = 2.0 if row.get("returning_event_candidate") else (
                1.0 if row.get("causal_control_exit_candidate") else 0.0
            )
        image = axis.imshow(grid, vmin=0, vmax=2, cmap=matplotlib.colors.ListedColormap(["#D9D9D9", "#F4A582", "#1B7837"]), aspect="auto")
        axis.set_xticks(range(len(multipliers)), [f"{value:g}" for value in multipliers])
        axis.set_yticks(range(len(durations)), [f"{value:g}" for value in durations])
        axis.set(xlabel=r"dose / $u_{ref}$", ylabel="pulse duration (ms)", title=f"phenotype {rank}")
    fig.colorbar(image, ax=axes.ravel().tolist(), ticks=[0, 1, 2], label="0 persistent | 1 exit | 2 returning event")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main():
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
            "每个格子对应有限阈值脉冲的剂量和时长：灰色为持续、橙色为配对因果退出、绿色为退出后出现 returning event。"
            "该图只描述 seed-1 development control。\n\n**关注点**：是否存在非永久静默、且能回到间期事件的有限剂量窗。"
        ),
    }
    text = [f"### {name}\n\n{descriptions[name]}" for name in generated]
    (FIG / "README.md").write_text("\n\n".join(text) + "\n")
    print(json.dumps({"generated": generated}, ensure_ascii=False))


if __name__ == "__main__":
    main()
