#!/usr/bin/env python3
"""Render the auditable saddle-node / OU-SNN organizer figure."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_patient_zm_figure import (  # noqa: E402
    FOLD,
    HIGH,
    LOW,
    RETURNED,
    draw_critical_manifold_trajectory,
    load_projection,
    sha256,
)


def _panel_label(ax, label):
    ax.text(-0.16, 1.08, label, transform=ax.transAxes,
            fontsize=16, fontweight="bold", ha="left", va="top")


def _save(fig, stem):
    outputs = {}
    for suffix, options in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.03,
                    facecolor="white", **options)
        outputs[suffix] = {"path": str(path), "sha256": sha256(path)}
    return outputs


def _append_readme(path, stem_name):
    header = f"### {stem_name}.png / .pdf / .svg"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if header in text:
        return
    section = (
        f"\n{header}\n\n"
        "四联审计图把同一个患者匹配空间 Z/M 问题拆成四个可核对层次：A 将"
        "三颗 OU-SNN 轨迹投到 1 mm frozen-q rate manifold；B 在 D–A 平面同时"
        "显示 q_core、q_mean、M 和 rE；C 给出包含真实突触延迟的高/低支线性稳定性；"
        "D 比较 2、1.33、1 mm 保守粗化下的同支与 fold。\n\n"
        "**关注点**：三种粗化都有 generic fold，且共同 branch 在共享锚点一致；"
        "但 fold rate 在 1.33 与 1 mm 间仍差 32.9%，不能写成精确临界状态已网格收敛。"
        "高支是 delay-unstable skeleton，而不是稳定高固定点；有限 OU-SNN 的 tonic "
        "runaway 只与该 skeleton 在时间次序和状态尺度上一致。\n")
    path.write_text(text.rstrip() + "\n" + section, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    phase_root = Path(
        "/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram")
    parser.add_argument("--phase-root", type=Path, default=phase_root)
    parser.add_argument("--stem", default="patient_zm_saddle_organizer_audit")
    args = parser.parse_args()
    phase_root = args.phase_root.resolve()
    projection_path = (
        phase_root / "dynamic_projection/patient_zm_snn_manifold_projection.json")
    delay_path = (
        phase_root / "deterministic_meanfield/patient_zm_delay_stability_audit.json")
    grid_path = (
        phase_root / "deterministic_meanfield/patient_zm_grid_convergence.json")
    projection, arrays = load_projection(projection_path)
    delay = json.loads(delay_path.read_text())
    grid = json.loads(grid_path.read_text())
    if projection["status"] != "SNN_TRAJECTORIES_CONSISTENT_WITH_REDUCED_FOLD_ORGANIZER":
        raise RuntimeError("dynamic projection is not accepted")
    if delay["status"] != "PATIENT_ZM_DELAY_STABILITY_AUDITED":
        raise RuntimeError("delay stability audit is not accepted")
    if not grid["gates"]["generic_fold_present_on_every_grid"]:
        raise RuntimeError("not every conservative grid contains the audited fold")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.3))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.09, top=0.90,
                        wspace=0.29, hspace=0.38)
    fig.suptitle(
        "Patient-matched spatial Z/M: a reduced saddle-node organizes OU-SNN runaway",
        x=0.075, ha="left", fontsize=16, fontweight="bold")

    # A: q-rate critical manifold plus all prospective SNN trajectories.
    ax = axes[0, 0]
    q_fold = float(projection["manifold"]["q_fold"])
    rate_fold = float(projection["manifold"]["mean_rate_e_hz_at_fold"])
    ax.plot(1.0 - arrays["manifold_low_q"],
            arrays["manifold_low_rate_e_hz"], color=LOW, lw=1.3,
            label="near-silent branch")
    ax.plot(1.0 - arrays["manifold_returned_q"],
            arrays["manifold_returned_rate_e_hz"], color=RETURNED,
            lw=1.45, ls="--", label="returned branch")
    ax.plot(1.0 - arrays["manifold_high_q"],
            arrays["manifold_high_rate_e_hz"], color=HIGH, lw=1.8,
            label="high-rate skeleton")
    seed_colors = {1841: "#4C78A8", 1842: "#2A9D8F", 1843: "#E07A5F"}
    for run in projection["runs"]:
        seed = int(run["seed"])
        prefix = f"seed{seed}"
        time = arrays[f"{prefix}_time_ms"]
        keep = time <= run["scientific_onset_ms"] + 800.0
        rate = arrays[f"{prefix}_rate_E_20ms_hz"][keep]
        ax.plot(1.0 - arrays[f"{prefix}_q_mean"][keep], rate,
                color=seed_colors[seed], lw=0.95, alpha=0.92,
                label=f"SNN seed {seed}")
        ax.plot(1.0 - arrays[f"{prefix}_q_core"][keep], rate,
                color=seed_colors[seed], lw=0.65, ls=":", alpha=0.65)
    ax.scatter(1.0 - q_fold, rate_fold, marker="*", s=90,
               color=FOLD, ec="white", lw=0.5, zorder=7)
    ax.axvline(1.0 - q_fold, color=FOLD, lw=0.8, ls=":")
    ax.set(xlabel=r"Disinhibition $D=1-q$", ylabel=r"Mean E rate $r_E$ (Hz)",
           xlim=(-0.005, 0.235), ylim=(-8, 455))
    ax.tick_params(labelsize=7.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper left")
    ax.set_title("Three SNN trajectories cross the reduced fold first",
                 fontsize=10, fontweight="bold")
    _panel_label(ax, "A")

    # B: exact q_core/q_mean/M/rE paper-facing projection.
    ax = axes[0, 1]
    draw_critical_manifold_trajectory(
        ax, projection, arrays, seed=1842,
        add_rate_colorbar=True, show_legend=True)
    ax.set_title("Spatial slow state approaches the high-rate skeleton",
                 fontsize=10, fontweight="bold")
    _panel_label(ax, "B")

    # C: delay-aware stability of the same branch.
    ax = axes[1, 0]
    full_path = next(
        Path(row["path"]) for row in delay["source"]["stability"]
        if float(row["history_dt_ms"]) == 0.5)
    full = json.loads(full_path.read_text())
    for branch_name, color, marker, label in (
            ("high", FOLD, "o", "high-rate branch"),
            ("low", LOW, "s", "near-silent branch")):
        rows = sorted(
            [row for row in full["points"] if row["branch"] == branch_name],
            key=lambda row: row["q"], reverse=True)
        ax.plot([1.0 - row["q"] for row in rows],
                [row["maximum_real_exponent_per_ms"] for row in rows],
                color=color, marker=marker, ms=3.4, lw=1.2, label=label)
    ax.axhline(0.0, color="0.25", lw=0.8, ls="--")
    ax.text(0.218, 0.011, "~27 Hz complex mode", color=FOLD,
            ha="right", fontsize=7.0)
    ax.set(xlabel=r"Disinhibition $D=1-q$",
           ylabel=r"Maximum Re$(\lambda)$ (ms$^{-1}$)",
           xlim=(0.102, 0.232), ylim=(-0.052, 0.068))
    ax.tick_params(labelsize=7.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    ax.set_title("Realized delays destabilize the high equilibrium",
                 fontsize=10, fontweight="bold")
    _panel_label(ax, "C")

    # D: same-branch continuation and grid-sensitive fold state.
    ax = axes[1, 1]
    grid_npz_path = Path(grid["arrays"]["path"])
    with np.load(grid_npz_path, allow_pickle=False) as grid_arrays:
        grid_colors = {10: "#6C757D", 15: "#3A86A8", 20: "#7A5195"}
        for model_row in grid["models"]:
            n_grid = int(model_row["n_grid"])
            anchors = model_row["branch_anchors"]
            q_anchor = np.asarray([row["q"] for row in anchors])
            r_anchor = np.asarray([row["mean_rate_e_hz"] for row in anchors])
            q_micro = grid_arrays[f"ngrid{n_grid}_micro_q"]
            r_micro = grid_arrays[f"ngrid{n_grid}_micro_rate_e_hz"]
            color = grid_colors[n_grid]
            label = ({10: "2 mm", 15: "1.33 mm", 20: "1 mm"}[n_grid])
            ax.plot(1.0 - q_anchor, r_anchor, color=color, lw=1.3,
                    label=label)
            # Explicitly join the last common fixed-q anchor to the local
            # pseudo-arclength refinement; the two pieces are the same branch.
            ax.plot(1.0 - np.r_[q_anchor[-1], q_micro],
                    np.r_[r_anchor[-1], r_micro], color=color, lw=1.3)
            fold = model_row["fold"]
            ax.scatter(1.0 - fold["q_from_eigenvalue_zero"],
                       fold["mean_rate_e_hz_from_eigenvalue_zero"],
                       marker="*", s=62, color=color, ec="white", lw=0.4,
                       zorder=6)
    ax.set(xlabel=r"Disinhibition $D=1-q$", ylabel=r"Mean E rate $r_E$ (Hz)",
           xlim=(0.095, 0.145), ylim=(55, 270))
    ax.tick_params(labelsize=7.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.0, title="cell width",
              title_fontsize=7.0, loc="upper left")
    ax.text(0.142, 67, "q_fold: fine pair agrees\nfold rate: not converged",
            fontsize=6.9, ha="right", va="bottom", color="0.30")
    ax.set_title("Fold existence survives conservative coarse-graining",
                 fontsize=10, fontweight="bold")
    _panel_label(ax, "D")

    output_dir = phase_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / args.stem
    outputs = _save(fig, stem)
    plt.close(fig)
    metadata = {
        "figure": args.stem,
        "status": "PATIENT_ZM_SADDLE_ORGANIZER_AUDIT_RENDERED",
        "panel_semantics": {
            "A": "three prospective OU-SNN q_core/q_mean-rate trajectories versus the 1-mm reduced critical manifold",
            "B": "representative q_core/q_mean/M/rE trajectory versus the same manifold in D-A coordinates",
            "C": "delay-aware linear stability; self-consistent stationary-diffusion variance closure",
            "D": "same-branch identity and fold sensitivity at 2, 1.33 and 1 mm conservative reductions",
        },
        "sources": {
            "projection": {"path": str(projection_path),
                           "sha256": sha256(projection_path)},
            "delay_stability": {"path": str(delay_path),
                                "sha256": sha256(delay_path)},
            "grid_convergence": {"path": str(grid_path),
                                 "sha256": sha256(grid_path)},
            "grid_arrays": {"path": str(grid_npz_path),
                            "sha256": sha256(grid_npz_path)},
            "producer": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256(Path(__file__).resolve())},
            "critical_manifold_renderer": {
                "path": str((ROOT / "src/topic4_patient_zm_figure.py").resolve()),
                "sha256": sha256(ROOT / "src/topic4_patient_zm_figure.py")},
        },
        "outputs": outputs,
        "claim_boundary": (
            "generic saddle-node in conservative deterministic reductions and "
            "organizer-level consistency with finite OU-SNN trajectories; not "
            "a thermodynamic phase transition, exact finite-SNN threshold, stable "
            "high fixed point or nonlinear limit-cycle proof"),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    metadata_path = stem.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    _append_readme(output_dir / "README.md", args.stem)
    print(json.dumps({
        "status": metadata["status"], "outputs": outputs,
        "metadata": str(metadata_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
