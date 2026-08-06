#!/usr/bin/env python3
"""Render the Stage-0D diagnostic figure from preserved numeric artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0d-plot")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_spatial_slowfast_stage0b import ForkClassifierThresholds, classify_rate_trace  # noqa: E402


RESULT = ROOT / "results/topic4_sef_hfo/spatial_slowfast_topology/stage0d_local_basin_replication"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads((RESULT / "stage0d_local_basin_summary.json").read_text(encoding="utf-8"))
    phase = json.loads((RESULT / "phase_source.json").read_text(encoding="utf-8"))
    points = json.loads((RESULT / "parameter_point_outcomes.json").read_text(encoding="utf-8"))
    rows = json.loads((RESULT / "fork_outcomes.json").read_text(encoding="utf-8"))
    source = np.load(RESULT / "phase_source_traces.npz")
    confirm = np.load(RESULT / "confirm_traces.npz")
    refined = np.load(RESULT / "dt_half_traces.npz")

    confirm_original = [index for index, row in enumerate(rows) if row["screen_status"] == "candidate_survives"]
    refined_original = [index for index, row in enumerate(rows) if row["dt_half_classification"] is not None]
    confirm_map = {original: local for local, original in enumerate(confirm_original)}
    refined_map = {original: local for local, original in enumerate(refined_original)}

    figures = RESULT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(source["time_ms"] / 1000.0, 1000.0 * source["rE_khz"][:, 0], color="#3b528b", lw=1.0)
    for row in phase["phase_selection"]:
        ax.axvline(float(row["time_ms"]) / 1000.0, color="#e76f51", lw=0.85, alpha=0.85)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="A  Locked phase source")

    ax = axes[0, 1]
    centre = next(
        index
        for index, row in enumerate(rows)
        if row["z"] == 0.85
        and row["alpha_G"] == 16.0
        and row["phase_id"] == "phase_050"
        and row["history"] == "phase_anchor"
    )
    centre_local = confirm_map[centre]
    centre_metrics = classify_rate_trace(
        confirm["time_ms"], confirm["rE_khz"][:, centre_local], ForkClassifierThresholds()
    )
    survivor = next(index for index, row in enumerate(rows) if row["final_status"] == "candidate_survives")
    survivor_local = refined_map[survivor]
    mask_confirm = confirm["time_ms"] >= 18000.0
    mask_refined = refined["time_ms"] >= 18000.0
    ax.plot(
        confirm["time_ms"][mask_confirm] / 1000.0,
        1000.0 * confirm["rE_khz"][mask_confirm, centre_local],
        color="#9e9e9e",
        lw=1.0,
        label=f"centre unresolved (power={centre_metrics['spectral_power_ratio']:.3f})",
    )
    ax.plot(
        refined["time_ms"][mask_refined] / 1000.0,
        1000.0 * refined["rE_khz"][mask_refined, survivor_local],
        color="#21918c",
        lw=0.9,
        alpha=0.85,
        label=r"survivor ($z=.85,\alpha_G=15$)",
    )
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="B  Similar waveform, different locked status")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1, 0]
    grid = np.zeros((3, 3), dtype=float)
    for row in points:
        zi = (0.84, 0.85, 0.86).index(round(float(row["z"]), 2))
        ai = (15.0, 16.0, 17.0).index(float(row["alpha_G"]))
        grid[ai, zi] = float(row["n_off_orbit_survivors"])
    image = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=16, aspect="auto")
    for ai in range(3):
        for zi in range(3):
            ax.text(zi, ai, f"{int(grid[ai, zi])}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(3), ("0.84", "0.85", "0.86"))
    ax.set_yticks(range(3), ("15", "16", "17"))
    ax.set(xlabel="z", ylabel=r"$\alpha_G$", title="C  Off-orbit final survivors")
    fig.colorbar(image, ax=ax, shrink=0.78)

    ax = axes[1, 1]
    for row in points:
        if row["mean_frequency_hz"] is None:
            continue
        marker = "*" if row["is_centre"] else ("o" if row["is_manhattan_neighbour"] else "s")
        color = "#2a9d8f" if row["open_local_basin_support"] else "#e9c46a"
        ax.scatter(row["mean_frequency_hz"], row["mean_amplitude_hz"], marker=marker, s=80, color=color)
        ax.annotate(f"{row['z']:.2f},{row['alpha_G']:.0f}", (row["mean_frequency_hz"], row["mean_amplitude_hz"]), fontsize=8)
    ax.set(xlabel="frequency (Hz)", ylabel="peak-to-trough amplitude (Hz)", title="D  Same-object gate")
    ax.text(
        0.03,
        0.95,
        "4 off-orbit survivors, but only one phase",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#8a5a00",
    )
    fig.suptitle(summary["verdict"].replace("_", " "), fontsize=11)
    png = figures / "stage0d_local_basin_replication.png"
    pdf = figures / "stage0d_local_basin_replication.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    readme = (
        "### stage0d_local_basin_replication.png\n\n"
        "A 显示锁定 phase source 及四个相位；B 对比中心点被锁定分类器判为 unresolved 的轨迹与唯一 survivor 组，说明 unresolved 不等于没有振荡。"
        "C 显示每个参数点最终 off-orbit survivor 数；D 显示 survivor 的频率和振幅。\n\n"
        "**关注点**：`z=0.85, alpha_G=15` 有 fast/pool 四条 off-orbit survivor，但都来自同一相位；中心点也未通过 confirm，因此不能称 open basin 或邻域复制。\n"
    )
    (figures / "README.md").write_text(readme, encoding="utf-8")
    metadata = {
        "schema_version": "topic4_stage0d_figure.v1",
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "numeric_summary": str((RESULT / "stage0d_local_basin_summary.json").resolve()),
        "numeric_summary_sha256": _sha256(RESULT / "stage0d_local_basin_summary.json"),
        "png": str(png.resolve()),
        "png_sha256": _sha256(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": _sha256(pdf),
        "visualization_only": True,
        "posthoc_diagnostic_not_acceptance": True,
        "displayed_centre_confirm_metrics": centre_metrics,
        "displayed_survivor_final_metrics": {
            "z": rows[survivor]["z"],
            "alpha_G": rows[survivor]["alpha_G"],
            "phase_id": rows[survivor]["phase_id"],
            "history": rows[survivor]["history"],
            "dt_half_tail_mean_hz": rows[survivor]["dt_half_tail_mean_hz"],
            "dt_half_frequency_hz": rows[survivor]["dt_half_frequency_hz"],
            "dt_half_amplitude_hz": rows[survivor]["dt_half_amplitude_hz"],
        },
    }
    (RESULT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
