#!/usr/bin/env python3
"""Aggregate the locked LC5v2.1 base map and boundary patch after both are complete."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5v2p1_phase_map as MAP  # noqa: E402


BASE_MANIFEST = ROOT / "config/topic4_fcxr_lc5v2p1_timescale_dose_map.json"
PATCH_MANIFEST = ROOT / "config/topic4_fcxr_lc5v2p1_boundary_patch.json"
OUT = MAP.PREFIX.U2.OUT / "lc5v2p1_joint_phase_map"
FIGURES = OUT / "figures"

OUTCOME_ORDER = [
    "ESCALATING_SATURATION",
    "CONTAINED_HIGH_NO_OFFSET",
    "FINITE_EXCURSION_CANDIDATE",
    "OFFSET_OUTSIDE_TARGET",
    "ENTRY_BLOCKED_WITH_IED",
    "BASELINE_SUPPRESSED",
]
OUTCOME_COLORS = {
    "ESCALATING_SATURATION": "#B2182B",
    "CONTAINED_HIGH_NO_OFFSET": "#E69F00",
    "FINITE_EXCURSION_CANDIDATE": "#2CA25F",
    "OFFSET_OUTSIDE_TARGET": "#66C2A4",
    "ENTRY_BLOCKED_WITH_IED": "#2166AC",
    "BASELINE_SUPPRESSED": "#6A51A3",
}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(MAP.PREFIX.json_sanitize(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def summary_path(manifest, cell, tau_ms, gamma):
    reuse = manifest["reuse"]["eligible_cells"][cell]
    if reuse is not None:
        return ROOT / reuse / "summary.json"
    tag = MAP.PREFIX._tag(gamma, "q099", tau_ms, manifest["experiment_id"])
    return MAP.PREFIX.U2.OUT / tag / "summary.json"


def collect_rows():
    rows = []
    for manifest_path in (BASE_MANIFEST, PATCH_MANIFEST):
        _, manifest, cells = MAP.load_manifest(manifest_path)
        for cell, (tau_ms, gamma) in cells.items():
            path = summary_path(manifest, cell, tau_ms, gamma)
            if not path.is_file():
                raise RuntimeError(f"PHASE_MAP_INCOMPLETE: missing {path}")
            summary = json.loads(path.read_text())
            if summary.get("status") != "COMPLETE":
                raise RuntimeError(f"non-complete summary: {path}")
            if not np.isclose(float(summary["tau_ms"]), tau_ms, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"tau mismatch: {path}")
            if not np.isclose(float(summary["gamma_nominal_dose"]), gamma, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Gamma mismatch: {path}")
            onset = summary.get("onset_ms")
            offset = summary.get("offset_ms")
            post_onset = None if onset is None else float(summary["T_ms"]) - float(onset)
            rows.append({
                "experiment_id": manifest["experiment_id"],
                "cell": cell,
                "tau_ms": float(tau_ms),
                "tau_s": float(tau_ms) / 1000.0,
                "gamma": float(gamma),
                "outcome": summary["outcome"],
                "T_ms": float(summary["T_ms"]),
                "onset_ms": None if onset is None else float(onset),
                "offset_ms": None if offset is None else float(offset),
                "post_onset_observed_ms": post_onset,
                "n_returning": int(summary["n_returning"]),
                "mean_rate_hz": float(summary["mean_rate_hz"]),
                "end_rate_hz": float(summary["end_rate_hz"]),
                "pump_current_peak_mean": float(summary["pump_current_peak_mean"]),
                "achieved_dose_median": summary.get("achieved_population_mean_ratio_median"),
                "achieved_dose_peak": summary.get("achieved_population_mean_ratio_peak"),
                "per_second_mean_rate_hz": summary["per_second_mean_rate_hz"],
                "source_summary": str(path),
            })
    if len(rows) != 20 or len({(r["tau_ms"], r["gamma"]) for r in rows}) != 20:
        raise RuntimeError("joint map must contain 20 unique cells")
    return sorted(rows, key=lambda r: (r["tau_ms"], r["gamma"]))


def choose_extension_candidate(rows):
    eligible = [
        row for row in rows
        if row["outcome"] in {"FINITE_EXCURSION_CANDIDATE", "CONTAINED_HIGH_NO_OFFSET"}
    ]
    if not eligible:
        return None
    def key(row):
        finite_priority = 0 if row["outcome"] == "FINITE_EXCURSION_CANDIDATE" else 1
        observed = row["post_onset_observed_ms"] or 0.0
        return (finite_priority, -observed, row["end_rate_hz"], row["tau_ms"], row["gamma"])
    return min(eligible, key=key)


def _csv(rows, path):
    fields = [key for key in rows[0] if key != "per_second_mean_rate_hz"]
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _plot(rows, candidate):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for outcome in OUTCOME_ORDER:
        selected = [r for r in rows if r["outcome"] == outcome]
        if not selected:
            continue
        ax.scatter(
            [r["gamma"] for r in selected], [r["tau_s"] for r in selected],
            s=90, color=OUTCOME_COLORS[outcome], edgecolor="white", linewidth=0.8,
            label=outcome.replace("_", " ").lower(), zorder=3,
        )
    if candidate is not None:
        ax.scatter([candidate["gamma"]], [candidate["tau_s"]], s=230, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([.001, .002, .003, .004, .005, .006, .008, .01, .02, .04, .06])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks([3, 8, 15])
    ax.set_xlabel(r"Nominal early-episode dose $\Gamma_U$")
    ax.set_ylabel(r"Episode-memory timescale $\tau_U$ (s)")
    ax.set_title("a  Dynamical outcome")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[0, 1]
    onset = [np.nan if r["onset_ms"] is None else r["onset_ms"] / 1000.0 for r in rows]
    sc = ax.scatter([r["gamma"] for r in rows], [r["tau_s"] for r in rows], c=onset,
                    cmap="viridis", vmin=10, vmax=25, s=95, edgecolor="black", linewidth=.4)
    for row, value in zip(rows, onset):
        if np.isnan(value):
            ax.text(row["gamma"], row["tau_s"], "×", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    ax.set_xscale("log"); ax.set_yticks([3, 8, 15])
    ax.set_xlabel(r"$\Gamma_U$"); ax.set_ylabel(r"$\tau_U$ (s)")
    ax.set_title("b  Natural-onset latency (× = no onset by 25 s)")
    fig.colorbar(sc, ax=ax, label="Onset time (s)")

    ax = axes[1, 0]
    for tau_s, color in [(3, "#B2182B"), (8, "#6A51A3"), (15, "#2166AC")]:
        selected = [r for r in rows if r["tau_s"] == tau_s]
        ax.plot([r["gamma"] for r in selected], [r["end_rate_hz"] for r in selected],
                "o-", color=color, label=fr"$\tau_U={tau_s}$ s")
    ax.axhline(250, color="0.45", linestyle="--", linewidth=1, label="registered saturation")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Gamma_U$"); ax.set_ylabel("Final 1-s E rate (Hz)")
    ax.set_title("c  End-state activity")
    ax.grid(alpha=.18, which="both"); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    representatives = []
    for outcome in ("ESCALATING_SATURATION", "CONTAINED_HIGH_NO_OFFSET", "ENTRY_BLOCKED_WITH_IED"):
        selected = [r for r in rows if r["outcome"] == outcome]
        if outcome == "CONTAINED_HIGH_NO_OFFSET" and candidate is not None:
            selected = [candidate]
        if selected:
            representatives.append(min(selected, key=lambda r: abs(r["tau_s"] - 8) + r["gamma"]))
    for row in representatives:
        rate = np.asarray(row["per_second_mean_rate_hz"], float)
        time_s = np.arange(rate.size) + .5
        ax.plot(time_s, rate, linewidth=1.8, color=OUTCOME_COLORS[row["outcome"]],
                label=fr"$\tau={row['tau_s']:g}$ s, $\Gamma={row['gamma']:g}$: "
                      + row["outcome"].replace("_", " ").lower())
    ax.set_yscale("log")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Mean E rate (Hz)")
    ax.set_title("d  Representative natural trajectories")
    ax.grid(alpha=.18, which="both"); ax.legend(frameon=False, fontsize=7)

    fig.suptitle("FCXR-LC5v2.1: cell-local episode memory opens a sharp entry boundary", fontsize=14)
    png = FIGURES / "lc5v2p1_joint_phase_map.png"
    pdf = FIGURES / "lc5v2p1_joint_phase_map.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run():
    rows = collect_rows()
    candidate = choose_extension_candidate(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    _csv(rows, OUT / "phase_map.csv")
    payload = {
        "status": "COMPLETE",
        "n_cells": len(rows),
        "outcome_counts": {name: sum(r["outcome"] == name for r in rows) for name in OUTCOME_ORDER},
        "extension_selection_rule": (
            "finite before contained; then longest observed post-onset duration; then lowest end rate"
        ),
        "primary_extension_candidate": candidate,
        "rows": rows,
        "claim_boundary": (
            "A contained cell is a lifecycle-scaffold candidate, not evidence of offset, recovery, "
            "returning IED recovery, or patient-like seizure morphology."
        ),
    }
    _write_json(OUT / "phase_map.json", payload)
    png, pdf = _plot(rows, candidate)
    readme = """### lc5v2p1_joint_phase_map.png

这张诊断图联合基础 3×3 与沿边界补的 11 格。a 显示每格最终动力学类别，黑圈是按预锁规则选出的唯一续跑候选；b 区分自然进入、延迟进入与 25 秒内未进入；c 显示末端活动是否仍在饱和；d 对照代表性自然轨迹。

**关注点**：这张图只定位逐细胞 episode-memory 的 containment/entry 边界。contained 不是自主终止，更不是完整 lifecycle；只有续跑出现 offset、爆后保护、Z 恢复和 returning IED，才可升级结论。

### lc5v2p1_joint_phase_map.pdf

与 PNG 内容相同的矢量版本，用于放大核对边界和轨迹。

**关注点**：图中 × 表示到 25 秒仍未自然进入，不等于永久无法进入。
"""
    (FIGURES / "README.md").write_text(readme)
    _write_json(FIGURES / "lc5v2p1_joint_phase_map_metadata.json", {
        "source": str(OUT / "phase_map.json"), "png": str(png), "pdf": str(pdf),
        "n_cells": len(rows), "candidate": candidate,
    })
    return payload


if __name__ == "__main__":
    print(json.dumps(MAP.PREFIX.json_sanitize(run()), indent=2, sort_keys=True))
