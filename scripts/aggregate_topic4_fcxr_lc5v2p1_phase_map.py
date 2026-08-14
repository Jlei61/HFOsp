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
EXTENSION_SUMMARY = (
    MAP.PREFIX.U2.OUT
    / "lc5v2p1_candidate_extension_tau3000_gamma0060"
    / "summary.json"
)

OUTCOME_ORDER = [
    "ESCALATING_SATURATION",
    "CONTAINED_HIGH_NO_OFFSET",
    "FINITE_EXCURSION_CANDIDATE",
    "OFFSET_OUTSIDE_TARGET",
    "RIGHT_CENSORED_CONTAINMENT_CANDIDATE",
    "ENTRY_BLOCKED_WITH_IED",
    "BASELINE_SUPPRESSED",
]
OUTCOME_COLORS = {
    "ESCALATING_SATURATION": "#B2182B",
    "CONTAINED_HIGH_NO_OFFSET": "#E69F00",
    "FINITE_EXCURSION_CANDIDATE": "#2CA25F",
    "OFFSET_OUTSIDE_TARGET": "#66C2A4",
    "RIGHT_CENSORED_CONTAINMENT_CANDIDATE": "#F2B701",
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
                "screen_outcome": summary["outcome"],
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


def merge_extension(rows, extension):
    """Merge the one pre-registered exact-state continuation without erasing screen evidence."""
    merged = [dict(row) for row in rows]
    for row in merged:
        row.setdefault("screen_outcome", row["outcome"])
        row["adjudicated_outcome"] = row["screen_outcome"]
        row["extension_outcome"] = None
        row["extension_summary"] = None
    if extension is None:
        return merged, None
    if extension.get("status") != "COMPLETE":
        raise RuntimeError("candidate extension is not complete")
    source = Path(extension["source_summary"]).resolve()
    matches = [row for row in merged if Path(row["source_summary"]).resolve() == source]
    if len(matches) != 1:
        raise RuntimeError(f"candidate extension source must match exactly one map row: {source}")
    row = matches[0]
    if not np.isclose(row["tau_ms"], float(extension["tau_ms"]), rtol=0.0, atol=1e-12):
        raise RuntimeError("candidate extension tau mismatch")
    if not np.isclose(
        row["gamma"], float(extension["gamma_nominal_dose"]), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("candidate extension Gamma mismatch")
    if row["screen_outcome"] != extension.get("source_outcome"):
        raise RuntimeError("candidate extension source-outcome mismatch")

    row["screen_post_onset_observed_ms"] = row["post_onset_observed_ms"]
    for key in (
        "T_ms", "onset_ms", "offset_ms", "n_returning", "mean_rate_hz", "end_rate_hz",
        "per_second_mean_rate_hz",
    ):
        row[f"screen_{key}"] = row[key]
        row[key] = extension[key]
    row["post_onset_observed_ms"] = (
        None if row["onset_ms"] is None else float(row["T_ms"]) - float(row["onset_ms"])
    )
    row["extension_outcome"] = extension["outcome"]
    row["adjudicated_outcome"] = extension["outcome"]
    row["extension_summary"] = str(EXTENSION_SUMMARY)
    row["extension_early_stop_reason"] = extension.get("early_stop_reason")
    return merged, {
        "source_summary": str(source),
        "extension_summary": str(EXTENSION_SUMMARY),
        "screen_outcome": extension["source_outcome"],
        "adjudicated_outcome": extension["outcome"],
        "T_ms": extension["T_ms"],
        "onset_ms": extension.get("onset_ms"),
        "offset_ms": extension.get("offset_ms"),
        "end_rate_hz": extension["end_rate_hz"],
        "early_stop_reason": extension.get("early_stop_reason"),
    }


def evidence_class(row):
    """Separate a late-onset, short-follow-up hint from demonstrated containment."""
    outcome = row.get("adjudicated_outcome", row["outcome"])
    if outcome == "CONTAINED_HIGH_NO_OFFSET":
        observed = row.get("post_onset_observed_ms")
        if observed is None or float(observed) < 7000.0:
            return "RIGHT_CENSORED_CONTAINMENT_CANDIDATE"
    return outcome


def load_extension():
    if not EXTENSION_SUMMARY.is_file():
        return None
    return json.loads(EXTENSION_SUMMARY.read_text())


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
    fields = sorted({key for row in rows for key in row if key != "per_second_mean_rate_hz"})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _plot(rows, extension_result):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for outcome in OUTCOME_ORDER:
        selected = [r for r in rows if r["final_evidence_class"] == outcome]
        if not selected:
            continue
        ax.scatter(
            [r["gamma"] for r in selected], [r["tau_s"] for r in selected],
            s=90, color=OUTCOME_COLORS[outcome], edgecolor="white", linewidth=0.8,
            label=outcome.replace("_", " ").lower(), zorder=3,
        )
    if extension_result is not None:
        selected = [r for r in rows if r.get("extension_summary")]
        ax.scatter([r["gamma"] for r in selected], [r["tau_s"] for r in selected], s=175,
                   marker="*", facecolors="white", edgecolors="black", linewidths=1.2,
                   label="exact-state extension", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([.001, .003, .006, .01, .02, .06])
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
                    color="#2166AC", fontsize=12, fontweight="bold")
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
    for outcome in (
        "ESCALATING_SATURATION", "RIGHT_CENSORED_CONTAINMENT_CANDIDATE",
        "ENTRY_BLOCKED_WITH_IED",
    ):
        selected = [r for r in rows if r["final_evidence_class"] == outcome]
        if selected:
            representatives.append(min(selected, key=lambda r: abs(r["tau_s"] - 8) + r["gamma"]))
    for row in representatives:
        rate = np.asarray(row["per_second_mean_rate_hz"], float)
        time_s = np.arange(rate.size) + .5
        ax.plot(time_s, rate, linewidth=1.8, color=OUTCOME_COLORS[row["final_evidence_class"]],
                label=fr"$\tau={row['tau_s']:g}$ s, $\Gamma={row['gamma']:g}$: "
                      + row["final_evidence_class"].replace("_", " ").lower())
    ax.set_yscale("log")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Mean E rate (Hz)")
    ax.set_title("d  Representative natural trajectories")
    ax.grid(alpha=.18, which="both"); ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "FCXR-LC5v2.1: sharp saturation-to-entry-blocked boundary, no offset", fontsize=14
    )
    png = FIGURES / "lc5v2p1_joint_phase_map.png"
    pdf = FIGURES / "lc5v2p1_joint_phase_map.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run():
    screen_rows = collect_rows()
    selected_candidate = choose_extension_candidate(screen_rows)
    rows, extension_result = merge_extension(screen_rows, load_extension())
    for row in rows:
        row["final_evidence_class"] = evidence_class(row)
    OUT.mkdir(parents=True, exist_ok=True)
    _csv(rows, OUT / "phase_map.csv")
    screen_counts = {
        name: sum(r["screen_outcome"] == name for r in rows) for name in OUTCOME_ORDER
    }
    adjudicated_counts = {
        name: sum(r["adjudicated_outcome"] == name for r in rows) for name in OUTCOME_ORDER
    }
    evidence_counts = {
        name: sum(r["final_evidence_class"] == name for r in rows) for name in OUTCOME_ORDER
    }
    payload = {
        "status": "COMPLETE",
        "n_cells": len(rows),
        "outcome_counts": adjudicated_counts,
        "screen_outcome_counts": screen_counts,
        "adjudicated_outcome_counts": adjudicated_counts,
        "final_evidence_class_counts": evidence_counts,
        "extension_selection_rule": (
            "finite before contained; then longest observed post-onset duration; then lowest end rate"
        ),
        "primary_extension_candidate": selected_candidate,
        "extension_result": extension_result,
        "open_extension_candidate": None,
        "rows": rows,
        "claim_boundary": (
            "The pre-registered extension reclassified tau=3 s, Gamma=0.060 as escalating "
            "saturation. The remaining late-onset tau=15 s, Gamma=0.003 cell has only 2 s of "
            "post-onset follow-up and is right-censored. No offset, postictal protection, Z "
            "recovery, or returning-IED recovery was observed."
        ),
    }
    _write_json(OUT / "phase_map.json", payload)
    png, pdf = _plot(rows, extension_result)
    readme = """### lc5v2p1_joint_phase_map.png

这张收口图联合基础 3×3、沿边界补的 11 格和唯一一次预注册续跑。a 的白色星号标出被续跑的 `tau=3 s, Gamma=0.060`：它在原 18 秒窗内短暂看似 contained，继续 1 秒便超过注册饱和线，最终按饱和计；黄色点是 `tau=15 s, Gamma=0.003`，但 onset 后只观察到 2 秒，因此仅作右删失线索。b 区分自然进入与 25 秒内未进入；c 显示末端活动；d 对照代表性轨迹。

**关注点**：20 个条件中没有观察到 offset。最终证据是 11 格升级饱和、8 格保持 IED 但阻断进入、1 格因晚进入而右删失；没有 postictal、Z 恢复或 returning IED recovery。

### lc5v2p1_joint_phase_map.pdf

与 PNG 内容相同的矢量版本，用于放大核对边界和轨迹。

**关注点**：图中 × 表示到 25 秒仍未自然进入，不等于永久无法进入。
"""
    (FIGURES / "README.md").write_text(readme)
    _write_json(FIGURES / "lc5v2p1_joint_phase_map_metadata.json", {
        "source": str(OUT / "phase_map.json"), "png": str(png), "pdf": str(pdf),
        "n_cells": len(rows), "candidate": selected_candidate,
        "extension_result": extension_result,
    })
    return payload


if __name__ == "__main__":
    print(json.dumps(MAP.PREFIX.json_sanitize(run()), indent=2, sort_keys=True))
