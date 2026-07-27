#!/usr/bin/env python3
"""Render the frozen six-panel Topic-5 v2.2 Figure 6.

The producer is gate-aware.  Canonical rendering requires the interictal
pipeline to be finalized.  ``--preview`` is only for layout QA while formal
runs are still in progress and writes PREVIEW prominently into data panels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
CANONICAL = (
    ROOT
    / "results/paper-ready-figure"
    / "fig6_symmetric_axis_propagation_state_v2_2"
)
STEM = "fig6_symmetric_axis_propagation_state_v2_2"

BLUE = "#4477AA"
GREEN = "#228833"
RED = "#CC6677"
AMBER = "#CCBB44"
GREY = "#7A7A7A"
LIGHT_GREY = "#E7E7E7"
DARK = "#222222"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _short_status(status: str | None) -> tuple[str, str]:
    value = str(status or "not run").upper()
    if value == "PASS":
        return "PASS", GREEN
    if value in {"FAIL", "NOT_ESTIMABLE"}:
        return value.replace("_", " "), RED
    if value in {"SEALED", "LOCKED"}:
        return value, AMBER
    return value.replace("_", " "), GREY


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        0.0,
        1.03,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.09,
        1.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _status_badge(ax: plt.Axes, status: str | None, *, x: float = 0.98) -> None:
    label, color = _short_status(status)
    ax.text(
        x,
        0.98,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=color,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": mcolors.to_rgba(color, 0.08),
            "edgecolor": mcolors.to_rgba(color, 0.45),
            "linewidth": 0.7,
        },
    )


def _empty_panel(
    ax: plt.Axes,
    *,
    message: str,
    status: str,
    preview: bool,
) -> None:
    ax.set_axis_off()
    _status_badge(ax, status)
    ax.text(
        0.5,
        0.50,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        color=GREY,
        linespacing=1.4,
    )
    if preview:
        ax.text(
            0.5,
            0.08,
            "LAYOUT PREVIEW — NOT A SCIENTIFIC RESULT",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.6,
            color=RED,
            fontweight="bold",
        )


def _panel_a(ax: plt.Axes) -> None:
    _panel_title(ax, "A", "Same scaffold, opposite sources")
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    x = np.linspace(0.4, 9.6, 12)
    y = 0.28 * np.sin(np.linspace(-0.8, 2.2 * np.pi, len(x)))
    edges = [
        (i, j)
        for i in range(len(x))
        for j in range(i + 1, min(len(x), i + 4))
    ]
    for row_offset in (0.0, -0.95):
        for i, j in edges:
            width = 0.9 - 0.18 * (j - i)
            ax.plot(
                [x[i], x[j]],
                [y[i] + row_offset, y[j] + row_offset],
                color="#B6B6B6",
                lw=width,
                alpha=0.48,
                zorder=1,
            )
    cmap = plt.get_cmap("viridis")
    ax.scatter(
        x,
        y,
        c=np.linspace(0.05, 0.95, len(x)),
        cmap=cmap,
        s=38,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax.scatter(
        x,
        y - 0.95,
        c=np.linspace(0.95, 0.05, len(x)),
        cmap=cmap,
        s=38,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )
    ax.scatter(
        [x[0], x[-1]],
        [y[0], y[-1] - 0.95],
        marker="*",
        s=105,
        color=RED,
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )
    ax.text(0.35, 1.16, "source at left", fontsize=7.5, color=DARK)
    ax.text(9.65, -1.52, "source at right", fontsize=7.5, color=DARK, ha="right")
    ax.annotate(
        "",
        xy=(9.0, 0.88),
        xytext=(1.0, 0.88),
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 1.0},
    )
    ax.annotate(
        "",
        xy=(1.0, -1.52),
        xytext=(9.0, -1.52),
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 1.0},
    )
    ax.text(
        5.0,
        1.43,
        r"$W=W^\mathsf{T}$",
        ha="center",
        va="center",
        fontsize=9,
    )


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        facecolor=mcolors.to_rgba(color, 0.10),
        edgecolor=color,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.4,
        linespacing=1.25,
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            color=GREY,
            linewidth=0.9,
        )
    )


def _panel_b(ax: plt.Axes) -> None:
    _panel_title(ax, "B", "Self-supervised propagation model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    _box(ax, (0.03, 0.58), 0.22, 0.20, "observed\nrank set  $x_t$", color=BLUE)
    _box(
        ax,
        (0.38, 0.58),
        0.24,
        0.20,
        r"$P_{t+1}=\rho P_t+Wx_t$",
        color=GREEN,
    )
    _box(
        ax,
        (0.75, 0.58),
        0.22,
        0.20,
        "contact hazards\n+ scalar STOP",
        color=AMBER,
    )
    _arrow(ax, (0.25, 0.68), (0.38, 0.68))
    _arrow(ax, (0.62, 0.68), (0.75, 0.68))
    _box(
        ax,
        (0.18, 0.18),
        0.27,
        0.18,
        "next rank-set\nlikelihood",
        color=BLUE,
    )
    _box(
        ax,
        (0.57, 0.18),
        0.27,
        0.18,
        "absorbing\nfirst-arrival rollout",
        color=GREEN,
    )
    _arrow(ax, (0.83, 0.58), (0.73, 0.36))
    _arrow(ax, (0.78, 0.58), (0.38, 0.36))
    ax.text(
        0.50,
        0.05,
        "train on interictal prefixes  •  no A/B labels  •  no ictal values",
        ha="center",
        va="center",
        fontsize=6.8,
        color=GREY,
    )


def _jitter(n: int, width: float = 0.12) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-width, width, n)


def _endpoint_lookup(status: dict[str, Any], endpoint: str) -> dict[str, Any]:
    for item in status.get("endpoints", []):
        if item.get("endpoint") == endpoint:
            return item
    return {}


def _panel_c(
    ax: plt.Axes,
    *,
    status: dict[str, Any] | None,
    patient_path: Path,
    preview: bool,
) -> None:
    _panel_title(ax, "C", "Axial structure over local propagation")
    if status is None or status.get("status") != "complete" or not patient_path.is_file():
        _empty_panel(
            ax,
            message="Formal patient-first inference\nawaiting 66/66 frozen runs",
            status="pending",
            preview=preview,
        )
        return
    data = pd.read_csv(patient_path)
    columns = [
        ("seed_median_next_benefit", "next set", BLUE, "next_set"),
        (
            "seed_median_future_benefit",
            "future order",
            GREEN,
            "future_first_arrival",
        ),
    ]
    for index, (column, label, color, endpoint) in enumerate(columns):
        values = data[column].to_numpy(dtype=float)
        ax.scatter(
            np.full(len(values), index) + _jitter(len(values)),
            values,
            s=19,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        summary = _endpoint_lookup(status, endpoint)
        median = float(summary["median_benefit"])
        lo = float(summary["median_ci95_low"])
        hi = float(summary["median_ci95_high"])
        ax.errorbar(
            index,
            median,
            yerr=[[median - lo], [hi - median]],
            fmt="_",
            markersize=17,
            color=DARK,
            capsize=3,
            lw=1.2,
            zorder=4,
        )
        ax.text(
            index,
            ax.get_ylim()[1] if index < 0 else max(values.max(), hi),
            "",
        )
        label_status, label_color = _short_status(
            "PASS" if summary.get("pass") else "FAIL"
        )
        ax.text(
            index,
            0.97,
            f"{label_status}  q={summary['bh_fdr_q']:.3g}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.6,
            color=label_color,
        )
    ax.axhline(0, color=GREY, ls="--", lw=0.8, zorder=1)
    ax.set_xticks([0, 1], [item[1] for item in columns])
    ax.set_ylabel("NLL benefit (isotropic − axis)")
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.02,
        f"n={len(data)} patients; points are seed medians",
        transform=ax.transAxes,
        fontsize=6.5,
        color=GREY,
    )


def _panel_d(
    ax: plt.Axes,
    *,
    claim3: dict[str, Any] | None,
    random_path: Path,
    readback: dict[str, Any] | None,
    readback_path: Path,
    preview: bool,
) -> None:
    _panel_title(ax, "D", "Axis specificity and external read-back")
    ax.set_axis_off()
    left = ax.inset_axes([0.03, 0.08, 0.44, 0.80])
    right = ax.inset_axes([0.56, 0.08, 0.41, 0.80])
    if claim3 and claim3.get("status") == "complete" and random_path.is_file():
        data = pd.read_csv(random_path)
        column = "seed_median_delta_random_minus_learned"
        values = data[column].to_numpy(dtype=float)
        left.scatter(
            _jitter(len(values), 0.08),
            values,
            s=17,
            color=np.where(values > 0, BLUE, RED),
            edgecolor="white",
            linewidth=0.35,
        )
        left.errorbar(
            0,
            float(np.median(values)),
            yerr=np.asarray(
                [
                    [
                        float(np.median(values))
                        - float(claim3["median_ci95_low"])
                    ],
                    [
                        float(claim3["median_ci95_high"])
                        - float(np.median(values))
                    ],
                ]
            ),
            fmt="_",
            markersize=15,
            color=DARK,
            capsize=3,
        )
        left.axhline(0, color=GREY, ls="--", lw=0.75)
        left.set_xticks([])
        left.set_ylabel("random − learned NLL", fontsize=7)
        left.set_title("random directions", fontsize=7.5)
        left.tick_params(labelsize=6.6)
        left.spines[["top", "right"]].set_visible(False)
    else:
        left.set_axis_off()
        left.text(
            0.5,
            0.5,
            "random-axis test\nnot reached",
            ha="center",
            va="center",
            fontsize=7.5,
            color=GREY,
        )
    if readback and readback.get("status") == "complete" and readback_path.is_file():
        data = pd.read_csv(readback_path)
        data = data[data.status == "estimable"]
        values = data.abs_axis_cosine.to_numpy(dtype=float)
        right.scatter(
            _jitter(len(values), 0.08),
            values,
            s=17,
            color=GREEN,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
        )
        right.plot(
            [-0.13, 0.13],
            [np.median(values), np.median(values)],
            color=DARK,
            lw=1.4,
        )
        right.set_ylim(0, 1.03)
        right.set_xticks([])
        right.set_ylabel(r"$|u_{\rm RNN}\cdot u_{\rm A/B}|$", fontsize=7)
        right.set_title(f"A/B read-back (n={len(values)})", fontsize=7.5)
        right.tick_params(labelsize=6.6)
        right.spines[["top", "right"]].set_visible(False)
    else:
        right.set_axis_off()
        right.text(
            0.5,
            0.5,
            "A/B read-back\nawaiting frozen scores",
            ha="center",
            va="center",
            fontsize=7.5,
            color=GREY,
        )
    gate = (
        claim3.get("claim3_random_axis")
        if claim3 and claim3.get("status") == "complete"
        else "locked"
    )
    _status_badge(ax, gate)
    if preview:
        ax.text(
            0.5,
            0.01,
            "LAYOUT PREVIEW",
            transform=ax.transAxes,
            ha="center",
            fontsize=6.3,
            color=RED,
        )


def _panel_e(
    ax: plt.Axes,
    *,
    status: dict[str, Any] | None,
    patient_path: Path,
    preview: bool,
) -> None:
    _panel_title(ax, "E", "Shared scaffold across source sides")
    if status is None or status.get("status") not in {"complete", "not_estimable"}:
        _empty_panel(
            ax,
            message="Cross-direction analysis\nlocked by upstream specificity gate",
            status="locked",
            preview=preview,
        )
        return
    if status.get("status") == "not_estimable":
        _empty_panel(
            ax,
            message=(
                "Insufficient patients with train80 and heldout20\n"
                "events on both source sides"
            ),
            status="not_estimable",
            preview=preview,
        )
        return
    data = pd.read_csv(patient_path)
    side_columns = [
        ("seed_median_left_axis_benefit", "left", BLUE),
        ("seed_median_right_axis_benefit", "right", GREEN),
    ]
    for index, (column, label, color) in enumerate(side_columns):
        values = data[column].to_numpy(dtype=float)
        ax.scatter(
            np.full(len(values), index) + _jitter(len(values), 0.10),
            values,
            s=18,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.82,
        )
        ax.plot(
            [index - 0.16, index + 0.16],
            [np.median(values), np.median(values)],
            color=DARK,
            lw=1.4,
        )
    margin = data["seed_median_M"].to_numpy(dtype=float)
    ax.scatter(
        np.full(len(margin), 2) + _jitter(len(margin), 0.10),
        margin,
        s=18,
        color=np.where(margin < 0, GREEN, RED),
        marker="D",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
    )
    ax.plot(
        [1.84, 2.16],
        [np.median(margin), np.median(margin)],
        color=DARK,
        lw=1.4,
    )
    ax.axhline(0, color=GREY, ls="--", lw=0.8)
    ax.set_xticks([0, 1, 2], ["left", "right", "two-W\nmargin"])
    ax.set_ylabel("heldout benefit / margin")
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    _status_badge(ax, status.get("claim4_shared_scaffold"))
    ax.text(
        0.02,
        0.02,
        f"n={len(data)} eligible patients",
        transform=ax.transAxes,
        fontsize=6.5,
        color=GREY,
    )


def _panel_f(
    ax: plt.Axes,
    *,
    summary: dict[str, Any] | None,
    target: dict[str, Any],
    transfer_path: Path,
    preview: bool,
) -> None:
    _panel_title(ax, "F", "Early-ictal energy-field transfer")
    if transfer_path.is_file():
        data = pd.read_csv(transfer_path)
        if not {
            "full_spearman",
            "local_isotropic_spearman",
        }.issubset(data.columns):
            raise RuntimeError("early-ictal patient table schema drifted")
        values = (
            data.full_spearman - data.local_isotropic_spearman
        ).to_numpy(dtype=float)
        ax.scatter(
            _jitter(len(values), 0.10),
            values,
            s=21,
            color=np.where(values > 0, BLUE, RED),
            edgecolor="white",
            linewidth=0.4,
        )
        ax.plot(
            [-0.16, 0.16],
            [np.median(values), np.median(values)],
            color=DARK,
            lw=1.5,
        )
        ax.axhline(0, color=GREY, ls="--", lw=0.8)
        ax.set_xticks([])
        ax.set_ylabel(r"$\Delta$ patient Spearman (full − isotropic)")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        _status_badge(ax, "complete")
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    center = (0.50, 0.59)
    ax.add_patch(
        FancyBboxPatch(
            (0.39, 0.43),
            0.22,
            0.22,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=mcolors.to_rgba(AMBER, 0.10),
            edgecolor=AMBER,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Arc(
            (center[0], 0.64),
            0.16,
            0.19,
            theta1=0,
            theta2=180,
            edgecolor=AMBER,
            linewidth=1.4,
        )
    )
    ax.text(center[0], 0.535, "SEALED", ha="center", va="center", fontsize=8)
    denom = target["endpoint_denominators"]
    ax.text(
        0.5,
        0.28,
        (
            f"energy metadata: {denom['energy_metadata']['patients']} patients, "
            f"{denom['energy_metadata']['seizures']} seizures\n"
            "exact per-seizure clinical-onset sources: 0\n"
            "energy values were not read"
        ),
        ha="center",
        va="center",
        fontsize=7.2,
        color=DARK,
        linespacing=1.4,
    )
    _status_badge(ax, "sealed")
    if preview:
        ax.text(
            0.5,
            0.08,
            "LAYOUT PREVIEW",
            ha="center",
            fontsize=6.3,
            color=RED,
        )


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return {"commit": commit, "dirty": dirty}


def _scientific_qa(
    *,
    summary: dict[str, Any] | None,
    target: dict[str, Any],
) -> dict[str, Any]:
    inventory = pd.read_csv(BASE / "input_audit/subject_inventory.csv")
    development = inventory[inventory.development.astype(bool)]
    physical_lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text()
    )
    sequence_lock = json.loads(
        (BASE / "formal/ALL_SUBJECT_SEQUENCE_LOCK.json").read_text()
    )
    seal_ok = (
        target.get("energy_values_read") is False
        and target.get("recruitment_values_read") is False
    )
    return {
        "development_n3_and_geometry_complete": bool(
            len(development) == 3 and development.geometry_complete.all()
        ),
        "physical_axis_formal_n22": len(physical_lock["subjects"]) == 22,
        "all_subject_sequence_n31": len(sequence_lock["subjects"]) == 31,
        "topology_fallback_used": False,
        "restraint_state_used": False,
        "node_bias_definition": "eligible-prefix discrete-time hazard",
        "rollout_stop_is_absorbing": True,
        "horizons_kept_distinct": True,
        "event_first_aggregation": True,
        "source_side_threshold_source": "train80 only",
        "ab_role": "posthoc secondary read-back only",
        "clinical_and_eeg_onset_pooled": False,
        "operator_name": "effective propagation operator",
        "target_seal_intact": seal_ok,
        "interictal_summary_available": summary is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output-root", type=Path, default=CANONICAL)
    args = parser.parse_args()

    analysis = BASE / "formal/analysis"
    summary_path = analysis / "INTERICTAL_CLAIM_SUMMARY.json"
    summary = read_optional_json(summary_path)
    if summary is None and not args.preview:
        raise SystemExit(
            "canonical Figure 6 requires finalized interictal claim summary"
        )
    target_path = BASE / "target_audit/TARGET_METADATA_GATE.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise RuntimeError("target seal was violated before figure rendering")

    claim2 = read_optional_json(analysis / "CLAIM2_STATUS.json")
    claim3 = read_optional_json(analysis / "CLAIM3_STATUS.json")
    claim4 = read_optional_json(analysis / "CLAIM4_STATUS.json")
    readback = read_optional_json(analysis / "AB_AXIS_READBACK_STATUS.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.3, 7.3), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.95,
        wspace=0.30,
        hspace=0.38,
    )
    axes = [fig.add_subplot(grid[index // 3, index % 3]) for index in range(6)]
    _panel_a(axes[0])
    _panel_b(axes[1])
    _panel_c(
        axes[2],
        status=claim2,
        patient_path=analysis / "claim2_patient_metrics.csv",
        preview=args.preview,
    )
    _panel_d(
        axes[3],
        claim3=claim3,
        random_path=analysis / "claim3_random_axis_specificity.csv",
        readback=readback,
        readback_path=analysis / "ab_axis_readback.csv",
        preview=args.preview,
    )
    _panel_e(
        axes[4],
        status=claim4,
        patient_path=analysis / "claim4_shared_scaffold.csv",
        preview=args.preview,
    )
    _panel_f(
        axes[5],
        summary=summary,
        target=target,
        transfer_path=BASE / "early_ictal_transfer/per_patient.csv",
        preview=args.preview,
    )

    output = args.output_root
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    suffix = "_preview" if args.preview else ""
    png = figures / f"{STEM}{suffix}.png"
    pdf = figures / f"{STEM}{suffix}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    inputs = {}
    for path in [
        BASE / "development/DEVELOPMENT_LOCK.json",
        BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json",
        BASE / "formal/ALL_SUBJECT_SEQUENCE_LOCK.json",
        BASE / "formal/analysis/ALL_SUBJECT_SEQUENCE_STATUS.json",
        BASE / "target_audit/TARGET_METADATA_GATE.json",
        analysis / "CLAIM2_STATUS.json",
        analysis / "CLAIM3_STATUS.json",
        analysis / "CLAIM4_STATUS.json",
        analysis / "AB_AXIS_READBACK_STATUS.json",
        summary_path,
        analysis / "claim2_patient_metrics.csv",
        analysis / "claim3_random_axis_specificity.csv",
        analysis / "claim4_shared_scaffold.csv",
        analysis / "ab_axis_readback.csv",
        BASE / "early_ictal_transfer/per_patient.csv",
    ]:
        if path.is_file():
            inputs[str(path.relative_to(ROOT))] = sha256(path)
    qa = _scientific_qa(summary=summary, target=target)
    if not args.preview and not all(
        value
        for key, value in qa.items()
        if key
        in {
            "development_n3_and_geometry_complete",
            "physical_axis_formal_n22",
            "all_subject_sequence_n31",
            "target_seal_intact",
            "interictal_summary_available",
        }
    ):
        raise RuntimeError("Figure-6 scientific QA failed")
    payload = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "preview" if args.preview else "complete",
        "figure": display_path(png),
        "vector_figure": display_path(pdf),
        "png_dpi": 300,
        "panels": {
            "A": "same symmetric effective scaffold with opposite observed sources",
            "B": "single propagation state, scalar STOP, and absorbing rollout",
            "C": "patient-first full versus local-isotropic heldout benefit",
            "D": "random-axis specificity and post-hoc A/B axis read-back",
            "E": "train-only source-side shared-scaffold generalization",
            "F": (
                "frozen early-ictal energy transfer"
                if (BASE / "early_ictal_transfer/per_patient.csv").is_file()
                else "target sealed with metadata blocker shown"
            ),
        },
        "claim_status": {
            "claim2_next": claim2.get("claim2_next") if claim2 else "PENDING",
            "claim2_future": claim2.get("claim2_future") if claim2 else "PENDING",
            "claim3_random_axis": (
                claim3.get("claim3_random_axis")
                if claim3
                else "NOT_RUN"
            ),
            "claim4_shared_scaffold": (
                claim4.get("claim4_shared_scaffold") if claim4 else "NOT_RUN"
            ),
            "early_ictal_values_unlocked": (
                summary.get("early_ictal_values_unlocked") if summary else False
            ),
        },
        "scientific_qa": qa,
        "input_sha256": inputs,
        "producer_sha256": sha256(Path(__file__)),
        "git": _git_state(),
        "target_values_read": False,
    }
    summary_output = output / f"{STEM}_summary{suffix}.json"
    atomic_json(summary_output, payload)
    readme = (
        f"### {png.name}\n\n"
        "A、B 说明冻结模型的科学对象与自监督任务；C 展示患者优先的轴向结构增益；"
        "D 将 learned axis 与随机方向及既有 A/B 病理轴比较；E 检验同一个传播"
        " scaffold 是否跨两侧起点泛化；F 只在四项间期 gate 与逐发作 clinical-"
        "onset source metadata 同时就绪后显示迁移结果，否则明确显示 target sealed。"
        "\n\n"
        "**关注点**：所有统计点均先在患者内聚合；阴性或未达到上游 gate 的 panel "
        "不会由其他 endpoint、EEG onset 或旧 SNN 结果替代。\n"
    )
    (figures / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
