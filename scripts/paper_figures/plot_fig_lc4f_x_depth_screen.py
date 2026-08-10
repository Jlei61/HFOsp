#!/usr/bin/env python3
"""Executed X1 diagnostic for FCXR-LC4f; never presents the run as a lifecycle."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
          / "lc4f_x_depth_closure")
OUT = RESULT / "figures"

RED = "#b2182b"
BLUE = "#2166ac"
ORANGE = "#d17c2f"
GREEN = "#348a53"
PURPLE = "#6f4c8b"
GREY = "#777777"


def _smooth(x, bins):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(bins) / bins, mode="same") if bins > 1 else x


def _time_line(ax, x, y, t):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.0)
    lc.set_array(np.asarray(t[:-1], dtype=float))
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def main():
    record = json.loads((RESULT / "x_depth_screen.json").read_text())
    with np.load(RESULT / "x_depth_screen_traces.npz") as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    OUT.mkdir(parents=True, exist_ok=True)

    dt = float(data["rate_dt_ms"][0])
    t = np.arange(data["rate_E"].size) * dt / 1000.0
    ts = data["snapshot_t_ms"].astype(float) / 1000.0
    onset = float(record["gate"]["onset_ms"]) / 1000.0
    boundary = 0.38

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.4), facecolor="white")

    # A — the accepted entry survives, but there is no autonomous offset.
    axes[0].axvspan(0, onset, color="#e8f1f7", lw=0)
    axes[0].axvspan(onset, 22, color="#f4e4e4", lw=0)
    axes[0].plot(t, _smooth(data["rate_E"], 8), color=RED, lw=1.0)
    axes[0].axvline(onset, color=RED, ls="--", lw=0.9)
    axes[0].text(0.04, 0.96, "29 returning IEDs\nbefore no-kick onset",
                 transform=axes[0].transAxes, va="top", fontsize=8)
    axes[0].text(0.97, 0.06, "high state continues\nto the 22 s boundary",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].set(xlim=(0, 22), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="entry is preserved; offset is absent")

    # B — the registered K=3 candidate does not reach the archived late-bout reference in this window.
    tx = np.arange(data["x_mean"].size) * dt / 1000.0
    axes[1].plot(tx, data["x_mean"], color=BLUE, lw=1.3,
                 label="population-mean X")
    axes[1].axhline(boundary, color=RED, ls="--", lw=1.0,
                    label="archived late-bout reference")
    i_min = int(np.argmin(data["x_mean"]))
    axes[1].scatter([tx[i_min]], [data["x_mean"][i_min]], s=38, color=BLUE,
                    edgecolor="white", zorder=3)
    axes[1].text(tx[i_min] - 0.25, data["x_mean"][i_min] + 0.035,
                 f"minimum = {data['x_mean'][i_min]:.3f}", fontsize=8, ha="right")
    axes[1].axvline(onset, color=GREY, ls=":", lw=0.8)
    axes[1].set(xlim=(9.5, 22), ylim=(0.30, 1.03), xlabel="time (s)",
                ylabel="relay availability X", title="archived X reference not reached\nwithin target window")
    axes[1].legend(frameon=False, fontsize=7.2, loc="upper right")

    # C — the slow trajectory goes into the high-state region and does not turn back.
    lc = _time_line(axes[2], data["snapshot_D_all"], data["snapshot_H_all"], ts)
    onset_i = int(np.argmin(np.abs(ts - onset)))
    axes[2].scatter(data["snapshot_D_all"][onset_i], data["snapshot_H_all"][onset_i],
                    s=42, color=RED, edgecolor="white", zorder=3, label="onset")
    axes[2].scatter(data["snapshot_D_all"][-1], data["snapshot_H_all"][-1],
                    s=42, color="black", edgecolor="white", zorder=3, label="record end")
    axes[2].set(xlabel=r"disinhibition $D=1-z$", ylabel="carrier H",
                title="slow path enters but does not close")
    axes[2].legend(frameon=False, fontsize=7.2, loc="upper left")
    cb = fig.colorbar(lc, ax=axes[2], fraction=0.045, pad=0.02)
    cb.set_label("time (s)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # D — regional means show a spatial mismatch, not a universal population threshold.
    regions = ("core A", "core B", "axial", "off-axis", "all")
    keys = ("core_A", "core_B", "axial", "off_axis", "all")
    vals = [float(data[f"snapshot_X_{k}"][-1]) for k in keys]
    colors = [RED, RED, ORANGE, BLUE, PURPLE]
    axes[3].bar(np.arange(len(vals)), vals, color=colors, alpha=0.9)
    axes[3].axhline(boundary, color=RED, ls="--", lw=1.0)
    axes[3].set_xticks(np.arange(len(vals)), regions, rotation=22)
    axes[3].set(ylim=(0.30, 0.56), ylabel="regional X at 21.75 s",
                title="X field and carrier support\nare spatially mismatched")
    for i, v in enumerate(vals):
        axes[3].text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=7.5)

    for letter, ax in zip("ABCD", axes):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "FCXR-LC4f: natural entry persists beyond the preregistered offset window",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4f_x_depth_screen.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "lc4f_x_depth_screen",
        "kind": "executed X1 diagnostic; explicitly not a lifecycle figure",
        "verdict": record["gate"]["verdict"],
        "panels": {
            "A": "fresh no-kick rate trace with cumulative entry and no offset",
            "B": "population mean X against the archived 0.380 late-bout reference; not a universal boundary",
            "C": "population-mean D-H slow trajectory, onset marker and record-end marker",
            "D": "regional X at the final stored snapshot, showing mismatch with the surviving carrier support",
        },
        "key_numbers": record["trace_summary"],
        "claim_boundary": (
            "One development-seed natural-entry screen: the empirical K_y=3 loop did not reach the archived "
            "late-bout X reference and did not autonomously offset within the preregistered 1--5 s target "
            "window. This does not establish an asymptotic X floor, a universal X threshold, or population-wide "
            "coverage as the unique failure mechanism."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4f_x_depth_screen_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4f_x_depth_screen.png")


if __name__ == "__main__":
    main()
