#!/usr/bin/env python3
"""Diagnostic closeout for the executed FCXR-LC4b exact-dead-zone chain.

This is deliberately not a paper-claim lifecycle figure.  It shows the three stages that
actually ran: exact baseline identity, the retained frozen-D entry surface, and the 70 s
continuous no-kick trajectory that entered but did not autonomously terminate.  The gated
exact-D continuation is not drawn.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
RESULT = BASE / "lc4b_deadzone_lifecycle"
LC4 = BASE / "lc4_lifecycle_gate"
OUT = RESULT / "figures"

GREY = "#777777"
BLUE = "#2166ac"
RED = "#b2182b"
ORANGE = "#d17c2f"
GREEN = "#348a53"
PURPLE = "#6f4c8b"


def _smooth(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bins <= 1:
        return x
    return np.convolve(x, np.ones(bins, dtype=float) / bins, mode="same")


def _load():
    baseline = json.loads((RESULT / "baseline_verdict.json").read_text())
    onset = json.loads((RESULT / "onset_surface_verdict.json").read_text())
    nominal = json.loads((RESULT / "nominal_lifecycle.json").read_text())
    traces = {
        "control": np.load(LC4 / "runs/baseline_control_traces.npz"),
        "baseline": np.load(RESULT / "runs/baseline_deadzone_traces.npz"),
        "d10": np.load(RESULT / "runs/onset_deadzone_D10_traces.npz"),
        "nominal": np.load(RESULT / "nominal_lifecycle_traces.npz"),
    }
    return baseline, onset, nominal, traces


def _time(z, key="rate_E", dt_key="rate_dt_ms"):
    return np.arange(z[key].size, dtype=float) * float(z[dt_key][0]) / 1000.0


def _colored_path(ax, x, y, t, *, cmap="viridis"):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=2.0)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def main() -> None:
    baseline, onset, nominal, z = _load()
    OUT.mkdir(parents=True, exist_ok=True)
    target = float(nominal["candidate"]["matched_ictal_current"])
    onset_ms = float(nominal["nominal_gate"]["onset_ms"])

    fig, axes = plt.subplots(1, 4, figsize=(17.8, 4.35), facecolor="white")

    # A — exact dead zone really is invisible at baseline.
    for key, label, colour, width in (
        ("control", "actuator off", GREY, 2.0),
        ("baseline", "exact dead zone", BLUE, 1.0),
    ):
        t = _time(z[key])
        axes[0].plot(t, _smooth(z[key]["rate_E"], 10), color=colour, lw=width,
                     label=label, alpha=0.92)
    axes[0].axvspan(0, 2, color="0.93", lw=0)
    axes[0].text(0.05, 0.96, "byte-identical rate/AF\nexecuted current = 0",
                 transform=axes[0].transAxes, va="top", ha="left", fontsize=8)
    axes[0].set(xlim=(0, 12), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="baseline is exactly preserved")
    axes[0].legend(frameon=False, fontsize=7.7, loc="upper right")

    # B — D/Z entry surface remains reachable under the mechanism.
    for key, label, colour in (
        ("baseline", r"healthy $D$", BLUE),
        ("d10", r"fixed $D_{10}$", RED),
    ):
        t = _time(z[key])
        axes[1].plot(t, _smooth(z[key]["rate_E"], 10), color=colour, lw=1.25,
                     label=label)
    d10_row = next(r for r in onset["rows"]
                   if r.get("role") == "candidate" and r.get("d_label") == "D10")
    departure_s = float(d10_row["lifecycle"]["bout"][0])
    axes[1].axvline(departure_s, color=RED, ls="--", lw=1.0)
    axes[1].text(departure_s + 0.15, 0.94, "departure", color=RED,
                 transform=axes[1].get_xaxis_transform(), va="top", fontsize=8)
    axes[1].set(xlim=(0, 12), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="frozen-D entry surface is retained")
    axes[1].legend(frameon=False, fontsize=7.7, loc="upper left")

    # C — one continuous no-kick trajectory: onset occurs, but the high-density train persists.
    zn = z["nominal"]
    t = _time(zn)
    rate = _smooth(zn["rate_E"], 10)
    axes[2].plot(t, rate, color=RED, lw=0.9, alpha=0.85, label="E-cell rate")
    axes[2].axvline(onset_ms / 1000.0, color=RED, ls="--", lw=1.0)
    axes[2].axvspan(62, 70, color="#f4e8e8", lw=0, alpha=0.75)
    axes[2].set(xlim=(0, 70), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="no-kick entry, no autonomous offset")
    ax2 = axes[2].twinx()
    ts = np.asarray(zn["snapshot_t_ms"], float) / 1000.0
    current_t = _time(zn, "adap_current", "trace_dt_ms")
    current = np.asarray(zn["adap_current"], float) / target
    ax2.plot(current_t, _smooth(current, 20), color=ORANGE, lw=1.2,
             label=r"dead-zone current / $I_{target}$")
    ax2.plot(ts, np.asarray(zn["snapshot_D_all"], float), color=PURPLE, lw=1.05,
             label=r"disinhibition $D$")
    ax2.plot(ts, 1.0 - np.asarray(zn["snapshot_X_all"], float), color=GREEN, lw=1.05,
             label=r"relay depletion $1-X$")
    ax2.axhline(1.0, color=ORANGE, ls=":", lw=0.8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("slow coordinate / target", fontsize=8)
    lines = axes[2].lines[:1] + ax2.lines[:3]
    ax2.legend(lines, [q.get_label() for q in lines], frameon=False, fontsize=6.8,
               loc="upper right")
    axes[2].text(66, 0.025, "fixed return window", transform=axes[2].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.2, color="0.35")

    # D — the slow path is open, not a completed loop.
    d = np.asarray(zn["snapshot_D_all"], float)
    i_norm = np.interp(ts, current_t, current)
    lc = _colored_path(axes[3], d, i_norm, ts)
    onset_i = int(np.argmin(np.abs(ts - onset_ms / 1000.0)))
    axes[3].scatter(d[0], i_norm[0], s=36, color=BLUE, edgecolor="white", zorder=3,
                    label="start")
    axes[3].scatter(d[onset_i], i_norm[onset_i], s=45, color=RED, marker="^",
                    edgecolor="white", zorder=3, label="onset")
    axes[3].scatter(d[-1], i_norm[-1], s=45, color="black", marker="X", zorder=3,
                    label="70 s end")
    axes[3].axhline(1.0, color=ORANGE, ls="--", lw=0.9, label=r"matched $I_{target}$")
    axes[3].set(xlabel=r"mean disinhibition $D=1-z$",
                ylabel=r"dead-zone current / $I_{target}$",
                title="slow path does not close")
    axes[3].legend(frameon=False, fontsize=7.1, loc="upper left")
    cbar = fig.colorbar(lc, ax=axes[3], fraction=0.046, pad=0.03)
    cbar.set_label("time (s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for letter, ax in zip("ABCD", axes):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "FCXR-LC4b: an exact dead zone preserves baseline and D/Z entry, "
        "but the closed trajectory does not terminate",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4b_deadzone_lifecycle_diagnostic.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    max_current = float(np.max(zn["adap_current"]))
    metadata = {
        "figure": "lc4b_deadzone_lifecycle_diagnostic",
        "kind": "executed-stage diagnostic; not a complete-lifecycle or paper-claim figure",
        "stage_verdicts": {
            "D1": baseline["verdict"],
            "D2": onset["gate"]["verdict"],
            "D3": nominal["nominal_gate"]["verdict"],
        },
        "panels": {
            "A": "paired control/candidate population rates; traces and active fraction are byte-identical and current is exactly zero",
            "B": "candidate at healthy D and the first departing frozen D10 field; dashed line is whole-record departure",
            "C": "70 s no-kick trajectory with rate, mean D, relay depletion and executed dead-zone current normalized by its matched target",
            "D": "mean D versus normalized executed current, coloured by time; the endpoint remains on the non-returned path",
        },
        "key_numbers": {
            "nominal_onset_ms": onset_ms,
            "nominal_offset_ms": nominal["nominal_gate"]["offset_ms"],
            "nominal_bout_ms": nominal["nominal_gate"]["bout_ms"],
            "max_executed_current": max_current,
            "matched_current_target": target,
            "max_current_fraction_of_target": max_current / target,
            "final_return_window_event_rate_hz": nominal["nominal_gate"]["return_window"]["reference"]["event_rate_hz"],
            "final_D_mean": nominal["final_D_mean"],
            "final_X_mean": nominal["final_X_mean"],
        },
        "claim_boundary": (
            "The exact dead zone is baseline-inert and the D10 onset surface remains reachable. "
            "The single 70 s trajectory entered after 5 s but did not autonomously offset; its "
            "final 8 s were a 9 Hz dense self-terminating event train, not the frozen returning-IED "
            "distribution.  The gated exact-D continuation was therefore not run."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4b_deadzone_lifecycle_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4b_deadzone_lifecycle_diagnostic.png")


if __name__ == "__main__":
    main()
