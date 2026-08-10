#!/usr/bin/env python3
"""Executed-stage diagnostic for the FCXR-LC4d latency screen.

Only L1 was authorised and executed.  The figure therefore does not draw a
nominal lifecycle or a frozen-D confirmation.  It shows why the one-point
open-loop dose transfer failed in the actual closed spatial network.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
          / "lc4d_offset_latency_alignment")
OUT = RESULT / "figures"

RED = "#b2182b"
BLUE = "#2166ac"
ORANGE = "#d17c2f"
GREEN = "#348a53"
PURPLE = "#6f4c8b"
GREY = "#777777"


def _smooth(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bins <= 1:
        return x
    return np.convolve(x, np.ones(bins, dtype=float) / bins, mode="same")


def main() -> None:
    record = json.loads((RESULT / "latency_screen.json").read_text())
    lock = json.loads((RESULT / "candidate_lock.json").read_text())
    z = np.load(RESULT / "latency_screen_traces.npz")
    OUT.mkdir(parents=True, exist_ok=True)

    dt = float(z["trace_dt_ms"][0])
    t = np.arange(z["rate_E"].size, dtype=float) * dt / 1000.0
    ts = np.asarray(z["snapshot_t_ms"], dtype=float) / 1000.0
    target = float(lock["candidate"]["matched_ictal_current"])
    current = np.asarray(z["adap_current"], dtype=float)
    onset_s = float(record["gate"]["onset_ms"]) / 1000.0
    align_s = float(lock["candidate"]["calibration"]["align_time_ms"]) / 1000.0
    align_i = int(round(align_s * 1000.0 / dt))
    reached = np.flatnonzero(current >= target)
    first_reach_s = (float(t[reached[0]]) if reached.size else None)

    core_y = 0.5 * (np.asarray(z["snapshot_y_core_A"], float)
                    + np.asarray(z["snapshot_y_core_B"], float))
    core_h = 0.5 * (np.asarray(z["snapshot_H_core_A"], float)
                    + np.asarray(z["snapshot_H_core_B"], float))

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.35), facecolor="white")

    # A: fresh entry remains intact, but the bout reaches the record boundary.
    axes[0].axvspan(0, onset_s, color="#e8f1f7", lw=0)
    axes[0].axvspan(onset_s, 18.0, color="#f4e4e4", lw=0)
    axes[0].plot(t, _smooth(z["rate_E"], 10), color=RED, lw=1.0)
    axes[0].axvline(onset_s, color=RED, ls="--", lw=0.9)
    axes[0].text(0.04, 0.96, "29 returning events\nbefore spontaneous onset",
                 transform=axes[0].transAxes, va="top", fontsize=8)
    axes[0].text(0.97, 0.06, "no offset by 18 s",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].set(xlim=(0, 18), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="entry survives; offset does not")

    # B: the open-loop one-point transfer is attenuated by its own feedback.
    axes[1].plot(t, current / target, color=ORANGE, lw=1.3,
                 label=r"executed current / $I_{target}$")
    axes[1].axhline(1.0, color=GREY, ls=":", lw=0.9)
    axes[1].axvline(onset_s, color=RED, ls="--", lw=0.9)
    axes[1].axvline(align_s, color=BLUE, ls="--", lw=0.9)
    axes[1].scatter([align_s], [current[align_i] / target], s=32, color=BLUE,
                    edgecolor="white", zorder=3)
    axes[1].text(align_s + 0.2, current[align_i] / target,
                 f"actual {current[align_i] / target:.2f}×", fontsize=8, va="center")
    if first_reach_s is not None:
        axes[1].text(0.97, 0.06, f"first reaches target: {first_reach_s:.2f} s",
                     transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
    axes[1].set(xlim=(10.5, 18), ylim=(0, 1.18), xlabel="time (s)",
                ylabel=r"terminator current / $I_{target}$",
                title="open-loop dose transfer fails in closed loop")

    # C: the sustained load moves from the two cores to off-axis tissue.
    axes[2].plot(ts, core_y, color=RED, lw=1.2, label="two-core mean")
    axes[2].plot(ts, z["snapshot_y_axial"], color=ORANGE, lw=1.2, label="axial band")
    axes[2].plot(ts, z["snapshot_y_off_axis"], color=BLUE, lw=1.2, label="off-axis")
    axes[2].axvline(onset_s, color=GREY, ls="--", lw=0.8)
    axes[2].set(xlim=(11, 18), xlabel="time (s)", ylabel="regional load $y$",
                title="regional load redistributes")
    axes[2].legend(frameon=False, fontsize=7.2, loc="upper right")

    # D: H confirms that the remaining carrier is outside the original core/axis.
    axes[3].plot(ts, core_h, color=RED, lw=1.2, label="two-core mean")
    axes[3].plot(ts, z["snapshot_H_axial"], color=ORANGE, lw=1.2, label="axial band")
    axes[3].plot(ts, z["snapshot_H_off_axis"], color=BLUE, lw=1.2, label="off-axis")
    axes[3].axvline(onset_s, color=GREY, ls="--", lw=0.8)
    axes[3].text(0.97, 0.96, "17.75 s: core H≈0.085\noff-axis H≈1.12",
                 transform=axes[3].transAxes, ha="right", va="top", fontsize=8)
    axes[3].set(xlim=(11, 18), xlabel="time (s)", ylabel="regional carrier $H$",
                title="local suppression leaves off-axis carrier")
    axes[3].legend(frameon=False, fontsize=7.2, loc="upper right", bbox_to_anchor=(1, 0.77))

    for letter, ax in zip("ABCD", axes):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "FCXR-LC4d: stronger cell-local termination suppresses the cores "
        "but does not terminate the spatial carrier",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4d_offset_latency_screen.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "lc4d_offset_latency_screen",
        "kind": "executed L1 diagnostic; not a lifecycle or paper-claim figure",
        "stage_verdicts": {
            "L0": lock["verdict"],
            "L1": record["gate"]["verdict"],
            "L2_nominal": "not_authorised",
            "L2_exact_D": "not_authorised",
        },
        "panels": {
            "A": "fresh no-intervention rate trace; onset at 11 s and no offset by record end",
            "B": "executed current relative to the locked target; actual current at 15 s is only 0.411 of target",
            "C": "regional relay/load coordinate y after onset",
            "D": "regional H carrier showing late off-axis persistence after core suppression",
        },
        "key_numbers": {
            "onset_ms": record["gate"]["onset_ms"],
            "offset_ms": record["gate"]["offset_ms"],
            "bout_ms_lower_bound": record["gate"]["bout_ms"],
            "pre_returning_events": record["gate"]["n_returning_before_onset"],
            "target_current": target,
            "actual_current_at_15s": float(current[align_i]),
            "actual_fraction_at_15s": float(current[align_i] / target),
            "first_target_reach_ms": (None if first_reach_s is None else first_reach_s * 1000.0),
            "max_current": record["max_adap_current"],
            "core_y_17p75s": float(core_y[-1]),
            "off_axis_y_17p75s": float(z["snapshot_y_off_axis"][-1]),
            "core_H_17p75s": float(core_h[-1]),
            "off_axis_H_17p75s": float(z["snapshot_H_off_axis"][-1]),
        },
        "claim_boundary": (
            "At one development seed, the exact-dead-zone mechanism preserves cumulative no-kick entry "
            "and remains numerically safe, but a 4.85x cell-local gain increase does not produce an offset "
            "within the 18 s screen.  The regional trace is consistent with core suppression accompanied by "
            "persistent off-axis load/carrier.  It does not by itself prove a causal migration mechanism or "
            "rule out a spatially coordinated termination pathway."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4d_offset_latency_screen_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4d_offset_latency_screen.png")


if __name__ == "__main__":
    main()
