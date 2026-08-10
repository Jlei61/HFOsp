#!/usr/bin/env python3
"""Executed-stage diagnostic for the FCXR-LC4e shared-executor screen."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
RESULT = BASE / "lc4e_spatially_shared_terminator"
LOCAL = BASE / "lc4d_offset_latency_alignment"
OUT = RESULT / "figures"

RED = "#b2182b"
BLUE = "#2166ac"
ORANGE = "#d17c2f"
GREEN = "#348a53"
GREY = "#777777"


def _smooth(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bins <= 1:
        return x
    return np.convolve(x, np.ones(bins, dtype=float) / bins, mode="same")


def _core_mean(z, prefix: str) -> np.ndarray:
    return 0.5 * (np.asarray(z[f"snapshot_{prefix}_core_A"], dtype=float)
                  + np.asarray(z[f"snapshot_{prefix}_core_B"], dtype=float))


def main() -> None:
    verdict = json.loads((RESULT / "architecture_verdict.json").read_text())
    shared_record = json.loads((RESULT / "latency_screen.json").read_text())
    local_record = json.loads((LOCAL / "latency_screen.json").read_text())
    shared = np.load(RESULT / "latency_screen_traces.npz")
    local = np.load(LOCAL / "latency_screen_traces.npz")
    OUT.mkdir(parents=True, exist_ok=True)

    dt = float(shared["trace_dt_ms"][0])
    t = np.arange(shared["rate_E"].size, dtype=float) * dt / 1000.0
    ts = np.asarray(shared["snapshot_t_ms"], dtype=float) / 1000.0
    onset_s = float(shared_record["gate"]["onset_ms"]) / 1000.0
    current_s = np.asarray(shared["adap_current"], dtype=float)
    current_l = np.asarray(local["adap_current"], dtype=float)
    shared_core_y = _core_mean(shared, "y")
    local_core_y = _core_mean(local, "y")

    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.35), facecolor="white")

    axes[0].axvspan(0, onset_s, color="#e8f1f7", lw=0)
    axes[0].axvspan(onset_s, 18, color="#f4e4e4", lw=0)
    axes[0].plot(t, _smooth(local["rate_E"], 10), color=ORANGE, lw=1.0,
                 label="cell-local")
    axes[0].plot(t, _smooth(shared["rate_E"], 10), color=BLUE, lw=1.0,
                 alpha=0.9, label="spatially shared")
    axes[0].axvline(onset_s, color=RED, ls="--", lw=0.9)
    axes[0].text(0.04, 0.96, "identical through first current\n29 returning events before onset",
                 transform=axes[0].transAxes, va="top", fontsize=8)
    axes[0].text(0.97, 0.06, "neither offsets by 18 s",
                 transform=axes[0].transAxes, ha="right", fontsize=8)
    axes[0].set(xlim=(0, 18), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="shared execution preserves entry, not offset")
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper right")

    axes[1].plot(t, current_l, color=ORANGE, lw=1.2, label="cell-local")
    axes[1].plot(t, current_s, color=BLUE, lw=1.2, label="spatially shared")
    axes[1].axvline(onset_s, color=RED, ls="--", lw=0.9)
    axes[1].text(0.97, 0.96,
                 f"peak: {current_l.max():.1f} local\n{current_s.max():.1f} shared",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    axes[1].set(xlim=(10.5, 18), xlabel="time (s)", ylabel="executed current",
                title="shared feedback limits its own delivered dose")
    axes[1].legend(frameon=False, fontsize=7.2, loc="upper left")

    axes[2].plot(ts, shared_core_y, color=RED, lw=1.2, label="two-core mean")
    axes[2].plot(ts, shared["snapshot_y_axial"], color=ORANGE, lw=1.2,
                 label="axial band")
    axes[2].plot(ts, shared["snapshot_y_off_axis"], color=BLUE, lw=1.2,
                 label="off-axis")
    axes[2].axvline(onset_s, color=GREY, ls="--", lw=0.8)
    axes[2].set(xlim=(11, 18), xlabel="time (s)", ylabel="regional load $y$",
                title="shared actuation removes core-selective collapse")
    axes[2].legend(frameon=False, fontsize=7.2, loc="upper left")

    regions = ("core A", "core B", "axial", "off-axis")
    local_h = [verdict["spatial"][f"local_final_H_{key}"]
               for key in ("core_A", "core_B", "axial", "off_axis")]
    shared_h = [verdict["spatial"][f"shared_final_H_{key}"]
                for key in ("core_A", "core_B", "axial", "off_axis")]
    x = np.arange(len(regions))
    width = 0.36
    axes[3].bar(x - width / 2, local_h, width, color=ORANGE, label="cell-local")
    axes[3].bar(x + width / 2, shared_h, width, color=BLUE, label="spatially shared")
    axes[3].set_xticks(x, regions, rotation=20)
    axes[3].set(ylabel="regional carrier $H$ at 17.75 s",
                title="off-axis escape is removed; the whole carrier survives")
    axes[3].legend(frameon=False, fontsize=7.2, loc="upper left")

    for letter, ax in zip("ABCD", axes):
        ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "FCXR-LC4e: matched spatial sharing removes local escape but still does not terminate",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4e_shared_executor_screen.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "lc4e_shared_executor_screen",
        "kind": "executed E1 causal architecture diagnostic; not a lifecycle figure",
        "verdict": verdict["verdict"],
        "panels": {
            "A": "archived local and fresh shared rate traces; causal prefix matches and neither offsets",
            "B": "executed local versus shared current after their identical first-current boundary",
            "C": "regional load in the shared arm after onset",
            "D": "local versus shared regional H at the final stored snapshot",
        },
        "key_numbers": {
            "onset_ms": shared_record["gate"]["onset_ms"],
            "offset_ms": shared_record["gate"]["offset_ms"],
            "pre_returning_events": shared_record["gate"]["n_returning_before_onset"],
            "local_peak_current": float(current_l.max()),
            "shared_peak_current": float(current_s.max()),
            "shared_to_local_peak_current": float(current_s.max() / current_l.max()),
            "local_final_core_y": float(local_core_y[-1]),
            "shared_final_core_y": float(shared_core_y[-1]),
            **verdict["spatial"],
        },
        "claim_boundary": (
            "At one development seed, changing only the spatial allocation of the matched cell-load "
            "actuator preserves the exact causal prefix and removes the archived core-suppression/off-axis-"
            "escape pattern, but it still does not produce autonomous offset within the 18 s screen. "
            "This rejects spatial allocation as the sole blocker; it does not reject X-mediated or "
            "recruited-area termination."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4e_shared_executor_screen_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4e_shared_executor_screen.png")


if __name__ == "__main__":
    main()
