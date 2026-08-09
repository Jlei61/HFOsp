#!/usr/bin/env python3
"""Executed-stage diagnostic for FCXR-LC4c.

The figure is intentionally not a complete-lifecycle claim.  It shows the passing
15 s entry gate and the only 70 s nominal trajectory.  The nominal trajectory
autonomously offsets, but only after a 55 s high-density bout and leaves four
post-offset seconds, so the gated frozen-D confirmation is not drawn.
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
RESULT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
          / "lc4c_entry_offset_alignment")
OUT = RESULT / "figures"

BLUE = "#2166ac"
RED = "#b2182b"
ORANGE = "#d17c2f"
GREEN = "#348a53"
PURPLE = "#6f4c8b"
GREY = "#777777"


def _smooth(x: np.ndarray, bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if bins <= 1:
        return x
    return np.convolve(x, np.ones(bins, dtype=float) / bins, mode="same")


def _time(z, key="rate_E", dt_key="rate_dt_ms"):
    return np.arange(z[key].size, dtype=float) * float(z[dt_key][0]) / 1000.0


def _colored_path(ax, x, y, t):
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    seg = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", linewidth=2.0)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    return lc


def _event_stats(events, lo_s, hi_s):
    rows = [e for e in events
            if bool(e.get("returned", False))
            and lo_s * 1000.0 <= float(e["t_on_ms"]) < hi_s * 1000.0]
    return {
        "n": len(rows),
        "rate_hz": len(rows) / (hi_s - lo_s),
        "duration_median_ms": (float(np.median([e["dur_ms"] for e in rows]))
                               if rows else None),
        "participation_median": (float(np.median([e["peak_ext"] for e in rows]))
                                 if rows else None),
    }


def main() -> None:
    entry = json.loads((RESULT / "entry_gate.json").read_text())
    nominal = json.loads((RESULT / "nominal_lifecycle.json").read_text())
    ze = np.load(RESULT / "entry_gate_traces.npz")
    zn = np.load(RESULT / "nominal_lifecycle_traces.npz")
    OUT.mkdir(parents=True, exist_ok=True)

    gate = nominal["nominal_gate"]
    onset_s = float(gate["onset_ms"]) / 1000.0
    offset_s = float(gate["offset_ms"]) / 1000.0
    target = float(nominal["candidate"]["matched_ictal_current"])
    events = nominal["event_ledger"]["events"]
    post = _event_stats(events, offset_s, 70.0)
    final8 = _event_stats(events, 62.0, 70.0)

    fig, axes = plt.subplots(1, 4, figsize=(18.1, 4.45), facecolor="white")

    # A: the preregistered entry repair worked on a fresh trajectory.
    te = _time(ze)
    axes[0].plot(te, _smooth(ze["rate_E"], 10), color=RED, lw=1.0)
    axes[0].axvspan(0, 4, color="#e8f1f7", lw=0)
    axes[0].axvline(float(entry["gate"]["onset_ms"]) / 1000.0,
                    color=RED, ls="--", lw=1.0)
    axes[0].text(0.04, 0.96, "executed current = 0\n(first 4 s)",
                 transform=axes[0].transAxes, va="top", fontsize=8, color=BLUE)
    axes[0].text(0.97, 0.96,
                 f"onset 11 s\n{entry['gate']['n_returning_before_onset']} pre-events",
                 transform=axes[0].transAxes, va="top", ha="right", fontsize=8)
    axes[0].set(xlim=(0, 15), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="entry remains spontaneous")

    # B: the full nominal path does offset, but far too late.
    tn = _time(zn)
    axes[1].axvspan(0, onset_s, color="#e8f1f7", lw=0, alpha=0.8)
    axes[1].axvspan(onset_s, offset_s, color="#f4e4e4", lw=0, alpha=0.9)
    axes[1].axvspan(offset_s, 70, color="#eeeeee", lw=0, alpha=0.9)
    axes[1].plot(tn, _smooth(zn["rate_E"], 10), color=RED, lw=0.9, alpha=0.88)
    axes[1].axvline(onset_s, color=RED, ls="--", lw=1.0)
    axes[1].axvline(offset_s, color="black", ls="--", lw=1.0)
    axes[1].text(onset_s + 0.7, 0.95, "onset", color=RED,
                 transform=axes[1].get_xaxis_transform(), va="top", fontsize=8)
    axes[1].text(offset_s - 0.7, 0.95, "offset", color="black", ha="right",
                 transform=axes[1].get_xaxis_transform(), va="top", fontsize=8)
    axes[1].text(0.04, 0.05, "55 s high bout\n(gate: 1–5 s)",
                 transform=axes[1].transAxes, fontsize=8, va="bottom")
    axes[1].set(xlim=(0, 70), xlabel="time (s)", ylabel="E-cell rate (Hz)",
                title="offset exists, but is too late")

    # C: executor and slow coordinates reveal delayed accumulation and a short tail.
    current_t = _time(zn, "adap_current", "trace_dt_ms")
    current = np.asarray(zn["adap_current"], float) / target
    ts = np.asarray(zn["snapshot_t_ms"], float) / 1000.0
    axes[2].plot(current_t, _smooth(current, 20), color=ORANGE, lw=1.3,
                 label=r"terminator current / $I_{target}$")
    axes[2].plot(ts, np.asarray(zn["snapshot_D_all"], float), color=PURPLE, lw=1.1,
                 label=r"disinhibition $D$")
    axes[2].plot(ts, 1.0 - np.asarray(zn["snapshot_X_all"], float),
                 color=GREEN, lw=1.1, label=r"relay depletion $1-X$")
    axes[2].axvline(onset_s, color=RED, ls="--", lw=0.9)
    axes[2].axvline(offset_s, color="black", ls="--", lw=0.9)
    axes[2].axhline(1.0, color=ORANGE, ls=":", lw=0.8)
    axes[2].set(xlim=(0, 70), ylim=(0, 1.02), xlabel="time (s)",
                ylabel="slow coordinate / target",
                title="termination load accumulates too slowly")
    axes[2].legend(frameon=False, fontsize=7.0, loc="upper right")

    # D: the path turns back, but the record ends before statistical return can be tested.
    d = np.asarray(zn["snapshot_D_all"], float)
    i_norm = np.interp(ts, current_t, current)
    lc = _colored_path(axes[3], d, i_norm, ts)
    idx_on = int(np.argmin(np.abs(ts - onset_s)))
    idx_off = int(np.argmin(np.abs(ts - offset_s)))
    axes[3].scatter(d[0], i_norm[0], s=36, color=BLUE, edgecolor="white", zorder=3,
                    label="start")
    axes[3].scatter(d[idx_on], i_norm[idx_on], s=45, color=RED, marker="^",
                    edgecolor="white", zorder=3, label="onset")
    axes[3].scatter(d[idx_off], i_norm[idx_off], s=45, color="black", marker="v",
                    edgecolor="white", zorder=3, label="offset")
    axes[3].scatter(d[-1], i_norm[-1], s=45, color=GREY, marker="X",
                    edgecolor="white", zorder=3, label="70 s end")
    axes[3].set(xlabel=r"mean disinhibition $D=1-z$",
                ylabel=r"terminator current / $I_{target}$",
                title="return path is only partially observed")
    axes[3].text(0.97, 0.05,
                 f"post-offset observed: 4 s (gate: 8 s)\n"
                 f"post-offset returning events: {post['n']}",
                 transform=axes[3].transAxes, ha="right", va="bottom", fontsize=8)
    axes[3].legend(frameon=False, fontsize=7.0, loc="upper left")
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
        "FCXR-LC4c: cumulative entry and autonomous offset occur, "
        "but not a 1–5 s lifecycle with verified return",
        fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"lc4c_entry_offset_alignment_diagnostic.{ext}", dpi=220,
                    bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "lc4c_entry_offset_alignment_diagnostic",
        "kind": "executed-stage diagnostic; not a complete-lifecycle or paper-claim figure",
        "stage_verdicts": {
            "C1": entry["gate"]["verdict"],
            "C2_nominal": gate["verdict"],
            "C2_exact_final_D": "not_authorised",
        },
        "panels": {
            "A": "fresh 15 s no-kick entry gate; the first 4 s have exactly zero executor current",
            "B": "70 s nominal rate trajectory with 11 s onset and 66 s autonomous offset",
            "C": "executed terminator current, mean D and relay depletion through the nominal path",
            "D": "mean D versus normalized terminator current; only four post-offset seconds are observed",
        },
        "key_numbers": {
            "onset_ms": gate["onset_ms"],
            "offset_ms": gate["offset_ms"],
            "bout_ms": gate["bout_ms"],
            "pre_returning_events": gate["n_returning_before_onset"],
            "postictal_rate_hz": gate["postictal_rate_hz"],
            "post_offset_observation_ms": 70000.0 - gate["offset_ms"],
            "post_offset_returning_events": post["n"],
            "final8": final8,
            "final_D_mean": nominal["final_D_mean"],
            "final_X_mean": nominal["final_X_mean"],
            "max_current_fraction_of_target": float(np.max(current)),
        },
        "claim_boundary": (
            "At one development seed, C1 proves aligned no-kick cumulative entry.  C2 shows an "
            "autonomous offset and four seconds of post-offset suppression, but the high bout "
            "lasts 55 seconds rather than 1–5 seconds and no returning event occurs after offset. "
            "Distributional recovery and the exact-D stability confirmation remain untested."
        ),
        "source": str(RESULT.relative_to(ROOT)),
    }
    (OUT / "lc4c_entry_offset_alignment_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(OUT / "lc4c_entry_offset_alignment_diagnostic.png")


if __name__ == "__main__":
    main()
