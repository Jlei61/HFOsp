#!/usr/bin/env python3
"""Fig5 variant: what an adaptation timescale does to a released Z/M episode.

Same visual grammar as the accepted early-bridge figure — a virtual-SEEG readout
beside a path through the slow-variable plane, over a row of panels underneath —
but the time axis runs over the whole episode rather than the approach to onset.

Scientific boundary, stated on the canvas: the elevated segment is a burst train
that spends about three quarters of its time near silence, so it fails the
locked deep-gap clause by a wide margin.  This figure shows entry into and exit
from an elevated state.  It does not show a qualified ictal carrier, and the
control that would attribute the exit to adaptation has not landed yet.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint/seed1"
OUT = ROOT / "results/paper-ready-figure/fig5_adaptation_timescale_episode"

SHAFT_COLOR = "#E08A2E"          # axial shaft, locked convention
CROSS_COLOR = "#2AA5B5"          # transverse shaft, locked convention
ONSET_COLOR = "crimson"
EPISODE_SHADE = "#C9D6E8"
QUIET_HZ = 10.0                  # a bin below this counts as near-silent
GAP_GATE = 0.20                  # locked deep-gap clause
BAND = (30.0, 80.0)


def _runs():
    """Released-Z/M arms keyed by (gain, timescale in seconds)."""
    found = {}
    for root in sorted(RUNS.glob("*pg0.32e60*")):
        if "freeze" in root.name or not (root / "summary.json").is_file():
            continue
        summary = json.loads((root / "summary.json").read_text())
        policy = summary.get("freeze_policy") or {}
        if policy.get("arm") != "dynamic_replay":
            continue                                  # slow variables must be free
        mech = summary["mechanism"]
        if mech.get("subtractive_pool", {}).get("beta_SG", 0.0):
            continue                                  # no subtractive term in this panel
        if int(mech["pv_som_inhibitory_subtypes"]["seed"]) != 1:
            continue                                  # one wiring, like for like
        if float(summary["T_ms"]) != 30000.0:
            continue
        flow = mech.get("dynamic_slow_flow", {})
        key = (float(flow.get("g_M", 1.0)), float(flow.get("tau_M_ms", 500.0)) / 1000.0)
        found[key] = root
    return found


def _load(root):
    with np.load(root / "traces.npz", allow_pickle=False) as data:
        out = {k: np.asarray(data[k], float) for k in (
            "lfp_raw_synaptic_proxy", "fine_core_rate_hz", "fine_time_ms",
            "trace_z_core_mean", "trace_m_core_mean", "coarse_kymo_axial",
        )}
        out["lfp_fs_hz"] = float(data["lfp_fs_hz"])
    return out


def _bandpass(x, fs):
    b, a = butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def _episode_window(rate, bin_ms=2.0, block_ms=2000.0, floor_hz=30.0):
    """First and last block whose mean rate clears an elevated floor."""
    width = int(block_ms / bin_ms)
    blocks = rate[: rate.size // width * width].reshape(-1, width).mean(1)
    hot = np.flatnonzero(blocks > floor_hz)
    if not hot.size:
        return None
    return hot[0] * block_ms / 1000.0, (hot[-1] + 1) * block_ms / 1000.0


def _quiet_fraction(rate, lo_s, hi_s, bin_ms=2.0):
    seg = rate[int(lo_s * 1000 / bin_ms):int(hi_s * 1000 / bin_ms)]
    return float((seg < QUIET_HZ).mean()) if seg.size else float("nan")


def _first_below(rate, level_hz, bin_ms=2.0, block_ms=2000.0):
    width = int(block_ms / bin_ms)
    blocks = rate[: rate.size // width * width].reshape(-1, width).mean(1)
    hot = np.flatnonzero(blocks > 30.0)
    if not hot.size:
        return None
    after = np.flatnonzero(blocks[hot[0]:] < level_hz)
    return None if not after.size else (hot[0] + after[0] + 1) * block_ms / 1000.0


def main():
    runs = _runs()
    lead_key = (3.0, 25.0)
    if lead_key not in runs:
        raise RuntimeError(f"lead arm {lead_key} not found among {sorted(runs)}")
    lead = _load(runs[lead_key])
    t = lead["fine_time_ms"] / 1000.0
    rate = lead["fine_core_rate_hz"]
    window = _episode_window(rate)
    quiet = _quiet_fraction(rate, *window)

    fig = plt.figure(figsize=(14.6, 8.6), facecolor="white")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95],
                           width_ratios=[1.85, 1.0], hspace=0.34, wspace=0.20,
                           left=0.055, right=0.975, top=0.90, bottom=0.075)

    # (a) virtual-SEEG readout over the whole episode
    ax = fig.add_subplot(gs[0, 0])
    lfp = _bandpass(lead["lfp_raw_synaptic_proxy"], lead["lfp_fs_hz"])
    tl = np.arange(lfp.shape[0]) / lead["lfp_fs_hz"]
    step = np.percentile(np.abs(lfp), 99.5) * 2.2
    n = lfp.shape[1]
    ax.axvspan(window[0], window[1], color=EPISODE_SHADE, alpha=0.45, lw=0, zorder=0)
    for i in range(n):
        colour = SHAFT_COLOR if i < n - 4 else CROSS_COLOR
        ax.plot(tl, lfp[:, i] + (n - 1 - i) * step, lw=0.35, color=colour, zorder=2)
    ax.axvline(window[1], color=ONSET_COLOR, lw=1.6, ls="--", zorder=6)
    ax.set(xlim=(0, t[-1]), xlabel="Simulation time (s)",
           ylabel=f"Virtual-SEEG ({BAND[0]:.0f}–{BAND[1]:.0f} Hz)")
    ax.set_yticks([])
    # Events do not stop at the boundary; they thin out and come back sparse,
    # so the line marks the end of the dense segment and nothing stronger.
    after = rate[int(window[1] * 500):]
    ax.legend(handles=[
        Line2D([0], [0], color=ONSET_COLOR, lw=1.6, ls="--",
               label="dense segment ends"),
        Line2D([0], [0], color=EPISODE_SHADE, lw=8, label="dense segment"),
        Line2D([0], [0], color="none",
               label=f"after: {after.mean():.0f} Hz mean, "
                     f"{(after < QUIET_HZ).mean():.0%} near silence"),
    ], loc="upper right", frameon=False, fontsize=8.5, ncol=3,
       handlelength=1.6, columnspacing=1.1)

    # (b) the path through the slow-variable plane; the whole finding lives here
    ax = fig.add_subplot(gs[0, 1])
    for key, style in ((lead_key, dict(lw=2.0)), ((1.0, 0.5), dict(lw=1.4))):
        if key not in runs:
            continue
        arm = _load(runs[key])
        d = 1.0 - arm["trace_z_core_mean"]
        a = arm["trace_m_core_mean"] * 1e-3 * key[0]      # accumulated current, mV
        sl = slice(0, None, 200)
        if key == lead_key:
            pts = ax.scatter(d[sl], a[sl], c=np.arange(d[sl].size) * 0.02,
                             cmap="viridis", s=7, zorder=4)
            cb = fig.colorbar(pts, ax=ax, pad=0.02, fraction=0.045)
            cb.ax.set_title("time (s)", fontsize=8.5, pad=6)
            ax.plot(d[sl], a[sl], color="#555", lw=0.6, alpha=0.5, zorder=3)
            ax.scatter([d[0]], [a[0]], marker="o", s=70, facecolor="white",
                       edgecolor="#333", lw=1.4, zorder=6)
            ax.scatter([d[-1]], [a[-1]], marker="s", s=70, facecolor=ONSET_COLOR,
                       edgecolor="white", lw=1.0, zorder=6)
        else:
            ax.plot(d[sl], a[sl], color="#6b7480", zorder=2, alpha=0.85, **style)
            ax.scatter([d[-1]], [a[-1]], marker="s", s=55, facecolor="#9aa5b1",
                       edgecolor="white", lw=1.0, zorder=5)
    ax.set(xlabel=r"Disinhibition   $D=1-\bar z$",
           ylabel="Accumulated adaptation current (mV)")
    ax.set_title("slow-variable path", fontsize=11, fontweight="bold")
    ax.annotate("25 s constant:\nreturns", xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=8.5, va="top", color="#333")
    ax.annotate("0.5 s constant:\nruns to full disinhibition\nand stops there",
                xy=(0.97, 0.30), xycoords="axes fraction", fontsize=8.5,
                ha="right", va="top", color="#6b7480")

    lower = gs[1, :].subgridspec(1, 3, wspace=0.26, width_ratios=[1.15, 1.0, 1.0])

    # (c) where the activity sits along the pathological axis
    ax = fig.add_subplot(lower[0, 0])
    kymo = lead["coarse_kymo_axial"]
    ax.imshow(np.sqrt(np.clip(kymo, 0, None)), origin="lower", aspect="auto",
              cmap="magma", interpolation="nearest",
              extent=[0, 0.025 * kymo.shape[1], 0, kymo.shape[0]])
    ax.axvline(window[1], color="white", lw=1.2, ls="--")
    ax.set(xlabel="Simulation time (s)", ylabel="Axis bin")
    ax.set_title("where the activity is", fontsize=11, fontweight="bold")

    # (d) which adaptation timescale ends the episode
    ax = fig.add_subplot(lower[0, 1])
    order = sorted(runs, key=lambda k: (k[1], k[0]))
    cmap = plt.get_cmap("plasma")
    taus = sorted({k[1] for k in order})
    for key in order:
        if key[0] not in (1.0, 3.0):
            continue
        arm = _load(runs[key])
        blocks = arm["fine_core_rate_hz"][:15000].reshape(-1, 1000).mean(1)
        colour = cmap(0.12 + 0.72 * taus.index(key[1]) / max(1, len(taus) - 1))
        ax.plot(np.arange(blocks.size) * 2 + 1, blocks, lw=1.7, color=colour,
                ls="-" if key[0] == 1.0 else (0, (4, 1.6)),
                label=f"{key[1]:g} s, {key[0]:g}×")
    ax.set(xlabel="Simulation time (s)", ylabel="Core rate (Hz)")
    ax.set_title("which timescale ends it", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, title="adaptation", title_fontsize=8)

    # (e) the honesty panel: the elevated segment against the locked clause
    ax = fig.add_subplot(lower[0, 2])
    ax.barh([0], [quiet], color="#B2182B", height=0.42)
    ax.axvline(GAP_GATE, color="#333", ls=":", lw=1.4)
    ax.annotate(f"locked clause\n≤ {GAP_GATE:g}", xy=(GAP_GATE, 0.42),
                fontsize=8.5, ha="center", va="bottom", color="#333")
    ax.annotate(f"{quiet:.2f}", xy=(quiet - 0.02, 0), ha="right", va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.set(xlim=(0, 1), ylim=(-0.6, 0.9), yticks=[],
           xlabel="fraction of the elevated segment spent near silence")
    ax.set_title("not a qualified carrier", fontsize=11, fontweight="bold",
                 color="#B2182B")

    fig.suptitle(
        "An adaptation slow enough to keep integrating ends the dense segment "
        "and lets inhibition return\n"
        "single wiring; the control that would attribute the exit to adaptation "
        "has not landed",
        fontsize=12.0, y=0.975, linespacing=1.35,
    )
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    stem = fig_dir / "fig5_adaptation_timescale_episode"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    meta = {
        "lead_arm": {"gain": lead_key[0], "tau_M_s": lead_key[1],
                     "run": runs[lead_key].name},
        "elevated_segment_s": list(window),
        "elevated_quiet_fraction": quiet,
        "locked_deep_gap_clause": GAP_GATE,
        "qualifies_as_carrier": bool(quiet <= GAP_GATE),
        "arms": {f"gain{g:g}_tau{s:g}s": {
            "run": runs[(g, s)].name,
            "ends_below_20hz_at_s": _first_below(_load(runs[(g, s)])["fine_core_rate_hz"], 20.0),
        } for (g, s) in sorted(runs)},
        "boundary": (
            "single SOM wiring, released Z/M from the pre-entry checkpoint; the "
            "zero-current control at the same timescale is still running, so the "
            "exit is not yet attributed to the adaptation current"
        ),
    }
    (fig_dir / "fig5_adaptation_timescale_episode_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(stem.with_suffix(".png"))
    print(json.dumps(meta["arms"], indent=2, sort_keys=True))
    print("quiet fraction of elevated segment:", round(quiet, 3),
          "-> qualifies as carrier:", meta["qualifies_as_carrier"])


if __name__ == "__main__":
    main()
