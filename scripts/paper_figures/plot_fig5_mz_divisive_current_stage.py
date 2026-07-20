#!/usr/bin/env python3
"""Paper-ready *diagnostic* of the current-based MZ-divisive lifecycle screen.

This producer deliberately separates two evidence layers:

1. ``failure_summary`` is plotting-only and consumes the locked v2/v3 traces.  It
   shows the operational sustained-recruitment phenotype, unresolved slow-state drift, and
   the exact linear-M ladder's prevention phenotype.
2. ``current_stage`` consumes the optional representative spatial-capture artifact
   produced by ``run_topic4_mz_divisive_figure_capture.py``.  It is rendered only
   when that artifact exists; the script never substitutes a population-rate trace
   for missing spatial data.

Both outputs are visual diagnostics, not a locked Figure 5 claim and not evidence
for a complete seizure lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/topic4_sef_hfo/mz_divisive_lifecycle"
DEFAULT_OUT = ROOT / "results/paper-ready-figure/fig5_mz_divisive_current_stage/figures"
V2_RUN = RESULT_ROOT / "runs/20260719T162035.230785Z_6ce230e_e1acc35592_slow_gate"
V3_RUN = RESULT_ROOT / "runs/20260719T172358.336529Z_6ce230e_80a127d772_slow_gate_m"
CAPTURE_NPZ = RESULT_ROOT / "figure_capture/current_stage_capture.npz"
CAPTURE_JSON = RESULT_ROOT / "figure_capture/current_stage_capture.json"

COL_RATE = "#3b3b3b"
COL_Z = "#3b6fb6"
COL_TG = "#c74343"
COL_AG = "#d8902f"
COL_ONSET = "#b2182b"
COL_EVENT = "#6F9FD8"
COL_RECRUIT = "#d62748"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _key(label: str, field: str) -> str:
    return f"{label.replace('.', 'p').replace('-', '_')}__{field}"


def _smooth(x: np.ndarray, dt_ms: float, win_ms: float) -> np.ndarray:
    x = np.asarray(x, float)
    n = max(1, int(round(float(win_ms) / float(dt_ms))))
    if n == 1:
        return x.copy()
    left = n // 2
    right = n - 1 - left
    xp = np.pad(x, (left, right), mode="edge")
    cs = np.r_[0.0, np.cumsum(xp, dtype=float)]
    return (cs[n:] - cs[:-n]) / float(n)


def _trace_time(row: dict, arr: np.ndarray) -> tuple[np.ndarray, float]:
    dt_ms = float(row["T_ms"]) / max(1, int(arr.size))
    return np.arange(arr.size, dtype=float) * dt_ms * 1e-3, dt_ms


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _capture_required_arrays() -> set[str]:
    return {
        "times_ms", "rate_E_hz", "rate_state_envelope_250ms_hz", "slow_z_mean",
        "slow_TG", "lfp_times_ms", "lfp_gamma_30_80", "lfp_quiet_median",
        "posE", "contacts", "contact_names", "valid_contacts", "contact_axis_mm",
        "src_xy", "snk_xy", "center_xy", "axis_unit",
        "pre_event_first_spike_latency_ms", "pre_event_contact_latency_ms",
        "pre_event_contact_readable", "recruited_neuron_rate_hz",
        "recruited_contact_energy", "axial_active_fraction", "axial_times_ms",
        "axial_centers_mm",
    }


def _validate_capture(data: np.lib.npyio.NpzFile, meta: dict) -> None:
    missing = sorted(_capture_required_arrays() - set(data.files))
    if missing:
        raise KeyError(f"capture is missing required arrays: {missing}")
    if any("spk_bool" in key.lower() or "spike_raster" in key.lower() for key in data.files):
        raise RuntimeError("paper producer refuses capture artifacts containing a full spike raster")
    if meta.get("schema_version") != "topic4_mz_divisive_current_stage_capture_v1":
        raise RuntimeError(f"unexpected capture schema: {meta.get('schema_version')!r}")
    sim = meta.get("simulation", {})
    cfg = sim.get("cell_config", {})
    locked = {
        "seed": 1, "T_ms": 20_000.0, "spontaneous_no_kick": True,
    }
    for key, expected in locked.items():
        if sim.get(key) != expected:
            raise RuntimeError(f"capture drift: simulation.{key}={sim.get(key)!r}, expected {expected!r}")
    locked_cfg = {
        "use_z": True, "use_m": False, "alpha_G": 2.0, "use_TG": True,
        "alpha_TG": 4.0, "tau_TG": 750.0, "eta_m": 0.0,
    }
    for key, expected in locked_cfg.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"capture drift: cell_config.{key}={cfg.get(key)!r}, expected {expected!r}")
    state = meta.get("recruited_state", {})
    if state.get("status") != "recruited_macrostate":
        raise RuntimeError(f"locked capture lost recruited macrostate: {state}")
    expected_hash = meta.get("artifact_contract", {}).get("npz_sha256")
    if expected_hash and _sha256(CAPTURE_NPZ) != expected_hash:
        raise RuntimeError("capture NPZ SHA256 does not match its JSON sidecar")

    t = np.asarray(data["times_ms"])
    n_t = t.size
    for key in ("rate_E_hz", "rate_state_envelope_250ms_hz", "slow_z_mean", "slow_TG"):
        if np.asarray(data[key]).shape != (n_t,):
            raise RuntimeError(f"unaligned capture trace {key}: {np.asarray(data[key]).shape}")
    pos = np.asarray(data["posE"])
    n_e = pos.shape[0]
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise RuntimeError(f"invalid posE shape: {pos.shape}")
    for key in ("pre_event_first_spike_latency_ms", "recruited_neuron_rate_hz"):
        if np.asarray(data[key]).shape != (n_e,):
            raise RuntimeError(f"unaligned neuron field {key}: {np.asarray(data[key]).shape}")
    contacts = np.asarray(data["contacts"])
    n_c = contacts.shape[0]
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise RuntimeError(f"invalid contacts shape: {contacts.shape}")
    for key in ("contact_names", "valid_contacts", "contact_axis_mm",
                "pre_event_contact_latency_ms", "pre_event_contact_readable",
                "recruited_contact_energy", "lfp_quiet_median"):
        if np.asarray(data[key]).shape != (n_c,):
            raise RuntimeError(f"unaligned contact field {key}: {np.asarray(data[key]).shape}")
    lfp = np.asarray(data["lfp_gamma_30_80"])
    if lfp.ndim != 2 or lfp.shape[1] != n_c or lfp.shape[0] != np.asarray(data["lfp_times_ms"]).size:
        raise RuntimeError(f"unaligned virtual-SEEG array: {lfp.shape}")


def _selected_contact_indices(
    valid: np.ndarray, axis: np.ndarray, max_contacts: int | None = None
) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(valid, bool))
    if idx.size == 0:
        raise RuntimeError("capture has no valid contacts")
    idx = idx[np.argsort(np.asarray(axis, float)[idx])]
    if max_contacts is None or idx.size <= max_contacts:
        return idx
    pick = np.unique(np.round(np.linspace(0, idx.size - 1, max_contacts)).astype(int))
    return idx[pick]


def _spatial_base(ax: plt.Axes, pos: np.ndarray, src: np.ndarray, snk: np.ndarray) -> None:
    ax.scatter(pos[:, 0], pos[:, 1], s=0.38, c="#e5e5e5", linewidths=0,
               rasterized=True, zorder=0)
    ax.scatter([src[0]], [src[1]], marker="o", s=64, facecolor="none",
               edgecolor="#2166ac", linewidth=1.4, zorder=5)
    ax.scatter([snk[0]], [snk[1]], marker="s", s=58, facecolor="none",
               edgecolor="#b2182b", linewidth=1.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")


def _causal_moving_mean_2d(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Trailing moving mean for a time x space array (no future leakage)."""
    values = np.asarray(values, float)
    if values.ndim != 2 or n_frames < 1:
        raise ValueError("values must be 2D and n_frames >= 1")
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    out = np.empty_like(values)
    for index in range(values.shape[0]):
        start = max(0, index - int(n_frames) + 1)
        out[index] = (csum[index + 1] - csum[start]) / float(index - start + 1)
    return out


def _axial_fast_sweep(
    active_fraction: np.ndarray,
    times_ms: np.ndarray,
    axial_mm: np.ndarray,
    onset_ms: float,
    *,
    threshold: float = 0.05,
    smooth_ms: float = 50.0,
    hold_ms: float = 30.0,
) -> dict:
    """Descriptive fast-event sweep at the operational transition.

    This is deliberately not called a tissue-recruitment wavefront.  It measures the first
    post-onset fast event using a causal active-fraction threshold, and is used to distinguish
    a tens-of-ms axial event wave from the desired seconds-scale state front.
    """
    field = np.asarray(active_fraction, float)
    times = np.asarray(times_ms, float)
    axial = np.asarray(axial_mm, float)
    if field.ndim != 2 or field.shape != (times.size, axial.size) or times.size < 3:
        raise ValueError("axial field, time, and coordinate arrays do not align")
    dt_ms = float(np.median(np.diff(times)))
    n_smooth = max(1, int(round(float(smooth_ms) / dt_ms)))
    n_hold = max(1, int(round(float(hold_ms) / dt_ms)))
    smoothed = _causal_moving_mean_2d(field, n_smooth)
    first_index = int(np.searchsorted(times, float(onset_ms), side="left"))
    crossing = np.full(axial.size, np.nan, float)
    kernel = np.ones(n_hold, dtype=int)
    for column in range(axial.size):
        above = smoothed[:, column] >= float(threshold)
        run = np.convolve(above.astype(int), kernel, mode="valid")
        eligible = np.flatnonzero((run >= n_hold) & (np.arange(run.size) >= first_index))
        if eligible.size:
            crossing[column] = times[int(eligible[0])]
    valid = np.isfinite(crossing)
    rho = spearmanr(axial[valid], crossing[valid]) if int(valid.sum()) >= 3 else None
    span_ms = float(np.ptp(crossing[valid])) if int(valid.sum()) >= 2 else None
    distance_mm = float(np.ptp(axial[valid])) if int(valid.sum()) >= 2 else None
    speed_mm_s = (
        float(distance_mm / (span_ms * 1e-3))
        if span_ms is not None and span_ms > 0.0 and distance_mm is not None
        else None
    )
    return {
        "crossing_ms": crossing,
        "n_bins": int(axial.size),
        "n_crossed": int(valid.sum()),
        "threshold_active_fraction": float(threshold),
        "causal_smooth_ms": float(smooth_ms),
        "hold_ms": float(hold_ms),
        "crossing_span_ms": span_ms,
        "axis_span_mm": distance_mm,
        "apparent_speed_mm_s": speed_mm_s,
        "axis_spearman": None if rho is None else float(rho.statistic),
        "axis_spearman_p": None if rho is None else float(rho.pvalue),
        "interpretation": "fast_axial_event_sweep_not_slow_tissue_recruitment_front",
    }


def plot_current_stage(out_dir: Path) -> dict:
    """Render the locked current-stage spatial diagnostic from one exact capture."""
    meta = _load_json(CAPTURE_JSON)
    with np.load(CAPTURE_NPZ, allow_pickle=False) as d:
        _validate_capture(d, meta)
        arrays = {key: np.asarray(d[key]) for key in d.files}

    t_s = arrays["times_ms"].astype(float) * 1e-3
    lfp_t_s = arrays["lfp_times_ms"].astype(float) * 1e-3
    event = meta["representative_pre_onset_event"]
    onset_s = float(meta["recruited_state"]["onset_ms"]) * 1e-3
    event_on_s = float(event["t_on"]) * 1e-3
    event_off_s = float(event["t_off"]) * 1e-3
    x0 = max(0.0, event_on_s - 0.75)
    x1 = float(t_s[-1])

    valid = arrays["valid_contacts"].astype(bool)
    contact_idx = _selected_contact_indices(valid, arrays["contact_axis_mm"])
    names = arrays["contact_names"].astype(str)
    lfp = arrays["lfp_gamma_30_80"].astype(float)[:, contact_idx]
    pre_onset_lfp = lfp_t_s < onset_s
    amp = np.nanpercentile(np.abs(lfp[pre_onset_lfp]), 95.0, axis=0)
    fallback = np.nanmedian(np.abs(lfp), axis=0)
    amp = np.where(np.isfinite(amp) & (amp > 1e-12), amp, fallback)
    amp = np.maximum(amp, np.finfo(float).eps)
    # Do not normalize on or numerically clip the recruited segment.  Large excursions are allowed
    # to overlap/leave the stacked axes, making a high-state amplitude jump visible rather than
    # cosmetically forcing it to look bounded.
    lfp_norm = lfp / amp[None, :]

    fig = plt.figure(figsize=(15.4, 10.0), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=(2.15, 1.0, 3.55),
        left=0.065, right=0.965, bottom=0.075, top=0.935, hspace=0.27,
    )
    spatial_gs = gs[2].subgridspec(1, 3, width_ratios=(1.0, 1.0, 1.18), wspace=0.34)
    axial_gs = spatial_gs[0, 2].subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.42)

    ax_lfp = fig.add_subplot(gs[0])
    spacing = 9.0
    offsets = np.arange(contact_idx.size, dtype=float) * spacing
    for j in range(contact_idx.size):
        ax_lfp.plot(lfp_t_s, lfp_norm[:, j] + offsets[j], color="#222222", lw=0.42,
                    rasterized=True)
    ax_lfp.axvspan(event_on_s, event_off_s, color=COL_EVENT, alpha=0.20, lw=0)
    ax_lfp.axvline(onset_s, color=COL_ONSET, lw=1.2, ls="--")
    ax_lfp.axvspan(onset_s, x1, color=COL_RECRUIT, alpha=0.055, lw=0)
    ax_lfp.set_xlim(x0, x1)
    ax_lfp.set_ylim(-0.7 * spacing, offsets[-1] + 0.8 * spacing)
    ax_lfp.set_yticks(offsets, names[contact_idx])
    ax_lfp.tick_params(axis="x", labelbottom=False)
    ax_lfp.set_ylabel("virtual contacts\n(source -> sink)")
    ax_lfp.set_title("A  Continuous virtual-SEEG readout (30–80 Hz current proxy)", loc="left",
                     fontsize=10.5, fontweight="bold")
    ax_lfp.text(event_on_s, offsets[-1] + 0.55 * spacing, "returning event", color="#2c5c91",
                fontsize=8.0, ha="left", va="center", fontweight="bold")
    ax_lfp.text(onset_s + 0.08, offsets[-1] + 0.55 * spacing, "strict recruited onset",
                color=COL_ONSET, fontsize=8.0, ha="left", va="center", fontweight="bold")
    ax_lfp.text(0.995, 0.03, "scale: each contact's pre-onset |signal| P95",
                transform=ax_lfp.transAxes, ha="right", va="bottom", fontsize=7.2, color="0.35")

    ax_rate = fig.add_subplot(gs[1], sharex=ax_lfp)
    ax_rate.plot(t_s, arrays["rate_state_envelope_250ms_hz"], color=COL_RATE, lw=1.05,
                 label="E rate (250-ms envelope)")
    ax_rate.axhline(20.0, color="0.58", lw=0.75, ls=":")
    ax_rate.axvspan(event_on_s, event_off_s, color=COL_EVENT, alpha=0.18, lw=0)
    ax_rate.axvline(onset_s, color=COL_ONSET, lw=1.1, ls="--")
    ax_rate.set_xlim(x0, x1)
    ax_rate.set_ylabel("E rate (Hz)")
    ax_rate.set_xlabel("time (s)")
    ax_rate.set_title("B  Operational recruitment with unresolved slow drift", loc="left",
                      fontsize=10.5, fontweight="bold")
    ax_slow = ax_rate.twinx()
    ax_slow.plot(t_s, arrays["slow_z_mean"], color=COL_Z, lw=1.0, label=r"$\langle z\rangle$")
    ax_slow.plot(t_s, arrays["slow_TG"], color=COL_TG, lw=1.0, label=r"$T_G$")
    ax_slow.set_ylim(0.0, 1.02)
    ax_slow.set_ylabel("slow state")
    handles = [
        Line2D([0], [0], color=COL_RATE, lw=1.2, label="E rate"),
        Line2D([0], [0], color=COL_Z, lw=1.2, label=r"$\langle z\rangle$"),
        Line2D([0], [0], color=COL_TG, lw=1.2, label=r"$T_G$"),
    ]
    ax_rate.legend(handles=handles, loc="upper left", frameon=False, ncol=3, fontsize=7.8)

    pos = arrays["posE"].astype(float)
    contacts = arrays["contacts"].astype(float)
    src = arrays["src_xy"].astype(float)
    snk = arrays["snk_xy"].astype(float)

    ax_event = fig.add_subplot(spatial_gs[0, 0])
    _spatial_base(ax_event, pos, src, snk)
    latency = arrays["pre_event_first_spike_latency_ms"].astype(float)
    active = np.isfinite(latency)
    readable = arrays["pre_event_contact_readable"].astype(bool) & valid
    c_latency = arrays["pre_event_contact_latency_ms"].astype(float)
    event_duration = max(1.0, float(event["duration_ms"]))
    observed_late = [event_duration]
    if np.any(active):
        observed_late.append(float(np.nanmax(latency[active])))
    if np.any(readable):
        observed_late.append(float(np.nanmax(c_latency[readable])))
    event_latency_vmax = max(observed_late)
    sc_event = ax_event.scatter(pos[active, 0], pos[active, 1], s=2.0, c=latency[active],
                                cmap="viridis", vmin=0.0, vmax=event_latency_vmax,
                                linewidths=0, rasterized=True, zorder=2)
    ax_event.scatter(contacts[valid & ~readable, 0], contacts[valid & ~readable, 1], s=25,
                     marker="D", facecolor="white", edgecolor="0.35", linewidth=0.55, zorder=4)
    ax_event.scatter(contacts[readable, 0], contacts[readable, 1], s=30, marker="D",
                     c=c_latency[readable], cmap="viridis", vmin=0.0, vmax=event_latency_vmax,
                     edgecolor="white", linewidth=0.6, zorder=5)
    cb = fig.colorbar(sc_event, ax=ax_event, fraction=0.047, pad=0.025)
    cb.set_label("event latency (ms)", fontsize=7.8)
    pos_axis = (pos - arrays["center_xy"]) @ arrays["axis_unit"]
    event_neuron_rho = spearmanr(pos_axis[active], latency[active]) if int(active.sum()) >= 3 else None
    ax_event.set_title("C  Returning event field", loc="left", fontsize=10.2, fontweight="bold")
    event_rho_text = "n/a" if event_neuron_rho is None else f"{event_neuron_rho.statistic:+.2f}"
    ax_event.text(0.02, 0.02,
                  f"{int(active.sum()):,} active E neurons\n"
                  f"neuron axis-latency rho={event_rho_text}; contact direction unresolved",
                  transform=ax_event.transAxes, fontsize=7.7, va="bottom",
                  bbox=dict(fc="white", ec="0.8", alpha=0.90, boxstyle="round,pad=0.22"))

    ax_rec = fig.add_subplot(spatial_gs[0, 1])
    _spatial_base(ax_rec, pos, src, snk)
    rec_rate = arrays["recruited_neuron_rate_hz"].astype(float)
    positive = rec_rate[np.isfinite(rec_rate) & (rec_rate > 0)]
    rec_vmax = float(np.nanpercentile(positive, 99.0)) if positive.size else 1.0
    rec_vmax = max(rec_vmax, 1.0)
    sc_rec = ax_rec.scatter(pos[:, 0], pos[:, 1], s=1.5, c=rec_rate, cmap="magma",
                            vmin=0.0, vmax=rec_vmax, linewidths=0, rasterized=True, zorder=2)
    energy = arrays["recruited_contact_energy"].astype(float)
    finite_energy = valid & np.isfinite(energy)
    energy_p50 = float(np.nanpercentile(energy[finite_energy], 50.0)) if np.any(finite_energy) else 0.0
    energy_p95 = float(np.nanpercentile(energy[finite_energy], 95.0)) if np.any(finite_energy) else 1.0
    denom = max(energy_p95, np.finfo(float).eps)
    marker_size = 16.0 + 64.0 * np.clip(energy[finite_energy] / denom, 0.0, 1.0)
    ax_rec.scatter(contacts[finite_energy, 0], contacts[finite_energy, 1], s=marker_size,
                   facecolor="#35c4d8", edgecolor="white", linewidth=0.7, alpha=0.80, zorder=5)
    cb = fig.colorbar(sc_rec, ax=ax_rec, fraction=0.047, pad=0.025)
    cb.set_label("E-neuron rate in first 1 s (Hz)", fontsize=7.8)
    ax_rec.set_title("D  Recruited-window field", loc="left", fontsize=10.2, fontweight="bold")
    size50 = float(np.sqrt(16.0 + 64.0 * np.clip(energy_p50 / denom, 0.0, 1.0)))
    size95 = float(np.sqrt(80.0))
    ax_rec.legend(
        handles=[
            Line2D([0], [0], marker="o", ls="", markersize=size50,
                   markerfacecolor="#35c4d8", markeredgecolor="white",
                   label=f"contact energy P50={energy_p50:.2g}"),
            Line2D([0], [0], marker="o", ls="", markersize=size95,
                   markerfacecolor="#35c4d8", markeredgecolor="white",
                   label=f"contact energy P95={energy_p95:.2g}"),
        ],
        title="gamma proxy²", loc="lower right", frameon=True, framealpha=0.9,
        fontsize=6.7, title_fontsize=6.8,
    )
    rec_active_fraction = float(np.mean(rec_rate > 0.0))
    rec_over_100_fraction = float(np.mean(rec_rate > 100.0))
    ax_rec.text(
        0.02, 0.98,
        f"active={100.0 * rec_active_fraction:.1f}%\n"
        f">100 Hz={100.0 * rec_over_100_fraction:.1f}%",
        transform=ax_rec.transAxes, ha="left", va="top", fontsize=7.4,
        bbox=dict(fc="white", ec="0.8", alpha=0.90, boxstyle="round,pad=0.22"),
    )

    ax_st = fig.add_subplot(axial_gs[0, 0])
    st = arrays["axial_active_fraction"].astype(float)
    st_t = arrays["axial_times_ms"].astype(float) * 1e-3
    axial = arrays["axial_centers_mm"].astype(float)
    time_sel = (st_t >= x0) & (st_t <= x1)
    st_show = st[time_sel]
    finite_st = st_show[np.isfinite(st_show)]
    st_vmax = float(np.nanpercentile(finite_st, 99.0)) if finite_st.size else 1.0
    st_vmax = max(st_vmax, 1e-3)
    im = ax_st.imshow(
        st_show.T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
        vmin=0.0, vmax=st_vmax,
        extent=(float(st_t[time_sel][0]), float(st_t[time_sel][-1]),
                float(axial[0]), float(axial[-1])), rasterized=True,
    )
    ax_st.axvspan(event_on_s, event_off_s, color=COL_EVENT, alpha=0.20, lw=0)
    ax_st.axvline(onset_s, color="white", lw=1.0, ls="--")
    src_axis = float((src - arrays["center_xy"]) @ arrays["axis_unit"])
    snk_axis = float((snk - arrays["center_xy"]) @ arrays["axis_unit"])
    ax_st.scatter([event_on_s - 0.12], [src_axis], s=30, marker="o", facecolor="none",
                  edgecolor="#4db8ff", linewidth=1.2, clip_on=False)
    ax_st.scatter([event_on_s - 0.12], [snk_axis], s=28, marker="s", facecolor="none",
                  edgecolor="#ff8c8c", linewidth=1.2, clip_on=False)
    ax_st.set(xlabel="time (s)", ylabel="source-to-sink axis (mm)")
    ax_st.set_title("E  Axial activity across transition", loc="left", fontsize=10.2,
                    fontweight="bold")
    cb = fig.colorbar(im, ax=ax_st, fraction=0.047, pad=0.025)
    cb.set_label("active E fraction / 10 ms", fontsize=7.8)

    sweep = _axial_fast_sweep(
        st,
        arrays["axial_times_ms"].astype(float),
        axial,
        float(meta["recruited_state"]["onset_ms"]),
    )
    ax_zoom = fig.add_subplot(axial_gs[1, 0])
    zoom0 = max(float(st_t[0]), onset_s - 0.04)
    zoom1 = min(float(st_t[-1]), onset_s + 0.20)
    zoom_sel = (st_t >= zoom0) & (st_t <= zoom1)
    ax_zoom.imshow(
        st[zoom_sel].T, origin="lower", aspect="auto", interpolation="nearest", cmap="magma",
        vmin=0.0, vmax=st_vmax,
        extent=(float(st_t[zoom_sel][0]), float(st_t[zoom_sel][-1]),
                float(axial[0]), float(axial[-1])), rasterized=True,
    )
    ax_zoom.axvline(onset_s, color="white", lw=1.0, ls="--")
    crossing_s = np.asarray(sweep["crossing_ms"], float) * 1e-3
    cross_valid = np.isfinite(crossing_s)
    ax_zoom.scatter(crossing_s[cross_valid], axial[cross_valid], s=7.0, facecolor="none",
                    edgecolor="#64d8ff", linewidth=0.55, zorder=4)
    ax_zoom.set(xlabel="time (s)", ylabel="axis (mm)")
    ax_zoom.set_title("F  Fast event sweep at operational onset", loc="left", fontsize=10.2,
                      fontweight="bold")
    speed = sweep["apparent_speed_mm_s"]
    speed_text = "n/a" if speed is None else f"~{speed:.0f} mm/s"
    ax_zoom.text(
        0.02, 0.96,
        f"{sweep['n_crossed']}/{sweep['n_bins']} bins in {sweep['crossing_span_ms']:.0f} ms; "
        f"rho={sweep['axis_spearman']:+.2f}\n{speed_text}: fast wave, not a slow recruitment front",
        transform=ax_zoom.transAxes, ha="left", va="top", fontsize=7.1, color="white",
        bbox=dict(fc="black", ec="white", alpha=0.54, boxstyle="round,pad=0.22"),
    )

    for ax in (ax_lfp, ax_rate, ax_event, ax_rec, ax_st, ax_zoom):
        ax.tick_params(labelsize=7.7, length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_slow.tick_params(labelsize=7.7, length=3)
    ax_slow.spines["top"].set_visible(False)
    fig.suptitle("Current Z–$T_G$ model: operational recruitment without observed recovery",
                 fontsize=13.0, fontweight="bold", y=0.982)
    reaches_end = bool(meta["recruited_state"].get("reaches_record_end", False))
    footer = ("Diagnostic only — recruited activity reaches the 20-s record end; "
              "no observed offset, recovery, or frozen ictal attractor."
              if reaches_end else
              "Diagnostic only — current-stage trace; spatial and frozen-state claims remain gated.")
    fig.text(0.5, 0.018, footer, ha="center", va="bottom", fontsize=8.5, color="#7f0000")

    stem = out_dir / "fig5_candidate_E1146_mz_divisive_current_stage"
    _save(fig, stem)
    figure_meta = {
        "status": "paper-ready current-stage diagnostic; not a locked Figure 5 panel",
        "figure": str(stem.with_suffix(".png").relative_to(ROOT)),
        "sources": {
            "capture_npz": str(CAPTURE_NPZ.relative_to(ROOT)),
            "capture_json": str(CAPTURE_JSON.relative_to(ROOT)),
            "capture_npz_sha256": meta["artifact_contract"].get("npz_sha256"),
        },
        "display": {
            "time_window_s": [x0, x1],
            "selected_contact_indices": contact_idx.astype(int).tolist(),
            "selected_contact_names": names[contact_idx].tolist(),
            "virtual_seeg_scale": "per-contact pre-onset absolute-amplitude P95; recruited data not clipped before plotting",
            "event_latency_vmax_ms": event_latency_vmax,
            "recruited_neuron_rate_vmax_hz_p99": rec_vmax,
            "recruited_contact_energy_p50_proxy_squared": energy_p50,
            "recruited_contact_energy_p95_proxy_squared": energy_p95,
            "axial_active_fraction_vmax_p99": st_vmax,
            "operational_onset_zoom_s": [zoom0, zoom1],
        },
        "scientific_readout": {
            "strict_recruited_state": meta["recruited_state"],
            "representative_pre_onset_event": event,
            "pre_event_contact_direction": meta["readout_contract"].get("pre_event_direction"),
            "pre_event_axis_spearman": meta["readout_contract"].get("pre_event_axis_spearman"),
            "pre_event_neuron_axis_spearman": (
                None if event_neuron_rho is None else float(event_neuron_rho.statistic)
            ),
            "pre_event_active_E_neurons": int(active.sum()),
            "recruited_first_1s": {
                "active_neuron_fraction": rec_active_fraction,
                "over_100hz_neuron_fraction": rec_over_100_fraction,
                "rate_hz_percentiles_all_neurons": {
                    str(percentile): float(np.percentile(rec_rate, percentile))
                    for percentile in (0, 5, 50, 95, 99, 100)
                },
            },
            "operational_onset_fast_axial_sweep": {
                key: value for key, value in sweep.items() if key != "crossing_ms"
            },
        },
        "claim_boundary": [
            "single locked seed and finite 20-s record",
            "virtual-SEEG is the existing pre-divisor synaptic-current proxy, not effective membrane current",
            "recruited segment reaching record end is not observed maintenance or termination",
            "spatial maps diagnose the current structure and do not establish a propagated ictal front",
            "the ordered tens-of-ms axial sweep is a fast event wave, not a slow tissue-state front",
        ],
    }
    (out_dir / f"{stem.name}_metadata.json").write_text(
        json.dumps(figure_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return figure_meta


def plot_failure_summary(out_dir: Path) -> dict:
    """Render the already-completed v2/v3 screen without re-running the SNN."""
    v2_summary = _load_json(V2_RUN / "summary.json")
    v2_strict = _load_json(V2_RUN / "strict_audit.json")
    v3_summary = _load_json(V3_RUN / "summary.json")
    v3_strict = _load_json(V3_RUN / "strict_audit.json")
    v2_row = next(r for r in v2_summary["rows"] if r["label"] == "slow_gate_aT4_tau750")
    v2_audit = next(r for r in v2_strict["rows"] if r["label"] == v2_row["label"])
    v3_rows = list(v3_summary["rows"])

    with np.load(V2_RUN / "traces_downsampled.npz") as v2, np.load(
        V3_RUN / "traces_downsampled.npz"
    ) as v3:
        rate = np.asarray(v2[_key(v2_row["label"], "rate")], float)
        t, dt_ms = _trace_time(v2_row, rate)
        rate50 = _smooth(rate, dt_ms, 50.0)
        z_mean = np.asarray(v2[_key(v2_row["label"], "z_mean")], float)
        tg = np.asarray(v2[_key(v2_row["label"], "TG")], float)
        ag = np.asarray(v2[_key(v2_row["label"], "AG")], float)
        tz = np.linspace(0.0, float(v2_row["T_ms"]) * 1e-3, z_mean.size, endpoint=False)

        ladder = []
        for row in v3_rows:
            arr = np.asarray(v3[_key(row["label"], "rate")], float)
            dti = float(row["T_ms"]) / max(1, arr.size)
            ladder.append(_smooth(arr, dti, 50.0))
        ladder = np.vstack(ladder)

    fig = plt.figure(figsize=(13.8, 4.25), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.25, 1.05, 1.38), wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    onset_s = float(v2_audit["onset_ms"]) * 1e-3
    ax.plot(t, rate50, color=COL_RATE, lw=1.05)
    ax.axhline(20.0, color="0.55", lw=0.8, ls=":")
    ax.axvline(onset_s, color=COL_ONSET, lw=1.15, ls="--")
    ax.axvspan(onset_s, t[-1], color=COL_RECRUIT, alpha=0.075, lw=0)
    ax.text(onset_s + 0.15, 112, "operational onset (no kick)", color=COL_ONSET, fontsize=8.4,
            fontweight="bold", va="top")
    ax.text(0.35, 112, "returning events", color="0.28", fontsize=8.4,
            fontweight="bold", va="top")
    ax.set(xlim=(0, 20), ylim=(0, 120), xlabel="time (s)", ylabel="E population rate (Hz)",
           title="Operational sustained-recruitment phenotype")
    ax.text(0.98, 0.05,
            f"dominant modulation = {v2_audit['burst_peak_hz']:.2f} Hz\nno offset by 20 s",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92))

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(tz, z_mean, color=COL_Z, lw=1.4, label=r"$\langle z\rangle$")
    ax.plot(tz, tg, color=COL_TG, lw=1.25, label=r"$T_G$")
    ax.plot(tz, ag, color=COL_AG, lw=1.05, label=r"$A_G$")
    ax.axvline(onset_s, color=COL_ONSET, lw=1.0, ls="--")
    ax.set(xlim=(0, 20), ylim=(0, 1.02), xlabel="time (s)", ylabel="state value",
           title="Slow state does not settle")
    ax.legend(frameon=False, fontsize=8.1, loc="center left")
    slopes = v2_audit["tail_slopes_per_s"]
    ax.text(0.97, 0.05,
            f"d<z>/dt = {slopes['z_mean']:+.3f} /s\n"
            f"dT_G/dt = {slopes['TG']:+.3f} /s",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92))

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(
        np.clip(ladder, 0.0, 100.0), origin="upper", aspect="auto",
        extent=(0.0, 25.0, len(v3_rows) - 0.5, -0.5), cmap="magma", norm=Normalize(0, 100),
        interpolation="nearest",
    )
    eta = [float(r["cfg"]["eta_m"]) for r in v3_rows]
    labels = ["0 (off)" if e == 0 else f"{e:.5f}" for e in eta]
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set(xlabel="time (s)", ylabel=r"linear $\eta_m$", title="Linear M changes entry, not exit")
    shoulders = {float(k): float(v) for k, v in _load_json(RESULT_ROOT / "pilot_summary.json")[
        "v3_locked_M_ladder"]["m_on_max_recruited_shoulder_ms"].items()}
    for i, e in enumerate(eta):
        if e == 0:
            note = "ongoing"
        else:
            note = f"{int(round(shoulders.get(e, 0.0)))} ms max"
        ax.text(24.55, i, note, ha="right", va="center", color="white", fontsize=7.0,
                fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.047, pad=0.025)
    cb.set_label("50-ms E rate (Hz)", fontsize=8.3)
    cb.ax.tick_params(labelsize=7.5)

    for a in fig.axes:
        if a is cb.ax:
            continue
        a.tick_params(labelsize=8.0, length=3)
        a.title.set_fontsize(10.2)
        a.title.set_fontweight("bold")
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle("Current Z–$T_G$–M screen: operational recruitment under unresolved slow drift",
                 fontsize=12.4, fontweight="bold", y=1.01)
    stem = out_dir / "fig5_candidate_E1146_mz_divisive_failure_summary"
    _save(fig, stem)

    meta = {
        "status": "paper-ready visual diagnostic; not a locked Figure 5 panel",
        "figure": str(stem.with_suffix(".png").relative_to(ROOT)),
        "sources": {
            "v2_summary": str((V2_RUN / "summary.json").relative_to(ROOT)),
            "v2_strict_audit": str((V2_RUN / "strict_audit.json").relative_to(ROOT)),
            "v2_trace": str((V2_RUN / "traces_downsampled.npz").relative_to(ROOT)),
            "v3_summary": str((V3_RUN / "summary.json").relative_to(ROOT)),
            "v3_strict_audit": str((V3_RUN / "strict_audit.json").relative_to(ROOT)),
            "v3_trace": str((V3_RUN / "traces_downsampled.npz").relative_to(ROOT)),
        },
        "v2": {
            "onset_ms": v2_audit["onset_ms"],
            "duration_ms": v2_audit["recruited_duration_ms"],
            "burst_peak_hz": v2_audit["burst_peak_hz"],
            "returned": v2_audit["returned_to_same_seed_slowoff"],
            "tail_slopes_per_s": v2_audit["tail_slopes_per_s"],
        },
        "v3": {
            "verdict": v3_strict["verdict"],
            "eta_m": eta,
            "interpretation": v3_strict["interpretation"],
        },
        "claim_boundary": [
            "single-seed operational sustained recruitment, not a settled ictal attractor",
            "linear M prevented a one-second recruited macrostate; it did not terminate an established state",
            "this summary contains no spatial claim",
        ],
    }
    (out_dir / f"{stem.name}_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def _write_readme(out_dir: Path, spatial_rendered: bool, spatial_meta: dict | None = None) -> None:
    blocks = [
        "# MZ-divisive current-stage visual diagnostics\n",
        "### fig5_candidate_E1146_mz_divisive_failure_summary.png / .pdf\n\n"
        "这张图只消费已经完成的 v2/v3 单 seed 轨迹。左图显示高状态选择性的慢除法器把 delayed runaway 改成约 5 Hz 的有限窗 recruited bursting；中图显示同一窗口里 `z` 仍下降、`T_G` 仍上升，因而它不是 settled branch；右图显示线性 M 的所有非零档都在进入前压低活动，没有先建立发作态再终止。\n\n"
        "**关注点**：当前结构在无 kick 条件下跨过了预注册的持续招募操作阈值，并出现约 5 Hz 主导调制，但缺失稳定高态与回到同一间期 basin 的 exit。右图的非零 M 结果应读成 prevention/containment，而不是 termination；centered 250-ms envelope 的 onset 不是因果分岔时刻。\n",
    ]
    if spatial_rendered:
        readout = (spatial_meta or {}).get("scientific_readout", {})
        sweep = readout.get("operational_onset_fast_axial_sweep", {})
        recruited = readout.get("recruited_first_1s", {})
        sweep_sentence = (
            f"操作性 onset 附近，{sweep.get('n_crossed')}/{sweep.get('n_bins')} 个轴向 bin "
            f"在 {sweep.get('crossing_span_ms'):.0f} ms 内依次跨过因果 50-ms activity 门，"
            f"轴向 Spearman rho={sweep.get('axis_spearman'):+.2f}；"
            if sweep else ""
        )
        recruited_sentence = (
            f"onset 后首个 1 s 中 {100.0 * recruited.get('active_neuron_fraction', 0.0):.1f}% 的 E 神经元发放，"
            f"其中 {100.0 * recruited.get('over_100hz_neuron_fraction', 0.0):.1f}% 的全体 E 神经元超过 100 Hz。"
            if recruited else ""
        )
        blocks.append(
            "\n### fig5_candidate_E1146_mz_divisive_current_stage.png / .pdf\n\n"
            "上方为同一条 20 s 自发轨迹的连续 virtual-SEEG；中间将 population-rate 定义的 recruited onset 与 `z/T_G` 慢漂移对齐；下方分别显示 onset 前一个机器选择的 returning event、onset 后 recruited window 的真实 E-neuron 空间读出、完整 source→sink 轴时空场，以及 onset 附近的放大图。returning-event 颗粒颜色是逐神经元 first-spike latency，菱形颜色是触点 30–80 Hz envelope-peak latency；两者共用毫秒色标但不是同一测量。所有颗粒、触点与空间热图都来自同一次 capture，未用一维 rate 伪造空间结果。\n\n"
            f"{sweep_sentence}这对应约数百 mm/s 的 fast event sweep，而不是秒级的 ictal tissue-recruitment front。{recruited_sentence}\n\n"
            "**关注点**：当前模型并非没有空间结构；它保留 returning-event 的空间颗粒，并在操作性转变处产生快速有序轴向波。真正缺少的是一个慢速、局部改变组织状态的 recruitment front，以及其后的 refractory wake、stall/annihilation 和 return。高率细胞尾部也说明 population mean 的约 60 Hz 不能单独证明一个生理性有界高态。\n"
        )
    blocks.append(
        "\n两张图都是 current-stage diagnostic，不是正式锁定的 Figure 5，也不支持 seizure lifecycle、limit cycle、患者机制或 cohort inference。\n"
    )
    (out_dir / "README.md").write_text("".join(blocks), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--require-spatial", action="store_true",
                    help="fail if the representative spatial-capture artifact is missing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_failure_summary(args.out)

    spatial_rendered = False
    spatial_meta = None
    if CAPTURE_NPZ.exists() and CAPTURE_JSON.exists():
        spatial_meta = plot_current_stage(args.out)
        spatial_rendered = True
    if args.require_spatial and not spatial_rendered:
        raise FileNotFoundError(
            f"representative spatial capture missing: {CAPTURE_NPZ}; run the gated capture first"
        )
    _write_readme(args.out, spatial_rendered, spatial_meta)
    print(f"wrote {args.out / 'fig5_candidate_E1146_mz_divisive_failure_summary.png'}")
    if spatial_rendered:
        print(f"wrote {args.out / 'fig5_candidate_E1146_mz_divisive_current_stage.png'}")
    else:
        print(f"spatial capture pending: {CAPTURE_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
