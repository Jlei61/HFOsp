#!/usr/bin/env python3
"""Diagnostic signed broadband field movie + snapshot montage for Topic 5.

This is intentionally separate from the existing field-dynamics movie:
  - values are signed per-channel baseline robust-z, not rank01;
  - broadband is recomputed as 1-150 Hz log power (closed [lo,hi] sum, notch
    50/100/150/200 applied at load, NO extra FFT-bin line mask -> matches the
    Fig3-B 1-150 feature contract);
  - the colormap is zero-centered blue-white-red.

Two renderings share ONE `compute_field_frames` pass (so they are bit-identical):
  - `--emit gif`     : animated GIF over the movie window (legacy default);
  - `--emit montage` : a static snapshot grid of the same frames (paper-facing
                       "snapshot of the movie"); `--stop-sec` zooms the tail to
                       e.g. the early-ictal rise and `--anchor eeg` labels frame
                       times relative to EEG onset.

Default target: epilepsiae_1146, longest eligible seizure, -120 s to EEG offset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_contact_plane_static import (  # noqa: E402
    _attach_real_coords,
    _display_points,
    _smooth_rank_field_mm,
    _subject_display_frame,
)
from scripts.run_topic5_t0_eligibility import (  # noqa: E402
    GUARD_SEC,
    ICTAL_REFERENCE,
    MIN_BASELINE_SEC,
    PRE_FLOOR,
    TARGET_BASELINE_SEC,
    _eeg_rel_inv,
    _inventory_rows,
)
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window  # noqa: E402


REAL_DIR = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
LONG_CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed/figures"


def _default_seizure_idx(ds_sid: str) -> int:
    meta_f = LONG_CACHE / f"{ds_sid}.json"
    if meta_f.exists():
        meta = json.loads(meta_f.read_text())
        eligible = meta.get("eligible_idxs", [])
        if eligible:
            return int(max(eligible, key=lambda i: meta["seizure"][str(i)]["eeg_duration_sec"]))
    return 0


def _pre_target(dataset: str, inv: dict, *, display_start: float) -> float:
    # Same core rule as Topic 5 T0, plus enough pre-window to display from display_start.
    return max(
        PRE_FLOOR,
        abs(float(display_start)) + 2.0,
        min(abs(_eeg_rel_inv(dataset, inv)), 300.0) + GUARD_SEC + TARGET_BASELINE_SEC,
    )


def _offset_rel(dataset: str, inv: dict) -> float:
    onset_field = "clin_onset_epoch" if dataset == "epilepsiae" else "eeg_onset_epoch"
    if inv.get("eeg_offset_epoch") and inv.get(onset_field):
        return float(inv["eeg_offset_epoch"]) - float(inv[onset_field])
    if inv.get("eeg_duration_sec"):
        return float(inv["eeg_duration_sec"])
    raise ValueError("cannot resolve seizure offset from inventory row")


def _band_power_trace_chunked(signal: np.ndarray, fs: float, *, band: tuple[float, float],
                              win_sec: float, hop_sec: float, chunk_ch: int) -> tuple[np.ndarray, np.ndarray]:
    outs: list[np.ndarray] = []
    t_ref = None
    for i0 in range(0, signal.shape[0], chunk_ch):
        bp, t = recruit.band_power_trace(
            signal[i0:i0 + chunk_ch],
            fs,
            band=band,
            win_sec=win_sec,
            hop_sec=hop_sec,
        )
        if t_ref is None:
            t_ref = np.asarray(t, float)
        elif not np.allclose(t_ref, t):
            raise RuntimeError("spectrogram time grid changed across chunks")
        outs.append(np.asarray(bp, np.float32))
    if t_ref is None:
        raise RuntimeError("empty signal")
    return np.vstack(outs), t_ref


def _load_geometry(ds_sid: str):
    axis_f = REAL_DIR / f"{ds_sid}_t_a.json"
    if not axis_f.exists():
        raise FileNotFoundError(axis_f)
    rec = json.loads(axis_f.read_text())
    _attach_real_coords([rec])
    frame = _subject_display_frame([rec])
    if frame is None:
        raise RuntimeError(f"{ds_sid}: no display frame")
    xs, ys = _display_points(rec, frame)
    names = [c["name"] for c in rec["channels"]]
    support = np.array([float(c.get("support", 1.0)) for c in rec["channels"]], float)
    soz = np.array([bool(c.get("is_soz")) for c in rec["channels"]], bool)
    return rec, frame, np.asarray(xs), np.asarray(ys), names, support, soz


def _window_values(z_sel: np.ndarray, relt: np.ndarray, starts: np.ndarray, win_sec: float) -> np.ndarray:
    rows = []
    for lo in starts:
        hi = lo + win_sec
        m = (relt >= lo) & (relt <= hi)
        if not m.any():
            rows.append(np.full(z_sel.shape[0], np.nan))
        else:
            with np.errstate(invalid="ignore"):
                rows.append(np.nanmean(z_sel[:, m], axis=1))
    return np.asarray(rows, float)


def _smooth(xs, ys, vals, support, frame):
    _, _, field, _, _ = _smooth_rank_field_mm(
        xs,
        ys,
        vals,
        support,
        frame["xlim"],
        frame["ylim"],
        frame["sigma_mm"],
    )
    return field


def compute_field_frames(args: argparse.Namespace) -> dict:
    """Load one seizure, compute per-contact signed baseline robust-z band power on a
    fine hop grid, project onto the subject contact plane, and window it into frames.

    Rendering-free, so the GIF and the snapshot montage consume bit-identical frames.
    Frame grid = [start_sec, stop] rel clinical onset; `stop` defaults to the seizure
    offset (full-seizure movie) and is zoomed by `--stop-sec` (e.g. early-ictal rise).
    """
    ds_sid = args.subject
    dataset, sid = ds_sid.split("_", 1)
    seizure_idx = _default_seizure_idx(ds_sid) if args.seizure_idx is None else int(args.seizure_idx)

    inv_rows, _ = _inventory_rows(dataset, sid)
    if not (0 <= seizure_idx < len(inv_rows)):
        raise IndexError(f"{ds_sid}: seizure_idx={seizure_idx} out of range (n={len(inv_rows)})")
    inv = inv_rows[seizure_idx]
    offset = _offset_rel(dataset, inv)
    pre_sec = _pre_target(dataset, inv, display_start=args.start_sec)

    sw = extract_seizure_window(
        f"{dataset}/{sid}",
        seizure_idx,
        pre_sec=pre_sec,
        post_sec=offset + 0.5,
        reference=ICTAL_REFERENCE[dataset],
    )
    if sw.fs / 2.0 <= args.band_hi:
        raise RuntimeError(f"{ds_sid}: fs={sw.fs} has Nyquist {sw.fs / 2.0}, cannot compute {args.band_hi} Hz")

    bp, t = _band_power_trace_chunked(
        sw.signal,
        sw.fs,
        band=(args.band_lo, args.band_hi),
        win_sec=args.spectral_win_sec,
        hop_sec=args.hop_sec,
        chunk_ch=args.chunk_ch,
    )
    eeg_rel = (sw.eeg_onset_epoch - sw.clin_onset_epoch) if sw.eeg_onset_epoch is not None else None
    bl = resolve_baseline_window(
        bp.shape[1],
        hop_sec=args.hop_sec,
        pre_sec=sw.pre_sec,
        buffer_sec=GUARD_SEC,
        eeg_onset_rel_sec=eeg_rel,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    if not bl.valid:
        raise RuntimeError(f"{ds_sid} sz{seizure_idx}: invalid baseline {bl}")
    z = recruit.baseline_robust_z(
        bp,
        (bl.start_idx, bl.end_idx),
        hop_sec=args.hop_sec,
        min_baseline_valid_sec=MIN_BASELINE_SEC,
    )
    relt = np.asarray(t, float) - float(sw.pre_sec)

    _rec, frame, xs_all, ys_all, geom_names, support_all, soz_all = _load_geometry(ds_sid)
    raw_names = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
    raw_index = {n: i for i, n in enumerate(raw_names)}
    keep = np.array([
        (n in raw_index) and np.isfinite(xs_all[i]) and np.isfinite(ys_all[i])
        for i, n in enumerate(geom_names)
    ])
    if keep.sum() < 3:
        raise RuntimeError(f"{ds_sid}: only {keep.sum()} geometry contacts matched to ictal channels")
    idx = np.array([raw_index[n] for n in np.asarray(geom_names)[keep]])
    z_sel = z[idx]
    finite_row = np.isfinite(z_sel).any(axis=1)
    xs = xs_all[keep][finite_row]
    ys = ys_all[keep][finite_row]
    names = list(np.asarray(geom_names)[keep][finite_row])
    support = support_all[keep][finite_row]
    soz = soz_all[keep][finite_row]
    z_sel = z_sel[finite_row]

    hard_stop = offset - args.smooth_sec        # never window past the last full window inside the seizure
    stop = hard_stop if args.stop_sec is None else min(float(args.stop_sec), hard_stop)
    if stop <= args.start_sec:
        raise RuntimeError(f"{ds_sid}: empty movie window start={args.start_sec} stop={stop}")
    starts = np.arange(args.start_sec, stop + 1e-9, args.frame_step_sec)
    if starts.size == 0 or abs(float(starts[-1]) - float(stop)) > 1e-6:
        starts = np.append(starts, stop)
    values = _window_values(z_sel, relt, starts, args.smooth_sec)

    if args.zlim is None:
        sel = values
        zlim_from = getattr(args, "zlim_from_sec", None)
        if zlim_from is not None:
            # Scale the colorbar to a time sub-window (e.g. the onset transition), so a late
            # high-energy plateau does not dictate the range and crush the early-rise detail.
            centers = np.asarray(starts, float) + args.smooth_sec / 2.0
            if args.anchor == "eeg" and eeg_rel is not None:
                centers = centers - float(eeg_rel)
            fmask = (centers >= float(zlim_from[0])) & (centers <= float(zlim_from[1]))
            if fmask.any():
                sel = values[fmask]
        selfin = np.isfinite(sel)
        zlim = float(np.nanpercentile(np.abs(sel[selfin]), args.zlim_percentile)) if selfin.any() else 1.0
        zlim = max(zlim, args.zlim_min)
    else:
        zlim = float(args.zlim)
    zlim = float(np.ceil(zlim * 10.0) / 10.0)

    return {
        "ds_sid": ds_sid,
        "dataset": dataset,
        "seizure_idx": seizure_idx,
        "seizure_id": sw.seizure_id,
        "fs": float(sw.fs),
        "starts": np.asarray(starts, float),
        "values": values,
        "xs": xs,
        "ys": ys,
        "support": support,
        "soz": soz,
        "names": names,
        "frame": frame,
        "zlim": zlim,
        "eeg_rel": (None if eeg_rel is None else float(eeg_rel)),
        "offset": float(offset),
        "bl_start_sec": float(bl.start_sec),
        "bl_end_sec": float(bl.end_sec),
        "bl_valid_sec": float(bl.valid_sec),
    }


def _frame_center_rel(bundle: dict, args: argparse.Namespace, k: int) -> float:
    """Window-center time of frame k, relative to the chosen anchor (EEG onset when
    --anchor eeg and it exists, else clinical onset)."""
    center = float(bundle["starts"][k]) + args.smooth_sec / 2.0
    if args.anchor == "eeg" and bundle["eeg_rel"] is not None:
        return center - float(bundle["eeg_rel"])
    return center


def _out_path(args: argparse.Namespace, bundle: dict, kind: str, suffix: str) -> Path:
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_stem:
        stem = args.out_stem + ("_montage" if kind == "montage" else "")
    elif kind == "gif":
        stem = (f"{bundle['ds_sid']}_sz{bundle['seizure_idx']}_signed_broadband_"
                f"{args.band_lo:g}_{args.band_hi:g}Hz_m120_to_offset")
    else:
        stem = (f"{bundle['ds_sid']}_sz{bundle['seizure_idx']}_signed_"
                f"{args.band_lo:g}_{args.band_hi:g}Hz_montage")
    return out_dir / f"{stem}{suffix}"


def _sidecar_payload(bundle: dict, args: argparse.Namespace, *, kind: str, extra: dict | None = None) -> dict:
    starts = bundle["starts"]
    anchor = "eeg_onset" if (args.anchor == "eeg" and bundle["eeg_rel"] is not None) else "clinical_onset"
    payload = {
        "subject": bundle["ds_sid"],
        "dataset": bundle["dataset"],
        "seizure_idx": bundle["seizure_idx"],
        "seizure_id": bundle["seizure_id"],
        "fs": bundle["fs"],
        "band_hz": [float(args.band_lo), float(args.band_hi)],
        "feature": ("log summed PSD band power, per-channel baseline robust-z; closed [lo,hi] sum, "
                    "notch 50/100/150/200 Hz at load, no FFT-bin line mask (Fig3-B 1-150 contract)"),
        "kind": kind,
        "display": {
            "signed_colormap": "bwr",
            "two_slope_center": 0.0,
            "zlim": bundle["zlim"],
            "zlim_percentile": None if args.zlim is not None else float(args.zlim_percentile),
            "zlim_from_sec": (None if getattr(args, "zlim_from_sec", None) is None
                              else [float(args.zlim_from_sec[0]), float(args.zlim_from_sec[1])]),
            "smooth_window_sec": float(args.smooth_sec),
            "frame_step_sec": float(args.frame_step_sec),
            "time_anchor": anchor,
            "eeg_onset_rel_clinical_sec": bundle["eeg_rel"],
            "window_start_stop_rel_clinical_sec": [float(starts[0]), float(starts[-1] + args.smooth_sec)],
            "n_frames": int(len(starts)),
            "frame_center_rel_anchor_sec": [round(_frame_center_rel(bundle, args, k), 3) for k in range(len(starts))],
        },
        "spectrogram": {"win_sec": float(args.spectral_win_sec), "hop_sec": float(args.hop_sec)},
        "baseline": {
            "start_sec": bundle["bl_start_sec"],
            "end_sec": bundle["bl_end_sec"],
            "valid_sec": bundle["bl_valid_sec"],
            "guard_sec": float(GUARD_SEC),
        },
        "contacts": {"n_matched": int(len(bundle["names"])), "names": bundle["names"]},
        "source": {
            "script": str(Path(__file__).relative_to(_ROOT)),
            "geometry": str((REAL_DIR / f"{bundle['ds_sid']}_t_a.json").relative_to(_ROOT)),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def render_gif(bundle: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    starts = bundle["starts"]
    values = bundle["values"]
    xs, ys, support, soz, names = bundle["xs"], bundle["ys"], bundle["support"], bundle["soz"], bundle["names"]
    frame = bundle["frame"]
    zlim = bundle["zlim"]
    offset = bundle["offset"]

    norm = TwoSlopeNorm(vmin=-zlim, vcenter=0.0, vmax=zlim)
    cmap = plt.get_cmap("bwr")
    first_field = _smooth(xs, ys, values[0], support, frame)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 6.6), layout="constrained")
    im = ax.imshow(
        first_field,
        origin="lower",
        extent=[frame["xlim"][0], frame["xlim"][1], frame["ylim"][0], frame["ylim"][1]],
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    sc = ax.scatter(
        xs,
        ys,
        c=values[0],
        cmap=cmap,
        norm=norm,
        s=70,
        edgecolors=["black" if z0 else "white" for z0 in soz],
        linewidths=[1.6 if z0 else 0.65 for z0 in soz],
        zorder=3,
    )
    ax.set_xlim(*frame["xlim"])
    ax.set_ylim(*frame["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("along propagation axis (mm)")
    ax.set_ylabel("transverse (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"signed robust-z log power, {args.band_lo:g}-{args.band_hi:g} Hz (clip +/-{zlim:g})")

    pretty = bundle["ds_sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(
        f"{pretty} seizure {bundle['seizure_idx']}: signed broadband field, {args.band_lo:g}-{args.band_hi:g} Hz\n"
        f"baseline {bundle['bl_start_sec']:.0f} to {bundle['bl_end_sec']:.0f}s; blue < baseline, red > baseline",
        fontsize=12,
    )

    anchor_lbl = "EEG onset" if (args.anchor == "eeg" and bundle["eeg_rel"] is not None) else "clinical onset"

    def update(k: int):
        lo = float(starts[k])
        hi = lo + args.smooth_sec
        vals = values[k]
        field = _smooth(xs, ys, vals, support, frame)
        im.set_data(field)
        sc.set_array(np.asarray(vals))
        phase = "PRE" if hi < 0 else ("ICTAL" if lo <= offset else "POST")
        tc = _frame_center_rel(bundle, args, k)
        ax.set_title(
            f"{phase}  window center {tc:+.1f} s rel {anchor_lbl}  (n={len(names)} contacts)",
            fontsize=10.5,
        )
        return im, sc

    update(0)
    anim = FuncAnimation(fig, update, frames=len(starts), interval=1000.0 / args.fps, blit=False)
    gif = _out_path(args, bundle, "gif", ".gif")
    anim.save(gif, writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    sidecar = gif.with_suffix(".json")
    sidecar.write_text(json.dumps(_sidecar_payload(bundle, args, kind="gif"), indent=2, ensure_ascii=False) + "\n")
    print(gif)
    print(sidecar)
    return gif, sidecar


def render_montage(bundle: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    starts = bundle["starts"]
    values = bundle["values"]
    xs, ys, support, soz, names = bundle["xs"], bundle["ys"], bundle["support"], bundle["soz"], bundle["names"]
    frame = bundle["frame"]
    zlim = bundle["zlim"]
    eeg_rel = bundle["eeg_rel"]

    n = len(starts)
    cols = int(args.montage_cols)
    rows = int(np.ceil(n / cols))
    norm = TwoSlopeNorm(vmin=-zlim, vcenter=0.0, vmax=zlim)
    cmap = plt.get_cmap("bwr")
    extent = [frame["xlim"][0], frame["xlim"][1], frame["ylim"][0], frame["ylim"][1]]

    centers_clin = np.asarray(starts, float) + args.smooth_sec / 2.0
    k_eeg = int(np.argmin(np.abs(centers_clin - eeg_rel))) if eeg_rel is not None else -1
    k_clin = int(np.argmin(np.abs(centers_clin - 0.0)))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6 + 1.6, rows * 2.6 + 1.6),
                             squeeze=False, layout="constrained")
    im = None
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        if k >= n:
            ax.axis("off")
            continue
        field = _smooth(xs, ys, values[k], support, frame)
        im = ax.imshow(field, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm)
        ax.scatter(
            xs, ys, c=values[k], cmap=cmap, norm=norm, s=24,
            edgecolors=["black" if z0 else "white" for z0 in soz],
            linewidths=[1.1 if z0 else 0.4 for z0 in soz], zorder=3,
        )
        ax.set_xlim(*frame["xlim"])
        ax.set_ylim(*frame["ylim"])
        ax.set_xticks([])
        ax.set_yticks([])
        tc = _frame_center_rel(bundle, args, k)
        title, color, weight, lw, edge = f"{tc:+.1f} s", "black", "normal", 0.4, None
        if k == k_eeg:
            title, color, weight, lw, edge = f"{tc:+.1f} s · EEG onset", "#B26A00", "bold", 3.0, "#E8A33D"
        elif k == k_clin:
            title, color, weight, lw, edge = f"{tc:+.1f} s · clinical", "#333333", "normal", 2.0, "#666666"
        ax.set_title(title, fontsize=9.5, pad=2.5, color=color, fontweight=weight)
        for s in ax.spines.values():
            s.set_linewidth(lw)
            if edge is not None:
                s.set_color(edge)

    anchor_lbl = "EEG onset" if (args.anchor == "eeg" and eeg_rel is not None) else "clinical onset"
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    cb.set_label(f"signed robust-z log power, {args.band_lo:g}-{args.band_hi:g} Hz (clip +/-{zlim:g})", fontsize=10)

    pretty = bundle["ds_sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(
        f"{pretty} seizure {bundle['seizure_idx']} ({bundle['seizure_id']}) · "
        f"signed {args.band_lo:g}-{args.band_hi:g} Hz energy field over early-ictal rise\n"
        f"panel = {args.smooth_sec:g}s window (center rel {anchor_lbl}); "
        f"red > baseline, blue < baseline; black-edge dot = SOZ; n={len(names)}",
        fontsize=10.5,
    )
    fig.supxlabel(
        f"baseline {bundle['bl_start_sec']:.0f}..{bundle['bl_end_sec']:.0f}s · "
        f"effective time resolution ~{args.spectral_win_sec:g}s (spectrogram window) · "
        f"hop {args.hop_sec:g}s · frame step {args.frame_step_sec:g}s · "
        f"gold border = EEG onset, gray = clinical onset",
        fontsize=8.5, color="#555555",
    )

    png = _out_path(args, bundle, "montage", ".png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    sidecar = png.with_suffix(".json")
    sidecar.write_text(json.dumps(
        _sidecar_payload(bundle, args, kind="montage",
                         extra={"montage": {"rows": rows, "cols": cols,
                                            "eeg_onset_frame": k_eeg, "clinical_onset_frame": k_clin}}),
        indent=2, ensure_ascii=False) + "\n")
    print(png)
    print(sidecar)
    return png, sidecar


def plot_movie(args: argparse.Namespace) -> tuple[dict, list]:
    bundle = compute_field_frames(args)
    outputs: list[tuple[Path, Path]] = []
    if args.emit in ("gif", "both"):
        outputs.append(render_gif(bundle, args))
    if args.emit in ("montage", "both"):
        outputs.append(render_montage(bundle, args))
    return bundle, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--seizure-idx", type=int, default=None,
                    help="default: longest eligible seizure in ictal_field_long_cache")
    ap.add_argument("--start-sec", type=float, default=-120.0)
    ap.add_argument("--stop-sec", type=float, default=None,
                    help="last window START rel clinical onset (default: seizure offset). Zooms the tail, e.g. early rise.")
    ap.add_argument("--band-lo", type=float, default=1.0)
    ap.add_argument("--band-hi", type=float, default=150.0)
    ap.add_argument("--spectral-win-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--smooth-sec", type=float, default=5.0)
    ap.add_argument("--frame-step-sec", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--zlim", type=float, default=None)
    ap.add_argument("--zlim-percentile", type=float, default=98.0)
    ap.add_argument("--zlim-min", type=float, default=3.0)
    ap.add_argument("--zlim-from-sec", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="auto-zlim from frames whose center (rel anchor) is in [LO,HI], e.g. -1 2 "
                         "to scale the colorbar to the onset transition, not the late plateau")
    ap.add_argument("--chunk-ch", type=int, default=16)
    ap.add_argument("--anchor", choices=("clinical", "eeg"), default="clinical",
                    help="label frame times relative to clinical onset (default) or EEG onset")
    ap.add_argument("--emit", choices=("gif", "montage", "both"), default="gif",
                    help="which output(s) to render (default gif = legacy behavior)")
    ap.add_argument("--montage-cols", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--out-stem", type=str, default=None)
    args = ap.parse_args()
    plot_movie(args)


if __name__ == "__main__":
    main()
