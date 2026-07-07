#!/usr/bin/env python3
"""Compute interictal-field similarity for the signed 1-150 Hz broadband movie.

This is a diagnostic sidecar for `plot_topic5_signed_broadband_movie.py`.
It uses the same signed robust-z per-contact values as the GIF, but computes
field correlations from the contact values on the formal normalized plane,
not from rendered GIF pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.plot_topic5_signed_broadband_movie import (  # noqa: E402
    REAL_DIR,
    _band_power_trace_chunked,
    _default_seizure_idx,
    _offset_rel,
    _pre_target,
    _window_values,
)
from scripts.run_topic5_t0_eligibility import (  # noqa: E402
    GUARD_SEC,
    ICTAL_REFERENCE,
    MIN_BASELINE_SEC,
    _inventory_rows,
)
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window  # noqa: E402
from src.propagation_contact_plane_readout import (  # noqa: E402
    OVERLAP_MIN,
    S_THRESH,
    R_smooth_rank,
    corr_pair_mirror_invariant_signed,
    make_plane_grid,
)
from src.topic5_axis_alignment import make_field_record, matched_channels  # noqa: E402


OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"


def _nan(v):
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _load_axis(ds_sid: str, template: str) -> dict | None:
    fp = REAL_DIR / f"{ds_sid}_{template}.json"
    if not fp.exists():
        return None
    rec = json.loads(fp.read_text())
    return rec if rec.get("channels") else None


def _compute_values(args: argparse.Namespace):
    ds_sid = args.subject
    dataset, sid = ds_sid.split("_", 1)
    seizure_idx = _default_seizure_idx(ds_sid) if args.seizure_idx is None else int(args.seizure_idx)
    inv_rows, _ = _inventory_rows(dataset, sid)
    if not (0 <= seizure_idx < len(inv_rows)):
        raise IndexError(f"{ds_sid}: seizure_idx={seizure_idx} out of range (n={len(inv_rows)})")
    inv = inv_rows[seizure_idx]
    offset = _offset_rel(dataset, inv)
    pre_sec = _pre_target(dataset, inv, display_start=args.start_sec)
    stop_sec = getattr(args, "stop_sec", None)
    if stop_sec is None:
        post_sec = offset + 0.5
    else:
        post_pad = max(30.0, float(args.smooth_sec) + 0.5)
        post_sec = min(offset + 0.5, float(stop_sec) + post_pad)
    sw = extract_seizure_window(
        f"{dataset}/{sid}",
        seizure_idx,
        pre_sec=pre_sec,
        post_sec=post_sec,
        reference=ICTAL_REFERENCE[dataset],
    )
    if sw.fs / 2.0 <= args.band_hi:
        raise RuntimeError(f"Nyquist {sw.fs / 2.0:g} <= requested band_hi {args.band_hi:g}")

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

    axis_a = _load_axis(ds_sid, "t_a")
    if axis_a is None:
        raise FileNotFoundError(REAL_DIR / f"{ds_sid}_t_a.json")
    raw_names = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
    raw_index = {n: i for i, n in enumerate(raw_names)}
    matched = matched_channels(axis_a, raw_index)
    names = [c["name"] for c in matched]
    raw_idx = np.array([raw_index[n] for n in names], int)
    z_sel = z[raw_idx]
    finite_row = np.isfinite(z_sel).any(axis=1)
    matched = [c for c, ok in zip(matched, finite_row) if ok]
    names = [c["name"] for c in matched]
    z_sel = z_sel[finite_row]
    if len(matched) < 6:
        raise RuntimeError(f"{ds_sid}: insufficient matched contacts ({len(matched)})")

    stop_at = offset if stop_sec is None else min(offset, float(stop_sec))
    stop_start = stop_at - args.smooth_sec
    starts = np.arange(args.start_sec, stop_start + 1e-9, args.frame_step_sec)
    if starts.size == 0 or abs(float(starts[-1]) - float(stop_start)) > 1e-6:
        starts = np.append(starts, stop_start)
    window_vals = _window_values(z_sel, relt, starts, args.smooth_sec)
    onset_vals = _window_values(z_sel, relt, np.array([0.0]), args.onset_win_sec)[0]
    return ds_sid, seizure_idx, sw, offset, bl, matched, names, starts, window_vals, onset_vals


def _scorer(ds_sid: str, matched: list[dict]):
    axis_a = _load_axis(ds_sid, "t_a")
    axis_b = _load_axis(ds_sid, "t_b")
    names = [c["name"] for c in matched]
    X, Y = make_plane_grid()
    rank_a = [float(c["typical_rank"]) for c in matched]
    F_a = R_smooth_rank(make_field_record(matched, rank_a), X, Y, None, S_THRESH)
    sigma = F_a["sigma_xy"]
    fields = {"A": F_a}
    if axis_b is not None:
        b_rank_by_name = {
            c["name"]: float(c["typical_rank"])
            for c in axis_b.get("channels", [])
            if np.isfinite(c.get("typical_rank", np.nan))
        }
        rank_b = [b_rank_by_name.get(n, np.nan) for n in names]
        if np.isfinite(rank_b).sum() >= 4:
            fields["B"] = R_smooth_rank(make_field_record(matched, rank_b), X, Y, sigma, S_THRESH)

    def score(vals):
        F_ict = R_smooth_rank(make_field_record(matched, vals), X, Y, sigma, S_THRESH)
        out = {}
        for key, F in fields.items():
            r = corr_pair_mirror_invariant_signed(
                F["T"], F["S"], F_ict["T"], F_ict["S"], S_THRESH, OVERLAP_MIN
            )
            out[key] = r
        abs_vals = [_nan(v.get("abs_corr")) for v in out.values()]
        signed_vals = [_nan(v.get("signed_corr")) for v in out.values()]
        if np.isfinite(abs_vals).any():
            best_i = int(np.nanargmax(abs_vals))
            best_key = list(out.keys())[best_i]
        else:
            best_key = None
        return out, best_key

    return score


def _summ(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0, "median": None, "mean": None, "max": None, "min": None}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "max": float(np.max(a)),
        "min": float(np.min(a)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--seizure-idx", type=int, default=None)
    ap.add_argument("--start-sec", type=float, default=-120.0)
    ap.add_argument("--band-lo", type=float, default=1.0)
    ap.add_argument("--band-hi", type=float, default=150.0)
    ap.add_argument("--spectral-win-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--smooth-sec", type=float, default=5.0)
    ap.add_argument("--frame-step-sec", type=float, default=2.0)
    ap.add_argument("--onset-win-sec", type=float, default=10.0)
    ap.add_argument("--chunk-ch", type=int, default=16)
    args = ap.parse_args()

    ds_sid, seizure_idx, sw, offset, bl, matched, names, starts, window_vals, onset_vals = _compute_values(args)
    score = _scorer(ds_sid, matched)

    rows = []
    for lo, vals in zip(starts, window_vals):
        per_template, best = score(vals)
        row = {
            "subject": ds_sid,
            "seizure_idx": seizure_idx,
            "window_start_sec": float(lo),
            "window_end_sec": float(lo + args.smooth_sec),
            "phase": "pre" if lo + args.smooth_sec < 0 else ("ictal" if lo <= offset else "post"),
            "n_contacts": len(names),
            "best_template": best,
        }
        for key in ("A", "B"):
            r = per_template.get(key, {})
            row[f"{key}_signed_corr"] = _nan(r.get("signed_corr"))
            row[f"{key}_abs_corr"] = _nan(r.get("abs_corr"))
            row[f"{key}_mirror_choice"] = r.get("mirror_choice")
        row["maxAB_abs_corr"] = max(row["A_abs_corr"], row["B_abs_corr"])
        row["maxAB_signed_corr"] = row[f"{best}_signed_corr"] if best in ("A", "B") else float("nan")
        rows.append(row)

    onset_per_template, onset_best = score(onset_vals)
    onset = {
        "window_sec": [0.0, float(args.onset_win_sec)],
        "best_template": onset_best,
        "A": onset_per_template.get("A"),
        "B": onset_per_template.get("B"),
        "maxAB_abs_corr": max(
            _nan(onset_per_template.get("A", {}).get("abs_corr")),
            _nan(onset_per_template.get("B", {}).get("abs_corr")),
        ),
        "maxAB_signed_corr": (
            _nan(onset_per_template[onset_best].get("signed_corr"))
            if onset_best in onset_per_template else float("nan")
        ),
    }

    def col(name, phase=None):
        return [float(r[name]) for r in rows
                if np.isfinite(float(r[name])) and (phase is None or r["phase"] == phase)]

    summary = {
        "subject": ds_sid,
        "seizure_idx": seizure_idx,
        "seizure_id": sw.seizure_id,
        "band_hz": [float(args.band_lo), float(args.band_hi)],
        "feature": "signed per-channel baseline robust-z log power",
        "metric": "corr_pair_mirror_invariant_signed on formal normalized contact plane",
        "rank01": False,
        "sign_flip": False,
        "contacts": {"n": len(names), "names": names},
        "movie_window_sec": [float(args.start_sec), float(offset)],
        "baseline": {
            "start_sec": float(bl.start_sec),
            "end_sec": float(bl.end_sec),
            "valid_sec": float(bl.valid_sec),
        },
        "onset_0_10": onset,
        "per_window": {
            "n_frames": len(rows),
            "A_abs": _summ(col("A_abs_corr")),
            "B_abs": _summ(col("B_abs_corr")),
            "maxAB_abs": _summ(col("maxAB_abs_corr")),
            "A_signed": _summ(col("A_signed_corr")),
            "B_signed": _summ(col("B_signed_corr")),
            "maxAB_signed": _summ(col("maxAB_signed_corr")),
            "pre_maxAB_abs": _summ(col("maxAB_abs_corr", "pre")),
            "ictal_maxAB_abs": _summ(col("maxAB_abs_corr", "ictal")),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{ds_sid}_sz{seizure_idx}_signed_broadband_{args.band_lo:g}_{args.band_hi:g}Hz_similarity"
    csv_f = OUT / f"{stem}_per_window.csv"
    json_f = OUT / f"{stem}_summary.json"
    with csv_f.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_f.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(csv_f)
    print(json_f)


if __name__ == "__main__":
    main()
