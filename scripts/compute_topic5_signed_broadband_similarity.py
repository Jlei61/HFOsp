#!/usr/bin/env python3
"""Compute shared-gradient field similarity for signed 1-150 Hz broadband data.

This is a diagnostic sidecar for `plot_topic5_signed_broadband_movie.py`.
It uses the same signed robust-z per-contact values as the GIF, but scores them
only against the fingerprint-checked frozen ``shared_a/shared_b`` gradient
fields.  Missing shared fields fail closed; this producer never falls back to
``own_a/own_b``.
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
from src.topic5_template_axis_field import (  # noqa: E402
    interictal_field_quality_tier,
    score_field,
    scorers_from_interictal_record,
)


OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
FROZEN_FIELD_DIR = (
    _ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)


def _nan(v):
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _load_frozen_shared(ds_sid: str) -> tuple[dict, dict]:
    """Load a fingerprint-valid, two-dimensional shared A/B field record.

    Shared field keys alone are not an analysis-eligibility contract.  The
    canonical frozen record must also identify the requested subject and pass
    the explicit two-dimensional geometry gate (at least two shafts and
    effective rank >= 2 for both template axes).  One-dimensional records are
    retained upstream as directional sensitivities but are forbidden here.
    """
    fp = FROZEN_FIELD_DIR / f"{ds_sid}.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    record = json.loads(fp.read_text())

    dataset, subject = ds_sid.split("_", 1)
    record_ds_sid = f"{record.get('dataset')}_{record.get('subject')}"
    if record_ds_sid != ds_sid or record.get("dataset") != dataset or str(record.get("subject")) != subject:
        raise ValueError(
            f"{ds_sid}: frozen_subject_identity_mismatch ({record_ds_sid})"
        )

    all_scorers = scorers_from_interictal_record(record)
    shared = {
        key: all_scorers[key]
        for key in ("shared_a", "shared_b")
        if key in all_scorers
    }
    if set(shared) != {"shared_a", "shared_b"}:
        raise ValueError(f"{ds_sid}: missing_shared_a_or_shared_b_field")
    if any(key.startswith("own_") for key in shared):
        raise AssertionError("own field leaked into shared-only scorer set")

    pair = record.get("axis_pair") or {}
    axes = [pair.get("axis_a") or {}, pair.get("axis_b") or {}]
    n_shafts = [axis.get("n_shafts") for axis in axes]
    effective_ranks = [axis.get("effective_rank") for axis in axes]
    geometry_ok = (
        pair.get("geometry_2d_supported") is True
        and all(isinstance(v, (int, float)) and int(v) >= 2 for v in n_shafts)
        and all(isinstance(v, (int, float)) and int(v) >= 2 for v in effective_ranks)
    )
    if not geometry_ok:
        raise ValueError(
            f"{ds_sid}: geometry_2d_unsupported "
            f"(flag={pair.get('geometry_2d_supported')}, "
            f"n_shafts={n_shafts}, effective_rank={effective_ranks})"
        )
    return record, shared


def _shared_geometry_metadata(record: dict) -> dict:
    """Return provenance fields already validated by ``_load_frozen_shared``."""
    pair = record["axis_pair"]
    axes = [pair["axis_a"], pair["axis_b"]]
    return {
        "geometry_2d_supported": True,
        "geometry_quality_tier": interictal_field_quality_tier(record),
        "minimum_axis_n_shafts": int(min(axis["n_shafts"] for axis in axes)),
        "minimum_axis_effective_rank": int(
            min(axis["effective_rank"] for axis in axes)
        ),
    }


def _load_axis(ds_sid: str, template: str) -> dict | None:
    """Legacy axis loader retained for non-Fig3-B diagnostic consumers."""
    fp = REAL_DIR / f"{ds_sid}_{template}.json"
    if not fp.exists():
        return None
    rec = json.loads(fp.read_text())
    return rec if rec.get("channels") else None


def _shared_window_extent(
    *, offset: float, stop_sec: float | None, smooth_sec: float
) -> tuple[float, float]:
    """Return analysis stop and loaded post-onset extent for shared trajectories.

    Without an explicit stop the single-seizure diagnostic ends at seizure
    offset.  With ``--stop-sec`` the requested onset-relative endpoint is the
    contract, even when the seizure ends earlier; extra post data protect the
    notch-filtered trace from the loaded-window boundary.
    """
    if stop_sec is None:
        return float(offset), float(offset) + 0.5
    stop_at = float(stop_sec)
    post_pad = max(30.0, float(smooth_sec) + 0.5)
    return stop_at, stop_at + post_pad


def _compute_shared_values(args: argparse.Namespace):
    """Extract window values in the frozen shared-field contact order."""
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
    stop_at, post_sec = _shared_window_extent(
        offset=offset,
        stop_sec=stop_sec,
        smooth_sec=args.smooth_sec,
    )
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

    field_record, _shared_scorers = _load_frozen_shared(ds_sid)
    target_names = [
        str(name) for name in field_record["interictal_field"]["contact_order"]
    ]
    raw_names = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
    if len(raw_names) != len(set(raw_names)):
        raise ValueError(f"{ds_sid}: raw_channel_aliases_not_unique")
    raw_index = {n: i for i, n in enumerate(raw_names)}
    names = [name for name in target_names if name in raw_index]
    raw_idx = np.array([raw_index[n] for n in names], int)
    z_sel = z[raw_idx]
    finite_row = np.isfinite(z_sel).any(axis=1)
    names = [name for name, ok in zip(names, finite_row) if ok]
    z_sel = z_sel[finite_row]
    if len(names) < 6:
        raise RuntimeError(f"{ds_sid}: insufficient matched contacts ({len(names)})")

    stop_start = stop_at - args.smooth_sec
    starts = np.arange(args.start_sec, stop_start + 1e-9, args.frame_step_sec)
    if starts.size == 0 or abs(float(starts[-1]) - float(stop_start)) > 1e-6:
        starts = np.append(starts, stop_start)
    window_vals = _window_values(z_sel, relt, starts, args.smooth_sec)
    onset_vals = _window_values(z_sel, relt, np.array([0.0]), args.onset_win_sec)[0]
    return ds_sid, seizure_idx, sw, offset, bl, field_record, names, starts, window_vals, onset_vals


def _shared_scorer(ds_sid: str, matched_names: list[str]):
    """Build a shared-only frozen-gradient scorer with no own-field fallback."""
    record, shared = _load_frozen_shared(ds_sid)
    target_names = [str(name) for name in record["interictal_field"]["contact_order"]]
    if len(matched_names) != len(set(matched_names)):
        raise ValueError(f"{ds_sid}: matched channel names are not unique")
    target_index = {name: idx for idx, name in enumerate(target_names)}
    missing = [name for name in matched_names if name not in target_index]
    if missing:
        raise ValueError(f"{ds_sid}: matched channels outside frozen contact order: {missing}")

    def score(vals):
        values = np.asarray(vals, float)
        if values.shape != (len(matched_names),):
            raise ValueError(
                f"{ds_sid}: activation shape {values.shape} != ({len(matched_names)},)"
            )
        aligned = np.full(len(target_names), np.nan, float)
        for name, value in zip(matched_names, values):
            aligned[target_index[name]] = value
        out = {}
        for label, key in (("A", "shared_a"), ("B", "shared_b")):
            result = score_field(shared[key], aligned)
            out[label] = {
                "signed_corr": result["signed_r"],
                "abs_corr": result["abs_r"],
                "mirror_choice": result["mirror_choice"],
            }
        abs_vals = [_nan(v.get("abs_corr")) for v in out.values()]
        if np.isfinite(abs_vals).any():
            best_i = int(np.nanargmax(abs_vals))
            best_key = list(out.keys())[best_i]
        else:
            best_key = None
        return out, best_key

    return score


def _compute_values(args: argparse.Namespace):
    """Legacy extraction contract retained for existing diagnostic consumers."""
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
    """Legacy own-plane scorer retained for existing non-Fig3-B analyses."""
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

    ds_sid, seizure_idx, sw, offset, bl, field_record, names, starts, window_vals, onset_vals = _compute_shared_values(args)
    score = _shared_scorer(ds_sid, names)

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
            "field_plane": "shared",
            "field_scorers": "shared_a,shared_b",
            "field_fingerprint_sha256": field_record["interictal_field"]["fingerprint_sha256"],
            **_shared_geometry_metadata(field_record),
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
        "metric": "frozen shared-gradient field score with identity/mirror reselection",
        "field_contract": field_record["contract"],
        "field_plane": "shared",
        "field_scorers": ["shared_a", "shared_b"],
        "own_field_fallback": False,
        "field_fingerprint_sha256": field_record["interictal_field"]["fingerprint_sha256"],
        "axis_definition": field_record["axis_definition"],
        "axis_direction_convention": field_record["axis_direction_convention"],
        **_shared_geometry_metadata(field_record),
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
