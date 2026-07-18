#!/usr/bin/env python3
"""Subject-level time course of shared-gradient 1-150 Hz field similarity.

For each eligible seizure, recompute 1-150 Hz per-channel baseline robust-z,
slice non-overlapping time windows, and compare the signed energy field to
the subject's interictal template A/B fields with maxAB |corr|.

Default target: E1146, 10 s windows from -120 s to seizure offset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _compute_shared_values,
    _nan,
    _shared_geometry_metadata,
    _shared_scorer,
)
from scripts.run_topic5_t0_eligibility import _inventory_rows  # noqa: E402


T0_CACHE = _ROOT / "results/topic5_ictal_recruitment/t0_feature_cache"
LONG_CACHE = _ROOT / "results/topic5_ictal_recruitment/ictal_field_long_cache"
OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
FIG = OUT / "figures"


def _eligibility_status(ds_sid: str) -> dict:
    dataset, sid = ds_sid.split("_", 1)
    inv_rows, _onset_field = _inventory_rows(dataset, sid)
    cache_files = []
    for root in (T0_CACHE, LONG_CACHE):
        fp = root / f"{ds_sid}.json"
        if fp.exists():
            cache_files.append(fp)
            meta = json.loads(fp.read_text())
            idxs = [int(x) for x in meta.get("eligible_idxs", [])]
            if idxs:
                return {
                    "status": "eligible_cache_ready",
                    "inventory_n": len(inv_rows),
                    "cache_path": str(fp.relative_to(_ROOT)),
                    "eligible_idxs": idxs,
                    "reason_code": None,
                }
    if cache_files:
        reason = "blocked_input:derived_cache_has_no_eligible_seizures"
    elif inv_rows:
        reason = "blocked_input:eligible_derived_cache_missing"
    else:
        reason = "blocked_input:seizure_inventory_missing"
    return {
        "status": "blocked_input",
        "inventory_n": len(inv_rows),
        "cache_path": None,
        "eligible_idxs": [],
        "reason_code": reason,
    }


def _eligible_idxs(ds_sid: str) -> list[int]:
    status = _eligibility_status(ds_sid)
    if status["eligible_idxs"]:
        return status["eligible_idxs"]
    raise FileNotFoundError(
        f"{ds_sid}: {status['reason_code']} (inventory_n={status['inventory_n']})"
    )


def _coverage_status(n_requested: int, n_processed: int) -> str:
    if n_processed <= 0:
        return "blocked_input"
    if n_processed == n_requested:
        return "complete_ok"
    if n_processed / n_requested < 0.5:
        return "severely_partial"
    return "partial_ok"


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=path.suffix,
        delete=False,
    )
    handle.close()
    return Path(handle.name)


def _atomic_dataframe(df: pd.DataFrame, path: Path) -> None:
    tmp = _temporary_sibling(path)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    tmp = _temporary_sibling(path)
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _score_row(ds_sid: str, seizure_idx: int, lo: float, hi: float, offset: float,
               vals: np.ndarray, score) -> dict:
    per_template, best = score(vals)
    row = {
        "subject": ds_sid,
        "seizure_idx": int(seizure_idx),
        "window_start_sec": float(lo),
        "window_end_sec": float(hi),
        "window_center_sec": float((lo + hi) / 2.0),
        "phase": "pre" if hi <= 0 else ("ictal" if lo < offset else "post"),
        "best_template": best,
    }
    for key in ("A", "B"):
        r = per_template.get(key, {})
        row[f"{key}_signed_corr"] = _nan(r.get("signed_corr"))
        row[f"{key}_abs_corr"] = _nan(r.get("abs_corr"))
        row[f"{key}_mirror_choice"] = r.get("mirror_choice")
    row["maxAB_abs_corr"] = max(row["A_abs_corr"], row["B_abs_corr"])
    row["maxAB_signed_corr"] = row[f"{best}_signed_corr"] if best in ("A", "B") else float("nan")
    return row


def _on_common_grid(lo: float, *, start_sec: float, step_sec: float, atol: float = 1e-6) -> bool:
    pos = (float(lo) - float(start_sec)) / float(step_sec)
    return abs(pos - round(pos)) <= atol


def _sec_tag(x: float) -> str:
    prefix = "m" if float(x) < 0 else "p"
    return f"{prefix}{abs(float(x)):g}".replace(".", "p")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lo, hi, cen), g in df.groupby(["window_start_sec", "window_end_sec", "window_center_sec"], sort=True):
        v = pd.to_numeric(g["maxAB_abs_corr"], errors="coerce").dropna().to_numpy(float)
        signed = pd.to_numeric(g["maxAB_signed_corr"], errors="coerce").dropna().to_numpy(float)
        rows.append({
            "window_start_sec": float(lo),
            "window_end_sec": float(hi),
            "window_center_sec": float(cen),
            "n_seizures": int(v.size),
            "mean_maxAB_abs_corr": float(np.mean(v)) if v.size else np.nan,
            "median_maxAB_abs_corr": float(np.median(v)) if v.size else np.nan,
            "sd_maxAB_abs_corr": float(np.std(v, ddof=1)) if v.size >= 2 else np.nan,
            "var_maxAB_abs_corr": float(np.var(v, ddof=1)) if v.size >= 2 else np.nan,
            "q25_maxAB_abs_corr": float(np.percentile(v, 25)) if v.size else np.nan,
            "q75_maxAB_abs_corr": float(np.percentile(v, 75)) if v.size else np.nan,
            "mean_maxAB_signed_corr": float(np.mean(signed)) if signed.size else np.nan,
            "median_maxAB_signed_corr": float(np.median(signed)) if signed.size else np.nan,
            "sd_maxAB_signed_corr": float(np.std(signed, ddof=1)) if signed.size >= 2 else np.nan,
            "var_maxAB_signed_corr": float(np.var(signed, ddof=1)) if signed.size >= 2 else np.nan,
        })
    return pd.DataFrame(rows)


def _plot(ds_sid: str, per: pd.DataFrame, agg: pd.DataFrame, out_png: Path, *, band_lo: float, band_hi: float,
          window_sec: float, step_sec: float) -> None:
    fig, (ax, axv) = plt.subplots(
        2, 1, figsize=(12.4, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        layout="constrained",
    )
    for _idx, g in per.groupby("seizure_idx"):
        g = g.sort_values("window_center_sec")
        ax.plot(g["window_center_sec"], g["maxAB_abs_corr"], color="0.72", lw=0.75, alpha=0.55)

    x = agg["window_center_sec"].to_numpy(float)
    med = agg["median_maxAB_abs_corr"].to_numpy(float)
    mean = agg["mean_maxAB_abs_corr"].to_numpy(float)
    sd = agg["sd_maxAB_abs_corr"].to_numpy(float)
    n = agg["n_seizures"].to_numpy(int)
    band = n >= 2
    ax.fill_between(x[band], np.clip(mean[band] - sd[band], 0, 1), np.clip(mean[band] + sd[band], 0, 1),
                    color="#d7301f", alpha=0.16, linewidth=0, label="mean +/- 1 SD")
    ax.plot(x, med, color="black", lw=2.2, label="median across seizures")
    ax.plot(x, mean, color="#d7301f", lw=1.5, label="mean across seizures")
    ax.axvline(0, color="0.15", ls="--", lw=1.0)
    ax.set_ylabel("maxAB field similarity |r|")
    ax.set_ylim(0, 1.02)
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    var = agg["var_maxAB_abs_corr"].to_numpy(float)
    axv.plot(x, var, color="#756bb1", lw=1.7, label="variance of |r|")
    axv.scatter(x, var, color="#756bb1", s=14, zorder=3)
    axv.set_ylabel("variance")
    axv.grid(True, color="0.9", lw=0.6)
    axn = axv.twinx()
    axn.step(x, n, where="mid", color="0.25", lw=1.0, alpha=0.75, label="n seizures")
    axn.set_ylabel("n seizures")
    axn.set_ylim(0, max(1, int(np.nanmax(n))) + 1)
    axv.axvline(0, color="0.15", ls="--", lw=1.0)
    axv.set_xlabel("time from clinical onset (s), window center")

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(
        f"{pretty}: 1-150 Hz signed robust-z field similarity to interictal templates\n"
        f"{window_sec:g}s windows, {step_sec:g}s step; gray=individual seizures; shaded=+/-1 SD",
        fontsize=13,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    ds_sid = args.subject
    out_dir_value = getattr(args, "out_dir", None)
    out_dir = Path(out_dir_value).resolve() if out_dir_value else OUT
    fig_dir = out_dir / "figures"
    eligibility = _eligibility_status(ds_sid)
    idxs = list(eligibility["eligible_idxs"])
    if not idxs:
        raise FileNotFoundError(
            f"{ds_sid}: {eligibility['reason_code']} "
            f"(inventory_n={eligibility['inventory_n']})"
        )
    rows = []
    drops = []
    for k, seizure_idx in enumerate(idxs, start=1):
        print(f"[{ds_sid}] seizure {seizure_idx} ({k}/{len(idxs)})", flush=True)
        per_args = SimpleNamespace(**vars(args))
        per_args.seizure_idx = int(seizure_idx)
        per_args.smooth_sec = float(args.window_sec)
        per_args.frame_step_sec = float(args.step_sec)
        try:
            _ds_sid, _idx, sw, offset, bl, field_record, names, starts, window_vals, _onset_vals = _compute_shared_values(per_args)
            score = _shared_scorer(ds_sid, names)
            for lo, vals in zip(starts, window_vals):
                if not _on_common_grid(lo, start_sec=args.start_sec, step_sec=args.step_sec):
                    # `_compute_values` appends an offset-aligned final window per seizure.
                    # That is useful for single-seizure movies, but cross-seizure variance
                    # requires common time bins; otherwise each seizure contributes a private
                    # n=1 endpoint.
                    continue
                row = _score_row(
                    ds_sid, seizure_idx, float(lo), float(lo + args.window_sec),
                    float(offset), vals, score,
                )
                row.update({
                    "field_plane": "shared",
                    "field_scorers": "shared_a,shared_b",
                    "field_contract": field_record["contract"],
                    "field_fingerprint_sha256": field_record["interictal_field"]["fingerprint_sha256"],
                    "axis_definition": field_record["axis_definition"],
                    "axis_direction_convention": field_record["axis_direction_convention"],
                    "own_field_fallback": False,
                    **_shared_geometry_metadata(field_record),
                })
                rows.append(row)
        except Exception as exc:
            drops.append({"seizure_idx": int(seizure_idx), "reason": f"{type(exc).__name__}: {exc}"})
            print(f"  drop: {drops[-1]['reason']}", flush=True)
    if not rows:
        raise RuntimeError(f"{ds_sid}: no seizure produced timecourse rows")

    per = pd.DataFrame(rows)
    agg = _aggregate(per)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if abs(float(args.step_sec) - float(args.window_sec)) <= 1e-9:
        window_tag = f"{args.window_sec:g}s"
    else:
        window_tag = f"{args.window_sec:g}s_step{args.step_sec:g}s"
    range_tag = ""
    if args.stop_sec is not None:
        range_tag = f"_{_sec_tag(args.start_sec)}_{_sec_tag(args.stop_sec)}"
    stem = (
        f"{ds_sid}_signed_broadband_{args.band_lo:g}_{args.band_hi:g}Hz_"
        f"similarity_timecourse{range_tag}_{window_tag}"
    )
    per_csv = out_dir / f"{stem}_per_seizure.csv"
    agg_csv = out_dir / f"{stem}_aggregate.csv"
    png = fig_dir / f"{stem}.png"
    summary_json = out_dir / f"{stem}_summary.json"
    _atomic_dataframe(per, per_csv)
    _atomic_dataframe(agg, agg_csv)
    tmp_png = _temporary_sibling(png)
    try:
        _plot(ds_sid, per, agg, tmp_png, band_lo=args.band_lo, band_hi=args.band_hi,
              window_sec=args.window_sec, step_sec=args.step_sec)
        os.replace(tmp_png, png)
    finally:
        tmp_png.unlink(missing_ok=True)

    early = agg[(agg["window_start_sec"] >= 0) & (agg["window_end_sec"] <= 30)]
    pre = agg[(agg["window_start_sec"] >= -120) & (agg["window_end_sec"] <= 0)]
    n_processed = int(per["seizure_idx"].nunique())
    coverage_status = _coverage_status(len(idxs), n_processed)
    summary = {
        "subject": ds_sid,
        "band_hz": [float(args.band_lo), float(args.band_hi)],
        "window_sec": float(args.window_sec),
        "step_sec": float(args.step_sec),
        "feature": "1-150 Hz log power, per-channel baseline robust-z; signed values, shared-gradient maxAB |r| similarity",
        "field_contract": str(per["field_contract"].iloc[0]),
        "field_plane": "shared",
        "field_scorers": ["shared_a", "shared_b"],
        "own_field_fallback": False,
        "field_fingerprint_sha256": str(per["field_fingerprint_sha256"].iloc[0]),
        "axis_definition": str(per["axis_definition"].iloc[0]),
        "axis_direction_convention": str(per["axis_direction_convention"].iloc[0]),
        "geometry_2d_supported": True,
        "geometry_quality_tier": str(per["geometry_quality_tier"].iloc[0]),
        "minimum_axis_n_shafts": int(per["minimum_axis_n_shafts"].iloc[0]),
        "minimum_axis_effective_rank": int(per["minimum_axis_effective_rank"].iloc[0]),
        "input_eligibility": eligibility,
        "coverage_status": coverage_status,
        "coverage_fraction": float(n_processed / len(idxs)),
        "n_eligible_requested": len(idxs),
        "n_seizures_processed": n_processed,
        "drops": drops,
        "outputs": {
            "figure": str(png.relative_to(_ROOT)),
            "per_seizure_csv": str(per_csv.relative_to(_ROOT)),
            "aggregate_csv": str(agg_csv.relative_to(_ROOT)),
        },
        "overall": {
            "median_of_aggregate_median": float(np.nanmedian(agg["median_maxAB_abs_corr"])),
            "median_of_aggregate_variance": float(np.nanmedian(agg["var_maxAB_abs_corr"])),
            "max_aggregate_median": float(np.nanmax(agg["median_maxAB_abs_corr"])),
        },
        "pre_m120_0": {
            "median_of_aggregate_median": float(np.nanmedian(pre["median_maxAB_abs_corr"])) if len(pre) else None,
            "median_of_aggregate_variance": float(np.nanmedian(pre["var_maxAB_abs_corr"])) if len(pre) else None,
        },
        "early_0_30": {
            "median_of_aggregate_median": float(np.nanmedian(early["median_maxAB_abs_corr"])) if len(early) else None,
            "median_of_aggregate_variance": float(np.nanmedian(early["var_maxAB_abs_corr"])) if len(early) else None,
        },
    }
    _atomic_json(summary, summary_json)
    print(png)
    print(per_csv)
    print(agg_csv)
    print(summary_json)
    return png, per_csv, agg_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--start-sec", type=float, default=-120.0)
    ap.add_argument("--stop-sec", type=float, default=None)
    ap.add_argument("--band-lo", type=float, default=1.0)
    ap.add_argument("--band-hi", type=float, default=150.0)
    ap.add_argument("--spectral-win-sec", type=float, default=1.0)
    ap.add_argument("--hop-sec", type=float, default=0.5)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--step-sec", type=float, default=10.0)
    ap.add_argument("--onset-win-sec", type=float, default=10.0)
    ap.add_argument("--chunk-ch", type=int, default=16)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="optional run-scoped output directory; defaults to canonical field_dynamics_signed",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
