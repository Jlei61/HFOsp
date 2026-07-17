#!/usr/bin/env python3
"""Plot signed A/B field similarity for the pre-onset window.

Consumes the per-seizure CSV produced by
`plot_topic5_signed_broadband_similarity_timecourse.py` and makes a cleaner
diagnostic figure for the common preictal interval only. No maxAB and no abs:
template A and template B signed correlations are reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


BASE = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
FIG = BASE / "figures"


def _agg_signed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lo, hi, cen), g in df.groupby(["window_start_sec", "window_end_sec", "window_center_sec"], sort=True):
        row = {
            "window_start_sec": float(lo),
            "window_end_sec": float(hi),
            "window_center_sec": float(cen),
            "n_seizures": int(g["seizure_idx"].nunique()),
        }
        for tmpl in ("A", "B"):
            vals = pd.to_numeric(g[f"{tmpl}_signed_corr"], errors="coerce").dropna().to_numpy(float)
            row[f"{tmpl}_mean_signed_corr"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{tmpl}_median_signed_corr"] = float(np.median(vals)) if vals.size else np.nan
            row[f"{tmpl}_sd_signed_corr"] = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan
            row[f"{tmpl}_var_signed_corr"] = float(np.var(vals, ddof=1)) if vals.size >= 2 else np.nan
            row[f"{tmpl}_q25_signed_corr"] = float(np.percentile(vals, 25)) if vals.size else np.nan
            row[f"{tmpl}_q75_signed_corr"] = float(np.percentile(vals, 75)) if vals.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(ds_sid: str, per: pd.DataFrame, agg: pd.DataFrame, out_png: Path,
          *, band_lo: float, band_hi: float, lo_sec: float, hi_sec: float, window_sec: float) -> None:
    fig, (ax, axv) = plt.subplots(
        2, 1,
        figsize=(10.8, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        layout="constrained",
    )
    colors = {"A": "#b2182b", "B": "#2166ac"}
    for tmpl in ("A", "B"):
        for _idx, g in per.groupby("seizure_idx"):
            g = g.sort_values("window_center_sec")
            ax.plot(
                g["window_center_sec"],
                g[f"{tmpl}_signed_corr"],
                color=colors[tmpl],
                lw=0.55,
                alpha=0.12,
            )
        x = agg["window_center_sec"].to_numpy(float)
        mean = agg[f"{tmpl}_mean_signed_corr"].to_numpy(float)
        med = agg[f"{tmpl}_median_signed_corr"].to_numpy(float)
        sd = agg[f"{tmpl}_sd_signed_corr"].to_numpy(float)
        ok = np.isfinite(mean) & np.isfinite(sd)
        ax.fill_between(
            x[ok],
            np.clip(mean[ok] - sd[ok], -1, 1),
            np.clip(mean[ok] + sd[ok], -1, 1),
            color=colors[tmpl],
            alpha=0.11,
            linewidth=0,
        )
        ax.plot(x, med, color=colors[tmpl], lw=2.2, label=f"template {tmpl} median")
        ax.plot(x, mean, color=colors[tmpl], lw=1.1, ls="--", label=f"template {tmpl} mean")
        axv.plot(
            x,
            agg[f"{tmpl}_var_signed_corr"],
            color=colors[tmpl],
            lw=1.7,
            marker="o",
            ms=4,
            label=f"template {tmpl} variance",
        )

    ax.axhline(0, color="0.25", lw=0.9)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlim(lo_sec, hi_sec)
    ax.set_ylabel("signed field similarity r")
    ax.grid(True, color="0.9", lw=0.6)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=9)

    axv.axhline(0, color="0.35", lw=0.7)
    axv.set_ylabel("variance")
    axv.grid(True, color="0.9", lw=0.6)
    axn = axv.twinx()
    axn.step(
        agg["window_center_sec"],
        agg["n_seizures"],
        where="mid",
        color="0.25",
        lw=1.0,
        alpha=0.8,
        label="n seizures",
    )
    axn.set_ylabel("n seizures")
    axn.set_ylim(0, max(1, int(agg["n_seizures"].max())) + 1)
    axv.set_xlabel("time from clinical onset (s), window center")

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(
        f"{pretty}: signed A/B field similarity before onset, {band_lo:g}-{band_hi:g} Hz\n"
        f"{window_sec:g}s windows; no maxAB, no abs; shaded bands = mean +/- 1 SD",
        fontsize=13,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    ds_sid = args.subject
    src = BASE / (
        f"{ds_sid}_signed_broadband_{args.band_lo:g}_{args.band_hi:g}Hz_"
        f"similarity_timecourse_{args.window_sec:g}s_per_seizure.csv"
    )
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    keep = (
        (df["window_start_sec"] >= float(args.lo_sec))
        & (df["window_end_sec"] <= float(args.hi_sec))
    )
    per = df.loc[keep].copy()
    if per.empty:
        raise RuntimeError(f"no windows in [{args.lo_sec}, {args.hi_sec}] from {src}")
    agg = _agg_signed(per)

    stem = (
        f"{ds_sid}_signed_AB_similarity_pre_m{abs(int(args.lo_sec))}_m{abs(int(args.hi_sec))}_"
        f"{args.band_lo:g}_{args.band_hi:g}Hz_{args.window_sec:g}s"
    )
    per_csv = BASE / f"{stem}_per_seizure.csv"
    agg_csv = BASE / f"{stem}_aggregate.csv"
    png = FIG / f"{stem}.png"
    summary_json = BASE / f"{stem}_summary.json"

    per.to_csv(per_csv, index=False)
    agg.to_csv(agg_csv, index=False)
    _plot(
        ds_sid,
        per,
        agg,
        png,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        lo_sec=args.lo_sec,
        hi_sec=args.hi_sec,
        window_sec=args.window_sec,
    )
    summary = {
        "subject": ds_sid,
        "source_csv": str(src.relative_to(_ROOT)),
        "band_hz": [float(args.band_lo), float(args.band_hi)],
        "window_sec": float(args.window_sec),
        "time_range_sec": [float(args.lo_sec), float(args.hi_sec)],
        "metric": "signed corr_pair_mirror_invariant_signed separately for template A and B",
        "maxAB": False,
        "absolute_value": False,
        "n_seizures": int(per["seizure_idx"].nunique()),
        "n_windows": int(agg.shape[0]),
        "A": {
            "median_of_window_medians": float(np.nanmedian(agg["A_median_signed_corr"])),
            "median_of_window_variances": float(np.nanmedian(agg["A_var_signed_corr"])),
        },
        "B": {
            "median_of_window_medians": float(np.nanmedian(agg["B_median_signed_corr"])),
            "median_of_window_variances": float(np.nanmedian(agg["B_var_signed_corr"])),
        },
        "outputs": {
            "figure": str(png.relative_to(_ROOT)),
            "per_seizure_csv": str(per_csv.relative_to(_ROOT)),
            "aggregate_csv": str(agg_csv.relative_to(_ROOT)),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(png)
    print(per_csv)
    print(agg_csv)
    print(summary_json)
    return png, per_csv, agg_csv, summary_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--lo-sec", type=float, default=-120.0)
    ap.add_argument("--hi-sec", type=float, default=-20.0)
    ap.add_argument("--band-lo", type=float, default=1.0)
    ap.add_argument("--band-hi", type=float, default=150.0)
    ap.add_argument("--window-sec", type=float, default=10.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
