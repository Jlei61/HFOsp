#!/usr/bin/env python3
"""Plot signed A/B field similarity for onset 0-20 s only.

Consumes the per-seizure CSV produced by
`plot_topic5_signed_broadband_similarity_timecourse.py`.
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


OUT = _ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
FIG = OUT / "figures"


def _summary(vals: pd.Series) -> dict:
    v = pd.to_numeric(vals, errors="coerce").dropna().to_numpy(float)
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "var": None}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "sd": float(np.std(v, ddof=1)) if v.size >= 2 else None,
        "var": float(np.var(v, ddof=1)) if v.size >= 2 else None,
    }


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    ds_sid = args.subject
    src = OUT / (
        f"{ds_sid}_signed_broadband_{args.band_lo:g}_{args.band_hi:g}Hz_"
        f"similarity_timecourse_{args.window_sec:g}s_per_seizure.csv"
    )
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_csv(src)
    sub = df[(df["window_start_sec"] >= args.start_sec) & (df["window_end_sec"] <= args.end_sec)].copy()
    if sub.empty:
        raise RuntimeError(f"no windows inside [{args.start_sec}, {args.end_sec}] in {src}")

    agg_rows = []
    for (lo, hi, cen), g in sub.groupby(["window_start_sec", "window_end_sec", "window_center_sec"], sort=True):
        row = {
            "window_start_sec": float(lo),
            "window_end_sec": float(hi),
            "window_center_sec": float(cen),
            "n_seizures": int(g["seizure_idx"].nunique()),
        }
        for tmpl, col in (("A", "A_signed_corr"), ("B", "B_signed_corr")):
            s = _summary(g[col])
            for k, v in s.items():
                row[f"{tmpl}_{k}"] = v
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)

    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{ds_sid}_signed_broadband_{args.band_lo:g}_{args.band_hi:g}Hz_"
        f"AB_signed_similarity_onset_{args.start_sec:g}_{args.end_sec:g}_{args.window_sec:g}s"
    )
    png = FIG / f"{stem}.png"
    agg_csv = OUT / f"{stem}_aggregate.csv"
    summary_json = OUT / f"{stem}_summary.json"
    agg.to_csv(agg_csv, index=False)

    fig, (ax, axv) = plt.subplots(
        2,
        1,
        figsize=(8.6, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
        layout="constrained",
    )
    x_windows = sorted(sub["window_center_sec"].unique())
    jitter = {"A": -0.35, "B": 0.35}
    colors = {"A": "#2166ac", "B": "#b2182b"}
    pale = {"A": "#92c5de", "B": "#f4a582"}

    for seizure_idx, g in sub.groupby("seizure_idx"):
        g = g.sort_values("window_center_sec")
        ax.plot(g["window_center_sec"] + jitter["A"], g["A_signed_corr"],
                color=pale["A"], lw=0.8, alpha=0.45)
        ax.plot(g["window_center_sec"] + jitter["B"], g["B_signed_corr"],
                color=pale["B"], lw=0.8, alpha=0.45)

    for tmpl in ("A", "B"):
        x = agg["window_center_sec"].to_numpy(float) + jitter[tmpl]
        mean = agg[f"{tmpl}_mean"].to_numpy(float)
        med = agg[f"{tmpl}_median"].to_numpy(float)
        sd = agg[f"{tmpl}_sd"].to_numpy(float)
        ax.errorbar(x, mean, yerr=sd, fmt="o-", color=colors[tmpl], lw=2.0,
                    capsize=4, label=f"template {tmpl}: mean +/- SD")
        ax.plot(x, med, marker="s", ms=5, lw=1.2, ls="--", color=colors[tmpl],
                alpha=0.75, label=f"template {tmpl}: median")
        axv.bar(x, agg[f"{tmpl}_var"].to_numpy(float), width=0.55,
                color=colors[tmpl], alpha=0.55, label=f"template {tmpl} variance")

    for axis in (ax, axv):
        axis.axhline(0, color="0.25", lw=0.9)
        axis.grid(True, color="0.9", lw=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("signed field similarity r")
    ax.set_title("A/B signed similarity; no abs, no maxAB selection")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    axv.set_ylabel("variance")
    axv.set_xlabel("time from clinical onset (s), window center")
    axv.set_xticks(x_windows)
    axv.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    for _, row in agg.iterrows():
        ax.text(row["window_center_sec"], 1.01, f"n={int(row['n_seizures'])}",
                ha="center", va="bottom", fontsize=8, color="0.25")

    pretty = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    fig.suptitle(
        f"{pretty}: signed A/B field similarity, {args.band_lo:g}-{args.band_hi:g} Hz, onset {args.start_sec:g}-{args.end_sec:g}s\n"
        f"{args.window_sec:g}s windows; individual seizures are pale lines",
        fontsize=12.5,
    )
    fig.savefig(png, dpi=150)
    plt.close(fig)

    payload = {
        "subject": ds_sid,
        "band_hz": [float(args.band_lo), float(args.band_hi)],
        "window_sec": float(args.window_sec),
        "time_range_sec": [float(args.start_sec), float(args.end_sec)],
        "metric": "signed corr_pair_mirror_invariant_signed per template A/B; no abs; no maxAB selection",
        "source_per_seizure_csv": str(src.relative_to(_ROOT)),
        "outputs": {
            "figure": str(png.relative_to(_ROOT)),
            "aggregate_csv": str(agg_csv.relative_to(_ROOT)),
        },
        "windows": agg.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(png)
    print(agg_csv)
    print(summary_json)
    return png, agg_csv, summary_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--band-lo", type=float, default=1.0)
    ap.add_argument("--band-hi", type=float, default=150.0)
    ap.add_argument("--window-sec", type=float, default=10.0)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=20.0)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
