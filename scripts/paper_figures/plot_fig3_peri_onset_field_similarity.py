#!/usr/bin/env python3
"""Paper-ready peri-onset field-similarity diagnostic for Fig3.

This is a compact companion figure for the Topic 5 ictal field-dynamics
exploration. It uses the same per-seizure table as the diagnostic plots, but
renders only the paper-facing readouts:

  a. maxAB sign-free scaffold similarity, max(|r_A|, |r_B|)
  b. signed similarity to template A and template B separately

The lower diagnostic variance / n-seizure panel is intentionally omitted.
Those values are written to the sidecar JSON and README instead.
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_style import FS_LABEL, FS_TICK, savefig_pub, style_panel  # noqa: E402

FIELD_DIR = ROOT / "results/topic5_ictal_recruitment/field_dynamics_signed"
OUT_DIR = ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity/figures"

LO_SEC = -120.0
HI_SEC = 20.0
WINDOW_SEC = 10.0
STEP_SEC = 2.0

COL_MAX = "#A35E48"
COL_A = "#B2182B"
COL_B = "#2166AC"
COL_INDIV = "#B5B5B5"


def _subject_label(ds_sid: str) -> str:
    return ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _source_csv(ds_sid: str) -> Path:
    return FIELD_DIR / (
        f"{ds_sid}_signed_broadband_1_150Hz_"
        "similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv"
    )


def _load_peri_onset(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src)
    keep = (df["window_start_sec"] >= LO_SEC) & (df["window_end_sec"] <= HI_SEC)
    out = df.loc[keep].copy()
    if out.empty:
        raise RuntimeError(f"no windows in [{LO_SEC}, {HI_SEC}] from {SRC}")
    return out


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lo, hi, cen), g in df.groupby(["window_start_sec", "window_end_sec", "window_center_sec"], sort=True):
        row = {
            "window_start_sec": float(lo),
            "window_end_sec": float(hi),
            "window_center_sec": float(cen),
            "n_seizures": int(g["seizure_idx"].nunique()),
        }
        specs = {
            "maxAB": "maxAB_abs_corr",
            "A": "A_signed_corr",
            "B": "B_signed_corr",
        }
        for prefix, col in specs.items():
            vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(float)
            row[f"{prefix}_mean"] = float(np.mean(vals))
            row[f"{prefix}_median"] = float(np.median(vals))
            row[f"{prefix}_q25"] = float(np.percentile(vals, 25))
            row[f"{prefix}_q75"] = float(np.percentile(vals, 75))
            row[f"{prefix}_sd"] = float(np.std(vals, ddof=1))
            row[f"{prefix}_var"] = float(np.var(vals, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_individual(ax: plt.Axes, df: pd.DataFrame, col: str, *, color: str = COL_INDIV) -> None:
    for _idx, g in df.groupby("seizure_idx"):
        g = g.sort_values("window_center_sec")
        ax.plot(
            g["window_center_sec"],
            g[col],
            color=color,
            lw=0.45,
            alpha=0.13,
            zorder=1,
        )


def _draw_band_line(
    ax: plt.Axes,
    agg: pd.DataFrame,
    prefix: str,
    *,
    color: str,
    label: str,
    band_label: str | None = None,
) -> None:
    x = agg["window_center_sec"].to_numpy(float)
    q25 = agg[f"{prefix}_q25"].to_numpy(float)
    q75 = agg[f"{prefix}_q75"].to_numpy(float)
    med = agg[f"{prefix}_median"].to_numpy(float)
    ax.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=0.16,
        linewidth=0,
        label=band_label,
        zorder=2,
    )
    ax.plot(x, med, color=color, lw=2.2, label=label, zorder=4)


def _make_figure(df: pd.DataFrame, agg: pd.DataFrame, *, subject_label: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True)
    ax0, ax1 = axes
    xlo = float(agg["window_center_sec"].min())
    xhi = float(agg["window_center_sec"].max())

    _draw_individual(ax0, df, "maxAB_abs_corr")
    _draw_band_line(
        ax0,
        agg,
        "maxAB",
        color=COL_MAX,
        label="median",
        band_label="IQR",
    )
    ax0.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_xlim(xlo, xhi)
    ax0.set_title("maxAB scaffold similarity", fontsize=FS_LABEL, pad=8)
    ax0.set_ylabel("field similarity |r|", fontsize=FS_LABEL)
    ax0.set_xlabel("window center from onset (s)", fontsize=FS_LABEL)
    ax0.set_xticks([-100, -80, -60, -40, -20, 0])
    ax0.legend(frameon=False, loc="lower right", fontsize=9, handlelength=1.7)
    style_panel(ax0, "a", label_x=-0.17, label_y=1.07)

    for _idx, g in df.groupby("seizure_idx"):
        g = g.sort_values("window_center_sec")
        ax1.plot(g["window_center_sec"], g["A_signed_corr"], color=COL_A, lw=0.4, alpha=0.075, zorder=1)
        ax1.plot(g["window_center_sec"], g["B_signed_corr"], color=COL_B, lw=0.4, alpha=0.075, zorder=1)
    ax1.axhline(0, color="0.35", lw=0.9, zorder=0)
    ax1.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    _draw_band_line(ax1, agg, "A", color=COL_A, label="template A")
    _draw_band_line(ax1, agg, "B", color=COL_B, label="template B")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlim(xlo, xhi)
    ax1.set_title("signed template similarity", fontsize=FS_LABEL, pad=8)
    ax1.set_ylabel("signed field similarity r", fontsize=FS_LABEL)
    ax1.set_xlabel("window center from onset (s)", fontsize=FS_LABEL)
    ax1.set_xticks([-100, -80, -60, -40, -20, 0])
    ax1.legend(frameon=False, loc="lower left", fontsize=9, handlelength=1.7)
    style_panel(ax1, "b", label_x=-0.17, label_y=1.07)

    for ax in axes:
        ax.tick_params(labelsize=FS_TICK - 2)
        ax.text(
            0.02,
            0.97,
            f"{subject_label}, {int(df['seizure_idx'].nunique())} seizures, 10-s windows, 2-s step",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="0.25",
        )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.84, wspace=0.34)
    return fig


def _plot(df: pd.DataFrame, agg: pd.DataFrame, out_png: Path, out_pdf: Path, *, subject_label: str) -> None:
    savefig_pub(_make_figure(df, agg, subject_label=subject_label), out_png, dpi=300)
    savefig_pub(_make_figure(df, agg, subject_label=subject_label), out_pdf, dpi=300)


def _write_outputs(
    ds_sid: str,
    src: Path,
    df: pd.DataFrame,
    agg: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> Path:
    summary = {
        "subject": ds_sid,
        "source_csv": str(src.relative_to(ROOT)),
        "time_range_sec": [LO_SEC, HI_SEC],
        "band_hz": [1.0, 150.0],
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "n_seizures": int(df["seizure_idx"].nunique()),
        "n_windows": int(agg.shape[0]),
        "readouts": {
            "maxAB_abs": {
                "definition": "max(|r_A|, |r_B|)",
                "median_of_window_medians": float(np.nanmedian(agg["maxAB_median"])),
                "median_of_window_variances": float(np.nanmedian(agg["maxAB_var"])),
            },
            "signed_A": {
                "definition": "signed r against template A",
                "median_of_window_medians": float(np.nanmedian(agg["A_median"])),
                "median_of_window_variances": float(np.nanmedian(agg["A_var"])),
            },
            "signed_B": {
                "definition": "signed r against template B",
                "median_of_window_medians": float(np.nanmedian(agg["B_median"])),
                "median_of_window_variances": float(np.nanmedian(agg["B_var"])),
            },
        },
        "outputs": {
            "png": str(out_png.relative_to(ROOT)),
            "pdf": str(out_pdf.relative_to(ROOT)),
        },
    }
    meta = OUT_DIR / f"{ds_sid}_peri_onset_field_similarity_paper_ready_summary.json"
    meta.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# Fig3-B peri-onset field similarity\n\n"
        "### `<subject>_peri_onset_field_similarity_paper_ready.png / .pdf`\n\n"
        "这类图把单个 subject 的合格 seizures 限定在 -120 到 +20 s 的共同 10 s 时间窗内,并以 2 s 步长滑动。"
        "Panel a 显示 `max(|r_A|, |r_B|)` 的 sign-free scaffold similarity; Panel b 分别显示 "
        "template A 和 template B 的 signed similarity。浅线是单次 seizure,粗线是跨 seizure median,"
        "阴影是 IQR; 0 s 虚线标记临床 onset。诊断用的方差和 seizure 数不放在图面,写入 summary JSON。\n\n"
        "**关注点**:Panel a 回答发作前能量场是否像 A/B 任一间期模板;Panel b 检查这种相似性是否具有稳定 polarity。"
        "加入 +20 s 后只解释 onset 附近早期变化,不解释完整发作期轨迹;完整发作期仍需要 duration warping。\n",
        encoding="utf-8",
    )
    return meta


def run(ds_sid: str) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = _source_csv(ds_sid)
    if not src.exists():
        raise FileNotFoundError(src)
    df = _load_peri_onset(src)
    agg = _agg(df)
    out_png = OUT_DIR / f"{ds_sid}_peri_onset_field_similarity_paper_ready.png"
    out_pdf = OUT_DIR / f"{ds_sid}_peri_onset_field_similarity_paper_ready.pdf"
    _plot(df, agg, out_png, out_pdf, subject_label=_subject_label(ds_sid))
    meta = _write_outputs(ds_sid, src, df, agg, out_png, out_pdf)
    print(out_png)
    print(out_pdf)
    print(meta)
    return out_png, out_pdf, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    args = ap.parse_args()
    run(args.subject)


if __name__ == "__main__":
    main()
