#!/usr/bin/env python3
"""Paper-ready peri-onset field-similarity diagnostic for Fig3.

This is a compact companion figure for the Topic 5 ictal field-dynamics
exploration. It uses the same per-seizure table as the diagnostic plots, but
renders only the paper-facing readouts:

  a. shared-gradient maxAB scaffold similarity, max(|r_A|, |r_B|)
  b. signed shared-gradient similarity to template A and template B separately

The lower diagnostic variance / n-seizure panel is intentionally omitted.
Those values are written to the sidecar JSON and README instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
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
from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _load_frozen_shared,
    _shared_geometry_metadata,
)

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

DESIGN_STANDARD = "standard"
DESIGN_JOURNAL_CLEAN = "journal_clean"
DESIGN_VARIANTS = (DESIGN_STANDARD, DESIGN_JOURNAL_CLEAN)


def _subject_label(ds_sid: str) -> str:
    return ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")


def _source_csv(ds_sid: str) -> Path:
    return FIELD_DIR / (
        f"{ds_sid}_signed_broadband_1_150Hz_"
        "similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv"
    )


def _require_unique_exact(df: pd.DataFrame, column: str, expected: object, src: Path) -> None:
    if column not in df:
        raise RuntimeError(f"{src}: missing required provenance column {column}")
    if df[column].isna().any():
        raise RuntimeError(f"{src}: null provenance in {column}")
    values = {str(value) for value in df[column].tolist()}
    if values != {str(expected)}:
        raise RuntimeError(f"{src}: {column}={sorted(values)} != {expected}")


def _require_strict_false(series: pd.Series, src: Path) -> None:
    if series.isna().any():
        raise RuntimeError(f"{src}: null provenance in own_field_fallback")
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            valid = not bool(value)
        elif isinstance(value, str):
            valid = value.strip().lower() == "false"
        else:
            valid = False
        if not valid:
            raise RuntimeError(
                f"{src}: own-field fallback is forbidden (value={value!r})"
            )


def _validate_window_grid(df: pd.DataFrame, src: Path) -> None:
    expected_start = np.arange(LO_SEC, HI_SEC - WINDOW_SEC + 1e-9, STEP_SEC)
    if len(expected_start) != 66:
        raise AssertionError("locked Fig3-B grid must contain 66 windows")
    key = ["seizure_idx", "window_start_sec", "window_end_sec"]
    if df.duplicated(key).any():
        raise RuntimeError(f"{src}: duplicate seizure/window rows")
    for seizure_idx, group in df.groupby("seizure_idx", sort=True):
        group = group.sort_values("window_start_sec")
        starts = pd.to_numeric(group["window_start_sec"], errors="coerce").to_numpy(float)
        ends = pd.to_numeric(group["window_end_sec"], errors="coerce").to_numpy(float)
        centers = pd.to_numeric(group["window_center_sec"], errors="coerce").to_numpy(float)
        if len(group) != 66 or not np.allclose(starts, expected_start, atol=1e-9, rtol=0):
            raise RuntimeError(
                f"{src}: seizure {seizure_idx} has incomplete/noncanonical 66-window grid"
            )
        if not np.allclose(ends, starts + WINDOW_SEC, atol=1e-9, rtol=0):
            raise RuntimeError(f"{src}: seizure {seizure_idx} window_end mismatch")
        if not np.allclose(centers, starts + WINDOW_SEC / 2.0, atol=1e-9, rtol=0):
            raise RuntimeError(f"{src}: seizure {seizure_idx} window_center mismatch")


def _load_peri_onset(src: Path, ds_sid: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(src)
    required = {
        "subject", "seizure_idx", "window_start_sec", "window_end_sec",
        "window_center_sec", "maxAB_abs_corr", "A_abs_corr", "B_abs_corr",
        "A_signed_corr", "B_signed_corr", "field_plane", "field_scorers",
        "field_contract", "field_fingerprint_sha256", "axis_definition",
        "axis_direction_convention", "own_field_fallback",
        "geometry_2d_supported", "geometry_quality_tier",
        "minimum_axis_n_shafts", "minimum_axis_effective_rank",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"{src}: missing required columns {missing}")
    if df.empty:
        raise RuntimeError(f"{src}: empty input CSV")
    if ds_sid is None:
        subjects = df["subject"].dropna().astype(str).unique().tolist()
        if len(subjects) != 1:
            raise RuntimeError(f"{src}: cannot infer one subject identity")
        ds_sid = subjects[0]

    record, _shared = _load_frozen_shared(ds_sid)
    geometry = _shared_geometry_metadata(record)
    _require_unique_exact(df, "subject", ds_sid, src)
    _require_unique_exact(df, "field_plane", "shared", src)
    _require_unique_exact(df, "field_scorers", "shared_a,shared_b", src)
    _require_unique_exact(df, "field_contract", record["contract"], src)
    _require_unique_exact(
        df,
        "field_fingerprint_sha256",
        record["interictal_field"]["fingerprint_sha256"],
        src,
    )
    _require_unique_exact(df, "axis_definition", record["axis_definition"], src)
    _require_unique_exact(
        df,
        "axis_direction_convention",
        record["axis_direction_convention"],
        src,
    )
    _require_unique_exact(df, "geometry_2d_supported", True, src)
    _require_unique_exact(
        df, "geometry_quality_tier", geometry["geometry_quality_tier"], src
    )
    _require_unique_exact(
        df, "minimum_axis_n_shafts", geometry["minimum_axis_n_shafts"], src
    )
    _require_unique_exact(
        df,
        "minimum_axis_effective_rank",
        geometry["minimum_axis_effective_rank"],
        src,
    )
    _require_strict_false(df["own_field_fallback"], src)

    numeric = [
        "seizure_idx", "window_start_sec", "window_end_sec", "window_center_sec",
        "maxAB_abs_corr", "A_abs_corr", "B_abs_corr", "A_signed_corr", "B_signed_corr",
    ]
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{src}: non-finite numeric values in {column}")
    expected_maxab = np.maximum(
        pd.to_numeric(df["A_abs_corr"]).to_numpy(float),
        pd.to_numeric(df["B_abs_corr"]).to_numpy(float),
    )
    if not np.allclose(
        expected_maxab,
        pd.to_numeric(df["maxAB_abs_corr"]).to_numpy(float),
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError(f"{src}: maxAB arithmetic mismatch")

    keep = (df["window_start_sec"] >= LO_SEC) & (df["window_end_sec"] <= HI_SEC)
    out = df.loc[keep].copy()
    if out.empty:
        raise RuntimeError(f"no windows in [{LO_SEC}, {HI_SEC}] from {src}")
    if len(out) != len(df):
        raise RuntimeError(f"{src}: rows outside locked [{LO_SEC}, {HI_SEC}] contract")
    _validate_window_grid(out, src)
    return out


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
            row[f"{prefix}_sd"] = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan
            row[f"{prefix}_var"] = float(np.var(vals, ddof=1)) if vals.size >= 2 else np.nan
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


def _make_figure(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    *,
    subject_label: str,
    design_variant: str = DESIGN_STANDARD,
) -> plt.Figure:
    if design_variant not in DESIGN_VARIANTS:
        raise ValueError(f"unknown design_variant={design_variant!r}")
    journal_clean = design_variant == DESIGN_JOURNAL_CLEAN

    figsize = (7.4, 2.55) if journal_clean else (7.4, 3.25)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
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
    if journal_clean:
        ax0.set_ylabel(
            "Field similarity\n" + r"$\max(|r_A|, |r_B|)$",
            fontsize=FS_LABEL - 2,
        )
        ax0.set_xlabel("Time (s)", fontsize=FS_LABEL - 2)
    else:
        ax0.set_title("shared-gradient maxAB", fontsize=FS_LABEL, pad=8)
        ax0.set_ylabel("field similarity |r|", fontsize=FS_LABEL)
        ax0.set_xlabel("window center from onset (s)", fontsize=FS_LABEL)
    ax0.set_xticks([-100, -80, -60, -40, -20, 0])
    ax0.legend(frameon=False, loc="lower right", fontsize=9, handlelength=1.7)

    for _idx, g in df.groupby("seizure_idx"):
        g = g.sort_values("window_center_sec")
        ax1.plot(g["window_center_sec"], g["A_signed_corr"], color=COL_A, lw=0.4, alpha=0.075, zorder=1)
        ax1.plot(g["window_center_sec"], g["B_signed_corr"], color=COL_B, lw=0.4, alpha=0.075, zorder=1)
    ax1.axhline(0, color="0.35", lw=0.9, zorder=0)
    ax1.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    _draw_band_line(
        ax1,
        agg,
        "A",
        color=COL_A,
        label="TA" if journal_clean else "template A",
    )
    _draw_band_line(
        ax1,
        agg,
        "B",
        color=COL_B,
        label="TB" if journal_clean else "template B",
    )
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_xlim(xlo, xhi)
    if journal_clean:
        ax1.set_ylabel("Signed field similarity, r", fontsize=FS_LABEL - 2)
        ax1.set_xlabel("Time (s)", fontsize=FS_LABEL - 2)
    else:
        ax1.set_title("signed shared-gradient A/B", fontsize=FS_LABEL, pad=8)
        ax1.set_ylabel("signed field similarity r", fontsize=FS_LABEL)
        ax1.set_xlabel("window center from onset (s)", fontsize=FS_LABEL)
    ax1.set_xticks([-100, -80, -60, -40, -20, 0])
    ax1.legend(frameon=False, loc="lower left", fontsize=9, handlelength=1.7)

    for ax in axes:
        style_panel(ax, "" if journal_clean else ("a" if ax is ax0 else "b"), label_x=-0.17, label_y=1.07)
        ax.tick_params(labelsize=FS_TICK - 2)
        if not journal_clean:
            ax.text(
                0.02,
                0.97,
                f"{subject_label} · shared A/B · n={int(df['seizure_idx'].nunique())} seizures",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color="0.25",
            )

    if journal_clean:
        fig.subplots_adjust(left=0.11, right=0.995, bottom=0.25, top=0.98, wspace=0.35)
    else:
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.18, top=0.84, wspace=0.34)
    return fig


def _plot(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    subject_label: str,
    design_variant: str = DESIGN_STANDARD,
) -> None:
    savefig_pub(
        _make_figure(
            df,
            agg,
            subject_label=subject_label,
            design_variant=design_variant,
        ),
        out_png,
        dpi=300,
    )
    savefig_pub(
        _make_figure(
            df,
            agg,
            subject_label=subject_label,
            design_variant=design_variant,
        ),
        out_pdf,
        dpi=300,
    )


def _build_summary(
    ds_sid: str,
    src: Path,
    df: pd.DataFrame,
    agg: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    *,
    design_variant: str = DESIGN_STANDARD,
) -> dict:
    source_summary_path = src.with_name(src.name.replace("_per_seizure.csv", "_summary.json"))
    if not source_summary_path.exists():
        raise FileNotFoundError(source_summary_path)
    source_summary = json.loads(source_summary_path.read_text())
    if source_summary.get("subject") != ds_sid:
        raise RuntimeError(f"{source_summary_path}: subject identity mismatch")
    if source_summary.get("field_fingerprint_sha256") != str(df["field_fingerprint_sha256"].iloc[0]):
        raise RuntimeError(f"{source_summary_path}: stale/mixed fingerprint")
    if int(source_summary.get("n_seizures_processed", -1)) != int(df["seizure_idx"].nunique()):
        raise RuntimeError(f"{source_summary_path}: seizure coverage mismatch")
    return {
        "subject": ds_sid,
        "design_variant": design_variant,
        "manuscript_panel": "Fig3E" if design_variant == DESIGN_JOURNAL_CLEAN else None,
        "source_csv": str(src.relative_to(ROOT)),
        "time_range_sec": [LO_SEC, HI_SEC],
        "band_hz": [1.0, 150.0],
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "field_contract": str(df["field_contract"].iloc[0]),
        "field_plane": "shared",
        "field_scorers": ["shared_a", "shared_b"],
        "own_field_fallback": False,
        "field_fingerprint_sha256": str(df["field_fingerprint_sha256"].iloc[0]),
        "axis_definition": str(df["axis_definition"].iloc[0]),
        "axis_direction_convention": str(df["axis_direction_convention"].iloc[0]),
        "geometry_2d_supported": True,
        "geometry_quality_tier": str(df["geometry_quality_tier"].iloc[0]),
        "minimum_axis_n_shafts": int(df["minimum_axis_n_shafts"].iloc[0]),
        "minimum_axis_effective_rank": int(df["minimum_axis_effective_rank"].iloc[0]),
        "coverage_status": source_summary["coverage_status"],
        "coverage_fraction": source_summary["coverage_fraction"],
        "n_eligible_requested": source_summary["n_eligible_requested"],
        "n_seizure_drops": len(source_summary.get("drops", [])),
        "seizure_drops": source_summary.get("drops", []),
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


def run(
    ds_sid: str,
    *,
    source_csv: Path | None = None,
    out_dir: Path | None = None,
    design_variant: str = DESIGN_STANDARD,
) -> tuple[Path, Path, Path]:
    if design_variant not in DESIGN_VARIANTS:
        raise ValueError(f"unknown design_variant={design_variant!r}")
    output_dir = Path(out_dir).resolve() if out_dir is not None else OUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source_csv).resolve() if source_csv is not None else _source_csv(ds_sid)
    if not src.exists():
        raise FileNotFoundError(src)
    df = _load_peri_onset(src, ds_sid)
    agg = _agg(df)
    suffix = "_journal_clean" if design_variant == DESIGN_JOURNAL_CLEAN else ""
    stem = f"{ds_sid}_peri_onset_field_similarity_paper_ready{suffix}"
    out_png = output_dir / f"{stem}.png"
    out_pdf = output_dir / f"{stem}.pdf"
    meta = output_dir / f"{stem}_summary.json"
    summary = _build_summary(
        ds_sid,
        src,
        df,
        agg,
        out_png,
        out_pdf,
        design_variant=design_variant,
    )
    tmp_png = _temporary_sibling(out_png)
    tmp_pdf = _temporary_sibling(out_pdf)
    tmp_meta = _temporary_sibling(meta)
    try:
        _plot(
            df,
            agg,
            tmp_png,
            tmp_pdf,
            subject_label=_subject_label(ds_sid),
            design_variant=design_variant,
        )
        tmp_meta.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        if not tmp_png.stat().st_size or not tmp_pdf.stat().st_size:
            raise RuntimeError(f"{ds_sid}: renderer produced an empty figure")
        os.replace(tmp_png, out_png)
        os.replace(tmp_pdf, out_pdf)
        os.replace(tmp_meta, meta)  # completion marker is replaced last
    finally:
        for tmp in (tmp_png, tmp_pdf, tmp_meta):
            tmp.unlink(missing_ok=True)
    print(out_png)
    print(out_pdf)
    print(meta)
    return out_png, out_pdf, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--source-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--design-variant", choices=DESIGN_VARIANTS, default=DESIGN_STANDARD)
    args = ap.parse_args()
    run(
        args.subject,
        source_csv=args.source_csv,
        out_dir=args.out_dir,
        design_variant=args.design_variant,
    )


if __name__ == "__main__":
    main()
