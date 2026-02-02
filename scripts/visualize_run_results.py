#!/usr/bin/env python3
"""
Visualize one pipeline run using the saved config/run summary.

Figures produced:
1) Group-event Hilbert envelope (core channels if provided)
2) Baseline-corrected wavelet TF maps with centroid paths
3) Lag/rank heatmaps (core channels)
4) Co-activation heatmap from groupAnalysis coact_* fields
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from pathlib import Path as _Path

# Ensure project root on sys.path so "src" imports work
_SCRIPT_DIR = _Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Match notebook defaults
plt.rcParams["figure.dpi"] = 200

import sys

# Fix module import by ensuring project root is in sys.path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("YAML config requires PyYAML (pip install pyyaml)") from exc
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_record_stem(record: str) -> str:
    return Path(record).stem


def _load_core_channels(cfg: Dict[str, Any], subject_dir: Path) -> Optional[list]:
    core_cfg = cfg.get("core_channels", {}) or {}
    source = str(core_cfg.get("source", "none")).lower().strip()
    if source == "manual":
        manual = core_cfg.get("manual_list", []) or []
        return [str(x) for x in manual]
    if source == "hist_meanx":
        hist_path = subject_dir / "hist_meanX.npz"
        if not hist_path.exists():
            print(f"[WARN] hist_meanX not found: {hist_path} (skip core channel filtering)")
            return None
        data = np.load(str(hist_path), allow_pickle=True)
        return [str(x) for x in data["pick_chns"].tolist()]
    return None


def _pick_path(run_summary: Dict[str, Any], keys: list) -> Optional[str]:
    for k in keys:
        if k in run_summary and run_summary[k]:
            return str(run_summary[k])
    return None


def _load_run_summary(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        print(f"[WARN] run_summary not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one pipeline run outputs.")
    parser.add_argument("--config", type=str, default=None, help="Path to config (yaml/json)")
    parser.add_argument("--run-summary", type=str, default=None, help="Path to run_summary.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for figures")
    parser.add_argument("--max-events", type=int, default=80, help="Max number of events to plot")
    args = parser.parse_args()

    run_summary_path = Path(args.run_summary).expanduser() if args.run_summary else None
    run_summary = _load_run_summary(run_summary_path)

    config_path = None
    if args.config:
        config_path = Path(args.config).expanduser()
    else:
        snap = _pick_path(run_summary, ["config_snapshot", "configSnapshot", "config_snapshot_path"])
        if snap:
            config_path = Path(snap).expanduser()

    if config_path is None or not config_path.exists():
        raise FileNotFoundError("Config not found. Provide --config or run_summary with config snapshot.")

    cfg = _load_config(config_path)
    subject = cfg.get("subject")
    record = cfg.get("record")
    if not subject or not record:
        raise ValueError("subject and record must be present in config.")

    data_root = Path(cfg.get("data_root", ".")).expanduser()
    subject_dir = data_root / str(subject)
    output_dir = Path(cfg.get("output_dir") or (subject_dir / "temp")).expanduser()
    output_prefix = cfg.get("output_prefix") or _resolve_record_stem(str(record))
    outputs_cfg = cfg.get("outputs", {}) or {}
    packed_suffix = outputs_cfg.get("packed_times_suffix", "_packedTimes_pipeline.npy")
    lagpat_suffix = outputs_cfg.get("lagpat_suffix", "_lagPat_pipeline.npz")

    analysis_cfg = cfg.get("analysis", {}) or {}
    band = str(analysis_cfg.get("band", "ripple"))
    reference = str(analysis_cfg.get("reference", "bipolar"))

    group_analysis_path = _pick_path(
        run_summary,
        ["groupAnalysis", "group_analysis", "group_analysis_path", "groupAnalysisPath"],
    ) or str(Path(output_dir) / f"{output_prefix}_groupAnalysis.npz")
    packed_times_path = _pick_path(
        run_summary,
        ["packedTimes", "packed_times", "packed_times_path", "packedTimesPath"],
    ) or str(Path(output_dir) / f"{output_prefix}{packed_suffix}")
    lagpat_path = _pick_path(
        run_summary,
        ["lagPat", "lagpat", "lagpat_path", "lagPatPath"],
    ) or str(Path(output_dir) / f"{output_prefix}{lagpat_suffix}")
    env_cache_path = _pick_path(
        run_summary,
        ["envCache", "env_cache", "env_cache_path", "envCachePath"],
    ) or str(Path(output_dir) / "temp" / f"{output_prefix}_envCache_{band}_{reference}.npz")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
    else:
        out_dir = Path("./results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    core_channels = _load_core_channels(cfg, subject_dir)
    if core_channels:
        print(f"[INFO] core channels: {len(core_channels)}")
    else:
        print("[INFO] core channels not specified; use all channels")

    from src.group_event_analysis import load_envelope_cache, load_group_analysis_results
    from src.visualization import (
        plot_coactivation_heatmap_from_group_analysis,
        plot_group_events_band_raster,
        plot_lag_heatmaps_from_group_analysis,
        plot_paper_fig1_bandpassed_traces,
        plot_paper_fig2_normalized_spectrogram,
    )

    # 1) Fig1 (paper style): bandpassed traces (requires x_band)
    cache_meta = load_envelope_cache(str(env_cache_path))
    packed = np.load(str(packed_times_path), allow_pickle=True)
    n_events_total = int(packed.shape[0])
    n_events = min(n_events_total, int(args.max_events))
    if n_events <= 0:
        raise ValueError("No packedTimes events available for Fig1.")
    event_indices = list(range(n_events))
    if cache_meta.get("x_band") is None:
        raise ValueError("env cache has no x_band; set group_analysis.visualization.save_bandpass=true")
    fig1 = plot_paper_fig1_bandpassed_traces(
        cache_npz_path=str(env_cache_path),
        packed_times_path=str(packed_times_path),
        channel_order=core_channels,
        event_indices=event_indices,
        max_events=int(args.max_events),
        figsize=(16, 10),
    )
    fig1_path = out_dir / f"{output_prefix}_env_raster.png"
    _ensure_parent(fig1_path)
    plt.suptitle(f"{subject}/{record} — Bandpassed Traces (80-250Hz)", fontsize=14, y=1.02)
    fig1.savefig(fig1_path, dpi=160)
    plt.close(fig1)
    print(f"[OK] env raster: {fig1_path}")

    # 2) Fig2 (notebook style): normalized spectrogram + centroid paths
    if cache_meta.get("x_band") is None:
        print("[WARN] env cache has no x_band; skip TF plot (need save_bandpass=True).")
    else:
        gpu_path = _pick_path(run_summary, ["gpu", "gpu_npz", "gpu_path"])
        fig2 = plot_paper_fig2_normalized_spectrogram(
            cache_npz_path=str(env_cache_path),
            packed_times_path=str(packed_times_path),
            detections_npz_path=str(gpu_path) if gpu_path else None,
            group_analysis_npz_path=str(group_analysis_path),
            channel_order=core_channels,
            event_indices=event_indices,
            max_events=int(args.max_events),
            freq_band=(80.0, 250.0) if band == "ripple" else (250.0, 500.0),
            centroid_marker_size=60.0,
            figsize=(16, 10),
        )
        fig2_path = out_dir / f"{output_prefix}_tf_centroid_paths.png"
        plt.suptitle(f"{subject}/{record} — TF Power + Centroid Paths", fontsize=14, y=1.02)
        fig2.savefig(fig2_path, dpi=160)
        plt.close(fig2)
        print(f"[OK] TF centroid paths: {fig2_path}")

    # 3) Lag/rank heatmaps (core channels)
    fig_e, fig_r, fig_l = plot_lag_heatmaps_from_group_analysis(
        group_analysis_npz=str(group_analysis_path),
        env_cache_npz=str(env_cache_path),
        packed_times_npy=str(packed_times_path),
        channel_names=core_channels,
        max_events=int(args.max_events),
        cmap_rank="Greens_r",
    )
    fig_e_path = out_dir / f"{output_prefix}_event_energy.png"
    fig_r_path = out_dir / f"{output_prefix}_lag_rank.png"
    fig_l_path = out_dir / f"{output_prefix}_lag_ms.png"
    fig_e.savefig(fig_e_path, dpi=160)
    fig_r.savefig(fig_r_path, dpi=160)
    fig_l.savefig(fig_l_path, dpi=160)
    plt.close(fig_e)
    plt.close(fig_r)
    plt.close(fig_l)
    print(f"[OK] lag/rank/energy: {fig_r_path}")

    # 4) Check coact_* fields + plot coactivation heatmap
    ga = load_group_analysis_results(str(group_analysis_path))
    coact_keys = [k for k in ga.keys() if k.startswith("coact_")]
    if not coact_keys:
        raise ValueError("coact_* fields not found in groupAnalysis.")
    print(f"[OK] coact fields: {sorted(coact_keys)}")

    fig4 = plot_coactivation_heatmap_from_group_analysis(
        group_analysis_npz=str(group_analysis_path),
        metric="time_ratio",
        channel_names=None,
        cmap="viridis",
        figsize=(8, 7),
        show_values=False,
    )
    fig4_path = out_dir / f"{output_prefix}_coact_time_ratio.png"
    fig4.savefig(fig4_path, dpi=160)
    plt.close(fig4)
    print(f"[OK] coactivation heatmap: {fig4_path}")

    # Quick lagPat existence check
    if Path(lagpat_path).exists():
        print(f"[OK] lagPat: {lagpat_path}")
    else:
        print(f"[WARN] lagPat not found: {lagpat_path}")


if __name__ == "__main__":
    main()
