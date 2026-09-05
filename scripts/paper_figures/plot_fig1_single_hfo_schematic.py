#!/usr/bin/env python3
"""Legacy HFO morphology panel for paper Figure 1 Panel a1.

This producer deliberately uses the exact manually annotated legacy artifact
shown in the supplied reference: 178 HFO snippets from
``zhangkexuan_pickSigs.npz`` selected by ``zhangkexuan_annot_v4.pik``.  The
waveform and spectrogram recipe follows ``p16_mechan_events_specComp.py``.

Unlike the historical plotting call, the spectrogram cell edges are extended
to the full 0--0.6 s snippet bounds.  This removes the misleading white strips
next to the x axis without changing any spectrogram value.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ["HOME"] = "/tmp/hfo-paper-home"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from fig1_spectrogram_utils import (
    ALGORITHM_ID,
    compute_smoothed_magnitude_spectrogram,
)


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ROOT / "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef"
DEFAULT_SNIPPETS = LEGACY_ROOT / "zhangkexuan_pickSigs.npz"
DEFAULT_ANNOTATIONS = LEGACY_ROOT / "zhangkexuan_annot_v4.pik"
DEFAULT_OUTPUT_DIR = ROOT / "results/paper-ready-figure/fig1_hfo_group_event_demo/figures"
DEFAULT_STEM = "legacy_hfo_n178_schematic"


def _load_annotated_hfos(
    snippets_path: Path,
    annotations_path: Path,
    label: int = 1,
) -> np.ndarray:
    data = np.load(snippets_path, allow_pickle=True)
    if "sigs" not in data.files:
        raise KeyError(f"{snippets_path} does not contain 'sigs'")
    with annotations_path.open("rb") as f:
        annotations = np.asarray(pickle.load(f))
    snippets = np.asarray(data["sigs"], dtype=np.float64)
    if snippets.ndim != 2:
        raise ValueError(f"expected snippets with shape (events, time), got {snippets.shape}")
    if annotations.shape != (snippets.shape[0],):
        raise ValueError(
            f"annotation/snippet mismatch: {annotations.shape} vs {snippets.shape[0]} events"
        )
    hfos = snippets[annotations == int(label)]
    if hfos.shape[0] != 178:
        raise ValueError(
            f"reference contract requires exactly 178 label-{label} HFOs, got {hfos.shape[0]}"
        )
    if not np.all(np.isfinite(hfos)):
        raise ValueError("HFO snippets contain non-finite values")
    return hfos


def _mean_spectrograms(
    snippets: np.ndarray,
    fs: float,
    freq_max: float = 240.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the legacy raw and baseline-normalized mean spectrograms."""
    win = int(round(0.18 * fs))
    overlap = int(round(0.16 * fs))
    specs = []
    spec_t = spec_f = None
    for row in snippets:
        freqs, times, magnitude = compute_smoothed_magnitude_spectrogram(
            row,
            fs,
            window="hann",
            window_sec=win / float(fs),
            overlap_sec=overlap / float(fs),
            freq_range_hz=(0.0, float(freq_max)),
            gaussian_sigma=1.5,
        )
        specs.append(magnitude)
        spec_t = times
        spec_f = freqs
    if spec_t is None or spec_f is None:
        raise ValueError("no HFO spectrograms were computed")
    mean_spec = np.mean(np.asarray(specs), axis=0)
    raw_spec = np.log(np.maximum(mean_spec, np.finfo(float).tiny))
    baseline = mean_spec[:, spec_t <= 0.15]
    if baseline.shape[1] == 0:
        raise ValueError("legacy baseline t<=0.15 s contains no spectrogram bins")
    baseline_mean = np.mean(baseline, axis=1, keepdims=True)
    norm_spec = mean_spec / np.maximum(baseline_mean, np.finfo(float).tiny) - 1.0
    return raw_spec, norm_spec, spec_t, spec_f


def _full_extent_edges(centers: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return monotonic pcolormesh edges with exact requested outer bounds."""
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("centers must be a strictly increasing 1D array with >=2 values")
    if not lower < values[0] or not values[-1] < upper:
        raise ValueError(f"outer bounds {lower, upper} must contain center range")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[0] = float(lower)
    edges[-1] = float(upper)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("derived pcolormesh edges are not increasing")
    return edges


def _frequency_edges(freqs: np.ndarray) -> np.ndarray:
    values = np.asarray(freqs, dtype=np.float64)
    step = float(np.median(np.diff(values)))
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = max(0.0, values[0] - 0.5 * step)
    edges[-1] = values[-1] + 0.5 * step
    return edges


def _plot(
    snippets: np.ndarray,
    fs: float,
    raw_spec: np.ndarray,
    norm_spec: np.ndarray,
    spec_t: np.ndarray,
    spec_f: np.ndarray,
    output_png: Path,
    output_pdf: Path,
) -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    duration = snippets.shape[1] / float(fs)
    waveform_t = np.arange(snippets.shape[1], dtype=np.float64) / float(fs)
    time_edges = _full_extent_edges(spec_t, 0.0, duration)
    freq_edges = _frequency_edges(spec_f)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(1.75, 4.22),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.43},
    )
    ax0, ax1, ax2 = axes

    ax0.plot(waveform_t, snippets.T, color="black", linewidth=0.28, alpha=0.23)
    ax0.plot(waveform_t, np.mean(snippets, axis=0), color="#FFD000", linewidth=1.05, zorder=5)
    ax0.set_title("HFO n = 178", color="red", fontsize=12.5, fontweight="normal", pad=4)

    raw_limits = np.nanpercentile(raw_spec, [1.0, 99.0])
    norm_abs = float(np.nanpercentile(np.abs(norm_spec), 99.0))
    ax1.pcolormesh(
        time_edges,
        freq_edges,
        raw_spec,
        cmap="coolwarm",
        shading="flat",
        vmin=float(raw_limits[0]),
        vmax=float(raw_limits[1]),
        rasterized=True,
    )
    ax2.pcolormesh(
        time_edges,
        freq_edges,
        norm_spec,
        cmap="coolwarm",
        shading="flat",
        vmin=-norm_abs,
        vmax=norm_abs,
        rasterized=True,
    )

    for ax, title in ((ax1, "raw Spec"), (ax2, "normalized Spec")):
        ax.set_title(title, fontsize=10.5, fontweight="normal", pad=3)
        ax.set_ylabel("Freq (Hz)", fontsize=9.5, labelpad=5)
        ax.set_ylim(0.0, 240.0)
        ax.set_yticks([0, 100, 200])

    ax2.set_xlabel("Time (s)", fontsize=9.5, labelpad=5)
    for ax in axes:
        ax.set_xlim(0.0, duration)
        ax.set_xticks([0.0, 0.25, 0.50])
        ax.set_xticklabels(["0.00", "0.25", "0.50"])
        ax.margins(x=0.0)
        ax.tick_params(axis="both", labelsize=9.0, length=2.5, width=0.7, pad=2.0)
        ax.tick_params(axis="x", labelbottom=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.55)
        ax.spines["bottom"].set_linewidth(0.55)

    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.11, top=0.95)
    fig.savefig(output_png, dpi=300, facecolor="white")
    fig.savefig(output_pdf, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snippets", type=Path, default=DEFAULT_SNIPPETS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--annotation-label", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    parser.add_argument("--fs", type=float, default=1000.0)
    args = parser.parse_args()

    snippets = _load_annotated_hfos(
        args.snippets.resolve(),
        args.annotations.resolve(),
        label=int(args.annotation_label),
    )
    raw_spec, norm_spec, spec_t, spec_f = _mean_spectrograms(snippets, float(args.fs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf"
    metadata_path = args.output_dir / f"{args.output_stem}_metadata.json"
    _plot(
        snippets,
        float(args.fs),
        raw_spec,
        norm_spec,
        spec_t,
        spec_f,
        output_png,
        output_pdf,
    )

    duration = snippets.shape[1] / float(args.fs)
    metadata = {
        "figure_panel": "Fig1-a1",
        "legacy_code_reference": str(
            LEGACY_ROOT / "p16_mechan_events_specComp.py"
        ),
        "source_paths": {
            "snippets_npz": str(args.snippets.resolve()),
            "annotations_pickle": str(args.annotations.resolve()),
        },
        "annotation_label": int(args.annotation_label),
        "n_hfo_snippets": int(snippets.shape[0]),
        "snippet_samples": int(snippets.shape[1]),
        "duration_sec": float(duration),
        "fs_hz": float(args.fs),
        "spectrogram": {
            "algorithm_id": ALGORITHM_ID,
            "window": "hann",
            "nperseg_sec": 0.18,
            "noverlap_sec": 0.16,
            "freq_range_hz": [0, 240],
            "gaussian_sigma": 1.5,
            "raw": "log(mean smoothed magnitude spectrogram)",
            "normalized": "mean_spec / mean(baseline bins t<=0.15 s) - 1",
            "x_cell_edges_sec": [0.0, float(duration)],
            "edge_policy": "extend first/last spectrogram cells to snippet bounds; no x-axis white strips",
        },
        "outputs": [str(output_png), str(output_pdf)],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {output_png}")
    print(f"[OK] wrote {output_pdf}")
    print(f"[OK] n_hfo={snippets.shape[0]} duration={duration:.3f}s")


if __name__ == "__main__":
    main()
