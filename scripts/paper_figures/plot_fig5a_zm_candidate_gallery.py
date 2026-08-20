#!/usr/bin/env python3
"""Render comparable Figure 5A diagnostics for frozen Z/M candidates."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ICL = "#F1783A"
SCL = "#29A6B5"
ONSET = "#D62745"
STATE_SHADE = "#F7E9ED"
ACTIVE = "#252525"
SHEET = "#6D7F91"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _signed_bandpass(raw: np.ndarray, dt_ms: float) -> np.ndarray:
    fs_hz = 1000.0 / float(dt_ms)
    sos = butter(4, (30.0, 80.0), btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(raw, float), axis=0)


def _contact_order(names: np.ndarray) -> np.ndarray:
    def number(name: str) -> int:
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else -1

    return np.asarray(sorted(
        range(len(names)),
        key=lambda index: (
            0 if str(names[index]).startswith("SCL") else 1,
            number(str(names[index])),
        ),
    ), dtype=int)


def _candidate(prefix: Path) -> dict:
    meta = json.loads(prefix.with_suffix(".json").read_text())
    data = _load_npz(prefix.with_suffix(".npz"))
    morphology = meta.get("runaway_morphology")
    if not isinstance(morphology, dict):
        raise RuntimeError(f"{prefix} has no morphology audit")
    onset_ms = float(morphology["scientific_onset_ms"])
    dt_ms = float(data["lfp_dt_ms"])
    filtered = _signed_bandpass(data["lfp_trace"], dt_ms)
    parameters = meta["parameters"]
    dose_recorded = "E_to_I_dose" in parameters
    return {
        "prefix": prefix,
        "meta": meta,
        "data": data,
        "morphology": morphology,
        "onset_ms": onset_ms,
        "dt_ms": dt_ms,
        "filtered": filtered,
        "dose": float(parameters.get("E_to_I_dose", 1.0)),
        "dose_provenance": "recorded" if dose_recorded else "legacy_default_1p0",
    }


def _display_window(candidate: dict) -> tuple[float, float]:
    onset = candidate["onset_ms"]
    duration = candidate["filtered"].shape[0] * candidate["dt_ms"]
    return max(0.0, onset - 1000.0), min(duration, onset + 1500.0)


def _shared_scale(candidates: list[dict]) -> float:
    values = []
    for candidate in candidates:
        onset = candidate["onset_ms"]
        dt_ms = candidate["dt_ms"]
        time_ms = np.arange(candidate["filtered"].shape[0]) * dt_ms
        mask = (time_ms >= onset - 500.0) & (time_ms < onset)
        values.append(np.abs(candidate["filtered"][mask]).ravel())
    scale = float(np.percentile(np.concatenate(values), 99.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("candidate gallery has no finite shared current scale")
    return scale


def _summary_text(candidate: dict) -> str:
    morphology = candidate["morphology"]
    recruitment = morphology["full_field_recruitment"]
    population = morphology["population_rate_frequency"]
    oscillation = morphology["contact_oscillation"]
    majority = 100.0 * min(
        float(recruitment["fraction_windows_majority_E_active"]),
        float(recruitment["fraction_windows_majority_sheet_recruited"]),
    )
    return (
        f"majority windows {majority:.0f}%  |  "
        f"population {population['median_rate_pre_hz']:.0f} to "
        f"{population['median_rate_post_hz']:.0f} Hz  |  "
        f"contacts {oscillation['median_spectral_centroid_pre_hz']:.0f} to "
        f"{oscillation['median_spectral_centroid_post_hz']:.0f} Hz"
    )


def _plot(candidate: dict, shared_scale: float) -> plt.Figure:
    data = candidate["data"]
    names = np.asarray(data["contact_names"]).astype(str)
    shafts = (
        np.asarray(data["shaft_ids"]).astype(str)
        if "shaft_ids" in data else
        np.asarray(["SCL" if name.startswith("SCL") else "ICL" for name in names])
    )
    order = _contact_order(names)
    start, stop = _display_window(candidate)
    onset = candidate["onset_ms"]
    dt_ms = candidate["dt_ms"]
    time_ms = np.arange(candidate["filtered"].shape[0]) * dt_ms
    mask = (time_ms >= start) & (time_ms <= stop)
    shown_time = time_ms[mask] - start
    traces = 0.72 * candidate["filtered"][mask][:, order] / shared_scale

    fig = plt.figure(figsize=(7.25, 3.35))
    grid = fig.add_gridspec(
        2, 1, height_ratios=(0.72, 2.75), hspace=0.05,
        left=0.115, right=0.985, bottom=0.16, top=0.84,
    )
    recruit_ax = fig.add_subplot(grid[0])
    trace_ax = fig.add_subplot(grid[1], sharex=recruit_ax)

    state_start = onset - start
    for axis in (recruit_ax, trace_ax):
        axis.axvspan(state_start, stop - start, color=STATE_SHADE, lw=0, zorder=0)
        axis.axvline(state_start, color=ONSET, lw=1.0, ls="--", zorder=4)

    recruitment_time = np.asarray(data["full_field_time_ms"], float)
    rmask = (recruitment_time >= start) & (recruitment_time <= stop)
    recruit_ax.plot(
        recruitment_time[rmask] - start,
        100.0 * np.asarray(data["active_neuron_fraction_20ms"], float)[rmask],
        color=ACTIVE, lw=0.9, label="active E neurons",
    )
    recruit_ax.plot(
        recruitment_time[rmask] - start,
        100.0 * np.asarray(data["recruited_spatial_fraction_1mm"], float)[rmask],
        color=SHEET, lw=0.9, label="recruited sheet",
    )
    recruit_ax.axhline(50.0, color=ONSET, lw=0.75, ls=":")
    recruit_ax.set_ylim(0.0, 103.0)
    recruit_ax.set_ylabel("Global\nrecruitment (%)", fontsize=7.5)
    recruit_ax.tick_params(axis="x", labelbottom=False)
    recruit_ax.tick_params(axis="y", labelsize=6.8, length=2.2)
    recruit_ax.spines[["top", "right"]].set_visible(False)
    recruit_ax.legend(
        loc="upper left", frameon=False, ncol=2, fontsize=6.8,
        handlelength=1.6, columnspacing=1.0,
    )

    offsets = np.arange(len(order), dtype=float) * 1.1
    for row, contact_index in enumerate(order):
        color = ICL if shafts[contact_index] == "ICL" else SCL
        trace_ax.plot(
            shown_time, traces[:, row] + offsets[row],
            color=color, lw=0.72, alpha=0.97,
        )
    trace_ax.set_xlim(0.0, stop - start)
    trace_ax.set_ylim(-0.8, offsets[-1] + 0.9)
    trace_ax.set_yticks(offsets)
    trace_ax.set_yticklabels(names[order], fontsize=6.8)
    for tick, contact_index in zip(trace_ax.get_yticklabels(), order):
        tick.set_color(ICL if shafts[contact_index] == "ICL" else SCL)
    trace_ax.set_xlabel("Time in continuous trajectory (ms)", fontsize=8.5)
    trace_ax.set_ylabel("Virtual-SEEG proxy (30-80 Hz)", fontsize=8.5)
    trace_ax.tick_params(axis="x", labelsize=7.2, length=2.5)
    trace_ax.spines[["top", "right"]].set_visible(False)
    trace_ax.text(
        state_start + 12.0, offsets[-1] + 0.45, "transition",
        color=ONSET, fontsize=6.8, ha="left", va="center",
    )

    dose_pct = 100.0 * candidate["dose"]
    fig.text(
        0.115, 0.975,
        f"E1146 data-driven Z/M | E-to-I dose {dose_pct:g}%",
        fontsize=10.0, fontweight="bold", ha="left", va="top",
    )
    fig.text(0.115, 0.918, _summary_text(candidate),
             fontsize=7.2, ha="left", va="top", color="0.25")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", required=True,
                        help="Candidate prefix without .json/.npz")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    candidates = [_candidate(Path(record).resolve()) for record in args.record]
    candidates.sort(key=lambda candidate: candidate["dose"])
    doses = [candidate["dose"] for candidate in candidates]
    if len(set(doses)) != len(doses):
        raise RuntimeError(f"candidate doses are not unique: {doses}")
    shared_scale = _shared_scale(candidates)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    pdf_path = out_dir / "fig5a-zm-etoi-dose-gallery.pdf"
    with PdfPages(pdf_path) as pdf:
        for candidate in candidates:
            dose_pct = 100.0 * candidate["dose"]
            token = f"{dose_pct:05.1f}".replace(".", "p")
            stem = out_dir / f"fig5a-zm-etoi-{token}pct"
            fig = _plot(candidate, shared_scale)
            fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
            fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            records.append({
                "dose": candidate["dose"],
                "dose_provenance": candidate["dose_provenance"],
                "source": str(candidate["prefix"]),
                "png": str(stem.with_suffix(".png")),
                "pdf": str(stem.with_suffix(".pdf")),
                "verdict": candidate["meta"]["verdict"],
                "summary": _summary_text(candidate),
            })

    metadata = {
        "status": "FIG5A_ZM_CANDIDATE_GALLERY_RENDERED",
        "shared_pretransition_p99_current_scale": shared_scale,
        "scale_contract": "one shared 30-80 Hz current-proxy scale across all doses",
        "candidate_count": len(records),
        "records": records,
        "multipage_pdf": str(pdf_path),
    }
    (out_dir / "fig5a-zm-etoi-dose-gallery.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    readme = [
        "### fig5a-zm-etoi-<dose>pct.png / .pdf",
        "每张图使用同一套 Fig5A 语法，比较同一冻结 E1146 data-driven Z/M 底物上不同 E→I 调制剂量。",
        "上轨给出全场活跃兴奋神经元和被招募空间网格比例；下轨给出连续 15 触点 30–80 Hz 虚拟读出。整组候选共用同一个间期电流放大系数。",
        "**关注点**：进入后是否长期保持多数神经元和多数空间网格参与，同时触点振荡是否明显加快。",
        "",
        "### fig5a-zm-etoi-dose-gallery.pdf",
        "多页 PDF 按 E→I 剂量从低到高排列，便于逐页比较。",
        "**关注点**：E→I 增强后全局高频状态是否被压缩为间歇或局部招募。",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme))
    print(json.dumps({
        "candidate_count": len(records),
        "out_dir": str(out_dir),
        "shared_scale": shared_scale,
    }))


if __name__ == "__main__":
    main()
