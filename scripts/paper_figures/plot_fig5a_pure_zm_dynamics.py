#!/usr/bin/env python3
"""Render the strongest full-edge, Z/M-only high-recruitment Fig. 5A.

The selected trajectory must retain the frozen learned EE and EI substrate.
Selection is made from the model-internal qualification table before looking at
the image: maximize one-second joint broad-recruitment duty among numerically
safe full-edge candidates that also pass the population-rate clause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
INK = "#252525"
SHEET = "#6D7F91"
ICL = "#F1783A"
SCL = "#29A6B5"
ONSET = "#D62745"
STATE_SHADE = "#F7E9ED"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_pure_zm_candidate(records: list[dict]) -> dict:
    """Select a full-edge high-recruitment trajectory without image review."""
    eligible = []
    for row in records:
        if not row.get("primary_zm_only") or row.get("edge_dose_comparator"):
            continue
        qualification = row.get("model_ictal_qualification") or {}
        clauses = qualification.get("clauses") or {}
        required = (
            clauses.get("complete_one_second_recruitment"),
            clauses.get("joint_broad_recruitment_duty"),
            clauses.get("population_rate_ratio"),
            clauses.get("numerically_safe"),
        )
        if not all(required):
            continue
        if qualification.get("joint_duty") is None:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError(
            "no numerically safe full-edge Z/M candidate passes the "
            "one-second broad-recruitment and population-rate clauses"
        )
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["model_ictal_qualification"]["joint_duty"]),
            -float(row["model_ictal_qualification"].get(
                "contact_centroid_shift_hz", float("-inf")
            )),
            str(row["candidate_id"]),
        ),
    )[0]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _signed_bandpass(raw: np.ndarray, dt_ms: float) -> np.ndarray:
    fs_hz = 1000.0 / float(dt_ms)
    sos = butter(4, (30.0, 80.0), btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(raw, dtype=float), axis=0)


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


def render_figure(record: dict, source_meta: dict,
                  arrays: dict[str, np.ndarray]) -> plt.Figure:
    qualification = record["model_ictal_qualification"]
    onset_ms = float(source_meta["operational_onset_ms"])
    dt_ms = float(arrays["lfp_dt_ms"])
    start_ms = max(0.0, onset_ms - 1000.0)
    stop_ms = min(float(len(arrays["rate_E_hz"]) * dt_ms), onset_ms + 1500.0)

    sample_time_ms = np.arange(len(arrays["rate_E_hz"]), dtype=float) * dt_ms
    shown = (sample_time_ms >= start_ms) & (sample_time_ms <= stop_ms)
    shown_time = sample_time_ms[shown] - onset_ms
    smooth_points = max(1, int(round(20.0 / dt_ms)))
    rate_20ms = uniform_filter1d(
        np.asarray(arrays["rate_E_hz"], dtype=float),
        size=smooth_points,
        mode="nearest",
    )

    recruit_time = np.asarray(arrays["full_field_time_ms"], dtype=float)
    recruit_mask = (recruit_time >= start_ms) & (recruit_time <= stop_ms)
    recruit_rel = recruit_time[recruit_mask] - onset_ms

    filtered = _signed_bandpass(arrays["lfp_trace"], dt_ms)
    pre_scale = ((sample_time_ms >= onset_ms - 500.0)
                 & (sample_time_ms < onset_ms))
    scale = float(np.percentile(np.abs(filtered[pre_scale]), 99.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("pure Z/M trajectory has no finite pre-onset scale")
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = (
        np.asarray(arrays["shaft_ids"]).astype(str)
        if "shaft_ids" in arrays else
        np.asarray(["SCL" if name.startswith("SCL") else "ICL"
                    for name in names])
    )
    order = _contact_order(names)
    traces = 0.68 * filtered[shown][:, order] / scale

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 3.85))
    grid = fig.add_gridspec(
        3, 1,
        height_ratios=(0.82, 0.78, 2.85),
        hspace=0.06,
        left=0.115,
        right=0.985,
        bottom=0.14,
        top=0.80,
    )
    rate_ax = fig.add_subplot(grid[0])
    recruit_ax = fig.add_subplot(grid[1], sharex=rate_ax)
    trace_ax = fig.add_subplot(grid[2], sharex=rate_ax)

    x_min = start_ms - onset_ms
    x_max = stop_ms - onset_ms
    for axis in (rate_ax, recruit_ax, trace_ax):
        axis.axvspan(0.0, x_max, color=STATE_SHADE, lw=0, zorder=0)
        axis.axvline(0.0, color=ONSET, lw=1.0, ls="--", zorder=5)
        axis.set_xlim(x_min, x_max)
        axis.spines[["top", "right"]].set_visible(False)

    rate_ax.plot(shown_time, rate_20ms[shown], color=INK, lw=0.9)
    rate_ax.axhline(120.0, color=ONSET, lw=0.75, ls=":")
    rate_max = max(160.0, float(np.percentile(rate_20ms[shown], 99.5)) * 1.08)
    rate_ax.set_ylim(0.0, rate_max)
    rate_ax.set_ylabel("E rate\n(Hz)", labelpad=5.0)
    rate_ax.tick_params(axis="x", labelbottom=False)
    rate_ax.text(
        x_max - 12.0, 120.0 + 0.025 * rate_max, "120 Hz",
        color=ONSET, fontsize=6.3, ha="right", va="bottom",
    )
    rate_ax.text(
        12.0, 0.90 * rate_max, "operational runaway",
        color=ONSET, fontsize=6.6, ha="left", va="top",
    )

    recruit_ax.plot(
        recruit_rel,
        100.0 * np.asarray(arrays["active_neuron_fraction_20ms"], float)[recruit_mask],
        color=INK,
        lw=0.85,
        label="active E neurons",
    )
    recruit_ax.plot(
        recruit_rel,
        100.0 * np.asarray(arrays["recruited_spatial_fraction_1mm"], float)[recruit_mask],
        color=SHEET,
        lw=0.85,
        label="recruited sheet",
    )
    recruit_ax.axhline(50.0, color=ONSET, lw=0.7, ls=":")
    recruit_ax.set_ylim(0.0, 103.0)
    recruit_ax.set_ylabel("Recruitment\n(%)", labelpad=5.0)
    recruit_ax.tick_params(axis="x", labelbottom=False)
    recruit_ax.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        fontsize=6.3,
        handlelength=1.5,
        columnspacing=0.9,
    )

    offsets = np.arange(len(order), dtype=float) * 1.0
    for row, contact_index in enumerate(order):
        color = SCL if shafts[contact_index] == "SCL" else ICL
        trace_ax.plot(
            shown_time,
            traces[:, row] + offsets[row],
            color=color,
            lw=0.66,
            alpha=0.98,
        )
    trace_ax.set_ylim(-0.72, offsets[-1] + 0.75)
    trace_ax.set_yticks(offsets)
    trace_ax.set_yticklabels(names[order], fontsize=6.5)
    for tick, contact_index in zip(trace_ax.get_yticklabels(), order):
        tick.set_color(SCL if shafts[contact_index] == "SCL" else ICL)
    trace_ax.set_xlabel("Time from operational runaway onset (ms)")
    trace_ax.set_ylabel("Virtual-contact current proxy\n(30–80 Hz)")

    fig.text(0.018, 0.965, "A", fontsize=13.0, fontweight="bold",
             ha="left", va="top")
    fig.text(
        0.115, 0.965,
        "Pure Z/M dynamics on the frozen E1146 scaffold",
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="top",
    )
    base_hz = float(qualification["contact_centroid_base_hz"])
    early_hz = float(qualification["contact_centroid_early_hz"])
    duty = float(qualification["joint_duty"])
    fig.text(
        0.115, 0.905,
        (f"1-s broad-recruitment duty = {duty:.2f}; contact frequency "
         f"{base_hz:.1f} → {early_hz:.1f} Hz (qualification not met)"),
        fontsize=7.2,
        color="0.28",
        ha="left",
        va="top",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rescore",
        default=(
            "results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
            "target_informed_bridge_v1/existing_candidate_rescore.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "results/paper-ready-figure/fig5_pure_zm_dynamics/figures"
        ),
    )
    args = parser.parse_args()

    rescore_path = (ROOT / args.rescore).resolve()
    rescore = _load_json(rescore_path)
    record = select_pure_zm_candidate(rescore["records"])
    source_json = (ROOT / record["source_json"]).resolve()
    source_npz = (ROOT / record["source_npz"]).resolve()
    source_meta = _load_json(source_json)
    arrays = _load_npz(source_npz)

    fig = render_figure(record, source_meta, arrays)
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig5a-pure-zm-high-recruitment"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)

    qualification = record["model_ictal_qualification"]
    source_parameters = dict(record.get("parameters") or {})
    metadata = {
        "status": "PURE_ZM_BROAD_HIGH_RATE_STATE_RENDERED_NOT_FIG5_QUALIFIED",
        "figure": "Fig5A pure Z/M high-recruitment diagnostic candidate",
        "candidate_id": record["candidate_id"],
        "selection": {
            "rule": (
                "among primary_zm_only full-edge candidates passing numerical, "
                "one-second recruitment, broad-duty, and population-rate clauses, "
                "maximize joint_duty; then contact-frequency shift; then candidate_id"
            ),
            "image_pixels_used": False,
            "patient_energy_target_used": False,
        },
        "substrate": {
            "description": "frozen data-driven Node + learned E-to-E + learned E-to-I",
            "E_to_E_dose": 1.0,
            "E_to_I_dose": 1.0,
            "dose_provenance": (
                "primary_zm_only full-edge role; legacy omitted dose fields mean 1.0"
            ),
            "only_varied_mechanism": "Z/M slow-variable parameters",
        },
        "zm_parameters": source_parameters,
        "operational_runaway": {
            "criterion": "population rate >= 120 Hz for >= 100 ms",
            "onset_ms": float(source_meta["operational_onset_ms"]),
        },
        "model_internal_qualification": qualification,
        "interpretation": {
            "supported": (
                "this full-edge Z/M-only trajectory enters a broad high-rate "
                "recruitment state and maintains the registered one-second duty"
            ),
            "not_supported": (
                "the virtual-contact frequency does not increase, so the trajectory "
                "does not reproduce the complete registered Fig5 high-frequency morphology"
            ),
        },
        "source_files": {
            "rescore": str(rescore_path.relative_to(ROOT)),
            "rescore_sha256": _sha256(rescore_path),
            "candidate_json": str(source_json.relative_to(ROOT)),
            "candidate_json_sha256": _sha256(source_json),
            "candidate_npz": str(source_npz.relative_to(ROOT)),
            "candidate_npz_sha256": _sha256(source_npz),
        },
        "outputs": {
            suffix: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for suffix, path in outputs.items()
        },
        "claim_boundary": (
            "single-seed model-internal diagnostic; operational runaway and broad "
            "high-rate recruitment, not clinical seizure reproduction, recovery, "
            "multi-seed confirmation, or a fully qualified Fig5 state"
        ),
    }
    metadata_path = out_dir / "fig5a-pure-zm-high-recruitment-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "### fig5a-pure-zm-high-recruitment.png / .pdf / .svg\n"
        "该图只使用完整保留 learned E→E/E→I 的 full-edge 候选，按冻结的模型内部"
        "资格表选取 1 秒 broad-recruitment duty 最高的 `ith080_ta0062`，没有读取图像"
        "外观或患者能量目标。连续轨迹显示群体率跨过 120 Hz/100 ms 操作性 runaway"
        "门槛，并进入广泛高率招募；但触点频率质心从 16.0 Hz 降到 13.9 Hz，因此"
        "它不是完整 Fig5 高频形态复现。\n\n"
        "**关注点**：这张 Fig5A 证明纯 Z/M 路径里存在操作性 runaway / 广泛高率态，"
        "同时把最关键的失败边界直接写在图上：没有触点频率加速，也没有恢复或多 seed"
        "确认。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "candidate_id": record["candidate_id"],
        "png": str(outputs["png"]),
        "pdf": str(outputs["pdf"]),
        "metadata": str(metadata_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
