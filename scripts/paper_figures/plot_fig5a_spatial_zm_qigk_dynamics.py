#!/usr/bin/env python3
"""Render a confirmed spatial Z/qI--M Fig. 5A without image-based selection."""
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
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
INK = "#252525"
SHEET = "#6D7F91"
ICL = "#F1783A"
SCL = "#29A6B5"
QCOL = "#7B4D6D"
QSUR = "#B78FA6"
MCOL = "#6F7E3C"
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


def select_confirmed_representative(aggregate: dict) -> tuple[dict, dict]:
    """Fail closed unless the aggregator locked an all-pass multi-seed family."""
    family = aggregate.get("primary_hybrid_family")
    record = aggregate.get("primary_hybrid_candidate")
    if not family or not family.get("eligible_multi_seed_family") or not record:
        raise RuntimeError(
            "no eligible multi-seed full-edge hybrid family; formal Fig5A is blocked"
        )
    if not (
        record.get("all_checks_pass")
        and record.get("full_edge")
        and record.get("mode") == "hybrid"
        and record.get("run_role") == "confirmation"
        and record.get("parameter_set_id") == family.get("parameter_set_id")
    ):
        raise RuntimeError("aggregate representative violates the Fig5A contract")
    if int(family.get("n_unique_seeds", 0)) < int(
            family.get("minimum_confirmation_seeds", 3)):
        raise RuntimeError("confirmation family does not contain enough unique seeds")
    if not family.get("single_frozen_config"):
        raise RuntimeError("confirmation parameter-set id contains config drift")
    if record.get("parameter_contract_sha256") != family.get(
            "parameter_contract_sha256"):
        raise RuntimeError("representative config differs from the frozen family config")
    return family, record


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


def render_figure(family: dict, record: dict, source: dict,
                  arrays: dict[str, np.ndarray]) -> plt.Figure:
    onset_ms = float(source["scientific_onset_ms"])
    dt_ms = float(arrays["lfp_dt_ms"])
    sample_time = np.arange(len(arrays["rate_E_hz"]), dtype=float) * dt_ms
    start_ms = max(0.0, onset_ms - 2000.0)
    stop_ms = min(float(sample_time[-1]), onset_ms + 1300.0)
    shown = (sample_time >= start_ms) & (sample_time <= stop_ms)
    time_rel = sample_time[shown] - onset_ms
    x_min, x_max = start_ms - onset_ms, stop_ms - onset_ms

    recruit_time = np.asarray(arrays["full_field_time_ms"], float)
    recruit_mask = (recruit_time >= start_ms) & (recruit_time <= stop_ms)
    recruit_rel = recruit_time[recruit_mask] - onset_ms
    slow_time = np.asarray(arrays["slow_time_ms"], float)
    slow_mask = (slow_time >= start_ms) & (slow_time <= stop_ms)
    slow_rel = slow_time[slow_mask] - onset_ms

    filtered = _signed_bandpass(arrays["lfp_trace"], dt_ms)
    visible = filtered[shown]
    scale = float(np.percentile(np.abs(visible), 98.5))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("confirmed trajectory has no finite 30-80 Hz readout")
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    order = _contact_order(names)
    traces = 0.72 * visible[:, order] / scale

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
    fig = plt.figure(figsize=(7.25, 4.25))
    grid = fig.add_gridspec(
        3, 1, height_ratios=(0.70, 0.82, 2.85), hspace=0.08,
        left=0.12, right=0.97, bottom=0.13, top=0.79,
    )
    slow_ax = fig.add_subplot(grid[0])
    recruit_ax = fig.add_subplot(grid[1], sharex=slow_ax)
    trace_ax = fig.add_subplot(grid[2], sharex=slow_ax)
    for axis in (slow_ax, recruit_ax, trace_ax):
        axis.axvspan(0.0, x_max, color=STATE_SHADE, lw=0, zorder=0)
        axis.axvline(0.0, color=ONSET, lw=1.0, ls="--", zorder=5)
        axis.set_xlim(x_min, x_max)
        axis.spines[["top", "right"]].set_visible(False)

    q_core_line = slow_ax.plot(
        slow_rel,
        100.0 * np.asarray(arrays["slow_q_core_mean"], float)[slow_mask],
        color=QCOL, lw=0.9, label="q, high-h tissue",
    )[0]
    q_surround_line = slow_ax.plot(
        slow_rel,
        100.0 * np.asarray(arrays["slow_q_surround_mean"], float)[slow_mask],
        color=QSUR, lw=0.85, ls="--", label="q, surrounding tissue",
    )[0]
    slow_ax.set_ylim(0.0, 103.0)
    slow_ax.set_ylabel("q (%)", color=QCOL, labelpad=4.0)
    slow_ax.tick_params(axis="y", colors=QCOL)
    slow_ax.tick_params(axis="x", labelbottom=False)
    m_ax = slow_ax.twinx()
    m_line = m_ax.plot(
        slow_rel,
        np.asarray(arrays["slow_adaptation_current_mean"], float)[slow_mask],
        color=MCOL, lw=0.85, label="M/gK current",
    )[0]
    m_hi = max(1.0, 1.08 * float(np.max(m_line.get_ydata(), initial=0.0)))
    m_ax.set_ylim(0.0, m_hi)
    m_ax.set_ylabel("M/gK current (a.u.)", color=MCOL, labelpad=4.0)
    m_ax.tick_params(axis="y", colors=MCOL)
    m_ax.spines["top"].set_visible(False)
    slow_ax.legend(
        [q_core_line, q_surround_line, m_line],
        [q_core_line.get_label(), q_surround_line.get_label(),
         m_line.get_label()],
        loc="upper left", frameon=False, ncol=3, fontsize=5.9,
        handlelength=1.4, columnspacing=0.75,
    )

    recruit_ax.plot(
        recruit_rel,
        100.0 * np.asarray(arrays["active_neuron_fraction_20ms"], float)[recruit_mask],
        color=INK, lw=0.85, label="active E neurons",
    )
    recruit_ax.plot(
        recruit_rel,
        100.0 * np.asarray(arrays["recruited_spatial_fraction_1mm"], float)[recruit_mask],
        color=SHEET, lw=0.85, label="recruited sheet",
    )
    recruit_ax.axhline(50.0, color=ONSET, lw=0.7, ls=":")
    recruit_ax.set_ylim(0.0, 103.0)
    recruit_ax.set_ylabel("Global recruitment\n(%)", labelpad=4.0)
    recruit_ax.tick_params(axis="x", labelbottom=False)
    recruit_ax.legend(
        loc="upper left", frameon=False, ncol=2, fontsize=6.2,
        handlelength=1.5, columnspacing=0.9,
    )

    offsets = np.arange(len(order), dtype=float)
    for row, contact_index in enumerate(order):
        color = SCL if shafts[contact_index] == "SCL" else ICL
        trace_ax.plot(time_rel, traces[:, row] + offsets[row],
                      color=color, lw=0.62)
    trace_ax.set_ylim(-0.75, offsets[-1] + 0.75)
    trace_ax.set_yticks(offsets)
    trace_ax.set_yticklabels(names[order], fontsize=6.3)
    for tick, contact_index in zip(trace_ax.get_yticklabels(), order):
        tick.set_color(SCL if shafts[contact_index] == "SCL" else ICL)
    trace_ax.set_xlabel("Time from model-state transition (ms)")
    trace_ax.set_ylabel("Virtual-contact current proxy\n(30–80 Hz)")
    trace_ax.text(12.0, offsets[-1] + 0.48, "transition", color=ONSET,
                  fontsize=6.5, ha="left", va="top")

    rhythm = source["contact_rhythm"]
    recruitment = source["global_recruitment"]
    fig.text(0.018, 0.965, "A", fontsize=13.0, fontweight="bold",
             ha="left", va="top")
    fig.text(
        0.12, 0.965,
        "Spatial Z/qI–M transition on frozen E1146 connectivity",
        fontsize=9.8, fontweight="bold", ha="left", va="top",
    )
    fig.text(
        0.12, 0.905,
        (f"full learned E→E/E→I  |  {family['n_passed_seeds']}/"
         f"{family['n_unique_seeds']} confirmation seeds  |  "
         f"contact rhythm {float(rhythm['median_contact_peak_hz']):.0f} Hz  |  "
         f"global duty {float(recruitment['joint_global_recruitment_duty']):.2f}"),
        fontsize=7.1, color="0.28", ha="left", va="top",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default=("results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
                 "spatial_zqim_hybrid/confirmation_aggregate.json"),
    )
    parser.add_argument(
        "--out-dir",
        default="results/paper-ready-figure/fig5a_spatial_zm_qigk_dynamics/figures",
    )
    args = parser.parse_args()
    aggregate_path = (ROOT / args.aggregate).resolve()
    aggregate = _load_json(aggregate_path)
    family, record = select_confirmed_representative(aggregate)
    source_json = (ROOT / record["path"]).resolve()
    source_npz = source_json.with_suffix(".npz")
    source = _load_json(source_json)
    arrays = _load_npz(source_npz)
    fig = render_figure(family, record, source, arrays)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig5a-spatial-zm-qigk-global-oscillation"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)

    metadata = {
        "status": "FIG5A_SPATIAL_ZM_QIGK_GLOBAL_OSCILLATION_RENDERED",
        "figure": "Fig5A spatial Z/qI--M model-state transition",
        "parameter_set_id": family["parameter_set_id"],
        "parameter_contract_sha256": family["parameter_contract_sha256"],
        "confirmation_seeds": family["seeds"],
        "confirmation_passed_seeds": family["passed_seeds"],
        "representative_seed": record["seed"],
        "selection": {
            "rule": "all-pass full-edge hybrid family, then median ordered seed",
            "image_pixels_used": False,
            "patient_ictal_waveform_used": False,
        },
        "full_edge_contract": source["full_edge_contract"],
        "hybrid_config": source["hybrid_config"],
        "protocol_contract": source["protocol_contract"],
        "scientific_onset_ms": source["scientific_onset_ms"],
        "state_rate": source["state_rate"],
        "global_recruitment": source["global_recruitment"],
        "contact_rhythm": {
            key: value for key, value in source["contact_rhythm"].items()
            if not key.startswith("per_")
        },
        "source_files": {
            "aggregate": str(aggregate_path.relative_to(ROOT)),
            "aggregate_sha256": _sha256(aggregate_path),
            "candidate_json": str(source_json.relative_to(ROOT)),
            "candidate_json_sha256": _sha256(source_json),
            "candidate_npz": str(source_npz.relative_to(ROOT)),
            "candidate_npz_sha256": _sha256(source_npz),
        },
        "outputs": {
            suffix: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for suffix, path in outputs.items()
        },
        "claim_boundary": (
            "synthetic multi-seed model-state morphology on one frozen patient-derived "
            "scaffold; not clinical seizure reproduction, patient waveform fitting, "
            "mechanism identification, or recovery/termination evidence"
        ),
    }
    metadata_path = out_dir / "fig5a-spatial-zm-qigk-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "### fig5a-spatial-zm-qigk-global-oscillation.png / .pdf / .svg\n"
        "该图显示冻结 E1146 节点场和 learned E→E/E→I 全开的连续模型轨迹：空间"
        "抑制资源 `q` 在高 h 组织与周边形成空间分化、局部空间 `M/gK` 电流响应，随后全局招募并在虚拟触点"
        "形成持续 30–80 Hz 节律。正式代表轨迹只从至少 3 个 confirmation seed 全部"
        "通过同一冻结数值门的参数家族中选择，代表 seed 按排序中位数锁定，不看图挑选。\n\n"
        "**关注点**：这是用户明确要求的连续动力学诊断 panel，因此不采用 Topic 4 默认四列传播布局；"
        "确认转变前有独立低活动驻留期、转变后多数空间和触点持续参与，"
        "同时不要把合成模型态解读成临床发作复现、患者波形拟合、恢复机制或机制鉴定。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "parameter_set_id": family["parameter_set_id"],
        "representative_seed": record["seed"],
        "png": str(outputs["png"]),
        "metadata": str(metadata_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
