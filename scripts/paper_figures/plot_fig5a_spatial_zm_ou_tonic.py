#!/usr/bin/env python3
"""Render Fig. 5A for the spatial Z/M + stationary-OU tonic runaway."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


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


def select_tonic_representative(aggregate: dict, *, allow_discovery_preview=False):
    """Select a formal confirmation record or an explicitly labelled preview."""
    family = aggregate.get("primary_confirmation_family")
    record = aggregate.get("primary_confirmation_candidate")
    if family is not None and record is not None:
        if not family.get("eligible_multi_seed_family"):
            raise RuntimeError("primary tonic family is not multi-seed eligible")
        if int(family.get("n_unique_seeds", 0)) < int(
                family.get("minimum_confirmation_seeds", 3)):
            raise RuntimeError("tonic confirmation family has too few seeds")
        if not family.get("single_frozen_config"):
            raise RuntimeError("tonic confirmation family contains config drift")
        if not (
            record.get("all_checks_pass")
            and record.get("run_role") == "confirmation"
            and record.get("mode") == "hybrid"
            and record.get("parameter_set_id") == family.get("parameter_set_id")
            and record.get("parameter_contract_sha256")
            == family.get("parameter_contract_sha256")
        ):
            raise RuntimeError("formal tonic representative violates its family")
        return family, record, False
    if not allow_discovery_preview:
        raise RuntimeError(
            "no eligible multi-seed tonic confirmation family; formal Fig5A is blocked")
    record = aggregate.get("primary_discovery_candidate")
    if not record or not (
            record.get("all_checks_pass")
            and record.get("run_role") == "discovery"
            and record.get("mode") == "hybrid"):
        raise RuntimeError("no eligible tonic discovery trajectory for preview")
    return None, record, True


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


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


def _normalised_contact_plateau(raw, *, dt_ms, onset_ms):
    """Show the tonic step without detrending or band-pass removal."""
    raw = np.asarray(raw, float)
    smooth_steps = max(1, int(round(5.0 / float(dt_ms))))
    smooth = uniform_filter1d(raw, size=smooth_steps, axis=0, mode="nearest")
    time = np.arange(len(smooth), dtype=float) * float(dt_ms)
    pre = (time >= max(0.0, float(onset_ms) - 500.0)) & (time < float(onset_ms))
    post = ((time >= float(onset_ms) + 300.0)
            & (time < float(onset_ms) + 1300.0))
    if not np.any(pre) or not np.any(post):
        raise RuntimeError("candidate lacks complete contact normalisation windows")
    baseline = np.median(smooth[pre], axis=0)
    plateau = np.median(smooth[post], axis=0)
    scale = plateau - baseline
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-9):
        raise RuntimeError("contact current proxy does not rise into a plateau")
    return (smooth - baseline[None, :]) / scale[None, :]


def render_figure(record, source, arrays, *, preview=False, family=None):
    onset_ms = float(source["scientific_onset_ms"])
    dt_ms = float(arrays["lfp_dt_ms"])
    time = np.arange(len(arrays["rate_E_hz"]), dtype=float) * dt_ms
    x_min, x_max = 0.0, float(time[-1])
    slow_time = np.asarray(arrays["slow_time_ms"], float)
    recruit_time = np.asarray(arrays["full_field_time_ms"], float)
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    order = _contact_order(names)
    contact = _normalised_contact_plateau(
        arrays["lfp_trace"], dt_ms=dt_ms, onset_ms=onset_ms)[:, order]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 5.25))
    grid = fig.add_gridspec(
        4, 1, height_ratios=(0.72, 0.72, 0.72, 2.75), hspace=0.20,
        left=0.13, right=0.93, bottom=0.105, top=0.81,
    )
    slow_ax = fig.add_subplot(grid[0])
    rate_ax = fig.add_subplot(grid[1], sharex=slow_ax)
    recruit_ax = fig.add_subplot(grid[2], sharex=slow_ax)
    trace_ax = fig.add_subplot(grid[3], sharex=slow_ax)
    for axis in (slow_ax, rate_ax, recruit_ax, trace_ax):
        axis.axvspan(onset_ms, x_max, color=STATE_SHADE, lw=0, zorder=0)
        axis.axvline(onset_ms, color=ONSET, lw=1.0, ls="--", zorder=8)
        axis.set_xlim(x_min, x_max)
        axis.spines[["top", "right"]].set_visible(False)

    q_core = slow_ax.plot(
        slow_time, 100.0 * np.asarray(arrays["slow_q_core_mean"], float),
        color=QCOL, lw=0.9, label="Z/q, high-h tissue",
    )[0]
    q_surround = slow_ax.plot(
        slow_time, 100.0 * np.asarray(arrays["slow_q_surround_mean"], float),
        color=QSUR, lw=0.85, ls="--", label="Z/q, surround",
    )[0]
    slow_ax.set_ylim(70.0, 102.0)
    slow_ax.set_ylabel("Z/q (%)", color=QCOL, labelpad=7.0,
                       fontsize=6.9)
    slow_ax.tick_params(axis="y", colors=QCOL)
    slow_ax.tick_params(axis="x", labelbottom=False)
    m_ax = slow_ax.twinx()
    m_line = m_ax.plot(
        slow_time,
        np.asarray(arrays["slow_adaptation_current_mean"], float),
        color=MCOL, lw=0.85, label="M/gK current",
    )[0]
    m_hi = max(0.12, 1.08 * float(np.max(m_line.get_ydata(), initial=0.0)))
    m_ax.set_ylim(0.0, m_hi)
    m_ax.set_ylabel("M/gK (a.u.)", color=MCOL, labelpad=4.0, fontsize=6.9)
    m_ax.tick_params(axis="y", colors=MCOL)
    m_ax.spines["top"].set_visible(False)
    slow_ax.legend(
        [q_core, q_surround, m_line],
        [q_core.get_label(), q_surround.get_label(), m_line.get_label()],
        loc="lower right", bbox_to_anchor=(1.0, 1.08), frameon=False,
        ncol=3, fontsize=5.8,
        handlelength=1.35, columnspacing=0.75,
    )

    smooth_steps = max(1, int(round(20.0 / dt_ms)))
    smoothed_rate = uniform_filter1d(
        np.asarray(arrays["rate_E_hz"], float), size=smooth_steps,
        mode="nearest")
    rate_ax.plot(time, smoothed_rate, color=INK, lw=0.85)
    rate_ax.axhline(120.0, color=ONSET, lw=0.65, ls=":")
    rate_ax.set_ylim(0.0, 500.0)
    rate_ax.set_ylabel("E rate (Hz)", labelpad=7.0, fontsize=6.9)
    rate_ax.tick_params(axis="x", labelbottom=False)
    rate_ax.text(
        onset_ms + 18.0, 478.0, "tonic transition", color=ONSET,
        fontsize=6.5, ha="left", va="top",
    )

    recruit_ax.plot(
        recruit_time,
        100.0 * np.asarray(arrays["active_neuron_fraction_20ms"], float),
        color=INK, lw=0.85, label="active E neurons",
    )
    recruit_ax.plot(
        recruit_time,
        100.0 * np.asarray(arrays["recruited_spatial_fraction_1mm"], float),
        color=SHEET, lw=0.85, label="recruited sheet",
    )
    recruit_ax.set_ylim(0.0, 103.0)
    recruit_ax.set_ylabel("recruitment (%)", labelpad=7.0, fontsize=6.9)
    recruit_ax.tick_params(axis="x", labelbottom=False)
    recruit_ax.legend(
        loc="lower right", frameon=False, ncol=2, fontsize=5.9,
        handlelength=1.4, columnspacing=0.8,
    )

    offsets = np.arange(len(order), dtype=float) * 1.18
    for row, contact_index in enumerate(order):
        color = SCL if shafts[contact_index] == "SCL" else ICL
        trace_ax.plot(
            time, np.clip(contact[:, row], -0.25, 1.25) + offsets[row],
            color=color, lw=0.62,
        )
    trace_ax.set_ylim(-0.6, offsets[-1] + 1.5)
    trace_ax.set_yticks(offsets)
    trace_ax.set_yticklabels(names[order], fontsize=6.2)
    for tick, contact_index in zip(trace_ax.get_yticklabels(), order):
        tick.set_color(SCL if shafts[contact_index] == "SCL" else ICL)
    trace_ax.set_xlabel("Time in continuous trajectory (ms)")
    trace_ax.set_ylabel("virtual-contact current proxy\n(normalized tonic level)",
                        fontsize=6.9)

    rate = source["state_rate"]
    recruitment = source["global_recruitment"]
    contact_rows = source.get("per_contact_diagnosis") or []
    n_contact = sum(
        float(row.get("local_rate_post_hz", 0.0)) >= 120.0
        and float(row.get("local_rate_ratio_post_over_pre", 0.0)) >= 2.0
        for row in contact_rows)
    fig.text(0.018, 0.972, "A", fontsize=13.0, fontweight="bold",
             ha="left", va="top")
    fig.text(
        0.13, 0.972,
        "Spatial Z/M under stationary OU enters a tonic global runaway",
        fontsize=9.6, fontweight="bold", ha="left", va="top",
    )
    confirmation_text = ""
    if family is not None:
        confirmation_text = (
            f"{int(family['n_passed_seeds'])}/"
            f"{int(family['n_unique_seeds'])} confirmation seeds  |  ")
    fig.text(
        0.13, 0.918,
        (f"full learned E→E/E→I  |  {confirmation_text}post rate "
         f"{float(rate['median_post_hz']):.0f} Hz  |  global duty "
         f"{float(recruitment['joint_global_recruitment_duty']):.2f}  |  "
         f"{n_contact}/15 contacts recruited"),
        fontsize=7.0, color="0.28", ha="left", va="top",
    )
    if preview:
        fig.text(0.97, 0.972, "discovery preview", fontsize=6.2,
                 color="0.45", ha="right", va="top")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default=("results/topic4_sef_hfo/data_driven_zm_ictal_transition/"
                 "spatial_zm_ou/tonic_runaway_aggregate.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=("results/paper-ready-figure/"
                 "fig5a_spatial_zm_ou_tonic/figures"),
    )
    parser.add_argument("--allow-discovery-preview", action="store_true")
    args = parser.parse_args()

    aggregate_path = (ROOT / args.aggregate).resolve()
    aggregate = _load_json(aggregate_path)
    family, record, preview = select_tonic_representative(
        aggregate, allow_discovery_preview=args.allow_discovery_preview)
    source_json = (ROOT / record["path"]).resolve()
    source_npz = source_json.with_suffix(".npz")
    source = _load_json(source_json)
    arrays = _load_npz(source_npz)
    fig = render_figure(
        record, source, arrays, preview=preview, family=family)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "fig5a-spatial-zm-ou-tonic-global-runaway"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)

    metadata = {
        "status": ("FIG5A_TONIC_RUNAWAY_DISCOVERY_PREVIEW_RENDERED" if preview
                   else "FIG5A_TONIC_RUNAWAY_CONFIRMED_RENDERED"),
        "endpoint": "persistent near-saturated tonic global runaway",
        "preview": preview,
        "endpoint_does_not_require": [
            "30-80 Hz contact rhythm",
            "deep population-rate modulation",
        ],
        "tonic_morphology_thresholds": aggregate[
            "tonic_morphology_thresholds"],
        "virtual_contact_recruitment_thresholds": aggregate[
            "virtual_contact_recruitment_thresholds"],
        "representative_seed": record["seed"],
        "parameter_set_id": record.get("parameter_set_id"),
        "confirmation_family": family,
        "selection": {
            "rule": (aggregate.get("discovery_selection_rule") if preview else
                     "eligible all-pass multi-seed confirmation family; seed nearest median onset"),
            "image_pixels_used": False,
        },
        "display_transforms": {
            "population_rate": "20-ms uniform smoothing; raw Hz retained",
            "global_recruitment": "raw archived 20-ms-window metrics",
            "virtual_contact_proxy": (
                "5-ms uniform smoothing; each contact mapped so its pre-onset "
                "median is 0 and onset+300..1300-ms median is 1; displayed at "
                "[-0.25,1.25] without detrending or band-pass filtering"),
        },
        "full_edge_contract": source["full_edge_contract"],
        "hybrid_config": source["hybrid_config"],
        "applied_spatial_ou": source["applied_spatial_ou"],
        "protocol_contract": source["protocol_contract"],
        "scientific_onset_ms": source["scientific_onset_ms"],
        "state_rate": source["state_rate"],
        "global_recruitment": source["global_recruitment"],
        "source_files": {
            "aggregate": str(aggregate_path.relative_to(ROOT)),
            "aggregate_sha256": _sha256(aggregate_path),
            "candidate_json": str(source_json.relative_to(ROOT)),
            "candidate_json_sha256": _sha256(source_json),
            "candidate_npz": str(source_npz.relative_to(ROOT)),
            "candidate_npz_sha256": _sha256(source_npz),
            "figure_producer": str(Path(__file__).resolve().relative_to(ROOT)),
            "figure_producer_sha256": _sha256(Path(__file__).resolve()),
            "aggregate_producer": (
                "scripts/aggregate_topic4_spatial_zm_ou_tonic.py"),
            "aggregate_producer_sha256": _sha256(
                ROOT / "scripts/aggregate_topic4_spatial_zm_ou_tonic.py"),
            "transition_runner": "scripts/run_topic4_spatial_zm_ou_transition.py",
            "transition_runner_sha256": _sha256(
                ROOT / "scripts/run_topic4_spatial_zm_ou_transition.py"),
            "tonic_classifier": "src/topic4_global_recruited_oscillation.py",
            "tonic_classifier_sha256": _sha256(
                ROOT / "src/topic4_global_recruited_oscillation.py"),
        },
        "outputs": {
            suffix: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for suffix, path in outputs.items()
        },
        "claim_boundary": (
            "synthetic tonic model-state transition on one frozen patient-derived "
            "scaffold; not clinical seizure reproduction, patient waveform fitting, "
            "or biological mechanism identification"),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    metadata_path = out_dir / "fig5a-spatial-zm-ou-tonic-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    readme = (
        "### fig5a-spatial-zm-ou-tonic-global-runaway.png / .pdf / .svg\n"
        "该图展示冻结 E1146 data-driven 底物在持续平稳空间 OU 背景下，由空间 "
        "Z/M 慢变量从低活动进入近饱和 tonic runaway。上三行依次显示慢变量、"
        "群体放电率和全片招募；底行保留虚拟触点 current proxy 的 tonic level，"
        "不做会把平台减掉的 detrend 或 30–80 Hz band-pass。"
        + ("当前文件是 discovery preview，正式 Fig5A 仍等待同一冻结参数的 3 个 "
           "confirmation seed 全部通过。\n\n" if preview else
           "代表轨迹来自至少 3 个同一冻结参数的 confirmation seed，并按 onset "
           "中位数选取，不看图挑选。\n\n")
        + "**关注点**：转变后群体率接近不应期上限、神经元和空间招募接近 100%，"
          "15/15 触点均升入持续高态；本图不要求深调制或 30–80 Hz 节律，也不能"
          "解读成临床发作波形或患者机制复现。\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": metadata["status"],
        "representative_seed": record["seed"],
        "png": str(outputs["png"]),
        "metadata": str(metadata_path),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
