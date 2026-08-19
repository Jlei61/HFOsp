#!/usr/bin/env python3
"""Render held-out patient and cohort views after the unattended v2 run."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_patient_specific_field_cohort import (  # noqa: E402
    atomic_json,
    load_config,
    load_subject_contract,
    patient_target_arrays,
    sha256,
    verify_inputs,
)


DEFAULT_CONFIG = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2.json"
MODE_COLORS = ("#d84a3a", "#168aad")


def _mean_rank(ranks: np.ndarray) -> np.ndarray:
    return np.nanmean(np.asarray(ranks, float), axis=0)


def _confirmation_files(output: Path, subject_id: str, winner_id: str) -> list[tuple[Path, Path]]:
    root = output / "per_subject" / subject_id / "workers" / "confirmation"
    rows = []
    for json_path in sorted(root.glob(f"{winner_id}_seed_*.json")):
        npz_path = json_path.with_suffix(".npz")
        if npz_path.exists():
            rows.append((json_path, npz_path))
    if not rows:
        raise RuntimeError(f"no confirmation files for {subject_id}")
    return rows


def _save(fig, stem: Path) -> list[str]:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return [str(stem.with_suffix(".png")), str(stem.with_suffix(".pdf"))]


def _aligned_event_rows(ranks: np.ndarray, natural: dict | None) -> tuple[np.ndarray, np.ndarray]:
    readable = np.isfinite(ranks).sum(axis=1) >= 3
    use = np.asarray(ranks[readable], float)
    labels = np.zeros(len(use), int)
    if natural is not None and natural.get("aligned_labels") is not None:
        labels = np.asarray(natural["aligned_labels"], int)
        if len(labels) != len(use):
            raise RuntimeError("natural KMeans labels do not align with readable events")
    order = np.argsort(labels, kind="stable")
    return use[order], labels[order]


def render_subject(config: dict, subject_id: str) -> dict:
    output = Path(config["output_root"])
    subject_root = output / "per_subject" / subject_id
    selection = json.loads((subject_root / "selection.json").read_text())
    winner_id = selection["winner"]["candidate_id"]
    files = _confirmation_files(output, subject_id, winner_id)
    records = [(json.loads(j.read_text()), n) for j, n in files]
    record, npz_path = max(records, key=lambda row: row[0]["n_returned_events"])
    with np.load(npz_path, allow_pickle=False) as loaded:
        ranks = np.asarray(loaded["ranks"], float)
        onsets = np.asarray(loaded["onsets"], float)
        contact_names = [str(value) for value in loaded["contact_names"]]
        contact_xy = np.asarray(loaded["contact_xy_mm"], float)
        positions = np.asarray(loaded["positions_E"], float)
        h_e = np.asarray(loaded["h_E"], float)
        envelope = np.asarray(loaded["contact_envelope"], float)
        envelope_dt = float(loaded["contact_envelope_dt_ms"])
    natural = record["score"].get("natural_kmeans")
    ordered_ranks, ordered_labels = _aligned_event_rows(ranks, natural)
    target = patient_target_arrays(
        load_subject_contract(config, subject_id)["target_npz_path"], "heldout",
    )["target"]
    figures = subject_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)
    field = axes[0].tricontourf(positions[:, 0], positions[:, 1], h_e, levels=16, cmap="magma")
    axes[0].scatter(contact_xy[:, 0], contact_xy[:, 1], s=18, c="white", edgecolor="black", linewidth=.5)
    readable_onset = np.isfinite(onsets).any(axis=1)
    if np.any(readable_onset):
        onset_values = onsets[readable_onset]
        earliest = np.argmin(np.where(np.isfinite(onset_values), onset_values, np.inf), axis=1)
        counts = np.bincount(earliest, minlength=len(contact_names))
        axes[0].scatter(contact_xy[:, 0], contact_xy[:, 1], s=18 + 7 * counts,
                        facecolor="none", edgecolor="#20b2aa", linewidth=1.0)
    axes[0].set(xlabel="sheet x (mm)", ylabel="sheet y (mm)", title="continuous field")
    fig.colorbar(field, ax=axes[0], label="node field h", fraction=.05)

    if len(ordered_ranks):
        image = np.ma.masked_invalid(ordered_ranks.T)
        axes[1].imshow(image, aspect="auto", interpolation="nearest", cmap="viridis")
        split = int(np.sum(ordered_labels == 0))
        axes[1].axvline(split - .5, color="white", lw=1)
    axes[1].set(xlabel="held-out events", ylabel="electrode contact", title="natural KMeans")
    axes[1].set_yticks(np.arange(len(contact_names)), contact_names, fontsize=6)

    for mode, color in enumerate(MODE_COLORS):
        subset = ordered_ranks[ordered_labels == mode]
        if len(subset):
            axes[2].plot(_mean_rank(subset), np.arange(len(contact_names)), color=color, lw=2,
                         label=f"model T{'AB'[mode]}")
        axes[2].plot(target["profiles"][mode], np.arange(len(contact_names)), color=color,
                     lw=1.5, ls="--", label=f"patient T{'AB'[mode]}")
    axes[2].invert_yaxis()
    axes[2].set(xlabel="mean rank (first to last)", ylabel="electrode contact",
                title="held-out rank profiles")
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    direct_files = _save(fig, figures / "fig4_style_field_and_kmeans")

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    plotted = 0
    readable = np.flatnonzero(np.isfinite(ranks).sum(axis=1) >= 3)
    aligned = np.asarray(natural.get("aligned_labels", []), int) if natural else np.empty(0, int)
    events = record.get("events", [])
    for mode, color in enumerate(MODE_COLORS):
        candidates = readable[aligned == mode] if len(aligned) == len(readable) else readable
        if not len(candidates):
            continue
        event_index = int(candidates[0])
        event = events[event_index]
        center = 0.5 * (float(event["t_on_ms"]) + float(event["t_off_ms"]))
        start = max(0, int(round((center - 100.0) / envelope_dt)))
        stop = min(envelope.shape[1], int(round((center + 180.0) / envelope_dt)))
        time_axis = np.arange(stop - start) * envelope_dt + plotted * 320.0
        scale = np.nanpercentile(np.abs(envelope[:, start:stop]), 95) or 1.0
        for index, trace in enumerate(envelope[:, start:stop]):
            ax.plot(time_axis, trace / scale + index, color=color, lw=.75, alpha=.85)
        plotted += 1
    ax.set_yticks(np.arange(len(contact_names)), contact_names, fontsize=7)
    ax.set(xlabel="time from representative held-out events (ms)",
           ylabel="virtual electrode contact", title="direct model-current readout")
    waveform_files = _save(fig, figures / "fig4_style_direct_waveforms")

    readme = figures / "README.md"
    readme.write_text(
        "### fig4_style_field_and_kmeans.png / .pdf\n"
        "展示该患者独立拟合的连续 node field、held-out 事件的自然 KMeans 分群，"
        "以及模型与患者 held-out 两个传播模板的触点秩次曲线。圆环大小表示事件最早触点密度。\n\n"
        "**关注点**：两个自然簇是否分别接近患者 TA/TB，而不是只恢复一个模板。\n\n"
        "### fig4_style_direct_waveforms.png / .pdf\n"
        "展示同一冻结 winner 在确认网络中的两类代表事件虚拟电极读出。\n\n"
        "**关注点**：波形仅为 model-current proxy，用于核对触点招募顺序，不代表临床 SEEG 振幅复现。\n"
    )
    metadata = {
        "subject_id": subject_id, "winner_id": winner_id,
        "confirmation_seed": record["seed"], "confirmation_npz": str(npz_path),
        "confirmation_npz_sha256": sha256(npz_path),
        "files": direct_files + waveform_files,
        "heldout_only": True, "figure_does_not_select_candidate": True,
    }
    atomic_json(metadata, figures / "metadata.json")
    return metadata


def render_cohort(config: dict, result: dict) -> dict:
    output = Path(config["output_root"])
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = [row for row in result["subjects"] if not row["development_source"]]
    delta = np.asarray([row["heldout_null_advantage"] for row in rows], float)
    ood = np.asarray([row["mean_ood_fraction"] for row in rows], float)
    k2 = np.asarray([row["same_network_k2_count_of_4"] for row in rows], int)
    order = np.argsort(delta)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.1), constrained_layout=True)
    colors = np.where(delta[order] > 0, "#2878b5", "#b8b8b8")
    axes[0].barh(np.arange(len(rows)), delta[order], color=colors, height=.76)
    axes[0].axvline(0, color="black", lw=.8)
    axes[0].set(xlabel="null loss - model loss", ylabel="patients",
                title="held-out match")
    axes[0].set_yticks([])
    axes[1].bar(["0/4", "1/4", "2/4", "3/4", "4/4"],
                [int(np.sum(k2 == value)) for value in range(5)], color="#168aad")
    axes[1].set(xlabel="confirming networks", ylabel="patients",
                title="two-mode recovery")
    axes[2].scatter(ood, delta, c=np.where(k2 >= 3, "#d84a3a", "#808080"), s=34)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set(xlabel="OOD event fraction", ylabel="null loss - model loss",
                title="support vs match")
    files = _save(fig, figures / "patient_specific_cohort_heldout")
    (figures / "README.md").write_text(
        "### patient_specific_cohort_heldout.png / .pdf\n"
        "以患者为独立单位汇总 27 位非开发来源患者的 held-out 结果。左图为模型相对杆内触点身份"
        "置换 null 的损失优势，中图为四张确认网络中自然 KMeans 双模式通过次数，右图显示患者"
        "支持范围与匹配优势的关系。\n\n"
        "**关注点**：正值表示模型优于配对 null；红点表示至少 3/4 确认网络恢复两种传播模式。\n"
    )
    metadata = {
        "files": files, "n_real_geometry_fitted": result["n_real_geometry_fitted"],
        "n_primary_nondevelopment": len(rows), "subject_is_inference_unit": True,
        "heldout_only": True,
    }
    atomic_json(metadata, figures / "metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    verify_inputs(config, code_root=ROOT)
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != expected:
        raise RuntimeError("finalizer HEAD differs from expected commit")
    result_path = Path(config["output_root"]) / "COHORT_RESULT.json"
    result = json.loads(result_path.read_text())
    subjects = [row["subject_id"] for row in result["subjects"]]
    per_subject = [render_subject(config, subject_id) for subject_id in subjects]
    cohort = render_cohort(config, result)
    atomic_json({
        "status": "PATIENT_SPECIFIC_COHORT_COMPLETE",
        "expected_git_commit": expected,
        "cohort_result_sha256": sha256(result_path),
        "n_subject_figures": len(per_subject),
        "cohort_figure": cohort,
    }, Path(config["output_root"]) / "DONE.json")


if __name__ == "__main__":
    main()
