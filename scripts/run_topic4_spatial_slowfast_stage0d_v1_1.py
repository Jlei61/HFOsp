#!/usr/bin/env python3
"""Run the independent Stage-0D v1.1 engineering repair."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0d-v1-1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_topic4_spatial_slowfast_stage0d as v1_runner  # noqa: E402
from scripts.run_topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    _atomic_json,
    _atomic_text,
    _sha256,
    _write_csv,
)
from src.topic4_spatial_slowfast_stage0d_v1_1 import (  # noqa: E402
    FIGURE_B_EMPTY_TEXT,
    centre_final_survivor_indices,
    compare_fork_outcomes,
    strict_temporal_amplitude_status,
)


DEFAULT_CONFIG = ROOT / "config/topic4_spatial_slowfast_stage0d_v1_1.yaml"
SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0d-local-basin-replication-v1_1-engineering-repair-design.md"
MODULE = ROOT / "src/topic4_spatial_slowfast_stage0d_v1_1.py"
RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0d_v1_1.py"
TESTS = ROOT / "tests/test_topic4_spatial_slowfast_stage0d_v1_1.py"

V1_PATHS = {
    "spec": ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0d-local-basin-replication-design.md",
    "config": ROOT / "config/topic4_spatial_slowfast_stage0d.yaml",
    "module": ROOT / "src/topic4_spatial_slowfast_stage0d.py",
    "runner": ROOT / "scripts/run_topic4_spatial_slowfast_stage0d.py",
    "tests": ROOT / "tests/test_topic4_spatial_slowfast_stage0d.py",
}


def _v1_artifact_paths(cfg: dict) -> dict[str, Path]:
    root = ROOT / cfg["stage0d_v1_result_root"]
    return {
        **V1_PATHS,
        "summary": root / "stage0d_local_basin_summary.json",
        "fork_outcomes": root / "fork_outcomes.json",
        "phase_source": root / "phase_source.json",
    }


def _validate_config(cfg: dict) -> dict:
    """Validate v1.1 and prove every scientific section equals v1."""

    v1_runner._validate_config(cfg)
    v1_config_path = ROOT / cfg["stage0d_v1_config"]
    v1_cfg = yaml.safe_load(v1_config_path.read_text(encoding="utf-8"))
    scientific_sections = (
        "root_continuation",
        "v1_1_root",
        "locked_hashes",
        "centre",
        "parameter_grid",
        "phase_source",
        "battery",
        "screen",
        "confirm",
        "dt_half",
        "classifier",
        "acceptance",
        "resource_contract",
        "scope",
    )
    mismatches = [name for name in scientific_sections if cfg.get(name) != v1_cfg.get(name)]
    if mismatches:
        raise ValueError(f"v1.1 scientific config drifted from v1: {mismatches}")
    expected_repair = {
        "confirm_dt_half_frequency_abs_hz": 0.25,
        "confirm_dt_half_frequency_relative": 0.10,
        "figure_b_empty_text": FIGURE_B_EMPTY_TEXT,
    }
    if cfg.get("repair") != expected_repair:
        raise ValueError("v1.1 repair contract drifted")
    if Path(cfg["result_root"]).name != "stage0d_local_basin_replication_v1_1":
        raise ValueError("v1.1 result root drifted")
    return v1_cfg


def _verify_v1_immutable(cfg: dict) -> dict[str, dict[str, str | bool]]:
    rows: dict[str, dict[str, str | bool]] = {}
    paths = _v1_artifact_paths(cfg)
    for name, path in paths.items():
        observed = _sha256(path)
        expected = str(cfg["stage0d_v1_immutable_hashes"][name])
        rows[name] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "pass": observed == expected,
        }
    if not all(bool(row["pass"]) for row in rows.values()):
        raise RuntimeError("immutable Stage0D v1 lineage hash mismatch")
    return rows


def _source_paths(config_path: Path) -> dict[str, Path]:
    return {
        "spec": SPEC,
        "config": config_path,
        "module": MODULE,
        "runner": RUNNER,
        "tests": TESTS,
    }


def _source_hashes(config_path: Path) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in _source_paths(config_path).items()
    }


def _write_execution_lock(
    output: Path,
    config_path: Path,
    v1_immutable: dict,
) -> dict:
    lock = {
        "schema_version": "topic4_stage0d_v1_1_execution_lock.v1",
        "locked_before_numerical_execution": True,
        "v1_immutable_inputs": v1_immutable,
        "v1_1_sources_pre_execution": _source_hashes(config_path),
        "v1_1_sources_post_execution": None,
        "sources_unchanged_during_execution": None,
    }
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return lock


def _finalize_execution_lock(output: Path, config_path: Path, lock: dict) -> bool:
    observed = _source_hashes(config_path)
    unchanged = observed == lock["v1_1_sources_pre_execution"]
    lock["v1_1_sources_post_execution"] = observed
    lock["sources_unchanged_during_execution"] = unchanged
    _atomic_json(output / "EXECUTION_LOCK.json", lock)
    return bool(unchanged)


def _plot_v1_1(output: Path, summary: dict, final_rows: list[dict], point_rows: list[dict]) -> dict:
    """Render the repaired figure; Figure B uses final centre survivors only."""

    phase = json.loads((output / "phase_source.json").read_text(encoding="utf-8"))
    source = np.load(output / "phase_source_traces.npz")
    refined_path = output / "dt_half_traces.npz"
    refined = np.load(refined_path) if refined_path.exists() else None
    dt_original = [index for index, row in enumerate(final_rows) if row.get("dt_half_classification") is not None]
    dt_lookup = {original: local for local, original in enumerate(dt_original)}
    centre_indices = centre_final_survivor_indices(final_rows)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(source["time_ms"] / 1000.0, 1000.0 * source["rE_khz"][:, 0], color="#3b528b", lw=1.0)
    for row in phase["phase_selection"]:
        ax.axvline(float(row["time_ms"]) / 1000.0, color="#e76f51", lw=0.85, alpha=0.85)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="A  Locked phase source")

    ax = axes[0, 1]
    ax.set_title("B  Centre dt/2 survivors")
    if centre_indices and refined is not None:
        colors = plt.cm.viridis(np.linspace(0.15, 0.90, len(centre_indices)))
        for color, original in zip(colors, centre_indices):
            local = dt_lookup[original]
            row = final_rows[original]
            ax.plot(
                refined["time_ms"] / 1000.0,
                1000.0 * refined["rE_khz"][:, local],
                color=color,
                lw=0.75,
                alpha=0.8,
                label=f"{row['phase_id']} {row['history']}",
            )
        ax.set(xlabel="time (s)", ylabel="E rate (Hz)")
        ax.legend(frameon=False, fontsize=7)
        ax.text(0.02, 0.95, f"n={len(centre_indices)}", transform=ax.transAxes, va="top")
    else:
        ax.set_axis_off()
        ax.set_title("B  Centre dt/2 survivors")
        ax.text(
            0.5,
            0.5,
            FIGURE_B_EMPTY_TEXT,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color="#9d3d38",
            weight="bold",
        )

    ax = axes[1, 0]
    grid = np.zeros((3, 3), dtype=float)
    for row in point_rows:
        zi = (0.84, 0.85, 0.86).index(round(float(row["z"]), 2))
        ai = (15.0, 16.0, 17.0).index(float(row["alpha_G"]))
        grid[ai, zi] = float(row["n_off_orbit_survivors"])
    image = ax.imshow(grid, origin="lower", cmap="viridis", vmin=0, vmax=16, aspect="auto")
    for ai in range(3):
        for zi in range(3):
            ax.text(zi, ai, f"{int(grid[ai, zi])}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(3), ("0.84", "0.85", "0.86"))
    ax.set_yticks(range(3), ("15", "16", "17"))
    ax.set(xlabel="z", ylabel=r"$\alpha_G$", title="C  Off-orbit final survivors")
    fig.colorbar(image, ax=ax, shrink=0.78)

    ax = axes[1, 1]
    status_counts = summary["final_status_counts"]
    labels = list(status_counts)
    values = [status_counts[label] for label in labels]
    ax.bar(range(len(labels)), values, color=["#9e9e9e" if "unresolved" in label else "#21918c" for label in labels])
    ax.set_xticks(range(len(labels)), [label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set(ylabel="histories", title="D  Final locked outcomes")
    for index, value in enumerate(values):
        ax.text(index, value + max(values) * 0.015, str(value), ha="center", fontsize=9)
    fig.suptitle(summary["verdict"].replace("_", " "), fontsize=11)
    png = figures / "stage0d_local_basin_replication_v1_1.png"
    pdf = figures / "stage0d_local_basin_replication_v1_1.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    # Overwrite the transient v1-engine filenames in this *new* result root so
    # no semantically empty Figure B remains beside the repaired figure.
    fig.savefig(figures / "stage0d_local_basin_replication.png", dpi=220)
    fig.savefig(figures / "stage0d_local_basin_replication.pdf")
    plt.close(fig)

    if centre_indices:
        panel_b_cn = f"B 显示中心点最终通过锁定 dt/2 gate 的 {len(centre_indices)} 条轨迹。"
    else:
        panel_b_cn = "B 明确显示 `none passed locked gate`：中心点没有任何轨迹通过最终锁定 dt/2 gate，未用邻点或较早阶段轨迹替代。"
    _atomic_text(
        figures / "README.md",
        "### stage0d_local_basin_replication_v1_1.png\n\n"
        "A 显示冻结 phase source 与四个选定相位。"
        + panel_b_cn
        + " C 显示 3×3 邻域每点的 off-orbit final survivor 数；D 汇总全部 180 条历史的最终锁定状态。\n\n"
        "**关注点**：Figure B 只接受中心点 `final_status=candidate_survives` 的 dt/2 轨迹；空面板文本是科学结果，不是缺图。\n",
    )
    metadata = {
        "schema_version": "topic4_stage0d_v1_1_figure.v1",
        "centre_final_survivor_count": len(centre_indices),
        "figure_b_empty_text_rendered": FIGURE_B_EMPTY_TEXT if not centre_indices else None,
        "figure_b_uses_only_final_centre_survivors": True,
        "png": str(png.resolve()),
        "png_sha256": _sha256(png),
        "pdf": str(pdf.resolve()),
        "pdf_sha256": _sha256(pdf),
    }
    _atomic_json(output / "figure_metadata.json", metadata)
    return metadata


def _comparison(
    cfg: dict,
    v1_summary: dict,
    repaired_summary: dict,
    v1_rows: list[dict],
    repaired_rows: list[dict],
) -> dict:
    fork = compare_fork_outcomes(v1_rows, repaired_rows)
    verdict_changed = str(v1_summary["verdict"]) != str(repaired_summary["verdict"])
    counts_changed = v1_summary["final_status_counts"] != repaired_summary["final_status_counts"]
    centre_changed = bool(v1_summary["centre_open_local_basin_support"]) != bool(
        repaired_summary["centre_open_local_basin_support"]
    )
    neighbours_changed = int(v1_summary["n_compatible_manhattan_neighbours"]) != int(
        repaired_summary["n_compatible_manhattan_neighbours"]
    )
    scientific_changed = bool(
        fork["any_fork_status_changed"] or verdict_changed or counts_changed or centre_changed or neighbours_changed
    )
    return {
        "schema_version": "topic4_stage0d_v1_vs_v1_1.v1",
        "v1_summary": str((ROOT / cfg["stage0d_v1_result_root"] / "stage0d_local_basin_summary.json").resolve()),
        "v1_verdict": v1_summary["verdict"],
        "v1_1_verdict": repaired_summary["verdict"],
        "verdict_changed": verdict_changed,
        "v1_final_status_counts": v1_summary["final_status_counts"],
        "v1_1_final_status_counts": repaired_summary["final_status_counts"],
        "counts_changed": counts_changed,
        "centre_open_basin_changed": centre_changed,
        "compatible_neighbour_count_changed": neighbours_changed,
        **fork,
        "scientific_result_changed": scientific_changed,
        "interpretation_cn": (
            "v1.1 严格 frequency gate 改变了至少一个最终状态或科学判定。"
            if scientific_changed
            else "v1.1 修复了实现偏差，但最终 fork 状态、open-basin 判定和科学 verdict 均未改变。"
        ),
    }


def run(config_path: Path) -> tuple[dict, Path]:
    wrapper_start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(cfg)
    v1_immutable = _verify_v1_immutable(cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    lock = _write_execution_lock(output, config_path, v1_immutable)

    strict_audits: dict[tuple[float, float, str, str], dict] = {}

    def strict_adapter(confirm_row, refined_row, *, exact_error_pass: bool) -> str:
        status, audit = strict_temporal_amplitude_status(
            confirm_row,
            refined_row,
            exact_error_pass=exact_error_pass,
        )
        key = (
            round(float(confirm_row["z"]), 2),
            float(confirm_row["alpha_G"]),
            str(confirm_row["phase_id"]),
            str(confirm_row["history"]),
        )
        strict_audits[key] = audit
        return status

    original_temporal_gate = v1_runner.temporal_amplitude_status
    v1_runner.temporal_amplitude_status = strict_adapter
    try:
        raw_summary, observed_output = v1_runner.run(config_path)
    finally:
        v1_runner.temporal_amplitude_status = original_temporal_gate
    if observed_output.resolve() != output.resolve():
        raise RuntimeError("v1.1 numeric engine wrote to the wrong root")

    final_path = output / "fork_outcomes.json"
    final_rows = json.loads(final_path.read_text(encoding="utf-8"))
    for row in final_rows:
        key = (round(float(row["z"]), 2), float(row["alpha_G"]), str(row["phase_id"]), str(row["history"]))
        row["strict_confirm_dt_half_gate"] = strict_audits.get(key)
    _atomic_json(final_path, final_rows)
    _write_csv(output / "fork_outcomes.csv", final_rows)
    point_rows = json.loads((output / "parameter_point_outcomes.json").read_text(encoding="utf-8"))

    v1_root = ROOT / cfg["stage0d_v1_result_root"]
    v1_summary = json.loads((v1_root / "stage0d_local_basin_summary.json").read_text(encoding="utf-8"))
    v1_rows = json.loads((v1_root / "fork_outcomes.json").read_text(encoding="utf-8"))
    comparison = _comparison(cfg, v1_summary, raw_summary, v1_rows, final_rows)
    _atomic_json(output / "v1_vs_v1_1_comparison.json", comparison)

    final_summary = {
        **raw_summary,
        "schema_version": "topic4_stage0d_local_basin.v1_1",
        "engineering_repair": {
            "v1_deviation": (
                "v1 confirm-vs-dt/2 frequency implementation used legacy max(0.5 Hz,15%) instead of "
                "the written max(0.25 Hz,10%) gate; authoritative Figure B was also semantically empty."
            ),
            "v1_1_frequency_gate": "max(0.25 Hz,10% of pair mean)",
            "rate_gate_unchanged": "max(1 Hz,10% of pair mean)",
            "amplitude_gate_unchanged": "max(5 Hz,10% of pair mean)",
            "scientific_contract_changed": False,
            "battery_or_integration_changed": False,
            "scientific_result_changed": comparison["scientific_result_changed"],
        },
        "v1_vs_v1_1": comparison,
        "execution_lock": lock,
        "stage0e_opened": False,
        "stage1_opened": False,
        "scientific_boundary_cn": (
            "v1.1 仅修复 Stage0D frequency gate 和 Figure B；模型、180 battery、阈值与 T/dt 未变，"
            "不含 Stage0E、slow 或 space。"
        ),
    }
    figure_metadata = _plot_v1_1(output, final_summary, final_rows, point_rows)
    final_summary["figure_contract"] = figure_metadata
    unchanged = _finalize_execution_lock(output, config_path, lock)
    v1_still_immutable = _verify_v1_immutable(cfg)
    final_summary["execution_lock"] = lock
    final_summary["v1_immutable_post_execution"] = v1_still_immutable
    final_summary["resource_usage"] = {
        **raw_summary["resource_usage"],
        "wrapper_wall_seconds": time.perf_counter() - wrapper_start,
        "max_rss_gib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2),
    }
    if not unchanged or not all(bool(row["pass"]) for row in v1_still_immutable.values()):
        final_summary["verdict"] = "STAGE0D_V1_1_ENGINEERING_OR_PROVENANCE_FAIL"
        final_summary["reason_cn"] = "v1.1 source 在运行中漂移，或 v1 immutable lineage 被改变。"
        final_summary["centre_open_local_basin_support"] = False
    if float(final_summary["resource_usage"]["max_rss_gib"]) >= 4.0:
        final_summary["verdict"] = "STAGE0D_V1_1_ENGINEERING_OR_PROVENANCE_FAIL"
        final_summary["reason_cn"] = "峰值内存超过 4 GiB 合同。"
        final_summary["centre_open_local_basin_support"] = False
    _atomic_json(output / "stage0d_local_basin_summary_v1_1.json", final_summary)
    _atomic_json(output / "stage0d_local_basin_summary.json", final_summary)
    _atomic_text(
        output / "STATUS.md",
        "# Stage 0D local-basin replication v1.1 状态\n\n"
        f"- 结论：`{final_summary['verdict']}`\n"
        f"- final counts：`{final_summary['final_status_counts']}`\n"
        f"- v1→v1.1 changed forks：{comparison['n_fork_status_changes']}\n"
        f"- scientific result changed：{comparison['scientific_result_changed']}\n"
        f"- centre open basin：{final_summary['centre_open_local_basin_support']}\n"
        f"- Figure B centre survivors：{figure_metadata['centre_final_survivor_count']}\n"
        f"- source hashes unchanged：{unchanged}\n"
        f"- peak RSS：{final_summary['resource_usage']['max_rss_gib']:.3f} GiB\n\n"
        "本轮没有 Stage0E、slow 或 space；相关节点保持关闭。\n",
    )
    return final_summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked Stage0D v1.1 repair")
    summary, output = run(args.config)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": summary["verdict"],
                "final_status_counts": summary["final_status_counts"],
                "scientific_result_changed": summary["engineering_repair"]["scientific_result_changed"],
                "wall_seconds": summary["resource_usage"]["wrapper_wall_seconds"],
                "max_rss_gib": summary["resource_usage"]["max_rss_gib"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
