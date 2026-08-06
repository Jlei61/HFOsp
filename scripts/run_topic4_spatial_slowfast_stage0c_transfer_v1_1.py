#!/usr/bin/env python3
"""Run the locked v1.1 extra-fine numerical repair for Stage-0C transfer support."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0c-transfer-v1-1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    LOCKED_POINTS,
    _atomic_json,
    _atomic_text,
    _classifier,
    _exact_reference_audit,
    _load_locked_forks,
    _overlap_lut_audit,
    _run_ablation,
    _run_resolution,
    _save_simulation,
    _save_transfer,
    _sha256,
    _write_csv,
)
from src.sef_hfo_lif import TAU_ME, TREF_E  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    ExtendedSiegertTransfer,
    TransferResolution,
    TransferSupport,
    direct_exact_error_audit,
    resolution_pair_status,
    stable_siegert_rate,
    temporal_refinement_status,
)


DEFAULT_CONFIG = ROOT / "config" / "topic4_spatial_slowfast_stage0c_transfer_v1_1.yaml"
EXTRA_FINE = TransferResolution("extra_fine", 0.125, 0.0625, 256)
V1_1_SPEC = ROOT / "docs/superpowers/specs/2026-07-20-topic4-stage0c-transfer-support-audit-v1_1-design.md"
V1_1_RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0c_transfer_v1_1.py"
SHARED_RUNNER = ROOT / "scripts/run_topic4_spatial_slowfast_stage0c_transfer.py"
TRANSFER_MODULE = ROOT / "src/topic4_spatial_slowfast_stage0c_transfer.py"


def _validate_v1_1_config(cfg: dict) -> TransferSupport:
    points = tuple((round(float(row["z"]), 2), float(row["alpha_G"])) for row in cfg["candidate_points"])
    if points != LOCKED_POINTS:
        raise ValueError("v1.1 candidate points drifted")
    support = TransferSupport(**cfg["transfer_support"]).validate()
    if support != TransferSupport():
        raise ValueError("v1.1 support must equal v1 support")
    observed = TransferResolution("extra_fine", **cfg["extra_fine"]).validate()
    if observed != EXTRA_FINE:
        raise ValueError("extra-fine resolution drifted")
    expected_runs = {
        "screen": (0.25, 6000.0, 20),
        "confirm": (0.125, 12000.0, 40),
        "dt_half": (0.0625, 12000.0, 80),
    }
    for name, expected in expected_runs.items():
        row = cfg[name]
        if (float(row["dt_ms"]), float(row["duration_ms"]), int(row["save_stride"])) != expected:
            raise ValueError(f"{name} contract drifted")
    acceptance = cfg["acceptance"]
    if any(float(acceptance[key]) != 0.25 for key in ("overlap_max_abs_error_hz", "trajectory_max_abs_error_hz")):
        raise ValueError("v1.1 absolute-error gate must remain 0.25 Hz")
    if any(float(acceptance[key]) != 0.02 for key in ("overlap_p99_relative_error", "trajectory_p99_relative_error")):
        raise ValueError("v1.1 relative-error gate must remain 2 percent")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("v1.1 scope expansion is forbidden")
    if int(cfg["resource_contract"]["blas_threads"]) != 1:
        raise ValueError("BLAS threads must equal one")
    return support


def _load_transfer(path: Path, expected_name: str) -> ExtendedSiegertTransfer:
    with np.load(path, allow_pickle=False) as payload:
        transfer = ExtendedSiegertTransfer(
            payload["mu_axis"],
            payload["sigma_axis"],
            payload["log_integral_table"],
            name=expected_name,
        )
        if not bool(payload["no_clip"]):
            raise RuntimeError("v1 fine transfer provenance says clipping was enabled")
    return transfer


def _load_traces(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _pair_fine_extra(
    fine_rows: list[dict],
    extra_rows: list[dict],
    *,
    fine_exact_audit: dict,
    extra_exact_audit: dict,
    transfer_validation_pass: bool,
    phase: str,
) -> list[dict]:
    if len(fine_rows) != len(extra_rows):
        raise RuntimeError("fine/extra-fine row count mismatch")
    output: list[dict] = []
    fine_exact_by_fork = {int(row["fork_index"]): row for row in fine_exact_audit.get("per_fork", [])}
    extra_exact_by_fork = {int(row["fork_index"]): row for row in extra_exact_audit.get("per_fork", [])}
    for index, (fine, extra) in enumerate(zip(fine_rows, extra_rows)):
        for key in ("z", "alpha_G", "initial_kind", "initial_label"):
            if fine[key] != extra[key]:
                raise RuntimeError(f"fine/extra-fine ordering mismatch at {index}: {key}")
        fork_exact_pass = bool(
            transfer_validation_pass
            and fine_exact_by_fork.get(index, {}).get("pass", False)
            and extra_exact_by_fork.get(index, {}).get("pass", False)
        )
        status = resolution_pair_status(fine, extra, exact_error_pass=fork_exact_pass)
        output.append(
            {
                "fork_index": index,
                "z": extra["z"],
                "alpha_G": extra["alpha_G"],
                "initial_kind": extra["initial_kind"],
                "initial_label": extra["initial_label"],
                "primary_classification": extra.get("primary_classification"),
                "phase": phase,
                "transfer_audit_status": status,
                "fine_classification": fine["classification"],
                "extra_fine_classification": extra["classification"],
                "fine_tail_mean_hz": fine.get("tail_mean_hz"),
                "extra_fine_tail_mean_hz": extra.get("tail_mean_hz"),
                "fine_tail_peak_hz": fine.get("tail_peak_hz"),
                "extra_fine_tail_peak_hz": extra.get("tail_peak_hz"),
                "fine_frequency_hz": fine.get("dominant_frequency_hz"),
                "extra_fine_frequency_hz": extra.get("dominant_frequency_hz"),
                "extra_fine_support_violation_step_count": extra["support_violation_step_count"],
                "extra_fine_over_100hz_tail_step_count": extra["over_100hz_tail_step_count"],
                "direct_exact_error_pass": fork_exact_pass,
                "fine_direct_exact_max_abs_error_hz": fine_exact_by_fork.get(index, {}).get("max_abs_error_hz"),
                "extra_fine_direct_exact_max_abs_error_hz": extra_exact_by_fork.get(index, {}).get("max_abs_error_hz"),
                "fine_direct_exact_p99_relative_error": fine_exact_by_fork.get(index, {}).get(
                    "p99_relative_error_meaningful"
                ),
                "extra_fine_direct_exact_p99_relative_error": extra_exact_by_fork.get(index, {}).get(
                    "p99_relative_error_meaningful"
                ),
            }
        )
    return output


def _point_outcomes(final_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for z, alpha in LOCKED_POINTS:
        subset = [row for row in final_rows if np.isclose(row["z"], z) and np.isclose(row["alpha_G"], alpha)]
        survivors = [row for row in subset if row["final_status"] == "candidate_survives"]
        rates = np.asarray([row["confirm_extra_fine_tail_mean_hz"] for row in survivors], dtype=float)
        frequencies = np.asarray([row["confirm_extra_fine_frequency_hz"] for row in survivors], dtype=float)
        supported = bool(
            len(survivors) >= 2
            and float(np.ptp(rates)) <= max(5.0, 0.20 * float(np.mean(rates)))
            and float(np.ptp(frequencies)) <= max(1.0, 0.25 * float(np.mean(frequencies)))
        )
        rows.append(
            {
                "z": z,
                "alpha_G": alpha,
                "n_forks": len(subset),
                "status_counts": dict(Counter(row["final_status"] for row in subset)),
                "n_candidate_survivors": len(survivors),
                "two_history_same_object_support": supported,
                "survivor_labels": [row["initial_label"] for row in survivors],
                "survivor_mean_rate_hz": float(np.mean(rates)) if rates.size else None,
                "survivor_frequency_hz": float(np.mean(frequencies)) if frequencies.size else None,
            }
        )
    return rows


def _plot(
    output: Path,
    metadata: list[dict],
    fine_screen: dict[str, np.ndarray],
    extra_screen: dict[str, np.ndarray] | None,
    point_rows: list[dict],
    audits: dict,
    verdict: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), constrained_layout=True)
    ax = axes[0, 0]
    mus = np.linspace(-100.0, 80.0, 500)
    for sigma, color in zip((3.0, 10.0, 30.0), ("#3b528b", "#21918c", "#fde725")):
        rates = [1000.0 * stable_siegert_rate(float(mu), sigma, TAU_ME, TREF_E) for mu in mus]
        ax.semilogy(mus, np.maximum(rates, 1e-300), color=color, lw=1.4, label=fr"$\sigma$={sigma:g} mV")
    ax.axvline(-40.0, color="0.5", ls="--", lw=0.9)
    ax.set(xlabel=r"input mean $\mu$ (mV)", ylabel="exact E rate (Hz)", title="Stable exact-Siegert reference")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    if extra_screen is not None:
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(LOCKED_POINTS)))
        for color, (z, alpha) in zip(colors, LOCKED_POINTS):
            indices = [
                index
                for index, row in enumerate(metadata)
                if np.isclose(row["z"], z)
                and np.isclose(row["alpha_G"], alpha)
                and row["initial_label"] == "probe_rest"
            ]
            index = indices[0]
            ax.plot(
                fine_screen["time_ms"] / 1000.0,
                1000.0 * fine_screen["rE_khz"][:, index],
                color=color,
                lw=0.8,
                ls="--",
                alpha=0.7,
            )
            ax.plot(
                extra_screen["time_ms"] / 1000.0,
                1000.0 * extra_screen["rE_khz"][:, index],
                color=color,
                lw=1.1,
                label=f"z={z:.2f}, a={alpha:g}",
            )
        ax.axhline(100.0, color="0.5", ls="--", lw=0.9)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="Fine (dashed) vs extra-fine (solid)")

    ax = axes[1, 0]
    statuses = ("candidate_survives", "collapses_low", "becomes_over_100", "numerical_unresolved")
    palette = ("#2ca02c", "#4c78a8", "#e45756", "#b279a2")
    bottom = np.zeros(len(point_rows))
    for status, color in zip(statuses, palette):
        values = np.asarray([row["status_counts"].get(status, 0) for row in point_rows])
        ax.bar(np.arange(len(point_rows)), values, bottom=bottom, color=color, label=status)
        bottom += values
    ax.set_xticks(
        np.arange(len(point_rows)),
        [f"{row['z']:.2f}\n{row['alpha_G']:g}" for row in point_rows],
    )
    ax.set(xlabel="z / alpha_G", ylabel="fixed non-exact forks", title="v1.1 authoritative outcomes")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    extra_overlap = audits.get("extra_fine_overlap", {})
    extra_direct = audits.get("extra_fine_screen_direct_exact", {})
    ax.text(
        0.02,
        0.98,
        "\n".join(
            [
                f"Verdict: {verdict}",
                "v1 preserved: provenance mismatch",
                f"extra-fine overlap: {extra_overlap.get('pass')}",
                f"extra-fine direct: {extra_direct.get('pass')}",
                f"max abs: {extra_direct.get('max_abs_error_hz', float('nan')):.3g} Hz",
                f"p99 relative: {extra_direct.get('p99_relative_error_meaningful', float('nan')):.3g}",
                "coarse is diagnostic only",
                "no clipping / no extrapolation",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
    )
    ax.set_title("v1.1 numerical repair boundary")
    path = figures / "stage0c_transfer_support_audit_v1_1.png"
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    _atomic_text(
        figures / "README.md",
        "### stage0c_transfer_support_audit_v1_1.png\n\n"
        "这张图是 v1 transfer 验证失败后的纯数值修复审计，不改变候选点或模型方程。"
        "左上是 stable exact reference；右上用固定 probe_rest 对比 v1 fine（虚线）与 extra-fine（实线）；"
        "左下给出 extra-fine authoritative 分类；右下列出数值门。它不包含 slow lifecycle 或空间耦合。\n\n"
        "**关注点**：extra-fine 是否先通过原 0.25 Hz / 2% 门，以及是否留下至少两初态支持的有限对象。\n",
    )


def _write_validation_failure(
    output: Path,
    cfg: dict,
    config_path: Path,
    start: float,
    provenance: dict,
    v1_provenance: dict,
    exact_reference: dict,
    extra_overlap: dict,
) -> dict:
    elapsed = time.perf_counter() - start
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    summary = {
        "schema_version": "topic4_stage0c_transfer_support_audit.v1_1",
        "verdict": "NUMERICAL_UNRESOLVED_EXTRA_FINE_VALIDATION_FAILED",
        "candidate_survives": False,
        "reason_cn": "extra-fine 未通过 v1.1 新锁定的 conservative 0.25 Hz / 2% transfer validation，按合同停止且不重放动力学。",
        "v1_provenance": v1_provenance,
        "provenance": provenance,
        "numerical_audits": {"exact_reference": exact_reference, "extra_fine_overlap": extra_overlap},
        "resource_usage": {
            "wall_seconds": elapsed,
            "max_rss_gib": rss,
            "max_memory_gib_contract": float(cfg["resource_contract"]["max_memory_gib"]),
            "within_memory_contract": rss < float(cfg["resource_contract"]["max_memory_gib"]),
        },
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "implementation_provenance": {
            "transfer_module_sha256": _sha256(TRANSFER_MODULE),
            "v1_1_runner_sha256": _sha256(V1_1_RUNNER),
            "shared_runner_sha256": _sha256(SHARED_RUNNER),
            "v1_1_spec_sha256": _sha256(V1_1_SPEC),
        },
    }
    _atomic_json(output / "stage0c_transfer_support_summary_v1_1.json", summary)
    _atomic_text(
        output / "STATUS.md",
        "# Stage 0C transfer-support audit v1.1\n\n"
        f"- 结论：`{summary['verdict']}`\n"
        "- 动力学重放：未运行（数值 stop rule）\n"
        f"- wall / peak RSS：{elapsed:.2f} s / {rss:.3f} GiB\n",
    )
    return summary


def run(config_path: Path) -> tuple[dict, Path]:
    start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    support = _validate_v1_1_config(cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    metadata, states, params, provenance = _load_locked_forks(cfg)
    thresholds = _classifier(cfg)

    v1_root = ROOT / cfg["v1_result_root"]
    v1_summary_path = v1_root / "stage0c_transfer_support_summary.json"
    v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
    if v1_summary["verdict"] != "NUMERICAL_UNRESOLVED_TRANSFER_VALIDATION_FAILED":
        raise RuntimeError("v1.1 requires the preserved v1 validation-failed parent")
    v1_provenance = {
        "summary": str(v1_summary_path.resolve()),
        "summary_sha256": _sha256(v1_summary_path),
        "verdict": v1_summary["verdict"],
        "interpretation": "validation_implementation_spec_provenance_mismatch_unresolved",
        "v1_runner_hash_available_in_artifact": False,
        "v1_1_gate_role": "new_locked_conservative_repair_not_retrospective_preregistration",
        "fine_transfer": str((v1_root / "extended_transfer_fine.npz").resolve()),
        "fine_transfer_sha256": _sha256(v1_root / "extended_transfer_fine.npz"),
        "fine_screen_rows_sha256": _sha256(v1_root / "state_fork_screen_fine.json"),
        "fine_screen_traces_sha256": _sha256(v1_root / "state_fork_screen_fine_traces.npz"),
        "coarse_role": "diagnostic_only_not_authoritative_in_v1_1",
    }

    exact_reference = _exact_reference_audit()
    extra_transfer = ExtendedSiegertTransfer.build(support, EXTRA_FINE)
    _save_transfer(output / "extended_transfer_extra_fine.npz", extra_transfer)
    extra_overlap = _overlap_lut_audit(extra_transfer)
    _atomic_json(
        output / "pre_replay_transfer_validation.json",
        {"exact_reference": exact_reference, "extra_fine_overlap": extra_overlap},
    )
    if not exact_reference["pass"] or not extra_overlap["pass"]:
        summary = _write_validation_failure(
            output, cfg, config_path, start, provenance, v1_provenance, exact_reference, extra_overlap
        )
        _plot(
            output,
            metadata,
            _load_traces(v1_root / "state_fork_screen_fine_traces.npz"),
            None,
            [],
            {"extra_fine_overlap": extra_overlap},
            summary["verdict"],
        )
        return summary, output

    fine_transfer = _load_transfer(v1_root / "extended_transfer_fine.npz", "fine")
    fine_screen_rows = json.loads((v1_root / "state_fork_screen_fine.json").read_text(encoding="utf-8"))
    fine_screen_sim = _load_traces(v1_root / "state_fork_screen_fine_traces.npz")
    v1_validation = json.loads((v1_root / "transfer_validation.json").read_text(encoding="utf-8"))
    fine_screen_exact = direct_exact_error_audit(fine_screen_sim, fine_transfer, max_points_per_fork=16)
    if not fine_screen_exact["pass"]:
        raise RuntimeError("v1 fine trajectory direct-exact must pass before v1.1 comparison")

    extra_screen_sim, extra_screen_rows, extra_screen_exact = _run_resolution(
        metadata, states, params, extra_transfer, cfg["screen"], thresholds, "screen"
    )
    _save_simulation(output / "state_fork_screen_extra_fine_traces.npz", extra_screen_sim)
    screen_pairs = _pair_fine_extra(
        fine_screen_rows,
        extra_screen_rows,
        fine_exact_audit=fine_screen_exact,
        extra_exact_audit=extra_screen_exact,
        transfer_validation_pass=bool(exact_reference["pass"] and extra_overlap["pass"]),
        phase="screen",
    )
    screen_survivors = [index for index, row in enumerate(screen_pairs) if row["transfer_audit_status"] == "candidate_survives"]

    confirm_sim: dict[str, dict[str, np.ndarray]] = {}
    confirm_rows: dict[str, list[dict]] = {"fine": [], "extra_fine": []}
    confirm_exact: dict[str, dict] = {}
    confirm_pairs: list[dict] = []
    if screen_survivors:
        subset_metadata = [metadata[index] for index in screen_survivors]
        subset_states = states[screen_survivors]
        subset_params = [params[index] for index in screen_survivors]
        for name, transfer in (("fine", fine_transfer), ("extra_fine", extra_transfer)):
            simulation, rows, exact = _run_resolution(
                subset_metadata, subset_states, subset_params, transfer, cfg["confirm"], thresholds, "confirm"
            )
            confirm_sim[name], confirm_rows[name], confirm_exact[name] = simulation, rows, exact
            _save_simulation(output / f"state_fork_confirm_{name}_traces.npz", simulation)
        confirm_pairs = _pair_fine_extra(
            confirm_rows["fine"],
            confirm_rows["extra_fine"],
            fine_exact_audit=confirm_exact["fine"],
            extra_exact_audit=confirm_exact["extra_fine"],
            transfer_validation_pass=True,
            phase="confirm",
        )

    confirm_survivor_local = [
        index for index, row in enumerate(confirm_pairs) if row["transfer_audit_status"] == "candidate_survives"
    ]
    confirm_survivor_indices = [screen_survivors[index] for index in confirm_survivor_local]
    dt_half_rows: list[dict] = []
    dt_half_exact: dict = {"pass": None, "reason": "not_run_no_confirm_survivor"}
    if confirm_survivor_indices:
        dt_half_sim, dt_half_rows, dt_half_exact = _run_resolution(
            [metadata[index] for index in confirm_survivor_indices],
            states[confirm_survivor_indices],
            [params[index] for index in confirm_survivor_indices],
            extra_transfer,
            cfg["dt_half"],
            thresholds,
            "dt_half",
        )
        _save_simulation(output / "state_fork_dt_half_extra_fine_traces.npz", dt_half_sim)

    ablation_rows: list[dict] = []
    if confirm_survivor_indices:
        ablation_rows = _run_ablation(
            confirm_survivor_indices,
            confirm_survivor_local,
            metadata,
            states,
            params,
            extra_transfer,
            confirm_sim["extra_fine"],
            cfg,
            thresholds,
        )
        expected_ablation_rows = 5 * len(confirm_survivor_indices)
        expected_arms = {"dynamic", "instantaneous", "clamped", "matched_subtractive", "mean_only"}
        if len(ablation_rows) != expected_ablation_rows:
            raise RuntimeError("five-arm ablation row count is incomplete")
        for original_index in confirm_survivor_indices:
            source = metadata[original_index]
            members = [
                row
                for row in ablation_rows
                if np.isclose(row["z"], source["z"])
                and np.isclose(row["alpha_G"], source["alpha_G"])
                and row["initial_kind"] == source["initial_kind"]
                and row["initial_label"] == source["initial_label"]
            ]
            if {row["mechanism"] for row in members} != expected_arms:
                raise RuntimeError("five-arm ablation labels are incomplete")

    confirm_lookup = {screen_survivors[local]: row for local, row in enumerate(confirm_pairs)}
    confirm_extra_lookup = {
        screen_survivors[local]: row for local, row in enumerate(confirm_rows["extra_fine"])
    }
    dt_lookup = {confirm_survivor_indices[local]: row for local, row in enumerate(dt_half_rows)}
    dt_exact_lookup = {
        confirm_survivor_indices[int(row["fork_index"])]: row
        for row in dt_half_exact.get("per_fork", [])
    }
    final_rows: list[dict] = []
    for index, screen in enumerate(screen_pairs):
        status = screen["transfer_audit_status"]
        confirm = confirm_lookup.get(index)
        if status == "candidate_survives":
            status = "numerical_unresolved" if confirm is None else confirm["transfer_audit_status"]
        dt_row = dt_lookup.get(index)
        if status == "candidate_survives":
            if dt_row is None:
                status = "numerical_unresolved"
            else:
                confirm_extra = confirm_extra_lookup[index]
                dt_exact_row = dt_exact_lookup.get(index, {"pass": False})
                status = temporal_refinement_status(
                    confirm_extra,
                    dt_row,
                    exact_error_pass=bool(dt_exact_row.get("pass", False)),
                )
        final_rows.append(
            {
                **screen,
                "final_status": status,
                "confirm_status": None if confirm is None else confirm["transfer_audit_status"],
                "confirm_extra_fine_classification": None if confirm is None else confirm["extra_fine_classification"],
                "confirm_extra_fine_tail_mean_hz": None if confirm is None else confirm["extra_fine_tail_mean_hz"],
                "confirm_extra_fine_tail_peak_hz": None if confirm is None else confirm["extra_fine_tail_peak_hz"],
                "confirm_extra_fine_frequency_hz": None if confirm is None else confirm["extra_fine_frequency_hz"],
                "dt_half_classification": None if dt_row is None else dt_row["classification"],
                "dt_half_tail_mean_hz": None if dt_row is None else dt_row.get("tail_mean_hz"),
                "dt_half_tail_peak_hz": None if dt_row is None else dt_row.get("tail_peak_hz"),
                "dt_half_vs_confirm_rate_abs_difference_hz": (
                    None
                    if dt_row is None or confirm is None
                    else abs(float(dt_row.get("tail_mean_hz")) - float(confirm["extra_fine_tail_mean_hz"]))
                ),
                "dt_half_vs_confirm_frequency_abs_difference_hz": (
                    None
                    if dt_row is None or confirm is None
                    else abs(float(dt_row.get("dominant_frequency_hz")) - float(confirm["extra_fine_frequency_hz"]))
                ),
                "dt_half_direct_exact_pass": None if dt_row is None else bool(dt_exact_lookup.get(index, {}).get("pass", False)),
            }
        )

    point_rows = _point_outcomes(final_rows)
    supported_points = [row for row in point_rows if row["two_history_same_object_support"]]
    status_counts = dict(Counter(row["final_status"] for row in final_rows))
    primary_invalid = [row for row in final_rows if row["primary_classification"] == "audit_invalid_candidate"]
    primary_invalid_counts = dict(Counter(row["final_status"] for row in primary_invalid))
    if supported_points:
        verdict = "TRANSFER_SUPPORTED_FINITE_FAST_OBJECT_CANDIDATE_V1_1"
        reason_cn = "extra-fine 数值门通过，且至少一个参数点有两初态支持的有限快态候选。"
    elif status_counts.get("numerical_unresolved", 0):
        verdict = "EXTRA_FINE_VALID_NO_SUPPORTED_OBJECT_WITH_UNRESOLVED_TRANSIENTS"
        reason_cn = "extra-fine transfer 数值验证通过，但没有两初态支持的有限对象，且仍有长瞬态或分类未决。"
    else:
        verdict = "EXTRA_FINE_VALID_CLEAN_NO_SUPPORTED_OBJECT"
        reason_cn = "extra-fine 将全部固定初态裁决为 low 或 >100 Hz，没有有限对象。"

    elapsed = time.perf_counter() - start
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    audits = {
        "exact_reference": exact_reference,
        "extra_fine_overlap": extra_overlap,
        "v1_fine_screen_direct_exact": fine_screen_exact,
        "v1_artifact_fine_screen_direct_exact": v1_validation["screen_direct_exact"]["fine"],
        "extra_fine_screen_direct_exact": extra_screen_exact,
        "confirm_direct_exact": confirm_exact,
        "dt_half_direct_exact": dt_half_exact,
    }
    summary = {
        "schema_version": "topic4_stage0c_transfer_support_audit.v1_1",
        "verdict": verdict,
        "reason_cn": reason_cn,
        "candidate_survives": bool(supported_points),
        "n_supported_parameter_points": len(supported_points),
        "supported_parameter_points": supported_points,
        "final_status_counts_all_102": status_counts,
        "primary_23_audit_invalid_candidate_final_status_counts": primary_invalid_counts,
        "n_primary_audit_invalid_candidates": len(primary_invalid),
        "n_screen_survivor_forks": len(screen_survivors),
        "n_confirm_survivor_forks": len(confirm_survivor_indices),
        "dt_half_run": bool(confirm_survivor_indices),
        "five_arm_ablation_run": bool(ablation_rows),
        "n_ablation_rows": len(ablation_rows),
        "point_outcomes": point_rows,
        "v1_provenance": v1_provenance,
        "provenance": provenance,
        "transfer_contract": {
            "extra_fine": cfg["extra_fine"],
            "support": cfg["transfer_support"],
            "acceptance": cfg["acceptance"],
            "fine_role": "resolution_comparator",
            "coarse_role": "diagnostic_only_not_authoritative",
            "outside_support": "NaN_fail_closed",
            "clipping": False,
            "extrapolation": False,
        },
        "integration_contract": {key: cfg[key] for key in ("screen", "confirm", "dt_half")},
        "numerical_audits": audits,
        "resource_usage": {
            "wall_seconds": elapsed,
            "max_rss_gib": rss,
            "max_memory_gib_contract": float(cfg["resource_contract"]["max_memory_gib"]),
            "within_memory_contract": rss < float(cfg["resource_contract"]["max_memory_gib"]),
            "execution": "single_process_blas_threads_1",
        },
        "scientific_boundary_cn": (
            "v1.1 只修复 transfer 分辨率并裁决六个既有参数点；它不含 slow lifecycle、噪声或空间耦合，"
            "无论结果如何都不直接开放 Stage1。"
        ),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "implementation_provenance": {
            "transfer_module": str(TRANSFER_MODULE.resolve()),
            "transfer_module_sha256": _sha256(TRANSFER_MODULE),
            "v1_1_runner": str(V1_1_RUNNER.resolve()),
            "v1_1_runner_sha256": _sha256(V1_1_RUNNER),
            "shared_runner": str(SHARED_RUNNER.resolve()),
            "shared_runner_sha256": _sha256(SHARED_RUNNER),
            "v1_1_spec": str(V1_1_SPEC.resolve()),
            "v1_1_spec_sha256": _sha256(V1_1_SPEC),
            "extra_fine_transfer_sha256": _sha256(output / "extended_transfer_extra_fine.npz"),
        },
    }
    if not summary["resource_usage"]["within_memory_contract"]:
        summary["verdict"] = "ENGINEERING_FAIL_MEMORY_CONTRACT_V1_1"
        summary["candidate_survives"] = False
        summary["reason_cn"] = "峰值内存超过 4 GiB 合同。"

    _atomic_json(output / "transfer_validation_v1_1.json", audits)
    _atomic_json(output / "state_fork_screen_extra_fine.json", extra_screen_rows)
    _atomic_json(output / "state_fork_screen_fine_vs_extra_fine.json", screen_pairs)
    _atomic_json(output / "state_fork_confirm_fine.json", confirm_rows["fine"])
    _atomic_json(output / "state_fork_confirm_extra_fine.json", confirm_rows["extra_fine"])
    _atomic_json(output / "state_fork_confirm_fine_vs_extra_fine.json", confirm_pairs)
    _atomic_json(output / "state_fork_dt_half_extra_fine.json", dt_half_rows)
    _atomic_json(output / "mechanism_ablation_extra_fine.json", ablation_rows)
    _atomic_json(output / "final_fork_outcomes_v1_1.json", final_rows)
    _atomic_json(output / "point_outcomes_v1_1.json", point_rows)
    _atomic_json(output / "stage0c_transfer_support_summary_v1_1.json", summary)
    _write_csv(output / "final_fork_outcomes_v1_1.csv", final_rows)
    _write_csv(output / "point_outcomes_v1_1.csv", point_rows)
    _write_csv(output / "mechanism_ablation_extra_fine.csv", ablation_rows)
    _plot(output, metadata, fine_screen_sim, extra_screen_sim, point_rows, audits, summary["verdict"])
    _atomic_text(
        output / "STATUS.md",
        "# Stage 0C transfer-support audit v1.1\n\n"
        f"- 结论：`{summary['verdict']}`\n"
        f"- all-102 final counts：`{status_counts}`\n"
        f"- primary-23 final counts：`{primary_invalid_counts}`\n"
        f"- supported points：{len(supported_points)}\n"
        f"- screen / confirm survivor forks：{len(screen_survivors)} / {len(confirm_survivor_indices)}\n"
        f"- extra-fine overlap / trajectory direct：{extra_overlap['pass']} / {extra_screen_exact['pass']}\n"
        f"- wall / peak RSS：{elapsed:.2f} s / {rss:.3f} GiB\n"
        f"- 解释：{summary['reason_cn']}\n\n"
        "v1 按 implementation/spec provenance mismatch 保留 unresolved；coarse 不参与 v1.1 authoritative 判定。\n",
    )
    return summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked v1.1 numerical repair")
    summary, output = run(args.config)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": summary["verdict"],
                "candidate_survives": summary["candidate_survives"],
                "wall_seconds": summary["resource_usage"]["wall_seconds"],
                "max_rss_gib": summary["resource_usage"]["max_rss_gib"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
