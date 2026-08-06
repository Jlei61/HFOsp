#!/usr/bin/env python3
"""Run the independent Stage-0C homogeneous dynamic-divisive-pool screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

# This screen is deliberately one process.  Fix thread limits before numpy/scipy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sef_hfo_lif import C_EE, TAU_ME, W_EE  # noqa: E402
from src.topic4_spatial_slowfast_stage0b import (  # noqa: E402
    FastParameters,
    ForkClassifierThresholds,
    build_state_forks as build_stage0b_forks,
    find_fixed_points as find_stage0b_fixed_points,
    simulate_forks as simulate_stage0b_forks,
)
from src.topic4_spatial_slowfast_stage0c import (  # noqa: E402
    E0_KHZ,
    E50_KHZ,
    N_PSI,
    S_MAX,
    TAU_FAST_MS,
    TAU_MU_MS,
    TAU_S_MS,
    PoolParameters,
    build_state_forks,
    classify_fork_batch,
    continuation_root_scan,
    select_confirm_candidates,
    simulate_forks,
    summarize_stage0c,
)


DEFAULT_CONFIG = ROOT / "config" / "topic4_spatial_slowfast_stage0c.yaml"


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, default=_json_default)
        stream.write("\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classifier(cfg: dict) -> ForkClassifierThresholds:
    return ForkClassifierThresholds(**cfg["classifier"]).validate()


def _validate_config(cfg: dict) -> tuple[list[float], list[float]]:
    axes = cfg["axes"]
    z_values = [float(value) for value in axes["z"]]
    alpha_values = [float(value) for value in axes["alpha_G"]]
    if not np.allclose(z_values, np.round(np.arange(1.0, 0.79, -0.01), 2)):
        raise ValueError("Stage0C z axis drifted from locked 1.00:-0.01:0.80")
    if alpha_values != [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
        raise ValueError("Stage0C alpha_G axis drifted from locked values")
    if float(axes["w_ee_mult"]) != 1.1 or float(axes["ratio"]) != 1.0:
        raise ValueError("Stage0C requires w_ee_mult=1.1 and ratio=1")
    locked_pool = {
        "tau_fast_ms": TAU_FAST_MS,
        "tau_mu_ms": TAU_MU_MS,
        "tau_S_ms": TAU_S_MS,
        "S_max": S_MAX,
        "e0_khz": E0_KHZ,
        "e50_khz": E50_KHZ,
        "n_psi": N_PSI,
        "numerical_clip": False,
    }
    for key, expected in locked_pool.items():
        observed = cfg["pool"][key]
        if isinstance(expected, bool):
            if bool(observed) != expected:
                raise ValueError(f"pool contract drift: {key}")
        elif not np.isclose(float(observed), float(expected)):
            raise ValueError(f"pool contract drift: {key}")
    if any(bool(cfg["scope"][key]) for key in ("noise", "local_recovery_r", "spatial_coupling", "dynamic_phi")):
        raise ValueError("Stage0C must keep noise/r/spatial/phi disabled")
    if int(cfg["resource_contract"]["blas_threads"]) != 1:
        raise ValueError("Stage0C requires one BLAS thread")
    return z_values, alpha_values


def _alpha0_parity_audit(
    root_rows: list[dict],
    z_values: list[float],
    *,
    dt_ms: float,
    duration_ms: float,
    save_stride: int,
    audit_tail_fraction: float,
) -> dict:
    """Compare alpha=0 roots and off-manifold trajectories to Stage0B."""

    root_count_match = True
    root_max_abs_khz = 0.0
    stability_match = True
    for z in z_values:
        stage0c_point = next(
            row for row in root_rows if row["alpha_G"] == 0.0 and np.isclose(row["z"], z)
        )
        stage0b_roots = find_stage0b_fixed_points(FastParameters(1.1, z, 1.0))
        stage0c_roots = stage0c_point["roots"]
        if len(stage0b_roots) != len(stage0c_roots):
            root_count_match = False
            continue
        for old, new in zip(stage0b_roots, stage0c_roots):
            root_max_abs_khz = max(
                root_max_abs_khz,
                abs(float(old["rE_khz"]) - float(new["rE_khz"])),
                abs(float(old["rI_khz"]) - float(new["rI_khz"])),
            )
            stability_match &= str(old["stability"]) == str(new["stability"])

    stage0b_points = [
        {"w_ee_mult": 1.1, "q": z, "ratio": 1.0, "roots": []} for z in z_values
    ]
    metadata_b, states_b, params_b = build_stage0b_forks(stage0b_points)
    keep_b = [index for index, row in enumerate(metadata_b) if row["initial_kind"] in {"probe", "off_manifold_probe"}]
    simulation_b = simulate_stage0b_forks(
        states_b[keep_b],
        [params_b[index] for index in keep_b],
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        save_stride=save_stride,
    )

    stage0c_points = [
        {"z": z, "alpha_G": 0.0, "w_ee_mult": 1.1, "ratio": 1.0, "roots": []}
        for z in z_values
    ]
    metadata_c, states_c, params_c = build_state_forks(stage0c_points)
    keep_c = [
        index
        for index, row in enumerate(metadata_c)
        if row["initial_kind"] in {"on_manifold_probe", "stage0b_off_manifold_probe"}
    ]
    simulation_c = simulate_forks(
        states_c[keep_c],
        [params_c[index] for index in keep_c],
        dt_ms=dt_ms,
        duration_ms=duration_ms,
        save_stride=save_stride,
        audit_tail_fraction=audit_tail_fraction,
    )
    label_b = [(metadata_b[index]["q"], metadata_b[index]["initial_label"]) for index in keep_b]
    label_c = [(metadata_c[index]["z"], metadata_c[index]["initial_label"]) for index in keep_c]
    labels_match = label_b == label_c
    rate_trace_max_abs_khz = float(
        max(
            np.max(np.abs(simulation_b["rE_khz"] - simulation_c["rE_khz"])),
            np.max(np.abs(simulation_b["rI_khz"] - simulation_c["rI_khz"])),
        )
    )
    final_first6_max_abs = float(
        np.max(np.abs(simulation_b["final_state"] - simulation_c["final_state"][:, :6]))
    )
    passed = bool(
        root_count_match
        and stability_match
        and labels_match
        and root_max_abs_khz <= 2e-6
        and rate_trace_max_abs_khz <= 1e-10
        and final_first6_max_abs <= 1e-10
    )
    return {
        "pass": passed,
        "n_z": len(z_values),
        "root_count_match": root_count_match,
        "root_stability_class_match": stability_match,
        "root_max_abs_khz": root_max_abs_khz,
        "fork_labels_match": labels_match,
        "n_common_forks": len(keep_b),
        "rate_trace_max_abs_khz": rate_trace_max_abs_khz,
        "final_first6_max_abs": final_first6_max_abs,
        "tolerance": {"root_khz": 2e-6, "trace_khz": 1e-10, "final_state": 1e-10},
    }


def _bidirectional_root_audit(forward: list[dict], reverse: list[dict]) -> dict:
    """Require root sets to agree under reversed z/alpha warm-continuation order."""

    reverse_lookup = {
        (round(float(point["z"]), 8), round(float(point["alpha_G"]), 8)): point
        for point in reverse
    }
    mismatches: list[dict] = []
    max_rate_difference_khz = 0.0
    for point in forward:
        key = (round(float(point["z"]), 8), round(float(point["alpha_G"]), 8))
        other = reverse_lookup.get(key)
        if other is None or len(point["roots"]) != len(other["roots"]):
            mismatches.append(
                {
                    "z": point["z"],
                    "alpha_G": point["alpha_G"],
                    "forward_n_roots": len(point["roots"]),
                    "reverse_n_roots": None if other is None else len(other["roots"]),
                    "reason": "root_count_mismatch",
                }
            )
            continue
        point_failed = False
        for left, right in zip(point["roots"], other["roots"]):
            difference = max(
                abs(float(left["rE_khz"]) - float(right["rE_khz"])),
                abs(float(left["rI_khz"]) - float(right["rI_khz"])),
            )
            max_rate_difference_khz = max(max_rate_difference_khz, difference)
            if (
                difference > 2e-5
                or left["stability"] != right["stability"]
                or left["branch_class"] != right["branch_class"]
            ):
                point_failed = True
        if point_failed:
            mismatches.append(
                {
                    "z": point["z"],
                    "alpha_G": point["alpha_G"],
                    "forward_n_roots": len(point["roots"]),
                    "reverse_n_roots": len(other["roots"]),
                    "reason": "root_set_or_stability_mismatch",
                }
            )
    return {
        "pass": not mismatches and len(forward) == len(reverse),
        "method": "dense_80_seed_multistart_plus_forward_and_reverse_warm_continuation",
        "n_parameter_points": len(forward),
        "n_mismatched_points": len(mismatches),
        "max_rate_difference_khz": max_rate_difference_khz,
        "rate_match_tolerance_khz": 2e-5,
        "mismatches": mismatches,
    }


def _run_ablation(
    confirmed_local_indices: list[int],
    candidate_indices: list[int],
    initial_states: np.ndarray,
    parameters: list[PoolParameters],
    metadata: list[dict],
    confirm_sim: dict[str, np.ndarray],
    confirm_cfg: dict,
    thresholds: ForkClassifierThresholds,
) -> list[dict]:
    """Run the pre-registered five-arm ablation only for confirmed candidates."""

    rows: list[dict] = []
    for local_index in confirmed_local_indices:
        screen_index = candidate_indices[local_index]
        params = parameters[screen_index]
        state = initial_states[screen_index : screen_index + 1]
        base_meta = metadata[screen_index]
        tail_start = max(1, int(np.floor((1.0 - thresholds.tail_fraction) * confirm_sim["time_ms"].size)))
        matched_s = float(np.mean(confirm_sim["S_G"][tail_start:, local_index]))
        final_s_ee = float(confirm_sim["final_state"][local_index, 2])
        rec_mean_mv = TAU_ME * C_EE * (params.w_ee_mult * W_EE) * final_s_ee
        divided_loss_mv = rec_mean_mv * (params.alpha_g * matched_s) / max(1.0 + params.alpha_g * matched_s, 1e-12)
        beta_mv = divided_loss_mv / max(matched_s, 1e-9)
        arms = (
            ("dynamic", None, None),
            ("instantaneous", None, None),
            ("clamped", matched_s, None),
            ("matched_subtractive", None, beta_mv),
            ("mean_only", None, None),
        )
        for mechanism, clamp_s, subtractive_beta in arms:
            simulation = simulate_forks(
                state,
                [params],
                dt_ms=float(confirm_cfg["dt_ms"]),
                duration_ms=float(confirm_cfg["duration_ms"]),
                save_stride=int(confirm_cfg["save_stride"]),
                mechanism=mechanism,
                clamp_s=clamp_s,
                subtractive_beta_mv=subtractive_beta,
                audit_tail_fraction=thresholds.tail_fraction,
            )
            classified = classify_fork_batch([base_meta], simulation, thresholds)[0]
            rows.append(
                {
                    **classified,
                    "mechanism": mechanism,
                    "matched_S_G": matched_s,
                    "matched_subtractive_beta_mV_per_SG": beta_mv,
                    "matched_reference_recurrent_mean_mV": rec_mean_mv,
                    "matched_reference_divisive_loss_mV": divided_loss_mv,
                    "ablation_dt_ms": float(confirm_cfg["dt_ms"]),
                    "ablation_duration_ms": float(confirm_cfg["duration_ms"]),
                }
            )
    return rows


def _plot_screen(
    output: Path,
    root_rows: list[dict],
    screen_rows: list[dict],
    metadata: list[dict],
    simulation: dict[str, np.ndarray],
    z_values: list[float],
    alpha_values: list[float],
    verdict: str,
) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    for point in root_rows:
        for root in point["roots"]:
            marker = "o" if root["stability"] == "stable" else "x"
            color = plt.cm.viridis(min(float(root["rE_hz"]), 100.0) / 100.0)
            ax.scatter(point["z"], point["alpha_G"], s=25, marker=marker, c=[color], linewidths=0.9)
    ax.set(xlabel="frozen z", ylabel=r"$\alpha_G$", title="9D root topology (color: E rate, capped at 100 Hz)")
    ax.invert_xaxis()

    ax = axes[0, 1]
    category = np.full((len(alpha_values), len(z_values)), np.nan)
    for ai, alpha in enumerate(alpha_values):
        for zi, z in enumerate(z_values):
            subset = [row for row in screen_rows if np.isclose(row["alpha_G"], alpha) and np.isclose(row["z"], z) and row["initial_kind"] != "exact_root"]
            labels = {row["classification"] for row in subset}
            if labels & {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}:
                category[ai, zi] = 2
            elif labels <= {"low_fixed_point", "saturation_or_over_100hz"}:
                category[ai, zi] = 1 if "saturation_or_over_100hz" in labels else 0
            else:
                category[ai, zi] = 3
    image = ax.imshow(category, aspect="auto", origin="lower", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(["#4c78a8", "#e45756", "#54a24b", "#b279a2"]), vmin=-0.5, vmax=3.5)
    ax.set_xticks(np.arange(0, len(z_values), 4), [f"{z_values[i]:.2f}" for i in range(0, len(z_values), 4)])
    ax.set_yticks(np.arange(len(alpha_values)), [f"{value:g}" for value in alpha_values])
    ax.set(xlabel="frozen z", ylabel=r"$\alpha_G$", title="Fork outcome: low / saturation / candidate / unresolved")

    trace_indices: list[int] = []
    for alpha in (0.0, 4.0, 12.0, 32.0):
        candidates = [
            index for index, row in enumerate(metadata)
            if np.isclose(row["z"], 0.90)
            and np.isclose(row["alpha_G"], alpha)
            and row["initial_label"] == "probe_boundary"
        ]
        if candidates:
            trace_indices.append(candidates[0])
    time_s = simulation["time_ms"] / 1000.0
    ax = axes[1, 0]
    for index in trace_indices:
        ax.plot(time_s, 1000.0 * simulation["rE_khz"][:, index], lw=1.25, label=fr"$\alpha_G$={metadata[index]['alpha_G']:g}")
    ax.axhline(100.0, color="0.5", lw=0.8, ls="--")
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="Boundary-probe trajectories at z=0.90")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for index in trace_indices:
        ax.plot(time_s, simulation["S_G"][:, index], lw=1.25, label=fr"$\alpha_G$={metadata[index]['alpha_G']:g}")
    ax.set(xlabel="time (s)", ylabel=r"$S_G$", title="Unclipped dynamic-pool trajectories")
    ax.text(0.01, 0.98, verdict, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    path = figures / "stage0c_dynamic_pool_topology.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    _atomic_text(
        figures / "README.md",
        "### stage0c_dynamic_pool_topology.png\n\n"
        "这张图汇总冻结 z、均匀 9D 快系统中动态 recurrent-E 除法池的 root 与 state-fork 结果。"
        "左上圆点/叉号分别表示稳定/不稳定 root，颜色只编码 E rate；右上是非 exact-root 轨迹的参数格分类；"
        "下排展示 z=0.90 的同一 boundary probe 在代表性 alpha_G 下的 E rate 与未裁剪 S_G。"
        "它只回答是否存在有限快态对象，不代表发作、自发转换、恢复或空间传播。\n\n"
        "**关注点**：是否存在低于 100 Hz、无 LUT/状态边界依赖、并通过 12 s confirm 的绿色 candidate 格。\n",
    )


def run(config_path: Path) -> tuple[dict, Path]:
    start = time.perf_counter()
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    z_values, alpha_values = _validate_config(cfg)
    axes = cfg["axes"]
    root_rows = continuation_root_scan(
        z_values,
        alpha_values,
        w_ee_mult=float(axes["w_ee_mult"]),
        ratio=float(axes["ratio"]),
    )
    reverse_root_rows = continuation_root_scan(
        list(reversed(z_values)),
        list(reversed(alpha_values)),
        w_ee_mult=float(axes["w_ee_mult"]),
        ratio=float(axes["ratio"]),
    )
    root_coverage_audit = _bidirectional_root_audit(root_rows, reverse_root_rows)
    metadata, initial_states, parameters = build_state_forks(root_rows)
    screen_cfg = cfg["screen"]
    thresholds = _classifier(cfg)
    screen_sim = simulate_forks(
        initial_states,
        parameters,
        dt_ms=float(screen_cfg["dt_ms"]),
        duration_ms=float(screen_cfg["duration_ms"]),
        save_stride=int(screen_cfg["save_stride"]),
        audit_tail_fraction=thresholds.tail_fraction,
    )
    screen_rows = classify_fork_batch(metadata, screen_sim, thresholds)
    candidate_indices = select_confirm_candidates(screen_rows)

    confirm_rows: list[dict] = []
    confirm_sim: dict[str, np.ndarray] | None = None
    if candidate_indices:
        confirm_cfg = cfg["confirm"]
        confirm_sim = simulate_forks(
            initial_states[candidate_indices],
            [parameters[index] for index in candidate_indices],
            dt_ms=float(confirm_cfg["dt_ms"]),
            duration_ms=float(confirm_cfg["duration_ms"]),
            save_stride=int(confirm_cfg["save_stride"]),
            audit_tail_fraction=thresholds.tail_fraction,
        )
        classified = classify_fork_batch(
            [metadata[index] for index in candidate_indices], confirm_sim, thresholds
        )
        for local_index, screen_index in enumerate(candidate_indices):
            confirm_rows.append(
                {
                    **classified[local_index],
                    "screen_classification": screen_rows[screen_index]["classification"],
                    "confirm_dt_ms": float(confirm_cfg["dt_ms"]),
                    "confirm_duration_ms": float(confirm_cfg["duration_ms"]),
                }
            )

    alpha0_parity = _alpha0_parity_audit(
        root_rows,
        z_values,
        dt_ms=float(screen_cfg["dt_ms"]),
        duration_ms=float(screen_cfg["duration_ms"]),
        save_stride=int(screen_cfg["save_stride"]),
        audit_tail_fraction=thresholds.tail_fraction,
    )
    confirmed_local = [
        index for index, row in enumerate(confirm_rows) if row["classification"] in {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}
    ]
    ablation_rows: list[dict] = []
    if confirmed_local and confirm_sim is not None:
        ablation_rows = _run_ablation(
            confirmed_local,
            candidate_indices,
            initial_states,
            parameters,
            metadata,
            confirm_sim,
            cfg["confirm"],
            thresholds,
        )

    summary = summarize_stage0c(
        root_rows,
        screen_rows,
        confirm_rows,
        alpha0_parity=alpha0_parity,
        ablation_rows=ablation_rows,
    )
    elapsed = time.perf_counter() - start
    max_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    summary.update(
        {
            "schema_version": "topic4_spatial_slowfast_stage0c.v1",
            "config": str(config_path.resolve()),
            "config_sha256": _fingerprint(config_path),
            "implementation_sha256": _fingerprint(ROOT / "src" / "topic4_spatial_slowfast_stage0c.py"),
            "axes": axes,
            "pool": cfg["pool"],
            "root_coverage_audit": root_coverage_audit,
            "coverage_boundary": {
                "alpha_G_sampled": alpha_values,
                "unsampled_open_interval": [0.0, 1.0],
                "interpretation_cn": (
                    "alpha_G=0与1之间未采样；即使本轮clean no-go，也只能裁决锁定alpha网格，"
                    "不能外推为整个动态池机制全局no-go。"
                ),
                "subunit_refinement_run": False,
            },
            "screen_contract": screen_cfg,
            "confirm_contract": cfg["confirm"],
            "n_screen_forks": len(screen_rows),
            "n_confirm_forks": len(confirm_rows),
            "resource_usage": {
                "wall_seconds": elapsed,
                "max_rss_gib": max_rss_gib,
                "max_memory_gib_contract": float(cfg["resource_contract"]["max_memory_gib"]),
                "within_memory_contract": max_rss_gib < float(cfg["resource_contract"]["max_memory_gib"]),
                "execution": "single_process_blas_threads_1",
            },
            "scientific_boundary_cn": (
                "Stage0C只检验冻结z、r=0、均匀无噪声9D快系统中的动态除法池是否产生有限快态对象；"
                "它不证明发作、自发转换、终止、retrigger或空间pattern。"
            ),
        }
    )
    if not root_coverage_audit["pass"]:
        summary["verdict"] = "INCONCLUSIVE_ROOT_COVERAGE_DIRECTION_MISMATCH"
        summary["stage0c_pass"] = False
        summary["open_phi_or_spatial"] = False
        summary["stop_rule_triggered"] = False
        summary["reason_cn"] = "正向/反向warm continuation的root集合不一致，不能作拓扑裁决。"
    if not summary["resource_usage"]["within_memory_contract"]:
        summary["verdict"] = "ENGINEERING_FAIL_MEMORY_CONTRACT"
        summary["stage0c_pass"] = False
        summary["open_phi_or_spatial"] = False
        summary["stop_rule_triggered"] = False
        summary["reason_cn"] = "峰值内存超过4 GiB合同，结果不可验收。"

    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "stage0c_summary.json", summary)
    _atomic_json(output / "root_continuation.json", root_rows)
    _atomic_json(output / "root_continuation_reverse.json", reverse_root_rows)
    _atomic_json(output / "state_fork_screen.json", screen_rows)
    _atomic_json(output / "state_fork_confirm.json", confirm_rows)
    _atomic_json(output / "mechanism_ablation.json", ablation_rows)
    flat_roots = [
        {"z": point["z"], "alpha_G": point["alpha_G"], "w_ee_mult": point["w_ee_mult"], "ratio": point["ratio"], **root}
        for point in root_rows
        for root in point["roots"]
    ]
    _write_csv(output / "root_table.csv", flat_roots)
    _write_csv(output / "state_fork_screen.csv", screen_rows)
    _write_csv(output / "state_fork_confirm.csv", confirm_rows)
    _write_csv(output / "mechanism_ablation.csv", ablation_rows)
    _plot_screen(
        output,
        root_rows,
        screen_rows,
        metadata,
        screen_sim,
        z_values,
        alpha_values,
        summary["verdict"],
    )
    status = (
        "# Stage 0C 动态除法池状态\n\n"
        f"- 结论：`{summary['verdict']}`\n"
        f"- 是否找到有限快态对象：`{summary['stage0c_pass']}`\n"
        f"- 参数点 / screen forks / confirm forks：{len(root_rows)} / {len(screen_rows)} / {len(confirm_rows)}\n"
        f"- alpha_G=0 复刻 Stage0B：`{alpha0_parity['pass']}`\n"
        f"- 双向 root-set 一致：`{root_coverage_audit['pass']}`\n"
        "- coverage 边界：alpha_G 的开放区间 (0,1) 未采样；本轮不自行扩参\n"
        f"- wall / peak RSS：{elapsed:.2f} s / {max_rss_gib:.3f} GiB\n"
        f"- 解释：{summary['reason_cn']}\n\n"
        "本阶段冻结 z、关闭 local recovery r、噪声、空间耦合和 dynamic phi。"
        "因此结果只裁决 M4 两级动态除法池能否在均匀快系统中造出有限对象，不能写成发作或恢复结论。\n"
    )
    _atomic_text(output / "STATUS.md", status)
    return summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked Stage0C screen")
    summary, output = run(args.config)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": summary["verdict"],
                "stage0c_pass": summary["stage0c_pass"],
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
