#!/usr/bin/env python3
"""Run the locked Stage-0C extended-Siegert transfer-support audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl-stage0c-transfer")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sef_hfo_lif import C_EE, TAU_ME, TREF_E, TREF_I, W_EE, lif_rate  # noqa: E402
from src.topic4_spatial_slowfast_stage0b import ForkClassifierThresholds  # noqa: E402
from src.topic4_spatial_slowfast_stage0c import build_state_forks  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import (  # noqa: E402
    COARSE_RESOLUTION,
    FINE_RESOLUTION,
    ExtendedSiegertTransfer,
    TransferResolution,
    TransferSupport,
    classify_extended_batch,
    direct_exact_error_audit,
    resolution_pair_status,
    simulate_extended_forks,
    stable_siegert_log_rate,
    stable_siegert_rate,
)


DEFAULT_CONFIG = ROOT / "config" / "topic4_spatial_slowfast_stage0c_transfer.yaml"
LOCKED_POINTS = ((0.80, 12.0), (0.85, 16.0), (0.81, 16.0), (0.84, 24.0), (0.82, 24.0), (0.84, 32.0))


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


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classifier(cfg: dict) -> ForkClassifierThresholds:
    return ForkClassifierThresholds(**cfg["classifier"]).validate()


def _validate_config(cfg: dict) -> tuple[TransferSupport, dict[str, TransferResolution]]:
    points = tuple((round(float(row["z"]), 2), float(row["alpha_G"])) for row in cfg["candidate_points"])
    if points != LOCKED_POINTS:
        raise ValueError(f"candidate points drifted: {points!r}")
    support = TransferSupport(**cfg["transfer_support"]).validate()
    if support != TransferSupport():
        raise ValueError("extended transfer support drifted from locked domain")
    resolutions = {
        name: TransferResolution(name, **settings).validate()
        for name, settings in cfg["resolutions"].items()
    }
    if resolutions != {"coarse": COARSE_RESOLUTION, "fine": FINE_RESOLUTION}:
        raise ValueError("coarse/fine transfer resolutions drifted")
    expected_runs = {
        "screen": (0.25, 6000.0, 20),
        "confirm": (0.125, 12000.0, 40),
        "dt_half": (0.0625, 12000.0, 80),
    }
    for name, expected in expected_runs.items():
        observed = cfg[name]
        if (float(observed["dt_ms"]), float(observed["duration_ms"]), int(observed["save_stride"])) != expected:
            raise ValueError(f"{name} integration contract drifted")
    if any(bool(value) for value in cfg["scope"].values()):
        raise ValueError("all out-of-scope expansion switches must remain false")
    if int(cfg["resource_contract"]["blas_threads"]) != 1:
        raise ValueError("BLAS threads must equal one")
    return support, resolutions


def _load_locked_forks(cfg: dict) -> tuple[list[dict], np.ndarray, list, dict]:
    root_path = ROOT / cfg["primary_root_continuation"]
    screen_path = ROOT / cfg["primary_screen"]
    root_rows = json.loads(root_path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in root_rows
        if (round(float(row["z"]), 2), float(row["alpha_G"])) in LOCKED_POINTS
    ]
    ordering = [(round(float(row["z"]), 2), float(row["alpha_G"])) for row in selected]
    if set(ordering) != set(LOCKED_POINTS) or len(selected) != len(LOCKED_POINTS):
        raise RuntimeError("primary root artifact does not contain exactly the six locked points")
    selected.sort(key=lambda row: LOCKED_POINTS.index((round(float(row["z"]), 2), float(row["alpha_G"]))))
    metadata_all, states_all, params_all = build_state_forks(selected)
    keep = [index for index, row in enumerate(metadata_all) if row["initial_kind"] != "exact_root"]
    metadata = [dict(metadata_all[index]) for index in keep]
    states = states_all[keep]
    params = [params_all[index] for index in keep]
    labels_by_point: dict[tuple[float, float], list[tuple[str, str]]] = defaultdict(list)
    for row in metadata:
        labels_by_point[(round(float(row["z"]), 2), float(row["alpha_G"]))].append(
            (str(row["initial_kind"]), str(row["initial_label"]))
        )
    label_sets = list(labels_by_point.values())
    if len(metadata) != 102 or any(len(labels) != 17 for labels in label_sets) or any(labels != label_sets[0] for labels in label_sets[1:]):
        raise RuntimeError("locked replay requires the same 17 non-exact forks at all six points")

    primary_rows = json.loads(screen_path.read_text(encoding="utf-8"))
    primary_lookup = {
        (
            round(float(row["z"]), 2),
            float(row["alpha_G"]),
            str(row["initial_kind"]),
            str(row["initial_label"]),
        ): row
        for row in primary_rows
        if row["initial_kind"] != "exact_root"
    }
    for row in metadata:
        key = (round(float(row["z"]), 2), float(row["alpha_G"]), row["initial_kind"], row["initial_label"])
        source = primary_lookup[key]
        row["primary_classification"] = source["classification"]
        row["primary_pre_audit_classification"] = source.get("pre_audit_classification")
        row["primary_tail_mean_hz"] = source.get("tail_mean_hz")
        row["primary_tail_peak_hz"] = source.get("tail_peak_hz")
        row["primary_lut_clip_tail_occupancy_stepwise"] = source.get("lut_clip_tail_occupancy_stepwise")
    provenance = {
        "root_continuation": str(root_path.resolve()),
        "root_continuation_sha256": _sha256(root_path),
        "state_fork_screen": str(screen_path.resolve()),
        "state_fork_screen_sha256": _sha256(screen_path),
        "n_parameter_points": len(selected),
        "n_nonexact_forks_per_point": 17,
        "n_nonexact_forks": len(metadata),
        "fixed_labels_identical_across_points": True,
        "n_primary_audit_invalid_candidates_in_replay": int(
            sum(row["primary_classification"] == "audit_invalid_candidate" for row in metadata)
        ),
    }
    return metadata, states, params, provenance


def _save_transfer(path: Path, transfer: ExtendedSiegertTransfer) -> None:
    np.savez_compressed(
        path,
        mu_axis=transfer.mu_axis,
        sigma_axis=transfer.sigma_axis,
        log_integral_table=transfer.log_integral_table,
        transfer_name=np.asarray(transfer.name),
        no_clip=np.asarray(True),
    )


def _exact_reference_audit() -> dict:
    finite_errors: list[float] = []
    relative_errors: list[float] = []
    n_nonfinite_canonical = 0
    for mu in (-40.0, -30.0, -10.0, 0.0, 10.0, 18.0, 40.0, 80.0, 120.0):
        for sigma in (0.5, 1.0, 3.0, 10.0, 20.0, 30.0):
            for tau_m, tau_ref in ((TAU_ME, TREF_E), (10.0, TREF_I)):
                exact = stable_siegert_rate(mu, sigma, tau_m, tau_ref)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    canonical = lif_rate(mu, sigma, tau_m, tau_ref)
                if not np.isfinite(canonical):
                    n_nonfinite_canonical += 1
                    continue
                finite_errors.append(abs(exact - canonical))
                if canonical >= 1e-12:
                    relative_errors.append(abs(exact - canonical) / canonical)
    mus = np.linspace(-2500.0, -40.0, 80)
    log_rates = np.asarray([stable_siegert_log_rate(float(mu), 3.0, TAU_ME, TREF_E) for mu in mus])
    branch_mu = 18.0 - 6.0 * 3.0
    branch_jump = abs(
        stable_siegert_log_rate(branch_mu - 1e-7, 3.0, TAU_ME, TREF_E)
        - stable_siegert_log_rate(branch_mu + 1e-7, 3.0, TAU_ME, TREF_E)
    )
    passed = bool(
        finite_errors
        and max(finite_errors) <= 2e-12
        and (not relative_errors or max(relative_errors) <= 2e-8)
        and np.all(np.diff(log_rates) > 0.0)
        and np.all(np.isfinite(log_rates))
        and branch_jump <= 2e-5
    )
    return {
        "pass": passed,
        "n_finite_canonical_comparisons": len(finite_errors),
        "n_nonfinite_canonical_extreme_points": n_nonfinite_canonical,
        "max_abs_rate_error_khz": max(finite_errors),
        "max_relative_rate_error_above_1e-12_khz": max(relative_errors) if relative_errors else 0.0,
        "extreme_low_mu_log_rate_finite": bool(np.all(np.isfinite(log_rates))),
        "extreme_low_mu_monotone": bool(np.all(np.diff(log_rates) > 0.0)),
        "endpoint_branch_log_rate_jump": float(branch_jump),
        "mu_range_monotonicity_audit_mv": [-2500.0, -40.0],
    }


def _overlap_lut_audit(transfer: ExtendedSiegertTransfer, *, seed: int = 20260720, n: int = 2048) -> dict:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-40.0, 120.0, n)
    sigma = rng.uniform(0.5, 30.0, n)
    rows = []
    for pop, tau_m, tau_ref in (("E", TAU_ME, TREF_E), ("I", 10.0, TREF_I)):
        approximate = transfer.rate(mu, sigma, pop)
        exact = np.asarray(
            [stable_siegert_rate(float(m), float(s), tau_m, tau_ref) for m, s in zip(mu, sigma)]
        )
        absolute_hz = 1000.0 * np.abs(approximate - exact)
        meaningful = exact >= 1e-9
        relative = np.abs(approximate[meaningful] - exact[meaningful]) / exact[meaningful]
        rows.append(
            {
                "population": pop,
                "n": n,
                "max_abs_error_hz": float(np.max(absolute_hz)),
                "p99_abs_error_hz": float(np.percentile(absolute_hz, 99.0)),
                "max_relative_error_meaningful": float(np.max(relative)) if relative.size else 0.0,
                "p99_relative_error_meaningful": float(np.percentile(relative, 99.0)) if relative.size else 0.0,
            }
        )
    return {
        "transfer": transfer.name,
        "domain": {"mu_mv": [-40.0, 120.0], "sigma_mv": [0.5, 30.0]},
        "seed": seed,
        "populations": rows,
        "pass": bool(
            all(row["max_abs_error_hz"] <= (0.5 if transfer.name == "coarse" else 0.25) for row in rows)
            and all(row["p99_relative_error_meaningful"] <= (0.05 if transfer.name == "coarse" else 0.02) for row in rows)
        ),
        "acceptance": {
            "max_abs_error_hz": 0.5 if transfer.name == "coarse" else 0.25,
            "p99_relative_error_meaningful": 0.05 if transfer.name == "coarse" else 0.02,
        },
    }


def _save_simulation(path: Path, simulation: dict[str, np.ndarray]) -> None:
    keep = {
        key: value
        for key, value in simulation.items()
        if key in {
            "time_ms",
            "rE_khz",
            "rI_khz",
            "rE_fast_khz",
            "mu_G",
            "S_G",
            "muE_mV",
            "sigmaE_mV",
            "muI_mV",
            "sigmaI_mV",
            "divisor",
        }
    }
    np.savez_compressed(path, **keep)


def _run_resolution(
    metadata: list[dict],
    states: np.ndarray,
    params: list,
    transfer: ExtendedSiegertTransfer,
    run_cfg: dict,
    thresholds: ForkClassifierThresholds,
    phase: str,
    *,
    mechanism: str = "dynamic",
    clamp_s: float | None = None,
    subtractive_beta_mv: float | None = None,
) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    simulation = simulate_extended_forks(
        states,
        params,
        transfer,
        dt_ms=float(run_cfg["dt_ms"]),
        duration_ms=float(run_cfg["duration_ms"]),
        save_stride=int(run_cfg["save_stride"]),
        audit_tail_fraction=thresholds.tail_fraction,
        mechanism=mechanism,
        clamp_s=clamp_s,
        subtractive_beta_mv=subtractive_beta_mv,
    )
    rows = classify_extended_batch(metadata, simulation, thresholds, transfer_name=transfer.name, phase=phase)
    exact = direct_exact_error_audit(simulation, transfer, max_points_per_fork=16)
    return simulation, rows, exact


def _pair_rows(
    coarse_rows: list[dict],
    fine_rows: list[dict],
    *,
    exact_error_pass: bool,
    phase: str,
) -> list[dict]:
    if len(coarse_rows) != len(fine_rows):
        raise RuntimeError("coarse/fine row count mismatch")
    paired: list[dict] = []
    for index, (coarse, fine) in enumerate(zip(coarse_rows, fine_rows)):
        for key in ("z", "alpha_G", "initial_kind", "initial_label"):
            if coarse[key] != fine[key]:
                raise RuntimeError(f"coarse/fine metadata mismatch at row {index}: {key}")
        status = resolution_pair_status(coarse, fine, exact_error_pass=exact_error_pass)
        paired.append(
            {
                "fork_index": index,
                "z": fine["z"],
                "alpha_G": fine["alpha_G"],
                "initial_kind": fine["initial_kind"],
                "initial_label": fine["initial_label"],
                "phase": phase,
                "transfer_audit_status": status,
                "coarse_classification": coarse["classification"],
                "fine_classification": fine["classification"],
                "coarse_tail_mean_hz": coarse.get("tail_mean_hz"),
                "fine_tail_mean_hz": fine.get("tail_mean_hz"),
                "coarse_tail_peak_hz": coarse.get("tail_peak_hz"),
                "fine_tail_peak_hz": fine.get("tail_peak_hz"),
                "coarse_frequency_hz": coarse.get("dominant_frequency_hz"),
                "fine_frequency_hz": fine.get("dominant_frequency_hz"),
                "fine_support_violation_step_count": fine["support_violation_step_count"],
                "fine_over_100hz_tail_step_count": fine["over_100hz_tail_step_count"],
                "direct_exact_error_pass": exact_error_pass,
            }
        )
    return paired


def _run_ablation(
    survivor_indices: list[int],
    fine_confirm_column_indices: list[int],
    metadata: list[dict],
    states: np.ndarray,
    params: list,
    fine_transfer: ExtendedSiegertTransfer,
    fine_confirm: dict[str, np.ndarray],
    cfg: dict,
    thresholds: ForkClassifierThresholds,
) -> list[dict]:
    rows: list[dict] = []
    if len(survivor_indices) != len(fine_confirm_column_indices):
        raise ValueError("survivor and confirm-column indices must align")
    groups: dict[tuple[float, float], list[tuple[int, int]]] = defaultdict(list)
    for original_index, confirm_column in zip(survivor_indices, fine_confirm_column_indices):
        groups[(float(metadata[original_index]["z"]), float(metadata[original_index]["alpha_G"]))].append(
            (original_index, confirm_column)
        )
    n_time = fine_confirm["time_ms"].size
    tail_start = max(1, int(np.floor((1.0 - thresholds.tail_fraction) * n_time)))
    for point, members in groups.items():
        original_indices = [member[0] for member in members]
        confirm_columns = [member[1] for member in members]
        matched_s = float(np.mean(fine_confirm["S_G"][tail_start:, confirm_columns]))
        final_s_ee = float(np.mean(fine_confirm["final_state"][confirm_columns, 2]))
        point_params = [params[index] for index in original_indices]
        rec_mean_mv = TAU_ME * C_EE * (point_params[0].w_ee_mult * W_EE) * final_s_ee
        divided_loss_mv = rec_mean_mv * (point_params[0].alpha_g * matched_s) / max(
            1.0 + point_params[0].alpha_g * matched_s, 1e-12
        )
        beta_mv = divided_loss_mv / max(matched_s, 1e-12)
        arms = (
            ("dynamic", None, None),
            ("instantaneous", None, None),
            ("clamped", matched_s, None),
            ("matched_subtractive", None, beta_mv),
            ("mean_only", None, None),
        )
        for mechanism, clamp_s, beta in arms:
            _simulation, classified, exact = _run_resolution(
                [metadata[index] for index in original_indices],
                states[original_indices],
                point_params,
                fine_transfer,
                cfg["confirm"],
                thresholds,
                "ablation",
                mechanism=mechanism,
                clamp_s=clamp_s,
                subtractive_beta_mv=beta,
            )
            exact_by_fork = {int(row["fork_index"]): row for row in exact.get("per_fork", [])}
            for fork_index, row in enumerate(classified):
                fork_exact = exact_by_fork.get(fork_index, {"pass": False})
                rows.append(
                    {
                        **row,
                        "mechanism": mechanism,
                        "matched_S_G_point": matched_s,
                        "matched_subtractive_beta_mV_per_SG": beta_mv,
                        "direct_exact_error_pass": bool(fork_exact.get("pass", False)),
                        "direct_exact_max_abs_error_hz": fork_exact.get("max_abs_error_hz"),
                        "direct_exact_p99_relative_error": fork_exact.get("p99_relative_error_meaningful"),
                    }
                )
    return rows


def _point_support(final_rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for z, alpha in LOCKED_POINTS:
        rows = [row for row in final_rows if np.isclose(row["z"], z) and np.isclose(row["alpha_G"], alpha)]
        survivors = [row for row in rows if row["final_status"] == "candidate_survives"]
        means = np.asarray([row["confirm_fine_tail_mean_hz"] for row in survivors], dtype=float)
        frequencies = np.asarray([row["confirm_fine_frequency_hz"] for row in survivors], dtype=float)
        same_object = bool(
            len(survivors) >= 2
            and float(np.ptp(means)) <= max(5.0, 0.20 * float(np.mean(means)))
            and float(np.ptp(frequencies)) <= max(1.0, 0.25 * float(np.mean(frequencies)))
        )
        output.append(
            {
                "z": z,
                "alpha_G": alpha,
                "n_forks": len(rows),
                "status_counts": dict(Counter(row["final_status"] for row in rows)),
                "n_candidate_survivors": len(survivors),
                "two_history_same_object_support": same_object,
                "survivor_labels": [row["initial_label"] for row in survivors],
                "survivor_mean_rate_hz": float(np.mean(means)) if means.size else None,
                "survivor_frequency_hz": float(np.mean(frequencies)) if frequencies.size else None,
            }
        )
    return output


def _plot_results(
    output: Path,
    metadata: list[dict],
    fine_screen: dict[str, np.ndarray],
    final_rows: list[dict],
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
        ax.semilogy(mus, np.maximum(rates, 1e-300), color=color, lw=1.5, label=fr"$\sigma$={sigma:g} mV")
    ax.axvline(-40.0, color="0.45", ls="--", lw=1.0, label="primary LUT lower edge")
    ax.set(xlabel=r"input mean $\mu$ (mV)", ylabel="exact E rate (Hz)", title="Log-domain exact Siegert reference")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    time_s = fine_screen["time_ms"] / 1000.0
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(LOCKED_POINTS)))
    for color, (z, alpha) in zip(colors, LOCKED_POINTS):
        candidate_indices = [
            index
            for index, row in enumerate(metadata)
            if np.isclose(row["z"], z)
            and np.isclose(row["alpha_G"], alpha)
            and row["primary_classification"] == "audit_invalid_candidate"
        ]
        if not candidate_indices:
            candidate_indices = [
                index
                for index, row in enumerate(metadata)
                if np.isclose(row["z"], z) and np.isclose(row["alpha_G"], alpha) and row["initial_label"] == "probe_rest"
            ]
        index = candidate_indices[0]
        ax.plot(time_s, 1000.0 * fine_screen["rE_khz"][:, index], color=color, lw=1.0, label=f"z={z:.2f}, a={alpha:g}")
    ax.axhline(100.0, color="0.45", ls="--", lw=0.9)
    ax.set(xlabel="time (s)", ylabel="E rate (Hz)", title="Fine extended-transfer replay")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 0]
    statuses = ("candidate_survives", "collapses_low", "becomes_over_100", "numerical_unresolved")
    palette = ("#2ca02c", "#4c78a8", "#e45756", "#b279a2")
    bottom = np.zeros(len(point_rows))
    labels = [f"{row['z']:.2f}\n{row['alpha_G']:g}" for row in point_rows]
    for status, color in zip(statuses, palette):
        values = np.asarray([row["status_counts"].get(status, 0) for row in point_rows])
        ax.bar(np.arange(len(point_rows)), values, bottom=bottom, color=color, label=status)
        bottom += values
    ax.set_xticks(np.arange(len(point_rows)), labels)
    ax.set(xlabel="z / alpha_G", ylabel="fixed non-exact forks", title="Final transfer-audit outcome")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    fine_overlap = audits["lut_overlap"]["fine"]
    fine_direct = audits["screen_direct_exact"]["fine"]
    lines = [
        f"Verdict: {verdict}",
        f"Stable exact parity: {audits['exact_reference']['pass']}",
        f"Fine overlap LUT: {fine_overlap['pass']}",
        f"Fine trajectory exact: {fine_direct['pass']}",
        f"Fine direct max abs: {fine_direct['max_abs_error_hz']:.3g} Hz",
        f"Fine direct p99 rel: {fine_direct['p99_relative_error_meaningful']:.3g}",
        "No clipping / no extrapolation",
        "Uniform frozen fast system only",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
    ax.set_title("Numerical acceptance and claim boundary")

    figure_path = figures / "stage0c_transfer_support_audit.png"
    fig.savefig(figure_path, dpi=190)
    fig.savefig(figure_path.with_suffix(".pdf"))
    plt.close(fig)
    _atomic_text(
        figures / "README.md",
        "### stage0c_transfer_support_audit.png\n\n"
        "这张诊断图只复核 Stage 0C 六个既有候选点是否依赖原始 transfer LUT 的低端裁剪。"
        "左上显示数值稳定的 exact-Siegert reference；右上是 fine extended transfer 下每个参数点一条"
        "预先存在的候选/固定代表初态；左下汇总 17 条固定 non-exact forks 的最终分类；右下列数值验收。"
        "本图没有 slow-variable lifecycle 或空间耦合，不能解释为发作、终止或空间传播。\n\n"
        "**关注点**：是否有绿色 survivor；若没有，区分其回到 low、越过 100 Hz，还是仍属数值未决。\n",
    )


def run(config_path: Path) -> tuple[dict, Path]:
    start = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    support, resolutions = _validate_config(cfg)
    output = ROOT / cfg["result_root"]
    output.mkdir(parents=True, exist_ok=True)
    metadata, states, params, provenance = _load_locked_forks(cfg)
    thresholds = _classifier(cfg)

    exact_reference = _exact_reference_audit()
    transfers: dict[str, ExtendedSiegertTransfer] = {}
    overlap_audits: dict[str, dict] = {}
    for name in ("coarse", "fine"):
        transfer = ExtendedSiegertTransfer.build(support, resolutions[name])
        transfers[name] = transfer
        _save_transfer(output / f"extended_transfer_{name}.npz", transfer)
        overlap_audits[name] = _overlap_lut_audit(transfer)

    screen_sim: dict[str, dict[str, np.ndarray]] = {}
    screen_rows: dict[str, list[dict]] = {}
    screen_exact: dict[str, dict] = {}
    for name in ("coarse", "fine"):
        simulation, rows, exact = _run_resolution(
            metadata, states, params, transfers[name], cfg["screen"], thresholds, "screen"
        )
        screen_sim[name], screen_rows[name], screen_exact[name] = simulation, rows, exact
        _save_simulation(output / f"state_fork_screen_{name}_traces.npz", simulation)
    screen_pairs = _pair_rows(
        screen_rows["coarse"],
        screen_rows["fine"],
        exact_error_pass=bool(screen_exact["fine"]["pass"] and overlap_audits["fine"]["pass"] and exact_reference["pass"]),
        phase="screen",
    )
    screen_survivors = [index for index, row in enumerate(screen_pairs) if row["transfer_audit_status"] == "candidate_survives"]

    confirm_sim: dict[str, dict[str, np.ndarray]] = {}
    confirm_rows: dict[str, list[dict]] = {"coarse": [], "fine": []}
    confirm_exact: dict[str, dict] = {}
    confirm_pairs: list[dict] = []
    if screen_survivors:
        confirm_metadata = [metadata[index] for index in screen_survivors]
        confirm_states = states[screen_survivors]
        confirm_params = [params[index] for index in screen_survivors]
        for name in ("coarse", "fine"):
            simulation, rows, exact = _run_resolution(
                confirm_metadata,
                confirm_states,
                confirm_params,
                transfers[name],
                cfg["confirm"],
                thresholds,
                "confirm",
            )
            confirm_sim[name], confirm_rows[name], confirm_exact[name] = simulation, rows, exact
            _save_simulation(output / f"state_fork_confirm_{name}_traces.npz", simulation)
        confirm_pairs = _pair_rows(
            confirm_rows["coarse"],
            confirm_rows["fine"],
            exact_error_pass=bool(confirm_exact["fine"]["pass"] and overlap_audits["fine"]["pass"] and exact_reference["pass"]),
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
            transfers["fine"],
            cfg["dt_half"],
            thresholds,
            "dt_half",
        )
        _save_simulation(output / "state_fork_dt_half_fine_traces.npz", dt_half_sim)

    ablation_rows: list[dict] = []
    if confirm_survivor_indices:
        ablation_rows = _run_ablation(
            confirm_survivor_indices,
            confirm_survivor_local,
            metadata,
            states,
            params,
            transfers["fine"],
            confirm_sim["fine"],
            cfg,
            thresholds,
        )

    confirm_lookup = {screen_survivors[local]: row for local, row in enumerate(confirm_pairs)}
    dt_half_lookup = {confirm_survivor_indices[local]: row for local, row in enumerate(dt_half_rows)}
    final_rows: list[dict] = []
    for index, screen_row in enumerate(screen_pairs):
        final_status = screen_row["transfer_audit_status"]
        confirm_row = confirm_lookup.get(index)
        if final_status == "candidate_survives":
            final_status = "numerical_unresolved" if confirm_row is None else confirm_row["transfer_audit_status"]
        dt_row = dt_half_lookup.get(index)
        if final_status == "candidate_survives":
            if dt_row is None:
                final_status = "numerical_unresolved"
            elif dt_row["classification"] == "low_fixed_point":
                final_status = "collapses_low"
            elif int(dt_row["over_100hz_tail_step_count"]) > 0:
                final_status = "becomes_over_100"
            elif (
                dt_row["classification"] not in {"bounded_tonic_candidate", "bounded_oscillatory_candidate"}
                or not bool(dt_half_exact.get("pass", False))
            ):
                final_status = "numerical_unresolved"
        final_rows.append(
            {
                **screen_row,
                "final_status": final_status,
                "confirm_status": None if confirm_row is None else confirm_row["transfer_audit_status"],
                "confirm_fine_classification": None if confirm_row is None else confirm_row["fine_classification"],
                "confirm_fine_tail_mean_hz": None if confirm_row is None else confirm_row["fine_tail_mean_hz"],
                "confirm_fine_tail_peak_hz": None if confirm_row is None else confirm_row["fine_tail_peak_hz"],
                "confirm_fine_frequency_hz": None if confirm_row is None else confirm_row["fine_frequency_hz"],
                "dt_half_classification": None if dt_row is None else dt_row["classification"],
                "dt_half_tail_mean_hz": None if dt_row is None else dt_row.get("tail_mean_hz"),
                "dt_half_tail_peak_hz": None if dt_row is None else dt_row.get("tail_peak_hz"),
            }
        )
    point_rows = _point_support(final_rows)
    supported_points = [row for row in point_rows if row["two_history_same_object_support"]]
    outcome_counts = dict(Counter(row["final_status"] for row in final_rows))
    if supported_points:
        verdict = "TRANSFER_SUPPORTED_FINITE_FAST_OBJECT_CANDIDATE"
        reason_cn = "至少一个锁定参数点有两条不同初态通过 fine/coarse、12 s、dt/2 与 direct-exact 数值门。"
    elif outcome_counts.get("numerical_unresolved", 0):
        verdict = "NUMERICALLY_RESOLVED_PARTIAL_NO_SUPPORTED_OBJECT"
        reason_cn = "没有参数点形成两初态支持的有限对象；部分初态仍是长瞬态或分辨率/分类未决。"
    else:
        verdict = "TRANSFER_AUDIT_CLEAN_NO_SUPPORTED_OBJECT"
        reason_cn = "extended exact-supported transfer 将全部固定初态裁决为 low 或 >100 Hz，未留下有限对象。"

    elapsed = time.perf_counter() - start
    max_rss_gib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0**2)
    audits = {
        "exact_reference": exact_reference,
        "lut_overlap": overlap_audits,
        "screen_direct_exact": screen_exact,
        "confirm_direct_exact": confirm_exact,
        "dt_half_direct_exact": dt_half_exact,
    }
    summary = {
        "schema_version": "topic4_stage0c_transfer_support_audit.v1",
        "verdict": verdict,
        "reason_cn": reason_cn,
        "candidate_survives": bool(supported_points),
        "n_supported_parameter_points": len(supported_points),
        "supported_parameter_points": supported_points,
        "final_status_counts": outcome_counts,
        "n_screen_survivor_forks": len(screen_survivors),
        "n_confirm_survivor_forks": len(confirm_survivor_indices),
        "dt_half_run": bool(confirm_survivor_indices),
        "five_arm_ablation_run": bool(ablation_rows),
        "n_ablation_rows": len(ablation_rows),
        "point_outcomes": point_rows,
        "provenance": provenance,
        "transfer_contract": {
            "support": cfg["transfer_support"],
            "resolutions": cfg["resolutions"],
            "interpolation": "linear_in_log_siegert_integral_on_irregular_mu_grid",
            "outside_support": "NaN_fail_closed",
            "clipping": False,
            "extrapolation": False,
            "candidate_points": cfg["candidate_points"],
            "nonexact_forks_per_point": 17,
        },
        "integration_contract": {key: cfg[key] for key in ("screen", "confirm", "dt_half")},
        "numerical_audits": audits,
        "resource_usage": {
            "wall_seconds": elapsed,
            "max_rss_gib": max_rss_gib,
            "max_memory_gib_contract": float(cfg["resource_contract"]["max_memory_gib"]),
            "within_memory_contract": max_rss_gib < float(cfg["resource_contract"]["max_memory_gib"]),
            "execution": "single_process_blas_threads_1",
        },
        "scientific_boundary_cn": (
            "本审计只判断六个既有 Stage0C 候选点是否由原 LUT 裁剪伪造。即使有 survivor，"
            "也只是假定 z 冻结、无空间、无噪声的均匀快系统对象，不能直接开放 Stage1 或写成发作生命周期。"
        ),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(ROOT / "src" / "topic4_spatial_slowfast_stage0c_transfer.py"),
    }
    if not exact_reference["pass"] or not overlap_audits["fine"]["pass"]:
        summary["verdict"] = "NUMERICAL_UNRESOLVED_TRANSFER_VALIDATION_FAILED"
        summary["candidate_survives"] = False
        summary["reason_cn"] = "exact reference 或 fine LUT overlap parity 未通过，禁止动力学裁决。"
    if not summary["resource_usage"]["within_memory_contract"]:
        summary["verdict"] = "ENGINEERING_FAIL_MEMORY_CONTRACT"
        summary["candidate_survives"] = False
        summary["reason_cn"] = "峰值内存超过 4 GiB 合同。"

    _atomic_json(output / "transfer_validation.json", audits)
    _atomic_json(output / "state_fork_screen_coarse.json", screen_rows["coarse"])
    _atomic_json(output / "state_fork_screen_fine.json", screen_rows["fine"])
    _atomic_json(output / "state_fork_screen_paired.json", screen_pairs)
    _atomic_json(output / "state_fork_confirm_coarse.json", confirm_rows["coarse"])
    _atomic_json(output / "state_fork_confirm_fine.json", confirm_rows["fine"])
    _atomic_json(output / "state_fork_confirm_paired.json", confirm_pairs)
    _atomic_json(output / "state_fork_dt_half.json", dt_half_rows)
    _atomic_json(output / "mechanism_ablation.json", ablation_rows)
    _atomic_json(output / "final_fork_outcomes.json", final_rows)
    _atomic_json(output / "point_outcomes.json", point_rows)
    _atomic_json(output / "stage0c_transfer_support_summary.json", summary)
    _write_csv(output / "final_fork_outcomes.csv", final_rows)
    _write_csv(output / "point_outcomes.csv", point_rows)
    _write_csv(output / "mechanism_ablation.csv", ablation_rows)
    _plot_results(output, metadata, screen_sim["fine"], final_rows, point_rows, audits, summary["verdict"])
    _atomic_text(
        output / "STATUS.md",
        "# Stage 0C transfer-support audit 状态\n\n"
        f"- 结论：`{summary['verdict']}`\n"
        f"- final fork counts：`{summary['final_status_counts']}`\n"
        f"- supported points：{summary['n_supported_parameter_points']}\n"
        f"- screen / confirm survivor forks：{len(screen_survivors)} / {len(confirm_survivor_indices)}\n"
        f"- exact reference / fine overlap：{exact_reference['pass']} / {overlap_audits['fine']['pass']}\n"
        f"- clipping / extrapolation：False / False\n"
        f"- wall / peak RSS：{elapsed:.2f} s / {max_rss_gib:.3f} GiB\n"
        f"- 解释：{summary['reason_cn']}\n\n"
        "这不是新参数搜索，也不含 slow lifecycle 或 spatial coupling。\n",
    )
    return summary, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_run:
        parser.error("pass --confirm-run to execute the locked transfer-support audit")
    summary, output = run(args.config)
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": summary["verdict"],
                "final_status_counts": summary["final_status_counts"],
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
