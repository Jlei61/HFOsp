#!/usr/bin/env python3
"""Run the mass-balanced P=2 frozen-sheet diagnostic for the additive MZ line."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from src.topic4_mz_entry_exit_nullclines import find_equilibria  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import (  # noqa: E402
    integrate_frozen_patch_batch,
    lift_product_history,
    sheet_label,
    summarize_local_state,
)
from src.topic4_mz_spatial_patch import (  # noqa: E402
    PatchKernels,
    PatchParameters,
    prepare_patch_rhs,
)
from src.topic4_mz_spatial_reduction import canonical_m3b_core_surround  # noqa: E402
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state  # noqa: E402
from src.topic4_spatial_slowfast_stage0c_transfer import ExtendedSiegertTransfer  # noqa: E402
from src.topic4_spatial_slowfast_stage0f import SmoothDomain  # noqa: E402
from src.topic4_spatial_slowfast_stage0f_v1_1 import SmoothSiegertTransferV11  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_spatial_p2_frozen_sheets.yaml"
COLORS = {"core": "#B2182B", "surround": "#2166AC"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_inputs(cfg: dict) -> dict[str, str]:
    keys = ("transfer_path", "orbit_cycle_path", "orbit_summary_path")
    if set(cfg["input_sha256"]) != set(keys):
        raise ValueError(f"input_sha256 must lock exactly {keys}")
    observed = {}
    for key in keys:
        path = ROOT / cfg[key]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed[key] = _sha256(path)
        if observed[key] != str(cfg["input_sha256"][key]):
            raise RuntimeError(f"locked input drift for {key}: {observed[key]}")
    return observed


def _load_transfer(cfg: dict) -> Any:
    with np.load(ROOT / cfg["transfer_path"], allow_pickle=False) as payload:
        if not bool(payload["no_clip"]):
            raise RuntimeError("source transfer did not assert no clipping")
        exact = ExtendedSiegertTransfer(
            payload["mu_axis"], payload["sigma_axis"], payload["log_integral_table"],
            name=str(payload["transfer_name"]),
        )
    smooth = cfg["smooth_transfer"]
    return SmoothSiegertTransferV11.from_extended(
        exact,
        domain=SmoothDomain(
            float(smooth["mu_min_mv"]), float(smooth["mu_max_mv"]),
            float(smooth["sigma_min_mv"]), float(smooth["sigma_max_mv"]),
        ),
        kx=int(smooth["spline_degree_mu"]),
        ky=int(smooth["spline_degree_sigma"]),
        smoothing=float(smooth["smoothing"]),
    )


def _parameters(cfg: dict) -> tuple[PoolParameters, PatchParameters]:
    model = cfg["model"]
    low = PoolParameters(
        float(model["z_low"]), float(model["alpha_G"]),
        float(model["w_ee_mult"]), float(model["ratio"]),
    )
    patch = PatchParameters(
        alpha_g=float(model["alpha_G"]),
        w_ee_mult=float(model["w_ee_mult"]),
        ratio=float(model["ratio"]),
        additive_max_mv=float(model["additive_max_mv"]),
        pool_p=float(model["pool_p"]),
    )
    return low, patch


def _low_template(cfg: dict, transfer: Any, params: PoolParameters) -> tuple[np.ndarray, dict]:
    roots = find_equilibria(params, transfer, 0.0)
    stable = [row for row in roots if row["stability"] == "stable" and row["rE_hz"] < 5.0]
    if not stable:
        raise RuntimeError("no stable low root at registered z_low")
    root = min(stable, key=lambda row: row["rE_hz"])
    state = equilibrium_state((1e-3 * root["rE_hz"], 1e-3 * root["rI_hz"]))
    return state, root


def _phase_state(trace: np.ndarray, fraction: float) -> np.ndarray:
    index = int(round(float(fraction) * float(trace.shape[0] - 1)))
    return np.asarray(trace[index], dtype=float)


def _case_specs(phases: list[float]) -> list[dict]:
    rows = [
        {"case": "LL_base", "initial_sheet": "LL", "core_phase": None, "surround_phase": None},
        {"case": "LL_antisymmetric", "initial_sheet": "LL", "core_phase": None,
         "surround_phase": None, "antisymmetric": True},
    ]
    rows.extend(
        {"case": f"CL_phase_{phase:.2f}", "initial_sheet": "CL", "core_phase": phase,
         "surround_phase": None}
        for phase in phases
    )
    rows.extend(
        {"case": f"LC_phase_{phase:.2f}", "initial_sheet": "LC", "core_phase": None,
         "surround_phase": phase}
        for phase in phases
    )
    rows.extend(
        {"case": f"CC_relative_{phase:.2f}", "initial_sheet": "CC", "core_phase": 0.0,
         "surround_phase": phase}
        for phase in phases
    )
    rows.append(
        {"case": "CC_sync_antisymmetric", "initial_sheet": "CC", "core_phase": 0.0,
         "surround_phase": 0.0, "antisymmetric": True}
    )
    return rows


def _build_initial_batch(
    cases: list[dict],
    low: np.ndarray,
    cycle: np.ndarray,
    kernels: PatchKernels,
    patch_params: PatchParameters,
    cfg: dict,
) -> np.ndarray:
    states = []
    z_low = float(cfg["model"]["z_low"])
    z_cycle = float(cfg["model"]["z_cycle"])
    perturb = float(cfg["initial_sheets"]["antisymmetric_perturbation_khz"])
    weights = kernels.weights()
    for case in cases:
        templates = []
        z = []
        for phase in (case["core_phase"], case["surround_phase"]):
            if phase is None:
                templates.append(low)
                z.append(z_low)
            else:
                templates.append(_phase_state(cycle, phase))
                z.append(z_cycle)
        state = lift_product_history(
            np.asarray(templates), kernels, z=z, parameters=patch_params
        )
        if case.get("antisymmetric", False):
            state[0] += perturb
            state[1] -= perturb * weights[0] / weights[1]
        states.append(state)
    return np.asarray(states)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_run(
    arm: str,
    dt_ms: float,
    cases: list[dict],
    result: dict[str, Any],
    discard_returns: int,
) -> list[dict]:
    rows = []
    for index, case in enumerate(cases):
        local_rows = []
        for patch_index, patch_name in enumerate(("core", "surround")):
            local_rows.append(
                summarize_local_state(
                    result["time_ms"],
                    result["rE_khz"][:, index, patch_index],
                    result["rE_fast_khz"][:, index, patch_index],
                    result["return_times_ms"][index][patch_index],
                    support_violation_count=int(result["support_violation_count"][index, patch_index]),
                    state_bound_violation_count=int(result["state_bound_violation_count"][index, patch_index]),
                    finite=bool(result["finite"][index]),
                    discard_returns=int(discard_returns),
                )
            )
        core, surround = local_rows
        rows.append(
            {
                "arm": arm,
                "dt_ms": float(dt_ms),
                "case": case["case"],
                "initial_sheet": case["initial_sheet"],
                "core_phase": case["core_phase"],
                "surround_phase": case["surround_phase"],
                "final_sheet": sheet_label(core["status"], surround["status"]),
                "core_status": core["status"],
                "surround_status": surround["status"],
                "core_returns": core["n_returns"],
                "surround_returns": surround["n_returns"],
                "core_period_ms": core["recent_period_ms"],
                "surround_period_ms": surround["recent_period_ms"],
                "core_period_cv": core["recent_period_cv"],
                "surround_period_cv": surround["recent_period_cv"],
                "core_peak_hz": core["peak_rE_hz"],
                "surround_peak_hz": surround["peak_rE_hz"],
                "core_fraction_over_100hz": core["fraction_over_100hz"],
                "surround_fraction_over_100hz": surround["fraction_over_100hz"],
                "core_tail_mean_hz": core["tail_mean_rE_hz"],
                "surround_tail_mean_hz": surround["tail_mean_rE_hz"],
                "core_support_violations": core["support_violation_count"],
                "surround_support_violations": surround["support_violation_count"],
                "core_bound_violations": core["state_bound_violation_count"],
                "surround_bound_violations": surround["state_bound_violation_count"],
                "core_return_times_ms": json.dumps(core["return_times_ms"]),
                "surround_return_times_ms": json.dumps(surround["return_times_ms"]),
            }
        )
    return rows


def _save_trace(path: Path, cases: list[dict], result: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        time_ms=result["time_ms"].astype(np.float32),
        case=np.asarray([case["case"] for case in cases]),
        rE_khz=result["rE_khz"],
        rI_khz=result["rI_khz"],
        rE_fast_khz=result["rE_fast_khz"],
        shared_state=result["shared_state"],
        final_state=result["final_state"].astype(np.float32),
        support_violation_count=result["support_violation_count"],
        state_bound_violation_count=result["state_bound_violation_count"],
        finite=result["finite"],
    )


def _outcome_counts(rows: list[dict], *, dt_ms: float, arm: str) -> dict[str, dict[str, int]]:
    output = {}
    for sheet in ("LL", "CL", "LC", "CC"):
        selected = [
            row for row in rows
            if row["arm"] == arm and row["dt_ms"] == dt_ms and row["initial_sheet"] == sheet
            and "antisymmetric" not in row["case"]
        ]
        labels = sorted({row["final_sheet"] for row in selected})
        output[sheet] = {label: sum(row["final_sheet"] == label for row in selected) for label in labels}
    return output


def _gates(rows: list[dict], dts: list[float]) -> dict[str, bool]:
    def selected(arm: str, case: str) -> list[dict]:
        return [row for row in rows if row["arm"] == arm and row["case"] == case]

    official = "fixed_m3b_coupling"
    off = "cross_zone_synaptic_coupling_off"
    ll_cases = ("LL_base", "LL_antisymmetric")
    cc_cases = ("CC_relative_0.00", "CC_sync_antisymmetric")
    cl_cases = sorted({row["case"] for row in rows if row["initial_sheet"] == "CL"})
    status_by_key = {}
    for row in rows:
        status_by_key[(row["arm"], row["case"], row["dt_ms"])] = row["final_sheet"]
    dt_stable = all(
        len({status_by_key[(arm, case, dt)] for dt in dts}) == 1
        for arm in (official, off) for case in {row["case"] for row in rows}
    )
    official_recruit = {
        case for case in cl_cases
        if all(row["final_sheet"] == "CC" for row in selected(official, case))
    }
    off_recruit = {
        case for case in cl_cases
        if all(row["final_sheet"] == "CC" for row in selected(off, case))
    }
    return {
        "ll_low_basin_preserved": all(
            row["final_sheet"] == "LL" for case in ll_cases for row in selected(official, case)
        ),
        "uniform_cc_cycle_preserved": all(
            row["final_sheet"] == "CC" for case in cc_cases for row in selected(official, case)
        ),
        "base_half_dt_labels_match": dt_stable,
        "core_cycle_survives_intact_surround": all(
            row["core_status"] == "C" for case in cl_cases for row in selected(official, case)
        ),
        "whole_surround_recruited_any_phase": bool(official_recruit),
        "whole_surround_recruited_all_phases": bool(cl_cases) and official_recruit == set(cl_cases),
        "coupling_specific_recruitment": bool(official_recruit - off_recruit),
        "cross_zone_off_has_no_false_recruitment": not bool(off_recruit),
    }


def _plot(
    figures: Path,
    reduction: Any,
    cases: list[dict],
    base_result: dict[str, Any],
    rows: list[dict],
    base_dt: float,
) -> Path:
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.7), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(2)
    width = 0.24
    ax.bar(x - width, reduction.kernels.weights(), width, color=[COLORS["core"], COLORS["surround"]],
           label="area weight")
    ee_cross = np.asarray([reduction.kernels.K_EE[0, 1], reduction.kernels.K_EE[1, 0]])
    i_cross = np.asarray([reduction.kernels.K_I[0, 1], reduction.kernels.K_I[1, 0]])
    ax.bar(x, ee_cross, width, color="#7B3294", label="E→E cross")
    ax.bar(x + width, i_cross, width, color="#008837", label="I cross")
    ax.set_xticks(x, ["core target", "surround target"])
    ax.set_ylabel("weight")
    ax.set_title("A  Mass-balanced whole-sheet reduction")
    ax.legend(frameon=False, fontsize=7)
    ax.text(0.02, 0.97, f"core area = {100*reduction.kernels.weights()[0]:.2f}%\n"
            f"surround←core E→E = {reduction.kernels.K_EE[1,0]:.4f}",
            transform=ax.transAxes, va="top", fontsize=7.5)

    lookup = {case["case"]: index for index, case in enumerate(cases)}
    representatives = [
        (axes[0, 1], "LL_base", "B  LL seed"),
        (axes[0, 2], "CL_phase_0.00", "C  CL seed"),
        (axes[1, 0], "CC_relative_0.00", "D  CC seed"),
        (axes[1, 1], "LC_phase_0.00", "E  LC seed"),
    ]
    time = base_result["time_ms"] * 1e-3
    for panel, case_name, title in representatives:
        index = lookup[case_name]
        panel.plot(time, 1000.0 * base_result["rE_khz"][:, index, 0],
                   color=COLORS["core"], lw=0.8, label="core")
        panel.plot(time, 1000.0 * base_result["rE_khz"][:, index, 1],
                   color=COLORS["surround"], lw=0.8, label="surround")
        panel.axhline(20.0, color="0.72", lw=0.7, ls="--")
        panel.set(xlabel="time (s)", ylabel="rE (Hz)", title=title, xlim=(time[0], time[-1]))
        panel.margins(x=0)
    axes[0, 1].legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    ax.axis("off")
    counts_fixed = _outcome_counts(rows, dt_ms=base_dt, arm="fixed_m3b_coupling")
    counts_off = _outcome_counts(rows, dt_ms=base_dt, arm="cross_zone_synaptic_coupling_off")
    lines = ["F  Frozen-sheet outcomes", "", "initial   fixed K        K=I (shared pool kept)"]
    for sheet in ("LL", "CL", "LC", "CC"):
        left = ", ".join(f"{key}:{value}" for key, value in counts_fixed[sheet].items()) or "—"
        right = ", ".join(f"{key}:{value}" for key, value in counts_off[sheet].items()) or "—"
        lines.append(f"{sheet:<7} {left:<14} {right}")
    lines.extend(["", "Whole-surround P=2 is a patch-constant diagnostic.",
                  "A negative CL→CC result cannot reject a local front."])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.2)

    fig.suptitle("Fixed spatial coupling: which fast sheet is reached?", fontsize=13, fontweight="bold")
    stem = figures / "mz_spatial_p2_frozen_sheets"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem.with_suffix(".png")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    low_params, patch_params = _parameters(cfg)
    orbit_summary = json.loads((ROOT / cfg["orbit_summary_path"]).read_text(encoding="utf-8"))
    if orbit_summary["model_contract"]["fast_system"] != "locked Stage0C nine-state system":
        raise RuntimeError("orbit input is not the locked Stage-0C system")
    low, low_root = _low_template(cfg, transfer, low_params)
    trace_key = str(cfg["initial_sheets"]["cycle_trace_key"])
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{trace_key}_state"], dtype=float)

    geometry = cfg["geometry"]
    reduction = canonical_m3b_core_surround(
        grid_n=int(geometry["grid_n"]),
        grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    weights = reduction.kernels.weights()
    identity = PatchKernels(np.eye(2), np.eye(2), weights).validate()
    arms = {
        "fixed_m3b_coupling": reduction.kernels,
        "cross_zone_synaptic_coupling_off": identity,
    }
    phases = [float(value) for value in cfg["initial_sheets"]["phase_fractions"]]
    cases = _case_specs(phases)
    if len(cases) != int(cfg["resource_contract"]["vectorized_forks"]):
        raise RuntimeError("case count drifted from the resource contract")

    output = ROOT / cfg["result_root"]
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    integration = cfg["integration"]
    all_rows: list[dict] = []
    base_result = None
    base_dt = float(integration["dt_ms"][0])
    for arm_name, kernels in arms.items():
        initial = _build_initial_batch(cases, low, cycle, kernels, patch_params, cfg)
        prepared = prepare_patch_rhs(kernels, patch_params)
        for dt_value in integration["dt_ms"]:
            dt_ms = float(dt_value)
            result = integrate_frozen_patch_batch(
                initial,
                prepared,
                transfer,
                dt_ms=dt_ms,
                duration_ms=float(integration["duration_ms"]),
                save_dt_ms=float(integration["save_dt_ms"]),
                section_level_khz=float(integration["section_level_rE_fast_khz"]),
                rearm_level_khz=float(integration["rearm_level_rE_fast_khz"]),
            )
            rows = _summarize_run(
                arm_name, dt_ms, cases, result, int(integration["discard_returns"])
            )
            all_rows.extend(rows)
            suffix = str(dt_ms).replace(".", "p")
            _save_trace(output / f"p2_{arm_name}_dt{suffix}_traces.npz", cases, result)
            if arm_name == "fixed_m3b_coupling" and dt_ms == base_dt:
                base_result = result
    if base_result is None:
        raise RuntimeError("missing primary trace")

    gates = _gates(all_rows, [float(value) for value in integration["dt_ms"]])
    mass_errors = {
        "K_EE_row_sum": float(np.max(np.abs(reduction.kernels.K_EE.sum(axis=1) - 1.0))),
        "K_I_row_sum": float(np.max(np.abs(reduction.kernels.K_I.sum(axis=1) - 1.0))),
        "K_EE_stationarity": float(np.max(np.abs(weights @ reduction.kernels.K_EE - weights))),
        "K_I_stationarity": float(np.max(np.abs(weights @ reduction.kernels.K_I - weights))),
    }
    if not gates["ll_low_basin_preserved"] or not gates["uniform_cc_cycle_preserved"]:
        status = "P2_FAST_SCAFFOLD_INVALID_STOP_BEFORE_SLOW_DYNAMICS"
    elif gates["coupling_specific_recruitment"]:
        status = "P2_WHOLE_SURROUND_COUPLING_SPECIFIC_RECRUITMENT_SUPPORTED"
    elif not gates["core_cycle_survives_intact_surround"]:
        status = "P2_FOCAL_CYCLE_NOT_MAINTAINED_BY_CURRENT_SHARED_POOL"
    else:
        status = "P2_WHOLE_SURROUND_NO_RECRUITMENT_DILUTION_LIMITED"

    _write_csv(output / "p2_frozen_sheet_outcomes.csv", all_rows)
    figure = _plot(figures, reduction, cases, base_result, all_rows, base_dt)
    summary = {
        "status": status,
        "scientific_layer": "mass_balanced_patch_constant_fast_sheet_diagnostic_not_wavefront_or_lifecycle",
        "geometry": {
            "grid_n": reduction.grid_n,
            "grid_L_mm": reduction.grid_L_mm,
            "grid_spacing_mm": reduction.grid_spacing_mm,
            "core_radius_mm": reduction.core_radius_mm,
            "core_cells": reduction.core_cells,
            "surround_cells": reduction.surround_cells,
            "patch_weights": weights.tolist(),
            "K_EE": reduction.kernels.K_EE.tolist(),
            "K_I": reduction.kernels.K_I.tolist(),
            "mass_balance_errors": mass_errors,
            "boundary": geometry["boundary"],
            "partition": geometry["partition"],
        },
        "anchors": {
            "low_z": float(cfg["model"]["z_low"]),
            "low_root": low_root,
            "cycle_z": float(cfg["model"]["z_cycle"]),
            "cycle_trace_key": trace_key,
            "cycle_phase_fractions": phases,
            "product_history_lift": "spatial K on synaptic histories; true area weights on one shared muG/SG",
        },
        "gates": gates,
        "outcome_counts_base_dt": {
            arm: _outcome_counts(all_rows, dt_ms=base_dt, arm=arm) for arm in arms
        },
        "claim_boundary": [
            "the exact core/full-complement Galerkin reduction preserves constant fields and tissue area mass",
            "K=I removes cross-zone synaptic coupling but deliberately keeps the shared global pool",
            "whole-surround averaging dilutes a local boundary wake; failure to recruit this patch cannot reject a local front",
            "z/p/m are frozen, so no onset, termination, recovery, latch, or lifecycle is claimed",
            "no E-E weight, width, direction, delay, relay, or conductance parameter was changed",
        ],
        "next_step": (
            "conditional_entry_exit_boundaries_then_P3_core_annulus_bath"
            if gates["ll_low_basin_preserved"] and gates["uniform_cc_cycle_preserved"]
            else "repair_fast_spatial_scaffold_before_any_slow_dynamics"
        ),
        "input_sha256": hashes,
        "resource_contract": cfg["resource_contract"],
        "artifacts": {
            "figure": str(figure.relative_to(ROOT)),
            "outcome_csv": str((output / "p2_frozen_sheet_outcomes.csv").relative_to(ROOT)),
            "trace_glob": str((output / "p2_*_traces.npz").relative_to(ROOT)),
        },
        "config": cfg,
    }
    (output / "p2_frozen_sheet_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (figures / "README.md").write_text(
        "### mz_spatial_p2_frozen_sheets.png\n\n"
        "这张图是 additive MZ 路线的 P=2 frozen-fast 诊断。A 显示 canonical M3B 单核/全补集约化的真实面积权重和跨区耦合；B–E 显示 LL、CL、CC、LC 代表初态在固定慢变量下落入哪个 fast sheet；F 汇总 fixed K 与 K=I 对照。\n\n"
        "它不是正式 Figure 4/5，也不声称完整发作生命周期。whole-surround 把全部远场压成一个变量，会稀释局部 front；因此 CL 未转 CC 只能指导下一步 P3 core–annulus–bath，不能否定空间传播。\n\n"
        "**关注点**：先看 LL 与同步 CC 是否保存，再看 CL 中 core 能否维持、surround 是否只在 fixed K 而非 K=I 下获得 bounded returns。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps({"status": summary["status"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
