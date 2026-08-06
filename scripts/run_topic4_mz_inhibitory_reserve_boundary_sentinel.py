#!/usr/bin/env python3
"""Resolve the lower fixed-q smooth-ramp boundary after the accepted R0b run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from scripts.run_topic4_mz_inhibitory_reserve_corridor_r0b import (  # noqa: E402
    _all_nonempty,
    _cartesian_complete,
    _frozen_view,
    _load_csv,
    _no_failclosed_violation,
    _ramp_parameters,
    _save_csv,
    _solve_folds,
)
from scripts.run_topic4_mz_spatial_regional_entry_exit import (  # noqa: E402
    _checkpoint,
    _cycle_initial,
    _load_transfer,
    _low_initial,
    _low_template,
    _model,
    _pattern_summary,
    _validate_inputs,
)
from src.topic4_mz_spatial_autonomous_latch import integrate_autonomous_latch_batch  # noqa: E402
from src.topic4_mz_spatial_frozen_sheets import integrate_frozen_patch_batch  # noqa: E402
from src.topic4_mz_spatial_patch import prepare_patch_rhs  # noqa: E402
from src.topic4_mz_spatial_reduction import canonical_m3b_core_annulus_bath  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_mz_inhibitory_reserve_corridor_r0b.yaml"


def _summarize_boundary_sentinel(
    rows: list[dict[str, Any]],
    known_failure_rows: list[dict[str, Any]],
    cfg: dict,
) -> dict[str, Any]:
    """Summarize confirmed anchors without claiming a monotone boundary."""

    screen = cfg["r0b"]
    q_axis = [float(value) for value in screen["lower_boundary_sentinel_q_axis"]]
    known_failure = float(screen["known_lower_failing_q"])
    unresolved = float(screen["unresolved_source_q"])
    phases = [float(value) for value in screen["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    sentinel_complete = _cartesian_complete(
        rows, ("q", "phase", "dt_ms"), (q_axis, phases, dts)
    )
    known_complete = _cartesian_complete(
        known_failure_rows,
        ("q", "phase", "dt_ms"),
        ([known_failure], phases, dts),
    )
    labels = {
        q: sorted({str(row["outcome"]) for row in rows if float(row["q"]) == q})
        for q in q_axis
    }
    safe = [
        q for q in q_axis
        if _all_nonempty(
            [row for row in rows if float(row["q"]) == q],
            lambda row: row["outcome"] == "LLL" and _no_failclosed_violation(row),
        )
    ]
    known_anchor_confirmed = bool(
        known_complete
        and _all_nonempty(
            known_failure_rows,
            lambda row: row["outcome"] == "physical_or_numerical_failure",
        )
    )
    anchor_ordered = bool(
        safe
        and known_failure < min(safe)
        and known_failure < unresolved < min(safe)
    )
    gates = {
        "sentinel_cartesian_product_complete": sentinel_complete,
        "known_failure_anchor_cartesian_product_complete": known_complete,
        "known_failure_anchor_confirmed_in_canonical_r0b": known_anchor_confirmed,
        "at_least_one_failclosed_safe_anchor": bool(safe),
        "unresolved_q_lies_strictly_between_confirmed_anchors": anchor_ordered,
    }
    return {
        "status": (
            "R0B_LOWER_RAMP_CONFIRMED_ANCHOR_BRACKET"
            if all(gates.values())
            else "R0B_LOWER_RAMP_ANCHOR_BRACKET_NOT_RESOLVED"
        ),
        "gates": gates,
        "q_axis": q_axis,
        "outcomes_by_q": {str(q): labels[q] for q in q_axis},
        "highest_confirmed_failing_q": known_failure if known_anchor_confirmed else None,
        "lowest_confirmed_safe_q": min(safe) if safe else None,
        "boundary_bracket_width": (
            min(safe) - known_failure if safe and known_anchor_confirmed else None
        ),
        "unresolved_source_q": unresolved,
        "unresolved_source_reason": (
            "base-dt Poincare closure 2.24-2.30e-5 exceeds the locked 2e-5 "
            "gate despite zero support/bound violations"
        ),
    }


def run(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    transfer = _load_transfer(cfg)
    parameters, low_parameters = _model(cfg)
    geometry = cfg["geometry"]
    reduction = canonical_m3b_core_annulus_bath(
        grid_n=int(geometry["grid_n"]), grid_L_mm=float(geometry["grid_L_mm"]),
        core_radius_mm=float(geometry["core_radius_mm"]),
        theta_rad=np.deg2rad(float(geometry["theta_deg"])),
    )
    prepared = prepare_patch_rhs(reduction.kernels, parameters)
    low, _ = _low_template(transfer, low_parameters)
    low_initial = _low_initial(low, float(cfg["model"]["z_interictal"]), reduction, parameters)
    inhibitory_baseline = np.asarray(low_initial[9:12], dtype=float)
    with np.load(ROOT / cfg["orbit_cycle_path"], allow_pickle=False) as payload:
        cycle = np.asarray(payload[f"{cfg['r0b']['cycle_trace_key']}_state"], dtype=float)

    q_axis = [float(value) for value in cfg["r0b"]["lower_boundary_sentinel_q_axis"]]
    phases = [float(value) for value in cfg["r0b"]["relative_phase_fractions"]]
    dts = [float(value) for value in cfg["integration"]["dt_ms"]]
    folds = _solve_folds(q_axis, cfg, prepared, parameters, transfer)
    ramp_arm = _ramp_parameters(cfg)
    rows: list[dict[str, Any]] = []

    for dt in dts:
        metadata = [(q, phase) for q in q_axis for phase in phases]
        source = integrate_frozen_patch_batch(
            np.asarray([
                _cycle_initial(low, cycle, phase, q, reduction, parameters)
                for q, phase in metadata
            ]),
            prepared, transfer, dt_ms=dt,
            duration_ms=float(cfg["integration"]["source_prelude_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
        )
        checkpoints = []
        for index, (q, phase) in enumerate(metadata):
            pattern = _pattern_summary(source, index, cfg, prepared, transfer)
            if pattern["outcome"] != "bounded_CCO":
                raise RuntimeError(f"boundary source is not bounded CCO: q={q}, phase={phase}, dt={dt}")
            checkpoint, _ = _checkpoint(
                source, index, int(cfg["r0b"]["source_min_returns_each_region"])
            )
            checkpoints.append(checkpoint)
        ramp = integrate_autonomous_latch_batch(
            np.asarray(checkpoints), prepared, transfer, [ramp_arm] * len(metadata), [],
            inhibitory_baseline_khz=inhibitory_baseline,
            dt_ms=dt, duration_ms=float(cfg["integration"]["ramp_post_ms"]),
            save_dt_ms=float(cfg["integration"]["save_dt_ms"]),
            section_level_khz=float(cfg["integration"]["section_level_rE_fast_khz"]),
            rearm_level_khz=float(cfg["integration"]["rearm_level_rE_fast_khz"]),
            max_trace_bytes=int(cfg["resource_contract"]["max_trace_bytes_per_batch"]),
            initial_latch_state=np.tile(np.asarray([[True, True, False]]), (len(metadata), 1)),
        )
        view = _frozen_view(ramp)
        for index, (q, phase) in enumerate(metadata):
            additive = float(cfg["model"]["additive_max_mv"]) * np.asarray(ramp["m"][:, index, 0], dtype=float)
            rows.append({
                "q": q, "phase": phase, "dt_ms": dt,
                **_pattern_summary(view, index, cfg, prepared, transfer),
                "low_fold_additive_mv": float(folds[q]["additive_mv"]),
                "max_additive_mv": float(np.max(additive)),
                "first_support_failure_ms": None if not np.isfinite(ramp["first_support_failure_ms"][index]) else float(ramp["first_support_failure_ms"][index]),
                "first_nonfinite_ms": None if not np.isfinite(ramp["first_nonfinite_ms"][index]) else float(ramp["first_nonfinite_ms"][index]),
            })

    output = ROOT / cfg["result_root"]
    canonical_rows = _load_csv(output / "r0b_smooth_ramp.csv")
    known_failure = float(cfg["r0b"]["known_lower_failing_q"])
    known_failure_rows = [
        row for row in canonical_rows if float(row["q"]) == known_failure
    ]
    summary = _summarize_boundary_sentinel(rows, known_failure_rows, cfg)
    _save_csv(output / "r0b_lower_boundary_sentinel.csv", rows)
    summary.update({
        "input_sha256": hashes,
        "claim_boundary": [
            "targeted dual-dt four-phase fixed-q ramp sentinel selected after R0b",
            "the q=.825 failing anchor comes from the canonical R0b stress arm",
            "q=.8275 is excluded as source-unresolved rather than relabeled safe or failed",
            "the anchors bracket an unresolved interval; they do not prove a monotone dynamical boundary",
            "this does not integrate reserve dynamics",
        ],
        "artifact": str((output / "r0b_lower_boundary_sentinel.csv").relative_to(ROOT)),
    })
    (output / "r0b_lower_boundary_sentinel.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def refresh_existing(config_path: Path) -> dict[str, Any]:
    """Rebuild the anchor summary from existing canonical and sentinel tables."""

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hashes = _validate_inputs(cfg)
    output = ROOT / cfg["result_root"]
    rows = _load_csv(output / "r0b_lower_boundary_sentinel.csv")
    canonical_rows = _load_csv(output / "r0b_smooth_ramp.csv")
    known_failure = float(cfg["r0b"]["known_lower_failing_q"])
    known_failure_rows = [
        row for row in canonical_rows if float(row["q"]) == known_failure
    ]
    summary = _summarize_boundary_sentinel(rows, known_failure_rows, cfg)
    summary.update({
        "input_sha256": hashes,
        "claim_boundary": [
            "targeted dual-dt four-phase fixed-q ramp sentinel selected after R0b",
            "the q=.825 failing anchor comes from the canonical R0b stress arm",
            "q=.8275 is excluded as source-unresolved rather than relabeled safe or failed",
            "the anchors bracket an unresolved interval; they do not prove a monotone dynamical boundary",
            "this does not integrate reserve dynamics",
        ],
        "artifact": str((output / "r0b_lower_boundary_sentinel.csv").relative_to(ROOT)),
    })
    (output / "r0b_lower_boundary_sentinel.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    summary = (
        refresh_existing(args.config.resolve())
        if args.refresh_existing
        else run(args.config.resolve())
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
