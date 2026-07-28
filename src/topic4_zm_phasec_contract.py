"""Immutable Phase-C experiment contract for the Z/M tonic-branch audit.

This module is deliberately simulator-free.  It reads the already accepted
Rev3.1 canonical configuration and anchor manifests, resolves every production
input by SHA256, and builds one deterministic contract.  It never edits a
canonical configuration, snapshot, or anchor.

The write-once helper uses an exclusive hard-link publication step.  Re-running
the locker is idempotent only when the complete normalized JSON object is
identical; any drift fails closed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from src.topic4_zm_noise_bank import PAIRED_REPLICATES, build_noise_bank


PHASEC_INPUT_VERSION = "zm_phasec_input_v1_2026-07-28"
PHASEC_CONTRACT_VERSION = "zm_phasec_contract_v1.3_2026-07-28"
PRIMARY_SEEDS = (1, 3, 4)
FAST_PHASES = ("rising", "peak")
NOISE_REPLICATES = tuple(PAIRED_REPLICATES)
C0_DURATION_MS = 8000.0
C0_BURN_IN_MS = 500.0
GAIN_MEASURE_MS = 1000.0
GAIN_BURN_IN_MS = 500.0
FINE_BIN_MS = 2.0
CURRENT_STRIDE_MS = 1.0

# These are diagnostic threshold offsets in membrane-potential units.  They do
# not modify the production V_th configuration or the recurrent substrate.
THRESHOLD_PERTURB_MV = (-0.10, -0.05, 0.0, 0.05, 0.10)

PRIMARY_STAGES = ("bounded_early", "bounded_mid", "bounded_late")
PRIMARY_CELL_NAMES = tuple(
    f"primary__{phase}__{name}"
    for phase in FAST_PHASES
    for name in (
        "bounded_early",
        "early_mid_midpoint",
        "bounded_mid",
        "mid_late_midpoint",
        "bounded_late",
    )
)
SHELL_DIRECTIONS = (
    "fullfield_mode2",
    "fullfield_mode3",
    "pathology_parallel",
    "pathology_perpendicular",
)
SHELL_CELL_NAMES = tuple(
    f"shell__bounded_mid__{direction}__{sign}0p25sd"
    for direction in SHELL_DIRECTIONS
    for sign in ("minus", "plus")
)

DEFAULT_UPSTREAM_ROOT = Path("results/topic4_sef_hfo/zm_branch_decision")
DEFAULT_OUTPUT = Path(
    "results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_manifest.json"
)
DEFAULT_INPUT_OUTPUT = Path(
    "results/topic4_sef_hfo/zm_phase_c_tonic_identity/"
    "phasec_input_manifest.json"
)
COORDINATE_MANIFEST_PATHS = {
    "dt": Path(
        "results/topic4_sef_hfo/zm_phase_c_tonic_identity/"
        "phasec1_coordinate_manifest_dt.json"
    ),
    "dt2": Path(
        "results/topic4_sef_hfo/zm_phase_c_tonic_identity/"
        "phasec1_coordinate_manifest_dt2.json"
    ),
}
PANEL_PATH = Path(
    "results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_panels.json"
)
SPEC_PATH = Path(
    "docs/superpowers/specs/"
    "2026-07-28-topic4-zm-phase-c-tonic-branch-identity-maturation-design.md"
)
PLAN_PATH = Path(
    "docs/superpowers/plans/"
    "2026-07-28-topic4-zm-phase-c-tonic-branch-identity-maturation.md"
)
PRODUCTION_PRODUCER_PATHS = (
    Path("scripts/lock_topic4_zm_phasec.py"),
    Path("scripts/lock_topic4_zm_phasec_panels.py"),
    Path("scripts/build_topic4_zm_phasec1_neighbourhood.py"),
    Path("scripts/run_topic4_zm_phasec_cell.py"),
    Path("scripts/run_topic4_zm_phasec0_parallel.py"),
    Path("scripts/run_topic4_zm_phasec1_parallel.py"),
    Path("scripts/run_topic4_zm_phasec_dt2_parallel.py"),
    Path("scripts/analyze_topic4_zm_phasec0.py"),
    Path("scripts/analyze_topic4_zm_phasec1.py"),
    Path("scripts/analyze_topic4_zm_phasec1_gain.py"),
    Path("scripts/analyze_topic4_zm_phasec_modal.py"),
    Path("scripts/adjudicate_topic4_zm_phasec.py"),
    Path("scripts/plot_topic4_zm_phasec.py"),
    Path("scripts/lock_topic4_zm_phasec1_dt2_confirmation.py"),
    Path("scripts/lock_topic4_zm_phasec1_gain_triggers.py"),
    Path("src/topic4_zm_phasec_contract.py"),
    Path("src/topic4_zm_phasec_metrics.py"),
    Path("src/topic4_zm_phasec_observation.py"),
    Path("src/topic4_zm_phasec_phenotype.py"),
    Path("src/topic4_zm_phasec_verdict.py"),
    Path("src/topic4_zm_phasec_neighbourhood.py"),
    Path("src/topic4_zm_phasec_modal.py"),
    Path("src/topic4_zm_phasec_plot.py"),
)


class ContractError(RuntimeError):
    """Base class for a fail-closed Phase-C contract error."""


class ImmutableManifestError(ContractError):
    """Raised when an existing write-once manifest differs from expectation."""


class ContractInputError(ContractError):
    """Raised when a required upstream input is absent or inconsistent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha(obj: Any) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError as exc:
        raise ContractInputError(f"required input is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractInputError(f"cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractInputError(f"required JSON must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractInputError(message)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ContractInputError(f"path escapes repository root: {path}") from exc


def _captured_lookup(anchor: Mapping[str, Any]) -> dict[tuple[str, str], dict]:
    rows = anchor.get("captured_states")
    _require(isinstance(rows, list), "anchor captured_states must be a list")
    out: dict[tuple[str, str], dict] = {}
    for row in rows:
        _require(isinstance(row, dict), "captured state row must be an object")
        key = (row.get("bin_name"), row.get("fast_phase"))
        _require(None not in key, "captured state lacks bin_name/fast_phase")
        _require(key not in out, f"duplicate captured state {key}")
        out[key] = row
    return out


def _state_ref(
    root: Path,
    row: Mapping[str, Any],
    *,
    expected_bin: str,
    expected_phase: str,
) -> dict:
    _require(row.get("bin_name") == expected_bin, "captured-state bin drift")
    _require(row.get("fast_phase") == expected_phase, "captured-state phase drift")
    rel = row.get("path")
    _require(isinstance(rel, str) and rel, "captured state lacks path")
    path = root / rel
    _require(path.is_file(), f"captured state is missing: {rel}")
    state_hash = row.get("state_hash")
    _require(
        isinstance(state_hash, str) and len(state_hash) == 64,
        f"captured state lacks semantic state_hash: {rel}",
    )
    t_step = row.get("t_step")
    _require(isinstance(t_step, int) and t_step >= 0, f"invalid t_step: {rel}")
    return {
        "bin_name": expected_bin,
        "fast_phase": expected_phase,
        "path": rel,
        "file_sha256": sha256_file(path),
        "state_hash": state_hash,
        "t_step": t_step,
        "t_ms": float(row["t_ms"]),
    }


def _noise_refs(config_sha: str, seed: int, state: Mapping[str, Any]) -> list[dict]:
    rows = []
    for replicate in NOISE_REPLICATES:
        bank = build_noise_bank(config_sha, seed, int(state["t_step"]), replicate)
        rows.append({
            "replicate": replicate,
            "bank_sha": bank["bank_sha"],
            "start_step": int(bank["start_step"]),
            "is_paired": bool(bank["is_paired"]),
            "ext_mean_only": bool(bank["ext_mean_only"]),
        })
    return rows


def _primary_cells(state_refs: Mapping[tuple[str, str], dict]) -> list[dict]:
    cells = []
    for phase in FAST_PHASES:
        early = state_refs[("bounded_early", phase)]
        mid = state_refs[("bounded_mid", phase)]
        late = state_refs[("bounded_late", phase)]
        rows = (
            ("bounded_early", early, early, (1.0, 0.0)),
            ("early_mid_midpoint", early, mid, (0.5, 0.5)),
            ("bounded_mid", mid, mid, (1.0, 0.0)),
            ("mid_late_midpoint", mid, late, (0.5, 0.5)),
            ("bounded_late", late, late, (1.0, 0.0)),
        )
        for label, left, right, weights in rows:
            cells.append({
                "cell_name": f"primary__{phase}__{label}",
                "tier": "primary_convex",
                "phase_trajectory": phase,
                "left_state_hash": left["state_hash"],
                "right_state_hash": right["state_hash"],
                "weights": list(weights),
                "clipping_allowed": False,
            })
    _require(
        tuple(row["cell_name"] for row in cells) == PRIMARY_CELL_NAMES,
        "internal primary-cell naming drift",
    )
    return cells


def _shell_cells() -> list[dict]:
    rows = []
    for direction in SHELL_DIRECTIONS:
        for sign, signed_step in (("minus", -0.25), ("plus", 0.25)):
            rows.append({
                "cell_name": (
                    f"shell__bounded_mid__{direction}__{sign}0p25sd"
                ),
                "tier": "secondary_shell",
                "basis_direction": direction,
                "step_robust_sd": signed_step,
                "clipping_allowed": False,
                "claim_ceiling": "nearby_extrapolated_candidate",
            })
    _require(
        tuple(row["cell_name"] for row in rows) == SHELL_CELL_NAMES,
        "internal shell-cell naming drift",
    )
    return rows


def _thresholds() -> dict:
    return {
        "confidence_level": 0.95,
        "hierarchical_bootstrap_draws": 5000,
        "time_block_ms": 500.0,
        "sliding_rate_window_ms": 250.0,
        "sliding_rate_window_stride_ms": 50.0,
        "active_area_window_ms": 25.0,
        "active_neuron_floor_hz": 5.0,
        "spatial_grid_n": 16,
        "local_active_floor_hz": 5.0,
        "ceiling_ratio": 0.80,
        "saturation_rho80_lcb": 0.50,
        "ai_rho80_ucb": 0.20,
        "saturation_gain_rel_ucb": 0.20,
        "ai_gain_rel_lcb": 0.50,
        "refractory_lock_lcb": 0.80,
        "ai_cv2_lcb": 0.70,
        "pairwise_abs_median_max": 0.10,
        "pairwise_shift_null_quantile": 0.975,
        "pairwise_shift_null_draws": 100,
        "pairwise_bin_ms": 5.0,
        "pairwise_shift_null_strata": [
            "core_core",
            "core_surround",
            "surround_surround",
        ],
        "pairwise_matched_stratum_max_excess": 0.0,
        "gain_linearity_relative_spread_max": 0.25,
        "whole_sheet_plateau_area_frac": 0.50,
        "phase_cell_min_passes": 2,
        "cell_total_min_passes": 5,
        "cell_total_replicates": 6,
        "cell_posterior_median_min": 0.80,
        "periodic_min_cycles": 10,
        "clonic_min_bursts": 6,
        "fine_rate_modulation_min": 0.20,
        "carrier_occupancy_min": 0.80,
        "period_cv_max": 0.20,
        "burst_interval_cv_max": 0.50,
        "c1_refractory_saturation_rho_min": 0.50,
        "c1_refractory_isi_fraction_min": 0.80,
        "c1_two_zone_occupancy_min": 0.80,
        "c1_periodic_cross_phase_period_rel_diff_max": 0.20,
        "c1_periodic_source_phase_bins": 16,
        "c1_periodic_source_phase_corr_min": 0.80,
        "c1_periodic_source_phase_alignment": (
            "maximum_circular_phase_shift"
        ),
        "c1_periodic_rest_reset_fraction_max": 0.20,
        "c1_rest_dwell_ms": 100.0,
        "c1_active_relative_to_p95": 0.15,
        "c1_rest_relative_to_p95": 0.10,
        "c1_rest_occupancy_max": 0.10,
        "c1_runaway_rate_hz": 250.0,
        "c1_hfo_train_min_events": 4,
        "c1_stationarity_abs_relative_drift_max": 0.50,
        "c1_stationarity_variance_ratio_max": 4.0,
        "c1_clonic_period_min_ms": 150.0,
        "c1_clonic_period_max_ms": 2000.0,
        "c1_periodic_frequency_max_hz": 150.0,
        "c1_periodic_period_max_ms": 200.0,
        "c1_primary_adjacent_cells_min": 2,
        "c1_primary_cross_seed_support_min": 2,
        "c1_shell_same_cell_cross_seed_support_min": 2,
        "c1_third_seed_policy": (
            "concordant_or_probabilistically_indeterminate_not_saturation_runaway"
        ),
        "c1_relay_location_permutations": 999,
        "c1_relay_permutation_p_max": 0.025,
        "c1_relay_abs_rho_min": 0.50,
        "c1_relay_flash_fraction_max": 0.80,
        "c1_relay_axis_span_fraction_min": 0.25,
        "c1_relay_first_passage_bins_min": 2,
    }


def validate_panel_manifest(
    panels: Mapping[str, Any],
    *,
    config_shas: Mapping[str, str],
) -> None:
    """Validate fixed, activity-independent panel IDs and their self hashes."""
    _require(isinstance(panels, Mapping), "panel manifest must be an object")
    _require(
        panels.get("schema") == "zm_phasec_panels_v1_2026-07-28",
        "panel schema drift",
    )
    panel_sha = panels.get("manifest_sha256")
    _require(
        isinstance(panel_sha, str) and len(panel_sha) == 64,
        "panel manifest lacks manifest_sha256",
    )
    panel_payload = {k: v for k, v in panels.items() if k != "manifest_sha256"}
    _require(_object_sha(panel_payload) == panel_sha, "panel manifest self-hash mismatch")
    rows = panels.get("seeds")
    _require(isinstance(rows, Mapping), "panel manifest lacks seeds")
    _require(
        sorted(int(k) for k in rows) == list(PRIMARY_SEEDS),
        "panel seed coverage drift",
    )
    for seed in PRIMARY_SEEDS:
        key = str(seed)
        row = rows[key]
        _require(isinstance(row, Mapping), f"seed {seed} panel row must be an object")
        _require(row.get("seed") == seed, f"seed {seed} panel seed drift")
        _require(
            row.get("config_sha") == config_shas[key],
            f"seed {seed} panel/canonical config drift",
        )
        _require(row.get("NE") == 32000, f"seed {seed} panel NE drift")
        _require(
            row.get("activity_independent") is True,
            f"seed {seed} panel is not activity-independent",
        )
        _require(
            row.get("selection")
            == "sha256(config_sha|seed|panel_stratum|E_local_id)",
            f"seed {seed} panel selection drift",
        )
        analysis = row.get("analysis_panel_E_ids")
        pairwise = row.get("pairwise_panel_E_ids")
        _require(
            isinstance(analysis, list) and len(analysis) == 1024,
            f"seed {seed} analysis panel size drift",
        )
        _require(
            isinstance(pairwise, list) and len(pairwise) == 256,
            f"seed {seed} pairwise panel size drift",
        )
        for label, ids in (("analysis", analysis), ("pairwise", pairwise)):
            _require(
                all(isinstance(i, int) and 0 <= i < row["NE"] for i in ids),
                f"seed {seed} {label} panel has invalid E ID",
            )
            _require(
                len(set(ids)) == len(ids),
                f"seed {seed} {label} panel contains duplicate IDs",
            )
        _require(
            row.get("analysis_panel_n_core") == 512
            and row.get("analysis_panel_n_surround") == 512,
            f"seed {seed} analysis strata drift",
        )
        _require(
            row.get("pairwise_panel_n_core") == 128
            and row.get("pairwise_panel_n_surround") == 128,
            f"seed {seed} pairwise strata drift",
        )
        row_sha = row.get("panel_sha256")
        _require(
            isinstance(row_sha, str) and len(row_sha) == 64,
            f"seed {seed} panel lacks panel_sha256",
        )
        row_payload = {k: v for k, v in row.items() if k != "panel_sha256"}
        _require(_object_sha(row_payload) == row_sha, f"seed {seed} panel self-hash mismatch")


def _resources() -> dict:
    return {
        "mem_available_reserve_gb": 96.0,
        "worker_rss_safety_factor": 1.25,
        "logical_cpu_reserve": 8,
        "swap_growth_allowed_bytes": 0,
        "thread_env": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "worker_limit_formula": (
            "min(logical_cpus-8,"
            "floor((MemAvailable_GB-96)/(1.25*measured_worker_RSS_GB)))"
        ),
        "atomic_part_per_cell": True,
        "resume_policy": "missing_or_technical_invalid_only",
    }


def build_input_manifest(root: Path | str) -> dict:
    """Build the non-production bootstrap lock from accepted upstream inputs."""
    root = Path(root).resolve()
    upstream = root / DEFAULT_UPSTREAM_ROOT
    canonical_path = upstream / "phase0/canonical_config.json"
    canonical = _read_json(canonical_path)
    seeds_obj = canonical.get("seeds")
    _require(isinstance(seeds_obj, dict), "canonical config lacks seeds object")
    _require(
        tuple(sorted(int(k) for k in seeds_obj)) == PRIMARY_SEEDS,
        f"canonical seed set must be exactly {PRIMARY_SEEDS}",
    )
    config_shas = {
        str(seed): seeds_obj[str(seed)].get("config_sha") for seed in PRIMARY_SEEDS
    }
    panels_path = root / PANEL_PATH
    panels = _read_json(panels_path)
    validate_panel_manifest(panels, config_shas=config_shas)

    spec = root / SPEC_PATH
    plan = root / PLAN_PATH
    module_path = Path(__file__).resolve()
    script_path = root / "scripts/lock_topic4_zm_phasec.py"
    producer_paths = tuple(root / path for path in PRODUCTION_PRODUCER_PATHS)
    for path in (spec, plan, module_path, script_path, *producer_paths):
        _require(path.is_file(), f"contract provenance input is missing: {path}")

    per_seed = {}
    for seed in PRIMARY_SEEDS:
        key = str(seed)
        seed_cfg = seeds_obj.get(key)
        _require(isinstance(seed_cfg, dict), f"canonical config lacks seed {seed}")
        config_sha = seed_cfg.get("config_sha")
        _require(
            isinstance(config_sha, str) and len(config_sha) == 64,
            f"seed {seed} canonical config_sha is invalid",
        )
        anchor_path = upstream / f"anchors/seed{seed}/anchor.json"
        anchor = _read_json(anchor_path)
        _require(anchor.get("seed") == seed, f"anchor seed mismatch for seed {seed}")
        _require(
            anchor.get("config_sha") == config_sha,
            f"anchor/canonical config drift for seed {seed}",
        )
        lookup = _captured_lookup(anchor)

        refs: dict[tuple[str, str], dict] = {}
        for stage in PRIMARY_STAGES:
            for phase in FAST_PHASES:
                key_state = (stage, phase)
                _require(
                    key_state in lookup,
                    f"seed {seed} lacks required state {stage}__{phase}",
                )
                refs[key_state] = _state_ref(
                    root, lookup[key_state],
                    expected_bin=stage, expected_phase=phase,
                )
        pre_key = ("pre_entry", "natural")
        _require(pre_key in lookup, f"seed {seed} lacks pre_entry__natural")
        pre = _state_ref(
            root, lookup[pre_key],
            expected_bin="pre_entry", expected_phase="natural",
        )

        c0_states = {}
        for phase in FAST_PHASES:
            state = refs[("bounded_mid", phase)]
            c0_states[phase] = {
                "state": state,
                "noise_banks": _noise_refs(config_sha, seed, state),
            }
        pre_control = {
            "state": pre,
            "noise_banks": _noise_refs(config_sha, seed, pre),
        }

        native_panel = dict(panels["seeds"][str(seed)])
        seed_payload = {
            "seed": seed,
            "canonical_config_sha": config_sha,
            "canonical_seed_object_sha256": _object_sha(seed_cfg),
            "I_th_EI_mV": float(seed_cfg["config"]["I_th_EI"]),
            "readout_kernel_width_mm": float(
                seed_cfg["config"]["params"]["Rr"]
            ),
            "readout_kernel_width_source": (
                "canonical_config.seeds[seed].config.params.Rr"
            ),
            "anchor_path": _relative(anchor_path, root),
            "anchor_file_sha256": sha256_file(anchor_path),
            "anchor_manifest_git_sha": anchor.get("git_sha"),
            "c0_carrier_states": c0_states,
            "c0_pre_entry_gain_control": pre_control,
            "c1_source_states": [
                refs[(stage, phase)]
                for phase in FAST_PHASES
                for stage in PRIMARY_STAGES
            ],
            "c1_primary_cells": _primary_cells(refs),
            "c1_secondary_shell_cells": _shell_cells(),
            # Embed the exact self-hashed panel row.  dt/2 reuses these
            # activity-independent anatomy IDs and never re-selects them with
            # the dt/2 configuration SHA.
            "fixed_panels": native_panel,
            "panel_selection_config_sha": config_sha,
        }
        if seed in (1, 3):
            dt2_config_path = upstream / f"phase0/dt2/seed{seed}_config.json"
            dt2_config = _read_json(dt2_config_path)
            _require(
                dt2_config.get("resolution") == "dt2"
                and dt2_config.get("seed") == seed,
                f"seed {seed} dt2 config identity drift",
            )
            _require(
                dt2_config.get("parent_config_sha") == config_sha,
                f"seed {seed} dt2 parent/native config drift",
            )
            dt2_config_sha = dt2_config.get("config_sha")
            _require(
                isinstance(dt2_config_sha, str) and len(dt2_config_sha) == 64,
                f"seed {seed} dt2 config_sha is invalid",
            )
            dt2_anchor_path = upstream / f"anchors_dt2/seed{seed}/anchor.json"
            dt2_anchor = _read_json(dt2_anchor_path)
            _require(
                dt2_anchor.get("resolution") == "dt2"
                and dt2_anchor.get("seed") == seed
                and dt2_anchor.get("config_sha") == dt2_config_sha,
                f"seed {seed} dt2 anchor/config drift",
            )
            dt2_lookup = _captured_lookup(dt2_anchor)
            dt2_refs: dict[tuple[str, str], dict] = {}
            for stage in PRIMARY_STAGES:
                for phase in FAST_PHASES:
                    state_key = (stage, phase)
                    _require(
                        state_key in dt2_lookup,
                        f"seed {seed} dt2 lacks {stage}__{phase}",
                    )
                    dt2_refs[state_key] = _state_ref(
                        root,
                        dt2_lookup[state_key],
                        expected_bin=stage,
                        expected_phase=phase,
                    )
            _require(
                ("pre_entry", "natural") in dt2_lookup,
                f"seed {seed} dt2 lacks pre_entry__natural",
            )
            dt2_pre = _state_ref(
                root,
                dt2_lookup[("pre_entry", "natural")],
                expected_bin="pre_entry",
                expected_phase="natural",
            )
            dt2_carrier = {}
            for phase in FAST_PHASES:
                dt2_state = dt2_refs[("bounded_mid", phase)]
                dt2_carrier[phase] = {
                    "state": dt2_state,
                    "noise_banks": _noise_refs(
                        dt2_config_sha, seed, dt2_state
                    ),
                }
            seed_payload["resolution_confirmations"] = {
                "dt2": {
                    "resolution": "dt2",
                    "dt_ms": float(dt2_config["dt"]),
                    "config_path": _relative(dt2_config_path, root),
                    "config_file_sha256": sha256_file(dt2_config_path),
                    "config_sha": dt2_config_sha,
                    "parent_config_sha": config_sha,
                    "anchor_path": _relative(dt2_anchor_path, root),
                    "anchor_file_sha256": sha256_file(dt2_anchor_path),
                    "c0_carrier_states": dt2_carrier,
                    "c0_pre_entry_gain_control": {
                        "state": dt2_pre,
                        "noise_banks": _noise_refs(
                            dt2_config_sha, seed, dt2_pre
                        ),
                    },
                    "c1_source_states": [
                        dt2_refs[(stage, phase)]
                        for phase in FAST_PHASES
                        for stage in PRIMARY_STAGES
                    ],
                    "fixed_panels": native_panel,
                    "panel_selection_config_sha": config_sha,
                    "panel_selection_resolution": "parent_native_dt",
                }
            }
        else:
            seed_payload["resolution_confirmations"] = {}
        per_seed[str(seed)] = seed_payload

    payload = {
        "schema": PHASEC_INPUT_VERSION,
        "production_authorized": False,
        "design_amendments": {
            "gain_probe": {
                "supersedes": (
                    "design draft section 4.2 external-drive percentage probe"
                ),
                "locked_implementation": (
                    "paired E-threshold diagnostic offsets "
                    "[-0.10,-0.05,0,+0.05,+0.10] mV"
                ),
                "reason": (
                    "the engine exposes an off-by-default threshold diagnostic; "
                    "the draft external-drive percentage probe is not the "
                    "authorized production implementation"
                ),
            },
            "z_physical_bound": {
                "locked_implementation": "0 <= z_i <= 1",
                "source": (
                    "src/snn_engine/mz_slow_vars.py and "
                    "src/snn_engine/slow_field.py Z update"
                ),
                "qI_q_min_not_applicable": True,
            },
        },
        "claim_boundary": {
            "phase_c0_c1_only": True,
            "observation_matched_ictal": False,
            "entry_tested": False,
            "offset_tested": False,
            "recovery_lifecycle_established": False,
            "phase_c2_authorized": False,
            "actuator_authorized": False,
        },
        "provenance": {
            "spec_path": str(SPEC_PATH),
            "spec_sha256": sha256_file(spec),
            "plan_path": str(PLAN_PATH),
            "plan_sha256": sha256_file(plan),
            "contract_module_path": _relative(module_path, root),
            "contract_module_sha256": sha256_file(module_path),
            "locker_script_path": _relative(script_path, root),
            "locker_script_sha256": sha256_file(script_path),
            "canonical_config_path": _relative(canonical_path, root),
            "canonical_config_file_sha256": sha256_file(canonical_path),
            "panel_manifest_path": _relative(panels_path, root),
            "panel_manifest_file_sha256": sha256_file(panels_path),
            "panel_manifest_sha256": panels["manifest_sha256"],
            "panel_manifest_schema": panels["schema"],
            "producer_file_sha256": {
                _relative(path, root): sha256_file(path) for path in producer_paths
            },
        },
        "c0": {
            "seeds": list(PRIMARY_SEEDS),
            "fast_phases": list(FAST_PHASES),
            "noise_replicates": list(NOISE_REPLICATES),
            "duration_ms": C0_DURATION_MS,
            "burn_in_ms": C0_BURN_IN_MS,
            "protocols": {
                "identity": {
                    "burn_in_ms": C0_BURN_IN_MS,
                    "measure_ms": C0_DURATION_MS,
                    "fine_bin_ms": FINE_BIN_MS,
                    "current_stride_ms": CURRENT_STRIDE_MS,
                },
                "gain": {
                    "burn_in_ms": GAIN_BURN_IN_MS,
                    "measure_ms": GAIN_MEASURE_MS,
                    "threshold_delta_abs_mV": [0.05, 0.10],
                },
            },
            "freeze_policy": {
                "freeze_z": True,
                "freeze_m": True,
                "freeze_sg_family": True,
                "membrane_effects_active": True,
            },
            "threshold_perturbation": {
                "target": "E_V_th_diagnostic_offset",
                "units": "mV",
                "values": list(THRESHOLD_PERTURB_MV),
                "changes_production_config": False,
                "draws_new_rng": False,
            },
            "required_identity_continuations": (
                len(PRIMARY_SEEDS) * len(FAST_PHASES) * len(NOISE_REPLICATES)
            ),
        },
        "c1": {
            "primary_tier": "primary_convex",
            "primary_cell_names": list(PRIMARY_CELL_NAMES),
            "primary_cell_count_per_seed": len(PRIMARY_CELL_NAMES),
            "secondary_tier": "secondary_shell",
            "secondary_shell_step_robust_sd": 0.25,
            "secondary_shell_cell_names": list(SHELL_CELL_NAMES),
            "secondary_shell_cell_count_per_seed": len(SHELL_CELL_NAMES),
            "secondary_claim_ceiling": "nearby_extrapolated_candidate",
            "clipping_allowed": False,
            "physical_bounds": {
                "z_min": 0.0,
                "z_min_source": "Z equation clip; not qI q_min",
                "z_max": 1.0,
                "m_min": 0.0,
                "S_G_min": 0.0,
                "S_G_max_source": "canonical slow_field.S_max",
                "empirical_envelope_iqr_pad": 0.25,
            },
        },
        "thresholds": _thresholds(),
        "resources": _resources(),
        "per_seed": per_seed,
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = _object_sha(payload)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate either the bootstrap input lock or final production contract."""
    if not isinstance(manifest, Mapping):
        raise ContractInputError("manifest must be an object")
    required = {
        "schema", "design_amendments", "claim_boundary", "provenance", "c0",
        "c1", "thresholds", "resources", "per_seed", "manifest_sha256",
        "production_authorized",
    }
    missing = required - set(manifest)
    _require(not missing, f"manifest lacks required fields: {sorted(missing)}")
    _require(
        manifest["schema"] in {PHASEC_INPUT_VERSION, PHASEC_CONTRACT_VERSION},
        "Phase-C contract schema drift",
    )
    if manifest["schema"] == PHASEC_INPUT_VERSION:
        _require(
            manifest["production_authorized"] is False,
            "input manifest cannot authorize production",
        )
    else:
        _require(
            manifest["production_authorized"] is True,
            "final manifest must authorize production",
        )
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    expected_sha = _object_sha(payload)
    _require(
        manifest["manifest_sha256"] == expected_sha,
        "manifest self-hash mismatch",
    )
    c0 = manifest["c0"]
    _require(c0["seeds"] == list(PRIMARY_SEEDS), "C0 seed drift")
    _require(c0["fast_phases"] == list(FAST_PHASES), "C0 phase drift")
    _require(
        c0["noise_replicates"] == list(NOISE_REPLICATES),
        "C0 noise-replicate drift",
    )
    _require(c0["duration_ms"] == C0_DURATION_MS, "C0 duration drift")
    _require(c0["burn_in_ms"] == C0_BURN_IN_MS, "C0 burn-in drift")
    _require(
        c0["protocols"]["identity"] == {
            "burn_in_ms": C0_BURN_IN_MS,
            "measure_ms": C0_DURATION_MS,
            "fine_bin_ms": FINE_BIN_MS,
            "current_stride_ms": CURRENT_STRIDE_MS,
        },
        "C0 identity protocol drift",
    )
    _require(
        c0["protocols"]["gain"] == {
            "burn_in_ms": GAIN_BURN_IN_MS,
            "measure_ms": GAIN_MEASURE_MS,
            "threshold_delta_abs_mV": [0.05, 0.10],
        },
        "C0 gain protocol drift",
    )
    _require(
        c0["threshold_perturbation"]["values"]
        == list(THRESHOLD_PERTURB_MV),
        "threshold-perturbation drift",
    )
    c1 = manifest["c1"]
    _require(
        c1["primary_cell_names"] == list(PRIMARY_CELL_NAMES),
        "C1 primary naming drift",
    )
    _require(
        c1["secondary_shell_cell_names"] == list(SHELL_CELL_NAMES),
        "C1 shell naming drift",
    )
    _require(
        c1["secondary_shell_step_robust_sd"] == 0.25,
        "C1 shell extent drift",
    )
    _require(c1["physical_bounds"]["z_min"] == 0.0, "Z physical-bound drift")
    _require(
        c1["physical_bounds"]["z_min_source"] == "Z equation clip; not qI q_min",
        "Z physical-bound source drift",
    )
    _require(
        sorted(int(k) for k in manifest["per_seed"]) == list(PRIMARY_SEEDS),
        "per-seed manifest coverage drift",
    )
    _require(
        manifest["thresholds"] == _thresholds(),
        "Phase-C decisive threshold manifest drift",
    )
    for seed in PRIMARY_SEEDS:
        row = manifest["per_seed"][str(seed)]
        _require(
            np.isfinite(float(row.get("readout_kernel_width_mm", np.nan)))
            and float(row["readout_kernel_width_mm"]) > 0.0,
            f"seed {seed} readout-kernel width missing",
        )
        _require(
            sorted(row["c0_carrier_states"]) == sorted(FAST_PHASES),
            f"seed {seed} C0 phase coverage drift",
        )
        for phase in FAST_PHASES:
            banks = row["c0_carrier_states"][phase]["noise_banks"]
            _require(
                [b["replicate"] for b in banks] == list(NOISE_REPLICATES),
                f"seed {seed}/{phase} noise coverage drift",
            )
            _require(
                all(b["is_paired"] for b in banks),
                f"seed {seed}/{phase} contains non-paired noise",
            )
        _require(
            [c["cell_name"] for c in row["c1_primary_cells"]]
            == list(PRIMARY_CELL_NAMES),
            f"seed {seed} primary-cell drift",
        )
        _require(
            [c["cell_name"] for c in row["c1_secondary_shell_cells"]]
            == list(SHELL_CELL_NAMES),
            f"seed {seed} shell-cell drift",
        )
        panels = row.get("fixed_panels")
        _require(isinstance(panels, Mapping), f"seed {seed} fixed panels missing")
        _require(
            len(panels.get("analysis_panel_E_ids", [])) == 1024,
            f"seed {seed} fixed analysis panel drift",
        )
        _require(
            len(panels.get("pairwise_panel_E_ids", [])) == 256,
            f"seed {seed} fixed pairwise panel drift",
        )
        _require(
            row.get("panel_selection_config_sha")
            == row.get("canonical_config_sha"),
            f"seed {seed} native panel-selection config drift",
        )
        if seed in (1, 3):
            dt2 = row.get("resolution_confirmations", {}).get("dt2")
            _require(
                isinstance(dt2, Mapping),
                f"seed {seed} dt2 confirmation lock missing",
            )
            _require(
                dt2.get("parent_config_sha") == row["canonical_config_sha"],
                f"seed {seed} dt2 parent/native config drift",
            )
            _require(
                dt2.get("panel_selection_config_sha")
                == row["canonical_config_sha"]
                and dt2.get("fixed_panels") == panels,
                f"seed {seed} dt2 panel reuse drift",
            )
            _require(
                len(dt2.get("c1_source_states", [])) == 6,
                f"seed {seed} dt2 C1 anchor coverage drift",
            )
        else:
            _require(
                row.get("resolution_confirmations") == {},
                f"seed {seed} unauthorized dt2 confirmation",
            )

    if manifest["schema"] == PHASEC_CONTRACT_VERSION:
        provenance = manifest.get("provenance", {})
        for key in (
            "phasec_input_manifest_path",
            "phasec_input_manifest_file_sha256",
            "phasec_input_manifest_manifest_sha256",
        ):
            _require(
                isinstance(provenance.get(key), str)
                and bool(provenance[key]),
                f"final manifest lacks {key}",
            )
        coordinate_manifests = manifest["c1"].get("coordinate_manifests")
        _require(
            isinstance(coordinate_manifests, Mapping)
            and set(coordinate_manifests) == {"dt", "dt2"},
            "final manifest lacks dt/dt2 coordinate-manifest locks",
        )
        for resolution in ("dt", "dt2"):
            ref = coordinate_manifests[resolution]
            _require(
                all(
                    isinstance(ref.get(key), str) and bool(ref[key])
                    for key in (
                        "path",
                        "file_sha256",
                        "manifest_sha256",
                        "semantic_sha256",
                    )
                ),
                f"final {resolution} coordinate lock incomplete",
            )
        file_map = manifest["c1"].get(
            "coordinate_npz_file_sha256_by_seed_by_resolution"
        )
        semantic_map = manifest["c1"].get(
            "coordinate_npz_semantic_sha256_by_seed_by_resolution"
        )
        _require(
            isinstance(file_map, Mapping)
            and isinstance(semantic_map, Mapping)
            and set(file_map) == {"dt", "dt2"}
            and set(semantic_map) == {"dt", "dt2"},
            "final coordinate NPZ file/semantic maps missing",
        )
        _require(
            sorted(int(k) for k in file_map["dt"]) == list(PRIMARY_SEEDS)
            and sorted(int(k) for k in semantic_map["dt"])
            == list(PRIMARY_SEEDS)
            and sorted(int(k) for k in file_map["dt2"]) == [1, 3]
            and sorted(int(k) for k in semantic_map["dt2"]) == [1, 3],
            "final coordinate NPZ resolution/seed coverage drift",
        )


def require_production_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate and require the sole final v1.3 production authority."""
    validate_manifest(manifest)
    _require(
        manifest.get("schema") == PHASEC_CONTRACT_VERSION
        and manifest.get("production_authorized") is True,
        "production requires final Phase-C v1.3 manifest",
    )


def _semantic_npz_sha(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with np.load(path, allow_pickle=False) as data:
            for name in sorted(data.files):
                array = np.ascontiguousarray(np.asarray(data[name]))
                h.update(
                    f"{name}|{array.dtype.str}|{array.shape}|".encode("utf-8")
                )
                h.update(array.tobytes())
    except (OSError, ValueError) as exc:
        raise ContractInputError(f"cannot validate coordinate NPZ {path}: {exc}") from exc
    return h.hexdigest()


def validate_coordinate_manifest(
    coordinate: Mapping[str, Any],
    *,
    root: Path,
    input_manifest: Mapping[str, Any],
    input_path: Path,
) -> None:
    """Validate one resolution-local coordinate lock and all NPZ payloads."""
    _require(
        coordinate.get("schema")
        == "zm_phasec1_coordinate_manifest_v2_2026-07-28",
        "coordinate manifest schema drift",
    )
    claimed = coordinate.get("manifest_sha256")
    body = {k: v for k, v in coordinate.items() if k != "manifest_sha256"}
    _require(
        isinstance(claimed, str) and _object_sha(body) == claimed,
        "coordinate manifest self-hash mismatch",
    )
    semantic = coordinate.get("semantic_sha256")
    semantic_body = {
        k: v for k, v in coordinate.items()
        if k not in {"manifest_sha256", "semantic_sha256"}
    }
    _require(
        isinstance(semantic, str) and _object_sha(semantic_body) == semantic,
        "coordinate manifest semantic hash mismatch",
    )
    _require(
        coordinate.get("parent_phasec_input_manifest_path")
        == _relative(input_path, root)
        and coordinate.get("parent_phasec_input_manifest_file_sha256")
        == sha256_file(input_path)
        and coordinate.get("parent_phasec_input_manifest_sha256")
        == input_manifest["manifest_sha256"],
        "coordinate/input-manifest provenance mismatch",
    )
    resolution = coordinate.get("resolution")
    _require(resolution in {"dt", "dt2"}, "coordinate resolution drift")
    expected_seeds = PRIMARY_SEEDS if resolution == "dt" else (1, 3)
    seeds = coordinate.get("seeds")
    _require(
        isinstance(seeds, Mapping)
        and sorted(int(k) for k in seeds) == list(expected_seeds),
        f"{resolution} coordinate seed coverage drift",
    )
    for seed in expected_seeds:
        row = seeds[str(seed)]
        npz_path = root / row.get("npz_path", "")
        _require(npz_path.is_file(), f"coordinate NPZ missing: {npz_path}")
        _require(
            row.get("npz_file_sha256") == sha256_file(npz_path),
            f"{resolution}/seed{seed} coordinate NPZ file hash drift",
        )
        _require(
            row.get("npz_semantic_sha256") == _semantic_npz_sha(npz_path),
            f"{resolution}/seed{seed} coordinate NPZ semantic hash drift",
        )
        native = input_manifest["per_seed"][str(seed)]
        expected_config = (
            native["canonical_config_sha"]
            if resolution == "dt"
            else native["resolution_confirmations"]["dt2"]["config_sha"]
        )
        _require(
            row.get("config_sha") == expected_config,
            f"{resolution}/seed{seed} coordinate config drift",
        )
        _require(
            row.get("panel_selection_config_sha")
            == native["canonical_config_sha"],
            f"{resolution}/seed{seed} coordinate anatomy-panel drift",
        )


def build_final_manifest(
    root: Path | str,
    *,
    input_path: Path | str | None = None,
    coordinate_paths: Mapping[str, Path | str] | None = None,
) -> dict:
    """Build the sole production-authorized v1.3 lock.

    This is intentionally the second stage.  It first proves that the
    write-once input manifest still equals the current live producer/upstream
    hashes, then locks both independent coordinate resolutions in one forward
    edge.  There is no coordinate -> final-manifest reference and therefore no
    hash cycle.
    """
    root = Path(root).resolve()
    input_path = (
        root / DEFAULT_INPUT_OUTPUT if input_path is None else Path(input_path)
    ).resolve()
    input_manifest = _read_json(input_path)
    validate_manifest(input_manifest)
    _require(
        input_manifest["schema"] == PHASEC_INPUT_VERSION
        and input_manifest["production_authorized"] is False,
        "final lock requires a non-production Phase-C input manifest",
    )
    expected_input = build_input_manifest(root)
    assert_manifest_matches(input_manifest, expected_input)
    coordinate_paths = (
        COORDINATE_MANIFEST_PATHS
        if coordinate_paths is None
        else coordinate_paths
    )
    coordinate_refs = {}
    npz_file_by_resolution = {}
    npz_semantic_by_resolution = {}
    for resolution in ("dt", "dt2"):
        raw_path = coordinate_paths.get(resolution)
        _require(raw_path is not None, f"missing {resolution} coordinate path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        coordinate = _read_json(path)
        validate_coordinate_manifest(
            coordinate,
            root=root,
            input_manifest=input_manifest,
            input_path=input_path,
        )
        _require(
            coordinate["resolution"] == resolution,
            f"{resolution} coordinate path contains wrong resolution",
        )
        coordinate_refs[resolution] = {
            "path": _relative(path, root),
            "file_sha256": sha256_file(path),
            "manifest_sha256": coordinate["manifest_sha256"],
            "semantic_sha256": coordinate["semantic_sha256"],
        }
        npz_file_by_resolution[resolution] = {
            str(seed): row["npz_file_sha256"]
            for seed, row in coordinate["seeds"].items()
        }
        npz_semantic_by_resolution[resolution] = {
            str(seed): row["npz_semantic_sha256"]
            for seed, row in coordinate["seeds"].items()
        }
    payload = {
        k: json.loads(json.dumps(v))
        for k, v in input_manifest.items()
        if k != "manifest_sha256"
    }
    payload["schema"] = PHASEC_CONTRACT_VERSION
    payload["production_authorized"] = True
    payload["provenance"].update({
        "phasec_input_manifest_path": _relative(input_path, root),
        "phasec_input_manifest_file_sha256": sha256_file(input_path),
        "phasec_input_manifest_manifest_sha256": input_manifest["manifest_sha256"],
    })
    payload["c1"]["coordinate_manifests"] = coordinate_refs
    payload["c1"]["coordinate_npz_file_sha256_by_seed_by_resolution"] = (
        npz_file_by_resolution
    )
    payload["c1"]["coordinate_npz_semantic_sha256_by_seed_by_resolution"] = (
        npz_semantic_by_resolution
    )
    final = dict(payload)
    final["manifest_sha256"] = _object_sha(payload)
    validate_manifest(final)
    return final


# Backwards-compatible name is deliberately the non-production stage.  Callers
# that need production authority must use build_final_manifest explicitly.
def build_manifest(root: Path | str) -> dict:
    return build_input_manifest(root)


def assert_manifest_matches(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Fail closed when normalized manifest content differs at all."""
    validate_manifest(existing)
    validate_manifest(expected)
    if _canonical_bytes(existing) != _canonical_bytes(expected):
        raise ImmutableManifestError(
            "existing Phase-C manifest differs from the current locked inputs: "
            f"existing={existing.get('manifest_sha256')} "
            f"expected={expected.get('manifest_sha256')}"
        )


def write_manifest_once(path: Path | str, manifest: Mapping[str, Any]) -> str:
    """Publish ``manifest`` once; exact reuse is allowed, overwrite is not.

    Returns ``"created"`` or ``"reused"``.
    """
    path = Path(path)
    validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False)
        + "\n"
    ).encode("utf-8")

    if path.exists():
        assert_manifest_matches(_read_json(path), manifest)
        return "reused"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(pretty)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            assert_manifest_matches(_read_json(path), manifest)
            return "reused"
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return "created"
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
