#!/usr/bin/env python3
"""Run one rev13 dynamic-Node arm through the frozen rev12 readout pipeline.

The rev12 worker remains the producer of record for substrate reconstruction,
causal-family event boundaries and contact readout.  This module composes over
four narrow hooks: substrate capture, controller injection, NPZ augmentation
and JSON augmentation.  It intentionally does not copy the rev12 analysis
pipeline.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import run_topic4_rev12_node_worker as rev12  # noqa: E402
from scripts import freeze_topic4_rev13_node_zero_sum_recovery as rev13_freezer  # noqa: E402
from src import topic4_zm_ictal_transition as transition_module  # noqa: E402
from src.topic4_continuous_field import continuous_field_h_with_queries  # noqa: E402
from src.topic4_node_accessibility import (  # noqa: E402
    FieldGatedZeroSumNodeRecovery,
    build_spatial_shift_permutation,
)


SCIENTIFIC_ROLE = "development_only_model_internal_node_recovery_canary"
FORBIDDEN_RUNTIME_INPUT_TOKENS = (
    "patient", "prototype", "classifier", "heldout", "ictal_target",
)
FORBIDDEN_CONTROLLER_TOKENS = (
    "patient", "prototype", "label", "contact", "shaft", "mode_a", "mode_b",
)
OFF_MODES = frozenset({"off", "exact_off", "none"})
MANIFEST_STATUS = "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN"
SPATIAL_SHIFT_MODE = "spatial_shift"
EXPLICIT_RUNTIME_PATHS = (
    "scripts/run_topic4_rev13_node_zero_sum_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "scripts/run_topic4_rev10_sa_spectral_field_worker.py",
    "scripts/run_topic4_rev9l_forced_source_worker.py",
    "scripts/run_topic4_core_field_stage3_fit.py",
    "scripts/run_topic4_rev9_node_kick_canary.py",
    "src/topic4_node_accessibility.py",
    "src/topic4_node_dualmode.py",
    "src/topic4_continuous_field.py",
    "src/topic4_core_field.py",
    "src/topic4_core_field_rev9.py",
    "src/topic4_core_field_runner.py",
    "src/topic4_graph_edge_flow.py",
    "src/topic4_local_connectivity.py",
    "src/topic4_spatial_ou_drive.py",
    "src/topic4_zm_d4.py",
    "src/sef_hfo_subject_placement.py",
    "src/sef_hfo_observation.py",
    "src/sef_hfo_events.py",
    "src/sef_hfo_snn_adapter.py",
    "src/snn_engine/kick_probe.py",
    "src/snn_engine/checkpoint.py",
    "src/snn_engine/model.py",
    "src/snn_engine/params.py",
    "src/snn_engine/connectivity.py",
    "src/snn_engine/connectivity_rot.py",
    "src/snn_engine/lfp.py",
    "src/topic4_zm_ictal_transition.py",
    "scripts/freeze_topic4_rev13_node_zero_sum_recovery.py",
)
SUBSTRATE_INPUT_KEYS = frozenset({
    "frozen_substrate_manifest",
    "rev9_base_config",
    "stage_config",
    "contact_contract",
    "common_detector_audit",
    "node_anchor_config",
    "placement_gradient_field",
    "placement_rank_displacement",
    "placement_geometry_t_a",
    "placement_geometry_t_b",
})


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_json_exact(actual: Any, expected: Any, *, path: str = "$") -> None:
    """Fail at the first field that differs in a frozen JSON-compatible tree."""
    if type(actual) is not type(expected):
        raise RuntimeError(
            f"frozen candidate field type changed at {path}: "
            f"{type(actual).__name__} != {type(expected).__name__}"
        )
    if isinstance(expected, Mapping):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"frozen candidate fields changed at {path}: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        for key in expected:
            _assert_json_exact(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"frozen candidate length changed at {path}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _assert_json_exact(
                actual_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if actual != expected:
        raise RuntimeError(
            f"frozen candidate value changed at {path}: {actual!r} != {expected!r}"
        )


def _git_output(arguments: list[str], *, text: bool = True):
    return subprocess.check_output(arguments, cwd=ROOT, text=text)


def _explicit_runtime_provenance(
    config_path: Path, expected_commit: str
) -> dict[str, Any]:
    """Freeze every rev13 composition path that can change numerical output."""
    expected = _git_output(
        ["git", "rev-parse", str(expected_commit)], text=True
    ).strip()
    current = _git_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if current != expected:
        raise RuntimeError("rev13 worker is not at the expected commit")
    try:
        config_relative = str(config_path.resolve().relative_to(ROOT))
    except ValueError as exc:
        raise RuntimeError("rev13 config must live inside the frozen worktree") from exc
    paths = [*EXPLICIT_RUNTIME_PATHS, config_relative]
    if len(paths) != len(set(paths)):
        raise RuntimeError("rev13 explicit provenance paths are duplicated")

    records: dict[str, dict[str, Any]] = {}
    for relative in paths:
        absolute = ROOT / relative
        if not absolute.is_file():
            raise RuntimeError(f"rev13 frozen runtime path is missing: {relative}")
        dirty_text = _git_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            text=True,
        ).strip()
        try:
            committed = _git_output(
                ["git", "show", f"{expected}:{relative}"], text=False
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"rev13 runtime path is not frozen at expected commit: {relative}"
            ) from exc
        expected_sha256 = hashlib.sha256(committed).hexdigest()
        observed_sha256 = _sha256_file(absolute)
        records[relative] = {
            "observed_sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "matches_expected_commit": observed_sha256 == expected_sha256,
            "dirty": bool(dirty_text),
        }
    dirty = [path for path, record in records.items() if record["dirty"]]
    mismatched = [
        path for path, record in records.items()
        if not record["matches_expected_commit"]
    ]
    if dirty or mismatched:
        raise RuntimeError(
            "rev13 explicit runtime freeze failed: "
            f"dirty={dirty}, mismatched={mismatched}"
        )
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "all_explicit_paths_clean": True,
        "all_explicit_paths_match_expected_commit": True,
        "explicit_runtime_files": records,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _validate_scientific_role(role: str) -> None:
    if str(role) != SCIENTIFIC_ROLE:
        raise RuntimeError("rev13 scientific role changed")


def _contains_token(value: object, tokens: tuple[str, ...]) -> bool:
    lowered = str(value).lower()
    return any(token in lowered for token in tokens)


def _assert_no_patient_runtime_inputs(config: Mapping[str, Any]) -> None:
    """Reject files that could expose patient labels/prototypes to this worker."""
    for name, record in config.get("inputs", {}).items():
        path = record.get("path", "") if isinstance(record, Mapping) else ""
        if _contains_token(name, FORBIDDEN_RUNTIME_INPUT_TOKENS) or _contains_token(
            path, FORBIDDEN_RUNTIME_INPUT_TOKENS
        ):
            raise RuntimeError(
                f"rev13 worker forbids patient target input {name!r}: {path!r}"
            )


def _assert_controller_is_model_internal(spec: Mapping[str, Any]) -> None:
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if _contains_token(key, FORBIDDEN_CONTROLLER_TOKENS):
                    raise RuntimeError(
                        f"node_accessibility uses forbidden runtime key {key!r}"
                    )
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and _contains_token(
            value, FORBIDDEN_CONTROLLER_TOKENS
        ):
            raise RuntimeError(
                f"node_accessibility uses forbidden runtime value {value!r}"
            )

    visit(spec)


def _assert_candidate_is_patient_free(candidate: Mapping[str, Any]) -> None:
    """Reject a dedicated manifest that embeds patient-side runtime targets."""
    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if _contains_token(key, FORBIDDEN_CONTROLLER_TOKENS):
                    raise RuntimeError(
                        f"rev13 candidate uses forbidden patient field {key!r}"
                    )
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and _contains_token(
            value, FORBIDDEN_CONTROLLER_TOKENS
        ):
            raise RuntimeError(
                f"rev13 candidate uses forbidden patient value {value!r}"
            )

    visit(candidate)


def _normalized_controller_spec(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = candidate.get("node_accessibility")
    if raw is None:
        return {"mode": "exact_off", "enabled": False}
    if not isinstance(raw, Mapping):
        raise RuntimeError("rev13 candidate lacks node_accessibility manifest payload")
    raw = dict(raw)
    nested = raw.pop("controller", None)
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise RuntimeError("node_accessibility.controller must be a mapping")
        merged = dict(raw)
        merged.update(nested)
        raw = merged
    _assert_controller_is_model_internal(raw)
    mode = str(raw.get("mode", raw.get("kind", ""))).lower()
    enabled = bool(raw.get("enabled", mode not in OFF_MODES))
    if not enabled:
        mode = "exact_off"
    if mode in OFF_MODES:
        if enabled:
            raise RuntimeError("an off node_accessibility mode cannot be enabled")
        mode = "exact_off"
    elif mode not in FieldGatedZeroSumNodeRecovery.MODES | {SPATIAL_SHIFT_MODE}:
        raise RuntimeError("unknown rev13 node_accessibility mode")
    if mode != "exact_off":
        if raw.get("kind") != "field_gated_bounded_node_recovery":
            raise RuntimeError("rev13 node_accessibility kind changed")
        if float(raw.get("a_max_multiplier", np.nan)) != 2.0:
            raise RuntimeError("rev13 node_accessibility trace bound changed")
        if raw.get("draws_random_numbers_at_runtime") is not False:
            raise RuntimeError("rev13 controller cannot draw random numbers at runtime")
    raw["mode"] = mode
    raw["enabled"] = mode != "exact_off"
    return raw


def _quantile_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    if int(n_bins) < 1:
        raise ValueError("stratification bin count must be positive")
    values = np.asarray(values, dtype=np.float64)
    if int(n_bins) == 1 or np.all(values == values[0]):
        return np.zeros(values.size, dtype=np.int16)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    if edges.size <= 2:
        return np.zeros(values.size, dtype=np.int16)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int16)


def _stratified_permutation(
    support: np.ndarray,
    signed_depth: np.ndarray,
    shuffle: Mapping[str, Any],
) -> np.ndarray:
    """Construct the frozen activity-location null without touching network RNG."""
    explicit = shuffle.get("permutation")
    if explicit is not None:
        permutation = np.asarray(explicit, dtype=np.int64)
    else:
        depth_key = (
            "signed_depth_quantile_bins"
            if "signed_depth_quantile_bins" in shuffle
            else "static_quantile_bins"
        )
        required = {"seed", "support_quantile_bins", depth_key}
        missing = required.difference(shuffle)
        if missing:
            raise RuntimeError(
                "stratified shuffle is missing " + ", ".join(sorted(missing))
            )
        support_code = _quantile_codes(
            support, int(shuffle["support_quantile_bins"])
        )
        static_code = _quantile_codes(
            signed_depth, int(shuffle[depth_key])
        )
        joint = (
            support_code.astype(np.int64)
            * (int(np.max(static_code, initial=0)) + 1)
            + static_code
        )
        permutation = np.arange(support.size, dtype=np.int64)
        rng = np.random.default_rng(int(shuffle["seed"]))
        for stratum in np.unique(joint):
            indices = np.flatnonzero(joint == stratum)
            permutation[indices] = rng.permutation(indices)
    if permutation.shape != support.shape or not np.array_equal(
        np.sort(permutation), np.arange(support.size)
    ):
        raise RuntimeError("frozen shuffle is not a complete E-neuron permutation")
    expected = shuffle.get("permutation_sha256")
    actual = _sha256_array(permutation)
    if expected is not None and str(expected) != actual:
        raise RuntimeError("frozen stratified-shuffle permutation hash changed")
    return permutation


def _spatial_shift_permutation(
    positions_e: np.ndarray,
    *,
    support_g: np.ndarray,
    L: float,
    contract: Mapping[str, Any],
) -> np.ndarray:
    """Build the config-defined toroidal, one-to-one spatial matched mapping."""
    del L  # Sheet scale is implicit in equal-width blocks of the frozen positions.
    required = {
        "mapping_method",
        "block_shape",
        "shift_blocks",
        "within_block_order",
        "permutation_scope",
        "dynamic_scale_match",
    }
    missing = required.difference(contract)
    if missing:
        raise RuntimeError(
            "spatial_shift contract is missing " + ", ".join(sorted(missing))
        )
    if contract["mapping_method"] != (
        "rank_matched_equal_width_spatial_blocks_toroidal_shift"
    ):
        raise RuntimeError("spatial_shift mapping method changed")
    if contract["within_block_order"] != "y_then_x_then_original_index":
        raise RuntimeError("spatial_shift within-block ordering changed")
    if contract["permutation_scope"] != "all_E_neurons":
        raise RuntimeError("spatial_shift permutation scope changed")
    if contract["dynamic_scale_match"] != (
        "support_weighted_centered_SD_each_step_to_unshifted_zero_sum"
    ):
        raise RuntimeError("spatial_shift dynamic amplitude matching changed")
    permutation = build_spatial_shift_permutation(
        positions_e,
        support_g=support_g,
        block_shape=tuple(contract["block_shape"]),
        shift_blocks=tuple(contract["shift_blocks"]),
    )
    if permutation.shape != (np.asarray(positions_e).shape[0],) or not np.array_equal(
        np.sort(permutation), np.arange(permutation.size)
    ):
        raise RuntimeError("spatial_shift is not a complete E-neuron permutation")
    expected = contract.get("permutation_sha256")
    actual = _sha256_array(permutation)
    if expected is not None and str(expected) != actual:
        raise RuntimeError("frozen spatial_shift permutation hash changed")
    return permutation


def _dispersion_field_on_substrate(candidate: Mapping[str, Any], substrate) -> np.ndarray:
    field = candidate.get("node_dispersion_field")
    if not isinstance(field, Mapping):
        raise RuntimeError("rev13 requires the frozen dual-channel Node substrate")
    h_e, _, _ = continuous_field_h_with_queries(
        field["coefficients"], substrate.positions_e, substrate.positions_i,
        n_basis=int(field["n_basis"]), degree=int(field["degree"]),
        target_count=float(substrate.stage["N_core_manual"]),
        L=float(substrate.engine["L"]),
    )
    return np.asarray(h_e, dtype=np.float64)


def _controller_support(candidate: Mapping[str, Any], substrate) -> tuple[np.ndarray, dict]:
    h_mean = np.asarray(substrate.h_e, dtype=np.float64)
    h_dispersion = _dispersion_field_on_substrate(candidate, substrate)
    raw = h_mean + h_dispersion
    maximum = float(np.max(raw, initial=0.0))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise RuntimeError("rev13 Node support has no positive finite mass")
    support = raw / maximum
    audit = {
        "contract": "normalized_union_of_frozen_mean_and_dispersion_fields",
        "h_mean_sha256": _sha256_array(h_mean),
        "h_dispersion_sha256": _sha256_array(h_dispersion),
        "support_sha256": _sha256_array(support),
        "support_mass": float(np.sum(support)),
        "support_max": float(np.max(support)),
    }
    raw_controller = candidate.get("node_accessibility")
    expected = (
        raw_controller.get("support_sha256")
        if isinstance(raw_controller, Mapping) else None
    )
    if expected is not None and str(expected) != audit["support_sha256"]:
        raise RuntimeError("rev13 controller support hash changed")
    return support, audit


def _support_weighted_centered_sd(
    support: np.ndarray, modulation: np.ndarray
) -> float:
    """Support-weighted centered SD used to scale dynamic Node amplitude."""
    mass = float(np.sum(support))
    if not np.isfinite(mass) or mass <= 0.0:
        raise RuntimeError("Node support has no finite positive mass")
    center = float(np.dot(support, modulation) / mass)
    return float(np.sqrt(np.dot(support, (modulation - center) ** 2) / mass))


def _static_rms(support: np.ndarray, modulation: np.ndarray) -> float:
    """Backward-compatible name for the centered-SD amplitude contract."""
    return _support_weighted_centered_sd(support, modulation)


def _frozen_signed_depth(substrate) -> np.ndarray:
    from src.topic4_core_field import (  # local import keeps provenance explicit
        core_thresholds, sample_core_quantiles, signed_depth,
    )

    quantiles = sample_core_quantiles(
        int(substrate.n_e), int(substrate.stage["quantile_seed"])
    )
    return np.asarray(signed_depth(
        core_thresholds(
            quantiles,
            float(substrate.engine["core_mean"]),
            float(substrate.engine["core_std"]),
        ),
        float(substrate.engine["v_base"]),
    ), dtype=np.float64)


@dataclass
class _ThresholdRecorder:
    """Controller proxy that audits bounds and static-modulation sign changes."""

    controller: FieldGatedZeroSumNodeRecovery
    static_modulation: np.ndarray
    reset_mV: float
    static_centered_sd_mV: float | None = None
    minimum_margin_mV: float = 1.0
    time_ms: list[float] = field(default_factory=list)
    threshold_min_mV: list[float] = field(default_factory=list)
    threshold_max_mV: list[float] = field(default_factory=list)
    zero_sum_error_mV: list[float] = field(default_factory=list)
    saturation_fraction: list[float] = field(default_factory=list)
    static_sign_flip_fraction: list[float] = field(default_factory=list)
    legacy_unweighted_saturation_fraction: list[float] = field(default_factory=list)
    legacy_unweighted_static_sign_flip_fraction: list[float] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.static_modulation = np.asarray(self.static_modulation, dtype=np.float64)
        if self.static_modulation.shape != (self.controller.n_e,):
            raise ValueError("static modulation must align to the E population")
        expected_sd = _support_weighted_centered_sd(
            self.controller.support_g, self.static_modulation
        )
        if self.static_centered_sd_mV is None:
            self.static_centered_sd_mV = expected_sd
        elif not np.isclose(
            float(self.static_centered_sd_mV), expected_sd, rtol=0.0, atol=1e-12
        ):
            raise ValueError("static centered SD does not match the frozen substrate")
        self.static_centered_sd_mV = float(self.static_centered_sd_mV)
        self.primary_diagnostic_mask = (
            (self.controller.support_g >= 0.05)
            & (np.abs(self.static_modulation) >= 0.1 * self.static_centered_sd_mV)
        )
        self.primary_diagnostic_weight = np.where(
            self.primary_diagnostic_mask, self.controller.support_g, 0.0
        )
        self.primary_diagnostic_weight_sum = float(
            np.sum(self.primary_diagnostic_weight)
        )
        if self.primary_diagnostic_weight_sum <= 0.0:
            raise ValueError("primary Node diagnostic support is empty")

    def _support_weighted_fraction(self, indicator: np.ndarray) -> float:
        indicator = np.asarray(indicator, dtype=bool)
        return float(
            np.dot(self.primary_diagnostic_weight, indicator.astype(np.float64))
            / self.primary_diagnostic_weight_sum
        )

    @property
    def KIND(self):  # checkpoint machinery identifies controllers by KIND
        return self.controller.KIND

    def threshold(self, base):
        base_array = np.asarray(base, dtype=np.float64)
        effective = self.controller.threshold(base_array)
        effective_e = np.asarray(effective[: self.controller.n_e], dtype=np.float64)
        if float(np.min(effective_e)) <= self.reset_mV + self.minimum_margin_mV:
            raise FloatingPointError("rev13 effective threshold violated reset margin")
        if (
            self.controller.step_index == 0
            or self.controller.step_index % self.controller._trace_every == 0
        ):
            dynamic = effective_e - base_array[: self.controller.n_e]
            legacy_valid = np.abs(self.static_modulation) > 1e-12
            combined = self.static_modulation + dynamic
            flips = np.zeros(legacy_valid.shape, dtype=bool)
            flips[legacy_valid] = np.signbit(combined[legacy_valid]) != np.signbit(
                self.static_modulation[legacy_valid]
            )
            saturated = self.controller.state_mV >= self.controller.a_max_mV - 1e-12
            self.time_ms.append(self.controller.step_index * self.controller.dt_ms)
            self.threshold_min_mV.append(float(np.min(effective_e)))
            self.threshold_max_mV.append(float(np.max(effective_e)))
            self.zero_sum_error_mV.append(abs(float(np.sum(dynamic, dtype=np.float64))))
            self.saturation_fraction.append(self._support_weighted_fraction(saturated))
            self.static_sign_flip_fraction.append(
                self._support_weighted_fraction(flips)
            )
            self.legacy_unweighted_saturation_fraction.append(
                float(np.mean(saturated))
            )
            self.legacy_unweighted_static_sign_flip_fraction.append(
                float(np.mean(flips[legacy_valid])) if np.any(legacy_valid) else 0.0
            )
        return effective

    def step(self, spikes_e, dt_ms):
        self.controller.step(spikes_e, dt_ms)

    def checkpoint_state(self):
        return self.controller.checkpoint_state()

    def restore_checkpoint_state(self, payload):
        self.controller.restore_checkpoint_state(payload)

    def trace_arrays(self) -> dict[str, np.ndarray]:
        return {
            "time_ms": np.asarray(self.time_ms, dtype=np.float64),
            "threshold_min_mV": np.asarray(self.threshold_min_mV, dtype=np.float64),
            "threshold_max_mV": np.asarray(self.threshold_max_mV, dtype=np.float64),
            "zero_sum_error_mV": np.asarray(self.zero_sum_error_mV, dtype=np.float64),
            "saturation_fraction": np.asarray(
                self.saturation_fraction, dtype=np.float64
            ),
            "static_modulation_sign_flip_fraction": np.asarray(
                self.static_sign_flip_fraction, dtype=np.float64
            ),
            "legacy_unweighted_saturation_fraction": np.asarray(
                self.legacy_unweighted_saturation_fraction, dtype=np.float64
            ),
            "legacy_unweighted_static_modulation_sign_flip_fraction": np.asarray(
                self.legacy_unweighted_static_sign_flip_fraction, dtype=np.float64
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        arrays = self.trace_arrays()
        base = self.controller.diagnostics()
        base.update({
            "saturation_fraction": (
                float(arrays["saturation_fraction"][-1])
                if arrays["saturation_fraction"].size else None
            ),
            "static_modulation_sign_flip_fraction": (
                float(arrays["static_modulation_sign_flip_fraction"][-1])
                if arrays["static_modulation_sign_flip_fraction"].size else None
            ),
            "legacy_unweighted_saturation_fraction": (
                float(arrays["legacy_unweighted_saturation_fraction"][-1])
                if arrays["legacy_unweighted_saturation_fraction"].size else None
            ),
            "legacy_unweighted_static_modulation_sign_flip_fraction": (
                float(arrays[
                    "legacy_unweighted_static_modulation_sign_flip_fraction"
                ][-1])
                if arrays[
                    "legacy_unweighted_static_modulation_sign_flip_fraction"
                ].size else None
            ),
            "threshold_min_mV_observed": (
                float(np.min(arrays["threshold_min_mV"]))
                if arrays["threshold_min_mV"].size else None
            ),
            "threshold_max_mV_observed": (
                float(np.max(arrays["threshold_max_mV"]))
                if arrays["threshold_max_mV"].size else None
            ),
            "maximum_zero_sum_error_mV": (
                float(np.max(arrays["zero_sum_error_mV"]))
                if arrays["zero_sum_error_mV"].size else None
            ),
            "maximum_saturation_fraction": (
                float(np.max(arrays["saturation_fraction"]))
                if arrays["saturation_fraction"].size else None
            ),
            "maximum_support_weighted_saturation_fraction": (
                float(np.max(arrays["saturation_fraction"]))
                if arrays["saturation_fraction"].size else None
            ),
            "maximum_static_modulation_sign_flip_fraction": (
                float(np.max(arrays["static_modulation_sign_flip_fraction"]))
                if arrays["static_modulation_sign_flip_fraction"].size else None
            ),
            "maximum_support_weighted_static_modulation_sign_flip_fraction": (
                float(np.max(arrays["static_modulation_sign_flip_fraction"]))
                if arrays["static_modulation_sign_flip_fraction"].size else None
            ),
            "maximum_legacy_unweighted_saturation_fraction": (
                float(np.max(arrays["legacy_unweighted_saturation_fraction"]))
                if arrays["legacy_unweighted_saturation_fraction"].size else None
            ),
            "maximum_legacy_unweighted_static_modulation_sign_flip_fraction": (
                float(np.max(arrays[
                    "legacy_unweighted_static_modulation_sign_flip_fraction"
                ]))
                if arrays[
                    "legacy_unweighted_static_modulation_sign_flip_fraction"
                ].size else None
            ),
            "primary_diagnostic_support_g_minimum": 0.05,
            "primary_diagnostic_static_absolute_minimum_in_static_sd": 0.1,
            "primary_diagnostic_support_weight": self.primary_diagnostic_weight_sum,
            "primary_diagnostic_neuron_count": int(np.sum(self.primary_diagnostic_mask)),
            "static_amplitude_definition": "support_weighted_centered_sd",
            "static_centered_sd_mV": self.static_centered_sd_mV,
            "threshold_reset_margin_mV": self.minimum_margin_mV,
        })
        return base


@dataclass
class _RunState:
    candidate: dict[str, Any]
    controller_spec: dict[str, Any]
    substrate: Any = None
    support_audit: dict[str, Any] = field(default_factory=dict)
    controller: FieldGatedZeroSumNodeRecovery | None = None
    recorder: _ThresholdRecorder | None = None
    static_centered_sd_mV: float | None = None
    source_config_text: str | None = None
    compatibility_config: dict[str, Any] = field(default_factory=dict)
    compatibility_audit: dict[str, Any] = field(default_factory=dict)
    rev13_provenance: dict[str, Any] = field(default_factory=dict)
    engineering_run_kind: str | None = None
    requested_duration_ms: float | None = None
    frozen_duration_ms: float = 20000.0


def _build_controller(state: _RunState, substrate) -> _ThresholdRecorder | None:
    spec = state.controller_spec
    if not spec["enabled"]:
        return None
    support, support_audit = _controller_support(state.candidate, substrate)
    static_modulation = np.asarray(
        substrate.delta_vtheta[: substrate.n_e], dtype=np.float64
    )
    sigma_node = _support_weighted_centered_sd(support, static_modulation)
    if "a_ref_mV" in spec:
        a_ref_mV = float(spec["a_ref_mV"])
    elif "amplitude_fraction_of_static_rms" in spec:
        a_ref_mV = float(spec["amplitude_fraction_of_static_rms"]) * sigma_node
    elif "c" in spec:
        a_ref_mV = float(spec["c"]) * sigma_node
    else:
        raise RuntimeError("active rev13 controller lacks a_ref_mV or amplitude fraction")
    declared = spec.get("a_ref_mV_expected")
    if declared is not None and not np.isclose(
        a_ref_mV, float(declared), rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("rev13 a_ref_mV no longer matches its frozen declaration")

    mode = str(spec["mode"])
    permutation = None
    spatial_shift_permutation = None
    if mode == "stratified_shuffle":
        shuffle = spec.get("shuffle", spec.get("stratified_shuffle"))
        if not isinstance(shuffle, Mapping):
            raise RuntimeError("stratified_shuffle lacks its frozen shuffle contract")
        permutation = _stratified_permutation(
            support, _frozen_signed_depth(substrate), shuffle
        )
        support_audit["shuffle_permutation_sha256"] = _sha256_array(permutation)
    elif mode == SPATIAL_SHIFT_MODE:
        spatial_shift = spec.get("spatial_shift")
        if not isinstance(spatial_shift, Mapping):
            raise RuntimeError("spatial_shift lacks its frozen mapping contract")
        spatial_shift_permutation = _spatial_shift_permutation(
            np.asarray(substrate.positions_e, dtype=np.float64),
            support_g=support,
            L=float(substrate.engine["L"]),
            contract=spatial_shift,
        )
        support_audit.update({
            "spatial_shift_permutation_sha256": _sha256_array(
                spatial_shift_permutation
            ),
            "spatial_shift_mapping": copy.deepcopy(dict(spatial_shift)),
            "spatial_shift_is_complete_permutation": True,
        })
    controller = FieldGatedZeroSumNodeRecovery(
        support,
        dt_ms=float(substrate.engine["dt"]),
        tau_ms=float(spec["tau_ms"]),
        a_ref_mV=a_ref_mV,
        r_ref_hz=float(spec.get("r_ref_hz", spec.get("reference_rate_hz", 50.0))),
        mode=mode,
        shuffle_permutation=permutation,
        spatial_shift_permutation=spatial_shift_permutation,
        trace_dt_ms=float(spec.get("trace_dt_ms", 10.0)),
    )
    state.support_audit = support_audit
    state.static_centered_sd_mV = sigma_node
    state.controller = controller
    state.recorder = _ThresholdRecorder(
        controller=controller,
        static_modulation=static_modulation,
        reset_mV=float(substrate.params.V_reset),
        static_centered_sd_mV=sigma_node,
        minimum_margin_mV=float(spec.get("minimum_reset_margin_mV", 1.0)),
    )
    return state.recorder


def _npz_additions(state: _RunState) -> dict[str, np.ndarray]:
    mode = state.controller_spec["mode"]
    result: dict[str, np.ndarray] = {
        "node_accessibility_enabled": np.asarray(state.recorder is not None, bool),
        "node_accessibility_mode": np.asarray(mode),
    }
    if state.recorder is None:
        empty = np.asarray([], dtype=np.float64)
        result.update({
            "node_accessibility_support_g": empty.copy(),
            "node_accessibility_state_final_mV": empty.copy(),
            "node_accessibility_dynamic_delta_final_mV": empty.copy(),
            "node_accessibility_effective_threshold_final_mV": empty.copy(),
            "node_accessibility_static_modulation_sign_flip_final": np.asarray(
                [], dtype=bool
            ),
            "node_accessibility_primary_diagnostic_support_mask": np.asarray(
                [], dtype=bool
            ),
            "node_accessibility_primary_support_weighted_sign_flip_final": empty.copy(),
            "node_accessibility_legacy_unweighted_sign_flip_final": empty.copy(),
        })
        for key in (
            "time_ms", "state_mean_mV", "state_max_mV", "saturation_fraction",
            "delta_sum_mV", "delta_min_mV", "delta_max_mV",
            "lowered_support_mass_fraction",
        ):
            result[f"node_accessibility_trace_{key}"] = empty.copy()
        for key in (
            "time_ms", "threshold_min_mV", "threshold_max_mV",
            "zero_sum_error_mV", "saturation_fraction",
            "static_modulation_sign_flip_fraction",
            "legacy_unweighted_saturation_fraction",
            "legacy_unweighted_static_modulation_sign_flip_fraction",
        ):
            result[f"node_accessibility_threshold_{key}"] = empty.copy()
        return result
    controller = state.controller
    dynamic_final = np.asarray(controller.delta_theta(), dtype=np.float64)
    static = np.asarray(
        state.substrate.delta_vtheta[: state.substrate.n_e], dtype=np.float64
    )
    valid = np.abs(static) > 1e-12
    sign_flip = np.zeros(valid.shape, dtype=bool)
    sign_flip[valid] = np.signbit((static + dynamic_final)[valid]) != np.signbit(
        static[valid]
    )
    primary_mask = np.asarray(state.recorder.primary_diagnostic_mask, dtype=bool)
    primary_sign_flip = state.recorder._support_weighted_fraction(sign_flip)
    legacy_sign_flip = float(np.mean(sign_flip[valid])) if np.any(valid) else 0.0
    result.update({
        "node_accessibility_support_g": np.asarray(controller.support_g),
        "node_accessibility_state_final_mV": np.asarray(controller.state_mV),
        "node_accessibility_dynamic_delta_final_mV": dynamic_final,
        "node_accessibility_effective_threshold_final_mV": (
            np.asarray(state.substrate.vtheta[: state.substrate.n_e], dtype=np.float64)
            + dynamic_final
        ),
        "node_accessibility_static_modulation_sign_flip_final": sign_flip,
        "node_accessibility_primary_diagnostic_support_mask": primary_mask,
        "node_accessibility_primary_support_weighted_sign_flip_final": np.asarray(
            primary_sign_flip, dtype=np.float64
        ),
        "node_accessibility_legacy_unweighted_sign_flip_final": np.asarray(
            legacy_sign_flip, dtype=np.float64
        ),
    })
    for key, values in state.controller.trace_arrays().items():
        result[f"node_accessibility_trace_{key}"] = np.asarray(values)
    for key, values in state.recorder.trace_arrays().items():
        result[f"node_accessibility_threshold_{key}"] = np.asarray(values)
    return result


def _validate_dynamic_diagnostics(state: _RunState) -> None:
    if state.recorder is None:
        return
    arrays = state.recorder.trace_arrays()
    for key, values in arrays.items():
        if values.ndim != 1 or not np.isfinite(values).all():
            raise RuntimeError(f"rev13 controller trace is invalid: {key}")
    if not arrays["threshold_min_mV"].size:
        raise RuntimeError("rev13 controller threshold trace is empty")
    for key in (
        "saturation_fraction",
        "static_modulation_sign_flip_fraction",
        "legacy_unweighted_saturation_fraction",
        "legacy_unweighted_static_modulation_sign_flip_fraction",
    ):
        if np.any((arrays[key] < 0.0) | (arrays[key] > 1.0)):
            raise RuntimeError(f"rev13 controller fraction is invalid: {key}")
    if state.controller_spec["mode"] in {
        "zero_sum", "stratified_shuffle", SPATIAL_SHIFT_MODE,
    }:
        tolerance = float(state.controller_spec.get("zero_sum_atol_mV", 1e-9))
        if float(np.max(arrays["zero_sum_error_mV"])) > tolerance:
            raise RuntimeError("rev13 zero-sum threshold budget drifted")


def _augment_npz_arrays(arrays: Mapping[str, Any], state: _RunState) -> dict[str, Any]:
    _validate_dynamic_diagnostics(state)
    output = dict(arrays)
    additions = _npz_additions(state)
    overlap = set(output).intersection(additions)
    if overlap:
        raise RuntimeError(f"rev13 NPZ fields collide with rev12: {sorted(overlap)}")
    output.update(additions)
    return output


def _controller_payload(state: _RunState) -> dict[str, Any]:
    if state.recorder is None:
        static = None
        if state.substrate is not None:
            static = np.asarray(
                state.substrate.vtheta[: state.substrate.n_e], dtype=np.float64,
            )
        return {
            "enabled": False,
            "mode": "exact_off",
            "manifest": dict(state.controller_spec),
            "support": None,
            "diagnostics": {
                "threshold_min_mV_observed": (
                    None if static is None else float(np.min(static))
                ),
                "threshold_max_mV_observed": (
                    None if static is None else float(np.max(static))
                ),
                "maximum_zero_sum_error_mV": 0.0,
                "maximum_saturation_fraction": 0.0,
                "maximum_support_weighted_saturation_fraction": 0.0,
                "maximum_static_modulation_sign_flip_fraction": 0.0,
                "maximum_support_weighted_static_modulation_sign_flip_fraction": 0.0,
            },
            "patient_runtime_inputs_used": False,
        }
    _validate_dynamic_diagnostics(state)
    return {
        "enabled": True,
        "mode": state.controller_spec["mode"],
        "manifest": dict(state.controller_spec),
        "support": dict(state.support_audit),
        "static_amplitude_definition": "support_weighted_centered_sd",
        "static_node_centered_sd_mV": state.static_centered_sd_mV,
        "static_node_rms_mV_legacy_name": state.static_centered_sd_mV,
        "diagnostics": state.recorder.diagnostics(),
        "trace_fields": sorted(_npz_additions(state)),
        "patient_runtime_inputs_used": False,
    }


def _augment_json_payload(payload: Mapping[str, Any], state: _RunState) -> dict:
    output = dict(payload)
    mechanism = dict(output.get("mechanism_freeze", {}))
    if mechanism.get("EE") != "off" or mechanism.get("E_to_I") != "off":
        raise RuntimeError("rev13 inherited an active learned edge pathway")
    if mechanism.get("Z_M") != "off" or not mechanism.get(
        "edge_coefficients_all_zero", False
    ):
        raise RuntimeError("rev13 inherited Z/M or nonzero edge coefficients")
    mechanism.update({
        "node_accessibility": state.controller_spec["mode"],
        "node_accessibility_active": state.recorder is not None,
    })
    output.update({
        "base_worker_status": output.get("status"),
        "status": "REV13_NODE_ZERO_SUM_WORKER_COMPLETE",
        "scientific_role": SCIENTIFIC_ROLE,
        "mechanism_freeze": mechanism,
        "node_accessibility": _controller_payload(state),
        "execution_duration": {
            "frozen_duration_ms": state.frozen_duration_ms,
            "requested_duration_ms": state.requested_duration_ms,
            "engineering_run_kind": state.engineering_run_kind,
            "duration_override_used": state.requested_duration_ms is not None,
        },
    })
    provenance = dict(output.get("provenance", {}))
    provenance["rev13_systemd_unit"] = os.environ.get("REV13_SYSTEMD_UNIT")
    provenance["composition_base_worker"] = "scripts/run_topic4_rev12_node_worker.py"
    provenance["compatibility_overlay"] = dict(state.compatibility_audit)
    provenance["rev13_explicit_runtime_freeze"] = copy.deepcopy(
        state.rev13_provenance
    )
    provenance["rev13_execution_duration"] = copy.deepcopy(
        output["execution_duration"]
    )
    output["provenance"] = provenance
    return output


def _compatibility_view(
    config: Mapping[str, Any], *, artifact_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add only fields required by rev12 execution, not its patient-side inputs."""
    stage_record = config["inputs"].get("stage_ak_config")
    if not isinstance(stage_record, Mapping):
        raise RuntimeError("rev13 config lacks its frozen Stage-AK execution contract")
    stage_path = rev12._resolve(artifact_root, stage_record["path"])
    if rev12._sha256(stage_path) != stage_record["sha256"]:
        raise RuntimeError("rev13 Stage-AK execution contract changed")
    stage_config = json.loads(stage_path.read_text())
    if config.get("event_unit") != stage_config.get("event_unit"):
        raise RuntimeError("rev13 event unit drifted from Stage-AK")
    if config.get("source_topology") != stage_config.get("source_topology"):
        raise RuntimeError("rev13 source topology drifted from Stage-AK")
    transition_record = stage_config["inputs"]["transition_config"]
    contact_readout = stage_config["search"]["contact_readout"]
    compatibility = copy.deepcopy(dict(config))
    compatibility["inputs"] = {
        "transition_config": copy.deepcopy(transition_record),
    }
    compatibility["search"]["contact_readout"] = copy.deepcopy(contact_readout)
    return compatibility, {
        "source": str(stage_path),
        "source_sha256": stage_record["sha256"],
        "added_fields": ["inputs.transition_config", "search.contact_readout"],
        "excluded_stage_ak_inputs": sorted(
            set(stage_config.get("inputs", {})).difference({"transition_config"})
        ),
        "patient_label_or_prototype_input_loaded": False,
    }


def _verify_rev13_substrate_inputs(
    config: Mapping[str, Any], *, artifact_root: Path | None, state: _RunState
) -> dict[str, Any]:
    """Hash only files consumed by frozen substrate/placement reconstruction."""
    inputs = config.get("inputs", {})
    missing = SUBSTRATE_INPUT_KEYS.difference(inputs)
    if missing:
        raise RuntimeError(
            "rev13 transition contract lacks substrate inputs: "
            + ", ".join(sorted(missing))
        )
    records = {}
    for key in sorted(SUBSTRATE_INPUT_KEYS):
        record = inputs[key]
        path = transition_module._input_path(record["path"], artifact_root)
        if not path.exists():
            raise RuntimeError(f"input missing: {record['path']}")
        digest = transition_module._sha256_file(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
        records[key] = {
            "path": record["path"],
            "expected": record["sha256"],
            "observed": digest,
            "match": True,
        }
    excluded = sorted(set(inputs).difference(SUBSTRATE_INPUT_KEYS))
    state.compatibility_audit["verified_transition_inputs"] = sorted(records)
    state.compatibility_audit["excluded_transition_inputs"] = excluded
    state.compatibility_audit[
        "patient_label_or_prototype_input_loaded"
    ] = False
    return {"all_match": True, "records": records}


class _JsonProxy:
    """Transform only the rev13 config parse; every other JSON parse is literal."""

    def __init__(self, source_text: str, replacement: Mapping[str, Any]):
        self.source_text = source_text
        self.replacement = copy.deepcopy(dict(replacement))

    def loads(self, text, *args, **kwargs):
        if text == self.source_text:
            return copy.deepcopy(self.replacement)
        return json.loads(text, *args, **kwargs)

    @staticmethod
    def dumps(value, *args, **kwargs):
        return json.dumps(value, *args, **kwargs)


def _validate_duration_override(
    duration_ms: float | None,
    engineering_run_kind: str | None,
    *,
    frozen_duration_ms: float,
) -> float | None:
    if not np.isclose(float(frozen_duration_ms), 20000.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("rev13 frozen scientific duration changed from 20000 ms")
    if duration_ms is None:
        if engineering_run_kind is not None:
            raise RuntimeError("engineering run kind requires --duration-ms")
        return None
    if engineering_run_kind not in {"sentinel", "parity"}:
        raise RuntimeError(
            "--duration-ms is restricted to engineering sentinel/parity runs"
        )
    duration = float(duration_ms)
    if (
        not np.isfinite(duration)
        or duration <= 0.0
        or duration > float(frozen_duration_ms)
        or duration > 20000.0
    ):
        raise RuntimeError("engineering duration must lie in (0, 20000] ms")
    return duration


def _rebuild_and_validate_manifest(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild every candidate from frozen Stage-AK/AL inputs and compare exactly."""
    manifest_path = artifact_root / str(config["candidate_manifest"])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError("rev13 candidate manifest status is not frozen")
    config_sha256 = _sha256_file(config_path)
    if manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("rev13 candidate manifest config_sha256 is stale")

    loaded: dict[str, Any] = {}
    input_audit: dict[str, Any] = {}
    for name, record in config.get("inputs", {}).items():
        path = rev12._resolve(artifact_root, record["path"])
        observed = rev12._sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"rev13 input hash changed: {name}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {
            "path": str(path),
            "expected_sha256": record["sha256"],
            "observed_sha256": observed,
            "match": True,
        }
    required = {"stage_ak_config", "stage_ak_manifest", "stage_al_manifest"}
    if set(loaded) != required:
        raise RuntimeError("rev13 manifest rebuild input set changed")
    rebuilt_candidates, rebuilt_substrate_audit = rev13_freezer.build_candidates(
        loaded["stage_ak_manifest"],
        loaded["stage_al_manifest"],
        loaded["stage_ak_config"],
        dict(config),
    )
    _assert_json_exact(
        manifest.get("candidates"), rebuilt_candidates, path="$.candidates"
    )
    _assert_json_exact(
        manifest.get("substrate_audit"),
        rebuilt_substrate_audit,
        path="$.substrate_audit",
    )
    for key in ("event_unit", "source_topology", "search", "pathways"):
        _assert_json_exact(manifest.get(key), config.get(key), path=f"$.{key}")
    return manifest, rebuilt_candidates, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_status": MANIFEST_STATUS,
        "config_sha256": config_sha256,
        "candidates_rebuilt_from_frozen_inputs": True,
        "candidate_exact_compare": True,
        "input_records": input_audit,
    }


def _preflight(argv: list[str] | None = None) -> tuple[argparse.Namespace, _RunState]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument(
        "--engineering-run-kind", choices=("sentinel", "parity")
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    source_config_text = config_path.read_text()
    config = json.loads(source_config_text)
    _validate_scientific_role(config.get("scientific_role"))
    _assert_no_patient_runtime_inputs(config)
    expected_pathways = {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off",
        "Z_M": "off",
    }
    if config.get("pathways") != expected_pathways:
        raise RuntimeError("rev13 config must keep EE, E-to-I and Z/M off")
    artifact_root = args.artifact_root.resolve()
    frozen_duration_ms = float(config["search"]["simulation"]["duration_ms"])
    duration_ms = _validate_duration_override(
        args.duration_ms,
        args.engineering_run_kind,
        frozen_duration_ms=frozen_duration_ms,
    )
    manifest, rebuilt_candidates, manifest_audit = _rebuild_and_validate_manifest(
        config, config_path=config_path, artifact_root=artifact_root
    )
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if manifest.get("provenance", {}).get("git_commit") != expected_commit:
        raise RuntimeError("rev13 manifest was not frozen at the expected commit")
    manifest_audit["manifest_git_commit"] = expected_commit
    matches = [
        row for row in rebuilt_candidates
        if row.get("candidate_id") == args.candidate_id
    ]
    if len(matches) != 1:
        parser.error("candidate is outside the frozen rev13 manifest")
    candidate = dict(matches[0])
    _assert_candidate_is_patient_free(candidate)
    if candidate.get("pathways") != expected_pathways:
        raise RuntimeError("rev13 candidate must keep EE, E-to-I and Z/M off")
    spec = _normalized_controller_spec(candidate)
    if duration_ms is not None:
        if args.candidate_id != "exact_off":
            raise RuntimeError(
                "engineering duration override is restricted to exact_off"
            )
        allowed_seed_key = (
            "canary_network_seeds"
            if args.engineering_run_kind == "sentinel"
            else "engineering_parity_network_seeds"
        )
        if int(args.seed) not in {
            int(seed) for seed in config["search"].get(allowed_seed_key, [])
        }:
            raise RuntimeError(
                f"engineering {args.engineering_run_kind} seed is outside its frozen pool"
            )
    rev13_provenance = _explicit_runtime_provenance(
        config_path, args.expected_commit
    )
    rev13_provenance["manifest_rebuild"] = manifest_audit
    compatibility, compatibility_audit = _compatibility_view(
        config, artifact_root=artifact_root
    )
    if duration_ms is not None:
        compatibility["search"]["simulation"]["duration_ms"] = duration_ms
        compatibility_audit["engineering_duration_override"] = {
            "kind": args.engineering_run_kind,
            "requested_duration_ms": duration_ms,
            "frozen_duration_ms": frozen_duration_ms,
        }
        if args.engineering_run_kind == "parity":
            compatibility["search"]["selection_network_seeds"] = [int(args.seed)]
            compatibility_audit["engineering_parity_seed_added_to_compatibility_pool"] = (
                int(args.seed)
            )
    return args, _RunState(
        candidate=candidate,
        controller_spec=spec,
        source_config_text=source_config_text,
        compatibility_config=compatibility,
        compatibility_audit=compatibility_audit,
        rev13_provenance=rev13_provenance,
        engineering_run_kind=args.engineering_run_kind,
        requested_duration_ms=duration_ms,
        frozen_duration_ms=frozen_duration_ms,
    )


def _run_rev12_composed(state: _RunState, base_module=rev12) -> None:
    original_build = base_module.build_substrate
    original_simulate = base_module.simulate_kick
    original_npz = base_module._atomic_npz
    original_json = base_module.atomic_write_json

    def build_substrate(*args, **kwargs):
        role = args[1] if len(args) > 1 else kwargs.get("role")
        if role != "node_baseline":
            raise RuntimeError("rev13 requires the frozen Node-only substrate")
        if float(kwargs.get("ee_dose", np.nan)) != 0.0:
            raise RuntimeError("rev13 learned EE redistribution must be off")
        if float(kwargs.get("etoi_dose", np.nan)) != 0.0:
            raise RuntimeError("rev13 learned E-to-I redistribution must be off")
        substrate = original_build(*args, **kwargs)
        if not np.allclose(substrate.edge_coefficients, 0.0, rtol=0.0, atol=0.0):
            raise RuntimeError("rev13 substrate contains active edge coefficients")
        state.substrate = substrate
        _build_controller(state, substrate)
        return substrate

    def simulate_kick(*args, **kwargs):
        if state.substrate is None:
            raise RuntimeError("rev13 controller injection preceded substrate construction")
        slow = kwargs.get("slow", args[3] if len(args) > 3 else None)
        if slow is not None:
            raise RuntimeError("rev13 requires Z/M and every slow mechanism off")
        if kwargs.get("node_accessibility") is not None:
            raise RuntimeError("base worker unexpectedly supplied node_accessibility")
        kwargs["node_accessibility"] = state.recorder
        return original_simulate(*args, **kwargs)

    def atomic_npz(path, **arrays):
        return original_npz(path, **_augment_npz_arrays(arrays, state))

    def atomic_json(payload, path):
        return original_json(_augment_json_payload(payload, state), path)

    with ExitStack() as stack:
        stack.enter_context(patch.object(base_module, "build_substrate", build_substrate))
        stack.enter_context(patch.object(base_module, "simulate_kick", simulate_kick))
        stack.enter_context(patch.object(base_module, "_atomic_npz", atomic_npz))
        stack.enter_context(patch.object(base_module, "atomic_write_json", atomic_json))
        stack.enter_context(patch.object(
            base_module, "_validate_scientific_role", _validate_scientific_role
        ))
        if state.source_config_text is not None:
            stack.enter_context(patch.object(
                base_module, "json", _JsonProxy(
                    state.source_config_text, state.compatibility_config
                )
            ))
        stack.enter_context(patch.object(
            transition_module,
            "verify_frozen_inputs",
            lambda config, *, artifact_root=None: _verify_rev13_substrate_inputs(
                config, artifact_root=artifact_root, state=state
            ),
        ))
        base_module.main()


def _base_worker_argv(args: argparse.Namespace) -> list[str]:
    """Translate rev13 CLI state to the strict rev12 compatibility surface."""
    argv = [
        str(Path(sys.argv[0])),
        "--config", str(args.config),
        "--candidate-id", str(args.candidate_id),
        "--seed", str(int(args.seed)),
        "--expected-commit", str(args.expected_commit),
        "--artifact-root", str(args.artifact_root),
    ]
    if args.out_json is not None:
        argv.extend(["--out-json", str(args.out_json)])
    if args.out_npz is not None:
        argv.extend(["--out-npz", str(args.out_npz)])
    return argv


def main() -> None:
    args, state = _preflight()
    with patch.object(sys, "argv", _base_worker_argv(args)):
        _run_rev12_composed(state)


if __name__ == "__main__":
    main()
