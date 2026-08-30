#!/usr/bin/env python3
"""Run one rev14 absolute-Fourier M3 field through the frozen rev12 worker.

This module does not implement a simulator. It composes over the rev12 static
Node worker, replacing only the E-neuron Node field after the frozen Stage-AK
``exact_off`` substrate has been reconstructed. Event boundaries, source maps
and contact readout remain rev12 outputs.
"""
from __future__ import annotations

import argparse
import copy
import csv
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src" / "snn_engine"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import freeze_topic4_rev14_m3_canary as freezer  # noqa: E402
from scripts import run_topic4_rev12_node_worker as rev12  # noqa: E402
from src.topic4_core_field import (  # noqa: E402
    core_thresholds,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_rev14_field_projection import (  # noqa: E402
    project_fourier_to_frozen_node,
)
from src.topic4_rev14_fourier_field import array_sha256, mode_inventory  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
WORKER_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_WORKER_COMPLETE"
PREPARE_STATUS = "REV14_M3_CANARY_WORKER_PREPARED_NO_SNN"
DOSE_TABLE_STATUS = "REV14_M3_PHYSICAL_DOSE_TABLE_PREPARED_NO_SNN"
BASE_COMPATIBILITY_ROLE = "development_only_orthogonal_free_field_screen"
EXPECTED_PATHWAYS = freezer.EXPECTED_PATHWAYS


def _field_design(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Fourier design while preserving legacy M3 manifests."""
    if "field_design" in config:
        if "m3_design" in config:
            raise RuntimeError("Fourier config cannot define both field_design and m3_design")
        return config["field_design"]
    return config["m3_design"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        freezer._jsonable(payload), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("RMS input is not finite")
    return float(np.sqrt(np.mean(array * array)))


def _assert_json_exact(actual: Any, expected: Any, *, path: str = "$") -> None:
    if type(actual) is not type(expected):
        raise RuntimeError(f"rev14 M3 manifest type changed at {path}")
    if isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise RuntimeError(f"rev14 M3 manifest keys changed at {path}")
        for key in expected:
            _assert_json_exact(actual[key], expected[key], path=f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"rev14 M3 manifest length changed at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_json_exact(left, right, path=f"{path}[{index}]")
    elif actual != expected:
        raise RuntimeError(f"rev14 M3 manifest value changed at {path}")


def _load_manifest(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    artifact_root: Path,
    prepare_only: bool,
    provenance: Mapping[str, Any],
) -> tuple[dict, dict]:
    rebuilt = freezer.build_manifest_payload(
        config_path,
        artifact_root=artifact_root,
        provenance=provenance,
        status=(freezer.PREPARE_STATUS if prepare_only else freezer.STATUS),
    )
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if prepare_only:
        return rebuilt, {
            "manifest_path": str(manifest_path),
            "manifest_read": False,
            "rebuilt_from_frozen_inputs": True,
            "prepare_only": True,
        }
    if not manifest_path.is_file():
        raise RuntimeError("rev14 M3 frozen candidate manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev14 M3 candidate manifest is not formally frozen")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev14 M3 candidate manifest config hash is stale")
    for key in (
        "candidates", "direction_audit", "exact_off_reconstruction",
        "signed_depth_audit", "event_unit", "source_topology", "search",
        "pathways", "inputs",
    ):
        _assert_json_exact(manifest.get(key), rebuilt.get(key), path=f"$.{key}")
    frozen_provenance = manifest.get("provenance", {})
    if not frozen_provenance.get("formal_ready"):
        raise RuntimeError("rev14 M3 manifest provenance is not formally frozen")
    if frozen_provenance.get("git_commit") != provenance.get("git_commit"):
        raise RuntimeError("rev14 M3 manifest and worker commits differ")
    return manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_read": True,
        "rebuilt_from_frozen_inputs": True,
        "candidate_exact_compare": True,
        "prepare_only": False,
    }


def _candidate(manifest: Mapping[str, Any], candidate_id: str) -> dict:
    rows = [
        row for row in manifest.get("candidates", [])
        if row.get("candidate_id") == candidate_id
    ]
    if len(rows) != 1:
        raise RuntimeError("candidate is outside or duplicated in the rev14 M3 manifest")
    candidate = copy.deepcopy(rows[0])
    if candidate.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("candidate activates a forbidden pathway")
    return candidate


def _frozen_original_signed_depth(
    substrate, config: Mapping[str, Any],
) -> np.ndarray:
    """Rebuild d_i from the original frozen quantiles, never from exact_off."""
    contract = config["node_mapping"]["signed_depth_contract"]
    n_e = int(substrate.n_e)
    expected = {
        "expected_n_e": n_e,
        "quantile_seed": int(substrate.stage["quantile_seed"]),
        "core_mean_mV": float(substrate.engine["core_mean"]),
        "core_std_mV": float(substrate.engine["core_std"]),
        "v_base_mV": float(substrate.engine["v_base"]),
    }
    for key, observed in expected.items():
        if contract[key] != observed:
            raise RuntimeError(f"rev14 M3 signed-depth contract changed: {key}")
    depth = np.asarray(signed_depth(
        core_thresholds(
            sample_core_quantiles(n_e, int(contract["quantile_seed"])),
            float(contract["core_mean_mV"]),
            float(contract["core_std_mV"]),
        ),
        float(contract["v_base_mV"]),
    ), dtype=np.float64)
    observed_hash = array_sha256(depth)
    if observed_hash != contract["sha256"]:
        raise RuntimeError("rev14 M3 original signed-depth array hash changed")
    return depth


def _project_candidate(candidate: Mapping[str, Any], substrate, config: Mapping[str, Any]) -> dict:
    design = _field_design(config)
    mapping_contract = config["node_mapping"]
    exact_h = np.asarray(substrate.h_e, dtype=np.float64)
    exact_vtheta = np.asarray(substrate.vtheta, dtype=np.float64)
    n_e = int(substrate.n_e)
    if exact_h.shape != (n_e,) or exact_vtheta.ndim != 1 or len(exact_vtheta) < n_e:
        raise RuntimeError("Stage-AK exact_off arrays have incompatible shapes")
    if not np.isfinite(exact_h).all() or np.any((exact_h <= 0.0) | (exact_h >= 1.0)):
        raise RuntimeError("Stage-AK exact_off h is outside (0,1)")
    v_base = float(substrate.engine["v_base"])
    if len(exact_vtheta) > n_e and not np.array_equal(
        exact_vtheta[n_e:], np.full(len(exact_vtheta) - n_e, v_base),
    ):
        raise RuntimeError("Stage-AK inhibitory thresholds drifted from v_base")
    target_mass = float(substrate.stage["N_core_manual"])
    if target_mass != float(mapping_contract["expected_target_h_mass"]):
        raise RuntimeError("rev14 M3 frozen target h mass changed")
    if not np.isclose(float(np.sum(exact_h)), target_mass, rtol=0.0, atol=1e-8):
        raise RuntimeError("Stage-AK exact_off h mass changed")
    signed_depth_values = _frozen_original_signed_depth(substrate, config)
    expected_exact_vtheta = v_base - exact_h * signed_depth_values
    exact_mapping_family_difference = float(np.max(np.abs(
        exact_vtheta[:n_e] - expected_exact_vtheta
    )))

    field_kind = str(candidate["field_kind"])
    coordinate = candidate.get("fourier_coordinate")
    modes = mode_inventory(int(design["maximum_order"]))
    if field_kind == "stage_ak_exact_off_benchmark":
        if coordinate is not None:
            raise RuntimeError("exact_off benchmark cannot contain Fourier coefficients")
        h = exact_h.copy()
        vtheta = exact_vtheta.copy()
        delta_vtheta = vtheta[:n_e] - v_base
        mapping = "stage_ak_exact_off_bypass_fourier"
        hashes = {
            "h_sha256": array_sha256(h),
            "vtheta_sha256": array_sha256(vtheta),
            "delta_vtheta_sha256": array_sha256(delta_vtheta),
            "frozen_signed_depth_sha256": array_sha256(signed_depth_values),
            "coefficients_sha256": None,
            "latent_at_neurons_sha256": None,
        }
        contract = {
            "schema_id": "topic4_rev14_exact_off_benchmark_projection_v1",
            "candidate_id": candidate["candidate_id"],
            "mapping": mapping,
            "target_h_mass": target_mass,
            "hashes": hashes,
        }
        hashes["projection_sha256"] = _canonical_sha256(contract)
        projection = {
            "h": h,
            "vtheta": vtheta,
            "delta_vtheta": delta_vtheta,
            "signed_depth": signed_depth_values,
            "latent_at_neurons": None,
            "hashes": hashes,
            "audit": {
                "mapping": mapping,
                "mapping_type": mapping,
                "field_kind": field_kind,
                "historical_field_inherited": True,
                "candidate_h_inherits_exact_off_geometry": True,
                "exact_off_bypasses_fourier": True,
                "target_h_mass": target_mass,
                "observed_h_mass": float(np.sum(h)),
                "h_mass_error": float(np.sum(h) - target_mass),
                "centered_latent_surface_rms": None,
                "latent_physical_sheet_mean": None,
                "h_rms": _rms(h),
                "h_centered_rms": _rms(h - float(np.mean(h))),
                "h_delta_from_uniform_rms": _rms(h - target_mass / n_e),
                "threshold_modulation_rms_mV": _rms(delta_vtheta),
                "h_delta_from_exact_off_rms": 0.0,
                "vtheta_delta_from_exact_off_rms_mV": 0.0,
                "uniform_h_value": None,
                "inhibitory_thresholds_unchanged": True,
                "exact_off_vs_original_signed_depth_mapping_max_abs_mV": (
                    exact_mapping_family_difference
                ),
                "exact_off_mapping_contract": (
                    "historical_dual_continuous_mean_dispersion_bypass"
                ),
                "exact_off_bypass_h_max_abs_error": 0.0,
                "exact_off_bypass_vtheta_max_abs_mV": 0.0,
            },
        }
    else:
        if not isinstance(coordinate, Mapping):
            raise RuntimeError("Fourier candidate lacks its frozen coordinate")
        frozen_modes = tuple(tuple(row) for row in coordinate["modes"])
        if frozen_modes != modes:
            raise RuntimeError("Fourier candidate M3 mode inventory changed")
        coefficients = np.asarray(coordinate["coefficients"], dtype=np.float64)
        if array_sha256(coefficients) != coordinate["coefficients_sha256"]:
            raise RuntimeError("Fourier candidate coefficient hash changed")
        projection = project_fourier_to_frozen_node(
            coefficients,
            np.asarray(substrate.positions_e, dtype=np.float64),
            modes,
            frozen_signed_depth=signed_depth_values,
            n_total=len(exact_vtheta),
            target_count=target_mass,
            v_base=v_base,
            tau_h=float(mapping_contract["tau_h"]),
            eps=float(mapping_contract["eps"]),
            L=float(design["sheet_length_mm"]),
            quadrature_n=int(design["quadrature_per_axis"]),
        )
        h = np.asarray(projection["h"], dtype=np.float64)
        vtheta = np.asarray(projection["vtheta"], dtype=np.float64)
        delta_vtheta = np.asarray(projection["delta_vtheta"], dtype=np.float64)
        target_rms = float(coordinate["target_centered_surface_rms"])
        if field_kind == "zero_fourier_uniform_benchmark":
            if np.count_nonzero(coefficients) != 0 or target_rms != 0.0:
                raise RuntimeError("uniform benchmark is not the zero Fourier field")
            expected_uniform = target_mass / n_e
            if not np.allclose(h, expected_uniform, rtol=0.0, atol=1e-14):
                raise RuntimeError("zero Fourier coefficients did not produce uniform h")
            mapping = "absolute_zero_fourier_uniform_mass_projection"
        elif field_kind in {
            "absolute_paired_phase_fourier_m3",
            "absolute_paired_phase_fourier_m4_shell_coordinate",
            "absolute_paired_phase_fourier_joint_m3_m4",
        }:
            if not candidate.get("selection_eligible"):
                raise RuntimeError("Fourier field unexpectedly became nonselectable")
            if not np.isclose(
                projection["audit"]["centered_latent_surface_rms"],
                target_rms, rtol=0.0, atol=2e-12,
            ):
                raise RuntimeError("Fourier centered surface RMS changed")
            mapping = "absolute_fourier_s_to_mass_projected_h"
        else:
            raise RuntimeError("unknown absolute Fourier field kind")
        projection["latent_at_neurons"] = projection.pop(
            "latent_surface_at_neurons"
        )
        projection["hashes"]["exact_off_h_sha256"] = array_sha256(exact_h)
        projection["hashes"]["exact_off_vtheta_sha256"] = array_sha256(exact_vtheta)
        projection["audit"].update({
            "mapping": mapping,
            "field_kind": field_kind,
            "candidate_h_inherits_exact_off_geometry": False,
            "exact_off_bypasses_fourier": False,
            "h_delta_from_exact_off_rms": _rms(h - exact_h),
            "vtheta_delta_from_exact_off_rms_mV": _rms(
                vtheta[:n_e] - exact_vtheta[:n_e]
            ),
            "uniform_h_value": (
                float(target_mass / n_e)
                if field_kind == "zero_fourier_uniform_benchmark" else None
            ),
            "inhibitory_thresholds_unchanged": True,
            "exact_off_vs_original_signed_depth_mapping_max_abs_mV": (
                exact_mapping_family_difference
            ),
            "exact_off_mapping_contract": (
                "historical_dual_continuous_mean_dispersion_benchmark_only"
            ),
        })

    if not np.isfinite(vtheta).all() or not np.isfinite(delta_vtheta).all():
        raise RuntimeError("rev14 M3 threshold projection is non-finite")
    if not np.array_equal(vtheta[n_e:], exact_vtheta[n_e:]):
        raise RuntimeError("rev14 M3 changed inhibitory thresholds")
    if projection["hashes"]["frozen_signed_depth_sha256"] != (
        mapping_contract["signed_depth_contract"]["sha256"]
    ):
        raise RuntimeError("rev14 M3 projection used the wrong frozen d_i")
    return projection


def _physical_dose_row(candidate: Mapping[str, Any], projection: Mapping[str, Any]) -> dict:
    h = np.asarray(projection["h"], dtype=np.float64)
    delta = np.asarray(projection["delta_vtheta"], dtype=np.float64)
    vtheta_e = np.asarray(projection["vtheta"], dtype=np.float64)[:len(h)]
    coordinate = candidate.get("fourier_coordinate") or {}
    h_q = np.quantile(h, [0.05, 0.5, 0.95])
    delta_q = np.quantile(delta, [0.05, 0.5, 0.95])
    return {
        "candidate_id": candidate["candidate_id"],
        "selection_eligible": bool(candidate["selection_eligible"]),
        "field_kind": candidate["field_kind"],
        "direction_index": coordinate.get("direction_index"),
        "sign": coordinate.get("sign"),
        "target_centered_surface_rms": coordinate.get(
            "target_centered_surface_rms"
        ),
        "target_h_mass": float(projection["audit"]["target_h_mass"]),
        "h_min": float(np.min(h)),
        "h_q05": float(h_q[0]),
        "h_median": float(h_q[1]),
        "h_q95": float(h_q[2]),
        "h_max": float(np.max(h)),
        "h_mean": float(np.mean(h)),
        "h_rms": _rms(h),
        "h_centered_rms": _rms(h - float(np.mean(h))),
        "delta_vtheta_min_mV": float(np.min(delta)),
        "delta_vtheta_q05_mV": float(delta_q[0]),
        "delta_vtheta_median_mV": float(delta_q[1]),
        "delta_vtheta_q95_mV": float(delta_q[2]),
        "delta_vtheta_max_mV": float(np.max(delta)),
        "delta_vtheta_mean_mV": float(np.mean(delta)),
        "delta_vtheta_rms_mV": _rms(delta),
        "abs_delta_vtheta_q95_mV": float(np.quantile(np.abs(delta), 0.95)),
        "threshold_lowering_fraction": float(np.mean(delta < 0.0)),
        "threshold_raising_fraction": float(np.mean(delta > 0.0)),
        "vtheta_e_min_mV": float(np.min(vtheta_e)),
        "vtheta_e_max_mV": float(np.max(vtheta_e)),
        "coefficients_sha256": coordinate.get("coefficients_sha256"),
        "h_sha256": projection["hashes"]["h_sha256"],
        "delta_vtheta_sha256": projection["hashes"]["delta_vtheta_sha256"],
        "frozen_signed_depth_sha256": projection["hashes"][
            "frozen_signed_depth_sha256"
        ],
        "projection_sha256": projection["hashes"]["projection_sha256"],
    }


def build_physical_dose_table(
    manifest: Mapping[str, Any], substrate, config: Mapping[str, Any], *, seed: int,
) -> dict:
    """Project every frozen coordinate on one CRN network without simulating it."""
    rows = [
        _physical_dose_row(candidate, _project_candidate(candidate, substrate, config))
        for candidate in manifest["candidates"]
    ]
    selectable = [row for row in rows if row["selection_eligible"]]
    benchmarks = [row for row in rows if not row["selection_eligible"]]
    design = _field_design(config)
    expected_total = int(design["candidate_count"])
    expected_selectable = int(design["selectable_candidate_count"])
    if (
        len(rows) != expected_total
        or len(selectable) != expected_selectable
        or len(benchmarks) != expected_total - expected_selectable
    ):
        raise RuntimeError("absolute-Fourier physical-dose table has the wrong candidate count")
    signed_depth_hashes = {
        row["frozen_signed_depth_sha256"] for row in rows
    }
    if signed_depth_hashes != {
        config["node_mapping"]["signed_depth_contract"]["sha256"]
    }:
        raise RuntimeError("rev14 M3 physical-dose rows do not share frozen d_i")
    payload = {
        "schema_id": "topic4_rev14_m3_physical_dose_table_v1",
        "status": DOSE_TABLE_STATUS,
        "network_seed": int(seed),
        "simulation_run": False,
        "n_selectable": len(selectable),
        "n_benchmarks": len(benchmarks),
        "dose_definition": (
            "absolute Fourier s -> frozen-mass h; DeltaVtheta_i=-h_i*d_i "
            "with original quantile-frozen signed depth"
        ),
        "frozen_signed_depth_sha256": next(iter(signed_depth_hashes)),
        "selectable_candidates": selectable,
        "nonselectable_benchmarks": benchmarks,
    }
    payload["table_sha256"] = _canonical_sha256(payload)
    return payload


def _atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        with Path(temporary).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _JsonProxy:
    def __init__(self, replacements: Mapping[str, Mapping[str, Any]]):
        self.replacements = {
            text: copy.deepcopy(dict(value)) for text, value in replacements.items()
        }

    def loads(self, text, *args, **kwargs):
        if text in self.replacements:
            return copy.deepcopy(self.replacements[text])
        return json.loads(text, *args, **kwargs)

    @staticmethod
    def dumps(value, *args, **kwargs):
        return json.dumps(value, *args, **kwargs)


def _compatibility_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any],
    *, artifact_root: Path,
) -> tuple[dict, dict]:
    rev13_record = config["inputs"]["rev13_config"]
    rev13_path = freezer._resolve(artifact_root, rev13_record["path"])
    if _sha256(rev13_path) != rev13_record["sha256"]:
        raise RuntimeError("rev13 compatibility config changed")
    rev13_config = json.loads(rev13_path.read_text())
    stage_record = rev13_config["inputs"]["stage_ak_config"]
    stage_path = freezer._resolve(artifact_root, stage_record["path"])
    if _sha256(stage_path) != stage_record["sha256"]:
        raise RuntimeError("Stage-AK execution contract changed")
    stage_config = json.loads(stage_path.read_text())
    if manifest["event_unit"] != stage_config["event_unit"]:
        raise RuntimeError("rev14 M3 event-unit contract drifted from Stage-AK")
    if manifest["source_topology"] != stage_config["source_topology"]:
        raise RuntimeError("rev14 M3 source-topology contract drifted from Stage-AK")
    compatibility = copy.deepcopy(dict(config))
    compatibility["scientific_role"] = BASE_COMPATIBILITY_ROLE
    compatibility["inputs"] = {
        "transition_config": copy.deepcopy(stage_config["inputs"]["transition_config"])
    }
    compatibility["event_unit"] = copy.deepcopy(manifest["event_unit"])
    compatibility["source_topology"] = copy.deepcopy(manifest["source_topology"])
    compatibility["search"] = copy.deepcopy(config["search"])
    compatibility["search"]["contact_readout"] = copy.deepcopy(
        stage_config["search"]["contact_readout"]
    )
    return compatibility, {
        "rev13_config": str(rev13_path),
        "stage_ak_config": str(stage_path),
        "stage_ak_config_sha256": stage_record["sha256"],
        "added_fields": [
            "inputs.transition_config", "search.contact_readout",
            "event_unit", "source_topology",
        ],
        "patient_target_input_loaded": False,
    }


def _compatibility_manifest(
    manifest: Mapping[str, Any], candidate: Mapping[str, Any], config_sha256: str,
) -> dict:
    base = copy.deepcopy(manifest["exact_off_reconstruction"])
    base.update({
        "candidate_id": candidate["candidate_id"],
        "selection_eligible": candidate["selection_eligible"],
        "role": "rev14_m3_base_reconstruction_only",
    })
    return {
        "status": freezer.STATUS,
        "config_sha256": config_sha256,
        "candidates": [base],
    }


@dataclass
class _RunState:
    config: dict
    candidate: dict
    manifest: dict
    provenance: dict
    manifest_audit: dict
    compatibility_config: dict
    compatibility_manifest: dict
    compatibility_audit: dict
    source_config_text: str
    source_manifest_text: str
    projection: dict | None = None
    substrate: Any = None


def _reconstruct_static_node_substrate(
    state: _RunState, *, seed: int, artifact_root: Path, base_module=rev12,
):
    """Reuse the frozen rev12 reconstruction path without entering simulate_kick."""
    transition_record = state.compatibility_config["inputs"]["transition_config"]
    transition_path = freezer._resolve(artifact_root, transition_record["path"])
    if _sha256(transition_path) != transition_record["sha256"]:
        raise RuntimeError("rev14 M3 transition config changed")
    transition = base_module.load_round_config(transition_path)
    base = state.compatibility_manifest["candidates"][0]
    node_mapping = base.get("node_mapping", {})
    substrate = base_module.build_substrate(
        transition,
        "node_baseline",
        int(seed),
        cache_dir=str(artifact_root / state.config["network_cache"]),
        ee_dose=0.0,
        etoi_dose=0.0,
        node_candidate_override=base["node_field"],
        node_depth_shrinkage=float(
            node_mapping.get("signed_depth_shrinkage", 1.0)
        ),
        node_gain=float(node_mapping.get("node_gain", 1.0)),
        node_dispersion_candidate_override=base.get("node_dispersion_field"),
        artifact_root=artifact_root,
    )
    if not np.array_equal(
        np.asarray(substrate.edge_coefficients),
        np.zeros_like(np.asarray(substrate.edge_coefficients)),
    ):
        raise RuntimeError("rev14 M3 dose-table substrate has active edge coefficients")
    return substrate


def _augment_npz_arrays(arrays: Mapping[str, np.ndarray], state: _RunState) -> dict:
    if state.projection is None:
        raise RuntimeError("rev14 M3 projection was not built before NPZ output")
    output = dict(arrays)
    coordinate = state.candidate.get("fourier_coordinate")
    modes = mode_inventory(int(_field_design(state.config)["maximum_order"]))
    coefficients = (
        np.zeros((0, 2), dtype=np.float64)
        if coordinate is None else np.asarray(coordinate["coefficients"], dtype=np.float64)
    )
    output.update({
        "rev14_fourier_modes": np.asarray(modes, dtype=np.int16),
        "rev14_fourier_coefficients": coefficients,
        "rev14_frozen_signed_depth": np.asarray(
            state.projection["signed_depth"], dtype=np.float64,
        ),
        "rev14_projection_sha256": np.asarray(
            state.projection["hashes"]["projection_sha256"], dtype="U64",
        ),
    })
    return output


def _augment_json_payload(payload: Mapping[str, Any], state: _RunState) -> dict:
    if state.projection is None:
        raise RuntimeError("rev14 M3 projection was not built before JSON output")
    output = copy.deepcopy(dict(payload))
    mechanism = dict(output.get("mechanism_freeze", {}))
    if mechanism.get("EE") != "off" or mechanism.get("E_to_I") != "off":
        raise RuntimeError("rev14 M3 inherited an active learned edge pathway")
    if mechanism.get("Z_M") != "off" or not mechanism.get(
        "edge_coefficients_all_zero", False,
    ):
        raise RuntimeError("rev14 M3 inherited Z/M or nonzero edge coefficients")
    output.update({
        "base_worker_status": output.get("status"),
        "status": WORKER_STATUS,
        "scientific_role": state.config["scientific_role"],
        "field_sha256": state.projection["hashes"]["h_sha256"],
        "candidate_selection_eligible": bool(state.candidate["selection_eligible"]),
        "fourier_field": copy.deepcopy(state.candidate.get("fourier_coordinate")),
        "field_projection": {
            "audit": copy.deepcopy(state.projection["audit"]),
            "hashes": copy.deepcopy(state.projection["hashes"]),
        },
        "mechanism_freeze": {
            **mechanism,
            "static_node_field": state.candidate["field_kind"],
        },
        "node_mapping": {
            "mapping_type": state.projection["audit"]["mapping"],
            "projection_sha256": state.projection["hashes"]["projection_sha256"],
            "signed_depth_source": "original_frozen_quantile_draw",
            "frozen_signed_depth_sha256": state.projection["hashes"][
                "frozen_signed_depth_sha256"
            ],
            "candidate_h_inherits_exact_off_geometry": state.projection["audit"][
                "candidate_h_inherits_exact_off_geometry"
            ],
        },
    })
    provenance = dict(output.get("provenance", {}))
    provenance.update({
        "rev14_systemd_unit": os.environ.get("REV14_SYSTEMD_UNIT"),
        "composition_base_worker": "scripts/run_topic4_rev12_node_worker.py",
        "rev14_explicit_runtime_freeze": copy.deepcopy(state.provenance),
        "rev14_manifest_audit": copy.deepcopy(state.manifest_audit),
        "rev14_compatibility_overlay": copy.deepcopy(state.compatibility_audit),
    })
    output["provenance"] = provenance
    return output


def _run_rev12_composed(state: _RunState, base_module=rev12) -> None:
    original_build = base_module.build_substrate
    original_simulate = base_module.simulate_kick
    original_npz = base_module._atomic_npz
    original_json = base_module.atomic_write_json

    def build_substrate(*args, **kwargs):
        role = args[1] if len(args) > 1 else kwargs.get("role")
        if role != "node_baseline":
            raise RuntimeError("rev14 M3 requires the frozen Node-only substrate")
        if float(kwargs.get("ee_dose", np.nan)) != 0.0:
            raise RuntimeError("rev14 M3 learned EE redistribution must be off")
        if float(kwargs.get("etoi_dose", np.nan)) != 0.0:
            raise RuntimeError("rev14 M3 learned E-to-I redistribution must be off")
        substrate = original_build(*args, **kwargs)
        if not np.allclose(substrate.edge_coefficients, 0.0, rtol=0.0, atol=0.0):
            raise RuntimeError("rev14 M3 substrate contains active edge coefficients")
        projection = _project_candidate(
            state.candidate, substrate, state.config,
        )
        substrate.h_e = np.asarray(projection["h"], dtype=np.float64)
        substrate.vtheta = np.asarray(projection["vtheta"], dtype=np.float64)
        substrate.delta_vtheta = np.asarray(
            projection["delta_vtheta"], dtype=np.float64,
        )
        substrate.extras["rev14_m3_projection"] = copy.deepcopy(projection["audit"])
        state.substrate = substrate
        state.projection = projection
        return substrate

    def simulate_kick(*args, **kwargs):
        slow = kwargs.get("slow", args[3] if len(args) > 3 else None)
        if slow is not None:
            raise RuntimeError("rev14 M3 requires every slow mechanism off")
        if kwargs.get("node_accessibility") is not None:
            raise RuntimeError("rev14 M3 cannot activate dynamic Node accessibility")
        return original_simulate(*args, **kwargs)

    def atomic_npz(path, **arrays):
        return original_npz(path, **_augment_npz_arrays(arrays, state))

    def atomic_json(payload, path):
        return original_json(_augment_json_payload(payload, state), path)

    proxy = _JsonProxy({
        state.source_config_text: state.compatibility_config,
        state.source_manifest_text: state.compatibility_manifest,
    })
    with ExitStack() as stack:
        stack.enter_context(patch.object(base_module, "build_substrate", build_substrate))
        stack.enter_context(patch.object(base_module, "simulate_kick", simulate_kick))
        stack.enter_context(patch.object(base_module, "_atomic_npz", atomic_npz))
        stack.enter_context(patch.object(base_module, "atomic_write_json", atomic_json))
        stack.enter_context(patch.object(base_module, "json", proxy))
        base_module.main()


def _preflight(argv: list[str] | None = None) -> tuple[argparse.Namespace, _RunState | None]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-npz", type=Path)
    parser.add_argument("--physical-dose-table-out", type=Path)
    parser.add_argument("--prepare-only", "--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal worker execution requires --expected-commit")
    config_path = args.config.resolve()
    source_config_text = config_path.read_text()
    config = json.loads(source_config_text)
    freezer._validate_config(config)
    if int(args.seed) not in {int(seed) for seed in config["search"]["active_network_seeds"]}:
        parser.error("seed is outside the frozen active rev14 M3 pool")
    artifact_root = args.artifact_root.resolve()
    provenance = freezer.runtime_provenance(
        config_path,
        expected_commit=args.expected_commit,
        require_clean=not args.prepare_only,
    )
    manifest, manifest_audit = _load_manifest(
        config_path, config, artifact_root=artifact_root,
        prepare_only=args.prepare_only, provenance=provenance,
    )
    candidate = _candidate(manifest, args.candidate_id)
    if args.prepare_only and args.physical_dose_table_out is None:
        print(json.dumps({
            "status": PREPARE_STATUS,
            "candidate_id": args.candidate_id,
            "selection_eligible": candidate["selection_eligible"],
            "field_kind": candidate["field_kind"],
            "seed": int(args.seed),
            "duration_ms": float(config["search"]["simulation"]["duration_ms"]),
            "mechanisms": copy.deepcopy(config["pathways"]),
            "manifest_read": manifest_audit["manifest_read"],
            "formal_ready": provenance["formal_ready"],
            "snn_started": False,
        }, indent=2))
        return args, None
    compatibility, compatibility_audit = _compatibility_config(
        config, manifest, artifact_root=artifact_root,
    )
    manifest_path = artifact_root / str(config["candidate_manifest"])
    source_manifest_text = (
        manifest_path.read_text()
        if manifest_audit["manifest_read"] else json.dumps(manifest)
    )
    compatibility_manifest = _compatibility_manifest(
        manifest, candidate, _sha256(config_path),
    )
    return args, _RunState(
        config=copy.deepcopy(config), candidate=candidate,
        manifest=manifest, provenance=provenance,
        manifest_audit=manifest_audit,
        compatibility_config=compatibility,
        compatibility_manifest=compatibility_manifest,
        compatibility_audit=compatibility_audit,
        source_config_text=source_config_text,
        source_manifest_text=source_manifest_text,
    )


def _base_worker_argv(args: argparse.Namespace) -> list[str]:
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


def main(argv: list[str] | None = None) -> None:
    args, state = _preflight(argv)
    if state is None:
        return
    if args.physical_dose_table_out is not None:
        artifact_root = args.artifact_root.resolve()
        substrate = _reconstruct_static_node_substrate(
            state, seed=int(args.seed), artifact_root=artifact_root,
        )
        table = build_physical_dose_table(
            state.manifest, substrate, state.config, seed=int(args.seed),
        )
        output_json = args.physical_dose_table_out.resolve()
        output_csv = output_json.with_suffix(".csv")
        freezer._atomic_json(output_json, table)
        _atomic_csv(output_csv, table["selectable_candidates"])
        print(json.dumps({
            "status": table["status"],
            "n_selectable": table["n_selectable"],
            "n_benchmarks": table["n_benchmarks"],
            "network_seed": table["network_seed"],
            "simulation_run": False,
            "json": str(output_json),
            "csv": str(output_csv),
            "table_sha256": table["table_sha256"],
        }, indent=2))
        return
    with patch.object(sys, "argv", _base_worker_argv(args)):
        _run_rev12_composed(state)


if __name__ == "__main__":
    main()
