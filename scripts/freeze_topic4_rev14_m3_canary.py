#!/usr/bin/env python3
"""Freeze the observation-free rev14 M3 static-Node canary coordinates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_core_field import (  # noqa: E402
    core_thresholds,
    sample_core_quantiles,
    signed_depth,
)
from src.topic4_rev14_fourier_field import (  # noqa: E402
    array_sha256,
    mode_inventory,
    normalize_shell_rms,
    spectral_roughness_surrogate,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_FROZEN"
PREPARE_STATUS = "REV14_M3_OBSERVATION_FREE_CANARY_PREPARED_NOT_FROZEN"
EXPECTED_SCHEMA = "topic4_rev14_m3_observation_free_canary_v1"
EXPECTED_PATHWAYS = {
    "learned_E_to_E_redistribution": "off",
    "learned_E_to_I_redistribution": "off",
    "Z_M": "off",
}
FORMAL_RUNTIME_PATHS = (
    "config/topic4_rev14_m3_canary.json",
    "scripts/freeze_topic4_rev14_m3_canary.py",
    "scripts/run_topic4_rev14_m3_canary_worker.py",
    "scripts/run_topic4_rev12_node_worker.py",
    "src/topic4_core_field.py",
    "src/topic4_rev14_fourier_field.py",
    "src/topic4_rev14_field_projection.py",
)
FORBIDDEN_GENERATION_TOKENS = (
    "patient", "contact", "shaft", "electrode", "gaussian", "prototype",
    "classifier", "heldout", "manual_core",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / str(relative)
    return local if local.exists() else artifact_root / str(relative)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git(arguments: list[str], *, text: bool = True):
    return subprocess.check_output(arguments, cwd=ROOT, text=text)


def _path_provenance(relative: str, expected_commit: str | None) -> dict[str, Any]:
    absolute = ROOT / relative
    if not absolute.is_file():
        raise RuntimeError(f"rev14 M3 runtime path is missing: {relative}")
    dirty = bool(_git([
        "git", "status", "--porcelain", "--untracked-files=all", "--", relative,
    ]).strip())
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    record = {
        "observed_sha256": _sha256(absolute),
        "tracked": tracked,
        "dirty": dirty,
        "expected_sha256": None,
        "matches_expected_commit": False,
    }
    if expected_commit is not None and tracked:
        try:
            committed = _git(
                ["git", "show", f"{expected_commit}:{relative}"], text=False,
            )
        except subprocess.CalledProcessError:
            committed = None
        if committed is not None:
            record["expected_sha256"] = hashlib.sha256(committed).hexdigest()
            record["matches_expected_commit"] = (
                record["observed_sha256"] == record["expected_sha256"]
            )
    return record


def runtime_provenance(
    config_path: Path,
    *,
    expected_commit: str | None,
    require_clean: bool,
) -> dict[str, Any]:
    relative_config = str(config_path.resolve().relative_to(ROOT))
    paths = tuple(dict.fromkeys((relative_config, *FORMAL_RUNTIME_PATHS)))
    current = _git(["git", "rev-parse", "HEAD"]).strip()
    expected = None
    if expected_commit is not None:
        expected = _git(["git", "rev-parse", str(expected_commit)]).strip()
    records = {
        relative: _path_provenance(relative, expected) for relative in paths
    }
    all_clean = all(not row["dirty"] for row in records.values())
    all_tracked = all(row["tracked"] for row in records.values())
    all_match = bool(expected) and all(
        row["matches_expected_commit"] for row in records.values()
    )
    commit_match = expected is not None and current == expected
    if require_clean and not (
        commit_match and all_clean and all_tracked and all_match
    ):
        raise RuntimeError(
            "rev14 M3 formal provenance is not clean and frozen: "
            f"commit_match={commit_match}, all_clean={all_clean}, "
            f"all_tracked={all_tracked}, all_match={all_match}"
        )
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "commit_matches": commit_match,
        "all_explicit_paths_clean": all_clean,
        "all_explicit_paths_tracked": all_tracked,
        "all_explicit_paths_match_expected_commit": all_match,
        "formal_ready": bool(commit_match and all_clean and all_tracked and all_match),
        "explicit_runtime_files": records,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_id") != EXPECTED_SCHEMA:
        raise RuntimeError("rev14 M3 config schema changed")
    if config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev14 M3 must keep EE, E-to-I and Z/M off")
    if set(config.get("inputs", {})) != {
        "rev13_config", "rev13_exact_off_manifest",
    }:
        raise RuntimeError("rev14 M3 input set changed")
    for name, record in config["inputs"].items():
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise RuntimeError(f"rev14 M3 input hash is not frozen: {name}")
    design = config["m3_design"]
    expected = {
        "basis_family": "absolute_paired_phase_whole_sheet_fourier",
        "maximum_order": 3,
        "expected_modes": 14,
        "expected_real_coefficients": 28,
        "direction_count": 8,
        "surface_rms_levels": [0.8, 1.4],
        "signs": [-1, 1],
        "sheet_length_mm": 20.0,
        "quadrature_per_axis": 128,
        "candidate_count": 34,
        "selectable_candidate_count": 32,
        "basis_uses_observation_geometry": False,
        "basis_uses_predeclared_objects": False,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise RuntimeError(f"rev14 M3 design field changed: {key}")
    if config.get("node_mapping") != {
        "h_formula": "sigmoid((s_at_E_neuron-lambda)/tau_h)",
        "lambda_constraint": "sum_E_h_equals_frozen_stage_ak_mass",
        "expected_target_h_mass": 1129.0,
        "tau_h": 0.25,
        "eps": 0.001,
        "signed_depth_source": "original_frozen_quantile_draw",
        "signed_depth_formula": (
            "signed_depth(core_thresholds(sample_core_quantiles(n_e,"
            "quantile_seed),core_mean,core_std),v_base)"
        ),
        "signed_depth_contract": {
            "expected_n_e": 32000,
            "quantile_seed": 20260806,
            "core_mean_mV": 17.5,
            "core_std_mV": 1.0,
            "v_base_mV": 18.0,
            "sha256": (
                "68430aff5ab740baa93964745cbaf089720a1eb4518b59f109160faf3fb018ce"
            ),
        },
        "vtheta_formula": "Vtheta_E=v_base-h*d_i",
        "node_gain": 1.0,
        "zero_fourier_is_uniform": True,
        "exact_off_bypasses_fourier": True,
    }:
        raise RuntimeError("rev14 M3 absolute Node mapping contract changed")
    generation_payload = json.dumps({
        "candidate_generation_inputs": design.get("candidate_generation_inputs"),
        "orthogonalization": design.get("orthogonalization"),
    }).lower()
    if any(token in generation_payload for token in FORBIDDEN_GENERATION_TOKENS):
        raise RuntimeError("rev14 M3 candidate generation uses observation geometry")
    search = config["search"]
    if search.get("active_network_seeds") != [2321]:
        raise RuntimeError("rev14 M3 initial CRN seed changed")
    if search.get("canary_network_seeds") != [2321]:
        raise RuntimeError("rev14 M3 canary seed changed")
    if float(search["simulation"]["duration_ms"]) != 20000.0:
        raise RuntimeError("rev14 M3 simulation duration changed")


def frozen_signed_depth_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the original position-independent d_i draw and verify its hash."""
    contract = config["node_mapping"]["signed_depth_contract"]
    n_e = int(contract["expected_n_e"])
    values = np.asarray(signed_depth(
        core_thresholds(
            sample_core_quantiles(n_e, int(contract["quantile_seed"])),
            float(contract["core_mean_mV"]),
            float(contract["core_std_mV"]),
        ),
        float(contract["v_base_mV"]),
    ), dtype=np.float64)
    observed = array_sha256(values)
    if observed != contract["sha256"]:
        raise RuntimeError("rev14 M3 frozen original signed-depth hash changed")
    return {
        "source": config["node_mapping"]["signed_depth_source"],
        "formula": config["node_mapping"]["signed_depth_formula"],
        "n_e": n_e,
        "quantile_seed": int(contract["quantile_seed"]),
        "core_mean_mV": float(contract["core_mean_mV"]),
        "core_std_mV": float(contract["core_std_mV"]),
        "v_base_mV": float(contract["v_base_mV"]),
        "sha256": observed,
        "minimum_mV": float(np.min(values)),
        "maximum_mV": float(np.max(values)),
        "mean_mV": float(np.mean(values)),
        "rms_mV": float(np.sqrt(np.mean(values * values))),
        "position_or_exact_off_geometry_used": False,
    }


def _load_inputs(config: Mapping[str, Any], artifact_root: Path) -> tuple[dict, dict, dict]:
    loaded: dict[str, dict] = {}
    audit: dict[str, dict] = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"rev14 M3 input changed: {name}")
        loaded[name] = json.loads(path.read_text())
        audit[name] = {
            "path": str(path),
            "expected_sha256": record["sha256"],
            "observed_sha256": observed,
            "match": True,
        }
    return loaded["rev13_config"], loaded["rev13_exact_off_manifest"], audit


def _exact_off_reconstruction(rev13_config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    if rev13_config.get("schema_id") != "topic4_rev13_node_zero_sum_recovery_v2":
        raise RuntimeError("rev13 source config changed")
    if rev13_config.get("pathways") != EXPECTED_PATHWAYS:
        raise RuntimeError("rev13 source pathways are not off")
    if manifest.get("status") != "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN":
        raise RuntimeError("rev13 exact_off manifest status changed")
    if manifest.get("config_sha256") != hashlib.sha256(
        (ROOT / "config/topic4_rev13_node_zero_sum_recovery.json").read_bytes()
    ).hexdigest():
        raise RuntimeError("rev13 exact_off manifest is stale")
    rows = [
        row for row in manifest.get("candidates", [])
        if row.get("candidate_id") == "exact_off"
    ]
    if len(rows) != 1:
        raise RuntimeError("rev13 exact_off candidate is absent or duplicated")
    source = rows[0]
    if source.get("selection_eligible") is not False:
        raise RuntimeError("rev13 exact_off unexpectedly became selectable")
    required = {
        "primary_substrate_id", "source_candidate_ids", "node_field",
        "node_dispersion_field", "node_mapping",
    }
    if not required.issubset(source):
        raise RuntimeError("rev13 exact_off reconstruction payload is incomplete")
    return {key: copy.deepcopy(source[key]) for key in sorted(required)}


def orthogonal_sobol_directions(design: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    modes = mode_inventory(int(design["maximum_order"]))
    dimension = 2 * len(modes)
    if len(modes) != int(design["expected_modes"]):
        raise RuntimeError("M3 mode count changed")
    if dimension != int(design["expected_real_coefficients"]):
        raise RuntimeError("M3 real dimension changed")
    count = int(design["direction_count"])
    if count < 1 or count > dimension or count & (count - 1):
        raise RuntimeError("Sobol direction count must be a power of two within M3")
    engine = qmc.Sobol(
        d=dimension,
        scramble=bool(design["sobol_scramble"]),
        seed=int(design["sobol_seed"]),
    )
    raw = 2.0 * engine.random_base2(int(np.log2(count))) - 1.0
    directions: list[np.ndarray] = []
    for row in raw:
        vector = np.asarray(row, dtype=np.float64).copy()
        for previous in directions:
            vector -= np.dot(vector, previous) * previous
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("Sobol directions are rank deficient")
        vector /= norm
        if float(np.dot(vector, row)) < 0.0:
            vector *= -1.0
        directions.append(vector)
    orthonormal = np.vstack(directions)
    gram = orthonormal @ orthonormal.T
    maximum_error = float(np.max(np.abs(gram - np.eye(count))))
    if maximum_error > 1e-12:
        raise RuntimeError("Sobol directions are not orthonormal")
    return orthonormal, {
        "modes": [list(mode) for mode in modes],
        "modes_sha256": array_sha256(np.asarray(modes, dtype=np.int64)),
        "raw_sobol_sha256": array_sha256(raw),
        "orthonormal_directions_sha256": array_sha256(orthonormal),
        "maximum_gram_error": maximum_error,
        "candidate_generation_inputs": design["candidate_generation_inputs"],
        "observation_geometry_used": False,
        "predeclared_object_basis_used": False,
    }


def _coordinate_record(
    *,
    candidate_id: str,
    field_kind: str,
    selectable: bool,
    modes: tuple[tuple[int, int], ...],
    coefficients: np.ndarray | None,
    direction_index: int | None = None,
    sign: int | None = None,
    target_rms: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_id": candidate_id,
        "field_kind": field_kind,
        "selection_eligible": bool(selectable),
        "pathways": copy.deepcopy(EXPECTED_PATHWAYS),
    }
    if coefficients is None:
        record["fourier_coordinate"] = None
        return record
    coeff = np.asarray(coefficients, dtype=np.float64)
    record["fourier_coordinate"] = {
        "modes": [list(mode) for mode in modes],
        "coefficients": coeff.tolist(),
        "coefficients_sha256": array_sha256(coeff),
        "direction_index": direction_index,
        "sign": sign,
        "target_centered_surface_rms": target_rms,
        "spectral_roughness_surrogate": spectral_roughness_surrogate(coeff, modes),
        "absolute_field_not_exact_off_residual": True,
        "observation_geometry_used": False,
    }
    return record


def build_candidates(config: Mapping[str, Any]) -> tuple[list[dict], dict]:
    _validate_config(config)
    design = config["m3_design"]
    modes = mode_inventory(int(design["maximum_order"]))
    orthonormal, direction_audit = orthogonal_sobol_directions(design)
    candidates = [
        _coordinate_record(
            candidate_id="exact_off", field_kind="stage_ak_exact_off_benchmark",
            selectable=False, modes=modes, coefficients=None,
        ),
        _coordinate_record(
            candidate_id="uniform_node", field_kind="zero_fourier_uniform_benchmark",
            selectable=False, modes=modes,
            coefficients=np.zeros((len(modes), 2), dtype=np.float64),
            target_rms=0.0,
        ),
    ]
    for direction_index, vector in enumerate(orthonormal):
        unit_surface = normalize_shell_rms(
            vector.reshape(len(modes), 2), modes, target_rms=1.0,
            n_per_axis=int(design["quadrature_per_axis"]),
            L=float(design["sheet_length_mm"]),
        )
        for target_rms in map(float, design["surface_rms_levels"]):
            for sign in map(int, design["signs"]):
                coefficients = sign * target_rms * unit_surface
                rms_tag = f"r{int(round(10 * target_rms)):02d}"
                sign_tag = "p" if sign > 0 else "m"
                candidates.append(_coordinate_record(
                    candidate_id=f"m3_d{direction_index:02d}_{sign_tag}_{rms_tag}",
                    field_kind="absolute_paired_phase_fourier_m3",
                    selectable=True,
                    modes=modes,
                    coefficients=coefficients,
                    direction_index=direction_index,
                    sign=sign,
                    target_rms=target_rms,
                ))
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("rev14 M3 candidate count changed")
    selectable = [row for row in candidates if row["selection_eligible"]]
    if len(selectable) != int(design["selectable_candidate_count"]):
        raise RuntimeError("rev14 M3 selectable count changed")
    coordinate_hashes = [
        row["fourier_coordinate"]["coefficients_sha256"]
        for row in candidates if row["fourier_coordinate"] is not None
    ]
    if len(coordinate_hashes) != len(set(coordinate_hashes)):
        raise RuntimeError("rev14 M3 contains duplicate Fourier coordinates")
    return candidates, direction_audit


def build_manifest_payload(
    config_path: Path,
    *,
    artifact_root: Path,
    provenance: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    _validate_config(config)
    rev13_config, rev13_manifest, input_audit = _load_inputs(config, artifact_root)
    reconstruction = _exact_off_reconstruction(rev13_config, rev13_manifest)
    candidates, direction_audit = build_candidates(config)
    signed_depth_audit = frozen_signed_depth_audit(config)
    return {
        "schema_id": "topic4_rev14_m3_observation_free_canary_manifest_v1",
        "status": status,
        "config": str(config_path.resolve().relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "direction_audit": direction_audit,
        "signed_depth_audit": signed_depth_audit,
        "exact_off_reconstruction": reconstruction,
        "event_unit": copy.deepcopy(rev13_config["event_unit"]),
        "source_topology": copy.deepcopy(rev13_config["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": input_audit,
        "provenance": copy.deepcopy(dict(provenance)),
        "claim_boundary": config["claim_boundary"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--prepare-only", "--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal freeze requires --expected-commit")
    provenance = runtime_provenance(
        config_path,
        expected_commit=args.expected_commit,
        require_clean=not args.prepare_only,
    )
    status = PREPARE_STATUS if args.prepare_only else STATUS
    payload = build_manifest_payload(
        config_path, artifact_root=artifact_root,
        provenance=provenance, status=status,
    )
    config = json.loads(config_path.read_text())
    output = artifact_root / config["candidate_manifest"]
    if not args.prepare_only:
        _atomic_json(output, payload)
    print(json.dumps({
        "status": status,
        "n_candidates": len(payload["candidates"]),
        "n_selectable": sum(
            bool(row["selection_eligible"]) for row in payload["candidates"]
        ),
        "formal_ready": bool(provenance["formal_ready"]),
        "output_written": not args.prepare_only,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
