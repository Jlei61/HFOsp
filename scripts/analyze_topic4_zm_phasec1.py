#!/usr/bin/env python3
"""Fail-closed Phase-C1 base-atlas and conditional-gain adjudication.

The base atlas is deliberately independent of conditional gain.  It consumes
one JSON + NPZ part for every valid slow-field cell, fast phase and future
noise, assigns the already locked run-level phenotype, and then applies the
5/6 cell rule.  A write-once trigger manifest may subsequently route tonic
cells that pass the spike-only AI screen to a separate gain experiment.

This module does not run the SNN and does not infer entry, offset, recovery, or
a lifecycle from a frozen-state phenotype.

Runner-facing C1 base-part schema
---------------------------------
JSON (``C1_BASE_PART_SCHEMA``):
  phasec_manifest_sha256, coordinate_manifest_sha256, seed, cell_id, tier,
  trajectory_id, path_index, path_direction, phase, noise, resolution,
  slow_state_sha256, config_sha, noise_bank_sha, burn_in_ms=500,
  measure_ms=8000, status, scientific_end_reason, observables_path,
  observables_sha256.

NPZ (``C1_OBSERVABLES_SCHEMA``):
  phasec1_observables_schema, bin_ms, E_rate_grid, I_rate_grid,
  source_rate_hz, rest_mask, active_area_fraction, kymograph, axis_positions,
  plus the complete hierarchical sufficient-statistic schema consumed by
  ``analyze_topic4_zm_phasec0._load_hierarchical_npz``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.analyze_topic4_zm_phasec0 as C0  # noqa: E402
import scripts.lock_topic4_zm_phasec1_dt2_confirmation as DT2LOCK  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402
import src.topic4_zm_phasec_phenotype as P  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
PHASEC_MANIFEST = OUT / "phasec_manifest.json"
COORDINATE_MANIFESTS = {
    "dt": OUT / "phasec1_coordinate_manifest_dt.json",
    "dt2": OUT / "phasec1_coordinate_manifest_dt2.json",
}
# Compatibility alias for callers that explicitly mean the native atlas.
COORDINATE_MANIFEST = COORDINATE_MANIFESTS["dt"]
GAIN_TRIGGER_MANIFEST = OUT / "c1_gain_trigger_manifest.json"
DT2_CONFIRMATION_MANIFEST = DT2LOCK.OUTPUT_PATH
SEEDS = (1, 3, 4)
DT2_SEEDS = (1, 3)
PHASES = N.DEFAULT_PHASES
NOISES = N.DEFAULT_NOISES
RESOLUTIONS = ("dt", "dt2")

C1_BASE_PART_SCHEMA = "zm_phasec1_base_part_v1_2026-07-28"
C1_OBSERVABLES_SCHEMA = "zm_phasec1_observables_v1_2026-07-28"
C1_BASE_ATLAS_SCHEMA = "zm_phasec1_base_atlas_v1_2026-07-28"
C1_SUMMARY_SCHEMA = "zm_phasec1_summary_v1_2026-07-28"
C1_RESOLUTION_GATE_SCHEMA = "zm_phasec1_resolution_gate_v2_2026-07-29"
C1_GAIN_STATUS_SCHEMA = "zm_phasec1_conditional_gain_status_v1_2026-07-28"
C1_GAIN_TRIGGER_SCHEMA = "zm_phasec1_gain_trigger_manifest_v1_2026-07-28"
C1_DT2_CONFIRMATION_SCHEMA = DT2LOCK.SCHEMA
C1_DT2_CONFIRMATION_SUMMARY_SCHEMA = (
    "zm_phasec1_dt2_confirmation_summary_v1_2026-07-28"
)
RESOURCE_RECEIPT_INDEX_SCHEMA = C0.RESOURCE_RECEIPT_INDEX_SCHEMA
PERIODIC_PHASE_MEDIAN_REL_DIFF_MAX = 0.20
PERIODIC_PHASE_STRUCTURE_CORR_MIN = 0.80

NON_TONIC_CLASSES = (
    "periodic_non_tonic_carrier",
    "clonic_or_bursting_carrier",
)
TERMINAL_CLASSES = (
    "tonic_non_AI",
    "periodic_non_tonic_carrier",
    "clonic_or_bursting_carrier",
    "refractory_saturated",
    "hfo_like_relaxation_train",
    "rest_or_silence",
    "runaway",
)
CELL_TERMINAL_CLASSES = TERMINAL_CLASSES + (
    "spike_AI_screen_candidate",
    "balanced_AI_tonic_cell",
)
PRIMARY_POSITIVE_STATUS = "local_maturation_window"
SHELL_POSITIVE_STATUS = "maturation_candidate_in_secondary_shell"


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_sha(value):
    body = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(body).hexdigest()


def _validate_self_hash(value, *, label):
    claimed = value.get("manifest_sha256")
    body = {key: row for key, row in value.items() if key != "manifest_sha256"}
    if not isinstance(claimed, str) or _object_sha(body) != claimed:
        raise ValueError(f"{label} self-hash mismatch")


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _base_task_key(
    *, resolution, seed, tier, cell_id, phase, noise
):
    if resolution == "dt2":
        return f"dt2|s{seed}|{tier}|{cell_id}|{phase}|{noise}"
    return f"base|s{seed}|{tier}|{cell_id}|{phase}|{noise}"


def _resource_receipt_ref(path, *, manifest_sha256, task_key):
    """Return one live-validated immutable part/receipt binding."""
    path = Path(path)
    receipt_path = PRES.resource_receipt_path(path)
    valid, reason, receipt = PRES.validate_resource_receipt(
        receipt_path,
        artifact_path=path,
        artifact_root=ROOT,
        manifest_sha256=manifest_sha256,
        task_key=task_key,
    )
    if not valid or not isinstance(receipt, dict):
        raise ValueError(reason)
    part = _load_json(path)
    ref = {
        "task_key": str(task_key),
        "part_path": os.path.relpath(path, ROOT),
        "part_file_sha256": _sha256(path),
        "resource_receipt_path": os.path.relpath(receipt_path, ROOT),
        "resource_receipt_file_sha256": _sha256(receipt_path),
        "resource_receipt_sha256": receipt["receipt_sha256"],
    }
    aux_ref = part.get("observables_path")
    aux_sha = part.get("observables_sha256")
    if aux_ref is not None or aux_sha is not None:
        aux_path = Path(str(aux_ref))
        if not aux_path.is_absolute():
            aux_path = ROOT / aux_path
        if (
            not aux_path.is_file()
            or not isinstance(aux_sha, str)
            or _sha256(aux_path) != aux_sha
        ):
            raise ValueError("resource_index_aux_observables_drift")
        ref.update({
            "aux_observables_path": os.path.relpath(aux_path, ROOT),
            "aux_observables_file_sha256": aux_sha,
        })
    return ref


def _resource_receipt_failure(path, *, manifest_sha256, task_key):
    """Return a technical resource-audit failure for one production part."""
    try:
        _resource_receipt_ref(
            path,
            manifest_sha256=manifest_sha256,
            task_key=task_key,
        )
    except (OSError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def build_resource_receipt_index(tasks, *, manifest_sha256):
    """Build a canonical full-part resource index without hiding blockers."""
    entries = []
    issues = []
    seen = set()
    logical = []
    normalized = [
        (
            str(row[0]),
            row[1],
            str(row[2]) if len(row) > 2 else "unspecified",
        )
        for row in tasks
    ]
    for task_key, path, role in sorted(
        normalized, key=lambda row: (row[0], row[2])
    ):
        logical.append({"task_key": task_key, "role": role})
        path = Path(path)
        if task_key in seen:
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": "duplicate_resource_task_key",
            })
            continue
        seen.add(task_key)
        if not path.is_file():
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": "missing_part",
            })
            continue
        try:
            entries.append(_resource_receipt_ref(
                path,
                manifest_sha256=manifest_sha256,
                task_key=task_key,
            ))
        except (OSError, TypeError, ValueError) as exc:
            issues.append({
                "task_key": task_key,
                "part_path": os.path.relpath(path, ROOT),
                "reason": str(exc),
            })
    body = {
        "schema": RESOURCE_RECEIPT_INDEX_SCHEMA,
        "manifest_sha256": str(manifest_sha256),
        "status": "complete" if not issues else "incomplete",
        "expected_task_count": len(seen),
        "validated_entry_count": len(entries),
        "expected_logical_consumption_count": len(logical),
        "logical_consumptions": logical,
        "entries": entries,
        "issues": issues,
    }
    return {**body, "index_sha256": _object_sha(body)}


def merge_resource_receipt_indexes(indexes, *, manifest_sha256):
    """Merge already canonical indexes, de-duplicating shared C0 denominators."""
    tasks = {}
    logical = []
    issues = []
    for index in indexes:
        if not isinstance(index, dict):
            continue
        if (
            index.get("schema") != RESOURCE_RECEIPT_INDEX_SCHEMA
            or index.get("manifest_sha256") != manifest_sha256
        ):
            issues.append({
                "task_key": None,
                "part_path": None,
                "reason": "invalid_child_resource_receipt_index",
            })
            continue
        body = {
            key: value for key, value in index.items()
            if key != "index_sha256"
        }
        if index.get("index_sha256") != _object_sha(body):
            issues.append({
                "task_key": None,
                "part_path": None,
                "reason": "child_resource_receipt_index_self_hash_mismatch",
            })
            continue
        issues.extend(index.get("issues", []))
        logical.extend(index.get("logical_consumptions", []))
        for entry in index.get("entries", []):
            key = entry.get("task_key")
            if key in tasks and tasks[key] != entry:
                issues.append({
                    "task_key": key,
                    "part_path": entry.get("part_path"),
                    "reason": "conflicting_resource_receipt_index_entry",
                })
            else:
                tasks[key] = entry
    entries = [tasks[key] for key in sorted(tasks)]
    body = {
        "schema": RESOURCE_RECEIPT_INDEX_SCHEMA,
        "manifest_sha256": str(manifest_sha256),
        "status": "complete" if not issues else "incomplete",
        "expected_task_count": len(entries),
        "validated_entry_count": len(entries),
        "expected_logical_consumption_count": len(logical),
        "logical_consumptions": logical,
        "entries": entries,
        "issues": issues,
    }
    return {**body, "index_sha256": _object_sha(body)}


def resource_receipt_index_failure(index, *, manifest_sha256):
    """Revalidate a canonical index and every live part/receipt binding."""
    if not isinstance(index, dict):
        return "missing_resource_receipt_index"
    body = {
        key: value for key, value in index.items()
        if key != "index_sha256"
    }
    if (
        index.get("schema") != RESOURCE_RECEIPT_INDEX_SCHEMA
        or index.get("manifest_sha256") != manifest_sha256
        or index.get("index_sha256") != _object_sha(body)
        or index.get("status") != "complete"
        or index.get("issues") != []
    ):
        return "invalid_or_incomplete_resource_receipt_index"
    entries = index.get("entries")
    logical = index.get("logical_consumptions")
    if (
        not isinstance(entries, list)
        or index.get("expected_task_count") != len(entries)
        or index.get("validated_entry_count") != len(entries)
        or not isinstance(logical, list)
        or index.get("expected_logical_consumption_count") != len(logical)
    ):
        return "resource_receipt_index_count_mismatch"
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            return "invalid_resource_receipt_index_entry"
        task_key = entry.get("task_key")
        if not isinstance(task_key, str) or task_key in seen:
            return "duplicate_or_invalid_resource_receipt_task"
        seen.add(task_key)
        part = ROOT / str(entry.get("part_path", ""))
        try:
            current = _resource_receipt_ref(
                part,
                manifest_sha256=manifest_sha256,
                task_key=task_key,
            )
        except (OSError, TypeError, ValueError) as exc:
            return f"resource_receipt_index_live_failure:{exc}"
        if current != entry:
            return "resource_receipt_index_live_binding_mismatch"
    if any(
        not isinstance(row, dict)
        or row.get("task_key") not in seen
        or not isinstance(row.get("role"), str)
        for row in logical
    ):
        return "resource_receipt_index_logical_consumption_mismatch"
    return None


def _coordinate_path_from_final(phasec_manifest, resolution):
    """Resolve one resolution-local coordinate lock from the final manifest."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown coordinate resolution: {resolution}")
    refs = phasec_manifest.get("c1", {}).get("coordinate_manifests")
    if not isinstance(refs, dict) or resolution not in refs:
        raise ValueError(
            f"final Phase-C manifest lacks {resolution} coordinate lock"
        )
    ref = refs[resolution]
    required = ("path", "file_sha256", "manifest_sha256", "semantic_sha256")
    if any(not isinstance(ref.get(key), str) or not ref[key] for key in required):
        raise ValueError(
            f"final Phase-C {resolution} coordinate lock is incomplete"
        )
    path = ROOT / ref["path"]
    if not path.is_file() or _sha256(path) != ref["file_sha256"]:
        raise ValueError(
            f"final Phase-C {resolution} coordinate file hash mismatch"
        )
    coordinate = _load_json(path)
    _validate_self_hash(coordinate, label=f"C1 {resolution} coordinate manifest")
    semantic_body = {
        key: value for key, value in coordinate.items()
        if key not in {"manifest_sha256", "semantic_sha256"}
    }
    if (
        coordinate.get("manifest_sha256") != ref["manifest_sha256"]
        or coordinate.get("semantic_sha256") != ref["semantic_sha256"]
        or _object_sha(semantic_body) != ref["semantic_sha256"]
        or coordinate.get("resolution") != resolution
    ):
        raise ValueError(
            f"final Phase-C {resolution} coordinate semantic mismatch"
        )
    return path, coordinate, ref


def _resolution_seed_inputs(
    phasec_manifest, *, resolution, seed, phase, noise
):
    """Return the exact fast-state/config/noise identity for one C1 arm."""
    native = phasec_manifest.get("per_seed", {}).get(str(int(seed)))
    if not isinstance(native, dict):
        raise ValueError(f"final Phase-C manifest lacks seed {seed}")
    if resolution == "dt":
        row = native
        config_sha = row.get("canonical_config_sha")
    elif resolution == "dt2":
        row = native.get("resolution_confirmations", {}).get("dt2")
        if not isinstance(row, dict):
            raise ValueError(f"seed {seed} lacks locked dt2 inputs")
        config_sha = row.get("config_sha")
    else:
        raise ValueError(f"unknown resolution: {resolution}")
    family = row.get("c0_carrier_states", {}).get(phase)
    if not isinstance(family, dict) or not isinstance(family.get("state"), dict):
        raise ValueError(
            f"{resolution}/seed{seed}/{phase} lacks fast base state"
        )
    state = family["state"]
    banks = {
        item.get("replicate"): item
        for item in family.get("noise_banks", [])
        if isinstance(item, dict)
    }
    bank = banks.get(noise)
    if not isinstance(bank, dict):
        raise ValueError(
            f"{resolution}/seed{seed}/{phase} lacks noise bank {noise}"
        )
    expected = {
        "config_sha": config_sha,
        "fast_base_state_hash": state.get("state_hash"),
        "state_file_sha256": state.get("file_sha256"),
        "noise_bank_sha": bank.get("bank_sha"),
    }
    if any(not isinstance(value, str) or len(value) != 64
           for value in expected.values()):
        raise ValueError(
            f"{resolution}/seed{seed}/{phase}/{noise} fast-input lock invalid"
        )
    return expected


def _resolution_preentry_inputs(
    phasec_manifest, *, resolution, seed, noise
):
    """Return exact matched C0 pre-entry denominator identity."""
    native = phasec_manifest.get("per_seed", {}).get(str(int(seed)))
    if not isinstance(native, dict):
        raise ValueError(f"final Phase-C manifest lacks seed {seed}")
    if resolution == "dt":
        row = native
        config_sha = row.get("canonical_config_sha")
    elif resolution == "dt2":
        row = native.get("resolution_confirmations", {}).get("dt2")
        if not isinstance(row, dict):
            raise ValueError(f"seed {seed} lacks locked dt2 inputs")
        config_sha = row.get("config_sha")
    else:
        raise ValueError(f"unknown resolution: {resolution}")
    family = row.get("c0_pre_entry_gain_control")
    if not isinstance(family, dict) or not isinstance(family.get("state"), dict):
        raise ValueError(
            f"{resolution}/seed{seed} lacks pre-entry gain control"
        )
    banks = {
        item.get("replicate"): item
        for item in family.get("noise_banks", [])
        if isinstance(item, dict)
    }
    bank = banks.get(noise)
    state = family["state"]
    expected = {
        "config_sha": config_sha,
        "fast_base_state_hash": state.get("state_hash"),
        "state_file_sha256": state.get("file_sha256"),
        "noise_bank_sha": (
            None if not isinstance(bank, dict) else bank.get("bank_sha")
        ),
    }
    if any(not isinstance(value, str) or len(value) != 64
           for value in expected.values()):
        raise ValueError(
            f"{resolution}/seed{seed}/{noise} pre-entry lock invalid"
        )
    return expected


def _coordinate_seed_provenance(coordinate, *, seed):
    row = coordinate.get("seeds", {}).get(str(int(seed)))
    if not isinstance(row, dict):
        raise ValueError(f"coordinate manifest lacks seed {seed}")
    required = {
        "coordinate_npz_file_sha256": row.get(
            "npz_file_sha256", row.get("npz_sha256")
        ),
        "coordinate_npz_semantic_sha256": row.get("npz_semantic_sha256"),
    }
    if any(not isinstance(value, str) or len(value) != 64
           for value in required.values()):
        raise ValueError(f"coordinate seed {seed} NPZ provenance incomplete")
    return required


def _expected_runtime_provenance(
    phasec_manifest,
    *,
    phasec_manifest_file_sha256,
    coordinate,
    coordinate_ref,
    coordinate_seed,
    fast_inputs,
    trigger=None,
    trigger_file_sha256=None,
):
    """Canonical runtime provenance expected from every C1 production part."""
    expected = {
        "manifest_sha256": phasec_manifest["manifest_sha256"],
        "manifest_file_sha256": phasec_manifest_file_sha256,
        "producer_sha256": phasec_manifest.get("provenance", {}).get(
            "producer_file_sha256"
        ),
        "state_file_sha256": fast_inputs["state_file_sha256"],
        "noise_bank_sha": fast_inputs["noise_bank_sha"],
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate["semantic_sha256"],
        "coordinate_manifest_file_sha256": coordinate_ref["file_sha256"],
        "coordinate_npz_file_sha256": coordinate_seed[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": coordinate_seed[
            "coordinate_npz_semantic_sha256"
        ],
        "coordinate_producer_sha256": coordinate.get(
            "producer_file_sha256"
        ),
    }
    if trigger is not None:
        expected.update({
            "trigger_manifest_sha256": trigger["manifest_sha256"],
            "trigger_manifest_file_sha256": trigger_file_sha256,
            "trigger_producer_sha256": trigger.get(
                "producer_file_sha256"
            ),
        })
    if not isinstance(expected["producer_sha256"], dict):
        raise ValueError("final Phase-C producer map is missing")
    if not isinstance(expected["coordinate_producer_sha256"], dict):
        raise ValueError("coordinate producer map is missing")
    if trigger is not None and not isinstance(
        expected["trigger_producer_sha256"], dict
    ):
        raise ValueError("trigger producer map is missing")
    return expected


def base_part_path(resolution, seed, tier, cell_id, phase, noise):
    return (
        OUT / "parts/c1_base" / resolution / f"seed{seed}" / tier / cell_id
        / phase / noise / "phenotype.json"
    )


def gain_status_path(resolution, seed, tier, cell_id):
    return (
        OUT / "parts/c1_conditional_gain" / resolution / f"seed{seed}"
        / tier / cell_id / "gain_status.json"
    )


def _npz_scalar(value):
    x = np.asarray(value)
    if x.size != 1:
        raise ValueError("NPZ scalar expected")
    return x.reshape(()).item()


def _load_phenotype_arrays(part):
    path_value = part.get("observables_path")
    if not isinstance(path_value, str) or not path_value:
        return {"status": "blocked", "reason": "missing_observables_path"}
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {"status": "blocked", "reason": "missing_observables_npz"}
    if part.get("observables_sha256") != _sha256(path):
        return {"status": "blocked", "reason": "observables_sha256_mismatch"}
    required = (
        "phasec1_observables_schema",
        "bin_ms",
        "E_rate_grid",
        "I_rate_grid",
        "source_rate_hz",
        "rest_mask",
        "active_area_fraction",
        "kymograph",
        "axis_positions",
        "readout_kernel_width_mm",
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data.files]
            if missing:
                return {
                    "status": "blocked",
                    "reason": "missing_phenotype_npz_fields:" + ",".join(missing),
                }
            if str(_npz_scalar(data["phasec1_observables_schema"])) != (
                C1_OBSERVABLES_SCHEMA
            ):
                return {
                    "status": "blocked",
                    "reason": "phasec1_observables_schema_mismatch",
                }
            arrays = {
                "bin_ms": float(_npz_scalar(data["bin_ms"])),
                "E_rate_grid": np.asarray(data["E_rate_grid"], float),
                "I_rate_grid": np.asarray(data["I_rate_grid"], float),
                "source_rate_hz": np.asarray(data["source_rate_hz"], float),
                "rest_mask": np.asarray(data["rest_mask"], bool),
                "active_area_fraction": np.asarray(
                    data["active_area_fraction"], float
                ),
                "kymograph": np.asarray(data["kymograph"], float),
                "axis_positions": np.asarray(data["axis_positions"], float),
                "readout_kernel_width_mm": float(
                    _npz_scalar(data["readout_kernel_width_mm"])
                ),
            }
    except (OSError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": f"invalid_observables_npz:{exc}"}
    return {"status": "ok", "path": path, **arrays}


def spike_ai_screen(hierarchical, *, terminal_class):
    """Apply the C0 spike-only AI conjunction to one C1 continuation."""
    if terminal_class != "tonic_non_AI":
        return {
            "pass": False,
            "reason": "population_envelope_not_tonic",
            "metrics": None,
        }
    try:
        values = C0._continuation_point(  # shared locked C0 definitions
            {
                "hierarchical": hierarchical,
                # Gain is deliberately absent from this pre-trigger screen.
                "gain_ratio_samples": np.asarray([1.0]),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "pass": None,
            "reason": f"invalid_spike_statistics:{exc}",
            "metrics": None,
        }
    spike_keys = (
        "rho80_active_core",
        "isi_cv2_median",
        "pairwise_observed_median",
        "pairwise_null_q97_5",
        "pairwise_stratum_max_excess",
        "active_area_fraction",
    )
    if not np.all(np.isfinite([values[key] for key in spike_keys])):
        return {
            "pass": None,
            "reason": "nonfinite_spike_statistics",
            "metrics": {key: values[key] for key in spike_keys},
        }
    passed = bool(
        values["rho80_active_core"] <= 0.20
        and values["isi_cv2_median"] >= 0.70
        and abs(values["pairwise_observed_median"]) < 0.10
        and values["pairwise_stratum_max_excess"] < 0.0
        and values["active_area_fraction"] < 0.50
    )
    return {
        "pass": passed,
        "reason": "all_spike_AI_conditions" if passed else "spike_AI_gate_failed",
        "metrics": {key: values[key] for key in spike_keys},
    }


def _scientific_terminal(reason):
    return {
        "runaway": "runaway",
        # A whole-sheet early stop carries no saved spike statistics, so it
        # cannot satisfy the registered rho80+refractory double threshold.
        "whole_sheet_plateau": "probabilistically_indeterminate",
        "empirical_rest_dwell": "hfo_like_relaxation_train",
    }.get(reason)


def classify_base_part(
    path,
    *,
    coordinate,
    coordinate_manifest,
    phasec_manifest,
    panels,
    seed,
    phase,
    noise,
    resolution,
    phasec_manifest_file_sha256=None,
    coordinate_manifest_file_sha256=None,
    coordinate_ref=None,
    dt2_confirmation_manifest_sha256=None,
    dt2_confirmation_manifest_file_sha256=None,
):
    """Validate and classify one expected C1 base part."""
    path = Path(path)
    evidence = {
        "part_path": _relative(path),
        "part_sha256": _sha256(path) if path.is_file() else None,
    }
    if not path.is_file():
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": "missing_base_part",
            **evidence,
        }
    try:
        part = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": f"invalid_base_part_json:{exc}",
            **evidence,
        }
    fast_inputs = _resolution_seed_inputs(
        phasec_manifest,
        resolution=resolution,
        seed=seed,
        phase=phase,
        noise=noise,
    )
    coordinate_seed = _coordinate_seed_provenance(
        coordinate_manifest, seed=seed
    )
    expected = {
        "schema": C1_BASE_PART_SCHEMA,
        "phasec_manifest_sha256": phasec_manifest["manifest_sha256"],
        "coordinate_manifest_sha256": coordinate_manifest["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate_manifest[
            "semantic_sha256"
        ],
        "seed": int(seed),
        "cell_id": coordinate["cell_id"],
        "tier": coordinate["tier"],
        "trajectory_id": coordinate["trajectory_id"],
        "path_index": int(coordinate["path_index"]),
        "path_direction": coordinate["path_direction"],
        "phase": phase,
        "noise": noise,
        "resolution": resolution,
        "slow_state_sha256": coordinate["state_sha256"],
        "coordinate_npz_file_sha256": coordinate_seed[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": coordinate_seed[
            "coordinate_npz_semantic_sha256"
        ],
        **fast_inputs,
        "burn_in_ms": 500.0,
        "measure_ms": 8000.0,
    }
    if phasec_manifest_file_sha256 is not None:
        expected["phasec_manifest_file_sha256"] = (
            phasec_manifest_file_sha256
        )
    if coordinate_manifest_file_sha256 is not None:
        expected["coordinate_manifest_file_sha256"] = (
            coordinate_manifest_file_sha256
        )
    if dt2_confirmation_manifest_sha256 is not None:
        expected["dt2_confirmation_manifest_sha256"] = (
            dt2_confirmation_manifest_sha256
        )
        expected["dt2_confirmation_manifest_file_sha256"] = (
            dt2_confirmation_manifest_file_sha256
        )
    for key, wanted in expected.items():
        got = part.get(key)
        if isinstance(wanted, float):
            valid = got is not None and np.isclose(float(got), wanted)
        else:
            valid = got == wanted
        if not valid:
            return {
                "status": "blocked",
                "terminal_class": "missing",
                "reason": f"base_part_field_mismatch:{key}",
                **evidence,
            }
    if coordinate_ref is None:
        coordinate_ref = {
            "file_sha256": coordinate_manifest_file_sha256,
            "manifest_sha256": coordinate_manifest.get("manifest_sha256"),
            "semantic_sha256": coordinate_manifest.get("semantic_sha256"),
        }
    try:
        expected_runtime = _expected_runtime_provenance(
            phasec_manifest,
            phasec_manifest_file_sha256=phasec_manifest_file_sha256,
            coordinate=coordinate_manifest,
            coordinate_ref=coordinate_ref,
            coordinate_seed=coordinate_seed,
            fast_inputs=fast_inputs,
        )
    except ValueError as exc:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": f"invalid_locked_runtime_provenance:{exc}",
            **evidence,
        }
    actual_runtime = part.get("runtime_provenance")
    if dt2_confirmation_manifest_sha256 is not None:
        expected_runtime.update({
            "dt2_confirmation_manifest_sha256": (
                dt2_confirmation_manifest_sha256
            ),
            "dt2_confirmation_manifest_file_sha256": (
                dt2_confirmation_manifest_file_sha256
            ),
        })
    runtime_mismatch = (
        [
            key for key, wanted in expected_runtime.items()
            if not isinstance(actual_runtime, dict)
            or actual_runtime.get(key) != wanted
        ]
    )
    if runtime_mismatch:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": (
                "runtime_provenance_mismatch:"
                + ",".join(runtime_mismatch)
            ),
            **evidence,
        }
    receipt_failure = _resource_receipt_failure(
        path,
        manifest_sha256=phasec_manifest["manifest_sha256"],
        task_key=_base_task_key(
            resolution=resolution,
            seed=seed,
            tier=coordinate["tier"],
            cell_id=coordinate["cell_id"],
            phase=phase,
            noise=noise,
        ),
    )
    if receipt_failure is not None:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": receipt_failure,
            **evidence,
        }
    if part.get("status") != "complete":
        terminal = _scientific_terminal(part.get("scientific_end_reason"))
        if terminal is None:
            return {
                "status": "blocked",
                "terminal_class": "missing",
                "reason": (
                    "technical_or_unregistered_end:"
                    + str(part.get("technical_end_reason"))
                ),
                **evidence,
            }
        return {
            "status": "complete",
            "terminal_class": terminal,
            "reason": str(part.get("scientific_end_reason")),
            "spike_ai_screen": {"pass": False, "reason": "scientific_end"},
            "spatial_relay": {"is_spatial_relay": False},
            "locked_arm_identity": expected,
            **evidence,
        }

    phenotype_arrays = _load_phenotype_arrays(part)
    if phenotype_arrays["status"] != "ok":
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": phenotype_arrays["reason"],
            **evidence,
        }
    hierarchical = C0._load_hierarchical_npz(
        part, expected_panel=panels["seeds"][str(seed)]
    )
    if hierarchical["status"] != "ok":
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": hierarchical["reason"],
            **evidence,
        }
    try:
        refractory_fraction = C0._pooled_refractory_isi_probability(
            hierarchical[
                "block_refractory_isi_numerator_by_stratum"
            ],
            hierarchical[
                "block_refractory_isi_denominator_by_stratum"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": f"invalid_refractory_isi_counts:{exc}",
            **evidence,
        }
    if not np.isfinite(refractory_fraction):
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": "missing_refractory_isi_counts",
            **evidence,
        }
    try:
        result = P.classify_phasec_run(
            phenotype_arrays["E_rate_grid"],
            phenotype_arrays["I_rate_grid"],
            bin_ms=phenotype_arrays["bin_ms"],
            source_rate_hz=phenotype_arrays["source_rate_hz"],
            rest_mask=phenotype_arrays["rest_mask"],
            active_area_fraction=phenotype_arrays["active_area_fraction"],
            kymograph=phenotype_arrays["kymograph"],
            axis_positions=phenotype_arrays["axis_positions"],
            readout_kernel_width_mm=phenotype_arrays[
                "readout_kernel_width_mm"
            ],
            runaway_early_stop_ms=part.get("runaway_early_stop_ms"),
            saturation_fraction=part.get("saturation_fraction"),
            refractory_fraction=refractory_fraction,
            relay_rng_seed=int(seed),
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": f"phenotype_classifier_error:{exc}",
            **evidence,
        }
    terminal = result["phenotype"]
    spike = spike_ai_screen(hierarchical, terminal_class=terminal)
    if spike["pass"] is None:
        return {
            "status": "blocked",
            "terminal_class": "missing",
            "reason": spike["reason"],
            **evidence,
        }
    return {
        "status": "complete",
        "terminal_class": terminal,
        "reason": "classified",
        "phenotype": result,
        "spike_ai_screen": spike,
        "_hierarchical": hierarchical,
        "spatial_relay": result["spatial_relay"],
        "locked_arm_identity": expected,
        **evidence,
    }


def _support(run_rows, pass_fn):
    rows = []
    for row in run_rows:
        rows.append({
            "phase": row["phase"],
            "noise": row["noise"],
            "status": row["status"],
            "cell_pass": (
                None if row["status"] != "complete" else bool(pass_fn(row))
            ),
            "maturation_direction": row.get("path_direction"),
        })
    out = N.aggregate_cell(rows, pass_key="cell_pass", required_successes=5)
    counts = {
        phase: sum(
            row["status"] == "complete" and bool(pass_fn(row))
            for row in run_rows if row["phase"] == phase
        )
        for phase in PHASES
    }
    out["per_phase_pass_count"] = counts
    out["passes_locked_cell_gate"] = bool(
        out["status"] == "pass"
        and out["posterior_median"] > 0.80
        and all(counts[phase] >= 2 for phase in PHASES)
    )
    return out


def _relay_support(run_rows):
    base = _support(
        run_rows,
        lambda row: row.get("spatial_relay", {}).get(
            "is_spatial_relay"
        ) is True,
    )
    signs = [
        int(row.get("spatial_relay", {}).get("direction_sign", 0))
        for row in run_rows
        if row["status"] == "complete"
        and row.get("spatial_relay", {}).get("is_spatial_relay") is True
    ]
    nonzero = [sign for sign in signs if sign in (-1, 1)]
    count = Counter(nonzero)
    direction_sign, n_same = (
        count.most_common(1)[0] if count else (None, 0)
    )
    base.update({
        "direction_sign": direction_sign,
        "same_direction_pass_count": int(n_same),
        "supported": bool(
            base["passes_locked_cell_gate"] and n_same >= 5
        ),
    })
    return base


def _periodic_phase_consistency(run_rows):
    """Require period and axial source-phase structure across fast phases."""
    per_phase = {}
    for phase in PHASES:
        periods = []
        signatures = []
        for row in run_rows:
            if (
                row.get("status") != "complete"
                or row.get("phase") != phase
                or row.get("terminal_class")
                != "periodic_non_tonic_carrier"
            ):
                continue
            period = (
                row.get("phenotype", {})
                .get("temporal_diagnostics", {})
                .get("periodic", {})
                .get("median_period_ms")
            )
            if period is not None and np.isfinite(float(period)):
                signature = (
                    row.get("phenotype", {})
                    .get("temporal_diagnostics", {})
                    .get("periodic", {})
                    .get("source_phase_signature", {})
                )
                profile = np.asarray(signature.get("profile"), float)
                if (
                    signature.get("status") == "ok"
                    and profile.ndim == 2
                    and profile.shape[0] >= 4
                    and profile.shape[1] >= 2
                    and np.all(np.isfinite(profile))
                    and np.linalg.norm(profile) > 0
                ):
                    periods.append(float(period))
                    signatures.append(profile)
        per_phase[phase] = {
            "periods": periods,
            "signatures": signatures,
        }
    if any(len(per_phase[phase]["periods"]) < 2 for phase in PHASES):
        return {
            "pass": False,
            "reason": (
                "fewer_than_two_period_and_source_phase_estimates_per_fast_phase"
            ),
            "per_phase_median_period_ms": {
                phase: (
                    float(np.median(row["periods"]))
                    if row["periods"] else None
                )
                for phase, row in per_phase.items()
            },
            "relative_difference": None,
            "maximum_relative_difference": (
                PERIODIC_PHASE_MEDIAN_REL_DIFF_MAX
            ),
            "source_phase_similarity_median": None,
            "minimum_source_phase_similarity": (
                PERIODIC_PHASE_STRUCTURE_CORR_MIN
            ),
        }
    medians = {
        phase: float(np.median(row["periods"]))
        for phase, row in per_phase.items()
    }
    denominator = 0.5 * sum(medians.values())
    relative = (
        np.inf if denominator <= 0
        else abs(medians[PHASES[0]] - medians[PHASES[1]]) / denominator
    )
    similarities = []
    for left in per_phase[PHASES[0]]["signatures"]:
        for right in per_phase[PHASES[1]]["signatures"]:
            if left.shape != right.shape:
                continue
            shifted = []
            for phase_shift in range(left.shape[0]):
                candidate = np.roll(right, phase_shift, axis=0)
                correlation = float(
                    np.corrcoef(left.ravel(), candidate.ravel())[0, 1]
                )
                if np.isfinite(correlation):
                    shifted.append(correlation)
            if shifted:
                similarities.append(max(shifted))
    source_similarity = (
        float(np.median(similarities)) if similarities else np.nan
    )
    period_pass = relative <= PERIODIC_PHASE_MEDIAN_REL_DIFF_MAX
    structure_pass = (
        np.isfinite(source_similarity)
        and source_similarity >= PERIODIC_PHASE_STRUCTURE_CORR_MIN
    )
    passed = bool(period_pass and structure_pass)
    return {
        "pass": passed,
        "reason": (
            "fast_phase_period_and_source_structure_concordant"
            if passed else (
                "fast_phase_periods_differ"
                if not period_pass else "fast_phase_source_structure_differs"
            )
        ),
        "per_phase_median_period_ms": medians,
        "relative_difference": float(relative),
        "maximum_relative_difference": PERIODIC_PHASE_MEDIAN_REL_DIFF_MAX,
        "source_phase_similarity_median": (
            float(source_similarity)
            if np.isfinite(source_similarity) else None
        ),
        "minimum_source_phase_similarity": (
            PERIODIC_PHASE_STRUCTURE_CORR_MIN
        ),
        "source_phase_similarity_n_pairs": len(similarities),
        "source_phase_similarity_alignment": "maximum_circular_phase_shift",
    }


def aggregate_cell_rows(run_rows, coordinate):
    """Assign one fail-closed cell class from exactly 2x3 base runs."""
    common = {
        "seed": (
            int(run_rows[0]["seed"])
            if run_rows else int(coordinate["seed"])
        ),
        "cell_id": coordinate["cell_id"],
        "tier": coordinate["tier"],
        "trajectory_id": coordinate["trajectory_id"],
        "path_index": int(coordinate["path_index"]),
        "path_direction": coordinate["path_direction"],
        "slow_state_sha256": coordinate["state_sha256"],
    }
    if coordinate["status"] != "valid":
        return {
            **common,
            "status": "invalid_physical",
            "cell_class": "invalid_physical_cell",
            "gain_trigger_eligible": False,
            "run_rows": [],
        }
    terminal_support = {
        label: _support(
            run_rows, lambda row, label=label: row["terminal_class"] == label
        )
        for label in TERMINAL_CLASSES
    }
    periodic_consistency = _periodic_phase_consistency(run_rows)
    if (
        terminal_support["periodic_non_tonic_carrier"][
            "passes_locked_cell_gate"
        ]
        and not periodic_consistency["pass"]
    ):
        terminal_support["periodic_non_tonic_carrier"][
            "passes_locked_cell_gate"
        ] = False
        terminal_support["periodic_non_tonic_carrier"][
            "phase_consistency_blocked"
        ] = True
    spike_support = _support(
        run_rows,
        lambda row: row.get("spike_ai_screen", {}).get("pass") is True,
    )
    spike_ci = None
    spike_ci_pass = False
    if spike_support["passes_locked_cell_gate"]:
        tonic_numeric = [
            {
                "phase": f"bounded_mid__{row['phase']}",
                "hierarchical": row.get("_hierarchical"),
                # Gain is intentionally tested only after this spike-only
                # trigger; a neutral finite placeholder is required by the
                # shared hierarchical bootstrap implementation.
                "gain_ratio_samples": np.asarray([1.0]),
            }
            for row in run_rows
            if row.get("status") == "complete"
            and row.get("terminal_class") == "tonic_non_AI"
            and isinstance(row.get("_hierarchical"), dict)
        ]
        try:
            spike_ci = C0.hierarchical_seed_bootstrap(
                tonic_numeric,
                seed=(
                    int(common["seed"]) * 1009
                    + int(common["path_index"]) * 17
                ),
                n_boot=C0.N_BOOT,
            )
        except (KeyError, TypeError, ValueError) as exc:
            spike_ci = {"status": "blocked", "reason": str(exc)}
        if spike_ci.get("status") != "blocked":
            spike_ci_pass = bool(
                spike_ci["rho80_active_core"]["hi"] is not None
                and spike_ci["rho80_active_core"]["hi"] <= 0.20
                and spike_ci["isi_cv2_median"]["lo"] is not None
                and spike_ci["isi_cv2_median"]["lo"] >= 0.70
                and spike_ci["pairwise_observed_median"]["point"] is not None
                and abs(
                    spike_ci["pairwise_observed_median"]["point"]
                ) < 0.10
                and spike_ci["pairwise_stratum_max_excess"]["hi"] is not None
                and spike_ci["pairwise_stratum_max_excess"]["hi"] < 0.0
                and spike_ci["active_area_fraction"]["hi"] is not None
                and spike_ci["active_area_fraction"]["hi"] < 0.50
            )
    non_tonic = [
        label for label in NON_TONIC_CLASSES
        if terminal_support[label]["passes_locked_cell_gate"]
    ]
    terminal = [
        label for label in TERMINAL_CLASSES
        if terminal_support[label]["passes_locked_cell_gate"]
    ]
    if any(row["status"] != "complete" for row in run_rows) or len(run_rows) != 6:
        status, cell_class = "blocked", "missing"
    elif len(non_tonic) == 1:
        status, cell_class = "complete", non_tonic[0]
    elif spike_support["passes_locked_cell_gate"] and spike_ci_pass:
        status, cell_class = "complete", "spike_AI_screen_candidate"
    elif len(terminal) == 1:
        status, cell_class = "complete", terminal[0]
    else:
        status, cell_class = "indeterminate", "probabilistically_indeterminate"
    relay_support = _relay_support(run_rows)
    return {
        **common,
        "status": status,
        "cell_class": cell_class,
        "gain_trigger_eligible": bool(
            cell_class == "spike_AI_screen_candidate"
        ),
        "terminal_support": terminal_support,
        "periodic_fast_phase_consistency": periodic_consistency,
        "spike_ai_screen_support": {
            **spike_support,
            "hierarchical_ci_pass": spike_ci_pass,
        },
        "spike_ai_hierarchical_ci": spike_ci,
        "spatial_relay_modifier": {
            "supported": relay_support["supported"],
            "support": relay_support,
        },
        "run_rows": [
            {
                key: value for key, value in row.items()
                if key != "_hierarchical"
            }
            for row in run_rows
        ],
    }


def _adjacent_windows(cells, phenotype):
    selected = sorted(
        (
            row for row in cells
            if row.get("status") == "complete"
            and row.get("cell_class") == phenotype
        ),
        key=lambda row: (
            str(row.get("trajectory_id")),
            int(row.get("path_index", -1)),
        ),
    )
    out = []
    for left, right in zip(selected[:-1], selected[1:]):
        if left["trajectory_id"] != right["trajectory_id"]:
            continue
        if int(right["path_index"]) != int(left["path_index"]) + 1:
            continue
        if left["path_direction"] != right["path_direction"]:
            continue
        out.append({
            "phenotype": phenotype,
            "trajectory_id": left["trajectory_id"],
            "direction": left["path_direction"],
            "cells": [left["cell_id"], right["cell_id"]],
        })
    return out


def _primary_third_seed_compatibility(
    cells, *, phenotype, direction, supporting_windows,
):
    """Fail closed on the non-majority seed for a primary C1 window.

    A two-seed window is not allowed to overrule a contrary third-seed
    *window*.  The comparison is made at the homologous adjacent cells of the
    supporting window, rather than over all ten primary cells: tonic cells on
    an unrelated trajectory are not evidence against the candidate.  A
    non-supporting third seed is admissible only when every cell in one
    homologous pair is either matching-complete or explicitly
    probabilistically indeterminate, with at least one indeterminate cell.
    Any non-tonic window with a different phenotype or aligned direction is
    an opposite seed-level outcome and blocks the majority.
    """
    cells = list(cells)
    if not cells:
        return {
            "compatible": False,
            "disposition": "missing",
            "reason": "third_seed_has_no_cell_evidence",
        }
    third_windows = [
        window
        for label in NON_TONIC_CLASSES
        for window in _adjacent_windows(cells, label)
    ]
    opposite = [
        window for window in third_windows
        if window["phenotype"] != phenotype
        or window["direction"] != direction
    ]
    if opposite:
        return {
            "compatible": False,
            "disposition": "opposite_window",
            "reason": (
                "third_seed_has_different_phenotype_or_direction_window"
            ),
        }
    by_id = {row["cell_id"]: row for row in cells}
    homologous_pairs = {
        tuple(window["cells"])
        for window in supporting_windows
        if window["phenotype"] == phenotype
        and window["direction"] == direction
    }
    for pair in sorted(homologous_pairs):
        pair_rows = [by_id.get(cell_id) for cell_id in pair]
        explicit_indeterminate = [
            row is not None
            and row.get("status") == "indeterminate"
            and row.get("cell_class")
            == "probabilistically_indeterminate"
            for row in pair_rows
        ]
        matching_complete = [
            row is not None
            and row.get("status") == "complete"
            and row.get("cell_class") == phenotype
            and row.get("path_direction") == direction
            for row in pair_rows
        ]
        if (
            all(
                is_matching or is_indeterminate
                for is_matching, is_indeterminate in zip(
                    matching_complete, explicit_indeterminate
                )
            )
            and any(explicit_indeterminate)
        ):
            return {
                "compatible": True,
                "disposition": "probabilistically_indeterminate",
                "reason": (
                    "third_seed_homologous_window_is_concordant_or_explicitly_"
                    "probabilistically_indeterminate"
                ),
                "homologous_cells": list(pair),
            }
    return {
        "compatible": False,
        "disposition": "contradictory_or_unresolved_homologous_window",
        "reason": (
            "third_seed_homologous_window_is_not_concordant_or_explicitly_"
            "probabilistically_indeterminate"
        ),
    }


def adjudicate_tier(cell_rows, tier):
    """Adjudicate same-phenotype adjacency and cross-seed direction."""
    rows = [row for row in cell_rows if row["tier"] == tier]
    if tier == "secondary_shell":
        seed_rows = {}
        for seed in SEEDS:
            cells = [row for row in rows if row["seed"] == seed]
            scientific_classes = {
                row.get("cell_class") for row in cells
                if row.get("status") == "complete"
            }
            seed_rows[str(seed)] = {
                "windows": [],
                "isolated_non_tonic_cells": [
                    row["cell_id"] for row in cells
                    if row.get("cell_class") in NON_TONIC_CLASSES
                ],
                "saturation_runaway_only": bool(
                    scientific_classes
                    and scientific_classes
                    <= {"refractory_saturated", "runaway"}
                ),
                "cell_status_counts": dict(
                    Counter(row["status"] for row in cells)
                ),
            }
        candidates = []
        for cell_id in sorted({
            row["cell_id"] for row in rows
            if row.get("cell_class") in NON_TONIC_CLASSES
        }):
            for phenotype in NON_TONIC_CLASSES:
                by_seed = {
                    seed: next((
                        row for row in rows
                        if row["seed"] == seed and row["cell_id"] == cell_id
                    ), None)
                    for seed in SEEDS
                }
                supporting = [
                    seed for seed, row in by_seed.items()
                    if row is not None
                    and row.get("status") == "complete"
                    and row.get("cell_class") == phenotype
                ]
                if len(supporting) < 2:
                    continue
                non_supporting = [
                    row for seed, row in by_seed.items()
                    if seed not in supporting
                ]
                third_is_indeterminate = all(
                    row is not None
                    and row.get("status") == "indeterminate"
                    and row.get("cell_class")
                    == "probabilistically_indeterminate"
                    for row in non_supporting
                )
                if non_supporting and not third_is_indeterminate:
                    continue
                candidate = {
                    "phenotype": phenotype,
                    # Shell points are not an ordered direction; the complete
                    # locked cell ID encodes basis direction and sign.
                    "direction": cell_id,
                    "cell_id": cell_id,
                    "homologous_cells": [cell_id],
                    "supporting_seeds": supporting,
                    "dt2_eligible": set(DT2_SEEDS).issubset(supporting),
                }
                candidates.append(candidate)
                for seed in supporting:
                    seed_rows[str(seed)]["windows"].append({
                        "phenotype": phenotype,
                        "direction": cell_id,
                        "cells": [cell_id],
                    })
                    seed_rows[str(seed)]["isolated_non_tonic_cells"] = [
                        value for value in seed_rows[str(seed)][
                            "isolated_non_tonic_cells"
                        ] if value != cell_id
                    ]
        any_isolated = any(
            row["isolated_non_tonic_cells"] for row in seed_rows.values()
        )
        eligible_candidates = [
            row for row in candidates if row["dt2_eligible"]
        ]
        if eligible_candidates:
            status = SHELL_POSITIVE_STATUS
        elif candidates:
            status = "resolution_confirmation_unavailable"
        elif any_isolated:
            status = "isolated_maturation_candidate"
        else:
            status = "no_window"
        return {
            "tier": tier,
            "status": status,
            "candidates": candidates,
            "dt2_eligible_candidates": eligible_candidates,
            "seed_results": seed_rows,
            "acceptance_semantics": (
                "same_locked_shell_cell_and_phenotype_across_at_least_2_of_3_seeds"
            ),
            "primary_reachability_established": False,
        }
    seed_rows = {}
    for seed in SEEDS:
        cells = [row for row in rows if row["seed"] == seed]
        windows = [
            window
            for label in NON_TONIC_CLASSES
            for window in _adjacent_windows(cells, label)
        ]
        isolated = [
            row["cell_id"] for row in cells
            if row.get("cell_class") in NON_TONIC_CLASSES
            and not any(row["cell_id"] in window["cells"] for window in windows)
        ]
        scientific_classes = {
            row.get("cell_class") for row in cells
            if row.get("status") == "complete"
        }
        saturation_runaway_only = bool(
            scientific_classes
            and scientific_classes <= {"refractory_saturated", "runaway"}
        )
        seed_rows[str(seed)] = {
            "windows": windows,
            "isolated_non_tonic_cells": isolated,
            "saturation_runaway_only": saturation_runaway_only,
            "cell_status_counts": dict(Counter(row["status"] for row in cells)),
        }

    candidates = []
    keys = {
        (
            window["phenotype"],
            window["direction"],
            tuple(window["cells"]),
        )
        for row in seed_rows.values() for window in row["windows"]
    }
    for phenotype, direction, homologous_cells in sorted(keys):
        supporting = [
            seed for seed in SEEDS
            if any(
                window["phenotype"] == phenotype
                and window["direction"] == direction
                and tuple(window["cells"]) == homologous_cells
                for window in seed_rows[str(seed)]["windows"]
            )
        ]
        third_seed_assessment = {}
        for seed in SEEDS:
            if seed in supporting:
                continue
            supporting_windows = [
                window
                for supporting_seed in supporting
                for window in seed_rows[str(supporting_seed)]["windows"]
                if window["phenotype"] == phenotype
                and window["direction"] == direction
                and tuple(window["cells"]) == homologous_cells
            ]
            third_seed_assessment[str(seed)] = (
                _primary_third_seed_compatibility(
                    [row for row in rows if row["seed"] == seed],
                    phenotype=phenotype,
                    direction=direction,
                    supporting_windows=supporting_windows,
                )
            )
        if (
            len(supporting) >= 2
            and all(
                assessment["compatible"]
                for assessment in third_seed_assessment.values()
            )
        ):
            candidates.append({
                "phenotype": phenotype,
                "direction": direction,
                "homologous_cells": list(homologous_cells),
                "supporting_seeds": supporting,
                "third_seed_assessment": third_seed_assessment,
                "dt2_eligible": set(DT2_SEEDS).issubset(supporting),
            })
    any_window = any(row["windows"] for row in seed_rows.values())
    any_isolated = any(
        row["isolated_non_tonic_cells"] for row in seed_rows.values()
    )
    eligible_candidates = [
        row for row in candidates if row["dt2_eligible"]
    ]
    eligible_labels = {
        (row["phenotype"], row["direction"])
        for row in eligible_candidates
    }
    if len(eligible_labels) == 1:
        status = PRIMARY_POSITIVE_STATUS
    elif len(eligible_labels) > 1:
        status = "seed_heterogeneous_maturation"
    elif candidates:
        status = "resolution_confirmation_unavailable"
    elif any_window:
        status = "seed_heterogeneous_maturation"
    elif any_isolated:
        status = "isolated_maturation_candidate"
    else:
        status = "no_window"
    return {
        "tier": tier,
        "status": status,
        "candidates": candidates,
        "dt2_eligible_candidates": eligible_candidates,
        "seed_results": seed_rows,
        "acceptance_semantics": (
            "same_phenotype_direction_and_homologous_cells_in_at_least_"
            "2_of_3_seeds;"
            "every_remaining_seed_must_be_concordant_or_explicitly_"
            "probabilistically_indeterminate"
        ),
    }


def _cell_inventory(coordinate_manifest, *, expected_seeds=SEEDS):
    out = {}
    for seed in expected_seeds:
        seed_row = coordinate_manifest["seeds"][str(seed)]
        for cell in seed_row["cells"]:
            key = (seed, cell["tier"], cell["cell_id"])
            if key in out:
                raise ValueError(f"duplicate coordinate cell: {key}")
            out[key] = {**cell, "seed": int(seed)}
    return out


def _matrix_complete(cell_rows, coordinates):
    by_key = {
        (row["seed"], row["tier"], row["cell_id"]): row
        for row in cell_rows
    }
    missing = sorted(set(coordinates) - set(by_key))
    blocked = sorted(
        key for key, row in by_key.items()
        if row["status"] == "blocked"
    )
    return {
        "complete": not missing and not blocked,
        "missing_cells": [list(key) for key in missing],
        "blocked_cells": [list(key) for key in blocked],
    }


def build_base_atlas(
    *,
    resolution="dt",
    phasec_manifest_path=PHASEC_MANIFEST,
    coordinate_manifest_path=None,
):
    """Build the gain-independent C1 atlas from all preregistered base parts."""
    phasec = _load_json(phasec_manifest_path)
    PCC.validate_manifest(phasec)
    if phasec.get("production_authorized") is not True:
        raise ValueError("C1 analysis requires the final production manifest")
    locked_coordinate_path, locked_coordinate, coordinate_ref = (
        _coordinate_path_from_final(phasec, resolution)
    )
    coordinate_manifest_path = (
        locked_coordinate_path
        if coordinate_manifest_path is None
        else Path(coordinate_manifest_path)
    )
    if Path(coordinate_manifest_path).resolve() != locked_coordinate_path.resolve():
        raise ValueError(
            f"{resolution} coordinate path differs from final forward lock"
        )
    coordinate = _load_json(coordinate_manifest_path)
    _validate_self_hash(coordinate, label="C1 coordinate manifest")
    phasec_file_sha = _sha256(phasec_manifest_path)
    coordinate_file_sha = _sha256(coordinate_manifest_path)
    required_coordinate_ref = {
        "path": _relative(coordinate_manifest_path),
        "file_sha256": coordinate_file_sha,
        "manifest_sha256": coordinate["manifest_sha256"],
        "semantic_sha256": coordinate["semantic_sha256"],
    }
    if coordinate_ref != required_coordinate_ref:
        raise ValueError("final Phase-C/C1 coordinate reference mismatch")
    bootstrap_ref = {
        "path": phasec.get("provenance", {}).get(
            "phasec_input_manifest_path"
        ),
        "file_sha256": phasec.get("provenance", {}).get(
            "phasec_input_manifest_file_sha256"
        ),
        "manifest_sha256": phasec.get("provenance", {}).get(
            "phasec_input_manifest_manifest_sha256"
        ),
    }
    coordinate_parent = {
        "path": coordinate.get("parent_phasec_input_manifest_path"),
        "file_sha256": coordinate.get(
            "parent_phasec_input_manifest_file_sha256"
        ),
        "manifest_sha256": coordinate.get(
            "parent_phasec_input_manifest_sha256"
        ),
    }
    if bootstrap_ref != coordinate_parent or any(
        value is None for value in bootstrap_ref.values()
    ):
        raise ValueError("coordinate/bootstrap Phase-C provenance mismatch")
    expected_seeds = SEEDS if resolution == "dt" else DT2_SEEDS
    expected_npz = {
        str(seed): coordinate["seeds"][str(seed)]["npz_file_sha256"]
        for seed in expected_seeds
    }
    if phasec.get("c1", {}).get(
        "coordinate_npz_file_sha256_by_seed_by_resolution",
        phasec.get("c1", {}).get(
            "coordinate_npz_sha256_by_seed_by_resolution", {}
        ),
    ).get(resolution) != expected_npz:
        raise ValueError("final Phase-C coordinate NPZ SHA map mismatch")
    expected_npz_semantic = {
        str(seed): coordinate["seeds"][str(seed)][
            "npz_semantic_sha256"
        ]
        for seed in expected_seeds
    }
    if phasec.get("c1", {}).get(
        "coordinate_npz_semantic_sha256_by_seed_by_resolution", {}
    ).get(resolution) != expected_npz_semantic:
        raise ValueError(
            "final Phase-C coordinate NPZ semantic SHA map mismatch"
        )
    panels = C0._load_panels()
    phasec_manifest_file_sha256 = phasec_file_sha
    coordinate_manifest_file_sha256 = coordinate_file_sha
    coordinates = _cell_inventory(
        coordinate, expected_seeds=expected_seeds
    )
    cells = []
    part_counts = Counter()
    resource_tasks = []
    for (seed, tier, cell_id), cell in sorted(coordinates.items()):
        if cell["status"] != "valid":
            cells.append(aggregate_cell_rows([], cell))
            continue
        run_rows = []
        for phase in PHASES:
            for noise in NOISES:
                path = base_part_path(
                    resolution, seed, tier, cell_id, phase, noise
                )
                resource_tasks.append((
                    _base_task_key(
                        resolution=resolution,
                        seed=seed,
                        tier=tier,
                        cell_id=cell_id,
                        phase=phase,
                        noise=noise,
                    ),
                    path,
                    "c1_base" if resolution == "dt" else "c1_dt2_base",
                ))
                classified = classify_base_part(
                    path,
                    coordinate=cell,
                    coordinate_manifest=coordinate,
                    phasec_manifest=phasec,
                    panels=panels,
                    seed=seed,
                    phase=phase,
                    noise=noise,
                    resolution=resolution,
                    phasec_manifest_file_sha256=(
                        phasec_manifest_file_sha256
                    ),
                    coordinate_manifest_file_sha256=(
                        coordinate_manifest_file_sha256
                    ),
                    coordinate_ref=coordinate_ref,
                )
                run_rows.append({
                    "seed": seed,
                    "cell_id": cell_id,
                    "tier": tier,
                    "trajectory_id": cell["trajectory_id"],
                    "path_index": int(cell["path_index"]),
                    "path_direction": cell["path_direction"],
                    "phase": phase,
                    "noise": noise,
                    **classified,
                })
                part_counts[classified["status"]] += 1
        cells.append(aggregate_cell_rows(run_rows, cell))
    matrix = _matrix_complete(cells, coordinates)
    primary = adjudicate_tier(cells, "primary_convex")
    shell = adjudicate_tier(cells, "secondary_shell")
    return {
        "schema": C1_BASE_ATLAS_SCHEMA,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "phasec_manifest_file_sha256": phasec_file_sha,
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate[
            "semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": coordinate_file_sha,
        "phasec_producer_file_sha256": phasec["provenance"][
            "producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": coordinate[
            "producer_file_sha256"
        ],
        "coordinate_npz_provenance_by_seed": {
            str(seed): _coordinate_seed_provenance(
                coordinate, seed=seed
            )
            for seed in expected_seeds
        },
        "resolution": resolution,
        "expected_seeds": list(SEEDS),
        "expected_phases": list(PHASES),
        "expected_noises": list(NOISES),
        "runner_base_part_schema": C1_BASE_PART_SCHEMA,
        "runner_observables_schema": C1_OBSERVABLES_SCHEMA,
        "matrix": matrix,
        "part_status_counts": dict(part_counts),
        "resource_receipt_index": build_resource_receipt_index(
            resource_tasks,
            manifest_sha256=phasec["manifest_sha256"],
        ),
        "cells": cells,
        "primary_base_adjudication": primary,
        "secondary_shell_base_adjudication": shell,
        "claim_boundary": (
            "frozen source-space identity/maturation only; not entry, offset, "
            "recovery, observation match, actuator efficacy, or lifecycle"
        ),
    }


def _validate_trigger_manifest(trigger, base_atlas, base_atlas_path):
    _validate_self_hash(trigger, label="C1 gain trigger manifest")
    required = {
        "schema": C1_GAIN_TRIGGER_SCHEMA,
        "base_atlas_sha256": _sha256(base_atlas_path),
        "phasec_manifest_sha256": base_atlas["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": base_atlas[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": base_atlas[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": base_atlas[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": base_atlas[
            "coordinate_manifest_file_sha256"
        ],
        "resolution": base_atlas["resolution"],
        "phasec_producer_file_sha256": base_atlas[
            "phasec_producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": base_atlas[
            "coordinate_producer_file_sha256"
        ],
    }
    for key, wanted in required.items():
        if trigger.get(key) != wanted:
            raise ValueError(f"C1 gain trigger mismatch: {key}")


def _gain_status_for(
    cell, trigger, *, resolution, trigger_manifest_file_sha256=None
):
    if not cell.get("gain_trigger_eligible"):
        return {
            "status": "not_triggered",
            "final_cell_class": cell["cell_class"],
        }
    if trigger is None:
        return {
            "status": "trigger_manifest_not_locked",
            "final_cell_class": "spike_AI_screen_candidate",
        }
    trigger_rows = {
        (int(row["seed"]), row["tier"], row["cell_id"]): row
        for row in trigger.get("triggered_cells", [])
    }
    key = (cell["seed"], cell["tier"], cell["cell_id"])
    if key not in trigger_rows:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "eligible_cell_absent_from_locked_trigger_manifest",
            "final_cell_class": "missing",
        }
    trigger_row = trigger_rows[key]
    path = gain_status_path(resolution, *key)
    if not path.is_file():
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "missing_conditional_gain_status",
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    try:
        row = _load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": f"invalid_conditional_gain_status:{exc}",
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    expected = {
        "schema": C1_GAIN_STATUS_SCHEMA,
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": trigger_manifest_file_sha256,
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger["coordinate_manifest_sha256"],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        "resolution": resolution,
        "seed": cell["seed"],
        "tier": cell["tier"],
        "cell_id": cell["cell_id"],
        "slow_state_sha256": cell["slow_state_sha256"],
        "phasec_producer_file_sha256": trigger[
            "phasec_producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": trigger[
            "coordinate_producer_file_sha256"
        ],
        "trigger_producer_file_sha256": trigger[
            "producer_file_sha256"
        ],
    }
    for field, wanted in expected.items():
        if row.get(field) != wanted:
            return {
                "status": "C1_blocked_conditional_gain",
                "reason": f"conditional_gain_field_mismatch:{field}",
                "gain_status_path": _relative(path),
                "final_cell_class": "missing",
            }
    expected_arms = {
        arm["path"] for arm in trigger_row.get(
            "expected_carrier_gain_arms", []
        )
    }
    arm_files = row.get("completed_arm_file_sha256")
    if (
        len(expected_arms) != 30
        or not isinstance(arm_files, dict)
        or set(arm_files) != expected_arms
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in arm_files.values()
        )
    ):
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "conditional_gain_arm_provenance_incomplete",
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    carrier_hash_drift = []
    for arm_path in sorted(expected_arms):
        arm_file = ROOT / arm_path
        if (
            not arm_file.is_file()
            or _sha256(arm_file) != arm_files.get(arm_path)
        ):
            carrier_hash_drift.append(arm_path)
    if carrier_hash_drift:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "conditional_gain_arm_file_hash_drift",
            "drifted_paths": carrier_hash_drift,
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    expected_denominators = {
        ref["path"]: ref["file_sha256"]
        for ref in trigger_row.get("reused_c0_preentry_denominators", [])
    }
    if (
        len(expected_denominators) != 15
        or row.get("reused_c0_preentry_denominator_sha256")
        != expected_denominators
    ):
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "conditional_gain_denominator_provenance_mismatch",
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    denominator_hash_drift = []
    for denominator_path, expected_sha in sorted(
        expected_denominators.items()
    ):
        denominator_file = ROOT / denominator_path
        if (
            not denominator_file.is_file()
            or _sha256(denominator_file) != expected_sha
        ):
            denominator_hash_drift.append(denominator_path)
    if denominator_hash_drift:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "conditional_gain_denominator_file_hash_drift",
            "drifted_paths": denominator_hash_drift,
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    resource_index = row.get("resource_receipt_index")
    resource_failure = resource_receipt_index_failure(
        resource_index,
        manifest_sha256=trigger["phasec_manifest_sha256"],
    )
    if resource_failure is not None:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": resource_failure,
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    gain_class = row.get("gain_class")
    if gain_class not in {
        "balanced_AI_tonic_cell",
        "tonic_non_AI",
        "tonic_gain_indeterminate",
    }:
        return {
            "status": "C1_blocked_conditional_gain",
            "reason": "unknown_or_nonterminal_gain_class",
            "gain_status_path": _relative(path),
            "final_cell_class": "missing",
        }
    return {
        "status": (
            "scientific_indeterminate"
            if gain_class == "tonic_gain_indeterminate" else "complete"
        ),
        "gain_status_path": _relative(path),
        "gain_status_sha256": _sha256(path),
        "final_cell_class": gain_class,
        "resource_receipt_index": resource_index,
    }


def _strict_negative(cells, *, tier):
    rows = [row for row in cells if row["tier"] == tier]
    expected = (
        len(SEEDS)
        * (len(N.PRIMARY_CELL_NAMES) if tier == "primary_convex"
           else len(N.SHELL_CELL_NAMES))
    )
    if len(rows) != expected:
        return False, "cell_coverage_incomplete"
    if tier == "primary_convex" and any(
        row["status"] == "invalid_physical" for row in rows
    ):
        return False, "primary_physical_cell_invalid"
    if any(row["status"] != "complete" for row in rows):
        return False, "nonterminal_or_blocked_cell"
    if any(row["cell_class"] in NON_TONIC_CLASSES for row in rows):
        return False, "non_tonic_candidate_present"
    if any(
        row.get("conditional_gain", {}).get("status")
        in {
            "trigger_manifest_not_locked",
            "C1_blocked_conditional_gain",
            "scientific_indeterminate",
        }
        for row in rows
    ):
        return False, "conditional_gain_not_terminal_resolved"
    return True, "complete_bounded_negative"


def apply_conditional_gain(
    base_atlas,
    *,
    base_atlas_path,
    trigger_manifest_path=GAIN_TRIGGER_MANIFEST,
):
    """Attach conditional gain terminal status without mutating base evidence."""
    trigger = None
    trigger_file_sha = None
    path = Path(trigger_manifest_path)
    if path.is_file():
        trigger = _load_json(path)
        _validate_trigger_manifest(trigger, base_atlas, base_atlas_path)
        trigger_file_sha = _sha256(path)
    cells = []
    for base_cell in base_atlas["cells"]:
        cell = dict(base_cell)
        gain = _gain_status_for(
            cell,
            trigger,
            resolution=base_atlas["resolution"],
            trigger_manifest_file_sha256=trigger_file_sha,
        )
        cell["conditional_gain"] = gain
        if gain["status"] in {"complete", "scientific_indeterminate"}:
            cell["cell_class"] = gain["final_cell_class"]
            if gain["status"] == "scientific_indeterminate":
                cell["status"] = "indeterminate"
        elif gain["status"] == "C1_blocked_conditional_gain":
            cell["status"] = "blocked"
        cells.append(cell)

    primary_base = base_atlas["primary_base_adjudication"]
    shell_base = base_atlas["secondary_shell_base_adjudication"]
    if base_atlas.get("matrix", {}).get("complete") is not True:
        return {
            "schema": C1_SUMMARY_SCHEMA,
            "phasec_manifest_sha256": base_atlas[
                "phasec_manifest_sha256"
            ],
            "phasec_manifest_file_sha256": base_atlas[
                "phasec_manifest_file_sha256"
            ],
            "coordinate_manifest_sha256": base_atlas[
                "coordinate_manifest_sha256"
            ],
            "coordinate_manifest_semantic_sha256": base_atlas[
                "coordinate_manifest_semantic_sha256"
            ],
            "coordinate_manifest_file_sha256": base_atlas[
                "coordinate_manifest_file_sha256"
            ],
            "resolution": base_atlas["resolution"],
            "base_atlas_path": _relative(base_atlas_path),
            "base_atlas_sha256": _sha256(base_atlas_path),
            "gain_trigger_manifest_sha256": (
                None if trigger is None else trigger["manifest_sha256"]
            ),
            "cells": cells,
            "primary_adjudication": primary_base,
            "secondary_shell_adjudication": shell_base,
            "verdict": "C1_blocked_manifest",
            "reason": "base_matrix_technically_incomplete",
            "claim_boundary": base_atlas["claim_boundary"],
        }
    # The trigger decision is itself preregistered evidence.  Even when the
    # base atlas contains zero eligible AI cells, an immutable *empty* trigger
    # manifest must close that decision before a negative, isolated, or
    # heterogeneous atlas can be terminal.  A complete non-tonic maturation
    # window is the sole exception because its acceptance is gain-independent.
    has_gain_independent_window = (
        primary_base["status"] == PRIMARY_POSITIVE_STATUS
        or shell_base["status"] == SHELL_POSITIVE_STATUS
    )
    if trigger is None and not has_gain_independent_window:
        return {
            "schema": C1_SUMMARY_SCHEMA,
            "phasec_manifest_sha256": base_atlas[
                "phasec_manifest_sha256"
            ],
            "phasec_manifest_file_sha256": base_atlas[
                "phasec_manifest_file_sha256"
            ],
            "coordinate_manifest_sha256": base_atlas[
                "coordinate_manifest_sha256"
            ],
            "coordinate_manifest_semantic_sha256": base_atlas[
                "coordinate_manifest_semantic_sha256"
            ],
            "coordinate_manifest_file_sha256": base_atlas[
                "coordinate_manifest_file_sha256"
            ],
            "resolution": base_atlas["resolution"],
            "base_atlas_path": _relative(base_atlas_path),
            "base_atlas_sha256": _sha256(base_atlas_path),
            "gain_trigger_manifest_sha256": None,
            "cells": cells,
            "primary_adjudication": primary_base,
            "secondary_shell_adjudication": shell_base,
            "verdict": "C1_gain_trigger_not_locked",
            "reason": "write_once_trigger_decision_missing",
            "claim_boundary": base_atlas["claim_boundary"],
        }
    if (
        primary_base["status"] == "resolution_confirmation_unavailable"
        or shell_base["status"] == "resolution_confirmation_unavailable"
    ):
        verdict = "C1_blocked_resolution_gate"
        unavailable_tier = (
            "primary_convex"
            if primary_base["status"]
            == "resolution_confirmation_unavailable"
            else "secondary_shell"
        )
    elif primary_base["status"] == PRIMARY_POSITIVE_STATUS:
        verdict = (
            "primary_maturation_candidate_requires_dt2"
            if base_atlas["resolution"] == "dt"
            else "primary_maturation_dt2_confirmation_only"
        )
    elif shell_base["status"] == SHELL_POSITIVE_STATUS:
        verdict = (
            "secondary_shell_candidate_requires_dt2"
            if base_atlas["resolution"] == "dt"
            else "secondary_shell_dt2_confirmation_only"
        )
    elif (
        primary_base["status"] == "isolated_maturation_candidate"
        or shell_base["status"] == "isolated_maturation_candidate"
    ):
        verdict = "isolated_maturation_candidate"
    elif (
        primary_base["status"] == "seed_heterogeneous_maturation"
        or shell_base["status"] == "seed_heterogeneous_maturation"
    ):
        verdict = "seed_heterogeneous_maturation"
    else:
        primary_negative, primary_reason = _strict_negative(
            cells, tier="primary_convex"
        )
        shell_negative, shell_reason = _strict_negative(
            cells, tier="secondary_shell"
        )
        if primary_negative:
            verdict = "no_maturation_in_tested_primary_neighbourhood"
        elif any(
            row.get("conditional_gain", {}).get("status")
            == "C1_blocked_conditional_gain"
            for row in cells
        ):
            verdict = "C1_blocked_conditional_gain"
        elif any(
            row.get("conditional_gain", {}).get("status")
            == "trigger_manifest_not_locked"
            for row in cells
        ):
            verdict = "C1_gain_trigger_not_locked"
        else:
            verdict = "C1_incomplete_or_indeterminate"
        return {
            "schema": C1_SUMMARY_SCHEMA,
            "phasec_manifest_sha256": base_atlas["phasec_manifest_sha256"],
            "phasec_manifest_file_sha256": base_atlas[
                "phasec_manifest_file_sha256"
            ],
            "coordinate_manifest_sha256": base_atlas[
                "coordinate_manifest_sha256"
            ],
            "coordinate_manifest_semantic_sha256": base_atlas[
                "coordinate_manifest_semantic_sha256"
            ],
            "coordinate_manifest_file_sha256": base_atlas[
                "coordinate_manifest_file_sha256"
            ],
            "resolution": base_atlas["resolution"],
            "base_atlas_path": _relative(base_atlas_path),
            "base_atlas_sha256": _sha256(base_atlas_path),
            "gain_trigger_manifest_sha256": (
                None if trigger is None else trigger["manifest_sha256"]
            ),
            "cells": cells,
            "primary_adjudication": {
                **primary_base,
                "strict_negative": primary_negative,
                "strict_negative_reason": primary_reason,
            },
            "secondary_shell_adjudication": {
                **shell_base,
                "strict_negative": shell_negative,
                "strict_negative_reason": shell_reason,
            },
            "verdict": verdict,
            "claim_boundary": base_atlas["claim_boundary"],
        }
    # A complete non-tonic positive is independent of conditional AI gain.
    extra = {}
    if verdict == "C1_blocked_resolution_gate":
        extra = {
            "reason": "resolution_confirmation_unavailable",
            "resolution_gate": "insufficient_homologous_native_support",
            "unavailable_tier": unavailable_tier,
            "required_dt2_seeds": list(DT2_SEEDS),
        }
    return {
        "schema": C1_SUMMARY_SCHEMA,
        "phasec_manifest_sha256": base_atlas["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": base_atlas[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": base_atlas[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": base_atlas[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": base_atlas[
            "coordinate_manifest_file_sha256"
        ],
        "resolution": base_atlas["resolution"],
        "base_atlas_path": _relative(base_atlas_path),
        "base_atlas_sha256": _sha256(base_atlas_path),
        "gain_trigger_manifest_sha256": (
            None if trigger is None else trigger["manifest_sha256"]
        ),
        "cells": cells,
        "primary_adjudication": primary_base,
        "secondary_shell_adjudication": shell_base,
        "verdict": verdict,
        **extra,
        "claim_boundary": base_atlas["claim_boundary"],
    }


def _native_positive_windows(native_summary):
    """Enumerate native windows supported by both dt2-enabled seeds."""
    selected = []
    for tier, key in (
        ("primary_convex", "primary_adjudication"),
        ("secondary_shell", "secondary_shell_adjudication"),
    ):
        adjudication = native_summary.get(key, {})
        expected_status = (
            PRIMARY_POSITIVE_STATUS
            if tier == "primary_convex" else SHELL_POSITIVE_STATUS
        )
        if adjudication.get("status") != expected_status:
            continue
        candidates = {
            (row["phenotype"], row["direction"])
            for row in adjudication.get("candidates", [])
        }
        for phenotype, direction in sorted(candidates):
            for seed in DT2_SEEDS:
                for window in adjudication.get("seed_results", {}).get(
                    str(seed), {}
                ).get("windows", []):
                    if (
                        window.get("phenotype") == phenotype
                        and window.get("direction") == direction
                        and isinstance(window.get("cells"), list)
                        and len(window["cells"]) == (
                            2 if tier == "primary_convex" else 1
                        )
                    ):
                        selected.append({
                            "tier": tier,
                            "phenotype": phenotype,
                            "direction": direction,
                            "seed": seed,
                            "cells": list(window["cells"]),
                        })
    labels = {
        (row["tier"], row["phenotype"], row["direction"])
        for row in selected
        if {
            candidate["seed"] for candidate in selected
            if (
                candidate["tier"], candidate["phenotype"],
                candidate["direction"],
            ) == (
                row["tier"], row["phenotype"], row["direction"]
            )
        } >= set(DT2_SEEDS)
    }
    return [
        row for row in selected
        if (row["tier"], row["phenotype"], row["direction"]) in labels
    ]


def build_dt2_confirmation_manifest(
    *,
    native_summary_path=None,
    phasec_manifest_path=PHASEC_MANIFEST,
    gain_trigger_path=GAIN_TRIGGER_MANIFEST,
):
    """Compatibility entry point for the one canonical dedicated lock."""
    return DT2LOCK.build_payload(
        phasec_path=phasec_manifest_path,
        native_summary_path=(
            native_summary_path or OUT / "phasec1_summary_dt.json"
        ),
        gain_trigger_path=gain_trigger_path,
    )


def lock_dt2_confirmation_manifest(
    *,
    native_summary_path=None,
    output_path=DT2_CONFIRMATION_MANIFEST,
    gain_trigger_path=GAIN_TRIGGER_MANIFEST,
):
    payload = build_dt2_confirmation_manifest(
        native_summary_path=native_summary_path,
        gain_trigger_path=gain_trigger_path,
    )
    return payload, N.write_json_once(output_path, payload)


def analyze_dt2_confirmation(
    *,
    selection_manifest_path=DT2_CONFIRMATION_MANIFEST,
    phasec_manifest_path=PHASEC_MANIFEST,
):
    """Analyze only the locked homologous subset, never a partial dt2 negative."""
    selection_manifest_path = Path(selection_manifest_path)
    phasec_manifest_path = Path(phasec_manifest_path)
    selection = _load_json(selection_manifest_path)
    _validate_self_hash(selection, label="C1 dt2 confirmation manifest")
    if (
        selection.get("schema") != C1_DT2_CONFIRMATION_SCHEMA
        or selection.get("resolution") != "dt2"
        or selection.get("selection_is_closed") is not True
    ):
        raise ValueError("invalid dt2 confirmation-only manifest")
    phasec = _load_json(phasec_manifest_path)
    PCC.validate_manifest(phasec)
    _, coordinate, dt2_ref = _coordinate_path_from_final(phasec, "dt2")
    _, _, dt_ref = _coordinate_path_from_final(
        phasec, "dt"
    )
    expected_phasec = {
        "path": _relative(phasec_manifest_path),
        "file_sha256": _sha256(phasec_manifest_path),
        "manifest_sha256": phasec["manifest_sha256"],
    }
    if selection.get("final_phasec") != expected_phasec:
        raise ValueError("dt2 selection/final Phase-C provenance mismatch")
    expected_coordinate = dict(dt2_ref)
    if selection.get("coordinate_manifests", {}).get(
        "dt2"
    ) != expected_coordinate:
        raise ValueError("dt2 selection/coordinate provenance mismatch")
    expected_native_coordinate = dict(dt_ref)
    if (
        selection.get("coordinate_manifests", {}).get("dt")
        != expected_native_coordinate
    ):
        raise ValueError("dt2 selection/native-coordinate provenance mismatch")
    if (
        selection.get("coordinate_producer_file_sha256")
        != coordinate.get("producer_file_sha256")
    ):
        raise ValueError("dt2 selection/coordinate producer provenance mismatch")
    trigger_ref = selection.get("gain_trigger_manifest")
    if not isinstance(trigger_ref, dict):
        raise ValueError("dt2 selection lacks canonical gain-trigger provenance")
    trigger_path = ROOT / str(trigger_ref.get("path", ""))
    try:
        trigger = DT2LOCK._validate_gain_trigger(
            trigger_path, phasec, phasec_manifest_path
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"dt2 selection/gain-trigger provenance mismatch: {exc}"
        ) from exc
    expected_trigger = {
        "path": _relative(trigger_path),
        "file_sha256": _sha256(trigger_path),
        "manifest_sha256": trigger["manifest_sha256"],
        "selection_is_closed": True,
    }
    if trigger_ref != expected_trigger:
        raise ValueError("dt2 selection/gain-trigger provenance mismatch")
    native_ref = selection["native_summary"]
    native_path = ROOT / native_ref["path"]
    if (
        not native_path.is_file()
        or _sha256(native_path) != native_ref["file_sha256"]
    ):
        raise ValueError("native summary provenance drift")
    native = _load_json(native_path)
    if (
        native.get("schema") != native_ref.get("schema")
        or native.get("resolution") != "dt"
        or native.get("phasec_manifest_sha256")
        != phasec["manifest_sha256"]
        or native.get("phasec_manifest_file_sha256")
        != _sha256(phasec_manifest_path)
        or native.get("coordinate_manifest_sha256")
        != dt_ref["manifest_sha256"]
        or native.get("coordinate_manifest_semantic_sha256")
        != dt_ref["semantic_sha256"]
        or native.get("coordinate_manifest_file_sha256")
        != dt_ref["file_sha256"]
        or native.get("gain_trigger_manifest_sha256")
        != trigger["manifest_sha256"]
    ):
        raise ValueError("native summary semantic provenance mismatch")
    panels = C0._load_panels()
    coordinates = _cell_inventory(coordinate, expected_seeds=DT2_SEEDS)
    expected_arms = {
        (
            int(row["seed"]), row["tier"], row["cell_id"],
            row["phase"], row["noise"],
        ): row
        for row in selection["expected_base_arms"]
    }
    expected_n = (
        len(selection["selected_cells"]) * len(PHASES) * len(NOISES)
    )
    if len(expected_arms) != expected_n:
        raise ValueError("dt2 expected-arm matrix is incomplete or duplicated")
    cell_rows, technical_blockers = [], []
    for selected in selection["selected_cells"]:
        if (
            not isinstance(selected.get("phenotypes"), list)
            or not selected["phenotypes"]
            or not isinstance(selected.get("directions"), list)
            or not selected["directions"]
        ):
            raise ValueError(
                "dt2 selected cell lacks closed phenotype/direction semantics"
            )
        key = (
            int(selected["seed"]), selected["tier"], selected["cell_id"]
        )
        coordinate_cell = coordinates.get(key)
        if coordinate_cell is None:
            raise ValueError(f"dt2 coordinate cell absent: {key}")
        runs = []
        for phase in PHASES:
            for noise in NOISES:
                arm = expected_arms.get((*key, phase, noise))
                if arm is None:
                    raise ValueError(f"dt2 arm absent from selection: {key}")
                classified = classify_base_part(
                    ROOT / arm["path"],
                    coordinate=coordinate_cell,
                    coordinate_manifest=coordinate,
                    phasec_manifest=phasec,
                    panels=panels,
                    seed=key[0],
                    phase=phase,
                    noise=noise,
                    resolution="dt2",
                    phasec_manifest_file_sha256=expected_phasec[
                        "file_sha256"
                    ],
                    coordinate_manifest_file_sha256=dt2_ref[
                        "file_sha256"
                    ],
                    coordinate_ref=dt2_ref,
                    dt2_confirmation_manifest_sha256=selection[
                        "manifest_sha256"
                    ],
                    dt2_confirmation_manifest_file_sha256=_sha256(
                        selection_manifest_path
                    ),
                )
                if (
                    classified.get("locked_arm_identity") is not None
                    and {
                        field: value
                        for field, value in classified[
                            "locked_arm_identity"
                        ].items()
                        if not field.startswith("dt2_confirmation_manifest_")
                    } != {
                        field: arm[field]
                        for field in classified["locked_arm_identity"]
                        if not field.startswith("dt2_confirmation_manifest_")
                    }
                ):
                    technical_blockers.append({
                        "key": [*key, phase, noise],
                        "reason": "selection_arm_identity_mismatch",
                    })
                runs.append({
                    "seed": key[0],
                    "cell_id": key[2],
                    "tier": key[1],
                    "trajectory_id": coordinate_cell["trajectory_id"],
                    "path_index": int(coordinate_cell["path_index"]),
                    "path_direction": coordinate_cell["path_direction"],
                    "phase": phase,
                    "noise": noise,
                    **classified,
                })
        cell = aggregate_cell_rows(runs, coordinate_cell)
        cell["expected_phenotypes"] = selected["phenotypes"]
        cell["expected_directions"] = selected["directions"]
        cell_rows.append(cell)
        if cell["status"] == "blocked":
            technical_blockers.append({
                "key": list(key), "reason": "cell_base_evidence_blocked"
            })
    confirmed = []
    window_assessments = []
    scientific_indeterminate_windows = []
    terminal_contradictions = []
    for window in selection["selected_windows"]:
        rows_by_cell = {
            row["cell_id"]: row
            for row in cell_rows
            if row["seed"] == window["seed"]
            and row["tier"] == window["tier"]
            and row["cell_id"] in window["cells"]
        }
        expected_cells = 2 if window["tier"] == "primary_convex" else 1
        selected_rows = [
            rows_by_cell.get(cell_id) for cell_id in window["cells"]
        ]
        if len(window["cells"]) != expected_cells or any(
            row is None for row in selected_rows
        ):
            technical_blockers.append({
                "key": [
                    window["seed"], window["tier"], *window["cells"]
                ],
                "reason": "selected_window_cell_evidence_missing",
            })
            assessment = "technical_block"
        elif any(row["status"] == "blocked" for row in selected_rows):
            assessment = "technical_block"
        elif any(
            row["status"] == "complete"
            and row["cell_class"] != window["phenotype"]
            for row in selected_rows
        ):
            # One completed, nonmatching member is sufficient to contradict
            # the preregistered adjacent/same-cell window at this resolution.
            assessment = "terminal_contradiction"
        elif any(row["status"] != "complete" for row in selected_rows):
            # A scientifically indeterminate cell is not evidence that the
            # native window changes identity at dt/2.
            assessment = "scientific_indeterminate"
        elif all(
            row["cell_class"] == window["phenotype"]
            for row in selected_rows
        ):
            assessment = "confirmed"
        else:
            assessment = "scientific_indeterminate"
        window_row = {
            **window,
            "assessment": assessment,
            "observed_cell_classes": {
                cell_id: (
                    None if rows_by_cell.get(cell_id) is None
                    else rows_by_cell[cell_id].get("cell_class")
                )
                for cell_id in window["cells"]
            },
            "observed_cell_statuses": {
                cell_id: (
                    None if rows_by_cell.get(cell_id) is None
                    else rows_by_cell[cell_id].get("status")
                )
                for cell_id in window["cells"]
            },
        }
        window_assessments.append(window_row)
        if assessment == "confirmed":
            confirmed.append(window)
        elif assessment == "terminal_contradiction":
            terminal_contradictions.append(window_row)
        elif assessment == "scientific_indeterminate":
            scientific_indeterminate_windows.append(window_row)
    matches = _homologous_window_matches(confirmed)
    verdict, label_assessments = _close_dt2_candidate_labels(
        window_assessments, matches, technical_blockers
    )
    resource_index = build_resource_receipt_index(
        [
            (
                _base_task_key(
                    resolution="dt2",
                    seed=int(arm["seed"]),
                    tier=arm["tier"],
                    cell_id=arm["cell_id"],
                    phase=arm["phase"],
                    noise=arm["noise"],
                ),
                ROOT / arm["path"],
                "c1_dt2_base",
            )
            for arm in expected_arms.values()
        ],
        manifest_sha256=phasec["manifest_sha256"],
    )
    return {
        "schema": C1_DT2_CONFIRMATION_SUMMARY_SCHEMA,
        "selection_manifest_path": _relative(selection_manifest_path),
        "selection_manifest_sha256": selection["manifest_sha256"],
        "selection_manifest_file_sha256": _sha256(
            selection_manifest_path
        ),
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "phasec_manifest_file_sha256": _sha256(phasec_manifest_path),
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate[
            "semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": dt2_ref["file_sha256"],
        "resolution": "dt2_confirmation_only",
        "matrix": {
            "expected_arms": expected_n,
            "technically_blocked": len(technical_blockers),
            "complete": not technical_blockers,
        },
        "cells": cell_rows,
        "confirmed_windows": confirmed,
        "window_assessments": window_assessments,
        "label_assessments": label_assessments,
        "matches": matches,
        "technical_blockers": technical_blockers,
        "terminal_contradictions": terminal_contradictions,
        "resource_receipt_index": resource_index,
        "scientific_indeterminate_windows": (
            scientific_indeterminate_windows
        ),
        "verdict": verdict,
        "reason": (
            "dt2_confirmation_subset_technical_block"
            if technical_blockers else
            "homologous_dt2_window_confirmed"
            if matches else
            "completed_dt2_terminal_contradiction"
            if verdict == "resolution_sensitive_maturation" else
            "dt2_scientifically_indeterminate"
        ),
        "claim_boundary": selection["claim_boundary"],
    }


def _homologous_window_matches(confirmed):
    """Return only exact-cell windows confirmed in both locked dt2 seeds."""
    window_seeds = defaultdict(set)
    for row in confirmed:
        window_seeds[
            (
                row["tier"],
                row["phenotype"],
                row["direction"],
                tuple(row["cells"]),
            )
        ].add(row["seed"])
    matches = [
        {
            "tier": window[0],
            "phenotype": window[1],
            "direction": window[2],
            "homologous_cells": list(window[3]),
            "homologous_supporting_seeds": sorted(seeds),
        }
        for window, seeds in sorted(window_seeds.items())
        if set(seeds) >= set(DT2_SEEDS)
    ]
    return matches


def _window_support_keys(adjudication):
    keys = {}
    for candidate in adjudication.get("candidates", []):
        label = (candidate["phenotype"], candidate["direction"])
        seed_windows = {}
        for seed, row in adjudication.get("seed_results", {}).items():
            windows = {
                tuple(window["cells"])
                for window in row.get("windows", [])
                if (
                    window["phenotype"],
                    window["direction"],
                ) == label
            }
            if windows:
                seed_windows[int(seed)] = windows
        keys[label] = seed_windows
    return keys


def _resolution_match(native_adjudication, dt2_adjudication):
    native = _window_support_keys(native_adjudication)
    dt2 = _window_support_keys(dt2_adjudication)
    matches = []
    for label in sorted(set(native) & set(dt2)):
        seeds = []
        for seed in sorted(set(native[label]) & set(dt2[label])):
            if native[label][seed] & dt2[label][seed]:
                seeds.append(seed)
        if len(seeds) >= 2:
            matches.append({
                "phenotype": label[0],
                "direction": label[1],
                "homologous_supporting_seeds": seeds,
            })
    return matches


def _close_dt2_candidate_labels(window_assessments, matches, technical_blockers):
    """Close homologous windows first, then their phenotype/direction label.

    A primary label may have several preregistered adjacent windows.  One
    contradicted window cannot turn a different unresolved homologous window
    into a negative; the label is contradicted only when every one of its
    homologous windows is terminally contradicted.
    """
    by_window = defaultdict(list)
    for row in window_assessments:
        window = (
            row.get("tier"),
            row.get("phenotype"),
            row.get("direction"),
            tuple(row.get("cells") or []),
        )
        by_window[window].append(str(row.get("assessment")))
    matched_windows = {
        (
            row.get("tier"),
            row.get("phenotype"),
            row.get("direction"),
            tuple(row.get("homologous_cells") or []),
        )
        for row in matches
    }
    window_rows = []
    for window, assessments in sorted(by_window.items()):
        if window in matched_windows:
            closure = "confirmed"
        elif "technical_block" in assessments:
            closure = "technical_block"
        elif "terminal_contradiction" in assessments:
            # Both fixed dt2 seeds are required.  One terminal mismatch makes
            # this exact homologous window unable to achieve 2/2 support.
            closure = "terminal_contradiction"
        else:
            closure = "scientific_indeterminate"
        window_rows.append({
            "tier": window[0],
            "phenotype": window[1],
            "direction": window[2],
            "homologous_cells": list(window[3]),
            "window_assessments": assessments,
            "closure": closure,
        })

    by_label = defaultdict(list)
    for row in window_rows:
        by_label[
            (row["tier"], row["phenotype"], row["direction"])
        ].append(row)
    labels = []
    for label, windows in sorted(by_label.items()):
        closures = {row["closure"] for row in windows}
        if "confirmed" in closures:
            closure = "confirmed"
        elif "technical_block" in closures:
            closure = "technical_block"
        elif "scientific_indeterminate" in closures:
            closure = "scientific_indeterminate"
        elif closures == {"terminal_contradiction"}:
            closure = "terminal_contradiction"
        else:
            closure = "scientific_indeterminate"
        labels.append({
            "tier": label[0],
            "phenotype": label[1],
            "direction": label[2],
            "homologous_windows": windows,
            "closure": closure,
        })
    if technical_blockers or any(
        row["closure"] == "technical_block" for row in labels
    ):
        verdict = "C1_blocked_resolution_gate"
    elif matches:
        verdict = (
            "maturation_window_at_primary_convex_states"
            if any(row.get("tier") == "primary_convex" for row in matches)
            else "maturation_candidate_in_secondary_shell"
        )
    elif any(
        row["closure"] == "scientific_indeterminate" for row in labels
    ):
        verdict = "C1_window_pending_dt2"
    elif labels and all(
        row["closure"] == "terminal_contradiction" for row in labels
    ):
        verdict = "resolution_sensitive_maturation"
    else:
        verdict = "C1_window_pending_dt2"
    return verdict, labels


def _layer_resolution_gate(native, dt2, *, tier, adjudication_key):
    """Close one C1 layer without borrowing evidence from the other layer."""
    adjudication = native.get(adjudication_key)
    if not isinstance(adjudication, dict):
        return {
            "tier": tier,
            "status": "blocked",
            "reason": "missing_native_layer_adjudication",
        }
    positive_status = (
        PRIMARY_POSITIVE_STATUS
        if tier == "primary_convex" else SHELL_POSITIVE_STATUS
    )
    native_status = adjudication.get("status")
    if native_status == "resolution_confirmation_unavailable":
        return {
            "tier": tier,
            "status": "blocked",
            "native_status": native_status,
            "reason": "resolution_confirmation_unavailable",
            "required_dt2_seeds": list(DT2_SEEDS),
        }
    if native_status != positive_status:
        return {
            "tier": tier,
            "status": "not_required",
            "native_status": native_status,
            "reason": "native_layer_has_no_positive_window",
        }

    required = set(DT2_SEEDS)
    eligible = [
        row for row in adjudication.get("candidates", [])
        if required.issubset(set(row.get("supporting_seeds") or []))
        and isinstance(row.get("homologous_cells"), list)
        and bool(row["homologous_cells"])
    ]
    if not eligible:
        return {
            "tier": tier,
            "status": "blocked",
            "native_status": native_status,
            "reason": "resolution_confirmation_unavailable",
            "required_dt2_seeds": sorted(required),
        }
    eligible_windows = {
        (
            str(row.get("phenotype")),
            str(row.get("direction")),
            tuple(row["homologous_cells"]),
        )
        for row in eligible
    }
    eligible_labels = {
        (phenotype, direction)
        for phenotype, direction, _cells in eligible_windows
    }
    common = {
        "tier": tier,
        "native_status": native_status,
        "required_dt2_seeds": sorted(required),
        "eligible_labels": [
            {"phenotype": phenotype, "direction": direction}
            for phenotype, direction in sorted(eligible_labels)
        ],
    }
    if dt2 is None:
        return {
            **common,
            "status": "indeterminate",
            "reason": "dt2_summary_missing",
        }
    if (
        not isinstance(dt2, dict)
        or dt2.get("schema") != C1_DT2_CONFIRMATION_SUMMARY_SCHEMA
        or dt2.get("resolution") != "dt2_confirmation_only"
        or dt2.get("phasec_manifest_sha256")
        != native.get("phasec_manifest_sha256")
        or dt2.get("phasec_manifest_file_sha256")
        != native.get("phasec_manifest_file_sha256")
    ):
        return {
            **common,
            "status": "blocked",
            "reason": "nonhomologous_or_invalid_dt2_summary",
        }

    matches = [
        row for row in dt2.get("matches", [])
        if row.get("tier") == tier
        and (
            str(row.get("phenotype")),
            str(row.get("direction")),
            tuple(row.get("homologous_cells") or []),
        ) in eligible_windows
        and required.issubset(
            set(row.get("homologous_supporting_seeds") or [])
        )
    ]
    if matches:
        return {
            **common,
            "status": "confirmed",
            "reason": "homologous_dt2_window_confirmed",
            "matches": matches,
        }

    label_rows = [
        row for row in dt2.get("label_assessments", [])
        if row.get("tier") == tier
        and (str(row.get("phenotype")), str(row.get("direction")))
        in eligible_labels
    ]
    technical = [
        row for row in dt2.get("technical_blockers", [])
        if (
            isinstance(row.get("key"), list)
            and len(row["key"]) >= 2
            and row["key"][1] == tier
        )
    ]
    closures = {str(row.get("closure")) for row in label_rows}
    if technical or "technical_block" in closures:
        return {
            **common,
            "status": "blocked",
            "reason": "dt2_layer_technical_block",
            "label_assessments": label_rows,
        }
    if label_rows and closures == {"terminal_contradiction"}:
        return {
            **common,
            "status": "contradicted",
            "reason": "native_window_not_reproduced_at_homologous_dt2",
            "label_assessments": label_rows,
        }
    return {
        **common,
        "status": "indeterminate",
        "reason": "dt2_layer_scientifically_indeterminate",
        "label_assessments": label_rows,
    }


def combine_resolution_summaries(native, dt2):
    """Apply independent preregistered dt/2 gates to primary and shell."""
    if native.get("schema") != C1_SUMMARY_SCHEMA:
        return {
            "schema": C1_RESOLUTION_GATE_SCHEMA,
            "verdict": "C1_blocked_resolution_gate",
            "reason": "missing_or_invalid_native_summary",
            "primary_gate": {
                "tier": "primary_convex",
                "status": "blocked",
                "reason": "missing_or_invalid_native_summary",
            },
            "shell_gate": {
                "tier": "secondary_shell",
                "status": "blocked",
                "reason": "missing_or_invalid_native_summary",
            },
        }
    primary_gate = _layer_resolution_gate(
        native, dt2,
        tier="primary_convex",
        adjudication_key="primary_adjudication",
    )
    shell_gate = _layer_resolution_gate(
        native, dt2,
        tier="secondary_shell",
        adjudication_key="secondary_shell_adjudication",
    )
    statuses = {
        primary_gate["status"], shell_gate["status"]
    }
    if primary_gate["status"] == "confirmed":
        verdict = "maturation_window_at_primary_convex_states"
        resolution_gate = "passed"
        reason = "primary_window_confirmed_at_homologous_dt2"
    elif shell_gate["status"] == "confirmed":
        verdict = "maturation_candidate_in_secondary_shell"
        resolution_gate = "passed"
        reason = "secondary_shell_window_confirmed_at_homologous_dt2"
    elif "blocked" in statuses:
        verdict = "C1_blocked_resolution_gate"
        blocked_reasons = [
            row.get("reason")
            for row in (primary_gate, shell_gate)
            if row.get("status") == "blocked"
        ]
        reason = str(blocked_reasons[0])
        resolution_gate = (
            "insufficient_homologous_native_support"
            if "resolution_confirmation_unavailable" in blocked_reasons
            else "layer_specific_resolution_block"
        )
    elif "indeterminate" in statuses:
        verdict = "C1_window_pending_dt2"
        resolution_gate = "scientifically_indeterminate_dt2_confirmation"
        reason = "dt2_scientifically_indeterminate"
    elif "contradicted" in statuses:
        verdict = "resolution_sensitive_maturation"
        resolution_gate = "passed"
        reason = "one_or_more_layer_windows_contradicted_at_homologous_dt2"
    else:
        verdict = native["verdict"]
        resolution_gate = "not_required_without_native_window"
        reason = "no_native_positive_window_requires_dt2"
    matches = [
        row
        for layer_gate in (primary_gate, shell_gate)
        for row in layer_gate.get("matches", [])
    ]
    return {
        "schema": C1_RESOLUTION_GATE_SCHEMA,
        "verdict": verdict,
        "resolution_gate": resolution_gate,
        "reason": reason,
        "primary_gate": primary_gate,
        "shell_gate": shell_gate,
        "matches": matches,
        "claim_boundary": native.get("claim_boundary"),
    }


def finalize_resolution_gate():
    native_path = OUT / "phasec1_summary_dt.json"
    dt2_path = OUT / "phasec1_dt2_confirmation_summary.json"
    native = _load_json(native_path) if native_path.is_file() else {}
    dt2 = _load_json(dt2_path) if dt2_path.is_file() else None
    payload = combine_resolution_summaries(native, dt2)
    payload.update({
        "native_summary_path": (
            _relative(native_path) if native_path.is_file() else None
        ),
        "native_summary_sha256": (
            _sha256(native_path) if native_path.is_file() else None
        ),
        "dt2_summary_path": (
            _relative(dt2_path) if dt2_path.is_file() else None
        ),
        "dt2_summary_sha256": (
            _sha256(dt2_path) if dt2_path.is_file() else None
        ),
    })
    path = OUT / "phasec1_resolution_gate.json"
    _write_json(path, payload)
    return payload


def analyze(resolution="dt"):
    if resolution not in RESOLUTIONS:
        raise ValueError(f"unknown resolution: {resolution}")
    if resolution == "dt2":
        raise ValueError(
            "dt2 is confirmation-only; lock and analyze the native-positive "
            "homologous subset instead of a full dt2 atlas"
        )
    base = build_base_atlas(resolution=resolution)
    base_path = OUT / f"phasec1_base_atlas_{resolution}.json"
    _write_json(base_path, base)
    summary = apply_conditional_gain(base, base_atlas_path=base_path)
    summary["resource_receipt_index"] = merge_resource_receipt_indexes(
        [base["resource_receipt_index"]] + [
            row.get("conditional_gain", {}).get(
                "resource_receipt_index"
            )
            for row in summary.get("cells", [])
            if row.get("conditional_gain", {}).get(
                "resource_receipt_index"
            ) is not None
        ],
        manifest_sha256=base["phasec_manifest_sha256"],
    )
    summary_path = OUT / f"phasec1_summary_{resolution}.json"
    _write_json(summary_path, summary)
    print(json.dumps({
        "resolution": resolution,
        "matrix_complete": base["matrix"]["complete"],
        "base_primary": base["primary_base_adjudication"]["status"],
        "base_shell": base["secondary_shell_base_adjudication"]["status"],
        "verdict": summary["verdict"],
        "base_atlas": _relative(base_path),
        "summary": _relative(summary_path),
    }, sort_keys=True))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=RESOLUTIONS, default="dt")
    parser.add_argument("--finalize-resolution", action="store_true")
    parser.add_argument("--lock-dt2-confirmation", action="store_true")
    parser.add_argument("--analyze-dt2-confirmation", action="store_true")
    parser.add_argument("--native-summary")
    parser.add_argument(
        "--dt2-confirmation-manifest",
        default=str(DT2_CONFIRMATION_MANIFEST),
    )
    args = parser.parse_args(argv)
    selected = sum((
        bool(args.finalize_resolution),
        bool(args.lock_dt2_confirmation),
        bool(args.analyze_dt2_confirmation),
    ))
    if selected > 1:
        raise SystemExit("choose only one dt2/finalization action")
    if args.finalize_resolution:
        print(json.dumps(finalize_resolution_gate(), sort_keys=True))
    elif args.lock_dt2_confirmation:
        payload = DT2LOCK.build_payload(
            phasec_path=PHASEC_MANIFEST,
            native_summary_path=(
                args.native_summary
                or OUT / "phasec1_summary_dt.json"
            ),
        )
        DT2LOCK._publish_once(
            Path(args.dt2_confirmation_manifest), payload
        )
        print(json.dumps({
            "status": "created",
            "manifest_sha256": payload["manifest_sha256"],
            "n_selected_cells": len(payload["selected_cells"]),
            "n_expected_arms": len(payload["expected_base_arms"]),
        }, sort_keys=True))
    elif args.analyze_dt2_confirmation:
        payload = analyze_dt2_confirmation(
            selection_manifest_path=Path(
                args.dt2_confirmation_manifest
            )
        )
        output = OUT / "phasec1_dt2_confirmation_summary.json"
        _write_json(output, payload)
        print(json.dumps({
            "verdict": payload["verdict"],
            "summary": _relative(output),
        }, sort_keys=True))
    else:
        analyze(args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
