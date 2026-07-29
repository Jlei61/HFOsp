#!/usr/bin/env python3
"""Build the seed-specific Phase-C modal audit from immutable artifacts.

No SNN is run here.  The representative manifest is a write-once routing
artifact prepared after the C0/C1 resolution gates close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_modal as M  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402


SELECTION_SCHEMA = "zm_phasec_modal_representatives_v1_2026-07-28"
OUTPUT_SCHEMA = "zm_phasec_modal_summary_v1_2026-07-28"
SELECTION_RULE = (
    "seed1_first_then_numeric_seed; accepted_C1_window_only_if_final_gate;"
    "otherwise_C0_identity; primary_before_shell; cell_phase_noise_"
    "lexicographic; completeness_and_provenance_only; no phenotype_"
    "strength_PSD_or_visual_selection"
)
DEFAULT_OUT = (
    ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
    / "phasec_seed_specific_modal.json"
)
DEFAULT_SELECTION = (
    ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
    / "phasec_modal_representatives.json"
)

C0_GATE_SCHEMA = "zm_phasec_c0_resolution_gate_v1"
C1_GATE_SCHEMA = "zm_phasec1_resolution_gate_v2_2026-07-29"
C0_POSITIVE = {
    "refractory_saturated_branch_supported",
    "balanced_AI_tonic_candidate_supported",
}
C0_NOT_REQUIRED = {
    "seed_heterogeneous_identity",
    "mixed_or_indeterminate_tonic_branch",
}
C1_POSITIVE = {
    "maturation_window_at_primary_convex_states",
    "maturation_candidate_in_secondary_shell",
}
C1_NOT_REQUIRED = {
    "no_maturation_in_tested_primary_neighbourhood",
    "isolated_maturation_candidate",
    "seed_heterogeneous_maturation",
    "C1_incomplete_or_indeterminate",
}


def _sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _self_hash(payload: Mapping[str, Any], *, label: str) -> None:
    claimed = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if not isinstance(claimed, str) or M.canonical_sha256(body) != claimed:
        raise ValueError(f"{label} self-hash mismatch")


def _semantic_npz_sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = np.ascontiguousarray(np.asarray(data[name]))
            h.update(
                f"{name}|{value.dtype.str}|{value.shape}|".encode("utf-8")
            )
            h.update(value.tobytes())
    return h.hexdigest()


def _artifact_semantic_sha(value: Mapping[str, Any]) -> str:
    return M.canonical_sha256(value)


def _production_task_key(part: Mapping[str, Any]) -> str:
    """Reconstruct the coordinator task key from one immutable part.

    Modal representatives can originate from either the C0 identity matrix or
    the native/dt2 C1 base matrix.  The receipt is only meaningful when its
    task identity is independently derivable from the selected artifact.
    """

    schema = part.get("schema")
    seed = int(part.get("seed", -1))
    if schema == "zm_phasec_identity_cell_v1":
        state = part.get("state_tag")
        noise = part.get("replicate")
        if seed < 0 or not isinstance(state, str) or not isinstance(noise, str):
            raise ValueError("selected C0 identity part lacks task-key fields")
        return f"identity|s{seed}|{state}|{noise}"
    if schema == "zm_phasec1_base_part_v1_2026-07-28":
        tier = part.get("tier")
        cell_id = part.get("cell_id")
        phase = part.get("phase")
        noise = part.get("noise")
        resolution = part.get("resolution")
        if (
            seed < 0
            or resolution not in {"dt", "dt2"}
            or not all(
                isinstance(value, str) and value
                for value in (tier, cell_id, phase, noise)
            )
        ):
            raise ValueError("selected C1 base part lacks task-key fields")
        prefix = "base" if resolution == "dt" else "dt2"
        return f"{prefix}|s{seed}|{tier}|{cell_id}|{phase}|{noise}"
    raise ValueError(f"unsupported modal production part schema: {schema!r}")


def _validate_and_lock_resource_receipt(
    part_path: Path,
    part: Mapping[str, Any],
    *,
    phasec_manifest_sha256: str,
) -> dict[str, str]:
    """Validate the adjacent receipt and return its immutable run lock."""

    task_key = _production_task_key(part)
    declared_manifest = (
        part.get("manifest_sha256")
        if part.get("schema") == "zm_phasec_identity_cell_v1"
        else part.get("phasec_manifest_sha256")
    )
    runtime = part.get("runtime_provenance")
    if (
        declared_manifest != phasec_manifest_sha256
        or not isinstance(runtime, Mapping)
        or runtime.get("manifest_sha256") != phasec_manifest_sha256
    ):
        raise ValueError(
            "selected modal part/receipt parent manifest provenance mismatch"
        )
    receipt_path = PRES.resource_receipt_path(part_path)
    ok, reason, receipt = PRES.validate_resource_receipt(
        receipt_path,
        artifact_path=part_path,
        artifact_root=ROOT,
        manifest_sha256=phasec_manifest_sha256,
        task_key=task_key,
    )
    if not ok or not isinstance(receipt, dict):
        raise ValueError(f"selected modal part resource receipt invalid: {reason}")
    return {
        "resource_task_key": task_key,
        "resource_receipt_path": _relative(receipt_path),
        "resource_receipt_file_sha256": _sha(receipt_path),
        "resource_receipt_sha256": str(receipt["receipt_sha256"]),
    }


def _validate_terminal_resolution_gates(
    c0_gate: Mapping[str, Any], c1_gate: Mapping[str, Any]
) -> None:
    """Fail closed before freezing write-once modal representatives."""
    c0_verdict = c0_gate.get("verdict")
    c0_resolution = c0_gate.get("resolution_gate")
    if c0_gate.get("schema") != C0_GATE_SCHEMA:
        raise ValueError("C0 resolution gate schema is not final")
    if c0_verdict in C0_POSITIVE:
        c0_terminal = c0_resolution == "passed"
    elif c0_verdict in C0_NOT_REQUIRED:
        c0_terminal = c0_resolution == "not_required_without_native_positive"
    else:
        c0_terminal = (
            c0_verdict == "resolution_sensitive_identity"
            and c0_resolution is None
        )
    if not c0_terminal:
        raise ValueError(
            "C0 resolution gate is pending, blocked, no-evidence, or not terminal"
        )

    if c1_gate.get("schema") != C1_GATE_SCHEMA:
        raise ValueError("C1 resolution gate schema is not final")
    layer_gates = [
        c1_gate.get("primary_gate"), c1_gate.get("shell_gate")
    ]
    c1_verdict = c1_gate.get("verdict")
    primary_gate, shell_gate = layer_gates
    structurally_closed = all(
        isinstance(row, Mapping)
        and row.get("status") in {
            "confirmed", "contradicted", "indeterminate",
            "blocked", "not_required",
        }
        for row in layer_gates
    )
    if c1_verdict == "maturation_window_at_primary_convex_states":
        c1_terminal = (
            structurally_closed
            and primary_gate.get("status") == "confirmed"
        )
    elif c1_verdict == "maturation_candidate_in_secondary_shell":
        c1_terminal = (
            structurally_closed
            and shell_gate.get("status") == "confirmed"
        )
    else:
        c1_terminal = (
            structurally_closed
            and all(
                row.get("status") in {"contradicted", "not_required"}
                for row in layer_gates
            )
        )
    if not c1_terminal:
        raise ValueError(
            "C1 resolution gate is pending, blocked, no-evidence, or not terminal"
        )


def _selection_producer_locks() -> dict[str, str]:
    return {
        str(Path(__file__).resolve().relative_to(CODE_ROOT)): _sha(
            Path(__file__).resolve()
        ),
        "src/topic4_zm_phasec_modal.py": _sha(
            CODE_ROOT / "src/topic4_zm_phasec_modal.py"
        ),
        "src/topic4_zm_modal_operator.py": _sha(
            CODE_ROOT / "src/topic4_zm_modal_operator.py"
        ),
        "src/topic4_zm_phasec_resources.py": _sha(
            CODE_ROOT / "src/topic4_zm_phasec_resources.py"
        ),
    }


def _relative(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _locked_path(ref: Mapping[str, Any], supplied: Path, *, label: str) -> None:
    expected = Path(str(ref.get("path", "")))
    if not expected.is_absolute():
        expected = ROOT / expected
    if supplied.resolve() != expected.resolve():
        raise ValueError(f"{label} path differs from representative lock")
    if not supplied.is_file() or _sha(supplied) != ref.get("file_sha256"):
        raise ValueError(f"{label} file SHA mismatch")


def _validate_live_producers(producers: Any, *, label: str) -> dict[str, str]:
    if not isinstance(producers, dict) or not producers:
        raise ValueError(f"{label} producer locks missing")
    for relative, expected in sorted(producers.items()):
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"{label} live producer drift: {relative}")
    return dict(producers)


def _seed_row(c0: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    rows = c0.get("seed_rows")
    if not isinstance(rows, list):
        raise ValueError("C0 native summary lacks seed_rows")
    found = [row for row in rows if int(row.get("seed", -1)) == int(seed)]
    if len(found) != 1:
        raise ValueError(f"C0 seed {seed} row is missing or duplicated")
    return found[0]


def _c1_cell(
    c1: Mapping[str, Any], seed: int, tier: str | None, cell_id: str | None
) -> Mapping[str, Any] | None:
    if cell_id is None:
        return None
    cells = c1.get("cells")
    if not isinstance(cells, list):
        raise ValueError("C1 native summary lacks cells")
    found = [
        row for row in cells
        if int(row.get("seed", -1)) == int(seed)
        and row.get("tier") == tier
        and row.get("cell_id") == cell_id
    ]
    if len(found) != 1:
        raise ValueError(f"C1 representative cell missing or duplicated: seed {seed}")
    return found[0]


def _load_observables(
    run: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_cell: str | None,
    phasec_manifest_sha256: str,
) -> dict[str, Any]:
    part_path = Path(str(run.get("part_path", "")))
    if not part_path.is_absolute():
        part_path = ROOT / part_path
    obs_path = Path(str(run.get("observables_path", "")))
    if not obs_path.is_absolute():
        obs_path = ROOT / obs_path
    if not part_path.is_file() or _sha(part_path) != run.get("part_file_sha256"):
        raise ValueError("representative part file SHA mismatch")
    part = _read(part_path)
    if _artifact_semantic_sha(part) != run.get("part_semantic_sha256"):
        raise ValueError("representative part semantic SHA mismatch")
    phase = part.get("phase", part.get("state_tag"))
    noise = part.get("noise", part.get("replicate"))
    if (
        int(part.get("seed", -1)) != int(expected_seed)
        or (expected_cell is not None and part.get("cell_id") != expected_cell)
        or phase != run.get("phase")
        or noise != run.get("noise")
        or part.get("observables_sha256") != run.get("observables_file_sha256")
    ):
        raise ValueError("representative part identity/provenance mismatch")
    current_receipt_lock = _validate_and_lock_resource_receipt(
        part_path,
        part,
        phasec_manifest_sha256=phasec_manifest_sha256,
    )
    locked_receipt = {
        key: run.get(key)
        for key in (
            "resource_task_key",
            "resource_receipt_path",
            "resource_receipt_file_sha256",
            "resource_receipt_sha256",
        )
    }
    if current_receipt_lock != locked_receipt:
        raise ValueError("representative resource receipt differs from run lock")
    part_observables = Path(str(part.get("observables_path", "")))
    if not part_observables.is_absolute():
        part_observables = ROOT / part_observables
    if part_observables.resolve() != obs_path.resolve():
        raise ValueError("representative observables path differs from part")
    if not obs_path.is_file() or _sha(obs_path) != run.get("observables_file_sha256"):
        raise ValueError("representative observables file SHA mismatch")
    if _semantic_npz_sha(obs_path) != run.get("observables_semantic_sha256"):
        raise ValueError("representative observables semantic SHA mismatch")
    runtime_producers = part.get("runtime_provenance", {}).get(
        "producer_sha256"
    )
    if runtime_producers != run.get("runtime_producer_file_sha256"):
        raise ValueError("representative runtime producer locks mismatch")
    with np.load(obs_path, allow_pickle=False) as data:
        required = {
            "phasec1_observables_schema", "bin_ms",
            "E_rate_grid", "I_rate_grid",
        }
        if not required.issubset(data.files):
            raise ValueError("representative observables lack modal fields")
        arrays = {
            "E_rate_grid": np.asarray(data["E_rate_grid"], dtype=float),
            "I_rate_grid": np.asarray(data["I_rate_grid"], dtype=float),
        }
        bin_ms = float(np.asarray(data["bin_ms"]).reshape(()).item())
    return {
        **dict(run),
        "observables": arrays,
        "bin_ms": bin_ms,
        "part_runtime_provenance": part.get("runtime_provenance"),
    }


def _input_ref(
    path: Path, value: Mapping[str, Any], *, phasec_manifest_sha256: str
) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "file_sha256": _sha(path),
        "semantic_sha256": _artifact_semantic_sha(value),
        "artifact_declared_manifest_sha256": (
            value.get("manifest_sha256")
            or value.get("phasec_manifest_sha256")
        ),
        "parent_phasec_manifest_sha256": phasec_manifest_sha256,
        "schema": value.get("schema"),
    }


def _accepted_c1_window(
    c1_gate: Mapping[str, Any],
    c1_native: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any] | None:
    verdict = c1_gate.get("verdict")
    if verdict == "maturation_window_at_primary_convex_states":
        tier, key = "primary_convex", "primary_adjudication"
    elif verdict == "maturation_candidate_in_secondary_shell":
        tier, key = "secondary_shell", "secondary_shell_adjudication"
    else:
        return None
    adjudication = c1_native.get(key)
    if not isinstance(adjudication, Mapping):
        return None
    gate_labels = {
        (str(row.get("phenotype")), str(row.get("direction")))
        for row in c1_gate.get("matches", [])
        if isinstance(row, Mapping)
        and row.get("tier") == tier
        and row.get("phenotype") in M.PERIODIC_PHENOTYPES
    }
    if len(gate_labels) != 1:
        return None
    windows = (
        adjudication.get("seed_results", {})
        .get(str(int(seed)), {})
        .get("windows", [])
    )
    labels = {
        (str(row.get("phenotype")), str(row.get("direction")))
        for row in windows
        if isinstance(row, Mapping)
        and isinstance(row.get("cells"), list)
        and row.get("phenotype") in M.PERIODIC_PHENOTYPES
        and (
            str(row.get("phenotype")),
            str(row.get("direction")),
        ) in gate_labels
    }
    if len(labels) != 1:
        return None
    phenotype, direction = next(iter(labels))
    cells = sorted({
        str(cell)
        for row in windows
        if (
            str(row.get("phenotype")),
            str(row.get("direction")),
        ) == (phenotype, direction)
        for cell in row["cells"]
    })
    return {
        "tier": tier,
        "phenotype": phenotype,
        "direction": direction,
        "cell_ids": cells,
    }


def _candidate_cells(
    c1_native: Mapping[str, Any],
    *,
    seed: int,
    accepted_window: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    cells = [
        row for row in c1_native.get("cells", [])
        if int(row.get("seed", -1)) == int(seed)
        and row.get("status") == "complete"
    ]
    if accepted_window is not None:
        allowed = set(accepted_window["cell_ids"])
        cells = [
            row for row in cells
            if row.get("tier") == accepted_window["tier"]
            and row.get("cell_id") in allowed
        ]
    return sorted(
        cells,
        key=lambda row: (
            0 if row.get("tier") == "primary_convex" else 1,
            str(row.get("cell_id")),
        ),
    )


def _lock_representative_run(
    row: Mapping[str, Any],
    *,
    seed: int,
    cell_id: str,
    phasec_manifest_sha256: str,
) -> dict[str, Any]:
    part_path = Path(str(row.get("part_path", "")))
    if not part_path.is_absolute():
        part_path = ROOT / part_path
    if (
        not part_path.is_file()
        or _sha(part_path) != row.get("part_sha256")
    ):
        raise ValueError("C1 summary part lock drift while selecting modal run")
    part = _read(part_path)
    if (
        part.get("status") != "complete"
        or int(part.get("seed", -1)) != int(seed)
        or part.get("cell_id") != cell_id
        or part.get("phase") != row.get("phase")
        or part.get("noise") != row.get("noise")
    ):
        raise ValueError("selected modal part is not the locked complete C1 run")
    obs_path = Path(str(part.get("observables_path", "")))
    if not obs_path.is_absolute():
        obs_path = ROOT / obs_path
    if (
        not obs_path.is_file()
        or _sha(obs_path) != part.get("observables_sha256")
    ):
        raise ValueError("selected modal observables drift from C1 part")
    producers = part.get("runtime_provenance", {}).get("producer_sha256")
    if not isinstance(producers, dict) or not producers:
        raise ValueError("selected modal part lacks runtime producer locks")
    receipt_lock = _validate_and_lock_resource_receipt(
        part_path,
        part,
        phasec_manifest_sha256=phasec_manifest_sha256,
    )
    return {
        "phase": str(row["phase"]),
        "noise": str(row["noise"]),
        "part_path": _relative(part_path),
        "part_file_sha256": _sha(part_path),
        "part_semantic_sha256": _artifact_semantic_sha(part),
        "observables_path": _relative(obs_path),
        "observables_file_sha256": _sha(obs_path),
        "observables_semantic_sha256": _semantic_npz_sha(obs_path),
        "runtime_producer_file_sha256": producers,
        **receipt_lock,
    }


def _run_triplet(
    cell: Mapping[str, Any],
    *,
    seed: int,
    phasec_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], str]:
    complete = [
        row for row in cell.get("run_rows", [])
        if row.get("status") == "complete"
        and isinstance(row.get("part_path"), str)
        and isinstance(row.get("part_sha256"), str)
    ]
    phases = sorted({str(row.get("phase")) for row in complete})
    for phase in phases:
        by_noise = {}
        for row in sorted(
            (item for item in complete if str(item.get("phase")) == phase),
            key=lambda item: str(item.get("noise")),
        ):
            noise = str(row.get("noise"))
            if noise in by_noise:
                raise ValueError(
                    f"seed {seed} cell {cell.get('cell_id')} duplicates {phase}/{noise}"
                )
            by_noise[noise] = row
        if len(by_noise) < 3:
            continue
        selected_noise = sorted(by_noise)[:3]
        locked = [
            _lock_representative_run(
                by_noise[noise],
                seed=seed,
                cell_id=str(cell["cell_id"]),
                phasec_manifest_sha256=phasec_manifest_sha256,
            )
            for noise in selected_noise
        ]
        for index, row in enumerate(locked):
            row["role"] = "fit" if index < 2 else "noise_heldout"
        return locked, (
            "primary_tier_then_cell_phase_noise_lexicographic_complete_triplet"
        )
    return [], "no_phase_has_three_complete_unique_noise_continuations"


def _locked_period_ms(
    cell: Mapping[str, Any], *, phenotype: str, selected_phase: str | None
) -> float | None:
    if phenotype not in M.PERIODIC_PHENOTYPES:
        return None
    if phenotype == "periodic_non_tonic_carrier":
        value = (
            cell.get("periodic_fast_phase_consistency", {})
            .get("per_phase_median_period_ms", {})
            .get(selected_phase)
        )
        if value is not None:
            return float(value)
    key = "periodic" if phenotype == "periodic_non_tonic_carrier" else "clonic"
    values = [
        row.get("phenotype", {})
        .get("temporal_diagnostics", {})
        .get(key, {})
        .get("median_period_ms")
        for row in cell.get("run_rows", [])
        if row.get("status") == "complete"
        and (selected_phase is None or row.get("phase") == selected_phase)
    ]
    finite = [
        float(value) for value in values
        if value is not None and np.isfinite(float(value)) and float(value) > 0
    ]
    return float(np.median(finite)) if finite else None


def build_representative_manifest(
    *,
    phasec_manifest_path: Path,
    c0_gate_path: Path,
    c0_native_path: Path,
    c1_gate_path: Path,
    c1_native_path: Path,
) -> dict[str, Any]:
    """Deterministically lock modal inputs without inspecting their waveforms."""

    phasec = _read(phasec_manifest_path)
    PCC.validate_manifest(phasec)
    if phasec.get("production_authorized") is not True:
        raise ValueError("representative lock requires final Phase-C authority")
    c0_gate = _read(c0_gate_path)
    c0_native = _read(c0_native_path)
    c1_gate = _read(c1_gate_path)
    c1_native = _read(c1_native_path)
    _validate_terminal_resolution_gates(c0_gate, c1_gate)
    supplied = {
        "phasec_manifest": (phasec_manifest_path, phasec),
        "c0_resolution_gate": (c0_gate_path, c0_gate),
        "c0_native_summary": (c0_native_path, c0_native),
        "c1_resolution_gate": (c1_gate_path, c1_gate),
        "c1_native_summary": (c1_native_path, c1_native),
    }
    seed_rows = sorted(
        c0_native.get("seed_rows", []),
        key=lambda row: (
            0 if int(row.get("seed", -1)) == 1 else 1,
            int(row.get("seed", -1)),
        ),
    )
    if not seed_rows:
        raise ValueError("C0 native summary contains no seed rows")
    selections = {}
    for c0_row in seed_rows:
        seed = int(c0_row["seed"])
        accepted_window = _accepted_c1_window(
            c1_gate, c1_native, seed=seed
        )
        source = "C1_cell" if accepted_window is not None else "C0_seed"
        phenotype = (
            accepted_window["phenotype"]
            if accepted_window is not None else c0_row.get("klass")
        )
        cells = _candidate_cells(
            c1_native, seed=seed, accepted_window=accepted_window
        )
        cell = cells[0] if cells else None
        runs, selection_reason = (
            _run_triplet(
                cell,
                seed=seed,
                phasec_manifest_sha256=phasec["manifest_sha256"],
            )
            if cell is not None else ([], "no_complete_C1_representative_cell")
        )
        selected_phase = runs[0]["phase"] if runs else None
        route = M.phenotype_route(str(phenotype))
        sensitivity = None
        if route["route"] == "saturated_sensitivity_only":
            ci = c0_row.get("hierarchical_ci", {})
            sensitivity = {
                "gain_relative_to_preentry": ci.get(
                    "gain_relative_to_preentry"
                ),
                "refractory_isi_fraction": ci.get(
                    "refractory_isi_fraction"
                ),
                "rho80_active_core": ci.get("rho80_active_core"),
                "source": "locked_C0_hierarchical_ci",
            }
        selections[str(seed)] = {
            "phenotype": phenotype,
            "phenotype_source": source,
            "route": route["route"],
            "tier": None if cell is None else cell.get("tier"),
            "cell_id": None if cell is None else cell.get("cell_id"),
            "pathology_axis_deg": float(c0_row.get("pathology_axis_deg", 0.0)),
            "pathology_axis_source": (
                "C0_seed_row" if "pathology_axis_deg" in c0_row
                else "zero_coordinate_reference_axis_not_pathology_claim"
            ),
            "period_ms": (
                None if cell is None else _locked_period_ms(
                    cell,
                    phenotype=str(phenotype),
                    selected_phase=selected_phase,
                )
            ),
            "locked_sensitivity": sensitivity,
            "runs": runs,
            "selection_reason": selection_reason,
        }
    body = {
        "schema": SELECTION_SCHEMA,
        "selection_is_closed": True,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "heldout_contract": M.HELDOUT_CONTRACT,
        "selection_rule": SELECTION_RULE,
        "inputs": {
            name: {
                "path": _relative(path),
                "file_sha256": _sha(path),
                "semantic_sha256": _artifact_semantic_sha(value),
                "manifest_sha256": (
                    value.get("manifest_sha256")
                    or value.get("phasec_manifest_sha256")
                ),
                "parent_phasec_manifest_sha256": phasec["manifest_sha256"],
            }
            for name, (path, value) in supplied.items()
        },
        "producer_file_sha256": _selection_producer_locks(),
        "seeds": selections,
    }
    return {**body, "manifest_sha256": M.canonical_sha256(body)}


def analyze(
    *,
    phasec_manifest_path: Path,
    c0_gate_path: Path,
    c0_native_path: Path,
    c1_gate_path: Path,
    c1_native_path: Path,
    representatives_path: Path,
) -> dict[str, Any]:
    phasec = _read(phasec_manifest_path)
    PCC.validate_manifest(phasec)
    if phasec.get("production_authorized") is not True:
        raise ValueError("modal audit requires the final Phase-C manifest")
    c0_gate = _read(c0_gate_path)
    c0_native = _read(c0_native_path)
    c1_gate = _read(c1_gate_path)
    c1_native = _read(c1_native_path)
    selection = _read(representatives_path)
    _self_hash(selection, label="modal representative manifest")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("selection_is_closed") is not True
        or selection.get("selection_rule") != SELECTION_RULE
        or selection.get("heldout_contract") != M.HELDOUT_CONTRACT
        or selection.get("phasec_manifest_sha256")
        != phasec.get("manifest_sha256")
    ):
        raise ValueError("modal representative manifest is not closed")

    locked_inputs = selection.get("inputs")
    if not isinstance(locked_inputs, dict):
        raise ValueError("modal representative manifest lacks input locks")
    supplied = {
        "phasec_manifest": phasec_manifest_path,
        "c0_resolution_gate": c0_gate_path,
        "c0_native_summary": c0_native_path,
        "c1_resolution_gate": c1_gate_path,
        "c1_native_summary": c1_native_path,
    }
    supplied_values = {
        "phasec_manifest": phasec,
        "c0_resolution_gate": c0_gate,
        "c0_native_summary": c0_native,
        "c1_resolution_gate": c1_gate,
        "c1_native_summary": c1_native,
    }
    for name, path in supplied.items():
        if name not in locked_inputs:
            raise ValueError(f"modal representative manifest lacks {name} lock")
        _locked_path(locked_inputs[name], path, label=name)
        if (
            locked_inputs[name].get("semantic_sha256")
            != _artifact_semantic_sha(supplied_values[name])
        ):
            raise ValueError(f"{name} semantic SHA mismatch")
        if (
            locked_inputs[name].get("parent_phasec_manifest_sha256")
            != phasec.get("manifest_sha256")
        ):
            raise ValueError(f"{name} parent Phase-C manifest SHA mismatch")
    if (
        locked_inputs["phasec_manifest"].get("manifest_sha256")
        != phasec.get("manifest_sha256")
    ):
        raise ValueError("Phase-C manifest semantic identity mismatch")
    selection_producers = _validate_live_producers(
        selection.get("producer_file_sha256"), label="modal selection"
    )
    if selection_producers != _selection_producer_locks():
        raise ValueError("modal selection producer map is not the locked builder")
    phasec_producers = _validate_live_producers(
        phasec.get("provenance", {}).get("producer_file_sha256"),
        label="Phase-C",
    )

    selections = selection.get("seeds")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("modal representative manifest has no seeds")
    seed_results = []
    representative_provenance = {}
    for seed_text, selected in sorted(selections.items(), key=lambda item: int(item[0])):
        seed = int(seed_text)
        c0_row = _seed_row(c0_native, seed)
        cell = _c1_cell(
            c1_native, seed, selected.get("tier"), selected.get("cell_id")
        )
        route = M.derive_seed_route(
            seed=seed,
            c0_seed_row=c0_row,
            c1_cell=cell,
            selected_phenotype=str(selected.get("phenotype")),
            phenotype_source=str(
                selected.get("phenotype_source", "auto")
            ),
        )
        requested = selected.get("route")
        if requested is not None and requested != route["route"]:
            raise ValueError(f"seed {seed}: selected modal route contradicts phenotype")

        loaded_runs = []
        run_provenance = []
        for run in selected.get("runs", []):
            loaded = _load_observables(
                run,
                expected_seed=seed,
                expected_cell=selected.get("cell_id"),
                phasec_manifest_sha256=phasec["manifest_sha256"],
            )
            loaded_runs.append(loaded)
            run_provenance.append({
                key: loaded.get(key)
                for key in (
                    "phase", "noise", "role", "part_path",
                    "part_file_sha256", "part_semantic_sha256",
                    "observables_path", "observables_file_sha256",
                    "observables_semantic_sha256",
                    "runtime_producer_file_sha256",
                    "resource_task_key", "resource_receipt_path",
                    "resource_receipt_file_sha256",
                    "resource_receipt_sha256",
                    "part_runtime_provenance",
                )
            })
        bin_values = {float(row["bin_ms"]) for row in loaded_runs}
        if len(bin_values) > 1:
            raise ValueError(f"seed {seed}: representative bin_ms mismatch")
        bin_ms = (
            next(iter(bin_values))
            if bin_values else float(selected.get("bin_ms", 2.0))
        )
        result = M.analyze_seed(
            route,
            loaded_runs,
            bin_ms=bin_ms,
            pathology_axis_deg=float(selected.get("pathology_axis_deg", 0.0)),
            period_ms=selected.get("period_ms"),
            locked_sensitivity=selected.get("locked_sensitivity"),
            maximum_rank=int(selected.get("maximum_rank", 8)),
        )
        if result["input_phenotype"] != selected.get("phenotype"):
            raise RuntimeError("modal analysis attempted a phenotype override")
        seed_results.append(result)
        representative_provenance[str(seed)] = run_provenance

    aggregate = M.aggregate_seed_modal(seed_results)
    return {
        "schema": OUTPUT_SCHEMA,
        **aggregate,
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "c0_final_verdict": c0_gate.get("verdict"),
        "c1_final_verdict": c1_gate.get("verdict"),
        "input_provenance": {
            name: _input_ref(
                path, value,
                phasec_manifest_sha256=phasec["manifest_sha256"],
            )
            for (name, path), value in zip(
                supplied.items(),
                (phasec, c0_gate, c0_native, c1_gate, c1_native),
            )
        },
        "representatives_manifest": {
            "path": _relative(representatives_path),
            "file_sha256": _sha(representatives_path),
            "manifest_sha256": selection["manifest_sha256"],
            "producer_file_sha256": selection_producers,
        },
        "phasec_producer_file_sha256": phasec_producers,
        "representative_run_provenance": representative_provenance,
        "analysis_producer_file_sha256": {
            _relative(Path(__file__)): _sha(Path(__file__)),
            str((CODE_ROOT / "src/topic4_zm_phasec_modal.py").relative_to(CODE_ROOT)): _sha(
                CODE_ROOT / "src/topic4_zm_phasec_modal.py"
            ),
            str((CODE_ROOT / "src/topic4_zm_modal_operator.py").relative_to(CODE_ROOT)): _sha(
                CODE_ROOT / "src/topic4_zm_modal_operator.py"
            ),
            str((CODE_ROOT / "src/topic4_zm_phasec_resources.py").relative_to(CODE_ROOT)): _sha(
                CODE_ROOT / "src/topic4_zm_phasec_resources.py"
            ),
        },
        "claim_boundary": (
            "seed-specific observational operator audit only; cannot alter "
            "C0/C1 phenotype and does not test entry, offset, recovery, or lifecycle"
        ),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return
        raise RuntimeError(f"refusing to overwrite different modal summary: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phasec-manifest", type=Path, required=True)
    parser.add_argument("--c0-resolution-gate", type=Path, required=True)
    parser.add_argument("--c0-native-summary", type=Path, required=True)
    parser.add_argument("--c1-resolution-gate", type=Path, required=True)
    parser.add_argument("--c1-native-summary", type=Path, required=True)
    parser.add_argument(
        "--representatives-manifest", type=Path, default=DEFAULT_SELECTION
    )
    parser.add_argument("--lock-representatives", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.lock_representatives:
        payload = build_representative_manifest(
            phasec_manifest_path=args.phasec_manifest,
            c0_gate_path=args.c0_resolution_gate,
            c0_native_path=args.c0_native_summary,
            c1_gate_path=args.c1_resolution_gate,
            c1_native_path=args.c1_native_summary,
        )
        _atomic_write(args.representatives_manifest, payload)
        print(json.dumps({
            "selection_is_closed": payload["selection_is_closed"],
            "manifest_sha256": payload["manifest_sha256"],
            "output": str(args.representatives_manifest),
        }, sort_keys=True))
        return
    payload = analyze(
        phasec_manifest_path=args.phasec_manifest,
        c0_gate_path=args.c0_resolution_gate,
        c0_native_path=args.c0_native_summary,
        c1_gate_path=args.c1_resolution_gate,
        c1_native_path=args.c1_native_summary,
        representatives_path=args.representatives_manifest,
    )
    _atomic_write(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "class_disagreement": payload["class_disagreement"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
