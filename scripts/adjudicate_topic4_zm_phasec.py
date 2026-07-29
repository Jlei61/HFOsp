#!/usr/bin/env python3
"""Re-hash immutable Phase-C summaries and call the pure adjudicator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.topic4_zm_phasec_verdict as V  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402
import scripts.analyze_topic4_zm_phasec0 as C0  # noqa: E402
import scripts.analyze_topic4_zm_phasec1 as C1  # noqa: E402


OUTPUT_SCHEMA = "zm_phasec_final_adjudication_v1_2026-07-28"
FINAL_INPUT_SCHEMA = "zm_phasec_final_input_v1_2026-07-28"
FINAL_INPUT_FILENAMES = {
    "c0": "phasec_final_input_c0.json",
    "c1_primary": "phasec_final_input_c1_primary.json",
    "c1_shell": "phasec_final_input_c1_shell.json",
    "coverage": "phasec_final_input_coverage.json",
}
C0_GATE_SCHEMA = "zm_phasec_c0_resolution_gate_v1"
C0_SUMMARY_SCHEMA = "zm_phasec_c0_summary_v1"
C1_GATE_SCHEMA = "zm_phasec1_resolution_gate_v2_2026-07-29"
C1_SUMMARY_SCHEMA = "zm_phasec1_summary_v1_2026-07-28"
MODAL_SUMMARY_SCHEMA = "zm_phasec_modal_summary_v1_2026-07-28"
RESOURCE_RECEIPT_INDEX_SCHEMA = C0.RESOURCE_RECEIPT_INDEX_SCHEMA


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _source_ref(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "file_sha256": _sha(path),
        "artifact_sha256": V.artifact_sha256(value),
        "schema": value.get("schema"),
        "declared_manifest_sha256": (
            value.get("manifest_sha256")
            or value.get("phasec_manifest_sha256")
        ),
    }


def _validate_modal_input_provenance(
    modal: Mapping[str, Any],
    supplied: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    phasec_manifest_sha256: str,
) -> None:
    """Require the modal audit and final adjudicator to consume one source set."""
    locked = modal.get("input_provenance")
    if not isinstance(locked, Mapping):
        raise ValueError("modal summary lacks locked input provenance")
    for name, (path, value) in supplied.items():
        row = locked.get(name)
        if not isinstance(row, Mapping):
            raise ValueError(f"modal summary lacks {name} input provenance")
        if (
            row.get("file_sha256") != _sha(path)
            or row.get("semantic_sha256") != _canonical_sha(value)
            or row.get("schema") != value.get("schema")
            or row.get("parent_phasec_manifest_sha256")
            != phasec_manifest_sha256
        ):
            raise ValueError(f"modal/{name} input provenance mismatch")


def _producer_locks() -> dict[str, str]:
    return {
        str(Path(__file__).resolve().relative_to(ROOT)): _sha(
            Path(__file__).resolve()
        ),
        "src/topic4_zm_phasec_verdict.py": _sha(
            ROOT / "src/topic4_zm_phasec_verdict.py"
        ),
    }


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _live_resource_ref(
    part_path: Path, *, task_key: str, manifest_sha256: str
) -> dict[str, Any]:
    """Independently reconstruct one part/aux/receipt binding."""
    part_path = Path(part_path)
    part = _read(part_path)
    receipt_path = PRES.resource_receipt_path(part_path)
    ok, reason, receipt = PRES.validate_resource_receipt(
        receipt_path,
        artifact_path=part_path,
        artifact_root=ROOT,
        manifest_sha256=manifest_sha256,
        task_key=task_key,
    )
    if not ok or not isinstance(receipt, dict):
        raise ValueError(reason)
    ref = {
        "task_key": str(task_key),
        "part_path": os.path.relpath(part_path, ROOT),
        "part_file_sha256": _sha(part_path),
        "resource_receipt_path": os.path.relpath(receipt_path, ROOT),
        "resource_receipt_file_sha256": _sha(receipt_path),
        "resource_receipt_sha256": receipt["receipt_sha256"],
    }
    aux_ref = part.get("observables_path")
    aux_sha = part.get("observables_sha256")
    if aux_ref is not None or aux_sha is not None:
        aux_path = _resolve(str(aux_ref))
        if (
            not aux_path.is_file()
            or not isinstance(aux_sha, str)
            or _sha(aux_path) != aux_sha
        ):
            raise ValueError("resource_index_aux_observables_drift")
        ref.update({
            "aux_observables_path": os.path.relpath(aux_path, ROOT),
            "aux_observables_file_sha256": aux_sha,
        })
    return ref


def _c1_base_resource_tasks(
    phasec: Mapping[str, Any], *, resolution: str
) -> list[tuple[str, Path, str]]:
    _, coordinate, _ = C1._coordinate_path_from_final(phasec, resolution)
    seeds = C1.SEEDS if resolution == "dt" else C1.DT2_SEEDS
    cells = C1._cell_inventory(coordinate, expected_seeds=seeds)
    return [
        (
            C1._base_task_key(
                resolution=resolution,
                seed=seed,
                tier=tier,
                cell_id=cell_id,
                phase=phase,
                noise=noise,
            ),
            C1.base_part_path(
                resolution, seed, tier, cell_id, phase, noise
            ),
            "c1_base" if resolution == "dt" else "c1_dt2_base",
        )
        for (seed, tier, cell_id), cell in sorted(cells.items())
        if cell.get("status") == "valid"
        for phase in C1.PHASES for noise in C1.NOISES
    ]


def _c1_gain_resource_tasks(
    summary: Mapping[str, Any],
) -> list[tuple[str, Path, str]]:
    claimed = summary.get("gain_trigger_manifest_sha256")
    if claimed is None:
        return []
    trigger_path = C1.GAIN_TRIGGER_MANIFEST
    trigger = _read(trigger_path)
    C1._validate_self_hash(trigger, label="C1 gain trigger manifest")
    if trigger.get("manifest_sha256") != claimed:
        raise ValueError("C1 summary/gain-trigger manifest mismatch")
    tasks = []
    for selected in trigger.get("triggered_cells", []):
        role_suffix = (
            f"s{selected['seed']}|{selected['tier']}|{selected['cell_id']}"
        )
        for ref in selected.get("expected_carrier_gain_arms", []):
            tasks.append((
                (
                    f"gain|s{ref['seed']}|{ref['tier']}|"
                    f"{ref['cell_id']}|{ref['phase']}|{ref['noise']}|"
                    f"{float(ref['delta_mV']):+g}"
                ),
                ROOT / ref["path"],
                "c1_gain_numerator|" + role_suffix,
            ))
        for ref in selected.get("reused_c0_preentry_denominators", []):
            tasks.append((
                C0._gain_task_key(
                    int(ref["seed"]),
                    ref["state_tag"],
                    ref["replicate"],
                    float(ref["signed_delta_abs_mV"]),
                    int(ref["sign"]),
                ),
                ROOT / ref["path"],
                "c1_gain_preentry_denominator|" + role_suffix,
            ))
    return tasks


def _expected_resource_tasks(
    kind: str,
    summary: Mapping[str, Any],
    phasec: Mapping[str, Any],
) -> list[tuple[str, Path, str]]:
    if kind == "c0_native":
        return C0.expected_resource_tasks("dt", C0.SEEDS)
    if kind == "c0_dt2":
        return C0.expected_resource_tasks("dt2", C1.DT2_SEEDS)
    if kind == "c1_native":
        return (
            _c1_base_resource_tasks(phasec, resolution="dt")
            + _c1_gain_resource_tasks(summary)
        )
    if kind == "c1_dt2":
        selection_path = _resolve(summary["selection_manifest_path"])
        selection = _read(selection_path)
        return [
            (
                C1._base_task_key(
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
            for arm in selection.get("expected_base_arms", [])
        ]
    raise ValueError(f"unknown resource-index summary kind: {kind}")


def _resource_index_issues(
    kind: str,
    summary: Mapping[str, Any],
    phasec: Mapping[str, Any],
) -> list[str]:
    """Rebuild the expected logical set and re-open every live binding."""
    index = summary.get("resource_receipt_index")
    if not isinstance(index, Mapping):
        return [f"{kind}:missing_resource_receipt_index"]
    body = {
        key: value for key, value in index.items()
        if key != "index_sha256"
    }
    manifest_sha = phasec["manifest_sha256"]
    issues = []
    if (
        index.get("schema") != RESOURCE_RECEIPT_INDEX_SCHEMA
        or index.get("manifest_sha256") != manifest_sha
        or index.get("index_sha256") != _canonical_sha(body)
        or index.get("status") != "complete"
        or index.get("issues") != []
    ):
        issues.append(f"{kind}:invalid_or_incomplete_resource_receipt_index")
    try:
        expected_rows = _expected_resource_tasks(kind, summary, phasec)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return issues + [f"{kind}:cannot_rebuild_expected_resource_set:{exc}"]
    expected_entries = {}
    expected_logical = []
    for task_key, path, role in expected_rows:
        task_key = str(task_key)
        expected_logical.append({"task_key": task_key, "role": str(role)})
        prior = expected_entries.get(task_key)
        resolved = Path(path)
        if prior is not None and prior.resolve() != resolved.resolve():
            issues.append(f"{kind}:conflicting_expected_task_path:{task_key}")
        expected_entries[task_key] = resolved
    actual_entries = index.get("entries")
    logical = index.get("logical_consumptions")
    if not isinstance(actual_entries, list):
        return issues + [f"{kind}:invalid_resource_entry_list"]
    actual_by_key = {
        row.get("task_key"): row
        for row in actual_entries if isinstance(row, Mapping)
    }
    if (
        len(actual_by_key) != len(actual_entries)
        or set(actual_by_key) != set(expected_entries)
        or index.get("expected_task_count") != len(expected_entries)
        or index.get("validated_entry_count") != len(actual_entries)
    ):
        issues.append(f"{kind}:resource_unique_task_set_mismatch")
    if (
        not isinstance(logical, list)
        or index.get("expected_logical_consumption_count")
        != len(expected_logical)
        or sorted(
            (
                str(row.get("task_key")),
                str(row.get("role")),
            )
            for row in logical if isinstance(row, Mapping)
        )
        != sorted(
            (row["task_key"], row["role"]) for row in expected_logical
        )
    ):
        issues.append(f"{kind}:resource_logical_consumption_set_mismatch")
    for task_key, path in sorted(expected_entries.items()):
        row = actual_by_key.get(task_key)
        if not isinstance(row, Mapping):
            continue
        if _resolve(str(row.get("part_path", ""))).resolve() != path.resolve():
            issues.append(f"{kind}:resource_part_path_mismatch:{task_key}")
            continue
        try:
            live = _live_resource_ref(
                path, task_key=task_key, manifest_sha256=manifest_sha
            )
        except (OSError, TypeError, ValueError) as exc:
            issues.append(f"{kind}:resource_live_binding_failure:{task_key}:{exc}")
            continue
        if dict(row) != live:
            issues.append(f"{kind}:resource_live_binding_mismatch:{task_key}")
    return issues


def _resource_source_ref(
    kind: str, path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "kind": kind,
        **_source_ref(path, value),
    }


def _audit_resource_sources(
    sources: Mapping[str, Any],
    phasec: Mapping[str, Any],
) -> list[str]:
    issues = []
    for name, ref in sorted(sources.items()):
        if not isinstance(ref, Mapping):
            issues.append(f"{name}:resource_summary_ref_missing")
            continue
        path = _resolve(str(ref.get("path", "")))
        if not path.is_file() or ref.get("file_sha256") != _sha(path):
            issues.append(f"{name}:resource_summary_file_drift")
            continue
        try:
            summary = _read(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"{name}:resource_summary_unreadable:{exc}")
            continue
        issues.extend(_resource_index_issues(
            str(ref.get("kind")), summary, phasec
        ))
    return issues


def _final_resource_source_contract(
    artifacts: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
) -> list[str]:
    """Tie the compact index sources to the exact final-input source graph."""
    issues = []
    coverage = artifacts.get("coverage", {})
    provenance = coverage.get("source_provenance")
    sources = resource_audit.get("summary_sources")
    if not isinstance(provenance, Mapping) or not isinstance(sources, Mapping):
        return ["resource_receipt_source_contract_missing"]
    expected = {}
    for key, source_name, kind in (
        ("c0_native", "c0_native_summary", "c0_native"),
        ("c1_native", "c1_native_summary", "c1_native"),
    ):
        ref = provenance.get(source_name)
        if not isinstance(ref, Mapping):
            issues.append(f"resource_receipt_source_missing:{source_name}")
            continue
        expected[key] = {
            "kind": kind,
            "path": ref.get("path"),
            "file_sha256": ref.get("file_sha256"),
        }
    for gate_name, dt2_key, field in (
        ("c0_resolution_gate", "c0_dt2", "dt2_summary"),
        ("c1_resolution_gate", "c1_dt2", "dt2_summary_path"),
    ):
        gate_ref = provenance.get(gate_name)
        if not isinstance(gate_ref, Mapping):
            issues.append(f"resource_receipt_source_missing:{gate_name}")
            continue
        gate_path = _resolve(str(gate_ref.get("path", "")))
        if (
            not gate_path.is_file()
            or gate_ref.get("file_sha256") != _sha(gate_path)
        ):
            issues.append(f"resource_receipt_gate_source_drift:{gate_name}")
            continue
        gate = _read(gate_path)
        dt2_path_ref = gate.get(field)
        if dt2_path_ref is not None:
            dt2_path = _resolve(str(dt2_path_ref))
            sha_field = (
                "dt2_summary_sha256"
                if gate_name == "c0_resolution_gate"
                else "dt2_summary_sha256"
            )
            expected[dt2_key] = {
                "kind": dt2_key,
                "path": str(dt2_path),
                "file_sha256": gate.get(sha_field),
            }
    if set(sources) != set(expected):
        issues.append("resource_receipt_summary_source_set_mismatch")
    for key, wanted in expected.items():
        got = sources.get(key)
        if not isinstance(got, Mapping):
            continue
        if (
            got.get("kind") != wanted["kind"]
            or _resolve(str(got.get("path", ""))).resolve()
            != _resolve(str(wanted["path"])).resolve()
            or got.get("file_sha256") != wanted["file_sha256"]
        ):
            issues.append(f"resource_receipt_summary_source_mismatch:{key}")
    for name in ("c0", "c1_primary", "c1_shell", "coverage"):
        artifact = artifacts.get(name)
        if (
            isinstance(artifact, Mapping)
            and artifact.get("resource_receipt_audit") != resource_audit
        ):
            issues.append(f"resource_receipt_audit_cross_input_mismatch:{name}")
    return issues


def _c0_final_input(
    gate: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    verdict = gate.get("verdict")
    native_aggregate = native.get("aggregate")
    if not isinstance(native_aggregate, Mapping):
        native_aggregate = {}
    seed_classes = (
        gate.get("seed_classes")
        or native_aggregate.get("seed_classes")
        or {}
    )
    known_complete = {
        "refractory_saturated_branch_supported",
        "balanced_AI_tonic_candidate_supported",
        "mixed_or_indeterminate_tonic_branch",
        "seed_heterogeneous_identity",
    }
    if verdict == "resolution_sensitive_identity":
        coverage = "resolution_sensitive"
    elif verdict in {"C0_blocked_observables"}:
        coverage = "blocked_observables"
    elif verdict in known_complete:
        coverage = "complete"
    elif verdict == "C0_no_evidence":
        verdict = "C0_insufficient_coverage"
        coverage = "insufficient"
    else:
        verdict = "C0_insufficient_coverage"
        coverage = "insufficient"
    return {
        "schema": FINAL_INPUT_SCHEMA,
        "layer": "c0",
        "verdict": verdict,
        "seed_classes": dict(seed_classes),
        "resolution_gate_verdict": gate.get("verdict"),
        **common,
    }, coverage


def _native_layer(
    native: Mapping[str, Any], *, layer: str
) -> tuple[str, str]:
    key = (
        "primary_adjudication"
        if layer == "c1_primary"
        else "secondary_shell_adjudication"
    )
    row = native.get(key)
    if not isinstance(row, Mapping):
        return "not_tested", "not_tested"
    status = row.get("status")
    if (
        status == "local_maturation_window"
        or (
            layer == "c1_shell"
            and status == "maturation_candidate_in_secondary_shell"
        )
    ):
        return "local_maturation_window", "complete"
    if status == "isolated_maturation_candidate":
        return "isolated_maturation_candidate", "complete"
    if status == "seed_heterogeneous_maturation":
        return "seed_heterogeneous_maturation", "complete"
    if status == "representation_sensitive":
        return "representation_sensitive_maturation", "complete"
    if status == "no_window":
        if row.get("strict_negative") is True:
            return "no_local_maturation_window", "complete"
        return "insufficient_coverage", (
            "complete" if layer == "c1_primary" else "incomplete"
        )
    return "insufficient_coverage", (
        "complete" if layer == "c1_primary" else "incomplete"
    )


def _c1_final_inputs(
    gate: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    common: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    gate_verdict = gate.get("verdict")
    layer_gates = {
        "c1_primary": gate.get("primary_gate"),
        "c1_shell": gate.get("shell_gate"),
    }

    def consume(layer):
        layer_gate = layer_gates[layer]
        if not isinstance(layer_gate, Mapping):
            return (
                ("C1_blocked_manifest", "blocked_manifest")
                if layer == "c1_primary"
                else ("insufficient_coverage", "incomplete")
            )
        status = layer_gate.get("status")
        if status == "confirmed":
            return "local_maturation_window", "complete"
        if status == "contradicted":
            return "representation_sensitive_maturation", "complete"
        if status == "not_required":
            return _native_layer(native, layer=layer)
        if status in {"blocked", "indeterminate"}:
            return (
                ("C1_blocked_manifest", "blocked_manifest")
                if layer == "c1_primary"
                else ("insufficient_coverage", "incomplete")
            )
        return (
            ("C1_blocked_manifest", "blocked_manifest")
            if layer == "c1_primary"
            else ("insufficient_coverage", "incomplete")
        )

    primary_verdict, primary_coverage = consume("c1_primary")
    shell_verdict, shell_coverage = consume("c1_shell")
    primary = {
        "schema": FINAL_INPUT_SCHEMA,
        "layer": "c1_primary",
        "verdict": primary_verdict,
        "native_adjudication": native.get("primary_adjudication"),
        "resolution_gate_verdict": gate_verdict,
        "layer_resolution_gate": layer_gates["c1_primary"],
        **common,
    }
    shell = {
        "schema": FINAL_INPUT_SCHEMA,
        "layer": "c1_shell",
        "verdict": shell_verdict,
        "native_adjudication": native.get(
            "secondary_shell_adjudication"
        ),
        "resolution_gate_verdict": gate_verdict,
        "layer_resolution_gate": layer_gates["c1_shell"],
        **common,
    }
    return primary, shell, primary_coverage, shell_coverage


def build_final_inputs(
    *,
    phasec_manifest_path: Path,
    c0_gate_path: Path,
    c0_native_path: Path,
    c1_gate_path: Path,
    c1_native_path: Path,
    modal_path: Path,
) -> dict[str, dict[str, Any]]:
    """Derive the four pure-verdict inputs from immutable nested summaries."""

    phasec = _read(phasec_manifest_path)
    phasec_body = {
        key: value for key, value in phasec.items()
        if key != "manifest_sha256"
    }
    if (
        phasec.get("manifest_sha256") != _canonical_sha(phasec_body)
        or phasec.get("production_authorized") is not True
    ):
        raise ValueError("build-inputs requires final self-hashed Phase-C authority")
    c0_gate = _read(c0_gate_path)
    c0_native = _read(c0_native_path)
    c1_gate = _read(c1_gate_path)
    c1_native = _read(c1_native_path)
    modal = _read(modal_path)
    manifest_sha = phasec["manifest_sha256"]
    phasec_file_sha = _sha(phasec_manifest_path)
    if (
        c0_native.get("schema") != C0_SUMMARY_SCHEMA
        or c0_native.get("manifest_sha256") != manifest_sha
        or c0_native.get("manifest_file_sha256") != phasec_file_sha
        or c0_native.get("resolution") != "dt"
    ):
        raise ValueError("C0 native summary parent manifest mismatch")
    if (
        c1_native.get("schema") != C1_SUMMARY_SCHEMA
        or c1_native.get("phasec_manifest_sha256") != manifest_sha
        or c1_native.get("phasec_manifest_file_sha256")
        != phasec_file_sha
    ):
        raise ValueError("C1 native summary parent manifest mismatch")
    if c0_gate.get("schema") != C0_GATE_SCHEMA:
        raise ValueError("C0 resolution gate schema mismatch")
    if c1_gate.get("schema") != C1_GATE_SCHEMA:
        raise ValueError("C1 resolution gate schema mismatch")
    if (
        modal.get("schema") != MODAL_SUMMARY_SCHEMA
        or modal.get("phasec_manifest_sha256") != manifest_sha
    ):
        raise ValueError("modal summary parent manifest mismatch")
    c0_locked_native = c0_gate.get("native_summary")
    if c0_locked_native is None:
        raise ValueError("C0 gate lacks locked native summary provenance")
    c0_locked_path = Path(str(c0_locked_native))
    if not c0_locked_path.is_absolute():
        c0_locked_path = ROOT / c0_locked_path
    if c0_locked_path.resolve() != c0_native_path.resolve():
        raise ValueError("C0 gate/native summary path mismatch")
    if c0_gate.get("native_summary_sha256") != _sha(c0_native_path):
        raise ValueError("C0 gate/native summary file SHA mismatch")
    c0_dt2_ref = c0_gate.get("dt2_summary")
    c0_dt2_sha = c0_gate.get("dt2_summary_sha256")
    c0_dt2_path = None
    c0_dt2 = None
    if c0_dt2_ref is None:
        if c0_dt2_sha is not None:
            raise ValueError("C0 gate has unpaired dt2 summary SHA")
    else:
        c0_dt2_path = Path(str(c0_dt2_ref))
        if not c0_dt2_path.is_absolute():
            c0_dt2_path = ROOT / c0_dt2_path
        if (
            not c0_dt2_path.is_file()
            or not isinstance(c0_dt2_sha, str)
            or c0_dt2_sha != _sha(c0_dt2_path)
        ):
            raise ValueError("C0 gate/dt2 summary file SHA mismatch")
        c0_dt2 = _read(c0_dt2_path)
        if (
            c0_dt2.get("schema") != C0_SUMMARY_SCHEMA
            or c0_dt2.get("manifest_sha256") != manifest_sha
            or c0_dt2.get("manifest_file_sha256") != phasec_file_sha
            or c0_dt2.get("resolution") != "dt2"
        ):
            raise ValueError("C0 dt2 summary parent provenance mismatch")
    c1_locked_native = c1_gate.get("native_summary_path")
    c1_locked_native_sha = c1_gate.get("native_summary_sha256")
    if not isinstance(c1_locked_native, str) or not isinstance(
        c1_locked_native_sha, str
    ):
        raise ValueError("C1 gate lacks locked native summary provenance")
    c1_locked_path = Path(c1_locked_native)
    if not c1_locked_path.is_absolute():
        c1_locked_path = ROOT / c1_locked_path
    if c1_locked_path.resolve() != c1_native_path.resolve():
        raise ValueError("C1 gate/native summary path mismatch")
    if c1_locked_native_sha != _sha(c1_native_path):
        raise ValueError("C1 gate/native summary file SHA mismatch")
    c1_dt2_ref = c1_gate.get("dt2_summary_path")
    c1_dt2_sha = c1_gate.get("dt2_summary_sha256")
    c1_layer_gates = {
        name: c1_gate.get(name)
        for name in ("primary_gate", "shell_gate")
    }
    allowed_layer_gate_statuses = {
        "confirmed", "contradicted", "indeterminate",
        "blocked", "not_required",
    }
    if any(
        not isinstance(row, Mapping)
        or row.get("status") not in allowed_layer_gate_statuses
        for row in c1_layer_gates.values()
    ):
        raise ValueError("C1 resolution gate lacks closed layer gates")
    c1_verdict_requires_dt2 = any(
        row.get("status") in {"confirmed", "contradicted"}
        for row in c1_layer_gates.values()
    )
    c1_dt2_path = None
    c1_dt2 = None
    if c1_dt2_ref is None:
        if c1_dt2_sha is not None:
            raise ValueError("C1 gate has unpaired dt2 summary SHA")
        if c1_verdict_requires_dt2:
            raise ValueError(
                "C1 terminal resolution verdict lacks dt2 summary provenance"
            )
    else:
        if not isinstance(c1_dt2_ref, str) or not isinstance(
            c1_dt2_sha, str
        ):
            raise ValueError("C1 gate dt2 summary provenance is incomplete")
        c1_dt2_path = Path(c1_dt2_ref)
        if not c1_dt2_path.is_absolute():
            c1_dt2_path = ROOT / c1_dt2_path
        if (
            not c1_dt2_path.is_file()
            or c1_dt2_sha != _sha(c1_dt2_path)
        ):
            raise ValueError("C1 gate/dt2 summary file SHA mismatch")
        c1_dt2 = _read(c1_dt2_path)
        if (
            c1_dt2.get("schema")
            != "zm_phasec1_dt2_confirmation_summary_v1_2026-07-28"
            or c1_dt2.get("phasec_manifest_sha256") != manifest_sha
            or c1_dt2.get("phasec_manifest_file_sha256")
            != phasec_file_sha
            or c1_dt2.get("resolution") != "dt2_confirmation_only"
        ):
            raise ValueError("C1 dt2 summary parent provenance mismatch")
    _validate_modal_input_provenance(
        modal,
        {
            "phasec_manifest": (phasec_manifest_path, phasec),
            "c0_resolution_gate": (c0_gate_path, c0_gate),
            "c0_native_summary": (c0_native_path, c0_native),
            "c1_resolution_gate": (c1_gate_path, c1_gate),
            "c1_native_summary": (c1_native_path, c1_native),
        },
        phasec_manifest_sha256=manifest_sha,
    )
    sources = {
        "phasec_manifest": _source_ref(phasec_manifest_path, phasec),
        "c0_resolution_gate": _source_ref(c0_gate_path, c0_gate),
        "c0_native_summary": _source_ref(c0_native_path, c0_native),
        "c1_resolution_gate": _source_ref(c1_gate_path, c1_gate),
        "c1_native_summary": _source_ref(c1_native_path, c1_native),
        "modal_summary": _source_ref(modal_path, modal),
    }
    resource_sources = {
        "c0_native": _resource_source_ref(
            "c0_native", c0_native_path, c0_native
        ),
        "c1_native": _resource_source_ref(
            "c1_native", c1_native_path, c1_native
        ),
    }
    if c0_dt2_path is not None and c0_dt2 is not None:
        resource_sources["c0_dt2"] = _resource_source_ref(
            "c0_dt2", c0_dt2_path, c0_dt2
        )
    if c1_dt2_path is not None and c1_dt2 is not None:
        resource_sources["c1_dt2"] = _resource_source_ref(
            "c1_dt2", c1_dt2_path, c1_dt2
        )
    resource_issues = _audit_resource_sources(
        resource_sources, phasec
    )
    common = {
        "phasec_manifest_sha256": manifest_sha,
        "source_provenance": sources,
        "producer_file_sha256": _producer_locks(),
        "derivation": "deterministic_runtime_rehash_then_final_write_once_lock",
        "resource_receipt_audit": {
            "status": "complete" if not resource_issues else "incomplete",
            "issues": resource_issues,
            "summary_sources": resource_sources,
        },
    }
    c0, c0_coverage = _c0_final_input(
        c0_gate, c0_native, common=common
    )
    primary, shell, primary_coverage, shell_coverage = _c1_final_inputs(
        c1_gate, c1_native, common=common
    )
    if resource_issues:
        c0["verdict"] = "C0_blocked_observables"
        c0_coverage = "blocked_observables"
        primary["verdict"] = "C1_blocked_manifest"
        shell["verdict"] = "not_tested"
        primary_coverage = "blocked_manifest"
        shell_coverage = "not_tested"
    modal_status = modal.get("status")
    if modal_status not in {"complete", "partial", "not_tested"}:
        modal_status = "not_tested"
    coverage = {
        "schema": FINAL_INPUT_SCHEMA,
        "layer": "coverage",
        "c0": {"status": c0_coverage},
        "c1_primary": {"status": primary_coverage},
        "c1_shell": {"status": shell_coverage},
        "modal": {"status": modal_status},
        **common,
    }
    return {
        "c0": c0,
        "c1_primary": primary,
        "c1_shell": shell,
        "coverage": coverage,
    }


def write_final_inputs(
    output_dir: Path, inputs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, filename in FINAL_INPUT_FILENAMES.items():
        path = output_dir / filename
        _atomic_write(path, inputs[name])
        paths[name] = path
    return paths


def adjudicate_files(
    *,
    c0_path: Path | None,
    c1_primary_path: Path | None,
    c1_shell_path: Path | None,
    modal_path: Path | None,
    coverage_path: Path | None,
    trigger_path: Path | None,
    phasec_manifest_path: Path,
) -> dict[str, Any]:
    paths = {
        "c0": c0_path,
        "c1_primary": c1_primary_path,
        "c1_shell": c1_shell_path,
        "modal": modal_path,
        "coverage": coverage_path,
    }
    phasec = _read(phasec_manifest_path)
    manifest_sha = phasec.get("manifest_sha256")
    manifest_body = {
        key: value for key, value in phasec.items()
        if key != "manifest_sha256"
    }
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or _canonical_sha(manifest_body) != manifest_sha
    ):
        raise ValueError("final Phase-C manifest self-hash mismatch")
    config_rows = phasec.get("per_seed")
    if not isinstance(config_rows, dict) or not config_rows:
        raise ValueError("final Phase-C per-seed config locks missing")
    config_contract = {
        str(seed): row.get("canonical_config_sha")
        for seed, row in sorted(config_rows.items(), key=lambda item: int(item[0]))
    }
    if any(not isinstance(value, str) or len(value) != 64
           for value in config_contract.values()):
        raise ValueError("final Phase-C canonical config SHA missing")
    wrapper_issues = []
    artifacts = {}
    for name, path in paths.items():
        if path is None or not Path(path).is_file():
            wrapper_issues.append(f"{name}_file_or_provenance_missing")
            artifacts[name] = {}
            continue
        try:
            artifacts[name] = _read(Path(path))
            if not artifacts[name]:
                wrapper_issues.append(f"{name}_file_or_provenance_invalid")
        except (OSError, ValueError, json.JSONDecodeError):
            wrapper_issues.append(f"{name}_file_or_provenance_invalid")
            artifacts[name] = {}

    trigger = {}
    if trigger_path is None or not Path(trigger_path).is_file():
        wrapper_issues.append("trigger_file_or_provenance_missing")
    else:
        try:
            trigger = _read(Path(trigger_path))
            trigger_body = {
                key: value for key, value in trigger.items()
                if key != "manifest_sha256"
            }
            if (
                trigger.get("selection_is_closed") is not True
                or trigger.get("phasec_manifest_sha256") != manifest_sha
                or trigger.get("manifest_sha256")
                != _canonical_sha(trigger_body)
                or not isinstance(
                    trigger.get("producer_file_sha256"), dict
                )
                or not trigger["producer_file_sha256"]
            ):
                wrapper_issues.append("trigger_file_or_provenance_invalid")
        except (OSError, ValueError, json.JSONDecodeError):
            trigger = {}
            wrapper_issues.append("trigger_file_or_provenance_invalid")
    resource_audit = artifacts.get("coverage", {}).get(
        "resource_receipt_audit"
    )
    if (
        not isinstance(resource_audit, Mapping)
        or resource_audit.get("status") != "complete"
        or resource_audit.get("issues") != []
        or not isinstance(resource_audit.get("summary_sources"), Mapping)
        or not resource_audit.get("summary_sources")
    ):
        wrapper_issues.append("resource_receipt_audit_missing_or_incomplete")
    else:
        wrapper_issues.extend(
            "resource_receipt_audit_contract_failure:" + issue
            for issue in _final_resource_source_contract(
                artifacts, resource_audit
            )
        )
        live_resource_issues = _audit_resource_sources(
            resource_audit["summary_sources"], phasec
        )
        wrapper_issues.extend(
            f"resource_receipt_audit_live_failure:{issue}"
            for issue in live_resource_issues
        )
    provenance = V.build_provenance(
        artifacts,
        manifest_sha256=manifest_sha,
        config_sha256=_canonical_sha(config_contract),
    )
    if wrapper_issues:
        provenance["status"] = "incomplete"
    verdict = V.adjudicate_phasec(
        c0=artifacts["c0"],
        c1_primary=artifacts["c1_primary"],
        c1_shell=artifacts["c1_shell"],
        modal=artifacts["modal"],
        coverage=artifacts["coverage"],
        provenance=provenance,
    )
    # These are duplicated at the wrapper boundary intentionally: a future
    # change in the pure adjudicator cannot silently authorize a lifecycle.
    locked = {
        "entry": "not_tested",
        "offset": "not_tested",
        "recovery_lifecycle": "not_established",
        "phase_c2_authorized": False,
        "actuator_authorized": False,
    }
    if any(verdict.get(key) != value for key, value in locked.items()):
        raise RuntimeError("pure adjudicator violated Phase-C lifecycle boundary")
    return {
        "schema": OUTPUT_SCHEMA,
        **verdict,
        **locked,
        "summary_authority_model": (
            "runtime_rehash_of_deterministic_derived_summaries_then_"
            "final_write_once_lock; summary_SHAs_were_not_preregistered"
        ),
        "input_file_provenance": {
            name: {
                "status": (
                    "complete"
                    if path is not None and Path(path).is_file()
                    and artifacts[name]
                    else "missing_or_invalid"
                ),
                "path": None if path is None else str(path),
                "file_sha256": (
                    _sha(Path(path))
                    if path is not None and Path(path).is_file()
                    else None
                ),
                "artifact_sha256": (
                    V.artifact_sha256(artifacts[name])
                    if artifacts[name] else None
                ),
                "artifact_declared_manifest_sha256": (
                    artifacts[name].get("manifest_sha256")
                    or artifacts[name].get("phasec_manifest_sha256")
                ),
                "parent_phasec_manifest_sha256": manifest_sha,
            }
            for name, path in paths.items()
        },
        "trigger_provenance": {
            "status": (
                "complete"
                if trigger
                and "trigger_file_or_provenance_invalid" not in wrapper_issues
                else "missing_or_invalid"
            ),
            "path": None if trigger_path is None else str(trigger_path),
            "file_sha256": (
                _sha(Path(trigger_path))
                if trigger_path is not None and Path(trigger_path).is_file()
                else None
            ),
            "manifest_sha256": trigger.get("manifest_sha256"),
            "producer_file_sha256": trigger.get("producer_file_sha256"),
            "parent_phasec_manifest_sha256": trigger.get(
                "phasec_manifest_sha256"
            ),
        },
        "wrapper_provenance_issues": wrapper_issues,
        "phasec_manifest_provenance": {
            "path": str(phasec_manifest_path),
            "file_sha256": _sha(phasec_manifest_path),
            "manifest_sha256": manifest_sha,
            "producer_file_sha256": phasec.get(
                "provenance", {}
            ).get("producer_file_sha256"),
            "config_contract_sha256": _canonical_sha(config_contract),
        },
        "adjudicator_producer_file_sha256": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha(
                Path(__file__).resolve()
            ),
            "src/topic4_zm_phasec_verdict.py": _sha(
                ROOT / "src/topic4_zm_phasec_verdict.py"
            ),
        },
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return
        raise RuntimeError(f"refusing to overwrite different adjudication: {path}")
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
    parser.add_argument("--build-inputs", action="store_true")
    parser.add_argument("--c0-gate", type=Path)
    parser.add_argument("--c0-native", type=Path)
    parser.add_argument("--c1-gate", type=Path)
    parser.add_argument("--c1-native", type=Path)
    parser.add_argument("--modal-summary", type=Path)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--c0", type=Path)
    parser.add_argument("--c1-primary", type=Path)
    parser.add_argument("--c1-shell", type=Path)
    parser.add_argument("--modal", type=Path)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--trigger", type=Path)
    parser.add_argument("--phasec-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.build_inputs:
        required = {
            "--c0-gate": args.c0_gate,
            "--c0-native": args.c0_native,
            "--c1-gate": args.c1_gate,
            "--c1-native": args.c1_native,
            "--modal-summary": args.modal_summary,
            "--inputs-dir": args.inputs_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--build-inputs requires " + ", ".join(missing))
        inputs = build_final_inputs(
            phasec_manifest_path=args.phasec_manifest,
            c0_gate_path=args.c0_gate,
            c0_native_path=args.c0_native,
            c1_gate_path=args.c1_gate,
            c1_native_path=args.c1_native,
            modal_path=args.modal_summary,
        )
        paths = write_final_inputs(args.inputs_dir, inputs)
        print(json.dumps({
            "status": "final_inputs_locked",
            "paths": {name: str(path) for name, path in paths.items()},
        }, sort_keys=True))
        return
    if args.output is None:
        parser.error("adjudication mode requires --output")
    payload = adjudicate_files(
        c0_path=args.c0,
        c1_primary_path=args.c1_primary,
        c1_shell_path=args.c1_shell,
        modal_path=args.modal,
        coverage_path=args.coverage,
        trigger_path=args.trigger,
        phasec_manifest_path=args.phasec_manifest,
    )
    _atomic_write(args.output, payload)
    print(json.dumps({
        "verdict": payload["verdict"],
        "next_route": payload["next_route"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
