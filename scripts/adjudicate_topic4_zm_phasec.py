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


OUTPUT_SCHEMA = "zm_phasec_final_adjudication_v1_2026-07-28"
FINAL_INPUT_SCHEMA = "zm_phasec_final_input_v1_2026-07-28"
FINAL_INPUT_FILENAMES = {
    "c0": "phasec_final_input_c0.json",
    "c1_primary": "phasec_final_input_c1_primary.json",
    "c1_shell": "phasec_final_input_c1_shell.json",
    "coverage": "phasec_final_input_coverage.json",
}


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


def _producer_locks() -> dict[str, str]:
    return {
        str(Path(__file__).resolve().relative_to(ROOT)): _sha(
            Path(__file__).resolve()
        ),
        "src/topic4_zm_phasec_verdict.py": _sha(
            ROOT / "src/topic4_zm_phasec_verdict.py"
        ),
    }


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
    pending_or_blocked = {
        "C1_window_pending_dt2",
        "C1_blocked_resolution_gate",
        "C1_blocked_manifest",
    }
    if gate_verdict in pending_or_blocked or gate_verdict is None:
        primary_verdict, primary_coverage = (
            "C1_blocked_manifest", "blocked_manifest"
        )
        shell_verdict, shell_coverage = "not_tested", "not_tested"
    elif gate_verdict == "maturation_window_at_primary_convex_states":
        primary_verdict, primary_coverage = (
            "local_maturation_window", "complete"
        )
        shell_verdict, shell_coverage = _native_layer(
            native, layer="c1_shell"
        )
        # A primary positive already closes the scientific decision.  An
        # incomplete shell remains explicit but cannot demote the primary.
    elif gate_verdict == "maturation_candidate_in_secondary_shell":
        primary_verdict, primary_coverage = _native_layer(
            native, layer="c1_primary"
        )
        if primary_verdict != "no_local_maturation_window":
            primary_verdict, primary_coverage = (
                "C1_blocked_manifest", "blocked_manifest"
            )
        shell_verdict, shell_coverage = (
            "local_maturation_window", "complete"
        )
    elif gate_verdict == "resolution_sensitive_maturation":
        reason = str(gate.get("reason", ""))
        if "secondary" in reason:
            primary_verdict, primary_coverage = _native_layer(
                native, layer="c1_primary"
            )
            shell_verdict, shell_coverage = (
                "representation_sensitive_maturation", "complete"
            )
        else:
            primary_verdict, primary_coverage = (
                "representation_sensitive_maturation", "complete"
            )
            shell_verdict, shell_coverage = "not_tested", "not_tested"
    else:
        primary_verdict, primary_coverage = _native_layer(
            native, layer="c1_primary"
        )
        shell_verdict, shell_coverage = _native_layer(
            native, layer="c1_shell"
        )
        # A native positive is never accepted unless the resolution gate
        # explicitly confirmed it.
        if primary_verdict == "local_maturation_window":
            primary_verdict, primary_coverage = (
                "C1_blocked_manifest", "blocked_manifest"
            )
            shell_verdict, shell_coverage = "not_tested", "not_tested"
        elif shell_verdict == "local_maturation_window":
            primary_verdict, primary_coverage = (
                "C1_blocked_manifest", "blocked_manifest"
            )
            shell_verdict, shell_coverage = "not_tested", "not_tested"
    primary = {
        "schema": FINAL_INPUT_SCHEMA,
        "layer": "c1_primary",
        "verdict": primary_verdict,
        "native_adjudication": native.get("primary_adjudication"),
        "resolution_gate_verdict": gate_verdict,
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
    if c0_native.get("manifest_sha256") != manifest_sha:
        raise ValueError("C0 native summary parent manifest mismatch")
    if c1_native.get("phasec_manifest_sha256") != manifest_sha:
        raise ValueError("C1 native summary parent manifest mismatch")
    if modal.get("phasec_manifest_sha256") != manifest_sha:
        raise ValueError("modal summary parent manifest mismatch")
    c0_locked_native = c0_gate.get("native_summary")
    if c0_locked_native is not None:
        c0_locked_path = Path(str(c0_locked_native))
        if not c0_locked_path.is_absolute():
            c0_locked_path = ROOT / c0_locked_path
        if c0_locked_path.resolve() != c0_native_path.resolve():
            raise ValueError("C0 gate/native summary path mismatch")
    c1_locked_native = c1_gate.get("native_summary_path")
    if c1_locked_native is not None:
        c1_locked_path = Path(str(c1_locked_native))
        if not c1_locked_path.is_absolute():
            c1_locked_path = ROOT / c1_locked_path
        if c1_locked_path.resolve() != c1_native_path.resolve():
            raise ValueError("C1 gate/native summary path mismatch")
    if (
        c1_gate.get("native_summary_sha256") is not None
        and c1_gate.get("native_summary_sha256") != _sha(c1_native_path)
    ):
        raise ValueError("C1 gate/native summary file SHA mismatch")
    sources = {
        "phasec_manifest": _source_ref(phasec_manifest_path, phasec),
        "c0_resolution_gate": _source_ref(c0_gate_path, c0_gate),
        "c0_native_summary": _source_ref(c0_native_path, c0_native),
        "c1_resolution_gate": _source_ref(c1_gate_path, c1_gate),
        "c1_native_summary": _source_ref(c1_native_path, c1_native),
        "modal_summary": _source_ref(modal_path, modal),
    }
    common = {
        "phasec_manifest_sha256": manifest_sha,
        "source_provenance": sources,
        "producer_file_sha256": _producer_locks(),
        "derivation": "deterministic_runtime_rehash_then_final_write_once_lock",
    }
    c0, c0_coverage = _c0_final_input(
        c0_gate, c0_native, common=common
    )
    primary, shell, primary_coverage, shell_coverage = _c1_final_inputs(
        c1_gate, c1_native, common=common
    )
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
