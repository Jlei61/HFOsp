#!/usr/bin/env python3
"""Lock the Phase-C whole-sheet-runaway analysis amendment.

This write-once artifact authorizes reuse of immutable raw SNN parts by the
versioned v2 analyzer.  It is created only after the complete native C1 base
matrix exists and before any formal C1 atlas has been written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import scripts.analyze_topic4_zm_phasec1_v2 as A2  # noqa: E402


SCHEMA = "zm_phasec_analysis_amendment_v1"
RESULT_ROOT = (
    CODE_ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
)
ORIGINAL_ANALYSIS_FILES = (
    "scripts/analyze_topic4_zm_phasec1.py",
    "src/topic4_zm_phasec_phenotype.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _resolve_observables(path_value, *, code_root=CODE_ROOT):
    path = Path(str(path_value))
    return path if path.is_absolute() else Path(code_root) / path


def audit_base_observables(result_root, *, expected_count, code_root=CODE_ROOT):
    """Prove every completed base part carries the independent runaway trace."""
    base = Path(result_root) / "parts/c1_base/dt"
    parts = sorted(base.rglob("phenotype.json"))
    if len(parts) != int(expected_count):
        raise ValueError(
            f"C1 base part count mismatch: {len(parts)} != {expected_count}"
        )
    issues = []
    complete = 0
    scientific_failure = 0
    all_sheet_bins = []
    all_sheet_bin_ms = []
    part_hashes = {}
    observable_hashes = {}
    for part_path in parts:
        part = _read(part_path)
        status = part.get("status")
        if status == "scientific_failure":
            scientific_failure += 1
        elif status == "complete":
            complete += 1
            obs_path = _resolve_observables(
                part.get("observables_path"), code_root=code_root
            )
            if (
                not obs_path.is_file()
                or part.get("observables_sha256") != _sha(obs_path)
            ):
                issues.append(f"{part_path}:observables_sha256_mismatch")
                continue
            try:
                with np.load(obs_path, allow_pickle=False) as data:
                    missing = [
                        key for key in (
                            "carrier_gate_r_all_hz",
                            "carrier_gate_bin_ms",
                            "source_rate_hz",
                            "bin_ms",
                        )
                        if key not in data.files
                    ]
                    if missing:
                        issues.append(
                            f"{part_path}:missing:{','.join(missing)}"
                        )
                        continue
                    all_rate = np.asarray(
                        data["carrier_gate_r_all_hz"], float
                    ).ravel()
                    all_bin = float(
                        np.asarray(
                            data["carrier_gate_bin_ms"]
                        ).reshape(()).item()
                    )
                    source = np.asarray(
                        data["source_rate_hz"], float
                    ).ravel()
                    source_bin = float(
                        np.asarray(data["bin_ms"]).reshape(()).item()
                    )
            except (OSError, TypeError, ValueError) as exc:
                issues.append(
                    f"{part_path}:invalid_npz:{type(exc).__name__}:{exc}"
                )
                continue
            if (
                all_rate.size < 4
                or source.size < 16
                or not np.all(np.isfinite(all_rate))
                or not np.all(np.isfinite(source))
                or not np.isfinite(all_bin)
                or all_bin <= 0
                or not np.isfinite(source_bin)
                or source_bin <= 0
            ):
                issues.append(f"{part_path}:invalid_rate_trace")
                continue
            all_sheet_bins.append(int(all_rate.size))
            all_sheet_bin_ms.append(all_bin)
            observable_hashes[str(obs_path)] = _sha(obs_path)
        else:
            issues.append(f"{part_path}:nonterminal_status:{status}")
        part_hashes[str(part_path)] = _sha(part_path)
    if issues:
        raise ValueError("C1 raw-observable audit failed: " + ";".join(issues))
    return {
        "status": "complete",
        "expected_part_count": int(expected_count),
        "validated_part_count": len(parts),
        "complete_part_count": complete,
        "scientific_failure_part_count": scientific_failure,
        "complete_parts_with_all_sheet_trace": len(all_sheet_bins),
        "all_sheet_n_bins_min": (
            min(all_sheet_bins) if all_sheet_bins else None
        ),
        "all_sheet_n_bins_max": (
            max(all_sheet_bins) if all_sheet_bins else None
        ),
        "all_sheet_bin_ms_values": sorted(set(all_sheet_bin_ms)),
        "part_set_sha256": _canonical_sha(part_hashes),
        "observable_set_sha256": _canonical_sha(observable_hashes),
    }


def build_payload(
    *,
    result_root,
    manifest_path,
    coordinator_summary_path,
    corrected_analysis_producers,
    amendment_producers,
    analysis_git_sha,
    created_at,
    code_root=CODE_ROOT,
):
    result_root = Path(result_root)
    manifest_path = Path(manifest_path)
    coordinator_summary_path = Path(coordinator_summary_path)
    if (result_root / "phasec1_base_atlas_dt.json").exists():
        raise ValueError(
            "formal C1 base atlas already exists; amendment is too late"
        )
    phasec = _read(manifest_path)
    manifest_body = {
        key: value for key, value in phasec.items()
        if key != "manifest_sha256"
    }
    if phasec.get("manifest_sha256") != _canonical_sha(manifest_body):
        raise ValueError("Phase-C manifest self-hash mismatch")
    coordinator = _read(coordinator_summary_path)
    if (
        coordinator.get("schema")
        != "zm_phasec1_coordinator_v1_2026-07-28"
        or coordinator.get("phase") != "base"
        or coordinator.get("phasec_manifest_sha256")
        != phasec["manifest_sha256"]
        or coordinator.get("n_pending_after_stop") != 0
        or coordinator.get("n_failures") != 0
    ):
        raise ValueError("C1 base coordinator is not terminal-success")
    expected = int(coordinator.get("n_expected_simulations", -1))
    if expected <= 0:
        raise ValueError("invalid expected C1 base count")
    audit = audit_base_observables(
        result_root, expected_count=expected, code_root=code_root
    )
    locked = phasec.get("provenance", {}).get("producer_file_sha256")
    if not isinstance(locked, dict):
        raise ValueError("Phase-C producer locks missing")
    original = {}
    for relative in ORIGINAL_ANALYSIS_FILES:
        expected_sha = locked.get(relative)
        path = Path(code_root) / relative
        if (
            not isinstance(expected_sha, str)
            or not path.is_file()
            or _sha(path) != expected_sha
        ):
            raise ValueError(f"locked original analyzer drift: {relative}")
        original[relative] = expected_sha
    if (
        not isinstance(corrected_analysis_producers, dict)
        or not corrected_analysis_producers
    ):
        raise ValueError("corrected analysis producer map missing")
    for relative, expected_sha in corrected_analysis_producers.items():
        path = Path(code_root) / relative
        if not path.is_file() or _sha(path) != expected_sha:
            raise ValueError(f"corrected analyzer drift: {relative}")
    if not isinstance(amendment_producers, dict) or not amendment_producers:
        raise ValueError("amendment producer map missing")
    for relative, expected_sha in amendment_producers.items():
        path = Path(code_root) / relative
        if not path.is_file() or _sha(path) != expected_sha:
            raise ValueError(f"amendment builder drift: {relative}")

    body = {
        "schema": SCHEMA,
        "created_at_utc": str(created_at),
        "analysis_git_sha": str(analysis_git_sha),
        "phasec_manifest_path": str(manifest_path),
        "phasec_manifest_file_sha256": _sha(manifest_path),
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "coordinator_summary_path": str(coordinator_summary_path),
        "coordinator_summary_file_sha256": _sha(coordinator_summary_path),
        "discovery_stage": "before_first_formal_c1_base_atlas",
        "issue": (
            "250_Hz_whole_sheet_runaway_gate_was_mapped_to_"
            "pathology_core_source_rate"
        ),
        "correction": (
            "runaway_and_tail_trend_use_carrier_gate_r_all_hz_at_"
            "carrier_gate_bin_ms;source_temporal_morphology_unchanged"
        ),
        "runaway_threshold_hz": 250.0,
        "threshold_changed": False,
        "simulation_grid_changed": False,
        "slow_fields_changed": False,
        "raw_snn_parts_reused": True,
        "snn_rerun_required": False,
        "original_locked_analysis_producer_file_sha256": original,
        "corrected_analysis_producer_file_sha256": dict(
            corrected_analysis_producers
        ),
        "amendment_producer_file_sha256": dict(amendment_producers),
        "raw_observable_audit": audit,
        "claim_boundary": (
            "analysis-only semantic correction; no entry, offset, recovery, "
            "observation match, or lifecycle claim"
        ),
    }
    return {**body, "amendment_sha256": _canonical_sha(body)}


def _publish_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def _latest_successful_coordinator(result_root):
    candidates = sorted(
        (Path(result_root) / "coordinator_runs").glob(
            "phasec1_base_summary_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            row = _read(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            row.get("phase") == "base"
            and row.get("n_pending_after_stop") == 0
            and row.get("n_failures") == 0
        ):
            return path
    raise ValueError("no successful terminal C1 base coordinator summary")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--coordinator-summary")
    parser.add_argument("--confirm-lock", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_lock:
        raise SystemExit("--confirm-lock is required")
    result_root = Path(args.result_root)
    coordinator = (
        Path(args.coordinator_summary)
        if args.coordinator_summary
        else _latest_successful_coordinator(result_root)
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        text=True,
    ).strip()
    payload = build_payload(
        result_root=result_root,
        manifest_path=result_root / "phasec_manifest.json",
        coordinator_summary_path=coordinator,
        corrected_analysis_producers=A2._analysis_producers(),
        amendment_producers={
            str(Path(__file__).resolve().relative_to(CODE_ROOT)): _sha(
                Path(__file__).resolve()
            )
        },
        analysis_git_sha=git_sha,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    output = result_root / "phasec_analysis_amendment.json"
    _publish_once(output, payload)
    print(json.dumps({
        "output": str(output),
        "amendment_sha256": payload["amendment_sha256"],
        "validated_part_count": payload[
            "raw_observable_audit"
        ]["validated_part_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
