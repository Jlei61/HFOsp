#!/usr/bin/env python3
"""Adjudicate the user-authorized Phase-C1 post-result futility stop.

This is deliberately not a partial C1 atlas.  It asks only whether the
already completed seed-1 primary runs make the preregistered primary
maturation GO logically unreachable.  The original production manifest,
classifier, analyzer, raw SNN parts, and resource receipts remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from src import topic4_zm_phasec_phenotype_v2 as P2  # noqa: E402
from src import topic4_zm_phasec_resources as R  # noqa: E402


SCHEMA = "zm_phasec_post_result_futility_stop_v1_2026-07-31"
RESULT_ROOT = (
    CODE_ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
)
OUTPUT_NAME = "phasec_futility_verdict.json"
ORIGINAL_ANALYSIS_FILES = (
    "scripts/analyze_topic4_zm_phasec1.py",
    "src/topic4_zm_phasec_phenotype.py",
)
POSITIVE_NON_TONIC_CLASSES = frozenset({
    "periodic_non_tonic_carrier",
    "clonic_or_bursting_carrier",
})
PHASES = ("rising", "peak")
NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _relative(path, *, code_root=CODE_ROOT):
    return os.path.relpath(Path(path).resolve(), Path(code_root).resolve())


def _validate_self_hash(payload, *, hash_field, label):
    body = {key: value for key, value in payload.items() if key != hash_field}
    if payload.get(hash_field) != _canonical_sha(body):
        raise ValueError(f"{label} self-hash mismatch")


def _part_identity(path, *, result_root):
    base = Path(result_root) / "parts/c1_base/dt"
    try:
        rel = Path(path).relative_to(base)
    except ValueError as exc:
        raise ValueError(f"part outside native C1 base root: {path}") from exc
    fields = rel.parts
    if len(fields) != 6 or fields[-1] != "phenotype.json":
        raise ValueError(f"unexpected C1 part path: {rel}")
    seed_name, tier, cell_id, phase, noise, _ = fields
    if not seed_name.startswith("seed"):
        raise ValueError(f"invalid C1 seed path: {seed_name}")
    return {
        "seed": int(seed_name.removeprefix("seed")),
        "tier": tier,
        "cell_id": cell_id,
        "phase": phase,
        "noise": noise,
    }


def _task_key(identity):
    return (
        f"base|s{identity['seed']}|{identity['tier']}|"
        f"{identity['cell_id']}|{identity['phase']}|{identity['noise']}"
    )


def _classify_part(path, *, result_root, phasec_manifest_sha256):
    identity = _part_identity(path, result_root=result_root)
    part = _read(path)
    for key, wanted in {
        "seed": identity["seed"],
        "tier": identity["tier"],
        "cell_id": identity["cell_id"],
        "phase": identity["phase"],
        "noise": identity["noise"],
        "resolution": "dt",
        "phasec_manifest_sha256": phasec_manifest_sha256,
        "status": "complete",
    }.items():
        if part.get(key) != wanted:
            raise ValueError(f"{path}: part field mismatch: {key}")

    receipt_path = R.resource_receipt_path(path)
    receipt_ok, receipt_reason, receipt = R.validate_resource_receipt(
        receipt_path,
        artifact_path=path,
        artifact_root=CODE_ROOT,
        manifest_sha256=phasec_manifest_sha256,
        task_key=_task_key(identity),
    )
    if not receipt_ok:
        raise ValueError(f"{path}: {receipt_reason}")

    arrays = A2._load_phenotype_arrays(part)
    if arrays.get("status") != "ok":
        raise ValueError(f"{path}: {arrays.get('reason')}")
    try:
        refractory_fraction = float(
            part["spike_metrics"]["isi"]["refractory_isi_fraction"]
        )
        rho80 = float(
            part["spike_metrics"]["firing"][
                "rho80_active_core_median"
            ]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid spike metrics") from exc
    if not np.isfinite(refractory_fraction) or not np.isfinite(rho80):
        raise ValueError(f"{path}: non-finite spike metrics")

    classified = P2.classify_phasec_run(
        arrays["E_rate_grid"],
        arrays["I_rate_grid"],
        bin_ms=arrays["bin_ms"],
        source_rate_hz=arrays["source_rate_hz"],
        rest_mask=arrays["rest_mask"],
        active_area_fraction=arrays["active_area_fraction"],
        kymograph=arrays["kymograph"],
        axis_positions=arrays["axis_positions"],
        readout_kernel_width_mm=arrays["readout_kernel_width_mm"],
        all_sheet_rate_hz=arrays["all_sheet_rate_hz"],
        all_sheet_bin_ms=arrays["all_sheet_bin_ms"],
        runaway_early_stop_ms=part.get("runaway_early_stop_ms"),
        saturation_fraction=part.get("saturation_fraction"),
        refractory_fraction=refractory_fraction,
        relay_rng_seed=identity["seed"],
    )
    modulation = float(
        classified["temporal_diagnostics"]["global_modulation_fraction"]
    )
    if not np.isfinite(modulation):
        raise ValueError(f"{path}: non-finite modulation depth")
    return {
        **identity,
        "part_path": _relative(path),
        "part_file_sha256": _sha(path),
        "receipt_path": _relative(receipt_path),
        "receipt_file_sha256": _sha(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "observables_path": _relative(arrays["path"]),
        "observables_file_sha256": _sha(arrays["path"]),
        "phenotype": classified["phenotype"],
        "runaway_rate_scope": classified["bounded_gate"][
            "runaway_rate_scope"
        ],
        "modulation_depth": modulation,
        "core_rate_mean_hz": float(np.mean(arrays["source_rate_hz"])),
        "all_sheet_rate_mean_hz": float(
            np.mean(arrays["all_sheet_rate_hz"])
        ),
        "all_sheet_rate_p95_hz": float(
            np.percentile(arrays["all_sheet_rate_hz"], 95)
        ),
        "rho80_active_core_median": rho80,
        "refractory_isi_fraction": refractory_fraction,
        "active_occupancy": float(
            classified["bounded_gate"]["active_occupancy"]
        ),
    }


def _validate_producer_hashes(phasec, *, code_root):
    locked = phasec.get("provenance", {}).get("producer_file_sha256")
    if not isinstance(locked, dict):
        raise ValueError("Phase-C producer locks missing")
    original = {}
    for relative in ORIGINAL_ANALYSIS_FILES:
        expected = locked.get(relative)
        path = Path(code_root) / relative
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or _sha(path) != expected
        ):
            raise ValueError(f"locked original analyzer drift: {relative}")
        original[relative] = expected
    corrected = A2._analysis_producers()
    for relative, expected in corrected.items():
        path = Path(code_root) / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"corrected analyzer drift: {relative}")
    return original, corrected


def _summary(values):
    x = np.asarray(list(values), float)
    return {
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
    }


def build_payload(
    *,
    result_root,
    manifest_path,
    coordinate_manifest_path,
    partial_abort_path,
    analysis_git_sha,
    created_at,
    code_root=CODE_ROOT,
):
    """Build a fail-closed futility proof from immutable partial evidence."""
    result_root = Path(result_root)
    manifest_path = Path(manifest_path)
    coordinate_manifest_path = Path(coordinate_manifest_path)
    partial_abort_path = Path(partial_abort_path)
    if (result_root / "phasec1_base_atlas_dt.json").exists():
        raise ValueError("formal C1 base atlas exists; futility stop is invalid")

    phasec = _read(manifest_path)
    _validate_self_hash(
        phasec, hash_field="manifest_sha256", label="Phase-C manifest"
    )
    coordinate = _read(coordinate_manifest_path)
    _validate_self_hash(
        coordinate,
        hash_field="manifest_sha256",
        label="C1 coordinate manifest",
    )
    coordinate_ref = phasec.get("c1", {}).get(
        "coordinate_manifests", {}
    ).get("dt", {})
    if (
        coordinate_ref.get("file_sha256") != _sha(coordinate_manifest_path)
        or coordinate_ref.get("manifest_sha256")
        != coordinate.get("manifest_sha256")
        or coordinate_ref.get("semantic_sha256")
        != coordinate.get("semantic_sha256")
    ):
        raise ValueError("native coordinate manifest is not Phase-C locked")

    abort = _read(partial_abort_path)
    if (
        abort.get("schema") != "zm_phasec1_coordinator_partial_abort_v1"
        or abort.get("phase") != "base"
        or abort.get("phasec_manifest_sha256")
        != phasec["manifest_sha256"]
        or int(abort.get("n_expected_simulations", -1)) <= 0
        or int(abort.get("n_owned_inflight_at_abort", -1)) < 0
    ):
        raise ValueError("invalid Phase-C partial-abort record")
    resource_log = Path(code_root) / abort["resource_log_path"]
    if not resource_log.is_file():
        raise ValueError("partial-abort resource log missing")

    original, corrected = _validate_producer_hashes(
        phasec, code_root=code_root
    )
    parts = sorted(
        (result_root / "parts/c1_base/dt").rglob("phenotype.json")
    )
    if len(parts) != int(abort.get("n_completed_before_abort", -1)):
        raise ValueError("completed part count differs from abort record")
    rows = [
        _classify_part(
            path,
            result_root=result_root,
            phasec_manifest_sha256=phasec["manifest_sha256"],
        )
        for path in parts
    ]
    if not rows:
        raise ValueError("no completed C1 evidence")
    if any(
        row["seed"] != 1 or row["tier"] != "primary_convex"
        for row in rows
    ):
        raise ValueError(
            "futility proof requires the observed set to be seed-1 primary"
        )

    thresholds = phasec["thresholds"]
    modulation_min = float(thresholds["fine_rate_modulation_min"])
    cell_success_min = int(thresholds["cell_total_min_passes"])
    expected_cells = list(phasec["c1"]["primary_cell_names"])
    if len(expected_cells) != 10:
        raise ValueError("unexpected seed-1 primary cell inventory")
    expected_run_ids = {
        (cell, phase, noise)
        for cell in expected_cells
        for phase in PHASES
        for noise in NOISES
    }
    observed_run_ids = {
        (row["cell_id"], row["phase"], row["noise"]) for row in rows
    }
    if (
        len(observed_run_ids) != len(rows)
        or not observed_run_ids.issubset(expected_run_ids)
    ):
        raise ValueError("duplicate or unregistered seed-1 primary run")

    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    cells = []
    for cell_id in expected_cells:
        observed = by_cell[cell_id]
        positive = sum(
            row["phenotype"] in POSITIVE_NON_TONIC_CLASSES
            for row in observed
        )
        missing = 6 - len(observed)
        max_possible_positive = positive + missing
        cells.append({
            "cell_id": cell_id,
            "n_expected": 6,
            "n_observed": len(observed),
            "n_missing": missing,
            "n_observed_non_tonic_positive": positive,
            "max_possible_non_tonic_positive": max_possible_positive,
            "cell_positive_required": cell_success_min,
            "cell_can_still_be_positive": bool(
                max_possible_positive >= cell_success_min
            ),
            "phenotype_counts": dict(
                Counter(row["phenotype"] for row in observed)
            ),
            "modulation_depth": (
                _summary(row["modulation_depth"] for row in observed)
                if observed else None
            ),
        })

    seed1_unrescuable = bool(
        all(not row["cell_can_still_be_positive"] for row in cells)
        and len(observed_run_ids) == len(rows)
        and set(by_cell) == set(expected_cells)
    )
    all_tonic = all(row["phenotype"] == "tonic_non_AI" for row in rows)
    all_below_modulation = all(
        row["modulation_depth"] < modulation_min for row in rows
    )
    all_sheet_scope = all(
        row["runaway_rate_scope"] == "all_sheet_E" for row in rows
    )
    if not (
        seed1_unrescuable
        and all_tonic
        and all_below_modulation
        and all_sheet_scope
    ):
        raise ValueError(
            "observed evidence does not establish registered seed-1 futility"
        )

    part_hashes = {row["part_path"]: row["part_file_sha256"] for row in rows}
    receipt_hashes = {
        row["receipt_path"]: row["receipt_file_sha256"] for row in rows
    }
    observable_hashes = {
        row["observables_path"]: row["observables_file_sha256"]
        for row in rows
    }
    script_path = Path(__file__).resolve()
    body = {
        "schema": SCHEMA,
        "created_at_utc": str(created_at),
        "analysis_git_sha": str(analysis_git_sha),
        "status": "post_result_futility_stopped_incomplete",
        "decision": "stop_phasec1_and_open_fast_carrier_repair_spec",
        "user_authorization": (
            "user_approved_prior_recommendation_to_change_spec_2026-07-31"
        ),
        "phasec_manifest_path": _relative(
            manifest_path, code_root=code_root
        ),
        "phasec_manifest_file_sha256": _sha(manifest_path),
        "phasec_manifest_sha256": phasec["manifest_sha256"],
        "coordinate_manifest_path": _relative(
            coordinate_manifest_path, code_root=code_root
        ),
        "coordinate_manifest_file_sha256": _sha(
            coordinate_manifest_path
        ),
        "coordinate_manifest_sha256": coordinate["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": coordinate[
            "semantic_sha256"
        ],
        "partial_abort_path": _relative(
            partial_abort_path, code_root=code_root
        ),
        "partial_abort_file_sha256": _sha(partial_abort_path),
        "resource_log_path": _relative(
            resource_log, code_root=code_root
        ),
        "resource_log_file_sha256": _sha(resource_log),
        "execution_coverage": {
            "expected_full_c1_base_runs": int(
                abort["n_expected_simulations"]
            ),
            "completed_runs": len(rows),
            "pending_at_abort": int(abort["n_pending_at_abort"]),
            "owned_inflight_cleaned_at_abort": int(
                abort["n_owned_inflight_at_abort"]
            ),
            "seed1_primary_expected": 60,
            "seed1_primary_completed": len(rows),
            "seed1_primary_missing": 60 - len(rows),
            "complete_phasec1_negative": False,
        },
        "analysis_correction": {
            "issue": (
                "250_Hz_whole_sheet_runaway_gate_was_mapped_to_"
                "pathology_core_source_rate"
            ),
            "correction": (
                "runaway_uses_carrier_gate_r_all_hz_at_"
                "carrier_gate_bin_ms;source_morphology_unchanged"
            ),
            "threshold_changed": False,
            "simulation_grid_changed": False,
            "raw_snn_parts_reused": True,
            "snn_rerun_required": False,
        },
        "registered_logic": {
            "positive_non_tonic_classes": sorted(
                POSITIVE_NON_TONIC_CLASSES
            ),
            "run_positive_modulation_min": modulation_min,
            "cell_positive_min_runs": cell_success_min,
            "cell_total_runs": 6,
            "primary_go_requires_native_seeds_1_and_3": True,
            "secondary_shell_cannot_establish_primary_reachability": True,
        },
        "seed1_primary_futility": {
            "established": True,
            "reason": (
                "all_59_completed_runs_are_tonic_non_AI_and_each_primary_"
                "cell_has_fewer_than_five_possible_non_tonic_positive_runs_"
                "even_if_the_single_missing_run_were_positive"
            ),
            "n_cells": len(cells),
            "n_unrescuable_cells": sum(
                not row["cell_can_still_be_positive"] for row in cells
            ),
            "phenotype_counts": dict(
                Counter(row["phenotype"] for row in rows)
            ),
            "runaway_rate_scope_counts": dict(
                Counter(row["runaway_rate_scope"] for row in rows)
            ),
            "modulation_depth": _summary(
                row["modulation_depth"] for row in rows
            ),
            "core_rate_mean_hz": _summary(
                row["core_rate_mean_hz"] for row in rows
            ),
            "all_sheet_rate_mean_hz": _summary(
                row["all_sheet_rate_mean_hz"] for row in rows
            ),
            "rho80_active_core_median": _summary(
                row["rho80_active_core_median"] for row in rows
            ),
            "refractory_isi_fraction": _summary(
                row["refractory_isi_fraction"] for row in rows
            ),
            "cells": cells,
        },
        "logical_consequence": {
            "registered_primary_maturation_go_reachable": False,
            "why": (
                "seed1_cannot_support_even_one_positive_primary_cell_and_"
                "the_registered_primary_go_requires_native_seeds_1_and_3"
            ),
            "conditional_gain_authorized": False,
            "dt2_confirmation_authorized": False,
            "modal_analysis_authorized": False,
            "full_phasec_adjudication_authorized": False,
        },
        "scientific_interpretation": {
            "supported": (
                "within_the_observed_seed1_primary_slow_field_neighbourhood_"
                "the_fast_network_remains_on_a_localised_high_rate_"
                "low_modulation_tonic_branch"
            ),
            "not_supported": [
                "complete_bounded_negative_over_all_three_seeds",
                "no_carrier_anywhere_in_the_SNN",
                "entry",
                "offset",
                "recovery",
                "ictal_lifecycle",
            ],
            "mechanistic_implication": (
                "moving_frozen_Z_M_S_G_coordinates_does_not_create_the_"
                "required_fast_non_tonic_carrier_in_seed1;the_next_spec_"
                "must_change_fast_inhibitory_membrane_feedback_without_"
                "changing_E_to_E"
            ),
        },
        "entry": "not_tested",
        "offset": "not_tested",
        "recovery_lifecycle": "not_established",
        "original_locked_analysis_producer_file_sha256": original,
        "corrected_analysis_producer_file_sha256": corrected,
        "futility_producer_file_sha256": {
            _relative(script_path, code_root=code_root): _sha(script_path),
        },
        "evidence_set_sha256": {
            "parts": _canonical_sha(part_hashes),
            "resource_receipts": _canonical_sha(receipt_hashes),
            "observables": _canonical_sha(observable_hashes),
        },
        "run_rows": rows,
        "claim_boundary": (
            "post-result logical futility stop from 59/60 seed-1 primary "
            "runs; not a complete C1 atlas or three-seed bounded negative"
        ),
    }
    return {**body, "verdict_sha256": _canonical_sha(body)}


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


def _latest_partial_abort(result_root):
    candidates = sorted(
        (Path(result_root) / "coordinator_runs").glob(
            "phasec1_base_partial_abort_*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise ValueError("no Phase-C1 partial-abort record")
    return candidates[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(RESULT_ROOT))
    parser.add_argument("--partial-abort")
    parser.add_argument("--confirm-futility-stop", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_futility_stop:
        raise SystemExit("--confirm-futility-stop is required")
    result_root = Path(args.result_root)
    partial_abort = (
        Path(args.partial_abort)
        if args.partial_abort
        else _latest_partial_abort(result_root)
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        text=True,
    ).strip()
    payload = build_payload(
        result_root=result_root,
        manifest_path=result_root / "phasec_manifest.json",
        coordinate_manifest_path=(
            result_root / "phasec1_coordinate_manifest_dt.json"
        ),
        partial_abort_path=partial_abort,
        analysis_git_sha=git_sha,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    output = result_root / OUTPUT_NAME
    _publish_once(output, payload)
    print(json.dumps({
        "output": str(output),
        "verdict_sha256": payload["verdict_sha256"],
        "status": payload["status"],
        "completed_runs": payload["execution_coverage"]["completed_runs"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
