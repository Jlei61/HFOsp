#!/usr/bin/env python3
"""Adjudicate the preregistered Phase-D zero-spike dominance panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_fast_carrier_calibration as CAL  # noqa: E402
from src import topic4_zm_fast_carrier_contract as C  # noqa: E402


BASE = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair"
RUN_ROOT = BASE / "calibration/dynamic_preentry"
INPUT_FINAL = BASE / "phaseD_input_manifest_v1_5.json"
INPUT_RUN = BASE / "phaseD_input_manifest_v1_4.json"
OUTPUT = BASE / "calibration/calibration_dominance_verdict.json"
SCALE_I = (0.8, 1.0, 1.2)
N_E = 32_000
BIN_MS = 25.0
T_MS = 8_500.0


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_receipt(path: Path) -> dict:
    payload = json.loads(path.read_text())
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("manifest_sha256") != _canonical_sha(body):
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    if _sha(ROOT / payload["array_path"]) != payload["array_file_sha256"]:
        raise RuntimeError(f"array hash mismatch: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-adjudicate", action="store_true")
    args = parser.parse_args()
    if not args.confirm_adjudicate:
        raise SystemExit("refusing verdict write without --confirm-adjudicate")

    final_lock = json.loads(INPUT_FINAL.read_text())
    C.validate_input_manifest(final_lock, ROOT)
    run_lock = json.loads(INPUT_RUN.read_text())
    if run_lock["manifest_sha256"] != "c454a84c8610e7b5d8a56bad37b947dbbc74d006326cbe443e73d06def5f249f":
        raise RuntimeError("v1.4 execution lock drift")
    reference_path = RUN_ROOT / "reference__noise_replay.json"
    reference = _read_receipt(reference_path)
    if reference["mode"] != "reference" or reference["T_ms"] != T_MS:
        raise RuntimeError("dynamic reference scope drift")
    if reference["candidate_outcomes_accessed"]:
        raise RuntimeError("dynamic reference accessed a candidate outcome")
    if reference["returning_events"] != {
        "median_duration_ms": 75.0,
        "median_peak_hz": 68.58024691358025,
        "n_events": 15,
    }:
        raise RuntimeError("dynamic reference no longer reproduces source IEDs")

    rows, evidence = [], []
    for scale_I in SCALE_I:
        stem = f"sE1.2_sI{scale_I:g}_sM1__noise_replay"
        receipt_path = RUN_ROOT / f"{stem}.json"
        receipt = _read_receipt(receipt_path)
        if receipt["input_manifest_sha256"] != run_lock["manifest_sha256"]:
            raise RuntimeError(f"cell input-lock drift: {stem}")
        if receipt["mode"] != "cell" or receipt["T_ms"] != T_MS:
            raise RuntimeError(f"cell scope drift: {stem}")
        if receipt["scales"] != [1.2, float(scale_I), 1.0]:
            raise RuntimeError(f"cell scale-label drift: {stem}")
        if receipt["candidate_outcomes_accessed"]:
            raise RuntimeError(f"cell accessed a candidate outcome: {stem}")
        if receipt["external_drive_sha256"] != reference["external_drive_sha256"]:
            raise RuntimeError(f"cell/reference external-drive mismatch: {stem}")
        arrays_path = ROOT / receipt["array_path"]
        with np.load(arrays_path, allow_pickle=False) as arrays:
            r_all = np.asarray(arrays["r_all"], float)
            active = np.asarray(arrays["active_fraction"], float)
        expected_bins = int(round(T_MS / BIN_MS))
        if r_all.shape != (expected_bins,) or active.shape != (expected_bins,):
            raise RuntimeError(f"cell summary-shape drift: {stem}")
        if not np.all(np.isfinite(r_all)) or not np.all(np.isfinite(active)):
            raise RuntimeError(f"cell contains non-finite observables: {stem}")
        if np.any(r_all < 0) or np.any((active < 0) | (active > 1)):
            raise RuntimeError(f"cell observable bounds violated: {stem}")
        total = int(round(float(np.sum(r_all) * N_E * BIN_MS / 1000.0)))
        row = {
            "scale_E": 1.2,
            "scale_I": float(scale_I),
            "scale_M": 1.0,
            "external_drive_sha256": receipt["external_drive_sha256"],
            "total_e_spikes": total,
            "returning_event_count": int(receipt["returning_events"]["n_events"]),
            "peak_active_fraction": float(np.max(active)),
            "runaway_early_stop_ms": receipt["runaway_early_stop_ms"],
        }
        rows.append(row)
        evidence.append(
            {
                **row,
                "receipt_path": str(receipt_path.relative_to(ROOT)),
                "receipt_file_sha256": _sha(receipt_path),
                "receipt_manifest_sha256": receipt["manifest_sha256"],
                "array_path": receipt["array_path"],
                "array_file_sha256": receipt["array_file_sha256"],
                "median_vinf_mv": receipt["diagnostics"]["median_vinf_mv"],
                "median_tau_eff_ms": receipt["diagnostics"]["median_tau_eff_ms"],
            }
        )
    verdict = CAL.zero_spike_dominance_stop(rows)
    if not verdict["stop"]:
        raise RuntimeError("dominance stop did not pass; full lattice is required")
    body = {
        "schema": "zm_fast_carrier_calibration_dominance_verdict_v1_2026-07-31",
        "status": "complete_preregistered_cheap_stop",
        "verdict": verdict["verdict"],
        "scientific_stage_reached": "baseline_preservation_only",
        "full_lattice_required": False,
        "full_lattice_completed": False,
        "reason_full_lattice_not_required": "registered first-spike dominance stop passed",
        "final_input_lock": {
            "path": str(INPUT_FINAL.relative_to(ROOT)),
            "file_sha256": _sha(INPUT_FINAL),
            "manifest_sha256": final_lock["manifest_sha256"],
        },
        "execution_input_lock": {
            "path": str(INPUT_RUN.relative_to(ROOT)),
            "file_sha256": _sha(INPUT_RUN),
            "manifest_sha256": run_lock["manifest_sha256"],
            "relationship": (
                "v1.5 adds the pre-result dominance adjudication rule only; "
                "cell equations, scales, source and 8.5s dynamic window are unchanged"
            ),
        },
        "reference": {
            "path": str(reference_path.relative_to(ROOT)),
            "file_sha256": _sha(reference_path),
            "manifest_sha256": reference["manifest_sha256"],
            "returning_events": reference["returning_events"],
            "median_e_rate_hz": reference["median_e_rate_hz"],
        },
        "dominance_proof": verdict,
        "evidence": evidence,
        "claim_boundary": {
            "baseline_conductance_calibration_passed": False,
            "fast_carrier_tested": False,
            "spatial_carrier_tested": False,
            "perturbation_return_tested": False,
            "entry_tested": False,
            "offset_tested": False,
            "recovery_tested": False,
            "ictal_lifecycle_established": False,
            "general_conductance_inhibition_falsified": False,
        },
        "production_authorized": False,
    }
    payload = {**body, "verdict_sha256": _canonical_sha(body)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if OUTPUT.exists() and OUTPUT.read_text() != text:
        raise RuntimeError(f"refusing to overwrite different verdict: {OUTPUT}")
    if not OUTPUT.exists():
        tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, OUTPUT)
    print(OUTPUT)
    print(payload["verdict_sha256"])


if __name__ == "__main__":
    main()
