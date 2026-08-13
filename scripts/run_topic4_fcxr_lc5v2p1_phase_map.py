#!/usr/bin/env python3
"""Manifest-only launcher for the LC5v2.1 timescale-dose map.

This file deliberately exposes no legacy stages or free parameter flags.  A cell must be present in
the locked JSON manifest; reusable 18-s cells are accepted only when their recorded onset leaves the
full seven-second post-onset observation inside the stored trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5v2_natural_prefix as PREFIX  # noqa: E402


DEFAULT_MANIFEST = ROOT / "config/topic4_fcxr_lc5v2p1_timescale_dose_map.json"
RECEIPTS = PREFIX.U2.OUT / "lc5v2p1_phase_map_reuse_receipts"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell_key(tau_ms, gamma):
    return f"tau{int(round(float(tau_ms)))}_gamma{int(round(float(gamma) * 1000.0)):04d}"


def load_manifest(path=DEFAULT_MANIFEST):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported LC5v2.1 manifest schema")
    if payload.get("experiment_id") != "lc5v2p1_timescale_dose_map":
        raise ValueError("unexpected experiment_id")
    model = payload["model"]
    frozen = {
        "online_from_t0": True, "u0": 0.0, "p0_mode": "temporal_q099",
        "dynamic_z": True, "dynamic_h": True, "x_frozen": 1.0,
        "m_enabled": False, "kick": False, "parameter_step": False,
        "connection_seed": 1, "noise_seed": 401, "dt_ms": 0.05, "hill_h": 3,
    }
    if model != frozen:
        raise ValueError("manifest model contract drift")
    cells = {
        _cell_key(tau, gamma): (float(tau), float(gamma))
        for tau in payload["matrix"]["tau_ms"] for gamma in payload["matrix"]["gamma"]
    }
    if len(cells) != 9 or set(cells) != set(payload["reuse"]["eligible_cells"]):
        raise ValueError("manifest must define one reuse entry for every 3x3 cell")
    return path, payload, cells


def _write_json(path, payload):
    PREFIX._write_json(path, payload)


def validate_reuse(root, summary_path, *, tau_ms, gamma, observation, expected_input):
    path = (root / summary_path / "summary.json").resolve()
    summary = json.loads(path.read_text())
    if not PREFIX.np.isclose(float(summary["tau_ms"]), float(tau_ms), rtol=0.0, atol=1e-12):
        raise ValueError("reuse tau mismatch")
    if not PREFIX.np.isclose(
        float(summary["gamma_nominal_dose"]), float(gamma), rtol=0.0, atol=1e-12
    ):
        raise ValueError("reuse Gamma mismatch")
    if summary.get("p0_policy") != "q099":
        raise ValueError("reuse candidate is not a q99 arm")
    onset = summary.get("onset_ms")
    if onset is None:
        raise ValueError("an 18-s no-onset arm cannot be reused for the event-aligned map")
    required_ms = min(
        float(observation["max_end_ms"]),
        max(float(observation["min_end_ms"]), float(onset) + float(observation["post_onset_ms"])),
    )
    if float(summary["T_ms"]) + 1e-9 < required_ms:
        raise ValueError("reuse candidate lacks the required post-onset observation")
    observed_input = summary.get("external_input_prefix_18s_sha256", summary["external_input_sha256"])
    if observed_input != expected_input:
        raise ValueError("reuse external-input prefix mismatch")
    return {
        "status": "REUSED_ELIGIBLE", "source_summary": str(path),
        "source_summary_sha256": _sha(path), "tau_ms": float(tau_ms), "gamma": float(gamma),
        "required_observation_ms": required_ms, "source_T_ms": float(summary["T_ms"]),
        "source_outcome": summary["outcome"], "source_spike_sha256": summary["spike_sha256"],
        "external_input_prefix_18s_sha256": observed_input,
    }


def validate_control(root, summary_path, observation):
    path = (root / summary_path / "summary.json").resolve()
    summary = json.loads(path.read_text())
    parity = summary.get("control_parity", {})
    if not (parity.get("spike_exact") and parity.get("rate_exact") and
            float(parity.get("rate_max_abs_diff_hz", np.nan)) == 0.0):
        raise ValueError("pump-off control lacks exact spike/rate parity")
    onset = summary.get("onset_ms")
    required_ms = min(
        float(observation["max_end_ms"]),
        max(float(observation["min_end_ms"]), float(onset) + float(observation["post_onset_ms"])),
    )
    if float(summary["T_ms"]) + 1e-9 < required_ms:
        raise ValueError("pump-off control lacks required post-onset observation")
    return {
        "status": "REUSED_CONTROL_EXACT", "source_summary": str(path),
        "source_summary_sha256": _sha(path), "required_observation_ms": required_ms,
        "source_T_ms": float(summary["T_ms"]), "source_outcome": summary["outcome"],
        "source_spike_sha256": summary["spike_sha256"],
        "external_input_hash_note": (
            "legacy control stores the full U1-run input hash, not an 18-s prefix hash; exact "
            "spike/rate parity is the control contract, while every new arm separately asserts "
            "the locked 18-s input-prefix hash"
        ),
    }


def run_cell(cell, manifest_path=DEFAULT_MANIFEST):
    manifest_path, manifest, cells = load_manifest(manifest_path)
    if cell == "control":
        receipt = validate_control(
            ROOT, Path(manifest["reuse"]["control"]), manifest["observation"]
        )
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        receipt["manifest_sha256"] = _sha(manifest_path)
        _write_json(RECEIPTS / "control.json", receipt)
        return receipt
    if cell not in cells:
        raise ValueError(f"cell {cell!r} is not in the locked 3x3 manifest")
    tau_ms, gamma = cells[cell]
    expected_input = manifest["source"]["expected_external_input_prefix_18s_sha256"]
    reuse = manifest["reuse"]["eligible_cells"][cell]
    if reuse is not None:
        receipt = validate_reuse(
            ROOT, Path(reuse), tau_ms=tau_ms, gamma=gamma,
            observation=manifest["observation"], expected_input=expected_input,
        )
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        receipt["manifest_sha256"] = _sha(manifest_path)
        _write_json(RECEIPTS / f"{cell}.json", receipt)
        return receipt
    observation = manifest["observation"]
    return PREFIX.stage_prefix(
        gamma, "q099", tau_ms,
        protocol_id=manifest["experiment_id"],
        min_run_ms=float(observation["min_end_ms"]),
        max_run_ms=float(observation["max_end_ms"]),
        post_onset_ms=float(observation["post_onset_ms"]),
        expected_input_prefix_18s=expected_input,
        baseline_eval_end_ms=float(manifest["source"]["baseline_window_ms"][1]),
        protocol_manifest_sha256=_sha(manifest_path),
        extra_mechanism_files=(Path(__file__), manifest_path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC5v2.1 40k/reuse publication requires --confirm-run")
    print(json.dumps(PREFIX.json_sanitize(run_cell(args.cell, args.manifest)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
