#!/usr/bin/env python3
"""Apply the pre-formal event-tolerant LOW gate to completed canary points."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_phase_diagram import (  # noqa: E402
    classify_stationary_branch,
    scientific_contract_digest,
    stationary_rate_metrics,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def reclassify(path):
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if record.get("status") != "SPATIAL_ZM_PHASE_POINT_COMPLETE":
        return None
    npz_path = Path(record["trajectory_npz"]["path"])
    observed_npz_sha = _sha256(npz_path)
    if observed_npz_sha != record["trajectory_npz"]["sha256"]:
        raise RuntimeError(f"trajectory NPZ hash changed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as handle:
        rate = np.asarray(handle["rate_E_hz"], float)
    rate_metrics = stationary_rate_metrics(
        rate, dt_ms=float(record["simulation"]["dt_ms"]),
        burn_in_ms=float(record["simulation"]["burn_in_ms"]),
    )
    prior = record["classification"]
    history = list(record.get("classification_history", []))
    if not any(row.get("contract_version") == "q95_low_v1" for row in history):
        history.append({
            "contract_version": prior.get("contract_version", "q95_low_v1"),
            "classification": prior,
            "reason_superseded": (
                "q95 rejected eventful interictal low states even when no "
                "120-Hz episode persisted for the operational 100-ms runaway hold"),
        })
    record["stationary_metrics"].update(rate_metrics)
    record["classification"] = classify_stationary_branch(
        record["stationary_metrics"],
        numerically_stable=bool(record["numerical_stability"]["all_checks_pass"]),
    )
    record["classification_history"] = history
    record["classification_amendment"] = {
        "date": "2026-09-03",
        "stage": "pre-formal Stage 0 instrument calibration",
        "changed_high_state_gate": False,
        "changed_low_state_gate": True,
        "old_low_tail_gate": "q95 rate <120 Hz",
        "new_low_persistence_gate": "longest 20-ms-smoothed rate >=120 Hz is <100 ms",
        "reason": "permit sparse interictal-like events while excluding sustained high activity",
    }
    digest, contract = scientific_contract_digest(record)
    record["scientific_contract_sha256"] = digest
    record["scientific_contract"] = contract
    _atomic_json(record, path)
    return {
        "path": str(path),
        "old_label": prior["label"],
        "new_label": record["classification"]["label"],
        "scientific_contract_sha256": digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    outputs = []
    for path in sorted(Path(args.input_dir).rglob("*.json")):
        result = reclassify(path)
        if result is not None:
            outputs.append(result)
    print(json.dumps({"status": "RECLASSIFICATION_COMPLETE", "records": outputs},
                     sort_keys=True))


if __name__ == "__main__":
    main()
