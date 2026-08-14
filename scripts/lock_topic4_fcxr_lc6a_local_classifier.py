#!/usr/bin/env python3
"""Freeze the C0-derived local companion classifier before reading Q-arm trajectories."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc5 import SparseSpikeStream  # noqa: E402
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    apply_local_classifier, calibrate_local_classifier, spatial_rate_maps,
)


OUT = NAT.OUT
LOCK = OUT / "local_classifier_manifest_addendum.json"
C0_READOUT = OUT / "local_classifier_readouts/C0.json"
DONE = OUT / "DONE_LC6A_LOCAL_CLASSIFIER_LOCK.json"
FAILED = OUT / "FAILED_LC6A_LOCAL_CLASSIFIER_LOCK.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    NAT._write_json(path, payload)


def _load_c0():
    arm = OUT / "trajectories/C0"
    summary_path = arm / "summary.json"
    spikes_path = arm / "spikes.npz"
    spatial_path = arm / "spatial_readouts.npz"
    for path in (summary_path, spikes_path, spatial_path):
        if not path.is_file():
            raise RuntimeError(f"C0 local-classifier source is missing: {path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "COMPLETE" or summary.get("condition") != "C0":
        raise RuntimeError("C0 natural trajectory is not a complete canonical source")
    if summary.get("control_parity", {}).get("spike_exact") is not True:
        raise RuntimeError("C0 control parity must be exact before local-classifier lock")
    with np.load(spikes_path, allow_pickle=False) as z:
        stream = SparseSpikeStream(
            np.asarray(z["steps"], np.int64), np.asarray(z["cells"], np.int32),
            int(z["n_steps"][0]), int(z["n_cells"][0]),
        )
        expected_stream_hash = str(z["sha256"][0])
    if stream.sha256 != expected_stream_hash or stream.sha256 != summary["spike_sha256"]:
        raise RuntimeError("C0 spike stream hash mismatch")
    with np.load(spatial_path, allow_pickle=False) as z:
        bins = np.asarray(z["cell_bins"], np.int32)
        occupancy = np.asarray(z["occupancy"], np.int32)
    if bins.size != stream.n_cells:
        raise RuntimeError("C0 spatial bins are not aligned to E cells")
    return summary, stream, bins, occupancy, summary_path, spikes_path, spatial_path


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    if LOCK.is_file() and C0_READOUT.is_file() and DONE.is_file():
        payload = json.loads(LOCK.read_text())
        if payload.get("manifest_sha256") != _sha(manifest_path):
            raise RuntimeError("existing local-classifier lock manifest hash mismatch")
        expected_sources = {
            str(Path(__file__).relative_to(ROOT)): _sha(__file__),
            "src/topic4_fcxr_lc6_trajectory.py": _sha(
                ROOT / "src/topic4_fcxr_lc6_trajectory.py"
            ),
        }
        if payload.get("source_sha256") != expected_sources:
            raise RuntimeError("existing local-classifier lock source hash mismatch")
        for key, hash_key in (
            ("C0_summary", "C0_summary_sha256"),
            ("C0_spikes", "C0_spikes_sha256"),
            ("C0_spatial_readouts", "C0_spatial_readouts_sha256"),
        ):
            if _sha(payload[key]) != payload[hash_key]:
                raise RuntimeError(f"existing local-classifier source drift: {key}")
        return payload
    summary, stream, bins, occupancy, summary_path, spikes_path, spatial_path = _load_c0()
    rate = NAT._rate_from_stream(stream)
    adjudication = NAT.PREFIX._adjudicate(stream, rate)
    onset_ms = adjudication.get("onset_ms")
    if onset_ms is None:
        raise RuntimeError("C0 has no global onset for pre-onset local-classifier calibration")
    maps = spatial_rate_maps(
        stream.steps, stream.cells, bins, occupancy, n_steps=stream.n_steps,
        dt_ms=NAT.U2.DT_MS, window_ms=100.0,
    )
    thresholds = calibrate_local_classifier(
        maps, occupancy, adjudication["returned"], onset_ms=float(onset_ms),
        sheet_size_mm=float(NAT.U2.PP.L), rate_quantile=.995, area_quantile=.99,
        window_ms=100.0, persistence_ms=500.0,
    )
    readout = apply_local_classifier(maps, occupancy, thresholds)
    payload = {
        "status": "LOCKED",
        "stage": "LC6A_LOCAL_CLASSIFIER_LOCK",
        "scientific_role": "C0_pre_onset_returning_IED_tail_companion_classifier",
        "manifest": str(manifest_path), "manifest_sha256": _sha(manifest_path),
        "C0_summary": str(summary_path), "C0_summary_sha256": _sha(summary_path),
        "C0_spikes": str(spikes_path), "C0_spikes_sha256": _sha(spikes_path),
        "C0_spatial_readouts": str(spatial_path),
        "C0_spatial_readouts_sha256": _sha(spatial_path),
        "C0_graph_sha256": summary["graph_sha256"],
        "C0_global_onset_ms": float(onset_ms),
        "thresholds": thresholds,
        "selection_used_Q_trajectory_outcomes": False,
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): _sha(__file__),
            "src/topic4_fcxr_lc6_trajectory.py": _sha(
                ROOT / "src/topic4_fcxr_lc6_trajectory.py"
            ),
        },
    }
    _write_json(LOCK, payload)
    _write_json(C0_READOUT, {
        "status": "COMPLETE", "condition": "C0",
        "lock": str(LOCK), "lock_sha256": _sha(LOCK),
        "readout": readout,
    })
    _write_json(DONE, {
        "status": "DONE", "lock": str(LOCK), "lock_sha256": _sha(LOCK),
        "C0_readout": str(C0_READOUT),
    })
    FAILED.unlink(missing_ok=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("local-classifier lock requires --confirm-run")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".local_classifier_lock.lock").open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("local-classifier lock is already running") from exc
        try:
            result = run(args.execution_manifest)
            print(json.dumps(NAT._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(FAILED, {
                "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            })
            raise


if __name__ == "__main__":
    main()
