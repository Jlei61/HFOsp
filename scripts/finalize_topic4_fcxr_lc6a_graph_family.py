#!/usr/bin/env python3
"""Atomically finalize the parallel LC6A graph-family build.

The original family runner builds conditions serially.  Q1/Q2/Q3 are now built
by independent, manifest-locked workers while the serial runner finishes C1.
This finalizer performs no graph construction and reads no trajectory outcome:
it verifies the five atomic graph artifacts and assembles the canonical family
audit once every independent condition is complete.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_topic4_fcxr_lc6a_graph_family as FAMILY  # noqa: E402
from src.topic4_fcxr_lc6_surround import EToIGraph, graph_sha256  # noqa: E402


GRAPH_IDS = ("C0", "C1", "Q1", "Q2", "Q3")
PARALLEL_IDS = ("Q1", "Q2", "Q3")


def _load_verified_graph(path: Path) -> tuple[EToIGraph, dict, str]:
    with np.load(path, allow_pickle=False) as z:
        graph = EToIGraph(
            np.asarray(z["sources"], np.int32),
            np.asarray(z["weights"]),
            np.asarray(z["delay_steps"], np.int32),
        )
        stored = str(z["graph_sha256"][0])
        metadata = json.loads(str(z["metadata_json"][0]))
    actual = graph_sha256(graph)
    if stored != actual or metadata.get("graph_sha256") != actual:
        raise RuntimeError(f"graph artifact hash mismatch: {path}")
    return graph, metadata, actual


def _same_json(left, right) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def finalize(manifest_path: Path) -> dict:
    manifest_path, _manifest = FAMILY._validate_manifest(manifest_path)
    started = time.time()
    audits: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for condition in GRAPH_IDS:
        graph_path = FAMILY.OUT / f"graphs/{condition}.npz"
        if not graph_path.is_file():
            raise RuntimeError(f"missing atomic graph artifact: {condition}")
        _graph, metadata, digest = _load_verified_graph(graph_path)
        if metadata.get("graph_legality", "PASS") != "PASS":
            raise RuntimeError(f"graph legality failed: {condition}")
        if condition in PARALLEL_IDS:
            done = FAMILY.OUT / f"DONE_LC6A_GRAPH_{condition}.json"
            audit_path = FAMILY.OUT / f"graph_condition_{condition}.json"
            if not done.is_file() or not audit_path.is_file():
                raise RuntimeError(f"parallel graph worker is not complete: {condition}")
            sidecar = json.loads(audit_path.read_text())
            if sidecar.get("condition") != condition:
                raise RuntimeError(f"condition sidecar identity mismatch: {condition}")
            if sidecar.get("manifest_sha256") != FAMILY._sha(manifest_path):
                raise RuntimeError(f"condition manifest hash mismatch: {condition}")
            if not _same_json(metadata, sidecar):
                raise RuntimeError(f"embedded/sidecar audit mismatch: {condition}")
        audits[condition] = metadata
        hashes[condition] = digest

    reference_candidates = [
        audits[key].get("frozen_reference_widths") for key in PARALLEL_IDS
    ]
    if any(value is None for value in reference_candidates):
        raise RuntimeError("parallel condition is missing frozen reference widths")
    reference = reference_candidates[0]
    if any(not _same_json(reference, value) for value in reference_candidates[1:]):
        raise RuntimeError("parallel conditions disagree on frozen reference widths")
    if not np.isclose(
        float(reference["c0_construction_q"]),
        float(audits["C0"]["construction_q"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("C0 construction coordinate differs from frozen references")

    payload = {
        "status": "COMPLETE",
        "stage": "LC6A_GRAPH_FAMILY",
        "build_mode": "C0_C1_SERIAL_PLUS_Q1_Q2_Q3_PARALLEL",
        "manifest": str(manifest_path),
        "manifest_sha256": FAMILY._sha(manifest_path),
        "trajectory_outcome_read": False,
        "graph_ids": list(GRAPH_IDS),
        "graph_sha256": hashes,
        "frozen_reference_widths": reference,
        "audits": audits,
        "all_graphs_legal": True,
        "resource_end": FAMILY._meminfo(),
        "finalizer_wall_s": time.time() - started,
    }
    FAMILY._write_json(FAMILY.OUT / "graph_audit.json", payload)
    FAMILY._write_json(
        FAMILY.OUT / "DONE_LC6A_GRAPH_FAMILY.json",
        {
            "status": "DONE",
            "all_graphs_legal": True,
            "build_mode": payload["build_mode"],
            "graph_audit": str(FAMILY.OUT / "graph_audit.json"),
            "graph_sha256": hashes,
        },
    )
    stale_running = FAMILY.OUT / "RUNNING_LC6A_GRAPH_FAMILY.json"
    if stale_running.is_file():
        FAMILY._write_json(
            FAMILY.OUT / "SUPERSEDED_LC6A_GRAPH_FAMILY_SERIAL.json",
            {
                "status": "SUPERSEDED_BY_PARALLEL_FINALIZER",
                "reason": "serial runner was intentionally stopped after atomic C1 save",
                "original_running": json.loads(stale_running.read_text()),
            },
        )
        stale_running.unlink()
    (FAMILY.OUT / "FAILED_LC6A_GRAPH_FAMILY.json").unlink(missing_ok=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-manifest",
        type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("parallel graph finalization requires --confirm-run")
    FAMILY.OUT.mkdir(parents=True, exist_ok=True)
    with (FAMILY.OUT / ".graph_family_finalize.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("LC6A graph finalizer is already running") from exc
        try:
            payload = finalize(args.execution_manifest)
            print(json.dumps(FAMILY._jsonable(payload), indent=2, sort_keys=True))
        except BaseException as exc:
            FAMILY._write_json(
                FAMILY.OUT / "FAILED_LC6A_GRAPH_FAMILY_FINALIZE.json",
                {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"},
            )
            raise
        else:
            (FAMILY.OUT / "FAILED_LC6A_GRAPH_FAMILY_FINALIZE.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
