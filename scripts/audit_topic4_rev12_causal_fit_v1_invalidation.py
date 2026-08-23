#!/usr/bin/env python3
"""Invalidate the approximate binned-contact causal field fit without rescoring it."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def accounting(*, expected: int, completed: int, parity_failed: int,
               resource_stopped: int, never_launched: int) -> dict:
    values = [completed, parity_failed, resource_stopped, never_launched]
    if any(value < 0 for value in values) or sum(values) != int(expected):
        raise RuntimeError("invalidated queue accounting does not close")
    return {
        "expected_workers": int(expected),
        "completed_but_invalid": int(completed),
        "contact_parity_failed": int(parity_failed),
        "resource_safety_stopped": int(resource_stopped),
        "never_launched": int(never_launched),
        "accounting_exact": True,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.artifact_root.resolve() / config["output_root"]
    manifest = json.loads((root / "candidate_manifest.json").read_text())
    expected = len(manifest["candidates"]) * len(
        config["search"]["fit_network_seeds"]
    )
    completed_paths = sorted((root / "workers").glob("*.json"))
    if len(completed_paths) != 14:
        raise RuntimeError("Stage-I v1 completed-artifact inventory drifted")
    for path in completed_paths:
        payload = json.loads(path.read_text())
        if payload["contact_readout"]["source"] != (
                "lineage_restricted_sheet_activity"):
            raise RuntimeError("Stage-I v1 contains an unexpected readout source")
    failed = [
        "stage_i_anchor_01_seed_2212",
        "stage_i_a00_d00_s00_p_seed_2212",
    ]
    stopped = [
        "stage_i_anchor_02_seed_2211", "stage_i_anchor_02_seed_2212",
        "stage_i_midpoint_00_seed_2211", "stage_i_midpoint_00_seed_2212",
        "stage_i_midpoint_01_seed_2211", "stage_i_midpoint_01_seed_2212",
        "stage_i_midpoint_02_seed_2211", "stage_i_midpoint_02_seed_2212",
    ]
    counts = accounting(
        expected=expected, completed=len(completed_paths),
        parity_failed=len(failed), resource_stopped=len(stopped),
        never_launched=expected - len(completed_paths) - len(failed) - len(stopped),
    )
    payload = {
        "schema_id": "topic4_rev12_causal_fit_v1_invalidation_v1",
        "status": "INVALIDATED_APPROXIMATE_BINNED_CONTACT_READOUT",
        **counts,
        "completed_worker_stems": [path.stem for path in completed_paths],
        "parity_failed_worker_stems": failed,
        "resource_safety_stopped_worker_stems": stopped,
        "scientific_reason": (
            "The 1 mm population-averaged contact sampler did not preserve the "
            "frozen per-neuron virtual-contact readout for newly explored spatial "
            "activity patterns. Completed artifacts are retained but cannot rank fields."
        ),
        "replacement": (
            "Reuse the identical 54 frozen fields with root assignment at 1 mm/2 ms "
            "and exact per-neuron Gaussian contact sampling."
        ),
    }
    output = root / "aggregate" / "fit_v1_invalidation.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
