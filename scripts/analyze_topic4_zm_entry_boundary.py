#!/usr/bin/env python
"""Aggregate conditional Z-entry probes and locate P_enter=0.5 fail-closed."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_boundaries as BD  # noqa: E402


OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    paths = sorted(
        glob.glob(os.path.join(OUT, "boundaries", "entry", "seed*", "entry_probes.json"))
    )
    manifests = [json.load(open(path)) for path in paths]
    complete = [manifest for manifest in manifests if manifest.get("complete")]
    rows = [
        row
        for manifest in complete
        for row in manifest.get("rows", [])
        if row.get("completed")
    ]
    if len(complete) < 2:
        boundary = {
            "status": "insufficient_seeds",
            "q_half": None,
            "q_half_ci": None,
        }
        crossing = {
            "crossed": False,
            "direction_ok": False,
        }
        verdict = "no_evidence"
    else:
        boundary = BD.bootstrap_half_boundary(
            rows,
            "lambda",
            "entered_carrier",
            expected_direction="increasing",
            n_boot=2000,
            seed=20260727,
            cluster_key="seed",
        )
        reachability = BD.boundary_reachability(
            boundary,
            [0.0, 1.0],
            expected_direction="increasing",
            reachable_range=(0.0, 1.0),
        )
        crossing = (
            reachability["crossing"]
            or {"crossed": False, "direction_ok": False}
        )
        verdict = (
            "conditional_Z_entry_boundary_crossed"
            if reachability["reached"]
            else "conditional_Z_entry_boundary_unresolved"
        )
    end_counts = {}
    for row in rows:
        key = row.get("end_reason") or "survived"
        end_counts[key] = end_counts.get(key, 0) + 1
    output = {
        "verdict": verdict,
        "n_complete_seeds": len(complete),
        "seeds": sorted({int(manifest["seed"]) for manifest in complete}),
        "n_rows": len(rows),
        "boundary": boundary,
        "trajectory_crossing": crossing,
        "reachability": (
            reachability if len(complete) >= 2 else None
        ),
        "end_reason_counts": end_counts,
        "coordinate_family": "conditional_z_slice",
        "inputs": [
            {
                "path": os.path.relpath(path, _ROOT),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
        "boundary_version": BD.BOUNDARY_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "claim_boundary": (
            "conditional actual-field Z slice with onset-context M/S_G held; "
            "not global Z sufficiency and not offset/recovery/lifecycle evidence"
        ),
    }
    path = os.path.join(OUT, "boundaries", "entry", "entry_boundary_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, path)
    print(
        f"[entry-summary] verdict={verdict} seeds={output['seeds']} "
        f"rows={len(rows)} -> {path}"
    )


if __name__ == "__main__":
    main()
