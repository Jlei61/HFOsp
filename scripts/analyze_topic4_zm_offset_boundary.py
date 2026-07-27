#!/usr/bin/env python
"""Aggregate existing-coordinate offset surfaces and dynamic Z/M realization."""

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
import src.topic4_zm_minimal_carrier as MC  # noqa: E402


OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")
FAMILIES = ("M_alone", "M_SG", "M_Z_recovery")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    paths = sorted(
        glob.glob(os.path.join(OUT, "boundaries", "offset", "seed*", "offset_probes.json"))
    )
    manifests = [json.load(open(path)) for path in paths]
    complete = [manifest for manifest in manifests if manifest.get("complete")]
    rows = [
        row
        for manifest in complete
        for row in manifest.get("rows", [])
        if row.get("completed") and row.get("response_valid")
    ]
    family_results = {}
    if len(complete) >= 2:
        for family in FAMILIES:
            active = [
                row for row in rows
                if row.get("family") == family
                and row.get("initial_kind") == "active"
                and "lambda" in row
            ]
            boundary = BD.bootstrap_half_boundary(
                active,
                "lambda",
                "remained_carrier",
                expected_direction="decreasing",
                n_boot=2000,
                seed=20260727,
            )
            q_half = boundary.get("q_half")
            actual_crossing = (
                BD.trajectory_crossing(
                    [0.0, 1.0], q_half, expected_direction="increasing"
                )
                if boundary.get("status") == "bracketed"
                and q_half is not None
                else {"crossed": False, "direction_ok": False}
            )
            low = [
                row for row in rows
                if row.get("family") == family
                and row.get("initial_kind") == "low"
            ]
            family_results[family] = {
                "boundary": boundary,
                "actual_0_to_1_crossing": actual_crossing,
                "n_active_rows": len(active),
                "n_low_rows": len(low),
                "low_basin_persistence_fraction": (
                    sum(bool(row.get("low_basin_persisted")) for row in low)
                    / len(low) if low else None
                ),
                "boundary_within_actual_range": bool(
                    boundary.get("status") == "bracketed"
                    and q_half is not None
                    and 0.0 <= q_half <= 1.0
                ),
                "boundary_in_locked_extension": bool(
                    boundary.get("status") == "bracketed"
                    and q_half is not None
                    and 1.0 < q_half <= 1.25
                ),
            }
    dynamic = [
        row for row in rows if row.get("family") == "dynamic_ZM"
    ]
    dynamic_offset_rows = [
        {
            "offset_reached": bool(
                row.get("end_reason") == "dead_in_rest_basin"
            )
        }
        for row in dynamic
    ]
    dynamic_posterior = (
        MC.jeffreys_posterior(
            sum(row["offset_reached"] for row in dynamic_offset_rows),
            len(dynamic_offset_rows),
        )
        if dynamic_offset_rows else None
    )
    dynamic_reached = bool(
        len({int(row["seed"]) for row in dynamic}) >= 2
        and dynamic_posterior is not None
        and dynamic_posterior["median"] > 0.8
    )

    verdict = "no_evidence"
    if len(complete) >= 2:
        if family_results["M_alone"]["boundary_within_actual_range"]:
            verdict = "M_sufficient_and_reached"
        elif family_results["M_SG"]["boundary_within_actual_range"]:
            verdict = "M_SG_joint_offset_reached"
        elif (
            family_results["M_Z_recovery"]["boundary_within_actual_range"]
            and dynamic_reached
        ):
            verdict = "M_Z_recovery_offset_reached"
        elif any(
            result["boundary_in_locked_extension"]
            for result in family_results.values()
        ):
            verdict = "M_boundary_near_but_unreached"
        elif (
            family_results["M_Z_recovery"]["boundary_within_actual_range"]
            and not dynamic_reached
        ):
            verdict = "M_Z_recovery_boundary_exists_but_unreached"
        else:
            verdict = "M_shapes_but_no_offset_surface"

    output = {
        "verdict": verdict,
        "n_complete_seeds": len(complete),
        "seeds": sorted({int(manifest["seed"]) for manifest in complete}),
        "family_results": family_results,
        "dynamic_ZM": {
            "n_rows": len(dynamic),
            "posterior_offset_reached": dynamic_posterior,
            "reached": dynamic_reached,
            "definition": (
                "dead_in_rest_basin under dynamic Z+M with the S_G family frozen"
            ),
        },
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
            "offset from the carrier basin only; dead/rest return is not "
            "returning interictal-event recovery and does not establish lifecycle"
        ),
    }
    path = os.path.join(OUT, "boundaries", "offset", "offset_boundary_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, path)
    print(
        f"[offset-summary] verdict={verdict} seeds={output['seeds']} -> {path}"
    )


if __name__ == "__main__":
    main()
