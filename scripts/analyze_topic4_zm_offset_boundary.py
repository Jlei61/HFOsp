#!/usr/bin/env python
"""Aggregate existing-coordinate offset surfaces and dynamic Z/M realization."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
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
PRIMARY_SEEDS = (1, 3, 4)
BASE_LEVELS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
BASE_INITIAL_KINDS = ("active", "low")
BASE_REPLICATE = "noise_replay"
REQUIRED_STATE_LABELS = (
    "pre_entry__natural",
    "bounded_early__peak",
    "bounded_mid__peak",
    "bounded_late__peak",
)
SPEC_PATH = os.path.join(
    _ROOT,
    "docs",
    "superpowers",
    "specs",
    "2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md",
)
CANONICAL_CONFIG_PATH = os.path.join(OUT, "phase0", "canonical_config.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_offset_manifest_contract(manifests, anchors):
    """Validate the locked three-seed Phase-2B coverage before adjudication."""

    reasons = []
    by_seed = {}
    for manifest in manifests:
        try:
            seed = int(manifest["seed"])
        except (KeyError, TypeError, ValueError):
            reasons.append("manifest_missing_seed")
            continue
        if seed in by_seed:
            reasons.append(f"duplicate_manifest_seed:{seed}")
        by_seed[seed] = manifest
    if set(by_seed) != set(PRIMARY_SEEDS):
        reasons.append(
            f"manifest_seed_set:{sorted(by_seed)}!=expected:{list(PRIMARY_SEEDS)}"
        )

    reference_engine = None
    common_fields = {
        "boundary_version": BD.BOUNDARY_VERSION,
        "metrics_version": MC.METRICS_VERSION,
        "response_ms": 8000.0,
        "dt": 0.1,
        "resolution": "dt",
        "state_schema": "zm_sim_state_v1",
        "families": list(FAMILIES),
        "levels": list(BASE_LEVELS),
    }
    required_base = {
        (family, float(level), initial_kind, BASE_REPLICATE)
        for family in FAMILIES
        for level in BASE_LEVELS
        for initial_kind in BASE_INITIAL_KINDS
    }
    row_counts = {}
    for seed in PRIMARY_SEEDS:
        manifest = by_seed.get(seed)
        if manifest is None:
            continue
        if manifest.get("complete") is not True:
            reasons.append(f"seed{seed}:manifest_not_complete")
        if manifest.get("pending_cells"):
            reasons.append(f"seed{seed}:pending_cells_nonempty")
        for field, expected in common_fields.items():
            if manifest.get(field) != expected:
                reasons.append(f"seed{seed}:{field}_mismatch")
        engine = manifest.get("engine_sha256")
        if not isinstance(engine, dict) or not engine:
            reasons.append(f"seed{seed}:engine_sha256_missing")
        elif reference_engine is None:
            reference_engine = engine
        elif engine != reference_engine:
            reasons.append(f"seed{seed}:engine_sha256_mismatch")
        source_hashes = manifest.get("source_state_hashes") or {}
        if any(not source_hashes.get(label) for label in REQUIRED_STATE_LABELS):
            reasons.append(f"seed{seed}:source_state_hashes_incomplete")
        anchor = anchors.get(seed) or {}
        if not anchor:
            reasons.append(f"seed{seed}:anchor_missing")
        elif manifest.get("config_sha") != anchor.get("config_sha"):
            reasons.append(f"seed{seed}:anchor_config_sha_mismatch")

        rows = manifest.get("rows") or []
        keys = [row.get("key") for row in rows]
        if None in keys or len(keys) != len(set(keys)):
            reasons.append(f"seed{seed}:row_keys_missing_or_duplicate")
        base_seen = set()
        for row in rows:
            if int(row.get("seed", -1)) != seed:
                reasons.append(f"seed{seed}:foreign_row_seed")
            if row.get("completed") and row.get("response_valid"):
                if not row.get("bank_sha"):
                    reasons.append(f"seed{seed}:valid_row_missing_bank_sha")
                if row.get("boundary_version") != BD.BOUNDARY_VERSION:
                    reasons.append(f"seed{seed}:row_boundary_version_mismatch")
            if (
                row.get("family") in FAMILIES
                and row.get("replicate") == BASE_REPLICATE
                and row.get("initial_kind") in BASE_INITIAL_KINDS
                and "lambda" in row
            ):
                cell = (
                    row["family"],
                    float(row["lambda"]),
                    row["initial_kind"],
                    row["replicate"],
                )
                if cell in required_base:
                    base_seen.add(cell)
                    if not row.get("completed") or not row.get("response_valid"):
                        reasons.append(f"seed{seed}:invalid_required_base_cell:{cell}")
        missing_base = sorted(required_base - base_seen)
        if missing_base:
            reasons.append(f"seed{seed}:missing_required_base_cells:{missing_base}")
        row_counts[str(seed)] = len(rows)

    return {
        "passed": not reasons,
        "reasons": reasons,
        "required_seeds": list(PRIMARY_SEEDS),
        "required_base_cells_per_seed": len(required_base),
        "row_counts": row_counts,
        "producer_git_sha_note": (
            "legacy rows may lack row-level producer_git_sha; manifest "
            "contributing_git_shas remain provenance only, never positive evidence"
        ),
    }


def main():
    paths = sorted(
        glob.glob(os.path.join(OUT, "boundaries", "offset", "seed*", "offset_probes.json"))
    )
    manifests = [json.load(open(path)) for path in paths]
    anchor_paths = sorted(
        glob.glob(os.path.join(OUT, "anchors", "seed*", "anchor.json"))
    )
    anchors = {
        int(anchor["seed"]): anchor
        for anchor in (json.load(open(path)) for path in anchor_paths)
    }
    contract_audit = validate_offset_manifest_contract(manifests, anchors)
    complete = [manifest for manifest in manifests if manifest.get("complete")]
    all_rows = [
        row
        for manifest in manifests
        for row in manifest.get("rows", [])
        if row.get("completed")
    ]
    rows = [
        row
        for row in all_rows
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
                cluster_key="seed",
            )
            q_half = boundary.get("q_half")
            reachability = BD.boundary_reachability(
                boundary,
                [0.0, 1.0],
                expected_direction="increasing",
                reachable_range=(0.0, 1.0),
            )
            actual_crossing = (
                reachability["crossing"]
                or {"crossed": False, "direction_ok": False}
            )
            low = [
                row for row in rows
                if row.get("family") == family
                and row.get("initial_kind") == "low"
            ]
            base_active_by_level = {
                float(row["lambda"]): bool(row["remained_carrier"])
                for row in active
                if row.get("replicate") == "noise_replay"
            }
            base_low_by_level = {
                float(row["lambda"]): bool(row.get("low_basin_persisted"))
                for row in low
                if row.get("replicate") == "noise_replay"
            }
            coexistence_levels = sorted(
                level
                for level in set(base_active_by_level) & set(base_low_by_level)
                if base_active_by_level[level] and base_low_by_level[level]
            )
            boundary_has_ci = bool(boundary.get("q_half_ci") is not None)
            q_half_ci = boundary.get("q_half_ci")
            boundary_within_actual_range = bool(
                reachability["within_reachable_range"]
            )
            boundary_reached = bool(reachability["reached"])
            family_results[family] = {
                "boundary": boundary,
                "actual_0_to_1_crossing": actual_crossing,
                "reachability": reachability,
                "n_active_rows": len(active),
                "n_low_rows": len(low),
                "low_basin_persistence_fraction": (
                    sum(bool(row.get("low_basin_persisted")) for row in low)
                    / len(low) if low else None
                ),
                "coexistence_levels": coexistence_levels,
                "basin_coexistence_observed": bool(coexistence_levels),
                "boundary_has_bootstrap_ci": boundary_has_ci,
                "boundary_within_actual_range": boundary_within_actual_range,
                "boundary_reached_by_actual_direction": boundary_reached,
                "boundary_in_locked_extension": bool(
                    boundary_has_ci
                    and q_half is not None
                    and 1.0 < q_half <= 1.25
                    and 1.0 < float(q_half_ci[0])
                    and float(q_half_ci[1]) <= 1.25
                ),
            }
    dynamic = [
        row for row in all_rows if row.get("family") == "dynamic_ZM"
    ]
    dynamic_summary = BD.dynamic_offset_summary(dynamic)
    decision = BD.adjudicate_offset_surface(
        family_results,
        dynamic_summary,
        contract_ok=contract_audit["passed"],
    )
    verdict = decision["verdict"]

    output = {
        "verdict": verdict,
        "diagnostic_status": decision.get("diagnostic_status"),
        "reason_code": decision.get("reason_code"),
        "ambiguous_family_statuses": decision.get(
            "ambiguous_family_statuses", {}
        ),
        "n_complete_seeds": len(complete),
        "seeds": sorted({int(manifest["seed"]) for manifest in complete}),
        "family_results": family_results,
        "dynamic_ZM": dynamic_summary,
        "contract_audit": contract_audit,
        "inputs": [
            {
                "path": os.path.relpath(path, _ROOT),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
        "anchor_inputs": [
            {
                "path": os.path.relpath(path, _ROOT),
                "sha256": _sha256(path),
            }
            for path in anchor_paths
        ],
        "spec": {
            "path": os.path.relpath(SPEC_PATH, _ROOT),
            "sha256": _sha256(SPEC_PATH),
        },
        "canonical_config": {
            "path": os.path.relpath(CANONICAL_CONFIG_PATH, _ROOT),
            "sha256": _sha256(CANONICAL_CONFIG_PATH),
        },
        "analysis_git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
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
        f"[offset-summary] verdict={verdict} "
        f"diagnostic={output['diagnostic_status']} "
        f"seeds={output['seeds']} -> {path}"
    )


if __name__ == "__main__":
    main()
