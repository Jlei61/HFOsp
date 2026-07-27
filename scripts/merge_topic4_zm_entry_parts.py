#!/usr/bin/env python
"""Fail-closed merger for isolated Z-entry cell manifests."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_topic4_zm_branch_decision as R  # noqa: E402


_ROW_IDENTITY_FIELDS = (
    "key",
    "seed",
    "lambda",
    "replicate",
    "bank_sha",
    "entered_carrier",
    "completed",
    "boundary_version",
    "survived",
    "stationarity_ok",
    "end_reason",
)


def _same_scientific_row(left, right):
    return all(left.get(key) == right.get(key) for key in _ROW_IDENTITY_FIELDS)


def merge_seed(seed: int):
    root = os.path.join(R.OUT, "boundaries", "entry", f"seed{int(seed)}")
    canonical_path = os.path.join(root, "entry_probes.json")
    canonical = json.load(open(canonical_path)) if os.path.exists(canonical_path) else None
    parts = [
        json.load(open(path))
        for path in sorted(glob.glob(os.path.join(root, "parts", "*.json")))
    ]
    complete_parts = [
        part for part in parts
        if part.get("complete") is True and isinstance(part.get("row"), dict)
    ]
    if canonical is None and not complete_parts:
        raise SystemExit(f"seed {seed}: no canonical manifest or complete parts")

    expected_version = R.BD.BOUNDARY_VERSION
    expected_hashes = (
        canonical.get("source_state_hashes") if canonical is not None
        else complete_parts[0].get("source_state_hashes")
    )
    if not expected_hashes:
        raise SystemExit(f"seed {seed}: source_state_hashes unavailable")
    rows = {}
    contributing = set()
    if canonical is not None:
        if canonical.get("boundary_version") != expected_version:
            raise SystemExit(f"seed {seed}: canonical boundary version drift")
        contributing.add(canonical.get("git_sha"))
        contributing.update(canonical.get("contributing_git_shas") or [])
        for row in canonical.get("rows", []):
            rows[row["key"]] = row
    for part in complete_parts:
        if part.get("boundary_version") != expected_version:
            raise SystemExit(f"seed {seed}: part boundary version drift")
        if part.get("source") != "canonical_row_reuse":
            if part.get("source_state_hashes") != expected_hashes:
                raise SystemExit(f"seed {seed}: part source-state hash drift")
            contributing.add(part.get("git_sha"))
        row = part["row"]
        if int(row.get("seed")) != int(seed):
            raise SystemExit(f"seed {seed}: foreign row {row.get('key')}")
        previous = rows.get(row["key"])
        if previous is not None and not _same_scientific_row(previous, row):
            raise SystemExit(
                f"seed {seed}: conflicting duplicate row {row['key']}"
            )
        rows[row["key"]] = previous or row

    base_rows = [
        {"lambda": row["lambda"], "entered_carrier": row["entered_carrier"]}
        for row in rows.values()
        if row.get("replicate") == R.ENTRY_BASE_REPLICATE
        and row.get("completed")
    ]
    bracket = None
    if len(base_rows) == len(R.ENTRY_LEVELS):
        curve = R.BD.jeffreys_probability_curve(
            base_rows, "lambda", "entered_carrier"
        )
        bracket = R.BD.half_boundary(curve, expected_direction="increasing")
    pending = []
    for lam in R.ENTRY_LEVELS:
        key = f"lambda={lam:g}|{R.ENTRY_BASE_REPLICATE}"
        if key not in rows:
            pending.append(
                {"lambda": float(lam), "replicate": R.ENTRY_BASE_REPLICATE}
            )
    if bracket and bracket["status"] == "bracketed":
        for lam in bracket["q_bracket"]:
            for replicate in R.ENTRY_EXPANSION_REPLICATES:
                key = f"lambda={float(lam):g}|{replicate}"
                if key not in rows:
                    pending.append(
                        {"lambda": float(lam), "replicate": replicate}
                    )

    if canonical is None:
        exemplar = complete_parts[0]
        canonical = {
            key: value for key, value in exemplar.items()
            if key not in {"complete", "row", "phase"}
        }
        canonical.update(
            phase="entry_boundary",
            coordinate_family="conditional_z_slice",
            scientific_assumption=(
                "interpolate the actual pre-entry-to-carrier z field while "
                "holding M and the full S_G family at onset-adjacent values"
            ),
            starting_fast_state="pre_entry__natural",
            levels=list(R.ENTRY_LEVELS),
            base_replicate=R.ENTRY_BASE_REPLICATE,
            expansion_replicates=list(R.ENTRY_EXPANSION_REPLICATES),
        )
    canonical["rows"] = sorted(rows.values(), key=lambda row: row["key"])
    canonical["cheap_bracket"] = bracket or {"status": "pending_base_cells"}
    canonical["pending_cells"] = pending
    canonical["complete"] = bool(not pending and bracket is not None)
    canonical["contributing_git_shas"] = sorted(
        sha for sha in contributing if sha
    )
    canonical["claim_boundary"] = (
        "conditional Z-entry slice on the observed trajectory context; no "
        "offset, recovery, observation match, or lifecycle implication"
    )
    R.write_json_atomic(canonical_path, canonical)
    print(
        f"[entry-merge] seed={seed} rows={len(rows)} "
        f"pending={len(pending)} complete={canonical['complete']} "
        f"bracket={canonical['cheap_bracket']['status']} -> {canonical_path}"
    )
    return canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=(1, 3, 4))
    args = parser.parse_args()
    merge_seed(args.seed)


if __name__ == "__main__":
    main()
