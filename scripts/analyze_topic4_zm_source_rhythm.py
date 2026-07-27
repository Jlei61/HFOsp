#!/usr/bin/env python
"""Aggregate replicated fine-source audits into an operator-routing decision."""

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

from src.topic4_zm_source_rhythm import adjudicate_source_rhythm  # noqa: E402


OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    verdict_path = os.path.join(OUT, "branch_verdict.json")
    verdict = json.load(open(verdict_path)) if os.path.exists(verdict_path) else {}
    authorized = bool(
        verdict.get("verdict") == "carrier_at_visited_states"
        and (verdict.get("confirmation") or {}).get("status") == "passed"
    )
    paths = sorted(glob.glob(os.path.join(OUT, "source_rhythm", "dt", "seed*", "source_rhythm.json")))
    rows = []
    inputs = []
    for path in paths:
        payload = json.load(open(path))
        metrics = payload.get("source_rhythm") or {}
        rows.append(
            {
                "seed": int(payload["seed"]),
                "source_temporal_class": metrics.get("source_temporal_class"),
            }
        )
        inputs.append(
            {
                "path": os.path.relpath(path, _ROOT),
                "sha256": _sha256(path),
                "seed": int(payload["seed"]),
            }
        )
    summary = (
        adjudicate_source_rhythm(rows)
        if authorized
        else {
            "status": "not_authorized",
            "carrier_type": None,
            "reason": "two-seed native carrier confirmation has not passed",
        }
    )
    output = {
        **summary,
        "authorized": authorized,
        "inputs": inputs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "branch_verdict_path": os.path.relpath(verdict_path, _ROOT),
        "claim_boundary": (
            "replicated fine-source operator routing only; not observation "
            "matching, ictal identity, entry, offset, or lifecycle evidence"
        ),
    }
    path = os.path.join(OUT, "source_rhythm", "source_rhythm_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
    os.replace(tmp, path)
    print(
        f"[source-rhythm] status={output['status']} "
        f"carrier_type={output.get('carrier_type')} inputs={len(inputs)} -> {path}"
    )


if __name__ == "__main__":
    main()
