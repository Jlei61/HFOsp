#!/usr/bin/env python
"""Resolve the real observation references or write the mandated blocked artifact.

This command never substitutes generic background or model traces for real
returning group events.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.topic4_zm_empirical_carrier as EC  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_returning_index(path):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    rows = rows.get("windows", []) if isinstance(rows, dict) else rows
    return [r for r in rows if isinstance(r, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seizure-inventory",
        default="/home/honglab/leijiaxin/HFOsp/results/epilepsiae_seizure_inventory.csv")
    ap.add_argument(
        "--geometry",
        default="/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                "field_swap_subject_snn/figdata_epilepsiae_1146_twoend_equal_tsrc_s3.npz")
    ap.add_argument("--returning-event-index",
                    help="JSON index of real returning group-event SEEG windows")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-early", type=int, default=6)
    ap.add_argument("--out", default=str(
        ROOT / "results/topic4_sef_hfo/zm_branch_decision/phase0c"))
    a = ap.parse_args()

    inv = Path(a.seizure_inventory)
    rows = []
    if inv.exists():
        rows = [r for r in csv.DictReader(inv.open()) if r.get("subject") == EC.SUBJECT]
    early = EC.resolve_early_ictal_windows(rows, max_n=a.max_early)
    returning = load_returning_index(a.returning_event_index)
    status = EC.reference_contract_status(early, returning, a.geometry, min_n=a.min_n)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(
        version=EC.LOCK_VERSION,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip(),
        seizure_inventory=str(inv),
        seizure_inventory_sha256=sha256(inv) if inv.exists() else None,
        geometry=str(a.geometry),
        geometry_sha256=sha256(a.geometry) if os.path.exists(a.geometry) else None,
        returning_event_index=a.returning_event_index,
        returning_event_index_sha256=(
            sha256(a.returning_event_index)
            if a.returning_event_index and os.path.exists(a.returning_event_index) else None),
        resolved_early_ictal_windows=early,
        contract=status,
    )
    if not status["sufficient"]:
        path = out / "blocked_reference_artifacts.json"
        provenance.update(
            verdict="blocked_reference_artifacts",
            reason=(
                "The early-ictal raw windows and E1146 geometry are resolvable, "
                "but no canonical real returning-group-event SEEG-window index "
                "was supplied. Generic background and model interictal traces "
                "are forbidden substitutes."
            ),
        )
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(provenance, indent=2))
        os.replace(tmp, path)
        print(f"[blocked] {status['missing']} -> {path}")
        return 0

    # The actual metric lock is intentionally not built here until each indexed
    # returning window and early-ictal crop has been loaded and validated.  A
    # mere path-count pass cannot masquerade as metric evidence.
    path = out / "reference_artifacts_resolved_pending_metrics.json"
    provenance.update(verdict="resolved_pending_metric_extraction")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, indent=2))
    os.replace(tmp, path)
    print(f"[resolved] metric extraction still required -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
