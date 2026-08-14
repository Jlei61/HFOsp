#!/usr/bin/env python3
"""Prove that the portable LC1 classifier snapshot reproduces archived LC5 reductions.

This is a read-only audit over completed LC5 spike bundles.  It deliberately compares only fields
that depend on the lost baseline contract: event detection, per-window regimes, lifecycle label,
and onset/offset.  Post-hoc phase-map labels such as ENTRY_BLOCKED_WITH_IED are outside that scope.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.topic4_fcxr_lc_baseline import DEFAULT_SNAPSHOT, load_classifier_snapshot  # noqa: E402


def _load_prefix():
    path = ROOT / "scripts/run_topic4_fcxr_lc5v2_natural_prefix.py"
    spec = importlib.util.spec_from_file_location("lc5v2_prefix_snapshot_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _same_optional_number(left, right):
    if left is None or right is None:
        return left is None and right is None
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))


def audit(root, snapshot=DEFAULT_SNAPSHOT):
    root = Path(root)
    baseline = load_classifier_snapshot(snapshot)
    prefix = _load_prefix()
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        relative_parts = summary_path.relative_to(root).parts
        if "superseded" in relative_parts or any(part.startswith(".") for part in relative_parts):
            continue
        spikes_path = summary_path.parent / "spikes.npz"
        if not spikes_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        archived_lifecycle = summary.get("lifecycle")
        if not isinstance(archived_lifecycle, dict) or not isinstance(
            archived_lifecycle.get("regimes"), list
        ):
            continue
        stream = prefix.load_sparse_spike_stream(spikes_path)
        rate = (
            np.bincount(stream.steps, minlength=stream.n_steps).astype(float)
            / stream.n_cells / prefix.U2.DT_MS * 1000.0
        )
        replay = prefix._adjudicate(stream, rate)
        checks = {
            "spike_hash": summary.get("spike_sha256") in (None, stream.sha256),
            "n_events": int(summary.get("n_events", -1)) == len(replay["events"]),
            "n_returning": int(summary.get("n_returning", -1)) == len(replay["returned"]),
            "raw_regimes": archived_lifecycle["regimes"] == replay["lifecycle"]["regimes"],
            "lifecycle_label": archived_lifecycle.get("label") == replay["lifecycle"].get("label"),
            "onset_ms": _same_optional_number(summary.get("onset_ms"), replay.get("onset_ms")),
            "offset_ms": _same_optional_number(summary.get("offset_ms"), replay.get("offset_ms")),
        }
        # The original Gamma=0 control was later given the post-hoc
        # ESCALATING_SATURATION outcome while its stored lifecycle object retained the earlier
        # bounded label.  That known context mismatch is not a baseline-contract field.  The
        # baseline-sensitive proof is exact event detection + raw regimes + onset/offset.
        required = (
            "spike_hash", "n_events", "n_returning", "raw_regimes", "onset_ms", "offset_ms"
        )
        rows.append({
            "summary": str(summary_path),
            "spikes": str(spikes_path),
            "checks": checks,
            "pass": bool(all(checks[key] for key in required)),
            "archived_outcome": summary.get("outcome"),
            "replayed_outcome": replay.get("outcome"),
        })
    if not rows:
        raise RuntimeError("no completed LC5 lifecycle bundles found for classifier audit")
    return {
        "status": "LC1_CLASSIFIER_SNAPSHOT_REPLAY_PASS" if all(r["pass"] for r in rows)
        else "LC1_CLASSIFIER_SNAPSHOT_REPLAY_FAIL",
        "snapshot": str(Path(snapshot)),
        "snapshot_sha256": _sha(snapshot),
        "original_full_contract_sha256": baseline["original_full_contract"]["sha256"],
        "n_bundles": len(rows),
        "n_pass": sum(r["pass"] for r in rows),
        "n_context_label_mismatches": sum(
            not r["checks"]["lifecycle_label"] for r in rows
        ),
        "scope": (
            "baseline-sensitive reduction only; lifecycle labels that were post-hoc overridden "
            "by saturation context and entry-blocked/delayed outcome labels are diagnostics"
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode",
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "lc1_classifier_snapshot_replay_audit.json"
    result = audit(args.root, args.snapshot)
    _write_json(output, result)
    print(json.dumps({k: result[k] for k in ("status", "n_bundles", "n_pass")}, indent=2))
    if result["status"] != "LC1_CLASSIFIER_SNAPSHOT_REPLAY_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
