#!/usr/bin/env python3
"""Create an immutable adjudicated ledger after an adaptive batch stop.

The raw coordinator ledger is intentionally preserved.  This pass separates
scientific adaptive cancellation from worker failure and resolves completed
children from their durable summary artifacts when the coordinator was paused
before reaping them.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/lifecycle_sprint"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _artifact_from_log(log_path):
    path = ROOT / log_path
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        candidate = Path(line.strip())
        if candidate.name == "summary.json" and candidate.is_file():
            return candidate
    return None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary_receipt(path):
    if path is None or not path.is_file():
        return {}
    summary = json.loads(path.read_text())
    return {
        "wall_s": summary.get("wall_s"),
        "peak_rss_gb": summary.get("peak_rss_gb"),
        "runtime_git_sha": summary.get("runtime_git_sha"),
        "runaway_early_stop_ms": summary.get("runaway_early_stop_ms"),
        "finite_control": summary.get("finite_control"),
        "observed_ms": summary.get("observed_ms"),
    }


def adjudicate(raw, decisions):
    cancelled = {
        config_id
        for decision in decisions.get("decisions", [])
        for config_id in decision.get("cancelled_config_ids", [])
    }
    rows = {}
    for config_id, source in raw["rows"].items():
        row = dict(source)
        status = row.get("status")
        artifact = _artifact_from_log(row.get("log_path", ""))
        if config_id in cancelled:
            row.update(
                adjudicated_status="adaptively_cancelled",
                scientific_evidence_eligible=False,
                adjudication_reason="cancelled after completed-wave phenotype review",
            )
        elif artifact is not None:
            row.update(
                adjudicated_status="success",
                scientific_evidence_eligible=True,
                artifact_path=str(artifact.relative_to(ROOT)),
                artifact_sha256=_sha256(artifact) if artifact.is_file() else None,
                **_summary_receipt(artifact),
            )
        elif status == "pending":
            row.update(
                adjudicated_status="not_run_after_adaptive_stop",
                scientific_evidence_eligible=False,
            )
        elif status == "worker_failed":
            row.update(
                adjudicated_status="worker_failed",
                scientific_evidence_eligible=False,
            )
        elif status == "running":
            row.update(
                adjudicated_status="unresolved_running_without_artifact",
                scientific_evidence_eligible=False,
            )
        else:
            row.update(
                adjudicated_status=status,
                scientific_evidence_eligible=status == "success",
            )
        rows[config_id] = row
    counts = {}
    for row in rows.values():
        key = row["adjudicated_status"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "topic4_zm_lifecycle_sprint_adjudicated_ledger_v1_2026-08-02",
        "created_at_utc": _now(),
        "raw_ledger_preserved": True,
        "raw_ledger_status_was_coordinator_state_not_final_scientific_status": True,
        "status_counts": counts,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-ledger", type=Path, default=OUT / "batch1_run_ledger.json")
    parser.add_argument(
        "--decisions", type=Path, default=OUT / "batch1_adaptation_decisions.json"
    )
    parser.add_argument(
        "--output", type=Path, default=OUT / "batch1_adjudicated_ledger.json"
    )
    args = parser.parse_args()
    raw_path = args.raw_ledger.resolve()
    decisions_path = args.decisions.resolve()
    payload = adjudicate(
        json.loads(raw_path.read_text()),
        json.loads(decisions_path.read_text()) if decisions_path.is_file() else {},
    )
    payload["raw_ledger_path"] = str(raw_path.relative_to(ROOT))
    payload["decisions_path"] = (
        str(decisions_path.relative_to(ROOT)) if decisions_path.is_file() else None
    )
    path = args.output.resolve()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
