#!/usr/bin/env python3
"""Publish agent C's own fitted arms into C's additive registry file.

Agent A owns the shared registry; C never rewrites it.  C's entries go in a
separate file next to it, one atomic merge per entry, so two agents writing at the
same time cannot produce a last-writer-wins registry.

Every entry carries the full contract field set, including the content hash of the
checkpoint it points at -- so a later reader can tell whether the file on disk is
still the one the entry describes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import file_hash, write_json_atomic  # noqa: E402
from src.topic5_group_event_state_h3.registry import (  # noqa: E402
    AGENT_C_REGISTRY_NAME,
    REQUIRED_FIELDS,
    SHARED_ROOT,
)

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    args = parser.parse_args()

    entries: dict[str, dict] = {}
    skipped: list[str] = []
    for path in sorted((OUT_ROOT / "machine" / args.tag).glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        checkpoint = Path(payload["checkpoint_file"])
        if not checkpoint.exists():
            skipped.append(payload["run_id"])
            continue
        entry = {
            "producer_id": f"agent_c_{payload['arm']}",
            "subject": payload["subject"],
            "seed": payload["seed"],
            "model_family": "h3_linear_state_future_block",
            "uses_waveform": payload["uses_waveform"],
            "uses_multiband": payload["uses_multiband"],
            "uses_background": payload["uses_background"],
            "event_update": payload["event_update"],
            "feedback_model": payload["feedback_model"],
            "physical_dt": payload["physical_dt"],
            "training_objective": [f"future_{h}m" for h in payload["horizons_minutes"]],
            "anchor_grid_minutes": 5,
            "source_commit": payload["source_commit"],
            "config_hash": payload["config_hash"],
            "checkpoint_hash": file_hash(checkpoint),
            "checkpoint_path": str(checkpoint),
            "support_hash": payload["support_hash"],
            "owner": "agent_c",
            "tier": "h3_internal_arm_not_a_shared_state_producer",
        }
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"{payload['run_id']}: entry missing {missing}")
        entries[f"{entry['producer_id']}|{entry['subject']}|{entry['seed']}"] = entry

    path = Path(args.shared_root) / AGENT_C_REGISTRY_NAME
    existing = {"owner": "agent_c", "producers": {}}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    existing.setdefault("producers", {}).update(entries)
    existing["note"] = (
        "agent C's own H3 arms, published here rather than into the shared registry "
        "file that agent A owns; these are the arms of the H3 comparison, not shared "
        "state producers for other lines to consume"
    )
    write_json_atomic(existing, path)
    print(f"published {len(entries)} entries to {path}")
    if skipped:
        print(f"skipped {len(skipped)} runs with no checkpoint on disk: {skipped[:5]}")


if __name__ == "__main__":
    main()
