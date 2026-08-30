#!/usr/bin/env python3
"""Aggregate rev16 crossed hotspot intervention and freeze Node if selective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev15_node_intervention as base  # noqa: E402
from scripts import run_topic4_rev16_node_intervention_worker as worker  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_intervention.json"
EXPECTED_CONFIG_SCHEMA = "topic4_rev16_node_crossed_intervention_v1"
OUTPUT_SCHEMA = "topic4_rev16_node_crossed_intervention_aggregate_v1"
FREEZE_SCHEMA = "topic4_rev16_frozen_node_field_v1"
FROZEN_STATUS = "REV16_NODE_FIELD_FROZEN"
NOT_SELECTIVE_STATUS = "REV16_NODE_INTERVENTION_NOT_SELECTIVE"


def aggregate(
    *, config_path: Path = DEFAULT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.FREEZE_SCHEMA,
        base.FROZEN_STATUS, base.NOT_SELECTIVE_STATUS, base.WORKER_STATUS,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    base.FREEZE_SCHEMA = FREEZE_SCHEMA
    base.FROZEN_STATUS = FROZEN_STATUS
    base.NOT_SELECTIVE_STATUS = NOT_SELECTIVE_STATUS
    base.WORKER_STATUS = worker.WORKER_STATUS
    try:
        return base.aggregate(
            config_path=config_path.resolve(), artifact_root=artifact_root.resolve(),
        )
    finally:
        (
            base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.FREEZE_SCHEMA,
            base.FROZEN_STATUS, base.NOT_SELECTIVE_STATUS, base.WORKER_STATUS,
        ) = previous


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = aggregate(config_path=args.config, artifact_root=args.artifact_root)
    print(json.dumps({
        "status": payload["status"], "candidate_id": payload["candidate_id"],
        "node_freeze_permitted": payload["node_freeze_permitted"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
