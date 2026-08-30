#!/usr/bin/env python3
"""Run one rev16 crossed same-checkpoint Node hotspot intervention."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic4_rev15_node_intervention_worker as base  # noqa: E402
from scripts import topic4_rev16_node_substrate_adapter as adapter  # noqa: E402


EXPECTED_CONFIG_SCHEMA = "topic4_rev16_node_crossed_intervention_v1"
OUTPUT_SCHEMA = "topic4_rev16_node_crossed_intervention_worker_v1"
WORKER_STATUS = "REV16_NODE_CROSSED_INTERVENTION_WORKER_COMPLETE"


def run_worker(**kwargs: Any) -> dict[str, Any]:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
        base.build_projected_node_substrate,
        base.verify_projection_against_worker,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    base.WORKER_STATUS = WORKER_STATUS
    base.build_projected_node_substrate = adapter.build_projected_node_substrate
    base.verify_projection_against_worker = adapter.verify_projection_against_worker
    try:
        return base.run_worker(**kwargs)
    finally:
        (
            base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
            base.build_projected_node_substrate,
            base.verify_projection_against_worker,
        ) = previous


def main(argv: list[str] | None = None) -> None:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
        base.build_projected_node_substrate,
        base.verify_projection_against_worker,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    base.WORKER_STATUS = WORKER_STATUS
    base.build_projected_node_substrate = adapter.build_projected_node_substrate
    base.verify_projection_against_worker = adapter.verify_projection_against_worker
    try:
        base.main(argv)
    finally:
        (
            base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
            base.build_projected_node_substrate,
            base.verify_projection_against_worker,
        ) = previous


if __name__ == "__main__":
    main()
