#!/usr/bin/env python3
"""Resource-safe controller for the three rev17 hotspot workers."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev17_node_intervention as aggregate  # noqa: E402
from scripts import monitor_topic4_rev16_node_intervention as base  # noqa: E402
from scripts import run_topic4_rev17_node_intervention_worker as worker  # noqa: E402


DEFAULT_UNIT_PREFIX = "codex-t4-r17-node-intervention"
CONTROLLER_SCHEMA = "topic4_rev17_node_intervention_controller_v1"
COMPLETE = "REV17_NODE_INTERVENTION_QUEUE_COMPLETE"
FAILED = "REV17_NODE_INTERVENTION_QUEUE_FAILED"
RUNNING = "REV17_NODE_INTERVENTION_QUEUE_RUNNING"
RESOURCE_WAIT = "REV17_NODE_INTERVENTION_RESOURCE_WAIT"
NETWORK_SEEDS = [2381, 2382, 2383]
FINAL_AUDIT_ADVANCE_STATUS = (
    "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
)


def _swap() -> tuple[Any, ...]:
    previous = (
        base.aggregate, base.worker, base.WORKER, base.DEFAULT_UNIT_PREFIX,
        base.CONTROLLER_SCHEMA, base.COMPLETE, base.FAILED, base.RUNNING,
        base.RESOURCE_WAIT, base.EXPECTED_NETWORK_SEEDS,
        base.FINAL_AUDIT_ADVANCE_STATUS,
        base.EXPECTED_PROJECTION_PARITY_KEYS, base.REVISION_LABEL,
    )
    base.aggregate = aggregate
    base.worker = worker
    base.WORKER = ROOT / "scripts/run_topic4_rev17_node_intervention_worker.py"
    base.DEFAULT_UNIT_PREFIX = DEFAULT_UNIT_PREFIX
    base.CONTROLLER_SCHEMA = CONTROLLER_SCHEMA
    base.COMPLETE = COMPLETE
    base.FAILED = FAILED
    base.RUNNING = RUNNING
    base.RESOURCE_WAIT = RESOURCE_WAIT
    base.EXPECTED_NETWORK_SEEDS = NETWORK_SEEDS
    base.FINAL_AUDIT_ADVANCE_STATUS = FINAL_AUDIT_ADVANCE_STATUS
    base.EXPECTED_PROJECTION_PARITY_KEYS = {"h", "delta_vtheta"}
    base.REVISION_LABEL = "rev17"
    return previous


def _restore(previous: tuple[Any, ...]) -> None:
    (
        base.aggregate, base.worker, base.WORKER, base.DEFAULT_UNIT_PREFIX,
        base.CONTROLLER_SCHEMA, base.COMPLETE, base.FAILED, base.RUNNING,
        base.RESOURCE_WAIT, base.EXPECTED_NETWORK_SEEDS,
        base.FINAL_AUDIT_ADVANCE_STATUS,
        base.EXPECTED_PROJECTION_PARITY_KEYS, base.REVISION_LABEL,
    ) = previous


def tick(**kwargs: Any) -> dict[str, Any]:
    previous = _swap()
    try:
        return base.tick(**kwargs)
    finally:
        _restore(previous)


def main(argv: list[str] | None = None) -> None:
    previous = _swap()
    try:
        base.main(argv)
    finally:
        _restore(previous)


if __name__ == "__main__":
    main()
