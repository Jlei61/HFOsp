#!/usr/bin/env python3
"""Run one rev17 crossed same-checkpoint Node hotspot intervention."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic4_rev15_node_intervention_worker as base  # noqa: E402
from scripts import topic4_rev17_node_substrate_adapter as adapter  # noqa: E402


EXPECTED_CONFIG_SCHEMA = "topic4_rev17_node_crossed_intervention_v1"
OUTPUT_SCHEMA = "topic4_rev17_node_crossed_intervention_worker_v1"
WORKER_STATUS = "REV17_NODE_CROSSED_INTERVENTION_WORKER_COMPLETE"
FINAL_AUDIT_ADVANCE_STATUS = (
    "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
)


def _swap() -> tuple[Any, ...]:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
        base.FINAL_AUDIT_ADVANCE_STATUS,
        base.build_projected_node_substrate,
        base.verify_projection_against_worker,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    base.WORKER_STATUS = WORKER_STATUS
    base.FINAL_AUDIT_ADVANCE_STATUS = FINAL_AUDIT_ADVANCE_STATUS
    base.build_projected_node_substrate = adapter.build_projected_node_substrate
    base.verify_projection_against_worker = adapter.verify_projection_against_worker
    return previous


def _restore(previous: tuple[Any, ...]) -> None:
    (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
        base.FINAL_AUDIT_ADVANCE_STATUS,
        base.build_projected_node_substrate,
        base.verify_projection_against_worker,
    ) = previous


def run_worker(**kwargs: Any) -> dict[str, Any]:
    previous = _swap()
    try:
        return base.run_worker(**kwargs)
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
