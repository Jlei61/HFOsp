#!/usr/bin/env python3
"""Run one frozen rev16 Node field on an unseen confirmation network."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev16_node_confirmation as freezer  # noqa: E402
from scripts import run_topic4_rev14_m3_canary_worker as base  # noqa: E402


WORKER_STATUS = "REV16_NODE_CONFIRMATION_WORKER_COMPLETE"
PREPARE_STATUS = "REV16_NODE_CONFIRMATION_WORKER_PREPARED_NO_SNN"


def configure_base() -> None:
    base.freezer = freezer
    base.EXPECTED_PATHWAYS = freezer.EXPECTED_PATHWAYS
    base.WORKER_STATUS = WORKER_STATUS
    base.PREPARE_STATUS = PREPARE_STATUS


def main(argv: list[str] | None = None) -> None:
    configure_base()
    base.main(argv)


if __name__ == "__main__":
    main()
