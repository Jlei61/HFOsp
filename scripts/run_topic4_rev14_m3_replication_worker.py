#!/usr/bin/env python3
"""Run one frozen rev14 M3 shortlist field on a replication network."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_replication as replication  # noqa: E402
from scripts import run_topic4_rev14_m3_canary_worker as worker  # noqa: E402


WORKER_STATUS = "REV14_M3_NETWORK_REPLICATION_WORKER_COMPLETE"
PREPARE_STATUS = "REV14_M3_NETWORK_REPLICATION_WORKER_PREPARED_NO_SNN"


def main(argv: list[str] | None = None) -> None:
    # The simulator composition is identical to the audited M3 canary worker.
    # Only the manifest freezer and output status identify the replication pool.
    previous = (
        worker.freezer, worker.EXPECTED_PATHWAYS, worker.WORKER_STATUS,
        worker.PREPARE_STATUS,
    )
    try:
        worker.freezer = replication
        worker.EXPECTED_PATHWAYS = replication.EXPECTED_PATHWAYS
        worker.WORKER_STATUS = WORKER_STATUS
        worker.PREPARE_STATUS = PREPARE_STATUS
        worker.main(argv)
    finally:
        (
            worker.freezer, worker.EXPECTED_PATHWAYS, worker.WORKER_STATUS,
            worker.PREPARE_STATUS,
        ) = previous


if __name__ == "__main__":
    main()
