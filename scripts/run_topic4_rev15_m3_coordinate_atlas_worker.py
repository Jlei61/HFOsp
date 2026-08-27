#!/usr/bin/env python3
"""Run one frozen rev15 M3 coordinate through the rev14 composed worker."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev15_m3_coordinate_atlas as freezer
from scripts import run_topic4_rev14_m3_canary_worker as base


WORKER_STATUS = "REV15_M3_COORDINATE_ATLAS_WORKER_COMPLETE"
PREPARE_STATUS = "REV15_M3_COORDINATE_ATLAS_WORKER_PREPARED_NO_SNN"


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
