#!/usr/bin/env python3
"""Run one frozen rev16 M4-shell coordinate through the composed Node worker."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev16_m4_shell_coordinate_atlas as freezer
from scripts import run_topic4_rev14_m3_canary_worker as base


WORKER_STATUS = "REV16_M4_SHELL_COORDINATE_WORKER_COMPLETE"
PREPARE_STATUS = "REV16_M4_SHELL_COORDINATE_WORKER_PREPARED_NO_SNN"


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
