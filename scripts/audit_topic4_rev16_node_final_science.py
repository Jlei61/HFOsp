#!/usr/bin/env python3
"""Run rev16 held-out distribution and source-topology audit once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev15_node_final_science as base  # noqa: E402
from scripts import run_topic4_rev16_node_confirmation_worker as worker  # noqa: E402
from scripts.prepare_topic4_rev16_node_final_science_config import (  # noqa: E402
    OUTPUT_SCHEMA as EXPECTED_CONFIG_SCHEMA,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_final_science.json"
OUTPUT_SCHEMA = "topic4_rev16_node_final_science_audit_v2"


def audit(
    config_path: Path = DEFAULT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.EXPECTED_WORKER_STATUS,
        base.OUTPUT_SCHEMA,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.EXPECTED_WORKER_STATUS = worker.WORKER_STATUS
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    try:
        payload = base.audit(config_path.resolve(), artifact_root.resolve())
    finally:
        (
            base.EXPECTED_CONFIG_SCHEMA, base.EXPECTED_WORKER_STATUS,
            base.OUTPUT_SCHEMA,
        ) = previous
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"], "candidate_id": payload["candidate_id"],
        "advances_to_intervention": payload["decision"][
            "accepted_for_same_checkpoint_intervention"
        ],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
