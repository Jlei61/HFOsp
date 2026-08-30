#!/usr/bin/env python3
"""Run rev16 same-network natural-KMeans audit without reranking fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_topic4_rev15_node_postselection as base  # noqa: E402
from scripts import run_topic4_rev16_joint_m3_m4_candidate_worker as worker  # noqa: E402
from scripts.prepare_topic4_rev16_node_postselection_config import (  # noqa: E402
    OUTPUT_SCHEMA as EXPECTED_CONFIG_SCHEMA,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_postselection.json"
OUTPUT_SCHEMA = "topic4_rev16_node_postselection_audit_v1"


def audit(
    *, config_path: Path = DEFAULT_CONFIG,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    previous = (
        base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
    )
    base.EXPECTED_CONFIG_SCHEMA = EXPECTED_CONFIG_SCHEMA
    base.OUTPUT_SCHEMA = OUTPUT_SCHEMA
    base.WORKER_STATUS = worker.WORKER_STATUS
    try:
        payload = base.audit(
            config_path=config_path.resolve(), artifact_root=artifact_root.resolve(),
        )
    finally:
        (
            base.EXPECTED_CONFIG_SCHEMA, base.OUTPUT_SCHEMA, base.WORKER_STATUS,
        ) = previous
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(
        config_path=args.config.resolve(), artifact_root=args.artifact_root.resolve(),
    )
    print(json.dumps({
        "status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "accepted": payload["acceptance"]["accepted"],
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
