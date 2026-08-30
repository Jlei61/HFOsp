#!/usr/bin/env python3
"""Render canonical Fig.4 panels for the accepted rev16 Node field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic4_rev16_joint_m3_m4_candidate_worker as worker  # noqa: E402
from scripts.paper_figures import plot_topic4_rev15_node_postselection_fig4 as base  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_postselection.json"
DEFAULT_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/postselection/analysis/node_postselection_audit.json"
)
REVISION_ID = "rev16"
NODE_DESCRIPTION = "rev16 joint M3+M4 Node"
SCIENTIFIC_ROLE = "development_only_rev16_node_postselection"
RENDER_STATUS = "REV16_NODE_POSTSELECTION_FIG4_RENDERED"


def render(
    *, config_path: Path = DEFAULT_CONFIG, audit_path: Path = DEFAULT_AUDIT,
    artifact_root: Path = ARTIFACT_ROOT,
    allow_rejected_diagnostic: bool = False,
) -> dict[str, Any]:
    previous = (
        base.post.WORKER_STATUS, base.REVISION_ID, base.NODE_DESCRIPTION,
        base.SCIENTIFIC_ROLE, base.RENDER_STATUS,
    )
    base.post.WORKER_STATUS = worker.WORKER_STATUS
    base.REVISION_ID = REVISION_ID
    base.NODE_DESCRIPTION = NODE_DESCRIPTION
    base.SCIENTIFIC_ROLE = SCIENTIFIC_ROLE
    base.RENDER_STATUS = RENDER_STATUS
    try:
        return base.render(
            config_path=config_path.resolve(), audit_path=audit_path.resolve(),
            artifact_root=artifact_root.resolve(),
            allow_rejected_diagnostic=allow_rejected_diagnostic,
        )
    finally:
        (
            base.post.WORKER_STATUS, base.REVISION_ID, base.NODE_DESCRIPTION,
            base.SCIENTIFIC_ROLE, base.RENDER_STATUS,
        ) = previous


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--allow-rejected-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    payload = render(
        config_path=args.config, audit_path=args.audit,
        artifact_root=args.artifact_root,
        allow_rejected_diagnostic=args.allow_rejected_diagnostic,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
