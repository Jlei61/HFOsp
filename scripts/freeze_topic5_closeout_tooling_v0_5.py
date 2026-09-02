#!/usr/bin/env python3
"""Freeze every post-score closeout producer before target authorization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
SOURCES = (
    "scripts/guard_topic5_preunseal_resume_v0_5.py",
    "scripts/run_topic5_multiscale_closeout_watcher_v0_5.py",
    "scripts/adjudicate_topic5_multiscale_claims_v0_5.py",
    "scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r2.py",
    "scripts/export_topic5_figure6_source_data_v0_5.py",
    "scripts/finalize_topic5_multiscale_closeout_report_v0_5.py",
    "scripts/audit_topic5_multiscale_closeout_v0_5.py",
    "scripts/sync_topic5_multiscale_closeout_docs_v0_5.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if os.environ.get("TOPIC5_V0_5_TARGET_SEALED") != "1":
        raise RuntimeError("closeout tooling must be frozen inside target embargo")
    if (OUT / "TARGET_UNSEAL_AUTHORIZATION.json").exists():
        raise RuntimeError("closeout tooling freeze is forbidden after authorization")
    payload = {
        "contract": "topic5_v0_5_closeout_tooling_prefreeze",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_TARGET_FREE",
        "target_values_read": False,
        "sources": {
            relative: sha256_file(ROOT / relative) for relative in SOURCES
        },
        "source_count": len(SOURCES),
        "producer": str(Path(__file__).resolve()),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
        "purpose": (
            "PREVENT_POSTUNSEAL_DRIFT_IN_ADJUDICATION_FIGURE_SOURCE_EXPORT_"
            "REPORT_AUDIT_AND_DOC_SYNC"
        ),
    }
    destination = OUT / "CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
