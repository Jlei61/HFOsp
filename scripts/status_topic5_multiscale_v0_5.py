#!/usr/bin/env python3
"""Read-only progress snapshot for the detached Topic 5.1 v0.5 pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def count(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.glob(pattern)) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    cache = out / "attenuation/unit_cache"
    matched = out / "attenuation/matched_local"
    markers = {
        name: (out / name).exists()
        for name in (
            "STAGE_E_TRAINING_COMPLETE.json",
            "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json",
            "MODE_FLOW_ATTENUATION_COMPLETE.json",
            "ATTENUATED_FIELDS_FROZEN.json",
            "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json",
            "STAGE_F_TARGET_FREE_COMPLETE.json",
            "TARGET_UNSEAL_AUTHORIZATION.json",
            "EARLY_ICTAL_SCORING_COMPLETE.json",
            "PIPELINE_COMPLETE.json",
        )
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attenuation": {
            "completed_unit_targets": count(cache, "*/*/*.json.gz"),
            "scheduled_unit_targets": 504,
            "matched_local_searches": count(matched, "*/*/match.json"),
            "scheduled_matched_local_searches": 126,
        },
        "markers": markers,
        "target_status": (
            "UNSEALED_AFTER_AUTHORIZATION"
            if markers["TARGET_UNSEAL_AUTHORIZATION.json"]
            else "PHYSICALLY_SEALED_TARGET_FREE"
        ),
        "failed_marker": (out / "POSTTRAINING_PIPELINE_FAILED.json").exists(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
