#!/usr/bin/env python3
"""Run the existing frozen H2b probe on a registered shared-state trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.contracts import atomic_json  # noqa: E402
from src.topic5_group_event_state.v035.seizure_transfer import run_seizure_transfer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--producer-unit", type=Path, required=True)
    ap.add_argument("--rate-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    source = json.loads((args.producer_unit / "card.json").read_text(encoding="utf-8"))
    subject, family, seed = source["subject"], source["family"], int(source["seed"])
    rate = args.rate_root / subject / f"seed{seed}" / "trajectory_and_scores.npz"
    card = run_seizure_transfer(
        subject, Path(source["state_trajectory"]), rate,
        out_dir=args.out_dir, overwrite=args.overwrite,
    )
    binding = {
        "format": "group_event_state_v0_3_5_shared_h2b_binding_v1",
        "subject": subject, "family": family, "producer_card": str(args.producer_unit / "card.json"),
        "producer_frozen": True, "seizure_gradient_to_producer": False,
        "h2b_card": str(args.out_dir / "card.json"),
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    atomic_json(args.out_dir / "shared_state_binding.json", binding)
    print(json.dumps({"subject": subject, "family": family,
                      "seizures_by_phase": card["distance_survival"]["seizures_by_phase"]}, indent=2))


if __name__ == "__main__":
    main()
