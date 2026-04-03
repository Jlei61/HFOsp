"""Aggregate Epilepsiae event-level synchrony into interval/window tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony_aggregation import run_epilepsiae_sync_aggregation


RESULTS_DIR = Path("results")
SYNC_DIR = RESULTS_DIR / "interictal_synchrony" / "epilepsiae_ready_full_artifacts"
OUTPUT_DIR = SYNC_DIR / "aggregated"


def main() -> None:
    summary = run_epilepsiae_sync_aggregation(
        seizure_inventory_csv=RESULTS_DIR / "epilepsiae_seizure_inventory.csv",
        sync_event_csv=SYNC_DIR
        / "epilepsiae_ready_full_artifacts_interictal_sync_events.csv",
        output_dir=OUTPUT_DIR,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
