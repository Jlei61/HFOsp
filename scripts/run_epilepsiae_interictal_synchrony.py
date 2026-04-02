"""Run PR4 interictal synchrony on Epilepsiae ready_full_artifacts subjects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_synchrony import (
    run_epilepsiae_interictal_synchrony_from_manifest,
    save_interictal_synchrony_summary,
)


RESULTS_DIR = Path("results")
MANIFEST_CSV = RESULTS_DIR / "epilepsiae_sync_subject_manifest.csv"
OUTPUT_DIR = RESULTS_DIR / "interictal_synchrony" / "epilepsiae_ready_full_artifacts"
SUMMARY_CSV = OUTPUT_DIR / "epilepsiae_ready_full_artifacts_interictal_sync_summary.csv"


def main() -> None:
    rows = run_epilepsiae_interictal_synchrony_from_manifest(
        str(MANIFEST_CSV),
        str(OUTPUT_DIR),
        tier="ready_full_artifacts",
    )
    summary_path = save_interictal_synchrony_summary(rows, str(SUMMARY_CSV))
    print(
        json.dumps(
            {
                "n_block_results": len(rows),
                "summary_csv": summary_path,
                "output_dir": str(OUTPUT_DIR),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
