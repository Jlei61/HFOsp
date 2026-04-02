"""Generate Epilepsiae inventories and synchrony-ready subject manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.epilepsiae_dataset import (
    build_epilepsiae_sync_subject_manifest,
    save_epilepsiae_inventory,
    save_epilepsiae_sync_subject_manifest,
    survey_epilepsiae_dataset,
)


RESULTS_DIR = Path("results")


def main() -> None:
    inventory = survey_epilepsiae_dataset()
    outputs = save_epilepsiae_inventory(inventory, RESULTS_DIR)
    manifest = build_epilepsiae_sync_subject_manifest(inventory)
    manifest_path = save_epilepsiae_sync_subject_manifest(
        manifest,
        RESULTS_DIR / "epilepsiae_sync_subject_manifest.csv",
    )

    print(json.dumps(inventory.summary, indent=2, ensure_ascii=False))
    print(json.dumps({"inventory_outputs": outputs, "manifest_csv": manifest_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
