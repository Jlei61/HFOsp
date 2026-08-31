#!/usr/bin/env python3
"""Build the outcome-value-blind H2b v0.3 attrition census."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_csv,
    atomic_json,
)
from src.topic5_continuous_marked_state_h2b.v03_attrition import (  # noqa: E402
    build_attrition_payload,
)
from src.topic5_continuous_marked_state_h2b.v03_contract import (  # noqa: E402
    assert_frozen_contract_matches,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    args = parser.parse_args()
    contract_path = args.result_root / "analysis_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"freeze analysis_contract.json before attrition audit: {contract_path}"
        )
    import json
    assert_frozen_contract_matches(json.loads(contract_path.read_text(encoding="utf-8")))
    payload = build_attrition_payload(args.v0_2_root)
    output = args.result_root / "manifests/attrition_audit.json"
    atomic_json(output, payload)
    atomic_csv(
        args.result_root / "manifests/attrition_by_cell.csv",
        payload["cell_rows"],
    )
    atomic_csv(
        args.result_root / "manifests/contrast_estimability.csv",
        payload["contrast_rows"],
    )
    print(f"COMPLETE {payload['funnel']} {output}")


if __name__ == "__main__":
    main()
