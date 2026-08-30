#!/usr/bin/env python3
"""Build one H2b v0.2 subject's exact queries and support denominator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.cohort_v02 import (
    prepare_subject_query_inputs,
)
from src.topic5_continuous_marked_state_h2b.contract import V0_2_RESULT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seizure-crosswalk", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--design-manifest", type=Path)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    args = parser.parse_args()
    result = prepare_subject_query_inputs(
        subject=args.subject,
        seizure_crosswalk_path=args.seizure_crosswalk,
        coverage_path=args.coverage,
        design_path=args.design,
        design_sha256=args.design_sha256,
        design_manifest_path=args.design_manifest,
        result_root=args.result_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
