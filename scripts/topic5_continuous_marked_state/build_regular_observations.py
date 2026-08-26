#!/usr/bin/env python3
"""Build event-independent regular observation grids for selected pilots."""
from __future__ import annotations

import argparse
import json

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_observation import (
    REGULAR_OBSERVATION_REVISION,
    write_regular_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True,
                        choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-kind", choices=("spectral", "raw", "both"),
                        default="spectral")
    args = parser.parse_args()
    root = contract.RESULT_ROOT / (
        "regular_observation/features" if args.feature_kind == "spectral"
        else f"regular_observation/features_{args.feature_kind}"
    )
    for subject in args.subjects:
        output = root / f"{subject}.npz"
        manifest = output.with_suffix(".manifest.json")
        if output.exists() and manifest.exists() and not args.overwrite:
            old = json.loads(manifest.read_text())
            expected_revision = f"{REGULAR_OBSERVATION_REVISION}__{args.feature_kind}"
            if old.get("regular_observation_revision") in (
                expected_revision,
                REGULAR_OBSERVATION_REVISION if args.feature_kind == "spectral" else "",
            ):
                print(json.dumps({"status": "SKIPPED", "subject": subject}))
                continue
        result = write_regular_observations(
            subject, output, feature_kind=args.feature_kind
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
