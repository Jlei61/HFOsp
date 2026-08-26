#!/usr/bin/env python3
"""Build full-event recurrent timelines for the frozen pilot subjects."""
from __future__ import annotations

import argparse
import json

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.long_sequence import (
    LONG_SEQUENCE_REVISION,
    write_full_event_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=contract.PILOT_SUBJECTS,
                        choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = contract.RESULT_ROOT / "long_sequence/features"
    for subject in args.subjects:
        output = root / f"{subject}.npz"
        manifest = output.with_suffix(".manifest.json")
        if output.exists() and manifest.exists() and not args.overwrite:
            old = json.loads(manifest.read_text())
            if old.get("long_sequence_revision") == LONG_SEQUENCE_REVISION:
                print(json.dumps({"status": "SKIPPED", "subject": subject}))
                continue
        result = write_full_event_sequence(subject, output)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
