#!/usr/bin/env python3
"""Build the v0.3.5 event-independent fixed-clock background cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.background_rate import (  # noqa: E402
    BACKGROUND_CACHE_ROOT,
    build_fixed_grid_background_cache,
)
from src.topic5_group_event_state.v035.contracts import RateTrainConfig, V035_SUBJECTS  # noqa: E402
from src.topic5_group_event_state.v035.dynamic_rate import load_rate_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=V035_SUBJECTS, required=True)
    parser.add_argument("--cache-root", type=Path, default=BACKGROUND_CACHE_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    data = load_rate_data(args.subject, RateTrainConfig())
    manifest = build_fixed_grid_background_cache(
        data, cache_root=args.cache_root, overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
