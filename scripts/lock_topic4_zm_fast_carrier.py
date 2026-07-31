#!/usr/bin/env python3
"""Build and publish the immutable Phase-D bootstrap input lock."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_fast_carrier_contract as C  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / C.DEFAULT_INPUT_OUTPUT
    )
    parser.add_argument("--confirm-input-lock", action="store_true")
    args = parser.parse_args()
    if not args.confirm_input_lock:
        raise SystemExit("--confirm-input-lock is required")
    payload = C.build_input_manifest(ROOT)
    C.validate_input_manifest(payload, ROOT, expected=payload)
    C.publish_once(args.output, payload)
    print(args.output)
    print(payload["manifest_sha256"])


if __name__ == "__main__":
    main()
