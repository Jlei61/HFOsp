#!/usr/bin/env python3
"""Paper-ready entry point for the interictal-axis representativeness figure."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_axis_representativeness import main  # noqa: E402


if __name__ == "__main__":
    main()
