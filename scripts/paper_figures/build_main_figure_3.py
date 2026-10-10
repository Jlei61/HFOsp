#!/usr/bin/env python3
"""Compatibility entry for the author-accepted Figure 3 A–E release."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.paper_figures.build_fig3_current import main

if __name__ == "__main__":
    main()
