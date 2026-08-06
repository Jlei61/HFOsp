#!/usr/bin/env python3
"""Plot Figure 5 supplementary spatial-dynamics candidates from accepted sidecars.

This entry point is plotting-only: it does not replay the SNN, solve new states, or change the
registered estimand. The scientific producer remains
``run_topic4_state_conditioned_susceptibility.py``.
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_topic4_state_conditioned_susceptibility as runner  # noqa: E402


if __name__ == "__main__":
    runner.main(["plot-paper-ready", *sys.argv[1:]])
