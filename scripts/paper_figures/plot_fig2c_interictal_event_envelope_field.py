#!/usr/bin/env python3
"""Build the paper-ready Fig. 2c candidate and its synchronized TA/TB GIF."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_event_envelope_field import (  # noqa: E402
    GIF_FPS,
    GIF_STEP_MS,
    run,
)


DEFAULT_OUT = (
    ROOT
    / "results"
    / "paper-ready-figure"
    / "fig2c_interictal_event_envelope_field"
    / "figures"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-step-ms", type=float, default=GIF_STEP_MS)
    ap.add_argument("--gif-fps", type=float, default=GIF_FPS)
    args = ap.parse_args()
    run(
        args.subject,
        paper_ready_dir=args.output_dir,
        make_gif=not args.no_gif,
        gif_step_ms=args.gif_step_ms,
        gif_fps=args.gif_fps,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
