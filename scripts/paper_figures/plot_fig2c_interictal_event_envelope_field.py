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
    / "interictal_propagation_masked"
    / "event_envelope_fields"
    / "paper_source"
    / "figures"
)
LOCKED_TA_EVENT_POS = {"epilepsiae_1146": 6344}
LOCKED_TB_EVENT_POS = {"epilepsiae_1146": 937}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-step-ms", type=float, default=GIF_STEP_MS)
    ap.add_argument("--gif-fps", type=float, default=GIF_FPS)
    ap.add_argument(
        "--ta-event-pos", type=int,
        help="explicit direction-qualified TA exemplar; E1146 defaults to accepted event 6344",
    )
    ap.add_argument(
        "--tb-event-pos", type=int,
        help="explicit full-field-qualified TB exemplar; E1146 defaults to accepted event 937",
    )
    ap.add_argument(
        "--use-medoid-ta", action="store_true",
        help="ignore the accepted E1146 override and recompute the original TA medoid",
    )
    ap.add_argument(
        "--use-medoid-tb", action="store_true",
        help="ignore the accepted E1146 override and recompute the original TB medoid",
    )
    args = ap.parse_args()
    ta_event_pos = None
    if not args.use_medoid_ta:
        ta_event_pos = (
            args.ta_event_pos
            if args.ta_event_pos is not None
            else LOCKED_TA_EVENT_POS.get(args.subject)
        )
    tb_event_pos = None
    if not args.use_medoid_tb:
        tb_event_pos = (
            args.tb_event_pos
            if args.tb_event_pos is not None
            else LOCKED_TB_EVENT_POS.get(args.subject)
        )
    run(
        args.subject,
        paper_ready_dir=args.output_dir,
        make_gif=not args.no_gif,
        gif_step_ms=args.gif_step_ms,
        gif_fps=args.gif_fps,
        ta_event_pos=ta_event_pos,
        tb_event_pos=tb_event_pos,
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
