"""Thin CLI: run the M3A-v2.2 approach-criticality transition sim and write the fail-closed
M3A->M3B handoff artifacts to --out-dir, then print the overlay_verdict.

The real export legitimately REFUSES the phase-map overlay (the slow->rate mapping for this
sim is uncalibrated) -- that is the honest verdict, not an error. This wrapper is intentionally
minimal; the subprocess smoke test lands later (T3a-6).

    python scripts/run_topic4_crit_export.py --out-dir results/topic4_criticality/handoff_v2_2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sef_hfo_transition_sim import default_transition_config  # noqa: E402
from src.topic4_criticality import export_v2_2_handoff  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="directory for the handoff artifacts")
    ap.add_argument("--layout", default="subject1146", choices=["stage5", "subject1146"])
    ap.add_argument("--top", default="qI", choices=["hG", "qI"])
    args = ap.parse_args()
    cfg = default_transition_config(layout=args.layout, top=args.top)
    verdict = export_v2_2_handoff(args.out_dir, cfg)
    print(f"overlay_verdict={verdict}  out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
