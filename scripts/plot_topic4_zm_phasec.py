#!/usr/bin/env python
"""Render the eight locked Z/M Phase-C diagnostic figures without SNN runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_zm_phasec_plot import render_phasec_figures  # noqa: E402


DEFAULT_RESULT_ROOT = (
    ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot source-space Phase-C identity/maturation diagnostics; "
            "this command never runs the SNN."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--c0-summary", type=Path)
    parser.add_argument("--c1-summary", type=Path)
    parser.add_argument("--modal-summary", type=Path)
    parser.add_argument("--final-verdict", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else result_root / "figures"
    )
    manifest = render_phasec_figures(
        repo_root=ROOT,
        output_dir=output_dir,
        c0_summary_path=(
            args.c0_summary or result_root / "c0_identity_summary_dt.json"
        ),
        c1_summary_path=(
            args.c1_summary or result_root / "phasec1_summary_dt.json"
        ),
        modal_summary_path=(
            args.modal_summary
            or result_root / "phasec_seed_specific_modal.json"
        ),
        final_verdict_path=(
            args.final_verdict
            or result_root / "phasec_final_adjudication.json"
        ),
        dpi=args.dpi,
    )
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "n_figures": len(manifest["figures"]),
        "representative": manifest["representative"],
        "claim_boundary": manifest["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
