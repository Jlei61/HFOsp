#!/usr/bin/env python3
"""Measure the effect-size sensitivity of the v0.3.2 positive-control assay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_group_event_state.v032_model.config import load_config  # noqa: E402
from src.topic5_group_event_state.v032_model.data import load_subject_bundle  # noqa: E402
from src.topic5_group_event_state.v032_model.paths import MODEL_ROOT, atomic_write_json  # noqa: E402
from src.topic5_group_event_state.v032_model.synthetic import (  # noqa: E402
    judge_synthetic,
    run_synthetic_assay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--architecture", default="leaky_bank", choices=("leaky_bank", "repaired_rnn"))
    parser.add_argument("--betas", nargs="+", type=float, default=(0.35, 0.70, 1.40))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = load_config(None, architecture=args.architecture)
    bundle = load_subject_bundle(args.subject, allow_provisional_h=False)
    root = MODEL_ROOT / "synthetic_sensitivity" / args.architecture / args.subject
    rows = []
    for beta in args.betas:
        beta_name = f"beta_{beta:.2f}".replace(".", "p")
        assays = []
        for replicate in range(args.replicates):
            out_dir = root / beta_name / f"replicate_{replicate}"
            assays.append(
                run_synthetic_assay(
                    bundle,
                    cfg,
                    kind="positive",
                    replicate=replicate,
                    seed=20261900 + replicate,
                    device=device,
                    out_dir=out_dir,
                    beta=beta,
                    overwrite=args.force,
                )
            )
        judgement = judge_synthetic(assays, "positive")
        judgement["beta"] = beta
        atomic_write_json(root / beta_name / "judgement.json", judgement)
        rows.append(judgement)
    summary = {
        "format": "group_event_state_v0_3_2_synthetic_effect_size_ladder",
        "subject": args.subject,
        "architecture": args.architecture,
        "rows": rows,
        "smallest_beta_passing": next((row["beta"] for row in rows if row["pass"]), None),
        "status": "complete",
    }
    atomic_write_json(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
