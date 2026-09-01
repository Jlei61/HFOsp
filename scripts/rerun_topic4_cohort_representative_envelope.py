#!/usr/bin/env python3
"""Re-run the representative subject's confirmation network keeping envelopes.

The formal workers drop the per-contact envelope, which is seventy times the
size of everything else they write, because the volume is nearly full. The
waveform panel needs it for exactly one network, so that one network is
repeated with the envelope kept and written beside the original: the worker is
deterministic given candidate, seed and config, and overwriting the original
would invalidate the aggregation's npz hash check.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
WORKER = ROOT / "scripts/run_topic4_data_driven_snn_cohort_formal_worker.py"
DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    result = json.loads((output_root / "cohort_result.json").read_text())
    subject_id = result["representative_subject"]["subject_id"]
    row = next(item for item in result["canonical_subjects"]
               if item["subject_id"] == subject_id)

    sys.path.insert(0, str(ROOT / "scripts/paper_figures"))
    from scripts.paper_figures.plot_topic4_cohort_representative_kmeans import (
        _representative_seed,
    )

    seed = _representative_seed(row)
    stem = output_root / "workers" / f"{row['candidate_id']}_seed_{seed}"
    target = stem.with_name(stem.name + "_envelope.npz")
    if target.exists():
        print(json.dumps({"status": "ALREADY_PRESENT", "npz": str(target)}))
        return
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    subprocess.run(
        [str(PYTHON), str(WORKER), "--config", str(args.config),
         "--candidate-id", row["candidate_id"], "--seed", str(seed),
         "--expected-commit", commit, "--store-contact-envelope",
         "--out-json", str(stem.with_name(stem.name + "_envelope.json")),
         "--out-npz", str(target)],
        cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV},
    )
    print(json.dumps({
        "status": "ENVELOPE_RERUN_COMPLETE", "subject_id": subject_id,
        "candidate_id": row["candidate_id"], "seed": seed,
        "npz": str(target.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
