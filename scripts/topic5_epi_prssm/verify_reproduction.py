#!/usr/bin/env python3
"""Re-run one completed arm under the current code and compare it to the archived run.

Model source files were edited after some Goal 1 arms had already started, so the
recorded package hash of those runs is older than the final package.  The edits
were additive and are expected to leave every arm except `static` bit-identical.
This script tests that expectation rather than asserting it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import OUTPUT_ROOT, atomic_write_json, load_tensors, package_hash, resolve_cohort, torch  # noqa: E402

import numpy as np  # noqa: E402

from src.topic5_epi_prssm.contracts import LeakageGuard  # noqa: E402
from src.topic5_epi_prssm.evaluate import evaluate  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM  # noqa: E402
from run_generator_ladder import ARMS  # noqa: E402
from src.topic5_epi_prssm.trainer import TrainConfig, make_split_batches, train_model  # noqa: E402

from _common import expected_load_vector  # noqa: E402

TARGET = OUTPUT_ROOT / "manifests/REPRODUCTION_CHECK.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="ct_ewma_g0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--max-epochs", type=int, default=2)
    args = parser.parse_args()

    archived = None
    for path in sorted((OUTPUT_ROOT / "generator_ladder/runs").glob("*.json")):
        record = json.loads(path.read_text())
        if (record.get("arm") == args.arm and record.get("seed") == args.seed
                and record.get("cohort") == args.cohort):
            archived = record
            break
    if archived is None:
        raise SystemExit(f"no archived run for {args.arm} seed {args.seed}")

    patients = load_tensors(resolve_cohort(args.cohort))
    config = TrainConfig(max_epochs=args.max_epochs, tbptt_length=64,
                         max_train_events_per_patient=30000, seed=args.seed)
    torch.manual_seed(args.seed)
    model = EpiPRSSM(feature_dim=patients[0].node_features.shape[-1], **ARMS[args.arm])
    report = train_model(model, patients, config, guard=LeakageGuard(stage="reproduction"))
    history = report.history

    reference = archived["train_report"]["history"][: len(history)]
    deltas = [abs(a["train_loss"] - b["train_loss"]) for a, b in zip(history, reference)]
    val_deltas = [abs(a["validation_loss"] - b["validation_loss"])
                  for a, b in zip(history, reference)]
    identical = bool(deltas and max(deltas) < 1e-9 and max(val_deltas) < 1e-9)
    payload = {
        "contract": "topic5_epi_prssm_v0_1_reproduction_check",
        "arm": args.arm, "seed": args.seed, "cohort": args.cohort,
        "epochs_compared": len(history),
        "archived_job_id": archived["job_id"],
        "archived_package_hash": archived["package_hash"],
        "current_package_hash": package_hash(),
        "package_hash_matches": archived["package_hash"] == package_hash(),
        "max_abs_train_loss_delta": float(max(deltas)) if deltas else None,
        "max_abs_validation_loss_delta": float(max(val_deltas)) if val_deltas else None,
        "bit_identical": identical,
        "interpretation": (
            "the source edits made after this arm started are behaviour-identical for it"
            if identical else
            "the source edits changed this arm's behaviour; the archived numbers must be "
            "re-run, not reused"),
        "per_epoch": [{"epoch": i, "archived_train": b["train_loss"],
                       "recomputed_train": a["train_loss"],
                       "archived_validation": b["validation_loss"],
                       "recomputed_validation": a["validation_loss"]}
                      for i, (a, b) in enumerate(zip(history, reference))],
    }
    atomic_write_json(TARGET, payload)
    print(json.dumps({k: payload[k] for k in
                      ("arm", "seed", "epochs_compared", "package_hash_matches",
                       "max_abs_train_loss_delta", "max_abs_validation_loss_delta",
                       "bit_identical", "interpretation")}, indent=2))


if __name__ == "__main__":
    main()
