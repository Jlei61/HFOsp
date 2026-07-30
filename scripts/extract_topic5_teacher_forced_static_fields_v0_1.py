#!/usr/bin/env python3
"""Extract target-blind teacher-forced probability fields from frozen GRUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    strict_clinical_inventory,
)
from scripts.train_topic5_interictal_rank_distribution import _model  # noqa: E402
from src.topic5_rnn_internal_state import (  # noqa: E402
    teacher_forced_probability_fields,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
RUN_ROOT = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs"
    / "formal_multiseed_20260725_v1"
)
OUT = (
    ROOT
    / "results/topic5_static_scaffold_fixed_readout_validation"
    / "teacher_forced_fields/per_seed"
)
SEEDS = (20260725, 20260726, 20260727)
CONTROLS = ("full_history_gru", "rank_shuffle_gru")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard_index < args.n_shards:
        raise ValueError("invalid shard assignment")
    subjects = sorted(strict_clinical_inventory())
    assigned = subjects[args.shard_index :: args.n_shards]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.20, device=device)
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    completed = []
    for subject_index, subject in enumerate(assigned):
        dataset_path = DATASET / "per_subject" / f"{subject}.npz"
        with np.load(dataset_path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            features = np.asarray(data["contact_features"], dtype=np.float32)
            groups = np.asarray(data["event_group_ids"], dtype=np.int16)
            counts = np.asarray(data["event_group_count"], dtype=np.int16)
            split = np.asarray(data["event_split"], dtype=np.uint8)
        eval_indices = np.flatnonzero(split == 1)
        feature_tensor = torch.as_tensor(features, device=device)
        for seed in SEEDS:
            for control in CONTROLS:
                checkpoint = (
                    RUN_ROOT
                    / f"seed_{seed}"
                    / subject
                    / f"{control}_checkpoint.pt"
                )
                payload = torch.load(
                    checkpoint, map_location=device, weights_only=False
                )
                if payload.get("ictal_target_read", True):
                    raise RuntimeError(
                        f"{subject}/{seed}/{control}: checkpoint target seal failed"
                    )
                model = _model(
                    control,
                    features.shape[1],
                    payload["model_kwargs"],
                ).to(device)
                model.load_state_dict(payload["model_state"])
                offset = payload["heldout_local_offset"].to(device)
                result = teacher_forced_probability_fields(
                    model,
                    feature_tensor,
                    offset,
                    groups,
                    counts,
                    eval_indices,
                    batch_size=args.batch_size,
                )
                output = OUT / f"{subject}_seed{seed}_{control}.npz"
                np.savez_compressed(
                    output,
                    contact_names=names,
                    union_participation=result["union_participation"],
                    summed_next_probability=result[
                        "summed_next_probability"
                    ],
                    event_union_mass=result["event_union_mass"],
                    eval_indices=eval_indices,
                )
                metadata = {
                    "contract": (
                        "topic5_static_scaffold_fixed_readout_validation_v0_1"
                    ),
                    "phase": "teacher_forced_field_extraction",
                    "subject": subject,
                    "seed": seed,
                    "control": control,
                    "target_values_read": False,
                    "field_definition": (
                        "event-first mean of 1-prod_t(1-p_i,t) along observed "
                        "heldout20 prefixes, including terminal STOP competition"
                    ),
                    "not_free_rollout": True,
                    "n_eval_events": int(len(eval_indices)),
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_sha256": sha256(checkpoint),
                    "dataset_npz_sha256": sha256(dataset_path),
                    "output_npz_sha256": sha256(output),
                }
                atomic_json(output.with_suffix(".json"), metadata)
                completed.append(metadata)
                del model, payload, offset
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        print(
            f"teacher-forced shard {args.shard_index}: "
            f"{subject_index + 1}/{len(assigned)} {subject}",
            flush=True,
        )
    status = {
        "status": "COMPLETE",
        "shard_index": args.shard_index,
        "n_shards": args.n_shards,
        "device": str(device),
        "n_subjects": len(assigned),
        "n_cells": len(completed),
        "target_values_read": False,
        "subjects": assigned,
    }
    atomic_json(
        OUT.parent / f"SHARD_{args.shard_index}_STATUS.json", status
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
