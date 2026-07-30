#!/usr/bin/env python3
"""Frozen-model rank-set tolerance sensitivity for one patient."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_minimal_sequence_kernel_cell_v0_2 import (  # noqa: E402
    DATASET,
    _batch,
    _checkpoint_path,
    _load_model,
)
from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_minimal_sequence_kernel import (  # noqa: E402
    decomposed_next_set_stop_loss,
    merge_frozen_groups_by_lag_tolerance,
)
from src.topic5_rank_distribution import next_set_stop_loss  # noqa: E402


TOLERANCES_MS = (0.0, 1.0, 2.0, 5.0, 10.0)
CONDITIONS = ("unordered_prefix", "history_3", "linear_state")
SEEDS = (20260725, 20260726, 20260727)


def _encode(
    frozen_groups: np.ndarray,
    frozen_counts: np.ndarray,
    lag_raw: np.ndarray,
    tolerance_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    return merge_frozen_groups_by_lag_tolerance(
        frozen_groups,
        frozen_counts,
        lag_raw,
        tolerance_seconds=float(tolerance_ms) / 1000.0,
    )


@torch.no_grad()
def _metrics(
    model,
    record,
    offset: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    values = {
        "event_total_nll": [],
        "event_contact_choice_nll": [],
        "event_stop_contribution_nll": [],
        "event_continue_nll": [],
        "event_terminal_stop_nll": [],
    }
    maximum_error = 0.0
    decisions = 0
    nonterminal = 0
    for start in range(0, len(record.eval_indices), int(batch_size)):
        chunk = record.eval_indices[start : start + int(batch_size)]
        batch = _batch(
            record,
            chunk,
            device,
            rank_shuffle=False,
            rng=np.random.default_rng(0),
        )
        output = model(**batch, local_offset=offset)
        original = next_set_stop_loss(
            output, batch["group_ids"], batch["group_count"]
        )
        split = decomposed_next_set_stop_loss(
            output, batch["group_ids"], batch["group_count"]
        )
        maximum_error = max(
            maximum_error,
            float(
                torch.max(
                    torch.abs(
                        original["event_nll"] - split["event_total_nll"]
                    )
                ).cpu()
            ),
        )
        for key in values:
            values[key].extend(split[key].cpu().numpy())
        decisions += int(torch.sum(split["decision_mask"]).cpu())
        nonterminal += int(torch.sum(split["nonterminal_mask"]).cpu())
    return {
        **{key: float(np.mean(value)) for key, value in values.items()},
        "n_events": int(len(record.eval_indices)),
        "n_decisions": decisions,
        "n_nonterminal_decisions": nonterminal,
        "maximum_reconstruction_error": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.14)
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(DATASET)
    record = records[args.subject]
    with np.load(record.path, allow_pickle=False) as data:
        lag_raw = np.asarray(data["event_lag_raw"], np.float64)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_memory_fraction)
        )
    torch.set_num_threads(int(args.cpu_threads))

    rows = []
    tie_rows = []
    for tolerance_ms in TOLERANCES_MS:
        groups, counts = _encode(
            record.group_ids, record.group_count, lag_raw, tolerance_ms
        )
        if tolerance_ms == 0.0:
            if not np.array_equal(groups, record.group_ids) or not np.array_equal(
                counts, record.group_count
            ):
                raise RuntimeError("zero-tolerance rebuild does not match frozen data")
        current_record = replace(
            record, group_ids=groups, group_count=counts
        )
        eval_groups = groups[record.eval_indices]
        eval_counts = counts[record.eval_indices]
        tie_sets = sum(
            int(np.sum(event == step) > 1)
            for event, count in zip(eval_groups, eval_counts)
            for step in range(int(count))
        )
        tie_rows.append(
            {
                "subject": record.subject,
                "dataset": record.dataset,
                "tolerance_ms": tolerance_ms,
                "n_eval_events": int(len(eval_groups)),
                "n_eval_rank_sets": int(np.sum(eval_counts)),
                "n_eval_tied_rank_sets": int(tie_sets),
                "mean_eval_group_count": float(np.mean(eval_counts)),
            }
        )
        for seed in SEEDS:
            for condition in CONDITIONS:
                checkpoint = _checkpoint_path(
                    condition, seed, record.subject
                )
                model, offset, _ = _load_model(
                    checkpoint,
                    condition=condition,
                    feature_dim=record.contact_features.shape[1],
                    subject=record.subject,
                    device=device,
                )
                rows.append(
                    {
                        "subject": record.subject,
                        "dataset": record.dataset,
                        "seed": int(seed),
                        "condition": condition,
                        "tolerance_ms": tolerance_ms,
                        **_metrics(
                            model,
                            current_record,
                            offset,
                            device=device,
                            batch_size=args.batch_size,
                        ),
                    }
                )
                del model, offset
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows)
    if float(metrics.maximum_reconstruction_error.max()) > 2e-5:
        raise RuntimeError("likelihood decomposition failed")
    metrics.to_csv(output_dir / "tolerance_metrics.csv", index=False)
    pd.DataFrame(tie_rows).to_csv(
        output_dir / "tolerance_cardinality.csv", index=False
    )
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "subject": record.subject,
        "dataset": record.dataset,
        "tolerances_ms": list(TOLERANCES_MS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "zero_tolerance_exact_rebuild": True,
        "target_values_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
