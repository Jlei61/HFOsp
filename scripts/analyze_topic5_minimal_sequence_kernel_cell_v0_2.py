#!/usr/bin/env python3
"""Re-score one patient/seed cell under the minimal sequence-kernel contract."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _batch,
    load_records,
)
from src.topic5_minimal_sequence_kernel import (  # noqa: E402
    block_hankel_from_lag_kernels,
    decomposed_next_set_stop_loss,
    hankel_singular_summary,
    linear_state_contact_lag_kernels,
    linear_state_lag_ablation_outputs,
)
from src.topic5_rank_distribution import (  # noqa: E402
    FullHistorySequenceGRU,
    LinearStateSequenceRNN,
    StaticSequenceContactQuery,
    WindowedHistorySequenceGRU,
    next_set_stop_loss,
)


DATASET = (
    ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
)
BASE_ROOT = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
HISTORY_ROOT = (
    ROOT
    / "results/topic5_interictal_scaffold_reliability_history_necessity/"
    "history_runs_v0_1"
)
HISTORY_SHUFFLE_ROOT = (
    ROOT
    / "results/topic5_interictal_scaffold_reliability_history_necessity/"
    "history3_rank_shuffle_runs_v0_1"
)
LINEAR_ROOT = (
    ROOT
    / "results/topic5_ordered_history_architecture_audit/formal/"
    "architecture_controls_formal_20260729/linear_state"
)
LINEAR_SHUFFLE_ROOT = (
    ROOT
    / "results/topic5_ordered_history_architecture_audit/rank_shuffle/"
    "selected_architecture_rank_shuffle_20260729/linear_state"
)


CONDITIONS = (
    "unordered_prefix",
    "history_1",
    "history_2",
    "history_3",
    "full_history",
    "history_3_rank_shuffle",
    "linear_state",
    "linear_state_rank_shuffle",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(condition: str, seed: int, subject: str) -> Path:
    seed_dir = f"seed_{int(seed)}"
    if condition == "unordered_prefix":
        return BASE_ROOT / seed_dir / subject / "unordered_prefix_checkpoint.pt"
    if condition == "full_history":
        return BASE_ROOT / seed_dir / subject / "full_history_gru_checkpoint.pt"
    if condition in {"history_1", "history_2", "history_3"}:
        window = int(condition[-1])
        return (
            HISTORY_ROOT
            / seed_dir
            / subject
            / f"history_{window}_gru"
            / "checkpoint.pt"
        )
    if condition == "history_3_rank_shuffle":
        return (
            HISTORY_SHUFFLE_ROOT
            / seed_dir
            / subject
            / "history_3_rank_shuffle_gru"
            / "checkpoint.pt"
        )
    if condition == "linear_state":
        return (
            LINEAR_ROOT
            / seed_dir
            / subject
            / "linear_state_checkpoint.pt"
        )
    if condition == "linear_state_rank_shuffle":
        return (
            LINEAR_SHUFFLE_ROOT
            / seed_dir
            / subject
            / "linear_state_rank_shuffle_checkpoint.pt"
        )
    raise ValueError(f"unknown condition: {condition}")


def _load_model(
    path: Path,
    *,
    condition: str,
    feature_dim: int,
    subject: str,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if bool(payload.get("ictal_target_read", True)):
        raise RuntimeError(f"{path}: target seal failed")
    if str(payload.get("heldout_subject")) != subject:
        raise RuntimeError(f"{path}: heldout subject mismatch")
    kwargs = dict(payload["model_kwargs"])
    kwargs.pop("contact_feature_dim", None)
    if condition == "unordered_prefix":
        model = StaticSequenceContactQuery(
            feature_dim, mode="unordered", **kwargs
        )
    elif condition == "full_history":
        model = FullHistorySequenceGRU(feature_dim, **kwargs)
    elif condition in {"history_1", "history_2", "history_3"}:
        model = WindowedHistorySequenceGRU(
            feature_dim, history_window=int(condition[-1]), **kwargs
        )
    elif condition == "history_3_rank_shuffle":
        model = WindowedHistorySequenceGRU(
            feature_dim, history_window=3, **kwargs
        )
    elif condition in {"linear_state", "linear_state_rank_shuffle"}:
        model = LinearStateSequenceRNN(feature_dim, **kwargs)
    else:
        raise ValueError(condition)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload["heldout_local_offset"].to(device), payload


def _mask_hex(mask: np.ndarray) -> str:
    return np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes().hex()


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    record,
    offset: torch.Tensor,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_path: Path,
    batch_size: int,
    output_transform=None,
) -> dict:
    """Evaluate and stream one row per heldout decision to compressed CSV."""

    event_total = []
    event_stop = []
    event_contact_contribution = []
    event_contact_choice = []
    event_continue = []
    event_terminal_stop = []
    original_event = []
    stop_probability = []
    stop_target = []
    n_decisions = 0
    n_nonterminal = 0
    n_tied_decisions = 0
    maximum_reconstruction_error = 0.0

    header = [
        "subject",
        "dataset",
        "seed",
        "condition",
        "event_index",
        "event_source_index",
        "prediction_step",
        "terminal",
        "n_candidates",
        "target_set_size",
        "candidate_mask_hex",
        "total_nll",
        "stop_decision_nll",
        "contact_choice_nll",
        "joint_stop_probability",
    ]
    with gzip.open(output_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        indices = record.eval_indices
        for start in range(0, len(indices), int(batch_size)):
            chunk = indices[start : start + int(batch_size)]
            batch = _batch(
                record,
                chunk,
                device,
                rank_shuffle=False,
                rng=np.random.default_rng(0),
            )
            if output_transform is None:
                outputs = model(**batch, local_offset=offset)
            else:
                outputs = output_transform(model, batch, offset)
            original = next_set_stop_loss(
                outputs, batch["group_ids"], batch["group_count"]
            )
            split = decomposed_next_set_stop_loss(
                outputs, batch["group_ids"], batch["group_count"]
            )
            error = torch.max(
                torch.abs(original["event_nll"] - split["event_total_nll"])
            )
            maximum_reconstruction_error = max(
                maximum_reconstruction_error, float(error.cpu())
            )

            arrays = {
                key: value.detach().cpu().numpy()
                for key, value in split.items()
                if isinstance(value, torch.Tensor) and value.ndim
            }
            candidates = outputs["candidate_mask"].detach().cpu().numpy()
            groups = record.group_ids[chunk]
            counts = record.group_count[chunk]
            original_event.extend(original["event_nll"].cpu().numpy())
            event_total.extend(arrays["event_total_nll"])
            event_stop.extend(arrays["event_stop_contribution_nll"])
            event_contact_contribution.extend(
                arrays["event_contact_contribution_nll"]
            )
            event_contact_choice.extend(arrays["event_contact_choice_nll"])
            event_continue.extend(arrays["event_continue_nll"])
            event_terminal_stop.extend(arrays["event_terminal_stop_nll"])

            for local, event_index in enumerate(chunk):
                count = int(counts[local])
                for step in range(count + 1):
                    terminal = step == count
                    target_size = (
                        0 if terminal else int(np.sum(groups[local] == step))
                    )
                    candidate = candidates[local, step]
                    probability = float(arrays["stop_probability"][local, step])
                    stop_probability.append(probability)
                    stop_target.append(float(terminal))
                    n_decisions += 1
                    n_nonterminal += int(not terminal)
                    n_tied_decisions += int(target_size > 1)
                    writer.writerow(
                        [
                            record.subject,
                            record.dataset,
                            int(seed),
                            condition,
                            int(event_index),
                            int(record.event_source_index[event_index]),
                            step,
                            int(terminal),
                            int(np.sum(candidate)),
                            target_size,
                            _mask_hex(candidate),
                            float(arrays["decision_total_nll"][local, step]),
                            float(arrays["decision_stop_nll"][local, step]),
                            float(
                                arrays["decision_contact_choice_nll"][local, step]
                            ),
                            probability,
                        ]
                    )

    stop_probability_array = np.asarray(stop_probability, dtype=np.float64)
    stop_target_array = np.asarray(stop_target, dtype=np.float64)
    return {
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(seed),
        "condition": condition,
        "n_events": int(len(record.eval_indices)),
        "n_decisions": int(n_decisions),
        "n_nonterminal_decisions": int(n_nonterminal),
        "n_tied_nonterminal_decisions": int(n_tied_decisions),
        "event_total_nll": float(np.mean(event_total)),
        "event_stop_contribution_nll": float(np.mean(event_stop)),
        "event_contact_contribution_nll": float(
            np.mean(event_contact_contribution)
        ),
        "event_contact_choice_nll": float(np.mean(event_contact_choice)),
        "event_continue_nll": float(np.mean(event_continue)),
        "event_terminal_stop_nll": float(np.mean(event_terminal_stop)),
        "joint_stop_brier": float(
            np.mean((stop_probability_array - stop_target_array) ** 2)
        ),
        "joint_terminal_stop_probability": float(
            np.mean(stop_probability_array[stop_target_array == 1])
        ),
        "joint_nonterminal_stop_probability": float(
            np.mean(stop_probability_array[stop_target_array == 0])
        ),
        "maximum_event_nll_reconstruction_error": float(
            maximum_reconstruction_error
        ),
        "original_event_nll": float(np.mean(original_event)),
    }


def _write_kernel_outputs(
    model: torch.nn.Module,
    record,
    offset: torch.Tensor,
    output_dir: Path,
    *,
    seed: int,
    checkpoint_sha256: str,
    device: torch.device,
) -> dict:
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    )
    kernels = linear_state_contact_lag_kernels(
        model, features, offset, max_lag=5
    )
    contact = kernels["contact"].cpu().numpy().astype(np.float64)
    stop = kernels["stop"].cpu().numpy().astype(np.float64)
    combined = np.concatenate([contact, stop], axis=1)
    contact_hankel = block_hankel_from_lag_kernels(contact)
    combined_hankel = block_hankel_from_lag_kernels(combined)
    contact_summary = hankel_singular_summary(contact_hankel)
    combined_summary = hankel_singular_summary(combined_hankel)
    np.savez_compressed(
        output_dir / "linear_state_lag_kernels.npz",
        contact_kernels=contact,
        stop_kernels=stop,
        persistence=kernels["persistence"].cpu().numpy(),
        contact_embedding=kernels["contact_embedding"].cpu().numpy(),
        contact_names=record.contact_names,
        contact_hankel=contact_hankel,
        combined_hankel=combined_hankel,
        contact_hankel_singular_values=contact_summary["singular_values"],
        combined_hankel_singular_values=combined_summary["singular_values"],
    )
    return {
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(seed),
        "checkpoint_sha256": checkpoint_sha256,
        "contact_hankel_rank90": int(contact_summary["rank90"]),
        "contact_hankel_rank95": int(contact_summary["rank95"]),
        "contact_hankel_effective_order": float(
            contact_summary["effective_order"]
        ),
        "combined_hankel_rank90": int(combined_summary["rank90"]),
        "combined_hankel_rank95": int(combined_summary["rank95"]),
        "combined_hankel_effective_order": float(
            combined_summary["effective_order"]
        ),
        "persistence_median": float(
            np.median(kernels["persistence"].cpu().numpy())
        ),
        "persistence_maximum": float(
            np.max(kernels["persistence"].cpu().numpy())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.18)
    args = parser.parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(
            float(args.gpu_memory_fraction)
        )
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(args.cpu_threads))
    records = load_records(DATASET)
    if args.subject not in records:
        raise RuntimeError(f"unknown subject: {args.subject}")
    record = records[args.subject]
    started = time.time()
    metrics = []
    checkpoint_manifest = {}
    kernel_summary = None
    for condition in args.conditions:
        checkpoint = _checkpoint_path(condition, args.seed, args.subject)
        model, offset, _ = _load_model(
            checkpoint,
            condition=condition,
            feature_dim=record.contact_features.shape[1],
            subject=args.subject,
            device=device,
        )
        checkpoint_hash = _sha256(checkpoint)
        checkpoint_manifest[condition] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": checkpoint_hash,
        }
        metrics.append(
            _evaluate(
                model,
                record,
                offset,
                condition=condition,
                seed=args.seed,
                device=device,
                output_path=output_dir / f"{condition}_decisions.csv.gz",
                batch_size=args.batch_size,
            )
        )
        if condition == "linear_state":
            kernel_summary = _write_kernel_outputs(
                model,
                record,
                offset,
                output_dir,
                seed=args.seed,
                checkpoint_sha256=checkpoint_hash,
                device=device,
            )
            for label, exact_lags, from_lag in (
                ("lag0_removed", [0], None),
                ("lag1_removed", [1], None),
                ("lag2_removed", [2], None),
                ("lag3plus_removed", None, 3),
            ):
                metrics.append(
                    _evaluate(
                        model,
                        record,
                        offset,
                        condition=label,
                        seed=args.seed,
                        device=device,
                        output_path=output_dir / f"{label}_decisions.csv.gz",
                        batch_size=args.batch_size,
                        output_transform=(
                            lambda current_model, batch, current_offset,
                            exact_lags=exact_lags, from_lag=from_lag:
                            linear_state_lag_ablation_outputs(
                                current_model,
                                batch["contact_features"],
                                batch["contact_mask"],
                                batch["group_ids"],
                                batch["group_count"],
                                current_offset,
                                ablate_lags=exact_lags,
                                ablate_from_lag=from_lag,
                            )
                        ),
                    )
                )
        del model, offset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame(metrics)
    if float(metrics_frame.maximum_event_nll_reconstruction_error.max()) > 2e-5:
        raise RuntimeError("joint likelihood decomposition failed reconstruction")
    metrics_frame.to_csv(output_dir / "component_metrics.csv", index=False)
    if kernel_summary is not None:
        pd.DataFrame([kernel_summary]).to_csv(
            output_dir / "kernel_summary.csv", index=False
        )
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "subject": record.subject,
        "dataset": record.dataset,
        "seed": int(args.seed),
        "conditions": list(args.conditions),
        "dataset_npz_sha256": record.input_sha256,
        "checkpoint_manifest": checkpoint_manifest,
        "target_values_read": False,
        "ab_labels_read": False,
        "resources": {
            "runtime_seconds": float(time.time() - started),
            "cpu_threads": int(torch.get_num_threads()),
            "peak_rss_gb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.type == "cuda"
                else 0
            ),
            "gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if device.type == "cuda"
                else 0
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
