#!/usr/bin/env python3
"""Run one patient/seed constructive source-conditioned generation cell."""
from __future__ import annotations

import argparse
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

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_constructive_event_generator import (  # noqa: E402
    VALID_CONDITIONS,
    constant_stop_hazard,
    event_length_wasserstein,
    remove_revealed_source,
    shaft_preserving_permutation,
    source_conditioned_rollout,
    stop_hazard_curve,
    train_progress_hazard,
    train_static_log_scaffold,
)
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    distribution_errors,
)


CONDITION_ORDER = [
    "full_constructive",
    "static_only",
    "static_shuffle",
    "history_h1",
    "history_h2",
    "constant_stop",
    "no_termination",
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(subject: str, seed: int, suffix: str) -> int:
    token = hashlib.sha256(f"{subject}:{seed}:{suffix}".encode()).hexdigest()
    return int(token[:8], 16)


def _equal_halves(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    midpoint = len(indices) // 2
    n = min(midpoint, len(indices) - midpoint)
    if n < 1:
        raise RuntimeError("heldout split has too few events")
    return indices[midpoint - n : midpoint], indices[midpoint : midpoint + n]


def _stop_curve_mae(
    predicted_count: np.ndarray,
    observed_count: np.ndarray,
    *,
    n_contacts: int,
) -> float:
    predicted = stop_hazard_curve(predicted_count, max_groups=n_contacts)
    observed = stop_hazard_curve(observed_count, max_groups=n_contacts)
    valid = np.isfinite(predicted) & np.isfinite(observed)
    return float(np.mean(np.abs(predicted[valid] - observed[valid])))


def _sink_distribution(groups: np.ndarray, counts: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups, dtype=int)
    counts = np.asarray(counts, dtype=int)
    out = np.zeros(groups.shape[1], dtype=float)
    valid_events = 0
    for event, length in zip(groups, counts):
        sink = np.flatnonzero(event == int(length) - 1)
        if sink.size:
            out[sink] += 1.0 / sink.size
            valid_events += 1
    if valid_events:
        out /= valid_events
    return out


def _source_sink_distance(
    groups: np.ndarray,
    counts: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or not np.all(np.isfinite(coords))
    ):
        return np.asarray([], dtype=float)
    distance = []
    for event, length in zip(np.asarray(groups, int), np.asarray(counts, int)):
        source = np.flatnonzero(event == 0)
        sink = np.flatnonzero(event == int(length) - 1)
        if source.size and sink.size:
            distance.append(
                np.linalg.norm(
                    np.mean(coords[sink], axis=0)
                    - np.mean(coords[source], axis=0)
                )
            )
    return np.asarray(distance, dtype=float)


def _distance_w1(predicted: np.ndarray, observed: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    if predicted.size == 0 or observed.size == 0:
        return float("nan")
    scale = max(float(np.ptp(observed)), 1.0)
    return float(wasserstein_distance(predicted, observed) / scale)


def _strict_jsonable(value):
    """Convert NumPy scalars and non-finite diagnostics for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()

    started = time.time()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    state_path = out_dir / "run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "contract": "topic5_constructive_event_generation_v0_1",
                "subject": args.subject,
                "seed": int(args.seed),
                "ictal_target_read": False,
                "ab_or_axis_used_during_rollout": False,
            },
            indent=2,
        )
        + "\n"
    )

    torch.set_num_threads(int(args.cpu_threads))
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_memory_fraction))
        torch.cuda.reset_peak_memory_stats()

    records = load_records(
        args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    )
    record = records[args.subject]
    with np.load(record.path, allow_pickle=False) as subject_npz:
        contact_coords = np.asarray(subject_npz["contact_coords"], dtype=float)
    if contact_coords.shape != (record.contact_features.shape[0], 3):
        raise RuntimeError("contact coordinate array is not aligned with contacts")
    checkpoint_path = (
        args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "linear_state":
        raise RuntimeError("constructive generator requires a linear_state checkpoint")
    if checkpoint.get("heldout_subject") != args.subject:
        raise RuntimeError("checkpoint subject mismatch")
    if int(checkpoint.get("seed")) != int(args.seed):
        raise RuntimeError("checkpoint seed mismatch")
    if checkpoint.get("ictal_target_read") is not False:
        raise RuntimeError("checkpoint is not target sealed")

    model = LinearStateSequenceRNN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    offset = checkpoint["heldout_local_offset"].to(device)
    features = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    ).unsqueeze(0)
    contact_mask = torch.ones(
        (1, record.contact_features.shape[0]), dtype=torch.bool, device=device
    )

    eval_indices = np.asarray(record.eval_indices, dtype=int)
    observed_groups = np.asarray(record.group_ids[eval_indices], dtype=np.int16)
    observed_count = np.asarray(record.group_count[eval_indices], dtype=np.int16)
    source_mask = observed_groups == 0
    n_events, n_contacts = observed_groups.shape
    random_seed = _seed(args.subject, args.seed, "paired_uniforms_v0_1")
    uniforms = np.random.default_rng(random_seed).random(
        (n_events, n_contacts), dtype=np.float64
    )
    uniforms_hash = _sha256_bytes(uniforms.tobytes())
    static = train_static_log_scaffold(record.group_ids, record.train_indices)
    progress = train_progress_hazard(
        record.group_count,
        record.train_indices,
        max_groups=n_contacts,
    )
    constant = constant_stop_hazard(record.group_count, record.train_indices)
    permutation = shaft_preserving_permutation(
        record.contact_names,
        seed=_seed(args.subject, args.seed, "static_shaft_shuffle"),
    )

    suffix_observed = remove_revealed_source(observed_groups, source_mask)
    suffix_observed_count = np.maximum(observed_count - 1, 0)
    observed_distance = _source_sink_distance(
        observed_groups, observed_count, contact_coords
    )
    observed_sink = _sink_distribution(observed_groups, observed_count)
    metric_rows = []
    payload: dict[str, np.ndarray] = {
        "observed_group_ids": observed_groups,
        "observed_group_count": observed_count,
        "revealed_source_mask": source_mask.astype(np.uint8),
        "eval_indices": eval_indices,
        "uniforms_seed": np.asarray(random_seed, dtype=np.uint64),
        "static_log_scaffold": static.astype(np.float32),
        "progress_hazard": progress.astype(np.float32),
        "constant_hazard": np.asarray(constant, dtype=np.float32),
        "static_permutation": permutation.astype(np.int16),
    }

    for condition in CONDITION_ORDER:
        rollout = source_conditioned_rollout(
            model,
            features,
            contact_mask,
            offset,
            source_mask,
            uniforms,
            static,
            progress,
            condition=condition,
            static_permutation=permutation,
            constant_hazard=constant,
            batch_size=int(args.batch_size),
            uniforms_sha256=uniforms_hash,
        )
        generated_groups = rollout.event_group_ids
        generated_count = rollout.event_group_count
        suffix_generated = remove_revealed_source(
            generated_groups, source_mask
        )
        suffix_generated_count = np.maximum(generated_count - 1, 0)
        whole = distribution_errors(
            generated_groups,
            generated_count,
            observed_groups,
            observed_count,
            bins=10,
        )
        suffix = distribution_errors(
            suffix_generated,
            suffix_generated_count,
            suffix_observed,
            suffix_observed_count,
            bins=10,
        )
        generated_distance = _source_sink_distance(
            generated_groups, generated_count, contact_coords
        )
        generated_sink = _sink_distribution(generated_groups, generated_count)
        metric_rows.append(
            {
                "subject": args.subject,
                "dataset": record.dataset,
                "seed": int(args.seed),
                "condition": condition,
                "n_events": int(n_events),
                "n_contacts": int(n_contacts),
                **{f"whole_{key}": value for key, value in whole.items()},
                **{f"suffix_{key}": value for key, value in suffix.items()},
                "event_length_wasserstein": event_length_wasserstein(
                    generated_count, observed_count
                ),
                "stop_hazard_mae": _stop_curve_mae(
                    generated_count, observed_count, n_contacts=n_contacts
                ),
                "sink_distribution_mae": float(
                    np.mean(np.abs(generated_sink - observed_sink))
                ),
                "source_sink_distance_wasserstein": _distance_w1(
                    generated_distance, observed_distance
                ),
                "generated_zero_suffix_fraction": float(
                    np.mean(generated_count == 1)
                ),
            }
        )
        payload[f"{condition}__event_group_ids"] = generated_groups
        payload[f"{condition}__event_group_count"] = generated_count

    first, second = _equal_halves(np.arange(n_events))
    empirical_whole = distribution_errors(
        observed_groups[first],
        observed_count[first],
        observed_groups[second],
        observed_count[second],
        bins=10,
    )
    empirical_suffix = distribution_errors(
        suffix_observed[first],
        suffix_observed_count[first],
        suffix_observed[second],
        suffix_observed_count[second],
        bins=10,
    )
    empirical_reference = {
        **{f"whole_{key}": value for key, value in empirical_whole.items()},
        **{f"suffix_{key}": value for key, value in empirical_suffix.items()},
        "event_length_wasserstein": event_length_wasserstein(
            observed_count[first], observed_count[second]
        ),
        "stop_hazard_mae": _stop_curve_mae(
            observed_count[first],
            observed_count[second],
            n_contacts=n_contacts,
        ),
        "sink_distribution_mae": float(
            np.mean(
                np.abs(
                    _sink_distribution(
                        observed_groups[first], observed_count[first]
                    )
                    - _sink_distribution(
                        observed_groups[second], observed_count[second]
                    )
                )
            )
        ),
        "source_sink_distance_wasserstein": _distance_w1(
            _source_sink_distance(
                observed_groups[first],
                observed_count[first],
                contact_coords,
            ),
            _source_sink_distance(
                observed_groups[second],
                observed_count[second],
                contact_coords,
            ),
        ),
        "n_events_per_half": int(len(first)),
    }

    np.savez_compressed(out_dir / "constructive_rollouts.npz", **payload)
    pd.DataFrame(metric_rows).to_csv(out_dir / "condition_metrics.csv", index=False)
    (out_dir / "empirical_reference.json").write_text(
        json.dumps(
            _strict_jsonable(empirical_reference),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_constructive_event_generation_v0_1",
        "subject": args.subject,
        "dataset": record.dataset,
        "seed": int(args.seed),
        "conditions": CONDITION_ORDER,
        "valid_conditions_match_spec": set(CONDITION_ORDER) == VALID_CONDITIONS,
        "n_events": int(n_events),
        "n_contacts": int(n_contacts),
        "n_train_events": int(record.train_indices.size),
        "revealed_source_only": True,
        "source_rows_identical_across_conditions": True,
        "uniforms_identical_across_conditions": True,
        "uniforms_sha256": uniforms_hash,
        "ictal_target_read": False,
        "ab_or_axis_used_during_rollout": False,
        "input_fingerprints": {
            "dataset_npz": record.input_sha256,
            "checkpoint": _sha256_file(checkpoint_path),
        },
        "resources": {
            "runtime_seconds": float(time.time() - started),
            "peak_rss_gb": float(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
            "gpu_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "gpu_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
            ),
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    state_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
