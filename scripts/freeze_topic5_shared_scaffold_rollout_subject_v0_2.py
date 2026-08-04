#!/usr/bin/env python3
"""Freeze one patient's exact-k bidirectional rollout fields without ictal data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_scaffold_field_readout import (  # noqa: E402
    bidirectional_rollout_fields,
    build_frozen_subject_field_record,
    learned_axis_source_pools,
    normalized_laplacian_source_pools,
)
from src.topic5_shared_scaffold_rnn import (  # noqa: E402
    OrdinaryDenseGRUBaseline,
    SharedScaffoldPropagationRNN,
)
from src.topic5_shared_scaffold_rollout import rollout_from_source_pool  # noqa: E402


DEFAULT_MODELS = ("structured", "ordinary_gru")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_rollout_seed(
    *, base: int, subject: str, model: str, training_seed: int, side: str
) -> int:
    text = f"{int(base)}|{subject}|{model}|{int(training_seed)}|{side}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**63 - 1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_model(checkpoint_path: Path, *, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model"])
    bias = np.asarray(checkpoint["participation_bias"], dtype=np.float32)
    state = checkpoint["model_state"]
    if model_name in {"structured", "structured_rank_shuffle"}:
        hyperparameters = checkpoint.get("model_hyperparameters", {})
        # The v0.3 operators are analytically rank two, so the checkpoint key
        # is provenance only and the constructor rejects any other value.
        low_rank = int(hyperparameters.get("low_rank", 2))
        model = SharedScaffoldPropagationRNN(
            fixed_adjacency=state["fixed_adjacency"],
            participation_bias=bias,
            low_rank=low_rank,
        )
    elif model_name == "ordinary_gru":
        hyperparameters = checkpoint.get("model_hyperparameters", {})
        hidden_size = int(
            hyperparameters.get("hidden_size", state["gru.weight_hh"].shape[1])
        )
        model = OrdinaryDenseGRUBaseline(
            participation_bias=bias,
            hidden_size=hidden_size,
        )
    else:
        raise ValueError(f"unsupported rollout model {model_name!r}")
    model.load_state_dict(state)
    model.to(device).eval()
    return checkpoint, model


def run_side(
    *,
    checkpoint_path: Path,
    model,
    source_indices: np.ndarray,
    side: str,
    horizon: int,
    n_rollouts: int,
    batch_size: int,
    rollout_seed: int,
    output_path: Path,
    resume: bool,
) -> np.ndarray:
    done_path = output_path.with_suffix(".DONE.json")
    checkpoint_sha = sha256_file(checkpoint_path)
    expected = {
        "checkpoint_sha256": checkpoint_sha,
        "source_indices_sha256": sha256_array(source_indices.astype("<i8")),
        "horizon": int(horizon),
        "n_rollouts": int(n_rollouts),
        "batch_size": int(batch_size),
        "rollout_seed": int(rollout_seed),
        "sampler": "exact_elementary_symmetric_dp",
        "first_arrival_dtype": "float64",
        "schema_version": 1,
    }
    if resume and output_path.exists() and done_path.exists():
        done = json.loads(done_path.read_text())
        if all(done.get(key) == value for key, value in expected.items()):
            if done.get("npz_sha256") == sha256_file(output_path):
                with np.load(output_path, allow_pickle=False) as data:
                    return np.asarray(data["first_arrival_mass"], dtype=np.float64)

    source = np.zeros(model.n_contacts, dtype=bool)
    source[np.asarray(source_indices, dtype=int)] = True
    result = rollout_from_source_pool(
        model,
        source_pool=source,
        horizon=int(horizon),
        n_rollouts=int(n_rollouts),
        seed=int(rollout_seed),
        batch_size=int(batch_size),
    )
    atomic_npz(
        output_path,
        event_group_ids=result.event_group_ids,
        event_group_count=result.event_group_count,
        first_arrival_mass=result.first_arrival_mass.astype(np.float64),
        source_at_step_zero=result.source_at_step_zero.astype(np.uint8),
        cumulative_participation_include_source=(
            result.cumulative_participation_include_source.astype(np.float32)
        ),
        cumulative_participation_post_source=(
            result.cumulative_participation_post_source.astype(np.float32)
        ),
        stop_step_histogram=result.stop_step_histogram,
        source_indices=np.asarray(source_indices, dtype=np.int16),
        side=np.asarray(side),
        horizon=np.asarray(horizon, dtype=np.int16),
        n_rollouts=np.asarray(n_rollouts, dtype=np.int32),
        rollout_seed=np.asarray(rollout_seed, dtype=np.int64),
        exact_conditional_k_subset_sampler=np.asarray(True),
        target_values_read=np.asarray(False),
    )
    atomic_json(
        done_path,
        {
            "status": "COMPLETE",
            **expected,
            "side": side,
            "npz_sha256": sha256_file(output_path),
            "target_values_read": False,
        },
    )
    return result.first_arrival_mass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    # Primary is the model's own signed axis; the diffusion-graph split is
    # kept as the pre-registered sensitivity rule and writes a parallel tree.
    parser.add_argument(
        "--source-pool-rule",
        choices=("learned_axis", "normalized_laplacian"),
        default="learned_axis",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    unknown = sorted(set(args.models).difference(config["models"]["names"]))
    if unknown:
        raise ValueError(f"unknown models {unknown}")
    seeds = list(map(int, config["training"]["seeds"]))
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    training_root = output_root / "per_subject" / args.subject
    freeze_dir = (
        "field_freeze"
        if args.source_pool_rule == "learned_axis"
        else "field_freeze_diffusion_graph_sensitivity"
    )
    subject_root = output_root / freeze_dir / "per_subject" / args.subject
    subject_root.mkdir(parents=True, exist_ok=True)
    n_rollouts = int(args.n_rollouts or config["rollout"]["n_rollouts_per_seed"])
    batch_size = int(args.batch_size or config["rollout"]["batch_size"])
    checkpoint_models = sorted(set(args.models).union({"structured"}))
    input_checkpoint_paths = {
        f"{model_name}/seed_{seed}": (
            training_root / model_name / f"seed_{seed}" / "checkpoint.pt"
        )
        for model_name in checkpoint_models
        for seed in seeds
    }
    missing_checkpoints = [
        str(path) for path in input_checkpoint_paths.values() if not path.exists()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(f"missing checkpoints: {missing_checkpoints}")
    input_checkpoint_hashes = {
        name: sha256_file(path) for name, path in input_checkpoint_paths.items()
    }
    code_hashes = {
        "config": sha256_file(config_path),
        "rollout_subject_script": sha256_file(Path(__file__).resolve()),
        "rollout_core": sha256_file(ROOT / "src/topic5_shared_scaffold_rollout.py"),
        "field_readout": sha256_file(
            ROOT / "src/topic5_shared_scaffold_field_readout.py"
        ),
    }
    done_path = subject_root / "DONE.json"
    if args.resume and done_path.exists():
        existing = json.loads(done_path.read_text())
        artifacts_valid = all(
            (subject_root / name).exists()
            and sha256_file(subject_root / name) == expected_hash
            for name, expected_hash in existing.get("artifact_sha256", {}).items()
        )
        if (
            existing.get("status") == "COMPLETE"
            and existing.get("models") == list(args.models)
            and existing.get("n_rollouts_per_seed_per_side") == n_rollouts
            and existing.get("rollout_batch_size") == batch_size
            and existing.get("input_checkpoint_sha256") == input_checkpoint_hashes
            and existing.get("code_sha256") == code_hashes
            and artifacts_valid
        ):
            print(json.dumps(existing, allow_nan=False), flush=True)
            return
    device = torch.device(args.device or config["resources"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        device_index = int(device.index or 0)
        torch.cuda.set_device(device_index)
        fraction = float(config["resources"].get("gpu_memory_fraction_per_process", 0.0))
        if 0.0 < fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
    else:
        device_index = None
    torch.set_num_threads(int(config["resources"]["torch_num_threads"]))
    started = time.time()

    structured_checkpoints = [
        training_root / "structured" / f"seed_{seed}" / "checkpoint.pt"
        for seed in seeds
    ]
    if any(not path.exists() for path in structured_checkpoints):
        missing = [str(path) for path in structured_checkpoints if not path.exists()]
        raise FileNotFoundError(f"missing structured checkpoints: {missing}")
    structured_operators = []
    structured_axes: dict[str, np.ndarray] = {}
    contact_names = None
    fit_indices = None
    structured_checkpoint_hashes = {}
    for seed, path in zip(seeds, structured_checkpoints):
        checkpoint, model = load_model(path, device=torch.device("cpu"))
        names = np.asarray(checkpoint["contact_names"]).astype(str)
        if contact_names is None:
            contact_names = names
            fit_indices = np.asarray(checkpoint["fit_indices"], dtype=np.int64)
        elif not np.array_equal(contact_names, names):
            raise RuntimeError("structured seeds do not share contact order")
        components = model.operator_components()
        structured_operators.append(
            components["W"].detach().cpu().numpy().astype(np.float64)
        )
        structured_axes[str(seed)] = (
            components["axis_coordinate"].detach().cpu().numpy().astype(np.float64)
        )
        structured_checkpoint_hashes[str(seed)] = sha256_file(path)
    assert contact_names is not None and fit_indices is not None
    W_ensemble = np.mean(structured_operators, axis=0)
    if not np.allclose(W_ensemble, W_ensemble.T, atol=1e-7, rtol=0.0):
        raise RuntimeError("seed-ensemble structured operator lost symmetry")
    if np.min(W_ensemble) < -1e-7:
        raise RuntimeError("seed-ensemble structured operator is negative")
    endpoint_fraction = float(config["rollout"]["endpoint_fraction"])
    if args.source_pool_rule == "learned_axis":
        diffusion = learned_axis_source_pools(
            structured_axes,
            contact_names=contact_names,
            endpoint_fraction=endpoint_fraction,
        )
    else:
        diffusion = normalized_laplacian_source_pools(
            W_ensemble,
            contact_names=contact_names,
            endpoint_fraction=endpoint_fraction,
        )

    dataset_path = (
        Path(config["dataset_artifact_root"]).resolve()
        / config["dataset_root"]
        / "per_subject"
        / f"{args.subject}.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        group_count = np.asarray(data["event_group_count"], dtype=np.int64)
        dataset_names = np.asarray(data["contact_names"]).astype(str)
    if not np.array_equal(contact_names, dataset_names):
        raise RuntimeError("checkpoint and dataset contact order differ")
    quantile = float(config["rollout"]["horizon_fit60_quantile"])
    try:
        raw_horizon = float(np.quantile(group_count[fit_indices], quantile, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        raw_horizon = float(
            np.quantile(group_count[fit_indices], quantile, interpolation="higher")
        )
    horizon = int(
        np.clip(
            int(raw_horizon),
            int(config["rollout"]["horizon_min"]),
            int(config["rollout"]["horizon_max"]),
        )
    )
    source_definition = {
        "contract": config["contract"],
        "subject": args.subject,
        "contact_names": contact_names.tolist(),
        "seeds": seeds,
        "structured_checkpoint_sha256_by_seed": structured_checkpoint_hashes,
        "W_ensemble": W_ensemble.tolist(),
        "W_ensemble_sha256": sha256_array(W_ensemble.astype("<f8")),
        "source_pool_rule": diffusion["source_pool_rule"],
        "structured_axis_by_seed": {
            seed: axis.tolist() for seed, axis in structured_axes.items()
        },
        "diffusion_coordinate": np.asarray(diffusion["diffusion_coordinate"]).tolist(),
        **{
            key: _jsonable(diffusion[key])
            for key in (
                "laplacian_eigenvalues",
                "first_nontrivial_eigenvalue",
                "seed_order",
                "seed_axis_sign_flipped",
                "seed_axis_pairwise_pearson",
                "min_seed_axis_pairwise_pearson",
            )
            if key in diffusion
        },
        "source_minus_indices": np.asarray(diffusion["source_minus_indices"]).tolist(),
        "source_plus_indices": np.asarray(diffusion["source_plus_indices"]).tolist(),
        "source_minus_contacts": diffusion["source_minus_contacts"],
        "source_plus_contacts": diffusion["source_plus_contacts"],
        "horizon": horizon,
        "horizon_rule": "fit60_rank_count_p90_clipped_3_12",
        "training_split_sha256": sha256_array(fit_indices.astype("<i8")),
        "target_values_read": False,
    }
    atomic_json(subject_root / "source_definition.json", source_definition)
    source_definition_sha = sha256_file(subject_root / "source_definition.json")

    record_paths = []
    aggregate_paths = []
    for model_name in args.models:
        checkpoint_hashes: dict[str, str] = {}
        seed_mass: dict[str, list[np.ndarray]] = {"minus": [], "plus": []}
        for training_seed in seeds:
            checkpoint_path = (
                training_root / model_name / f"seed_{training_seed}" / "checkpoint.pt"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint, model = load_model(checkpoint_path, device=device)
            if not np.array_equal(
                np.asarray(checkpoint["contact_names"]).astype(str), contact_names
            ):
                raise RuntimeError(f"{model_name} seed {training_seed}: contact mismatch")
            checkpoint_hashes[str(training_seed)] = sha256_file(checkpoint_path)
            for side in ("minus", "plus"):
                source_indices = np.asarray(
                    diffusion[f"source_{side}_indices"], dtype=np.int64
                )
                rollout_seed = stable_rollout_seed(
                    base=int(config["rollout"]["random_seed_base"]),
                    subject=args.subject,
                    model=model_name,
                    training_seed=training_seed,
                    side=side,
                )
                side_path = (
                    subject_root
                    / "per_seed"
                    / model_name
                    / f"seed_{training_seed}"
                    / f"{side}.npz"
                )
                mass = run_side(
                    checkpoint_path=checkpoint_path,
                    model=model,
                    source_indices=source_indices,
                    side=side,
                    horizon=horizon,
                    n_rollouts=n_rollouts,
                    batch_size=batch_size,
                    rollout_seed=rollout_seed,
                    output_path=side_path,
                    resume=bool(args.resume),
                )
                seed_mass[side].append(mass)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        ensemble_minus = np.mean(seed_mass["minus"], axis=0)
        ensemble_plus = np.mean(seed_mass["plus"], axis=0)
        fields = bidirectional_rollout_fields(
            first_arrival_minus=ensemble_minus,
            first_arrival_plus=ensemble_plus,
            source_minus_indices=diffusion["source_minus_indices"],
            source_plus_indices=diffusion["source_plus_indices"],
        )
        aggregate_path = subject_root / f"{model_name}_fields.npz"
        atomic_npz(
            aggregate_path,
            contact_names=contact_names,
            horizon=np.asarray(horizon, dtype=np.int16),
            W_ensemble=W_ensemble.astype(np.float32),
            diffusion_coordinate=np.asarray(
                diffusion["diffusion_coordinate"], dtype=np.float32
            ),
            source_minus_indices=np.asarray(
                diffusion["source_minus_indices"], dtype=np.int16
            ),
            source_plus_indices=np.asarray(
                diffusion["source_plus_indices"], dtype=np.int16
            ),
            first_arrival_mass_minus=ensemble_minus.astype(np.float32),
            first_arrival_mass_plus=ensemble_plus.astype(np.float32),
            field_minus=np.asarray(fields["field_minus"], dtype=np.float32),
            field_plus=np.asarray(fields["field_plus"], dtype=np.float32),
            seeds=np.asarray(seeds, dtype=np.int32),
            n_rollouts_per_seed=np.asarray(n_rollouts, dtype=np.int32),
            source_definition_sha256=np.asarray(source_definition_sha),
            target_values_read=np.asarray(False),
        )
        aggregate_paths.append(aggregate_path)
        record = build_frozen_subject_field_record(
            subject_id=args.subject,
            model_name=model_name,
            contact_names=contact_names,
            operator=W_ensemble,
            diffusion_result=diffusion,
            field_minus=fields["field_minus"],
            field_plus=fields["field_plus"],
            horizon=horizon,
            checkpoint_sha256_by_seed=checkpoint_hashes,
            training_split_sha256=source_definition["training_split_sha256"],
        )
        record_path = subject_root / f"{model_name}_field_record.json"
        atomic_json(record_path, record)
        record_paths.append(record_path)

    artifacts = [subject_root / "source_definition.json", *aggregate_paths, *record_paths]
    done = {
        "status": "COMPLETE",
        "contract": config["contract"],
        "subject": args.subject,
        "models": list(args.models),
        "seeds": seeds,
        "horizon": horizon,
        "n_rollouts_per_seed_per_side": n_rollouts,
        "rollout_batch_size": batch_size,
        "subset_sampler": "exact_elementary_symmetric_dp",
        "source_operator": "structured_seed_ensemble_mean",
        "source_pool_rule": diffusion["source_pool_rule"],
        "source_pool_rule_tier": (
            "primary" if args.source_pool_rule == "learned_axis" else "sensitivity"
        ),
        "source_definition_sha256": source_definition_sha,
        "input_checkpoint_sha256": input_checkpoint_hashes,
        "code_sha256": code_hashes,
        "artifact_sha256": {path.name: sha256_file(path) for path in artifacts},
        "target_values_read": False,
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device_index) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(subject_root / "DONE.json", done)
    print(json.dumps(done, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
