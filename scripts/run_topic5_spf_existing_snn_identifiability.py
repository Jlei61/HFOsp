#!/usr/bin/env python3
"""System-identification test on existing SNN rank-event artifacts.

No simulator is imported or called. Models are fit only to paired-source/sink
rank events from SNN seeds 1--15, selected on seeds 16--18, and evaluated on
seeds 19--21. Direction labels and geometry are sealed during fitting and are
opened only for readout after checkpoint selection.

The source-only and sink-only families are an OOD conditional-reuse test:
their first rank, event length, and rank cardinalities are supplied to the
frozen paired-family model. This is not a blind prediction of a lesion from an
internal weight deletion, and therefore cannot by itself close the full G0
perturbation-prediction gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required; use the cuda_env environment") from exc

from scripts.run_topic5_spf_model_ladder import (  # noqa: E402
    _build,
    _model_seed,
    _score_repeated,
    _seed_everything,
)
from scripts.run_topic5_spf_nested_learning_curve import _fit_model  # noqa: E402
from src.sef_hfo_observation import endpoint_centroid_axis  # noqa: E402
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    sha256_file,
    validate_rank_event_arrays,
)

CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
INPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/snn_positive_control"
    / "existing_artifact_reuse"
)
OUTPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/snn_positive_control"
    / "existing_artifact_system_identification"
)
MODELS = (
    "m0_static",
    "m1_markov_phase",
    "m2_markov_mixture_phase",
    "m3_template",
    "m4_field",
    "m4_field_phase",
)
FIT_SEEDS = (20260731, 20260732, 20260733)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n")


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def _load_family(
    name: str, *, include_direction_readout: bool = True
) -> dict[str, np.ndarray]:
    path = INPUT_ROOT / f"rank_events_{name}.npz"
    with np.load(path, allow_pickle=False) as artifact:
        keys = {
            "contact_names",
            "event_group_ids",
            "event_group_count",
            "source_seed",
            "source_event_index",
        }
        if include_direction_readout:
            keys.update(
                {
                    "event_direction_sign",
                    "event_direction_sign_reported",
                }
            )
        missing = keys.difference(artifact.files)
        if missing:
            raise RuntimeError(
                f"{path}: missing required arrays {sorted(missing)}"
            )
        values = {key: np.asarray(artifact[key]) for key in sorted(keys)}
    values["artifact_sha256"] = np.asarray(sha256_file(path))
    validate_rank_event_arrays(
        values["event_group_ids"], values["event_group_count"]
    )
    return values


def _geometry() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    inventory = json.loads(
        (INPUT_ROOT / "existing_snn_artifact_inventory.json").read_text()
    )
    paired = next(
        family
        for family in inventory["families"]
        if family["family"] == "paired_source_sink"
    )
    path = ROOT / paired["files"][0]["figdata"]
    with np.load(path, allow_pickle=True) as artifact:
        coordinates = np.asarray(artifact["contacts"], dtype=float)
        names = [str(value) for value in artifact["names"]]
        axis = np.asarray(artifact["reg"].item()["axis_unit"], dtype=float)
    return coordinates, axis, names, sha256_file(path)


def _direction(
    group_ids: np.ndarray,
    coordinates: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    output = np.full(len(group_ids), np.nan, dtype=float)
    for index, ranks in enumerate(group_ids):
        vector = endpoint_centroid_axis(
            ranks,
            ranks >= 0,
            coordinates,
            k_dir=2,
            eps_deg=2.0,
        )
        if vector is not None:
            output[index] = float(np.sign(np.dot(vector, axis)))
    return output


def _split_by_source_seed(source_seed: np.ndarray) -> tuple[np.ndarray, ...]:
    train = np.flatnonzero(np.isin(source_seed, np.arange(1, 16)))
    monitor = np.flatnonzero(np.isin(source_seed, np.arange(16, 19)))
    test = np.flatnonzero(np.isin(source_seed, np.arange(19, 22)))
    if min(len(train), len(monitor), len(test)) == 0:
        raise RuntimeError("SNN source-seed split is empty")
    return train, monitor, test


def _batch(
    family: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(
            family["event_group_ids"][indices], dtype=torch.long
        ),
        torch.as_tensor(
            family["event_group_count"][indices], dtype=torch.long
        ),
    )


def _direction_summary(
    observed: np.ndarray,
    generated: list[np.ndarray],
    coordinates: np.ndarray,
    axis: np.ndarray,
) -> dict[str, Any]:
    observed_sign = _direction(observed, coordinates, axis)
    generated_sign = np.stack(
        [_direction(value, coordinates, axis) for value in generated], axis=0
    )
    observed_readable = np.isfinite(observed_sign)
    generated_readable = np.isfinite(generated_sign)
    observed_forward = float(np.mean(observed_sign[observed_readable] > 0))
    generated_forward = float(
        np.sum((generated_sign > 0) & generated_readable)
        / max(np.sum(generated_readable), 1)
    )
    event_forward_probability = np.divide(
        np.sum((generated_sign > 0) & generated_readable, axis=0),
        np.sum(generated_readable, axis=0),
        out=np.full(generated_sign.shape[1], np.nan, dtype=float),
        where=np.sum(generated_readable, axis=0) > 0,
    )
    calibration_valid = observed_readable & np.isfinite(
        event_forward_probability
    )
    brier = (
        float(
            np.mean(
                (
                    event_forward_probability[calibration_valid]
                    - (observed_sign[calibration_valid] > 0).astype(float)
                )
                ** 2
            )
        )
        if np.any(calibration_valid)
        else float("nan")
    )
    return {
        "n_events": int(len(observed)),
        "observed_readable_fraction": float(np.mean(observed_readable)),
        "generated_readable_fraction": float(np.mean(generated_readable)),
        "observed_forward_fraction": observed_forward,
        "generated_forward_fraction": generated_forward,
        "absolute_forward_fraction_error": abs(
            generated_forward - observed_forward
        ),
        "event_direction_brier": brier,
    }


def _worker(fit_seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    # Do not even decompress the direction arrays until all checkpoints have
    # been selected.  The fit dictionary contains only ranks, counts, contact
    # names, and source-seed provenance.
    paired = _load_family(
        "paired_source_sink", include_direction_readout=False
    )
    train, monitor, test = _split_by_source_seed(paired["source_seed"])
    _seed_everything(fit_seed)
    scaffold = fit_static_scaffold_ml(
        paired["event_group_ids"],
        paired["event_group_count"],
        train,
        steps=int(config["ladder"]["scaffold_steps"]),
        learning_rate=float(config["ladder"]["scaffold_learning_rate"]),
        seed=fit_seed,
        device=torch.device("cpu"),
    )
    model_config = dict(config["model"])
    model_config["mixture_components"] = int(
        config["ladder"]["mixture_components"]
    )
    # The SNN pool has only ~150 training events. Keeping the human-pilot
    # batch size (256) would yield one optimizer update per epoch and make the
    # nominal 400-epoch budget incomparable to the many-update human fits.
    # Use a small batch and a wider ceiling; checkpoint selection remains
    # monitor-based and the adequacy gate still requires a plateau.
    snn_training = dict(config["training"])
    snn_training.update(
        {
            "batch_events": 32,
            "epochs": 1600,
            "early_stopping_patience": 40,
            "lr_scheduler_patience": 15,
            "minimum_training_epochs": 40,
        }
    )
    trained = {}
    for name in MODELS:
        model_seed = _model_seed(fit_seed, name)
        _seed_everything(model_seed)
        model = _build(
            name,
            paired["event_group_ids"].shape[1],
            scaffold,
            model_config,
        )
        started = time.time()
        model, fitted = _fit_model(
            name,
            model,
            paired["event_group_ids"],
            paired["event_group_count"],
            train,
            monitor,
            device=torch.device("cpu"),
            training=snn_training,
            evaluation=config["evaluation"],
            seed=model_seed,
        )
        trained[name] = (model, fitted, time.time() - started)

    coordinates, axis, geometry_names, geometry_sha = _geometry()
    paired_readout = _load_family(
        "paired_source_sink", include_direction_readout=True
    )
    for key in (
        "contact_names",
        "event_group_ids",
        "event_group_count",
        "source_seed",
        "source_event_index",
    ):
        if not np.array_equal(paired[key], paired_readout[key]):
            raise RuntimeError(f"paired fit/readout reload drift: {key}")
    paired = paired_readout
    if geometry_names != [str(value) for value in paired["contact_names"]]:
        raise RuntimeError("SNN geometry/rank contact order mismatch")
    test_groups, test_counts = _batch(paired, test)
    families = {
        "paired_source_sink": (paired, test),
    }
    for family_name in ("source_only", "sink_only"):
        family = _load_family(family_name)
        family_test = np.flatnonzero(
            np.isin(family["source_seed"], np.arange(19, 22))
        )
        families[family_name] = (family, family_test)

    first_rank_shortcut = {}
    for family_name, (family, family_test) in families.items():
        family_train = np.flatnonzero(
            np.isin(family["source_seed"], np.arange(1, 16))
        )
        lookup: dict[bytes, list[float]] = {}
        for row, sign in zip(
            family["event_group_ids"][family_train],
            family["event_direction_sign"][family_train],
        ):
            if np.isfinite(sign):
                key = np.packbits(row == 0).tobytes()
                lookup.setdefault(key, []).append(float(sign))
        train_sign = family["event_direction_sign"][family_train]
        global_sign = (
            1.0 if float(np.nanmean(train_sign)) >= 0.0 else -1.0
        )
        predicted = []
        observed = []
        known = []
        for row, sign in zip(
            family["event_group_ids"][family_test],
            family["event_direction_sign"][family_test],
        ):
            if not np.isfinite(sign):
                continue
            key = np.packbits(row == 0).tobytes()
            predicted.append(
                float(np.sign(np.mean(lookup[key])))
                if key in lookup
                else global_sign
            )
            observed.append(float(sign))
            known.append(key in lookup)
        predicted_array = np.asarray(predicted, dtype=float)
        observed_array = np.asarray(observed, dtype=float)
        known_array = np.asarray(known, dtype=bool)
        first_rank_shortcut[family_name] = {
            "n_readable_test_events": int(len(observed_array)),
            "n_known_first_rank_test_events": int(np.sum(known_array)),
            "direction_accuracy": float(
                np.mean(predicted_array == observed_array)
            ),
            "known_first_rank_direction_accuracy": float(
                np.mean(
                    predicted_array[known_array] == observed_array[known_array]
                )
            )
            if np.any(known_array)
            else float("nan"),
            "n_train_first_rank_keys": int(len(lookup)),
            "role": (
                "post-selection shortcut diagnostic; direction labels were "
                "not available to any generative model"
            ),
        }
    results = {}
    states = {}
    for model_index, (name, (model, fitted, elapsed)) in enumerate(
        trained.items()
    ):
        score = _score_repeated(
            model,
            test_groups,
            test_counts,
            prior_samples=128,
            importance_samples=128,
            repeats=3,
            seed=fit_seed + 211,
        )
        family_results = {}
        for family_index, (family_name, (family, indices)) in enumerate(
            families.items()
        ):
            groups, counts = _batch(family, indices)
            generated = []
            for repeat in range(32):
                with torch.no_grad():
                    value = model.generate_conditioned(
                        groups,
                        counts,
                        seed=fit_seed
                        + 10001
                        + model_index * 100003
                        + family_index * 1009
                        + repeat,
                    )
                generated.append(value.cpu().numpy())
            family_results[family_name] = _direction_summary(
                groups.cpu().numpy(), generated, coordinates, axis
            )
        results[name] = {
            "paired_test_nll_per_decision": score["nll_per_decision"],
            "paired_test_prior_predictive_nll_per_decision": score[
                "prior_predictive_nll_per_decision"
            ],
            "training_adequacy": fitted["adequacy"],
            "training_elapsed_seconds": elapsed,
            "direction_readout": family_results,
        }
        states[name] = {
            "model_state": {
                key: value.detach().cpu()
                for key, value in fitted["best_state"].items()
            },
            "training_adequacy": fitted["adequacy"],
        }
    checkpoint_path = OUTPUT_ROOT / "per_run" / f"seed{fit_seed}_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "topic5_spf_existing_snn_identifiability_v0_1",
            "fit_seed": int(fit_seed),
            "static_scaffold_ml": torch.as_tensor(scaffold),
            "models": states,
            "train_source_seeds": list(range(1, 16)),
            "monitor_source_seeds": list(range(16, 19)),
            "test_source_seeds": list(range(19, 22)),
        },
        checkpoint_path,
    )
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_spf_existing_snn_identifiability_v0_1",
        "fit_seed": int(fit_seed),
        "n_train_events": int(len(train)),
        "n_monitor_events": int(len(monitor)),
        "n_test_events": int(len(test)),
        "train_indices_sha256": _indices_sha256(train),
        "monitor_indices_sha256": _indices_sha256(monitor),
        "test_indices_sha256": _indices_sha256(test),
        "direction_labels_read_during_fit_or_selection": False,
        "geometry_read_during_fit_or_selection": False,
        "simulator_called": False,
        "direction_readout_contract": (
            "endpoint centroid, explicit k_dir=2, eps_deg=2.0"
        ),
        "snn_training_override": {
            key: snn_training[key]
            for key in (
                "batch_events",
                "epochs",
                "early_stopping_patience",
                "lr_scheduler_patience",
                "minimum_training_epochs",
            )
        },
        "geometry_artifact_sha256": geometry_sha,
        "input_rank_event_sha256": {
            family_name: str(family["artifact_sha256"])
            for family_name, (family, _) in families.items()
        },
        "first_rank_direction_shortcut": first_rank_shortcut,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "models": results,
    }
    _write_json(OUTPUT_ROOT / "per_run" / f"seed{fit_seed}.json", payload)
    return payload


def _aggregate(outputs: list[dict[str, Any]]) -> None:
    rows = []
    for output in outputs:
        for name, model in output["models"].items():
            for family, direction in model["direction_readout"].items():
                rows.append(
                    {
                        "fit_seed": output["fit_seed"],
                        "model": name,
                        "evaluation_family": family,
                        "paired_test_nll_per_decision": model[
                            "paired_test_nll_per_decision"
                        ],
                        "paired_test_prior_predictive_nll_per_decision": model[
                            "paired_test_prior_predictive_nll_per_decision"
                        ],
                        "training_verdict": model["training_adequacy"][
                            "verdict"
                        ],
                        **direction,
                    }
                )
    with (OUTPUT_ROOT / "snn_system_identification_runs.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = []
    grouping = lambda row: (row["model"], row["evaluation_family"])
    for key, values in itertools.groupby(sorted(rows, key=grouping), key=grouping):
        selected = list(values)
        means.append(
            {
                "model": key[0],
                "evaluation_family": key[1],
                "paired_test_nll_per_decision_mean": float(
                    np.mean(
                        [
                            row["paired_test_nll_per_decision"]
                            for row in selected
                        ]
                    )
                ),
                "paired_test_prior_predictive_nll_per_decision_mean": float(
                    np.mean(
                        [
                            row[
                                "paired_test_prior_predictive_nll_per_decision"
                            ]
                            for row in selected
                        ]
                    )
                ),
                "generated_forward_fraction_mean": float(
                    np.mean(
                        [row["generated_forward_fraction"] for row in selected]
                    )
                ),
                "absolute_forward_fraction_error_mean": float(
                    np.mean(
                        [
                            row["absolute_forward_fraction_error"]
                            for row in selected
                        ]
                    )
                ),
                "event_direction_brier_mean": float(
                    np.mean([row["event_direction_brier"] for row in selected])
                ),
                "generated_readable_fraction_mean": float(
                    np.mean(
                        [
                            row["generated_readable_fraction"]
                            for row in selected
                        ]
                    )
                ),
                "n_fit_seeds": len(selected),
            }
        )
    with (OUTPUT_ROOT / "snn_system_identification_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    _write_json(
        OUTPUT_ROOT / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 5,
            "question": (
                "Can a model fit to held-in SNN paired-repertoire seeds recover "
                "held-out direction occupancy and reuse that structure under "
                "source-only/sink-only conditional inputs?"
            ),
            "n_fit_seeds": len(outputs),
            "simulator_called": False,
            "direction_labels_read_during_fit_or_selection": False,
            "geometry_read_during_fit_or_selection": False,
            "test_source_seeds": [19, 20, 21],
            "input_rank_event_sha256": {
                family: sha256_file(INPUT_ROOT / f"rank_events_{family}.npz")
                for family in (
                    "paired_source_sink",
                    "source_only",
                    "sink_only",
                )
            },
            "interpretation_limit": (
                "source/sink evaluation supplies the altered condition's "
                "first rank and nuisance schedule; it is OOD conditional reuse, "
                "not a blind internal-lesion forecast, so full G0 remains open"
            ),
            "g0_current_model_status": (
                "NOT_PASSED_UNLESS_M4_OUTPERFORMS_SIMPLE_CONTROLS_ON_THE_"
                "KNOWN_STRUCTURE_PAIRED_FAMILY"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--fit-seeds", type=int, nargs="+", default=FIT_SEEDS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_seeds = sorted({int(value) for value in args.fit_seeds})
    outputs = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(_worker, seed): seed for seed in fit_seeds
        }
        for future in as_completed(futures):
            output = future.result()
            outputs.append(output)
            print(f"complete SNN fit seed={output['fit_seed']}", flush=True)
    _aggregate(outputs)


if __name__ == "__main__":
    main()
