#!/usr/bin/env python3
"""Run the pre-specified d={2,4,6} SPF latent-dimension sensitivity.

The d=4 row is reused from the frozen v0.4 pilot. Only d=2 and d=6 are newly
fit. Data splits, static scaffold, optimizer contract, and likelihood estimator
are held fixed. This is a robustness check, not an architecture search.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
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
    _subsample_chronological,
)
from scripts.run_topic5_spf_nested_learning_curve import (  # noqa: E402
    _fit_model,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    load_subject_rank_events,
    sha256_file,
)

CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
PILOT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4"
)
OUTPUT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development"
    / "multiround_review_2026-07-31/round6_latent_dimension_sensitivity"
)
MODELS = ("m3_template", "m4_field", "m4_field_phase")


def _write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            return item.item()
        if isinstance(item, dict):
            return {str(key): convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, ensure_ascii=False) + "\n")


def _pilot_runs() -> list[Path]:
    values = sorted(PILOT_ROOT.glob("*_seed*/summary.json"))
    if len(values) != 18:
        raise RuntimeError("frozen v0.4 pilot is incomplete")
    return [path.parent for path in values]


def _worker(run_dir_text: str, latent_dim: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    run_dir = Path(run_dir_text)
    summary = json.loads((run_dir / "summary.json").read_text())
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if checkpoint["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("v0.4 checkpoint/config fingerprint drift")
    record = load_subject_rank_events(
        ROOT / config["data"]["dataset_dir"], summary["subject"]
    )
    train, monitor, test = record.development_split(
        float(config["data"]["inner_validation_fraction"]),
        float(config["data"]["inner_test_fraction"]),
    )
    ladder = config["ladder"]
    train = _subsample_chronological(train, int(ladder["max_train_events"]))
    monitor = _subsample_chronological(
        monitor, int(ladder["max_validation_events"])
    )
    test = _subsample_chronological(test, int(ladder["max_test_events"]))
    if np.intersect1d(
        np.r_[train, monitor, test], record.old_heldout20_indices
    ).size:
        raise RuntimeError("old heldout20 entered dimension sensitivity")
    scaffold = checkpoint["static_scaffold_ml"].detach().cpu().numpy()
    model_config = dict(config["model"])
    model_config["latent_dim"] = int(latent_dim)
    model_config["mixture_components"] = int(ladder["mixture_components"])
    training = dict(config["training"])
    evaluation = dict(config["evaluation"])
    trained: dict[str, tuple[torch.nn.Module, dict[str, Any], float]] = {}
    for name in MODELS:
        model_seed = _model_seed(int(summary["seed"]), name)
        _seed_everything(model_seed)
        model = _build(
            name, len(record.contact_names), scaffold, model_config
        )
        started = time.time()
        model, fitted = _fit_model(
            name,
            model,
            record.group_ids,
            record.group_count,
            train,
            monitor,
            device=torch.device("cpu"),
            training=training,
            evaluation=evaluation,
            seed=model_seed,
        )
        trained[name] = (model, fitted, time.time() - started)
    test_groups = torch.as_tensor(record.group_ids[test], dtype=torch.long)
    test_counts = torch.as_tensor(record.group_count[test], dtype=torch.long)
    results = {}
    for name, (model, fitted, elapsed) in trained.items():
        score = _score_repeated(
            model,
            test_groups,
            test_counts,
            prior_samples=64,
            importance_samples=64,
            repeats=2,
            seed=int(summary["seed"]) + 211,
        )
        results[name] = {
            "monitor_nll_per_event": fitted["adequacy"][
                "best_validation_nll"
            ],
            "nll_per_decision": score["nll_per_decision"],
            "prior_predictive_nll_per_decision": score[
                "prior_predictive_nll_per_decision"
            ],
            "training_adequacy": fitted["adequacy"],
            "training_elapsed_seconds": elapsed,
        }
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_spf_latent_dimension_sensitivity_v0_1",
        "subject": summary["subject"],
        "seed": int(summary["seed"]),
        "latent_dim": int(latent_dim),
        "input_sha256": record.input_sha256,
        "config_sha256": sha256_file(CONFIG_PATH),
        "static_scaffold_source_checkpoint": str(
            (run_dir / "checkpoint.pt").relative_to(ROOT)
        ),
        "old_heldout20_scored": False,
        "models": results,
    }
    _write_json(
        OUTPUT_ROOT
        / "per_run"
        / f"{summary['subject']}_seed{summary['seed']}_d{latent_dim}.json",
        payload,
    )
    return payload


def _d4_rows() -> list[dict[str, Any]]:
    rows = []
    for run_dir in _pilot_runs():
        summary = json.loads((run_dir / "summary.json").read_text())
        for name in MODELS:
            model = summary["models"][name]
            rows.append(
                {
                    "subject": summary["subject"],
                    "seed": int(summary["seed"]),
                    "latent_dim": 4,
                    "model": name,
                    "monitor_nll_per_event": model["training_adequacy"][
                        "best_validation_nll"
                    ],
                    "nll_per_decision": model[
                        "development_test_nll_per_decision"
                    ],
                    "prior_predictive_nll_per_decision": model[
                        "prior_predictive_nll_per_decision"
                    ],
                    "training_verdict": model["training_adequacy"]["verdict"],
                    "source": "reused_frozen_v0_4",
                }
            )
    return rows


def _aggregate(outputs: list[dict[str, Any]]) -> None:
    rows = _d4_rows()
    for output in outputs:
        for name, model in output["models"].items():
            rows.append(
                {
                    "subject": output["subject"],
                    "seed": output["seed"],
                    "latent_dim": output["latent_dim"],
                    "model": name,
                    "monitor_nll_per_event": model["monitor_nll_per_event"],
                    "nll_per_decision": model["nll_per_decision"],
                    "prior_predictive_nll_per_decision": model[
                        "prior_predictive_nll_per_decision"
                    ],
                    "training_verdict": model["training_adequacy"]["verdict"],
                    "source": "new_sensitivity_fit",
                }
            )
    with (OUTPUT_ROOT / "dimension_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    patient_rows = []
    grouping = lambda row: (row["subject"], row["latent_dim"], row["model"])
    for key, values in itertools.groupby(sorted(rows, key=grouping), key=grouping):
        selected = list(values)
        patient_rows.append(
            {
                "subject": key[0],
                "latent_dim": key[1],
                "model": key[2],
                "monitor_nll_per_event_mean": float(
                    np.mean([row["monitor_nll_per_event"] for row in selected])
                ),
                "nll_per_decision_mean": float(
                    np.mean([row["nll_per_decision"] for row in selected])
                ),
                "prior_predictive_nll_per_decision_mean": float(
                    np.mean(
                        [
                            row["prior_predictive_nll_per_decision"]
                            for row in selected
                        ]
                    )
                ),
                "n_converged": int(
                    np.sum(
                        [
                            row["training_verdict"]
                            in ("CONVERGED", "NO_FREE_PARAMETERS")
                            for row in selected
                        ]
                    )
                ),
                "n_seeds": len(selected),
            }
        )
    with (OUTPUT_ROOT / "dimension_patient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(patient_rows[0]))
        writer.writeheader()
        writer.writerows(patient_rows)
    lookup = {
        (row["subject"], row["latent_dim"], row["model"]): row
        for row in patient_rows
    }
    contrasts = []
    for subject in sorted({row["subject"] for row in patient_rows}):
        for dimension in (2, 4, 6):
            for left, right, label in (
                ("m4_field", "m3_template", "m4_minus_m3"),
                (
                    "m4_field_phase",
                    "m3_template",
                    "m4phase_minus_m3",
                ),
            ):
                left_row = lookup[(subject, dimension, left)]
                right_row = lookup[(subject, dimension, right)]
                contrasts.append(
                    {
                        "subject": subject,
                        "latent_dim": dimension,
                        "comparison": label,
                        "delta_nll_per_decision": (
                            left_row["nll_per_decision_mean"]
                            - right_row["nll_per_decision_mean"]
                        ),
                        "delta_prior_predictive_nll_per_decision": (
                            left_row["prior_predictive_nll_per_decision_mean"]
                            - right_row[
                                "prior_predictive_nll_per_decision_mean"
                            ]
                        ),
                    }
                )
    with (OUTPUT_ROOT / "dimension_contrasts.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    _write_json(
        OUTPUT_ROOT / "ROUND_STATE.json",
        {
            "status": "COMPLETE",
            "round": 6,
            "question": (
                "Is the template-versus-field ordering robust across the "
                "pre-specified latent dimensions d={2,4,6}?"
            ),
            "dimensions": [2, 4, 6],
            "d4_source": "reused frozen v0.4 pilot",
            "new_dimensions": [2, 6],
            "n_new_jobs": len(outputs),
            "old_heldout20_scored": False,
            "interpretation_limit": (
                "pre-specified robustness sensitivity, not architecture search"
            ),
            "source_sha256": sha256_file(Path(__file__)),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dimensions", type=int, nargs="+", default=(2, 6))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dimensions = sorted({int(value) for value in args.dimensions})
    if not dimensions or any(value not in (2, 6) for value in dimensions):
        raise ValueError("only the missing pre-specified d=2/d=6 fits are allowed")
    jobs = list(itertools.product(_pilot_runs(), dimensions))
    outputs = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(_worker, str(run_dir), dimension): (
                run_dir,
                dimension,
            )
            for run_dir, dimension in jobs
        }
        for future in as_completed(futures):
            payload = future.result()
            outputs.append(payload)
            print(
                f"complete {payload['subject']} seed={payload['seed']} "
                f"d={payload['latent_dim']}",
                flush=True,
            )
    _aggregate(outputs)


if __name__ == "__main__":
    main()
