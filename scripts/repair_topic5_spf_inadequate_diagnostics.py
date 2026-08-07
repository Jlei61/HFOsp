#!/usr/bin/env python3
"""Refit only inadequate Round-3/Round-6 diagnostic shards.

The canonical v0.4 pilot remains untouched. This repair starts each affected
diagnostic model from its original deterministic initialization and tries a
small, pre-declared lower-learning-rate ladder. It refuses to aggregate until
every free-parameter fit has a converged adequacy verdict.
"""
from __future__ import annotations

import copy
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_spf_latent_dimension_sensitivity import (  # noqa: E402
    OUTPUT_ROOT as DIMENSION_ROOT,
    _aggregate as aggregate_dimension,
)
from scripts.run_topic5_spf_model_ladder import (  # noqa: E402
    _build,
    _model_seed,
    _score_repeated,
    _seed_everything,
    _subsample_chronological,
    _train_one,
)
from scripts.run_topic5_spf_nested_learning_curve import (  # noqa: E402
    OUTPUT_ROOT as LEARNING_ROOT,
    _aggregate as aggregate_learning,
    _nested_order,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    fit_static_scaffold_ml,
    load_subject_rank_events,
    sha256_file,
)

CONFIG_PATH = ROOT / "config/topic5_shared_propagation_field_v0_1.yaml"
PILOT_ROOT = (
    ROOT
    / "results/topic5_shared_propagation_field/development/ladder_pilot_v0_4"
)
LEARNING_RATE_FACTORS = (0.2, 0.04, 0.008)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _inadequate(payload: dict[str, Any]) -> list[str]:
    return [
        name
        for name, value in payload["models"].items()
        if not bool(value["training_adequacy"]["converged"])
        and value["training_adequacy"]["verdict"] != "NO_FREE_PARAMETERS"
    ]


def _refit(
    *,
    subject: str,
    seed: int,
    model_name: str,
    latent_dim: int,
    groups: np.ndarray,
    counts: np.ndarray,
    train: np.ndarray,
    monitor: np.ndarray,
    test: np.ndarray,
    scaffold: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_config = dict(config["model"])
    model_config["latent_dim"] = int(latent_dim)
    model_config["mixture_components"] = int(
        config["ladder"]["mixture_components"]
    )
    model_seed = _model_seed(int(seed), model_name)
    _seed_everything(model_seed)
    model = _build(model_name, groups.shape[1], scaffold, model_config)
    initial = copy.deepcopy(model.state_dict())
    attempts = []
    accepted = None
    for factor in LEARNING_RATE_FACTORS:
        training = dict(config["training"])
        training["learning_rate"] = float(
            config["training"]["learning_rate"]
        ) * float(factor)
        training["epochs"] = 800
        training["early_stopping_patience"] = 40
        training["lr_scheduler_patience"] = 15
        model.load_state_dict(initial)
        _seed_everything(model_seed)
        fitted = _train_one(
            model_name,
            model,
            groups,
            counts,
            train,
            monitor,
            device=torch.device("cpu"),
            training=training,
            evaluation=config["evaluation"],
            seed=model_seed,
        )
        attempts.append(
            {
                "learning_rate_factor": factor,
                "learning_rate": training["learning_rate"],
                "adequacy": fitted["adequacy"],
            }
        )
        if bool(fitted["adequacy"]["converged"]):
            accepted = fitted
            break
    if accepted is None:
        raise RuntimeError(
            f"{subject} seed={seed} {model_name} d={latent_dim}: "
            "lower-learning-rate repair did not converge"
        )
    model.load_state_dict(accepted["best_state"])
    model.eval()
    test_groups = torch.as_tensor(groups[test], dtype=torch.long)
    test_counts = torch.as_tensor(counts[test], dtype=torch.long)
    score = _score_repeated(
        model,
        test_groups,
        test_counts,
        prior_samples=64,
        importance_samples=64,
        repeats=2,
        seed=int(seed) + 211,
    )
    return (
        {
            "monitor_nll_per_event": accepted["adequacy"][
                "best_validation_nll"
            ],
            "nll_per_decision": score["nll_per_decision"],
            "nll_per_decision_mc_sd": score["nll_per_decision_mc_sd"],
            "prior_predictive_nll_per_decision": score[
                "prior_predictive_nll_per_decision"
            ],
            "prior_predictive_nll_per_decision_mc_sd": score[
                "prior_predictive_nll_per_decision_mc_sd"
            ],
            "estimator": score["estimator"],
            "training_adequacy": {
                **accepted["adequacy"],
                "rescue_used": True,
                "n_training_attempts": len(attempts),
                "primary_attempt_verdict": "INHERITED_INADEQUATE_DIAGNOSTIC",
            },
            "training_attempts": attempts,
            "training_elapsed_seconds": None,
        },
        attempts,
    )


def repair_learning(config: dict[str, Any]) -> int:
    repaired = 0
    for path_text in sorted(glob.glob(str(LEARNING_ROOT / "per_run/*.json"))):
        path = Path(path_text)
        payload = _read(path)
        names = _inadequate(payload)
        if not names:
            continue
        record = load_subject_rank_events(
            ROOT / config["data"]["dataset_dir"], payload["subject"]
        )
        train_pool, monitor, test = record.development_split(
            float(config["data"]["inner_validation_fraction"]),
            float(config["data"]["inner_test_fraction"]),
        )
        nested = _nested_order(
            payload["subject"],
            train_pool,
            int(config["ladder"]["max_train_events"]),
        )
        train = nested[: int(payload["n_train_events"])]
        monitor = _subsample_chronological(
            monitor, int(config["ladder"]["max_validation_events"])
        )
        test = _subsample_chronological(
            test, int(config["ladder"]["max_test_events"])
        )
        scaffold = fit_static_scaffold_ml(
            record.group_ids,
            record.group_count,
            train,
            steps=int(config["ladder"]["scaffold_steps"]),
            learning_rate=float(config["ladder"]["scaffold_learning_rate"]),
            seed=int(payload["seed"]),
            device=torch.device("cpu"),
        )
        for name in names:
            replacement, attempts = _refit(
                subject=payload["subject"],
                seed=int(payload["seed"]),
                model_name=name,
                latent_dim=int(config["model"]["latent_dim"]),
                groups=record.group_ids,
                counts=record.group_count,
                train=train,
                monitor=monitor,
                test=test,
                scaffold=scaffold,
                config=config,
            )
            payload["models"][name] = replacement
            payload.setdefault("adequacy_repairs", {})[name] = attempts
            repaired += 1
        _write(path, payload)
    return repaired


def repair_dimension(config: dict[str, Any]) -> int:
    repaired = 0
    for path_text in sorted(glob.glob(str(DIMENSION_ROOT / "per_run/*.json"))):
        path = Path(path_text)
        payload = _read(path)
        names = _inadequate(payload)
        if not names:
            continue
        record = load_subject_rank_events(
            ROOT / config["data"]["dataset_dir"], payload["subject"]
        )
        train, monitor, test = record.development_split(
            float(config["data"]["inner_validation_fraction"]),
            float(config["data"]["inner_test_fraction"]),
        )
        train = _subsample_chronological(
            train, int(config["ladder"]["max_train_events"])
        )
        monitor = _subsample_chronological(
            monitor, int(config["ladder"]["max_validation_events"])
        )
        test = _subsample_chronological(
            test, int(config["ladder"]["max_test_events"])
        )
        checkpoint = torch.load(
            ROOT / payload["static_scaffold_source_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        scaffold = checkpoint["static_scaffold_ml"].detach().cpu().numpy()
        for name in names:
            replacement, attempts = _refit(
                subject=payload["subject"],
                seed=int(payload["seed"]),
                model_name=name,
                latent_dim=int(payload["latent_dim"]),
                groups=record.group_ids,
                counts=record.group_count,
                train=train,
                monitor=monitor,
                test=test,
                scaffold=scaffold,
                config=config,
            )
            payload["models"][name] = replacement
            payload.setdefault("adequacy_repairs", {})[name] = attempts
            repaired += 1
        _write(path, payload)
    return repaired


def _assert_all_adequate(
    root: Path, *, expected_payloads: int
) -> list[dict[str, Any]]:
    payloads = [
        _read(Path(path))
        for path in sorted(glob.glob(str(root / "per_run/*.json")))
    ]
    if len(payloads) != int(expected_payloads):
        raise RuntimeError(
            f"{root}: expected {expected_payloads} diagnostic shards, "
            f"found {len(payloads)}"
        )
    bad = [
        (value["subject"], value["seed"], name)
        for value in payloads
        for name in _inadequate(value)
    ]
    if bad:
        raise RuntimeError(f"inadequate diagnostic fits remain: {bad}")
    return payloads


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    learning_repairs = repair_learning(config)
    dimension_repairs = repair_dimension(config)
    learning_payloads = _assert_all_adequate(
        LEARNING_ROOT, expected_payloads=6 * 3 * 6
    )
    dimension_payloads = _assert_all_adequate(
        DIMENSION_ROOT, expected_payloads=6 * 3 * 2
    )
    fractions = sorted({float(value["fraction"]) for value in learning_payloads})
    aggregate_learning(learning_payloads, fractions)
    aggregate_dimension(dimension_payloads)
    repair = {
        "status": "COMPLETE",
        "learning_repairs": learning_repairs,
        "dimension_repairs": dimension_repairs,
        "learning_rate_factors": LEARNING_RATE_FACTORS,
        "all_diagnostic_fits_adequate": True,
        "source_sha256": sha256_file(Path(__file__)),
    }
    target = (
        ROOT
        / "results/topic5_shared_propagation_field/development"
        / "multiround_review_2026-07-31"
        / "ADEQUACY_REPAIR_STATE.json"
    )
    _write(target, repair)
    for state_path in (
        LEARNING_ROOT / "ROUND_STATE.json",
        DIMENSION_ROOT / "ROUND_STATE.json",
    ):
        state = _read(state_path)
        state["adequacy_repair_state"] = str(target.relative_to(ROOT))
        state["all_fits_adequate_after_repair"] = True
        _write(state_path, state)


if __name__ == "__main__":
    main()
