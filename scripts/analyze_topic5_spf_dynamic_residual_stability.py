#!/usr/bin/env python3
"""Subtract the static M0 response before assessing observable seed stability.

The first Round-4 analysis intentionally used the full observable response. A
high correlation there can be inherited from the shared participation scaffold
rather than a stable learned dynamic residual. This diagnostic repeats the
same future-blind rollouts, subtracts each run's own M0 response, and compares
only the remaining response across optimization seeds.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_spf_multiround import (  # noqa: E402
    OUTPUT_ROOT,
    RESPONSE_MODELS,
    _corr,
    _first_key,
    _load_context,
    _load_model,
    _response_vector,
    _run_dirs,
)
from src.topic5_shared_propagation_field import (  # noqa: E402
    MarkovMixtureModel,
    sha256_file,
)

TARGET = OUTPUT_ROOT / "round4b_dynamic_residual_seed_stability"
ROLLOUT_REPEATS = 12
MINIMUM_TRAIN_SUPPORT = 20


def _m0(context: dict[str, Any]) -> MarkovMixtureModel:
    checkpoint = context["checkpoint"]
    scaffold = checkpoint["static_scaffold_ml"].detach().cpu().numpy()
    model = MarkovMixtureModel(
        len(context["record"].contact_names),
        scaffold,
        n_components=1,
        use_transition=False,
    )
    model.load_state_dict(checkpoint["models"]["m0_static"]["model_state"])
    model.eval()
    return model


def _worker(run_dir_text: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    context = _load_context(Path(run_dir_text))
    record = context["record"]
    train = context["train"]
    test = context["test"]
    test_groups = record.group_ids[test]
    train_support: dict[bytes, int] = {}
    for row in record.group_ids[train]:
        key = _first_key(row)
        train_support[key] = train_support.get(key, 0) + 1
    strata: dict[bytes, list[int]] = {}
    for index, row in enumerate(test_groups):
        key = _first_key(row)
        if train_support.get(key, 0) >= MINIMUM_TRAIN_SUPPORT:
            strata.setdefault(key, []).append(index)
    strata = {key: value for key, value in strata.items() if len(value) >= 5}
    if not strata:
        raise RuntimeError(f"{record.subject}: no supported response stratum")
    groups = context["groups"]
    counts = context["counts"]
    models: dict[str, torch.nn.Module] = {"m0_static": _m0(context)}
    models.update({name: _load_model(context, name) for name in RESPONSE_MODELS})
    vectors: dict[str, dict[str, np.ndarray]] = {}
    fidelity_rows = []
    for model_index, (name, model) in enumerate(models.items()):
        generated = []
        for repeat in range(ROLLOUT_REPEATS):
            with torch.no_grad():
                value = model.generate_conditioned(
                    groups,
                    counts,
                    seed=int(context["summary"]["seed"])
                    + 32001
                    + model_index * 10007
                    + repeat * 1009,
                )
            generated.append(value.cpu().numpy())
        vectors[name] = {}
        for stratum_index, (key, indices) in enumerate(
            sorted(strata.items(), key=lambda item: item[0])
        ):
            label = f"stratum_{stratum_index:03d}"
            index = np.asarray(indices, dtype=int)
            vector, _ = _response_vector(
                np.concatenate([value[index] for value in generated], axis=0)
            )
            vectors[name][label] = vector
    residuals = {}
    for name in RESPONSE_MODELS:
        residuals[name] = {}
        for stratum, vector in vectors[name].items():
            residual = vector - vectors["m0_static"][stratum]
            residuals[name][stratum] = residual
            indices = np.asarray(
                list(sorted(strata.items(), key=lambda item: item[0]))[
                    int(stratum.rsplit("_", 1)[1])
                ][1],
                dtype=int,
            )
            observed, _ = _response_vector(test_groups[indices])
            observed_residual = observed - vectors["m0_static"][stratum]
            correlation, n_features = _corr(residual, observed_residual)
            valid = np.isfinite(residual) & np.isfinite(observed_residual)
            fidelity_rows.append(
                {
                    "subject": record.subject,
                    "seed": int(context["summary"]["seed"]),
                    "model": name,
                    "stratum": stratum,
                    "train_support": int(
                        train_support[
                            list(
                                sorted(strata.items(), key=lambda item: item[0])
                            )[int(stratum.rsplit("_", 1)[1])][0]
                        ]
                    ),
                    "test_events": int(len(indices)),
                    "dynamic_residual_correlation_to_observed": correlation,
                    "dynamic_residual_mae_to_observed": float(
                        np.mean(
                            np.abs(residual[valid] - observed_residual[valid])
                        )
                    )
                    if np.any(valid)
                    else float("nan"),
                    "dynamic_residual_rms": float(
                        np.sqrt(np.nanmean(residual**2))
                    ),
                    "n_response_features": n_features,
                }
            )
    return {
        "subject": record.subject,
        "seed": int(context["summary"]["seed"]),
        "residuals": residuals,
        "fidelity": fidelity_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    outputs = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_worker, str(path)): path for path in _run_dirs()
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    fidelity = [row for item in outputs for row in item["fidelity"]]
    pair_rows = []
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for item in outputs:
        by_subject.setdefault(item["subject"], []).append(item)
    for subject, runs in sorted(by_subject.items()):
        for name in RESPONSE_MODELS:
            for left, right in itertools.combinations(runs, 2):
                values = []
                common = sorted(
                    set(left["residuals"][name]).intersection(
                        right["residuals"][name]
                    )
                )
                for stratum in common:
                    correlation, _ = _corr(
                        left["residuals"][name][stratum],
                        right["residuals"][name][stratum],
                    )
                    if np.isfinite(correlation):
                        values.append(correlation)
                pair_rows.append(
                    {
                        "subject": subject,
                        "model": name,
                        "seed_left": left["seed"],
                        "seed_right": right["seed"],
                        "n_common_strata": len(common),
                        "mean_dynamic_residual_correlation": float(
                            np.mean(values)
                        )
                        if values
                        else float("nan"),
                        "min_dynamic_residual_correlation": float(np.min(values))
                        if values
                        else float("nan"),
                    }
                )
    _write_csv(TARGET / "dynamic_residual_fidelity.csv", fidelity)
    _write_csv(TARGET / "dynamic_residual_seed_pairs.csv", pair_rows)
    state = {
        "status": "COMPLETE",
        "round": "4b",
        "question": (
            "Does observable seed stability survive subtraction of each "
            "checkpoint's static M0 response?"
        ),
        "n_runs": len(outputs),
        "rollout_repeats": ROLLOUT_REPEATS,
        "minimum_train_support": MINIMUM_TRAIN_SUPPORT,
        "old_heldout20_scored": False,
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            "scripts/analyze_topic5_spf_multiround.py": sha256_file(
                ROOT / "scripts/analyze_topic5_spf_multiround.py"
            ),
        },
    }
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "ROUND_STATE.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
