#!/usr/bin/env python3
"""D1 audit of existing SIG v2 artifacts without fitting a new generator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_propagation_field import (  # noqa: E402
    load_subject_rank_events,
    sha256_file,
    suffix_log_likelihood,
)
from src.topic5_stable_interaction_graph import (  # noqa: E402
    MatchedPhaseMarkovMixtureModel,
    cardinality_schedule,
)


DEVELOPMENT = ROOT / "results/topic5_stable_interaction_graph/development"
SIG_ROOT = DEVELOPMENT / "human_graph_increment_pilot_v0_3_provenance"
LADDER_ROOT = (
    DEVELOPMENT / "human_matched_baseline_ladder_v0_2_training_adequacy"
)
DATASET_DIR = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUTPUT = DEVELOPMENT / "v2_1_existing_artifact_audit"
MODELS = (
    "m1_markov_matched_phase",
    "m2_mixture_matched_phase",
    "m3_latent_template",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _subsample(indices: np.ndarray, limit: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(limit):
        return values
    return values[np.linspace(0, len(values) - 1, int(limit)).astype(int)]


def _first_key(event: np.ndarray) -> str:
    packed = np.packbits(np.asarray(event == 0, dtype=np.uint8))
    return hashlib.sha1(packed.tobytes()).hexdigest()[:16]


def _suffix_signature(event: np.ndarray) -> bytes:
    value = np.asarray(event, dtype=np.int16).copy()
    value[value == 0] = -1
    value[value > 0] -= 1
    return value.tobytes()


def _entropy(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    values = values[values > 0]
    if not len(values):
        return float("nan")
    probability = values / values.sum()
    return float(-np.sum(probability * np.log2(probability)))


def _event_vector(event: np.ndarray, count: int) -> np.ndarray:
    value = np.full(event.shape, 1.25, dtype=np.float32)
    present = event >= 0
    value[present] = event[present] / max(int(count) - 1, 1)
    return value


def _pair_distance(
    events: np.ndarray, counts: np.ndarray, indices: np.ndarray, *, seed: int
) -> float:
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    if len(indices) > 256:
        indices = np.sort(rng.choice(indices, 256, replace=False))
    values = np.stack([_event_vector(events[i], counts[i]) for i in indices])
    left = rng.integers(0, len(values), size=512)
    right = rng.integers(0, len(values), size=512)
    valid = left != right
    return float(np.mean(np.abs(values[left[valid]] - values[right[valid]])))


def _within_start_metrics(
    events: np.ndarray, counts: np.ndarray, *, seed: int, min_group: int = 5
) -> dict[str, float | int]:
    keys = np.asarray([_first_key(event) for event in events])
    rows = []
    for ordinal, key in enumerate(sorted(set(keys))):
        indices = np.flatnonzero(keys == key)
        if len(indices) < int(min_group):
            continue
        signatures = [_suffix_signature(events[index]) for index in indices]
        _, signature_counts = np.unique(signatures, return_counts=True)
        rows.append(
            {
                "weight": len(indices),
                "unique_fraction": len(signature_counts) / len(indices),
                "entropy_bits": _entropy(signature_counts),
                "pair_distance": _pair_distance(
                    events, counts, indices, seed=seed + ordinal * 997
                ),
            }
        )
    if not rows:
        return {
            "n_start_groups": 0,
            "n_events": 0,
            "unique_fraction": float("nan"),
            "entropy_bits": float("nan"),
            "pair_distance": float("nan"),
        }
    weights = np.asarray([row["weight"] for row in rows], dtype=float)
    return {
        "n_start_groups": len(rows),
        "n_events": int(weights.sum()),
        **{
            name: float(
                np.average(
                    np.asarray([row[name] for row in rows], dtype=float),
                    weights=weights,
                )
            )
            for name in ("unique_fraction", "entropy_bits", "pair_distance")
        },
    }


def _component_event_scores(
    model: MatchedPhaseMarkovMixtureModel,
    groups: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    values = []
    for component in range(model.n_components):
        def logit_fn(step, previous, active, component=component):
            return model.component_logits(
                component, previous, step=step, group_count=counts
            )

        values.append(
            suffix_log_likelihood(logit_fn, groups, counts)[
                "event_log_probability"
            ]
        )
    return torch.stack(values, dim=1) + torch.log_softmax(
        model.mixture_logit, dim=0
    )[None, :]


def _route_features(
    groups: np.ndarray, counts: np.ndarray, *, include_schedule: bool
) -> np.ndarray:
    first = (groups == 0).astype(np.float32)
    normalized_count = (
        counts.astype(np.float32) / max(float(np.max(counts)), 1.0)
    )[:, None]
    output = [first, normalized_count]
    if include_schedule:
        schedule = cardinality_schedule(groups, counts).astype(np.float32)
        schedule /= max(float(np.max(schedule)), 1.0)
        output.append(schedule)
    return np.concatenate(output, axis=1)


def _route_predictability(
    train_groups: np.ndarray,
    train_counts: np.ndarray,
    test_groups: np.ndarray,
    test_counts: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    *,
    include_schedule: bool,
    seed: int,
) -> dict[str, Any]:
    classes = np.unique(train_labels)
    occupancy = np.bincount(train_labels, minlength=3)
    result: dict[str, Any] = {
        "train_occupancy": occupancy.tolist(),
        "train_occupancy_entropy_bits": _entropy(occupancy),
        "n_train_classes": int(len(classes)),
    }
    if len(classes) < 2:
        return {
            **result,
            "status": "ONE_POSTERIOR_ROUTE_ONLY",
            "balanced_accuracy": float("nan"),
            "cross_entropy": float("nan"),
        }
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=int(seed),
        ),
    )
    classifier.fit(
        _route_features(
            train_groups, train_counts, include_schedule=include_schedule
        ),
        train_labels,
    )
    prediction = classifier.predict(
        _route_features(test_groups, test_counts, include_schedule=include_schedule)
    )
    probability = classifier.predict_proba(
        _route_features(test_groups, test_counts, include_schedule=include_schedule)
    )
    return {
        **result,
        "status": "EVALUATED",
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, prediction)
        ),
        "cross_entropy": float(
            log_loss(test_labels, probability, labels=classifier.classes_)
        ),
    }


def _load_m2(
    subject: str, seed: int, contacts: int
) -> MatchedPhaseMarkovMixtureModel:
    run_dir = LADDER_ROOT / "per_run" / subject / f"seed_{seed}"
    checkpoint = torch.load(
        run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["models"]["m2_mixture_matched_phase"]["state_dict"]
    model = MatchedPhaseMarkovMixtureModel(
        contacts, state["static_bias"].numpy(), n_components=3
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _audit_subject(subject: str, seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = load_subject_rank_events(DATASET_DIR, subject)
    train, validation, test = record.development_split(0.15, 0.15)
    train = _subsample(train, 9600)
    validation = _subsample(validation, 2048)
    test = _subsample(test, 2048)
    if np.intersect1d(np.r_[train, validation, test], record.old_heldout20_indices).size:
        raise RuntimeError(f"{subject}: outer heldout20 leakage")

    validation_scores = {name: [] for name in MODELS}
    generation: dict[str, list[dict[str, Any]]] = {
        "sig1_feedback_graph": [],
        **{name: [] for name in MODELS},
    }
    route_rows = []
    observed_reference = None
    observed_counts = None
    for seed in seeds:
        baseline_run = LADDER_ROOT / "per_run" / subject / f"seed_{seed}"
        baseline_summary = json.loads((baseline_run / "summary.json").read_text())
        for name in MODELS:
            validation_scores[name].append(
                baseline_summary["models"][name]["training_adequacy"][
                    "best_validation_nll"
                ]
            )
        with np.load(baseline_run / "conditioned_generation.npz") as artifact:
            observed = np.asarray(artifact["observed_group_ids"], dtype=np.int16)
            counts = np.asarray(artifact["observed_group_count"], dtype=np.int16)
            if observed_reference is None:
                observed_reference, observed_counts = observed.copy(), counts.copy()
            elif not (
                np.array_equal(observed_reference, observed)
                and np.array_equal(observed_counts, counts)
            ):
                raise RuntimeError(f"{subject}: baseline test arrays differ by seed")
            for name in MODELS:
                generation[name].append(
                    _within_start_metrics(
                        np.asarray(artifact[name], dtype=np.int16),
                        counts,
                        seed=seed,
                    )
                )
        sig_run = SIG_ROOT / "per_run" / subject / f"seed_{seed}"
        with np.load(sig_run / "conditioned_generation.npz") as artifact:
            if not (
                np.array_equal(observed_reference, artifact["observed_group_ids"])
                and np.array_equal(observed_counts, artifact["observed_group_count"])
            ):
                raise RuntimeError(f"{subject}: SIG and ladder test arrays differ")
            generation["sig1_feedback_graph"].append(
                _within_start_metrics(
                    np.asarray(artifact["sig1_feedback_graph"], dtype=np.int16),
                    observed_counts,
                    seed=seed,
                )
            )

        model = _load_m2(subject, seed, len(record.contact_names))
        train_groups = torch.as_tensor(record.group_ids[train], dtype=torch.long)
        train_counts = torch.as_tensor(record.group_count[train], dtype=torch.long)
        test_groups = torch.as_tensor(record.group_ids[test], dtype=torch.long)
        test_counts = torch.as_tensor(record.group_count[test], dtype=torch.long)
        with torch.no_grad():
            train_labels = _component_event_scores(
                model, train_groups, train_counts
            ).argmax(1).numpy()
            test_labels = _component_event_scores(
                model, test_groups, test_counts
            ).argmax(1).numpy()
        basic = _route_predictability(
            record.group_ids[train], record.group_count[train],
            record.group_ids[test], record.group_count[test],
            train_labels, test_labels, include_schedule=False, seed=seed,
        )
        full = _route_predictability(
            record.group_ids[train], record.group_count[train],
            record.group_ids[test], record.group_count[test],
            train_labels, test_labels, include_schedule=True, seed=seed,
        )
        route_rows.append(
            {
                "subject": subject,
                "fit_seed": seed,
                "basic_balanced_accuracy": basic["balanced_accuracy"],
                "full_balanced_accuracy": full["balanced_accuracy"],
                "schedule_increment_balanced_accuracy": (
                    full["balanced_accuracy"] - basic["balanced_accuracy"]
                ),
                "basic_cross_entropy": basic["cross_entropy"],
                "full_cross_entropy": full["cross_entropy"],
                "schedule_gain_cross_entropy": (
                    basic["cross_entropy"] - full["cross_entropy"]
                ),
                "route_occupancy_entropy_bits": basic[
                    "train_occupancy_entropy_bits"
                ],
                "train_occupancy": "|".join(map(str, basic["train_occupancy"])),
                "status": basic["status"],
            }
        )

    assert observed_reference is not None and observed_counts is not None
    observed_diversity = _within_start_metrics(
        observed_reference, observed_counts, seed=20260731
    )
    selected = min(
        MODELS,
        key=lambda name: float(np.median(validation_scores[name])),
    )
    diversity = {"observed": observed_diversity}
    for name, rows in generation.items():
        diversity[name] = {
            key: float(np.nanmedian([row[key] for row in rows]))
            for key in ("n_start_groups", "n_events", "unique_fraction", "entropy_bits", "pair_distance")
        }
        for metric in ("unique_fraction", "entropy_bits", "pair_distance"):
            real = float(observed_diversity[metric])
            diversity[name][f"{metric}_ratio_to_real"] = (
                float(diversity[name][metric]) / real if real > 0 else float("nan")
            )
    route_medians = {
        key: float(np.nanmedian([row[key] for row in route_rows]))
        for key in (
            "basic_balanced_accuracy",
            "full_balanced_accuracy",
            "schedule_increment_balanced_accuracy",
            "basic_cross_entropy",
            "full_cross_entropy",
            "schedule_gain_cross_entropy",
            "route_occupancy_entropy_bits",
        )
    }
    return (
        {
            "subject": subject,
            "n_contacts": len(record.contact_names),
            "n_train_events": len(train),
            "n_validation_events": len(validation),
            "n_test_events": len(test),
            "validation_selected_baseline": selected,
            **{
                f"validation_median_{name}": float(
                    np.median(validation_scores[name])
                )
                for name in MODELS
            },
            "diversity": diversity,
            "m2_route_predictability": route_medians,
        },
        route_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ladder = json.loads((LADDER_ROOT / "MATCHED_BASELINE_LADDER.json").read_text())
    sig = json.loads((SIG_ROOT / "HUMAN_GRAPH_INCREMENT_PILOT.json").read_text())
    subjects = [row["subject"] for row in ladder["patient_rows"]]
    seeds = sorted({int(row["fit_seed"]) for row in ladder["run_rows"]})
    sig_patients = {row["subject"]: row for row in sig["patient_rows"]}
    ladder_patients = {row["subject"]: row for row in ladder["patient_rows"]}

    patients = []
    route_rows = []
    pairwise_rows = []
    for subject in subjects:
        audited, per_seed_route = _audit_subject(subject, seeds)
        route_rows.extend(per_seed_route)
        old = ladder_patients[subject]
        sig_row = sig_patients[subject]
        selected = audited["validation_selected_baseline"]
        audited["test"] = {
            "sig1_nll": sig_row["sig1_nll"],
            "sig1_precedence_mae": sig_row["sig1_precedence_mae"],
            "validation_selected_baseline_nll": old[f"{selected}_nll"],
            "validation_selected_baseline_precedence_mae": old[
                f"{selected}_precedence_mae"
            ],
            "sig1_nll_gain_vs_validation_selected": (
                old[f"{selected}_nll"] - sig_row["sig1_nll"]
            ),
            "sig1_precedence_gain_vs_validation_selected": (
                old[f"{selected}_precedence_mae"]
                - sig_row["sig1_precedence_mae"]
            ),
        }
        audited["historical_test_oracle"] = {
            "nll_family_selected_on_test": min(
                MODELS, key=lambda name: old[f"{name}_nll"]
            ),
            "precedence_family_selected_on_test": min(
                MODELS, key=lambda name: old[f"{name}_precedence_mae"]
            ),
            "status": "ENDPOINT_SPECIFIC_ORACLE_STRESS_TEST_ONLY",
        }
        for name in MODELS:
            pairwise_rows.append(
                {
                    "subject": subject,
                    "baseline": name,
                    "sig1_nll_gain": old[f"{name}_nll"] - sig_row["sig1_nll"],
                    "sig1_precedence_gain": (
                        old[f"{name}_precedence_mae"]
                        - sig_row["sig1_precedence_mae"]
                    ),
                }
            )
        patients.append(audited)

    pairwise_counts = {
        name: {
            "nll_better": sum(
                row["sig1_nll_gain"] > 0
                for row in pairwise_rows if row["baseline"] == name
            ),
            "precedence_better": sum(
                row["sig1_precedence_gain"] > 0
                for row in pairwise_rows if row["baseline"] == name
            ),
            "both_better": sum(
                row["sig1_nll_gain"] > 0 and row["sig1_precedence_gain"] > 0
                for row in pairwise_rows if row["baseline"] == name
            ),
        }
        for name in MODELS
    }
    selected_counts = {
        "nll_better": sum(
            row["test"]["sig1_nll_gain_vs_validation_selected"] > 0
            for row in patients
        ),
        "precedence_better": sum(
            row["test"]["sig1_precedence_gain_vs_validation_selected"] > 0
            for row in patients
        ),
        "both_better": sum(
            row["test"]["sig1_nll_gain_vs_validation_selected"] > 0
            and row["test"]["sig1_precedence_gain_vs_validation_selected"] > 0
            for row in patients
        ),
    }
    schedule_increments = np.asarray(
        [
            row["m2_route_predictability"][
                "schedule_increment_balanced_accuracy"
            ]
            for row in patients
        ],
        dtype=float,
    )
    payload = {
        "contract": "topic5_stable_interaction_identifiability_v2_1_d1",
        "status": "COMPLETE_EXISTING_ARTIFACT_AUDIT",
        "n_subjects": len(patients),
        "n_fit_seeds": len(seeds),
        "historical_comparator_status": (
            "ENDPOINT_SPECIFIC_DEVELOPMENT_TEST_ORACLE_STRESS_TEST_ONLY"
        ),
        "pairwise_counts": pairwise_counts,
        "validation_selected_baseline_counts": selected_counts,
        "validation_selection_boundary": (
            "One family is selected per patient by seed-median inner-validation "
            "NLL and the same family is used for both test endpoints. M1/M2 "
            "validation likelihood is exact; M3 uses the frozen finite-sample "
            "future-blind estimator, so this is a corrected predictive audit, "
            "not a structure Gate."
        ),
        "future_schedule_route_audit": {
            "median_balanced_accuracy_increment": float(
                np.nanmedian(schedule_increments)
            ),
            "n_positive_increment": int(np.sum(schedule_increments > 0)),
            "interpretation": (
                "The classifier predicts M2 posterior route labels. It audits "
                "whether adding the full cardinality schedule to X1+T exposes "
                "route identity; it does not prove causal use by the generator."
            ),
        },
        "patients": patients,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "sources": {
            "sig": {"path": str(SIG_ROOT), "sha256": sha256_file(SIG_ROOT / "HUMAN_GRAPH_INCREMENT_PILOT.json")},
            "ladder": {"path": str(LADDER_ROOT), "sha256": sha256_file(LADDER_ROOT / "MATCHED_BASELINE_LADDER.json")},
        },
        "source_sha256": sha256_file(Path(__file__)),
    }
    _write_json(args.output_dir / "D1_EXISTING_ARTIFACT_AUDIT.json", payload)
    _write_csv(args.output_dir / "pairwise_effects.csv", pairwise_rows)
    _write_csv(args.output_dir / "m2_route_predictability_per_seed.csv", route_rows)
    print(json.dumps({
        "pairwise_counts": pairwise_counts,
        "validation_selected_baseline_counts": selected_counts,
        "future_schedule_route_audit": payload["future_schedule_route_audit"],
    }, indent=2))


if __name__ == "__main__":
    main()
