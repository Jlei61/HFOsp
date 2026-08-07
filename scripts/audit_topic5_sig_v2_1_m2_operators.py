#!/usr/bin/env python3
"""D2 audit of M2 observable operators, permutations, and shared backbone."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

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
)


DEVELOPMENT = ROOT / "results/topic5_stable_interaction_graph/development"
LADDER_ROOT = (
    DEVELOPMENT / "human_matched_baseline_ladder_v0_2_training_adequacy"
)
DATASET_DIR = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUTPUT = DEVELOPMENT / "v2_1_m2_operator_audit"


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


def _load_model(subject: str, seed: int, contacts: int) -> MatchedPhaseMarkovMixtureModel:
    checkpoint = torch.load(
        LADDER_ROOT / "per_run" / subject / f"seed_{seed}/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["models"]["m2_mixture_matched_phase"]["state_dict"]
    model = MatchedPhaseMarkovMixtureModel(
        contacts, state["static_bias"].numpy(), n_components=3
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _component_scores(
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


@torch.no_grad()
def _supported_component_influence(
    model: MatchedPhaseMarkovMixtureModel,
    groups: torch.Tensor,
    counts: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal sender response over occupied prefixes in contact coordinates."""
    contacts = model.n_contacts
    numerator = torch.zeros(
        model.n_components, contacts, contacts, dtype=model.static_bias.dtype
    )
    denominator = torch.zeros_like(numerator)
    recruited = groups == 0
    previous = recruited.clone()
    impossible = torch.finfo(model.static_bias.dtype).min / 4.0
    for step in range(1, int(counts.max().item())):
        active = counts > step
        candidate = ~recruited
        for component in range(model.n_components):
            do_logits = model.component_logits(
                component, previous, step=step, group_count=counts
            )
            do_probability = torch.softmax(
                torch.where(candidate, do_logits, impossible), dim=1
            )
            for source in range(contacts):
                context = active & previous[:, source]
                if not torch.any(context):
                    continue
                control_previous = previous[context].clone()
                control_previous[:, source] = False
                control_logits = model.component_logits(
                    component,
                    control_previous,
                    step=step,
                    group_count=counts[context],
                )
                context_candidate = candidate[context]
                control_probability = torch.softmax(
                    torch.where(
                        context_candidate, control_logits, impossible
                    ),
                    dim=1,
                )
                difference = do_probability[context] - control_probability
                valid_target = context_candidate.to(difference.dtype)
                numerator[component, :, source] += (
                    difference * valid_target
                ).sum(0)
                denominator[component, :, source] += valid_target.sum(0)
        target = groups == step
        recruited = recruited | target
        previous = target
    output = torch.full_like(numerator, float("nan"))
    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]
    return output.numpy(), denominator.numpy()


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 10 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
        return float("nan")
    return float(spearmanr(a[valid], b[valid]).statistic)


def _match_components(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.asarray(
        [
            [_spearman(left[i], right[j]) for j in range(right.shape[0])]
            for i in range(left.shape[0])
        ],
        dtype=float,
    )
    cost = -np.nan_to_num(similarity, nan=-1.0)
    row, col = linear_sum_assignment(cost)
    permutation = np.empty(left.shape[0], dtype=int)
    permutation[row] = col
    return permutation, similarity


def _nanmean_axis0(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    numerator = np.where(finite, values, 0.0).sum(axis=0)
    denominator = finite.sum(axis=0)
    output = np.full(values.shape[1:], np.nan, dtype=float)
    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]
    return output


def _backbone_fraction(components: np.ndarray) -> dict[str, float]:
    backbone = _nanmean_axis0(components)
    backbone_energy = float(np.nanmean(backbone**2))
    residual_energy = float(
        np.nanmean((components - backbone[None, :, :]) ** 2)
    )
    total = backbone_energy + residual_energy
    return {
        "backbone_energy": backbone_energy,
        "component_residual_energy": residual_energy,
        "shared_backbone_energy_fraction": (
            backbone_energy / total if total > 0 else float("nan")
        ),
    }


def _audit_subject(subject: str, seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = load_subject_rank_events(DATASET_DIR, subject)
    train, _, _ = record.development_split(0.15, 0.15)
    train = _subsample(train, 9600)
    groups = torch.as_tensor(record.group_ids[train], dtype=torch.long)
    counts = torch.as_tensor(record.group_count[train], dtype=torch.long)
    operators = {}
    occupancies = {}
    supports = {}
    fit_rows = []
    for seed in seeds:
        model = _load_model(subject, seed, len(record.contact_names))
        component, support = _supported_component_influence(model, groups, counts)
        operators[seed] = component
        supports[seed] = support
        labels = _component_scores(model, groups, counts).argmax(1).numpy()
        occupancy = np.bincount(labels, minlength=3)
        occupancies[seed] = occupancy
        energy = _backbone_fraction(component)
        fit_rows.append(
            {
                "subject": subject,
                "fit_seed": seed,
                "posterior_occupancy_0": int(occupancy[0]),
                "posterior_occupancy_1": int(occupancy[1]),
                "posterior_occupancy_2": int(occupancy[2]),
                "posterior_occupancy_min_fraction": float(
                    occupancy.min() / occupancy.sum()
                ),
                "supported_pair_fraction": float(np.mean(support > 0)),
                **energy,
            }
        )

    pair_rows = []
    backbone_stability = []
    component_stability = []
    for left_seed, right_seed in itertools.combinations(seeds, 2):
        left = operators[left_seed]
        right = operators[right_seed]
        permutation, similarity = _match_components(left, right)
        matched = [similarity[i, permutation[i]] for i in range(3)]
        left_backbone = _nanmean_axis0(left)
        right_backbone = _nanmean_axis0(right)
        backbone_rho = _spearman(left_backbone, right_backbone)
        backbone_stability.append(backbone_rho)
        component_stability.extend(matched)
        pair_rows.append(
            {
                "subject": subject,
                "left_seed": left_seed,
                "right_seed": right_seed,
                "permutation": "|".join(map(str, permutation.tolist())),
                "component_rho_mean": float(np.nanmean(matched)),
                "component_rho_min": float(np.nanmin(matched)),
                "backbone_rho": backbone_rho,
            }
        )

    subject_row = {
        "subject": subject,
        "n_contacts": len(record.contact_names),
        "n_train_events": len(train),
        "component_seed_stability_median": float(
            np.nanmedian(component_stability)
        ),
        "component_seed_stability_min": float(np.nanmin(component_stability)),
        "backbone_seed_stability_median": float(
            np.nanmedian(backbone_stability)
        ),
        "backbone_seed_stability_min": float(np.nanmin(backbone_stability)),
        "shared_backbone_energy_fraction_median": float(
            np.nanmedian(
                [row["shared_backbone_energy_fraction"] for row in fit_rows]
            )
        ),
        "minimum_component_occupancy_fraction": float(
            min(row["posterior_occupancy_min_fraction"] for row in fit_rows)
        ),
        "supported_pair_fraction_min": float(
            min(row["supported_pair_fraction"] for row in fit_rows)
        ),
        "fit_rows": fit_rows,
        "pair_rows": pair_rows,
    }
    return subject_row, pair_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ladder = json.loads((LADDER_ROOT / "MATCHED_BASELINE_LADDER.json").read_text())
    subjects = [row["subject"] for row in ladder["patient_rows"]]
    seeds = sorted({int(row["fit_seed"]) for row in ladder["run_rows"]})
    patients = []
    pair_rows = []
    for subject in subjects:
        row, pairs = _audit_subject(subject, seeds)
        patients.append(row)
        pair_rows.extend(pairs)
        print(json.dumps({
            "subject": subject,
            "component_seed_stability": row["component_seed_stability_median"],
            "backbone_seed_stability": row["backbone_seed_stability_median"],
        }))

    payload = {
        "contract": "topic5_stable_interaction_identifiability_v2_1_d2",
        "status": "COMPLETE_EXISTING_M2_OPERATOR_AUDIT",
        "n_subjects": len(patients),
        "n_fit_seeds": len(seeds),
        "operator": (
            "For each occupied training prefix, remove one sender from the "
            "previous rank while holding the candidate set, phase, and other "
            "senders fixed; average the change in next-contact probability."
        ),
        "primary_representation": "SUPPORTED_OBSERVABLE_CONTACT_INFLUENCE",
        "raw_parameter_similarity_is_primary": False,
        "scope": (
            "Seed stability only. Chronological early/late stability and "
            "real-over-null specificity remain untested and cannot be inferred "
            "from this artifact reuse audit."
        ),
        "cohort_summary": {
            "component_seed_stability_median": float(
                np.median([row["component_seed_stability_median"] for row in patients])
            ),
            "backbone_seed_stability_median": float(
                np.median([row["backbone_seed_stability_median"] for row in patients])
            ),
            "shared_backbone_energy_fraction_median": float(
                np.median([row["shared_backbone_energy_fraction_median"] for row in patients])
            ),
        },
        "patients": patients,
        "old_heldout20_scored": False,
        "forbidden_inputs_read": False,
        "snn_inputs_read": False,
        "source_artifact": {
            "path": str(LADDER_ROOT / "MATCHED_BASELINE_LADDER.json"),
            "sha256": sha256_file(LADDER_ROOT / "MATCHED_BASELINE_LADDER.json"),
        },
        "source_sha256": sha256_file(Path(__file__)),
    }
    _write_json(args.output_dir / "D2_M2_OPERATOR_AUDIT.json", payload)
    _write_csv(args.output_dir / "component_matching_pairs.csv", pair_rows)
    print(json.dumps(payload["cohort_summary"], indent=2))


if __name__ == "__main__":
    main()
