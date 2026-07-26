#!/usr/bin/env python3
"""Localize the bounded-negative persistent path-mode RNN result."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_persistent_path_pilot import (  # noqa: E402
    METRICS,
    _count_gate,
)
from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    load_records,
)
from scripts.train_topic5_persistent_path_rnn import (  # noqa: E402
    _batch,
    load_path_mode_priors,
)
from src.topic5_persistent_path_rnn import (  # noqa: E402
    PersistentPathModeRNN,
    persistent_mixture_loss,
)
from src.topic5_rank_distribution import contact_rank_distribution  # noqa: E402

SUBJECTS = (
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
)
SEEDS = (20260726, 20260727, 20260728)
DIAGNOSTIC_K = 2
BASELINES = ("no_history", "merged_path", "weight_shuffle", "mode_shuffle")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _primary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics.lesion.astype(str) == "none"].copy()


def _reference(
    primary: pd.DataFrame, mode_count: int, baseline: str
) -> pd.DataFrame:
    if baseline == "no_history":
        return primary[
            (primary.mode_count == 0) & (primary.control == "no_history")
        ]
    if baseline == "merged_path":
        return primary[
            (primary.mode_count == 1) & (primary.control == "merged_path")
        ]
    return primary[
        (primary.mode_count == mode_count) & (primary.control == baseline)
    ]


def local_transition_benefits(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = _primary(metrics)
    rows = []
    for mode_count in range(1, 5):
        intact = primary[
            (primary.mode_count == mode_count)
            & (primary.control == "intact")
        ].set_index(["subject", "seed"])
        baselines = BASELINES if mode_count >= 2 else BASELINES[:-1]
        for baseline in baselines:
            reference = _reference(primary, mode_count, baseline).set_index(
                ["subject", "seed"]
            )
            left, right = intact.heldout_event_nll.align(
                reference.heldout_event_nll, join="inner"
            )
            for (subject, seed), benefit in (right - left).items():
                rows.append(
                    {
                        "mode_count": mode_count,
                        "baseline": baseline,
                        "subject": subject,
                        "seed": int(seed),
                        "nll_benefit": float(benefit),
                    }
                )
    return pd.DataFrame(rows)


def mode_identifiability(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = _primary(metrics)
    frame = frame[
        (frame.control == "intact") & frame.mode_count.between(1, 4)
    ].copy()
    frame["n_components"] = 2 * frame.mode_count
    frame["maximum_entropy"] = np.log(frame.n_components)
    frame["normalized_posterior_entropy"] = (
        frame.posterior_entropy_mean / frame.maximum_entropy
    )
    frame["posterior_information_fraction"] = (
        1.0 - frame.normalized_posterior_entropy
    )
    frame["uniform_component_probability"] = 1.0 / frame.n_components
    frame["posterior_max_excess_uniform"] = (
        frame.posterior_max_mean - frame.uniform_component_probability
    )
    return frame[
        [
            "subject",
            "dataset",
            "seed",
            "mode_count",
            "n_components",
            "posterior_max_mean",
            "uniform_component_probability",
            "posterior_max_excess_uniform",
            "posterior_entropy_mean",
            "maximum_entropy",
            "normalized_posterior_entropy",
            "posterior_information_fraction",
            "rollout_component_entropy",
        ]
    ]


def node_distribution(
    pilot_root: Path,
    records: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for subject in SUBJECTS:
        record = records[subject]
        observed = contact_rank_distribution(
            record.group_ids[record.eval_indices],
            record.group_count[record.eval_indices],
            bins=10,
        )
        for seed in SEEDS:
            path = (
                pilot_root
                / f"seed_{seed}"
                / f"k_{DIAGNOSTIC_K}"
                / "intact"
                / subject
                / "free_rollouts.npz"
            )
            with np.load(path, allow_pickle=False) as z:
                generated = contact_rank_distribution(
                    z["event_group_ids"], z["event_group_count"], bins=10
                )
                axis = np.asarray(z["axis_coordinate"], float)
            for contact_index, contact_name in enumerate(record.contact_names):
                rows.append(
                    {
                        "subject": subject,
                        "dataset": record.dataset,
                        "seed": seed,
                        "contact_index": contact_index,
                        "contact_name": str(contact_name),
                        "axis_coordinate": float(axis[contact_index]),
                        "observed_participation": float(
                            observed["participation_probability"][contact_index]
                        ),
                        "generated_participation": float(
                            generated["participation_probability"][contact_index]
                        ),
                        "observed_mean_rank": float(
                            observed["mean_rank"][contact_index]
                        ),
                        "generated_mean_rank": float(
                            generated["mean_rank"][contact_index]
                        ),
                    }
                )
    all_seed = pd.DataFrame(rows)
    contact = (
        all_seed.groupby(
            [
                "subject",
                "dataset",
                "contact_index",
                "contact_name",
                "axis_coordinate",
                "observed_participation",
                "observed_mean_rank",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            generated_participation=("generated_participation", "median"),
            generated_mean_rank=("generated_mean_rank", "median"),
        )
    )
    summary_rows = []
    for subject, frame in all_seed.groupby("subject"):
        for seed, seed_frame in frame.groupby("seed"):
            for field in ("participation", "mean_rank"):
                observed = seed_frame[f"observed_{field}"].to_numpy(float)
                generated = seed_frame[f"generated_{field}"].to_numpy(float)
                valid = np.isfinite(observed) & np.isfinite(generated)
                summary_rows.append(
                    {
                        "subject": subject,
                        "seed": int(seed),
                        "field": field,
                        "n_contacts": int(valid.sum()),
                        "mae": float(
                            np.mean(np.abs(generated[valid] - observed[valid]))
                        ),
                        "pearson_r": (
                            float(np.corrcoef(observed[valid], generated[valid])[0, 1])
                            if valid.sum() >= 3
                            else np.nan
                        ),
                    }
                )
    return contact, pd.DataFrame(summary_rows)


def lesion_gate_matrix(
    metrics: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intact = metrics[
        (metrics.control == "intact")
        & (metrics.lesion.astype(str) == "none")
    ].set_index(["mode_count", "subject", "seed"])
    benefits = []
    for mode_count in range(1, 5):
        current = metrics[
            (metrics.mode_count == mode_count)
            & (metrics.control == "intact")
        ]
        direction = current[
            current.lesion.isin(("drop_forward", "drop_reverse"))
        ].groupby(["subject", "seed"])[list(METRICS)].mean()
        base = intact.xs(mode_count, level="mode_count")
        for metric in METRICS:
            left, right = base[metric].align(direction[metric], join="inner")
            for (subject, seed), value in (right - left).items():
                benefits.append(
                    {
                        "mode_count": mode_count,
                        "lesion": "direction_removal_mean",
                        "metric": metric,
                        "subject": subject,
                        "seed": int(seed),
                        "benefit": float(value),
                    }
                )
        for lesion in (
            "graph",
            "inhibition",
            "mode_collapse",
            "drop_dominant_mode",
        ):
            lesion_frame = current[current.lesion == lesion].set_index(
                ["mode_count", "subject", "seed"]
            )
            if lesion_frame.empty:
                continue
            for metric in METRICS:
                left, right = intact[metric].align(
                    lesion_frame[metric], join="inner"
                )
                for (k, subject, seed), value in (right - left).items():
                    benefits.append(
                        {
                            "mode_count": int(k),
                            "lesion": lesion,
                            "metric": metric,
                            "subject": subject,
                            "seed": int(seed),
                            "benefit": float(value),
                        }
                    )
    benefit_frame = pd.DataFrame(benefits)
    checks = []
    for keys, frame in benefit_frame.groupby(
        ["mode_count", "lesion", "metric"]
    ):
        check = _count_gate(
            frame,
            min_patient_seed=cfg["evaluation"][
                "pilot_min_patient_seed_better"
            ],
            min_subjects=cfg["evaluation"]["pilot_min_subjects_better"],
        )
        checks.append(
            {
                "mode_count": int(keys[0]),
                "lesion": keys[1],
                "metric": keys[2],
                **check,
            }
        )
    return benefit_frame, pd.DataFrame(checks)


@torch.no_grad()
def posterior_trajectory(
    pilot_root: Path,
    records: dict,
    priors: dict,
    cfg: dict,
    *,
    device: torch.device,
) -> pd.DataFrame:
    accumulator = defaultdict(
        lambda: {
            "entropy": 0.0,
            "state": 0.0,
            "inhibition": 0.0,
            "n": 0,
        }
    )
    batch_size = int(cfg["training"]["batch_events"])
    for subject in SUBJECTS:
        record = records[subject]
        prior = priors[subject]
        for seed in SEEDS:
            run_dir = (
                pilot_root
                / f"seed_{seed}"
                / f"k_{DIAGNOSTIC_K}"
                / "intact"
                / subject
            )
            checkpoint = torch.load(
                run_dir / "checkpoint.pt",
                map_location=device,
                weights_only=False,
            )
            if bool(checkpoint.get("ictal_target_read", True)):
                raise RuntimeError("checkpoint does not preserve target seal")
            model = PersistentPathModeRNN(
                record.contact_features.shape[1],
                local_offset_dim=int(cfg["model"]["local_offset_dim"]),
                use_recurrence=True,
            ).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            offset = checkpoint["heldout_local_offset"].to(device)
            for start in range(0, len(record.eval_indices), batch_size):
                index = record.eval_indices[start : start + batch_size]
                batch = _batch(record, prior, index, device)
                output = model(**batch, local_offset=offset)
                loss = persistent_mixture_loss(
                    output,
                    batch["group_ids"],
                    batch["group_count"],
                    stop_calibration_weight=float(
                        cfg["model"]["stop_calibration_weight"]
                    ),
                    endpoint_source_weight=float(
                        cfg["model"]["endpoint_source_weight"]
                    ),
                )
                posterior = (
                    loss["component_posterior_trajectory"].detach().cpu().numpy()
                )
                entropy = -np.sum(
                    posterior * np.log(np.clip(posterior, 1e-12, 1.0)),
                    axis=2,
                ) / np.log(posterior.shape[2])
                state = (
                    output["latent_state"]
                    .detach()
                    .abs()
                    .mean(3)
                    .permute(0, 2, 1)
                    .cpu()
                    .numpy()
                )
                inhibition = (
                    output["inhibitory_state"]
                    .detach()
                    .abs()
                    .permute(0, 2, 1)
                    .cpu()
                    .numpy()
                )
                state = np.sum(posterior * state, axis=2)
                inhibition = np.sum(posterior * inhibition, axis=2)
                counts = record.group_count[index]
                for local, count in enumerate(counts):
                    denominator = max(int(count), 1)
                    for step in range(int(count) + 1):
                        bin_index = min(
                            5, max(0, int(np.rint(5 * step / denominator)))
                        )
                        key = (subject, seed, bin_index)
                        accumulator[key]["entropy"] += float(
                            entropy[local, step]
                        )
                        accumulator[key]["state"] += float(state[local, step])
                        accumulator[key]["inhibition"] += float(
                            inhibition[local, step]
                        )
                        accumulator[key]["n"] += 1
    rows = []
    for (subject, seed, bin_index), value in accumulator.items():
        rows.append(
            {
                "subject": subject,
                "seed": int(seed),
                "prefix_fraction": bin_index / 5.0,
                "normalized_component_entropy": (
                    value["entropy"] / value["n"]
                ),
                "posterior_weighted_state_abs": value["state"] / value["n"],
                "posterior_weighted_inhibition_abs": (
                    value["inhibition"] / value["n"]
                ),
                "n_event_steps": int(value["n"]),
            }
        )
    return pd.DataFrame(rows)


def comparison_gate_matrix(
    comparison_checks: pd.DataFrame,
    lesion_checks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for mode_count in range(1, 5):
        for baseline in BASELINES:
            use = comparison_checks[
                (comparison_checks.mode_count == mode_count)
                & (comparison_checks.baseline == baseline)
            ]
            if len(use) != 2:
                continue
            values = use.set_index("metric").to_dict(orient="index")
            rows.append(
                {
                    "section": "comparison",
                    "row": baseline,
                    "mode_count": mode_count,
                    "precedence_n_better": int(
                        values["precedence_mae"]["n_patient_seed_better"]
                    ),
                    "whole_path_n_better": int(
                        values[
                            "path_sliced_wasserstein"
                        ]["n_patient_seed_better"]
                    ),
                    "precedence_subjects_better": int(
                        values["precedence_mae"]["n_subject_median_better"]
                    ),
                    "whole_path_subjects_better": int(
                        values[
                            "path_sliced_wasserstein"
                        ]["n_subject_median_better"]
                    ),
                    "both_metric_gate_pass": bool(
                        values["precedence_mae"]["pass"]
                        and values["path_sliced_wasserstein"]["pass"]
                    ),
                }
            )
        for lesion in (
            "graph",
            "inhibition",
            "direction_removal_mean",
            "mode_collapse",
            "drop_dominant_mode",
        ):
            use = lesion_checks[
                (lesion_checks.mode_count == mode_count)
                & (lesion_checks.lesion == lesion)
            ]
            if len(use) != 2:
                continue
            values = use.set_index("metric").to_dict(orient="index")
            rows.append(
                {
                    "section": "lesion",
                    "row": lesion,
                    "mode_count": mode_count,
                    "precedence_n_better": int(
                        values["precedence_mae"]["n_patient_seed_better"]
                    ),
                    "whole_path_n_better": int(
                        values[
                            "path_sliced_wasserstein"
                        ]["n_patient_seed_better"]
                    ),
                    "precedence_subjects_better": int(
                        values["precedence_mae"]["n_subject_median_better"]
                    ),
                    "whole_path_subjects_better": int(
                        values[
                            "path_sliced_wasserstein"
                        ]["n_subject_median_better"]
                    ),
                    "both_metric_gate_pass": bool(
                        values["precedence_mae"]["pass"]
                        and values["path_sliced_wasserstein"]["pass"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _benefit_summary(
    frame: pd.DataFrame, value: str, group: list[str]
) -> list[dict]:
    rows = []
    for keys, values in frame.groupby(group):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(group, keys)),
                "median": float(values[value].median()),
                "n_positive": int((values[value] > 0).sum()),
                "n_total": int(len(values)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v0_9.yaml",
    )
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    pilot_root = (
        args.pilot_root
        if args.pilot_root is not None
        else ROOT / cfg["outputs"]["pilot"]
    )
    analysis_root = pilot_root / "analysis"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else analysis_root / "bounded_negative"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = json.loads(
        (analysis_root / "pilot_gate_summary.json").read_text()
    )
    if bool(gate["hard_gate_pass"]):
        raise RuntimeError("bounded-negative analysis requested after a pass")
    if bool(gate.get("ictal_target_read", True)):
        raise RuntimeError("ictal target seal failed")
    metrics = pd.read_csv(analysis_root / "all_seed_metrics.csv")
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    nll = local_transition_benefits(metrics)
    nll.to_csv(output_dir / "local_transition_nll_benefits.csv", index=False)
    identifiability = mode_identifiability(metrics)
    identifiability.to_csv(
        output_dir / "mode_identifiability.csv", index=False
    )
    node, node_summary = node_distribution(pilot_root, records)
    node.to_csv(output_dir / "node_distribution_k2.csv", index=False)
    node_summary.to_csv(
        output_dir / "node_distribution_k2_summary.csv", index=False
    )
    lesion_benefit, lesion_checks = lesion_gate_matrix(metrics, cfg)
    lesion_benefit.to_csv(
        output_dir / "lesion_benefits_complete.csv", index=False
    )
    lesion_checks.to_csv(
        output_dir / "lesion_gate_checks_complete.csv", index=False
    )
    comparison_checks = pd.read_csv(
        analysis_root / "comparison_gate_checks.csv"
    )
    matrix = comparison_gate_matrix(comparison_checks, lesion_checks)
    matrix.to_csv(output_dir / "hard_gate_matrix.csv", index=False)
    device = torch.device(args.device)
    priors = load_path_mode_priors(
        ROOT / cfg["inputs"]["path_mode_prior"],
        records,
        mode_count=DIAGNOSTIC_K,
        control="intact",
        seed=SEEDS[0],
        axis_floor=float(cfg["prior"]["axis_floor"]),
        neighbors=int(cfg["prior"]["neighbors"]),
    )
    trajectory = posterior_trajectory(
        pilot_root, records, priors, cfg, device=device
    )
    trajectory.to_csv(
        output_dir / "posterior_state_trajectory_k2.csv", index=False
    )
    prior_audit = pd.read_csv(
        ROOT
        / cfg["outputs"]["prior"]
        / "path_mode_prior_audit.csv"
    )
    prior_summary = (
        prior_audit.groupby("mode_count")
        .agg(
            split_half_cosine=(
                "split_half_aligned_mode_cosine_median",
                "median",
            ),
            heldout_reconstruction=(
                "heldout_soft_reconstruction_cosine_median",
                "median",
            ),
            within_patient_mode_cosine=(
                "pairwise_mode_cosine_median",
                "median",
            ),
        )
        .reset_index()
    )
    prior_summary.to_csv(
        output_dir / "path_mode_prior_summary.csv", index=False
    )
    summary = {
        "status": "complete",
        "result_tier": "bounded_negative",
        "diagnostic_mode_count": DIAGNOSTIC_K,
        "diagnostic_k_rationale": (
            "smallest explicit multi-path model, fixed independently of outcome"
        ),
        "n_pilot_patients": len(SUBJECTS),
        "n_seeds": len(SEEDS),
        "n_runs": 117,
        "hard_gate_pass": False,
        "selected_mode_count": None,
        "local_transition_nll_benefits": _benefit_summary(
            nll, "nll_benefit", ["mode_count", "baseline"]
        ),
        "mode_identifiability": (
            identifiability.groupby("mode_count")
            .agg(
                normalized_entropy_median=(
                    "normalized_posterior_entropy",
                    "median",
                ),
                information_fraction_median=(
                    "posterior_information_fraction",
                    "median",
                ),
                posterior_max_excess_uniform_median=(
                    "posterior_max_excess_uniform",
                    "median",
                ),
            )
            .reset_index()
            .to_dict("records")
        ),
        "node_distribution_k2": (
            node_summary.groupby("field")
            .agg(
                pearson_r_median=("pearson_r", "median"),
                mae_median=("mae", "median"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "seed_rank_stability_median": {
            metric: float(
                pd.read_csv(
                    analysis_root / "seed_rank_stability.csv"
                )
                .loc[lambda frame: frame.metric == metric, "spearman"]
                .median()
            )
            for metric in METRICS
        },
        "next_action": "stop_no_formal_no_ictal",
        "formal_34x3_started": False,
        "ictal_target_read": False,
    }
    (output_dir / "bounded_negative_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default)
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
