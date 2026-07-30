#!/usr/bin/env python3
"""Run the v0.3 synthetic-mode and rank-shuffle Phase-0 sanity checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_state_conditioned_rnn import derive_prefix_axis  # noqa: E402


def synthetic_forward_reverse(seed: int = 20260724) -> dict:
    rng = np.random.default_rng(int(seed))
    n_contacts, n_events = 8, 400
    true_mode = np.arange(n_events) % 2
    participation = rng.random((n_contacts, n_events)) < 0.88
    participation[:4] = True
    ranks = np.zeros((n_contacts, n_events), float)
    forward = np.arange(n_contacts, dtype=float)
    reverse = forward[::-1]
    for event in range(n_events):
        template = forward if true_mode[event] == 0 else reverse
        noisy = template + rng.normal(0.0, 0.08, n_contacts)
        ranks[:, event] = np.argsort(np.argsort(noisy))
        phantom = ~participation[:, event]
        ranks[phantom, event] = 1000 + rng.integers(0, 100, int(np.sum(phantom)))
    axis = derive_prefix_axis(
        ranks,
        participation,
        np.arange(n_events),
        seed=int(seed),
        min_cluster_fraction=0.10,
    )
    labels = np.asarray(axis["labels"], int)
    accuracy = max(
        float(np.mean(labels == true_mode)),
        float(np.mean((1 - labels) == true_mode)),
    )
    ta = np.asarray(axis["template_a"], float)
    tb = np.asarray(axis["template_b"], float)
    valid = np.isfinite(ta) & np.isfinite(tb)
    template_correlation = float(np.corrcoef(ta[valid], tb[valid])[0, 1])
    passed = bool(
        accuracy >= 0.95
        and float(axis["seed_ami"]) >= 0.95
        and template_correlation <= -0.90
    )
    return {
        "pass": passed,
        "n_contacts": n_contacts,
        "n_events": n_events,
        "label_accuracy_up_to_swap": accuracy,
        "seed_ami": float(axis["seed_ami"]),
        "template_correlation": template_correlation,
        "cluster_fractions": np.asarray(axis["cluster_fractions"]).tolist(),
        "phantom_values_injected": True,
        "phantom_values_masked_before_clustering": True,
    }


def real_rank_shuffle(dataset_dir: Path, subject: str, seed: int = 20260724) -> dict:
    path = dataset_dir / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as z:
        groups = np.asarray(z["event_group_ids"], np.int16)
        split = np.asarray(z["event_split"], np.uint8)
    groups = groups[split == 0]
    if groups.shape[0] > 2000:
        take = np.linspace(0, groups.shape[0] - 1, 2000).round().astype(int)
        groups = groups[np.unique(take)]
    rng = np.random.default_rng(int(seed))
    shuffled = groups.copy()
    for event_index, event in enumerate(groups):
        participating = np.flatnonzero(event >= 0)
        shuffled[event_index, participating] = rng.permutation(event[participating])

    mask_equal = bool(np.array_equal(groups >= 0, shuffled >= 0))
    group_size_equal = True
    correlations = []
    for original, control in zip(groups, shuffled):
        participant = original >= 0
        if not np.array_equal(
            np.sort(original[participant]), np.sort(control[participant])
        ):
            group_size_equal = False
        if np.sum(participant) >= 3:
            tau = kendalltau(original[participant], control[participant]).statistic
            if np.isfinite(tau):
                correlations.append(float(tau))
    support_original = np.mean(groups >= 0, axis=0)
    support_control = np.mean(shuffled >= 0, axis=0)
    median_tau = float(np.median(correlations)) if correlations else np.nan
    passed = bool(
        mask_equal
        and group_size_equal
        and np.array_equal(support_original, support_control)
        and abs(median_tau) <= 0.10
    )
    return {
        "pass": passed,
        "subject": subject,
        "n_events": int(groups.shape[0]),
        "participation_mask_exactly_preserved": mask_equal,
        "per_event_group_size_multiset_preserved": group_size_equal,
        "contact_support_exactly_preserved": bool(
            np.array_equal(support_original, support_control)
        ),
        "median_original_vs_shuffle_kendall_tau": median_tau,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_operator_static_readout/dataset_v0_3",
    )
    parser.add_argument("--subject", default="epilepsiae_1084")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    dataset_dir = (
        args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    )
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    synthetic = synthetic_forward_reverse(args.seed)
    shuffle = real_rank_shuffle(dataset_dir, args.subject, args.seed)
    report = {
        "contract": manifest["contract"],
        "dataset_phase0_pass": bool(manifest["phase0_pass"]),
        "target_values_read": bool(manifest["target_values_read"]),
        "synthetic_forward_reverse": synthetic,
        "participation_preserving_rank_shuffle": shuffle,
        "phase0_sanity_pass": bool(
            manifest["phase0_pass"]
            and not manifest["target_values_read"]
            and synthetic["pass"]
            and shuffle["pass"]
        ),
    }
    (dataset_dir / "phase0_sanity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (dataset_dir / "PHASE0_DONE.json").write_text(
        json.dumps(
            {
                "status": "complete" if report["phase0_sanity_pass"] else "failed",
                "scientific_scope": "data_and_sanity_only_not_event_dynamics_gate",
                "target_values_read": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["phase0_sanity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
