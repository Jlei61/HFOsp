#!/usr/bin/env python3
"""Build frozen node-rank fields after the formal interictal gate passes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records
from scripts.train_topic5_persistent_path_rnn import (
    _rollout,
    load_path_mode_priors,
)
from src.topic5_persistent_path_rnn import PersistentPathModeRNN
from src.topic5_rank_distribution import contact_rank_distribution


SEEDS = (20260726, 20260727, 20260728)
GENERATED_CONDITIONS = (
    "intact",
    "no_history",
    "graph_lesion",
    "mode_collapse_lesion",
)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location="cpu")


def _distribution_rows(
    groups: np.ndarray,
    counts: np.ndarray,
    *,
    subject: str,
    dataset: str,
    seed: int,
    condition: str,
    contact_names: np.ndarray,
) -> list[dict]:
    distribution = contact_rank_distribution(groups, counts, bins=10)
    rows = []
    for contact, name in enumerate(contact_names):
        participation = float(
            distribution["participation_probability"][contact]
        )
        histogram = distribution["rank_histogram"][contact]
        row = {
            "subject": subject,
            "dataset": dataset,
            "seed": int(seed),
            "condition": condition,
            "contact_index": int(contact),
            "contact_name": str(name),
            "nonparticipation_probability": 1.0 - participation,
            "participation_probability": participation,
            "mean_rank": float(distribution["mean_rank"][contact]),
            "early_probability_conditional": float(np.sum(histogram[:3])),
            "middle_probability_conditional": float(np.sum(histogram[3:7])),
            "late_probability_conditional": float(np.sum(histogram[7:])),
        }
        for bin_index, probability in enumerate(histogram):
            row[f"joint_rank_bin_{bin_index}"] = float(
                participation * probability
            )
        probability_sum = row["nonparticipation_probability"] + sum(
            row[f"joint_rank_bin_{bin_index}"] for bin_index in range(10)
        )
        if not np.isclose(probability_sum, 1.0, atol=1e-7):
            raise RuntimeError(
                f"{subject}:{condition}:{name}: node distribution not normalized"
            )
        rows.append(row)
    return rows


def _saved_rollout(
    run_dir: Path, contact_names: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(run_dir / "free_rollouts.npz", allow_pickle=False) as z:
        if bool(z["ictal_target_read"]):
            raise RuntimeError(f"{run_dir}: ictal target entered rollout")
        if not np.array_equal(
            np.asarray(z["contact_names"]).astype(str),
            np.asarray(contact_names).astype(str),
        ):
            raise RuntimeError(f"{run_dir}: rollout contact order mismatch")
        return (
            np.asarray(z["event_group_ids"], np.int16),
            np.asarray(z["event_group_count"], np.int16),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_persistent_path_mode_rnn_v1_0.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.root.resolve()
    analysis = root / "analysis"
    gate_path = analysis / "formal_gate_summary.json"
    if not gate_path.exists():
        raise RuntimeError("formal interictal analysis is incomplete")
    gate = json.loads(gate_path.read_text())
    if gate.get("formal_interictal_gate_pass") is not True:
        raise RuntimeError(
            "interictal gate failed; cross-state feature stage remains sealed"
        )
    if gate.get("ictal_target_read") is not False:
        raise RuntimeError("formal interictal target seal was violated")

    config_path = (
        args.config if args.config.is_absolute() else ROOT / args.config
    )
    cfg = yaml.safe_load(config_path.read_text())
    records = load_records(ROOT / cfg["inputs"]["dataset"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = []
    for subject, record in records.items():
        rows.extend(
            _distribution_rows(
                record.group_ids[record.train_indices],
                record.group_count[record.train_indices],
                subject=subject,
                dataset=record.dataset,
                seed=-1,
                condition="empirical_train80",
                contact_names=record.contact_names,
            )
        )

    for seed in SEEDS:
        priors = load_path_mode_priors(
            ROOT / cfg["inputs"]["path_mode_prior"],
            records,
            mode_count=2,
            control="intact",
            seed=int(seed),
            axis_floor=float(cfg["prior"]["axis_floor"]),
            neighbors=int(cfg["prior"]["neighbors"]),
        )
        for subject, record in records.items():
            intact_dir = (
                root / f"seed_{seed}" / "k_2" / "intact" / subject
            )
            no_history_dir = (
                root / f"seed_{seed}" / "k_0" / "no_history" / subject
            )
            for condition, run_dir in (
                ("intact", intact_dir),
                ("no_history", no_history_dir),
            ):
                groups, counts = _saved_rollout(
                    run_dir, record.contact_names
                )
                rows.extend(
                    _distribution_rows(
                        groups,
                        counts,
                        subject=subject,
                        dataset=record.dataset,
                        seed=int(seed),
                        condition=condition,
                        contact_names=record.contact_names,
                    )
                )

            checkpoint = _load_checkpoint(intact_dir / "checkpoint.pt")
            if checkpoint.get("ictal_target_read") is not False:
                raise RuntimeError(f"{subject}: ictal target entered checkpoint")
            model = PersistentPathModeRNN(
                record.contact_features.shape[1],
                local_offset_dim=int(cfg["model"]["local_offset_dim"]),
                use_recurrence=True,
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            offset = checkpoint["heldout_local_offset"].to(device)
            for condition, lesion in (
                ("graph_lesion", "graph"),
                ("mode_collapse_lesion", "mode_collapse"),
            ):
                groups, counts, _ = _rollout(
                    model,
                    record,
                    priors[subject],
                    offset,
                    device=device,
                    n_events=int(cfg["evaluation"]["formal_rollouts"]),
                    seed=int(seed) + 700_000,
                    lesion=lesion,
                )
                rows.extend(
                    _distribution_rows(
                        groups,
                        counts,
                        subject=subject,
                        dataset=record.dataset,
                        seed=int(seed),
                        condition=condition,
                        contact_names=record.contact_names,
                    )
                )
            del model, offset

    frame = pd.DataFrame(rows)
    expected = {"empirical_train80", *GENERATED_CONDITIONS}
    if set(frame.condition) != expected:
        raise RuntimeError("cross-state feature condition inventory mismatch")
    out = analysis / "cross_state_frozen_node_rank_features.csv"
    frame.to_csv(out, index=False)
    summary = {
        "status": "complete",
        "contract": "topic5_rnn_frozen_ictal_static_readout_v1_0",
        "n_patients": int(frame.subject.nunique()),
        "conditions": sorted(expected),
        "generated_seeds": list(SEEDS),
        "feature": (
            "nonparticipation probability plus 10 joint normalized-rank-bin "
            "probabilities"
        ),
        "ictal_target_read": False,
        "output": str(out.relative_to(ROOT)),
    }
    (analysis / "cross_state_frozen_node_rank_features_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
