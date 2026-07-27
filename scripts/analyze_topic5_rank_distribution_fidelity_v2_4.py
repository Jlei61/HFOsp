#!/usr/bin/env python3
"""Audit what node-level rank information the frozen v2.4 rollouts retain."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
REPRESENTATIONS = BASE / "representations/per_subject"
OUT = BASE / "representations"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_rank(distribution: np.ndarray) -> np.ndarray:
    participation = distribution[:, 1:].sum(axis=1)
    bins = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
    numerator = distribution[:, 1:] @ bins
    return np.divide(
        numerator,
        participation,
        out=np.ones_like(numerator),
        where=participation > 0,
    )


def mean_contact_total_variation(
    first: np.ndarray, second: np.ndarray
) -> float:
    return float(np.mean(0.5 * np.abs(first - second).sum(axis=1)))


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    value = float(spearmanr(first, second).statistic)
    if not np.isfinite(value):
        raise ValueError("rank-distribution summary is constant")
    return value


def summarize(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(float)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "iqr": np.quantile(array, [0.25, 0.75]).tolist(),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for subject in audit["target_metadata_eligible_patients"]:
        path = REPRESENTATIONS / f"{subject}.npz"
        with np.load(path, allow_pickle=False) as data:
            full = np.asarray(data["full_fixed_axis"], dtype=np.float64)
            empirical = np.asarray(
                data["empirical_train80"], dtype=np.float64
            )
            no_history = np.asarray(
                data["no_history"], dtype=np.float64
            )
            isotropic = np.asarray(
                data["local_isotropic"], dtype=np.float64
            )
            node_only = np.asarray(data["node_only"], dtype=np.float64)
        for name, distribution in (
            ("full", full),
            ("empirical", empirical),
            ("no_history", no_history),
            ("isotropic", isotropic),
            ("node_only", node_only),
        ):
            if not np.allclose(distribution.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"{subject}: {name} rows do not close")
        rows.append(
            {
                "subject": subject,
                "n_contacts": len(full),
                "full_empirical_mean_contact_tv": (
                    mean_contact_total_variation(full, empirical)
                ),
                "full_no_history_mean_contact_tv": (
                    mean_contact_total_variation(full, no_history)
                ),
                "full_isotropic_mean_contact_tv": (
                    mean_contact_total_variation(full, isotropic)
                ),
                "full_node_only_mean_contact_tv": (
                    mean_contact_total_variation(full, node_only)
                ),
                "full_empirical_participation_spearman": rank_correlation(
                    1.0 - full[:, 0], 1.0 - empirical[:, 0]
                ),
                "full_empirical_expected_rank_spearman": rank_correlation(
                    expected_rank(full), expected_rank(empirical)
                ),
                "target_values_read": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values("subject")
    frame.to_csv(OUT / "rank_distribution_fidelity.csv", index=False)
    metric_names = [
        column
        for column in frame.columns
        if column not in {"subject", "n_contacts", "target_values_read"}
    ]
    payload = {
        "contract": "topic5_rank_distribution_fidelity_v2_4",
        "status": "COMPLETE",
        "n_patients": int(len(frame)),
        "metrics": {
            metric: summarize(frame[metric]) for metric in metric_names
        },
        "bounded_interpretation": (
            "The full rollout preserved the broad contact ordering of the "
            "empirical interictal distribution, but remained extremely close "
            "to the isotropic rollout. The larger probability-level distance "
            "from the empirical distribution identifies smoothing/compression "
            "as a plausible source of the failed static transfer; this audit "
            "does not use the ictal target and is not a new predictive gate."
        ),
        "target_values_read": False,
    }
    atomic_json(OUT / "RANK_DISTRIBUTION_FIDELITY.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
