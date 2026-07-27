#!/usr/bin/env python3
"""Compute frozen categorical node/Markov benchmarks for formal v2.3."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_transition_decomposition_v0_1 import (  # noqa: E402
    estimate_pair_residual,
    history_contacts,
)


DATASET = (
    ROOT
    / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)
BASE = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
AUDIT = BASE / "input_audit"
HISTORY = (
    ROOT
    / "results/topic5_interictal_transition_decomposition_v0_1"
    / "history_depth_metrics.csv"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def has_non_source_tie(event: np.ndarray) -> bool:
    ranks = event[event > 0]
    if ranks.size == 0:
        return False
    _, counts = np.unique(ranks, return_counts=True)
    return bool(np.any(counts > 1))


def event_nll(
    event: np.ndarray,
    *,
    node_logit: np.ndarray,
    residual: np.ndarray,
    history_mode: str,
    decay: float,
) -> float:
    n_steps = int(np.max(event[event >= 0])) + 1
    terms = []
    for step in range(n_steps - 1):
        seen = (event >= 0) & (event <= step)
        eligible = ~seen
        contacts, weights = history_contacts(
            event, step, history_mode, decay=decay
        )
        weights = weights / weights.sum()
        drive = np.average(residual[contacts], axis=0, weights=weights)
        score = node_logit[eligible] + drive[eligible]
        target = (event == (step + 1))[eligible]
        if int(target.sum()) != 1:
            raise ValueError("categorical benchmark requires one next contact")
        terms.append(float(logsumexp(score) - score[target][0]))
    if not terms:
        raise ValueError("event has no next-contact decision")
    return float(np.mean(terms))


def compute_subject(subject: str) -> list[dict[str, Any]]:
    path = DATASET / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
    keep = np.asarray(
        [not has_non_source_tie(event) for event in groups], dtype=bool
    )
    train = np.flatnonzero((split == 0) & keep)
    heldout = np.flatnonzero((split == 1) & keep)
    pair = estimate_pair_residual(groups, train)
    history = pd.read_csv(HISTORY)
    selected = history.loc[history.subject.astype(str) == subject]
    if len(selected) != 1:
        raise ValueError(f"{subject}: selected history decay missing")
    decay = float(selected.selected_decay.iloc[0])
    zero = np.zeros_like(pair.residual)
    rows = []
    for model, residual, mode in (
        ("node_bias_categorical", zero, "last_rank"),
        ("empirical_last_rank_markov", pair.residual, "last_rank"),
        ("empirical_ordered_history_markov", pair.residual, "ordered_full_prefix"),
    ):
        values = np.asarray(
            [
                event_nll(
                    groups[index],
                    node_logit=pair.node_logit,
                    residual=residual,
                    history_mode=mode,
                    decay=decay,
                )
                for index in heldout
            ],
            dtype=np.float64,
        )
        rows.append(
            {
                "subject": subject,
                "model": model,
                "heldout_categorical_nll": float(values.mean()),
                "heldout_event_median_nll": float(np.median(values)),
                "n_train_events": int(len(train)),
                "n_heldout_events": int(len(heldout)),
                "selected_history_decay": decay,
                "finite": bool(np.all(np.isfinite(values))),
                "target_values_read": False,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("--workers must be in [1, 12]")
    audit = json.loads(
        (AUDIT / "INPUT_AUDIT_STATUS.json").read_text(encoding="utf-8")
    )
    subjects = list(map(str, audit["physical_axis_formal_patients"]))
    if len(subjects) != 22 or audit.get("target_values_read"):
        raise SystemExit("physical cohort or target seal drifted")
    output = BASE / "formal/markov_benchmarks.csv"
    status_path = BASE / "formal/MARKOV_BENCHMARK_STATE.json"
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "started_unix": time.time(),
            "n_subjects": len(subjects),
            "workers": args.workers,
            "target_values_read": False,
        },
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_subject, subject): subject
            for subject in subjects
        }
        for index, future in enumerate(as_completed(futures), start=1):
            subject = futures[future]
            rows.extend(future.result())
            print(f"[{index:02d}/{len(subjects)}] {subject}", flush=True)
    table = pd.DataFrame(rows).sort_values(["subject", "model"])
    if len(table) != 66 or not table.finite.all():
        raise SystemExit("categorical Markov benchmark is incomplete/non-finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    atomic_json(
        status_path,
        {
            "status": "COMPLETE",
            "finished_unix": time.time(),
            "n_subjects": len(subjects),
            "n_rows": len(table),
            "workers": args.workers,
            "target_values_read": False,
        },
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
