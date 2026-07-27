#!/usr/bin/env python3
"""Run and analyze the nonblocking formal Claim-1 node-bias control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_formal_node_control_v2_2 import (  # noqa: E402
    evaluate_node_control,
    estimate_node_hazard,
    fit_loso_stop,
    stop_histogram,
)
from src.topic5_symmetric_axis_propagation_state_v2_2 import (  # noqa: E402
    node_bias_fingerprint,
)


BASE = ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2"
DATASET = (
    ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_subject(subject: str) -> dict[str, Any]:
    path = DATASET / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        groups = np.asarray(data["event_group_ids"], dtype=np.int64)
        split = np.asarray(data["event_split"], dtype=np.uint8)
    train = np.flatnonzero(split == 0)
    heldout = np.flatnonzero(split == 1)
    if len(train) == 0 or len(heldout) == 0:
        raise RuntimeError(f"{subject}: empty formal split")
    return {
        "groups": groups,
        "train": train,
        "heldout": heldout,
        "sha256": sha256(path),
    }


def _bootstrap_median_ci(
    values: np.ndarray, seed: int = 20260726
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(values, size=(20_000, len(values)), replace=True), axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def build_baseline(subjects: list[str]) -> pd.DataFrame:
    output = BASE / "formal/claim1_node_control"
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "node_control_patient_metrics.csv"
    complete_path = output / "COMPLETE"
    if baseline_path.is_file() and complete_path.is_file():
        frame = pd.read_csv(baseline_path)
        if (
            set(frame.subject) == set(subjects)
            and len(frame) == len(subjects)
            and "fit_weighting" in frame
            and set(frame.fit_weighting)
            == {"patient-balanced_event-first_eligible-normalized"}
        ):
            return frame

    data = {subject: load_subject(subject) for subject in subjects}
    histograms = {
        subject: stop_histogram(record["groups"], record["train"])
        for subject, record in data.items()
    }
    rows = []
    state_path = output / "run_state.json"
    atomic_json(
        state_path,
        {
            "status": "RUNNING",
            "expected_patients": len(subjects),
            "target_values_read": False,
            "started_unix": time.time(),
        },
    )
    try:
        for index, subject in enumerate(subjects, start=1):
            stop = fit_loso_stop(
                histograms[other] for other in subjects if other != subject
            )
            if not stop.optimizer_success:
                raise RuntimeError(f"{subject}: LOSO node-control STOP failed")
            record = data[subject]
            hazard = estimate_node_hazard(
                record["groups"], record["train"]
            )
            event_nll = evaluate_node_control(
                groups=record["groups"],
                heldout_indices=record["heldout"],
                node_hazard=hazard,
                stop=stop,
            )
            bias = np.log(hazard) - np.log1p(-hazard)
            rows.append(
                {
                    "subject": subject,
                    "n_train_events": len(record["train"]),
                    "n_heldout_events": len(record["heldout"]),
                    "node_bias_next_nll": float(event_nll.mean()),
                    "node_bias_next_nll_median_event": float(
                        np.median(event_nll)
                    ),
                    "c0": stop.c0,
                    "c_n": stop.c_n,
                    "shared_training_patients": len(subjects) - 1,
                    "shared_training_excludes_heldout": True,
                    "shared_stop_train_decisions": stop.n_decisions,
                    "shared_stop_train_terminal": stop.n_terminal,
                    "fit_weighting": (
                        "patient-balanced_event-first_eligible-normalized"
                    ),
                    "node_bias_sha256": node_bias_fingerprint(bias),
                    "input_sha256": record["sha256"],
                    "target_values_read": False,
                }
            )
            print(
                f"[{index:02d}/{len(subjects)}] {subject}: "
                f"node NLL={event_nll.mean():.6g}",
                flush=True,
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(baseline_path, index=False)
        atomic_json(
            state_path,
            {
                "status": "COMPLETE",
                "patients": len(frame),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        complete_path.write_text("COMPLETE\n", encoding="utf-8")
        return frame
    except Exception as exc:
        atomic_json(
            state_path,
            {
                "status": "FAILED",
                "error": repr(exc),
                "target_values_read": False,
                "finished_unix": time.time(),
            },
        )
        raise


def analyze(subjects: list[str], baseline: pd.DataFrame) -> None:
    claim2_path = BASE / "formal/analysis/CLAIM2_STATUS.json"
    if not claim2_path.is_file():
        print("Claim 1 baseline complete; Claim-2 scores still absent")
        return
    claim2 = json.loads(claim2_path.read_text(encoding="utf-8"))
    if claim2.get("status") != "complete":
        print("Claim 1 baseline complete; Claim-2 scores are not frozen")
        return
    rows = []
    baseline_index = baseline.set_index("subject")
    for subject in subjects:
        node = baseline_index.loc[subject]
        full_values = []
        for seed in (17, 29, 43):
            path = (
                BASE
                / "formal/claim2_runs"
                / subject
                / f"seed_{seed}"
                / "metrics.json"
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["node_bias_sha256"] != node["node_bias_sha256"]:
                raise RuntimeError(f"{subject}/seed_{seed}: bias mismatch")
            full_values.append(
                float(
                    record["models"]["full"]["heldout_fit"]["metrics"][
                        "heldout20"
                    ]["next_nll"]
                )
            )
        full = float(np.median(full_values))
        node_nll = float(node["node_bias_next_nll"])
        rows.append(
            {
                "subject": subject,
                "seed_median_full_next_nll": full,
                "node_bias_next_nll": node_nll,
                "full_benefit_over_node_bias": node_nll - full,
                "node_bias_sha256": node["node_bias_sha256"],
                "target_values_read": False,
            }
        )
    patient = pd.DataFrame(rows)
    analysis = BASE / "formal/analysis"
    patient.to_csv(analysis / "claim1_sequence_predictability.csv", index=False)
    values = patient.full_benefit_over_node_bias.to_numpy(dtype=float)
    pvalue = float(
        wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        ).pvalue
    )
    ci_low, ci_high = _bootstrap_median_ci(values)
    n_positive = int(np.sum(values > 0))
    passed = bool(
        np.median(values) > 0
        and n_positive > len(values) / 2
        and pvalue < 0.05
    )
    status = {
        "contract": "topic5_symmetric_axis_propagation_state_rnn",
        "version": "2.2",
        "status": "complete",
        "role": "sanity_replication_nonblocking",
        "claim1_sequence_predictability": "PASS" if passed else "FAIL",
        "n_patients": len(values),
        "median_benefit": float(np.median(values)),
        "median_ci95_low": ci_low,
        "median_ci95_high": ci_high,
        "n_positive": n_positive,
        "fraction_positive": float(np.mean(values > 0)),
        "wilcoxon_one_sided_p": pvalue,
        "seed_aggregation": "median full NLL within patient",
        "node_control_seed_dependence": "none; convex LOSO STOP fit",
        "fit_weighting": (
            "patient-balanced_event-first_eligible-normalized"
        ),
        "used_for_target_unlock": False,
        "claim2_status_sha256": sha256(claim2_path),
        "target_values_read": False,
    }
    atomic_json(analysis / "CLAIM1_STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    lock = json.loads(
        (BASE / "formal/PHYSICAL_AXIS_FORMAL_LOCK.json").read_text(
            encoding="utf-8"
        )
    )
    subjects = list(map(str, lock["subjects"]))
    if len(subjects) != 22:
        raise SystemExit("formal physical-axis cohort drifted")
    target = json.loads(
        (BASE / "target_audit/TARGET_METADATA_GATE.json").read_text(
            encoding="utf-8"
        )
    )
    if target.get("energy_values_read") or target.get("recruitment_values_read"):
        raise SystemExit("target seal violated")
    baseline = build_baseline(subjects)
    if not args.baseline_only:
        analyze(subjects, baseline)


if __name__ == "__main__":
    main()
